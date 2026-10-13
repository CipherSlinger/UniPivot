"""风控拦截持久化台账与冷却状态跨重启恢复引擎 (OPT-011) 测试套件

验证：
1. 严重度分级 (classify_interception_severity)：auth > ratelimit > waf 及五家上游已知原因串
2. 冷却状态原子落盘/读取往返 (POSIX os.replace)
3. 跨重启冷却状态恢复（新建 CooldownTracker 仍保有避让）
4. JSONL 台账追加与 recent_interceptions 时间窗聚合
5. 台账超限自动裁剪 (max_lines)
6. GATEWAY_DISABLE_RISK_LEDGER 完全关闭持久化门控
7. 认证失效仅在连续 ≥3 次后升级为长效冷却（避免误锁健康节点）
8. record_success 同时清空内存与磁盘状态
9. 普通 5xx/超时不入账（保护风控画像）
10. GET /v1/diagnostics 与 GET /health 暴露 risk_ledger 段

运行: .venv/bin/python tests/test_risk_ledger.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["GATEWAY_DISABLE_AUTO_REFRESH"] = "1"
os.environ["TESTING"] = "1"
os.environ["GATEWAY_DISABLE_PACING"] = "1"
os.environ["QWEN_AUTH_TOKEN"] = "fake_qwen"
os.environ["DEEPSEEK_AUTH_TOKEN"] = "fake_ds"
os.environ["DOUBAO_AUTH_TOKEN"] = "fake_doubao"
os.environ["KIMI_AUTH_TOKEN"] = "fake_kimi"
os.environ["GLM_AUTH_TOKEN"] = "fake_glm"

from fastapi.testclient import TestClient  # noqa: E402

import server as srv  # noqa: E402
import providers.failover as fov  # noqa: E402
from providers.risk_ledger import (  # noqa: E402
    SEVERITY_AUTH,
    SEVERITY_RATE_LIMIT,
    SEVERITY_WAF,
    RiskLedger,
    classify_interception_severity,
)


def _fresh_ledger() -> RiskLedger:
    return RiskLedger(Path(tempfile.mkdtemp(prefix="risk_ledger_test_")))


def _bind_global(ledger: RiskLedger):
    """将 failover 模块解析到的全局台账指向测试实例，返回还原函数。"""
    original = fov.get_risk_ledger
    fov.get_risk_ledger = lambda: ledger  # type: ignore[assignment]
    return original


def test_severity_classification_matrix():
    """auth > ratelimit > waf 的优先级与已知原因串识别。"""
    # 认证失效（401 与令牌/账号级文案）
    assert classify_interception_severity(False, "", 401, "") == SEVERITY_AUTH
    assert classify_interception_severity(False, "", 0, "auth expiration detected") == SEVERITY_AUTH
    assert classify_interception_severity(False, "", 0, "令牌失效，请重新登录") == SEVERITY_AUTH
    assert classify_interception_severity(False, "", 403, "authorization failed") == SEVERITY_AUTH

    # 频率限制（429 与频控文案）
    assert classify_interception_severity(False, "", 429, "too many requests") == SEVERITY_RATE_LIMIT
    assert classify_interception_severity(False, "", 0, "rate limit exceeded") == SEVERITY_RATE_LIMIT
    assert classify_interception_severity(False, "", 0, "请求过于频繁，触发频控") == SEVERITY_RATE_LIMIT

    # WAF 人机验证（默认归类）
    assert classify_interception_severity(True, "RGV587 人机验证", 403, "rgv587 captcha") == SEVERITY_WAF
    assert classify_interception_severity(True, "滑块验证") == SEVERITY_WAF

    # 认证优先于频控
    assert classify_interception_severity(True, "", 401, "rate limit") == SEVERITY_AUTH
    print("test_severity_classification_matrix: PASSED")


def test_atomic_state_roundtrip():
    """冷却状态原子落盘后能被重新读取。"""
    ledger = _fresh_ledger()
    state = {"cooldowns": {"deepseek": time.time() + 900}, "failure_counts": {"deepseek": 2}}
    ledger.save_state(state)
    assert ledger.cooldown_path.exists()
    loaded = ledger.load_state()
    assert loaded["failure_counts"] == {"deepseek": 2}
    assert abs(loaded["cooldowns"]["deepseek"] - state["cooldowns"]["deepseek"]) < 1e-6
    # 文件权限必须严格为 0o600
    mode = ledger.cooldown_path.stat().st_mode & 0o777
    assert mode == 0o600, oct(mode)
    print("test_atomic_state_roundtrip: PASSED")


def test_cross_restart_cooldown_recovery():
    """进程重启（新建 CooldownTracker）后避让冷却状态依然生效。"""
    ledger = _fresh_ledger()
    original = _bind_global(ledger)
    try:
        tracker1 = fov.CooldownTracker()

        async def _triple_auth_fail():
            for _ in range(3):
                await tracker1.record_failure("deepseek", 401, "auth expiration, please relogin")

        asyncio.run(_triple_auth_fail())
        rem_before = tracker1.get_remaining_cooldown("deepseek")
        assert rem_before > 1000, f"第 3 次认证失败应升级为长效冷却，实际 {rem_before}"

        # 模拟重启：全新 CooldownTracker 从同一磁盘目录恢复
        tracker2 = fov.CooldownTracker()
        assert tracker2.is_in_cooldown("deepseek"), "重启后必须仍处于冷却避让期"
        rem_after = tracker2.get_remaining_cooldown("deepseek")
        assert rem_after > 900, f"重启后应恢复接近的冷却时长，实际 {rem_after}"
        assert tracker2._failure_counts.get("deepseek") == 3
    finally:
        fov.get_risk_ledger = original  # type: ignore[assignment]
    print("test_cross_restart_cooldown_recovery: PASSED")


def test_ledger_append_and_window():
    """JSONL 追加与 recent_interceptions 时间窗聚合。"""
    ledger = _fresh_ledger()
    now = 1_000_000.0
    ledger.record_interception("qwen", 403, "RGV587", SEVERITY_WAF, 45.0, "rgv587 captcha", now=now - 7200)
    ledger.record_interception("deepseek", 401, "expired", SEVERITY_AUTH, 1800.0, "auth expiration", now=now - 60)
    ledger.record_interception("qwen", 429, "freq", SEVERITY_RATE_LIMIT, 60.0, "rate limit", now=now - 10)

    # 台账为纯 JSONL 追加
    assert len(ledger.ledger_path.read_text(encoding="utf-8").strip().splitlines()) == 3

    window = ledger.recent_interceptions(hours=1.0, now=now)
    assert window["total"] == 2, window  # 1 小时前的那条被过滤
    assert window["per_provider"]["qwen"]["count"] == 1
    assert window["per_provider"]["deepseek"]["last_severity"] == SEVERITY_AUTH

    wide = ledger.recent_interceptions(hours=24.0, now=now)
    assert wide["total"] == 3
    assert wide["per_provider"]["qwen"]["severities"] == {SEVERITY_WAF: 1, SEVERITY_RATE_LIMIT: 1}
    print("test_ledger_append_and_window: PASSED")


def test_ledger_pruning():
    """台账行数超限时裁剪保留后一半。"""
    ledger = _fresh_ledger()
    ledger.max_lines = 5
    now = time.time()
    for i in range(8):
        ledger.record_interception("glm", 429, f"freq-{i}", SEVERITY_RATE_LIMIT, 30.0, now=now + i)
    lines = ledger.ledger_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) <= ledger.max_lines, f"裁剪后不应超过上限，实际 {len(lines)}"
    assert ledger._stats["ledger_prunes"] >= 1
    # 最新一条必须被保留
    assert "freq-7" in lines[-1]
    print("test_ledger_pruning: PASSED")


def test_disable_gate():
    """GATEWAY_DISABLE_RISK_LEDGER=1 时完全关闭持久化。"""
    ledger = _fresh_ledger()
    os.environ["GATEWAY_DISABLE_RISK_LEDGER"] = "1"
    try:
        assert ledger.is_disabled() is True
        ledger.save_state({"cooldowns": {"qwen": time.time() + 300}, "failure_counts": {"qwen": 1}})
        ledger.record_interception("qwen", 429, "freq", SEVERITY_RATE_LIMIT, 30.0)
        assert not ledger.cooldown_path.exists(), "禁用时不应写入冷却状态"
        assert not ledger.ledger_path.exists(), "禁用时不应写入台账"
        assert ledger.load_state() == {}
        assert ledger.get_diagnostics()["enabled"] is False
    finally:
        os.environ.pop("GATEWAY_DISABLE_RISK_LEDGER", None)
    print("test_disable_gate: PASSED")


def test_auth_escalation_only_after_repeated_failures():
    """首次认证失败维持短冷却，连续 ≥3 次才升级为长效冷却。"""
    os.environ["GATEWAY_AUTH_COOLDOWN_BASE"] = "1800"
    os.environ["GATEWAY_AUTH_COOLDOWN_MAX"] = "86400"
    ledger = _fresh_ledger()
    original = _bind_global(ledger)
    try:
        tracker = fov.CooldownTracker()
        first = asyncio.run(tracker.record_failure("glm", 401, "auth expiration"))
        assert first < 1000, f"首次认证失败不应长效锁定，实际 {first}"
        second = asyncio.run(tracker.record_failure("glm", 401, "auth expiration"))
        assert second < 1000, f"第 2 次仍未达升级阈值，实际 {second}"
        third = asyncio.run(tracker.record_failure("glm", 401, "auth expiration"))
        assert third >= 1800, f"第 3 次应升级为长效冷却 (>=1800s)，实际 {third}"
    finally:
        fov.get_risk_ledger = original  # type: ignore[assignment]
        os.environ.pop("GATEWAY_AUTH_COOLDOWN_BASE", None)
        os.environ.pop("GATEWAY_AUTH_COOLDOWN_MAX", None)
    print("test_auth_escalation_only_after_repeated_failures: PASSED")


def test_record_success_clears_memory_and_disk():
    """record_success 同时清空内存冷却与磁盘状态。"""
    ledger = _fresh_ledger()
    original = _bind_global(ledger)
    try:
        tracker = fov.CooldownTracker()
        asyncio.run(tracker.record_failure("doubao", 429, "rate limit"))

        async def _recover():
            await tracker.record_success("doubao")

        asyncio.run(_recover())
        assert not tracker.is_in_cooldown("doubao")
        state = ledger.load_state()
        assert state.get("cooldowns") == {}
        assert state.get("failure_counts", {}).get("doubao", 0) == 0
    finally:
        fov.get_risk_ledger = original  # type: ignore[assignment]
    print("test_record_success_clears_memory_and_disk: PASSED")


def test_plain_transport_failure_not_ledgered():
    """普通 5xx/超时不计入风控台账（保护风控画像）。"""
    ledger = _fresh_ledger()
    original = _bind_global(ledger)
    try:
        tracker = fov.CooldownTracker()
        asyncio.run(tracker.record_failure("kimi", 500, "upstream internal error"))
        assert not ledger.ledger_path.exists() or not ledger.ledger_path.read_text(encoding="utf-8").strip()
    finally:
        fov.get_risk_ledger = original  # type: ignore[assignment]
    print("test_plain_transport_failure_not_ledgered: PASSED")


def test_diagnostics_and_health_expose_risk_ledger():
    """GET /v1/diagnostics 与 GET /health 暴露 risk_ledger 段。"""
    client = TestClient(srv.app)
    r = client.get("/v1/diagnostics")
    assert r.status_code == 200
    diag = r.json()["risk_ledger"]
    for key in ("enabled", "directory", "cooldown_path", "ledger_path", "recent", "stats"):
        assert key in diag, key
    assert "per_provider" in diag["recent"]

    h = client.get("/health")
    assert h.status_code == 200
    health = h.json()["risk_ledger"]
    for key in ("enabled", "recent_24h_total", "recent_24h_per_provider"):
        assert key in health, key
    print("test_diagnostics_and_health_expose_risk_ledger: PASSED")


def run_all():
    test_severity_classification_matrix()
    test_atomic_state_roundtrip()
    test_cross_restart_cooldown_recovery()
    test_ledger_append_and_window()
    test_ledger_pruning()
    test_disable_gate()
    test_auth_escalation_only_after_repeated_failures()
    test_record_success_clears_memory_and_disk()
    test_plain_transport_failure_not_ledgered()
    test_diagnostics_and_health_expose_risk_ledger()
    print("\n✅ All risk ledger & durable cooldown tests (OPT-011) passed!")


if __name__ == "__main__":
    run_all()
