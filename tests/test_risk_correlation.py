"""跨节点风控关联与系统性升级研判 (OPT-024) 测试套件

动机：OPT-011~OPT-023 的全部避让逻辑都以「单个 Provider 账号」为单位。但风控拦截存在
第二种成因——**我们自己的客户端足迹**（请求头指纹、UA、设备参数、出口 IP、时间分布）
被平台标记。此时各账号被拦并非彼此独立，而是同一根因的多个表现。

二者需要**完全相反**的应对：单账号成因下「换一个节点继续打」是正确的；系统性成因下
换节点毫无用处——新节点会以同样的足迹被立即拦下，只会逐个把剩下的账号也送进封禁。
单账号视角永远看不到这一层，于是一个接一个地消耗掉全部五个账号。

本套件验证 `providers/risk_correlation.py`：
1. 关联性来自**去重节点数**而非拦截次数（同一账号被拦百次仍是单账号问题）
2. 分级：none < localized < widespread < systemic，突发度（强信号）优先判定
3. 全局步调压制：systemic ×2.0、widespread ×1.25、无证据 ×1.0（零回归）
4. 应对策略正确：单账号成因下「换节点有效」，系统性成因下必须为「换节点无效」
5. 全局倍率经 TTL 缓存生效且确实作用于 compute_pacing_interval 与 TrafficPacer
6. 窗口外证据不参与判定；开关 GATEWAY_DISABLE_RISK_CORRELATION
7. 只读性：研判不得写台账
8. 台账异常降级为「无证据」而非中断
9. 遥测：/v1/diagnostics 与 GET /v1/admin/risk/correlation

运行: .venv/bin/python tests/test_risk_correlation.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["TESTING"] = "1"
os.environ["GATEWAY_DISABLE_AUTO_REFRESH"] = "1"
os.environ["GATEWAY_DISABLE_PACING"] = "1"
os.environ.pop("GATEWAY_DISABLE_RISK_LEDGER", None)
os.environ.pop("GATEWAY_DISABLE_RISK_CORRELATION", None)
os.environ.pop("GATEWAY_DISABLE_ADAPTIVE_PACING", None)
os.environ["QWEN_AUTH_TOKEN"] = "fake_qwen"
os.environ["DEEPSEEK_AUTH_TOKEN"] = "fake_ds"
os.environ["DOUBAO_AUTH_TOKEN"] = "fake_doubao"
os.environ["KIMI_AUTH_TOKEN"] = "fake_kimi"
os.environ["GLM_AUTH_TOKEN"] = "fake_glm"

from fastapi.testclient import TestClient  # noqa: E402

import providers.failover as fov  # noqa: E402
import server as srv  # noqa: E402
from providers.risk_correlation import (  # noqa: E402
    ESCALATION_LOCALIZED,
    ESCALATION_NONE,
    ESCALATION_PLAYBOOK,
    ESCALATION_SYSTEMIC,
    ESCALATION_WIDESPREAD,
    RiskCorrelationDetector,
)
from providers.risk_ledger import (  # noqa: E402
    SEVERITY_AUTH,
    SEVERITY_RATE_LIMIT,
    SEVERITY_WAF,
    RiskLedger,
)


def _isolated():
    """返回 (detector, ledger, tmpdir) 隔离组合。"""
    tmp = Path(tempfile.mkdtemp(prefix="correlation_"))
    ledger = RiskLedger(directory=tmp)
    detector = RiskCorrelationDetector(ledger=ledger)
    return detector, ledger, tmp


def _cleanup(tmp):
    shutil.rmtree(tmp, ignore_errors=True)


def _hit(ledger, provider, severity=SEVERITY_WAF, ago=0.0, status=403, reason="rgv587"):
    ledger.record_interception(
        provider=provider, status=status, reason=reason, severity=severity,
        cooldown_seconds=45.0, evidence="", now=time.time() - ago,
    )


# ------------------------------------------------------------------ 1. 去重语义

def test_distinct_providers_not_interception_count():
    """同一账号被拦一百次仍是「单账号事件」——关联性来自去重节点数。"""
    detector, ledger, tmp = _isolated()
    try:
        for i in range(100):
            _hit(ledger, "qwen", ago=i * 0.5)
        r = detector.evaluate()
        assert r["level"] == ESCALATION_LOCALIZED, r["level"]
        assert r["distinct_provider_count"] == 1
        assert r["burst_provider_count"] == 1
        assert r["window_total"] == 100
        assert r["pacing_multiplier"] == 1.0, "单账号事件不得触发全局压制"
        assert ESCALATION_PLAYBOOK[r["level"]]["rotate_node_helps"] is True
    finally:
        _cleanup(tmp)
    print("test_distinct_providers_not_interception_count: PASSED")


def test_empty_ledger_is_none():
    detector, ledger, tmp = _isolated()
    try:
        r = detector.evaluate()
        assert r["level"] == ESCALATION_NONE
        assert r["pacing_multiplier"] == 1.0
        assert r["distinct_providers"] == []
        assert ESCALATION_PLAYBOOK[r["level"]]["rotate_node_helps"] is True
    finally:
        _cleanup(tmp)
    print("test_empty_ledger_is_none: PASSED")


# ------------------------------------------------------------------ 2. 分级

def test_systemic_on_tight_burst():
    """三个不同账号在突发窗口内被集中拦截 -> systemic（强信号）。"""
    detector, ledger, tmp = _isolated()
    try:
        _hit(ledger, "qwen", ago=30)
        _hit(ledger, "glm", ago=20)
        assert detector.evaluate()["level"] == ESCALATION_LOCALIZED, "两个节点尚未构成系统性判定"

        _hit(ledger, "kimi", ago=10)
        r = detector.evaluate()
        assert r["level"] == ESCALATION_SYSTEMIC
        assert r["burst_provider_count"] == 3
        assert sorted(r["burst_providers"]) == ["glm", "kimi", "qwen"]
        assert r["pacing_multiplier"] == 2.0
        assert ESCALATION_PLAYBOOK[r["level"]]["rotate_node_helps"] is False, (
            "系统性成因下必须明确「换节点无效」"
        )
        assert ESCALATION_PLAYBOOK[r["level"]]["per_account_avoidance_helps"] is False
    finally:
        _cleanup(tmp)
    print("test_systemic_on_tight_burst: PASSED")


def test_widespread_when_spread_over_time():
    """三个账号被拦但时间上分散（无突发）-> widespread，仅轻量压制。

    关键区分：时间集中的多账号同时被拦几乎不可能彼此独立；时间分散则可能是
    各自独立的原因，因此只轻量放大并提示人工确认，不施加重压。
    """
    detector, ledger, tmp = _isolated()
    try:
        # 突发窗口默认 120s；把三次拦截摊开到 10 分钟内
        _hit(ledger, "qwen", ago=600)
        _hit(ledger, "glm", ago=400)
        _hit(ledger, "kimi", ago=200)
        r = detector.evaluate()
        assert r["burst_provider_count"] == 1, f"分散事件不应产生突发，实际 {r['burst_provider_count']}"
        assert r["level"] == ESCALATION_WIDESPREAD, r["level"]
        assert r["pacing_multiplier"] == 1.25
        assert r["distinct_provider_count"] == 3
        assert ESCALATION_PLAYBOOK[r["level"]]["rotate_node_helps"] is False
    finally:
        _cleanup(tmp)
    print("test_widespread_when_spread_over_time: PASSED")


def test_burst_window_boundary():
    """恰好越过突发窗口的记录不得被算作突发成员。"""
    detector, ledger, tmp = _isolated()
    os.environ["GATEWAY_RISK_CORRELATION_BURST_WINDOW"] = "60"
    try:
        now = time.time()
        ledger.record_interception("qwen", 403, "rgv587", SEVERITY_WAF, 45.0, now=now - 100)
        ledger.record_interception("glm", 403, "rgv587", SEVERITY_WAF, 45.0, now=now - 20)
        ledger.record_interception("kimi", 403, "rgv587", SEVERITY_WAF, 45.0, now=now - 10)
        r = detector.evaluate(now)
        assert r["burst_provider_count"] == 2, r["burst_provider_count"]
        assert "qwen" not in r["burst_providers"]
        assert r["level"] == ESCALATION_WIDESPREAD, "窗口内 3 个节点但突发仅 2 个 -> widespread"
    finally:
        os.environ["GATEWAY_RISK_CORRELATION_BURST_WINDOW"] = "120"
        _cleanup(tmp)
    print("test_burst_window_boundary: PASSED")


def test_window_excludes_old_evidence():
    """窗口外的拦截证据不参与关联判定。"""
    detector, ledger, tmp = _isolated()
    os.environ["GATEWAY_RISK_CORRELATION_WINDOW"] = "300"
    try:
        now = time.time()
        for pk in ("qwen", "glm", "kimi"):
            ledger.record_interception(pk, 403, "rgv587", SEVERITY_WAF, 45.0, now=now - 7200)
        r = detector.evaluate(now)
        assert r["level"] == ESCALATION_NONE, "两小时前的证据不得进入 5 分钟窗口"
        assert r["window_total"] == 0
    finally:
        os.environ["GATEWAY_RISK_CORRELATION_WINDOW"] = "900"
        _cleanup(tmp)
    print("test_window_excludes_old_evidence: PASSED")


def test_severity_breakdown_reported():
    detector, ledger, tmp = _isolated()
    try:
        _hit(ledger, "qwen", severity=SEVERITY_AUTH, status=401, reason="invalid token", ago=30)
        _hit(ledger, "glm", severity=SEVERITY_WAF, ago=20)
        _hit(ledger, "kimi", severity=SEVERITY_RATE_LIMIT, status=429, reason="rate limit", ago=10)
        r = detector.evaluate()
        assert set(r["by_severity"]) == {SEVERITY_AUTH, SEVERITY_WAF, SEVERITY_RATE_LIMIT}
        assert r["by_severity"][SEVERITY_WAF]["provider_count"] == 1
        assert r["by_severity"][SEVERITY_WAF]["providers"] == ["glm"]
        assert r["level"] == ESCALATION_SYSTEMIC
    finally:
        _cleanup(tmp)
    print("test_severity_breakdown_reported: PASSED")


# ------------------------------------------------------------------ 3. 步调联通

def test_global_multiplier_applies_to_pacing():
    """全局压制倍率必须真正作用于步调计算，而不只是被报告出来。"""
    detector, ledger, tmp = _isolated()
    original = fov.get_risk_correlation
    fov.get_risk_correlation = lambda: detector  # type: ignore[assignment]
    try:
        tracker = fov.CooldownTracker()
        tracker.quarantine = srv.quarantine_store
        tracker.risk_ledger = ledger
        tracker.reset()

        detector.clear_cache_for_tests()
        baseline = tracker.compute_pacing_interval("qwen", jitter=0.0)
        assert detector.global_pacing_multiplier() == 1.0

        _hit(ledger, "qwen", ago=30)
        _hit(ledger, "glm", ago=20)
        _hit(ledger, "kimi", ago=10)
        detector.clear_cache_for_tests()
        assert detector.global_pacing_multiplier() == 2.0

        widened = tracker.compute_pacing_interval("qwen", jitter=0.0)
        assert abs(widened - baseline * 2.0) < 1e-6, (
            f"全局系统性压制必须作用于步调: {baseline} -> {widened}"
        )

        # TrafficPacer 路径同样遵循
        stats = fov.TrafficPacer().get_stats()
        info = stats["providers"]["qwen"]
        assert info["global_risk_multiplier"] == 2.0
        assert abs(info["effective_interval_s"] - info["min_interval_s"] * 2.0) < 1e-6
    finally:
        fov.get_risk_correlation = original  # type: ignore[assignment]
        _cleanup(tmp)
    print("test_global_multiplier_applies_to_pacing: PASSED")


def test_cache_ttl_and_invalidation():
    """全局倍率带 TTL 缓存（步调路径调用频繁），失效后可重新研判。"""
    detector, ledger, tmp = _isolated()
    os.environ["GATEWAY_RISK_CORRELATION_CACHE_TTL"] = "3600"
    try:
        detector.clear_cache_for_tests()
        assert detector.cached_global_pacing_multiplier() == 1.0
        _hit(ledger, "qwen", ago=30)
        _hit(ledger, "glm", ago=20)
        _hit(ledger, "kimi", ago=10)
        # TTL 内仍返回缓存值（这正是「避免每次派发全量扫描台账」的意义）
        assert detector.cached_global_pacing_multiplier() == 1.0
        detector.clear_cache_for_tests()
        assert detector.cached_global_pacing_multiplier() == 2.0

        # TTL=0 -> 每次都重新研判
        os.environ["GATEWAY_RISK_CORRELATION_CACHE_TTL"] = "0"
        assert detector.cached_global_pacing_multiplier() == 2.0
    finally:
        os.environ["GATEWAY_RISK_CORRELATION_CACHE_TTL"] = "15"
        _cleanup(tmp)
    print("test_cache_ttl_and_invalidation: PASSED")


# ------------------------------------------------------------------ 4. 只读与健壮

def test_read_only():
    detector, ledger, tmp = _isolated()
    try:
        _hit(ledger, "qwen", ago=30)
        before = ledger.ledger_path.read_text(encoding="utf-8")
        for _ in range(3):
            detector.evaluate()
            detector.global_pacing_multiplier()
        after = ledger.ledger_path.read_text(encoding="utf-8")
        assert after == before, "关联研判不得写台账"
    finally:
        _cleanup(tmp)
    print("test_read_only: PASSED")


def test_survives_broken_ledger():
    detector, ledger, tmp = _isolated()

    class _Boom(RiskLedger):
        def _read_tail(self, limit):  # type: ignore[override]
            raise RuntimeError("boom")

    try:
        boom = RiskCorrelationDetector(ledger=_Boom(directory=tmp))
        r = boom.evaluate()
        assert r["level"] == ESCALATION_NONE
        assert boom.global_pacing_multiplier() == 1.0, "台账异常必须降级为无压制，绝不影响请求"
    finally:
        _cleanup(tmp)
    print("test_survives_broken_ledger: PASSED")


# ------------------------------------------------------------------ 5. 开关

def test_disable_switch():
    detector, ledger, tmp = _isolated()
    os.environ["GATEWAY_DISABLE_RISK_CORRELATION"] = "1"
    try:
        _hit(ledger, "qwen", ago=30)
        _hit(ledger, "glm", ago=20)
        _hit(ledger, "kimi", ago=10)
        r = detector.evaluate()
        assert r["level"] == ESCALATION_NONE
        assert r["pacing_multiplier"] == 1.0, "开关关闭后不得施加全局压制"
    finally:
        os.environ.pop("GATEWAY_DISABLE_RISK_CORRELATION", None)
        _cleanup(tmp)
    print("test_disable_switch: PASSED")


# ------------------------------------------------------------------ 6. 端到端

def test_endpoint_exposure():
    ledger = srv.risk_ledger
    try:
        ledger.clear()
        srv.risk_correlation.clear_cache_for_tests()

        diag = TestClient(srv.app).get("/v1/diagnostics").json()
        assert "risk_correlation" in diag
        assert diag["risk_correlation"]["level"] == ESCALATION_NONE

        r = TestClient(srv.app).get("/v1/admin/risk/correlation")
        assert r.status_code == 200, r.text
        body = r.json()
        for key in ("level", "playbook", "burst_providers", "pacing_multiplier", "by_severity"):
            assert key in body
        assert body["playbook"]["rotate_node_helps"] is True

        now = time.time()
        for i, pk in enumerate(("qwen", "glm", "kimi")):
            ledger.record_interception(pk, 403, "rgv587", SEVERITY_WAF, 45.0, now=now - 30 + i)
        srv.risk_correlation.clear_cache_for_tests()

        body2 = TestClient(srv.app).get("/v1/admin/risk/correlation").json()
        assert body2["level"] == ESCALATION_SYSTEMIC
        assert body2["pacing_multiplier"] == 2.0
        assert body2["playbook"]["rotate_node_helps"] is False
        assert len(body2["burst_providers"]) == 3
    finally:
        ledger.clear()
        srv.risk_correlation.clear_cache_for_tests()
    print("test_endpoint_exposure: PASSED")


def run_all():
    test_distinct_providers_not_interception_count()
    test_empty_ledger_is_none()
    test_systemic_on_tight_burst()
    test_widespread_when_spread_over_time()
    test_burst_window_boundary()
    test_window_excludes_old_evidence()
    test_severity_breakdown_reported()
    test_global_multiplier_applies_to_pacing()
    test_cache_ttl_and_invalidation()
    test_read_only()
    test_survives_broken_ledger()
    test_disable_switch()
    test_endpoint_exposure()
    print("\n✅ All risk-correlation tests (OPT-024) passed!")


if __name__ == "__main__":
    run_all()
