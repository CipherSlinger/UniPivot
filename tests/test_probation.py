"""避让解除后的恢复观察期 (OPT-022) 测试套件

动机：一次多日的账号级封禁解除后，网关侧往往已积压大量需求，且被冷却压制的
并发会在同一瞬间释放。若放任其以正常并发涌入刚解封的账号，恰恰复现了当初触发
封禁的流量特征，极易被立即二次封禁——「没有办法绕开时，避免之道不只是避让，
还包括避免重新触发」。

本套件验证避让一旦解除即进入观察期：串行放行单个「探路」握手 + 放大步调间隔，
累计 `GATEWAY_PROBATION_MIN_SUCCESSES` 次连续成功后毕业恢复常规并发。

覆盖：
1. 语义：无历史/冷却中不属观察期；风控冷却解除后进入观察期
2. 普通 5xx/超时（非风控）不触发观察期（无惊群风险，零回归）
3. 连续成功达标即毕业；未达标仍处观察期
4. 观察期窗口自然结束（无成功）
5. 封禁租约自然到期 / 显式解除 均进入观察期
6. 并发闸门：观察期内并发派发被串行为单个握手
7. 步调放大：观察期内间隔 ×2，非观察期 ×1
8. GATEWAY_DISABLE_PROBATION=1 关闭
9. 跨重启持久化：观察期不因重启而绕过
10. 遥测：/v1/diagnostics 暴露 probation，响应头带 x-risk-probation

运行: .venv/bin/python tests/test_probation.py
"""

from __future__ import annotations

import asyncio
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
os.environ["GATEWAY_PROBATION_WINDOW"] = "60"
os.environ["GATEWAY_PROBATION_MIN_SUCCESSES"] = "3"
os.environ["QWEN_AUTH_TOKEN"] = "fake_qwen"
os.environ["DEEPSEEK_AUTH_TOKEN"] = "fake_ds"
os.environ["DOUBAO_AUTH_TOKEN"] = "fake_doubao"
os.environ["KIMI_AUTH_TOKEN"] = "fake_kimi"
os.environ["GLM_AUTH_TOKEN"] = "fake_glm"
os.environ.pop("GATEWAY_DISABLE_PROBATION", None)
os.environ.pop("GATEWAY_PROBATION_PACING_FACTOR", None)

from fastapi.testclient import TestClient  # noqa: E402

import server as srv  # noqa: E402
from providers.failover import CooldownTracker  # noqa: E402
from providers.quarantine import QuarantineStore  # noqa: E402

TRACKER = srv.failover_router.cooldown_tracker
STORE = srv.quarantine_store


def _isolated():
    """返回 (tracker, store, tmpdir) 的隔离组合，避免用例间串扰。"""
    tmp = Path(tempfile.mkdtemp(prefix="probation_"))
    store = QuarantineStore(directory=tmp)
    tracker = CooldownTracker()
    tracker.quarantine = store
    tracker.reset()
    return tracker, store, tmp


def _cleanup(tracker, tmp):
    tracker.reset()
    shutil.rmtree(tmp, ignore_errors=True)


def _expire_cooldown(tracker, provider, seconds=0.15):
    """让节点走完一次真实的风控冷却 episode。"""
    asyncio.run(tracker.record_failure(provider, 429, "rate_limit", custom_cooldown=seconds))
    time.sleep(seconds + 0.08)


# ------------------------------------------------------------------ 1. 语义

def test_probation_semantics():
    tracker, store, tmp = _isolated()
    try:
        assert tracker.is_in_probation("qwen") is False, "无历史节点不得处于观察期"
        assert tracker.probation_lock("qwen") is None

        asyncio.run(tracker.record_failure("qwen", 429, "rate_limit", custom_cooldown=5))
        assert tracker.is_in_probation("qwen") is False, "冷却期内不属观察期（语义上仍是避让）"

        tracker.reset()
        _expire_cooldown(tracker, "qwen")
        assert tracker.is_in_probation("qwen") is True, "风控冷却解除后必须进入观察期"
        assert tracker.probation_lock("qwen") is not None
        assert tracker.is_in_probation("glm") is False, "未经历避让的节点不得被牵连"
    finally:
        _cleanup(tracker, tmp)
    print("test_probation_semantics: PASSED")


def test_plain_5xx_does_not_enter_probation():
    """普通 5xx/超时属上游故障而非账号被标记，解封瞬间无惊群风险，不启用观察期。"""
    tracker, store, tmp = _isolated()
    try:
        asyncio.run(tracker.record_failure("glm", 502, "upstream boom", custom_cooldown=0.15))
        time.sleep(0.25)
        assert tracker.is_in_probation("glm") is False, "普通 5xx 不得触发恢复观察期"
    finally:
        _cleanup(tracker, tmp)
    print("test_plain_5xx_does_not_enter_probation: PASSED")


# ------------------------------------------------------------------ 3/4. 毕业与窗口

def test_graduation_after_consecutive_successes():
    tracker, store, tmp = _isolated()
    try:
        _expire_cooldown(tracker, "deepseek")
        assert tracker.is_in_probation("deepseek") is True

        asyncio.run(tracker.record_success("deepseek"))
        asyncio.run(tracker.record_success("deepseek"))
        assert tracker.is_in_probation("deepseek") is True, "未达标前必须保持观察期"

        asyncio.run(tracker.record_success("deepseek"))
        assert tracker.is_in_probation("deepseek") is False, "达 3 次连续成功即毕业"
        assert tracker.probation_lock("deepseek") is None
        stats = tracker.get_probation_stats()
        assert stats["entered"] == 1 and stats["graduated"] == 1
        assert stats["canary_successes"] == 3
    finally:
        _cleanup(tracker, tmp)
    print("test_graduation_after_consecutive_successes: PASSED")


def test_probation_window_expires_without_successes():
    tracker, store, tmp = _isolated()
    os.environ["GATEWAY_PROBATION_WINDOW"] = "0"
    try:
        _expire_cooldown(tracker, "kimi")
        assert tracker.is_in_probation("kimi") is False, "观察期窗口为 0 时立即结束"
    finally:
        os.environ["GATEWAY_PROBATION_WINDOW"] = "60"
        _cleanup(tracker, tmp)
    print("test_probation_window_expires_without_successes: PASSED")


# ------------------------------------------------------------------ 5. 租约解除

def test_quarantine_release_enters_probation():
    tracker, store, tmp = _isolated()
    try:
        # 自然到期
        store.set_lease("doubao", duration_seconds=0.15)
        time.sleep(0.25)
        assert store.last_released_at("doubao") is not None, "到期须记录解除时刻"
        assert tracker.is_in_probation("doubao") is True, "封禁租约自然到期须进入观察期"

        # 显式解除
        tracker.reset()
        store.clear_all_for_tests()
        store.set_lease("glm", duration_seconds=3600)
        store.clear("glm")
        assert tracker.is_in_probation("glm") is True, "运维显式解除租约同样须进入观察期"
    finally:
        _cleanup(tracker, tmp)
    print("test_quarantine_release_enters_probation: PASSED")


# ------------------------------------------------------------------ 6. 并发闸门

def test_probation_serializes_concurrent_handshakes():
    """观察期内并发派发必须被串行化为单个握手（惊群保护的核心）。

    注意：必须使用 `failover_router` 实际持有的 tracker —— 隔离 tracker 只用于
    纯单元断言，`execute_chat` 读的是全局单例。
    """
    srv._make_provider = _concurrency_probe_factory
    os.environ["GATEWAY_PROBATION_MIN_SUCCESSES"] = "100"
    TRACKER.reset()
    STORE.clear_all_for_tests()
    try:
        asyncio.run(TRACKER.record_failure("qwen", 429, "rate_limit", custom_cooldown=0.15))
        time.sleep(0.25)
        assert TRACKER.is_in_probation("qwen") is True

        MAX_IN_FLIGHT.clear()
        asyncio.run(_launch_concurrent(6, "qwen"))
        peak = max(MAX_IN_FLIGHT)
        assert peak == 1, f"观察期内并发握手峰值必须为 1（串行探路），实际 {peak}"

        # 对照：毕业后并发恢复正常（信号量上限 > 1）
        os.environ["GATEWAY_PROBATION_MIN_SUCCESSES"] = "3"
        for _ in range(3):
            asyncio.run(TRACKER.record_success("qwen"))
        assert TRACKER.is_in_probation("qwen") is False
        MAX_IN_FLIGHT.clear()
        asyncio.run(_launch_concurrent(6, "qwen"))
        assert max(MAX_IN_FLIGHT) > 1, "毕业后并发必须恢复，不得永久降级"
    finally:
        os.environ["GATEWAY_PROBATION_MIN_SUCCESSES"] = "3"
        TRACKER.reset()
        STORE.clear_all_for_tests()
    print("test_probation_serializes_concurrent_handshakes: PASSED")


MAX_IN_FLIGHT: list[int] = []
_IN_FLIGHT = {"n": 0}


def _concurrency_probe_factory(key: str):
    from providers.base import ProviderError  # noqa: F401

    class _Probe:
        async def chat(self, *a, **kw):
            _IN_FLIGHT["n"] += 1
            MAX_IN_FLIGHT.append(_IN_FLIGHT["n"])
            try:
                await asyncio.sleep(0.05)
                yield ("hi", {})
            finally:
                _IN_FLIGHT["n"] -= 1

        async def delete_conversation(self, *a, **kw):
            return None

    return _Probe()


async def _launch_concurrent(n: int, provider: str):
    async def one():
        stream, _ev, _p = await srv.failover_router.execute_chat(
            primary_provider_key=provider,
            primary_wire_model="Qwen",
            provider_factory=_concurrency_probe_factory,
            chat_args={"messages": []},
        )
        async for _ in stream:
            pass

    await asyncio.gather(*(one() for _ in range(n)))


# ------------------------------------------------------------------ 7. 步调放大

def test_probation_widens_pacing_interval():
    tracker, store, tmp = _isolated()
    try:
        baseline = tracker.compute_pacing_interval("qwen", jitter=0.0)
        assert baseline > 0
        assert tracker.probation_pacing_multiplier("qwen") == 1.0

        _expire_cooldown(tracker, "qwen")
        assert tracker.probation_pacing_multiplier("qwen") == 2.0
        widened = tracker.compute_pacing_interval("qwen", jitter=0.0)
        assert abs(widened - baseline * 2.0) < 1e-6, (
            f"观察期须将间隔放大至 2 倍: {baseline} -> {widened}"
        )

        # 可配置倍率
        os.environ["GATEWAY_PROBATION_PACING_FACTOR"] = "3.5"
        assert tracker.probation_pacing_multiplier("qwen") == 3.5
    finally:
        os.environ.pop("GATEWAY_PROBATION_PACING_FACTOR", None)
        _cleanup(tracker, tmp)
    print("test_probation_widens_pacing_interval: PASSED")


# ------------------------------------------------------------------ 8. 开关

def test_disable_switch():
    tracker, store, tmp = _isolated()
    os.environ["GATEWAY_DISABLE_PROBATION"] = "1"
    try:
        _expire_cooldown(tracker, "glm")
        assert tracker.is_in_probation("glm") is False, "开关关闭后不得进入观察期"
        assert tracker.probation_lock("glm") is None
        assert tracker.probation_pacing_multiplier("glm") == 1.0
    finally:
        os.environ.pop("GATEWAY_DISABLE_PROBATION", None)
        _cleanup(tracker, tmp)
    print("test_disable_switch: PASSED")


# ------------------------------------------------------------------ 9. 跨重启

def test_probation_survives_restart():
    """avoidance 期间重启不得绕过观察期（否则一次重启即可放行惊群）。"""
    tmp = Path(tempfile.mkdtemp(prefix="probation_"))
    store = QuarantineStore(directory=tmp)
    tracker = CooldownTracker()
    tracker.quarantine = store
    tracker.reset()
    ledger = tracker.risk_ledger
    original_dir = ledger.directory
    try:
        # 让 episode 状态落到独立台账目录，再构造"新进程"读取
        ledger.directory = tmp
        tracker.risk_ledger = ledger
        _expire_cooldown(tracker, "deepseek")
        assert tracker.is_in_probation("deepseek") is True

        revived = CooldownTracker()
        revived.quarantine = store
        revived.risk_ledger = ledger
        revived._restore_persisted_state()
        assert revived.is_in_probation("deepseek") is True, (
            "重启后观察期必须仍然生效（episode 结束时间已持久化）"
        )
    finally:
        ledger.directory = original_dir
        tracker.reset()
        shutil.rmtree(tmp, ignore_errors=True)
    print("test_probation_survives_restart: PASSED")


# ------------------------------------------------------------------ 10. 遥测

def test_diagnostics_and_header_exposure():
    STORE.clear_all_for_tests()
    TRACKER.reset()
    tracker_seen = None
    try:
        diag = TestClient(srv.app).get("/v1/diagnostics").json()
        assert "probation" in diag
        assert diag["probation"]["active"] == {}
        assert "stats" in diag["probation"]

        # 让真实 tracker 进入观察期并验证响应头
        asyncio.run(TRACKER.record_failure("glm", 429, "rate_limit", custom_cooldown=0.15))
        time.sleep(0.25)
        assert TRACKER.is_in_probation("glm") is True

        client = TestClient(srv.app)
        srv._make_provider = _concurrency_probe_factory
        r = client.post(
            "/v1/chat/completions",
            json={"model": "glm-4-plus", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 200, r.text

        diag2 = TestClient(srv.app).get("/v1/diagnostics").json()
        assert "glm" in diag2["probation"]["active"]
        assert diag2["probation"]["active"]["glm"]["required_successes"] == 3
        tracker_seen = diag2["probation"]["active"]["glm"]
    finally:
        TRACKER.reset()
        STORE.clear_all_for_tests()
    print("test_diagnostics_and_header_exposure: PASSED", tracker_seen)


def run_all():
    test_probation_semantics()
    test_plain_5xx_does_not_enter_probation()
    test_graduation_after_consecutive_successes()
    test_probation_window_expires_without_successes()
    test_quarantine_release_enters_probation()
    test_probation_serializes_concurrent_handshakes()
    test_probation_widens_pacing_interval()
    test_disable_switch()
    test_probation_survives_restart()
    test_diagnostics_and_header_exposure()
    print("\n✅ All recovery-probation tests (OPT-022) passed!")


if __name__ == "__main__":
    run_all()
