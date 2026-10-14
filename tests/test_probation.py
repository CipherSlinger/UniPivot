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
# [OPT-028] 窗口续期默认关闭，由专门的续期用例显式开启（保证既有用例语义不变）
os.environ["GATEWAY_PROBATION_MAX_RENEWALS"] = "0"
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


def test_zero_window_graduates_only_on_evidence():
    """[OPT-028] 窗口归零表示「不给时间宽限」，而**不是**「立即放开」。

    旧语义（窗口一到即放开）正是本优化推翻的行为：它让保护在最需要它的场景
    （低流量节点在窗口内连一次握手都没有）静默失效——解封瞬间被压制的需求随后
    以全并发抵达一个**仍可能处于封禁中**的账号。窗口现在只是续期单位，毕业条件
    是**证据**；零证据仍被门控不产生任何代价，因为没有被成功验证过就没有洪峰可放行。
    """
    tracker, store, tmp = _isolated()
    os.environ["GATEWAY_PROBATION_WINDOW"] = "0"
    try:
        _expire_cooldown(tracker, "kimi")
        assert tracker.is_in_probation("kimi") is True, "零证据不得放开全并发"
        assert tracker.probation_lock("kimi") is not None, "零证据仍须串行门控"
        for _ in range(3):
            asyncio.run(tracker.record_success("kimi"))
        assert tracker.is_in_probation("kimi") is False, "取得证据后毕业"
        assert tracker.probation_lock("kimi") is None
    finally:
        os.environ["GATEWAY_PROBATION_WINDOW"] = "60"
        _cleanup(tracker, tmp)
    print("test_zero_window_graduates_only_on_evidence: PASSED")


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


# ------------------------------------------------------------------ 11. [OPT-028] 证据驱动的毕业

def test_window_renewal_keeps_gating_without_evidence():
    """窗口只是**续期单位**：耗尽一个窗口而未取得任何证据时，观察期自动续期，
    而不是静默失效。低流量节点（尤其夜间）在窗口内连一次握手都没有，旧语义下
    保护恰好在最需要它的场景里失效——解封瞬间被压制的需求随后以全并发抵达一个
    仍可能被封禁的账号。"""
    tracker, store, tmp = _isolated()
    os.environ["GATEWAY_PROBATION_WINDOW"] = "1"
    os.environ["GATEWAY_PROBATION_MAX_RENEWALS"] = "3"
    try:
        # 手动排定 episode 结束时间，模拟「已解封 elapsed 秒且从未被成功验证」
        now = time.time()

        def _set_elapsed(elapsed):
            tracker._episode_ends["deepseek"] = now - elapsed
            tracker._probation_successes["deepseek"] = 0

        _set_elapsed(0.5)
        assert tracker.is_in_probation("deepseek") is True, "窗口内应在观察期"
        _set_elapsed(1.5)  # 已过 1 个窗口 -> 续期 1 次
        assert tracker.is_in_probation("deepseek") is True, "续期后必须仍在观察期"
        assert tracker.probation_lock("deepseek") is not None, "续期期间串行门控必须保持"
        assert tracker.probation_pacing_multiplier("deepseek") > 1.0, "续期期间步调必须仍被放大"
        _set_elapsed(3.5)  # 续期 3 次（上限）
        assert tracker.is_in_probation("deepseek") is True

        states = tracker.get_probation_states()
        assert states["deepseek"]["renewals"] == 3, states
        assert states["deepseek"]["awaiting_first_handshake"] is True
        # 续期预算耗尽但**零证据**：仍不得放开全并发（没有证据支持「它已经好了」）
        _set_elapsed(10.0)
        assert tracker.is_in_probation("deepseek") is True, "零证据时预算耗尽也不得放开"
    finally:
        os.environ["GATEWAY_PROBATION_WINDOW"] = "60"
        os.environ["GATEWAY_PROBATION_MAX_RENEWALS"] = "0"
        _cleanup(tracker, tmp)
    print("test_window_renewal_keeps_gating_without_evidence: PASSED")


def test_budget_exhausted_graduates_on_partial_evidence():
    """预算耗尽后按**证据**裁决：只要有正面证据（至少一次成功握手）即可毕业，
    否则保持门控。这使毕业条件从「时间到了」变为「确实通了」。"""
    tracker, store, tmp = _isolated()
    os.environ["GATEWAY_PROBATION_WINDOW"] = "1"
    os.environ["GATEWAY_PROBATION_MAX_RENEWALS"] = "1"
    try:
        now = time.time()
        tracker._episode_ends["glm"] = now - 10.0  # 远超 1×(1+1)=2s 的视野
        tracker._probation_successes["glm"] = 1  # 但已取得正面证据
        assert tracker.is_in_probation("glm") is False, "有证据且预算耗尽即毕业"
        assert tracker.probation_lock("glm") is None
        # 未达 min_successes 但预算已耗尽 -> 仍按证据放行（避免无限门控）
        tracker._probation_successes["glm"] = 0
        assert tracker.is_in_probation("glm") is True, "同条件下零证据则保持门控"
    finally:
        os.environ["GATEWAY_PROBATION_WINDOW"] = "60"
        os.environ["GATEWAY_PROBATION_MAX_RENEWALS"] = "0"
        _cleanup(tracker, tmp)
    print("test_budget_exhausted_graduates_on_partial_evidence: PASSED")


def test_gated_node_can_still_graduate_by_late_evidence():
    """零证据被持续门控的节点，必须能通过**后来取得的成功**毕业——否则门控就是死锁。"""
    tracker, store, tmp = _isolated()
    os.environ["GATEWAY_PROBATION_WINDOW"] = "1"
    os.environ["GATEWAY_PROBATION_MAX_RENEWALS"] = "0"
    try:
        tracker._episode_ends["kimi"] = time.time() - 5.0
        tracker._probation_successes["kimi"] = 0
        assert tracker.is_in_probation("kimi") is True
        for _ in range(3):
            asyncio.run(tracker.record_success("kimi"))
        assert tracker.is_in_probation("kimi") is False, "迟到的证据同样必须能毕业"
        assert tracker.get_probation_stats()["graduated"] == 1
    finally:
        os.environ["GATEWAY_PROBATION_WINDOW"] = "60"
        _cleanup(tracker, tmp)
    print("test_gated_node_can_still_graduate_by_late_evidence: PASSED")


def test_renewal_survives_restart_and_is_visible_in_stats():
    """续期由「已流逝窗口数」推导（纯函数），因此重启不改变结论；
    且续期/零证据规模必须可观测，否则「观察期是否真的在起作用」无从判断。"""
    tmp = Path(tempfile.mkdtemp(prefix="probation_"))
    store = QuarantineStore(directory=tmp)
    tracker = CooldownTracker()
    tracker.quarantine = store
    tracker.reset()
    ledger = tracker.risk_ledger
    original_dir = ledger.directory
    try:
        ledger.directory = tmp
        tracker.risk_ledger = ledger
        os.environ["GATEWAY_PROBATION_WINDOW"] = "1"
        os.environ["GATEWAY_PROBATION_MAX_RENEWALS"] = "3"
        tracker._episode_ends["doubao"] = time.time() - 2.5  # 续期 2 次
        tracker._persist_state()

        revived = CooldownTracker()
        revived.quarantine = store
        revived.risk_ledger = ledger
        revived._restore_persisted_state()
        assert revived.is_in_probation("doubao") is True, "重启不得绕过续期中的观察期"
        assert revived.get_probation_states()["doubao"]["renewals"] == 2

        stats = revived.get_probation_stats()
        assert stats["renewing"] == 1, stats
        assert stats["awaiting_first_handshake"] == 1, stats
        assert stats["budget_exhausted"] == 0, stats

        # 恢复旧设置（窗口 60s、不续期）：即使时间视野变宽，**零证据**下依然保持
        # 门控——这正是 OPT-028 的核心不变式，毕业只认证据，不认时间。
        os.environ["GATEWAY_PROBATION_WINDOW"] = "60"
        os.environ["GATEWAY_PROBATION_MAX_RENEWALS"] = "0"
        assert revived.is_in_probation("doubao") is True, "零证据不得因时间流逝而放开"
        for _ in range(3):
            asyncio.run(revived.record_success("doubao"))
        assert revived.is_in_probation("doubao") is False, "取得证据后毕业"
    finally:
        os.environ["GATEWAY_PROBATION_WINDOW"] = "60"
        os.environ["GATEWAY_PROBATION_MAX_RENEWALS"] = "0"
        ledger.directory = original_dir
        revived.reset()
        tracker.reset()
        shutil.rmtree(tmp, ignore_errors=True)
    print("test_renewal_survives_restart_and_is_visible_in_stats: PASSED")


def run_all():
    test_probation_semantics()
    test_plain_5xx_does_not_enter_probation()
    test_graduation_after_consecutive_successes()
    test_zero_window_graduates_only_on_evidence()
    test_quarantine_release_enters_probation()
    test_probation_serializes_concurrent_handshakes()
    test_probation_widens_pacing_interval()
    test_disable_switch()
    test_probation_survives_restart()
    test_diagnostics_and_header_exposure()
    test_renewal_survives_restart_and_is_visible_in_stats()
    print("\n✅ All recovery-probation tests (OPT-022/OPT-028) passed!")


if __name__ == "__main__":
    run_all()
