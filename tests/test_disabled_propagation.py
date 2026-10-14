"""运维禁用贯穿全部上游自维护触点 (OPT-034) 测试套件

动机是一条实测出来的漏口：`DISABLED_PROVIDERS` 是运维「这个节点不属于本次部署」的显式
声明，派发边界（OPT-033）与健康探针（OPT-012）都遵守它——**但另外三条自维护通道只问了
冷却、没问禁用**。于是一个被声明出局、却恰好没有冷却的节点，网关仍会对它发起真实的上游
请求：

  - 启动静默令牌刷新 → 一次真实的上游登录请求
  - 匿名调用的临时会话删除 → 一次真实的删除请求
  - 自愈大脑选举 → 一次真实的上游补全

实测复现（修复前）：`DISABLED_PROVIDERS=kimi` 且无冷却时，
`_refresh_stale_provider("kimi", …)` 真的调用了 refresher，打印
「kimi 令牌已超过 6h，尝试静默刷新…」。运维已宣布不用的账号仍在被持续触碰——这正是
本项目「绝不触碰被封禁/已下线节点」这条约束针对的失效模式。

根因是**同一条避让声明只在部分通道生效**，与 OPT-020（租约折入冷却读数使所有门控自动
遵循）同属一类。修复的落点是「一条声明，一个判据」：新增
`CooldownTracker.touch_avoidance_reason()` 作为后台自维护触点的**唯一**判据
（运维禁用 ∪ 运维/自动租约 ∪ 冷却，heaviest-wins），四条通道全部向它求值。

覆盖：
1. 令牌刷新：被禁用节点不做刷新（复现并锁死原始漏口，零上游调用）
2. 令牌刷新：被禁用**不进入延迟重试**（与冷却的区别：等的是人改配置，不是等窗口到期）
3. 会话回收：被禁用节点的临时会话删除被推迟，零删除请求
4. 自愈选举：被禁用节点绝不被选为自愈大脑；候选耗尽时安全推迟（返回 None）
5. 探针：判据统一——直接改 tracker 集合（不经环境变量）也必须让探针跳过
6. 成因判读只有一份且如实：disabled > quarantined > cooldown，健康节点为空
7. 零回归：健康节点照常刷新/删除/被选举；冷却与租约仍按既有语义推迟
8. 冷却是「暂时」而禁用是「声明」：冷却受冷却总开关约束，禁用不受

运行: .venv/bin/python tests/test_disabled_propagation.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["TESTING"] = "1"
os.environ["GATEWAY_DISABLE_AUTO_REFRESH"] = "1"
os.environ["GATEWAY_DISABLE_PACING"] = "1"
os.environ["GATEWAY_DISABLE_SINGLEFLIGHT"] = "1"
for _name in (
    "DISABLED_PROVIDERS",
    "GATEWAY_QUARANTINE",
    "GATEWAY_DISABLE_COOLDOWN_GATE",
    "GATEWAY_DISABLE_PROBE_COOLDOWN_GATE",
    "GATEWAY_DISABLE_CONVERSATION_GC_COOLDOWN_GATE",
    "GATEWAY_DISABLE_BAN_INFERENCE",
    "GATEWAY_DISABLE_RISK_LEDGER",
):
    os.environ.pop(_name, None)
os.environ["GATEWAY_PROBATION_MAX_RENEWALS"] = "0"
os.environ["QWEN_AUTH_TOKEN"] = "fake_qwen"
os.environ["DEEPSEEK_AUTH_TOKEN"] = "fake_ds"
os.environ["DOUBAO_AUTH_TOKEN"] = "fake_doubao"
os.environ["KIMI_AUTH_TOKEN"] = "fake_kimi"
os.environ["GLM_AUTH_TOKEN"] = "fake_glm"

from healing.healer import pick_healer_model  # noqa: E402
from healing.monitor import HealthMonitor  # noqa: E402

import server as srv  # noqa: E402
from providers.conversation_gc import ConversationGC  # noqa: E402


class _MockSession:
    token = "fake_token"
    age = 10 ** 9  # 远超 MAX_AGE，必然进入刷新分支


def _run(coro):
    return asyncio.run(coro)


def _reset_tracker():
    tracker = srv.failover_router.cooldown_tracker
    tracker.reset()
    srv.quarantine_store.clear_all_for_tests()
    try:
        tracker.credential_watch.clear_all_for_tests()
    except Exception:  # noqa: BLE001
        pass
    return tracker, set(tracker.disabled_providers)


def _restore(tracker, snapshot):
    tracker.disabled_providers = set(snapshot)
    tracker.reset()
    srv.quarantine_store.clear_all_for_tests()


# ------------------------------------------------- 1. 令牌刷新：被禁用不刷新

def test_disabled_provider_is_not_refreshed():
    """**本套件的直接靶点**：修复前这里真的发出了上游刷新请求。"""
    tracker, snap = _reset_tracker()
    try:
        tracker.disabled_providers = {"kimi"}
        called: list = []

        def _refresher(timeout=60):
            called.append(timeout)
            return None

        verdict = srv._refresh_stale_provider("kimi", lambda: _MockSession(), _refresher)
        assert verdict == "disabled", f"被禁用节点应报 disabled，实际: {verdict}"
        assert called == [], f"被运维禁用的节点绝不能被刷新，实际发起了 {len(called)} 次上游请求"
    finally:
        _restore(tracker, snap)
    print("test_disabled_provider_is_not_refreshed: PASSED")


def test_disabled_is_not_deferred_because_there_is_nothing_to_wait_for():
    """禁用与冷却的区别在**处置方式**：冷却是在等一个会到期的窗口（推迟重试），
    禁用是在等运维改配置（没有可等期限）。若被误判为 cooling，就会被加入最多 24h 的
    延迟重试队列，每隔 poll 秒再问一次——对禁用节点而言纯属浪费，且一旦运维中途改回，
    反而在一个未经验证的时刻突然发起登录请求。"""
    tracker, snap = _reset_tracker()
    try:
        tracker.disabled_providers = {"kimi"}
        verdict = srv._refresh_stale_provider("kimi", lambda: _MockSession(), lambda **kw: None)
        assert verdict != "cooling", "禁用必须与冷却可区分，否则会被误纳入延迟重试队列"

        # 对照：冷却中的节点返回 cooling（既有语义不变）
        _run(tracker.record_failure("glm", 429, "rate limit exceeded"))
        verdict_cd = srv._refresh_stale_provider("glm", lambda: _MockSession(), lambda **kw: None)
        assert verdict_cd == "cooling", f"冷却节点仍须暂缓并重试，实际: {verdict_cd}"
    finally:
        _restore(tracker, snap)
    print("test_disabled_is_not_deferred_because_there_is_nothing_to_wait_for: PASSED")


# ------------------------------------------------- 2. 会话回收：被禁用不删除

def test_disabled_provider_conversation_is_not_deleted():
    tracker, snap = _reset_tracker()
    gc = ConversationGC()
    deleted: list = []
    resolved: list = []

    class _P:
        async def delete_conversation(self, cid):
            deleted.append(cid)
            return True

    def _resolver(pk):
        resolved.append(pk)
        return _P()

    try:
        tracker.disabled_providers = {"kimi"}
        gc.set_resolver(_resolver)
        gc.set_avoidance_gate(tracker.touch_avoidance_reason)

        _run(gc.drain_once())  # 队列尚未有记录：仅为确保无副作用
        gc.enqueue("kimi", "conv-1")
        _run(gc.drain_once())

        assert deleted == [], f"被禁用节点的临时会话绝不能被发起删除，实际: {deleted}"
        assert resolved == [], "被禁用节点甚至不该被构造（构造即触碰的前提）"
        diag = gc.get_diagnostics()
        assert diag["avoidance_gate_enabled"] is True
        assert diag["deferred_by_cooldown"]["per_cause"].get("disabled", 0) == 1, diag
        assert diag["pending_count"] == 1, "记录必须留在队列等待禁用解除，而不是被丢弃"
    finally:
        _restore(tracker, snap)
        gc.clear()
    print("test_disabled_provider_conversation_is_not_deleted: PASSED")


# ------------------------------------------------- 3. 自愈选举：被禁用不入选

def _make_monitor() -> HealthMonitor:
    monitor = HealthMonitor(check_interval=100.0, max_consecutive_failures=1)
    for pk in ("qwen", "deepseek", "doubao", "kimi", "glm"):
        monitor.register_provider(
            pk, resolver=lambda: _MockSession(), factory=lambda: None
        )
    return monitor


def test_disabled_provider_is_never_elected_as_healer():
    tracker, snap = _reset_tracker()
    try:
        monitor = _make_monitor()
        monitor.set_disabled_gate(tracker.is_disabled)
        monitor.set_touch_avoidance_gate(tracker.touch_avoidance_reason)

        # 只禁用首选（qwen），次选应顶上——且绝不能是 qwen
        tracker.disabled_providers = {"qwen"}
        chosen = pick_healer_model(monitor, "glm")
        assert chosen is not None, "仍有可用候选时应选出备选，而不是直接放弃"
        assert chosen != "qwen3.7-max", f"被运维禁用的节点绝不能当选自愈大脑: {chosen}"
        assert monitor.get_healer_skip_stats()["per_provider"].get("qwen", 0) >= 1

        # 除失败节点外全部禁用 → 必须安全推迟（None），而不是触碰任何被禁用节点
        monitor2 = _make_monitor()
        monitor2.set_disabled_gate(tracker.is_disabled)
        monitor2.set_touch_avoidance_gate(tracker.touch_avoidance_reason)
        tracker.disabled_providers = {"qwen", "deepseek", "doubao", "kimi"}
        assert pick_healer_model(monitor2, "glm") is None, "候选耗尽时必须推迟自愈而非触碰禁用节点"
    finally:
        _restore(tracker, snap)
    print("test_disabled_provider_is_never_elected_as_healer: PASSED")


# ------------------------------------------------- 4. 探针判据统一

def test_probe_skips_disabled_set_without_env_var():
    """判读只有一份：直接改 tracker 集合（**不动环境变量**）也必须让探针跳过。

    修复前健康监控自行现读 DISABLED_PROVIDERS，与 CooldownTracker 的集合是同一份配置的
    两份解读——进程内改集合时探针照样会触碰被禁用节点。
    """
    tracker, snap = _reset_tracker()
    try:
        tracker.disabled_providers = {"qwen"}
        assert os.getenv("DISABLED_PROVIDERS", "") == "", "本用例刻意不经环境变量"

        monitor = _make_monitor()
        monitor.set_disabled_gate(tracker.is_disabled)
        monitor.set_touch_avoidance_gate(tracker.touch_avoidance_reason)
        probe = AsyncMock(return_value=(True, None))
        monitor.probe_provider = probe

        is_healthy, err = _run(monitor.check_provider("qwen"))
        assert probe.call_count == 0, "被禁用节点绝不能发起真实探针"
        assert is_healthy is False
        assert "已被禁用" in err
        assert monitor.get_status("qwen")["status"] == "disabled"
        assert monitor.touch_avoidance_reason("qwen") == "disabled"
    finally:
        _restore(tracker, snap)
    print("test_probe_skips_disabled_set_without_env_var: PASSED")


# ------------------------------------------------- 5. 成因判读只有一份且如实

def test_avoidance_cause_resolution_is_ordered_and_honest():
    tracker, snap = _reset_tracker()
    try:
        # 健康节点：无成因
        assert tracker.touch_avoidance_reason("qwen") == ""

        # 自动冷却 → cooldown
        _run(tracker.record_failure("glm", 429, "rate limit exceeded"))
        assert tracker.touch_avoidance_reason("glm") == "cooldown"

        # 运维租约 → quarantined（且优先于冷却）
        srv.quarantine_store.set_lease("glm", time.time() + 3600, source="operator")
        assert tracker.touch_avoidance_reason("glm") == "quarantined"

        # 运维禁用 → disabled（最高优先级，压过租约与冷却）
        tracker.disabled_providers = {"glm"}
        assert tracker.touch_avoidance_reason("glm") == "disabled"

        report = tracker.touch_gate_report()
        assert report["blocked_by_cause"].get("disabled", 0) >= 1
        assert report["blocked_by_cause"].get("quarantined", 0) >= 1
        assert report["blocked_by_cause"].get("cooldown", 0) >= 1
    finally:
        _restore(tracker, snap)
    print("test_avoidance_cause_resolution_is_ordered_and_honest: PASSED")


# ------------------------------------------------- 6. 零回归

def test_healthy_provider_is_still_refreshed_deleted_and_elected():
    tracker, snap = _reset_tracker()
    try:
        tracker.disabled_providers = {"deepseek"}
        # 刷新：健康节点照常刷新（refresher 返回新会话即为成功）
        assert (
            srv._refresh_stale_provider("qwen", lambda: _MockSession(), lambda **kw: _MockSession())
            == "refreshed"
        ), "健康节点的刷新绝不能被新门控误伤"

        # 删除：健康节点照常删除
        gc = ConversationGC()
        deleted: list = []

        class _P:
            async def delete_conversation(self, cid):
                deleted.append(cid)
                return True

        gc.set_resolver(lambda pk: _P())
        gc.set_avoidance_gate(tracker.touch_avoidance_reason)
        gc.enqueue("qwen", "conv-ok")
        _run(gc.drain_once())
        assert deleted == ["conv-ok"], f"健康节点的删除不得被误伤: {deleted}"

        # 选举：健康节点照常当选
        monitor = _make_monitor()
        monitor.set_disabled_gate(tracker.is_disabled)
        monitor.set_touch_avoidance_gate(tracker.touch_avoidance_reason)
        assert pick_healer_model(monitor, "glm") == "qwen3.7-max"
    finally:
        _restore(tracker, snap)
    print("test_healthy_provider_is_still_refreshed_deleted_and_elected: PASSED")


# ------------------------------------------------- 7. 冷却受总开关约束，禁用不受

def test_cooldown_master_switch_does_not_release_a_disabled_node():
    """`GATEWAY_DISABLE_COOLDOWN_GATE=1` 的语义是「回退为无冷却感知」，
    而禁用是部署声明——一个开关不得悄悄放开一个已被声明出局的节点。"""
    tracker, snap = _reset_tracker()
    try:
        os.environ["GATEWAY_DISABLE_COOLDOWN_GATE"] = "1"
        try:
            tracker.disabled_providers = {"kimi"}
            monitor = _make_monitor()
            monitor.set_disabled_gate(tracker.is_disabled)
            monitor.set_touch_avoidance_gate(tracker.touch_avoidance_reason)

            assert monitor.touch_avoidance_reason("kimi") == "disabled", (
                "冷却总开关关闭时，禁用判定必须依然成立"
            )
            probe = AsyncMock(return_value=(True, None))
            monitor.probe_provider = probe
            _run(monitor.check_provider("kimi"))
            assert probe.call_count == 0, "总开关关闭也不得让探针触碰被禁用节点"

            # 对照：同一总开关下，单纯的冷却确实不再阻断（既有语义保持不变）
            _run(tracker.record_failure("glm", 429, "rate limit exceeded"))
            assert monitor.touch_avoidance_reason("glm") == "", (
                "总开关关闭时冷却感知应整体回退"
            )
        finally:
            os.environ.pop("GATEWAY_DISABLE_COOLDOWN_GATE", None)
    finally:
        _restore(tracker, snap)
    print("test_cooldown_master_switch_does_not_release_a_disabled_node: PASSED")


# ------------------------------------------------- 8. 与派发闸门同源

def test_dispatch_and_background_share_one_disabled_judgement():
    """派发闸门与后台触点判据必须对「谁被禁用」给出一致答案（同一份判读者）。"""
    tracker, snap = _reset_tracker()
    try:
        tracker.disabled_providers = {"doubao"}
        router = srv.failover_router
        gate = router.evaluate_dispatch_gate("doubao")
        assert gate["blocked_by"] == "disabled"
        assert tracker.touch_avoidance_reason("doubao") == "disabled"
        for pk in ("qwen", "kimi", "glm", "deepseek"):
            assert router.evaluate_dispatch_gate(pk)["blocked_by"] != "disabled"
            assert tracker.touch_avoidance_reason(pk) != "disabled"
    finally:
        _restore(tracker, snap)
    print("test_dispatch_and_background_share_one_disabled_judgement: PASSED")


def run_all():
    tests = [
        test_disabled_provider_is_not_refreshed,
        test_disabled_is_not_deferred_because_there_is_nothing_to_wait_for,
        test_disabled_provider_conversation_is_not_deleted,
        test_disabled_provider_is_never_elected_as_healer,
        test_probe_skips_disabled_set_without_env_var,
        test_avoidance_cause_resolution_is_ordered_and_honest,
        test_healthy_provider_is_still_refreshed_deleted_and_elected,
        test_cooldown_master_switch_does_not_release_a_disabled_node,
        test_dispatch_and_background_share_one_disabled_judgement,
    ]
    for fn in tests:
        fn()
    print(f"\n✅ All disabled-propagation tests (OPT-034) passed! ({len(tests)} cases)")


if __name__ == "__main__":
    run_all()
