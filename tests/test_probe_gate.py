"""风控感知健康探针门控 (OPT-012) 测试套件

验证：
1. 处于风控避让冷却期的节点，周期探针跳过真实网络调用（不再触碰已封禁节点）
2. 冷却结束（剩余 0）后探针恢复正常执行
3. 冷却门控期间既不累计失败计数、也不触发自愈（杜绝风控噪声掩盖为「节点故障」）
4. GATEWAY_DISABLE_PROBE_COOLDOWN_GATE=1 可关闭门控（退化为历史行为）
5. 探针跳过统计 (get_probe_skip_stats) 按节点累计
6. 未注册门控时探针行为与历史完全一致（零行为回退）
7. 服务端已将 CooldownTracker 接入门控，且 /v1/diagnostics 暴露跳过统计
8. [OPT-018] 自愈大脑选举的风控冷却避让：冷却节点绝不作为修复引擎被触碰
9. [OPT-018] 读取时冷却状态叠加：未探测的冷却节点不再被误判为 healthy
10. [OPT-018] 独立开关 GATEWAY_DISABLE_HEALER_COOLDOWN_GATE 与门控异常零阻塞

运行: .venv/bin/python tests/test_probe_gate.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock

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
from healing.healer import pick_healer_model  # noqa: E402
from healing.monitor import HealthMonitor  # noqa: E402


class _MockSession:
    token = "fake_token"


class _HealthyProvider:
    async def chat(self, *a, **kw):
        yield "pong", {}


def _make_monitor() -> HealthMonitor:
    monitor = HealthMonitor(check_interval=100.0, max_consecutive_failures=1)
    monitor.register_provider(
        "qwen",
        resolver=lambda: _MockSession(),
        factory=lambda: _HealthyProvider(),
    )
    return monitor


def test_cooling_node_skips_real_probe():
    """冷却期内的节点绝不发起真实探针。"""
    monitor = _make_monitor()
    monitor.set_cooldown_probe_gate(lambda name: 120.0 if name == "qwen" else 0.0)
    probe = AsyncMock(return_value=(True, None))
    monitor.probe_provider = probe

    is_healthy, err = asyncio.run(monitor.check_provider("qwen"))
    assert probe.call_count == 0, "冷却期内绝不能发起真实网络探针"
    assert is_healthy is False
    assert "冷却" in err
    st = monitor.get_status("qwen")
    assert st["status"] == "cooling"
    assert st["cooldown_remaining"] == 120.0
    assert st["consecutive_failures"] == 0
    print("test_cooling_node_skips_real_probe: PASSED")


def test_probe_runs_when_not_cooling():
    """冷却结束后探针恢复正常执行。"""
    monitor = _make_monitor()
    remaining = {"qwen": 30.0}
    monitor.set_cooldown_probe_gate(lambda name: remaining.get(name, 0.0))
    probe = AsyncMock(return_value=(True, None))
    monitor.probe_provider = probe

    asyncio.run(monitor.check_provider("qwen"))
    assert probe.call_count == 0
    assert monitor.get_status("qwen")["status"] == "cooling"

    remaining["qwen"] = 0.0  # 冷却结束
    asyncio.run(monitor.check_provider("qwen"))
    assert probe.call_count == 1, "冷却结束后必须恢复正常探针"
    st = monitor.get_status("qwen")
    assert st["status"] == "healthy"
    assert st["cooldown_remaining"] == 0.0
    print("test_probe_runs_when_not_cooling: PASSED")


def test_cooling_does_not_trigger_healing():
    """冷却门控期间绝不累计失败计数、也不触发自愈。"""
    monitor = _make_monitor()  # max_consecutive_failures=1 → 单次失败即触发
    triggered = {"count": 0}

    async def _trigger(name, err):
        triggered["count"] += 1

    monitor.set_healer_trigger(_trigger)
    monitor.set_cooldown_probe_gate(lambda name: 300.0)
    monitor.probe_provider = AsyncMock(return_value=(False, "不应被调用"))

    async def _run_many():
        for _ in range(3):
            await monitor.check_provider("qwen")

    asyncio.run(_run_many())
    assert triggered["count"] == 0, "冷却期绝不应触发自愈"
    assert monitor.get_status("qwen")["consecutive_failures"] == 0
    print("test_cooling_does_not_trigger_healing: PASSED")


def test_disable_gate_env():
    """GATEWAY_DISABLE_PROBE_COOLDOWN_GATE=1 关闭门控。"""
    monitor = _make_monitor()
    monitor.set_cooldown_probe_gate(lambda name: 600.0)
    probe = AsyncMock(return_value=(True, None))
    monitor.probe_provider = probe

    os.environ["GATEWAY_DISABLE_PROBE_COOLDOWN_GATE"] = "1"
    try:
        asyncio.run(monitor.check_provider("qwen"))
        assert probe.call_count == 1, "关闭门控后应正常探针"
    finally:
        os.environ.pop("GATEWAY_DISABLE_PROBE_COOLDOWN_GATE", None)
    print("test_disable_gate_env: PASSED")


def test_probe_skip_stats_counters():
    """探针跳过统计按节点累计。"""
    monitor = _make_monitor()
    monitor.set_cooldown_probe_gate(lambda name: 45.0)
    monitor.probe_provider = AsyncMock(return_value=(True, None))

    async def _run_many():
        for _ in range(4):
            await monitor.check_provider("qwen")

    asyncio.run(_run_many())
    stats = monitor.get_probe_skip_stats()
    assert stats["total"] == 4
    assert stats["per_provider"]["qwen"] == 4
    print("test_probe_skip_stats_counters: PASSED")


def test_no_gate_registered_probes_normally():
    """未注册门控时行为与历史完全一致（零行为回退）。"""
    monitor = _make_monitor()
    probe = AsyncMock(return_value=(True, None))
    monitor.probe_provider = probe
    monitor.set_cooldown_probe_gate(lambda name: 9999.0)  # 注册了…

    # 但显式清空回调，模拟未接入场景
    monitor._cooldown_probe_gate = None
    asyncio.run(monitor.check_provider("qwen"))
    assert probe.call_count == 1
    assert monitor.get_status("qwen")["status"] == "healthy"
    print("test_no_gate_registered_probes_normally: PASSED")


def test_server_gate_wired_and_diagnostics():
    """服务端已将 CooldownTracker 接入门控，并暴露跳过统计。"""
    assert srv.health_monitor._cooldown_probe_gate is not None, "服务端必须注册风控冷却门控"

    tracker = srv.failover_router.cooldown_tracker
    monitor = srv.health_monitor
    probe = AsyncMock(return_value=(True, None))
    original_probe = monitor.probe_provider
    monitor.probe_provider = probe
    try:
        # 将 qwen 打入冷却避让期
        asyncio.run(tracker.record_failure("qwen", 429, "rate limit exceeded"))
        assert tracker.get_remaining_cooldown("qwen") > 0

        skips_before = monitor.get_probe_skip_stats()["total"]
        is_healthy, err = asyncio.run(monitor.check_provider("qwen"))
        assert probe.call_count == 0, "服务端门控必须拦截处于冷却期的节点探针"
        assert is_healthy is False
        assert monitor.get_status("qwen")["status"] == "cooling"
        assert monitor.get_probe_skip_stats()["total"] == skips_before + 1

        # 冷却解除后探针恢复
        asyncio.run(tracker.record_success("qwen"))
        asyncio.run(monitor.check_provider("qwen"))
        assert probe.call_count == 1
    finally:
        monitor.probe_provider = original_probe
        asyncio.run(tracker.record_success("qwen"))

    # /v1/diagnostics 暴露冷却跳过统计
    client = TestClient(srv.app)
    r = client.get("/v1/diagnostics")
    assert r.status_code == 200
    healing = r.json()["healing_status"]
    assert "cooldown_probe_skips" in healing
    assert "per_provider" in healing["cooldown_probe_skips"]
    print("test_server_gate_wired_and_diagnostics: PASSED")


# ---------------------------------------------------------------- [OPT-018] 自愈大脑选举冷却避让

def test_healer_elects_healthy_baseline():
    """无冷却时选举行为与历史一致（零行为回退）。"""
    monitor = _make_monitor()
    # 未注册门控 → 不应有任何跳过
    assert pick_healer_model(monitor, "glm") == "qwen3.7-max"
    assert pick_healer_model(monitor, "qwen") == "deepseek-chat"
    assert monitor.get_healer_skip_stats()["total"] == 0
    print("test_healer_elects_healthy_baseline: PASSED")


def test_healer_skips_cooling_node():
    """处于冷却期的首选节点必须被跳过，选举落到下一个健康节点。"""
    monitor = _make_monitor()
    monitor.set_cooldown_probe_gate(lambda name: 200.0 if name == "qwen" else 0.0)

    picked = pick_healer_model(monitor, "glm")
    assert picked == "deepseek-chat", f"冷却的 qwen 必须被跳过，实际选中 {picked}"
    stats = monitor.get_healer_skip_stats()
    assert stats["total"] == 1
    assert stats["per_provider"]["qwen"] == 1
    print("test_healer_skips_cooling_node: PASSED")


def test_healer_defers_when_all_candidates_cooling():
    """所有候选均处于冷却期时返回 None（安全推迟自愈，而非触碰封禁节点）。"""
    monitor = _make_monitor()
    monitor.set_cooldown_probe_gate(lambda name: 600.0)

    assert pick_healer_model(monitor, "glm") is None, "无可用候选时必须推迟而非强行触碰"
    assert monitor.get_healer_skip_stats()["total"] == 4, "除 failing 外的 4 个候选均被跳过"
    print("test_healer_defers_when_all_candidates_cooling: PASSED")


def test_healer_resumes_after_cooldown_clears():
    """冷却解除后节点重新进入候选。"""
    monitor = _make_monitor()
    remaining = {"qwen": 120.0}
    monitor.set_cooldown_probe_gate(lambda name: remaining.get(name, 0.0))

    assert pick_healer_model(monitor, "glm") == "deepseek-chat"
    remaining["qwen"] = 0.0  # 冷却解除
    assert pick_healer_model(monitor, "glm") == "qwen3.7-max", "冷却解除后首选节点应恢复"
    print("test_healer_resumes_after_cooldown_clears: PASSED")


def test_healer_gate_disable_switch():
    """GATEWAY_DISABLE_COOLDOWN_GATE=1 时选举不避让（可回退）。"""
    monitor = _make_monitor()
    monitor.set_cooldown_probe_gate(lambda name: 300.0)
    os.environ["GATEWAY_DISABLE_COOLDOWN_GATE"] = "1"
    try:
        assert pick_healer_model(monitor, "glm") == "qwen3.7-max", "关闭开关后应恢复历史选举行为"
        assert monitor.get_healer_skip_stats()["total"] == 0
    finally:
        os.environ.pop("GATEWAY_DISABLE_COOLDOWN_GATE", None)
    print("test_healer_gate_disable_switch: PASSED")


def test_healer_gate_exception_never_blocks():
    """门控查询异常时按「无冷却」处理，绝不让自愈整体失效。"""
    monitor = _make_monitor()

    def _boom(name):
        raise RuntimeError("gate exploded")

    monitor.set_cooldown_probe_gate(_boom)
    assert pick_healer_model(monitor, "glm") == "qwen3.7-max"
    print("test_healer_gate_exception_never_blocks: PASSED")


def test_status_overlay_reports_cooling_without_probe():
    """读取时冷却叠加：未经探针覆盖的默认 healthy 节点也必须报告 cooling。"""
    monitor = _make_monitor()
    # ProviderHealth 默认 status="healthy"，且此处从未执行任何探针
    assert monitor.get_status("kimi")["status"] == "healthy"
    monitor.set_cooldown_probe_gate(lambda name: 90.0 if name == "kimi" else 0.0)

    st = monitor.get_status("kimi")
    assert st["status"] == "cooling", "未探测的冷却节点不得继续报告 healthy"
    assert st["cooldown_remaining"] == 90.0
    all_st = monitor.get_all_statuses()
    assert all_st["kimi"]["status"] == "cooling"
    assert all_st["glm"]["status"] != "cooling", "健康节点不得被误伤"

    # 叠加是只读的，不得粘滞为底层状态
    monitor.set_cooldown_probe_gate(lambda name: 0.0)
    assert monitor.get_status("kimi")["status"] == "healthy"
    print("test_status_overlay_reports_cooling_without_probe: PASSED")


def test_server_healer_gate_wired_and_diagnostics():
    """服务端接线：真实 CooldownTracker 冷却的节点不会被选为自愈大脑。"""
    monitor = srv.health_monitor
    tracker = srv.failover_router.cooldown_tracker
    try:
        # deepseek 是 qwen 之后的第二优先候选：将其打入冷却
        asyncio.run(tracker.record_failure("deepseek", 429, "rate limit exceeded"))
        assert tracker.get_remaining_cooldown("deepseek") > 0

        picked = pick_healer_model(monitor, "qwen")
        assert picked != "deepseek-chat", "风控冷却中的节点绝不能被选为自愈修复引擎"
        assert picked is not None
    finally:
        asyncio.run(tracker.record_success("deepseek"))

    client = TestClient(srv.app)
    healing = client.get("/v1/diagnostics").json()["healing_status"]
    assert "cooldown_healer_skips" in healing
    assert "per_provider" in healing["cooldown_healer_skips"]
    print("test_server_healer_gate_wired_and_diagnostics: PASSED")


def run_all():
    test_cooling_node_skips_real_probe()
    test_probe_runs_when_not_cooling()
    test_cooling_does_not_trigger_healing()
    test_disable_gate_env()
    test_probe_skip_stats_counters()
    test_no_gate_registered_probes_normally()
    test_server_gate_wired_and_diagnostics()
    test_healer_elects_healthy_baseline()
    test_healer_skips_cooling_node()
    test_healer_defers_when_all_candidates_cooling()
    test_healer_resumes_after_cooldown_clears()
    test_healer_gate_disable_switch()
    test_healer_gate_exception_never_blocks()
    test_status_overlay_reports_cooling_without_probe()
    test_server_healer_gate_wired_and_diagnostics()
    print("\n✅ All cooldown-aware health probe gate tests (OPT-012 / OPT-018) passed!")


if __name__ == "__main__":
    run_all()
