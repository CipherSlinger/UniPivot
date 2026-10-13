"""风控感知健康探针门控 (OPT-012) 测试套件

验证：
1. 处于风控避让冷却期的节点，周期探针跳过真实网络调用（不再触碰已封禁节点）
2. 冷却结束（剩余 0）后探针恢复正常执行
3. 冷却门控期间既不累计失败计数、也不触发自愈（杜绝风控噪声掩盖为「节点故障」）
4. GATEWAY_DISABLE_PROBE_COOLDOWN_GATE=1 可关闭门控（退化为历史行为）
5. 探针跳过统计 (get_probe_skip_stats) 按节点累计
6. 未注册门控时探针行为与历史完全一致（零行为回退）
7. 服务端已将 CooldownTracker 接入门控，且 /v1/diagnostics 暴露跳过统计

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


def run_all():
    test_cooling_node_skips_real_probe()
    test_probe_runs_when_not_cooling()
    test_cooling_does_not_trigger_healing()
    test_disable_gate_env()
    test_probe_skip_stats_counters()
    test_no_gate_registered_probes_normally()
    test_server_gate_wired_and_diagnostics()
    print("\n✅ All cooldown-aware health probe gate tests (OPT-012) passed!")


if __name__ == "__main__":
    run_all()
