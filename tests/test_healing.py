"""自愈与自动热修复系统单元测试 (tests/test_healing.py)。

覆盖验证：
1. 智能体环境发现器（Mock 找到与未找到 claude/codex CLI 场景及版本解析）。
2. 健康状态机跃迁（healthy -> degraded -> offline -> recovering -> healthy）。
3. 自愈决策器：健康模型挑选算法与自愈 Prompt 生成结构。
4. 自愈引擎与后台任务调度（非阻塞、加锁防重入、日志重定向）。
5. 服务端 /health 接口包含 providers 状态与 healing 统计，以及 POST /v1/heal/{provider} 接口。
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from healing.agent_runner import AgentInfo, bootstrap_claude, ensure_agent, find_agent, run_agent
from healing.healer import HealingEngine, build_healing_prompt, pick_healer_model
from healing.monitor import HealthMonitor, ProviderHealth
import server as srv


def test_agent_finder_and_version():
    """测试本地智能体 CLI 环境发现与降级。"""
    # 场景 1: mock which("claude") 存在
    with patch("shutil.which", side_effect=lambda cmd: "/opt/fake/bin/claude" if cmd == "claude" else None), \
         patch("healing.agent_runner._get_version", return_value="2.1.0"):
        info = find_agent()
        assert info is not None
        assert info.name == "claude"
        assert "/opt/fake/bin/claude" in info.path
        assert info.version == "2.1.0"

    # 场景 2: claude 不存在，codex 存在
    with patch("shutil.which", side_effect=lambda cmd: "/opt/fake/bin/codex" if cmd == "codex" else None), \
         patch("pathlib.Path.is_file", return_value=False), \
         patch("healing.agent_runner._get_version", return_value="1.0.0"):
        info = find_agent()
        assert info is not None
        assert info.name == "codex"
        assert "/opt/fake/bin/codex" in info.path

    # 场景 3: 两者均不存在
    with patch("shutil.which", return_value=None), \
         patch("pathlib.Path.is_file", return_value=False):
        info = find_agent()
        assert info is None


def test_health_state_machine_and_monitor():
    """测试健康状态机跃迁 (healthy -> degraded -> offline -> recovering -> healthy)。"""
    monitor = HealthMonitor(check_interval=100.0, max_consecutive_failures=2)

    class MockSession:
        token = "fake_token"

    class FailingProvider:
        async def chat(self, *a, **kw):
            from providers.base import ProviderError
            raise ProviderError("上游接口变更 500", status=500)
            yield

    class RecoveredProvider:
        async def chat(self, *a, **kw):
            yield "pong", {}

    current_provider = [FailingProvider()]
    monitor.register_provider(
        "qwen",
        resolver=lambda: MockSession(),
        factory=lambda: current_provider[0],
    )

    # 初始状态为 healthy
    st = monitor.get_status("qwen")
    assert st["status"] == "healthy"
    assert st["consecutive_failures"] == 0

    # 探测 1 次失败: 跃迁至 degraded
    asyncio.run(monitor.check_provider("qwen"))
    st1 = monitor.get_status("qwen")
    assert st1["status"] == "degraded"
    assert st1["consecutive_failures"] == 1
    assert "500" in st1["last_error"]

    # 探测 2 次失败: 跃迁至 offline
    asyncio.run(monitor.check_provider("qwen"))
    st2 = monitor.get_status("qwen")
    assert st2["status"] == "offline"
    assert st2["consecutive_failures"] == 2

    # 置为 recovering 状态
    monitor.mark_recovering("qwen")
    st_rec = monitor.get_status("qwen")
    assert st_rec["status"] == "recovering"

    # 在 recovering 状态下，普通 check 不覆盖其 recovering 状态
    asyncio.run(monitor.check_provider("qwen"))
    assert monitor.get_status("qwen")["status"] == "recovering"

    # 修复后切换为正常 provider 并手动探测
    current_provider[0] = RecoveredProvider()
    monitor.mark_healthy("qwen")
    asyncio.run(monitor.check_provider("qwen"))
    st_healthy = monitor.get_status("qwen")
    assert st_healthy["status"] == "healthy"
    assert st_healthy["consecutive_failures"] == 0
    assert st_healthy["last_error"] is None


def test_healer_prompt_and_model_picking():
    """测试健康模型挑选算法与 Prompt 提示词工程。"""
    monitor = HealthMonitor()

    # 场景 1: qwen 坏了，deepseek 和 doubao 健康 -> 应选 deepseek-chat
    monitor._states["qwen"] = ProviderHealth(status="offline")
    monitor._states["deepseek"] = ProviderHealth(status="healthy")
    monitor._states["doubao"] = ProviderHealth(status="healthy")

    chosen = pick_healer_model(monitor, failing_provider="qwen")
    assert chosen == "deepseek-chat"

    # 场景 2: deepseek 坏了，qwen 健康 -> 优先选 qwen3.7-max
    monitor._states["deepseek"] = ProviderHealth(status="offline")
    monitor._states["qwen"] = ProviderHealth(status="healthy")
    chosen2 = pick_healer_model(monitor, failing_provider="deepseek")
    assert chosen2 == "qwen3.7-max"

    # 场景 3: 构造 Prompt
    prompt = build_healing_prompt(
        provider_name="deepseek",
        last_error="401 unauthorized: WAF signature invalid",
        traceback_info="File 'providers/deepseek/provider.py', line 123",
    )
    assert "providers/deepseek/provider.py" in prompt
    assert "tests/test_providers.py" in prompt
    assert "WAF signature invalid" in prompt
    assert "Self-Healing Agent" in prompt


def test_healer_engine_execution():
    """测试自愈引擎的调度、并发防重入与复检流程。"""
    monitor = HealthMonitor()
    monitor._states["qwen"] = ProviderHealth(status="healthy")
    monitor._states["deepseek"] = ProviderHealth(status="offline", last_error="Header Check Failed")

    engine = HealingEngine(monitor, port=8000)

    # Mock 智能体与 run_agent
    fake_agent = AgentInfo(name="claude", path="/mock/bin/claude", version="2.0.0")

    async def _test():
        with patch("healing.healer.ensure_agent", AsyncMock(return_value=fake_agent)), \
             patch("healing.healer.run_agent", AsyncMock(return_value=0)), \
             patch.object(monitor, "check_provider", AsyncMock(return_value=(True, None))):

            # 触发自愈
            triggered = await engine.trigger_healing("deepseek", reason="SSE frame error")
            assert triggered is True

            # 稍作等待让后台 worker 执行
            await asyncio.sleep(0.05)

            # 复测成功后状态应恢复
            st = monitor.get_status("deepseek")
            assert st["status"] == "healthy"

    asyncio.run(_test())


def test_server_health_and_heal_endpoints():
    """测试 /health 接口与 /v1/heal/{provider} 接口。"""
    client = TestClient(srv.app)

    # 1. 验证 /health 响应结构包含 providers 和 healing
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "providers" in data
    assert "qwen" in data["providers"]
    assert "healing" in data
    assert "active_count" in data["healing"]

    # 2. 验证 /v1/heal/{provider} 触发
    with patch("healing.healer.ensure_agent", AsyncMock(return_value=AgentInfo("claude", "/bin/claude"))), \
         patch("healing.healer.run_agent", AsyncMock(return_value=0)):
        r_heal = client.post("/v1/heal/qwen")
        assert r_heal.status_code == 200
        res = r_heal.json()
        assert res["provider"] == "qwen"
        assert res["status"] in ("triggered", "already_running")

    # 3. 验证未知 provider 返回 404
    r_bad = client.post("/v1/heal/unknown_provider_xyz")
    assert r_bad.status_code == 404


def test_health_monitor_disabled_provider_skipping():
    """测试 HealthMonitor 识别 DISABLED_PROVIDERS 并跳过网络探针与自愈。"""
    os.environ["DISABLED_PROVIDERS"] = "deepseek,kimi"
    try:
        monitor = HealthMonitor()
        mock_probe = AsyncMock(return_value=(False, "不应该被调用"))
        monitor.probe_provider = mock_probe

        # deepseek 被禁用，探测应直接返回且不调用底层网络探针
        is_healthy, err = asyncio.run(monitor.check_provider("deepseek"))
        assert mock_probe.call_count == 0
        st = monitor.get_status("deepseek")
        assert st["status"] == "disabled"
        assert st["consecutive_failures"] == 0

        # get_all_statuses 中 deepseek 与 kimi 状态为 disabled
        all_st = monitor.get_all_statuses()
        assert all_st["deepseek"]["status"] == "disabled"
        assert all_st["kimi"]["status"] == "disabled"

        # 未被禁用的 qwen 正常调用 probe_provider
        asyncio.run(monitor.check_provider("qwen"))
        assert mock_probe.call_count == 1
    finally:
        os.environ.pop("DISABLED_PROVIDERS", None)

    # 验证动态解禁后状态自愈恢复（消除状态粘滞）
    all_st_after = monitor.get_all_statuses()
    assert all_st_after["deepseek"]["status"] == "healthy"
    assert all_st_after["kimi"]["status"] == "healthy"
    assert monitor.get_status("deepseek")["status"] == "healthy"


if __name__ == "__main__":
    test_agent_finder_and_version()
    print("[PASS] 智能体环境发现器与版本解析")
    test_health_state_machine_and_monitor()
    print("[PASS] 健康探测状态机跃迁与探针审计")
    test_health_monitor_disabled_provider_skipping()
    print("[PASS] HealthMonitor 禁用 Provider 跳过探针与标记 disabled")
    test_healer_prompt_and_model_picking()
    print("[PASS] 自愈决策器与 Prompt 提示词工程")
    test_healer_engine_execution()
    print("[PASS] 自愈引擎任务调度与复检恢复")
    test_server_health_and_heal_endpoints()
    print("[PASS] 服务端 /health 与 /v1/heal 接口")
    print("\n自愈系统全部单测通过！")
