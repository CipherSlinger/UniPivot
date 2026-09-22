"""对等互备网格与风险/速率配额步调测试 (tests/test_failover.py)。

覆盖内容:
- Test 1: classify_capability_ring:
  - 深度思考推理 (REASONING): deepseek-reasoner, qwen3.7-max (think), kimi-explore, doubao-think, glm-zero-preview
  - 旗舰全能 (FLAGSHIP): qwen3.7-max, kimi, doubao-pro, glm-4-plus, deepseek-chat
  - 极速经济 (SPEED): qwen3.6-flash, doubao-lite, glm-4-flash
- Test 2: ProviderRiskMeta & CooldownTracker:
  - 五大模型风险级别与限流参数 (qwen, deepseek, kimi, doubao, glm)
  - record_failure:
    - HTTP 429 或 "人机验证" / "RGV587" -> 加权指数冷却 (>= 45s)
    - HTTP 401 (auth) -> 认证避让冷却 (>= 60s)
    - HTTP 502/network -> 比例冷却 (min(base_cd, 15s * cur_fails))
  - is_in_cooldown & get_remaining_cooldown
  - record_success 重置失败计数并清除冷却状态
  - apply_pacing_and_acquire 遵循 min_call_interval 与 jitter 随机抖动
  - get_semaphore 节点并发控制
- Test 3: FailoverRouter.get_fallback_candidates:
  - 主选 REASONING 环模型 (如 deepseek-reasoner)，候选排除主选并优先非冷却、低风险节点 (LOW > MEDIUM > HIGH)
  - 节点进入冷却避让后自动后置到列表末尾
- Test 4: FailoverRouter.execute_chat (Mocked):
  - Scenario A: 主选节点首包失败 (401 或 429) -> 透明平滑转移到同能力环次级节点 -> 流式成功输出 -> FailoverEvent 精确填充
  - Scenario B: 能力环内所有候选节点均故障 -> 抛出 ProviderError 复合异常报告
  - Scenario C: 主选节点首次调用成功 -> FailoverEvent 为 None
- Test 5: Server 端点集成测试:
  - GET /health 上报各 Provider 风险级别、冷却状态与剩余秒数
  - /v1/chat/completions 与 /v1/messages 故障转移元数据及标头集成

运行方式:
  .venv/bin/python tests/test_failover.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import AsyncIterator, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 禁用自动化后台探测与外部依赖，注入假 Token
os.environ["GATEWAY_DISABLE_AUTO_REFRESH"] = "1"
os.environ["QWEN_AUTH_TOKEN"] = "fake_qwen"
os.environ["DEEPSEEK_AUTH_TOKEN"] = "fake_ds"
os.environ["DOUBAO_AUTH_TOKEN"] = "fake_doubao"
os.environ["KIMI_AUTH_TOKEN"] = "fake_kimi"
os.environ["GLM_AUTH_TOKEN"] = "fake_glm"

from fastapi.testclient import TestClient  # noqa: E402

import server as srv  # noqa: E402
from providers.base import ProviderError  # noqa: E402
from providers.failover import (  # noqa: E402
    CapabilityRing,
    CooldownTracker,
    FailoverCandidate,
    FailoverEvent,
    FailoverRouter,
    PROVIDER_RISK_SPECS,
    ProviderRiskMeta,
    RING_DEFINITIONS,
    RiskLevel,
    classify_capability_ring,
)


# ---------------------------------------------------------------- Mock Providers

class MockSuccessStreamProvider:
    """模拟返回成功流式输出的 Provider"""

    def __init__(self, name: str, response_text: str = "默认成功响应"):
        self.name = name
        self.response_text = response_text

    async def chat(self, *args, **kwargs) -> AsyncIterator[Tuple[str, dict]]:
        yield self.response_text, {"provider": self.name}
        yield "", {"conversation_id": f"conv_{self.name}_999"}

    async def delete_conversation(self, conversation_id: str) -> bool:
        return True


class MockErrorProvider:
    """���拟在握手或首包即刻抛出 ProviderError 的 Provider"""

    def __init__(
        self,
        name: str,
        status: int = 502,
        err_msg: str = "上游接口响应异常",
        err_type: str = "upstream_error",
    ):
        self.name = name
        self.status = status
        self.err_msg = err_msg
        self.err_type = err_type

    async def chat(self, *args, **kwargs) -> AsyncIterator[Tuple[str, dict]]:
        raise ProviderError(f"[{self.name}] {self.err_msg}", status=self.status, err_type=self.err_type)
        yield  # 保留 async generator 结构


# ---------------------------------------------------------------- Test 1: classify_capability_ring

def test_classify_capability_ring():
    """Test 1: classify_capability_ring 分类测试"""
    # 1. 深度思考推理环 (REASONING)
    assert classify_capability_ring("deepseek", "deepseek-reasoner") == CapabilityRing.REASONING
    assert classify_capability_ring("qwen", "qwen3.7-max (think)") == CapabilityRing.REASONING
    assert classify_capability_ring("kimi", "kimi-explore") == CapabilityRing.REASONING
    assert classify_capability_ring("doubao", "doubao-think") == CapabilityRing.REASONING
    assert classify_capability_ring("glm", "glm-zero-preview") == CapabilityRing.REASONING

    # 2. 旗舰全能环 (FLAGSHIP)
    assert classify_capability_ring("qwen", "qwen3.7-max") == CapabilityRing.FLAGSHIP
    assert classify_capability_ring("kimi", "kimi") == CapabilityRing.FLAGSHIP
    assert classify_capability_ring("doubao", "doubao-pro") == CapabilityRing.FLAGSHIP
    assert classify_capability_ring("glm", "glm-4-plus") == CapabilityRing.FLAGSHIP
    assert classify_capability_ring("deepseek", "deepseek-chat") == CapabilityRing.FLAGSHIP

    # 3. 极速经济环 (SPEED)
    assert classify_capability_ring("qwen", "qwen3.6-flash") == CapabilityRing.SPEED
    assert classify_capability_ring("doubao", "doubao-lite") == CapabilityRing.SPEED
    assert classify_capability_ring("glm", "glm-4-flash") == CapabilityRing.SPEED

    # 4. 环成员结构完整性
    for ring in (CapabilityRing.REASONING, CapabilityRing.FLAGSHIP, CapabilityRing.SPEED):
        assert ring in RING_DEFINITIONS
        assert len(RING_DEFINITIONS[ring]) >= 3

    print("[PASS] Test 1: classify_capability_ring 三级对等能力环识别分类通过")


# ---------------------------------------------------------------- Test 2: ProviderRiskMeta & CooldownTracker

def test_provider_risk_meta_and_cooldown_tracker():
    """Test 2: ProviderRiskMeta 与 CooldownTracker 动态冷却避让与并发控制测试"""
    # 2.1 验证五大模型风控级别与步调参数
    assert "qwen" in PROVIDER_RISK_SPECS
    assert "deepseek" in PROVIDER_RISK_SPECS
    assert "kimi" in PROVIDER_RISK_SPECS
    assert "doubao" in PROVIDER_RISK_SPECS
    assert "glm" in PROVIDER_RISK_SPECS

    # 高风控高频控节点
    for p in ("qwen", "deepseek", "kimi"):
        meta = PROVIDER_RISK_SPECS[p]
        assert meta.risk_level == RiskLevel.HIGH
        assert meta.min_call_interval >= 0.3
        assert meta.jitter_range[0] > 0.0
        assert meta.max_concurrency <= 3
        assert meta.default_cooldown >= 45.0

    # 低风控宽松节点
    for p in ("doubao", "glm"):
        meta = PROVIDER_RISK_SPECS[p]
        assert meta.risk_level == RiskLevel.LOW
        assert meta.min_call_interval <= 0.1
        assert meta.max_concurrency >= 5
        assert meta.default_cooldown <= 35.0

    async def _test_cooldown():
        tracker = CooldownTracker()

        # 初始状态
        assert not tracker.is_in_cooldown("qwen")
        assert tracker.get_remaining_cooldown("qwen") == 0.0

        # Case A: 429 或 WAF 人机挑战 (人机验证 / RGV587) -> 加权指数冷却 (>= 45s)
        cd_waf1 = await tracker.record_failure("qwen", 429, "触发 RGV587_ERROR 滑块验证")
        assert cd_waf1 >= 45.0
        assert tracker.is_in_cooldown("qwen")
        assert tracker.get_remaining_cooldown("qwen") > 40.0

        # 再次触发人机验证 -> 指数加权退避 (1.2 ** 1 * 45 = 54.0)
        cd_waf2 = await tracker.record_failure("qwen", 200, "触发人机验证阻断")
        assert cd_waf2 > cd_waf1
        assert cd_waf2 >= 50.0

        # Case B: 401 认证过期 -> 认证避让冷却 (>= 60s)
        tracker.reset()
        cd_auth = await tracker.record_failure("deepseek", 401, "Token expired or unauthorized")
        assert cd_auth >= 60.0
        assert tracker.is_in_cooldown("deepseek")
        assert tracker.get_remaining_cooldown("deepseek") > 55.0

        # Case C: 502 / 普通网络错误 -> 比例冷却 (min(base_cd, 15.0 * cur_fails))
        tracker.reset()
        cd_net1 = await tracker.record_failure("glm", 502, "Bad Gateway")
        assert cd_net1 == 15.0
        cd_net2 = await tracker.record_failure("glm", 502, "Bad Gateway")
        assert cd_net2 == 30.0  # min(30.0, 15.0 * 2)

        # Case D: record_success 重置失败计数并清除冷却
        assert tracker.is_in_cooldown("glm")
        assert tracker.get_remaining_cooldown("glm") > 0.0
        await tracker.record_success("glm")
        assert not tracker.is_in_cooldown("glm")
        assert tracker.get_remaining_cooldown("glm") == 0.0
        assert tracker._failure_counts.get("glm", 0) == 0

        # Case E: apply_pacing_and_acquire 步调与 Jitter 耗时
        tracker.reset()
        # 记录一次调用时间
        tracker._last_call_time["doubao"] = time.time()
        start = time.time()
        await tracker.apply_pacing_and_acquire("doubao")
        elapsed = time.time() - start
        # doubao min_call_interval 是 0.08，jitter 是 0.01-0.05，预计至少耗时 0.05s
        assert elapsed >= 0.05

        # Case F: get_semaphore 并发信号量
        sem_ds = tracker.get_semaphore("deepseek")
        assert sem_ds._value == PROVIDER_RISK_SPECS["deepseek"].max_concurrency
        sem_doubao = tracker.get_semaphore("doubao")
        assert sem_doubao._value == PROVIDER_RISK_SPECS["doubao"].max_concurrency

        # 未知节点返回默认并发
        sem_unknown = tracker.get_semaphore("custom_provider_xyz")
        assert sem_unknown._value == 4

    asyncio.run(_test_cooldown())
    print("[PASS] Test 2: ProviderRiskMeta 与 CooldownTracker 动态冷却避让机制通过")


# ---------------------------------------------------------------- Test 3: FailoverRouter.get_fallback_candidates

def test_fallback_candidates():
    """Test 3: FailoverRouter.get_fallback_candidates 候选排序与冷却降级测试"""
    async def _test():
        router = FailoverRouter()

        # 给定主选 deepseek-reasoner (属于 REASONING 环)
        # REASONING 环成员: deepseek(deepseek-reasoner), kimi(kimi-explore), qwen(Qwen3.7-Max), doubao(doubao-think), glm(glm-zero-preview)
        candidates = router.get_fallback_candidates("deepseek", "deepseek-reasoner")

        # 1. 验证排除了主选 provider
        cand_keys = [c.provider_key for c in candidates]
        assert "deepseek" not in cand_keys
        assert set(cand_keys) == {"kimi", "qwen", "doubao", "glm"}

        # 2. 验证低风险健康节点优先于��风险健康节点
        # doubao 与 glm 风险为 LOW (排在前面)
        # kimi 与 qwen 风险为 HIGH (排在后面)
        assert candidates[0].risk_level == RiskLevel.LOW
        assert candidates[1].risk_level == RiskLevel.LOW
        assert candidates[2].risk_level == RiskLevel.HIGH
        assert candidates[3].risk_level == RiskLevel.HIGH
        assert set([candidates[0].provider_key, candidates[1].provider_key]) == {"doubao", "glm"}
        assert set([candidates[2].provider_key, candidates[3].provider_key]) == {"kimi", "qwen"}

        # 3. 当低风险节点进入冷却避让时，其优先级降到最后
        await router.cooldown_tracker.record_failure("doubao", 429, "Rate limit")
        assert router.cooldown_tracker.is_in_cooldown("doubao")

        new_candidates = router.get_fallback_candidates("deepseek", "deepseek-reasoner")
        # 冷却节点被沉底到列表末尾
        assert new_candidates[-1].provider_key == "doubao"
        assert new_candidates[-1].in_cooldown is True

        # 列表前部全部为健康未冷却节点
        for cand in new_candidates[:-1]:
            assert cand.in_cooldown is False

        # 4. 指定 excluded_providers 集合
        filtered = router.get_fallback_candidates(
            "deepseek",
            "deepseek-reasoner",
            excluded_providers={"glm", "kimi"},
        )
        filtered_keys = [c.provider_key for c in filtered]
        assert "glm" not in filtered_keys
        assert "kimi" not in filtered_keys

    asyncio.run(_test())
    print("[PASS] Test 3: FailoverRouter.get_fallback_candidates 优先级排序与冷却降级通过")


# ---------------------------------------------------------------- Test 4: FailoverRouter.execute_chat

def test_execute_chat_scenarios():
    """Test 4: FailoverRouter.execute_chat 故障转移执行器场景验证"""
    async def _test():
        # -------------------------------------------------------------
        # Scenario A: 主选节点首包失败 (401 或 429) -> 透明转移到次级健康节点 -> 成功输出
        # -------------------------------------------------------------
        router_a = FailoverRouter()

        def factory_scenario_a(pkey: str):
            if pkey == "deepseek":
                # 主选节点模拟 429 WAF 阻断
                return MockErrorProvider("deepseek", status=429, err_msg="WAF 阻断 RGV587", err_type="rate_limit_error")
            elif pkey == "doubao":
                # 次选低风险健康节点 (doubao-think) 成功响应
                return MockSuccessStreamProvider("doubao", "豆包深度思考备用服务接力成功！")
            return MockErrorProvider(pkey)

        stream_a, event_a, active_a = await router_a.execute_chat(
            primary_provider_key="deepseek",
            primary_wire_model="deepseek-reasoner",
            provider_factory=factory_scenario_a,
            chat_args={"messages": [{"role": "user", "content": "9.9 与 9.11 比较"}]},
        )

        # 验证故障转移事件被正确填充
        assert event_a is not None
        assert event_a.from_provider == "deepseek"
        assert event_a.from_model == "deepseek-reasoner"
        assert event_a.to_provider == "doubao"
        assert event_a.to_model == "doubao-think"
        assert event_a.ring == CapabilityRing.REASONING
        assert "WAF 阻断" in event_a.reason

        # 验证原主选节点已被记入冷却避让
        assert router_a.cooldown_tracker.is_in_cooldown("deepseek") is True

        # 验证流式正常读取，且首包带有故障转移元数据
        chunks_a = []
        metas_a = []
        async for delta, meta in stream_a:
            if delta:
                chunks_a.append(delta)
            if meta:
                metas_a.append(meta)

        full_text_a = "".join(chunks_a)
        assert full_text_a == "豆包深度思考备用服务接力成功！"
        assert metas_a[0]["x-failover-from"] == "deepseek"
        assert metas_a[0]["x-failover-to"] == "doubao"
        assert metas_a[0]["x-failover-model"] == "doubao-think"
        assert metas_a[0]["x-failover-ring"] == "reasoning"

        # -------------------------------------------------------------
        # Scenario B: 能力环内所有候选节点全部故障 -> 抛出 ProviderError 复合报告
        # -------------------------------------------------------------
        router_b = FailoverRouter()

        def factory_scenario_b(pkey: str):
            return MockErrorProvider(pkey, status=503, err_msg="节点超载熔断", err_type="service_unavailable")

        try:
            await router_b.execute_chat(
                primary_provider_key="qwen",
                primary_wire_model="qwen3.6-flash",  # SPEED 环: qwen, doubao, glm
                provider_factory=factory_scenario_b,
                chat_args={"messages": [{"role": "user", "content": "速算测试"}]},
            )
            assert False, "所有节点故障必须抛出 ProviderError"
        except ProviderError as e:
            assert e.status == 503
            assert "对等能力环 [speed] 内所有节点均已尝试且全部失效" in e.message
            assert "qwen" in e.message
            assert "doubao" in e.message
            assert "glm" in e.message

        # -------------------------------------------------------------
        # Scenario C: 主选节点首次调用成功 -> FailoverEvent 为 None
        # -------------------------------------------------------------
        router_c = FailoverRouter()

        def factory_scenario_c(pkey: str):
            return MockSuccessStreamProvider(pkey, "主节点初次调用成功响应")

        stream_c, event_c, active_c = await router_c.execute_chat(
            primary_provider_key="deepseek",
            primary_wire_model="deepseek-chat",
            provider_factory=factory_scenario_c,
            chat_args={"messages": [{"role": "user", "content": "你好"}]},
        )

        assert event_c is None
        assert active_c.name == "deepseek"

        chunks_c = []
        async for delta, meta in stream_c:
            if delta:
                chunks_c.append(delta)
        assert "".join(chunks_c) == "主节点初次调用成功响应"

    asyncio.run(_test())
    print("[PASS] Test 4: FailoverRouter.execute_chat 场景 (切换/全熔断/直连成功) 验证通过")


# ---------------------------------------------------------------- Test 5: Server 端点集成与 /health 审计

def test_server_health_and_failover_integration():
    """Test 5: 服务端 GET /health 元数据报送及端点故障转移透传"""
    client = TestClient(srv.app)

    # 1. 验证 GET /health 包含风险等级、冷却状态与剩余冷却秒数
    # 先制造一个处于冷却状态的 Provider
    asyncio.run(srv.failover_router.cooldown_tracker.record_failure("deepseek", 429, "WAF 滑块挑战"))
    r_health = client.get("/health")
    assert r_health.status_code == 200
    h_data = r_health.json()

    assert h_data["status"] == "ok"
    assert "providers" in h_data
    providers = h_data["providers"]

    for name in ("qwen", "deepseek", "doubao", "kimi", "glm"):
        assert name in providers
        p_info = providers[name]
        assert "risk_level" in p_info
        assert p_info["risk_level"] in ("low", "medium", "high")
        assert "in_cooldown" in p_info
        assert "cooldown_status" in p_info
        assert "remaining_cooldown" in p_info
        assert "remaining_cooldown_seconds" in p_info

    # deepseek 处于冷却中
    assert providers["deepseek"]["in_cooldown"] is True
    assert providers["deepseek"]["cooldown_status"] == "in_cooldown"
    assert providers["deepseek"]["remaining_cooldown"] > 0.0

    # doubao 风险为 low，未冷却
    assert providers["doubao"]["risk_level"] == "low"
    assert providers["doubao"]["in_cooldown"] is False
    assert providers["doubao"]["cooldown_status"] == "ready"

    # 清除测试产生的冷却，避免干扰后续测试
    srv.failover_router.cooldown_tracker.reset()

    # 2. 验证 /v1/chat/completions (OpenAI) 自动故障转移
    def mock_server_factory(key: str):
        if key == "deepseek":
            return MockErrorProvider("deepseek", status=401, err_msg="认证凭据失效", err_type="auth_error")
        elif key == "doubao":
            return MockSuccessStreamProvider("doubao", "豆包在线接力回答！")
        return MockErrorProvider(key)

    srv._make_provider = mock_server_factory

    r_chat = client.post(
        "/v1/chat/completions",
        json={"model": "deepseek-reasoner", "messages": [{"role": "user", "content": "1+1"}]},
    )
    assert r_chat.status_code == 200
    chat_resp = r_chat.json()
    assert chat_resp["choices"][0]["message"]["content"] == "豆包在线接力回答！"
    assert "failover" in chat_resp
    assert chat_resp["failover"]["from_provider"] == "deepseek"
    assert chat_resp["failover"]["to_provider"] == "doubao"
    assert r_chat.headers.get("x-failover-from") == "deepseek"
    assert r_chat.headers.get("x-failover-to") == "doubao"

    # 3. 验证 /v1/messages (Anthropic) 自动故障转移
    r_msg = client.post(
        "/v1/messages",
        headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
        json={
            "model": "deepseek-reasoner",
            "messages": [{"role": "user", "content": "Anthropic 测试"}],
            "max_tokens": 512,
        },
    )
    assert r_msg.status_code == 200
    msg_resp = r_msg.json()
    assert msg_resp["type"] == "message"
    assert msg_resp["content"][0]["text"] == "豆包在线接力回答！"
    assert "failover" in msg_resp
    assert msg_resp["failover"]["from_provider"] == "deepseek"
    assert msg_resp["failover"]["to_provider"] == "doubao"
    assert r_msg.headers.get("x-failover-from") == "deepseek"
    assert r_msg.headers.get("x-failover-to") == "doubao"

    # 重置环境
    srv.failover_router.cooldown_tracker.reset()
    print("[PASS] Test 5: 服务端 GET /health 状态报送与端点透明故障转移透传通过")


def main():
    print("=" * 60)
    print("开始运行对等互备网格与风险频控步调系统单测...")
    print("=" * 60)

    test_classify_capability_ring()
    test_provider_risk_meta_and_cooldown_tracker()
    test_fallback_candidates()
    test_execute_chat_scenarios()
    test_server_health_and_failover_integration()

    print("=" * 60)
    print("所有对等互备网格与风险频控单测全部成功通过！(Exit 0)")
    print("=" * 60)


if __name__ == "__main__":
    main()
