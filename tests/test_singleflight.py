"""智能并发语义去重与流式请求合并引擎 (OPT-008) 测试套件

验证：
1. 请求指纹计算 (Deterministic Fingerprint Hashing) 确定性与区分度
2. SingleFlightManager 角色判定 (Leader vs Follower)
3. 流式单飞多路广播 (Streaming Multi-Subscriber Fanout) 与历史重放缓存 (Replay Buffer)
4. 流式异常在 Leader 与 Followers 间的安全隔离与传递
5. 非流式单飞请求合并 (Non-Streaming Coalescing)
6. 多协议端点 (OpenAI / Anthropic / Responses) 的 SingleFlight 遥测响应头
7. GET /v1/diagnostics 中的 singleflight 诊断遥测指标
8. GATEWAY_DISABLE_SINGLEFLIGHT 环境变量开关

运行: .venv/bin/python tests/test_singleflight.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["GATEWAY_DISABLE_AUTO_REFRESH"] = "1"
os.environ["TESTING"] = "1"
os.environ["GATEWAY_SINGLEFLIGHT_DEBOUNCE_TTL"] = "0.5"
os.environ["QWEN_AUTH_TOKEN"] = "fake_qwen"
os.environ["DEEPSEEK_AUTH_TOKEN"] = "fake_ds"
os.environ["DOUBAO_AUTH_TOKEN"] = "fake_doubao"
os.environ["KIMI_AUTH_TOKEN"] = "fake_kimi"
os.environ["GLM_AUTH_TOKEN"] = "fake_glm"

from fastapi.testclient import TestClient  # noqa: E402

import server as srv  # noqa: E402
from providers.singleflight import (  # noqa: E402
    FlightGroup,
    SingleFlightDecision,
    SingleFlightManager,
    broadcast_stream_wrapper,
    singleflight_manager,
    stream_follower_gen,
)

singleflight_manager.debounce_ttl = 0.5


class DummySlowProvider:
    """模拟慢速流式输出的 Provider，用于验证并发广播与重放。"""

    def __init__(self, key: str = "qwen", delay: float = 0.05):
        self.key = key
        self.delay = delay

    async def chat(self, *a, **kw):
        for chunk in ["Hello", " ", "World", "!"]:
            if self.delay > 0:
                await asyncio.sleep(self.delay)
            yield chunk, {"conversation_id": "conv_dummy_123"}


def test_singleflight_key_computation():
    """测试请求特征指纹计算的确定性与敏感性。"""
    mgr = SingleFlightManager()

    messages1 = [{"role": "user", "content": "9.11 和 9.9 哪个大"}]
    messages2 = [{"role": "user", "content": "9.11 和 9.9 哪个大"}]
    messages3 = [{"role": "user", "content": "9.11 和 9.8 哪个大"}]

    k1 = mgr.compute_key("qwen", "Qwen3.7-Max", messages1, temperature=0.7, stream=True)
    k2 = mgr.compute_key("qwen", "Qwen3.7-Max", messages2, temperature=0.7, stream=True)
    k3 = mgr.compute_key("qwen", "Qwen3.7-Max", messages3, temperature=0.7, stream=True)
    k_stream_diff = mgr.compute_key("qwen", "Qwen3.7-Max", messages1, temperature=0.7, stream=False)
    k_temp_diff = mgr.compute_key("qwen", "Qwen3.7-Max", messages1, temperature=0.2, stream=True)

    # 确定性：相同参数哈希一致
    assert k1 == k2
    assert len(k1) == 64
    # 敏感性：消息内容、stream 模式或温度改变时哈希不同
    assert k1 != k3
    assert k1 != k_stream_diff
    assert k1 != k_temp_diff
    print("test_singleflight_key_computation: PASSED")


def test_singleflight_manager_roles():
    """测试并发请求时的 Leader 与 Follower 角色分配及防抖过期清理。"""
    async def _run():
        mgr = SingleFlightManager(debounce_ttl=0.1)

        key = "test_flight_key_001"
        # 1. 首次请求：Leader
        d1 = await mgr.get_or_create(key)
        assert d1.is_leader is True
        assert d1.flight_key == key
        assert d1.short_key == key[:12]

        # 2. 第二次请求：Follower
        d2 = await mgr.get_or_create(key)
        assert d2.is_leader is False
        assert d2.group is d1.group
        assert d1.group.follower_count == 1

        # Leader 完成
        await d1.group.finish(result={"status": "ok"})
        assert d1.group.leader_done is True

        # 3. 在 debounce_ttl 内再次请求，仍命中该已完成组
        d3 = await mgr.get_or_create(key)
        assert d3.is_leader is False

        # 4. 等待 TTL 超时，下一次请求重新成为 Leader
        await asyncio.sleep(0.15)
        d4 = await mgr.get_or_create(key)
        assert d4.is_leader is True
        assert d4.group is not d1.group

    asyncio.run(_run())
    print("test_singleflight_manager_roles: PASSED")


def test_singleflight_stream_broadcast_and_replay():
    """测试流式单飞广播：Leader 输出，Follower 1 提前订阅，Follower 2 中途加入并重放历史。"""
    async def _run():
        group = FlightGroup("test_key_stream")

        # 模拟 Leader 生产数据流
        async def leader_producer():
            for item in ["frame_1", "frame_2", "frame_3", "frame_4"]:
                await asyncio.sleep(0.02)
                yield item

        # Follower 1 提早订阅
        f1_received = []

        async def follower_1_task():
            async for chunk in stream_follower_gen(group):
                f1_received.append(chunk)

        # Follower 2 中途加入
        f2_received = []

        async def follower_2_task():
            await asyncio.sleep(0.05)  # 等待 frame_1 和 frame_2 已产生
            async for chunk in stream_follower_gen(group):
                f2_received.append(chunk)

        # Leader 包装广播
        leader_received = []

        async def leader_task():
            wrapped = broadcast_stream_wrapper(leader_producer(), group)
            async for chunk in wrapped:
                leader_received.append(chunk)

        await asyncio.gather(leader_task(), follower_1_task(), follower_2_task())

        expected = ["frame_1", "frame_2", "frame_3", "frame_4"]
        assert leader_received == expected
        assert f1_received == expected
        # 验证 Follower 2 即使中途加入，也通过历史重放完整收到所有数据帧
        assert f2_received == expected

    asyncio.run(_run())
    print("test_singleflight_stream_broadcast_and_replay: PASSED")


def test_singleflight_stream_error_propagation():
    """测试上游发生异常时在 Leader 与 Followers 间的异常透传。"""
    async def _run():
        group = FlightGroup("test_key_err")

        async def faulty_producer():
            yield "chunk_ok"
            await asyncio.sleep(0.01)
            raise RuntimeError("上游连接异常中断")

        follower_received = []
        follower_err = None

        async def follower_task():
            nonlocal follower_err
            try:
                async for chunk in stream_follower_gen(group):
                    follower_received.append(chunk)
            except Exception as e:
                follower_err = e

        leader_err = None

        async def leader_task():
            nonlocal leader_err
            try:
                wrapped = broadcast_stream_wrapper(faulty_producer(), group)
                async for _ in wrapped:
                    pass
            except Exception as e:
                leader_err = e

        await asyncio.gather(leader_task(), follower_task())

        assert isinstance(leader_err, RuntimeError)
        assert isinstance(follower_err, RuntimeError)
        assert "上游连接异常中断" in str(follower_err)
        assert follower_received == ["chunk_ok"]

    asyncio.run(_run())
    print("test_singleflight_stream_error_propagation: PASSED")


def test_singleflight_non_stream_coalescing():
    """测试非流式请求并发去重与结果共享。"""
    async def _run():
        group = FlightGroup("test_key_non_stream")

        async def follower_coro():
            await group.done_event.wait()
            return group.completed_result

        f_task = asyncio.create_task(follower_coro())

        # Leader 计算完毕
        await asyncio.sleep(0.02)
        expected_result = {"id": "chatcmpl-123", "choices": [{"message": {"content": "42"}}]}
        await group.finish(result=expected_result)

        f_res = await f_task
        assert f_res == expected_result

    asyncio.run(_run())
    print("test_singleflight_non_stream_coalescing: PASSED")


def test_singleflight_chat_completions_headers():
    """测试 OpenAI /v1/chat/completions 响应头中的 x-singleflight 遥测。"""
    singleflight_manager.clear()
    orig_make_provider = srv._make_provider
    srv._make_provider = lambda pkey, client=None: DummySlowProvider(pkey, delay=0.0)

    try:
        client = TestClient(srv.app)
        payload = {
            "model": "qwen",
            "messages": [{"role": "user", "content": "SingleFlight 测试"}],
            "temperature": 0.5,
        }

        # 1. 发起非流式请求
        r1 = client.post("/v1/chat/completions", json=payload)
        assert r1.status_code == 200, r1.text
        assert r1.headers.get("x-singleflight") == "leader"
        assert "x-singleflight-key" in r1.headers

        # 2. 在防抖窗口内再次发起完全相同的非流式请求，应命中 Follower 共享
        r2 = client.post("/v1/chat/completions", json=payload)
        assert r2.status_code == 200, r2.text
        assert r2.headers.get("x-singleflight") == "follower"
        assert r2.headers.get("x-singleflight-key") == r1.headers.get("x-singleflight-key")
        assert "x-singleflight-followers" in r2.headers
    finally:
        srv._make_provider = orig_make_provider
        singleflight_manager.clear()

    print("test_singleflight_chat_completions_headers: PASSED")


def test_singleflight_anthropic_messages_headers():
    """测试 Anthropic /v1/messages 响应头中的 x-singleflight 遥测。"""
    singleflight_manager.clear()
    orig_make_provider = srv._make_provider
    srv._make_provider = lambda pkey, client=None: DummySlowProvider(pkey, delay=0.0)

    try:
        client = TestClient(srv.app)
        payload = {
            "model": "claude-3-7-sonnet",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Anthropic SingleFlight 测试"}],
            "temperature": 0.3,
        }

        # 1. 首次非流式请求
        r1 = client.post(
            "/v1/messages",
            headers={"anthropic-version": "2023-06-01"},
            json=payload,
        )
        assert r1.status_code == 200, r1.text
        assert r1.headers.get("x-singleflight") == "leader"
        assert "x-singleflight-key" in r1.headers

        # 2. 相同请求在防抖 TTL 内再次调用：Follower 共享
        r2 = client.post(
            "/v1/messages",
            headers={"anthropic-version": "2023-06-01"},
            json=payload,
        )
        assert r2.status_code == 200, r2.text
        assert r2.headers.get("x-singleflight") == "follower"
        assert r2.headers.get("x-singleflight-key") == r1.headers.get("x-singleflight-key")
    finally:
        srv._make_provider = orig_make_provider
        singleflight_manager.clear()

    print("test_singleflight_anthropic_messages_headers: PASSED")


def test_singleflight_responses_endpoint_headers():
    """测试 OpenAI Responses (/v1/responses) 响应头中的 x-singleflight 遥测。"""
    singleflight_manager.clear()
    orig_make_provider = srv._make_provider
    srv._make_provider = lambda pkey, client=None: DummySlowProvider(pkey, delay=0.0)

    try:
        client = TestClient(srv.app)
        payload = {
            "model": "gpt-4o",
            "input": "Responses SingleFlight 测试",
            "temperature": 0.4,
        }

        r1 = client.post("/v1/responses", json=payload)
        assert r1.status_code == 200, r1.text
        assert r1.headers.get("x-singleflight") == "leader"
        assert "x-singleflight-key" in r1.headers

        # 第二次请求
        r2 = client.post("/v1/responses", json=payload)
        assert r2.status_code == 200, r2.text
        assert r2.headers.get("x-singleflight") == "follower"
        assert r2.headers.get("x-singleflight-key") == r1.headers.get("x-singleflight-key")
    finally:
        srv._make_provider = orig_make_provider
        singleflight_manager.clear()

    print("test_singleflight_responses_endpoint_headers: PASSED")


def test_singleflight_diagnostics_telemetry():
    """测试 /v1/diagnostics 中的 singleflight 遥测指标。"""
    client = TestClient(srv.app)
    r = client.get("/v1/diagnostics")
    assert r.status_code == 200
    data = r.json()
    assert "singleflight" in data
    sf = data["singleflight"]
    assert "enabled" in sf
    assert "active_flights" in sf
    assert "total_flights" in sf
    assert "leader_executions" in sf
    assert "follower_shares" in sf
    assert "saved_ratio" in sf
    assert "debounce_ttl_seconds" in sf
    print("test_singleflight_diagnostics_telemetry: PASSED")


def test_singleflight_disabled_env():
    """测试 GATEWAY_DISABLE_SINGLEFLIGHT=1 时的禁用行为。"""
    singleflight_manager.clear()
    orig_make_provider = srv._make_provider
    srv._make_provider = lambda pkey, client=None: DummySlowProvider(pkey, delay=0.0)

    os.environ["GATEWAY_DISABLE_SINGLEFLIGHT"] = "1"
    try:
        client = TestClient(srv.app)
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen",
                "messages": [{"role": "user", "content": "禁用单飞测试"}],
            },
        )
        assert r.status_code == 200
        # 禁用时不应包含 x-singleflight 响应头
        assert "x-singleflight" not in r.headers
    finally:
        os.environ.pop("GATEWAY_DISABLE_SINGLEFLIGHT", None)
        srv._make_provider = orig_make_provider
        singleflight_manager.clear()

    print("test_singleflight_disabled_env: PASSED")


def run_all():
    test_singleflight_key_computation()
    test_singleflight_manager_roles()
    test_singleflight_stream_broadcast_and_replay()
    test_singleflight_stream_error_propagation()
    test_singleflight_non_stream_coalescing()
    test_singleflight_chat_completions_headers()
    test_singleflight_anthropic_messages_headers()
    test_singleflight_responses_endpoint_headers()
    test_singleflight_diagnostics_telemetry()
    test_singleflight_disabled_env()
    print("\n✅ All singleflight request debouncing & stream fanout tests (OPT-008) passed!")


if __name__ == "__main__":
    run_all()
