"""会话粘滞路由与跨模型状态迁移引擎 (OPT-007) 测试套件

验证：
1. SessionAffinityManager 核心生命周期与状态机 (fresh -> bound -> handoff)
2. LRU 驱逐与 TTL 超时失效机制
3. 跨模型状态平滑迁移与异构上游 ID 剥离
4. 服务端多协议端点 (OpenAI / Anthropic / Responses) 粘滞响应头透传
5. GET /v1/diagnostics 遥测统计指标

运行: .venv/bin/python tests/test_affinity.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["GATEWAY_DISABLE_AUTO_REFRESH"] = "1"
os.environ["TESTING"] = "1"
os.environ["QWEN_AUTH_TOKEN"] = "fake_qwen"
os.environ["DEEPSEEK_AUTH_TOKEN"] = "fake_ds"
os.environ["DOUBAO_AUTH_TOKEN"] = "fake_doubao"
os.environ["KIMI_AUTH_TOKEN"] = "fake_kimi"
os.environ["GLM_AUTH_TOKEN"] = "fake_glm"

from fastapi.testclient import TestClient  # noqa: E402

import server as srv  # noqa: E402
from providers.session_affinity import (  # noqa: E402
    AffinityDecision,
    SessionAffinityManager,
    session_affinity_manager,
)


class DummyEchoProvider:
    """模拟返回上游 conversation_id 的 Provider。"""

    def __init__(self, key: str = "qwen"):
        self.key = key

    async def chat(self, *a, **kw):
        conv_id = kw.get("conversation_id") or f"{self.key}_internal_conv_123"
        yield "收到消息", {"conversation_id": conv_id}


def test_session_affinity_manager_lifecycle():
    """测试会话粘滞管理器基础生命周期：fresh -> bound -> handoff"""
    mgr = SessionAffinityManager(ttl_seconds=10.0, max_entries=5)

    # 1. 无 session_id 时始终为 fresh
    dec0 = mgr.resolve_affinity(
        session_id=None,
        primary_provider="qwen",
        primary_model="Qwen3.7-Max",
        is_provider_available=lambda p: True,
    )
    assert dec0.status == "fresh"
    assert dec0.provider_key == "qwen"
    assert dec0.wire_model == "Qwen3.7-Max"
    assert dec0.turn_count == 1
    assert not dec0.is_handoff

    # 2. 首次带 session_id 请求（尚未记录）
    session_id = "test_user_session_001"
    dec1 = mgr.resolve_affinity(
        session_id=session_id,
        primary_provider="qwen",
        primary_model="Qwen3.7-Max",
        is_provider_available=lambda p: True,
    )
    assert dec1.status == "fresh"
    assert dec1.provider_key == "qwen"
    assert dec1.upstream_conv_id == session_id

    # 记录第一轮对话
    mgr.record_turn(
        session_id=session_id,
        provider_key="qwen",
        model_name="Qwen3.7-Max",
        upstream_conv_id="qwen_cloud_conv_abc",
    )

    # 3. 第二轮对话：应命中 bound 粘滞
    dec2 = mgr.resolve_affinity(
        session_id=session_id,
        primary_provider="doubao",  # 哪怕调度算法推荐了 doubao
        primary_model="doubao-pro",
        is_provider_available=lambda p: True,
    )
    assert dec2.status == "bound"
    assert dec2.provider_key == "qwen"
    assert dec2.wire_model == "Qwen3.7-Max"
    assert dec2.upstream_conv_id == "qwen_cloud_conv_abc"
    assert dec2.turn_count == 1
    assert not dec2.is_handoff

    # 记录第二轮对话更新轮次
    mgr.record_turn(
        session_id=session_id,
        provider_key="qwen",
        model_name="Qwen3.7-Max",
        upstream_conv_id="qwen_cloud_conv_abc",
    )

    # 4. 原节点发生故障或进入风控冷却：触发平滑跨模型迁移 (handoff)
    def is_available(pkey: str) -> bool:
        return pkey != "qwen"  # qwen 故障

    dec3 = mgr.resolve_affinity(
        session_id=session_id,
        primary_provider="doubao",
        primary_model="doubao-think",
        is_provider_available=is_available,
    )
    assert dec3.status == "handoff"
    assert dec3.is_handoff is True
    assert dec3.provider_key == "doubao"
    assert dec3.wire_model == "doubao-think"
    assert dec3.upstream_conv_id is None  # 异构 ID 剥离，避免上游冲突
    assert dec3.original_provider == "qwen"
    assert dec3.turn_count == 2

    # 验证诊断遥测
    diag = mgr.get_diagnostics()
    assert diag["total_recorded_sessions"] == 1
    assert diag["affinity_hits"] >= 1
    assert diag["affinity_handoffs"] >= 1
    print("test_session_affinity_manager_lifecycle: PASSED")


def test_session_affinity_ttl_and_lru():
    """测试会话粘滞管理器 TTL 过期与 LRU 淘汰"""
    mgr = SessionAffinityManager(ttl_seconds=0.1, max_entries=2)

    # 添加两个条目
    mgr.record_turn("s1", "qwen", "Qwen", "c1")
    mgr.record_turn("s2", "doubao", "doubao-pro", "c2")

    # 验证立即访问处于有效状态
    d1 = mgr.resolve_affinity("s1", "qwen", "Qwen", lambda p: True)
    assert d1.status == "bound"

    # 添加第三个条目，应淘汰 s2 (最久未访问)
    mgr.record_turn("s3", "kimi", "kimi", "c3")
    diag = mgr.get_diagnostics()
    assert diag["active_sessions"] <= 2

    # 等待 TTL 超时
    time.sleep(0.15)
    d_expired = mgr.resolve_affinity("s1", "qwen", "Qwen", lambda p: True)
    assert d_expired.status == "fresh"
    print("test_session_affinity_ttl_and_lru: PASSED")


def test_chat_completions_affinity_headers():
    """测试 OpenAI /v1/chat/completions 会话粘滞响应头透传"""
    session_affinity_manager.clear()
    orig_make_provider = srv._make_provider
    srv._make_provider = lambda pkey, client=None: DummyEchoProvider(pkey)

    try:
        client = TestClient(srv.app)
        session_id = "test_conv_chat_001"

        # Turn 1: 首次请求，返回 fresh
        r1 = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen",
                "messages": [{"role": "user", "content": "你好"}],
                "conversation_id": session_id,
            },
        )
        assert r1.status_code == 200, r1.text
        assert r1.headers.get("x-session-affinity") == "fresh"
        assert r1.headers.get("x-session-turns") == "1"

        # Turn 2: 第二次带相同 conversation_id，返回 bound
        r2 = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen",
                "messages": [
                    {"role": "user", "content": "你好"},
                    {"role": "assistant", "content": "收到消息"},
                    {"role": "user", "content": "再见"},
                ],
                "conversation_id": session_id,
            },
        )
        assert r2.status_code == 200, r2.text
        assert r2.headers.get("x-session-affinity") == "bound"

        # Turn 3: 模拟原 provider 被风控禁用，应返回 handoff 与 x-session-handoff-from
        os.environ["DISABLED_PROVIDERS"] = "qwen"
        try:
            r3 = client.post(
                "/v1/chat/completions",
                json={
                    "model": "qwen",
                    "messages": [{"role": "user", "content": "换节点测试"}],
                    "conversation_id": session_id,
                },
            )
            assert r3.status_code == 200, r3.text
            assert r3.headers.get("x-session-affinity") == "handoff"
            assert r3.headers.get("x-session-handoff-from") == "qwen"
        finally:
            os.environ["DISABLED_PROVIDERS"] = ""
    finally:
        srv._make_provider = orig_make_provider
        session_affinity_manager.clear()

    print("test_chat_completions_affinity_headers: PASSED")


def test_anthropic_messages_affinity_headers():
    """测试 Anthropic /v1/messages 会话粘滞响应头透传"""
    session_affinity_manager.clear()
    orig_make_provider = srv._make_provider
    srv._make_provider = lambda pkey, client=None: DummyEchoProvider(pkey)

    try:
        client = TestClient(srv.app)
        session_id = "test_anthropic_session_002"

        # 非流式 Turn 1
        r1 = client.post(
            "/v1/messages",
            headers={"anthropic-version": "2023-06-01"},
            json={
                "model": "claude-3-7-sonnet",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "你好 Claude"}],
                "conversation_id": session_id,
            },
        )
        assert r1.status_code == 200, r1.text
        assert r1.headers.get("x-session-affinity") == "fresh"

        # 非流式 Turn 2: 应变为 bound
        r2 = client.post(
            "/v1/messages",
            headers={"anthropic-version": "2023-06-01"},
            json={
                "model": "claude-3-7-sonnet",
                "max_tokens": 1024,
                "messages": [
                    {"role": "user", "content": "你好 Claude"},
                    {"role": "assistant", "content": "收到消息"},
                    {"role": "user", "content": "第二轮提问"},
                ],
                "conversation_id": session_id,
            },
        )
        assert r2.status_code == 200, r2.text
        assert r2.headers.get("x-session-affinity") == "bound"

        # 流式 Turn 3
        r3 = client.post(
            "/v1/messages",
            headers={"anthropic-version": "2023-06-01"},
            json={
                "model": "claude-3-7-sonnet",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "流式测试"}],
                "conversation_id": session_id,
                "stream": True,
            },
        )
        assert r3.status_code == 200, r3.text
        assert r3.headers.get("x-session-affinity") == "bound"
    finally:
        srv._make_provider = orig_make_provider
        session_affinity_manager.clear()

    print("test_anthropic_messages_affinity_headers: PASSED")


def test_responses_endpoint_affinity_headers():
    """测试 OpenAI Responses (/v1/responses) 会话粘滞响应头透传"""
    session_affinity_manager.clear()
    orig_make_provider = srv._make_provider
    srv._make_provider = lambda pkey, client=None: DummyEchoProvider(pkey)

    try:
        client = TestClient(srv.app)
        session_id = "test_resp_session_003"

        r1 = client.post(
            "/v1/responses",
            json={
                "model": "gpt-4o",
                "input": "Write a python script",
                "conversation_id": session_id,
            },
        )
        assert r1.status_code == 200, r1.text
        assert r1.headers.get("x-session-affinity") == "fresh"

        # Turn 2
        r2 = client.post(
            "/v1/responses",
            json={
                "model": "gpt-4o",
                "input": "Optimize it",
                "conversation_id": session_id,
            },
        )
        assert r2.status_code == 200, r2.text
        assert r2.headers.get("x-session-affinity") == "bound"
    finally:
        srv._make_provider = orig_make_provider
        session_affinity_manager.clear()

    print("test_responses_endpoint_affinity_headers: PASSED")


def test_diagnostics_telemetry():
    """测试 /v1/diagnostics 中的 session_affinity 遥测字段"""
    client = TestClient(srv.app)
    r = client.get("/v1/diagnostics")
    assert r.status_code == 200
    data = r.json()
    assert "session_affinity" in data
    aff = data["session_affinity"]
    assert "active_sessions" in aff
    assert "total_recorded_sessions" in aff
    assert "affinity_hits" in aff
    assert "affinity_handoffs" in aff
    print("test_diagnostics_telemetry: PASSED")


def run_all():
    test_session_affinity_manager_lifecycle()
    test_session_affinity_ttl_and_lru()
    test_chat_completions_affinity_headers()
    test_anthropic_messages_affinity_headers()
    test_responses_endpoint_affinity_headers()
    test_diagnostics_telemetry()
    print("\n✅ All session affinity & state handoff tests (OPT-007) passed!")


if __name__ == "__main__":
    run_all()
