"""服务端集成测试：验证错误转换、模型路由、SSE 帧结构（无需真实网络）。

运行: .venv/bin/python tests/test_server.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["GATEWAY_DISABLE_AUTO_REFRESH"] = "1"
os.environ["QWEN_AUTH_TOKEN"] = "fake_qwen"
os.environ["DEEPSEEK_AUTH_TOKEN"] = "fake_ds"
os.environ["DOUBAO_AUTH_TOKEN"] = "fake_doubao"

from fastapi.testclient import TestClient  # noqa: E402

import server as srv  # noqa: E402
from providers.base import ProviderError  # noqa: E402


class BoomProvider:
    """立即抛错的假 Provider，验证服务端把上游错误转成规范响应。"""

    async def chat(self, *a, **kw):
        raise ProviderError("上游 401: token 无效或已过期", status=401, err_type="auth_error")
        yield  # 保持 async generator 语义


class ToolCallingProvider:
    """模拟输出工具调用的 Provider。"""

    async def chat(self, *a, **kw):
        yield "准备读取文件：\n```tool_call\n", {}
        yield '{"name": "Read", "arguments": {"file_path": "/tmp/test.txt"}}\n', {}
        yield "```", {}


class EchoProvider:
    """回显假 Provider：验证 SSE 帧结构与 conversation_id 透传。"""

    async def chat(self, *a, **kw):
        yield "你好", {}
        yield "，世界", {"conversation_id": "qwen:chat_x:resp_y"}


def test_unknown_model_404():
    client = TestClient(srv.app)
    r = client.post(
        "/v1/chat/completions",
        json={"model": "unknown-nonexistent-model", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 404
    assert r.json()["error"]["type"] == "model_not_found"


def test_empty_messages_400():
    client = TestClient(srv.app)
    r = client.post("/v1/chat/completions", json={"model": "qwen", "messages": []})
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "invalid_request_error"


def test_upstream_error_json():
    srv._make_provider = lambda key: BoomProvider()
    client = TestClient(srv.app)
    r = client.post(
        "/v1/chat/completions",
        json={"model": "qwen-max-latest", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 401, r.text
    assert r.json()["error"]["type"] == "auth_error"


def test_upstream_error_sse():
    srv._make_provider = lambda key: BoomProvider()
    client = TestClient(srv.app)
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={"model": "deepseek-chat", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    ) as r:
        assert r.status_code == 200
        text = "".join(r.iter_text())
    assert '"error"' in text
    assert "token 无效" in text


def test_stream_sse_structure():
    srv._make_provider = lambda key: EchoProvider()
    client = TestClient(srv.app)
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={"model": "qwen-max-latest", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    ) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        lines = [ln for ln in r.iter_lines() if ln]
    frames = [ln[5:] for ln in lines if ln.startswith("data:") and ln[5:].strip() != "[DONE]"]
    assert frames, lines
    import json

    first = json.loads(frames[0])
    assert first["choices"][0]["delta"]["role"] == "assistant"
    body = "".join(
        json.loads(f)["choices"][0]["delta"].get("content", "")
        for f in frames[:-1]
        if json.loads(f).get("choices")
    )
    assert body == "你好，世界", body
    last = json.loads(frames[-1])
    assert last["choices"][0]["finish_reason"] == "stop"
    assert last["conversation_id"] == "qwen:chat_x:resp_y"
    assert lines[-1] == "data: [DONE]"


def test_doubao_routing_and_models():
    # 测试模型路由
    pk, wm = srv.resolve_model("doubao")
    assert pk == "doubao" and wm == "doubao-pro"

    pk, wm = srv.resolve_model("doubao-think")
    assert pk == "doubao" and wm == "doubao-think"

    pk, wm = srv.resolve_model("doubao-custom-variant")
    assert pk == "doubao" and wm == "doubao-custom-variant"

    client = TestClient(srv.app)
    # 测试 /health
    r_health = client.get("/health")
    assert r_health.status_code == 200
    assert r_health.json()["doubao_configured"] is True

    # 测试 /v1/models 包含 doubao
    r_models = client.get("/v1/models")
    assert r_models.status_code == 200
    model_ids = [m["id"] for m in r_models.json()["data"]]
    assert "doubao" in model_ids
    assert "doubao-pro" in model_ids
    assert "doubao-think" in model_ids

    # 测试 Doubao 补全分发
    srv._make_provider = lambda key: EchoProvider()
    r = client.post(
        "/v1/chat/completions",
        json={"model": "doubao-pro", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "你好，世界"


def test_claude_model_routing():
    # 1. 默认后端 (qwen)
    os.environ["CLAUDE_BACKEND"] = "qwen"
    pk, wm = srv.resolve_model("claude-3-7-sonnet-20250219")
    assert pk == "qwen" and wm == "Qwen3.7-Max"

    pk, wm = srv.resolve_model("claude-3-5-haiku-20241022")
    assert pk == "qwen" and wm == "Qwen3.6-Flash"

    pk, wm = srv.resolve_model("claude-3-5-sonnet")
    assert pk == "qwen" and wm == "Qwen"

    pk, wm = srv.resolve_model("claude-3-opus")
    assert pk == "qwen" and wm == "Qwen3.7-Max"

    # 2. DeepSeek 后端
    os.environ["CLAUDE_BACKEND"] = "deepseek"
    pk, wm = srv.resolve_model("claude-3-opus")
    assert pk == "deepseek" and wm == "deepseek-reasoner"

    pk, wm = srv.resolve_model("claude-3-5-sonnet")
    assert pk == "deepseek" and wm == "deepseek-chat"

    # 3. 豆包后端
    os.environ["CLAUDE_BACKEND"] = "doubao"
    pk, wm = srv.resolve_model("claude-3-opus")
    assert pk == "doubao" and wm == "doubao-think"

    pk, wm = srv.resolve_model("claude-3-5-haiku")
    assert pk == "doubao" and wm == "doubao-lite"

    pk, wm = srv.resolve_model("claude-3-5-sonnet")
    assert pk == "doubao" and wm == "doubao-pro"

    # 还原
    os.environ["CLAUDE_BACKEND"] = "qwen"


def test_anthropic_non_streaming():
    srv._make_provider = lambda key: EchoProvider()
    client = TestClient(srv.app)
    r = client.post(
        "/v1/messages",
        headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
        json={
            "model": "claude-3-7-sonnet",
            "max_tokens": 1024,
            "system": "你是一个助手",
            "messages": [{"role": "user", "content": "你好"}],
        },
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["type"] == "message"
    assert data["role"] == "assistant"
    assert data["model"] == "claude-3-7-sonnet"
    assert len(data["content"]) == 1
    assert data["content"][0]["type"] == "text"
    assert data["content"][0]["text"] == "你好，世界"
    assert data["stop_reason"] == "end_turn"
    assert data["usage"]["input_tokens"] > 0
    assert data["usage"]["output_tokens"] > 0


def test_anthropic_streaming_sse_sequence():
    import json

    srv._make_provider = lambda key: EchoProvider()
    client = TestClient(srv.app)
    with client.stream(
        "POST",
        "/v1/messages",
        headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
        json={
            "model": "claude-3-5-haiku",
            "max_tokens": 512,
            "stream": True,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "你好"},
                        {"type": "tool_result", "content": "工具执行结果"},
                    ],
                }
            ],
        },
    ) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        raw_text = "".join(r.iter_text())

    # 验证不包含 OpenAI 的 [DONE]
    assert "[DONE]" not in raw_text

    # 解析 SSE 事件对 (event, data)
    events: list[tuple[str, dict]] = []
    blocks = raw_text.strip().split("\n\n")
    for block in blocks:
        if not block.strip():
            continue
        ev_name = None
        ev_data = None
        for line in block.split("\n"):
            if line.startswith("event: "):
                ev_name = line[len("event: ") :].strip()
            elif line.startswith("data: "):
                ev_data = json.loads(line[len("data: ") :].strip())
        if ev_name and ev_data:
            events.append((ev_name, ev_data))

    event_names = [e[0] for e in events]
    expected_order = [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert event_names == expected_order, f"事件流顺序不符合规范: {event_names}"

    # 细粒度断言
    msg_start = events[0][1]
    assert msg_start["type"] == "message_start"
    assert msg_start["message"]["role"] == "assistant"
    assert msg_start["message"]["content"] == []

    cb_start = events[1][1]
    assert cb_start["type"] == "content_block_start"
    assert cb_start["content_block"]["type"] == "text"

    deltas = [e[1]["delta"]["text"] for e in events if e[0] == "content_block_delta"]
    assert "".join(deltas) == "你好，世界"

    msg_delta = events[5][1]
    assert msg_delta["type"] == "message_delta"
    assert msg_delta["delta"]["stop_reason"] == "end_turn"
    assert msg_delta["usage"]["output_tokens"] > 0

    assert events[6][0] == "message_stop"


def test_anthropic_tool_calling_streaming():
    import json

    srv._make_provider = lambda key: ToolCallingProvider()
    client = TestClient(srv.app)
    with client.stream(
        "POST",
        "/v1/messages",
        headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen3.7-max",
            "messages": [{"role": "user", "content": "帮我读取 /tmp/test.txt"}],
            "tools": [
                {
                    "name": "Read",
                    "description": "读取本地文件",
                    "input_schema": {
                        "type": "object",
                        "properties": {"file_path": {"type": "string"}},
                        "required": ["file_path"],
                    },
                }
            ],
            "stream": True,
        },
    ) as r:
        assert r.status_code == 200
        raw_text = "".join(r.iter_text())

    events: list[tuple[str, dict]] = []
    for block in raw_text.strip().split("\n\n"):
        if not block.strip():
            continue
        ev_name = None
        ev_data = None
        for line in block.split("\n"):
            if line.startswith("event: "):
                ev_name = line[len("event: ") :].strip()
            elif line.startswith("data: "):
                ev_data = json.loads(line[len("data: ") :].strip())
        if ev_name and ev_data:
            events.append((ev_name, ev_data))

    event_names = [e[0] for e in events]
    assert "message_start" in event_names
    assert "content_block_start" in event_names
    assert "message_delta" in event_names
    assert "message_stop" in event_names

    # 验证提取到的 tool_use block
    tool_blocks = [
        e[1]["content_block"]
        for e in events
        if e[0] == "content_block_start" and e[1].get("content_block", {}).get("type") == "tool_use"
    ]
    assert len(tool_blocks) == 1
    assert tool_blocks[0]["name"] == "Read"

    msg_delta = [e[1] for e in events if e[0] == "message_delta"][0]
    assert msg_delta["delta"]["stop_reason"] == "tool_use"


def test_anthropic_tool_calling_non_streaming():
    srv._make_provider = lambda key: ToolCallingProvider()
    client = TestClient(srv.app)
    r = client.post(
        "/v1/messages",
        headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen3.7-max",
            "messages": [{"role": "user", "content": "帮我读取 /tmp/test.txt"}],
            "tools": [
                {
                    "name": "Read",
                    "description": "读取本地文件",
                    "input_schema": {
                        "type": "object",
                        "properties": {"file_path": {"type": "string"}},
                        "required": ["file_path"],
                    },
                }
            ],
            "stream": False,
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["stop_reason"] == "tool_use"
    tool_use_blocks = [b for b in data["content"] if b["type"] == "tool_use"]
    assert len(tool_use_blocks) == 1
    assert tool_use_blocks[0]["name"] == "Read"
    assert tool_use_blocks[0]["input"] == {"file_path": "/tmp/test.txt"}


def test_anthropic_count_tokens():
    client = TestClient(srv.app)
    r = client.post(
        "/v1/messages/count_tokens",
        json={
            "model": "claude-3-7-sonnet",
            "system": "你是一个智能编码助手",
            "messages": [{"role": "user", "content": "写一段 Python 代码计算斐波那契数列"}],
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert "input_tokens" in data
    assert data["input_tokens"] > 5


def test_anthropic_models_dispatch():
    client = TestClient(srv.app)
    # 携带 anthropic-version 头，返回 Anthropic 规范
    r_ant = client.get("/v1/models", headers={"anthropic-version": "2023-06-01"})
    assert r_ant.status_code == 200
    ant_data = r_ant.json()
    assert "data" in ant_data
    assert ant_data["has_more"] is False
    assert any(m["id"] == "claude-3-7-sonnet-20250219" for m in ant_data["data"])

    # 不携带该头，返回标准 OpenAI 格式
    r_oai = client.get("/v1/models")
    assert r_oai.status_code == 200
    oai_data = r_oai.json()
    assert oai_data["object"] == "list"
    assert any(m["id"] == "claude-3-7-sonnet-20250219" for m in oai_data["data"])


def test_anthropic_error_mapping():
    client = TestClient(srv.app)

    # 1. 未知模型 404
    r_404 = client.post(
        "/v1/messages",
        json={"model": "invalid-claude-model", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r_404.status_code == 404
    assert r_404.json()["type"] == "error"
    assert r_404.json()["error"]["type"] == "not_found_error"

    # 2. 空消息 400
    r_400 = client.post(
        "/v1/messages",
        json={"model": "claude-3-7-sonnet", "messages": []},
    )
    assert r_400.status_code == 400
    assert r_400.json()["type"] == "error"
    assert r_400.json()["error"]["type"] == "invalid_request_error"

    # 3. 上游未授权 401（非流式）
    srv._make_provider = lambda key: BoomProvider()
    r_401 = client.post(
        "/v1/messages",
        json={"model": "claude-3-7-sonnet", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r_401.status_code == 401
    assert r_401.json()["type"] == "error"
    assert r_401.json()["error"]["type"] == "authentication_error"

    # 4. 上游异常（流式内抛错，吐出 event: error）
    with client.stream(
        "POST",
        "/v1/messages",
        json={"model": "claude-3-7-sonnet", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    ) as r_stream_err:
        assert r_stream_err.status_code == 200
        text = "".join(r_stream_err.iter_text())
    assert "event: error" in text
    assert "authentication_error" in text


def test_openai_model_routing():
    # 1. 默认后端 (qwen)
    os.environ["OPENAI_BACKEND"] = "qwen"
    pk, wm = srv.resolve_model("gpt-4o")
    assert pk == "qwen" and wm == "Qwen3.7-Max"

    pk, wm = srv.resolve_model("gpt-4-turbo")
    assert pk == "qwen" and wm == "Qwen3.7-Max"

    pk, wm = srv.resolve_model("o1")
    assert pk == "qwen" and wm == "Qwen3.7-Max"

    pk, wm = srv.resolve_model("gpt-4o-mini")
    assert pk == "qwen" and wm == "Qwen3.6-Flash"

    pk, wm = srv.resolve_model("gpt-3.5-turbo")
    assert pk == "qwen" and wm == "Qwen3.6-Flash"

    pk, wm = srv.resolve_model("auto")
    assert pk == "qwen" and wm == "Qwen"

    # 2. DeepSeek 后端
    os.environ["OPENAI_BACKEND"] = "deepseek"
    pk, wm = srv.resolve_model("gpt-4o")
    assert pk == "deepseek" and wm == "deepseek-chat"

    pk, wm = srv.resolve_model("o1")
    assert pk == "deepseek" and wm == "deepseek-reasoner"

    pk, wm = srv.resolve_model("gpt-4o-mini")
    assert pk == "deepseek" and wm == "deepseek-chat"

    # 3. 豆包后端
    os.environ["OPENAI_BACKEND"] = "doubao"
    pk, wm = srv.resolve_model("gpt-4o")
    assert pk == "doubao" and wm == "doubao-pro"

    pk, wm = srv.resolve_model("o1")
    assert pk == "doubao" and wm == "doubao-think"

    pk, wm = srv.resolve_model("gpt-4o-mini")
    assert pk == "doubao" and wm == "doubao-lite"

    # 还原
    os.environ["OPENAI_BACKEND"] = "qwen"


def test_models_includes_openai_aliases():
    client = TestClient(srv.app)
    r = client.get("/v1/models")
    assert r.status_code == 200
    model_ids = [m["id"] for m in r.json()["data"]]
    assert "gpt-4o" in model_ids
    assert "gpt-4o-mini" in model_ids
    assert "o1" in model_ids
    assert "gpt-3.5-turbo" in model_ids
    assert "auto" in model_ids


class StatefulTrackingProvider:
    def __init__(self):
        self.deleted_conversations: list[str] = []
        self.last_conversation_id: str | None = None

    async def chat(self, messages, model, conversation_id=None, **kw):
        self.last_conversation_id = conversation_id
        cid = conversation_id or "tracked_conv_123"
        yield "测试回复", {"conversation_id": cid}

    async def delete_conversation(self, cid: str) -> bool:
        self.deleted_conversations.append(cid)
        return True


def test_anthropic_stateful_vs_stateless_lifecycle():
    import asyncio
    import time

    tracker = StatefulTrackingProvider()
    srv._make_provider = lambda key: tracker
    client = TestClient(srv.app)

    # 1. 无状态请求（未提供 conversation_id）：自动触发后台删除
    r_stateless = client.post(
        "/v1/messages",
        json={"model": "claude-3-7-sonnet", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r_stateless.status_code == 200
    assert tracker.last_conversation_id is None
    # 等待微任务协程派发
    time.sleep(0.05)
    assert "tracked_conv_123" in tracker.deleted_conversations

    # 2. 有状态请求（显式提供 conversation_id）：保留会话不删除，并透传给客户端
    tracker.deleted_conversations.clear()
    r_stateful = client.post(
        "/v1/messages",
        json={
            "model": "claude-3-7-sonnet",
            "conversation_id": "client_session_xyz",
            "messages": [{"role": "user", "content": "续聊问候"}],
        },
    )
    assert r_stateful.status_code == 200
    assert tracker.last_conversation_id == "client_session_xyz"
    time.sleep(0.05)
    # 不应被加入删除列表
    assert "client_session_xyz" not in tracker.deleted_conversations
    assert r_stateful.json().get("conversation_id") == "client_session_xyz"


class XmlToolCallingProvider:
    async def chat(self, *a, **kw):
        yield "分析完成，执行工具：\n<tool_call>\n", {}
        yield '{"name": "Edit", "arguments": {"file_path": "/tmp/a.py", "new_string": "x = 1"}}\n', {}
        yield "</tool_call>", {}


def test_anthropic_xml_tool_calling():
    srv._make_provider = lambda key: XmlToolCallingProvider()
    client = TestClient(srv.app)
    r = client.post(
        "/v1/messages",
        json={
            "model": "claude-3-7-sonnet",
            "messages": [{"role": "user", "content": "修改文件"}],
            "tools": [
                {
                    "name": "Edit",
                    "description": "修改文件",
                    "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}}},
                }
            ],
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["stop_reason"] == "tool_use"
    tool_blocks = [b for b in data["content"] if b["type"] == "tool_use"]
    assert len(tool_blocks) == 1
    assert tool_blocks[0]["name"] == "Edit"
    assert tool_blocks[0]["input"]["file_path"] == "/tmp/a.py"


def test_flatten_messages_prefill():
    from providers.base import flatten_messages

    # 普通单条
    assert flatten_messages([{"role": "user", "content": "hi"}]) == "hi"

    # 多轮，最后一条为 user
    m1 = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "how are you?"},
    ]
    res1 = flatten_messages(m1)
    assert res1.endswith("Assistant:")

    # 多轮，最后一条为 assistant（prefill 模式）
    m2 = [
        {"role": "user", "content": "输出 JSON"},
        {"role": "assistant", "content": "{"},
    ]
    res2 = flatten_messages(m2)
    assert res2.endswith("Assistant: {")
    assert not res2.endswith("Assistant: {\n\nAssistant:")


def test_openai_stream_options_usage():
    srv._make_provider = lambda key: EchoProvider()
    client = TestClient(srv.app)
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "qwen3.7-max",
            "messages": [{"role": "user", "content": "你好"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    ) as r:
        assert r.status_code == 200
        raw_text = "".join(r.iter_text())

    chunks = []
    for line in raw_text.splitlines():
        if line.startswith("data: ") and not line.endswith("[DONE]"):
            chunks.append(json.loads(line[len("data: ") :]))

    # 包含 choices 为空、带 usage 的 chunk
    usage_chunks = [c for c in chunks if "usage" in c and c.get("choices") == []]
    assert len(usage_chunks) == 1, "未找到有效的流式 usage 统计 chunk"
    u = usage_chunks[0]["usage"]
    assert "prompt_tokens" in u and "completion_tokens" in u and "total_tokens" in u
    assert u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"]


def test_openai_tool_calling_support():
    srv._make_provider = lambda key: ToolCallingProvider()
    client = TestClient(srv.app)

    # 1. 非流式带 tools
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "读取 /tmp/test.txt"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "Read",
                        "description": "读取本地文件",
                        "parameters": {
                            "type": "object",
                            "properties": {"file_path": {"type": "string"}},
                            "required": ["file_path"],
                        },
                    },
                }
            ],
            "stream": False,
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["choices"][0]["finish_reason"] == "tool_calls"
    tcs = data["choices"][0]["message"].get("tool_calls", [])
    assert len(tcs) == 1
    assert tcs[0]["function"]["name"] == "Read"
    args = json.loads(tcs[0]["function"]["arguments"])
    assert args["file_path"] == "/tmp/test.txt"

    # 2. 流式带 tools
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "读取 /tmp/test.txt"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "Read",
                        "description": "读取本地文件",
                        "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}}},
                    },
                }
            ],
            "stream": True,
        },
    ) as r_stream:
        assert r_stream.status_code == 200
        raw_text = "".join(r_stream.iter_text())

    stream_chunks = [
        json.loads(line[len("data: ") :])
        for line in raw_text.splitlines()
        if line.startswith("data: ") and not line.endswith("[DONE]")
    ]
    tool_chunks = [
        c for c in stream_chunks
        if c.get("choices") and c["choices"][0].get("delta", {}).get("tool_calls")
    ]
    assert len(tool_chunks) == 1
    assert tool_chunks[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "Read"


def test_retrieve_model_endpoint():
    client = TestClient(srv.app)
    # 合法模型
    r = client.get("/v1/models/qwen3.7-max")
    assert r.status_code == 200
    assert r.json()["id"] == "qwen3.7-max"
    assert r.json()["owned_by"] == "qwen"

    # OpenAI 兼容别名
    r_alias = client.get("/models/gpt-4o")
    assert r_alias.status_code == 200
    assert r_alias.json()["id"] == "gpt-4o"

    # 未知模型 404
    r_unknown = client.get("/v1/models/non-existent-xyz")
    assert r_unknown.status_code == 404
    assert r_unknown.json()["error"]["type"] == "model_not_found"


def test_embeddings_endpoint():
    client = TestClient(srv.app)
    # 单条文本输入
    r1 = client.post("/v1/embeddings", json={"input": "你好世界", "model": "text-embedding-3-small"})
    assert r1.status_code == 200
    d1 = r1.json()
    assert d1["object"] == "list"
    assert len(d1["data"]) == 1
    assert len(d1["data"][0]["embedding"]) == 1536
    assert d1["usage"]["total_tokens"] > 0

    # 批量输入与指定维度
    r2 = client.post("/embeddings", json={"input": ["文本A", "文本B"], "dimensions": 512})
    assert r2.status_code == 200
    d2 = r2.json()
    assert len(d2["data"]) == 2
    assert len(d2["data"][0]["embedding"]) == 512
    assert len(d2["data"][1]["embedding"]) == 512


def test_route_aliases():
    srv._make_provider = lambda key: EchoProvider()
    client = TestClient(srv.app)

    # 1. /chat/completions (无 /v1 前缀)
    r1 = client.post("/chat/completions", json={"model": "qwen", "messages": [{"role": "user", "content": "hi"}]})
    assert r1.status_code == 200

    # 2. /models (无 /v1 前缀)
    r2 = client.get("/models")
    assert r2.status_code == 200
    assert r2.json()["object"] == "list"

    # 3. /messages (无 /v1 前缀)
    r3 = client.post("/messages", json={"model": "claude-3-7-sonnet", "messages": [{"role": "user", "content": "hi"}]})
    assert r3.status_code == 200

    # 4. /messages/count_tokens (无 /v1 前缀)
    r4 = client.post("/messages/count_tokens", json={"model": "claude-3-7-sonnet", "messages": [{"role": "user", "content": "hi"}]})
    assert r4.status_code == 200
    assert "input_tokens" in r4.json()


class MultiToolCallingProvider:
    """输出多个工具调用代码块的假 Provider。"""
    async def chat(self, *a, **kw):
        yield "我将读取两个配置文件：\n```tool_call\n", {}
        yield '{"name": "Read", "arguments": {"file_path": "/tmp/a.txt"}}\n```\n', {}
        yield "接下来读取第二个：\n```tool_call\n", {}
        yield '{"name": "Read", "arguments": {"file_path": "/tmp/b.txt"}}\n```', {}


class ComplexAndEmptyToolCallingProvider:
    """输出空参数、尾随逗号与复杂嵌套 JSON 的假 Provider。"""
    async def chat(self, *a, **kw):
        yield "调用工具：\n```tool_call\n", {}
        yield (
            '{"name": "ComplexTool", "arguments": {'
            '"empty_obj": {}, '
            '"nested": {"levels": [1, 2, {"key": "val"}]}, '
            '"flag": true, '
            '"trailing": "comma", '
            '}}\n```\n'
        ), {}
        yield "空参数工具：\n```tool_call\n", {}
        yield '{"name": "NoArgsTool", "arguments": ""}\n```', {}


class CrashProvider:
    """抛出未预期的通用 Exception，用于验证 500 全局异常转换。"""
    async def chat(self, *a, **kw):
        raise RuntimeError("未预期的底层硬件故障")
        yield


def test_unified_exception_handling_pydantic_validation():
    client = TestClient(srv.app)

    # 1. Anthropic 协议下的缺失字段校验异常 -> 400 invalid_request_error
    r_ant = client.post("/v1/messages", json={"model": "claude-3-7-sonnet"})
    assert r_ant.status_code == 400
    assert r_ant.json()["type"] == "error"
    assert r_ant.json()["error"]["type"] == "invalid_request_error"
    assert "messages" in r_ant.json()["error"]["message"]

    # 2. OpenAI 协议下的缺失字段校验异常 -> 400 invalid_request_error
    r_oai = client.post("/v1/chat/completions", json={"model": "gpt-4o"})
    assert r_oai.status_code == 400
    assert "error" in r_oai.json()
    assert r_oai.json()["error"]["type"] == "invalid_request_error"
    assert "messages" in r_oai.json()["error"]["message"]

    # 3. 携带 anthropic-version 请求头的非法请求 -> 格式化为 Anthropic 错误格式
    r_header = client.post(
        "/v1/chat/completions",
        headers={"anthropic-version": "2023-06-01"},
        json={"model": "gpt-4o"},
    )
    assert r_header.status_code == 400
    assert r_header.json()["type"] == "error"
    assert r_header.json()["error"]["type"] == "invalid_request_error"


def test_unified_exception_handling_uncaught_500():
    srv._make_provider = lambda key: CrashProvider()
    client = TestClient(srv.app)

    # 1. Anthropic 端点未��获异常 -> 500 / 502 api_error
    r_ant = client.post(
        "/v1/messages",
        json={"model": "claude-3-7-sonnet", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r_ant.status_code in (500, 502)
    assert r_ant.json()["type"] == "error"
    assert r_ant.json()["error"]["type"] in ("api_error", "upstream_error")

    # 2. OpenAI 端点未捕获异常 -> 500 / 502 server_error
    r_oai = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r_oai.status_code in (500, 502)
    assert "error" in r_oai.json()
    assert r_oai.json()["error"]["type"] in ("server_error", "upstream_error")


def test_multi_tool_calling_openai():
    srv._make_provider = lambda key: MultiToolCallingProvider()
    client = TestClient(srv.app)

    tools_decl = [
        {
            "type": "function",
            "function": {
                "name": "Read",
                "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}}},
            },
        }
    ]

    # 1. 非流式：返回两个工具调用
    r = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "读取两个配置"}], "tools": tools_decl},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["choices"][0]["finish_reason"] == "tool_calls"
    tcs = data["choices"][0]["message"]["tool_calls"]
    assert len(tcs) == 2
    assert tcs[0]["function"]["name"] == "Read"
    assert json.loads(tcs[0]["function"]["arguments"]) == {"file_path": "/tmp/a.txt"}
    assert tcs[1]["function"]["name"] == "Read"
    assert json.loads(tcs[1]["function"]["arguments"]) == {"file_path": "/tmp/b.txt"}

    # 2. 流式：返回两个工具调用
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "读取两个配置"}], "tools": tools_decl, "stream": True},
    ) as r_stream:
        assert r_stream.status_code == 200
        raw_text = "".join(r_stream.iter_text())

    stream_chunks = [
        json.loads(line[len("data: ") :])
        for line in raw_text.splitlines()
        if line.startswith("data: ") and not line.endswith("[DONE]")
    ]
    tool_chunks = [
        c for c in stream_chunks
        if c.get("choices") and c["choices"][0].get("delta", {}).get("tool_calls")
    ]
    assert len(tool_chunks) == 1
    stream_tcs = tool_chunks[0]["choices"][0]["delta"]["tool_calls"]
    assert len(stream_tcs) == 2
    assert stream_tcs[0]["function"]["name"] == "Read"
    assert stream_tcs[1]["function"]["name"] == "Read"


def test_multi_tool_calling_anthropic():
    srv._make_provider = lambda key: MultiToolCallingProvider()
    client = TestClient(srv.app)

    tools_decl = [
        {
            "name": "Read",
            "description": "读取文件",
            "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}}},
        }
    ]

    # 1. 非流式
    r = client.post(
        "/v1/messages",
        json={"model": "claude-3-7-sonnet", "messages": [{"role": "user", "content": "读取配置"}], "tools": tools_decl},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["stop_reason"] == "tool_use"
    tool_blocks = [b for b in data["content"] if b["type"] == "tool_use"]
    assert len(tool_blocks) == 2
    assert tool_blocks[0]["name"] == "Read"
    assert tool_blocks[0]["input"] == {"file_path": "/tmp/a.txt"}
    assert tool_blocks[1]["name"] == "Read"
    assert tool_blocks[1]["input"] == {"file_path": "/tmp/b.txt"}

    # 2. 流式
    with client.stream(
        "POST",
        "/v1/messages",
        json={"model": "claude-3-7-sonnet", "messages": [{"role": "user", "content": "读取配置"}], "tools": tools_decl, "stream": True},
    ) as r_stream:
        assert r_stream.status_code == 200
        raw_text = "".join(r_stream.iter_text())

    events = []
    for block in raw_text.strip().split("\n\n"):
        if not block.strip():
            continue
        ev_name = None
        ev_data = None
        for line in block.split("\n"):
            if line.startswith("event: "):
                ev_name = line[len("event: ") :].strip()
            elif line.startswith("data: "):
                ev_data = json.loads(line[len("data: ") :].strip())
        if ev_name and ev_data:
            events.append((ev_name, ev_data))

    tool_use_starts = [
        e[1]["content_block"] for e in events
        if e[0] == "content_block_start" and e[1].get("content_block", {}).get("type") == "tool_use"
    ]
    assert len(tool_use_starts) == 2
    assert tool_use_starts[0]["name"] == "Read"
    assert tool_use_starts[1]["name"] == "Read"


def test_tool_calling_empty_and_complex_nested_json():
    srv._make_provider = lambda key: ComplexAndEmptyToolCallingProvider()
    client = TestClient(srv.app)

    tools_decl = [
        {"name": "ComplexTool", "input_schema": {"type": "object"}},
        {"name": "NoArgsTool", "input_schema": {"type": "object"}},
    ]

    r = client.post(
        "/v1/messages",
        json={"model": "claude-3-7-sonnet", "messages": [{"role": "user", "content": "调用工具"}], "tools": tools_decl},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["stop_reason"] == "tool_use"
    tool_blocks = [b for b in data["content"] if b["type"] == "tool_use"]
    assert len(tool_blocks) == 2
    # 验证复杂嵌套结构被完整解析，尾随逗号被正确容错
    complex_inp = tool_blocks[0]["input"]
    assert complex_inp["nested"]["levels"] == [1, 2, {"key": "val"}]
    assert complex_inp["flag"] is True
    assert complex_inp["trailing"] == "comma"
    # 验证空参数归一化为 {}
    assert tool_blocks[1]["input"] == {}


class ReasoningEchoProvider:
    """模拟输出思考过程与最终回复的假 Provider。"""
    async def chat(self, *a, **kw):
        yield "第一步思考，", {"reasoning": True}
        yield "第二步思考结束。", {"reasoning": True}
        yield "这是正式回复内容。", {}
        yield "", {"conversation_id": "test_conv_id"}


def test_openai_reasoning_streaming_and_non_streaming():
    """测试 OpenAI 接口下的思考透传（流式 reasoning_content 与非流式字段聚合）。"""
    srv._make_provider = lambda key: ReasoningEchoProvider()
    client = TestClient(srv.app)

    # 1. 流式测试
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "deepseek-reasoner",
            "messages": [{"role": "user", "content": "测试思考"}],
            "stream": True,
        },
    ) as r:
        assert r.status_code == 200
        raw_text = "".join(r.iter_text())

    chunks = []
    for line in raw_text.splitlines():
        if line.startswith("data: ") and not line.endswith("[DONE]"):
            chunks.append(json.loads(line[len("data: ") :]))

    reasoning_parts = []
    content_parts = []
    for c in chunks:
        delta = c["choices"][0].get("delta", {})
        if "reasoning_content" in delta:
            reasoning_parts.append(delta["reasoning_content"])
        if "content" in delta and delta["content"]:
            content_parts.append(delta["content"])

    assert "".join(reasoning_parts) == "第一步思考，第二步思考结束。"
    assert "".join(content_parts) == "这是正式回复内容。"

    # 2. 非流式测试
    r_non_stream = client.post(
        "/v1/chat/completions",
        json={
            "model": "deepseek-reasoner",
            "messages": [{"role": "user", "content": "测试思考"}],
            "stream": False,
        },
    )
    assert r_non_stream.status_code == 200
    data = r_non_stream.json()
    msg = data["choices"][0]["message"]
    assert msg["reasoning_content"] == "第一步思考，第二步思考结束。"
    assert msg["content"] == "这是正式回复内容。"
    assert data.get("conversation_id") == "test_conv_id"
    print("[PASS] server: OpenAI 深度思考流式 choices[0].delta.reasoning_content 与非流式 message.reasoning_content 透传")


def test_anthropic_thinking_state_machine_streaming_and_non_streaming():
    """测试 Anthropic Messages 协议下思考状态机流式与非流式 content block 规范。"""
    srv._make_provider = lambda key: ReasoningEchoProvider()
    client = TestClient(srv.app)

    # 1. 流式测试：验证 Thinking -> Text 状态机流转与严格 index 递增
    with client.stream(
        "POST",
        "/v1/messages",
        headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
        json={
            "model": "claude-3-7-sonnet",
            "messages": [{"role": "user", "content": "测试思考过程"}],
            "stream": True,
        },
    ) as r:
        assert r.status_code == 200
        raw_text = "".join(r.iter_text())

    events: list[tuple[str, dict]] = []
    for block in raw_text.strip().split("\n\n"):
        if not block.strip():
            continue
        ev_name = None
        ev_data = None
        for line in block.split("\n"):
            if line.startswith("event: "):
                ev_name = line[len("event: ") :].strip()
            elif line.startswith("data: "):
                ev_data = json.loads(line[len("data: ") :].strip())
        if ev_name and ev_data:
            events.append((ev_name, ev_data))

    event_names = [e[0] for e in events]
    expected_order = [
        "message_start",
        "content_block_start",  # block 0 (thinking)
        "content_block_delta",  # delta 0
        "content_block_delta",  # delta 0
        "content_block_stop",   # stop 0
        "content_block_start",  # block 1 (text)
        "content_block_delta",  # delta 1
        "content_block_stop",   # stop 1
        "message_delta",
        "message_stop",
    ]
    assert event_names == expected_order, f"事件序列不匹配: {event_names}"

    # 检验 thinking block 0
    cb_start_0 = events[1][1]
    assert cb_start_0["index"] == 0
    assert cb_start_0["content_block"]["type"] == "thinking"

    th_delta_0 = events[2][1]
    assert th_delta_0["index"] == 0
    assert th_delta_0["delta"]["type"] == "thinking_delta"
    assert th_delta_0["delta"]["thinking"] == "第一步思考，"

    th_delta_1 = events[3][1]
    assert th_delta_1["index"] == 0
    assert th_delta_1["delta"]["type"] == "thinking_delta"
    assert th_delta_1["delta"]["thinking"] == "第二步思考结束。"

    assert events[4][1]["index"] == 0  # content_block_stop 0

    # 检验 text block 1
    cb_start_1 = events[5][1]
    assert cb_start_1["index"] == 1
    assert cb_start_1["content_block"]["type"] == "text"

    txt_delta = events[6][1]
    assert txt_delta["index"] == 1
    assert txt_delta["delta"]["type"] == "text_delta"
    assert txt_delta["delta"]["text"] == "这是正式回复内容。"

    assert events[7][1]["index"] == 1  # content_block_stop 1

    msg_delta = events[8][1]
    assert msg_delta["delta"]["stop_reason"] == "end_turn"

    # 2. 非流式测试：验证 content 数组优先包含 thinking block，再包含 text block
    r_non_stream = client.post(
        "/v1/messages",
        headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
        json={
            "model": "claude-3-7-sonnet",
            "messages": [{"role": "user", "content": "测试思考过程"}],
            "stream": False,
        },
    )
    assert r_non_stream.status_code == 200
    data_ant = r_non_stream.json()
    blocks = data_ant["content"]
    assert len(blocks) == 2
    assert blocks[0]["type"] == "thinking"
    assert blocks[0]["thinking"] == "第一步思考，第二步思考结束。"
    assert blocks[1]["type"] == "text"
    assert blocks[1]["text"] == "这是正式回复内容。"
    assert data_ant.get("conversation_id") == "test_conv_id"
    print("[PASS] server: Anthropic 思考状态机流式 content_block_start(thinking)->content_block_start(text) 与非流式 content 结构验证")


def test_peer_to_peer_mutual_failover():
    """验证五大模型对等互备（Mutual Peer Failover）：
    当第一顺位主节点抛错（如 401/429/502）时，网关动态无感切换至同能力环备选节点，并附带 failover 元数据。
    """
    class FailingPrimaryProvider:
        async def chat(self, *a, **kw):
            raise ProviderError("主节点网络超时或令牌过期", status=401, err_type="auth_error")
            yield

    class HealthyFallbackProvider:
        def __init__(self, name: str):
            self.name = name

        async def chat(self, *a, **kw):
            yield f"来自后备节点[{self.name}]的回复", {"conversation_id": f"conv_{self.name}"}

    def custom_factory(key: str):
        if key == "kimi":
            return FailingPrimaryProvider()
        return HealthyFallbackProvider(key)

    srv._make_provider = custom_factory
    # 重置冷却与连续失败记录，确保后备节点立即可用
    srv.failover_router.cooldown_tracker.reset()

    client = TestClient(srv.app)

    # 1. OpenAI 接口容灾测试（指定 kimi 为第一顺位）
    srv.failover_router.cooldown_tracker.reset()
    if hasattr(srv, "load_balancer"):
        srv.load_balancer.reset()
    r_openai = client.post(
        "/v1/chat/completions",
        json={"model": "kimi", "messages": [{"role": "user", "content": "你好"}]},
    )
    assert r_openai.status_code == 200, r_openai.text
    data_oa = r_openai.json()
    assert "来自后备节点" in data_oa["choices"][0]["message"]["content"]
    assert "failover" in data_oa
    assert data_oa["failover"]["from_provider"] == "kimi"
    assert r_openai.headers.get("x-failover-from") == "kimi"
    print(f"[PASS] 对等互备 (OpenAI 协议): {data_oa['failover']['from_provider']} -> {data_oa['failover']['to_provider']}")

    # 2. Anthropic 接口容灾测试（指定 kimi 为第一顺位）
    srv.failover_router.cooldown_tracker.reset()
    if hasattr(srv, "load_balancer"):
        srv.load_balancer.reset()
    r_ant = client.post(
        "/v1/messages",
        headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
        json={"model": "kimi", "messages": [{"role": "user", "content": "你好"}]},
    )
    assert r_ant.status_code == 200, r_ant.text
    data_ant = r_ant.json()
    assert "来自后备节点" in data_ant["content"][0]["text"]
    assert "failover" in data_ant
    assert data_ant["failover"]["from_provider"] == "kimi"
    assert r_ant.headers.get("x-failover-from") == "kimi"
    print(f"[PASS] 对等互备 (Anthropic 协议): {data_ant['failover']['from_provider']} -> {data_ant['failover']['to_provider']}")

    # 3. Responses (Codex CLI) 接口容灾测试
    srv.failover_router.cooldown_tracker.reset()
    if hasattr(srv, "load_balancer"):
        srv.load_balancer.reset()
    r_resp = client.post(
        "/v1/responses",
        json={"model": "kimi", "input": "生成代码"},
    )
    assert r_resp.status_code == 200, r_resp.text
    data_resp = r_resp.json()
    assert "来自后备节点" in data_resp["output"][0]["content"][0]["text"]
    assert "failover" in data_resp
    assert data_resp["failover"]["from_provider"] == "kimi"
    assert r_resp.headers.get("x-failover-from") == "kimi"
    print(f"[PASS] 对等互备 (Responses 协议): {data_resp['failover']['from_provider']} -> {data_resp['failover']['to_provider']}")


def test_kimi_and_glm_model_routing():
    """测试 Kimi 和 智谱 GLM 的模型路由与别名映射。"""
    # 1. Kimi 模型路由
    pk, wm = srv.resolve_model("kimi")
    assert pk == "kimi" and wm == "kimi"

    pk, wm = srv.resolve_model("kimi-explore")
    assert pk == "kimi" and wm == "kimi-explore"

    pk, wm = srv.resolve_model("kimi-k1")
    assert pk == "kimi" and wm == "kimi-explore"

    pk, wm = srv.resolve_model("moonshot-v1-8k")
    assert pk == "kimi" and wm == "kimi"

    # 2. GLM 模型路由
    pk, wm = srv.resolve_model("glm")
    assert pk == "glm" and wm == "glm-4-plus"

    pk, wm = srv.resolve_model("glm-4-plus")
    assert pk == "glm" and wm == "glm-4-plus"

    pk, wm = srv.resolve_model("glm-4-flash")
    assert pk == "glm" and wm == "glm-4-flash"

    pk, wm = srv.resolve_model("glm-zero-preview")
    assert pk == "glm" and wm == "glm-zero-preview"

    pk, wm = srv.resolve_model("chatglm")
    assert pk == "glm" and wm == "glm-4-plus"

    # 3. 环境变量后端路由切换测试
    os.environ["CLAUDE_BACKEND"] = "kimi"
    pk, wm = srv.resolve_model("claude-3-7-sonnet")
    assert pk == "kimi" and wm == "kimi"
    pk, wm = srv.resolve_model("claude-3-opus")
    assert pk == "kimi" and wm == "kimi-explore"

    os.environ["CLAUDE_BACKEND"] = "glm"
    pk, wm = srv.resolve_model("claude-3-7-sonnet")
    assert pk == "glm" and wm == "glm-4-plus"
    pk, wm = srv.resolve_model("claude-3-5-haiku")
    assert pk == "glm" and wm == "glm-4-flash"

    os.environ["CLAUDE_BACKEND"] = "qwen"
    print("[PASS] Kimi 和 智谱 GLM 模型路由解析验证通过")


def test_diagnostics_endpoint():
    """验证 GET /v1/diagnostics 与 /diagnostics 诊断接口：
    包含全部 5 个提供方状态、三级能力环、自愈状态、智能体 CLI 和 HTTP 连接池配置。
    """
    client = TestClient(srv.app)

    for endpoint in ("/v1/diagnostics", "/diagnostics"):
        r = client.get(endpoint)
        assert r.status_code == 200, f"{endpoint} 返回状态码 {r.status_code}: {r.text}"
        data = r.json()
        assert data["status"] == "ok"
        assert "timestamp" in data

        # 1. 验证 5 大 Provider
        providers = data["providers"]
        for p in ("qwen", "deepseek", "doubao", "kimi", "glm"):
            assert p in providers, f"缺少 Provider: {p}"
            p_info = providers[p]
            assert p_info["health_status"] in ("healthy", "degraded", "failing", "unhealthy")
            assert p_info["risk_level"] in ("HIGH", "MEDIUM", "LOW")
            assert isinstance(p_info["in_cooldown"], bool)
            assert isinstance(p_info["remaining_cooldown_s"], (float, int))
            assert isinstance(p_info["consecutive_failures"], int)
            assert isinstance(p_info["min_call_interval"], (float, int))
            assert p_info["min_call_interval"] > 0

        # 2. 验证三级对等能力环
        rings = data["capability_rings"]
        assert isinstance(rings, list)
        ring_names = [r["ring"] for r in rings]
        assert "reasoning" in ring_names
        assert "flagship" in ring_names
        assert "speed" in ring_names
        for r_entry in rings:
            assert "models" in r_entry and len(r_entry["models"]) > 0
            assert "members" in r_entry and len(r_entry["members"]) > 0

        # 3. 验证自愈状态
        healing = data["healing_status"]
        assert "last_check_timestamp" in healing
        assert "check_count" in healing
        assert "current_healing_task" in healing

        # 4. 验证智能体 CLI 信息
        agent = data["agent_cli"]
        assert "name" in agent
        assert "path" in agent
        assert "available" in agent

        # 5. 验证 HTTP 连接池配置与状态
        pool = data["http_pool"]
        assert pool["max_connections"] >= 100
        assert pool["max_keepalive_connections"] >= 40
        assert pool["keepalive_expiry"] >= 30.0
        assert "status" in pool
        assert "active_connections" in pool

        # 6. 验证自适应负载均衡器统计指标 (load_balancer)
        assert "load_balancer" in data
        lb = data["load_balancer"]
        assert "providers" in lb
        assert "total_requests" in lb
        assert "high_watermark_ratio" in lb
        assert "total_active_concurrency" in lb
        assert "total_qps" in lb
        for p in ("qwen", "deepseek", "doubao", "kimi", "glm"):
            assert p in lb["providers"]
            p_metric = lb["providers"][p]
            assert "active_concurrency" in p_metric
            assert "max_concurrency" in p_metric
            assert "concurrency_ratio" in p_metric
            assert "avg_ttft_ms" in p_metric
            assert "error_rate" in p_metric

        # 7. 验证 disabled_providers 列表
        assert "disabled_providers" in data
        assert isinstance(data["disabled_providers"], list)

        # 8. 验证 prompt_optimizer 统计指标
        assert "prompt_optimizer" in data
        po = data["prompt_optimizer"]
        assert "total_inspections" in po
        assert "total_folded_requests" in po
        assert "total_saved_tokens" in po
        assert "max_tokens" in po
        assert "preserve_recent_turns" in po
        assert "folding_enabled" in po

    print("[PASS] server: GET /v1/diagnostics 与 /diagnostics 诊断指标接口验证通过")


def test_diagnostics_with_disabled_provider():
    """验证当环境变量设置 DISABLED_PROVIDERS 时，/v1/diagnostics 暴露被禁用节点列表及对应节点状态；
    并在动态解禁后，状态自愈恢复，不发生状态粘滞。
    """
    os.environ["DISABLED_PROVIDERS"] = "deepseek"
    srv.failover_router.cooldown_tracker.reset()
    client = TestClient(srv.app)
    try:
        # 1. 禁用状态校验
        r = client.get("/v1/diagnostics")
        assert r.status_code == 200
        data = r.json()
        assert "deepseek" in data["disabled_providers"]
        assert data["providers"]["deepseek"]["health_status"] == "disabled"
    finally:
        os.environ.pop("DISABLED_PROVIDERS", None)
        srv.failover_router.cooldown_tracker.reset()

    # 2. 动态解禁后校验（验证状态粘滞消除，健康状态恢复）
    r_after = client.get("/v1/diagnostics")
    assert r_after.status_code == 200
    data_after = r_after.json()
    assert "deepseek" not in data_after["disabled_providers"]
    assert data_after["providers"]["deepseek"]["health_status"] != "disabled"
    assert data_after["providers"]["deepseek"]["health_status"] in ("healthy", "degraded", "failing", "unhealthy")
    print("[PASS] server: /v1/diagnostics 节点禁用与动态解禁自愈恢复校验通过")


def test_reports_latest_endpoint():
    """验证 GET /v1/reports/latest 与 /reports/latest：
    返回最新生成的项目优化空间建议书 JSON 结构。
    """
    client = TestClient(srv.app)

    for endpoint in ("/v1/reports/latest", "/reports/latest"):
        r = client.get(endpoint)
        assert r.status_code == 200, f"{endpoint} 返回状态码 {r.status_code}: {r.text}"
        data = r.json()
        assert "title" in data
        assert "optimization_opportunities" in data
        assert isinstance(data["optimization_opportunities"], list)
        assert len(data["optimization_opportunities"]) > 0
        opp = data["optimization_opportunities"][0]
        assert "id" in opp
        assert "title" in opp
        assert "priority" in opp
        assert "description" in opp

    print("[PASS] server: GET /v1/reports/latest 与 /reports/latest 优化报告接口验证通过")


def test_disabled_provider_zero_latency_switch():
    """测试当某个提供方被禁用 (DISABLED_PROVIDERS) 时：
    1. 首选被禁节点在主入口被 0ms 内部瞬时重定向至同能力环健康备用节点（如 deepseek -> qwen/doubao）；
    2. 被禁节点不产生任何实际调用，杜绝 401/429 无效网络惩罚；
    3. 响应头透传调度决策：
       - x-load-balancing-action (shedded)
       - x-load-balancing-reason
       - x-dispatched-provider
       - x-dispatched-model
    4. 覆盖 OpenAI (/v1/chat/completions)、Anthropic (/v1/messages) 与 Responses (/v1/responses) 协议，
       以及流式和非流式两种模式。
    """
    os.environ["DISABLED_PROVIDERS"] = "deepseek"
    srv.failover_router.cooldown_tracker.reset()
    if hasattr(srv, "load_balancer"):
        srv.load_balancer.cooldown_tracker.reset()
        srv.load_balancer.reset()

    called_providers: list[str] = []

    def mock_factory(key: str):
        called_providers.append(key)
        if key == "deepseek":
            raise RuntimeError("Fatal: DeepSeek should NOT be called when disabled!")
        return EchoProvider()

    orig_factory = srv._make_provider
    srv._make_provider = mock_factory

    try:
        client = TestClient(srv.app)

        # 1. OpenAI 协议非流式
        r_chat = client.post(
            "/v1/chat/completions",
            json={"model": "deepseek-chat", "messages": [{"role": "user", "content": "你好"}]},
        )
        assert r_chat.status_code == 200, r_chat.text
        assert r_chat.headers.get("x-load-balancing-action") == "shedded"
        dispatched_chat = r_chat.headers.get("x-dispatched-provider")
        assert dispatched_chat in ("qwen", "doubao", "glm", "kimi")
        assert dispatched_chat != "deepseek"
        assert r_chat.headers.get("x-dispatched-model") is not None
        assert r_chat.headers.get("x-load-balancing-reason") is not None
        assert "deepseek" not in called_providers

        # 2. OpenAI 协议流式
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={"model": "deepseek-chat", "messages": [{"role": "user", "content": "你好"}], "stream": True},
        ) as r_stream:
            assert r_stream.status_code == 200
            assert r_stream.headers.get("x-load-balancing-action") == "shedded"
            assert r_stream.headers.get("x-dispatched-provider") in ("qwen", "doubao", "glm", "kimi")
            assert r_stream.headers.get("x-dispatched-provider") != "deepseek"
            text = "".join(r_stream.iter_text())
            assert "你好" in text
            assert "deepseek" not in called_providers

        # 3. Anthropic 协议非流式
        r_ant = client.post(
            "/v1/messages",
            headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
            json={"model": "deepseek-chat", "messages": [{"role": "user", "content": "你好"}]},
        )
        assert r_ant.status_code == 200, r_ant.text
        assert r_ant.headers.get("x-load-balancing-action") == "shedded"
        assert r_ant.headers.get("x-dispatched-provider") != "deepseek"
        assert "deepseek" not in called_providers

        # 4. Anthropic 协议流式
        with client.stream(
            "POST",
            "/v1/messages",
            headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
            json={"model": "deepseek-chat", "messages": [{"role": "user", "content": "你好"}], "stream": True},
        ) as r_ant_stream:
            assert r_ant_stream.status_code == 200
            assert r_ant_stream.headers.get("x-load-balancing-action") == "shedded"
            assert r_ant_stream.headers.get("x-dispatched-provider") != "deepseek"
            text_ant = "".join(r_ant_stream.iter_text())
            assert "你好" in text_ant
            assert "deepseek" not in called_providers

        # 5. Responses (Codex CLI) 协议非流式
        r_resp = client.post(
            "/v1/responses",
            json={"model": "deepseek-chat", "input": "生成一段测试代码"},
        )
        assert r_resp.status_code == 200, r_resp.text
        assert r_resp.headers.get("x-load-balancing-action") == "shedded"
        assert r_resp.headers.get("x-dispatched-provider") != "deepseek"
        assert "deepseek" not in called_providers

        # 6. Responses (Codex CLI) 协议流式
        with client.stream(
            "POST",
            "/v1/responses",
            json={"model": "deepseek-chat", "input": "生成一段测试代码", "stream": True},
        ) as r_resp_stream:
            assert r_resp_stream.status_code == 200
            assert r_resp_stream.headers.get("x-load-balancing-action") == "shedded"
            assert r_resp_stream.headers.get("x-dispatched-provider") != "deepseek"
            text_resp = "".join(r_resp_stream.iter_text())
            assert "你好" in text_resp
            assert "deepseek" not in called_providers

        print("[PASS] test_disabled_provider_zero_latency_switch: 节点禁用瞬时转移与响应头透传验证通过")
    finally:
        os.environ.pop("DISABLED_PROVIDERS", None)
        srv.failover_router.cooldown_tracker.reset()
        srv._make_provider = orig_factory


def test_adaptive_load_balancer_metrics_and_degradation():
    """测试自适应负载均衡调度：
    1. 高水位与轻量降级 (degraded)：旗舰模型轻量 Prompt 遇到高负载时自适应降级至 SPEED 环；
    2. 生命周期追踪：acquire / release 并发计数平衡（调用后保底归零）；
    3. TTFT 首包耗时成功记录入滑动窗口统计器。
    """
    srv._make_provider = lambda key: EchoProvider()
    client = TestClient(srv.app)

    # 1. 正常路由 (normal)
    r_norm = client.post(
        "/v1/chat/completions",
        json={"model": "qwen", "messages": [{"role": "user", "content": "你好"}]},
    )
    assert r_norm.status_code == 200
    assert r_norm.headers.get("x-load-balancing-action") == "normal"
    assert r_norm.headers.get("x-dispatched-provider") == "qwen"

    # 2. 模拟高负载触发轻量降级 (degraded)
    # 将 qwen 并发拉高超过 high_watermark
    qwen_metrics = srv.load_balancer.get_metrics("qwen")
    for _ in range(8):
        qwen_metrics.acquire()

    try:
        # 短 Prompt (<300 tokens) 且无 thinking，请求旗舰模型 qwen-max
        r_deg = client.post(
            "/v1/chat/completions",
            json={"model": "qwen-max", "messages": [{"role": "user", "content": "1+1=?"}]},
        )
        assert r_deg.status_code == 200
        assert r_deg.headers.get("x-load-balancing-action") == "degraded"
        # 降级至 SPEED 环模型
        dispatched_model = r_deg.headers.get("x-dispatched-model")
        assert dispatched_model in ("Qwen3.6-Flash", "doubao-lite", "glm-4-flash")
    finally:
        for _ in range(8):
            qwen_metrics.release()

    # 3. 验证并发计数归零
    for pk in ("qwen", "doubao", "kimi", "glm", "deepseek"):
        assert srv.load_balancer.get_metrics(pk).active_concurrency == 0

    # 4. 验证 TTFT 耗时记录入快照
    snap = qwen_metrics.get_snapshot()
    assert snap.avg_ttft_ms >= 0.0

    print("[PASS] test_adaptive_load_balancer_metrics_and_degradation: 自适应降级与生命周期度量验证通过")


def test_prompt_folding_headers_and_truncation():
    """测试提示词自适应压缩与历史折叠响应头透传与截断功能：
    1. 正常短会话：返回 x-prompt-folding-applied: false，saved-tokens: 0；
    2. 中间包含超长工具调用日志的长会话：触发阶段一截断，返回 x-prompt-folding-applied: true，且 saved-tokens > 0；
    3. 覆盖 OpenAI (/v1/chat/completions)、Anthropic (/v1/messages) 与 Responses (/v1/responses)；
    4. 环境变量 GATEWAY_DISABLE_PROMPT_FOLDING=1 零损耗穿透。
    """
    srv._make_provider = lambda key: EchoProvider()
    client = TestClient(srv.app)

    # 1. 正常短对话
    r_short = client.post(
        "/v1/chat/completions",
        json={"model": "qwen", "messages": [{"role": "user", "content": "你好"}]},
    )
    assert r_short.status_code == 200
    assert r_short.headers.get("x-prompt-folding-applied") == "false"
    assert r_short.headers.get("x-prompt-folding-saved-tokens") == "0"

    # 2. 构造包含超长中间 tool 输出的对话
    huge_tool_data = "Log line " * 800  # ~7200 字符
    long_messages = [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "Please check the build logs."},
        {
            "role": "assistant",
            "content": "Checking logs...",
            "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "Bash"}}],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": huge_tool_data},
        {"role": "assistant", "content": "I see the build logs."},
        {"role": "user", "content": "Question 1"},
        {"role": "assistant", "content": "Answer 1"},
        {"role": "user", "content": "What is the final status?"},
    ]

    # 临时调小 max_tokens 以便在测试中稳定触发折叠
    orig_max_tokens = srv.prompt_optimizer.max_tokens
    srv.prompt_optimizer.max_tokens = 500
    try:
        # OpenAI 协议测试
        r_folded_chat = client.post(
            "/v1/chat/completions",
            json={"model": "qwen", "messages": long_messages},
        )
        assert r_folded_chat.status_code == 200
        assert r_folded_chat.headers.get("x-prompt-folding-applied") == "true"
        saved = int(r_folded_chat.headers.get("x-prompt-folding-saved-tokens", "0"))
        assert saved > 0
        orig_toks = int(r_folded_chat.headers.get("x-prompt-folding-original-tokens", "0"))
        final_toks = int(r_folded_chat.headers.get("x-prompt-folding-final-tokens", "0"))
        assert orig_toks > final_toks
        assert orig_toks - final_toks == saved

        # Anthropic 协议测试
        ant_messages = [
            {"role": "user", "content": "Please check the build logs."},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Running bash command..."},
                    {"type": "tool_use", "id": "tu_1", "name": "Bash", "input": {"cmd": "build"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "tu_1", "content": huge_tool_data}
                ],
            },
            {"role": "assistant", "content": "I see the result."},
            {"role": "user", "content": "Recent 1"},
            {"role": "assistant", "content": "Answer 1"},
            {"role": "user", "content": "What is the result?"},
        ]
        r_ant = client.post(
            "/v1/messages",
            headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
            json={
                "model": "claude-3-7-sonnet",
                "system": "You are a coding agent.",
                "messages": ant_messages,
            },
        )
        assert r_ant.status_code == 200
        assert r_ant.headers.get("x-prompt-folding-applied") == "true"
        assert int(r_ant.headers.get("x-prompt-folding-saved-tokens", "0")) > 0

        # Responses 协议测试
        resp_input = [
            {"type": "message", "role": "user", "content": "Please check build"},
            {"type": "message", "role": "assistant", "content": "Running..."},
            {"type": "message", "role": "tool", "content": huge_tool_data},
            {"type": "message", "role": "assistant", "content": "Done."},
            {"type": "message", "role": "user", "content": "Query 1"},
            {"type": "message", "role": "assistant", "content": "Answer 1"},
            {"type": "message", "role": "user", "content": "Summary?"},
        ]
        r_resp = client.post(
            "/v1/responses",
            json={"model": "gpt-4o", "input": resp_input},
        )
        assert r_resp.status_code == 200
        assert r_resp.headers.get("x-prompt-folding-applied") == "true"
        assert int(r_resp.headers.get("x-prompt-folding-saved-tokens", "0")) > 0

        # 环境变量禁用折叠测试
        os.environ["GATEWAY_DISABLE_PROMPT_FOLDING"] = "1"
        try:
            r_disabled = client.post(
                "/v1/chat/completions",
                json={"model": "qwen", "messages": long_messages},
            )
            assert r_disabled.status_code == 200
            assert r_disabled.headers.get("x-prompt-folding-applied") == "false"
            assert r_disabled.headers.get("x-prompt-folding-saved-tokens") == "0"
        finally:
            os.environ.pop("GATEWAY_DISABLE_PROMPT_FOLDING", None)

    finally:
        srv.prompt_optimizer.max_tokens = orig_max_tokens

    print("[PASS] test_prompt_folding_headers_and_truncation: 自适应折叠与全协议响应头透传验证通过")


if __name__ == "__main__":
    test_unknown_model_404()
    test_empty_messages_400()
    test_upstream_error_json()
    test_upstream_error_sse()
    test_stream_sse_structure()
    test_doubao_routing_and_models()
    test_claude_model_routing()
    test_openai_model_routing()
    test_models_includes_openai_aliases()
    test_anthropic_non_streaming()
    test_anthropic_streaming_sse_sequence()
    test_anthropic_count_tokens()
    test_anthropic_models_dispatch()
    test_anthropic_error_mapping()
    test_anthropic_tool_calling_streaming()
    test_anthropic_tool_calling_non_streaming()
    test_anthropic_stateful_vs_stateless_lifecycle()
    test_anthropic_xml_tool_calling()
    test_flatten_messages_prefill()
    test_openai_stream_options_usage()
    test_openai_tool_calling_support()
    test_retrieve_model_endpoint()
    test_embeddings_endpoint()
    test_route_aliases()
    test_unified_exception_handling_pydantic_validation()
    test_unified_exception_handling_uncaught_500()
    test_multi_tool_calling_openai()
    test_multi_tool_calling_anthropic()
    test_tool_calling_empty_and_complex_nested_json()
    test_openai_reasoning_streaming_and_non_streaming()
    test_anthropic_thinking_state_machine_streaming_and_non_streaming()
    test_kimi_and_glm_model_routing()
    test_peer_to_peer_mutual_failover()
    test_diagnostics_with_disabled_provider()
    test_diagnostics_endpoint()
    test_reports_latest_endpoint()
    test_disabled_provider_zero_latency_switch()
    test_adaptive_load_balancer_metrics_and_degradation()
    test_prompt_folding_headers_and_truncation()
    print("服务端集成测试全部通过")
