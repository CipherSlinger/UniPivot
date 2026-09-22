"""OpenAI Responses API (/v1/responses) 离线单测。

覆盖功能：
1. 请求格式转译（字符串 input、复杂 content block 数组、instructions、tools、function_call_output）
2. 非流式响应结构与 usage 统计
3. 流式 SSE 完整事件时序：
   response.created
   -> response.output_item.added
   -> response.content_part.added
   -> response.text.delta
   -> response.content_part.done
   -> response.output_item.done
   -> response.completed
   -> [DONE]
4. 工具调用模式（非流式与流式 SSE）
5. 错误转译与边界情况（未知模型、空请求）

运行: .venv/bin/python tests/test_responses.py
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
from providers.responses_compat import (  # noqa: E402
    ResponsesRequest,
    convert_responses_to_gateway_messages,
    extract_responses_content,
    format_responses_completion,
)


class EchoProvider:
    """回显 Provider 用于测试标准文本流与元数据传递。"""

    async def chat(self, *a, **kw):
        yield "Hello", {}
        yield "! How can I help you?", {"conversation_id": "conv_resp_123"}


class ReasoningEchoProvider:
    """包含深度思考过程与回答的 Provider。"""

    async def chat(self, *a, **kw):
        yield "Thinking process...", {"reasoning": True}
        yield "Here is the answer.", {"conversation_id": "conv_resp_reasoning"}


class ToolCallingProvider:
    """输出符合工具调用标记的 Provider。"""

    async def chat(self, *a, **kw):
        yield "Let me run the code for you:\n```tool_call\n", {}
        yield '{"name": "bash", "arguments": {"command": "echo hello"}}\n', {}
        yield "```", {"conversation_id": "conv_resp_tools"}


class BoomProvider:
    """抛出 ProviderError 模拟上游鉴权或网络异常。"""

    async def chat(self, *a, **kw):
        raise ProviderError("上游令牌失效", status=401, err_type="auth_error")
        yield


def test_convert_responses_request_string_input():
    req = ResponsesRequest(
        model="gpt-4o",
        input="Hello world",
        instructions="You are a helpful coding assistant.",
    )
    messages = convert_responses_to_gateway_messages(req)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are a helpful coding assistant."
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Hello world"


def test_convert_responses_request_complex_input_blocks():
    req = ResponsesRequest(
        model="o1",
        instructions=[{"type": "text", "text": "Follow system rules."}],
        input=[
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "What is 2+2?"}],
            },
            {
                "type": "message",
                "role": "assistant",
                "content": "Let me calculate.",
            },
            {
                "type": "function_call",
                "name": "calculator",
                "arguments": {"expr": "2+2"},
            },
            {
                "type": "function_call_output",
                "call_id": "call_calc_1",
                "output": {"result": 4},
            },
        ],
    )
    messages = convert_responses_to_gateway_messages(req)
    assert len(messages) == 5
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "Follow system rules."
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "What is 2+2?"
    assert messages[2]["role"] == "assistant"
    assert messages[2]["content"] == "Let me calculate."
    assert messages[3]["role"] == "assistant"
    assert "```tool_call" in messages[3]["content"]
    assert '"name": "calculator"' in messages[3]["content"]
    assert messages[4]["role"] == "user"
    assert "[Tool Result for call_calc_1]" in messages[4]["content"]
    assert '"result": 4' in messages[4]["content"]


def test_convert_responses_request_tools():
    req = ResponsesRequest(
        model="qwen3.7-max",
        input="List files",
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "list_files",
                    "description": "List directory contents",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                        },
                        "required": ["path"],
                    },
                },
            }
        ],
    )
    messages = convert_responses_to_gateway_messages(req)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "## Tools Available" in messages[0]["content"]
    assert "list_files(path: string (required))" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "List files"


def test_non_streaming_responses_endpoint():
    srv._make_provider = lambda key: EchoProvider()
    client = TestClient(srv.app)
    resp = client.post(
        "/v1/responses",
        json={
            "model": "qwen3.7-max",
            "instructions": "Be concise.",
            "input": "Hello",
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["object"] == "response"
    assert data["status"] == "completed"
    assert data["model"] == "qwen3.7-max"
    assert data["id"].startswith("resp_")
    assert len(data["output"]) == 1
    out_item = data["output"][0]
    assert out_item["type"] == "message"
    assert out_item["role"] == "assistant"
    assert out_item["content"][0]["type"] == "text"
    assert out_item["content"][0]["text"] == "Hello! How can I help you?"
    assert data["usage"]["input_tokens"] > 0
    assert data["usage"]["output_tokens"] > 0
    assert data["usage"]["total_tokens"] == data["usage"]["input_tokens"] + data["usage"]["output_tokens"]
    assert data["conversation_id"] == "conv_resp_123"


def test_responses_alternate_path():
    srv._make_provider = lambda key: EchoProvider()
    client = TestClient(srv.app)
    resp = client.post(
        "/responses",
        json={
            "model": "deepseek-chat",
            "input": "Test /responses",
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["object"] == "response"
    assert data["output"][0]["content"][0]["text"] == "Hello! How can I help you?"


def test_streaming_sse_event_sequence():
    srv._make_provider = lambda key: EchoProvider()
    client = TestClient(srv.app)

    with client.stream(
        "POST",
        "/v1/responses",
        json={
            "model": "gpt-4o",
            "input": "Tell me a joke",
            "stream": True,
        },
    ) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        raw_text = "".join(r.iter_text())

    # 解析 SSE 事件
    events = []
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
                payload = line[len("data: ") :].strip()
                if payload == "[DONE]":
                    ev_data = "[DONE]"
                else:
                    ev_data = json.loads(payload)
        if ev_name and ev_data is not None:
            events.append((ev_name, ev_data))
        elif ev_data == "[DONE]":
            events.append(("done", "[DONE]"))

    # 验证事件时序
    event_names = [e[0] for e in events]
    expected = [
        "response.created",
        "response.output_item.added",
        "response.content_part.added",
        "response.text.delta",
        "response.text.delta",
        "response.content_part.done",
        "response.output_item.done",
        "response.completed",
        "done",
    ]
    assert event_names == expected, f"Actual events: {event_names}"

    # 验证数据内容
    created_ev = events[0][1]
    assert created_ev["response"]["status"] == "in_progress"
    assert created_ev["response"]["model"] == "gpt-4o"
    resp_id = created_ev["response"]["id"]

    deltas = [e[1]["delta"] for e in events if e[0] == "response.text.delta"]
    assert "".join(deltas) == "Hello! How can I help you?"

    part_done = [e[1] for e in events if e[0] == "response.content_part.done"][0]
    assert part_done["part"]["text"] == "Hello! How can I help you?"

    item_done = [e[1] for e in events if e[0] == "response.output_item.done"][0]
    assert item_done["item"]["status"] == "completed"
    assert item_done["item"]["content"][0]["text"] == "Hello! How can I help you?"

    completed_ev = [e[1] for e in events if e[0] == "response.completed"][0]
    assert completed_ev["response"]["id"] == resp_id
    assert completed_ev["response"]["status"] == "completed"
    assert completed_ev["response"]["usage"]["output_tokens"] > 0


def test_tool_calling_non_streaming():
    srv._make_provider = lambda key: ToolCallingProvider()
    client = TestClient(srv.app)
    resp = client.post(
        "/v1/responses",
        json={
            "model": "qwen3.7-max",
            "input": "Run echo hello",
            "tools": [{"type": "function", "function": {"name": "bash"}}],
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert len(data["output"]) >= 1
    # 查找 function_call 项
    fn_items = [item for item in data["output"] if item["type"] == "function_call"]
    assert len(fn_items) == 1
    assert fn_items[0]["name"] == "bash"
    args = json.loads(fn_items[0]["arguments"])
    assert args["command"] == "echo hello"


def test_tool_calling_streaming():
    srv._make_provider = lambda key: ToolCallingProvider()
    client = TestClient(srv.app)
    with client.stream(
        "POST",
        "/v1/responses",
        json={
            "model": "doubao-pro",
            "input": "Run echo hello",
            "tools": [{"type": "function", "function": {"name": "bash"}}],
            "stream": True,
        },
    ) as r:
        assert r.status_code == 200
        raw_text = "".join(r.iter_text())

    assert "response.created" in raw_text
    assert "response.function_call_arguments.delta" in raw_text
    assert "response.function_call_arguments.done" in raw_text
    assert "response.output_item.done" in raw_text
    assert "response.completed" in raw_text
    assert "data: [DONE]" in raw_text


def test_error_handling_unknown_model():
    client = TestClient(srv.app)
    resp = client.post(
        "/v1/responses",
        json={"model": "unknown_test_model", "input": "hi"},
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["type"] == "model_not_found"


def test_error_handling_empty_input():
    client = TestClient(srv.app)
    resp = client.post(
        "/v1/responses",
        json={"model": "gpt-4o"},
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "invalid_request_error"


def test_error_handling_upstream_failure():
    srv._make_provider = lambda key: BoomProvider()
    client = TestClient(srv.app)
    resp = client.post(
        "/v1/responses",
        json={"model": "gpt-4o", "input": "hi"},
    )
    assert resp.status_code == 401
    assert resp.json()["error"]["type"] == "auth_error"


def test_responses_failover_non_streaming():
    """测试 Responses API 非流式调用时的对等互备网格故障转移。"""
    class FailingProvider:
        async def chat(self, *a, **kw):
            raise ProviderError("主选节点遭遇 429 频繁请求限流", status=429, err_type="rate_limit_error")
            yield

    class FallbackProvider:
        def __init__(self, name: str):
            self.name = name

        async def chat(self, *a, **kw):
            yield f"来自后备节点[{self.name}]的回复", {"conversation_id": f"conv_fallback_{self.name}"}

    def factory(key: str):
        if key == "kimi":
            return FailingProvider()
        return FallbackProvider(key)

    srv._make_provider = factory
    srv.failover_router.cooldown_tracker.reset()

    client = TestClient(srv.app)
    resp = client.post(
        "/v1/responses",
        json={"model": "kimi", "input": "帮我写代码"},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "来自后备节点" in data["output"][0]["content"][0]["text"]
    assert "failover" in data
    assert data["failover"]["from_provider"] == "kimi"
    assert resp.headers.get("x-failover-from") == "kimi"
    assert resp.headers.get("x-failover-to") == data["failover"]["to_provider"]
    assert resp.headers.get("x-failover-model") == data["failover"]["to_model"]
    assert resp.headers.get("x-failover-ring") == data["failover"]["ring"]
    srv.failover_router.cooldown_tracker.reset()


def test_responses_failover_streaming():
    """测试 Responses API 流式 SSE 调用时的对等互备网格故障转移与响应头/元数据注入。"""
    class FailingProvider:
        async def chat(self, *a, **kw):
            raise ProviderError("主选节点遭遇 429 频控熔断", status=429, err_type="rate_limit_error")
            yield

    class FallbackProvider:
        def __init__(self, name: str):
            self.name = name

        async def chat(self, *a, **kw):
            yield "Hello from ", {}
            yield f"fallback node [{self.name}]!", {"conversation_id": f"conv_fallback_{self.name}"}

    def factory(key: str):
        if key == "kimi":
            return FailingProvider()
        return FallbackProvider(key)

    srv._make_provider = factory
    srv.failover_router.cooldown_tracker.reset()

    client = TestClient(srv.app)
    with client.stream(
        "POST",
        "/v1/responses",
        json={"model": "kimi", "input": "Hello", "stream": True},
    ) as r:
        assert r.status_code == 200
        assert r.headers.get("x-failover-from") == "kimi"
        assert r.headers.get("x-failover-to") is not None
        assert r.headers.get("x-failover-model") is not None
        assert r.headers.get("x-failover-ring") is not None
        raw_text = "".join(r.iter_text())

    # 解析 SSE 事件
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
                payload = line[len("data: ") :].strip()
                if payload == "[DONE]":
                    ev_data = "[DONE]"
                else:
                    ev_data = json.loads(payload)
        if ev_name and ev_data is not None:
            events.append((ev_name, ev_data))
        elif ev_data == "[DONE]":
            events.append(("done", "[DONE]"))

    completed_events = [e[1] for e in events if e[0] == "response.completed"]
    assert len(completed_events) == 1
    completed = completed_events[0]
    fo = completed["response"].get("failover") or completed.get("failover")
    assert fo is not None
    assert fo["from_provider"] == "kimi"
    assert fo["to_provider"] == r.headers.get("x-failover-to")

    deltas = [e[1]["delta"] for e in events if e[0] == "response.text.delta"]
    full = "".join(deltas)
    assert "fallback node" in full
    srv.failover_router.cooldown_tracker.reset()


def test_tool_calling_xml_non_streaming():
    """测试 Responses API 非流式调用下 XML 格式工具调用的提取（包含前置文本）。"""
    class XmlToolProvider:
        async def chat(self, *a, **kw):
            yield "Querying weather database:\n", {}
            yield '<tool_call>{"name": "fetch_weather", "arguments": {"city": "Beijing"}}</tool_call>', {}

    srv._make_provider = lambda key: XmlToolProvider()
    client = TestClient(srv.app)
    resp = client.post(
        "/v1/responses",
        json={
            "model": "qwen3.7-max",
            "input": "What is the weather in Beijing?",
            "tools": [{"type": "function", "function": {"name": "fetch_weather"}}],
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert len(data["output"]) == 2
    # output[0] 为文本消息
    assert data["output"][0]["type"] == "message"
    assert data["output"][0]["content"][0]["text"] == "Querying weather database:"
    # output[1] 为 function_call
    assert data["output"][1]["type"] == "function_call"
    assert data["output"][1]["name"] == "fetch_weather"
    assert data["output"][1]["call_id"].startswith("call_")
    assert data["output"][1]["id"] == data["output"][1]["call_id"]
    assert data["output"][1]["status"] == "completed"
    args = json.loads(data["output"][1]["arguments"])
    assert args["city"] == "Beijing"


def test_tool_calling_xml_streaming():
    """测试 Responses API 流式调用下 XML 格式工具调用的完整时序与事件流。"""
    class XmlStreamToolProvider:
        async def chat(self, *a, **kw):
            yield '<function_call>{"name": "calculator", "arguments": {"expr": "100*42"}}</function_call>', {}

    srv._make_provider = lambda key: XmlStreamToolProvider()
    client = TestClient(srv.app)
    with client.stream(
        "POST",
        "/v1/responses",
        json={
            "model": "deepseek-chat",
            "input": "Calculate 100*42",
            "tools": [{"type": "function", "function": {"name": "calculator"}}],
            "stream": True,
        },
    ) as r:
        assert r.status_code == 200
        raw_text = "".join(r.iter_text())

    assert "response.created" in raw_text
    assert "response.output_item.added" in raw_text
    assert "response.function_call_arguments.delta" in raw_text
    assert "response.function_call_arguments.done" in raw_text
    assert "response.output_item.done" in raw_text
    assert "response.completed" in raw_text
    assert "data: [DONE]" in raw_text
    assert "100*42" in raw_text
    assert "calculator" in raw_text


def test_tool_calling_without_prefix_non_streaming():
    """测试无前置说明文本纯工具调用时的输出格式（仅 output[0] 为 function_call）。"""
    class DirectToolProvider:
        async def chat(self, *a, **kw):
            yield '```tool_call\n{"name": "bash", "arguments": {"command": "ls -la"}}\n```', {}

    srv._make_provider = lambda key: DirectToolProvider()
    client = TestClient(srv.app)
    resp = client.post(
        "/v1/responses",
        json={
            "model": "qwen3.7-max",
            "input": "List files",
            "tools": [{"type": "function", "function": {"name": "bash"}}],
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert len(data["output"]) == 1
    assert data["output"][0]["type"] == "function_call"
    assert data["output"][0]["name"] == "bash"
    assert data["output"][0]["status"] == "completed"
    assert data["output"][0]["id"].startswith("call_")
    assert data["output"][0]["call_id"] == data["output"][0]["id"]
    args = json.loads(data["output"][0]["arguments"])
    assert args["command"] == "ls -la"


def test_responses_stateless_session_cleanup():
    """测试 Responses API 无状态请求结束后自动触发会话清理 delete_conversation。"""
    import time
    deleted = []

    class CleanupProvider:
        async def chat(self, *a, **kw):
            yield "Hello", {"conversation_id": "conv_clean_me"}

        async def delete_conversation(self, conv_id: str):
            deleted.append(conv_id)
            return True

    srv._make_provider = lambda key: CleanupProvider()
    client = TestClient(srv.app)

    # 1. 无 conversation_id（stateless）请求 -> 应触发 delete_conversation
    resp = client.post(
        "/v1/responses",
        json={"model": "qwen3.7-max", "input": "Hello clean"},
    )
    assert resp.status_code == 200
    time.sleep(0.05)
    assert "conv_clean_me" in deleted

    # 2. 携带显式 conversation_id（stateful）请求 -> 不应删除
    deleted.clear()
    resp2 = client.post(
        "/v1/responses",
        json={"model": "qwen3.7-max", "input": "Hello keep", "conversation_id": "my_stateful_conv"},
    )
    assert resp2.status_code == 200
    time.sleep(0.05)
    assert len(deleted) == 0


def main():
    print("运行 OpenAI Responses 协议兼容层单测...")
    test_convert_responses_request_string_input()
    test_convert_responses_request_complex_input_blocks()
    test_convert_responses_request_tools()
    test_non_streaming_responses_endpoint()
    test_responses_alternate_path()
    test_streaming_sse_event_sequence()
    test_tool_calling_non_streaming()
    test_tool_calling_streaming()
    test_error_handling_unknown_model()
    test_error_handling_empty_input()
    test_error_handling_upstream_failure()
    test_responses_failover_non_streaming()
    test_responses_failover_streaming()
    test_tool_calling_xml_non_streaming()
    test_tool_calling_xml_streaming()
    test_tool_calling_without_prefix_non_streaming()
    test_responses_stateless_session_cleanup()
    print("[PASS] 全��� Responses API 单测通过！")


if __name__ == "__main__":
    main()
