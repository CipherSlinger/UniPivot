"""多轮 Agent 工具循环端到端测试 (AGENT-LOOP)

动机来自「网关连接 agent 效果差 / 工具调用没跑起来」。前几套件分别验证了
单点正确性（提示词序列化、伪造剔除、事件契约、schema 保真），但 agent 的真实
故障是**跨轮次**的：第 1 轮模型发工具调用 → 客户端执行 → 第 2 轮把
assistant.tool_calls + role:"tool" 结果回传 → 模型必须看到完整链条才能继续。

本套件用三个协议各自跑一遍真实的两轮循环，断言「上游真正收到的提示词」里
包含整条工具链，而不是只看网关自己的返回值：

  A. OpenAI Chat：第 1 轮发起调用；第 2 轮回传 tool_calls + tool 结果
  B. Anthropic：第 1 轮 tool_use；第 2 轮回传 tool_use + tool_result（带 id）
  C. Responses：第 1 轮 function_call；第 2 轮回传 function_call + function_call_output

并覆盖：
  D. tool_choice="none" / 指定工具 的约束确实进入了上游提示词
  E. 续聊（conversation_id）模式下第 2 轮只发送工具结果，不重发旧提问
  F. 客户端未声明的工具名不会被回传成 tool_calls（伪造防护的端到端形态）

运行: .venv/bin/python tests/test_agent_loop.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["TESTING"] = "1"
os.environ["GATEWAY_DISABLE_CREDENTIAL_WATCH"] = "1"
os.environ["GATEWAY_DISABLE_AUTO_REFRESH"] = "1"
os.environ["GATEWAY_DISABLE_PROMPT_FOLDING"] = "1"
os.environ["QWEN_AUTH_TOKEN"] = "fake_qwen"
os.environ["DEEPSEEK_AUTH_TOKEN"] = "fake_ds"
os.environ["DOUBAO_AUTH_TOKEN"] = "fake_doubao"

from fastapi.testclient import TestClient  # noqa: E402

import server as srv  # noqa: E402
from providers.base import flatten_messages, continuation_content  # noqa: E402
from providers.anthropic_compat import convert_anthropic_to_gateway_messages  # noqa: E402
from providers.responses_compat import (  # noqa: E402
    ResponsesRequest,
    convert_responses_to_gateway_messages,
)

TOOLS_OPENAI = [
    {
        "type": "function",
        "function": {
            "name": "Read",
            "description": "Read a file",
            "parameters": {
                "type": "object",
                "properties": {"file_path": {"type": "string"}},
                "required": ["file_path"],
            },
        },
    }
]


class CapturingProvider:
    """记录上游实际收到的 messages，并按脚本产出增量。"""

    last_messages: list = []
    last_conversation_id = None
    last_kwargs: dict = {}
    script: list = []

    def __init__(self, *a, **kw):
        pass

    async def chat(self, messages, model, **kwargs):
        CapturingProvider.last_messages = messages
        CapturingProvider.last_conversation_id = kwargs.get("conversation_id")
        CapturingProvider.last_kwargs = kwargs
        for delta, meta in CapturingProvider.script:
            yield delta, meta


def _with_provider(script, fn):
    """在捕获 provider 下执行 fn，返回被捕获的上游 messages。"""
    CapturingProvider.script = script
    CapturingProvider.last_messages = []
    srv._make_provider = lambda key: CapturingProvider()
    try:
        fn()
        return CapturingProvider.last_messages
    finally:
        srv._make_provider = _default_factory


_default_factory = srv._make_provider


def _client():
    return TestClient(srv.app)


# ---------------------------------------------------------------- A. OpenAI

def test_openai_two_round_loop():
    """第 2 轮的上游提示词必须同时包含 assistant 的调用与工具结果。"""
    history = [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "read /etc/hosts"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_abc",
                    "type": "function",
                    "function": {"name": "Read", "arguments": '{"file_path": "/etc/hosts"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_abc", "content": "127.0.0.1 localhost"},
    ]
    captured = _with_provider(
        [("The file contains one line.", {"conversation_id": "conv_openai"})],
        lambda: _client().post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": history, "tools": TOOLS_OPENAI},
        ),
    )
    prompt = flatten_messages(captured)
    assert "```tool_call" in prompt, f"assistant.tool_calls 在提示词里消失了:\n{prompt}"
    assert '"name": "Read"' in prompt, prompt
    assert '"/etc/hosts"' in prompt, prompt
    assert "call_abc" in prompt, "调用 id 必须保留，否则工具结果无法与调用对应"
    assert "Tool (Read)" in prompt, f"工具结果没有标明对应工具:\n{prompt}"
    assert "127.0.0.1 localhost" in prompt, prompt
    print("test_openai_two_round_loop: PASSED")


def test_openai_continuation_sends_only_tool_result():
    """续聊模式下第 2 轮只发送工具结果，绝不重发上一轮旧提问。"""
    captured = _with_provider(
        [("done", {})],
        lambda: _client().post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "conversation_id": "conv_cont",
                "tools": TOOLS_OPENAI,
                "messages": [
                    {"role": "user", "content": "read /etc/hosts"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_x",
                                "type": "function",
                                "function": {"name": "Read", "arguments": '{"file_path": "/etc/hosts"}'},
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "call_x", "content": "127.0.0.1 localhost"},
                ],
            },
        ),
    )
    cont = continuation_content(captured)
    assert "127.0.0.1 localhost" in cont, f"工具结果没有送达上游: {cont!r}"
    assert "read /etc/hosts" not in cont, f"旧提问被重复发送: {cont!r}"
    print("test_openai_continuation_sends_only_tool_result: PASSED")


# ---------------------------------------------------------------- B. Anthropic

def test_anthropic_two_round_loop():
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "read /etc/hosts"}],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "I'll read it."},
                {
                    "type": "tool_use",
                    "id": "toolu_01",
                    "name": "Read",
                    "input": {"file_path": "/etc/hosts"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_01",
                    "content": [{"type": "text", "text": "127.0.0.1 localhost"}],
                }
            ],
        },
    ]
    captured = _with_provider(
        [("OK", {})],
        lambda: _client().post(
            "/v1/messages",
            json={
                "model": "claude-3-7-sonnet",
                "max_tokens": 256,
                "messages": messages,
                "tools": [
                    {
                        "name": "Read",
                        "description": "Read a file",
                        "input_schema": {
                            "type": "object",
                            "properties": {"file_path": {"type": "string"}},
                            "required": ["file_path"],
                        },
                    }
                ],
            },
        ),
    )
    prompt = "\n".join(str(m.get("content", "")) for m in captured)
    assert "toolu_01" in prompt, f"tool_use id 没有进入提示词:\n{prompt}"
    assert "[Tool Result for toolu_01]" in prompt, f"工具结果没有送达:\n{prompt}"
    assert "127.0.0.1 localhost" in prompt
    assert "I'll read it." in prompt
    # schema 保真：required 必须出现在系统提示词里
    assert '"required":["file_path"]' in prompt, f"工具 schema 丢失 required:\n{prompt}"
    print("test_anthropic_two_round_loop: PASSED")


# ---------------------------------------------------------------- C. Responses

def test_responses_two_round_loop():
    req = ResponsesRequest(
        model="gpt-4o",
        tools=TOOLS_OPENAI,
        input=[
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "read /etc/hosts"}]},
            {"type": "function_call", "call_id": "call_r1", "name": "Read", "arguments": '{"file_path": "/etc/hosts"}'},
            {"type": "function_call_output", "call_id": "call_r1", "output": "127.0.0.1 localhost"},
        ],
    )
    msgs = convert_responses_to_gateway_messages(req)
    prompt = flatten_messages(msgs)
    assert "call_r1" in prompt, f"function_call 的 call_id 丢失:\n{prompt}"
    assert "[Tool Result for call_r1]" in prompt
    assert '"name": "Read"' in prompt
    print("test_responses_two_round_loop: PASSED")


def test_responses_stream_second_round_uses_tool_history():
    body = {
        "model": "gpt-4o",
        "stream": True,
        "tools": TOOLS_OPENAI,
        "input": [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "read /etc/hosts"}]},
            {"type": "function_call", "call_id": "call_r2", "name": "Read", "arguments": '{"file_path": "/etc/hosts"}'},
            {"type": "function_call_output", "call_id": "call_r2", "output": "127.0.0.1 localhost"},
        ],
    }

    def _post():
        with _client().stream("POST", "/v1/responses", json=body) as r:
            assert r.status_code == 200, r.read()
            "".join(r.iter_text())

    captured = _with_provider([("One line.", {})], _post)
    prompt = flatten_messages(captured)
    assert "call_r2" in prompt and "127.0.0.1 localhost" in prompt, prompt
    print("test_responses_stream_second_round_uses_tool_history: PASSED")


# ---------------------------------------------------------------- D. tool_choice

def test_tool_choice_none_reaches_prompt():
    captured = _with_provider(
        [("plain answer", {})],
        lambda: _client().post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [
                    {"role": "user", "content": "say hi"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {"id": "c1", "type": "function",
                             "function": {"name": "Read", "arguments": "{}"}}
                        ],
                    },
                    {"role": "tool", "tool_call_id": "c1", "content": "x"},
                ],
                "tools": TOOLS_OPENAI,
                "tool_choice": "none",
            },
        ),
    )
    prompt = flatten_messages(captured)
    assert "Do not call any tool" in prompt, f"tool_choice=none 未生效:\n{prompt}"
    print("test_tool_choice_none_reaches_prompt: PASSED")


def test_tool_choice_named_tool_reaches_prompt():
    captured = _with_provider(
        [("ok", {})],
        lambda: _client().post(
            "/v1/messages",
            json={
                "model": "claude-3-7-sonnet",
                "max_tokens": 128,
                "messages": [{"role": "user", "content": "go"}],
                "tools": [
                    {
                        "name": "Read",
                        "description": "Read a file",
                        "input_schema": {"type": "object", "properties": {}},
                    }
                ],
                "tool_choice": {"type": "tool", "name": "Read"},
            },
        ),
    )
    prompt = "\n".join(str(m.get("content", "")) for m in captured)
    assert "You MUST call the tool `Read`" in prompt, f"tool_choice 未生效:\n{prompt}"
    print("test_tool_choice_named_tool_reaches_prompt: PASSED")


# ---------------------------------------------------------------- F. 伪造防护

def test_undeclared_tool_is_not_returned_as_tool_call():
    """模型臆造的工具名不得变成客户端无法执行的 tool_calls。"""
    script = [
        ("Let me use a tool.\n```tool_call\n", {}),
        ('{"name": "EvilTool", "arguments": {"x": 1}}\n', {}),
        ("```", {}),
    ]

    def _post():
        resp = _client().post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "do it"}],
                "tools": TOOLS_OPENAI,
            },
        )
        data = resp.json()
        msg = data["choices"][0]["message"]
        assert not msg.get("tool_calls"), f"未声明的工具被透传成 tool_calls: {msg}"
        assert "EvilTool" in (msg.get("content") or ""), (
            f"被拒绝的调用必须以正文形式交还用户，而不是凭空消失: {msg}"
        )

    _with_provider(script, _post)
    print("test_undeclared_tool_is_not_returned_as_tool_call: PASSED")


def run_all():
    tests = [
        test_openai_two_round_loop,
        test_openai_continuation_sends_only_tool_result,
        test_anthropic_two_round_loop,
        test_responses_two_round_loop,
        test_responses_stream_second_round_uses_tool_history,
        test_tool_choice_none_reaches_prompt,
        test_tool_choice_named_tool_reaches_prompt,
        test_undeclared_tool_is_not_returned_as_tool_call,
    ]
    for fn in tests:
        fn()
    print(f"\n✅ All agent-loop tests passed! ({len(tests)} cases)")


if __name__ == "__main__":
    run_all()