"""Responses 历史回放中的工具调用测试 (tests/test_responses_history_tools.py)。

[TOOL-LOOP] S1.5/S1.6：客户端把上一轮的 output 原样回放时，工具调用有两种非常常见
的挂载方式，网关此前只认其中一种：

  - **`message.tool_calls`**（OpenAI chat 兼容形状，多数 SDK 与自建客户端用它）：
    整个分支只取了 `content`，`tool_calls` 被静默丢弃。于是模型下一轮只看到
    `[Tool Result for c1]`，却找不到自己发过什么调用——它不知道 c1 是什么工具、
    带什么参数，多轮 agent 循环在这里断掉，而且没有任何报错。
  - **`role:"tool"`** 的 message 项（同样来自 chat 形状）：只带 `tool_call_id`，
    原样落成一条 **user 发言**，模型看到的是一段无主的文本输出。

本套件锁住：两种历史形状都还原成上游能识别的 assistant 调用 + 带 id 的工具结果，
调用 id 与参数逐字保留，且 `message.tool_calls` 与 `function_call` 两条路径使用
**同一份**渲染规则（同一事实不在两处解释）。

运行: .venv/bin/python tests/test_responses_history_tools.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["TESTING"] = "1"
os.environ["GATEWAY_DISABLE_AUTO_REFRESH"] = "1"
os.environ["GATEWAY_DISABLE_CREDENTIAL_WATCH"] = "1"
os.environ["GATEWAY_DISABLE_PROBING"] = "1"
os.environ["GATEWAY_DISABLE_PACING"] = "1"
os.environ["QWEN_AUTH_TOKEN"] = "fake_qwen"
os.environ["DEEPSEEK_AUTH_TOKEN"] = "fake_ds"
os.environ["DOUBAO_AUTH_TOKEN"] = "fake_doubao"
os.environ["KIMI_AUTH_TOKEN"] = "fake_kimi"
os.environ["GLM_AUTH_TOKEN"] = "fake_glm"

from fastapi.testclient import TestClient  # noqa: E402

import server as srv  # noqa: E402
from providers.base import flatten_messages  # noqa: E402
from providers.responses_compat import (  # noqa: E402
    ResponsesRequest,
    convert_responses_to_gateway_messages,
)

ARGS = '{"path": "hello.py"}'


def _call_blocks(messages: list) -> list:
    """收集全部 ```tool_call 块（一条消息里可能有多次调用）。"""
    blocks = []
    for m in messages:
        if m["role"] != "assistant":
            continue
        parts = m["content"].split("```tool_call\n")[1:]
        for part in parts:
            blocks.append(json.loads(part.split("\n```", 1)[0]))
    return blocks


def test_assistant_message_tool_calls_are_replayed():
    req = ResponsesRequest.model_validate({
        "model": "gpt-4o",
        "input": [
            {"type": "message", "role": "user", "content": "读一下 hello.py"},
            {"type": "message", "role": "assistant", "content": "我来读",
             "tool_calls": [{"id": "c1", "type": "function",
                             "function": {"name": "read_file", "arguments": ARGS}}]},
            {"type": "message", "role": "tool", "tool_call_id": "c1", "content": "print(1)"},
            {"type": "message", "role": "user", "content": "现在呢？"},
        ],
    })
    msgs = convert_responses_to_gateway_messages(req)

    calls = _call_blocks(msgs)
    assert len(calls) == 1, f"message.tool_calls 被丢弃了：模型找不到自己发过的调用。{msgs}"
    assert calls[0]["name"] == "read_file"
    assert calls[0]["id"] == "c1", "调用 id 丢失，工具结果无法对应"
    assert calls[0]["arguments"] == {"path": "hello.py"}

    # 调用所在的 assistant 消息不得丢掉它原本的文本
    caller = [m for m in msgs if m["role"] == "assistant" and "```tool_call" in m["content"]][0]
    assert "我来读" in caller["content"], "assistant 的说明文字被工具块覆盖了"

    # role:"tool" 必须是带 id 的工具结果，而不是一句用户发言
    results = [m for m in msgs if "[Tool Result for c1]" in m["content"]]
    assert len(results) == 1, f"role:'tool' 没有还原成工具结果：{msgs}"
    assert results[0]["role"] == "user"
    tool_as_user = [m for m in msgs if m["role"] == "user" and m["content"].strip() == "print(1)"]
    assert not tool_as_user, "工具输出仍以裸用户发言的形式出现"
    print("test_assistant_message_tool_calls_are_replayed: PASSED")


def test_both_history_shapes_render_identically():
    # 同一轮调用，一种以 message.tool_calls 回放，一种以 function_call 项回放；
    # 渲染结果必须逐字节相同，否则同一次调用在两条路径上有两套规则。
    a = convert_responses_to_gateway_messages(ResponsesRequest.model_validate({
        "model": "gpt-4o",
        "input": [{"type": "message", "role": "assistant", "content": "",
                   "tool_calls": [{"id": "c1", "type": "function",
                                   "function": {"name": "read_file", "arguments": ARGS}}]}],
    }))
    b = convert_responses_to_gateway_messages(ResponsesRequest.model_validate({
        "model": "gpt-4o",
        "input": [{"type": "function_call", "name": "read_file",
                   "arguments": ARGS, "call_id": "c1"}],
    }))
    assert _call_blocks(a) == _call_blocks(b) == [
        {"name": "read_file", "arguments": {"path": "hello.py"}, "id": "c1"}
    ]
    print("test_both_history_shapes_render_identically: PASSED")


def test_multiple_tool_calls_in_one_message_all_survive():
    req = ResponsesRequest.model_validate({
        "model": "gpt-4o",
        "input": [{"type": "message", "role": "assistant", "content": "并行读两个",
                   "tool_calls": [
                       {"id": "c1", "function": {"name": "read_file", "arguments": '{"path":"a"}'}},
                       {"id": "c2", "function": {"name": "read_file", "arguments": '{"path":"b"}'}},
                   ]}],
    })
    calls = _call_blocks(convert_responses_to_gateway_messages(req))
    assert [c["id"] for c in calls] == ["c1", "c2"], f"并行的第二次调用丢失: {calls}"
    print("test_multiple_tool_calls_in_one_message_all_survive: PASSED")


def test_tool_call_reaches_upstream_prompt_end_to_end():
    captured = {}

    class P:
        async def chat(self, messages, model, conversation_id=None, **kw):
            captured["prompt"] = flatten_messages(messages)
            yield "ok", {}
            yield "", {"conversation_id": None}

    srv._make_provider = lambda key, client=None: P()
    r = TestClient(srv.app).post("/v1/responses", json={
        "model": "gpt-4o",
        "input": [
            {"type": "message", "role": "user", "content": "读一下 hello.py"},
            {"type": "message", "role": "assistant", "content": "我来读",
             "tool_calls": [{"id": "c1", "type": "function",
                             "function": {"name": "read_file", "arguments": ARGS}}]},
            {"type": "message", "role": "tool", "tool_call_id": "c1", "content": "print(1)"},
        ],
        "tools": [{"type": "function", "function": {
            "name": "read_file", "parameters": {"type": "object",
                                                "properties": {"path": {"type": "string"}}}}}],
    })
    assert r.status_code == 200, r.text
    prompt = captured["prompt"]
    assert "read_file" in prompt, f"调用没有到达上游提示词:\n{prompt}"
    assert "c1" in prompt, "调用 id 没有到达上游，工具结果无法对应"
    assert "[Tool Result for c1]" in prompt
    print("test_tool_call_reaches_upstream_prompt_end_to_end: PASSED")


def test_server_side_action_records_are_not_user_messages():
    """服务端动作回执（web_search_call 等）不得以原始 JSON 落成**用户发言**。

    第 43 节为 `reasoning` 修过一次的正是同一个兜底分支，但其余类型仍然漏：
    客户端把上一轮 output 原样回放时，`web_search_call` / `file_search_call` /
    `computer_call` / `mcp_call` / `image_generation_call` / `code_interpreter_call`
    全部落进 `else`，被 `extract_responses_content` 序列化成 JSON 塞进 user 通道。
    模型于是看到「用户说了一段自己没有发过的调用记录」——上下文污染，且它无法
    知道那次搜索**实际上发生过**，可能立刻重复同一请求。

    回执必须以 **assistant** 身份出现，并保留动作内容（谁搜了什么）。
    """
    for item_type in ("web_search_call", "file_search_call", "computer_call",
                      "mcp_call", "image_generation_call", "code_interpreter_call"):
        req = ResponsesRequest.model_validate({
            "model": "gpt-4o",
            "input": [
                {"type": "message", "role": "user", "content": "查一下"},
                {"type": item_type, "id": "s1", "status": "completed",
                 "action": {"type": "search", "query": "claude docs"}},
            ],
        })
        msgs = convert_responses_to_gateway_messages(req)
        users = [m for m in msgs if m["role"] == "user"]
        leaked = [m for m in users if '"type":' in str(m.get("content", ""))]
        assert not leaked, (
            f"{item_type} 以原始 JSON 落进了用户通道: {leaked[0]['content'][:90]!r}"
        )
        last = msgs[-1]
        assert last["role"] == "assistant", (
            f"{item_type} 的回执不是 assistant 身份: {last}"
        )
        assert "claude docs" in last["content"], (
            f"{item_type} 的回执丢掉了动作内容，模型会重复同一请求: {last['content']!r}"
        )
    print("test_server_side_action_records_are_not_user_messages: PASSED")


def test_truly_unknown_items_are_announced_not_dumped():
    """完全未知的类型：既不得灌原始 JSON，也不得静默消失。

    静默消失让模型把「它没看到」当成「它不存在」，然后据此作答。
    """
    req = ResponsesRequest.model_validate({
        "model": "gpt-4o",
        "input": [
            {"type": "message", "role": "user", "content": "hi"},
            {"type": "some_future_item_type", "payload": {"secret": 1}},
        ],
    })
    msgs = convert_responses_to_gateway_messages(req)
    # 断言必须落在**消息正文**上。第一版这里 `json.dumps(msgs)` 后再找 `"secret"`，
    # 而嵌套字符串里的引号会被转义成 `\"secret\"`，于是无论代码怎么写都找不到——
    # 一条永远为真的断言（它在修复前的代码上照样通过）。
    blob = "\n".join(str(m.get("content", "")) for m in msgs)
    assert "secret" not in blob, f"未知条目被整体序列化进了提示词: {blob[-160:]!r}"
    assert "some_future_item_type" in blob, "未知条目完全消失了，模型无从知道有内容缺失"
    print("test_truly_unknown_items_are_announced_not_dumped: PASSED")


def test_unknown_anthropic_blocks_are_announced_not_dropped():
    """Anthropic 侧的同一类缺陷：未知块类型此前**静默消失**。"""
    from providers.anthropic_compat import extract_anthropic_content

    text = extract_anthropic_content([
        {"type": "text", "text": "before"},
        {"type": "server_tool_use", "id": "s1"},
        {"type": "web_search_tool_result", "content": [{"title": "x"}]},
    ])
    assert "before" in text
    for probe in ("server_tool_use", "web_search_tool_result"):
        assert probe in text, f"未知块 {probe} 静默消失了，模型会以为它不存在"


def test_item_without_a_type_is_a_message():
    """**靶点**：`{"role": "user", "content": "hi"}` 是合法输入，此前是 HTTP 500。

    OpenAI SDK 的 `EasyInputMessage` 根本不带 `type`。`item.get("type")` 返回 None，
    随后 `item_type.endswith("_call")` 抛 `AttributeError: 'NoneType' object has no
    attribute 'endswith'`，整个请求变成 500 —— 一个合法请求被网关自己的内部异常拒绝，
    错误体里还带着 traceback。OpenAI 自己的定义给出了等价关系：条目有 `role` 就是
    一条 message。
    """
    captured = {}

    class P:
        async def chat(self, messages, model, conversation_id=None, **kw):
            captured["prompt"] = flatten_messages(messages)
            yield "ok", {}
            yield "", {"conversation_id": None}

    before = srv._make_provider
    srv._make_provider = lambda key, client=None: P()
    try:
        r = TestClient(srv.app).post("/v1/responses", json={
            "model": "gpt-4o",
            "input": [{"role": "user", "content": "hi"}],
        })
    finally:
        srv._make_provider = before

    assert r.status_code == 200, (
        f"不带 type 的 SDK 形状被拒绝为 {r.status_code}：{r.text[:300]}"
    )
    assert "hi" in captured.get("prompt", ""), (
        f"用户内容没有到达上游提示词: {captured.get('prompt')!r}"
    )
    print("test_item_without_a_type_is_a_message: PASSED")


def test_typed_and_untyped_messages_render_identically():
    """同一份内容，带不带 `type` 都必须渲染成同一条上游消息（一条规则，不是两条）。"""
    typed = ResponsesRequest(**{
        "model": "gpt-4o",
        "input": [{"type": "message", "role": "user", "content": "hi"}],
    })
    untyped = ResponsesRequest(**{"model": "gpt-4o", "input": [{"role": "user", "content": "hi"}]})
    a = convert_responses_to_gateway_messages(typed)
    b = convert_responses_to_gateway_messages(untyped)
    assert a == b, (
        "同一个 message 因为带不带 type 而渲染成两种结果——这正是[第 43 节]"
        f"（一种事实两处解释）的形态。带 type: {a} / 不带: {b}"
    )
    print("test_typed_and_untyped_messages_render_identically: PASSED")


def run_all():
    test_assistant_message_tool_calls_are_replayed()
    test_both_history_shapes_render_identically()
    test_multiple_tool_calls_in_one_message_all_survive()
    test_tool_call_reaches_upstream_prompt_end_to_end()
    test_server_side_action_records_are_not_user_messages()
    test_truly_unknown_items_are_announced_not_dumped()
    test_unknown_anthropic_blocks_are_announced_not_dropped()
    test_item_without_a_type_is_a_message()
    test_typed_and_untyped_messages_render_identically()
    print("\n✅ All Responses history tool-call tests passed!")


if __name__ == "__main__":
    run_all()