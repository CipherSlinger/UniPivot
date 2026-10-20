"""Responses freeform / custom 工具往返测试 (tests/test_responses_custom_tools.py)。

[TOOL-LOOP] S1.3：Codex CLI 的 `apply_patch` 是 Responses 的 **custom（freeform）**
工具——声明里没有 JSON schema，调用项的 `type` 是 `custom_tool_call`，参数是一段
自由文本 `input`，流式事件叫 `custom_tool_call_input.delta`。网关此前对它一无所知：

  - 声明被渲染成一个空的 `{}` schema（模型被告知「这个工具不要参数」）；
  - 回放（客户端把上一轮 output 原样送回）没有分支，落进兜底，被序列化成一段原始
    JSON 当成 **user 输入**——模型看到一段自己从未写过的补丁 JSON，既认不出这是
    自己的调用，也接不上紧随其后的执行结果；
  - 出口把所有工具都报成 `function_call` + JSON `arguments`，客户端按 `type` 分派
    时找不到自己声明的工具，本轮调用被静默丢弃。

三个症状都在客户端侧表现为「agent 不会改文件 / 工具没跑起来」，且没有一条报错。
本套件锁住四条不变量：

1. freeform 声明在提示词里有**可执行的**参数契约（`{"input": "<原文>"}`）；
2. 回放的 `custom_tool_call` 仍是 assistant 的调用，且原文逐字符保留；
3. 回放的 `custom_tool_call_output` 是工具结果，不是用户发言；
4. 出口（非流式与流式）对 freeform 工具发 `custom_tool_call` 族事件，对普通工具
   仍然发 `function_call` 族——两者不得互相污染。

运行: .venv/bin/python tests/test_responses_custom_tools.py
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
from providers.anthropic_compat import format_tools_system_prompt  # noqa: E402
from providers.responses_compat import (  # noqa: E402
    ResponsesRequest,
    convert_responses_to_gateway_messages,
    custom_tool_input_text,
    format_responses_completion,
    freeform_tool_names,
)

PATCH = """*** Begin Patch
*** Update File: hello.py
@@
-print("hi")
+print("hello")
*** End Patch"""


class PatchToolProvider:
    """上游模型按提示词契约写出 apply_patch 的 freeform 调用。"""

    def __init__(self):
        self.last_prompt = ""
        self.last_tools_prompt_has_input = False

    async def chat(self, messages, model, conversation_id=None, **kw):
        from providers.base import flatten_messages

        self.last_prompt = flatten_messages(messages)
        # 模型只能看到提示词里的工具声明：它必须被告知原文放进 `input` 字段。
        self.last_tools_prompt_has_input = '"input"' in self.last_prompt
        payload = json.dumps(
            {"name": "apply_patch", "arguments": {"input": PATCH}}, ensure_ascii=False
        )
        yield f"好的，我来改。\n```tool_call\n{payload}\n```", {}
        yield "", {"conversation_id": None}


def test_freeform_declaration_gets_an_executable_schema():
    prompt = format_tools_system_prompt(
        [{"type": "custom", "name": "apply_patch", "description": "Apply a patch"}]
    )
    assert "`apply_patch`" in prompt, "freeform 工具没有出现在工具声明里"
    assert '"input"' in prompt, (
        "freeform 工具没有参数契约——模型被告知它不接受任何参数，于是永远不会调用它"
    )
    assert "freeform" in prompt.lower(), "提示词没有告诉模型这是原文直传的工具"
    print("test_freeform_declaration_gets_an_executable_schema: PASSED")


def test_freeform_call_replay_is_assistant_not_user():
    req = ResponsesRequest.model_validate({
        "model": "gpt-4o",
        "input": [
            {"type": "message", "role": "user", "content": "改一下 hello.py"},
            {"type": "custom_tool_call", "call_id": "call_1", "name": "apply_patch", "input": PATCH},
            {"type": "custom_tool_call_output", "call_id": "call_1", "output": "Done!"},
            {"type": "message", "role": "user", "content": "继续"},
        ],
    })
    msgs = convert_responses_to_gateway_messages(req)
    roles = [m["role"] for m in msgs]
    assert "user" in roles and "assistant" in roles

    # 回放出来的调用必须是 assistant 说的，且原文逐字符保留（补丁对空白敏感）
    call_msgs = [m for m in msgs if m["role"] == "assistant" and "```tool_call" in m["content"]]
    assert len(call_msgs) == 1, f"custom_tool_call 没有被回放成 assistant 调用: {msgs}"
    block = call_msgs[0]["content"]
    payload = json.loads(block.split("```tool_call\n", 1)[1].rsplit("\n```", 1)[0])
    assert payload["name"] == "apply_patch"
    assert payload["arguments"] == {"input": PATCH}, (
        f"补丁原文被改写了（换行/引号必须原样）: {payload['arguments']!r}"
    )
    assert payload.get("id") == "call_1"

    # 执行结果必须是工具结果标签，而不是一句用户发言
    result_msgs = [m for m in msgs if "[Tool Result for call_1]" in m["content"]]
    assert len(result_msgs) == 1, f"custom_tool_call_output 没有回放成工具结果: {msgs}"
    assert result_msgs[0]["role"] == "user"

    # 兜底路径不再把原始 item JSON 当成用户输入
    raw_json_leaks = [
        m for m in msgs
        if m["role"] == "user" and '"custom_tool_call"' in m["content"]
    ]
    assert not raw_json_leaks, f"原始 item JSON 泄漏成了用户输入: {raw_json_leaks}"
    print("test_freeform_call_replay_is_assistant_not_user: PASSED")


def test_custom_tool_input_text_unwraps_three_shapes():
    # 契约形状
    assert custom_tool_input_text({"input": PATCH}) == PATCH
    # 模型把原文写成一个整串 arguments（被 _normalize_tool_args 折成 {"input": ...}）
    assert custom_tool_input_text(PATCH) == PATCH
    # 模型自己编了一个单字段名
    assert custom_tool_input_text({"patch": PATCH}) == PATCH
    # 真·结构化参数：序列化回原文，而不是丢掉
    assert json.loads(custom_tool_input_text({"a": 1})) == {"a": 1}
    print("test_custom_tool_input_text_unwraps_three_shapes: PASSED")


def test_freeform_names_only_custom_declarations():
    names = freeform_tool_names([
        {"type": "custom", "name": "apply_patch"},
        {"type": "function", "function": {"name": "bash"}},
        {"name": "shell"},
        "not-a-dict",
    ])
    assert names == {"apply_patch"}, names
    print("test_freeform_names_only_custom_declarations: PASSED")


def test_completion_emits_custom_tool_call_item():
    resp = format_responses_completion(
        resp_id="resp_1",
        model="gpt-4o",
        output_text="",
        input_tokens=10,
        output_tokens=5,
        tool_calls=[{"id": "c1", "type": "function",
                     "function": {"name": "apply_patch", "arguments": json.dumps({"input": PATCH})}}],
        freeform_tools={"apply_patch"},
    )
    items = [i for i in resp["output"] if i["type"] != "message"]
    assert len(items) == 1
    item = items[0]
    assert item["type"] == "custom_tool_call", (
        f"freeform 工具被报成了 {item['type']}，客户端按 type 分派会丢弃它"
    )
    assert item["input"] == PATCH, "补丁原文必须逐字符保留"
    assert item["name"] == "apply_patch"
    assert "arguments" not in item, "custom_tool_call 不该带 JSON arguments"

    # 对照：不在 freeform 集合里的工具仍然是 function_call
    resp2 = format_responses_completion(
        resp_id="resp_2", model="gpt-4o", output_text="", input_tokens=1, output_tokens=1,
        tool_calls=[{"id": "c2", "function": {"name": "bash", "arguments": "{\"command\": \"ls\"}"}}],
        freeform_tools={"apply_patch"},
    )
    fn_items = [i for i in resp2["output"] if i["type"] == "function_call"]
    assert len(fn_items) == 1
    assert json.loads(fn_items[0]["arguments"])["command"] == "ls"
    print("test_completion_emits_custom_tool_call_item: PASSED")


def test_stream_emits_custom_tool_events_end_to_end():
    provider = PatchToolProvider()
    srv._make_provider = lambda key, client=None: provider
    client = TestClient(srv.app)
    with client.stream(
        "POST",
        "/v1/responses",
        json={
            "model": "gpt-4o",
            "input": "改一下 hello.py",
            "tools": [{"type": "custom", "name": "apply_patch", "description": "Apply a patch"}],
            "stream": True,
        },
    ) as r:
        assert r.status_code == 200
        raw = "".join(r.iter_text())

    assert provider.last_tools_prompt_has_input, (
        "上游提示词里没有 freeform 工具的参数契约——模型无从知道原文放哪个字段"
    )
    assert "response.custom_tool_call_input.delta" in raw, (
        f"流里没有任何 custom_tool_call 事件，客户端收不到这次调用:\n{raw[:1200]}"
    )
    assert "response.custom_tool_call_input.done" in raw
    assert "response.function_call_arguments.delta" not in raw, (
        "freeform 工具被当成 function 工具发增量：客户端会按 JSON 解析补丁文本并失败"
    )

    # 逐事件核对 output_item.done 里的项形状
    done_items = []
    for line in raw.splitlines():
        if not line.startswith("data: "):
            continue
        body = line[6:]
        if body.strip() == "[DONE]":
            continue
        evt = json.loads(body)
        if evt.get("type") == "response.output_item.done":
            done_items.append(evt["item"])
    custom_items = [i for i in done_items if i.get("type") == "custom_tool_call"]
    assert len(custom_items) == 1, f"没有 custom_tool_call 输出项: {done_items}"
    assert custom_items[0]["input"] == PATCH
    assert custom_items[0]["name"] == "apply_patch"
    print("test_stream_emits_custom_tool_events_end_to_end: PASSED")


def run_all():
    test_freeform_declaration_gets_an_executable_schema()
    test_freeform_call_replay_is_assistant_not_user()
    test_custom_tool_input_text_unwraps_three_shapes()
    test_freeform_names_only_custom_declarations()
    test_completion_emits_custom_tool_call_item()
    test_stream_emits_custom_tool_events_end_to_end()
    print("\n✅ All Responses custom/freeform tool tests passed!")


if __name__ == "__main__":
    run_all()