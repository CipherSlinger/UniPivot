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


def test_nameless_builtin_tools_do_not_render_as_empty_bullets():
    r"""[TOOL-LOOP] 真实 Codex 请求里有两类**没有 `name`** 的内置工具声明。

    实测（codex-cli 0.145.0，`codex exec` 原始 `/v1/responses` 请求体）：
      - `{"type": "tool_search", "execution": "client", "parameters": {...}}`
      - `{"type": "web_search", "external_web_access": false}`
    两者都没有 `name` 字段。旧的渲染分支无条件 `t.get("name", "")`，于是提示词
    末尾出现 `- \`\`: ` + `Args: {}` 这样的**空名条目**——模型看到的是一个可以
    调用、却没有任何名字的工具。这正是「声明列表末尾的空名条目」的真正来源：
    它不是名字被损坏，而是内置工具本来就没有名字。

    处理必须与「模型能调用什么」一致：没有名字的声明**不可能**被调用（模型写不出
    调用名，`tool_names_of` 也不会放行），所以它不该作为可调用工具出现在提示词里。
    但它携带的信息（`tool_search` 会「让下一轮看到更多工具」）仍然值得让模型知道，
    因此降级为一段说明，而不是静默丢弃，更不能是空名条目。
    """
    tool_search = {
        "type": "tool_search",
        "execution": "client",
        "description": "# Tool discovery\n\nSearches over deferred tool metadata.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
    }
    web_search = {"type": "web_search", "external_web_access": False}
    real = {
        "type": "function",
        "name": "exec_command",
        "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}},
    }

    prompt = format_tools_system_prompt([real, tool_search, web_search])

    assert "- ``" not in prompt, (
        "提示词里出现了空名工具条目——模型看到一个可调用却没有名字的工具:\n"
        + prompt
    )
    assert "`exec_command`" in prompt, "有名字的正常工具不该被牵连"
    print("test_nameless_builtin_tools_do_not_render_as_empty_bullets: PASSED")


# ---------------------------------------------------------------------------
# [TOOL-LOOP] `namespace` 工具声明（真实 Codex 0.145.0 的 `multi_agent_v1`）
#
# 实测（`codex exec` 原始 `/v1/responses` 请求体，10 个工具里的第 6 个）：
#
#   {"type": "namespace", "name": "multi_agent_v1",
#    "description": "Tools for spawning and managing sub-agents.",
#    "tools": [{"type": "function", "name": "close_agent", "parameters": {...}},
#              ... resume_agent / send_input / spawn_agent / wait_agent ...]}
#
# 也就是说：这不是一个工具，而是**五个可调用的 function 子工具的容器**。
# 网关此前把 `namespace` 落进通用分支，取它的顶层 `name`/`description`、找不到
# `parameters`，于是渲染成
#
#   - `multi_agent_v1`: Tools for spawning and managing sub-agents.
#     Args: {}
#
# 两个独立的错误同时发生：
#   1. 五个子工具的名字与**全部参数 schema** 消失 —— 模型不知道它们存在，即使
#      猜得到名字也不知道参数（与 `apply_patch` 被告知 `Args: {}` 同一缺陷类）；
#   2. 它被告知可调用的名字是 `multi_agent_v1`，而真实客户端**拒绝**这个名字。
#
# 客户端路由实测（同一个 mock 上游，只改回传的 call item 形状；判据是工具**有没
# 有真的执行**——每一臂里 mock 模型都照样说 "Sub-agent spawned."，回复措辞无法区分）：
#   - `name="multi_agent_v1"`                    -> `unsupported call: multi_agent_v1`
#   - 裸名 `spawn_agent`（不带 namespace）        -> `unsupported call: spawn_agent`
#   - 裸名 + 真实 `"namespace"`                   -> `collab: SpawnAgent` -> `{"agent_id":…}`
#   - 裸名 + `"bogus_ns"`                        -> `unsupported call: bogus_nsspawn_agent`
#   - 普通工具 `exec_command` + 任意 namespace    -> `unsupported call: multi_agent_v1exec_command`
#   - `multi_agent_v1.spawn_agent` / `::` / `__`  -> 全部 `unsupported call`
#     （客户端把 namespace 与 name **无分隔符拼接**后再查表：这既要求子工具带上
#       namespace，也要求普通工具绝不能带）
#
# 所以**声明侧**的正确形状是：把子工具按**裸名**、带完整 schema 放进可调用列表，
# namespace 自己不可调用（只作为分组说明）；**出口侧**则要把 namespace 还原回
# 调用项。两侧规则相反，只修一侧会把另一侧变成回归——两个方向都由
# `test_namespaced_sub_tool_call_carries_its_namespace_on_the_wire` 锁住。
# ---------------------------------------------------------------------------

NS_FIXTURE = (
    Path(__file__).resolve().parent / "fixtures" / "codex_multi_agent_v1_namespace.json"
)
NS_SUB_TOOLS = ("close_agent", "resume_agent", "send_input", "spawn_agent", "wait_agent")


def _namespace_decl() -> dict:
    return json.loads(NS_FIXTURE.read_text(encoding="utf-8"))


def _tool_block(prompt: str, name: str) -> str:
    """取出某个工具**自己那一块**（`- `name`: ...` 到下一个条目之前）。

    断言必须打在该工具自己的块上：整段提示词里找 `"targets"` 之类的字符串，
    可能由另一个工具的参数提供，用例就在为错误的原因通过。
    """
    marker = f"- `{name}`"
    start = prompt.find(marker)
    if start < 0:
        return ""
    nxt = prompt.find("\n- `", start + len(marker))
    return prompt[start:] if nxt < 0 else prompt[start:nxt]


def test_namespace_declaration_renders_its_sub_tools():
    ns = _namespace_decl()
    tools = [ns, {"type": "function", "name": "exec_command",
                  "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}}}]

    prompt = format_tools_system_prompt(tools)

    assert "- `multi_agent_v1`: Tools for spawning and managing sub-agents." not in prompt, (
        "namespace 仍被渲染成一个可调用工具，且参数是空的 Args: {}——"
        "模型会去调 `multi_agent_v1`，而真实 Codex 回 `unsupported call: multi_agent_v1`。"
        f"实测渲染:\n{_tool_block(prompt, 'multi_agent_v1')}"
    )
    for sub in NS_SUB_TOOLS:
        block = _tool_block(prompt, sub)
        assert block, (
            f"namespace 的子工具 `{sub}` 完全没有出现在提示词里——模型不可能调用它。"
            f"实测提示词里的工具条目: {[l for l in prompt.splitlines() if l.startswith('- `')]}"
        )
    # 参数 schema 必须真的到达模型（不是只补了名字）
    assert '"required":["targets"]' in _tool_block(prompt, "wait_agent"), (
        "wait_agent 的必填参数 targets 没有保真:\n" + _tool_block(prompt, "wait_agent")
    )
    assert '"required":["target"]' in _tool_block(prompt, "close_agent"), (
        "close_agent 的必填参数 target 没有保真:\n" + _tool_block(prompt, "close_agent")
    )
    assert '"required":["id"]' in _tool_block(prompt, "resume_agent"), (
        "resume_agent 的必填参数 id 没有保真:\n" + _tool_block(prompt, "resume_agent")
    )
    # spawn_agent 没有 required，但可选参数是它唯一的契约
    spawn_block = _tool_block(prompt, "spawn_agent")
    for prop in ("fork_context", "items", "message", "model", "service_tier"):
        assert f'"{prop}"' in spawn_block, (
            f"spawn_agent 的可选参数 `{prop}` 丢失:\n{spawn_block}"
        )
    # 嵌套结构（send_input.items 是对象数组）必须保真
    send_block = _tool_block(prompt, "send_input")
    assert '"audio_url"' in send_block and '"image_url"' in send_block, (
        "send_input.items 的嵌套字段没有保真:\n" + send_block
    )
    # 同一个请求里的普通工具不受影响
    assert "- `exec_command`" in prompt
    print("test_namespace_declaration_renders_its_sub_tools: PASSED")


def test_namespace_is_not_a_callable_name_but_is_explained():
    ns = _namespace_decl()
    prompt = format_tools_system_prompt([ns])
    # 名字仍然让模型看得见（分组信息），但必须写明不可调用
    ns_line = [l for l in prompt.splitlines() if "multi_agent_v1" in l]
    assert ns_line, "namespace 的分组信息被静默丢弃了"
    joined = " ".join(ns_line).lower()
    assert "not callable" in joined or "not directly callable" in joined, (
        "提示词提到了 multi_agent_v1，却没有说明它本身不可调用——"
        f"模型会去调用它（真实客户端回 unsupported call）。实测: {ns_line}"
    )
    print("test_namespace_is_not_a_callable_name_but_is_explained: PASSED")


def test_namespace_sub_tools_enter_the_parse_allowlist():
    """解析层白名单必须与提示词一致。

    [TOOL-LOOP] 提示词只是「降低概率」，白名单才是「保证结果」。如果只把子工具
    渲染进提示词、白名单里仍是 `multi_agent_v1`，模型写出的 `spawn_agent` 调用会被
    `_extract_tools_from_data(allowed=...)` 当作未声明工具丢弃，客户端什么也收不到
    —— 症状与修复前一样，只是更难追。所以白名单的输入必须是**展平后**的声明，
    而且展平只有一份定义（`flatten_namespace_tools`），由请求模型统一应用。
    """
    from providers.tool_streamer import resolve_allowed_tools

    req = ResponsesRequest.model_validate({
        "model": "gpt-5-codex",
        "input": "spawn an agent",
        "tools": [_namespace_decl(), {"type": "function", "name": "exec_command"}],
    })
    allowed = resolve_allowed_tools(req.tools, "auto")
    assert allowed is not None
    assert set(NS_SUB_TOOLS) <= allowed, (
        f"namespace 子工具不在解析白名单里，模型调了也会被丢弃: {sorted(allowed)}"
    )
    assert "multi_agent_v1" not in allowed, (
        "namespace 自己的名字进了白名单——那是一个真实客户端会拒绝的调用名"
        f"（unsupported call）: {sorted(allowed)}"
    )
    assert "exec_command" in allowed, "普通工具被牵连"
    print("test_namespace_sub_tools_enter_the_parse_allowlist: PASSED")


class SpawnAgentProvider:
    """上游模型按提示词契约写出 namespace 子工具的调用。"""

    def __init__(self):
        self.last_prompt = ""

    async def chat(self, messages, model, conversation_id=None, **kw):
        from providers.base import flatten_messages

        self.last_prompt = flatten_messages(messages)
        payload = json.dumps(
            {"name": "spawn_agent", "arguments": {"message": "say pong"}},
            ensure_ascii=False,
        )
        yield f"```tool_call\n{payload}\n```", {}
        yield "", {"conversation_id": None}


def test_namespace_sub_tool_call_round_trips_end_to_end():
    provider = SpawnAgentProvider()
    original = srv._make_provider
    srv._make_provider = lambda key, client=None, **kw: provider
    try:
        client = TestClient(srv.app)
        r = client.post(
            "/v1/responses",
            json={
                "model": "gpt-5-codex",
                "input": "spawn an agent to say pong",
                "tools": [_namespace_decl()],
                "stream": False,
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
    finally:
        srv._make_provider = original

    # 1) 模型确实被告知了这个工具的参数契约
    assert "`spawn_agent`" in provider.last_prompt, (
        "上游真正收到的提示词里没有 spawn_agent——模型无从调用它"
    )
    # 2) 调用被解析并通过白名单（否则会被当作未声明工具，作为文本返回）
    calls = [i for i in body.get("output", []) if i.get("type") == "function_call"]
    assert calls, (
        "namespace 子工具的调用没有作为 function_call 回给客户端（被白名单丢弃了？）: "
        f"{json.dumps(body, ensure_ascii=False)[:800]}"
    )
    assert calls[0]["name"] == "spawn_agent", (
        f"调用名不是裸名，客户端会拒收: {calls[0]['name']!r}"
    )
    assert json.loads(calls[0]["arguments"])["message"] == "say pong"
    print("test_namespace_sub_tool_call_round_trips_end_to_end: PASSED")


class PlainToolProvider:
    """上游模型调用一个**普通**（非 namespace）工具。"""

    def __init__(self):
        self.last_prompt = ""

    async def chat(self, messages, model, conversation_id=None, **kw):
        from providers.base import flatten_messages

        self.last_prompt = flatten_messages(messages)
        payload = json.dumps(
            {"name": "exec_command", "arguments": {"cmd": "echo hi"}}, ensure_ascii=False
        )
        yield f"```tool_call\n{payload}\n```", {}
        yield "", {"conversation_id": None}


def _emitted_items(tools: list, provider_cls) -> list:
    provider = provider_cls()
    original = srv._make_provider
    srv._make_provider = lambda key, client=None, **kw: provider
    try:
        client = TestClient(srv.app)
        r = client.post(
            "/v1/responses",
            json={
                "model": "gpt-5-codex",
                "input": "go",
                "tools": tools,
                "stream": False,
            },
        )
        assert r.status_code == 200, r.text
        return r.json().get("output", [])
    finally:
        srv._make_provider = original


def test_namespaced_sub_tool_call_carries_its_namespace_on_the_wire():
    """[TOOL-LOOP] namespace 子工具的调用项必须带 `namespace`，普通工具**绝不能**带。

    这条规则是被真实 Codex 0.145.0 路由出来的，不是猜的：客户端的工具查表键是
    `namespace + name` **无分隔符拼接**。同一条源码规则（`parse_full_name` 在
    拿不到 namespace 时返回 `(None, name)`）造成两个方向相反的后果：

      实测（codex exec，同一份 mock 上游，只改调用项的字段形状）
        `exec_command`  无 namespace                -> 真正执行（`hi`）
        `exec_command`  + `namespace`              -> `unsupported call: multi_agent_v1exec_command`
        `spawn_agent`   无 namespace                -> `unsupported call: spawn_agent`
        `spawn_agent`   + `"bogus_ns"`             -> `unsupported call: bogus_nsspawn_agent`
        `spawn_agent`   + 真实 namespace            -> `collab: SpawnAgent` -> `{"agent_id":…}`
        `spawn_agent`   + 真实 namespace（功能开关关闭的 v1 配置）-> 同样被路由

    所以「裸名 + 完整 schema」只是**声明侧**的正确形态（提示词与解析白名单，见上一
    条用例）；出口侧必须把 namespace 还原回去，而且只还原在真正属于某个 namespace
    的工具上——给普通工具加上它会立刻打断它。两个方向一起锁，因为只修一个方向会
    把另一个方向的正确行为变成回归。
    """
    ns = _namespace_decl()
    ns_name = ns["name"]
    plain = {
        "type": "function",
        "name": "exec_command",
        "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}},
    }
    tools = [ns, plain]

    # --- 方向一：namespace 子工具的调用项带 namespace，名字仍是裸名
    items = _emitted_items(tools, SpawnAgentProvider)
    calls = [i for i in items if i.get("type") == "function_call"]
    assert calls, f"namespace 子工具的调用没有回给客户端: {json.dumps(items, ensure_ascii=False)[:600]}"
    sub = calls[0]
    assert sub["name"] == "spawn_agent", (
        f"名字被改写了，客户端的查表键是 namespace+name，改坏任何一个都查不到: {sub['name']!r}"
    )
    assert sub.get("namespace") == ns_name, (
        "namespace 子工具的调用项没有携带 namespace——实测真实 Codex 会回 "
        f"`unsupported call: {sub['name']}`（同一份 mock 上游下 2 次请求、工具一次都没执行）。"
        f"实际发出的项: {json.dumps(sub, ensure_ascii=False)}"
    )

    # --- 方向二（控制组）：普通工具的调用项绝不能带 namespace
    items = _emitted_items(tools, PlainToolProvider)
    calls = [i for i in items if i.get("type") == "function_call"]
    assert calls, "普通工具的调用没有回给客户端"
    assert "namespace" not in calls[0], (
        "普通工具被加上了 namespace——实测客户端拼成 "
        f"`{ns_name}exec_command` 并回 `unsupported call`，一个本来正常的工具因此被打断。"
        f"实际发出的项: {json.dumps(calls[0], ensure_ascii=False)}"
    )

    # --- 方向三：namespace 容器本身永远不是可调用的名字
    all_names = {i.get("name") for i in _emitted_items(tools, SpawnAgentProvider)
                 if i.get("type") == "function_call"}
    assert ns_name not in all_names, (
        f"namespace 自己的名字 {ns_name!r} 变成了调用名——真实客户端回 unsupported call"
    )
    print("test_namespaced_sub_tool_call_carries_its_namespace_on_the_wire: PASSED")


def run_all():
    test_freeform_declaration_gets_an_executable_schema()
    test_freeform_call_replay_is_assistant_not_user()
    test_custom_tool_input_text_unwraps_three_shapes()
    test_freeform_names_only_custom_declarations()
    test_completion_emits_custom_tool_call_item()
    test_stream_emits_custom_tool_events_end_to_end()
    test_nameless_builtin_tools_do_not_render_as_empty_bullets()
    test_namespace_declaration_renders_its_sub_tools()
    test_namespace_is_not_a_callable_name_but_is_explained()
    test_namespace_sub_tools_enter_the_parse_allowlist()
    test_namespace_sub_tool_call_round_trips_end_to_end()
    test_namespaced_sub_tool_call_carries_its_namespace_on_the_wire()
    print("\n✅ All Responses custom/freeform tool tests passed!")


if __name__ == "__main__":
    run_all()