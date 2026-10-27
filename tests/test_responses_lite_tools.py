"""[RESPONSES-LITE] 最新的 Codex 模型把工具声明放在 `input` 里，网关必须去那里找

Codex 0.145.0 的模型目录里有三个 `use_responses_lite: true` 的条目
（`gpt-5.6-sol` / `gpt-5.6-terra` / `gpt-5.6-luna`，实测 `codex debug models`）。
它们换了一种请求形状：顶层 `tools` 是 `None`、`instructions` 是空串，工具声明
装在 `input` 的第 0 个条目里——

    {"type": "additional_tools", "role": "developer",
     "tools": [{"type": "custom", "name": "exec", ...},
               {"type": "function", "name": "wait", ...},
               {"type": "namespace", "name": "collaboration", "tools": [...]},
               {"type": "function", "name": "request_user_input", ...}]}

网关只读 `req.tools`，于是一条 `additional_tools` 落进转换器的 `elif "role" in item`
分支，变成一条内容为空的 `developer` 消息，声明整段消失。

实测（真实 codex 0.145.0，`-m gpt-5.6-sol`，脚本化上游）：

    has_tools_section: False   declared_tools: []   prompt_len: 23955
    客户端退出码 0，`E2E_MARKER` 没有被创建

即：模型从未被告知存在任何工具，agent 的每一轮都是一个**无声的空转**——HTTP 200、
正常的正文回答、没有任何错误，而 `exec_command` / `apply_patch` 一次都不会被调用。

本套件把声明搬进 `tools`（与 `_flatten_namespaces` 同一个入口层），因此下游的
**每一处**读取都自动看见它：提示词、解析白名单、freeform 名单、namespace 映射。
用例分别钉住这四处，因为只修提示词会让模型写出调用、再被白名单当作未声明工具
丢弃——症状与修复前一模一样，只是更难追。

运行: .venv/bin/python tests/test_responses_lite_tools.py
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
os.environ["GATEWAY_DISABLE_PROMPT_FOLDING"] = "1"
os.environ["GATEWAY_DISABLE_SINGLEFLIGHT"] = "1"
for _k in ("QWEN", "DEEPSEEK", "DOUBAO", "KIMI", "GLM"):
    os.environ[f"{_k}_AUTH_TOKEN"] = f"fake_{_k.lower()}"

from fastapi.testclient import TestClient  # noqa: E402

import server as srv  # noqa: E402
from providers.anthropic_compat import format_tools_system_prompt  # noqa: E402
from providers.base import flatten_messages  # noqa: E402
from providers.responses_compat import (  # noqa: E402
    ResponsesRequest,
    convert_responses_to_gateway_messages,
)

FIXTURE = (
    Path(__file__).resolve().parent / "fixtures" / "codex_responses_lite_additional_tools.json"
)
# 真实抓包：`codex exec -m gpt-5.6-sol` 发往 `/v1/responses` 的 `input[0]`，逐字保存。
ADDITIONAL_TOOLS = json.loads(FIXTURE.read_text(encoding="utf-8"))

_GUARD_HITS: list[str] = []


def _guarded_factory(key, client=None, **kw):
    """绊线：没有安装脚本化 provider 的用例在这里失败，而不是真���触碰上游。"""
    _GUARD_HITS.append(key)
    raise AssertionError("测试发出了未安装假 provider 的上游请求")


srv._make_provider = _guarded_factory


def _lite_body(**over) -> dict:
    body = {
        "model": "gpt-5.6-sol",
        "instructions": "",
        "tools": None,
        "input": [
            ADDITIONAL_TOOLS,
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "list the files"}],
            },
        ],
        "stream": False,
    }
    body.update(over)
    return body


class RecordingProvider:
    """记录上游真正收到的提示词，并按脚本回答。"""

    script: list = []

    def __init__(self):
        self.prompt = ""

    async def chat(self, messages, model, **kw):
        self.prompt = flatten_messages(messages)
        for delta, meta in RecordingProvider.script:
            yield delta, meta


def _post(body: dict, script: list):
    provider = RecordingProvider()
    RecordingProvider.script = script
    srv._make_provider = lambda key, client=None, **kw: provider
    try:
        r = TestClient(srv.app).post("/v1/responses", json=body)
        assert r.status_code == 200, r.text
        return r, provider
    finally:
        srv._make_provider = _guarded_factory


def _tool_block(prompt: str, name: str) -> str:
    """取出某个工具自己那一块（`- `name`: ...` 到下一个条目之前）。"""
    marker = f"- `{name}`"
    start = prompt.find(marker)
    if start < 0:
        return ""
    nxt = prompt.find("\n- `", start + len(marker))
    return prompt[start:] if nxt < 0 else prompt[start:nxt]


def _text_answer(text: str = "done") -> list:
    return [(text, {}), ("", {"conversation_id": None})]


def _tool_answer(name: str, args: dict) -> list:
    payload = json.dumps({"name": name, "arguments": args}, ensure_ascii=False)
    return [
        (f"```tool_call\n{payload}\n```", {}),
        ("", {"conversation_id": None}),
    ]


# ------------------------------------------------ 1. 声明必须到达模型

def test_additional_tools_reach_the_prompt_the_upstream_receives():
    """真实抓包里的四个声明（含一个 namespace 容器）都必须出现在上游提示词里。

    修复前实测：
        has_tools_section: False   declared_tools: []
        （`input[0]` 被渲染成 `{"role": "developer", "content": ""}`）

    `exec` 是 freeform（custom）工具，`collaboration` 是装有六个子工具的
    namespace 容器——两者都不能按普通 function 渲染（§39 / §62）。
    """
    r, provider = _post(_lite_body(), _text_answer())
    prompt = provider.prompt

    assert "## Tools Available" in prompt, (
        "上游提示词里没有工具声明段——responses-lite 的 `additional_tools` 被丢掉了，"
        "模型从未被告知存在任何工具（实测 codex 退出码 0、一个工具都没执行）"
    )
    for name in ("exec", "wait", "request_user_input"):
        assert _tool_block(prompt, name), f"顶层声明的工具 `{name}` 没有出现在提示词里"
    for name in ("spawn_agent", "wait_agent", "send_message"):
        assert _tool_block(prompt, name), f"namespace 子工具 `{name}` 没有展平进提示词"
    # namespace 容器本身不可调用：它可以在「不可调用」分组里出现，但绝不能带上
    # `Args:` 变成一个可调用条目（§62：模型调它，客户端回 unsupported call）。
    callable_ns = [l for l in prompt.splitlines() if l.startswith("- `collaboration`:")]
    assert not callable_ns, (
        "namespace 容器被渲染成了一个可调用条目（§62：模型会去调它，客户端回 "
        f"unsupported call）: {callable_ns!r}"
    )
    print("test_additional_tools_reach_the_prompt_the_upstream_receives: PASSED")


def test_the_declaration_is_consumed_not_replayed_as_an_empty_message():
    """搬运完成后，`additional_tools` 条目不得在消息里留下一条空发言。

    修复前它落进 `elif "role" in item` 分支，产出一条 `content: ""` 的
    `developer` 消息——一个什么都不说的发言被当成常驻指令压进上下文。
    """
    msgs = convert_responses_to_gateway_messages(ResponsesRequest(**_lite_body()))
    empties = [m for m in msgs if isinstance(m.get("content"), str) and not m["content"].strip()]
    assert not empties, (
        f"`additional_tools` 条目留下了内容为空的消息: {empties!r}——"
        "声明已经搬进 tools，这个壳子不该再作为发言出现"
    )
    print("test_the_declaration_is_consumed_not_replayed_as_an_empty_message: PASSED")


# ------------------------------------------- 2. 白名单必须看到同一份声明

def test_a_declared_lite_tool_call_is_not_discarded_as_undeclared():
    """模型调用 `wait`，客户端必须收到 `function_call`（而不是被白名单当文本）。

    这是「只修提示词」的失败形态：模型学会了调用，网关却按未声明工具把它退成
    正文——用户看到的仍然是「agent 不动」，而错误面比修复前更隐蔽。
    """
    r, provider = _post(_lite_body(), _tool_answer("wait", {"cell_id": "c1", "yield_time_ms": 1000}))
    calls = [i for i in r.json().get("output", []) if i.get("type") == "function_call"]
    assert calls, (
        "`additional_tools` 里声明的 `wait` 调用没有作为 function_call 回给客户端"
        f"（被解析白名单丢弃了？）: {json.dumps(r.json(), ensure_ascii=False)[:600]}"
    )
    assert calls[0]["name"] == "wait", f"调用名被改写: {calls[0]['name']!r}"
    print("test_a_declared_lite_tool_call_is_not_discarded_as_undeclared: PASSED")


def test_the_hoisted_freeform_tool_keeps_its_freeform_contract():
    """`exec` 是 custom 工具：调用必须报成 `custom_tool_call`，参数是原文。

    声明被搬运后，§39 的三条链路（freeform 契约、`freeform_tool_names` 出口分派、
    `custom_tool_input_text` 反解）都必须自动生效——它们读的都是 `req.tools`。
    """
    script = _tool_answer("exec", {"input": "return await tools.exec_command('ls')"})
    r, provider = _post(_lite_body(), script)
    assert '"input"' in _tool_block(provider.prompt, "exec"), (
        "freeform 工具 `exec` 没有被赋予 `{\"input\": ...}` 参数契约——"
        "模型会被引导去调一个不收参数的工具"
    )
    items = r.json().get("output", [])
    custom = [i for i in items if i.get("type") == "custom_tool_call"]
    assert custom, (
        "`exec` 的调用被报成了别的东西（客户端按 type 分派，会整条丢弃）: "
        f"{json.dumps(items, ensure_ascii=False)[:600]}"
    )
    assert custom[0]["name"] == "exec"
    print("test_the_hoisted_freeform_tool_keeps_its_freeform_contract: PASSED")


def test_the_hoisted_namespace_sub_tool_carries_its_namespace():
    """`collaboration` 的子工具调用必须带上 namespace（客户端查表键是拼接的）。"""
    r, _provider = _post(_lite_body(), _tool_answer("spawn_agent", {"message": "say pong"}))
    calls = [i for i in r.json().get("output", []) if i.get("type") == "function_call"]
    assert calls, "namespace 子工具的调用没有回给客户端"
    assert calls[0]["name"] == "spawn_agent", f"名字被改写成 {calls[0]['name']!r}"
    assert calls[0].get("namespace") == ADDITIONAL_TOOLS["tools"][-1]["name"], (
        "调用项没有携带 namespace——真实 Codex 的查表键是 namespace+name 无分隔符拼接，"
        f"缺了它会回 `unsupported call: {calls[0]['name']}`（实测）"
    )
    print("test_the_hoisted_namespace_sub_tool_carries_its_namespace: PASSED")


def test_a_streaming_lite_tool_call_reaches_the_client():
    """流式路径读的也是同一份 `req.tools`，但它是独立的收集点（`freeform_names`
    在流的起点算一次），必须单独确认。

    用 `exec`（freeform）而不是普通 function 是有意的：修复前 `req.tools` 为
    `None`，`freeform_tool_names` 返回空集，于是同一个调用被报成
    `function_call` + JSON arguments（实测）。普通 function 在流式路径上碰巧
    能过，这条用例必须钉住「声明被搬运之后，出口分派也一起生效」。
    """
    script = _tool_answer("exec", {"input": "return await tools.exec_command('ls')"})
    r, _provider = _post(_lite_body(stream=True), script)
    events = []
    for line in r.text.splitlines():
        if line.startswith("data: "):
            try:
                events.append(json.loads(line[len("data: "):]))
            except json.JSONDecodeError:
                pass
    added = [
        e for e in events
        if e.get("type") == "response.output_item.added"
        and (e.get("item") or {}).get("type") == "custom_tool_call"
    ]
    assert added, (
        "流式路径没有把 `exec` 报成 `custom_tool_call`（客户端按 type 分派，"
        "function_call 会被整条丢弃）: "
        f"{[(e.get('type'), (e.get('item') or {}).get('type')) for e in events][:20]}"
    )
    assert added[0]["item"]["name"] == "exec"
    print("test_a_streaming_lite_tool_call_reaches_the_client: PASSED")


# --------------------------------------------------------- 3. 零回归

def test_a_flat_tools_array_is_unchanged():
    """控制方向：顶层 `tools` 数组（`gpt-5.5` 的形状）逐字不变。

    合并逻辑写在 `tools` 的入口层，一旦写错就会连普通请求一起改坏——而普通形状
    是绝大多数请求。
    """
    tools = [
        {
            "type": "function",
            "name": "exec_command",
            "description": "Run a command",
            "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}},
        }
    ]
    body = {
        "model": "gpt-5.5",
        "input": "go",
        "tools": tools,
        "stream": False,
    }
    _r, provider = _post(body, _text_answer())
    assert provider.prompt.count("## Tools Available") == 1
    assert _tool_block(provider.prompt, "exec_command"), "普通工具声明被改坏了"
    expected = format_tools_system_prompt(tools)
    assert expected in provider.prompt, (
        "普通形状渲染出的工具契约与修复前不一致——合并逻辑改动了不该改动的请求"
    )
    print("test_a_flat_tools_array_is_unchanged: PASSED")


def test_top_level_declarations_win_over_a_duplicate():
    """同名声明：顶层 `tools` 是显式通道，优先；同一个工具不得出现两次。"""
    dup = {
        "type": "function",
        "name": "wait",
        "description": "TOP-LEVEL-CONTRACT",
        "parameters": {"type": "object", "properties": {"cell_id": {"type": "string"}}},
    }
    body = _lite_body(tools=[dup])
    msgs = convert_responses_to_gateway_messages(ResponsesRequest(**body))
    prompt = "\n".join(m["content"] for m in msgs if isinstance(m.get("content"), str))
    assert _tool_block(prompt, "wait").count("TOP-LEVEL-CONTRACT") == 1, (
        "同名声明没有按「顶层优先、只出现一次」合并: "
        f"{_tool_block(prompt, 'wait')[:200]!r}"
    )
    assert _tool_block(prompt, "exec"), "合并同名声明时把其余声明一起丢了"
    print("test_top_level_declarations_win_over_a_duplicate: PASSED")


def run_all():
    test_additional_tools_reach_the_prompt_the_upstream_receives()
    test_the_declaration_is_consumed_not_replayed_as_an_empty_message()
    test_a_declared_lite_tool_call_is_not_discarded_as_undeclared()
    test_the_hoisted_freeform_tool_keeps_its_freeform_contract()
    test_the_hoisted_namespace_sub_tool_carries_its_namespace()
    test_a_streaming_lite_tool_call_reaches_the_client()
    test_a_flat_tools_array_is_unchanged()
    test_top_level_declarations_win_over_a_duplicate()
    assert not _GUARD_HITS, (
        f"有 {len(_GUARD_HITS)} 次上游调用未经脚本化 provider: {sorted(set(_GUARD_HITS))}"
    )
    print("\n✅ All responses-lite tool tests passed! (8 cases)")


if __name__ == "__main__":
    sys.exit(run_all())
