r"""`tool_choice` 的两个裁判必须给出同一个判断 (TOOL-LOOP)

§34b 把 `tool_choice` 拆成两件事：解析端的**白名单**（保证结果）与提示词里的
**一句话**（降低发生概率）。两者读同一个客户端字段，却是两处独立实现，于是
在两种形状上判得不一样——而这两处都不会报错：

  A. `{"type": "none"}` —— Anthropic **合法**的「本回合不要调用工具」写法。
     白名单认（`set()`），提示词不认（没有 `c_type == "none"` 分支），
     模型因此从没被告知别调工具。
  B. 点名了一个**根本没声明**的工具（`{"type": "tool", "name": "Nope"}`）。
     白名单按「无法识别」失败开放（保留全部已声明工具），提示词却仍然写
     `You MUST call the tool \`Nope\``——要求模型调用一个它连 schema 都
     没见过的工具，而解析端一定会把那个调用挡掉。

实测（脚本化上游，零真实上游调用）：

    shape                  allowed   directive
    none                   none      Do not call any tool …
    {type:none}            none      （空）                        ← A
    {type:tool,Nope}       all       You MUST call the tool `Nope` ← B
    {fn:{name:Nope}}       all       You MUST call the tool `Nope` ← B
    {flat:Nope}            all       You MUST call the tool `Nope` ← B

本套件断言的不只是这两格修好了，而是**两个裁判在所有形状上一致**：提示词
禁止时解析端必须禁止，提示词点名时该名字必须在白名单内，白名单为空时提示词
必须说出来。逐格枚举只能覆盖想到的形状，而这类缺陷每次都是漏掉一格。

运行: .venv/bin/python tests/test_tool_choice_agreement.py
"""

from __future__ import annotations

import json
import os
import re
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
from providers.anthropic_compat import format_tools_system_prompt  # noqa: E402
from providers.tool_streamer import resolve_allowed_tools  # noqa: E402

TOOLS_ANTHROPIC = [
    {
        "name": "Read",
        "description": "Read a file",
        "input_schema": {
            "type": "object",
            "properties": {"file_path": {"type": "string"}},
            "required": ["file_path"],
        },
    }
]
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

# 客户端能发出的、语义各不相同的 tool_choice 形状（三个协议的并集）。
SHAPES = [
    None,
    "auto",
    "none",
    "required",
    "any",
    {"type": "none"},
    {"type": "any"},
    {"type": "auto"},
    {"type": "tool", "name": "Read"},
    {"type": "tool", "name": "Nope"},
    {"type": "function", "function": {"name": "Read"}},
    {"type": "function", "function": {"name": "Nope"}},
    {"type": "function", "name": "Read"},
    {"type": "function", "name": "Nope"},
    {},
    {"type": "weird"},
    ["not", "a", "choice"],
]

_NO_TOOL = "Do not call any tool"
_MUST_ONE = "MUST call at least one tool"


class PromptProvider:
    """记录上游真正收到的 messages，然后产出一句普通回答。"""

    last_prompt = ""

    async def chat(self, messages, model, **kw):
        PromptProvider.last_prompt = json.dumps(messages, ensure_ascii=False)
        yield "ok", {}


_GUARD_HITS: list[str] = []


def _guarded_factory(key):
    """绊线：没有先安装脚本化 provider 的请求在这里立刻失败。

    与 §35 的约定一致——一个忘记安装假 provider 的用例会带着**真实凭据**向上游
    发一次真实请求，那正是「绝不加深任何供应商风控分数」所禁止的。
    """
    _GUARD_HITS.append(key)
    raise AssertionError(
        "测试发出了未安装假 provider 的上游请求（所有请求必须经 _run(fn)）"
    )


srv._make_provider = _guarded_factory


def _run(fn):
    PromptProvider.last_prompt = ""
    srv._make_provider = lambda key: PromptProvider()
    try:
        return fn()
    finally:
        srv._make_provider = _guarded_factory


_CLIENT = TestClient(srv.app)


def _post(path: str, body: dict) -> dict:
    def go():
        r = _CLIENT.post(path, json=body)
        assert r.status_code == 200, f"{path} -> {r.status_code}: {r.text[:400]}"
        return json.loads(PromptProvider.last_prompt)

    return _run(go)


def _anthropic_prompt(tool_choice) -> str:
    msgs = _post(
        "/v1/messages",
        {
            "model": "claude-3-7-sonnet",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "say hello in prose"}],
            "tools": TOOLS_ANTHROPIC,
            "tool_choice": tool_choice,
        },
    )
    return json.dumps(msgs, ensure_ascii=False)


def _openai_prompt(tool_choice) -> str:
    msgs = _post(
        "/v1/chat/completions",
        {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "say hello in prose"}],
            "tools": TOOLS_OPENAI,
            "tool_choice": tool_choice,
        },
    )
    return json.dumps(msgs, ensure_ascii=False)


# ---------------------------------------------------------------- A. dict none

def test_anthropic_dict_none_reaches_the_prompt():
    """Anthropic 的 `{"type":"none"}` 必须和字符串 `"none"` 说同一句话。"""
    by_shape = {
        "str": _anthropic_prompt("none"),
        "dict": _anthropic_prompt({"type": "none"}),
    }
    assert _NO_TOOL in by_shape["str"], (
        f"字符串形式的 none 没有进入上游提示词（对照组失败，说明测量本身有问题）: "
        f"{by_shape['str'][:400]}"
    )
    assert _NO_TOOL in by_shape["dict"], (
        "tool_choice={\"type\":\"none\"} 没有追加禁调用指令：白名单会挡掉模型写出的"
        f"调用，但模型从未被告知别调，提示词与解析端判得不一样。实测提示词: "
        f"{by_shape['dict'][:400]}"
    )
    print("test_anthropic_dict_none_reaches_the_prompt: PASSED")


def test_openai_dict_none_reaches_the_prompt():
    """OpenAI 路径读同一个提示词构造器，两个形状也必须一致。"""
    by_shape = {
        "str": _openai_prompt("none"),
        "dict": _openai_prompt({"type": "none"}),
    }
    assert _NO_TOOL in by_shape["str"], by_shape["str"][:400]
    assert _NO_TOOL in by_shape["dict"], (
        f"tool_choice={{\"type\":\"none\"}} 没有进入上游提示词: {by_shape['dict'][:400]}"
    )
    print("test_openai_dict_none_reaches_the_prompt: PASSED")


# ------------------------------------------------- B. named but undeclared

def test_named_undeclared_tool_is_never_demanded():
    """点名一个未声明的工具：两侧都必须失败开放，而不是一边放手一边下命令。"""
    for prompt_fn, tools, name in (
        (_openai_prompt, TOOLS_OPENAI, "openai"),
        (_anthropic_prompt, TOOLS_ANTHROPIC, "anthropic"),
    ):
        prompt = prompt_fn({"type": "tool", "name": "Nope"})
        assert f"MUST call the tool `Nope`" not in prompt, (
            f"[{name}] 提示词要求模型调用一个它从未见过的工具 `Nope`："
            f"白名单会挡掉这个调用，模型却被下了命令。实测提示词: {prompt[:400]}"
        )
        # 失败开放：白名单没有因此收紧（收紧只会把一个可用回合变成空回合）
        assert resolve_allowed_tools(tools, {"type": "tool", "name": "Nope"}) == {"Read"}, (
            f"[{name}] 点名未声明工具时白名单被收紧了"
        )
    print("test_named_undeclared_tool_is_never_demanded: PASSED")


# ------------------------------------------- the property, not the two cells

def test_the_two_judges_agree_over_every_shape():
    """两个裁判在所有形状上一致——按性质断言，而不是逐格枚举。"""
    problems = []
    for tools, dialect in ((TOOLS_ANTHROPIC, "anthropic"), (TOOLS_OPENAI, "openai")):
        for shape in SHAPES:
            allowed = resolve_allowed_tools(tools, shape)
            prompt = format_tools_system_prompt(tools, tool_choice=shape)
            named = re.search(r"You MUST call the tool `([^`]+)`", prompt)
            # 解析端禁止一切，提示词就必须说出来
            if allowed == set() and _NO_TOOL not in prompt:
                problems.append(
                    f"[{dialect}] {shape!r}: 白名单禁止全部工具，提示词却没有说"
                )
            # 提示词禁止调用，解析端就必须禁止
            if _NO_TOOL in prompt and allowed != set():
                problems.append(
                    f"[{dialect}] {shape!r}: 提示词禁止调用，白名单却放行 "
                    f"{sorted(allowed) if allowed else 'unfiltered'}"
                )
            # 提示词点名要求调用，该名字就必须在白名单里
            if named and allowed is not None and named.group(1) not in allowed:
                problems.append(
                    f"[{dialect}] {shape!r}: 提示词要求 `{named.group(1)}`，"
                    f"白名单只允许 {sorted(allowed)}"
                )
            # 反向：白名单只留一个工具时，提示词不应要求别的名字
            if named and allowed == set():
                problems.append(f"[{dialect}] {shape!r}: 提示词在禁调用回合里点名要求调用")
    assert not problems, "tool_choice 的两个裁判判断不一致:\n  " + "\n  ".join(problems)
    print("test_the_two_judges_agree_over_every_shape: PASSED")


# ------------------------------------------------------------ zero regression

def test_allowlist_semantics_are_unchanged():
    """白名单语义零回归（本次只允许「提示词开始同意解析端」，不允许动白名单）。"""
    tools = TOOLS_OPENAI
    assert resolve_allowed_tools(tools, None) == {"Read"}
    assert resolve_allowed_tools(tools, "auto") == {"Read"}
    assert resolve_allowed_tools(tools, "none") == set()
    assert resolve_allowed_tools(tools, "required") == {"Read"}
    assert resolve_allowed_tools(tools, {"type": "none"}) == set()
    assert resolve_allowed_tools(tools, {"type": "any"}) == {"Read"}
    assert resolve_allowed_tools(tools, {"type": "function", "function": {"name": "Read"}}) == {"Read"}
    assert resolve_allowed_tools(tools, {"type": "function", "name": "Read"}) == {"Read"}
    assert resolve_allowed_tools(tools, {"type": "bogus"}) == {"Read"}
    assert resolve_allowed_tools(tools, {"type": "tool", "name": "Nope"}) == {"Read"}
    assert resolve_allowed_tools(None, "none") is None
    # 未识别形状不得改变提示词（与 test_tool_schema 同一约定）
    for junk in ({}, {"type": "weird"}, ["not", "a", "choice"], 42):
        assert format_tools_system_prompt(tools, tool_choice=junk) == \
            format_tools_system_prompt(tools), f"未知 tool_choice 改变了输出: {junk!r}"
    print("test_allowlist_semantics_are_unchanged: PASSED")


def run_all():
    tests = [
        test_anthropic_dict_none_reaches_the_prompt,
        test_openai_dict_none_reaches_the_prompt,
        test_named_undeclared_tool_is_never_demanded,
        test_the_two_judges_agree_over_every_shape,
        test_allowlist_semantics_are_unchanged,
    ]
    for fn in tests:
        fn()
    # 故障转移会把绊线的断言包成 502/503，所以「没碰上游」的直接证据是计数器
    assert not _GUARD_HITS, (
        f"有 {len(_GUARD_HITS)} 次上游调用未经脚本化 provider: {sorted(set(_GUARD_HITS))}"
    )
    print(f"\n✅ All tool_choice agreement tests passed! ({len(tests)} cases)")


if __name__ == "__main__":
    run_all()
