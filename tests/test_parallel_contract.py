r"""提示词里的工具契约必须与解析端/客户端对「并行调用」的判断一致 (TOOL-LOOP)

§68 让**背靠背的两个 ```tool_call 块**真正能被解析、被回传给客户端；§34b 又定下
`parallel_tool_calls` 的语义——只有客户端**明确声明**禁止时才裁剪到第一个，默认
允许。三个协议的两个默认值都是允许（OpenAI/Responses `parallel_tool_calls` 缺省
true，Anthropic `disable_parallel_tool_use` 缺省 false）。

但注入给上游的那份工具契约从未随之更新，它仍然写着：

    2. Only call ONE tool at a time (one tool per fenced block).
    4. After outputting a tool call, stop and wait for the tool execution result.

五家网页端上游**没有原生工具调用**，这份提示词是模型唯一能看到的契约，所以它
不是一句建议的措辞问题：模型被告知「一次只能调一个、调完就停」，就会把一个本可
并行完成的回合拆成 N 个回合——每个回合都要重新付一遍完整历史的 token，而这正是
本网关存在的 agent 场景里最贵的那部分。更糟的是解析端并不这么判：模型若照自己的
习惯一次写出两个块，网关两个都会回传（`cap_parallel_tool_calls(TWO, False) -> 2`），
于是提示词与网关的实际行为互相矛盾，而那处矛盾对客户端不可见。

实测（`format_tools_system_prompt`，默认 `parallel_allowed=None`）：

    parallel ALLOWED (default)     rule2_one-at-a-time=True  rule4_wait=True  disallow-directive=False
    parser behaviour               cap_parallel_tool_calls(TWO, False) -> 2 / 2

本套件断言的不是某一句话改好了，而是**契约与解析端在每一种设置上一致**：解析端
会放行多个调用时，提示词不得禁止；解析端会裁到一个时，提示词必须说出来。

运行: .venv/bin/python tests/test_parallel_contract.py
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
from providers.anthropic_compat import format_tools_system_prompt  # noqa: E402
from providers.tool_streamer import (  # noqa: E402
    cap_parallel_tool_calls,
    parallel_calls_disallowed,
)

TOOLS = [
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
    },
    {
        "type": "function",
        "function": {
            "name": "Grep",
            "description": "Search files",
            "parameters": {
                "type": "object",
                "properties": {"pattern": {"type": "string"}},
                "required": ["pattern"],
            },
        },
    },
]

TWO_CALLS = [("Read", {"file_path": "/tmp/a"}), ("Grep", {"pattern": "x"})]

#: 只有客户端禁止并行时，契约里才允许出现的禁止句。
FORBID = "Do NOT emit more than one tool call in a turn"
#: 客户端允许并行时，契约必须说明「多个独立调用可以写在同一回合」。
ALLOW = "back to back"


# ------------------------------------------------------- direct: the contract

def test_contract_permits_parallel_calls_when_the_client_allows():
    """客户端允许并行（三个协议的默认值）时，契约不得禁止一个它不会裁掉的形状。"""
    for setting in (None, True):
        prompt = format_tools_system_prompt(TOOLS, parallel_allowed=setting)
        assert FORBID not in prompt, (
            f"parallel_allowed={setting!r}: 客户端没有禁止并行，契约却禁止了"
        )
        assert ALLOW in prompt, (
            f"parallel_allowed={setting!r}: 解析端会保留两个调用"
            f"（cap_parallel_tool_calls -> {len(cap_parallel_tool_calls(TWO_CALLS, False))}），"
            "契约却仍写着「一次只调一个、调完就停」，模型因此把一个回合拆成 N 个，"
            f"每个回合重新付一遍完整历史的 token。实测契约片段:\n"
            + "\n".join(l for l in prompt.splitlines() if l.strip().startswith(("2.", "4.")))
        )
        # 旧的措辞本身也必须消失——它比「没有许可」更强，直接下令只调一个。
        assert "one at a time" not in prompt, (
            f"parallel_allowed={setting!r}: 契约仍在下令「一次只调一个」"
        )
    print("test_contract_permits_parallel_calls_when_the_client_allows: PASSED")


def test_contract_forbids_parallel_calls_when_the_client_disallows():
    """客户端**明确**禁止并行时，契约必须说出来（防止上一条被过度修复）。

    这条在修复前后都必须是绿的：修复只允许把禁止句收拢到这一个条件分支里，
    不允许把它一起删掉。
    """
    prompt = format_tools_system_prompt(TOOLS, parallel_allowed=False)
    assert FORBID in prompt, (
        "客户端声明 parallel_tool_calls=false，契约却没有禁止多个调用；"
        "解析端会把第二个裁掉，而模型从没被告知过"
    )
    assert ALLOW not in prompt, (
        "客户端禁止并行，契约却仍在鼓励把多个块写在一起，与紧邻的禁止句自相矛盾"
    )
    print("test_contract_forbids_parallel_calls_when_the_client_disallows: PASSED")


# ------------------------------------------------- the property, not the cells

def test_contract_and_parser_agree_over_every_setting():
    """契约与解析端在每一种并行设置上一致——按性质断言，而不是逐格枚举。"""
    # 每个设置既通过「协议调用点会怎么派生 parallel_allowed」表达，也覆盖
    # tool_choice 携带 disable_parallel_tool_use 的那条路径。
    settings = [
        ({"parallel_tool_calls": None, "tool_choice": None}, None),
        ({"parallel_tool_calls": True, "tool_choice": None}, True),
        ({"parallel_tool_calls": False, "tool_choice": None}, False),
        ({"parallel_tool_calls": None,
          "tool_choice": {"type": "auto", "disable_parallel_tool_use": True}}, False),
        ({"parallel_tool_calls": None,
          "tool_choice": {"type": "auto", "disable_parallel_tool_use": False}}, None),
    ]
    problems = []
    for kwargs, parallel_allowed in settings:
        disallowed = parallel_calls_disallowed(
            kwargs["parallel_tool_calls"], kwargs["tool_choice"])
        kept = len(cap_parallel_tool_calls(TWO_CALLS, disallowed))
        prompt = format_tools_system_prompt(TOOLS, parallel_allowed=parallel_allowed)
        if kept > 1:
            if FORBID in prompt:
                problems.append(
                    f"{kwargs}: 解析端保留 {kept} 个调用，契约却禁止多个")
            if ALLOW not in prompt:
                problems.append(
                    f"{kwargs}: 解析端保留 {kept} 个调用，契约没有说明可以并行")
        else:
            if FORBID not in prompt:
                problems.append(
                    f"{kwargs}: 解析端会裁到 {kept} 个调用，契约却没有说")
            if ALLOW in prompt:
                problems.append(
                    f"{kwargs}: 解析端会裁到 {kept} 个调用，契约却在鼓励并行")
    assert not problems, "并行调用的契约与解析端不一致:\n  " + "\n  ".join(problems)
    print("test_contract_and_parser_agree_over_every_setting: PASSED")


# ------------------------------------------------ end to end: what upstream sees

class PromptProvider:
    """记录上游真正收到的 messages，然后产出一句普通回答。"""

    last_prompt = ""

    async def chat(self, messages, model, **kw):
        PromptProvider.last_prompt = json.dumps(messages, ensure_ascii=False)
        yield "ok", {}


_GUARD_HITS: list[str] = []


def _guarded_factory(key):
    """绊线：没有先安装脚本化 provider 的请求在这里立刻失败。"""
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


def _openai_prompt(body_extra: dict) -> str:
    body = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "read both files"}],
        "tools": TOOLS,
    }
    body.update(body_extra)

    def go():
        r = _CLIENT.post("/v1/chat/completions", json=body)
        assert r.status_code == 200, f"{r.status_code}: {r.text[:400]}"
        return PromptProvider.last_prompt

    return _run(go)


def test_openai_path_forwards_the_clients_parallel_declaration():
    """`/v1/chat/completions` 必须把客户端的并行声明真正送进上游提示词。"""
    default = _openai_prompt({})
    assert FORBID not in default and ALLOW in default, (
        "默认请求（parallel_tool_calls 缺省 = 允许并行）的契约仍然禁止并行。"
        f"实测提示词里的规则行:\n"
        + "\n".join(l for l in default.splitlines() if l.strip().startswith(("2.", "4.")))
    )
    forbidden = _openai_prompt({"parallel_tool_calls": False})
    assert FORBID in forbidden, (
        "parallel_tool_calls=false 没有进入上游提示词；解析端会裁掉第二个调用，"
        "模型却从没被告知"
    )
    print("test_openai_path_forwards_the_clients_parallel_declaration: PASSED")


def run_all():
    tests = [
        test_contract_permits_parallel_calls_when_the_client_allows,
        test_contract_forbids_parallel_calls_when_the_client_disallows,
        test_contract_and_parser_agree_over_every_setting,
        test_openai_path_forwards_the_clients_parallel_declaration,
    ]
    for fn in tests:
        fn()
    # 故障转移会把绊线的断言包成 502/503，所以「没碰上游」的直接证据是计数器
    assert not _GUARD_HITS, (
        f"有 {len(_GUARD_HITS)} 次上游调用未经脚本化 provider: {sorted(set(_GUARD_HITS))}"
    )
    print(f"\n✅ All parallel contract tests passed! ({len(tests)} cases)")


if __name__ == "__main__":
    run_all()
