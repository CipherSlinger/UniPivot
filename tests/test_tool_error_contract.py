r"""工具结果的每一种标记，必须被注入的工具契约解释 (TOOL-LOOP)

五家网页端上游**没有原生工具调用**：工具调用是网关把一段 ```tool_call 契约注入
提示词换来的。这份契约因此是模型唯一读得到的说明书——它必须描述**网关真正会写进
历史的东西**，否则模型看到的是一个它从未被告知过含义的标记。

网关的回放路径会写三种形态的标记：

    [Tool Result for <id>]        chat(有 id) / anthropic 成功 / responses(有 id)
    [Tool Error  for <id>]        anthropic `is_error: true`
    [Tool Result] (Bash)          chat 的 `role:"tool"` 没有 tool_call_id 时

契约第 5 条此前只写了第一种：`When you receive a \`[Tool Result for <id>]\` block,
continue the task using that result.`

`[Tool Error for <id>]` 不是「没被提到」这么轻——它是**每一次失败的 Claude Code
工具调用**的形状（非零退出码、被拒绝的 Edit、读不到的文件）。实测（真实转换链路）：

    User: [Tool Error for toolu_1]:
          ls: cannot access '/nope'
          [Tool Result for toolu_2]:
          a.py b.py

模型把一段失败当作「结果」读，而契约里唯一那句「continue the task using that
result」说的正是这个动作。§38 已经为**折叠**标记定过这条纪律：标记是模型下一步
推理的输入，把失败呈现成成功，模型就会从一个从未成立的结论继续往下走。这里是同
一个缺陷在契约层的另一面，而且更严重——折叠标记只是偶尔出现在历史里，`is_error`
出现在每一次失败的调用上。

本套件钉住的契约：
1. **性质**：回放路径能写出的每一种标记形状，契约都必须点名（不是枚举几种情形，
   而是对真实转换链路的输出做全称断言——下一种标记漏掉时它会变红）
2. 失败标记必须被**单独**说明，并且说明它意味着「调用跑过了、失败了」
3. 无 id 的 `[Tool Result]` 形态也必须被点名（契约唯一承诺的形状是 for <id>），
   并且网关自己不得写出 `[Tool Result for ]` 这种**残缺**标记——性质用例把它归一
   成合法形状，所以看不见它，这一条只能直接看输出
4. 端到端：`/v1/messages` 上游**真正收到**的提示词里，同时有失败标记和它的解释
5. 零回归：成功标记那句「继续用这个结果」原样保留

运行: .venv/bin/python tests/test_tool_error_contract.py
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["TESTING"] = "1"
os.environ["GATEWAY_DISABLE_CREDENTIAL_WATCH"] = "1"
os.environ["GATEWAY_DISABLE_AUTO_REFRESH"] = "1"
os.environ["GATEWAY_DISABLE_PROMPT_FOLDING"] = "1"
os.environ["GATEWAY_DISABLE_PACING"] = "1"
os.environ["GATEWAY_DISABLE_RISK_LEDGER"] = "1"
#: 端到端用例必须真的走到派发（它断言的是上游收到的提示词），所以摆假凭证。
for _pk in ("QWEN", "DEEPSEEK", "DOUBAO", "KIMI", "GLM"):
    os.environ[f"{_pk}_AUTH_TOKEN"] = f"test-token-{_pk.lower()}"

import httpx  # noqa: E402

import server as srv  # noqa: E402
from providers.anthropic_compat import (  # noqa: E402
    AnthropicMessage,
    convert_anthropic_to_gateway_messages,
    format_tools_system_prompt,
)
from providers.base import flatten_messages  # noqa: E402
from providers.responses_compat import (  # noqa: E402
    ResponsesRequest,
    convert_responses_to_gateway_messages,
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "Bash",
            "description": "Run a shell command.",
            "parameters": {
                "type": "object",
                "properties": {"cmd": {"type": "string"}},
                "required": ["cmd"],
            },
        },
    }
]

#: 回放路径写进提示词的标记。`[^\]\n]*` 让 `[Tool Result] (Bash)` 只取到 `]` 为止。
_MARKER_RE = re.compile(r"\[Tool (?:Result|Error)[^\]\n]*\]")


# --------------------------------------------------------------- harness

_GUARD_HITS: list[str] = []


def _guarded_factory(key, *a, **kw):
    _GUARD_HITS.append(key)
    raise AssertionError("测试发出了未安装假 provider 的上游请求")


_DEFAULT_FACTORY = srv._make_provider
srv._make_provider = _guarded_factory


def _contract(**kw) -> str:
    return format_tools_system_prompt(TOOLS, **kw)


def _stem(marker: str) -> str:
    """把一个具体标记归一成契约里应当出现的那种形状。"""
    if marker.startswith("[Tool Error for "):
        return "[Tool Error for <id>]"
    if marker.startswith("[Tool Result for "):
        return "[Tool Result for <id>]"
    if marker.startswith("[Tool Result]"):
        return "[Tool Result]"
    return marker


def _markers(text: str) -> set[str]:
    return {m.group(0) for m in _MARKER_RE.finditer(text)}


def _chat_history(*, call_id: str | None, name: str | None) -> str:
    """OpenAI Chat 协议的一条工具结果（是否带 id/名称由参数决定）。"""
    assistant = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "Bash", "arguments": '{"cmd":"ls"}'},
            }
        ],
    }
    tool = {"role": "tool", "content": "a.py"}
    if call_id:
        tool["tool_call_id"] = call_id
    if name:
        tool["name"] = name
    return flatten_messages([{"role": "user", "content": "go"}, assistant, tool])


def _anthropic_history(*, is_error: bool, call_id: str | None = "toolu_1") -> str:
    result: dict = {"type": "tool_result", "is_error": is_error,
                    "content": "ls: cannot access '/nope'"}
    if call_id:
        result["tool_use_id"] = call_id
    raw = [
        {"role": "user", "content": "list /nope"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "Bash",
                    "input": {"cmd": "ls /nope"},
                }
            ],
        },
        {"role": "user", "content": [result]},
    ]
    gw = convert_anthropic_to_gateway_messages(
        [AnthropicMessage(**m) for m in raw], tools=TOOLS
    )
    return flatten_messages(gw)


def _responses_history(*, call_id: str | None) -> str:
    call = {"type": "function_call", "name": "Bash", "arguments": '{"cmd":"ls"}'}
    if call_id:
        call["call_id"] = call_id
    out = {"type": "function_call_output", "output": "a.py"}
    if call_id:
        out["call_id"] = call_id
    req = ResponsesRequest(
        model="gpt-4o",
        input=[{"type": "message", "role": "user", "content": "go"}, call, out],
    )
    return flatten_messages(convert_responses_to_gateway_messages(req))


def _emitted_markers() -> dict[str, str]:
    """每一种标记形状（归一后）-> 它来自哪条回放路径。"""
    sources = {
        _chat_history(call_id="c1", name="Bash"): "chat / role=tool 带 id",
        _chat_history(call_id=None, name="Bash"): "chat / role=tool 无 id 有名字",
        _chat_history(call_id=None, name=None): "chat / role=tool 无 id 无名字",
        _anthropic_history(is_error=False): "anthropic / tool_result 成功",
        _anthropic_history(is_error=True): "anthropic / tool_result is_error",
        _anthropic_history(is_error=False, call_id=None): "anthropic / tool_result 无 id",
        _anthropic_history(is_error=True, call_id=None): "anthropic / tool_result 无 id 且失败",
        _responses_history(call_id="c1"): "responses / function_call_output 带 id",
        _responses_history(call_id=None): "responses / function_call_output 无 id",
    }
    out: dict[str, str] = {}
    for prompt, source in sources.items():
        for marker in _markers(prompt):
            out.setdefault(_stem(marker), source)
    return out


#: 回放路径已知会写出的三种形状（见模块 docstring）。它不是被测对象，而是**防真空**
#: 的把手：全称断言（「每一种被写出的标记都被契约点名」）在一条标记都写不出来时也是
#: 真的。故意只列出三种而**不**包含无 id 的失败形态 `[Tool Error]`——那条形态在
#: 修复前根本写不出来（网关写的是残缺的 `[Tool Error for ]`），把它列进地板会让本
#: 用例在基线树上以「真空通过」失败，从而盖掉真正要说的那句话。无 id 形态由
#: `test_an_idless_result_never_renders_a_half_built_marker` 单独钉住。
_EXPECTED_STEMS = {
    "[Tool Result for <id>]",
    "[Tool Error for <id>]",
    "[Tool Result]",
}


# ------------------------------------------------------------- the contract

def test_every_marker_the_replay_emits_is_named_by_the_contract():
    """[性质] 回放路径能写出的每一种标记形状，契约都必须点名。

    逐个写用例只能覆盖「我想得到的形状」；这里对**真实转换链路**的输出做全称断言，
    于是下一个新增的回放分支（某个客户端又发明一种结果形状）会让套件变红，而不是
    等运维去发现「模型又答非所问了」。
    """
    contract = _contract()
    emitted = _emitted_markers()
    assert _EXPECTED_STEMS <= set(emitted), (
        f"回放路径没有写出预期的标记形状，断言会真空通过: {sorted(emitted)}"
    )
    missing = sorted(s for s in emitted if s not in contract)
    assert not missing, (
        "这些标记网关会写进历史，而契约一个字都没提："
        + "、".join(f"{s}（来自 {emitted[s]}）" for s in missing)
    )
    print("test_every_marker_the_replay_emits_is_named_by_the_contract: PASSED")


def test_a_failed_tool_call_is_not_described_as_a_success():
    """失败标记必须被单独解释：调用跑过了、并且失败了。

    「继续使用这个结果」对失败结果是**有害的指令**——它让模型把一段错误信息当作
    任务的产出继续往下推理（§38 的同一条纪律：标记是模型下一步推理的输入）。
    因此断言两件事：契约里出现了失败标记本身，且解释它的那句话说的是失败。
    """
    contract = _contract()
    assert "[Tool Error for <id>]" in contract, (
        "契约没有点名 `[Tool Error for <id>]`——而这正是每一次失败的 Claude Code "
        "工具调用被回放时的形状"
    )
    # 解释句必须落在错误标记自己的那一行上，不能只靠成功标记那句顺带一句「或失败」。
    error_lines = [ln for ln in contract.splitlines() if "[Tool Error for <id>]" in ln]
    assert error_lines, "错误标记没有自己的一行说明"
    joined = " ".join(error_lines).lower()
    assert "fail" in joined, (
        f"点名了错误标记却没有说它会失败，模型只能把它当成普通结果读: {error_lines}"
    )
    assert "not" in joined or "do not" in joined, (
        f"没有告诉模型该拿失败结果怎么办（重发同一个调用只会得到同一个失败）: {error_lines}"
    )
    print("test_a_failed_tool_call_is_not_described_as_a_success: PASSED")


def test_idless_result_marker_is_named_by_the_contract():
    """`[Tool Result]`（无 id）也必须被点名。

    契约唯一承诺的形状是 `[Tool Result for <id>]`，而 `role:"tool"` 缺少
    tool_call_id 时网关写的是 `[Tool Result] (Bash)`。两者字面不同，模型没有理由
    把它们当成同一种东西。
    """
    contract = _contract()
    assert "[Tool Result]" in contract, (
        "契约没有点名无 id 的 `[Tool Result]` 形态（chat 的 role=tool 缺 "
        "tool_call_id 时网关就是这么写的）"
    )
    print("test_idless_result_marker_is_named_by_the_contract: PASSED")


def test_an_idless_result_never_renders_a_half_built_marker():
    """没有调用 id 时不得写出 `[Tool Result for ]` 这种残缺标记。

    性质用例看不见这一条：把 `[Tool Result for ]` 归一成契约形状，它确实就是
    `[Tool Result for <id>]`。而模型看到的是一个**写着 "for" 却没有下文**的标记，
    既不匹配契约承诺的形状，也不是另一条路径（chat / Responses）在同样情况下写出的
    `[Tool Result]`。同一个事实在三条回放路径上不能有第三种读法。
    """
    for is_error in (False, True):
        text = _anthropic_history(is_error=is_error, call_id=None)
        assert "for ]" not in text, (
            f"缺少 tool_use_id 时写出了残缺标记（实测 {'[Tool Error for ]' if is_error else '[Tool Result for ]'}）: {text[-120:]!r}"
        )
        expected = "[Tool Error]" if is_error else "[Tool Result]"
        assert expected in text, f"无 id 的结果没有降级成 {expected}: {text[-120:]!r}"
    print("test_an_idless_result_never_renders_a_half_built_marker: PASSED")


# ------------------------------------------------------------- end to end

class _RecordingProvider:
    """记录上游**真正收到**的提示词（与 provider 内部逐字相同的算法）。"""

    def __init__(self, key: str):
        self.key = key

    async def chat(self, messages, model, **kw):
        from providers.base import continuation_content

        cid = kw.get("conversation_id")
        prompt = continuation_content(messages) if cid else flatten_messages(messages)
        SEEN.append({"cid": cid, "prompt": prompt})
        yield "ok", {}
        yield "", {"conversation_id": "conv-1"}

    async def delete_conversation(self, *a, **kw):
        return None


SEEN: list[dict] = []


def test_end_to_end_the_upstream_prompt_explains_the_error_marker():
    """端到端：模型真正读到的提示词里，失败标记与它的解释同时在场。

    这一步不能只断言契约函数——§67(a) 的教训是「断言在网关自己的簿记上」会漏掉
    提示词根本没送到上游的情形。
    """
    SEEN.clear()
    srv._make_provider = lambda key, *a, **kw: _RecordingProvider(key)
    try:
        async def run():
            transport = httpx.ASGITransport(app=srv.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
                return await c.post(
                    "/v1/messages",
                    json={
                        "model": "claude-3-7-sonnet",
                        "max_tokens": 64,
                        "tools": TOOLS,
                        "messages": [
                            {"role": "user", "content": "list /nope"},
                            {
                                "role": "assistant",
                                "content": [
                                    {
                                        "type": "tool_use",
                                        "id": "toolu_1",
                                        "name": "Bash",
                                        "input": {"cmd": "ls /nope"},
                                    }
                                ],
                            },
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "tool_result",
                                        "tool_use_id": "toolu_1",
                                        "is_error": True,
                                        "content": "ls: cannot access '/nope'",
                                    }
                                ],
                            },
                        ],
                    },
                )

        resp = asyncio.run(run())
    finally:
        srv._make_provider = _guarded_factory

    assert resp.status_code == 200, resp.text
    assert SEEN, "上游一次都没被调用，用例没有测到提示词"
    prompt = SEEN[0]["prompt"]
    assert "[Tool Error for toolu_1]" in prompt, (
        f"回放的失败标记没有到达上游: {prompt[-300:]!r}"
    )
    assert "[Tool Error for <id>]" in prompt, (
        "上游收到的提示词里没有失败标记的契约说明——模型读到的是一段它从未被告知"
        "过含义的文本"
    )
    print("test_end_to_end_the_upstream_prompt_explains_the_error_marker: PASSED")


# ---------------------------------------------------------- zero regression

def test_success_marker_rule_is_unchanged():
    """[守卫] 成功标记那句「继续用这个结果」必须原样保留。

    本案只是在契约里**增加**失败与无 id 两种形态；成功路径的语义一个字都不能变，
    否则修复本身会改写模型对正常回程的理解。
    """
    contract = _contract()
    assert "[Tool Result for <id>]" in contract, "成功标记被漏掉了"
    success_lines = [
        ln for ln in contract.splitlines() if "[Tool Result for <id>]" in ln and "[Tool Error" not in ln
    ]
    assert success_lines, "成功标记没有自己的一行说明"
    joined = " ".join(success_lines).lower()
    assert "continue" in joined, f"成功标记的「继续」语义被改写了: {success_lines}"
    assert "fail" not in joined, (
        f"成功标记那一行被混进了失败语义，两种事实又变回一句话: {success_lines}"
    )
    print("test_success_marker_rule_is_unchanged: PASSED")


def run_all():
    tests = [
        test_every_marker_the_replay_emits_is_named_by_the_contract,
        test_a_failed_tool_call_is_not_described_as_a_success,
        test_idless_result_marker_is_named_by_the_contract,
        test_an_idless_result_never_renders_a_half_built_marker,
        test_end_to_end_the_upstream_prompt_explains_the_error_marker,
        test_success_marker_rule_is_unchanged,
    ]
    for fn in tests:
        fn()
    srv._make_provider = _DEFAULT_FACTORY
    assert not _GUARD_HITS, (
        f"有 {len(_GUARD_HITS)} 次上游调用未经脚本化 provider: {sorted(set(_GUARD_HITS))}"
    )
    print(f"\n✅ All tool error contract tests passed! ({len(tests)} cases)")


if __name__ == "__main__":
    run_all()
