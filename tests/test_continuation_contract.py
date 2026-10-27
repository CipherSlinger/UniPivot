"""续聊模式下，每轮的常驻契约必须继续送达 (SESSION-CONTINUITY / TOOL-LOOP)

## 缺陷：第二轮起，系统提示与工具契约从线上消失

五个 provider 的 `chat()` 都是这一行：

    prompt = continuation_content(messages) if conversation_id else flatten_messages(messages)

而 `continuation_content` 只取「最后一条 user 之后」的增量——**系统提示整段被丢掉**。
首轮它还在（`messages[0]` 会被展平），次轮起就不在了：

    首轮  prompt = "System: You are Claude Code. NEVER delete files.\\n\\n
                    ## Tools Available\\n...\\n\\nUser: q1"    (1069 字符)
    次轮  prompt = "q2"                                        (2 字符)

实测（真实 HTTP 路径，脚本化 provider，`TESTING=1`，零上游调用）：绑定会话的第二轮，
上游收到的提示词里 `## Tools Available` / `Bash` / `System:` / `STOP` 全部为 False。

这不是「省 token 的优化」，它有三条各自的后果，全部无报错：

  * **工具契约丢失** —— 第二轮起模型看不到自己有哪些工具，只能凭首轮记忆；
    首轮之后才声明的工具（MCP 动态挂载、子 agent 的工具集）**从未**出现过。
  * **系统提示变更丢失** —— Claude Code 每轮都发一份新的 system 块（实测 8662 字符
    与 8594+119 字符两轮不同），里面是环境、规则、文件状态。这些是*当轮*指令，
    上游会话里只有首轮那一份，后续更新永远到不了模型。
  * **当轮约束丢失** —— `tool_choice:"none"`、`response_format` 都是并被注入到
    系统块的，第二轮起一并消失：客户端刚说「这轮别调工具」，模型根本没被告知。

对 agent 客户端这是最要命的形态：首轮还能用，从第二轮起「模型突然不调工具了」，
而 HTTP 200、`finish_reason:"stop"`、内容看起来完全正常。

## 正确语义

续聊只应**省掉历史轮次**（上游会话已经存着），不应省掉**常驻契约**：系统提示、
工具声明、当轮约束都是"现在这一刻生效的指令"，不属于历史。上游没有保存它们之后
的任何更新，网关也无从得知上游存的是哪一版——所以每轮如实送达当前版本。

代价是每轮重发一次系统块；这仍然是纯收益：agent 客户端本来每轮就重发全量历史，
而丢掉的契约没有任何补偿机制。

运行: .venv/bin/python tests/test_continuation_contract.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["TESTING"] = "1"
os.environ["GATEWAY_DISABLE_AUTO_REFRESH"] = "1"
os.environ["GATEWAY_DISABLE_CREDENTIAL_WATCH"] = "1"
os.environ["GATEWAY_DISABLE_PROBING"] = "1"
os.environ["GATEWAY_DISABLE_PACING"] = "1"
os.environ["GATEWAY_DISABLE_RISK_LEDGER"] = "1"
os.environ["GATEWAY_DISABLE_SINGLEFLIGHT"] = "1"
os.environ["QWEN_AUTH_TOKEN"] = "fake_qwen"
os.environ["DEEPSEEK_AUTH_TOKEN"] = "fake_ds"
os.environ["DOUBAO_AUTH_TOKEN"] = "fake_doubao"
os.environ["KIMI_AUTH_TOKEN"] = "fake_kimi"
os.environ["GLM_AUTH_TOKEN"] = "fake_glm"

import httpx  # noqa: E402

import server as srv  # noqa: E402
from providers.base import continuation_content  # noqa: E402

_DEFAULT_FACTORY = srv._make_provider
SEEN: list[dict] = []

SYSTEM = "You are Claude Code. NEVER delete files."
BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "Bash",
        "description": "run a shell command",
        "parameters": {
            "type": "object",
            "properties": {"cmd": {"type": "string"}},
            "required": ["cmd"],
        },
    },
}
WRITE_TOOL = {
    "type": "function",
    "function": {
        "name": "Write",
        "description": "write a file",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
}


class _RecordingProvider:
    """记录上游**真正收到**的提示词（与 provider 内部逐字相同的算法）。"""

    def __init__(self, key: str):
        self.key = key

    async def chat(self, messages, model, **kw):
        from providers.base import flatten_messages

        cid = kw.get("conversation_id")
        prompt = continuation_content(messages) if cid else flatten_messages(messages)
        SEEN.append({"cid": cid, "prompt": prompt})
        yield "ok", {}
        yield "", {"conversation_id": "c-1"}

    async def delete_conversation(self, *a, **kw):
        return None


def _install():
    SEEN.clear()
    srv._make_provider = lambda key, *a, **kw: _RecordingProvider(key)


def _restore():
    srv._make_provider = _DEFAULT_FACTORY
    srv.session_affinity_manager.clear()


def _post(payload: dict) -> httpx.Response:
    async def run():
        transport = httpx.ASGITransport(app=srv.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            return await c.post("/v1/chat/completions", json=payload)

    return asyncio.run(run())


# ---------------------------------------------------------------- 1. 提示词层


def test_continuation_keeps_the_system_contract():
    """**靶点**：续聊增量必须带上当轮的系统提示，而不是只剩最后一句提问。"""
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
    ]
    prompt = continuation_content(msgs)

    assert SYSTEM in prompt, (
        f"续聊增量里没有系统提示，模型从第二轮起就读不到常驻指令了。实测: {prompt!r}"
    )
    assert "q2" in prompt, f"本轮的增量不能丢: {prompt!r}"
    assert "q1" not in prompt, (
        f"续聊不该重发历史轮次（那是它存在的意义），实测把上一轮也发了: {prompt!r}"
    )


def test_continuation_keeps_a_mid_conversation_system_update():
    """会话中途下达的新规则（Claude Code 的动态 system-reminder）必须送达。"""
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "system", "content": "IMPORTANT NEW RULE: never delete files."},
        {"role": "user", "content": "q2"},
    ]
    prompt = continuation_content(msgs)
    assert "IMPORTANT NEW RULE" in prompt, (
        f"中途下达的系统规则没有送达——它只存在于当轮，上游会话里没有。实测: {prompt!r}"
    )


def test_continuation_without_system_is_unchanged():
    """零回归：没有系统消息时，续聊内容与旧行为逐值相同。"""
    msgs = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！"},
        {"role": "user", "content": "今天天气如何"},
    ]
    assert continuation_content(msgs) == "今天天气如何"


def test_continuation_keeps_the_tool_result_contract():
    """零回归：agent 工具循环的续聊仍要送达工具结果（第 32 节的承重断言）。"""
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "read foo.py"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "c1",
                "type": "function",
                "function": {"name": "Bash", "arguments": '{"cmd": "cat foo.py"}'},
            }],
        },
        {"role": "tool", "tool_call_id": "c1", "name": "Bash", "content": "print(1)"},
    ]
    prompt = continuation_content(msgs)
    assert "print(1)" in prompt, f"工具结果必须送达: {prompt!r}"
    assert "Bash" in prompt, f"工具调用上下文必须送达: {prompt!r}"
    assert SYSTEM in prompt, f"系统契约必须送达: {prompt!r}"


# ---------------------------------------------------------------- 2. 线上路径


def test_bound_session_turn_two_carries_system_and_tools():
    """**靶点（端到端）**：绑定会话的第二轮，上游收到的提示词里契约仍在。

    两轮声明**同一套**工具：这既是真实客户端的形状（Claude Code 两轮都是 47 个，
    Codex 每个会话稳定），也是「续聊」这条路径唯一能被验证的形状——声明的工具集
    一变，请求就属于另一条对话线（[SESSION-IDENTITY]，见
    `tests/test_session_lines.py`），第二轮会走新建会话 + 展平，那时契约当然还在，
    但证明的不是续聊会补发契约，而是展平本来就带着它。这条用例此前用「首轮无工具、
    次轮有工具」来构造，等于在断言一个已经不成立的续聊（见下一条用例）。
    """
    _install()
    try:
        first = {"model": "qwen", "conversation_id": "sess-1", "stream": False,
                 "tools": [BASH_TOOL],
                 "messages": [{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": "q1"}]}
        r1 = _post(first)
        assert r1.status_code == 200, r1.text
        assert srv.session_affinity_manager.has_upstream_session(
            "sess-1", srv.tool_contract_key([BASH_TOOL]))
        SEEN.clear()

        r2 = _post({"model": "qwen", "conversation_id": "sess-1", "stream": False,
                    "tools": [BASH_TOOL],
                    "messages": [{"role": "system", "content": SYSTEM},
                                 {"role": "user", "content": "q1"},
                                 {"role": "assistant", "content": "a1"},
                                 {"role": "user", "content": "q2"}]})
        assert r2.status_code == 200, r2.text
    finally:
        _restore()

    assert SEEN, "第二轮没有打到上游"
    turn2 = SEEN[-1]
    assert turn2["cid"] is not None, "本轮应当走续聊分支（已绑定上游会话）"
    prompt = turn2["prompt"]
    assert SYSTEM in prompt, (
        f"绑定会话的第二轮丢了系统提示。上游实测只收到 {len(prompt)} 字符: {prompt!r}"
    )
    assert "## Tools Available" in prompt, (
        f"绑定会话的第二轮丢了工具契约——模型不知道自己有工具，只能凭首轮记忆。"
        f"实测: {prompt!r}"
    )
    assert "Bash" in prompt, f"工具名必须出现在提示词里: {prompt!r}"


def test_a_mid_session_tool_change_starts_a_new_line_but_still_ships_the_contract():
    """[SESSION-IDENTITY] 「首轮之后才声明的工具」现在的正确行为是**新开一条对话线**，
    并把全量历史 + 当前工具契约整份送上去。

    这不是把第 76 节的承诺削弱了，而是把它落到正确的层：那条承诺是「模型必须知道
    新工具存在」，它由展平满足（下面三条断言），而不是必须由「续聊同一段上游会话」
    来满足。上游会话 id 是节点私有的续聊凭据，声明的工具契约变了就意味着模型看到的
    契约变了，此时继续那段上游会话反而是在旧契约之上作答。代价是工具集变化的那一轮
    重建一次上下文（此后同契约的轮次照常续聊），换来的是 Claude Code 自动模式分类器
    （`tools: []`，与父会话共用同一个 session id）永远不会被并进用户的上游对话。
    """
    _install()
    try:
        _post({"model": "qwen", "conversation_id": "sess-4", "stream": False,
               "messages": [{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": "q1"}]})
        assert srv.session_affinity_manager.has_upstream_session("sess-4")
        SEEN.clear()
        r2 = _post({"model": "qwen", "conversation_id": "sess-4", "stream": False,
                    "tools": [BASH_TOOL],
                    "messages": [{"role": "system", "content": SYSTEM},
                                 {"role": "user", "content": "q1"},
                                 {"role": "assistant", "content": "a1"},
                                 {"role": "user", "content": "q2"}]})
        assert r2.status_code == 200, r2.text
    finally:
        _restore()

    turn2 = SEEN[-1]
    assert turn2["cid"] is None, (
        f"工具契约变了却仍续用旧的上游会话 {turn2['cid']!r}：模型将在旧契约之上作答"
    )
    prompt = turn2["prompt"]
    assert "## Tools Available" in prompt and "Bash" in prompt, (
        f"新线没有把工具契约送上去: {prompt[:400]!r}"
    )
    assert "q1" in prompt, (
        f"新线必须展平全量历史（否则模型丢了前文）: {prompt[:400]!r}"
    )


def test_tool_declared_only_after_turn_one_reaches_upstream():
    """首轮之后才声明的工具，此前**从未**出现在上游提示词里。"""
    _install()
    try:
        _post({"model": "qwen", "conversation_id": "sess-2", "stream": False,
               "tools": [BASH_TOOL],
               "messages": [{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": "q1"}]})
        SEEN.clear()
        r2 = _post({"model": "qwen", "conversation_id": "sess-2", "stream": False,
                    "tools": [BASH_TOOL, WRITE_TOOL],
                    "messages": [{"role": "system", "content": SYSTEM},
                                 {"role": "user", "content": "q1"},
                                 {"role": "assistant", "content": "a1"},
                                 {"role": "user", "content": "q2"}]})
        assert r2.status_code == 200, r2.text
    finally:
        _restore()

    prompt = SEEN[-1]["prompt"]
    assert "Write" in prompt, (
        f"第二轮新声明的工具 [Write] 没有送达上游——首轮没声明过，所以它从未"
        f"在会话里出现过，模型不可能知道它存在。实测: {prompt!r}"
    )
    # [SESSION-IDENTITY] 工具集变了就是另一条对话线：此刻展平全量历史是必要的，
    # 不是巧合。断言机制本身，避免这条用例因为别的原因碰巧为绿。
    assert SEEN[-1]["cid"] is None, (
        f"工具集变化后仍续用旧上游会话 {SEEN[-1]['cid']!r}——契约变了，旧会话是在"
        "另一份工具契约之下建立的"
    )


def test_per_turn_tool_choice_reaches_upstream_on_a_bound_session():
    """`tool_choice:"none"` 是当轮约束，绑定会话的第二轮里必须仍然可见。"""
    _install()
    try:
        _post({"model": "qwen", "conversation_id": "sess-3", "stream": False,
               "tools": [BASH_TOOL],
               "messages": [{"role": "user", "content": "q1"}]})
        SEEN.clear()
        r2 = _post({"model": "qwen", "conversation_id": "sess-3", "stream": False,
                    "tools": [BASH_TOOL],
                    "tool_choice": "none",
                    "messages": [{"role": "user", "content": "q1"},
                                 {"role": "assistant", "content": "a1"},
                                 {"role": "user", "content": "q2"}]})
        assert r2.status_code == 200, r2.text
    finally:
        _restore()

    prompt = SEEN[-1]["prompt"].lower()
    assert "do not call any tool" in prompt, (
        f"客户端本轮声明了 tool_choice:none，但上游提示词里没有任何对应指令——"
        f"模型会照常调用工具，而那是一个客户端明确表示不会执行的调用。实测: {prompt[:300]!r}"
    )


def run_all():
    test_continuation_keeps_the_system_contract()
    test_continuation_keeps_a_mid_conversation_system_update()
    test_continuation_without_system_is_unchanged()
    test_continuation_keeps_the_tool_result_contract()
    test_bound_session_turn_two_carries_system_and_tools()
    test_a_mid_session_tool_change_starts_a_new_line_but_still_ships_the_contract()
    test_tool_declared_only_after_turn_one_reaches_upstream()
    test_per_turn_tool_choice_reaches_upstream_on_a_bound_session()
    print("\n✅ All continuation-contract tests passed! (8 cases)")


if __name__ == "__main__":
    run_all()