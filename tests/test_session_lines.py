"""[SESSION-IDENTITY] A session id names a session, not a dialogue line.

Claude Code 2.1.293's auto mode sends an extra `/v1/messages` request for every
tool-approval decision: non-streaming, `tools: []`, `max_tokens: 2112`,
`stop_sequences: ["</block>"]`, its own 139 702-character "security monitor"
system prompt demanding a `<block>yes|no</block>` verdict, and a single user
message wrapping a `<transcript>` — carrying **the same**
`metadata.user_id.session_id` as the parent conversation (measured on three
bodies captured from the real binary: parent turn 1, classifier, parent turn 2;
system sha 0270404a12 / 2d052caeb7, 47 tools vs 0).

The affinity registry keyed only on that session id, so the gateway handed the
parent's live upstream conversation to the classifier and then continued the
same conversation on the parent's next turn. Measured through this gateway with
a scripted provider (wire prompt built by the real
`flatten_messages` / `continuation_content`, never a re-implementation):

    parent#1   affinity=fresh  wire conv=None  44 763 chars
    classifier affinity=bound  wire conv='up1' 140 302 chars  <-- appended to the parent
    parent#2   affinity=bound  wire conv='up1'  44 543 chars  <-- continues the polluted line

Two harms, neither of them an error: the user's upstream conversation permanently
gains a 140 KB security-monitor turn (the answer quality of every later turn is
measured against it), and the classifier is served as a *continuation* of a
conversation whose history has nothing to do with the transcript it is grading.

A client's session id cannot be narrowed (there is no finer identifier on the
wire), so the fix is the conservative direction the dedup key already uses
(§53/§75): **one more fact in the identity of "which conversation is this"** —
the tool contract the request declares. A conversation opened under 47 declared
tools is not continued by a request declaring none; the two lines are separate
entries, each reusing its own upstream conversation. Adding a discriminator can
only split conversations, never merge two that should have stayed apart.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["TESTING"] = "1"
os.environ["GATEWAY_DISABLE_CREDENTIAL_WATCH"] = "1"
os.environ["GATEWAY_DISABLE_AUTO_REFRESH"] = "1"
os.environ["GATEWAY_DISABLE_PROMPT_FOLDING"] = "1"
for _pk in ("QWEN", "DEEPSEEK", "DOUBAO", "KIMI", "GLM"):
    os.environ[f"{_pk}_AUTH_TOKEN"] = f"fake_{_pk.lower()}"

from fastapi.testclient import TestClient  # noqa: E402

import server as srv  # noqa: E402
from providers.base import continuation_content, flatten_messages  # noqa: E402

# ── the three measured requests, reproduced by shape ────────────────────────────
# Only the *shape* is needed (tools 0 vs 47, same session id, one user message);
# the real capture's system prompts are 139 702 / 5 757 characters and are not
# reproduced here. The numbers quoted in the docstring are the measured ones.
SESSION = "273cc882-a36c-40ad-a766-ac533978d4ad"
META = {
    "user_id": {
        "device_id": "0be3804e9d2c4c9d8a0f",
        "account_uuid": "",
        "session_id": SESSION,
    }
}
PARENT_SYSTEM = (
    "You are Claude Code, Anthropic's official CLI for Claude.\n\n"
    "## Tools Available\n... (47 declarations; this case needs three)\n"
)
# The classifier's own document, structurally what the real one is: a rule list
# ending in a strict output contract. CLASSIFIER_MARK is what must never appear
# in the parent's prompt.
CLASSIFIER_MARK = "CLASSIFIER-SYSTEM: your ENTIRE response MUST begin with <block>"
CLASSIFIER_SYSTEM = (
    "You are a security monitor for auto mode.\n"
    + CLASSIFIER_MARK
    + "\nBlock if ANY rule could apply. Do NOT output any analysis before <block>."
)


def _tool(name: str) -> dict:
    return {"name": name, "description": f"{name} tool", "input_schema": {"type": "object"}}


PARENT_TOOLS = [_tool("Bash"), _tool("Read"), _tool("Write")]


def _parent_body(turn: int) -> dict:
    if turn == 1:
        messages = [{"role": "user", "content": "Use the Agent tool to reply PONG, then tell me."}]
    else:
        messages = [
            {"role": "user", "content": "Use the Agent tool to reply PONG, then tell me."},
            {"role": "assistant", "content": [
                {"type": "text", "text": "Spawning a sub-agent."},
                {"type": "tool_use", "id": "toolu_01", "name": "Agent",
                 "input": {"prompt": "reply PONG"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_01", "content": "PONG"},
            ]},
        ]
    return {
        "model": "claude-3-7-sonnet",
        "max_tokens": 32000,
        "stream": True,
        "system": PARENT_SYSTEM,
        "metadata": META,
        "tools": PARENT_TOOLS,
        "messages": messages,
    }


def _classifier_body() -> dict:
    return {
        "model": "claude-3-7-sonnet",
        "max_tokens": 2112,
        "stream": False,
        "system": CLASSIFIER_SYSTEM,
        "metadata": META,
        "tools": [],
        "stop_sequences": ["</block>"],
        "messages": [{"role": "user", "content":
                      "<transcript>\nBash: rm -rf /tmp/x\n</transcript>\nShould this run?"}],
    }


# ── scripted provider: records what the provider really puts on the wire ────────
_CALLS: list[dict] = []
_GUARD_HITS: list[str] = []
_COUNTER = {"n": 0}


class _Recorder:
    async def chat(self, messages, model, **kw):
        cid = kw.get("conversation_id")
        prompt = continuation_content(messages) if cid else flatten_messages(messages)
        # A real upstream mints a *distinct* id per new conversation (§66: a stub
        # returning one constant makes "did the child inherit the parent's
        # conversation" untestable).
        if cid:
            served = cid
        else:
            _COUNTER["n"] += 1
            served = f"up{_COUNTER['n']}"
        _CALLS.append({"conv": cid, "served": served, "model": model, "prompt": prompt})
        yield "PONG", {}
        yield "", {"conversation_id": served}


def _guarded_factory(key):
    """Tripwire (§35): a case that forgot the scripted provider must not reach an
    upstream account. Failover wraps this into a 502; the zero-hit counter is the
    direct evidence that nothing was dispatched."""
    _GUARD_HITS.append(key)
    raise AssertionError(f"测试未安装假 provider，真实上游节点 [{key}] 被构造了")


srv._make_provider = _guarded_factory


def _client() -> TestClient:
    return TestClient(srv.app)


def _reset() -> None:
    srv.session_affinity_manager.clear()
    _CALLS.clear()
    _GUARD_HITS.clear()
    _COUNTER["n"] = 0
    srv._make_provider = lambda *a, **kw: _Recorder()


def _post(body: dict):
    r = _client().post("/v1/messages", json=body)
    assert r.status_code == 200, f"网关拒绝了真实形状的 Anthropic 请求: {r.status_code} {r.text[:300]}"
    return r, _CALLS[-1]


# ── cases ───────────────────────────────────────────────────────────────────────
def test_the_auto_mode_classifier_does_not_join_the_parent_conversation():
    """The measured defect: the classifier was handed the parent's conversation."""
    _reset()
    _, parent1 = _post(_parent_body(1))
    assert parent1["conv"] is None, parent1["conv"]

    _, classifier = _post(_classifier_body())
    assert classifier["conv"] is None, (
        "自动模式分类器请求被并进了父会话的上游对话：它拿到了 "
        f"conversation_id={classifier['conv']!r}（实测修复前为 'up1'）。两个后果："
        "用户的上游对话被永久写入一段 14 万字符的安全审查提示，而且分类器被当成"
        "「续聊」——它要评判的 transcript 与那条对话的历史毫无关系。"
    )


def test_the_parent_and_the_classifier_occupy_two_upstream_conversations():
    """Three requests, two lines: the parent keeps its own conversation and the
    classifier keeps its own. Pre-fix there was **one** — the parent's — so its
    next turn continued a conversation that now held a 140 KB security-monitor turn
    (`conv sequence: [None, 'up1', 'up1']`)."""
    _reset()
    _, parent1 = _post(_parent_body(1))
    _, classifier = _post(_classifier_body())
    _, parent2 = _post(_parent_body(2))

    parent_line = _CALLS[0]["served"]
    assert parent2["conv"] == parent_line, (
        "父会话在分类器插入后没有回到自己的上游对话："
        f"期望 {parent_line!r}，实际收到 {parent2['conv']!r}"
    )
    assert classifier["served"] != parent_line, (
        f"分类器与父会话共用同一条上游对话 {parent_line!r}——父会话此后每一轮"
        "都在那段 14 万字符的安全审查文本之上作答，而分类器是在「续聊」一条与它"
        "要评判的 transcript 毫无关系的对话"
    )


def test_repeated_classifier_calls_share_one_dedicated_line():
    """The classifier's own calls must reuse *their* conversation, not open a new
    one per approval decision (that would be one orphaned upstream conversation per
    tool call — §PROBE-GC's leak class, and 'many one-shot sessions' is itself the
    automation signature the risk line exists to avoid)."""
    _reset()
    _post(_parent_body(1))
    _, first = _post(_classifier_body())
    _, second = _post(_classifier_body())
    assert first["served"] != _CALLS[0]["served"], (
        "第一次分类器请求就并进了父会话的上游对话"
    )
    assert second["conv"] == first["served"], (
        "第二次分类器请求又新开了一个上游对话：每次工具审批都会在账号里留下一个"
        f"一次性会话（期望续聊 {first['served']!r}，实际 {second['conv']!r}）"
    )


def test_a_session_with_a_stable_tool_set_still_continues():
    """Zero-regression guard (§59/§69: green on the pre-fix tree by design, and it is
    recorded as a guard rather than dressed up as RED-first). The discriminator is a
    *widening* of the line identity, so the direction it must not break is the
    ordinary one: a session whose declared tools do not change keeps one upstream
    conversation. Measured: Claude Code declares the same 47 tools on both captured
    turns — digest 035695cc210236e5 — so this is the shape of real traffic."""
    _reset()
    _, t1 = _post(_parent_body(1))
    _, t2 = _post(_parent_body(2))
    _, t3 = _post(_parent_body(2))
    assert t1["conv"] is None
    assert t2["conv"] is not None, "工具集稳定的会话没有粘住自己的上游对话"
    assert t3["conv"] == t2["conv"], (
        f"同一对话线的上游 conversation_id 发生了变化: {t2['conv']!r} -> {t3['conv']!r}"
    )
    lines = {_CALLS[0]["served"], _CALLS[1]["served"], _CALLS[2]["served"]}
    assert len(lines) == 1, (
        f"工具集稳定的三次调用开了 {len(lines)} 条上游对话 {sorted(lines)}——"
        "把身份键加宽到「几乎不合并」是比原缺陷更糟的回归（§53 的反向错误）"
    )


def test_the_tool_contract_is_passed_at_every_call_site():
    """Structural (§53/§75): widening the signature alone passes every direct-call
    unit test while the real request path keeps handing the parent's conversation to
    the classifier. Every call site must pass it."""
    src = (Path(__file__).resolve().parents[1] / "server.py").read_text(encoding="utf-8")
    lines = src.splitlines()
    for fn, want in (("resolve_affinity(", 3), ("record_turn(", 8), ("has_upstream_session(", 1)):
        sites = [i for i, ln in enumerate(lines)
                 if fn in ln and "def " not in ln and "session_affinity_manager." in ln]
        assert len(sites) == want, f"{fn} 调用点有 {len(sites)} 处，期望 {want} 处"
        # Balance the parentheses from the call site: the argument may be wrapped,
        # and a fixed-size window could pick up the *next* call's argument.
        missing = []
        for i in sites:
            blob = "\n".join(lines[i:])
            start = blob.index(fn)
            # Balance from the call's own opening paren. Starting the loop at the
            # name would let `depth == 0` fire on the very first character (the
            # name is not a paren), truncating the slice to one character — an
            # assertion that then always reports "missing" and measures nothing.
            open_pos = blob.index("(", start)
            depth, end = 0, None
            for pos in range(open_pos, len(blob)):
                ch = blob[pos]
                depth += ch == "("
                depth -= ch == ")"
                if depth == 0:
                    end = pos
                    break
            call = blob[start: (end + 1 if end is not None else len(blob))]
            if "tool_contract" not in call:
                missing.append(lines[i].strip())
        assert not missing, (
            f"{fn} 有调用点没有传 tool_contract，真实请求路径仍然只按会话 id 认线: {missing[:2]}"
        )


def test_the_line_identity_is_observable_in_the_response():
    """A mechanism nobody can see is a mechanism nobody can debug. Both lines carry
    their own `x-session-tool-contract`, so "this session suddenly has two upstream
    conversations" can be read off the wire instead of inferred from a registry dump.
    The classifier declares no tools, so it deliberately carries **no** header —
    the absent header is the fact "this line has no tool contract", not an omission."""
    _reset()
    r_parent, _ = _post(_parent_body(1))
    r_classifier, _ = _post(_classifier_body())
    parent_key = r_parent.headers.get("x-session-tool-contract")
    assert parent_key, (
        "父会话（声明了工具）没有带上 x-session-tool-contract：对话线身份不可观测"
    )
    assert r_classifier.headers.get("x-session-tool-contract") is None, (
        "未声明任何工具的请求也带了工具契约头：空契约不发送，否则「契约存在但为空」"
        "与「没有契约」在线上无法区分"
    )


def test_one_definition_of_the_tool_contract():
    """`tool_names_of` already answers "which tools did the client declare" (all
    three protocol shapes, custom/freeform included). A second extractor here would
    be the recurring "one fact, two definitions" defect."""
    sa = (Path(__file__).resolve().parents[1] / "providers" / "session_affinity.py").read_text(encoding="utf-8")
    assert "tool_names_of" in sa, (
        "工具契约摘要没有复用 tool_names_of——它自己又数了一遍工具名，"
        "于是自定义工具 / namespace 容器在两条路径上会得到不同的契约"
    )


def run_all():
    test_the_auto_mode_classifier_does_not_join_the_parent_conversation()
    test_the_parent_and_the_classifier_occupy_two_upstream_conversations()
    test_repeated_classifier_calls_share_one_dedicated_line()
    test_a_session_with_a_stable_tool_set_still_continues()
    test_the_tool_contract_is_passed_at_every_call_site()
    test_the_line_identity_is_observable_in_the_response()
    test_one_definition_of_the_tool_contract()
    assert not _GUARD_HITS, f"有请求走到了真实上游节点: {_GUARD_HITS}"
    print("\n✅ All session-line tests passed! (7 cases)")


if __name__ == "__main__":
    run_all()
