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

## [MODEL-LINE] The second discriminator: which model this line is speaking to

The same argument applies to the model, and there the failure is worse than a
split conversation — it is a *silently ignored instruction*. `resolve_affinity`'s
bound branch returns `entry.model_name` and `server.py` assigns
`dispatched_model = affinity_decision.wire_model`, so a client that names a
different model on a bound session is dispatched to the **old** one. Measured
(same session id, same tool set, Anthropic path):

    turn 1  claude-3-5-haiku-20241022  affinity=fresh  wire model Qwen3.6-Flash  conv=None
    turn 2  claude-3-7-sonnet          affinity=bound  wire model Qwen3.6-Flash  conv='up1'

`resolve_model` resolves turn 2 to `Qwen3.7-Max`; the provider received
`Qwen3.6-Flash`. No error, no telemetry naming the substitution — the client's
`/model` switch simply did not happen. The mirror image is a background
small-model request (Claude Code sends these under the *same*
`metadata.user_id.session_id`) landing on the parent's line and then being served
with the parent's model, or worse, dragging the parent's next turn onto the small
model.

The line identity therefore carries the model too — the **resolved wire model
before load balancing** (`wire_model`), not the client's raw string and not the
post-LB model. Two aliases that resolve to the same wire model produce
interchangeable answers and stay on one line; a load-balancer downgrade is
transient and must never split a conversation (§M3's rationale, one layer over).
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


# ── [MODEL-LINE] the model is part of the line identity ─────────────────────────
def _parent_body_with_model(turn: int, model: str) -> dict:
    body = _parent_body(turn)
    body["model"] = model
    return body


def test_a_mid_session_model_change_is_dispatched_to_the_requested_model():
    """**靶点**：绑定会话里换模型，网关必须把请求交给**新声明的**模型。

    修复前实测：第二轮声明 `claude-3-7-sonnet`（`resolve_model` 解析为
    `Qwen3.7-Max`），上游实际收到 `Qwen3.6-Flash`（第一轮的模型）——`/model`
    切换静默失效，没有任何错误，遥测里也只有 `x-session-affinity: bound`。
    对 agent 客户端这是「我换了模型，它还是老样子，而且变笨/变贵了」。
    """
    _reset()
    _post(_parent_body_with_model(1, "claude-3-5-haiku-20241022"))
    _, turn2 = _post(_parent_body_with_model(2, "claude-3-7-sonnet"))
    assert turn2["model"] == "Qwen3.7-Max", (
        "会话中途换模型被静默忽略了：客户端声明 claude-3-7-sonnet（→ "
        f"Qwen3.7-Max），上游实际收到 {turn2['model']!r}"
    )


def test_a_model_change_starts_a_new_line_with_the_full_history():
    """换模型的那一轮必须新开对话线并展平历史——不能把新模型塞进旧模型建立的
    上游会话（对方既不认识这段会话的模型，也无法保证上下文被正确解释）。"""
    _reset()
    _, turn1 = _post(_parent_body_with_model(1, "claude-3-7-sonnet"))
    _, turn2 = _post(_parent_body_with_model(2, "claude-3-5-haiku-20241022"))
    assert turn2["conv"] is None, (
        f"换模型后仍续用旧模型的上游会话 {turn2['conv']!r}（第一轮为 {turn1['served']!r}）"
    )
    assert "Use the Agent tool to reply PONG" in turn2["prompt"], (
        "新线没有展平全量历史，新模型看不到前一轮的提问——它只收到一份增量: "
        f"{turn2['prompt'][:300]!r}"
    )


def test_a_background_small_model_request_does_not_hijack_the_parent_line():
    """Claude Code 的后台请求（标题、配额、自动模式分类）用**同一个会话 id**
    但用小的快模型。它既不能并进父会话的上游对话，也不能把父会话下一轮拖到
    小模型上。实测（修复前）：三轮的 wire conv 全是 'up1'，父会话第三轮被
    bind 在后台请求建立的模型上。"""
    _reset()
    _, parent1 = _post(_parent_body_with_model(1, "claude-3-7-sonnet"))
    _, background = _post(_parent_body_with_model(2, "claude-3-5-haiku-20241022"))
    _, parent3 = _post(_parent_body_with_model(2, "claude-3-7-sonnet"))
    assert background["conv"] is None, (
        f"后台快模型请求并进了父会话的上游对话（{background['conv']!r}）"
    )
    assert background["model"] == "Qwen3.6-Flash", background["model"]
    assert parent3["conv"] == parent1["served"], (
        f"父会话在后台请求之后没有回到自己的上游对话: 期望 {parent1['served']!r}，"
        f"实际 {parent3['conv']!r}"
    )
    assert parent3["model"] == "Qwen3.7-Max", parent3["model"]


def test_two_aliases_resolving_to_one_wire_model_share_one_line():
    """Zero-regression guard (§59/§69: green on the pre-fix tree by design, recorded
    as a guard). The key is the **resolved** wire model, not the client's raw string:
    `claude-3-7-sonnet` and `claude-3-opus` both resolve to `Qwen3.7-Max`, their
    answers are interchangeable, and they must not split a conversation. Keying the
    raw alias would make the identity key so specific that ordinary sessions stop
    continuing — the worse regression (§53)."""
    _reset()
    _, t1 = _post(_parent_body_with_model(1, "claude-3-7-sonnet"))
    _, t2 = _post(_parent_body_with_model(2, "claude-3-opus"))
    assert t2["conv"] == t1["served"], (
        f"两个解析到同一个上游模型的别名被拆成了两条线: {t1['served']!r} -> {t2['conv']!r}"
    )
    assert t2["model"] == "Qwen3.7-Max", t2["model"]


def test_the_dispatched_model_header_names_the_model_the_provider_received():
    """`x-dispatched-model` 是「这次真正派给了谁」的唯一可观测出口，而它在粘滞
    覆盖**之前**就构造好了（`_make_lb_headers(decision)`），所以绑定会话里它
    报的是负载均衡的初选，不是真实派发目标。实测：头部 `Qwen3.7-Max`，provider
    收到 `Qwen3.6-Flash`。断言按线路上的事实（provider 实收模型）来对，而不是
    按网关自己的记账（§67/§81）。"""
    _reset()
    _post(_parent_body_with_model(1, "claude-3-5-haiku-20241022"))
    r, turn2 = _post(_parent_body_with_model(2, "claude-3-7-sonnet"))
    assert r.headers.get("x-dispatched-model") == turn2["model"], (
        "响应头声明的模型与 provider 实收的模型不一致: 头 "
        f"{r.headers.get('x-dispatched-model')!r} vs 线路 {turn2['model']!r}"
    )
    assert r.headers.get("x-session-affinity") in ("fresh", "bound"), r.headers


def test_the_tool_contract_is_passed_at_every_call_site():
    """Structural (§53/§75): widening the signature alone passes every direct-call
    unit test while the real request path keeps handing the parent's conversation to
    the classifier. Every call site must pass it — and the same for the model."""
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
            for arg in ("tool_contract", "requested_model"):
                if arg not in call:
                    missing.append(f"{lines[i].strip()}  -> 缺 {arg}")
        assert not missing, (
            f"{fn} 有调用点没有把对话线身份的组成部分传全（工具契约 / 声明的模型），"
            f"真实请求路径仍会认错线: {missing[:2]}"
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
    test_a_mid_session_model_change_is_dispatched_to_the_requested_model()
    test_a_model_change_starts_a_new_line_with_the_full_history()
    test_a_background_small_model_request_does_not_hijack_the_parent_line()
    test_two_aliases_resolving_to_one_wire_model_share_one_line()
    test_the_dispatched_model_header_names_the_model_the_provider_received()
    test_the_tool_contract_is_passed_at_every_call_site()
    test_the_line_identity_is_observable_in_the_response()
    test_one_definition_of_the_tool_contract()
    assert not _GUARD_HITS, f"有请求走到了真实上游节点: {_GUARD_HITS}"
    print("\n✅ All session-line tests passed! (12 cases)")


if __name__ == "__main__":
    run_all()
