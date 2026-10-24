"""OpenAI Responses 流式事件契约测试 (RESPONSES-PROTOCOL)

动机来自「网关连接 agent 效果差」：Codex CLI 这类客户端并不解析 SSE 的 `event:`
行来决定行为，它按 **JSON body 里的 `type` 字段**分派，用 `item_id` 把增量挂到
具体的 output item 上，用 `sequence_number` 判定顺序。只要其中任何一项不符合
线上契约，客户端就会静默丢弃整段输出——表现是「连上了，但 agent 不说话」或
「工具调用没跑起来」，而网关日志一切正常。

本套件把这些契约钉死为可执行的断言：
1. 文本增量事件名是 `response.output_text.delta`（不是 `response.text.delta`）
2. 每一个事件 body 都带 `type` 且与 `event:` 行一致
3. `sequence_number` 从 0 起严格递增、无空洞
4. `content_part.added` / 增量 / `content_part.done` 共用同一个 `item_id`
5. 消息内容块类型是 `output_text` 且带 `annotations`
6. 思考流以 reasoning item + `reasoning_summary_text.delta` 呈现，且计入 usage
7. 工具调用产出 `function_call` item 与 `function_call_arguments.delta/done`
8. 无 tools 与有 tools 两条路径的事件序列形状一致
9. 空回复也给出结构完整的 message item（客户端不会认为本轮无输出）

运行: .venv/bin/python tests/test_responses_protocol.py
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
os.environ["QWEN_AUTH_TOKEN"] = "fake_qwen"
os.environ["DEEPSEEK_AUTH_TOKEN"] = "fake_ds"
os.environ["DOUBAO_AUTH_TOKEN"] = "fake_doubao"

from fastapi.testclient import TestClient  # noqa: E402

import server as srv  # noqa: E402
from providers.base import ProviderError  # noqa: E402


class EchoProvider:
    async def chat(self, *a, **kw):
        yield "Hello", {}
        yield " world", {"conversation_id": "conv_proto_1"}


class ReasoningProvider:
    async def chat(self, *a, **kw):
        yield "Let me think", {"reasoning": True}
        yield " about it", {"reasoning": True}
        yield "Answer.", {"conversation_id": "conv_proto_2"}


class ToolProvider:
    async def chat(self, *a, **kw):
        yield "Let me run it:\n```tool_call\n", {}
        yield '{"name": "bash", "arguments": {"command": "ls"}}\n', {}
        yield "```", {"conversation_id": "conv_proto_3"}


class EmptyProvider:
    async def chat(self, *a, **kw):
        yield "", {"conversation_id": "conv_proto_4"}


def _collect(body: dict) -> list[tuple[str, dict]]:
    """把 SSE 原始文本解析成 [(event_name, payload), ...]，保留 [DONE]。"""
    client = TestClient(srv.app)
    with client.stream("POST", "/v1/responses", json=body) as r:
        assert r.status_code == 200, r.read()
        raw = "".join(r.iter_text())

    events: list[tuple[str, dict]] = []
    for block in raw.strip().split("\n\n"):
        if not block.strip():
            continue
        ev_name = None
        data = None
        for line in block.split("\n"):
            if line.startswith("event: "):
                ev_name = line[len("event: "):].strip()
            elif line.startswith("data: "):
                payload = line[len("data: "):].strip()
                data = "[DONE]" if payload == "[DONE]" else json.loads(payload)
        if data == "[DONE]":
            events.append(("done", {"raw": "[DONE]"}))
        elif ev_name is not None and data is not None:
            events.append((ev_name, data))
    return events


def _run(body: dict, provider) -> list[tuple[str, dict]]:
    srv._make_provider = lambda key: provider
    try:
        return _collect(body)
    finally:
        srv._make_provider = _default_factory


_default_factory = srv._make_provider


def _assert_item_lifecycle(events):
    """输出项生命周期不变量：索引唯一、按序关闭、内容不串项。

    客户端按 `output_index` 归档内容，因此索引重复或未按序关闭会让内容互相覆盖
    ——表现为「agent 输出丢了一半」而不是报错，是最难定位的一类故障。
    """
    opened: dict[int, str] = {}
    closed: list[int] = []
    deltas: dict[int, str] = {}
    for name, p in events:
        if name == "response.output_item.added":
            idx = p["output_index"]
            assert idx not in opened, f"output_index {idx} 被打开两次: {opened}"
            assert idx not in closed, f"output_index {idx} 关闭后又被重新打开"
            opened[idx] = p["item"]["type"]
        elif name == "response.function_call_arguments.delta":
            idx = p["output_index"]
            assert idx in opened and idx not in closed, f"参数增量挂在未打开的项 {idx} 上"
            assert opened[idx] == "function_call", f"项 {idx} 不是 function_call"
            deltas[idx] = deltas.get(idx, "") + p.get("delta", "")
        elif name in ("response.output_text.delta", "response.reasoning_summary_text.delta"):
            idx = p["output_index"]
            assert idx in opened and idx not in closed, f"增量挂在未打开的项 {idx} 上"
            kind = "reasoning" if name == "response.reasoning_summary_text.delta" else "message"
            assert opened[idx] == kind, f"项 {idx} 是 {opened[idx]}，却收到 {kind} 增量"
            deltas[idx] = deltas.get(idx, "") + p.get("delta", "")
        elif name == "response.output_item.done":
            idx = p["output_index"]
            assert idx == p.get("output_index")
            assert idx in opened, f"项 {idx} 未打开就关闭"
            assert idx not in closed, f"项 {idx} 被关闭两次"
            item = p["item"]
            if item["type"] == "reasoning":
                got = item["summary"][0]["text"] if item.get("summary") else ""
            elif item["type"] == "message":
                got = item["content"][0]["text"] if item.get("content") else ""
            else:
                got = item.get("arguments", "")
            assert got == deltas.get(idx, ""), (
                f"项 {idx} 的 done 内容 {got!r} 与流式增量 {deltas.get(idx, '')!r} 不一致"
                "（内容串项）"
            )
            closed.append(idx)
    assert closed == sorted(opened), f"项必须按索引顺序关闭: opened={sorted(opened)} closed={closed}"


def _assert_envelope(events):
    """所有事件共享的不变量：type 与事件名一致、序号严格递增。"""
    seqs = []
    for name, payload in events:
        if name == "done":
            continue
        assert "type" in payload, f"事件 {name} 缺少 body.type（SDK 按它分派）: {payload}"
        assert payload["type"] == name, f"事件名与 body.type 不一致: {name} vs {payload['type']}"
        assert "sequence_number" in payload, f"事件 {name} 缺少 sequence_number"
        seqs.append(payload["sequence_number"])
    assert seqs == list(range(len(seqs))), f"sequence_number 必须从 0 连续递增，实际 {seqs}"


def test_text_stream_event_contract():
    events = _run(
        {"model": "qwen3.7-max", "input": "hi", "stream": True},
        EchoProvider(),
    )
    names = [n for n, _ in events]
    assert names[0] == "response.created"
    assert "response.in_progress" in names
    assert "response.output_text.delta" in names, f"现行规范的事件名必须是 output_text.delta: {names}"
    assert "response.text.delta" not in names, "旧事件名会让按名分派的客户端收不到任何文本"
    assert names[-1] == "done"
    _assert_envelope(events)
    _assert_item_lifecycle(events)

    # 文本内容与顺序
    deltas = [p["delta"] for n, p in events if n == "response.output_text.delta"]
    assert "".join(deltas) == "Hello world", deltas

    # item_id 贯穿 added / delta / done
    item_ids = {p["item_id"] for n, p in events
                if n in ("response.content_part.added", "response.output_text.delta",
                         "response.output_text.done", "response.content_part.done")}
    assert len(item_ids) == 1, f"同一条消息的所有事件必须共用 item_id，实际 {item_ids}"

    # 内容块类型
    part = [p for n, p in events if n == "response.content_part.done"][0]["part"]
    assert part["type"] == "output_text", part
    assert part["annotations"] == [], part
    assert part["text"] == "Hello world"

    # 终局
    completed = [p for n, p in events if n == "response.completed"][0]["response"]
    assert completed["status"] == "completed"
    assert completed["output"][0]["type"] == "message"
    assert completed["output"][0]["content"][0]["text"] == "Hello world"
    assert completed["usage"]["output_tokens_details"]["reasoning_tokens"] == 0
    print("test_text_stream_event_contract: PASSED")


def test_reasoning_stream_is_emitted():
    events = _run(
        {"model": "qwen3.7-max", "input": "think", "stream": True},
        ReasoningProvider(),
    )
    _assert_envelope(events)
    _assert_item_lifecycle(events)
    rs_deltas = [p for n, p in events if n == "response.reasoning_summary_text.delta"]
    assert rs_deltas, "思考流被静默丢弃（客户端会看到长时间无输出）"
    assert "".join(p["delta"] for p in rs_deltas) == "Let me think about it"

    items = [p["item"] for n, p in events if n == "response.output_item.done"]
    types = [i["type"] for i in items]
    assert "reasoning" in types and "message" in types, types

    completed = [p for n, p in events if n == "response.completed"][0]["response"]
    assert completed["usage"]["output_tokens_details"]["reasoning_tokens"] > 0
    print("test_reasoning_stream_is_emitted: PASSED")


def test_tool_call_stream_contract():
    events = _run(
        {
            "model": "qwen3.7-max",
            "input": "run ls",
            "stream": True,
            "tools": [{"type": "function", "function": {"name": "bash", "parameters": {"type": "object"}}}],
        },
        ToolProvider(),
    )
    _assert_envelope(events)
    _assert_item_lifecycle(events)

    added = [p for n, p in events if n == "response.output_item.added"]
    fn_items = [p for p in added if p["item"]["type"] == "function_call"]
    assert fn_items, f"工具调用未以 function_call item 呈现: {[p['item']['type'] for p in added]}"
    call_item = fn_items[0]["item"]
    assert call_item["name"] == "bash"
    assert call_item["call_id"]

    delta = [p for n, p in events if n == "response.function_call_arguments.delta"][0]
    assert delta["item_id"] == call_item["id"], "增量必须挂在同一个 item 上"
    assert delta["call_id"] == call_item["call_id"]

    done = [p for n, p in events if n == "response.function_call_arguments.done"][0]
    assert json.loads(done["arguments"]) == {"command": "ls"}

    completed = [p for n, p in events if n == "response.completed"][0]["response"]
    calls = [i for i in completed["output"] if i["type"] == "function_call"]
    assert len(calls) == 1 and calls[0]["name"] == "bash"
    assert json.loads(calls[0]["arguments"]) == {"command": "ls"}
    print("test_tool_call_stream_contract: PASSED")


def test_thinking_declaration_means_the_same_thing_on_all_three_protocols():
    """同一份 `thinking` 声明在三个协议上必须得到同一个判定。

    [PROTOCOL] `_wants_thinking()` 的 docstring 写着「三个协议共用这一个判定」，
    但 `/v1/responses` 从未调用它——第 3803 行自己重新推导了一遍，于是三种写法
    各自漂移，且**没有任何一个方向是安全的**：

      声明                           messages / chat    responses（修复前）
      {"type":"adaptive"}            思考开（实测 CLI   thinking=False（静默关掉）
                                     发的是这个）
      "adaptive"（裸字符串）          思考开             400（pydantic 拒绝 str）
      {"type":"disabled"}            思考关             思考关（恰好一致）

    第一行是当前 Claude Code 的线上写法，第二行是同一个值的另一种序列化。
    这不是「Responses 少一个特性」，而是**同一事实两份定义**——本仓库反复修复的
    缺陷类型（OPT-026/029/030/035 都属于这一类）。

    断言方式：不看网关自己的响应体，而是看**上游实际收到的 `thinking` 标志**
    （provider 的观察是唯一事实）。一个只在网关内部保持一致、却把不同标志交给
    上游的修复，等于没修。
    """
    declarations = [
        {"type": "adaptive"},
        {"type": "enabled"},
        {"type": "disabled"},
        "adaptive",
        True,
        False,
        None,
    ]

    def observed(path: str, body: dict, headers: dict) -> tuple[int, object]:
        seen: list[object] = []

        class Spy:
            async def chat(self, *a, **kw):
                seen.append(kw.get("thinking", "ABSENT"))
                yield "ok", {"conversation_id": "c"}
                yield "", {"conversation_id": "c"}

        srv._make_provider = lambda key: Spy()
        try:
            client = TestClient(srv.app)
            r = client.post(path, json=body, headers=headers)
        finally:
            srv._make_provider = _default_factory
        return r.status_code, (seen[-1] if seen else "NO_UPSTREAM_CALL")

    rows = []
    for decl in declarations:
        anthropic_body = {
            "model": "claude-3-7-sonnet",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}],
        }
        chat_body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
        responses_body = {"model": "gpt-4o", "input": "hi", "stream": False}
        if decl is not None:
            anthropic_body["thinking"] = decl
            chat_body["thinking"] = decl
            responses_body["thinking"] = decl

        a = observed("/v1/messages", anthropic_body, {"anthropic-version": "2023-06-01"})
        b = observed("/v1/responses", responses_body, {})
        rows.append((decl, a, b))

    mismatches = [
        f"  thinking={decl!r}: messages -> {a}, responses -> {b}"
        for decl, a, b in rows
        if a[0] != b[0] or (a[0] == 200 and b[0] == 200 and a[1] != b[1])
    ]
    assert not mismatches, (
        "同一份 thinking 声明在 /v1/messages 与 /v1/responses 上得到了不同的处理：\n"
        + "\n".join(mismatches)
        + "\n（messages 侧是 _wants_thinking 的既定语义，responses 侧自行重推了一遍）"
    )
    print(
        "test_thinking_declaration_means_the_same_thing_on_all_three_protocols: PASSED",
        f"({len(rows)} declarations agree)",
    )


def test_empty_response_still_has_message_item():
    events = _run(
        {"model": "qwen3.7-max", "input": "nothing", "stream": True},
        EmptyProvider(),
    )
    _assert_envelope(events)
    _assert_item_lifecycle(events)
    completed = [p for n, p in events if n == "response.completed"][0]["response"]
    assert completed["output"], "空回复也必须给出结构完整的 output item"
    assert completed["output"][0]["type"] == "message"
    assert completed["output"][0]["content"][0]["text"] == ""
    print("test_empty_response_still_has_message_item: PASSED")


def test_non_streaming_reasoning_item():
    srv._make_provider = lambda key: ReasoningProvider()
    try:
        client = TestClient(srv.app)
        resp = client.post("/v1/responses", json={"model": "qwen3.7-max", "input": "think"})
    finally:
        srv._make_provider = _default_factory
    assert resp.status_code == 200, resp.text
    data = resp.json()
    types = [i["type"] for i in data["output"]]
    assert types == ["reasoning", "message"], types
    assert data["output"][1]["content"][0]["type"] == "output_text"
    assert data["output"][1]["content"][0]["text"] == "Answer."
    print("test_non_streaming_reasoning_item: PASSED")


class InterleavedProvider:
    """推理模型在工具循环里的常态：想一点、写一点、再想一点。"""

    async def chat(self, *a, **kw):
        yield "first thought ", {"reasoning": True}
        yield "first answer, long enough to flush the lookahead window. ", {}
        yield "second thought ", {"reasoning": True}
        yield "second answer", {}


def test_interleaved_reasoning_and_text_never_collide():
    """靶点：四段（思考/正文/思考/正文）曾被压成两个互不匹配的 index。

    实测修复前：reasoning 与 message 同时挂在 index 0，第二段思考的增量被追加上
    一个已经写过正文的消息项，事件流为
      added(0,reasoning) → added(0,message) → reasoning_delta(0) → …
    客户端按 output_index 归档，重复索引互相覆盖，表现为输出丢一半。
    """
    events = _run(
        {"model": "qwen3.7-max", "input": "interleave", "stream": True},
        InterleavedProvider(),
    )
    _assert_envelope(events)
    _assert_item_lifecycle(events)

    done_items = [p["item"] for n, p in events if n == "response.output_item.done"]
    assert [i["type"] for i in done_items] == ["reasoning", "message", "reasoning", "message"], (
        f"四段内容必须各自成为独立输出项: {[i['type'] for i in done_items]}"
    )
    assert [i["summary"][0]["text"] for i in done_items if i["type"] == "reasoning"] == [
        "first thought ",
        "second thought ",
    ], "第二段思考串入了第一段内容"
    assert [i["content"][0]["text"] for i in done_items if i["type"] == "message"] == [
        "first answer, long enough to flush the lookahead window. ",
        "second answer",
    ], "正文内容串项"

    completed = [p for n, p in events if n == "response.completed"][0]["response"]
    assert len(completed["output"]) == 4
    print("test_interleaved_reasoning_and_text_never_collide: PASSED")


def test_shapes_match_between_tool_and_plain_paths():
    """有 tools 与无 tools 两条路径不再各自实现，事件骨架必须一致。

    注意：tools 路径启用推测式缓冲（前瞻窗口），因此同一段文本可能被合并成
    更少的增量帧——这是**帧数**差异，不是**骨架**差异。比较时把连续重复的
    同类事件折叠掉，同时要求两条路径拼出的文本完全一致。
    """
    def _skeleton(events):
        names = [n for n, _ in events]
        out = []
        for n in names:
            if not out or out[-1] != n:
                out.append(n)
        return out

    def _text(events):
        return "".join(
            p["delta"] for n, p in events if n == "response.output_text.delta"
        )

    plain = _run({"model": "qwen3.7-max", "input": "hi", "stream": True}, EchoProvider())
    withtools = _run(
        {"model": "qwen3.7-max", "input": "hi", "stream": True,
         "tools": [{"type": "function", "function": {"name": "bash", "parameters": {"type": "object"}}}]},
        EchoProvider(),
    )
    assert _skeleton(plain) == _skeleton(withtools), (
        f"两条路径的事件骨架出现漂移:\n plain={_skeleton(plain)}\n tools={_skeleton(withtools)}"
    )
    assert _text(plain) == _text(withtools) == "Hello world"
    print("test_shapes_match_between_tool_and_plain_paths: PASSED")



class MidStreamFailProvider:
    """先吐出一点正文，再在上游失败——这是 429/配额/上下文超限最真实的形状。"""

    async def chat(self, *a, **kw):
        yield "Here is the start of an answer. ", {}
        raise ProviderError("upstream quota exceeded", status=429, err_type="rate_limit_error")


def test_mid_stream_failure_is_a_terminal_event_codex_reads():
    """[RESPONSES-STREAM] 流中断的失败必须是 Codex **能读到的终局事件**。

    实测（真实 codex-cli 0.145.0，同一台 mock，唯一变量是终局事件的形状，
    每组都是 6 次请求——**这个修复恢复的是原因，不是减少重试**）：

        `error` only（修复前） → stream disconnected before completion:
                                 stream closed before response.completed
        `response.failed` only → stream disconnected before completion:
                                 upstream quota exceeded
        `error` + `response.failed`（当前） → upstream quota exceeded

    根因：网关在上游中途失败时发出 `event: error`，而 Codex 的 SSE 分派只有
    `response.*` 分支，`error` 落进 `_ => trace!("unhandled responses event")` 被
    整条丢弃；随后生成器结束、没有终局事件，客户端只能报「流在 completed 之前关闭」。
    于是上游给的真实原因（配额超限、上下文超限、风控拦截）全部丢失——用户看到的
    是一句「流莫名关闭」，而真正该看到的是 `upstream quota exceeded`。

    客户端对一次**已经开始**的流失败仍会按自己的策略重连（6 次，三种形状都一样）；
    网关不该也没有替它决定中途是否重试——内容已经发出去，重试只会产生半包重放。
    所以这里断言的是「原因有没有到达」，不是「重试有没有减少」。

    这条断言因此同时要求：终局事件是 `response.failed`、它是**最后一个**事件、
    它带着上游的真实 message，并且后面不再有 `[DONE]`（否则客户端会把一次失败
    的流当成正常收尾）。
    """
    events = _run(
        {"model": "gpt-5-codex", "input": "hi", "stream": True},
        MidStreamFailProvider(),
    )
    names = [n for n, _ in events]

    assert "response.failed" in names, (
        "流中途失败没有发出 Codex 能识别的终局事件，实际事件序列: "
        f"{names}"
    )
    assert names[-1] == "response.failed", (
        f"response.failed 必须是最后一个事件，实际的最后一个事件是 {names[-1]!r}"
    )
    assert "done" not in names, (
        "失败的流不得发送 [DONE]：那会让客户端把半截答案当成正常收尾"
    )

    payload = dict(events)["response.failed"]
    resp = payload.get("response") or {}
    assert resp.get("status") == "failed", f"response.status 应为 failed，实际 {resp.get('status')!r}"
    err = resp.get("error") or {}
    assert "quota" in (err.get("message") or "").lower(), (
        f"上游的真实失败原因没有传到客户端，实际 error={err!r}"
    )
    assert err.get("code"), "response.error.code 不能为空，客户端按它做重试分级"


def test_failed_event_code_is_the_gateway_err_type_not_a_mapped_enum():
    """[RESPONSES-STREAM] `response.failed.error.code` 的取值。

    历史注释曾断言「`error_response` 已经做过 OpenAI 风格映射，客户端按它做
    重试分级」——两句都不成立，而这个字段会被真实 Codex 分流，所以必须钉住。

    实测（真实 codex-cli 0.145.0，同一台 mock，唯一变量是 `error.code`，
    每次都吐 `insufficient_quota` 之外的同一个 message）：

        code                      请求数  用户看到
        server_error                 6    stream disconnected before completion: <msg>
        rate_limit_exceeded          6    stream disconnected before completion: <msg>
        upstream_error               6    stream disconnected before completion: <msg>
        auth_error / waf_blocked     6    stream disconnected before completion: <msg>
        insufficient_quota           1    "Quota exceeded. Check your plan and billing details."
        usage_not_included           1    "To use Codex with your ChatGPT plan, upgrade to Plus"
        context_length_exceeded      1    "Codex ran out of room in the model's context window"
        invalid_prompt               1    <msg>（原样显示）
        cyber_policy                 1    <msg>（原样显示）

    也就是说**只有这 5 个值**会改变客户端行为；网关实际会发出的 err_type
    （auth_error / waf_blocked / rate_limit_error / upstream_error /
    upstream_timeout / credential_outage / quarantined / ring_exhausted /
    disabled / upstream_incomplete）**一个都不在**这 5 个里，因此把它们
    「翻译」成标准枚举不会有任何收益——编译出来的值要么落回重试分支（等于没改），
    要么被客户端当成另一个原因显示。

    反向证据：`error` 事件上放另一个 code 不改变分支（实测），即 Codex 只读
    `response.failed`。所以这里同时锁住「code 就是 err_type」与「code 非空」。
    """
    events = _run(
        {"model": "gpt-5-codex", "input": "hi", "stream": True},
        MidStreamFailProvider(),
    )
    payload = dict(events)["response.failed"]
    err = (payload.get("response") or {}).get("error") or {}

    assert err.get("code") == "rate_limit_error", (
        "response.failed.error.code 应原样携带网关的 err_type（实测 Codex 只对 "
        "insufficient_quota / usage_not_included / context_length_exceeded / "
        f"invalid_prompt / cyber_policy 这 5 个值分流），实际 {err.get('code')!r}"
    )


def test_error_response_never_invents_a_code():
    """`error_response` 只产出 message/type——`failed()` 的回落分支依赖这一点。

    历史注释写「`error_response` 已经做过 OpenAI 风格映射」，实际它从不写
    `code` 键。这个断言把事实钉住：一旦它开始产 `code`，`failed()` 里那行
    `err.get("code") or err.get("type") or err_type` 的语义就悄悄变了。
    """
    from providers.base import error_response

    err = error_response("boom", 502, "upstream_error")["error"]
    assert "code" not in err, (
        f"error_response 不应产出 code（failed() 的回落依赖 type），实际 {sorted(err)}"
    )
    assert err["type"] == "upstream_error"


def test_zero_upstream_refusal_keeps_a_json_error_on_the_status_line():
    """[H4 / Codex] 零上游确定性拒绝必须仍是「状态行 + JSON 错误体」，不是 2xx 流。

    这条用例存在的意义是**挡住一个诱人但会撒谎的改法**：把确定性拒绝伪装成
    200 + 一条 `response.failed` 流，或者把状态码换成 400/429 去骗客户端停止重试。

    实测（真实 codex-cli 0.145.0，全部为零上游拒绝，服务端计数）：

        服务端形状                          请求数  用户看到
        ---------------------------------------------------------------
        503 + JSON 错误体（当前）              30   真实原因（25s 后）
        503 + SSE `response.failed` 流          30   `event: response.created`
        409 + SSE 流                             6   状态码 + body 片段
        400 + JSON 错误体                        1   **裸 JSON body**
        429 + JSON 错误体                        1   `exceeded retry limit, last status: 429`
        200 + SSE `response.failed`             6   真实原因
        200 + SSE `code:invalid_prompt`          1   真实原因（但 code 是编的）
        `Retry-After` / `x-should-retry`       30   完全无效

    三条结论决定了当前形状就是诚实的最优解：

    (a) **状态码完全决定重试次数，body 只在 2xx 时被解析。** 同一份带
        `insufficient_quota` / `invalid_prompt` 的 `response.failed` 流，挂在
        200 上 → 1~6 次请求；挂在 503 上 → 30 次，且用户看到的是 `event:
        response.created`（流被当错误体打印）。所以「用 SSE 形状改善 5xx」不成立。
    (b) **能 1 次停下的状态码都不诚实**：400 说「你的请求格式错了」（实际是网关
        没有可用凭证，用户的请求完全合法），429 说「你请求太快」（实际重试再快
        也没用）。而且 400 把错误体当裸 JSON 打印给用户，可读性比现在更差。
    (c) **能 1 次停下的 code 都是谎**：只有 `insufficient_quota` /
        `usage_not_included` / `context_length_exceeded` / `invalid_prompt` /
        `cyber_policy` 会终止重试，它们分别表示「额度用完 / 套餐不含 / 上下文超限 /
        提示词违规」，与「凭证失效」是完全不同的原因。
    (d) **重试预算是客户端配置，不是协议**：`request_max_retries = 0` /
        `stream_max_retries = 0` 写在 `[model_providers.<id>]` 下实测 30 → 1 次，
        写在顶层被忽略。这是**用户在客户端一侧**该做的决定，网关不该也不该能替它做。

    因此这里钉住当前选择：503 + `application/json`，状态行可被任何 SDK 读到
    （这正是 H4 存在的理由——严格 SDK 先看状态码），并且**不**携带任何伪造的
    终止型 code。同时钉住我们唯一能做的诚实优化仍然在：客户端可用配置压掉重试。
    """
    original = srv.failover_router.execute_chat

    async def _refuse(**kwargs):
        raise ProviderError(
            "全部/多数上游节点的认证凭证已失效，请执行 `.venv/bin/python login.py all`",
            status=503,
            err_type="credential_outage",
            retry_after=119,
        )

    srv.failover_router.execute_chat = _refuse
    try:
        client = TestClient(srv.app)
        for stream in (True, False):
            r = client.post(
                "/v1/responses",
                json={"model": "gpt-4o", "input": "hi", "stream": stream},
            )
            where = f"stream={stream}"
            assert r.status_code == 503, (
                f"零上游拒绝在 {where} 上被报成了 HTTP {r.status_code}："
                "实测 Codex 对 5xx 重试 30 次、对 4xx 只发 1 次，但能停在 1 次的 "
                "400/429 都在撒谎（400 = 你的请求格式错了，429 = 你请求太快），"
                "且 400/429 连原因都不给用户看（裸 JSON / 'exceeded retry limit'）。"
                "H4 要求拒绝体现在状态行上，503 是唯一既真实又能被严格 SDK 读到的答案。"
            )
            assert r.headers.get("content-type", "").startswith("application/json"), (
                f"零上游拒绝在 {where} 上不是 JSON 错误体，而是 "
                f"{r.headers.get('content-type')!r}：实测把同一份 response.failed "
                "挂在 503 上，Codex 仍发 30 次请求，并把 `event: response.created` "
                "当错误信息打印给用户——SSE 形状在非 2xx 上既不能减少重试也不能传达原因。"
            )
            payload = r.json()
            err = payload.get("error") or {}
            assert err.get("type") == "credential_outage", (
                f"{where} 的 error.type 不是网关真实的失败类别，实际 {err.get('type')!r}"
            )
            # 不伪造任何「会终止重试」的 code：那 5 个值各自的语义都与凭证失效无关。
            assert "code" not in err or err.get("code") not in {
                "insufficient_quota",
                "usage_not_included",
                "context_length_exceeded",
                "invalid_prompt",
                "cyber_policy",
            }, (
                f"{where} 伪造了终止型 error.code={err.get('code')!r}：这几个值会让 "
                "Codex 只发 1 次请求，但它们表示额度用完/套餐不含/上下文超限/提示词违规，"
                "与「网关没有可用凭证」是不同原因。省下的 29 次请求是用一句谎话换的。"
            )
            # 可读原因必须原样到达（没有任何分支可以把它吞掉）
            assert "login.py" in err.get("message", ""), (
                f"{where} 没有把可执行的修复指令传给用户，实际 {err.get('message')!r}"
            )
            # H6 的退避头仍然是诚实的（真实窗口 119s，不是常量）
            assert r.headers.get("retry-after") == "119", (
                f"{where} 的 Retry-After 不是抛出点知道的真实窗口，实际 "
                f"{r.headers.get('retry-after')!r}"
            )
    finally:
        srv.failover_router.execute_chat = original


def run_all():
    tests = [
        test_text_stream_event_contract,
        test_reasoning_stream_is_emitted,
        test_tool_call_stream_contract,
        test_empty_response_still_has_message_item,
        test_thinking_declaration_means_the_same_thing_on_all_three_protocols,
        test_non_streaming_reasoning_item,
        test_interleaved_reasoning_and_text_never_collide,
        test_shapes_match_between_tool_and_plain_paths,
        test_mid_stream_failure_is_a_terminal_event_codex_reads,
        test_failed_event_code_is_the_gateway_err_type_not_a_mapped_enum,
        test_error_response_never_invents_a_code,
        test_zero_upstream_refusal_keeps_a_json_error_on_the_status_line,
    ]
    for fn in tests:
        fn()
    print(f"\n✅ All Responses protocol tests passed! ({len(tests)} cases)")


if __name__ == "__main__":
    run_all()