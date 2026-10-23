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


def run_all():
    tests = [
        test_text_stream_event_contract,
        test_reasoning_stream_is_emitted,
        test_tool_call_stream_contract,
        test_empty_response_still_has_message_item,
        test_non_streaming_reasoning_item,
        test_interleaved_reasoning_and_text_never_collide,
        test_shapes_match_between_tool_and_plain_paths,
        test_mid_stream_failure_is_a_terminal_event_codex_reads,
    ]
    for fn in tests:
        fn()
    print(f"\n✅ All Responses protocol tests passed! ({len(tests)} cases)")


if __name__ == "__main__":
    run_all()