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


def run_all():
    tests = [
        test_text_stream_event_contract,
        test_reasoning_stream_is_emitted,
        test_tool_call_stream_contract,
        test_empty_response_still_has_message_item,
        test_non_streaming_reasoning_item,
        test_shapes_match_between_tool_and_plain_paths,
    ]
    for fn in tests:
        fn()
    print(f"\n✅ All Responses protocol tests passed! ({len(tests)} cases)")


if __name__ == "__main__":
    run_all()