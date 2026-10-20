"""Anthropic 流式事件契约测试 (ANTHROPIC-STREAM)

Claude Code 走的就是这条路径，而它的流式状态机比 Responses 更严格：每个
`content_block` 必须先 `content_block_start`、再若干 delta、最后
`content_block_stop`，**索引单调递增且不得重复开始**。事件流一旦非法，SDK
要么抛错，要么把后续内容错位到别的块上——用户看到的是「agent 输出乱掉/丢内容」。

本套件钉住的契约：
1. `block` 生命周期严格成对，索引从 0 连续递增，无重复 start
2. 思考块与正文块**互斥**：任意方向切换都必须先关旧块再开新块
   （实测修复前：思考→正文→再思考会输出 start(1,text) → start(1,thinking)，
   同一个索引被打开两次，非法）
3. 纯正文、纯思考、正文↔思考多次往返、空回复四种形态都合法
4. 工具模式与非工具模式产出同样合法的骨架
5. 工具调用产出 `tool_use` 块 + `input_json_delta`，`message_delta.stop_reason=tool_use`
6. `message_start` 必须是第一个事件，`message_stop` 必须是最后一个
7. 终局 `message_delta` 必带 `usage.output_tokens`

运行: .venv/bin/python tests/test_anthropic_stream.py
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

from fastapi.testclient import TestClient  # noqa: E402

import server as srv  # noqa: E402

TOOLS = [
    {
        "name": "Read",
        "description": "Read a file",
        "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}}},
    }
]


class ScriptedProvider:
    script: list = []

    async def chat(self, *a, **kw):
        for delta, meta in ScriptedProvider.script:
            yield delta, meta


def _events(script, *, tools=None) -> list[dict]:
    """跑一次 /v1/messages 流式请求，返回全部 data 事件（按序）。"""
    from fastapi.testclient import TestClient as _TC

    ScriptedProvider.script = script
    default_factory = srv._make_provider
    srv._make_provider = lambda key: ScriptedProvider()
    body = {
        "model": "claude-3-7-sonnet",
        "max_tokens": 256,
        "stream": True,
        "messages": [{"role": "user", "content": "hi"}],
    }
    if tools is not None:
        body["tools"] = tools
    try:
        client = _TC(srv.app)
        with client.stream("POST", "/v1/messages", json=body) as r:
            assert r.status_code == 200, r.read()
            raw = "".join(r.iter_text())
    finally:
        srv._make_provider = default_factory

    out = []
    for block in raw.split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data: "):
                try:
                    out.append(json.loads(line[len("data: "):]))
                except json.JSONDecodeError:
                    pass
    return out


def _assert_block_lifecycle(events: list[dict]) -> list[dict]:
    """校验块生命周期，返回打开的块（按开始顺序）。"""
    opened: list[dict] = []
    current = None
    for ev in events:
        t = ev.get("type")
        if t == "content_block_start":
            assert current is None, (
                f"索引 {ev['index']} 在上一个块 ({current['index']}) 关闭前就开始，事件流非法"
            )
            assert ev["index"] == len(opened), (
                f"块索引必须从 0 连续递增，期望 {len(opened)}，实际 {ev['index']}"
            )
            current = ev
            opened.append(ev)
        elif t == "content_block_delta":
            assert current is not None, f"delta 出现在任何块打开之前: {ev}"
            assert ev["index"] == current["index"], (
                f"delta 索引 {ev['index']} 不属于当前块 {current['index']}"
            )
        elif t == "content_block_stop":
            assert current is not None, f"stop 出现在任何块打开之前: {ev}"
            assert ev["index"] == current["index"], (
                f"stop 索引 {ev['index']} 不属于当前块 {current['index']}"
            )
            current = None
    assert current is None, f"块 {current and current['index']} 没有关闭，事件流被截断"
    first = events[0]
    assert first.get("type") == "message_start", f"第一个事件必须是 message_start: {first.get('type')}"
    assert events[-1].get("type") == "message_stop", f"最后一个事件必须是 message_stop: {events[-1].get('type')}"
    return opened


def test_plain_text_stream_is_valid():
    events = _events([("Hello", {}), (" world", {"conversation_id": "c1"})])
    opened = _assert_block_lifecycle(events)
    assert [b["content_block"]["type"] for b in opened] == ["text"]
    text = "".join(
        e["delta"]["text"] for e in events
        if e.get("type") == "content_block_delta" and e["delta"]["type"] == "text_delta"
    )
    assert text == "Hello world"
    print("test_plain_text_stream_is_valid: PASSED")


def test_thinking_only_stream_is_valid():
    events = _events([("thinking...", {"reasoning": True}), ("done", {"reasoning": True})])
    opened = _assert_block_lifecycle(events)
    assert [b["content_block"]["type"] for b in opened] == ["thinking"]
    assert all(
        e["delta"]["type"] == "thinking_delta"
        for e in events if e.get("type") == "content_block_delta"
    )
    print("test_thinking_only_stream_is_valid: PASSED")


def test_reasoning_text_interleaving_never_reopens_index():
    """靶点：思考→正文→再思考曾被渲染成 start(1,text) 后紧接 start(1,thinking)。

    推理模型在工具循环里反复「想一点、写一点、再想一点」是常态，因此这不是
    边角情况，而是 Claude Code 跑 agent 时的主路径。
    """
    events = _events([
        ("Let me think. ", {"reasoning": True}),
        ("I will answer at length so the lookahead flushes. ", {}),
        ("and reconsider. ", {"reasoning": True}),
        ("Final answer.", {}),
    ])
    opened = _assert_block_lifecycle(events)
    types = [b["content_block"]["type"] for b in opened]
    assert types == ["thinking", "text", "thinking", "text"], types
    # 内容不丢失
    text = "".join(
        e["delta"].get("text", "") for e in events
        if e.get("type") == "content_block_delta" and e["delta"]["type"] == "text_delta"
    )
    assert "Final answer." in text, text
    print("test_reasoning_text_interleaving_never_reopens_index: PASSED")


def test_empty_stream_still_has_one_block():
    events = _events([("", {"conversation_id": "c-empty"})])
    opened = _assert_block_lifecycle(events)
    assert len(opened) == 1 and opened[0]["content_block"]["type"] == "text"
    print("test_empty_stream_still_has_one_block: PASSED")


def test_tool_stream_produces_valid_tool_use_block():
    events = _events(
        [
            ("I'll read it:\n```tool_call\n", {}),
            ('{"name": "Read", "arguments": {"file_path": "/tmp/a"}}\n', {}),
            ("```", {"conversation_id": "c-tool"}),
        ],
        tools=TOOLS,
    )
    opened = _assert_block_lifecycle(events)

    tool_blocks = [b for b in opened if b["content_block"]["type"] == "tool_use"]
    assert tool_blocks, f"工具调用未以 tool_use 块呈现: {[b['content_block']['type'] for b in opened]}"
    tb = tool_blocks[0]["content_block"]
    assert tb["name"] == "Read" and tb["id"]

    deltas = [
        e["delta"] for e in events
        if e.get("type") == "content_block_delta" and e["delta"]["type"] == "input_json_delta"
    ]
    assert deltas, "tool_use 块必须有 input_json_delta"
    assert json.loads(deltas[0]["partial_json"]) == {"file_path": "/tmp/a"}

    md = [e for e in events if e.get("type") == "message_delta"][0]
    assert md["delta"]["stop_reason"] == "tool_use", md
    print("test_tool_stream_produces_valid_tool_use_block: PASSED")


def test_tool_stream_with_reasoning_is_valid():
    """工具模式下思考与正文交替同样不得重开索引。"""
    events = _events(
        [
            ("Considering. ", {"reasoning": True}),
            ("I'll read it:\n```tool_call\n", {}),
            ('{"name": "Read", "arguments": {"file_path": "/tmp/b"}}\n', {}),
            ("```", {}),
        ],
        tools=TOOLS,
    )
    opened = _assert_block_lifecycle(events)
    types = [b["content_block"]["type"] for b in opened]
    assert types[0] == "thinking" and "tool_use" in types, types
    print("test_tool_stream_with_reasoning_is_valid: PASSED")


def test_terminal_delta_carries_usage():
    events = _events([("hello", {})])
    md = [e for e in events if e.get("type") == "message_delta"][-1]
    assert md["usage"]["output_tokens"] >= 1
    print("test_terminal_delta_carries_usage: PASSED")


def run_all():
    tests = [
        test_plain_text_stream_is_valid,
        test_thinking_only_stream_is_valid,
        test_reasoning_text_interleaving_never_reopens_index,
        test_empty_stream_still_has_one_block,
        test_tool_stream_produces_valid_tool_use_block,
        test_tool_stream_with_reasoning_is_valid,
        test_terminal_delta_carries_usage,
    ]
    for fn in tests:
        fn()
    print(f"\n✅ All Anthropic stream tests passed! ({len(tests)} cases)")


if __name__ == "__main__":
    run_all()