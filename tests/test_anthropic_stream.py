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


def _events(script, *, tools=None, thinking=None) -> list[dict]:
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
    if thinking is not None:
        body["thinking"] = thinking
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


# ------------------------------------------- 8. 停用词下的块生命周期


def _events_with_stop(script, stop_sequences, *, tools=None) -> list[dict]:
    """同 `_events`，但带上客户端停用词（Claude Code 常同时给 tools + stop_sequences）。"""
    from fastapi.testclient import TestClient as _TC

    ScriptedProvider.script = script
    default_factory = srv._make_provider
    srv._make_provider = lambda key: ScriptedProvider()
    body = {
        "model": "claude-3-7-sonnet",
        "max_tokens": 256,
        "stream": True,
        "messages": [{"role": "user", "content": "hi"}],
        "stop_sequences": stop_sequences,
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


def _text_of(events: list[dict]) -> str:
    return "".join(
        e["delta"].get("text", "") for e in events
        if e.get("type") == "content_block_delta" and e["delta"].get("type") == "text_delta"
    )


def test_stop_sequence_stream_keeps_block_lifecycle_valid():
    """[ANTHROPIC-STREAM] 停用词扫描器扣住尾巴时，不能把已关闭的块重新打开。

    实测缺陷：`feed()` 在尾巴短于保留长度时返回空串，调用方于是**关掉刚打开的
    正文块**并把 `current_block_type` 置空；命中之后走到收口处，`else` 分支又用
    硬编码的索引 0 开了一个空块——索引 0 早已被使用并关闭过。对 Claude Code 来说
    这是非法事件流：实测同一索引被复用后，客户端只留下了后一个块的内容，前面
    整段正文被静默丢弃（用户看到的是「答到一半没了」，没有任何报错）。
    """
    # 第一块短于保留窗口（"END" 保留 2 字符），必然触发空返回
    script = [("Hi", {}), (" there", {}), ("END", {}), ("never mind", {})]
    events = _events_with_stop(script, ["END"])
    opened = _assert_block_lifecycle(events)
    assert opened, "停用词路径必须产出至少一个正文块"
    text = _text_of(events)
    assert text == "Hi there", f"命中点之后的正文泄漏或正文丢失: {text!r}"
    md = [e for e in events if e.get("type") == "message_delta"][-1]
    assert md["delta"]["stop_reason"] == "stop_sequence", md
    print("test_stop_sequence_stream_keeps_block_lifecycle_valid: PASSED")


def test_stop_sequence_stream_does_not_fabricate_empty_blocks():
    """命中后不得继续把上游残料切成成串的空块：那既浪费事件、又踩到索引复用的路径。"""
    script = [("short", {}), ("END", {}), ("x" * 50, {}), ("y" * 50, {})]
    events = _events_with_stop(script, ["END"])
    opened = _assert_block_lifecycle(events)
    assert len(opened) == 1, [b["content_block"]["type"] for b in opened]
    assert _text_of(events) == "short"
    print("test_stop_sequence_stream_does_not_fabricate_empty_blocks: PASSED")


def test_stop_sequence_is_never_reported_as_an_empty_string():
    """[PROTOCOL] 只要 `stop_reason` 是 `stop_sequence`，就必须带上命中的原文。

    `SpeculativeToolStreamer.truncate_at_stop()` 此前只返回布尔值，调用方于是
    `mark_hit()` 无参调用，扫描器用默认的 `matched=""` 建了一条命中记录——线契约
    要求这个字段是命中的字符串，空串只告诉客户端「被截了」，不说被什么截。
    `truncate_final()` 本来就算出了命中的原文，只是被丢掉了。

    这里断言的是**性质**而不是某一个形状：两个模式 × 各种分块方式，凡是报告
    `stop_sequence` 的，`stop_sequence` 必须非空。单点用例只能守住想得到的那
    一条路径，而这条字段由两个模式、四个收口点共同产出。
    """
    scripts = [
        [("just prose", {}), ("STOPHERE", {})],
        [("pro", {}), ("se STOP", {}), ("HERE tail", {})],
        [("ok\n```tool_call\n", {}), ('{"name": "Read"}\n```\n', {}), ("STOPHERE", {})],
    ]
    checked = 0
    for script in scripts:
        for tools in (None, TOOLS):
            events = _events_with_stop(script, ["STOPHERE"], tools=tools)
            _assert_block_lifecycle(events)
            mds = [e for e in events if e.get("type") == "message_delta"]
            assert mds, f"缺少 message_delta: {script}"
            delta = mds[-1]["delta"]
            if delta.get("stop_reason") == "stop_sequence":
                assert delta.get("stop_sequence"), (
                    f"报告了 stop_sequence 却没带命中的原文 (tools={bool(tools)}): {delta}"
                )
                assert delta["stop_sequence"] == "STOPHERE", delta
                checked += 1
    assert checked >= 2, f"样本没有覆盖到 stop_sequence 报告路径，只命中 {checked} 次"
    print("test_stop_sequence_is_never_reported_as_an_empty_string: PASSED")


# ------------------------------------------- 9. count_tokens 与 messages 同构


def test_count_tokens_accepts_every_system_shape_messages_accepts():
    """[PROTOCOL] 同一个请求体，`/v1/messages` 接受而 `/v1/messages/count_tokens` 400。

    两个端点描述的是**同一个** Anthropic 请求体，却各有一份 pydantic 模型：
    `AnthropicCountTokensRequest.system` 是 `Optional[Union[str, List[dict]]]`，
    `AnthropicMessagesRequest.system` 是 `Optional[Union[str, List[Union[dict, str]]]]`。
    于是 `"system": ["you are terse"]`（字符串列表，Claude Code 与 SDK 都会发）
    在 messages 上合法、在 count_tokens 上被 400 拒绝：

        system -> list[dict[any,any]] -> 0: Input should be a valid dictionary

    Claude Code 在每次请求前用 count_tokens 做上下文预算判断，400 会让它直接
    失败，而用户看到的只是「agent 不动了」。这里断言的是**性质**：凡是 messages
    接受的 system 形状，count_tokens 必须接受（两处同构，不是逐个补类型）。
    """
    shapes = [
        "you are terse",                                            # 裸字符串
        ["you are terse"],                                          # 字符串列表（实测 400）
        [{"type": "text", "text": "you are terse"}],                # 块列表
        [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
        ["a", {"type": "text", "text": "b"}],                       # 混合
    ]

    default_factory = srv._make_provider
    ScriptedProvider.script = [("ok", {})]
    srv._make_provider = lambda key: ScriptedProvider()
    try:
        client = TestClient(srv.app)
        for system in shapes:
            body = {
                "model": "claude-3-7-sonnet",
                "max_tokens": 16,
                "system": system,
                "messages": [{"role": "user", "content": "hi"}],
            }
            r_msg = client.post("/v1/messages", json=body)
            assert r_msg.status_code == 200, (
                f"/v1/messages 拒绝了 system={system!r}: {r_msg.status_code} {r_msg.text[:200]}"
            )
            r_cnt = client.post("/v1/messages/count_tokens", json=body)
            assert r_cnt.status_code == 200, (
                f"/v1/messages/count_tokens 拒绝了 messages 接受的 system={system!r}: "
                f"{r_cnt.status_code} {r_cnt.text[:200]}"
            )
            assert isinstance(r_cnt.json().get("input_tokens"), int), r_cnt.text[:200]
    finally:
        srv._make_provider = default_factory
    print("test_count_tokens_accepts_every_system_shape_messages_accepts: PASSED")


def test_count_tokens_cache_key_covers_tool_choice():
    """[PROTOCOL] 同一端点上的第二处「同一份事实两种解释」：`tool_choice` 不进指纹。

    `compute_count_key` 只取 messages/system/tools，而 `tool_choice` 会改变送给
    模型的工具提示词（`_tool_choice_directive` 追加一句约束）。于是缓存键相同、
    实际提示词不同：先到的 `auto` 估计被 `none` 请求命中。实测（清空缓存后逐个
    测量）：`none` = 314、`auto` = 290；先发 `none` 再发 `auto`，两次都得到 314，
    `hits` 涨了 1——第二个请求拿到的是一个它自己的提示词算不出来的数。

    Claude Code 用这个数判断上下文预算，系统性偏低/偏高都会让它误判何时压缩。
    与 SingleFlight 的 `compute_key` 同一条纪律：**能改变输出的字段必须进指纹**，
    且加入字段只会减少共享、永远不会造成错误共享。
    """
    from providers.token_counter_cache import token_counter_cache

    tools = [{"name": "Read", "description": "read",
              "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}}}}]
    base = {"model": "claude-3-7-sonnet",
            "messages": [{"role": "user", "content": "hi"}], "tools": tools}

    client = TestClient(srv.app)
    token_counter_cache.clear()

    # 逐个清空缓存测量：这两个请求的提示词不同，token 数必须不同。
    token_counter_cache.clear()
    none_only = client.post("/v1/messages/count_tokens",
                            json={**base, "tool_choice": "none"}).json()["input_tokens"]
    token_counter_cache.clear()
    auto_only = client.post("/v1/messages/count_tokens",
                            json={**base, "tool_choice": "auto"}).json()["input_tokens"]
    assert none_only != auto_only, (
        f"tool_choice 会改变工具提示词，估计值却相同 ({none_only})——"
        "说明估算路径根本没看到 tool_choice"
    )

    # 不清缓存连发：若指纹漏掉 tool_choice，第二个请求会命中第一个的缓存。
    token_counter_cache.clear()
    first = client.post("/v1/messages/count_tokens",
                        json={**base, "tool_choice": "none"}).json()["input_tokens"]
    second = client.post("/v1/messages/count_tokens",
                         json={**base, "tool_choice": "auto"}).json()["input_tokens"]
    assert (first, second) == (none_only, auto_only), (
        "缓存命中把不同 tool_choice 的估计混为一谈："
        f"逐测 ({none_only}, {auto_only}) vs 连发 ({first}, {second})"
    )
    token_counter_cache.clear()
    print("test_count_tokens_cache_key_covers_tool_choice: PASSED")


def test_count_tokens_sees_the_same_tools_prompt_as_messages():
    """[PROTOCOL] 估算路径与真实路径必须看到同一份提示词。

    `tool_choice.disable_parallel_tool_use: true` 会让 `/v1/messages` 在工具提示词
    末尾追加「Do NOT emit more than one tool call in a turn」一段，而 `count_tokens`
    从不传 `parallel_allowed`，于是同一份请求报文在两条路径上生成**不同**的提示词
    （实测 1267 字符 vs 1122 字符）。估算值系统性偏低，而 Claude Code 正是用它判断
    上下文预算、决定何时压缩——偏低的数字会让它在该压缩时继续硬塞。

    断言选的是**方向**（带约束的提示词必然更长），而不是具体数值：数值会随提示词
    文案演进漂移，方向不会。
    """
    from providers.token_counter_cache import token_counter_cache

    tools = [{"name": "Read", "description": "read",
              "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}}}}]
    base = {"model": "claude-3-7-sonnet",
            "messages": [{"role": "user", "content": "hi"}], "tools": tools}

    client = TestClient(srv.app)

    # 用同一份请求分别走两条路径，比较真正送给模型的提示词，而不是比较接口返回。
    seen: list = []
    original = srv.convert_anthropic_to_gateway_messages

    def spy(messages, system=None, tools=None, tool_choice=None, parallel_allowed=None):
        out = original(messages, system=system, tools=tools,
                       tool_choice=tool_choice, parallel_allowed=parallel_allowed)
        seen.append(srv.flatten_messages(out))
        return out

    # `/v1/messages` 必须被打桩：本套件此前有测试因漏装脚本 provider 而向真实 Qwen
    # 发了请求（CLAUDE.md 记录的 tripwire 事故），这里显式装上，绝不让提示词断言
    # 以一次真实上游调用为代价。
    ScriptedProvider.script = [("ok", {})]
    default_factory = srv._make_provider
    srv._make_provider = lambda key: ScriptedProvider()
    srv.convert_anthropic_to_gateway_messages = spy
    try:
        token_counter_cache.clear()
        named = {"type": "tool", "name": "Read"}
        parallel = {"type": "tool", "name": "Read", "disable_parallel_tool_use": True}

        client.post("/v1/messages/count_tokens", json={**base, "tool_choice": named})
        count_named = seen[-1]
        client.post("/v1/messages/count_tokens", json={**base, "tool_choice": parallel})
        count_serial = seen[-1]

        client.post("/v1/messages", json={**base, "tool_choice": parallel})
        msg_serial = seen[-1]
    finally:
        srv.convert_anthropic_to_gateway_messages = original
        srv._make_provider = default_factory
        token_counter_cache.clear()

    assert count_serial == msg_serial, (
        "count_tokens 与 /v1/messages 对同一份报文生成了不同的提示词"
        f"（{len(count_serial)} vs {len(msg_serial)} 字符）——估算的是另一个请求"
    )
    # 这里原本比较的是 `len(count_serial) > len(count_named)`（禁止并行的那份更长）。
    # 长度只是「约束段在不在」的代理：§101 让**允许**并行的那一份也变长了（规则 2
    # 多了一句「多个独立调用可以背靠背写出来」），一个正当的措辞改进就把这个代理
    # 推翻了，而它想断言的事实没有变。直接断言那一段文字本身，既不受措辞长度影响，
    # 也比长度更贴近用例名字里的主张。
    serial_marker = "Do NOT emit more than one tool call in a turn"
    assert serial_marker in count_serial, (
        "`disable_parallel_tool_use` 的约束段没有进入估算路径的提示词"
    )
    assert serial_marker not in count_named, (
        "客户端没有禁止并行，估算路径的提示词却带着禁止段"
    )
    print("test_count_tokens_sees_the_same_tools_prompt_as_messages: PASSED")


def test_count_tokens_sees_the_folded_history_messages_actually_sends():
    """[PROTOCOL] 估算路径没走过折叠，于是它报的是一个**已经被网关丢弃的**历史。

    真实 `claude -p` 15 轮 soak（3 工具调用/轮，每条工具输出填充 2400 字符）抓到的
    16 份请求体，逐份回放给 `/v1/messages/count_tokens`，与同一份请求真实
    `message_start.usage.input_tokens`（= 送进上游的展平提示词估算值）对比：

        turn   client_msgs   count_tokens   真实 input_tokens
          0         2            8695             8695
          3        11           21539            21539
          4        14          **25820**        **21855**    <- 折叠开始生效
          5        17          **30102**        **17378**
          ...
         15        47          **72926**        **17528**    <- 4.2 倍

    `/v1/messages` 在 `optimize_chat_messages` 之后才算 `input_tokens`，折叠把
    30k 字符的历史压成一行标记；`count_tokens` 则直接 `flatten_messages` 原始
    转换结果，从不折叠。超过 `GATEWAY_MAX_CONTEXT_TOKENS` 之后两者的差距随轮次
    单调放大，到第 15 轮偏 **+55400 tokens（4.2 倍）**。

    这个数字是客户端用来决定「还能不能再塞一轮 / 该不该压缩」的。系统性偏高会让
    Claude Code 在上下文其实还有大量余量时就开始激进压缩（丢掉真实上下文），
    和被修过的「偏低」是同一个错误的两个方向。

    断言用的是**同一条请求的两条路径必须给出同一个数**，而不是某个具体数值：
    数值随提示词文案漂移，这个等式不会。
    """
    from fastapi.testclient import TestClient as _TC

    # 构造一段超过上下文预算的历史：14 轮工具往返，每轮工具输入/输出都远超
    # 阶段一的折叠阈值（300/400 字符），确保折叠一定会被触发。
    pad = 4000
    messages = [{"role": "user", "content": "Do the long task."}]
    for i in range(14):
        messages.append({
            "role": "assistant",
            "content": [
                {"type": "text", "text": f"step {i}"},
                {"type": "tool_use", "id": f"toolu_{i}", "name": "Bash",
                 "input": {"command": f"echo {i} " + "x" * pad}},
            ],
        })
        messages.append({
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": f"toolu_{i}",
                 "content": f"out {i} " + "y" * pad},
            ],
        })

    body = {
        "model": "claude-3-7-sonnet",
        "max_tokens": 256,
        "tools": [{"name": "Bash", "description": "run",
                   "input_schema": {"type": "object",
                                    "properties": {"command": {"type": "string"}}}}],
        "messages": messages,
    }

    from providers.token_counter_cache import token_counter_cache

    ScriptedProvider.script = [("ok", {"conversation_id": "c1"})]
    default_factory = srv._make_provider
    srv._make_provider = lambda key: ScriptedProvider()
    try:
        token_counter_cache.clear()
        client = _TC(srv.app)

        # 折叠裁决共用，但**记账**只属于真实派发：一次估算调用若把「节省」计入
        # 折叠遥测，诊断就会声称省下了从未发生的 token。
        before = dict(srv.prompt_optimizer.get_stats())
        counted = client.post("/v1/messages/count_tokens", json=body).json()["input_tokens"]
        after = dict(srv.prompt_optimizer.get_stats())
        assert after == before, (
            "count_tokens 改写了折叠遥测（估算不是派发）："
            f"{ {k: (before[k], after[k]) for k in before if before[k] != after[k]} }"
        )

        # 从真实流式响应里读 `message_start.usage.input_tokens`：这是网关**真的**
        # 送进上游的那个展平提示词的估算值，不是我们重新算一遍。
        with client.stream("POST", "/v1/messages", json={**body, "stream": True}) as r:
            assert r.status_code == 200, r.read()
            raw = "".join(r.iter_text())
    finally:
        srv._make_provider = default_factory
        token_counter_cache.clear()

    real = None
    for chunk in raw.split("\n\n"):
        for line in chunk.split("\n"):
            if line.startswith("data: "):
                try:
                    obj = json.loads(line[6:])
                except Exception:
                    continue
                if obj.get("type") == "message_start":
                    real = obj["message"]["usage"]["input_tokens"]
    assert real is not None, f"流里没有 message_start: {raw[:300]}"

    assert counted == real, (
        "count_tokens 与真实请求的 input_tokens 不一致："
        f"count_tokens={counted} vs message_start={real}"
        f"（偏 {counted - real}, {counted / real:.2f} 倍）。"
        "估算路径没有应用 `optimize_chat_messages`，报的是一个已经被网关折叠掉的"
        "历史——客户端据此做的上下文预算判断是错的。"
    )
    print("test_count_tokens_sees_the_folded_history_messages_actually_sends: PASSED")


# ------------------------------------------- 14. thinking.display = "omitted"


def _thinking_text(events: list[dict]) -> str:
    return "".join(
        e["delta"]["thinking"] for e in events
        if e.get("type") == "content_block_delta" and e["delta"].get("type") == "thinking_delta"
    )


def test_thinking_display_omitted_sends_no_thinking_text():
    """Claude Code 每一轮都发 `{"type":"adaptive","display":"omitted"}`。

    实测（真实 claude 2.1.293，`--model claude-opus-5-5`）修复前：网关照常把
    思考文本作为 `thinking_delta` 发给客户端，客户端把它存进对话历史，下一轮
    `messages` 里就多出一个 `{"type":"thinking","thinking":"Let me think..."}`
    块——**尽管它明确要求 omitted**。工具循环里这个块被反复回放（实测一次会话
    长到 154 个 assistant 轮次、每个都带 thinking 块）。

    `display` 的语义由 Claude Code 二进制自己的 schema 定义：
    `display: z(["summarized","omitted"]).optional()`，且文档写
    "'omitted'（no thinking text）"。所以 omitted 时**一个 thinking_delta
    都不该出现**——不是少发，是零发。
    """
    events = _events(
        [("thinking...", {"reasoning": True}), ("answer", {"conversation_id": "c1"})],
        thinking={"type": "adaptive", "display": "omitted"},
    )
    _assert_block_lifecycle(events)
    assert _thinking_text(events) == "", (
        f"客户端要求 display=omitted，网关仍然发出了思考文本: {_thinking_text(events)!r}"
    )
    # 正文必须照常送达——抑制思考不能把答案一起吞掉
    text = "".join(
        e["delta"]["text"] for e in events
        if e.get("type") == "content_block_delta" and e["delta"]["type"] == "text_delta"
    )
    assert text == "answer", f"抑制思考时正文被吞: {text!r}"


def test_thinking_display_summarized_still_sends_thinking():
    """对照组：没有 display 或 display != omitted 时，思考照常发出。

    这条是上一条的**反向保险**：如果实现写成「只要有 thinking 字段就抑制」，
    这里会立刻变红。
    """
    for thinking in (
        {"type": "adaptive", "display": "summarized"},
        {"type": "adaptive"},
        {"type": "enabled"},
        True,
    ):
        events = _events(
            [("thinking...", {"reasoning": True}), ("answer", {"conversation_id": "c1"})],
            thinking=thinking,
        )
        _assert_block_lifecycle(events)
        assert _thinking_text(events) == "thinking...", (
            f"thinking={thinking} 时思考文本不该被抑制，实际: {_thinking_text(events)!r}"
        )


def test_thinking_display_omitted_still_recovers_tool_from_reasoning():
    """抑制**只在发送边界**，绝不在累积边界。

    这一条守住的是 [TOOL-LOOP] 的思考通道兜底：推理模型把整个 ```tool_call
    写在思考流里、正文为空时，网关要从思考流把它救回来。如果实现图省事在
    `accumulated_thinking` 那一层就把内容丢掉，这里会变成「两者都空」的回合
    ——客户端既没有答案也没有调用，循环当场停住。
    """
    events = _events(
        [
            ("```tool_call\n", {"reasoning": True}),
            ('{"name": "Read", "arguments": {"file_path": "/tmp/x"}}\n', {"reasoning": True}),
            # 收尾的 ``` 也必须带 reasoning 标记：思考流里的每一块都是思考，
            # 漏标会让它落进正文通道，正文不再为空，兜底规则自然不触发
            # （这正是这条用例第一版写错的地方——它测的其实不是抑制）。
            ("```", {"reasoning": True, "conversation_id": "c1"}),
        ],
        tools=TOOLS,
        thinking={"type": "adaptive", "display": "omitted"},
    )
    _assert_block_lifecycle(events)
    starts = [e for e in events if e.get("type") == "content_block_start"]
    assert [b["content_block"]["type"] for b in starts] == ["tool_use"], (
        f"omitted 抑制了思考文本，却把思考流里的工具调用也一起丢了: "
        f"{[b['content_block']['type'] for b in starts]}"
    )
    assert starts[0]["content_block"]["name"] == "Read"
    assert _thinking_text(events) == "", "工具路径同样不能发思考文本"


def test_non_streaming_thinking_display_omitted_sends_no_thinking_block():
    """非流式路径同一条语义：`content` 里不能有 thinking 块。"""
    from fastapi.testclient import TestClient as _TC

    ScriptedProvider.script = [
        ("thinking...", {"reasoning": True}),
        ("answer", {"conversation_id": "c1"}),
    ]
    default_factory = srv._make_provider
    srv._make_provider = lambda key: ScriptedProvider()
    try:
        client = _TC(srv.app)
        r = client.post(
            "/v1/messages",
            json={
                "model": "claude-3-7-sonnet",
                "max_tokens": 256,
                "stream": False,
                "thinking": {"type": "adaptive", "display": "omitted"},
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
    finally:
        srv._make_provider = default_factory
    assert r.status_code == 200, r.text
    data = r.json()
    kinds = [b["type"] for b in data["content"]]
    assert "thinking" not in kinds, (
        f"非流式路径把思考块发给了要求 omitted 的客户端: {data['content']}"
    )
    assert any(b["type"] == "text" and b["text"] == "answer" for b in data["content"]), (
        f"正文丢失: {data['content']}"
    )


def test_thinking_omitted_does_not_change_the_token_count():
    """抑制发送不改 `usage`：思考确实发生了，只是不展示。

    客户端用 output_tokens 做预算判断；把被抑制的思考从 usage 里也扣掉，
    会让它对上下文的估计系统性偏低。
    """
    suppressed = _events(
        [("thinking...", {"reasoning": True}), ("answer", {"conversation_id": "c1"})],
        thinking={"type": "adaptive", "display": "omitted"},
    )
    shown = _events(
        [("thinking...", {"reasoning": True}), ("answer", {"conversation_id": "c1"})],
        thinking={"type": "adaptive", "display": "summarized"},
    )
    sup_tok = [e for e in suppressed if e.get("type") == "message_delta"][0]["usage"]["output_tokens"]
    shown_tok = [e for e in shown if e.get("type") == "message_delta"][0]["usage"]["output_tokens"]
    assert sup_tok == shown_tok, (
        f"抑制展示改变了 output_tokens（{sup_tok} vs {shown_tok}）——"
        "思考确实产生了，只是不展示"
    )


def test_metadata_session_id_keys_the_conversation():
    """`metadata.user_id.session_id` 是 Claude Code **唯一**的会话身份，必须被用上。

    实测（真实 `claude` 2.1.293，19 份抓包，见 `/tmp/e2e/claude_bodies.jsonl` 等）：
    Claude Code 从不发送 `conversation_id` —— 它把会话身份放在
    `metadata.user_id` 这个 JSON 字符串里（`{"device_id":…, "session_id":"<uuid>"}`，
    device_id 跨会话恒定，session_id 每个会话一个）。而网关的会话粘滞、续聊判定、
    匿名临时会话删除全部只读 `req.conversation_id`，于是这条身份**一个字符都没被读过**。

    修复前的实测症状（同一份抓包驱动，5 轮）：
      - 每轮 `conversation_id` 传给上游都是 `None` → provider 永远走「新建会话」分支；
      - 每轮都重发**全量展平历史**：20852 / 21643 / 22434 / 23225 / 24016 字符，
        合计 112,170 字符；而带上会话身份后是 20852 + 4×10 = 20,892 字符
        （5.4 倍的输入 token 白付，且随轮次线性增长）；
      - 每轮都把「匿名临时会话」排进删除队列（实测 3 轮 = 3 次删除），
        而上游会话其实本该被续用。

    这条测试断言的是**上游真的收到了会话身份**，不是网关自己的头部：
    一个只改了遥测、没改 `chat_args` 的修复会在这里红。
    """
    from fastapi.testclient import TestClient as _TC

    seen = []

    class _ConvProvider:
        async def chat(self, messages, model, **kw):
            seen.append(kw.get("conversation_id"))
            yield "ok", {}
            yield "", {"conversation_id": "UPSTREAM-CONV"}

        async def delete_conversation(self, *a, **kw):
            seen.append("DELETED")
            return True

    default_factory = srv._make_provider
    srv._make_provider = lambda key: _ConvProvider()
    srv.session_affinity_manager.clear()
    # 真实 Claude Code 的形状：没有 conversation_id，只有 metadata。
    meta = {
        "user_id": json.dumps(
            {"device_id": "29c47c8beabc", "account_uuid": "", "session_id": "sess-AAA"}
        )
    }
    try:
        client = _TC(srv.app)
        # [SESSION-IDENTITY] 对话线的身份包含「线的起手用户轮」（同一父会话顺序派生
        # 的两个同级子 agent 只有任务不同，起手轮是唯一能把它们分开的事实，见
        # `tests/test_subagent_affinity_isolation.py`）。因此每轮重发**全量历史**、
        # 起手轮不变，才是真实 Anthropic 客户端的形状——协议没有服务端历史，客户端
        # 每轮自己带。以前这里每轮只发一句 `turn {i}`，等于每轮开一条新线，续聊
        # 永远不会发生，测的其实是「新会话」而不是本用例声称的「会话身份生效」。
        history: list[dict] = []
        for i in range(3):
            history.append({"role": "user", "content": f"turn {i}"})
            r = client.post(
                "/v1/messages",
                json={
                    "model": "claude-3-5-sonnet",
                    "max_tokens": 64,
                    "stream": False,
                    "metadata": meta,
                    "messages": list(history),
                },
            )
            assert r.status_code == 200, r.text
            history.append({"role": "assistant", "content": "ok"})
    finally:
        srv._make_provider = default_factory
        srv.session_affinity_manager.clear()

    continued = [c for c in seen if c != "DELETED"]
    deletes = [c for c in seen if c == "DELETED"]
    assert continued[0] is None, (
        f"首轮不该带上游会话 id（还没观测到过）: {continued}"
    )
    assert continued[1:] == ["UPSTREAM-CONV", "UPSTREAM-CONV"], (
        "`metadata.user_id.session_id` 没有被当作会话身份：上游从没收到过续聊 id，"
        f"每轮都在重建上下文。实测 = {continued}"
    )
    assert not deletes, (
        f"会话身份缺失导致每轮都把活跃会话当成匿名临时会话排进删除队列: {len(deletes)} 次"
    )
    print("test_metadata_session_id_keys_the_conversation: PASSED")


def run_all():
    tests = [
        test_plain_text_stream_is_valid,
        test_thinking_only_stream_is_valid,
        test_reasoning_text_interleaving_never_reopens_index,
        test_empty_stream_still_has_one_block,
        test_tool_stream_produces_valid_tool_use_block,
        test_tool_stream_with_reasoning_is_valid,
        test_terminal_delta_carries_usage,
        test_stop_sequence_stream_keeps_block_lifecycle_valid,
        test_stop_sequence_stream_does_not_fabricate_empty_blocks,
        test_stop_sequence_is_never_reported_as_an_empty_string,
        test_count_tokens_accepts_every_system_shape_messages_accepts,
        test_count_tokens_cache_key_covers_tool_choice,
        test_count_tokens_sees_the_same_tools_prompt_as_messages,
        test_count_tokens_sees_the_folded_history_messages_actually_sends,
        test_thinking_display_omitted_sends_no_thinking_text,
        test_thinking_display_summarized_still_sends_thinking,
        test_thinking_display_omitted_still_recovers_tool_from_reasoning,
        test_non_streaming_thinking_display_omitted_sends_no_thinking_block,
        test_thinking_omitted_does_not_change_the_token_count,
        test_metadata_session_id_keys_the_conversation,
    ]
    for fn in tests:
        fn()
    print(f"\n✅ All Anthropic stream tests passed! ({len(tests)} cases)")


if __name__ == "__main__":
    run_all()