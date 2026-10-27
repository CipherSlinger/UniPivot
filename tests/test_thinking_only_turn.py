"""[THINKING-TURN] 被抑制的思考仍然是这一回合产出的内容

`thinking.display == "omitted"` 是真实 Claude Code 每一次请求都带的声明（§58 实测
6/6 个请求体），网关因此不把思考文本发给客户端。但「不想看」不等于「没有产出」：
推理模型经常把整个回合只写在思考通道里（没有任何正文），而这条回合必须仍然是
一个**有内容的合法响应**。

修复前的两个症状，都是同一条根因的两个出口——某个分支只从「正文」推导这一回合：

1. 非流式 `/v1/messages` 返回 `content: []` + `stop_reason: "end_turn"` + 非零
   usage。客户端看到的是一个「正常结束、什么都没有」的助手回合，既没有可读的
   回答，也没有可继续的东西。同一条上游输出在**流式**路径上却是合法的空正文块
   （`content_block_start(0,text)` / `stop`），即同一份输出在两个传输上给出两种
   契约——本仓库反复记录的「一处事实、两处解释」。

2. 带 tools 的**流式**分支的 `message_delta.usage.output_tokens` 只计入正文，
   思考 tokens 全部漏掉；同一份上游输出在非流式与非工具流式路径上都计入思考，
   三个口径里有一个是错的。

钉住的规则只有一条：**思考被抑制，只是它的显示被拒绝，它的内容仍然存在**——
它必须让回合不再是空的，也必须计入 usage。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# [OPT-019] 隔离运行：真实 session/ 目录不得被测试写入。
os.environ["TESTING"] = "1"
os.environ["GATEWAY_DISABLE_CREDENTIAL_WATCH"] = "1"
os.environ["GATEWAY_DISABLE_AUTO_REFRESH"] = "1"
os.environ["GATEWAY_DISABLE_PROMPT_FOLDING"] = "1"
os.environ["QWEN_AUTH_TOKEN"] = "fake_qwen"
os.environ["DEEPSEEK_AUTH_TOKEN"] = "fake_ds"

from fastapi.testclient import TestClient  # noqa: E402

import server as srv  # noqa: E402

TOOLS = [
    {
        "name": "Read",
        "description": "Read a file",
        "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}}},
    }
]

OMITTED = {"type": "adaptive", "display": "omitted"}
ENABLED = {"type": "enabled"}

# 只写在思考通道里的回合：没有任何正文。
REASONING_ONLY = [("thinking about it", {"reasoning": True, "conversation_id": "c1"})]
REASONING_AND_TEXT = [("thinking about it", {"reasoning": True}), ("ab", {"conversation_id": "c1"})]

_GUARD_HITS: list[str] = []


def _guarded_factory(key, client=None):
    """绊线：没有安装脚本化 provider 的用例会在这里失败，而不是真的触碰上游。"""
    _GUARD_HITS.append(key)
    raise AssertionError("测试发出了未安装假 provider 的上游请求")


srv._make_provider = _guarded_factory


class ScriptedProvider:
    script: list = []

    async def chat(self, *a, **kw):
        for delta, meta in ScriptedProvider.script:
            yield delta, meta


def _post(script, *, stream, tools=None, thinking=None):
    ScriptedProvider.script = script
    srv._make_provider = lambda key, client=None: ScriptedProvider()
    body = {
        "model": "claude-3-7-sonnet",
        "max_tokens": 256,
        "stream": stream,
        "messages": [{"role": "user", "content": "hi"}],
    }
    if tools is not None:
        body["tools"] = tools
    if thinking is not None:
        body["thinking"] = thinking
    try:
        client = TestClient(srv.app)
        if stream:
            with client.stream("POST", "/v1/messages", json=body) as r:
                assert r.status_code == 200, r.read()
                raw = "".join(r.iter_text())
            events = []
            for block in raw.split("\n\n"):
                for line in block.splitlines():
                    if line.startswith("data: "):
                        try:
                            events.append(json.loads(line[len("data: "):]))
                        except json.JSONDecodeError:
                            pass
            return events
        r = client.post("/v1/messages", json=body)
        assert r.status_code == 200, r.text
        return r.json()
    finally:
        srv._make_provider = _guarded_factory


def _stream_blocks(events) -> list[tuple[str, str]]:
    """把事件流折叠成与响应体同构的 `[(block_type, text)]`。"""
    blocks: list[list[str]] = []
    current = None
    for ev in events:
        kind = ev.get("type")
        if kind == "content_block_start":
            payload = ev.get("content_block") or {}
            current = [payload.get("type", ""), payload.get("text") or payload.get("thinking") or ""]
            blocks.append(current)
        elif kind == "content_block_delta" and current is not None:
            delta = ev.get("delta") or {}
            current[1] += delta.get("text") or delta.get("thinking") or ""
        elif kind == "content_block_stop":
            current = None
    return [(b[0], b[1]) for b in blocks]


def _body_blocks(msg) -> list[tuple[str, str]]:
    return [
        (b.get("type", ""), b.get("text") or b.get("thinking") or "")
        for b in (msg.get("content") or [])
    ]


# ------------------------------------------------- 1. 回合不能是空的

def test_a_reasoning_only_turn_is_never_an_empty_message():
    """非流式、无 tools：思考被抑制时，回合仍然必须至少有一个内容块。

    修复前实测：
        content=[]  stop_reason=end_turn  usage={'input_tokens': 1, 'output_tokens': 4}
    而同一份上游输出在流式路径上是
        content_block_start(0,text) → content_block_stop(0)
    客户端读到的「HTTP 200 + 空数组 + end_turn」是一个正常结束的空回合：没有回答、
    没有错误、也没有任何可以继续的东西。
    """
    msg = _post(REASONING_ONLY, stream=False, thinking=OMITTED)
    assert msg["content"], (
        f"只有思考内容的回合被返回成空 content，stop_reason={msg.get('stop_reason')!r}，"
        f"usage={msg.get('usage')!r}——客户端看到的是「正常结束但什么都没有」"
    )
    assert [b["type"] for b in msg["content"]] == ["text"], (
        f"空回合应当补一个合法的空正文块（与流式一致），实际为 {msg['content']!r}"
    )
    print("test_a_reasoning_only_turn_is_never_an_empty_message: PASSED")


def test_the_tools_path_cannot_go_empty_either():
    """带 tools 但没收出工具调用时，同样不能是空回合。

    工具分支有自己的提前 return（tool_use 响应），空正文块那段守卫在它下面——
    只修无工具分支会让「声明了工具但没有调用」的回合继续为空。
    """
    msg = _post(REASONING_ONLY, stream=False, tools=TOOLS, thinking=OMITTED)
    assert msg["content"], (
        f"带 tools 的空回合仍然是 content=[]，stop_reason={msg.get('stop_reason')!r}"
    )
    assert [b["type"] for b in msg["content"]] == ["text"], f"实际块: {msg['content']!r}"
    print("test_the_tools_path_cannot_go_empty_either: PASSED")


def test_the_two_transports_agree_on_the_same_upstream_output():
    """同一份上游输出，流式与非流式必须给出同一个骨架。

    这是本套件的核心断言：修复若只落在一个传输上，就会重新制造「一处事实、两处
    解释」——而这正是修复前的形态（流式合法、非流式空）。对同一个脚本，两个
    传输的块类型序列与文本必须逐字相等。
    """
    for tools in (None, TOOLS):
        label = "tools" if tools else "no-tools"
        streamed = _stream_blocks(_post(REASONING_ONLY, stream=True, tools=tools, thinking=OMITTED))
        body = _body_blocks(_post(REASONING_ONLY, stream=False, tools=tools, thinking=OMITTED))
        assert streamed == body, (
            f"[{label}] 两个传输对同一份上游输出给出了不同的骨架："
            f"stream={streamed!r} body={body!r}"
        )
    print("test_the_two_transports_agree_on_the_same_upstream_output: PASSED")


# --------------------------------------------- 2. usage 口径必须一致

def test_tools_streaming_usage_counts_the_reasoning():
    """带 tools 的流式分支的 output_tokens 必须包含思考 tokens。

    修复前实测（同一份上游输出：100 字思考 + "ab"）：
        tools  streaming message_delta: {'output_tokens': 1}
        notool streaming message_delta: {'output_tokens': 150}
        non-stream usage:               {'output_tokens': 150}
    工具回合是 agent 最常走的路径，于是最常走的路径上报的成本少了整段推理——
    并且与同一份输出的非流式响应自相矛盾。
    """
    stream_evs = _post(REASONING_AND_TEXT, stream=True, tools=TOOLS)
    stream_tools = [e["usage"]["output_tokens"] for e in stream_evs if e.get("type") == "message_delta"]
    stream_plain = [
        e["usage"]["output_tokens"]
        for e in _post(REASONING_AND_TEXT, stream=True)
        if e.get("type") == "message_delta"
    ]
    body = _post(REASONING_AND_TEXT, stream=False, tools=TOOLS)["usage"]["output_tokens"]
    assert stream_tools and stream_plain, f"缺少 message_delta: {stream_evs!r}"
    assert stream_tools[-1] == body, (
        f"工具流式的 output_tokens={stream_tools[-1]} 与非流式的 {body} 不一致："
        f"思考 tokens 只在一个分支里被计入"
    )
    assert stream_tools[-1] == stream_plain[-1], (
        f"同为流式，工具分支 {stream_tools[-1]} 与非工具分支 {stream_plain[-1]} 口径不同"
    )
    print("test_tools_streaming_usage_counts_the_reasoning: PASSED")


# --------------------------------------------------------- 3. 零回归

def test_a_reasoning_only_turn_with_display_enabled_keeps_its_thinking_block():
    """控制方向：`display` 未声明时，思考块本身就是这一回合，不得再补空正文块。

    修复若写成「正文为空就补一个空 text 块」，会把这里变成 thought+empty-text
    两个块——那是把一个正确的响应改成了另一个形状。
    """
    msg = _post(REASONING_ONLY, stream=False, thinking=ENABLED)
    assert _body_blocks(msg) == [("thinking", "thinking about it")], (
        f"display 未抑制时思考块被动过: {msg['content']!r}"
    )
    print("test_a_reasoning_only_turn_with_display_enabled_keeps_its_thinking_block: PASSED")


def test_a_plain_text_turn_is_unchanged():
    """控制方向：普通正文回合与 usage 不受影响。"""
    script = [("hello there", {"conversation_id": "c1"})]
    msg = _post(script, stream=False, thinking=OMITTED)
    assert _body_blocks(msg) == [("text", "hello there")], f"正文回合被改动: {msg['content']!r}"
    assert msg["stop_reason"] == "end_turn", msg
    assert _stream_blocks(_post(script, stream=True, thinking=OMITTED)) == [("text", "hello there")], (
        "流式正文回合被改动"
    )
    print("test_a_plain_text_turn_is_unchanged: PASSED")


def run_all():
    test_a_reasoning_only_turn_is_never_an_empty_message()
    test_the_tools_path_cannot_go_empty_either()
    test_the_two_transports_agree_on_the_same_upstream_output()
    test_tools_streaming_usage_counts_the_reasoning()
    test_a_reasoning_only_turn_with_display_enabled_keeps_its_thinking_block()
    test_a_plain_text_turn_is_unchanged()
    assert not _GUARD_HITS, (
        f"有 {len(_GUARD_HITS)} 次上游调用未经脚本化 provider: {sorted(set(_GUARD_HITS))}"
    )
    print("\n✅ All thinking-only turn tests passed! (6 cases)")


if __name__ == "__main__":
    sys.exit(run_all())
