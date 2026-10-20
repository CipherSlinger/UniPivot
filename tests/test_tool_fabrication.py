"""伪工具调用与流失锁 (TOOL-FABRICATION) 测试套件

动机来自「工具调用没跑起来」的第二类表现：模型明明在正常回答，客户端却收到一个
它从未声明过、也无法执行的工具调用，于是整个回合被判无效；或者回答说到一半停住，
剩下的内容在最后一次性乱序吐出。

两类缺陷同源：**「看起来像工具调用」被当成了「是工具调用」**。

  A. 解析端过度识别。围栏正则原先匹配任意语言标签，`_extract_single_tool` 又把
     任何带 `name` 键的对象当成调用、并把「其余所有键」当参数。实测：
       '```json\\n{"name": "Alice", "age": 30}\\n```'
       → tool=Alice(args={'age': 30})，且块前正文被截断丢弃
       '```tool_call\\n{"name": "EvilTool", "arguments": {}}\\n```'
       → 原样透传一个客户端未声明的工具名，客户端拒绝整轮

  B. 流式端永久锁存。`tool_detected` 一旦置位永不复位，一次误触发之后整段回答被吞。

修复分三层，各管一件事：
  1. 格式白名单：普通代码块（python/ts/...）不再进入工具通道；json 仍然允许，
     因为真实模型确实常这么写——**不靠猜格式防伪造**。
  2. 严格形状：只接受 `{"name":..., "arguments"|...: ...}` 或 OpenAI 的
     `{"function": {"name":..., "arguments":...}}`，不再有「其余键当参数」的兜底。
  3. 工具名白名单：名字必须是客户端在 `tools` 里声明过的。

覆盖：
1. 普通数据对象不再被伪造成工具调用（靶点 A）
2. python/ts 等代码块不再进入工具通道
3. 客户端未声明的工具名被丢弃（靶点 A）
4. 声明过的工具名照常识别（零回归）
5. OpenAI `{"function": {...}}` 形状照常识别（零回归）
6. ```json 仍是有效的工具调用承载格式（真实模型行为，零回归）
7. 流式：一个普通 ```json 数据块之后，后续文本必须继续正常流出（靶点 B）
8. 流式：误触发被证伪后解除锁存并放出已缓冲内容
9. 流式：真正的工具调用仍然被正确缓冲与解析（零回归）

运行: .venv/bin/python tests/test_tool_fabrication.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["TESTING"] = "1"

from providers.tool_streamer import (  # noqa: E402
    parse_tool_calls_with_healing,
    tool_names_of,
)

# 客户端声明了这两个工具（Claude Code 风格）
TOOLS = [
    {"type": "function", "function": {"name": "Read", "parameters": {"type": "object"}}},
    {"type": "function", "function": {"name": "Bash", "parameters": {"type": "object"}}},
]
ALLOWED = tool_names_of(TOOLS)


# ------------------------------------------------- 1. 数据对象不得被伪装

def test_plain_data_object_is_not_a_tool_call():
    """**靶点 A**：修复前这里会产出 `Alice(age=30)` 并吞掉周围正文。"""
    text = (
        "Here is the config:\n```json\n"
        '{"name": "Alice", "age": 30}\n'
        "```\nHope that helps!"
    )
    parsed, _ = parse_tool_calls_with_healing(text, ALLOWED)
    assert parsed is None, f"普通数据对象被误判为工具调用: {parsed}"
    print("test_plain_data_object_is_not_a_tool_call: PASSED")


def test_python_fence_is_not_a_tool_channel():
    text = '```python\nd = {"name": "x"}\n```'
    parsed, _ = parse_tool_calls_with_healing(text, ALLOWED)
    assert parsed is None, f"python 代码块被当成了工具调用: {parsed}"
    print("test_python_fence_is_not_a_tool_channel: PASSED")


def test_undeclared_tool_name_is_rejected():
    """**靶点 A**：模型臆造的工具名对客户端毫无意义，必须丢弃而不是透传。"""
    text = '```tool_call\n{"name": "EvilTool", "arguments": {}}\n```'
    parsed, _ = parse_tool_calls_with_healing(text, ALLOWED)
    assert parsed is None, f"未声明的工具名被透传: {parsed}"
    print("test_undeclared_tool_name_is_rejected: PASSED")


# ------------------------------------------------- 2. 零回归

def test_declared_tool_is_recognized():
    text = '```tool_call\n{"name": "Read", "arguments": {"file_path": "/etc/hosts"}}\n```'
    parsed, _ = parse_tool_calls_with_healing(text, ALLOWED)
    assert parsed is not None, "声明过的工具必须照常识别"
    _, tools = parsed
    assert tools == [("Read", {"file_path": "/etc/hosts"})], tools
    print("test_declared_tool_is_recognized: PASSED")


def test_openai_function_shape_is_recognized():
    text = (
        '```tool_call\n'
        '{"function": {"name": "Bash", "arguments": {"command": "ls"}}}\n'
        "```"
    )
    parsed, _ = parse_tool_calls_with_healing(text, ALLOWED)
    assert parsed is not None
    _, tools = parsed
    assert tools == [("Bash", {"command": "ls"})], tools
    print("test_openai_function_shape_is_recognized: PASSED")


def test_json_fence_is_still_a_valid_carrier():
    """真实模型常把调用写成 ```json，禁掉会误伤；防伪造靠校验而非格式。"""
    text = '```json\n{"name": "Bash", "arguments": {"command": "pwd"}}\n```'
    parsed, _ = parse_tool_calls_with_healing(text, ALLOWED)
    assert parsed is not None
    _, tools = parsed
    assert tools == [("Bash", {"command": "pwd"})], tools
    print("test_json_fence_is_still_a_valid_carrier: PASSED")


# ------------------------------------------------- 3. 流式锁存

def test_json_data_block_does_not_swallow_the_rest_of_the_stream():
    """**靶点 B**：修复前一个 ```json 数据块会把后面的回答全部吞掉。"""
    from providers.tool_streamer import SpeculativeToolStreamer

    streamer = SpeculativeToolStreamer(tools_enabled=True, allowed=ALLOWED)
    emitted = []

    def _emit(chunk):
        for a in streamer.feed_chunk(chunk):
            if a.action_type == "text_delta":
                emitted.append(a.content)

    _emit("配置如下：\n")
    _emit('```json\n{"name": "Alice", "age": 30}\n```\n')
    _emit("以上是配置。")
    _emit("后续还有更多说明文字。")

    rem, tools, _ = streamer.finalize()
    assert tools is None, f"普通数据块不得产出工具调用: {tools}"

    all_text = "".join(emitted) + rem
    assert "后续还有更多说明文字" in all_text, (
        f"误触发之后的文本被吞掉了（这正是「说到一半停住」）:\n{all_text!r}"
    )
    print("test_json_data_block_does_not_swallow_the_rest_of_the_stream: PASSED")


def test_false_trigger_unlatches_and_releases_buffer():
    from providers.tool_streamer import SpeculativeToolStreamer

    streamer = SpeculativeToolStreamer(tools_enabled=True, allowed=ALLOWED)
    actions1 = streamer.feed_chunk('```tool_call\n{"name": "NotAMyTool", "arguments": {}}\n```')
    # 锁存后必须能被证伪并放出
    released = "".join(a.content for a in actions1 if a.action_type == "text_delta")
    assert "NotAMyTool" in released, f"被证伪的缓冲内容必须放行而不是丢弃: {released!r}"
    assert streamer.tool_detected is False, "锁存必须已解除"

    after = streamer.feed_chunk("继续正常回答。")
    # 短于前瞻窗口的尾部会留到 finalize 送出，这是既有的推测式缓冲语义
    remainder, tools, _ = streamer.finalize()
    tail = "".join(a.content for a in after if a.action_type == "text_delta") + remainder
    assert "继续正常回答" in tail, f"解除锁存后文本必须继续流出，实际: {tail!r}"
    assert tools is None
    print("test_false_trigger_unlatches_and_releases_buffer: PASSED")


def test_real_tool_call_is_still_buffered_and_parsed():
    from providers.tool_streamer import SpeculativeToolStreamer

    streamer = SpeculativeToolStreamer(tools_enabled=True, allowed=ALLOWED)
    streamer.feed_chunk("我来读取：\n```tool_call\n")
    streamer.feed_chunk('{"name": "Read", "arguments": {"file_path": "/etc/hosts"}}\n```')
    rem, tools, _ = streamer.finalize()

    assert tools == [("Read", {"file_path": "/etc/hosts"})], tools
    print("test_real_tool_call_is_still_buffered_and_parsed: PASSED")


def run_all():
    tests = [
        test_plain_data_object_is_not_a_tool_call,
        test_python_fence_is_not_a_tool_channel,
        test_undeclared_tool_name_is_rejected,
        test_declared_tool_is_recognized,
        test_openai_function_shape_is_recognized,
        test_json_fence_is_still_a_valid_carrier,
        test_json_data_block_does_not_swallow_the_rest_of_the_stream,
        test_false_trigger_unlatches_and_releases_buffer,
        test_real_tool_call_is_still_buffered_and_parsed,
    ]
    for fn in tests:
        fn()
    print(f"\n✅ All tool-fabrication tests passed! ({len(tests)} cases)")


if __name__ == "__main__":
    run_all()