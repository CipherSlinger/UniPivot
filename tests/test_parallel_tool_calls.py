"""一个回复里的**两个**工具调用，背靠背写出时必须都被解析 (TOOL-LOOP)

动机来自真实 codex 0.145.0 的一次抓包（CLAUDE.md 第 68 节）。上游模型在一次
assistant 回合里连着输出两个工具块，这是 agent 的常态（并行调用、一次读两个文件），
而网关只把**第一个**回传给了客户端，第二个连同它的调用一起消失，没有任何报错。

根因不在协议层，而在所有三个协议共用的代码块正则里：

    _BLOCK_TOOL_PATTERN = r"```+\\s*(?:tool_call|...)\\s*\\n(.*?)(?:\\n?```+|$)"

收尾的 ```` ```+ ```` 是**贪婪**的：面对

    ```tool_call\\n{...}\\n``````tool_call\\n{...}\\n```

第一块的收尾把后面那串反引号整个吃掉（6 个），于是第二个块的开头被吞进第一块
的收尾里，`finditer` 从它之后继续扫描，第二个块再也不匹配。两个块之间只要有
一个换行/空格/正文，反引号就不相邻，缺陷不出现——这正是它长期没被发现的原因：
现有用例（`test_parallel_tool_calls_false_caps_to_one_everywhere` 等）都用
`\\n` 分隔两个块。

本套件钉住的是**观测到的真实形状**（零分隔），并在三个协议的真实请求路径上
断言「客户端真的收到两个调用」，而不是只看解析函数的返回值——修复若只活在
辅助函数里，单点测试照样绿。
"""

from __future__ import annotations

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
os.environ["DOUBAO_AUTH_TOKEN"] = "fake_doubao"

from fastapi.testclient import TestClient  # noqa: E402

import server as srv  # noqa: E402
from providers.anthropic_compat import parse_tool_calls  # noqa: E402
from providers.tool_streamer import (  # noqa: E402
    SpeculativeToolStreamer,
    parse_tool_calls_with_healing,
)

TOOLS_OPENAI = [
    {
        "type": "function",
        "function": {
            "name": "Read",
            "description": "Read a file",
            "parameters": {
                "type": "object",
                "properties": {"file_path": {"type": "string"}},
                "required": ["file_path"],
            },
        },
    }
]
TOOLS_ANTHROPIC = [
    {
        "name": "Read",
        "description": "Read a file",
        "input_schema": {
            "type": "object",
            "properties": {"file_path": {"type": "string"}},
            "required": ["file_path"],
        },
    }
]


def _block(path: str) -> str:
    return '```tool_call\n{"name": "Read", "arguments": {"file_path": "%s"}}\n```' % path


# 三种真实出现过的分隔形状。`""` 是 codex 抓包里观测到的那个；另外两个是既有
# 用例用的，必须继续工作（零回归的方向）。
SEPARATORS = {"back_to_back": "", "single_newline": "\n", "blank_line": "\n\n"}


class CapturingProvider:
    """按脚本产出增量，并记录上游实际收到的 messages。"""

    script: list = []
    last_messages: list = []

    def __init__(self, *a, **kw):
        pass

    async def chat(self, messages, model, **kwargs):
        CapturingProvider.last_messages = messages
        for delta, meta in CapturingProvider.script:
            yield delta, meta


_default_factory = srv._make_provider


def _run(script, path, payload):
    CapturingProvider.script = script
    CapturingProvider.last_messages = []
    srv._make_provider = lambda key, client=None: CapturingProvider()
    try:
        return TestClient(srv.app).post(path, json=payload).json()
    finally:
        srv._make_provider = _default_factory


def _chunks(sep: str):
    """上游按真实流式节奏交付：正文一块，两个工具块各一块。"""
    return [
        ("I'll read both.\n", {}),
        (_block("a"), {}),
        (_block("b"), {"conversation_id": "c1"}),
    ]


# ------------------------------------------------------------ 1. 解析层（纯函数）

def test_adjacent_blocks_parse_to_two_calls():
    """零分隔的两个工具块必须解析出两个调用。

    这是 codex 抓包里的原始字节形状。修复前 `parse_tool_calls` 与
    `parse_tool_calls_with_healing` 都只返回第一个调用。
    """
    for label, sep in SEPARATORS.items():
        text = _block("a") + sep + _block("b")
        got = parse_tool_calls_with_healing(text, {"Read"})
        assert got is not None, f"[{label}] 整段被判成没有工具调用"
        _, tools = got[0]
        assert tools == [("Read", {"file_path": "a"}), ("Read", {"file_path": "b"})], (
            f"[{label}] 分隔符 {sep!r} 下第二个工具调用丢失: {tools}"
        )
        legacy = parse_tool_calls(text)
        assert legacy is not None and len(legacy[1]) == 2, (
            f"[{label}] anthropic_compat.parse_tool_calls 只解析出 {legacy} —— "
            "两个解析器共用同一个正则，必须一起绿"
        )
    print("test_adjacent_blocks_parse_to_two_calls: PASSED")


def test_three_adjacent_blocks_parse_to_three_calls():
    """三个块连写：修复前是两个（中间那个被吞），不是三个。

    两个块时"只解析出一个"还可以被误读成"只该有一个"；三个块把规律钉死：
    被吞的永远是**紧挨着前一块收尾**的那一个，和位置无关。
    """
    text = _block("a") + _block("b") + _block("c")
    got = parse_tool_calls_with_healing(text, {"Read"})
    assert got is not None
    paths = [a["file_path"] for _, a in got[0][1]]
    assert paths == ["a", "b", "c"], f"三个连写的块只解析出 {paths}"
    print("test_three_adjacent_blocks_parse_to_three_calls: PASSED")


def test_fence_shapes_are_unchanged():
    """零回归：围栏长度、语言别名、未闭合、正文里的行内反引号都物归原处。

    修复只允许改变"相邻块"这一种情形。下面每一条在修复前后必须给出同一个答案，
    否则就是把一个缺陷换成另一个。
    """
    # 四个反引号的围栏
    four = '````tool_call\n{"name": "Read", "arguments": {"file_path": "a"}}\n````'
    assert len(parse_tool_calls(four)[1]) == 1, "四反引号围栏被解析坏了"

    # 语言别名 json / json5 / tool / function
    for lang in ("json", "json5", "tool", "function", "tool_call"):
        t = '```%s\n{"name": "Read", "arguments": {"file_path": "a"}}\n```' % lang
        got = parse_tool_calls(t)
        assert got is not None and len(got[1]) == 1, f"```{lang} 围栏不再被识别: {got}"

    # 未闭合（截断的流）：末块必须照样解析（自愈路径依赖它）
    truncated = '```tool_call\n{"name": "Read", "arguments": {"file_path": "a"}}'
    got = parse_tool_calls(truncated)
    assert got is not None and got[1][0][1] == {"file_path": "a"}, f"截断块丢了: {got}"

    # 正文里的行内三反引号不能把后续块吃掉
    with_prose = (_block("a") + "\n\nUse ```code``` inline.\n" + _block("b"))
    got = parse_tool_calls(with_prose)
    assert [a["file_path"] for _, a in got[1]] == ["a", "b"], f"行内反引号改变了结果: {got}"

    # 名字相近的语言标记不能被当成同一族（`tool_calling` 不是 `tool_call` 的延续）
    suffix = _block("a") + "\n```tool_calling\nx\n```"
    got = parse_tool_calls(suffix)
    assert [a["file_path"] for _, a in got[1]] == ["a"], f"未知语言标记被吞进来了: {got}"

    print("test_fence_shapes_are_unchanged: PASSED")


# -------------------------------------------------------------- 2. 流式状态机

def test_streamer_keeps_both_calls_when_delivered_chunk_by_chunk():
    """真实交付节奏：每个工具块是**一个** chunk，两块背靠背。

    这是本缺陷在线上唯一可见的形态。`finalize()` 对累积全文重新解析，所以
    解析层的缺陷会原样穿透到每一个协议——单元测试必须走同一段累积路径。
    """
    streamer = SpeculativeToolStreamer(True, {"Read"})
    for delta in ["I'll read both.\n", _block("a"), _block("b")]:
        streamer.feed_text(delta)
    _, tools, _ = streamer.finalize()
    assert tools == [("Read", {"file_path": "a"}), ("Read", {"file_path": "b"})], (
        f"流式累积后只解析出 {tools} —— 第二个工具调用被静默丢弃"
    )
    print("test_streamer_keeps_both_calls_when_delivered_chunk_by_chunk: PASSED")


# ------------------------------------------------------- 3. 三个协议的真实路径

def test_all_three_protocols_return_both_calls():
    """客户端真的收到**两个**调用，而不只是解析函数返回了两个。

    一个只修辅助函数的补丁会让上一条用例变绿，而线上仍然只发出一个调用：
    三个协议的收集点各自决定把什么写进响应。所以这里走完整 HTTP 路径。
    """
    for label, sep in SEPARATORS.items():
        script = _chunks(sep)

        d = _run(script, "/v1/chat/completions",
                 {"model": "gpt-4o", "messages": [{"role": "user", "content": "x"}],
                  "tools": TOOLS_OPENAI})
        calls = d["choices"][0]["message"].get("tool_calls") or []
        assert [c["function"]["arguments"] for c in calls] == [
            '{"file_path": "a"}', '{"file_path": "b"}'
        ], f"[{label}] OpenAI 只回传了 {len(calls)} 个调用: {calls}"

        d = _run(script, "/v1/messages",
                 {"model": "claude-3-7-sonnet", "max_tokens": 64,
                  "messages": [{"role": "user", "content": "x"}],
                  "tools": TOOLS_ANTHROPIC})
        uses = [b for b in d["content"] if b["type"] == "tool_use"]
        assert [u["input"]["file_path"] for u in uses] == ["a", "b"], (
            f"[{label}] Anthropic 只回传了 {len(uses)} 个 tool_use: {uses}"
        )
        assert len({u["id"] for u in uses}) == 2, f"[{label}] 两个 tool_use 共用了 id"

        d = _run(script, "/v1/responses",
                 {"model": "gpt-4o", "input": "x", "tools": TOOLS_OPENAI})
        fc = [i for i in d["output"] if i["type"] == "function_call"]
        assert [i["arguments"] for i in fc] == [
            '{"file_path": "a"}', '{"file_path": "b"}'
        ], f"[{label}] Responses 只回传了 {len(fc)} 个 function_call: {fc}"
        assert len({i["call_id"] for i in fc}) == 2, f"[{label}] 两个调用共用了 call_id"
    print("test_all_three_protocols_return_both_calls: PASSED")


def test_parallel_disallowed_still_caps_adjacent_blocks():
    """`parallel_tool_calls:false` 的裁剪在相邻形状下依旧生效。

    修复把"两个调用"变得可见，于是必须同时确认唯一的裁剪者仍然是客户端声明，
    而不是解析器的偶然行为：否则这个修复会顺手破坏 [TOOL-LOOP] 的并行禁用。
    """
    script = _chunks("")
    d = _run(script, "/v1/chat/completions",
             {"model": "gpt-4o", "messages": [{"role": "user", "content": "x"}],
              "tools": TOOLS_OPENAI, "parallel_tool_calls": False})
    calls = d["choices"][0]["message"].get("tool_calls") or []
    assert len(calls) == 1 and calls[0]["function"]["arguments"] == '{"file_path": "a"}', (
        f"parallel_tool_calls=false 没有裁到第一个: {calls}"
    )

    d = _run(script, "/v1/responses",
             {"model": "gpt-4o", "input": "x", "tools": TOOLS_OPENAI,
              "parallel_tool_calls": False})
    fc = [i for i in d["output"] if i["type"] == "function_call"]
    assert len(fc) == 1, f"Responses 的 parallel_tool_calls=false 没有裁剪: {fc}"
    print("test_parallel_disallowed_still_caps_adjacent_blocks: PASSED")


def run_all():
    tests = [
        test_adjacent_blocks_parse_to_two_calls,
        test_three_adjacent_blocks_parse_to_three_calls,
        test_fence_shapes_are_unchanged,
        test_streamer_keeps_both_calls_when_delivered_chunk_by_chunk,
        test_all_three_protocols_return_both_calls,
        test_parallel_disallowed_still_caps_adjacent_blocks,
    ]
    for fn in tests:
        fn()
    print(f"\n✅ All parallel-tool-call tests passed! ({len(tests)} cases)")


if __name__ == "__main__":
    run_all()