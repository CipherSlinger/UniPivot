"""工具参数里的代码围栏是**数据**，不是块分隔符 (TOOL-FENCE)

动机来自一次真实二进制复现：让 agent 写一个内含代码围栏的文件（README、配置、
任何文档），两个客户端都静默失败——

    claude: exit 0，文件存在，内容却是 'before fence\\n'（第一个围栏之后全丢）
    codex:  exit 0，apply_patch 的补丁被同样截断，文件根本没被创建

根因在三个协议共用的代码块正则里。它的收尾是「任意一个不被语言标记跟随的三反引号」，
对正文一无所知；当模型写出

    ```tool_call
    {"name": "Write", "arguments": {"content": "```bash\\necho hi\\n```"}}
    ```

收尾匹配到的是 JSON **字符串字面量内部**那对反引号（`` ```bash `` 的 `bash` 不在
语言别名表里，所以它被当成合法收尾），参数在第一个代码围栏处被截断，剩下的
`..."}}` 加真正的块收尾被当成正文丢弃。更糟的是自愈解析器会把截断后的 JSON 修好并
报告「已修复」，于是一份**残缺的文件内容**以成功的样子交到客户端手里。

这就是本仓库反复记录的那一类缺陷：块边界必须由 JSON 结构决定，不能由字节形状猜。
本套件在解析层（三个入口）、流式状态机、以及三个协议的真实 HTTP 路径上各钉一次，
因为只修辅助函数会让单点测试变绿而线上照旧。

运行: .venv/bin/python tests/test_tool_fence_json.py
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
            "name": "Write",
            "description": "Write a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["file_path", "content"],
            },
        },
    }
]
TOOLS_ANTHROPIC = [
    {
        "name": "Write",
        "description": "Write a file",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["file_path", "content"],
        },
    }
]

# 一份真实文档的形状：围栏语言别名（bash 会被旧正则当成收尾）、裸围栏、串内的
# ```` ```tool_call ```` 字面量、以及最后的尾标记。任何一处被当成块收尾，后面的
# 内容就消失。
CONTENT = (
    "# README\n\n"
    "Run this:\n\n"
    "```bash\n"
    "echo hi\n"
    "```\n\n"
    "Then check the call shape:\n\n"
    "```tool_call\n"
    '{"name": "Bash", "arguments": {"cmd": "echo hi"}}\n'
    "```\n\n"
    "```json\n"
    '{"a": 1, "b": [2, 3]}\n'
    "```\n\n"
    "TAIL_MARKER_9F3C\n"
)


def _block(content: str = CONTENT, path: str = "README.md") -> str:
    payload = {"name": "Write", "arguments": {"file_path": path, "content": content}}
    return "```tool_call\n" + json.dumps(payload, ensure_ascii=False) + "\n```"


# --------------------------------------------------------------- 1. 解析层

def test_a_fence_inside_the_arguments_is_not_a_delimiter():
    """三个解析入口都必须保住围栏之后的全部内容。

    修复前实测三个入口给出同一份截断结果：
        ('', [('Write', {'file_path': 'README.md', 'content': '# README\\n\\nRun this:\\n\\n'})])

    自愈入口还会报告 `was_healed=True`——它把截断后的 JSON 修好了，于是「修好了」
    掩盖了「内容已经丢了」。
    """
    text = _block()

    legacy = parse_tool_calls(text)
    assert legacy is not None, "整段被判成没有工具调用"
    name, args = legacy[1][0]
    assert name == "Write" and args["content"] == CONTENT, (
        f"anthropic_compat.parse_tool_calls 在第一个代码围栏处截断了参数: {args!r}"
    )

    healed = parse_tool_calls_with_healing(text, {"Write"})
    assert healed is not None
    args2 = healed[0][1][0][1]
    assert args2["content"] == CONTENT, (
        f"自愈解析入口同样截断了参数（并且报告 was_healed={healed[1]}）: {args2!r}"
    )

    # 控制：没有围栏的内容逐字不变——修复不能变成一个「永远延长到最后一个反引号」
    # 的启发式。
    plain = "# Title\n\nplain text only\n"
    args3 = parse_tool_calls(_block(plain))[1][0][1]
    assert args3["content"] == plain, f"普通内容被改坏了: {args3!r}"
    print("test_a_fence_inside_the_arguments_is_not_a_delimiter: PASSED")


def test_a_truncated_json_object_still_heals():
    """模型真的截断了 JSON（不是被围栏骗到）时，自愈路径必须照旧工作。

    这是本次修复最容易弄坏的方向：把「以 { 开头但还没闭合」一律当成「块还没结束」，
    会让所有截断输出的自愈失效。
    """
    trunc = '```tool_call\n{"name": "Write", "arguments": {"file_path": "a", "content": "abc\n```'
    got = parse_tool_calls_with_healing(trunc, {"Write"})
    assert got is not None and got[0][1][0][1]["content"] == "abc", (
        f"截断的 JSON 不再被自愈: {got}"
    )
    print("test_a_truncated_json_object_still_heals: PASSED")


def test_two_calls_where_the_first_contains_a_fence():
    """与第 68 节（背靠背两个调用）叠加：第一个调用内含围栏，两个都必须完整。

    修复若把「延长到 JSON 终点」写成「延长到最后一个反引号」，第二个调用会被并进
    第一个的参数里——把丢一个调用换成了另一种丢法。
    """
    text = _block(path="a.md") + _block(path="b.md")
    got = parse_tool_calls_with_healing(text, {"Write"})
    assert got is not None
    tools = got[0][1]
    assert [n for n, _ in tools] == ["Write", "Write"], f"只解析出 {tools}"
    assert [a["file_path"] for _, a in tools] == ["a.md", "b.md"], (
        f"两个调用的边界错了: {[a['file_path'] for _, a in tools]}"
    )
    for _, a in tools:
        assert a["content"] == CONTENT, "带围栏的正文被截断了"
    print("test_two_calls_where_the_first_contains_a_fence: PASSED")


# ------------------------------------------------------------ 2. 流式状态机

def test_streamer_preserves_the_body_when_fed_in_small_chunks():
    """真实交付节奏：块被切成任意大小的增量，`finalize()` 必须给出完整参数。

    实测（修复前）：`finalize()` 返回的 content 只有第一个围栏之前的部分。
    """
    text = _block()
    s = SpeculativeToolStreamer(True, {"Write"})
    for i in range(0, len(text), 13):
        s.feed_text(text[i:i + 13])
    _prefix, tools, _healed = s.finalize()
    assert tools and tools[0][1]["content"] == CONTENT, (
        f"流式累积后参数被截断: {tools!r}"
    )
    print("test_streamer_preserves_the_body_when_fed_in_small_chunks: PASSED")


def test_streamer_never_releases_raw_tool_json_at_a_nested_fence():
    """缓冲停在串内围栏上时不得判定「已闭合」，否则裸 JSON 被当正文吐给客户端。

    这一条在修复前后都必须为 False，它防的是「延长正文」这个新逻辑把闭合判定带偏。
    """
    text = _block()
    cut = text[: text.index("```bash") + len("```bash")]
    s = SpeculativeToolStreamer(True, {"Write"})
    for i in range(0, len(cut), 11):
        acts = s.feed_text(cut[i:i + 11])
        leaked = "".join(a.content for a in acts if a.action_type == "text_delta")
        assert '"name"' not in leaked, f"半个工具调用被当成正文放出去了: {leaked!r}"
    assert s._should_unlatch() is False, "缓冲停在 JSON 字符串内部却被判定为已闭合"
    print("test_streamer_never_releases_raw_tool_json_at_a_nested_fence: PASSED")


def test_prose_code_block_still_streams_as_text():
    """零回归：普通 ```json 代码块仍按正文透传，一个字符都不能少。

    `_TRIGGER_PATTERNS` 含 `json`，所以这种文本会先被锁存、随后被证伪放行——本次
    改动了「证伪」的判定，必须确认这条既有路径没被改成吞正文。
    """
    prose = 'Here is an example:\n```json\n{"a": 1, "b": [2,3]}\n```\nDone.'
    s = SpeculativeToolStreamer(True, {"Write"})
    emitted = []
    for i in range(0, len(prose), 6):
        emitted.extend(a.content for a in s.feed_text(prose[i:i + 6])
                       if a.action_type in ("text_delta", "flush_text"))
    tail, tools, _ = s.finalize()
    assert tools is None, f"普通代码块被当成了工具调用: {tools}"
    assert "".join(emitted) + tail == prose, (
        f"普通代码块的正文被改动: {(''.join(emitted) + tail)!r} != {prose!r}"
    )
    print("test_prose_code_block_still_streams_as_text: PASSED")


# ------------------------------------------------------- 3. 三个协议的真实路径

_GUARD_HITS: list[str] = []


def _guarded_factory(key):
    """绊线：没有安装脚本化 provider 的请求在这里失败，而不是真的触碰上游。"""
    _GUARD_HITS.append(key)
    raise AssertionError("测试发出了未安装假 provider 的上游请求")


srv._make_provider = _guarded_factory


class ScriptedProvider:
    script: list = []

    async def chat(self, messages, model, **kw):
        for delta, meta in ScriptedProvider.script:
            yield delta, meta


def _run(script, path, payload):
    ScriptedProvider.script = script
    srv._make_provider = lambda key, client=None: ScriptedProvider()
    try:
        return TestClient(srv.app).post(path, json=payload)
    finally:
        srv._make_provider = _guarded_factory


def test_all_three_protocols_deliver_the_untruncated_arguments():
    """客户端真的收到完整参数，而不只是解析函数返回了完整参数。"""
    script = [(_block(), {"conversation_id": "c1"})]

    r = _run(script, "/v1/chat/completions",
             {"model": "gpt-4o", "messages": [{"role": "user", "content": "write it"}],
              "tools": TOOLS_OPENAI})
    call = r.json()["choices"][0]["message"]["tool_calls"][0]
    assert json.loads(call["function"]["arguments"])["content"] == CONTENT, (
        f"OpenAI 路径上的参数被截断: {call['function']['arguments'][:120]!r}…"
    )

    r = _run(script, "/v1/messages",
             {"model": "claude-3-7-sonnet", "max_tokens": 256,
              "messages": [{"role": "user", "content": "write it"}],
              "tools": TOOLS_ANTHROPIC})
    use = [b for b in r.json()["content"] if b["type"] == "tool_use"][0]
    assert use["input"]["content"] == CONTENT, (
        f"Anthropic 路径上的参数被截断: {use['input']!r}"
    )

    r = _run(script, "/v1/responses",
             {"model": "gpt-4o", "input": "write it", "tools": TOOLS_OPENAI})
    item = [i for i in r.json()["output"] if i["type"] == "function_call"][0]
    assert json.loads(item["arguments"])["content"] == CONTENT, (
        f"Responses 路径上的参数被截断: {item['arguments'][:120]!r}…"
    )
    print("test_all_three_protocols_deliver_the_untruncated_arguments: PASSED")


def test_streaming_path_delivers_the_untruncated_arguments():
    """流式路径：客户端拼起来的 arguments 必须完整。

    非流式与流式走不同的收集点（一个用 `finalize`，一个逐 delta 累积 JSON），
    只测非流式会让「流式仍然截断」的修复通过。
    """
    r = _run([(_block(), {"conversation_id": "c1"})], "/v1/chat/completions",
             {"model": "gpt-4o", "stream": True,
              "messages": [{"role": "user", "content": "write it"}],
              "tools": TOOLS_OPENAI})
    assert r.status_code == 200, r.text
    args = ""
    for line in r.text.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        ev = json.loads(line[6:])
        for tc in (ev["choices"][0]["delta"].get("tool_calls") or []):
            args += tc.get("function", {}).get("arguments", "")
    assert args, f"流式响应里没有 tool_calls: {r.text[:300]!r}"
    assert json.loads(args)["content"] == CONTENT, (
        f"流式路径上的参数被截断: {args[:120]!r}…"
    )
    print("test_streaming_path_delivers_the_untruncated_arguments: PASSED")


def run_all():
    test_a_fence_inside_the_arguments_is_not_a_delimiter()
    test_a_truncated_json_object_still_heals()
    test_two_calls_where_the_first_contains_a_fence()
    test_streamer_preserves_the_body_when_fed_in_small_chunks()
    test_streamer_never_releases_raw_tool_json_at_a_nested_fence()
    test_prose_code_block_still_streams_as_text()
    test_all_three_protocols_deliver_the_untruncated_arguments()
    test_streaming_path_delivers_the_untruncated_arguments()
    assert not _GUARD_HITS, (
        f"有 {len(_GUARD_HITS)} 次上游调用未经脚本化 provider: {sorted(set(_GUARD_HITS))}"
    )
    print("\n✅ All tool-fence tests passed! (8 cases)")


if __name__ == "__main__":
    sys.exit(run_all())
