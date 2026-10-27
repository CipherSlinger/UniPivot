"""`response_format` / `text.format` 的兑现 (PROTOCOL)

动机：`response_format` 是**硬的**输出契约。客户端拿到 `message.content`
（Responses 是 `output_text`）之后直接 `JSON.parse`。第 35 节只把它翻译成一条
提示词送进上游，解析端什么也没做——实测（真实 HTTP 路径，脚本化 provider 返回
`Sure! Here is the data you asked for:\\n\\n```json\\n{"name": "Ada", "age": 36}\\n```\\nHope it helps!`）：

    /v1/chat/completions -> 200，content 就是上面那段大白话
    /v1/responses        -> 200，output_text 同样是那段大白话

客户端一行之后崩掉，整轮报废，而且没有任何可读的成因。与 `tool_choice` / `stop` /
`parallel_tool_calls` 是同一条纪律：**提示词降低发生率，解析端保证结果**。

兑现分两步，顺序是刻意的：先把模型包在围栏/寒暄里的那个值**回收**出来（确定性，
不改变模型说了什么）；实在取不出合法 JSON 时**诚实**报错（502 `invalid_output`
带上模型原文），绝不把大白话伪装成结构化输出。

本套件同时锁定两条边界：

  * **不补截断**：`repair_truncated_json` 会给缺失字段补 `null`，那是替模型编数据。
    被截断的答案必须表现为失败，而不是一个看起来完整的对象（第 52 节）。
  * **零回归**：未声明、`{"type": "text"}`、以及任何不认识的形状必须逐字节不变。

并且锁定「提示词与解析端读同一个顶层类型」：`json_schema` 声明 `type: "array"`
时，提示词必须说 array（实测修复前它说的是 "a single valid JSON object"，与紧跟
其后的 `Schema:` 自相矛盾），解析端也必须能从散文里取出一个顶层数组（实测修复前
取不到，把一个**正确**的回答判成违约）。

运行: .venv/bin/python tests/test_response_format.py
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
os.environ["GATEWAY_DISABLE_PROMPT_FOLDING"] = "1"
os.environ["GATEWAY_DISABLE_CONVERSATION_GC"] = "1"
os.environ["GATEWAY_DISABLE_RISK_LEDGER"] = "1"
os.environ["GATEWAY_DISABLE_SINGLEFLIGHT"] = "1"
os.environ["GATEWAY_DISABLE_ADAPTIVE_PACING"] = "1"
os.environ["QWEN_AUTH_TOKEN"] = "fake_qwen"
os.environ["DEEPSEEK_AUTH_TOKEN"] = "fake_ds"
os.environ["DOUBAO_AUTH_TOKEN"] = "fake_doubao"

from fastapi.testclient import TestClient  # noqa: E402

import server as srv  # noqa: E402

#: 模型对「给我 JSON」最常见的回答形态：一段寒暄 + 一个 json 围栏 + 一段安慰。
PROSE_WITH_FENCE = (
    'Sure! Here is the data you asked for:\n\n'
    '```json\n{"name": "Ada", "age": 36}\n```\nHope this helps!'
)
#: 完全取不出 JSON 的回答。
PROSE_ONLY = "I'm sorry, I can't produce JSON for that."
#: 严格合法、但排版不规整（用来证明「原样交付」而不是「重新序列化」）。
STRICT_ODD_SPACING = '{"b": 2,  "a": 1}'

JSON_OBJECT = {"type": "json_object"}

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


class ScriptedProvider:
    """按脚本产出一个正文块，并记录上游**实际收到**的提示词。"""

    answer: str = ""
    last_prompt: str = ""

    async def chat(self, messages, model, **kw):
        # 只在首轮（无上游会话）记录：续聊时 provider 只会拿到增量，那不是本套件
        # 要断言的「模型看到的契约」。
        if not kw.get("conversation_id"):
            ScriptedProvider.last_prompt = json.dumps(messages, ensure_ascii=False)
        yield ScriptedProvider.answer, {}
        yield "", {"conversation_id": "rf-conv"}

    async def delete_conversation(self, *args, **kwargs):
        return True


_GUARD_HITS: list[str] = []


def _guarded_factory(*args, **kwargs):
    """绊线：任何**没有**先安装假 provider 的请求在这里立刻失败。

    这条纪律是被真实事故换来的：早先有一份用例忘了套 `_run`，网关便带着真实的
    session 凭据向上游发出了一次真实请求。把「忘记安装假 provider」变成一次断言
    失败，而不是一次对上游账号的触碰。
    """
    _GUARD_HITS.append(str(args[0]) if args else "?")
    raise AssertionError(
        "测试发出了未安装假 provider 的上游请求（所有请求必须经 _run(answer, fn)）"
    )


# 模块导入即安装绊线：此后只有 _run 内部会换成脚本化 provider
srv._make_provider = _guarded_factory


def _run(answer, fn):
    ScriptedProvider.answer = answer
    ScriptedProvider.last_prompt = ""
    srv._make_provider = lambda *a, **kw: ScriptedProvider()
    try:
        return fn()
    finally:
        srv._make_provider = _guarded_factory


def _client():
    return TestClient(srv.app)


# ------------------------------------------------------------- 请求/响应工具


def _chat(answer, **extra):
    body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "give me ada"}]}
    body.update(extra)

    def go():
        with _client().stream("POST", "/v1/chat/completions", json=body) as r:
            return r.status_code, r.read().decode()

    return _run(answer, go)


def _chat_frames(answer, **extra):
    body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "give me ada"}]}
    body.update(dict(extra, stream=True))

    def go():
        with _client().stream("POST", "/v1/chat/completions", json=body) as r:
            raw = r.read().decode()
            return r.status_code, raw

    status, raw = _run(answer, go)
    frames: list = []
    for line in raw.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            frames.append("[DONE]")
            continue
        try:
            frames.append(json.loads(payload))
        except ValueError:
            pass
    return status, frames


def _responses(answer, **body):
    def go():
        with _client().stream("POST", "/v1/responses", json=body) as r:
            return r.status_code, r.read().decode()

    return _run(answer, go)


def _responses_events(answer, **body):
    status, raw = _responses(answer, **dict(body, stream=True))
    events: list = []
    for line in raw.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            events.append("[DONE]")
            continue
        try:
            events.append(json.loads(payload))
        except ValueError:
            pass
    return status, events


def _event_names(events) -> list[str]:
    return [e.get("type", "") for e in events if isinstance(e, dict)]


def _content_deltas(frames) -> list[str]:
    out: list[str] = []
    for f in frames:
        if not isinstance(f, dict):
            continue
        choices = f.get("choices") or []
        if not choices:
            continue
        content = (choices[0].get("delta") or {}).get("content")
        if content:
            out.append(content)
    return out


# ------------------------------------------------- A. 回收：取出发给客户端的值


def test_openai_non_stream_recovers_a_fenced_payload():
    """核心用例：寒暄 + 围栏 + 寒暄，客户端拿到的 `content` 必须能直接 JSON.parse。

    修复前实测：200，content 就是 `PROSE_WITH_FENCE` 原文。
    """
    status, raw = _chat(PROSE_WITH_FENCE, response_format=JSON_OBJECT)
    assert status == 200, raw
    content = json.loads(raw)["choices"][0]["message"]["content"]
    parsed = json.loads(content)  # 客户端的下一步就是这一行
    assert parsed == {"name": "Ada", "age": 36}, (
        f"客户端拿到的 content 不是可解析的 JSON 对象: {content!r}"
    )


def test_openai_stream_delivers_exactly_one_parseable_content_chunk():
    """流式路径：正文必须**整体**缓冲后交付，不能把半截 JSON 先发出去。

    缓冲是刻意的——在保证合法 JSON 之前发出任何一个字节，都等于让客户端对着半个
    对象解析。因此断言「恰好一个非空正文增量」而不是「拼起来能解析」。
    """
    status, frames = _chat_frames(PROSE_WITH_FENCE, response_format=JSON_OBJECT)
    assert status == 200, frames
    deltas = _content_deltas(frames)
    assert len(deltas) == 1, f"JSON 模式下正文被拆成了多段增量: {deltas!r}"
    assert json.loads(deltas[0]) == {"name": "Ada", "age": 36}, deltas
    assert "[DONE]" in frames, "正常结束必须发 [DONE]"


def test_openai_stream_with_tools_still_enforces_json():
    """声明了 tools 的流式分支是另一条代码路径，同样必须兑现（实测修复前它原样吐出寒暄）。

    这条用例的意义在于：工具的投机流式与正文缓冲是两套状态机，只修一条等于没修。
    """
    status, frames = _chat_frames(
        PROSE_WITH_FENCE, response_format=JSON_OBJECT, tools=TOOLS_OPENAI
    )
    assert status == 200, frames
    deltas = [d for d in _content_deltas(frames) if d.strip()]
    assert len(deltas) == 1, f"带 tools 的 JSON 模式正文增量数量不对: {deltas!r}"
    assert json.loads(deltas[0]) == {"name": "Ada", "age": 36}, deltas


def test_responses_non_stream_text_format_is_enforced():
    """`text.format` 与 `response_format` 是同一个契约的两种拼写。"""
    status, raw = _responses(
        PROSE_WITH_FENCE,
        model="gpt-4o",
        input="give me ada",
        text={"format": {"type": "json_object"}},
    )
    assert status == 200, raw
    body = json.loads(raw)
    text = "".join(
        part.get("text", "")
        for item in body.get("output", [])
        for part in item.get("content", [])
        if part.get("type") == "output_text"
    )
    assert json.loads(text) == {"name": "Ada", "age": 36}, text


def test_responses_stream_text_format_is_enforced():
    status, events = _responses_events(
        PROSE_WITH_FENCE,
        model="gpt-4o",
        input="give me ada",
        text={"format": {"type": "json_object"}},
    )
    assert status == 200, events
    deltas = [
        e["delta"]
        for e in events
        if isinstance(e, dict) and e.get("type") == "response.output_text.delta"
    ]
    assert len(deltas) == 1, f"Responses JSON 模式的正文增量数量不对: {deltas!r}"
    assert json.loads(deltas[0]) == {"name": "Ada", "age": 36}, deltas
    assert _event_names(events)[-1] == "response.completed", _event_names(events)
    assert "[DONE]" in events, "协议结束帧不能省"


def test_strict_payload_is_delivered_byte_for_byte():
    """模型已经给对的字节不许改写：只回收，不重排、不重新序列化。"""
    status, raw = _chat(STRICT_ODD_SPACING, response_format=JSON_OBJECT)
    assert status == 200, raw
    content = json.loads(raw)["choices"][0]["message"]["content"]
    assert content == STRICT_ODD_SPACING, (
        f"严格合法的 JSON 被改写了（客户端可能按原始字节做签名/比对）: {content!r}"
    )


# ------------------------------------------- B. 诚实：取不出来就要报错，不伪装


def test_unrecoverable_prose_is_a_502_carrying_the_models_own_words():
    """客户端声明了它要 JSON.parse——递过去一段大白话等于让它一行之后崩掉。"""
    status, raw = _chat(PROSE_ONLY, response_format=JSON_OBJECT)
    assert status == 502, f"模型没兑现契约时网关必须报错，而不是 200 递大白话: {raw}"
    err = json.loads(raw)["error"]
    assert err["type"] == "invalid_output", err
    assert "I'm sorry" in err["message"], (
        f"错误文案没有带出模型原文，运维无从判断成因: {err['message']!r}"
    )


def test_failed_stream_emits_an_in_band_error_and_never_done():
    """流已经开启，状态行改不了：只能是「错误帧 + 永不 `[DONE]`」。

    发了 `[DONE]` 就等于告诉客户端「这是一段完整答案」，而客户端会拿着它去
    JSON.parse 并崩溃。
    """
    status, frames = _chat_frames(PROSE_ONLY, response_format=JSON_OBJECT)
    assert status == 200, frames  # 上游流已开启，状态行无法再改
    assert "[DONE]" not in frames, "没有兑现契约的流绝不能以 [DONE] 冒充正常结束"
    errors = [f for f in frames if isinstance(f, dict) and "error" in f]
    assert errors, f"流式失败没有给出可读的错误帧: {frames!r}"
    assert errors[-1]["error"]["type"] == "invalid_output", errors[-1]


def test_failed_responses_stream_ends_with_response_failed():
    """Codex 0.145.0 只认 `response.*` 事件；裸 `event: error` 会被整条丢弃（第 57 节）。"""
    status, events = _responses_events(
        PROSE_ONLY,
        model="gpt-4o",
        input="give me ada",
        text={"format": {"type": "json_object"}},
    )
    assert status == 200, events
    names = _event_names(events)
    assert names[-1] == "response.failed", f"最后一个事件必须是 response.failed: {names}"
    assert "[DONE]" not in events, "失败的流绝不能看起来像干净结束"


def test_truncated_json_is_not_repaired_into_a_fabricated_object():
    """截断补全会给缺失字段填 `null`——那是替模型编数据。

    如果这里放行，客户端会收到 `{"name": "Ada", "age": null}` 并把它当成模型
    真的给出的答案（第 52 节的「截断不得冒充完整」）。
    """
    status, raw = _chat('{"name": "Ada", "age":', response_format=JSON_OBJECT)
    assert status == 502, (
        f"被截断的 JSON 被补成了看起来完整的对象，客户端会把编造的 null 当成答案: {raw}"
    )
    err = json.loads(raw)["error"]
    assert err["type"] == "invalid_output", err
    assert '"age":' in err["message"], f"没有保留模型被截断的原文: {err['message']!r}"
    assert "null" not in err["message"], (
        f"截断补全给缺失字段编了一个 null（这正是不能开截断修复的原因）: {err['message']!r}"
    )


# ------------------------------ C. 提示词与解析端必须读同一个顶层类型


SCHEMA_ARRAY = {
    "type": "json_schema",
    "json_schema": {
        "name": "items",
        "schema": {"type": "array", "items": {"type": "integer"}},
    },
}


def test_json_schema_array_is_declared_as_an_array_to_the_model():
    """提示词是这些网页端上游**唯一**看得到的契约。

    修复前实测：`type: "array"` 的请求，提示词写的是 "a single valid JSON object"，
    紧跟着的 `Schema:` 又写着 `{"type":"array"}`——自相矛盾，模型必然违约。
    """
    status, raw = _chat(
        'Here:\n```json\n[1, 2, 3]\n```', response_format=SCHEMA_ARRAY
    )
    prompt = ScriptedProvider.last_prompt
    assert "## Output Format" in prompt, f"结构化输出约束没有进入上游提示词: {prompt}"
    assert "JSON array" in prompt, (
        f"声明了 array 的 schema，提示词却没有说 array: {prompt}"
    )
    assert "JSON object" not in prompt, (
        f"声明了 array 的 schema，提示词却让模型输出 object（自相矛盾）: {prompt}"
    )
    assert status == 200, raw
    content = json.loads(raw)["choices"][0]["message"]["content"]
    assert json.loads(content) == [1, 2, 3], (
        f"顶层数组没有被回收出来（修复前实测 payload is None，正确答案被判违约）: {content!r}"
    )


def test_json_schema_array_rejects_an_object():
    """声明了 array 就必须是 array：顶层类型校验不能形同虚设。"""
    status, raw = _chat('{"a": 1}', response_format=SCHEMA_ARRAY)
    assert status == 502, f"schema 声明 array 却接受了 object: {raw}"
    assert json.loads(raw)["error"]["type"] == "invalid_output", raw


# ------------------------------------------------------ D. 零回归（失败开放）


def test_absent_or_text_format_passes_prose_through_unchanged():
    """没有声明结构化输出时，一个字节都不许动。

    这一条与第 35 节「扫描器未配置时返回空串，把三个协议的正文全部吞掉」是同一
    类事故的防线：失败开放必须用**正文内容**证明，而不是用「没报错」证明。
    """
    for extra in ({}, {"response_format": {"type": "text"}}, {"response_format": {}}):
        status, raw = _chat(PROSE_ONLY, **extra)
        assert status == 200, (extra, raw)
        content = json.loads(raw)["choices"][0]["message"]["content"]
        assert content == PROSE_ONLY, f"{extra}: 未声明 JSON 的请求被改写了: {content!r}"

    status, events = _responses_events(
        PROSE_ONLY, model="gpt-4o", input="go", text={"format": {"type": "text"}}
    )
    assert status == 200, events
    text = "".join(
        e["delta"]
        for e in events
        if isinstance(e, dict) and e.get("type") == "response.output_text.delta"
    )
    assert text == PROSE_ONLY, f"text.format=text 的请求被改写了: {text!r}"


def run_all():
    tests = [
        test_openai_non_stream_recovers_a_fenced_payload,
        test_openai_stream_delivers_exactly_one_parseable_content_chunk,
        test_openai_stream_with_tools_still_enforces_json,
        test_responses_non_stream_text_format_is_enforced,
        test_responses_stream_text_format_is_enforced,
        test_strict_payload_is_delivered_byte_for_byte,
        test_unrecoverable_prose_is_a_502_carrying_the_models_own_words,
        test_failed_stream_emits_an_in_band_error_and_never_done,
        test_failed_responses_stream_ends_with_response_failed,
        test_truncated_json_is_not_repaired_into_a_fabricated_object,
        test_json_schema_array_is_declared_as_an_array_to_the_model,
        test_json_schema_array_rejects_an_object,
        test_absent_or_text_format_passes_prose_through_unchanged,
    ]
    for fn in tests:
        fn()
        print(f"{fn.__name__}: PASSED")
    # 故障转移会把绊线的断言包成 502，因此单看返回值不足以证明「没碰上游」。
    # 计数器是零上游调用的直接证据：任何一次命中都说明有用例绕过了 _run。
    assert not _GUARD_HITS, (
        f"有 {len(_GUARD_HITS)} 次上游调用未经脚本化 provider: {sorted(set(_GUARD_HITS))}"
    )
    print(f"\n✅ All response_format tests passed! ({len(tests)} cases)")


if __name__ == "__main__":
    run_all()
