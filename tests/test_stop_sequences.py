"""客户端停用词与输出格式的端到端契约 (PROTOCOL)

动机：「工具调用没跑起来」的一部分不是工具本身的问题，而是**客户端声明的约束
只被解析、从未被应用**。本套件锁定本次修掉的三处：

  A. `stop` / `stop_sequences`：修复前三个协议都把它解析进模型然后完全忽略，
     上游吐出终止符之后的全部内容原样回给客户端（实测 `stop:["END"]` 时
     正文是 "Here you go.\nEND\nnever mind"）。停用词是硬语义，必须在网关侧
     截断，且流式路径要能处理**终止符被切在两个 chunk 之间**的情况。
  B. 终止符之后藏着工具调用：截断点对客户端不存在，那里解析出的
     ```tool_call 若被执行，就是一次客户端从未预期过的调用。
  C. 各协议自己的终止枚举：OpenAI 没有 `stop_sequence`（只能是 "stop"）、
     Anthropic 有且必须同时给出被命中的字符串、Responses 是
     status="incomplete" + incomplete_details.reason="stop"。
  D. `response_format` / `text.format` 必须真的进入上游提示词。

外加零回归：未声明 `stop` 时正文逐字不变（本次曾因扫描器未配置时返回空串，
把三个协议的流式正文全部吞成空串——正是这条用例要防住的形态）。

运行: .venv/bin/python tests/test_stop_sequences.py
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
os.environ["QWEN_AUTH_TOKEN"] = "fake_qwen"
os.environ["DEEPSEEK_AUTH_TOKEN"] = "fake_ds"
os.environ["DOUBAO_AUTH_TOKEN"] = "fake_doubao"

from fastapi.testclient import TestClient  # noqa: E402

import server as srv  # noqa: E402

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
    {"name": "Read", "description": "Read a file",
     "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}}}}
]
TOOLS_RESPONSES = [
    {"type": "function", "name": "Read",
     "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}}}}
]


class StopProvider:
    """按脚本产出增量，并记录上游实际收到的内容。"""

    script: list = []
    last_content: str = ""

    async def chat(self, messages, model, **kw):
        StopProvider.last_content = json.dumps(messages, ensure_ascii=False)
        for delta, meta in StopProvider.script:
            yield delta, meta


_GUARD_HITS: list[str] = []


def _guarded_factory(key):
    """绊线：任何**没有**先安装假 provider 的请求在这里立刻失败。

    这一条是被真实事故换来的：早先的用例直接调 `_client().stream(...)` 而忘了套
    `_run`，网关便带着真实的 session 凭据向上游发出了一次真实请求。测试套件绝不
    允许触碰真实供应商——把「忘记安装假 provider」变成一次断言失败，而不是一次
    对上游账号的触碰。
    """
    _GUARD_HITS.append(key)
    raise AssertionError(
        "测试发出了未安装假 provider 的上游请求（所有请求必须经 _run(script, fn)）"
    )


# 模块导入即安装绊线：此后只有 _run 内部会换成脚本化 provider
srv._make_provider = _guarded_factory


def _run(script, fn):
    StopProvider.script = script
    StopProvider.last_content = ""
    srv._make_provider = lambda key: StopProvider()
    try:
        return fn()
    finally:
        srv._make_provider = _guarded_factory


def _client():
    return TestClient(srv.app)


def _openai_body(**extra) -> dict:
    body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "go"}]}
    body.update(extra)
    return body


def _openai_sse(script, body: dict):
    """跑一次 OpenAI 流式请求，返回 (拼接正文, 末个带 finish_reason 的帧, 全部帧)。"""
    frames: list[dict] = []

    def go():
        with _client().stream(
            "POST", "/v1/chat/completions", json={**body, "stream": True}
        ) as r:
            assert r.status_code == 200, r.text
            for line in r.iter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    continue
                frames.append(json.loads(payload))

    _run(script, go)
    text = "".join(
        f["choices"][0]["delta"].get("content") or ""
        for f in frames
        if f.get("choices") and f["choices"][0].get("delta")
    )
    terminal = [
        f for f in frames if f.get("choices") and f["choices"][0].get("finish_reason")
    ][-1]
    return text, terminal, frames


def _anthropic_sse(script, body: dict) -> list[dict]:
    events: list[dict] = []

    def go():
        with _client().stream("POST", "/v1/messages", json={**body, "stream": True}) as r:
            assert r.status_code == 200, r.text
            for line in r.iter_lines():
                if line.startswith("data:"):
                    events.append(json.loads(line[5:].strip()))

    _run(script, go)
    return events


def _anthropic_text(events: list[dict]) -> str:
    return "".join(
        e["delta"]["text"]
        for e in events
        if e["type"] == "content_block_delta" and e["delta"].get("type") == "text_delta"
    )


def _responses_sse(script, body: dict) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []

    def go():
        with _client().stream("POST", "/v1/responses", json={**body, "stream": True}) as r:
            assert r.status_code == 200, r.text
            for line in r.iter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    continue
                obj = json.loads(payload)
                events.append((obj.get("type", ""), obj))

    _run(script, go)
    return events


def _responses_text(events: list[tuple[str, dict]]) -> str:
    return "".join(
        p["delta"] for n, p in events if n == "response.output_text.delta" and p.get("delta")
    )


# ------------------------------------------------ A. 三协议截断 + 终止枚举

SCRIPT_STOP = [("Here you go.\n", {}), ("END", {}), ("never mind", {})]


def test_openai_stop_truncates_and_reports_stop():
    text, terminal, _ = _openai_sse(SCRIPT_STOP, _openai_body(stop=["END"]))
    assert text == "Here you go.\n", text
    assert terminal["choices"][0]["finish_reason"] == "stop", terminal
    text_str, terminal_str, _ = _openai_sse(SCRIPT_STOP, _openai_body(stop="END"))
    assert text_str == "Here you go.\n", "字符串形态的 stop 也必须生效"
    assert terminal_str["choices"][0]["finish_reason"] == "stop"
    # 非流式
    data = _run(SCRIPT_STOP, lambda: _client().post(
        "/v1/chat/completions", json=_openai_body(stop=["END"])).json())
    assert data["choices"][0]["message"]["content"] == "Here you go.\n", data
    assert data["choices"][0]["finish_reason"] == "stop"
    print("test_openai_stop_truncates_and_reports_stop: PASSED")


def test_anthropic_stop_sequence_names_the_matched_string():
    body = {"model": "claude-3-7-sonnet", "max_tokens": 64,
            "messages": [{"role": "user", "content": "go"}], "stop_sequences": ["END"]}
    events = _anthropic_sse(SCRIPT_STOP, body)
    assert _anthropic_text(events) == "Here you go.\n", _anthropic_text(events)
    delta = [e for e in events if e["type"] == "message_delta"][-1]["delta"]
    assert delta["stop_reason"] == "stop_sequence", delta
    # 线上契约：stop_sequence 终止原因必须携带命中的字符串，不能是 null
    assert delta["stop_sequence"] == "END", delta

    data = _run(SCRIPT_STOP, lambda: _client().post("/v1/messages", json=body).json())
    assert data["content"][0]["text"] == "Here you go.\n", data
    assert data["stop_reason"] == "stop_sequence", data
    assert data["stop_sequence"] == "END", data
    print("test_anthropic_stop_sequence_names_the_matched_string: PASSED")


def test_responses_stop_marks_incomplete():
    body = {"model": "gpt-4o", "input": "go", "stop": ["END"]}
    events = _responses_sse(SCRIPT_STOP, body)
    assert _responses_text(events) == "Here you go.\n", _responses_text(events)
    completed = [p for n, p in events if n == "response.completed"][0]["response"]
    assert completed["status"] == "incomplete", completed.get("status")
    assert completed["incomplete_details"] == {"reason": "stop"}, completed.get("incomplete_details")

    data = _run(SCRIPT_STOP, lambda: _client().post("/v1/responses", json=body).json())
    assert data["status"] == "incomplete", data
    assert data["incomplete_details"] == {"reason": "stop"}, data
    text = [c["text"] for i in data["output"] if i["type"] == "message" for c in i["content"]]
    assert text == ["Here you go.\n"], text
    print("test_responses_stop_marks_incomplete: PASSED")


# ------------------------------------------------ B. 终止符被切在两个 chunk 之间


def test_terminator_split_across_chunks():
    """上游吐 "EN" 然后 "D"：已发出的字节收不回来，扫描器必须扣住尾巴。"""
    script = [("Here you go.\n", {}), ("EN", {}), ("D", {}), ("never mind", {})]
    text, terminal, _ = _openai_sse(script, _openai_body(stop=["END"]))
    assert text == "Here you go.\n", f"跨块终止符泄漏或被吞: {text!r}"
    assert terminal["choices"][0]["finish_reason"] == "stop"

    events = _anthropic_sse(script,
        {"model": "claude-3-7-sonnet", "max_tokens": 64,
         "messages": [{"role": "user", "content": "go"}], "stop_sequences": ["END"]})
    assert _anthropic_text(events) == "Here you go.\n", _anthropic_text(events)
    delta = [e for e in events if e["type"] == "message_delta"][-1]["delta"]
    assert delta["stop_sequence"] == "END", delta
    print("test_terminator_split_across_chunks: PASSED")


def test_unmatched_stop_does_not_eat_the_tail():
    """未命中时被扣住的尾巴必须原样交还，不能丢字。"""
    script = [("abcdef", {}), ("ghij", {})]
    text, _, _ = _openai_sse(script, _openai_body(stop="ZZZZ"))
    assert text == "abcdefghij", f"未命中停用词却丢了正文: {text!r}"
    print("test_unmatched_stop_does_not_eat_the_tail: PASSED")


# ------------------------------------------------ C. 截断点之后的工具调用不得执行


def test_tool_call_after_terminator_is_not_executed():
    script = [
        ("Here you go.\n", {}),
        ("END", {}),
        ('\n```tool_call\n{"name": "Read", "arguments": {"file_path": "/etc/passwd"}}\n```', {}),
    ]
    body = _openai_body(tools=TOOLS_OPENAI, stop=["END"])

    data = _run(script, lambda: _client().post("/v1/chat/completions", json=body).json())
    choice = data["choices"][0]
    assert not choice["message"].get("tool_calls"), (
        f"截断点之后的工具调用被执行: {choice['message']}"
    )
    assert choice["message"]["content"] == "Here you go.\n", choice["message"]

    text, terminal, _ = _openai_sse(script, body)
    assert terminal["choices"][0]["finish_reason"] == "stop", terminal
    assert text == "Here you go.\n", text

    anth = _run(script, lambda: _client().post("/v1/messages", json={
        "model": "claude-3-7-sonnet", "max_tokens": 64,
        "messages": [{"role": "user", "content": "go"}],
        "tools": TOOLS_ANTHROPIC, "stop_sequences": ["END"],
    }).json())
    assert anth["stop_reason"] == "stop_sequence", anth
    assert not [b for b in anth["content"] if b["type"] == "tool_use"], anth["content"]

    resp = _run(script, lambda: _client().post("/v1/responses", json={
        "model": "gpt-4o", "input": "go", "tools": TOOLS_RESPONSES, "stop": ["END"],
    }).json())
    assert not [i for i in resp["output"] if i["type"] == "function_call"], resp["output"]
    print("test_tool_call_after_terminator_is_not_executed: PASSED")


# ------------------------------------------------ D. response_format 进入提示词


def test_response_format_reaches_upstream_prompt():
    """客户端要求 JSON 输出时，上游必须被告知——此前该字段解析后被完全丢弃。"""
    for body, needle, label in [
        (_openai_body(response_format={"type": "json_object"}),
         "single valid JSON object", "OpenAI json_object"),
        (_openai_body(response_format={
            "type": "json_schema",
            "json_schema": {"name": "answer", "schema": {"type": "object"}},
        }), "answer", "OpenAI json_schema name"),
        ({"model": "gpt-4o", "input": "go",
          "text": {"format": {"type": "json_object"}}},
         "single valid JSON object", "Responses text.format"),
    ]:
        path = "/v1/responses" if "input" in body else "/v1/chat/completions"
        _run([("{}", {})], lambda: _client().post(path, json=body))
        assert needle in StopProvider.last_content, (
            f"{label}: response_format 没有进入上游提示词: {StopProvider.last_content[:400]}"
        )
    print("test_response_format_reaches_upstream_prompt: PASSED")


# ------------------------------------------------ 零回归


def test_no_stop_declared_passes_text_through():
    """没有声明停用词时正文必须逐字透传（扫描器未启用不等于吞掉输出）。"""
    script = [("你好，", {}), ("世界", {})]
    text, terminal, _ = _openai_sse(script, _openai_body())
    assert text == "你好，世界", f"未声明 stop 却吞掉了正文: {text!r}"
    assert terminal["choices"][0]["finish_reason"] == "stop"

    events = _anthropic_sse(script,
        {"model": "claude-3-7-sonnet", "max_tokens": 64,
         "messages": [{"role": "user", "content": "go"}]})
    assert _anthropic_text(events) == "你好，世界", _anthropic_text(events)
    delta = [e for e in events if e["type"] == "message_delta"][-1]["delta"]
    assert delta["stop_reason"] == "end_turn", delta
    assert delta["stop_sequence"] is None, delta
    print("test_no_stop_declared_passes_text_through: PASSED")


def run_all():
    tests = [
        test_openai_stop_truncates_and_reports_stop,
        test_anthropic_stop_sequence_names_the_matched_string,
        test_responses_stop_marks_incomplete,
        test_terminator_split_across_chunks,
        test_unmatched_stop_does_not_eat_the_tail,
        test_tool_call_after_terminator_is_not_executed,
        test_response_format_reaches_upstream_prompt,
        test_no_stop_declared_passes_text_through,
    ]
    for fn in tests:
        fn()
    # 故障转移会把绊线的断言包成 502，因此单看返回值不足以证明「没碰上游」。
    # 计数器是零上游调用的直接证据：任何一次命中都说明有用例绕过了 _run。
    assert not _GUARD_HITS, (
        f"有 {len(_GUARD_HITS)} 次上游调用未经脚本化 provider: {sorted(set(_GUARD_HITS))}"
    )
    print(f"\n✅ All stop-sequence / output-format tests passed! ({len(tests)} cases)")


if __name__ == "__main__":
    run_all()