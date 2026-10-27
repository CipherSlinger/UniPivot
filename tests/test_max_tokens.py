"""客户端声明的输出 token 上界被解析、被传递、然后**没有任何人执行** (PROTOCOL)

## 实测：声明了 5 个 token 的上限，拿回 2000 个字符，还被告知「说完了」

真实 ASGI 路径（脚本化 provider 一次吐出 `"word " * 400`，`TESTING=1`，零真实
上游调用）：

    chat  max_tokens=5         -> 200 content_len=2000 finish_reason=stop
    anth  max_tokens=5         -> 200 text_len=2000    stop_reason=end_turn
    resp  max_output_tokens=5  -> 200 output_text_len=2000
                                  status=completed incomplete_details=None

`max_tokens` / `max_output_tokens` 一路被接受：pydantic 模型声明了它，`server.py`
把它算进 `chat_args`，五个 provider 的 `chat()` 签名都收下它——然后它不出现在任何
上游请求体里（`providers/qwen/provider.py` 的 `payload` 里既没有 `temperature` 也
没有 `max_tokens`），网关自己也不执行它。**解析了、然后没人用**，本项目记录得最多
的那一类缺陷。

## 为什么这是产品问题，而不是少一个可选特性

`max_tokens` 在 Anthropic 协议里是**必填字段**，所以 Claude Code 的**每一次**请求都
带着它——它用这个数字做上下文预算。网关无视它带来两种后果：

1. **成本失控**：客户端用 5 个 token 预算做一次短分类，拿回两千字符；
2. **无法判断「这轮结束了没有」**：真实语义是「被上限截断」，客户端据此决定重试、
   分段或压缩上下文。网关报的是 `stop` / `end_turn` / `completed`——**它宣称说完了**。
   这正是第 52 节那条规则的反面：被截断的答案不许冒充完整的答案，只不过这次的截断
   点是客户端自己声明的。

## 修法的三条纪律

* **只在客户端显式声明时执行**。Anthropic 的 `max_tokens: int = 4096` 有默认值，
  所以「没声明」和「声明了 4096」在字段值上无法区分——用 `model_fields_set` 判。
  执行一个隐式默认值等于截断一个从来没有人限定过的回答。
* **在网关侧执行，不写进提示词**。和 `stop` / `tool_choice` / `response_format`
  同一条纪律：提示词降低发生率，边界保证结果。模型数不清 token，写进提示词只会
  增加噪声而不会兑现契约。
* **截断必须发生在工具解析之前**（第 35 节）。截断点之后对客户端不存在，若那里
  藏着一个 ```` ```tool_call ````，把它回传出去就是一次客户端从未预期的执行。

## 终止信号按各自的协议如实报告

    OpenAI    finish_reason = "length"
    Anthropic stop_reason   = "max_tokens"
    Responses status="incomplete" + incomplete_details.reason="max_output_tokens"

三条都是各自协议规范里真实存在的枚举值，不是网关发明的（第 35 节踩过同一个坑：
OpenAI 的 `finish_reason` 里没有 `stop_sequence`）。

## 零回归

用例 7/8 是**守卫**，在修复前的树上也是绿的：没有声明上界的请求永远不被截断
（包括 Anthropic 省略 `max_tokens` 时那个 4096 的默认值），声明了一个大于回答的
上界也永远不截断。它们在自己的 docstring 里标明是守卫，不冒充 RED-first 的缺陷
复现（§59/§69/§79 的纪律）。

运行: .venv/bin/python tests/test_max_tokens.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["TESTING"] = "1"
for _k in (
    "GATEWAY_DISABLE_CREDENTIAL_WATCH",
    "GATEWAY_DISABLE_AUTO_REFRESH",
    "GATEWAY_DISABLE_PROMPT_FOLDING",
    "GATEWAY_DISABLE_CONVERSATION_GC",
    "GATEWAY_DISABLE_ADAPTIVE_PACING",
    "GATEWAY_DISABLE_PROBING",
    "GATEWAY_DISABLE_BAN_INFERENCE",
    "GATEWAY_DISABLE_STREAM_COMPLETION_CHECK",
    "GATEWAY_DISABLE_LONG_AVOIDANCE_FAST_FAIL",
    "GATEWAY_DISABLE_RISK_LEDGER",
    "GATEWAY_DISABLE_SINGLEFLIGHT",
):
    os.environ[_k] = "1"
for _k in ("QWEN", "DEEPSEEK", "DOUBAO", "KIMI", "GLM"):
    os.environ[f"{_k}_AUTH_TOKEN"] = "fake"

from fastapi.testclient import TestClient  # noqa: E402

import server as srv  # noqa: E402
from providers.base import _est_tokens  # noqa: E402

#: 上游一次吐出的回答：2000 字符，约 571 个 token。
LONG = "word " * 400
#: 超过 Anthropic `max_tokens` 默认值 4096 的回答：约 5714 token。
HUGE = "word " * 4000

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


srv._make_provider = _guarded_factory


class _Provider:
    """按脚本产出增量；脚本是**分块**的，好让截断落在块中间。"""

    script: list = []

    async def chat(self, messages, model, **kw):
        for delta, meta in _Provider.script:
            yield delta, meta

    async def delete_conversation(self, *args, **kwargs):
        return True


def _run(answer: str, fn, chunk: int = 7):
    _Provider.script = [
        (answer[i:i + chunk], {}) for i in range(0, len(answer), chunk)
    ] + [("", {"conversation_id": "mt-conv"})]
    srv._make_provider = lambda *a, **kw: _Provider()
    try:
        return fn()
    finally:
        srv._make_provider = _guarded_factory


def _client():
    return TestClient(srv.app)


# ------------------------------------------------------------- 请求/响应工具


def _chat_non_stream(answer: str, budget) -> dict:
    body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "go"}]}
    if budget is not None:
        body["max_tokens"] = budget
    return _run(answer, lambda: _client().post(
        "/v1/chat/completions", json=body).json())


def _chat_stream(answer: str, budget) -> tuple[str, dict]:
    body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "go"}],
            "stream": True}
    if budget is not None:
        body["max_tokens"] = budget
    frames: list[dict] = []

    def go():
        with _client().stream("POST", "/v1/chat/completions", json=body) as r:
            assert r.status_code == 200, r.text
            for line in r.iter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    continue
                frames.append(json.loads(payload))

    _run(answer, go)
    text = "".join(
        f["choices"][0]["delta"].get("content") or ""
        for f in frames
        if f.get("choices") and f["choices"][0].get("delta")
    )
    terminal = [
        f for f in frames if f.get("choices") and f["choices"][0].get("finish_reason")
    ][-1]
    return text, terminal


def _anth_body(budget) -> dict:
    body = {"model": "claude-3-5-sonnet", "messages": [{"role": "user", "content": "go"}]}
    if budget is not None:
        body["max_tokens"] = budget
    return body


def _anth_non_stream(answer: str, budget) -> dict:
    return _run(answer, lambda: _client().post(
        "/v1/messages", json=_anth_body(budget)).json())


def _anth_stream(answer: str, budget) -> tuple[str, dict]:
    events: list[dict] = []

    def go():
        with _client().stream(
            "POST", "/v1/messages", json={**_anth_body(budget), "stream": True}
        ) as r:
            assert r.status_code == 200, r.text
            for line in r.iter_lines():
                if line.startswith("data:"):
                    events.append(json.loads(line[5:].strip()))

    _run(answer, go)
    text = "".join(
        e["delta"]["text"]
        for e in events
        if e["type"] == "content_block_delta" and e["delta"].get("type") == "text_delta"
    )
    delta = [e for e in events if e["type"] == "message_delta"][-1]["delta"]
    return text, delta


def _resp_body(budget) -> dict:
    body = {"model": "gpt-4o", "input": [{"role": "user", "content": "go"}]}
    if budget is not None:
        body["max_output_tokens"] = budget
    return body


def _resp_non_stream(answer: str, budget) -> dict:
    return _run(answer, lambda: _client().post(
        "/v1/responses", json=_resp_body(budget)).json())


def _resp_stream(answer: str, budget) -> tuple[str, dict]:
    events: list[dict] = []

    def go():
        with _client().stream(
            "POST", "/v1/responses", json={**_resp_body(budget), "stream": True}
        ) as r:
            assert r.status_code == 200, r.text
            for line in r.iter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    continue
                events.append(json.loads(payload))

    _run(answer, go)
    text = "".join(
        e["delta"]
        for e in events
        if e.get("type") == "response.output_text.delta" and e.get("delta")
    )
    done = [e for e in events if e.get("type") == "response.completed"][-1]["response"]
    return text, done


def _resp_text(body: dict) -> str:
    return "".join(
        part.get("text", "")
        for item in body.get("output", [])
        for part in item.get("content", [])
        if part.get("type") == "output_text"
    )


# ------------------------------------------------ A. 三个协议都必须执行上界


def test_chat_non_streaming_honours_max_tokens_and_reports_length():
    """实测修复前：`max_tokens=5` 得到 2000 字符，`finish_reason="stop"`。

    `finish_reason` 是客户端唯一的「这轮为什么结束」信号：`stop` 说「说完了」，
    而真实情况是客户端自己划的那条线到了。
    """
    body = _chat_non_stream(LONG, 5)
    content = body["choices"][0]["message"]["content"]
    assert _est_tokens(content) <= 5, (
        f"客户端声明了 5 个 token 的上界，网关却回了 {len(content)} 字符"
        f"（约 {_est_tokens(content)} token）：{content[:60]!r}…"
    )
    assert body["choices"][0]["finish_reason"] == "length", (
        f"被客户端预算截断的答案被报成了「说完了」: "
        f"{body['choices'][0]['finish_reason']!r}"
    )
    print("test_chat_non_streaming_honours_max_tokens_and_reports_length: PASSED")


def test_chat_streaming_honours_max_tokens_and_reports_length():
    """流式路径同一个契约：已发出的字节收不回来，所以更必须在发之前判定。"""
    text, terminal = _chat_stream(LONG, 5)
    assert _est_tokens(text) <= 5, (
        f"流式正文突破了客户端声明的上界：{len(text)} 字符（约 "
        f"{_est_tokens(text)} token）"
    )
    assert terminal["choices"][0]["finish_reason"] == "length", (
        f"流式终局的 finish_reason 是 {terminal['choices'][0]['finish_reason']!r}，"
        f"应当如实报告被预算截断"
    )
    print("test_chat_streaming_honours_max_tokens_and_reports_length: PASSED")


def test_anthropic_non_streaming_reports_max_tokens_stop_reason():
    """实测修复前：`stop_reason="end_turn"`，正文 2000 字符。

    Anthropic 的枚举里**有** `max_tokens` 这个值，它存在的理由就是让客户端能区分
    「模型说完了」和「预算用完了」。
    """
    body = _anth_non_stream(LONG, 5)
    text = "".join(b.get("text", "") for b in body.get("content", []))
    assert _est_tokens(text) <= 5, (
        f"Anthropic 路径无视了 max_tokens=5：拿到 {len(text)} 字符"
    )
    assert body["stop_reason"] == "max_tokens", (
        f"stop_reason 是 {body['stop_reason']!r}，应当是 max_tokens"
    )
    assert body.get("stop_sequence") is None, (
        "被预算截断不是停用词命中，stop_sequence 必须是 None"
    )
    print("test_anthropic_non_streaming_reports_max_tokens_stop_reason: PASSED")


def test_anthropic_streaming_reports_max_tokens_stop_reason():
    text, delta = _anth_stream(LONG, 5)
    assert _est_tokens(text) <= 5, (
        f"Anthropic 流式正文突破了上界：{len(text)} 字符"
    )
    assert delta["stop_reason"] == "max_tokens", (
        f"message_delta 的 stop_reason 是 {delta['stop_reason']!r}"
    )
    assert delta.get("stop_sequence") is None, delta
    print("test_anthropic_streaming_reports_max_tokens_stop_reason: PASSED")


def test_responses_non_streaming_reports_incomplete_max_output_tokens():
    """Responses 的 `status` 只有 `completed` / `incomplete` 两个值。

    被预算截断属于 `incomplete` + `incomplete_details.reason`，而规范的枚举值是
    `max_output_tokens`——和第 35 节给 `stop` 用的 `{"reason": "stop"}` 是同一套
    机制，同一个字段。
    """
    body = _resp_non_stream(LONG, 5)
    text = _resp_text(body)
    assert _est_tokens(text) <= 5, (
        f"Responses 路径无视了 max_output_tokens=5：拿到 {len(text)} 字符"
    )
    assert body.get("status") == "incomplete", (
        f"status 是 {body.get('status')!r}：被截断的回答冒充了完整回答"
    )
    assert body.get("incomplete_details") == {"reason": "max_output_tokens"}, (
        f"incomplete_details 是 {body.get('incomplete_details')!r}"
    )
    print("test_responses_non_streaming_reports_incomplete_max_output_tokens: PASSED")


def test_responses_streaming_reports_incomplete_max_output_tokens():
    text, done = _resp_stream(LONG, 5)
    assert _est_tokens(text) <= 5, (
        f"Responses 流式正文突破了上界：{len(text)} 字符"
    )
    assert done.get("status") == "incomplete", (
        f"response.completed 的 status 是 {done.get('status')!r}"
    )
    assert done.get("incomplete_details") == {"reason": "max_output_tokens"}, (
        f"response.completed 的 incomplete_details 是 "
        f"{done.get('incomplete_details')!r}"
    )
    print("test_responses_streaming_reports_incomplete_max_output_tokens: PASSED")


# ------------------------------------------------ B. 零回归（守卫）


def test_an_undeclared_budget_is_never_truncated():
    """**这条用例在修复前的树上也是绿的**——它是守卫，不是缺陷复现。

    Anthropic 的 `max_tokens` 有 4096 的默认值，所以「没声明」与「声明了 4096」
    在字段值上无法区分。若修法直接读字段值，一个省略了它的请求会被一个从来没有人
    限定过的数字截断。判据必须是 `model_fields_set`（客户端真的写过这个键）。
    """
    chat = _chat_non_stream(LONG, None)
    assert chat["choices"][0]["message"]["content"] == LONG, "未声明上界的请求被截断了"
    assert chat["choices"][0]["finish_reason"] == "stop"

    anth = _anth_non_stream(HUGE, None)
    atext = "".join(b.get("text", "") for b in anth.get("content", []))
    assert atext == HUGE, (
        f"省略 max_tokens 的 Anthropic 请求被默认值 4096 截断了："
        f"{len(atext)} < {len(HUGE)} 字符"
    )
    assert anth["stop_reason"] == "end_turn"

    resp = _resp_non_stream(LONG, None)
    assert _resp_text(resp) == LONG, "未声明上界的 Responses 请求被截断了"
    assert resp.get("status") == "completed"
    print("test_an_undeclared_budget_is_never_truncated: PASSED")


def test_a_budget_larger_than_the_answer_never_truncates():
    """**同样是守卫**：上界大于回答时，一个字都不许动。"""
    chat = _chat_non_stream(LONG, 100000)
    assert chat["choices"][0]["message"]["content"] == LONG
    assert chat["choices"][0]["finish_reason"] == "stop"

    text, delta = _anth_stream(LONG, 100000)
    assert text == LONG, f"上界充足却少了 {len(LONG) - len(text)} 个字符"
    assert delta["stop_reason"] == "end_turn"

    text, done = _resp_stream(LONG, 100000)
    assert text == LONG
    assert done.get("status") == "completed"
    print("test_a_budget_larger_than_the_answer_never_truncates: PASSED")


# ------------------------------------------------ C. 截断点之后的工具调用不存在


def test_a_tool_call_beyond_the_cut_is_never_delivered():
    """第 35 节的规则在这里再考一次：截断点之后对客户端不存在。

    脚本是「一句话 + 一个 ```tool_call 围栏」。预算只够前面那几个词，围栏落在
    截断点之后——它不能被解析成一次真实调用，否则客户端会去执行一个它从未看到
    过、也无法和上下文对应的动作。
    """
    script = "Look here. " + '```tool_call\n{"name": "Bash", "args": {"cmd": "rm -rf /"}}\n```'
    _Provider.script = [(script[i:i + 7], {}) for i in range(0, len(script), 7)] + [
        ("", {"conversation_id": "mt-conv"})]
    srv._make_provider = lambda *a, **kw: _Provider()
    try:
        body = _client().post("/v1/chat/completions", json={
            "model": "gpt-4o",
            "max_tokens": 3,
            "messages": [{"role": "user", "content": "go"}],
            "tools": [{"type": "function", "function": {
                "name": "Bash", "description": "run",
                "parameters": {"type": "object",
                               "properties": {"cmd": {"type": "string"}}}}}],
        }).json()
    finally:
        srv._make_provider = _guarded_factory

    choice = body["choices"][0]
    assert not choice["message"].get("tool_calls"), (
        f"截断点之后的工具调用被执行了——客户端从未看到过它: {choice!r}"
    )
    assert choice["finish_reason"] == "length", (
        f"finish_reason 是 {choice['finish_reason']!r}，应当报告被预算截断"
    )
    print("test_a_tool_call_beyond_the_cut_is_never_delivered: PASSED")


# ------------------------------------------------ D. 上界只有一份定义


def test_the_token_budget_has_exactly_one_definition():
    """行为用例只能覆盖我想到的路径；「判定只有一份」才是让下一个协议无法
    悄悄退回「忽略上界」的那个属性（§68 的 `_TOOL_FENCE_LANG`、§95 的措辞同理）。
    """
    root = Path(__file__).resolve().parent.parent
    defs = []
    for p in root.glob("**/*.py"):
        if any(part in {".venv", "tests", "__pycache__"} for part in p.parts):
            continue
        if "class TokenBudget" in p.read_text(encoding="utf-8"):
            defs.append(p.relative_to(root).as_posix())
    assert defs == ["providers/stop_sequences.py"], (
        f"输出预算判定应当只定义一次，实测出现在 {defs}"
    )
    print("test_the_token_budget_has_exactly_one_definition: PASSED")


def test_the_dedup_key_distinguishes_absent_from_declared_default():
    """[S3] 本节修好的字段必须同时进 SingleFlight 的指纹——否则这个修法自己
    会造出第 53 节的缺陷。

    判据只有一条：**指纹相同 ⇒ 两条请求的正确答案可以互换**。修复前 `max_tokens`
    不改变任何输出（被解析后无人读），所以「省略」与「写了 4096」共用指纹是**无害**
    的；修复之后两者行为相反（一个不截断、一个截断在 4096），而 Anthropic 的模型
    字段有 `= 4096` 默认值，用 `req.max_tokens` 做指纹会让这两条并发请求合并，
    Follower 领到一条与它自己的声明不符的答案。

    所以指纹里放的必须是**判定本身**（`_declared_token_budget`），不是字段值。
    本用例同时钉住两半：字段值层面的判据（真模型解析）与三个调用点的源码形状，
    因为只改签名而调用点照旧，正是第 53 节记录过的那个「单测全绿、真实路径仍丢
    字段」的形状。

    注：这条用例在修复前的树上因为 `_declared_token_budget` 不存在而变红（结构性），
    它守卫的是**本次修法可能引入**的缺陷，不是修复前就存在的缺陷。
    """
    from providers.anthropic_compat import AnthropicMessagesRequest

    absent = AnthropicMessagesRequest(
        model="claude-3-5-sonnet", messages=[{"role": "user", "content": "go"}]
    )
    declared = AnthropicMessagesRequest(
        model="claude-3-5-sonnet", messages=[{"role": "user", "content": "go"}],
        max_tokens=4096,
    )
    assert absent.max_tokens == declared.max_tokens == 4096, (
        "前提不成立：两个请求的字段值必须相同，否则这条用例测的不是「省略 vs 声明」"
    )
    assert srv._declared_token_budget(absent) is None
    assert srv._declared_token_budget(declared) == 4096

    kw = dict(
        provider="qwen", model="Qwen", messages=[{"role": "user", "content": "go"}],
        protocol="anthropic_messages",
    )
    assert srv.singleflight_manager.compute_key(
        constraints={"max_tokens": srv._declared_token_budget(absent)}, **kw
    ) != srv.singleflight_manager.compute_key(
        constraints={"max_tokens": srv._declared_token_budget(declared)}, **kw
    ), "「省略 max_tokens」与「声明 4096」的指纹相同——它们的正确答案并不相同"

    src = (Path(__file__).resolve().parent.parent / "server.py").read_text(encoding="utf-8")
    blocks = src.split("constraints={")
    assert len(blocks) == 4, f"应当正好 3 个 constraints 调用点，实测 {len(blocks) - 1}"
    for i, block in enumerate(blocks[1:], start=1):
        head = block.split("},", 1)[0]
        assert "_declared_token_budget(req)" in head, (
            f"第 {i} 个协议的指纹没有用声明判据: {head.strip()[:120]}"
        )
        assert "req.max_tokens," not in head and "req.max_output_tokens," not in head, (
            f"第 {i} 个协议的指纹用了字段值而不是「客户端真写过这个键」: {head.strip()[:120]}"
        )
    print("test_the_dedup_key_distinguishes_absent_from_declared_default: PASSED")


def run_all():
    test_chat_non_streaming_honours_max_tokens_and_reports_length()
    test_chat_streaming_honours_max_tokens_and_reports_length()
    test_anthropic_non_streaming_reports_max_tokens_stop_reason()
    test_anthropic_streaming_reports_max_tokens_stop_reason()
    test_responses_non_streaming_reports_incomplete_max_output_tokens()
    test_responses_streaming_reports_incomplete_max_output_tokens()
    test_an_undeclared_budget_is_never_truncated()
    test_a_budget_larger_than_the_answer_never_truncates()
    test_a_tool_call_beyond_the_cut_is_never_delivered()
    test_the_token_budget_has_exactly_one_definition()
    test_the_dedup_key_distinguishes_absent_from_declared_default()
    assert not _GUARD_HITS, f"有请求绕过了脚本化 provider: {_GUARD_HITS}"
    print("\n✅ All max_tokens budget tests passed! (11 cases)")


if __name__ == "__main__":
    sys.exit(run_all())
