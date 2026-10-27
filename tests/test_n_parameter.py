"""`n` 是 OpenAI 协议里唯一一条网关**无法兑现**的输出约束，却一直静默返回一个 choice (PROTOCOL)

## 实测：声明要三个候选，拿到一个，没有任何错误

真实 ASGI 路径（脚本化 provider，`TESTING=1`，零真实上游调用），
`POST /v1/chat/completions`：

    n=None   status=200  choices=1  indices=[0]  upstream_calls=1
    n=1      status=200  choices=1  indices=[0]  upstream_calls=1
    n=0      status=200  choices=1  indices=[0]  upstream_calls=1
    n=-1     status=200  choices=1  indices=[0]  upstream_calls=1
    n=3      status=200  choices=1  indices=[0]  upstream_calls=1

    stream=True, n=3
    status=200  content-type=text/event-stream
    data: {..., "choices": [{"index": 0, ...

`n` 根本不在 `ChatCompletionRequest` 上（`server.py:586-611`），只被
`extra="allow"` 收进 `model_extra`，然后**没有任何代码读过**；而全部 17 处
`choices` 发射点都把 `index: 0` 写死。所以 `n` 既没有被拒绝，也没有被执行——
它是"解析了、然后没人用"这一类缺陷（§34b/§35/§73）里最需要**拒绝**而不是
补全的那一种。

## 为什么是拒绝，而不是补造 n 个候选（§58 的判据）

五个上游全是网页端聊天模型，**一次调用只产出一条回答**。要凑出 `choices: [0,1,2]`
只有两种做法，而两种都是造假：

* 复制同一条回答三次 —— 客户端拿到的是"三个独立候选"，实际是一个答案的三份
  副本。任何做 best-of-n 重排的客户端（这正是 `n` 的用途）都会把它当成三次独立
  采样来评分，得到的结论是凭空生成的。
* 连发 n 次上游请求 —— 那是 n 倍的风控暴露面。CLAUDE.md 的常设约束是"绝不让
  另一个厂商的风控分数升高"，用一次客户端请求换三倍上游调用正好相反。

§73（`previous_response_id`）已经确立了这条纪律：当一个字段的契约是网关**可证明
无法兑现**的，正确的处理是拒绝，并给出客户端已经具备恢复逻辑的错误，而不是静默
降级成"看起来成功了"。静默返回一个 choice 是其中最糟的一种：客户端按 `n=3` 规划
了重排，拿到一个候选，没有任何地方能看出少了两个。

## 错误形状

`400 / invalid_request_error`，`param="n"`。这是 OpenAI 线上契约里"参数值不被支持"
的标准形状——SDK 的 `BadRequestError` 分支会接到它，`param` 让调用方知道是哪个字段，
消息里写明本网关一次只产出一条回答。零上游调用（H4）：拒绝发生在任何 provider 被
构造之前，且流式请求同样在**状态行**上拒绝，而不是先发 200 再在 SSE 里塞错误。

## 为什么去重键不需要 `n`

因为拒绝把 `n` 变成惰性的：`n=1` 与不声明 `n` 产出完全相同的请求，其余取值根本
到不了派发。这正是 §53 判据（"指纹相同 ⇒ 两条请求的正确答案可互换"）在此处自然
成立的原因——不是忘了把它放进键里，而是它已经无法改变答案。

## 零回归

`n=1` 与不声明 `n` 必须逐字节等价（用例 3、4）：一个客户端显式写出它本来就得到的
东西，不该因此被拒绝。这两条用例在修复前的树上也是绿的——它们**是守卫**，不是
RED-first 的缺陷复现（§59/§69/§79 的纪律），存在是因为最容易写错的修法是
`if req.n:` 或 `if req.n is not None: refuse`，那样 `n=1` 会被误伤。

运行: .venv/bin/python tests/test_n_parameter.py
"""

from __future__ import annotations

import asyncio
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

import httpx  # noqa: E402

import server as srv  # noqa: E402

_GUARD_HITS: list[str] = []


def _guarded_factory(*args, **kwargs):
    _GUARD_HITS.append(str(args[0]) if args else "?")
    raise AssertionError("测试发出了未安装假 provider 的上游请求")


srv._make_provider = _guarded_factory

CALLS: list[int] = []


class _Provider:
    async def chat(self, messages, model, **kw):
        CALLS.append(1)
        yield "answer", {}
        yield "", {"conversation_id": "n-param-conv"}

    async def delete_conversation(self, *args, **kwargs):
        return True


async def _post(n, stream: bool = False) -> httpx.Response:
    """驱动真实 /v1/chat/completions 路径。"""
    srv._make_provider = lambda *a, **kw: _Provider()
    try:
        body = {
            "model": "gpt-4o",
            "stream": stream,
            "messages": [{"role": "user", "content": "hi"}],
        }
        if n is not None:
            body["n"] = n
        transport = httpx.ASGITransport(app=srv.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            return await c.post("/v1/chat/completions", json=body)
    finally:
        srv._make_provider = _guarded_factory


def _calls(fn, *a, **kw):
    CALLS.clear()
    return fn(*a, **kw)


# ------------------- 1. n>1 必须被拒绝，而不是静默塌成一个候选

def test_n_greater_than_one_is_refused_not_silently_collapsed():
    """实测修复前：`n=3` -> 200，`choices` 长度 1。

    客户端按三个候选规划了 best-of-n 重排，拿到一个，没有任何地方能看出少了
    两个——一次没有报错的错答案。上游一次调用只产出一条回答，所以这个契约
    可证明无法兑现，只能拒绝。
    """
    CALLS.clear()
    r = asyncio.run(_post(3))
    assert r.status_code == 400, (
        f"`n=3` 没有被拒绝（实测修复前 200 且 choices=1，实测本次 {r.status_code}）: "
        f"{r.text[:200]}"
    )
    err = r.json().get("error") or {}
    assert err.get("type") == "invalid_request_error", f"错误类型不是 invalid_request_error: {err}"
    assert err.get("param") == "n", f"错误没有指向 `param='n'`，调用方无法知道是哪个字段: {err}"
    assert not CALLS, f"拒绝必须发生在任何上游调用之前，实测发生了 {len(CALLS)} 次"
    print("test_n_greater_than_one_is_refused_not_silently_collapsed: PASSED")


# ------------------- 2. n<=0 是取值本身非法，同样必须拒绝

def test_a_non_positive_n_is_refused():
    """`n=0` / `n=-1` 在 OpenAI 契约里本身就是非法取值。

    实测修复前两者都返回 200 + 一个 choice，即把一个非法声明当成了默认值。
    """
    for bad in (0, -1):
        CALLS.clear()
        r = asyncio.run(_post(bad))
        assert r.status_code == 400, (
            f"`n={bad}` 没有被拒绝（实测修复前 200 + choices=1，本次 {r.status_code}）"
        )
        assert not CALLS, f"`n={bad}` 的拒绝之前发生了 {len(CALLS)} 次上游调用"
    print("test_a_non_positive_n_is_refused: PASSED")


# ------------------- 3. n=1 必须逐字节零回归（守卫）

def test_n_one_is_value_for_value_unchanged():
    """显式写出 `n=1` 的客户端，得到的必须与不声明时完全相同。

    **这条用例在修复前的树上也是绿的**——它是一条守卫，不是缺陷复现。它存在
    是因为最容易写错的修法是 `if req.n is not None: refuse`，那样一个只是
    把默认值写出来的客户端会被误伤。
    """
    CALLS.clear()
    r1 = asyncio.run(_post(1))
    assert r1.status_code == 200, f"`n=1` 被拒绝了，这是误伤: {r1.text[:200]}"
    assert len(CALLS) == 1, f"`n=1` 应当正常派发一次上游，实测 {len(CALLS)} 次"
    choices = r1.json()["choices"]
    assert len(choices) == 1 and choices[0]["index"] == 0, f"`n=1` 的 choices 形状变了: {choices}"

    CALLS.clear()
    r2 = asyncio.run(_post(None))
    assert r2.status_code == 200 and len(CALLS) == 1
    assert r1.json()["choices"] == r2.json()["choices"], (
        "`n=1` 与不声明 `n` 的答案形状不一致——显式默认值必须逐值等价"
    )
    print("test_n_one_is_value_for_value_unchanged: PASSED")


# ------------------- 4. 不声明 n 必须零回归（守卫）

def test_absent_n_is_value_for_value_unchanged():
    """没有 `n` 的请求（绝大多数客户端）必须完全不受影响。

    **这条用例在修复前的树上也是绿的**，是守卫。
    """
    CALLS.clear()
    r = asyncio.run(_post(None))
    assert r.status_code == 200
    assert len(CALLS) == 1
    assert len(r.json()["choices"]) == 1
    print("test_absent_n_is_value_for_value_unchanged: PASSED")


# ------------------- 5. 流式同样在状态行上拒绝（H4）

def test_streaming_n_is_refused_on_the_status_line():
    """实测修复前：`stream=True, n=3` -> 200 + `text/event-stream`。

    流式请求先发 200、再在 SSE 里塞错误，正是 H4 记录的形状：多数 SDK 先看状态
    行，200 一律当成功，于是客户端拿到的是一条看起来完整的单候选流，永远不会
    知道自己少要了两个候选。确定性拒绝必须在状态行上，且是 `application/json`。
    """
    CALLS.clear()
    r = asyncio.run(_post(3, stream=True))
    assert r.status_code == 400, (
        f"流式 `n=3` 没有在状态行上被拒绝（实测修复前 200 + text/event-stream，"
        f"本次 {r.status_code}）: {r.text[:200]}"
    )
    ctype = r.headers.get("content-type", "")
    assert "application/json" in ctype, f"确定性拒绝应当返回 JSON，实测 content-type={ctype!r}"
    assert not CALLS, f"流式拒绝之前发生了 {len(CALLS)} 次上游调用"
    print("test_streaming_n_is_refused_on_the_status_line: PASSED")


# ----------- 6. 字段必须显式声明，不能退回 extra="allow"（结构性）

def test_n_is_declared_on_the_request_model():
    """`extra="allow"` 正是这个缺陷存在的原因：字段被收下，却没有归属。

    结构性断言，因为行为用例只能覆盖我想得到的取值；"请求模型上真的声明了 `n`"
    才是让下一个人读代码时看得见它的那个属性。
    """
    fields = srv.ChatCompletionRequest.model_fields
    assert "n" in fields, (
        "`n` 没有在 ChatCompletionRequest 上声明——它会重新落回 model_extra，"
        "成为一个没有归属、没有读者的字段"
    )
    print("test_n_is_declared_on_the_request_model: PASSED")


def run_all():
    test_n_greater_than_one_is_refused_not_silently_collapsed()
    test_a_non_positive_n_is_refused()
    test_n_one_is_value_for_value_unchanged()
    test_absent_n_is_value_for_value_unchanged()
    test_streaming_n_is_refused_on_the_status_line()
    test_n_is_declared_on_the_request_model()
    assert not _GUARD_HITS, f"有请求绕过了脚本化 provider: {_GUARD_HITS}"
    print("\n✅ All n-parameter tests passed! (6 cases)")


if __name__ == "__main__":
    sys.exit(run_all())