"""上游 4xx 拒绝的是**请求体**，不是节点 (REQUEST-REJECT)

## 缺陷：客户端写错的一句话，被报成了「五个节点全都坏了」

`classify_stream_status` 只认识 401 / 403 / 429 三类风控状态，其余一律
`("upstream_error", False)`，而所有 provider 的抛出点都是

    raise ProviderError(..., status=resp.status_code if is_risk else 502, err_type=err_type)

于是上游对**请求体**的拒绝（400 / 413 / 422：参数不合法、上下文超长、内容被
策略拒绝）在网关里变成了 `502 upstream_error` —— 一个「服务器坏了，重试也许会好」
的结论，而事实是「这句话上游不接受，重发一百次也一样」。

实测（脚本化 provider 抛出 provider 层对 400 的真实产物，`TESTING=1`，零真实上游
调用，六个协议/传输路径）：

    chat-nonstream        -> 502  calls=5  cooldowns={qwen, glm, doubao, kimi, deepseek}
    chat-stream           -> 502  calls=5  cooldowns={五家}
    anthropic-nonstream   -> 502  calls=5  cooldowns={五家}
    anthropic-stream      -> 502  calls=5  cooldowns={五家}
    responses-nonstream   -> 502  calls=5  cooldowns={五家}
    responses-stream      -> 502  calls=3  cooldowns={三家}

一次被拒绝的请求产生了**五次真实的上游调用**，并把环内**每一个健康节点**都写进了
冷却。三条后果各自都够严重：

  * **自动重试风暴**：客户端收到 5xx 会立刻重试同一条请求，于是「5 家 × N 次」。
    这正是 [H7] 用 `x-should-retry: false` 消灭的形态，只是成因从「环已走完」换成
    「请求本身被拒绝」。对一个长会话的 agent 来说，上下文超限是**必然会遇到**的
    事件，不是边缘情况。
  * **诬告健康节点**：节点什么都没做错，却被记了失败计数与冷却，后续请求被
    `get_fallback_candidates` 无谓地绕开。这与 [BUDGET-3]/[BUDGET-4] 记录的
    「我们家自己的判断被写成了上游的罪状」是同一类错误。
  * **误导运维**：报错里写着「对等能力环内所有节点均已尝试且全部失效」，运维会去
    查网络和上游，而真正该改的是客户端发来的那条请求。

## 正确语义

状态码本身就是**事实**，不需要猜 body：400 / 413 / 422 是「这条请求不合法」，
401 / 403 / 429 是「这个账号被拦下了」，5xx / 网络错误是「这个节点坏了」。三者的
处置互不相同：

  * 请求级拒绝 → **保留上游状态码**（400/413/422），`err_type="invalid_request"`，
    **不故障转移、不记失败、不上冷却、不进风控台账**，并带上
    `x-should-retry: false`（客户端不该原地重试同一份请求）；
  * 风控拦截 → 保留状态码 + 进风控台账（既有行为，逐值不变）；
  * 上游故障 → 502 + 故障转移 + 冷却（既有行为，逐值不变）。

不做故障转移是一个**有意的取舍**：换一家厂商确实可能接受被另一家拒绝的内容，
但那需要付出「五次带凭证的上游触点」，而这条产品线存在的理由恰恰是不为一次
猜测去消耗账号。运维需要换厂商时，换模型重发即可。

运行: .venv/bin/python tests/test_invalid_request.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["TESTING"] = "1"
os.environ["GATEWAY_DISABLE_AUTO_REFRESH"] = "1"
os.environ["GATEWAY_DISABLE_CREDENTIAL_WATCH"] = "1"
os.environ["GATEWAY_DISABLE_PROBING"] = "1"
os.environ["GATEWAY_DISABLE_PACING"] = "1"
os.environ["GATEWAY_DISABLE_RISK_LEDGER"] = "1"
os.environ["GATEWAY_DISABLE_SINGLEFLIGHT"] = "1"
os.environ["QWEN_AUTH_TOKEN"] = "fake_qwen"
os.environ["DEEPSEEK_AUTH_TOKEN"] = "fake_ds"
os.environ["DOUBAO_AUTH_TOKEN"] = "fake_doubao"
os.environ["KIMI_AUTH_TOKEN"] = "fake_kimi"
os.environ["GLM_AUTH_TOKEN"] = "fake_glm"

import httpx  # noqa: E402

import server as srv  # noqa: E402
from providers.base import ProviderError, classify_stream_status  # noqa: E402

_DEFAULT_FACTORY = srv._make_provider
CALLS: list[str] = []


class _UpstreamFault:
    """按 provider 层**真实的**产物抛错。

    400/413/422 的换算来自 `classify_stream_status` 本身，而不是在这里手写
    `ProviderError(status=502)` —— 否则这条用例会把「网关错在哪」写死在测试里，
    产品改了、测试却仍然按旧假设构造，两边一起漂移。
    """

    def __init__(self, key: str, status: int):
        self.key = key
        self.status = status

    async def chat(self, messages, model, **kw):
        CALLS.append(self.key)
        err_type, keep_status = classify_stream_status(self.status)
        raise ProviderError(
            f"[{self.key}] 上游拒绝了这次请求: HTTP {self.status} invalid payload",
            status=self.status if keep_status else 502,
            err_type=err_type,
        )
        yield  # pragma: no cover

    async def delete_conversation(self, *a, **kw):
        return None


def _install(status: int):
    CALLS.clear()
    _reset_risk()
    srv._make_provider = lambda key, *a, **kw: _UpstreamFault(key, status)


def _restore():
    """只卸载桩与亲和性状态。

    **不在这里清冷却**：清掉之后再去断言「没有冷却」是**同一反复构**的断言——
    它恒为真，产品再怎么诬告节点它也绿。状态必须在请求结束后、`_restore` 之前
    读走（见各用例里的 `cools = _cooldowns()`）。
    """
    srv._make_provider = _DEFAULT_FACTORY
    srv.session_affinity_manager.clear()


def _reset_risk():
    tr = srv.failover_router.cooldown_tracker
    tr._cooldowns.clear()
    tr._cooldown_severity.clear()
    tr._failure_counts.clear()


def _cooldowns() -> dict:
    tr = srv.failover_router.cooldown_tracker
    return {k: v for k, v in tr._cooldowns.items() if v}


def _semaphores() -> dict:
    """节点信号量的当前值。

    新增的这条提前上抛路径绕开了原 `except` 分支的后半段，因此必须证明它**没有
    顺带绕开资源归还**（[BUDGET-3] 的教训：资源泄漏在一次「被正确拒绝」的请求后
    同样是永久的，而且比错误码更难发现）。归还发生在更内层的 `finally` 里，
    `sem_held and not sem_handed_off` 那一句，这里只是把结果量出来。
    """
    tr = srv.failover_router.cooldown_tracker
    return {k: tr.get_semaphore(k)._value for k in
            ("qwen", "deepseek", "doubao", "kimi", "glm")}


def _post(path: str, payload: dict) -> httpx.Response:
    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=srv.app), base_url="http://t"
        ) as c:
            return await c.post(path, json=payload)

    return asyncio.run(run())


_CASES = [
    ("chat-nonstream", "/v1/chat/completions",
     {"model": "qwen", "messages": [{"role": "user", "content": "hi"}]}, 400),
    ("chat-stream", "/v1/chat/completions",
     {"model": "qwen", "stream": True, "messages": [{"role": "user", "content": "hi"}]}, 400),
    ("anthropic-nonstream", "/v1/messages",
     {"model": "claude-3-5-sonnet", "max_tokens": 64,
      "messages": [{"role": "user", "content": "hi"}]}, 400),
    ("anthropic-stream", "/v1/messages",
     {"model": "claude-3-5-sonnet", "max_tokens": 64, "stream": True,
      "messages": [{"role": "user", "content": "hi"}]}, 400),
    ("responses-nonstream", "/v1/responses", {"model": "gpt-4o", "input": "hi"}, 400),
    ("responses-stream", "/v1/responses",
     {"model": "gpt-4o", "input": "hi", "stream": True}, 400),
]


# ---------------------------------------------------------------- 1. 契约层


def test_classify_stream_status_has_three_semantics():
    """状态码分类必须表达三种事实，而不是两种。"""
    for status in (400, 413, 422):
        err_type, keep = classify_stream_status(status)
        assert err_type == "invalid_request", (
            f"上游 {status} 是「请求被拒绝」，不是「服务器坏了」，实测被归成了 {err_type!r}"
        )
        assert keep is True, (
            f"上游 {status} 必须把原始状态码还给客户端（4xx 才会让 SDK 停止重试），"
            f"而不是塌成 502"
        )

    for status, expected in ((401, "auth_error"), (403, "waf_blocked"), (429, "rate_limit_error")):
        err_type, keep = classify_stream_status(status)
        assert (err_type, keep) == (expected, True), (
            f"风控状态的分类发生了回归: {status} -> {(err_type, keep)!r}"
        )

    for status in (500, 502, 503):
        err_type, keep = classify_stream_status(status)
        assert (err_type, keep) == ("upstream_error", False), (
            f"上游故障的分类发生了回归: {status} -> {(err_type, keep)!r}"
        )


# ---------------------------------------------------------------- 2. 线上路径


def test_a_rejected_request_is_returned_as_a_4xx_on_every_path():
    """**靶点**：六个路径都必须把上游的 4xx 如实还成 4xx，而不是 502。"""
    for name, path, payload, status in _CASES:
        _install(status)
        try:
            # 基线必须在**驱动请求之前**:在请求之后读基线,就是拿结果和它自己比,
            # 恒为真 ([BUDGET-3] 里那条测试 bug 的形状)。
            sem_before = _semaphores()
            r = _post(path, payload)
            calls = list(CALLS)
            cools = _cooldowns()
            sem_after = _semaphores()
        finally:
            _restore()

        assert sem_after == sem_before, (
            f"{name}: 被拒绝的请求泄漏了节点并发信号量 {sem_before} -> {sem_after} —— "
            f"提前上抛绕过了返回错误码的那段代码,不能顺带绕过资源归还"
        )

        assert r.status_code == status, (
            f"{name}: 上游 {status}（请求被拒绝）被报成了 {r.status_code} —— 客户端会把它"
            f"当成上游故障并反复重试同一条请求。实测 body: {r.text[:200]}"
        )
        assert r.headers.get("x-should-retry") == "false", (
            f"{name}: 请求级拒绝没有带 `x-should-retry: false`。实测头: "
            f"{dict(r.headers)}"
        )
        # 唯一的例外：这一类的修复方**就是客户端自己**，改完立刻能发，因此不能给它
        # 一个「等 600 秒」的倒计时（其余确定性拒绝都在等一个会自己到期的窗口）。
        assert "Retry-After" not in r.headers, (
            f"{name}: 请求级拒绝带了 `Retry-After: {r.headers.get('Retry-After')}` —— "
            f"客户端改完请求就能立刻重发，倒计时只会让它白等"
        )
        assert len(calls) == 1, (
            f"{name}: 一次被上游拒绝的请求打了 {len(calls)} 个节点（应只打 1 个）。"
            f"拒绝的是**请求体**，换节点只会把同一份请求再送一遍 —— 五家 × 客户端重试"
            f"正是风控线要消灭的自动重试签名。实测: {calls}"
        )
        assert not cools, (
            f"{name}: 请求被拒绝后节点被写进了冷却 {cools} —— 节点没有坏，"
            f"是请求不合法；后续请求会被无谓地绕开这些健康节点"
        )


def test_the_refusal_message_names_the_request_not_the_ring():
    """报错必须让运维看见「请求被拒绝」，而不是「整环全挂」。"""
    _install(400)
    try:
        r = _post("/v1/chat/completions",
                  {"model": "qwen", "messages": [{"role": "user", "content": "hi"}]})
    finally:
        _restore()

    text = r.text
    assert "invalid payload" in text, (
        f"上游给出的拒绝原因没有送到客户端: {text[:300]}"
    )
    assert "所有节点均已尝试且全部失效" not in text, (
        f"一条被拒绝的请求被报成了「整环失效」，运维会去查网络和上游: {text[:300]}"
    )
    assert "invalid_request" in text, f"错误类型不是 invalid_request: {text[:300]}"


# ---------------------------------------------------------------- 3. 零回归


def test_an_upstream_5xx_still_fails_over_and_cools():
    """零回归：真正的上游故障仍然要故障转移并记冷却。"""
    _install(503)
    try:
        r = _post("/v1/chat/completions",
                  {"model": "qwen", "messages": [{"role": "user", "content": "hi"}]})
        calls = list(CALLS)
        cools = _cooldowns()
    finally:
        _restore()

    assert r.status_code == 502, f"上游 5xx 应报 502，实测 {r.status_code}: {r.text[:200]}"
    assert len(calls) > 1, f"上游故障必须继续故障转移，实测只打了 {len(calls)} 个节点"
    assert cools, "上游故障必须记冷却，实测没有"


def run_all():
    test_classify_stream_status_has_three_semantics()
    test_a_rejected_request_is_returned_as_a_4xx_on_every_path()
    test_the_refusal_message_names_the_request_not_the_ring()
    test_an_upstream_5xx_still_fails_over_and_cools()
    print("\n✅ All request-rejection tests passed! (4 cases)")


if __name__ == "__main__":
    run_all()
