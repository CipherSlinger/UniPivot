"""并发额度必须覆盖整段流，而不是只覆盖握手 (STREAM-AUDIT / C1)

本套件锁定三个由同一次流式生命周期审计抓出的缺陷，三者的共同点是
**「资源的所有权边界画错了」**，而症状全都落在用户体感上（「不好用，稳定性差」）：

C1. 节点并发信号量在 `execute_chat` 返回时就归还了。
    `PROVIDER_RISK_SPECS` 里 qwen 3 / deepseek 2 / kimi 2 这些上限，原意是
    「同一账号同时最多背几条上游会话」——这是反爬系统最直接的自动化并发特征，
    也是本项目「不把另外四家打进风控」约束的直接手段。但原实现把信号量挂在
    `AsyncExitStack` 上、在 `async with` **内部** return，`__aexit__` 随即把额度
    还了回去，而主体一个字节都还没读。实测：并发 6 条长流，峰值在途上游会话是 6，
    而配置写的上限是 3。等于并发治理对真正的并发流完全不存在。

C2. 上游一帧都不吐被记成成功。
    五家的 `chat()` 结尾都**无条件**产出一帧末位元数据（`("", {conversation_id})`），
    所以「首个元素就 StopAsyncIteration」只可能是上游连上了却什么都没给。原实现
    照样 `record_success` 并交出一个空流：一次真实的失败被记成健康，失败计数
    永远不增长，下一轮仍然选它。

C3. 中途断流不给节点记冷却。
    一个每几次请求就吐到一半断掉的节点，失败计数永远是 0，于是被反复选为主选，
    客户端反复看到「回复说到一半停住」。风险类拦截（401/429/WAF）由 server 侧
    统一记账，这里只断言非风险类断流会进入冷却，且不会被记两遍。

运行: .venv/bin/python tests/test_stream_concurrency.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["TESTING"] = "1"

# [DAY-ZERO] 派发闸门有一道「本机有没有凭证」的硬阻断。本套件的上游是内存假节点，
# 与凭证无关；但不摆出凭证，请求会在进入被测逻辑前就以 `unconfigured` 返回，
# 套件便不再测它声称要测的东西。五个节点一律摆上假 token。
for _pk in ("QWEN", "DEEPSEEK", "DOUBAO", "KIMI", "GLM"):
    os.environ[f"{_pk}_AUTH_TOKEN"] = f"test-token-{_pk.lower()}"

os.environ["GATEWAY_DISABLE_RISK_LEDGER"] = "1"
os.environ["GATEWAY_DISABLE_AUTO_REFRESH"] = "1"
os.environ["GATEWAY_DISABLE_CREDENTIAL_WATCH"] = "1"
os.environ["GATEWAY_DISABLE_PROBATION"] = "1"
os.environ["GATEWAY_DISABLE_PACING"] = "1"

from providers.base import ProviderError  # noqa: E402
from providers.failover import FailoverRouter  # noqa: E402

ROUTER = FailoverRouter()


# ------------------------------------------------------------------ 工具


class _StreamProbe:
    """脚本化的上游：记录同时在途的会话数，并按脚本产出。"""

    in_flight = 0
    peak = 0
    chunks: list = []
    hold = 0.0
    fail_after: int | None = None

    def __init__(self, key: str):
        self.key = key

    async def chat(self, messages, model, **kw):
        type(self).in_flight += 1
        type(self).peak = max(type(self).peak, type(self).in_flight)
        try:
            for idx, item in enumerate(type(self).chunks):
                if type(self).hold:
                    await asyncio.sleep(type(self).hold)
                if type(self).fail_after is not None and idx >= type(self).fail_after:
                    raise ProviderError("上游连接中途断开", status=502, err_type="upstream_error")
                yield item
            # 与真实 provider 一致：结尾无条件产出末位元数据帧
            yield "", {"conversation_id": f"{self.key}-conv"}
        finally:
            type(self).in_flight -= 1

    async def delete_conversation(self, *a, **kw):
        return True


def _script(chunks, hold=0.0, fail_after=None):
    _StreamProbe.chunks = chunks
    _StreamProbe.hold = hold
    _StreamProbe.fail_after = fail_after
    _StreamProbe.peak = 0


async def _consume(n: int, provider: str = "qwen"):
    async def one():
        stream, _ev, _p = await ROUTER.execute_chat(
            primary_provider_key=provider,
            primary_wire_model="Qwen",
            provider_factory=lambda k: _StreamProbe(k),
            chat_args={"messages": []},
        )
        async for _delta, _meta in stream:
            pass

    await asyncio.gather(*(one() for _ in range(n)))


# ------------------------------------------------------------------ C1


def test_semaphore_covers_the_whole_stream():
    """并发 6 条长流，qwen 的在途上限 3 必须真的生效。"""
    ROUTER.cooldown_tracker.reset()
    _script([("hi", {}), ("there", {})], hold=0.02)
    asyncio.run(_consume(6))
    limit = 3  # PROVIDER_RISK_SPECS["qwen"].max_concurrency
    assert _StreamProbe.peak <= limit, (
        f"并发额度在握手后就被归还：同时在途上游会话峰值 {_StreamProbe.peak}，配置上限 {limit}"
    )
    assert _StreamProbe.peak > 1, "串行执行说明测试没有真正并发，断言无意义"
    print(f"test_semaphore_covers_the_whole_stream: PASSED (peak={_StreamProbe.peak})")


def test_slot_is_returned_after_the_stream_finishes():
    """流结束后额度必须归还，否则几轮之后节点被自己永久占满。"""
    ROUTER.cooldown_tracker.reset()
    _script([("a", {}), ("b", {})])
    for _ in range(6):
        asyncio.run(_consume(3))
    sem = ROUTER.cooldown_tracker.get_semaphore("qwen")
    assert sem._value == 3, f"流结束后并发额度没有全部归还: 剩余 {sem._value}/3"
    print("test_slot_is_returned_after_the_stream_finishes: PASSED")


def test_slot_is_returned_when_client_disconnects_mid_stream():
    """客户端读了两帧就断连（aclose）→ 额度同样必须归还。"""
    ROUTER.cooldown_tracker.reset()
    _script([("x", {}) for _ in range(50)], hold=0.01)

    async def early_close():
        stream, _ev, _p = await ROUTER.execute_chat(
            primary_provider_key="qwen",
            primary_wire_model="Qwen",
            provider_factory=lambda k: _StreamProbe(k),
            chat_args={"messages": []},
        )
        got = 0
        async for _d, _m in stream:
            got += 1
            if got >= 2:
                break
        await stream.aclose()

    asyncio.run(early_close())
    # aclose 后活着的内层生成器要等下一轮事件循环才被收尾（实测），
    # 因此在同一循环里再跑一次空的 asyncio.run 之前先让出若干次。
    sem = ROUTER.cooldown_tracker.get_semaphore("qwen")

    async def settle():
        for _ in range(20):
            await asyncio.sleep(0)
        for _ in range(20):
            await asyncio.sleep(0.01)

    asyncio.run(settle())
    assert sem._value == 3, f"断连后并发额度泄漏: 剩余 {sem._value}/3"
    print("test_slot_is_returned_when_client_disconnects_mid_stream: PASSED")


# ------------------------------------------------------------------ C2


class _EmptyProbe:
    def __init__(self):
        self.calls = 0

    async def chat(self, messages, model, **kw):
        self.calls += 1
        if False:  # pragma: no cover - 让本方法成为异步生成器
            yield None

    async def delete_conversation(self, *a, **kw):
        return True


def test_empty_stream_is_not_recorded_as_success():
    """上游一帧都没有 = 失败，不是成功；失败计数必须增长。"""
    ROUTER.cooldown_tracker.reset()
    probe = _EmptyProbe()

    async def go():
        try:
            stream, _ev, _p = await ROUTER.execute_chat(
                primary_provider_key="qwen",
                primary_wire_model="Qwen",
                provider_factory=lambda k: probe,
                chat_args={"messages": []},
            )
        except ProviderError:
            return "raised"
        # 若没抛，至少要把流读完看是否给出内容
        async for _d, _m in stream:
            return "leaked"
        return "empty_ok"

    outcome = asyncio.run(go())
    fails = ROUTER.cooldown_tracker._failure_counts.get("qwen", 0)
    # 环内其他节点也会被尝试（它们同样是空流），但 qwen 这一格必须计入失败
    assert outcome == "raised", (
        f"空流被当成了正常完成（{outcome}）—— 一次真实失败被记成健康"
    )
    assert fails >= 1, "空流没有计入失败计数，节点永远不会被降级"
    print(f"test_empty_stream_is_not_recorded_as_success: PASSED (fails={fails})")


# ------------------------------------------------------------------ C3


def test_midstream_failure_cools_the_node():
    """吐了两帧就断流：节点必须进入冷却，下一轮不再被选为主选。"""
    ROUTER.cooldown_tracker.reset()
    _script([("a", {}), ("b", {}), ("c", {}), ("d", {})], fail_after=2)

    async def go():
        stream, _ev, _p = await ROUTER.execute_chat(
            primary_provider_key="qwen",
            primary_wire_model="Qwen",
            provider_factory=lambda k: _StreamProbe(k),
            chat_args={"messages": []},
        )
        seen = 0
        try:
            async for _d, _m in stream:
                seen += 1
        except ProviderError:
            pass
        return seen

    seen = asyncio.run(go())
    assert seen == 2, f"断流前的帧必须如实交给客户端，实际 {seen}"
    fails = ROUTER.cooldown_tracker._failure_counts.get("qwen", 0)
    assert fails >= 1, (
        "中途断流没有给节点记冷却：一个每几次就断流的节点失败计数恒为 0，"
        "会被反复选为主选，客户端反复看到回复说到一半停住"
    )
    assert ROUTER.cooldown_tracker.is_in_cooldown("qwen"), "断流后节点未进入冷却"
    print(f"test_midstream_failure_cools_the_node: PASSED (fails={fails})")


def test_midstream_risk_failure_is_not_double_counted():
    """风险类断流由 server 侧统一记账，这一层不得重复记（否则冷却被放大）。"""
    ROUTER.cooldown_tracker.reset()

    class _WafProbe(_StreamProbe):
        async def chat(self, messages, model, **kw):
            yield "a", {}
            yield "b", {}
            raise ProviderError(
                "FAIL_SYS_USER_VALIDATE::RGV587_ERROR", status=429, err_type="rate_limit_error"
            )

    async def go():
        stream, _ev, _p = await ROUTER.execute_chat(
            primary_provider_key="qwen",
            primary_wire_model="Qwen",
            provider_factory=lambda k: _WafProbe(k),
            chat_args={"messages": []},
        )
        try:
            async for _d, _m in stream:
                pass
        except ProviderError:
            pass

    asyncio.run(go())
    fails = ROUTER.cooldown_tracker._failure_counts.get("qwen", 0)
    assert fails == 0, (
        f"风险类断流在 execute_chat 层被记账了 {fails} 次：server 侧还会再记一次，"
        f"同一次拦截被记两遍会让冷却按 2 的幂次虚高（对已被封禁的节点尤其危险）"
    )
    print("test_midstream_risk_failure_is_not_double_counted: PASSED")


def run_all():
    tests = [
        test_semaphore_covers_the_whole_stream,
        test_slot_is_returned_after_the_stream_finishes,
        test_slot_is_returned_when_client_disconnects_mid_stream,
        test_empty_stream_is_not_recorded_as_success,
        test_midstream_failure_cools_the_node,
        test_midstream_risk_failure_is_not_double_counted,
    ]
    for fn in tests:
        fn()
    print(f"\n✅ All stream concurrency/quota tests passed! ({len(tests)} cases)")


if __name__ == "__main__":
    run_all()