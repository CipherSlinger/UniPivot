"""握手阶段必须有**整条请求**的预算，而不只是「每次尝试」的上限 (tests/test_handshake_budget.py)。

[M2] 给 `GATEWAY_FIRST_CHUNK_TIMEOUT`（默认 180s）装的是**每一个候选节点**的上限。
派发循环会逐个走完能力环里的候选，于是最坏情况是：

    candidates × GATEWAY_FIRST_CHUNK_TIMEOUT = 5 × 180s = 900s

在这 900 秒里，客户端收不到**任何**字节——响应头都还没发出（`execute_chat` 在
构造响应之前就被 await，见 H4），所以 `sse_keepalive` 根本没有运行的机会；同时
负载均衡配额、节点并发信号量一直被这条注定失败的请求占着。

实测（`GATEWAY_FIRST_CHUNK_TIMEOUT=1.0`，五节点全挂）：

    [对等互备网格] 尝试失败: [qwen:Qwen] ... 在 1s 内未返回首包（握手超时）
    ... 逐个 qwen / glm / doubao / kimi / deepseek，各 1s
    failed after 5.0s: 504 upstream_timeout
    per-attempt timeout: 1.0

「每次尝试 1 秒」与「整条请求 5 秒」之间的差距就是缺陷本身：单个上限不是请求
的边界，候选数量才是乘数，而候选数量由能力环定义决定、不由客户端请求决定。

**本案测的是整条请求的总耗时，不是任何单次尝试的耗时**——这两者的差正是要被
消灭的东西。同时钉住三件不能破的事：

  1. 预算仍然允许**一次健康的慢握手**成功（推理模型正常思考几十秒是健康状态，
     不能因为加了预算就把慢节点判死）；
  2. 预算耗尽时的错误必须说清楚「试了几个节点、预算多少」，而不是一句超时；
  3. 单节点成功、单节点失败后转健康的路径逐值不变（零回归）。

运行: .venv/bin/python tests/test_handshake_budget.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["TESTING"] = "1"
os.environ["GATEWAY_DISABLE_AUTO_REFRESH"] = "1"
os.environ["GATEWAY_DISABLE_PROBING"] = "1"
os.environ["GATEWAY_DISABLE_PACING"] = "1"
os.environ["GATEWAY_DISABLE_SINGLEFLIGHT"] = "1"
os.environ["GATEWAY_DISABLE_PROBATION"] = "1"
os.environ["GATEWAY_DISABLE_RISK_CORRELATION"] = "1"
os.environ["GATEWAY_DISABLE_CREDENTIAL_WATCH"] = "1"
os.environ["GATEWAY_DISABLE_CREDENTIAL_RENEWAL"] = "1"
os.environ["GATEWAY_DISABLE_DEVICE_IDENTITY"] = "1"

from providers.failover import FailoverRouter, ProviderError  # noqa: E402


class _HungProvider:
    """连上了但永不产出任何数据（握手挂住）。"""

    def __init__(self, key: str, closed_flag: dict):
        self.key = key
        self.flag = closed_flag

    async def chat(self, *a, **kw):
        try:
            await asyncio.Event().wait()
            yield "", {}
        finally:
            self.flag["closed"] = self.flag.get("closed", 0) + 1


class _HealthyProvider:
    def __init__(self, key: str, delay: float = 0.0):
        self.key = key
        self.calls = 0
        self.delay = delay

    async def chat(self, *a, **kw):
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        yield "来自健康节点", {"conversation_id": f"{self.key}_conv"}
        yield "", {"conversation_id": f"{self.key}_conv"}


def _reset_risk_state() -> FailoverRouter:
    router = FailoverRouter()
    tracker = router.cooldown_tracker
    tracker._cooldowns.clear()
    tracker._cooldown_severity.clear()
    tracker._failure_counts.clear()
    tracker.disabled_providers.clear()
    return router


def _set_env(**kv: str) -> None:
    for k, v in kv.items():
        os.environ[k] = v


def test_dead_ring_stops_at_the_request_budget():
    """整环全挂时，总耗时必须由**请求预算**决定，而不是候选数 × 每次上限。

    修复前：`GATEWAY_FIRST_CHUNK_TIMEOUT=1.0` + 五节点全挂 = 5.0s。
    预算设为 1.2s 时，总耗时必须落在 [1.2, 约 1.6] 区间（允许一次判决的余量），
    而不是继续乘到 5s。
    """
    router = _reset_risk_state()
    _set_env(
        GATEWAY_FIRST_CHUNK_TIMEOUT="1.0",
        GATEWAY_HANDSHAKE_BUDGET="1.2",
        GATEWAY_STREAM_IDLE_TIMEOUT="5",
    )
    flag: dict = {}
    providers = {k: _HungProvider(k, flag) for k in ("qwen", "deepseek", "doubao", "kimi", "glm")}

    async def go():
        return await router.execute_chat(
            "qwen", "Qwen", lambda k: providers[k], {"model": "Qwen"}
        )

    t0 = time.monotonic()
    error: Exception | None = None
    try:
        asyncio.run(asyncio.wait_for(go(), timeout=12.0))
    except ProviderError as exc:
        error = exc
    elapsed = time.monotonic() - t0

    assert error is not None, "整环挂死却没有任何错误"
    assert getattr(error, "status", None) == 504, error
    # 修复前实测 5.0s；预算 1.2s 必须把它压下来
    assert elapsed < 2.5, (
        f"整条请求耗时 {elapsed:.2f}s —— 仍是「候选数 × 每次上限」"
        f"（{len(providers)} × 1.0s），请求预算没有生效"
    )
    assert elapsed >= 1.0, f"预算设了 1.2s 却在 {elapsed:.2f}s 就放弃了，判决过早"
    print(f"test_dead_ring_stops_at_the_request_budget: PASSED ({elapsed:.2f}s)")


def test_budget_error_names_the_budget_and_the_attempts():
    """预算耗尽必须说清「试了几个节点、预算多少」，运维才不用去猜。"""
    router = _reset_risk_state()
    _set_env(
        GATEWAY_FIRST_CHUNK_TIMEOUT="1.0",
        GATEWAY_HANDSHAKE_BUDGET="1.2",
        GATEWAY_STREAM_IDLE_TIMEOUT="5",
    )
    flag: dict = {}
    providers = {k: _HungProvider(k, flag) for k in ("qwen", "deepseek", "doubao", "kimi", "glm")}

    async def go():
        return await router.execute_chat(
            "qwen", "Qwen", lambda k: providers[k], {"model": "Qwen"}
        )

    error: ProviderError | None = None
    try:
        asyncio.run(asyncio.wait_for(go(), timeout=12.0))
    except ProviderError as exc:
        error = exc

    assert error is not None
    # `ProviderError` 的原始信息挂在 `message` 上（与其它派发错误的取法一致）
    text = str(getattr(error, "message", error))
    assert "预算" in text or "budget" in text.lower(), f"错误没说明是预算耗尽: {text}"
    assert "1" in text, f"错误没说清预算数值: {text}"
    print("test_budget_error_names_the_budget_and_the_attempts: PASSED")


def test_budget_allows_one_healthy_slow_handshake():
    """预算不得误伤健康的慢握手：推理模型正常思考几十秒仍是健康状态。"""
    router = _reset_risk_state()
    _set_env(
        GATEWAY_FIRST_CHUNK_TIMEOUT="10",
        GATEWAY_HANDSHAKE_BUDGET="10",
        GATEWAY_STREAM_IDLE_TIMEOUT="5",
    )
    providers = {"qwen": _HealthyProvider("qwen", delay=0.6)}

    async def go():
        stream, _event, _prov = await router.execute_chat(
            "qwen", "Qwen", lambda k: providers[k], {"model": "Qwen"}
        )
        return [item async for item in stream]

    got = asyncio.run(asyncio.wait_for(go(), timeout=6.0))
    assert any("健康节点" in delta for delta, _meta in got), got
    print("test_budget_allows_one_healthy_slow_handshake: PASSED")


def test_budget_does_not_break_single_node_failover():
    """零回归：首节点超时后转到健康节点的既有路径逐值不变。"""
    router = _reset_risk_state()
    _set_env(
        GATEWAY_FIRST_CHUNK_TIMEOUT="0.3",
        GATEWAY_HANDSHAKE_BUDGET="10",
        GATEWAY_STREAM_IDLE_TIMEOUT="5",
    )
    flag: dict = {}
    providers = {
        "qwen": _HungProvider("qwen", flag),
        "glm": _HealthyProvider("glm"),
        "deepseek": _HealthyProvider("deepseek"),
    }
    router.cooldown_tracker.disabled_providers.update({"doubao", "kimi"})

    async def go():
        stream, event, _prov = await router.execute_chat(
            "qwen", "Qwen", lambda k: providers[k], {"model": "Qwen"}
        )
        got = [item async for item in stream]
        return got, event

    got, event = asyncio.run(asyncio.wait_for(go(), timeout=6.0))
    assert any("健康节点" in delta for delta, _meta in got), got
    assert event is not None and event.from_provider == "qwen", event
    assert flag.get("closed") == 1, "挂住的上游必须被关闭"
    print("test_budget_does_not_break_single_node_failover: PASSED")


def test_budget_zero_disables_it():
    """`GATEWAY_HANDSHAKE_BUDGET=0` 逐值回退为旧行为（纯按候选逐个计时）。"""
    router = _reset_risk_state()
    _set_env(
        GATEWAY_FIRST_CHUNK_TIMEOUT="0.3",
        GATEWAY_HANDSHAKE_BUDGET="0",
        GATEWAY_STREAM_IDLE_TIMEOUT="5",
    )
    flag: dict = {}
    providers = {k: _HungProvider(k, flag) for k in ("qwen", "deepseek", "doubao", "kimi", "glm")}

    async def go():
        return await router.execute_chat(
            "qwen", "Qwen", lambda k: providers[k], {"model": "Qwen"}
        )

    t0 = time.monotonic()
    try:
        asyncio.run(asyncio.wait_for(go(), timeout=12.0))
    except ProviderError:
        pass
    elapsed = time.monotonic() - t0
    assert elapsed >= 1.2, (
        f"预算关闭后应在 {len(providers)}×0.3=1.5s 左右放弃，实际 {elapsed:.2f}s —— "
        "关闭开关必须逐值回退，而不是变成另一种行为"
    )
    print(f"test_budget_zero_disables_it: PASSED ({elapsed:.2f}s)")


def run_all():
    tests = [
        test_dead_ring_stops_at_the_request_budget,
        test_budget_error_names_the_budget_and_the_attempts,
        test_budget_allows_one_healthy_slow_handshake,
        test_budget_does_not_break_single_node_failover,
        test_budget_zero_disables_it,
    ]
    for fn in tests:
        fn()
    print(f"\n✅ All handshake budget tests passed! ({len(tests)} cases)")


if __name__ == "__main__":
    run_all()