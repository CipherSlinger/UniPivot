"""步调节奏竞态测试 (tests/test_pacing_race.py)。

[M1] 问题：调用节奏有**两条**实现，各自维护自己的「上次调用时间」，却只有
下游那条的语义是真的。

  1. `server.py` 在派发前先 `traffic_pacer.acquire(dispatched_provider)`
     ——这是一段**只等待、随后立即释放**的 `async with`，它模拟了「等一下」，
     但真正的上游调用发生在后面 `execute_chat` 里。于是它对节奏的实际约束是：
     请求 A 等完 0.3s 就让开，请求 B 看到「空闲」随即也等 0.3s 让开，两条请求
     可以**同时**进入上游。整条链路上没有任何东西真正串行化调用起始时刻。

  2. `execute_chat` 内部又调 `cooldown_tracker.apply_pacing_and_acquire()`，
     但它读 `self._last_call_time` 是在**锁外**、睡觉在锁外、写回也在锁外：
     两个并发调用者读到同一个 `last`，算出同一个 `elapsed`，于是同时睡完、
     同时写回——`min_call_interval` 这一档（qwen 0.3s / deepseek 0.4s /
     kimi 0.35s）在并发下被完全绕过。

这与「同一事实被两处解释」是同一种缺陷：两处都声称在做频控，实测两处都没做到。

实证（修复前）：
  * 两个并发请求，间隔上限 0.3s，测得的**上游调用起始时刻**间隔可以接近 0；
  * `apply_pacing_and_acquire` 并发 8 路，全部在同一时刻完成（应被拉成阶梯）。

修复语义：
  * 节奏只有**一处**裁决，且裁决点紧贴上游调用（`execute_chat` 的派发循环）；
  * 该处必须**在锁内**读取、计算、并**预约**下一次可调用时刻——并发调用者
    得到的是「依次排队」而不是「同时放行」；
  * `server.py` 侧的 `traffic_pacer` 不再对同一节点重复计一次等待；它保留的是
    并发上限与统计职责，以及响应头里的观测值（不改变真实节奏）。

运行: .venv/bin/python tests/test_pacing_race.py
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
os.environ["GATEWAY_DISABLE_PACING"] = "0"  # 明确打开：本套件测的就是节奏
os.environ["GATEWAY_DISABLE_SINGLEFLIGHT"] = "1"
os.environ["GATEWAY_DISABLE_PROBATION"] = "1"
os.environ["GATEWAY_DISABLE_RISK_CORRELATION"] = "1"
os.environ["GATEWAY_DISABLE_CREDENTIAL_WATCH"] = "1"
os.environ["GATEWAY_DISABLE_CREDENTIAL_RENEWAL"] = "1"
os.environ["GATEWAY_DISABLE_DEVICE_IDENTITY"] = "1"

# [DAY-ZERO] 派发闸门有一道「本机有没有凭证」的硬阻断。本套件的上游是内存假节点，
# 与凭证无关；但不摆出凭证，请求会在进入被测逻辑前就以 `unconfigured` 返回，
# 套件便不再测它声称要测的东西。五个节点一律摆上假 token。
for _pk in ("QWEN", "DEEPSEEK", "DOUBAO", "KIMI", "GLM"):
    os.environ[f"{_pk}_AUTH_TOKEN"] = f"test-token-{_pk.lower()}"

os.environ["GATEWAY_DISABLE_QUARANTINE_HARD_STOP"] = "1"
os.environ["GATEWAY_DISABLE_LONG_AVOIDANCE_FAST_FAIL"] = "1"

from providers import failover as fov  # noqa: E402
from providers.failover import FailoverRouter  # noqa: E402


# ------------------------------------------------------------------ 替身

class _StampProvider:
    """在**被调用那一刻**记录时间戳的健康上游。"""

    def __init__(self, key: str, stamps: list):
        self.key = key
        self.stamps = stamps

    async def chat(self, *a, **kw):
        self.stamps.append(time.time())
        yield "ok", {"conversation_id": f"{self.key}_conv"}
        yield "", {"conversation_id": f"{self.key}_conv"}


def _fresh_router() -> FailoverRouter:
    router = FailoverRouter()
    tracker = router.cooldown_tracker
    tracker.reset()
    tracker._cooldowns.clear()
    tracker._cooldown_severity.clear()
    tracker._failure_counts.clear()
    tracker.disabled_providers.clear()
    return router


def _tighten(provider_key: str, interval: float) -> fov.ProviderRiskMeta:
    """把某节点的步调间隔固定成测试值（无抖动），返回原值以便还原。"""
    original = fov.PROVIDER_RISK_SPECS[provider_key]
    fov.PROVIDER_RISK_SPECS[provider_key] = fov.ProviderRiskMeta(
        provider_name=original.provider_name,
        risk_level=original.risk_level,
        min_call_interval=interval,
        jitter_range=(0.0, 0.0),
        max_concurrency=original.max_concurrency,
        default_cooldown=original.default_cooldown,
    )
    return original


async def _drain(stream) -> None:
    async for _delta, _meta in stream:
        pass


# ------------------------------------------------------------------ 用例

def test_pacing_serializes_concurrent_dispatch():
    """并发派发必须被拉成阶梯：上游调用起始时刻的间隔不得小于节奏间隔。"""
    interval = 0.30
    original = _tighten("qwen", interval)
    try:
        router = _fresh_router()
        stamps: list = []
        provider = _StampProvider("qwen", stamps)
        router.cooldown_tracker.disabled_providers.update({"deepseek", "doubao", "kimi", "glm"})

        async def one():
            stream, _ev, _p = await router.execute_chat(
                "qwen", "Qwen", lambda k: provider, {"model": "Qwen"}
            )
            await _drain(stream)

        async def go():
            # 先打一枪，让第一个调用者占住「下一次可调用时刻」
            await asyncio.gather(one(), one(), one())

        asyncio.run(asyncio.wait_for(go(), timeout=10.0))
        assert len(stamps) == 3, stamps
        gaps = [b - a for a, b in zip(stamps, stamps[1:])]
        worst = min(gaps)
        assert worst >= interval * 0.8, (
            f"并发派发绕过了节奏：最小间隔 {worst:.3f}s < {interval * 0.8:.3f}s"
            f"（三次调用时刻: {[round(s - stamps[0], 3) for s in stamps]}）"
        )
    finally:
        fov.PROVIDER_RISK_SPECS["qwen"] = original
    print("test_pacing_serializes_concurrent_dispatch: PASSED")


def test_pacing_decision_is_under_lock():
    """节奏裁决必须在锁内完成读取 + 预约，而不是读锁外的快照。"""
    original = _tighten("qwen", 0.12)
    try:
        tracker = _fresh_router().cooldown_tracker

        async def go():
            await asyncio.gather(
                *[tracker.apply_pacing_and_acquire("qwen") for _ in range(4)]
            )

        start = time.time()
        asyncio.run(asyncio.wait_for(go(), timeout=10.0))
        elapsed = time.time() - start
        # 4 个调用者、间隔 0.12s：串行化后至少 ~0.36s；并发放行只会是 ~0.12s
        assert elapsed >= 0.12 * 3 * 0.75, (
            f"apply_pacing_and_acquire 并发下未串行化：4 路共耗时 {elapsed:.3f}s，"
            f"应至少 {0.12 * 3 * 0.75:.3f}s"
        )
    finally:
        fov.PROVIDER_RISK_SPECS["qwen"] = original
    print("test_pacing_decision_is_under_lock: PASSED")


def test_single_pacing_authority():
    """节奏只能有一处裁决：`server.py` 不应再对同一节点重复注入等待。"""
    server_src = (Path(__file__).resolve().parent.parent / "server.py").read_text(encoding="utf-8")
    assert "apply_pacing_and_acquire" in Path(
        Path(__file__).resolve().parent.parent / "providers" / "failover.py"
    ).read_text(encoding="utf-8")
    # server 侧的 acquire 若仍带 await 等待，则两处又在同时计一次
    for marker in ("await traffic_pacer.acquire", "async with traffic_pacer.acquire"):
        assert marker not in server_src, (
            f"server.py 仍在执行第二处节奏等待（{marker!r}）：节奏裁决必须唯一，"
            f"否则两处各自为政、并发下互相绕过"
        )
    print("test_single_pacing_authority: PASSED")


def run_all():
    test_pacing_serializes_concurrent_dispatch()
    test_pacing_decision_is_under_lock()
    test_single_pacing_authority()
    print("\n✅ All pacing race tests passed!")


if __name__ == "__main__":
    run_all()