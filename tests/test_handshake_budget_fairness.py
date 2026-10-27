"""预算不得让「第一个慢候选」独占全部机会 (tests/test_handshake_budget_fairness.py)。

[BUDGET-2] 上一轮加的 `GATEWAY_HANDSHAKE_BUDGET` 把「整条请求的握手总时长」
限住了，但它把预算**一次性全给第一个候选**：默认配置
(`GATEWAY_FIRST_CHUNK_TIMEOUT=180`、`GATEWAY_HANDSHAKE_BUDGET=180`) 下，
`min(180, 180)` = 180，第一个节点挂住 180s 就耗尽全部预算，第二到第五个候选
**一次都不再尝试**。

实测（`GATEWAY_FIRST_CHUNK_TIMEOUT=30`、`GATEWAY_HANDSHAKE_BUDGET=15`，
真实 `claude -p` 对着五节点全挂的环）：

    [握手预算] 总预算 15.0s 已耗尽（已尝试 1 个节点），停止派发以保证客户端等待有上界。

这是**修复前不存在**的行为：旧代码会逐个尝试五个候选（900s），虽然慢，但
「第一个节点慢、第二个节点快」这种最常见的真实故障形态能自愈。新代码把它变成
了「一慢全断」——可用性低于修复前，方向错了。

正确的语义：预算是一个**总量**，而不是第一个候选的额度。剩下的每个未尝试候选
都必须至少拿到一份公平的份额，否则预算就从「给等待上界」变成了「只许试一个」。
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

# [DAY-ZERO] 本套件用内存里的假上游驱动 `execute_chat`，与凭证毫无关系；但派发闸门
# 现在有一道「本机有没有凭证」的硬阻断。不摆出凭证的话，请求会在进入预算逻辑之前
# 就以 `unconfigured` 返回（实测：`第一个候选超时后请求直接失败（unconfigured）`），
# 于是这个套件测的不再是预算，而是一道与它无关的闸门。给五个节点摆上假 token，
# 让被测对象真的是被测对象。
for _pk in ("QWEN", "DEEPSEEK", "DOUBAO", "KIMI", "GLM"):
    os.environ[f"{_pk}_AUTH_TOKEN"] = f"test-token-{_pk.lower()}"

from providers.failover import FailoverRouter, ProviderError  # noqa: E402


class _HungProvider:
    def __init__(self, key: str):
        self.key = key

    async def chat(self, *a, **kw):
        await asyncio.Event().wait()
        yield "", {}


class _HealthyProvider:
    def __init__(self, key: str):
        self.key = key
        self.calls = 0

    async def chat(self, *a, **kw):
        self.calls += 1
        yield "来自健康节点", {"conversation_id": f"{self.key}_conv"}
        yield "", {"conversation_id": f"{self.key}_conv"}


def _router() -> FailoverRouter:
    router = FailoverRouter()
    t = router.cooldown_tracker
    t._cooldowns.clear()
    t._cooldown_severity.clear()
    t._failure_counts.clear()
    t.disabled_providers.clear()
    return router


def _env(**kv: str) -> None:
    for k, v in kv.items():
        os.environ[k] = v


def test_budget_is_shared_so_a_late_candidate_still_gets_a_chance():
    """第一个候选慢、第二个候选健康时，必须能自愈到第二个。

    修复前（预算全给第一个）：只尝试 1 个节点，请求直接 504。
    修复后：预算被公平分配，健康的第二个候选仍被尝试并成功。
    """
    router = _router()
    _env(
        GATEWAY_FIRST_CHUNK_TIMEOUT="3.0",
        GATEWAY_HANDSHAKE_BUDGET="3.0",
        GATEWAY_STREAM_IDLE_TIMEOUT="5",
    )
    hung = _HungProvider("qwen")
    healthy = _HealthyProvider("glm")
    providers = {"qwen": hung, "glm": healthy, "deepseek": healthy,
                 "doubao": healthy, "kimi": healthy}
    router.cooldown_tracker.disabled_providers.update({"doubao", "kimi", "deepseek"})

    async def go():
        stream, event, _prov = await router.execute_chat(
            "qwen", "Qwen", lambda k: providers[k], {"model": "Qwen"}
        )
        return [item async for item in stream], event

    try:
        got, event = asyncio.run(asyncio.wait_for(go(), timeout=20.0))
    except ProviderError as exc:
        raise AssertionError(
            f"第一个候选超时后请求直接失败（{exc.err_type}），"
            f"健康的第二候选一次都没被尝试——预算被第一个候选独占"
        )

    assert any("健康节点" in d for d, _m in got), got
    assert healthy.calls >= 1, "健康的第二候选没有被尝试"
    print("test_budget_is_shared_so_a_late_candidate_still_gets_a_chance: PASSED",
          f"(healthy called {healthy.calls}x)")


def test_shared_budget_still_bounds_total_time():
    """公平分配之后，总时长仍然必须有上界（预算的核心承诺不能丢）。

    五个节点全挂、预算 1.5s：总耗时仍须远小于「5 × 单次上限」。
    """
    router = _router()
    _env(
        GATEWAY_FIRST_CHUNK_TIMEOUT="5.0",
        GATEWAY_HANDSHAKE_BUDGET="1.5",
        GATEWAY_STREAM_IDLE_TIMEOUT="5",
    )
    providers = {k: _HungProvider(k)
                 for k in ("qwen", "deepseek", "doubao", "kimi", "glm")}

    async def go():
        return await router.execute_chat(
            "qwen", "Qwen", lambda k: providers[k], {"model": "Qwen"}
        )

    t0 = time.monotonic()
    try:
        asyncio.run(asyncio.wait_for(go(), timeout=20.0))
        raise AssertionError("整环挂死却没有任何错误")
    except ProviderError as exc:
        assert exc.status == 504, exc
    elapsed = time.monotonic() - t0
    assert elapsed < 4.0, (
        f"总耗时 {elapsed:.2f}s —— 公平分配后预算不再约束总时长"
    )
    print(f"test_shared_budget_still_bounds_total_time: PASSED ({elapsed:.2f}s)")


def test_fast_failures_leave_the_whole_remainder_to_the_survivor():
    """先失败的候选必须**把时间还回去**，让后面的节点拿到更大份额。

    这是公平分摊的安全性质：真实故障里前几个节点通常「立刻断/立刻 5xx」，
    几十毫秒就结束，若它们预占了份额而不归还，真正健康的那个节点反而只能拿到
    一小份、被误判为超时。份额按**剩余预算 / 剩余未尝试数**计算，每失败一次
    分母就变小，所以最后一个候选恒拿到全部剩余预算。

    这里用「4 个立刻失败 + 1 个需要 2.5s 才吐首包」验证：若份额不归还，
    健康节点只有 budget/5 = 1.0s，会被杀掉。
    """
    router = _router()
    _env(
        GATEWAY_FIRST_CHUNK_TIMEOUT="30",
        GATEWAY_HANDSHAKE_BUDGET="5.0",
        GATEWAY_STREAM_IDLE_TIMEOUT="5",
    )

    class _FastFail:
        def __init__(self, key):
            self.key = key

        async def chat(self, *a, **kw):
            raise RuntimeError(f"{self.key} 立刻失败")
            yield "", {}  # noqa: unreachable - 让它保持异步生成器

    class _SlowButHealthy:
        def __init__(self, key):
            self.key = key

        async def chat(self, *a, **kw):
            await asyncio.sleep(2.5)
            yield "慢但健康", {"conversation_id": f"{self.key}_c"}
            yield "", {"conversation_id": f"{self.key}_c"}

    # FLAGSHIP 环序固定为 qwen -> glm -> kimi -> doubao -> deepseek，因此把慢节点放在
    # **最后一位**：前面四个各自立刻失败，每失败一个分母就小一，最后一位拿全部剩余。
    providers = {
        "qwen": _FastFail("qwen"),
        "glm": _FastFail("glm"),
        "kimi": _FastFail("kimi"),
        "doubao": _FastFail("doubao"),
        "deepseek": _SlowButHealthy("deepseek"),
    }

    async def go():
        stream, _e, _p = await router.execute_chat(
            "qwen", "Qwen", lambda k: providers[k], {"model": "Qwen"}
        )
        return [item async for item in stream]

    try:
        got = asyncio.run(asyncio.wait_for(go(), timeout=25.0))
    except ProviderError as exc:
        raise AssertionError(
            f"健康但较慢的候选被误杀（{exc.err_type}）：失败候选没有把预占的"
            f"份额还回来，导致它只拿到 budget/5 = 1.0s，而它需要 2.5s"
        )
    assert any("慢但健康" in d for d, _m in got), got
    print("test_fast_failures_leave_the_whole_remainder_to_the_survivor: PASSED")


def run_all():
    test_budget_is_shared_so_a_late_candidate_still_gets_a_chance()
    test_shared_budget_still_bounds_total_time()
    test_fast_failures_leave_the_whole_remainder_to_the_survivor()
    print("\n✅ All handshake budget fairness tests passed! (3 cases)")


if __name__ == "__main__":
    run_all()