"""整环/单次握手超时的终局必须带重试抑制信号 (tests/test_ring_exhaustion_backoff.py)。

[H7] 「网关已经确定结论」的拒绝不止四类风控拒绝，还有一种最朴素的：
**这个能力环里的节点全都试过并且全都超时了**。

实测（真实 Claude Code CLI v2.1.293，对着五节点全挂的环，
`GATEWAY_FIRST_CHUNK_TIMEOUT=30`、`GATEWAY_HANDSHAKE_BUDGET=15`，同一条命令、
同一个服务端，唯一变量是响应头）：

    仅 504                       → 240s 内客户端发了 **8 次**请求（每次内部再走 5 个
                                   节点），脚本超时前用户看不到任何结论
    + `x-should-retry: false`    → **18s**、**2 次**请求，EXIT=1，用户读到完整原因

8 次重试 × 5 个候选 = 40 次打在一条已知全挂的环上的上游调用。上游看到的是
「同一条请求被反复发起」，正是风控线一直在消灭的那种风暴形态——所以这不只是
「让用户少等一会儿」，而是**别再制造无效上游流量**。

为什么这两类超时属于确定性拒绝：
- `upstream_timeout` 且**环内已无未尝试候选**（报错文案为「全部后备节点熔断」）：
  网关已经把环走完了，同一个环再走一遍不会得到不同答案；退避时长取冷却窗口
  （超时冷却 `GATEWAY_TIMEOUT_COOLDOWN`，默认 60s），这是抛出点真正知道的窗口。
- 单手节点超时、但环内还有候选：**不是**确定性拒绝（下一次可能换个节点就好了），
  绝不能带 `x-should-retry: false`，否则客户端会放弃一个仍可能成功的请求。

边界与风险模块的既有不变量一致：只抑制「重试无用」的那一类，普通上游抖动一律不抑制。
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

# [DAY-ZERO] 派发闸门现在有一道「本机有没有凭证」的硬阻断。本套件的上游全是内存假节点，
# 与凭证无关；但不摆出凭证的话，请求会在进入被测逻辑之前就以 `unconfigured` 返回，
# 于是套件测的变成一道与它无关的闸门（实测：期望 504，实得 503 unconfigured）。
# 给五个节点摆上假 token，让被测对象真的是被测对象。
for _pk in ("QWEN", "DEEPSEEK", "DOUBAO", "KIMI", "GLM"):
    os.environ[f"{_pk}_AUTH_TOKEN"] = f"test-token-{_pk.lower()}"


from fastapi.testclient import TestClient  # noqa: E402

import server as srv  # noqa: E402
from providers.failover import FailoverRouter  # noqa: E402


class _HungProvider:
    """连上就永不吐字。"""

    def __init__(self, key: str):
        self.key = key

    async def chat(self, *a, **kw):
        await asyncio.Event().wait()
        yield "", {}

    async def delete_conversation(self, *a, **kw):
        return None


def _clean(router: FailoverRouter) -> None:
    t = router.cooldown_tracker
    t._cooldowns.clear()
    t._cooldown_severity.clear()
    t._failure_counts.clear()
    t.disabled_providers.clear()


def _env(**kv: str) -> None:
    for k, v in kv.items():
        os.environ[k] = v


def test_ring_exhaustion_by_timeout_is_a_deterministic_refusal():
    """环内全部节点握手超时 ⇒ 终局必须带 `x-should-retry: false` 与真实退避窗口。

    修复前实测：504 且**没有任何**重试抑制头，客户端在 240s 内重试 8 次。
    """
    router = srv.failover_router
    _clean(router)
    _env(
        GATEWAY_FIRST_CHUNK_TIMEOUT="0.3",
        GATEWAY_HANDSHAKE_BUDGET="10",
        GATEWAY_STREAM_IDLE_TIMEOUT="5",
    )
    original = srv._make_provider
    srv._make_provider = lambda key: _HungProvider(key)
    client = TestClient(srv.app)
    try:
        r = client.post(
            "/v1/messages",
            json={"model": "claude-3-7-sonnet", "max_tokens": 16,
                  "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 504, (r.status_code, r.text[:300])
        assert r.headers.get("x-should-retry") == "false", (
            "整环握手超时的终局没有带上 x-should-retry: false —— "
            "实测这会让真实 Claude Code 在 240s 内重试 8 次（同一条件加了这个头是 2 次/18s），"
            "每次都要把整环再走一遍，正是风控线一直在消灭的无效上游流量"
        )
        ra = r.headers.get("retry-after")
        assert ra is not None, "整环超时拒绝没有携带 Retry-After"
        assert int(ra) > 0, f"Retry-After 必须是正整数秒，实际 {ra!r}"
    finally:
        srv._make_provider = original
        _clean(router)
    print("test_ring_exhaustion_by_timeout_is_a_deterministic_refusal: PASSED")


def test_healthy_ring_still_serves_and_carries_no_suppression_header():
    """零回归：健康节点正常返回，且绝不带重试抑制头。"""

    class _Healthy:
        def __init__(self, key):
            self.key = key

        async def chat(self, *a, **kw):
            yield "ok", {"conversation_id": "c1"}
            yield "", {"conversation_id": "c1"}

        async def delete_conversation(self, *a, **kw):
            return None

    router = srv.failover_router
    _clean(router)
    original = srv._make_provider
    srv._make_provider = lambda key: _Healthy(key)
    client = TestClient(srv.app)
    try:
        r = client.post(
            "/v1/chat/completions",
            json={"model": "qwen-max-latest",
                  "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 200, (r.status_code, r.text[:200])
        assert "x-should-retry" not in r.headers, (
            "健康响应上出现了重试抑制头 —— 这会让客户端放弃一次成功的请求"
        )
    finally:
        srv._make_provider = original
        _clean(router)
    print("test_healthy_ring_still_serves_and_carries_no_suppression_header: PASSED")


def run_all():
    test_ring_exhaustion_by_timeout_is_a_deterministic_refusal()
    test_healthy_ring_still_serves_and_carries_no_suppression_header()
    print("\n✅ All ring-exhaustion backoff tests passed! (2 cases)")


if __name__ == "__main__":
    run_all()