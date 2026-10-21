"""请求超时边界与「超时不是风控」测试 (tests/test_request_timeouts.py)。

[M2] 问题：整条链路上**没有任何超时边界**属于网关自己。
  * 握手（等首包）是一个裸 `await`，唯一的边界是 httpx 的 600 秒读超时；
  * 流式阶段按帧拉取，httpx 的 read timeout 是**每个 socket 读**的，每到一个字节
    就重置一次，因此一个每 599 秒吐一字节、或只吐 `: keepalive` 的上游可以无限期
    占住连接；
  * 这段时间里，节点并发信号量、观察期串行闸门、负载均衡配额和一条 HTTP/2 连接
    全被一个「半死不活」的上游持有——同账号下的其它请求会被排在后面，客户端则因为
    网关自己的心跳机制（`sse_keepalive` 无上限）永远不会放弃。

于是「稳定性差」的一半来自上游故障无法被**及时判定**：它既不成功、也不失败，
只是永久挂着。本套件钉住四件事：

  1. 握手超时必须被判为上游故障并**故障转移**到健康节点，且资源（并发槽位）在
     失败路径上被归还——否则超时只是把泄漏从「永久」变成「每次一次」；
  2. 流式空闲超时必须在**已经开流之后**触发，按同一条身份报错，并且上游生成器被
     显式关闭（不留幽灵任务/连接）；
  3. `=0` 关闭两个边界，逐值回退为旧行为（零回归）；
  4. 超时**不是风控拦截**：不写风控台账、不给节点上风控成因的冷却，档案研判
     报 `timeout` 而**不是** `waf`——否则运维会被建议去登录一个从未拦截过他的
     平台、甚至去找验证码。

运行: .venv/bin/python tests/test_request_timeouts.py
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

from providers.failover import (  # noqa: E402
    FailoverRouter,
    ProviderError,
    get_risk_ledger,
)
from providers.risk_dossier import RiskDossier  # noqa: E402


# ------------------------------------------------------------------ 测试替身

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


class _TrickleProvider:
    """首包正常，随后长时间沉默（半死不活的上游）。"""

    def __init__(self, key: str, closed_flag: dict):
        self.key = key
        self.flag = closed_flag

    async def chat(self, *a, **kw):
        try:
            yield "首包", {"conversation_id": f"{self.key}_conv"}
            await asyncio.Event().wait()
            yield "", {}
        finally:
            self.flag["closed"] = self.flag.get("closed", 0) + 1


class _HealthyProvider:
    def __init__(self, key: str):
        self.key = key
        self.calls = 0

    async def chat(self, *a, **kw):
        self.calls += 1
        yield "来自健康节点", {"conversation_id": f"{self.key}_conv"}
        yield "", {"conversation_id": f"{self.key}_conv"}


# ------------------------------------------------------------------ 工具

_ALL = ("qwen", "deepseek", "doubao", "kimi", "glm")


def _reset_risk_state() -> None:
    router = FailoverRouter()
    tracker = router.cooldown_tracker
    tracker._cooldowns.clear()
    tracker._cooldown_severity.clear()
    tracker._failure_counts.clear()
    tracker.disabled_providers.clear()
    for lease in list(getattr(tracker.quarantine, "_leases", {}) or {}):
        try:
            tracker.quarantine.clear(lease)
        except Exception:  # noqa: BLE001
            pass
    return router


def _set_env(**kv: str) -> None:
    for k, v in kv.items():
        os.environ[k] = v


async def _drain(stream):
    out = []
    async for delta, meta in stream:
        out.append(delta)
    return out


# ------------------------------------------------------------------ 用例

def test_handshake_timeout_fails_over_and_releases_resources():
    """握手挂住 → 判为上游故障、故障转移到健康节点、并发槽位被归还。"""
    router = _reset_risk_state()
    _set_env(
        GATEWAY_FIRST_CHUNK_TIMEOUT="0.25",
        GATEWAY_STREAM_IDLE_TIMEOUT="5",
        GATEWAY_DISABLE_QUARANTINE_HARD_STOP="1",
        GATEWAY_DISABLE_LONG_AVOIDANCE_FAST_FAIL="1",
    )
    hung_flag: dict = {}
    healthy = _HealthyProvider("glm")

    providers = {"qwen": _HungProvider("qwen", hung_flag), "glm": healthy}

    def factory(key):
        return providers[key]

    async def judge():
        stream, event, provider = await router.execute_chat(
            "qwen", "Qwen", lambda k: factory(k), {"model": "Qwen"}
        )
        text = "".join(await _drain(stream))
        return text, event, provider

    router.cooldown_tracker.disabled_providers.update({"deepseek", "doubao", "kimi"})
    sem_limit = router.cooldown_tracker.get_semaphore("qwen")._value
    text, event, provider = asyncio.run(
        asyncio.wait_for(judge(), timeout=3.0)
    )
    assert text == "来自健康节点", text
    assert event is not None and event.to_provider == "glm", "必须故障转移到健康节点"
    assert hung_flag.get("closed") == 1, "挂住的上游必须被关闭，不能留成幽灵连接"

    # 失败节点进入冷却，成因是 timeout 而非风控
    tracker = router.cooldown_tracker
    assert tracker.get_remaining_cooldown("qwen") > 0, "超时的节点必须被避让"
    assert tracker.cooldown_cause("qwen") == "timeout", (
        f"超时的成因应记为 timeout，实际 {tracker.cooldown_cause('qwen')!r}"
    )

    # 并发槽位必须归还：握手超时发生在信号量交接之前，若失败路径漏释放，
    # 每挂一次就永久吃掉一个槽位，节点最终会被幽灵请求占满而永久不可用。
    sem = router.cooldown_tracker.get_semaphore("qwen")
    assert sem._value == sem_limit, (
        f"握手超时后并发槽位必须归还：期望 {sem_limit}，实际 {sem._value}"
    )
    print("test_handshake_timeout_fails_over_and_releases_resources: PASSED")


def test_handshake_timeout_is_not_recorded_as_risk():
    """超时不得写入风控台账（它不是风控拦截），也不得按人机验证处置。"""
    router = _reset_risk_state()
    _set_env(GATEWAY_FIRST_CHUNK_TIMEOUT="0.2", GATEWAY_STREAM_IDLE_TIMEOUT="5")
    flag: dict = {}
    providers = {"qwen": _HungProvider("qwen", flag), "glm": _HealthyProvider("glm")}

    async def go():
        stream, _, _ = await router.execute_chat(
            "qwen", "Qwen", lambda k: providers[k], {"model": "Qwen"}
        )
        await _drain(stream)

    router.cooldown_tracker.disabled_providers.update({"deepseek", "doubao", "kimi"})
    ledger = get_risk_ledger()
    before = sum(
        int(v.get("count", 0))
        for v in (ledger.recent_interceptions(hours=24.0).get("per_provider") or {}).values()
    )
    asyncio.run(asyncio.wait_for(go(), timeout=3.0))
    after = sum(
        int(v.get("count", 0))
        for v in (ledger.recent_interceptions(hours=24.0).get("per_provider") or {}).values()
    )
    assert after == before, (
        f"超时是上游故障，不是风控拦截，不得入台账（{before} -> {after}）"
    )
    print("test_handshake_timeout_is_not_recorded_as_risk: PASSED")


def test_stream_idle_timeout_after_first_chunk():
    """开流之后上游沉默 → 空闲超时报错、上游被关闭、不进入无限等待。"""
    router = _reset_risk_state()
    _set_env(GATEWAY_FIRST_CHUNK_TIMEOUT="5", GATEWAY_STREAM_IDLE_TIMEOUT="0.25")
    flag: dict = {}
    providers = {"qwen": _TrickleProvider("qwen", flag)}

    async def go():
        stream, _, _ = await router.execute_chat(
            "qwen", "Qwen", lambda k: providers[k], {"model": "Qwen"}
        )
        got = []
        try:
            async for delta, _meta in stream:
                got.append(delta)
        except ProviderError as exc:
            return got, exc
        return got, None

    router.cooldown_tracker.disabled_providers.update({"deepseek", "doubao", "kimi", "glm"})
    got, exc = asyncio.run(asyncio.wait_for(go(), timeout=3.0))
    assert got == ["首包"], f"首包必须已交付给客户端，实际 {got}"
    assert exc is not None, "空闲超时必须报错而不是永久挂起"
    assert getattr(exc, "status", None) == 504, exc
    assert getattr(exc, "err_type", None) == "upstream_timeout", exc
    assert flag.get("closed") == 1, "沉默的上游必须被关闭"
    # 中途断流被记入冷却，且成因是 timeout
    tracker = router.cooldown_tracker
    assert tracker.cooldown_cause("qwen") == "timeout", (
        f"流式空闲超时的成因应记为 timeout，实际 {tracker.cooldown_cause('qwen')!r}"
    )
    print("test_stream_idle_timeout_after_first_chunk: PASSED")


def test_zero_disables_both_boundaries():
    """两个超时都设为 0 → 逐值回退为旧行为（不受限）。"""
    router = _reset_risk_state()
    _set_env(GATEWAY_FIRST_CHUNK_TIMEOUT="0", GATEWAY_STREAM_IDLE_TIMEOUT="0")
    assert router._first_chunk_timeout() == 0.0
    assert router._stream_idle_timeout() == 0.0

    providers = {"qwen": _HealthyProvider("qwen")}

    async def go():
        stream, _, _ = await router.execute_chat(
            "qwen", "Qwen", lambda k: providers[k], {"model": "Qwen"}
        )
        return "".join(await _drain(stream))

    router.cooldown_tracker.disabled_providers.update({"deepseek", "doubao", "kimi", "glm"})
    text = asyncio.run(asyncio.wait_for(go(), timeout=3.0))
    assert text == "来自健康节点", text
    print("test_zero_disables_both_boundaries: PASSED")


def test_dossier_reports_timeout_not_waf():
    """档案研判：超时避让的节点报 timeout，且明确「重新登录无效」。"""
    router = _reset_risk_state()
    tracker = router.cooldown_tracker
    tracker._cooldowns["qwen"] = time.time() + 120.0
    tracker._cooldown_severity["qwen"] = "timeout"

    dossier = RiskDossier(
        ledger=get_risk_ledger(), quarantine=tracker.quarantine, cooldown_tracker=tracker
    )
    entry = dossier.provider_entry("qwen")
    verdict = entry["avoidance"]
    assert verdict["verdict"] == "timeout", (
        f"超时不是 waf，档案必须报 timeout，实际 {verdict['verdict']!r}"
    )
    assert verdict["recoverable_by_relogin"] is False, "超时重登无效，绝不能建议重登"
    assert entry["cooldown_cause"] == "timeout", entry.get("cooldown_cause")
    advice = " ".join(entry["recommended_actions"])
    assert "无需重新登录" in advice or "重新登录无效" in advice, advice
    assert "滑块" not in advice, f"超时不该建议完成滑块验证：{advice}"
    print("test_dossier_reports_timeout_not_waf: PASSED")


def run_all():
    test_handshake_timeout_fails_over_and_releases_resources()
    test_handshake_timeout_is_not_recorded_as_risk()
    test_stream_idle_timeout_after_first_chunk()
    test_zero_disables_both_boundaries()
    test_dossier_reports_timeout_not_waf()
    print("\n✅ All request timeout tests passed!")


if __name__ == "__main__":
    run_all()