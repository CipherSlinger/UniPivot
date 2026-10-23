"""握手总预算的资源回收与「预算裁短超时」误判审计 (tests/test_budget_resource_reclaim.py)。

[CLAUDE.md 第 59 节] 新加的 `GATEWAY_HANDSHAKE_BUDGET` 在 `execute_chat` 的派发循环
**顶部**（`try` 之外）判决预算耗尽并 raise。本模块的历史缺陷都是同一条：
**资源在响应体存在之前就被持有，而释放写在 body 生成器的 finally 里**（C2）。
于是这条新出口必须先被证明：它 raise 的那一刻，一条资源都不在手上；而每个失败候选
在它自己的失败分支上，也确实把节点并发信号量与上游流还了回去。

本案测三件事：

  1. **回收**：整环挂住、预算耗尽时，每个被尝试的候选的上游生成器 `finally` 都把
     自己的连接关掉了（打开次数 == 关闭次数），五个节点的并发信号量回到初始值，
     调用方（三个协议端点）的负载均衡配额归零。全部断言**计数器**，不断言「没抛异常」
     —— 泄漏在构造上不可见（`SKILL.md` 的硬性工作方法）。
  2. **预算裁短超时不得把健康节点记成失败**：`[BUDGET-2]` 的公平分摊让第一个候选只
     拿到「剩余预算 / 剩余候选数」这一份。默认配置（单次 180s、总预算 180s、五节点）
     下第一份只有 36s，而推理模型正常思考几十秒是**健康**的（`_first_chunk_timeout`
     的 docstring 明确写了这一点）。若被裁短的超时仍按「节点超时」记账，一个健康节点
     会拿到 60s 冷却并被后续请求绕开 —— 这正是本项目反复记录的「把我们自己的决定
     写成上游的失败」。
  3. **0.05s 地板不得发起注定失败的调用**：份额低于可用下限时仍按 `max(share, 0.05)`
     发起一次 50ms 的尝试，对任何真实上游都是必败；它只换来一次白打的上游调用和一次
     假的超时冷却。此时应当像其它零上游拒绝一样如实报错（section 59 的原话：
     「不再发起注定超时的尝试」）。

运行: .venv/bin/python tests/test_budget_resource_reclaim.py
"""

from __future__ import annotations

import asyncio
import os
import sys
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

_KEYS = ("qwen", "deepseek", "doubao", "kimi", "glm")


# ------------------------------------------------------------------ 测试替身

class _TrackedProvider:
    """记录「打开了几条上游流 / 真正关闭了几条」，以及首包延迟。

    `delay=None` 表示连上了但永不产出（挂死）；数字表示首包延迟（健康但慢）。
    """

    def __init__(self, key: str, delay: float | None = None):
        self.key = key
        self.delay = delay
        self.opened = 0
        self.closed = 0

    async def chat(self, *args, **kwargs):
        self.opened += 1
        try:
            if self.delay is None:
                await asyncio.Event().wait()
            else:
                await asyncio.sleep(self.delay)
            yield "来自健康节点", {"conversation_id": f"{self.key}_conv"}
            yield "", {"conversation_id": f"{self.key}_conv"}
        finally:
            self.closed += 1

    async def delete_conversation(self, *args, **kwargs):
        return None


# ------------------------------------------------------------------ 工具

def _router() -> FailoverRouter:
    router = FailoverRouter()
    tracker = router.cooldown_tracker
    tracker._cooldowns.clear()
    tracker._cooldown_severity.clear()
    tracker._failure_counts.clear()
    tracker.disabled_providers.clear()
    tracker.reset()
    return router


def _env(**kv: str) -> None:
    for k, v in kv.items():
        os.environ[k] = v


def _sem_snapshot(router: FailoverRouter) -> dict:
    return {k: router.cooldown_tracker.get_semaphore(k)._value for k in _KEYS}


def _run_to_error(router: FailoverRouter, providers: dict, timeout: float = 20.0):
    async def go():
        return await router.execute_chat(
            "qwen", "Qwen", lambda k: providers[k], {"model": "Qwen"}
        )

    try:
        return asyncio.run(asyncio.wait_for(go(), timeout=timeout)), None
    except ProviderError as exc:
        return None, exc


# ------------------------------------------------------------------ 用例

def test_budget_exhaustion_reclaims_every_candidate_resource():
    """预算耗尽 raise 的那一刻：上游流全部关闭、信号量回到初始值。

    「预算耗尽」这条出口在 `try` 之外（第 59 节），所以它本身不持有资源；真正的风险
    在它**之前**的每个失败候选上——那五条路径各自握着信号量、开着上游流。
    实测（修复前）：5 开 / 5 关，信号量无泄漏（本用例先绿，钉住的是不变量）。
    """
    router = _router()
    _env(
        GATEWAY_FIRST_CHUNK_TIMEOUT="30",
        GATEWAY_HANDSHAKE_BUDGET="1.2",
        GATEWAY_STREAM_IDLE_TIMEOUT="5",
    )
    providers = {k: _TrackedProvider(k) for k in _KEYS}
    before = _sem_snapshot(router)

    _result, error = _run_to_error(router, providers)

    assert error is not None, "整环挂死却没有任何错误"
    assert getattr(error, "status", None) == 504, error

    leaks = []
    for key in _KEYS:
        p = providers[key]
        if p.opened == 0:
            continue  # 未被派发的节点不参与本断言
        if p.closed != p.opened:
            leaks.append(f"{key}: 上游流未关闭 {p.opened} 开 / {p.closed} 关")
    after = _sem_snapshot(router)
    for key in _KEYS:
        if after[key] != before[key]:
            leaks.append(f"{key}: 节点信号量泄漏 {before[key]} -> {after[key]}")

    attempted = [k for k in _KEYS if providers[k].opened]
    assert attempted, "预算耗尽的用例里一个候选都没被尝试，测不到任何回收路径"
    assert not leaks, "预算耗尽路径上的资源泄漏:\n" + "\n".join(leaks)
    print(
        "test_budget_exhaustion_reclaims_every_candidate_resource: PASSED",
        f"(attempted {attempted}, opened={sum(p.opened for p in providers.values())},"
        f" closed={sum(p.closed for p in providers.values())})",
    )


def test_budget_capped_timeout_does_not_cool_a_healthy_node():
    """被总预算裁短的超时是**请求**的边界，不是**节点**的失败。

    预算 0.3s、剩余候选 2 个 → 首个候选只分到 0.1s。首个候选健康但首包要 0.4s：
    它在 0.1s 处被判超时是预算造成的，节点本身没有任何证据表明它有故障。
    修复前实测：qwen 冷却 60.0s、成因 timeout、failure_count 1（健康节点被绕开 60s）。
    修复后：qwen 无冷却、failure_count 0，请求照常故障转移到健康候选并成功。
    """
    router = _router()
    _env(
        GATEWAY_FIRST_CHUNK_TIMEOUT="30",
        GATEWAY_HANDSHAKE_BUDGET="0.3",
        GATEWAY_STREAM_IDLE_TIMEOUT="5",
    )
    slow_but_healthy = _TrackedProvider("qwen", delay=0.4)
    fast = _TrackedProvider("glm", delay=0.0)
    providers = {"qwen": slow_but_healthy, "glm": fast, "deepseek": fast,
                 "doubao": fast, "kimi": fast}
    router.cooldown_tracker.disabled_providers.update({"deepseek", "doubao", "kimi"})

    async def go():
        stream, event, _prov = await router.execute_chat(
            "qwen", "Qwen", lambda k: providers[k], {"model": "Qwen"}
        )
        return [item async for item in stream], event

    got, event = asyncio.run(asyncio.wait_for(go(), timeout=20.0))
    assert any("健康节点" in delta for delta, _meta in got), got
    assert event is not None and event.to_provider == "glm", event

    tracker = router.cooldown_tracker
    cooldown = tracker.get_remaining_cooldown("qwen")
    cause = tracker.cooldown_cause("qwen")
    failures = tracker._failure_counts.get("qwen", 0)
    assert cooldown == 0.0 and failures == 0, (
        f"总预算裁短导致的超时被记成了节点失败：qwen 冷却 {cooldown:.1f}s、"
        f"成因 {cause!r}、failure_count {failures}。\n"
        f"预算份额 0.1s 是**请求**的边界，节点在 0.4s 内本可以正常应答；"
        f"把它记成上游故障会让一个健康节点无谓地冷却 60s。"
    )
    assert slow_but_healthy.opened == 1 and slow_but_healthy.closed == 1
    print(
        "test_budget_capped_timeout_does_not_cool_a_healthy_node: PASSED",
        f"(qwen cooldown={cooldown:.1f}s failure_count={failures})",
    )


def test_budget_floor_never_dispatches_a_doomed_attempt():
    """份额低于可用下限时不得按 0.05s 地板发起一次必败的上游调用。

    预算 0.12s、五候选 → 首份 0.024s，低于 0.05s 地板。原实现把它抬到 0.05s 派发，
    对任何真实上游都是必败，结果是「一次白打的上游调用 + 一次假的超时冷却」。
    正确行为与 section 59 的其它零上游拒绝一致：如实报错、零上游调用、零冷却。

    修复前实测：五个候选全被以 0.05s 超时派发、各记 60s 超时冷却、5 次上游调用。
    """
    router = _router()
    _env(
        GATEWAY_FIRST_CHUNK_TIMEOUT="30",
        GATEWAY_HANDSHAKE_BUDGET="0.12",
        GATEWAY_STREAM_IDLE_TIMEOUT="5",
    )
    # 健康但需要 0.2s 应答：在 50ms 的必败尝试里同样会被判超时
    providers = {k: _TrackedProvider(k, delay=0.2) for k in _KEYS}

    _result, error = _run_to_error(router, providers)

    assert error is not None, "必败的尝试既没有成功也没有报错"
    text = str(getattr(error, "message", error))
    assert "预算" in text, f"错误没有说明是预算不足导致不派发: {text}"

    calls = sum(p.opened for p in providers.values())
    assert calls == 0, (
        f"预算份额低于可用下限时仍然发起了 {calls} 次注定失败的上游调用"
        f"（0.05s 地板），只换来白打的凭证流量与假冷却"
    )
    cooled = {k: router.cooldown_tracker.get_remaining_cooldown(k) for k in _KEYS}
    assert not any(cooled.values()), f"零上游拒绝不得给任何节点上冷却: {cooled}"
    print(
        "test_budget_floor_never_dispatches_a_doomed_attempt: PASSED",
        f"(upstream calls={calls}, cooled={cooled})",
    )


# ------------------------------------------------------------------ 调用方回收

def _asgi_drive(provider_map: dict, path: str, body: bytes) -> dict:
    """驱动真实 ASGI 栈（三个协议端点各一次），返回观测到的计数器。"""
    import server as srv

    srv._make_provider = lambda key, client=None: provider_map[key]

    async def _run():
        for key in _KEYS:
            metrics = srv.load_balancer.get_metrics(key)
            while metrics.active_concurrency > 0:
                metrics.release(success=False)

        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.4"},
            "http_version": "1.1",
            "method": "POST",
            "headers": [(b"content-type", b"application/json")],
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "root_path": "",
            "client": ("127.0.0.1", 1),
            "server": ("127.0.0.1", 8000),
            "scheme": "http",
        }
        state = {"body_sent": False, "status": None}

        async def receive():
            if not state["body_sent"]:
                state["body_sent"] = True
                return {"type": "http.request", "body": body, "more_body": False}
            await asyncio.sleep(0.1)
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == "http.response.start":
                state["status"] = message["status"]

        try:
            await asyncio.wait_for(srv.app(scope, receive, send), timeout=25)
        except BaseException:  # noqa: BLE001 - 结束路径本身不是本用例的断言对象
            pass
        # 给独立收尾任务留出调度机会（与 test_stream_start_reclaim 同一处理）
        for _ in range(50):
            await asyncio.sleep(0.01)

        return {
            "status": state["status"],
            "lb": {k: srv.load_balancer.get_metrics(k).active_concurrency for k in _KEYS},
            "sem": {k: srv.failover_router.cooldown_tracker.get_semaphore(k)._value
                    for k in _KEYS},
        }

    return asyncio.run(_run())


def test_callers_reclaim_lb_quota_when_budget_refuses():
    """预算拒绝时，三个协议端点的负载均衡配额都必须归零。

    调用方（`server.py`）的 LB 配额获取点有两类：流式路径在 `gen()` **内部**获取
    （`execute_chat` 在其之前，拒绝时根本没 acquire），非流式路径在 `execute_chat`
    之前 acquire 并由 `finally` 归还。两条都必须归零，且节点信号量不得残留。
    """
    os.environ["QWEN_AUTH_TOKEN"] = "fake_qwen"
    os.environ["DEEPSEEK_AUTH_TOKEN"] = "fake_ds"
    os.environ["DOUBAO_AUTH_TOKEN"] = "fake_doubao"
    os.environ["KIMI_AUTH_TOKEN"] = "fake_kimi"
    os.environ["GLM_AUTH_TOKEN"] = "fake_glm"
    _env(
        GATEWAY_FIRST_CHUNK_TIMEOUT="30",
        GATEWAY_HANDSHAKE_BUDGET="0.9",
        GATEWAY_STREAM_IDLE_TIMEOUT="5",
    )

    import server as srv  # noqa: F401  (先导入以便核对信号量基线)

    cases = [
        ("/v1/chat/completions",
         b'{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}],"stream":true}'),
        ("/v1/messages",
         b'{"model":"claude-3-5-sonnet","max_tokens":16,'
         b'"messages":[{"role":"user","content":"hi"}],"stream":true}'),
        ("/v1/responses", b'{"model":"gpt-4o","input":"hi","stream":true}'),
    ]

    problems = []
    for path, body in cases:
        srv.failover_router.cooldown_tracker.reset()
        providers = {k: _TrackedProvider(k) for k in _KEYS}   # 全部挂死，预算必然耗尽
        # 基线必须在驱动**之前**取：驱动之后再读，等于拿结果和它自己比（曾是本案
        # 第一版的实际缺陷——那条断言恒真，泄漏时也不会红）。
        baseline = {
            k: srv.failover_router.cooldown_tracker.get_semaphore(k)._value for k in _KEYS
        }
        observed = _asgi_drive(providers, path, body)
        if observed["status"] is None:
            problems.append(f"{path}: 没有任何响应状态行")
        elif observed["status"] < 400:
            problems.append(f"{path}: 预算拒绝被报成了 {observed['status']}")
        leaked_lb = {k: v for k, v in observed["lb"].items() if v != 0}
        if leaked_lb:
            problems.append(f"{path}: LB 配额泄漏 {leaked_lb}")
        for key in _KEYS:
            p = providers[key]
            if p.opened and p.closed != p.opened:
                problems.append(
                    f"{path}: {key} 上游流未关闭 {p.opened} 开 / {p.closed} 关"
                )
        # 信号量本身在 ASGI 路径上必须回到满值（每个节点最多被尝试一次）
        for key in _KEYS:
            if observed["sem"][key] != baseline[key]:
                problems.append(
                    f"{path}: {key} 节点信号量残留 {baseline[key]} -> {observed['sem'][key]}"
                )

    assert not problems, "\n".join(problems)
    print("test_callers_reclaim_lb_quota_when_budget_refuses: PASSED",
          f"({len(cases)} protocols, LB=0, sem baseline held)")


def run_all():
    test_budget_exhaustion_reclaims_every_candidate_resource()
    test_budget_capped_timeout_does_not_cool_a_healthy_node()
    test_budget_floor_never_dispatches_a_doomed_attempt()
    test_callers_reclaim_lb_quota_when_budget_refuses()
    print("\n✅ All budget resource-reclaim tests passed! (4 cases)")


if __name__ == "__main__":
    run_all()