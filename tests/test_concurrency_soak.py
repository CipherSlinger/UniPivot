"""并发压力下的资源归属验证 (tests/test_concurrency_soak.py)。

前面的套件把生命周期不变量一条条钉住了，但几乎都是 **N=1 或 N=6** 的定点用例。
而本模块存在的理由就是「多个 agent 会话同时长跑」——那才是真实负载。定点用例能
证明「这条路径会回收」，证明不了「同时有十几条路径时每条都回收」，也证明不了
**信号量真的是并发的上界**（而不是恰好没有并发）。

本套件用计数器回答四个问题，全部经**真实 ASGI 栈**驱动，上游一律脚本化：

  1. **并发浸泡**（整环 + 单节点）：N=14 条并发流同时长跑，逐节点峰值在途上游
     会话必须 ≤ `PROVIDER_RISK_SPECS` 的 `max_concurrency`；单节点场景下峰值必须
     **恰好等于**上限（若小于上限，说明并发没有真正发生，断言无意义）。全部结束
     后：信号量回到满值、LB `active_concurrency` 归零、上游流 `opened == closed`。
  2. **断连风暴**：一轮 12 个客户端在「响应头之前 / 流中途」消失，连打三轮。
     每一轮结束后 LB 配额、节点信号量、上游流、在途计数器都必须回到驱动前的值，
     且第二轮、第三轮不得比第一轮多漏一点（泄漏会累积，一次性用例看不出来）。
  3. **心跳覆盖长沉默**：上游沉默时长远超心跳间隔后才恢复，`content` 帧一个都不能
     少、顺序不能变，且沉默期真的有 `: keepalive` 字节（客户端计时器被复位）。
  4. **SingleFlight 扇出背压**（真实栈）：Leader + 多个跟随者，其中一个消费很慢。
     慢跟随者被摘除，其余跟随者收全，Leader 不被拖住，结束时资源全部归位。

运行: .venv/bin/python tests/test_concurrency_soak.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["TESTING"] = "1"
os.environ["GATEWAY_DISABLE_AUTO_REFRESH"] = "1"
os.environ["GATEWAY_DISABLE_PROBING"] = "1"
os.environ["GATEWAY_DISABLE_PACING"] = "1"
os.environ["GATEWAY_DISABLE_PROBATION"] = "1"
os.environ["GATEWAY_DISABLE_RISK_CORRELATION"] = "1"
os.environ["GATEWAY_DISABLE_CREDENTIAL_WATCH"] = "1"
os.environ["GATEWAY_DISABLE_CREDENTIAL_RENEWAL"] = "1"
os.environ["GATEWAY_DISABLE_DEVICE_IDENTITY"] = "1"
os.environ["GATEWAY_DISABLE_BAN_INFERENCE"] = "1"
os.environ["GATEWAY_DISABLE_CONVERSATION_GC"] = "1"
# 用例若是自己写错而挂住，应当快速失败，而不是等到 180s 默认值
os.environ["GATEWAY_FIRST_CHUNK_TIMEOUT"] = "15"
os.environ["GATEWAY_STREAM_IDLE_TIMEOUT"] = "15"
os.environ["GATEWAY_HANDSHAKE_BUDGET"] = "60"

import server as srv  # noqa: E402
from providers.failover import PROVIDER_RISK_SPECS, ProviderError  # noqa: E402
from providers.singleflight import (  # noqa: E402
    get_follower_backpressure_stats,
    singleflight_manager,
)

_KEYS = ("qwen", "deepseek", "doubao", "kimi", "glm")

OPENAI_BODY = b'{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}],"stream":true}'
ANTHROPIC_BODY = (
    b'{"model":"claude-3-5-sonnet","max_tokens":16,'
    b'"messages":[{"role":"user","content":"hi"}],"stream":true}'
)
RESPONSES_BODY = b'{"model":"gpt-4o","input":"hi","stream":true}'


# ------------------------------------------------------------------ 测试替身

class _NodeStats:
    """一个节点的上游会话账本（跨 provider 实例累计）。

    真实 `_make_provider` 每次派发都会**新建** provider 对象，所以计数器不能挂在
    provider 实例上，否则每个请求都是 1/1，等于什么都没测。
    """

    def __init__(self, key: str) -> None:
        self.key = key
        self.in_flight = 0
        self.peak = 0
        self.opened = 0
        self.closed = 0
        self.frames = 0


class _Script:
    """脚本化上游的行为描述。"""

    def __init__(
        self,
        head: str = "start",
        tail: tuple = ("end",),
        *,
        hold: "asyncio.Event | None" = None,
        delay: float = 0.0,
        complete: bool = True,
    ) -> None:
        self.head = head
        self.tail = tuple(tail)
        self.hold = hold
        self.delay = delay
        # 元数据帧里是否带 `upstream_complete`。只有 qwen/kimi 这类真有完成标记的
        # 协议会带；这里的脚本化上游代表「没有标记」的三家，故默认不带。
        self.complete = complete


class _ScriptedProvider:
    """脚本化上游：记录在途会话峰值，并在 `hold` 事件上挂住整段流。"""

    def __init__(self, stats: _NodeStats, script: _Script) -> None:
        self.stats = stats
        self.script = script

    async def chat(self, *args, **kwargs):
        st = self.stats
        st.opened += 1
        st.in_flight += 1
        st.peak = max(st.peak, st.in_flight)
        conv = {"conversation_id": f"{st.key}-conv"}
        try:
            yield self.script.head, conv
            st.frames += 1
            if self.script.hold is not None:
                # 首包已经产出（握手成功），但整段流仍占着节点并发槽位——
                # 这正是要测的窗口：信号量必须覆盖整段流，而不是只覆盖握手。
                await self.script.hold.wait()
            for delta in self.script.tail:
                if self.script.delay:
                    await asyncio.sleep(self.script.delay)
                yield delta, conv
                st.frames += 1
            # 与五家真实 provider 一致：结尾无条件产出一帧末位元数据
            yield "", {**conv, "upstream_complete": self.script.complete}
        finally:
            st.closed += 1
            st.in_flight -= 1

    async def delete_conversation(self, *args, **kwargs):
        return True


_TRIPWIRE_HITS = 0


def _install_tripwire() -> None:
    """任何一条用例忘了安装脚本化上游时，立刻炸掉而不是发出真实请求。"""

    def factory(key, client=None):
        global _TRIPWIRE_HITS
        _TRIPWIRE_HITS += 1
        raise AssertionError(f"用例未安装脚本化上游（{key}），几乎发出了真实上游请求")

    srv._make_provider = factory


def _install(registry: dict, script: _Script) -> None:
    def factory(key, client=None):
        stats = registry.get(key)
        if stats is None:
            stats = registry[key] = _NodeStats(key)
        return _ScriptedProvider(stats, script)

    srv._make_provider = factory


# ------------------------------------------------------------------ ASGI 客户端

def _scope(path: str) -> dict:
    return {
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


class _Client:
    """一个可观测的 ASGI 客户端。

    `close_after` 用来模拟「客户端消失」：第 N 次 send 时抛 `OSError`。Starlette 在
    `spec_version >= 2.4` 下把 send 的 `OSError` 转成 `ClientDisconnect`，这是最
    忠实的消失方式（不是我们主动取消任务，而是管道断了）。
    `slow` 模拟一个消费很慢的 SSE 客户端：每次 send 之前睡一会儿。
    """

    def __init__(self, *, close_after: int | None = None, slow: float = 0.0) -> None:
        self.close_after = close_after
        self.slow = slow
        self.sends = 0
        self.status: int | None = None
        self.bodies: list[bytes] = []
        self.disconnected = False
        self.error: str | None = None

    async def send(self, message: dict) -> None:
        self.sends += 1
        if self.close_after is not None and self.sends > self.close_after:
            self.disconnected = True
            raise OSError("client gone")
        if message["type"] == "http.response.start":
            self.status = message["status"]
            return
        if message["type"] == "http.response.body":
            if self.slow:
                await asyncio.sleep(self.slow)
            chunk = message.get("body", b"")
            if chunk:
                self.bodies.append(chunk)

    @property
    def text(self) -> str:
        return b"".join(self.bodies).decode("utf-8", "replace")


def _receiver(body: bytes):
    state = {"sent": False}

    async def receive():
        if not state["sent"]:
            state["sent"] = True
            return {"type": "http.request", "body": body, "more_body": False}
        # 长流在 spec 2.4 下不会被反复询问；这里只是兜底
        await asyncio.sleep(0.2)
        return {"type": "http.disconnect"}

    return receive


async def _drive(path: str, body: bytes, client: _Client) -> _Client:
    try:
        await asyncio.wait_for(
            srv.app(_scope(path), _receiver(body), client.send), timeout=40
        )
    except asyncio.CancelledError:
        raise
    except BaseException as exc:  # noqa: BLE001 - 断连/异常本身是被测行为
        client.error = f"{type(exc).__name__}: {exc}"
    return client


async def _settle(loops: int = 80) -> None:
    """给独立收尾任务（shield 的 aclose、create_task 的 delete 等）调度机会。"""
    for _ in range(loops):
        await asyncio.sleep(0.01)


async def _wait_until(predicate, timeout: float = 8.0, interval: float = 0.02) -> bool:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(interval)
    return predicate()


# ------------------------------------------------------------------ 计数器

def _sem_snapshot() -> dict:
    tracker = srv.failover_router.cooldown_tracker
    return {k: tracker.get_semaphore(k)._value for k in _KEYS}


def _sem_limit(key: str) -> int:
    return PROVIDER_RISK_SPECS[key].max_concurrency


def _lb_snapshot() -> dict:
    return {k: srv.load_balancer.get_metrics(k).active_concurrency for k in _KEYS}


def _reset_lb() -> None:
    for key in _KEYS:
        srv.load_balancer.get_metrics(key).reset()


def _clear_tracker() -> None:
    tracker = srv.failover_router.cooldown_tracker
    tracker.reset()
    tracker.disabled_providers.clear()
    # 每个用例各自一次 `asyncio.run`，即各自一个事件循环；而 `asyncio.Semaphore`
    # 在**首次使用**时绑定当时的循环。全局 router 的信号量会在第一个用例的循环里
    # 绑定，第二个用例再用就是 `is bound to a different event loop`（实测会把请求
    # 变成 502 并给节点记冷却，直接污染后面的测量）。这是**测试隔离**问题，不是产品
    # 缺陷：生产里只有一个循环。重建信号量表即可让每个用例拿到属于自己循环的实例。
    from providers.failover import PROVIDER_RISK_SPECS as _SPECS

    tracker._semaphores = {
        key: asyncio.Semaphore(meta.max_concurrency)
        for key, meta in _SPECS.items()
    }
    srv.session_affinity_manager.clear()
    singleflight_manager.clear()


def _leaks(registry: dict, sem_before: dict, lb_after: dict) -> list[str]:
    problems: list[str] = []
    tracker = srv.failover_router.cooldown_tracker
    for key in _KEYS:
        sem_now = tracker.get_semaphore(key)._value
        if sem_now != sem_before[key]:
            problems.append(
                f"{key}: 节点信号量未归位 {sem_before[key]} -> {sem_now}"
                f"（并发上限 {_sem_limit(key)}）"
            )
        if lb_after.get(key, 0) != 0:
            problems.append(f"{key}: LB 配额泄漏 active_concurrency={lb_after[key]}")
        st = registry.get(key)
        if st is not None and st.opened and st.closed != st.opened:
            problems.append(f"{key}: 上游流未关闭 {st.opened} 开 / {st.closed} 关")
        if st is not None and st.in_flight != 0:
            problems.append(f"{key}: 上游在途计数残留 in_flight={st.in_flight}")
    if srv._inflight_requests != 0:
        problems.append(f"在途请求计数残留 inflight_requests={srv._inflight_requests}")
    return problems


# ------------------------------------------------------------------ 1. 并发浸泡

def test_ring_soak_never_exceeds_per_node_concurrency():
    """整环并发 14 条长流：逐节点峰值 ≤ 上限，结束后全部归位。

    这是本模块最核心的一条：`PROVIDER_RISK_SPECS` 的 `max_concurrency` 是
    「同一账号同时最多背几条上游会话」，也是本项目「不把另外四家打进风控」的直接
    手段。它只有在真实并发下仍然成立才有意义。
    """
    os.environ["GATEWAY_DISABLE_SINGLEFLIGHT"] = "1"  # 同构请求否则会被合并成一条
    os.environ["GATEWAY_STREAM_KEEPALIVE"] = "0"
    _clear_tracker()
    _install_tripwire()
    _reset_lb()

    registry: dict = {}
    gate = asyncio.Event()
    _install(registry, _Script(head="start", tail=("end",), hold=gate))
    sem_before = _sem_snapshot()
    n = 14

    async def _run():
        clients = [
            _Client() for _ in range(n)
        ]
        tasks = []
        # 错开启动。`select_route` 读的是 `active_concurrency`，而 LB 配额是在
        # `gen()` 里 acquire 的（execute_chat 之后），所以**同一 tick 内**起来的请求
        # 全都看到 0 并发，必然全压到同一个节点上，环根本不会分摊。真实的并发 agent
        # 会话天然是错开的，这里按 0.12s 铺开，让负载均衡看到负载。
        for index, c in enumerate(clients):
            tasks.append(
                asyncio.ensure_future(_drive("/v1/chat/completions", OPENAI_BODY, c))
            )
            if index < n - 1:
                await asyncio.sleep(0.12)
        # 等到「在途总数」稳定（该起的都起了，该排队的都排上了）再测峰值
        samples: list[int] = []
        stable = 0
        for _ in range(400):
            await asyncio.sleep(0.02)
            total = sum(st.in_flight for st in registry.values())
            samples.append(total)
            if len(samples) >= 10 and len(set(samples[-10:])) == 1:
                stable += 1
                if stable >= 2:
                    break
        peak_at_barrier = sum(st.in_flight for st in registry.values())
        per_node = {k: st.peak for k, st in registry.items()}

        gate.set()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        await _settle()
        return results, per_node, peak_at_barrier

    results, peak, peak_at_barrier = asyncio.run(_run())

    problems: list[str] = []
    if min(peak.values() or [0]) < 0:
        problems.append("负峰值")
    for key, value in peak.items():
        if value > _sem_limit(key):
            problems.append(
                f"{key}: 并发峰值 {value} 超过配置上限 {_sem_limit(key)}"
                f"——信号量没有覆盖整段流"
            )
    if max(peak.values() or [0]) < 2:
        problems.append(
            f"没有任何节点同时跑过两条流（逐节点峰值 {peak}），"
            f"说明这次浸泡没有真正并发，断言无意义"
        )
    if len(registry) < 2:
        problems.append(f"只用到 {sorted(registry)} 一个节点，环没有分摊，测不到整环并发")

    for client in results:
        if isinstance(client, BaseException):
            problems.append(f"请求被执行期异常打断: {type(client).__name__}: {client}")
            continue
        if client.status != 200:
            problems.append(f"并发流的状态行不是 200: {client.status} ({client.error})")
        elif "start" not in client.text or "end" not in client.text:
            problems.append("并发流的内容不完整（疑似被回收动作截断）")

    problems.extend(_leaks(registry, sem_before, _lb_snapshot()))
    assert not problems, "整环并发浸泡暴露的资源问题:\n" + "\n".join(problems)
    print(
        "test_ring_soak_never_exceeds_per_node_concurrency: PASSED",
        f"(n={n}, barrier in-flight={peak_at_barrier},"
        f" per-node peak={ {k: v for k, v in peak.items()} },"
        f" opened={sum(s.opened for s in registry.values())})",
    )


def test_single_node_soak_peak_equals_the_semaphore_limit():
    """只剩一个节点时，14 条并发流的峰值必须**恰好等于**它的上限。

    「≤ 上限」在并发没有真正发生时会白白通过（串行执行峰值只有 1）。这里把其余
    四个节点禁用，让全部请求都指向 qwen（上限 3），于是 3 是唯一正确的答案：
    小于 3 说明并发没发生，大于 3 说明并发治理失效。
    """
    os.environ["GATEWAY_DISABLE_SINGLEFLIGHT"] = "1"
    os.environ["GATEWAY_STREAM_KEEPALIVE"] = "0"
    _clear_tracker()
    _install_tripwire()
    _reset_lb()

    tracker = srv.failover_router.cooldown_tracker
    tracker.disabled_providers.update({"deepseek", "doubao", "kimi", "glm"})

    registry: dict = {}
    gate = asyncio.Event()
    _install(registry, _Script(head="start", tail=("end",), hold=gate))
    sem_before = _sem_snapshot()
    limit = _sem_limit("qwen")
    n = 14

    async def _run():
        clients = [_Client() for _ in range(n)]
        tasks = [
            asyncio.ensure_future(_drive("/v1/chat/completions", OPENAI_BODY, c))
            for c in clients
        ]
        reached = await _wait_until(
            lambda: registry.get("qwen") is not None
            and registry["qwen"].in_flight >= limit,
            timeout=10.0,
        )
        held = registry["qwen"].in_flight if "qwen" in registry else 0
        peak = registry["qwen"].peak if "qwen" in registry else 0
        gate.set()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        await _settle()
        return results, reached, held, peak

    results, reached, held, peak = asyncio.run(_run())
    tracker.disabled_providers.clear()

    problems: list[str] = []
    if not reached:
        problems.append(
            f"没能观察到 {limit} 条流同时在途（峰值 {peak}），"
            f"并发浸泡没有真正发生"
        )
    if peak > limit:
        problems.append(f"qwen 峰值 {peak} 超过上限 {limit}")
    if peak < limit:
        problems.append(
            f"qwen 峰值只有 {peak} < 上限 {limit}：要么没并发起来，"
            f"要么有别的因素在串行化"
        )
    for client in results:
        if isinstance(client, BaseException) or client.status != 200:
            problems.append(f"并发流未正常完成: {client}")
    problems.extend(_leaks(registry, sem_before, _lb_snapshot()))
    assert not problems, "单节点并发浸泡暴露的问题:\n" + "\n".join(problems)
    print(
        "test_single_node_soak_peak_equals_the_semaphore_limit: PASSED",
        f"(n={n}, limit={limit}, held={held}, peak={peak})",
    )


# ------------------------------------------------------------------ 2. 断连风暴

def test_disconnect_storm_returns_every_resource():
    """三轮断连风暴：每轮 12 个客户端在最难受的三个时刻消失。

    「响应头之前」「流中途」「正常收完」三种消失方式各有不同的回收路径
    （`on_close` 同步槽位回收 / `guard_stream` / 正常 finally）。定点用例各证明
    一次；这里证明**连续三轮不会累积**——真实泄漏几乎总是这样：单次看起来数字
    对得上，压力一上来配额就被一点点吃掉，直到负载均衡把节点判成高负载。
    """
    os.environ["GATEWAY_DISABLE_SINGLEFLIGHT"] = "1"  # 否则同构请求会合并，测不到 12 条流
    os.environ["GATEWAY_STREAM_KEEPALIVE"] = "0"
    _clear_tracker()
    _install_tripwire()
    _reset_lb()

    registry: dict = {}
    _install(registry, _Script(head="hello", tail=("world",)))
    sem_before = _sem_snapshot()
    rounds = 3

    async def _burst():
        jobs = []
        for _ in range(4):  # 响应头之前就消失
            jobs.append(_drive("/v1/chat/completions", OPENAI_BODY, _Client(close_after=0)))
        for _ in range(4):  # 流到一半消失
            jobs.append(_drive("/v1/chat/completions", OPENAI_BODY, _Client(close_after=2)))
        jobs.append(_drive("/v1/messages", ANTHROPIC_BODY, _Client(close_after=0)))
        jobs.append(_drive("/v1/messages", ANTHROPIC_BODY, _Client(close_after=2)))
        jobs.append(_drive("/v1/responses", RESPONSES_BODY, _Client(close_after=0)))
        jobs.append(_drive("/v1/responses", RESPONSES_BODY, _Client(close_after=2)))
        return await asyncio.gather(*jobs, return_exceptions=True)

    async def _run():
        per_round = []
        for _ in range(rounds):
            await _burst()
            await _settle(40)
            per_round.append(
                {
                    "sem": _sem_snapshot(),
                    "lb": _lb_snapshot(),
                    "inflight": srv._inflight_requests,
                    "opened": {k: s.opened for k, s in registry.items()},
                    "closed": {k: s.closed for k, s in registry.items()},
                }
            )
        await _settle()
        return per_round

    per_round = asyncio.run(_run())

    problems: list[str] = []
    for idx, snap in enumerate(per_round, start=1):
        for key in _KEYS:
            if snap["sem"][key] != sem_before[key]:
                problems.append(
                    f"第 {idx} 轮: {key} 信号量 {sem_before[key]} -> {snap['sem'][key]}"
                )
            if snap["lb"][key] != 0:
                problems.append(
                    f"第 {idx} 轮: {key} LB 配额泄漏 {snap['lb'][key]}"
                )
            opened = snap["opened"].get(key, 0)
            closed = snap["closed"].get(key, 0)
            if opened != closed:
                problems.append(
                    f"第 {idx} 轮: {key} 上游流未关闭 {opened} 开 / {closed} 关"
                )
        if snap["inflight"] != 0:
            problems.append(f"第 {idx} 轮: 在途请求计数残留 {snap['inflight']}")

    # 三轮必须真的打出了流量，否则「没有泄漏」只是因为什么都没发生
    total_opened = sum(s.opened for s in registry.values())
    if total_opened < rounds * 6:
        problems.append(
            f"三轮风暴只打开了 {total_opened} 条上游流，压力不足以验证回收"
        )

    assert not problems, "断连风暴暴露的资源泄漏:\n" + "\n".join(problems)
    print(
        "test_disconnect_storm_returns_every_resource: PASSED",
        f"(rounds={rounds}, streams opened={total_opened},"
        f" per-round sem held, LB=0, inflight=0)",
    )


# ------------------------------------------------------------------ 3. 心跳覆盖长沉默

def test_keepalive_covers_long_silence_without_losing_chunks():
    """上游沉默 6 个心跳周期后恢复：内容一个不少，且沉默期真有字节。

    定点用例已经证明心跳会产出、且 `wait_for` 写法会永久丢帧。这条用例问的是
    另一个问题：**沉默跨越多个心跳周期**时（推理模型想几十秒的真实形状），
    恢复之后的上游数据是否仍然完整，以及客户端在这段时间里是否真的收到了字节。
    """
    os.environ["GATEWAY_DISABLE_SINGLEFLIGHT"] = "1"
    os.environ["GATEWAY_STREAM_KEEPALIVE"] = "0.05"
    _clear_tracker()
    _install_tripwire()
    _reset_lb()

    registry: dict = {}
    silence = 0.35  # 7 个心跳周期
    _install(registry, _Script(head="a", tail=("b", "c"), delay=silence))
    client = _Client()

    async def _run():
        await _drive("/v1/chat/completions", OPENAI_BODY, client)
        await _settle(20)

    asyncio.run(_run())
    os.environ["GATEWAY_STREAM_KEEPALIVE"] = "0"

    raw = client.text
    contents: list[str] = []
    for line in raw.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[len("data: "):].strip()
        if payload == "[DONE]":
            continue
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue
        for choice in obj.get("choices", []):
            delta = choice.get("delta") or {}
            piece = delta.get("content")
            if isinstance(piece, str) and piece:
                contents.append(piece)

    expected_keepalives = 4  # 两段沉默，每段 >= 4 帧（0.35s / 0.05s = 7）
    got_keepalives = raw.count(": keepalive")
    problems: list[str] = []
    if contents != ["a", "b", "c"]:
        problems.append(f"沉默恢复后内容丢失或乱序: {contents!r}（期望 ['a','b','c']）")
    if got_keepalives < expected_keepalives:
        problems.append(
            f"沉默 0.35s（7 个心跳周期）只产出了 {got_keepalives} 帧心跳，"
            f"客户端计时器没有被复位"
        )
    if client.status != 200:
        problems.append(f"状态行不是 200: {client.status}")
    problems.extend(_leaks(registry, _sem_snapshot(), _lb_snapshot()))
    assert not problems, "心跳覆盖长沉默的问题:\n" + "\n".join(problems)
    print(
        "test_keepalive_covers_long_silence_without_losing_chunks: PASSED",
        f"(silence={silence}s, keepalive frames={got_keepalives}, contents={contents})",
    )


# ------------------------------------------------------------------ 4. SingleFlight 扇出背压

def test_singleflight_fanout_backpressure_through_the_real_stack():
    """Leader + 4 跟随者（其一消费很慢）走真实 ASGI 栈。

    定点用例在飞行组层面证明了「摘除而非阻塞」。这条用例问的是**线上后果**：
    慢跟随者是否真的被摘掉、快跟随者是否收全、Leader 是否被拖住、以及结束时
    资源是否归位。跟随者挂在共享流上，其中一个卡住绝不能连累其余。
    """
    os.environ["GATEWAY_DISABLE_SINGLEFLIGHT"] = "0"
    os.environ["GATEWAY_SINGLEFLIGHT_BACKLOG"] = "4"
    os.environ["GATEWAY_STREAM_KEEPALIVE"] = "0"
    _clear_tracker()
    _install_tripwire()
    _reset_lb()

    registry: dict = {}
    tail = tuple(f"chunk{i}" for i in range(40))
    _install(registry, _Script(head="first", tail=tail, delay=0.01))
    sem_before = _sem_snapshot()
    baseline_detached = get_follower_backpressure_stats()["followers_detached"]

    # 请求体必须完全一致，否则指纹不同、根本不会合并成一组
    body = b'{"model":"gpt-4o","messages":[{"role":"user","content":"same"}],"stream":true}'
    leader = _Client()
    fast = [_Client() for _ in range(3)]
    slow = _Client(slow=0.08)

    async def _run():
        started = time.monotonic()
        tasks = [asyncio.ensure_future(_drive("/v1/chat/completions", body, leader))]
        # 让 Leader 先拿到组，再挂跟随者（真实客户端也是这个顺序）
        await asyncio.sleep(0.05)
        tasks.append(asyncio.ensure_future(_drive("/v1/chat/completions", body, slow)))
        for c in fast:
            tasks.append(asyncio.ensure_future(_drive("/v1/chat/completions", body, c)))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = time.monotonic() - started
        await _settle()
        return results, elapsed

    results, elapsed = asyncio.run(_run())
    os.environ.pop("GATEWAY_SINGLEFLIGHT_BACKLOG", None)
    os.environ["GATEWAY_DISABLE_SINGLEFLIGHT"] = "1"

    detached = (
        get_follower_backpressure_stats()["followers_detached"] - baseline_detached
    )
    problems: list[str] = []
    if detached < 1:
        problems.append(
            "慢跟随者没有被摘除（背压上限 4 帧，Leader 产出 41 帧，"
            "慢客户端每帧 0.08s）"
        )
    if leader.status != 200 or "first" not in leader.text or tail[-1] not in leader.text:
        problems.append("Leader 自己没收到完整回答")
    for idx, c in enumerate(fast):
        if c.status != 200 or "first" not in c.text or tail[-1] not in c.text:
            problems.append(f"跟随者 #{idx} 没有收全（不该被摘除）: status={c.status}")
    # Leader 不能被慢跟随者拖住：生产 41 帧 × 0.01s ≈ 0.4s；慢客户端的 0.08s/帧
    # 若参与同步，总时长会到 3s 以上。
    if elapsed > 6.0:
        problems.append(f"扇出被慢跟随者拖住，总耗时 {elapsed:.2f}s")
    problems.extend(_leaks(registry, sem_before, _lb_snapshot()))

    assert not problems, "SingleFlight 扇出背压的问题:\n" + "\n".join(problems)
    print(
        "test_singleflight_fanout_backpressure_through_the_real_stack: PASSED",
        f"(detached={detached}, elapsed={elapsed:.2f}s,"
        f" slow got {len(slow.bodies)} body frames, leader got {len(leader.bodies)})",
    )


# ------------------------------------------------------------------ 主入口

def run_all():
    _install_tripwire()
    try:
        test_ring_soak_never_exceeds_per_node_concurrency()
        test_single_node_soak_peak_equals_the_semaphore_limit()
        test_disconnect_storm_returns_every_resource()
        test_keepalive_covers_long_silence_without_losing_chunks()
        test_singleflight_fanout_backpressure_through_the_real_stack()
    finally:
        _install_tripwire()
    assert _TRIPWIRE_HITS == 0, (
        f"有 {_TRIPWIRE_HITS} 次派发落到了 tripwire 上——某个用例没有安装脚本化上游，"
        f"在真实环境里就是一次带凭证的真实上游请求"
    )
    print("\n✅ All concurrency soak tests passed! (5 cases)")


if __name__ == "__main__":
    run_all()