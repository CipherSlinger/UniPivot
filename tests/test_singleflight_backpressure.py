"""SingleFlight 跟随者背压测试 (tests/test_singleflight_backpressure.py)。

[M4] 问题：跟随者的订阅队列是**无界**的（`asyncio.Queue()` 默认 maxsize=0），
而 Leader 广播用的是 `put_nowait` —— 它在这个队列上**永不失败**，所以「满了怎么
办」的分支（`except Exception: pass`）实际上是一段死代码。于是一个消费很慢、
或者干脆一帧都不读的跟随者，其队列会被 Leader 逐帧灌满，没有上限。

实测（修复前）：Leader 广播 20000 帧 × 512 字节后，单个不消费跟随者的队列里
躺着 20000 帧，约 10 MB，且看不到任何上限。正常的速度差——客户端在慢网络上、
SSE 消费端处理一次卡顿——会比任何人手写的测试更早地把它堆起来。这不是理论风险，
是「网关连接 agent 效果差」里一个安静的放大器：它不会报错，只会一直吃内存。

修复语义：
  * 每个跟随者积压超过 `GATEWAY_SINGLEFLIGHT_BACKLOG`（默认 256 帧）即被**摘除**
    并收到一个明确错误，而不是被阻塞或被静默截断；
  * **绝不阻塞 Leader**：那等于让一个慢客户端拖住所有跟随者，正是 SingleFlight
    要消除的问题；
  * 其余跟随者不受影响，Leader 也不受影响（这里是「这台机器上只有它一个」的
    局部决定，而不是给所有跟随者改判）；
  * 被摘除的跟随者收到的是**错误**而非静默结束——半截回答被当成完整回答，
    比报错更糟。

运行: .venv/bin/python tests/test_singleflight_backpressure.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["TESTING"] = "1"

from providers.singleflight import (  # noqa: E402
    FlightGroup,
    broadcast_stream_wrapper,
    follower_backlog_limit,
    get_follower_backpressure_stats,
    singleflight_manager,
    stream_follower_gen,
)


def test_slow_follower_is_detached_not_unbounded():
    """消费过慢的跟随者：队列有界、被摘除、并收到明确错误；其余跟随者不受影响。"""

    async def _run():
        limit = 8
        os.environ["GATEWAY_SINGLEFLIGHT_BACKLOG"] = str(limit)
        group = FlightGroup("k" * 64)

        # 跟随者 A：真的在消费，但很慢（模拟慢网络 / 卡顿的 SSE 客户端）
        slow_received = []
        slow_err = None

        async def slow_consumer():
            nonlocal slow_err
            try:
                async for chunk in stream_follower_gen(group):
                    slow_received.append(chunk)
                    await asyncio.sleep(0.01)
            except Exception as exc:  # noqa: BLE001
                slow_err = exc

        # 跟随者 B：同步排空自己的队列（模拟跟得上节奏的客户端），用来证明
        # 摘除是**局部**的，不会牵连其它跟随者。
        slow_task = asyncio.ensure_future(slow_consumer())
        await asyncio.sleep(0)  # 让它先完成挂载
        _, live_q, _ = await group.add_subscriber()

        max_seen = 0
        for i in range(200):
            await group.broadcast_chunk((f"frame_{i}", {}))
            max_seen = max(max_seen, live_q.qsize())
            # 排空 B 的队列
            while not live_q.empty():
                live_q.get_nowait()
            await asyncio.sleep(0)

        assert max_seen <= limit, f"单个跟随者的积压必须有界，实测峰值 {max_seen}"
        assert live_q in group.subscribers, "跟得上节奏的跟随者不得被摘除"
        assert get_follower_backpressure_stats()["followers_detached"] >= 1

        await group.finish()
        await asyncio.wait_for(slow_task, timeout=5.0)

        assert slow_err is not None, "被摘除的跟随者必须收到错误，不能静默截断成半截回答"
        assert "消费过慢" in str(slow_err) or "摘除" in str(slow_err), str(slow_err)
        assert len(slow_received) < 200, "被摘除者不应假装收全了"
        assert len(group.history) == 200, "历史重放缓冲区仍须完整（Leader 不受影响）"
        os.environ.pop("GATEWAY_SINGLEFLIGHT_BACKLOG", None)

    asyncio.run(_run())
    print("test_slow_follower_is_detached_not_unbounded: PASSED")


def test_leader_is_never_blocked_by_a_slow_follower():
    """摘除而非阻塞：慢跟随者不能拖慢 Leader 的广播。"""

    async def _run():
        os.environ["GATEWAY_SINGLEFLIGHT_BACKLOG"] = "8"
        group = FlightGroup("m" * 64)
        await group.add_subscriber()  # 永不消费

        frames = 5000
        start = asyncio.get_running_loop().time()

        async def produce():
            async for _ in broadcast_stream_wrapper(_gen(frames), group):
                pass

        await asyncio.wait_for(produce(), timeout=5.0)
        elapsed = asyncio.get_running_loop().time() - start
        # 只需证明没有随帧数线性阻塞（若阻塞在慢跟随者上，5000 帧会远超 5 秒超时）
        assert elapsed < 5.0, f"Leader 被慢跟随者拖住了：{elapsed:.2f}s"
        assert len(group.history) == frames, "历史重放缓冲区仍须完整"
        os.environ.pop("GATEWAY_SINGLEFLIGHT_BACKLOG", None)

    async def _gen(n):
        for i in range(n):
            yield (f"frame_{i}", {})

    asyncio.run(_run())
    print("test_leader_is_never_blocked_by_a_slow_follower: PASSED")


def test_backpressure_does_not_leak_between_flights():
    """摘除计数与配置不串扰：新跟随者挂载时重置本组的摘除标记，诊断可见。"""

    async def _run():
        os.environ["GATEWAY_SINGLEFLIGHT_BACKLOG"] = "4"
        singleflight_manager.clear()

        group = FlightGroup("n" * 64)
        await group.add_subscriber()
        for i in range(20):
            await group.broadcast_chunk((f"f{i}", {}))
        assert group._followers_detached == 1

        # 上一轮慢跟随者已退出，新跟随者挂载后本组应可再次服务
        _, q, done = await group.add_subscriber()
        assert done is False
        assert group._followers_detached == 0
        assert follower_backlog_limit() == 4

        diag = singleflight_manager.get_diagnostics()
        assert diag["follower_backlog_limit"] == 4
        assert "followers_detached" in diag
        os.environ.pop("GATEWAY_SINGLEFLIGHT_BACKLOG", None)

    asyncio.run(_run())
    print("test_backpressure_does_not_leak_between_flights: PASSED")


def run_all():
    test_slow_follower_is_detached_not_unbounded()
    test_leader_is_never_blocked_by_a_slow_follower()
    test_backpressure_does_not_leak_between_flights()
    print("\n✅ All singleflight backpressure tests passed!")


if __name__ == "__main__":
    run_all()