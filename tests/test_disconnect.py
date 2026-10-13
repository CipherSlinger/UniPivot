"""客户端断连传播与上游资源回收引擎 (OPT-009) 测试套件

验证：
1. 上游异步生成器的安全关闭 (safe_aclose)
2. 断连传播包装器 (guard_stream) 的正常完成 / 提前断连 / 取消 / 回调异常路径
3. SingleFlight Leader 断连后飞行组终局化（回归 OPT-008 的 Follower 永久挂起缺陷）
4. 陈旧飞行组回收与 Follower 唤醒
5. 断连回收遥测指标与禁用开关

运行: .venv/bin/python tests/test_disconnect.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["GATEWAY_DISABLE_AUTO_REFRESH"] = "1"
os.environ["TESTING"] = "1"

from providers.disconnect_guard import DisconnectGuard, disconnect_guard, guard_stream  # noqa: E402
from providers.singleflight import (  # noqa: E402
    FlightGroup,
    SingleFlightManager,
    broadcast_stream_wrapper,
    stream_follower_gen,
)


def _reset_guard() -> None:
    disconnect_guard.clear()


# ------------------------------------------------------------------ safe_aclose


def test_safe_aclose_closes_generator():
    """被 aclose 的生成器其 finally 执行，且 reclaimed_streams 计数递增。"""
    _reset_guard()
    closed = {"flag": False}

    async def upstream():
        try:
            yield "a"
            yield "b"
        finally:
            closed["flag"] = True

    async def _run():
        agen = upstream()
        assert await agen.__anext__() == "a"
        ok = await disconnect_guard.safe_aclose(agen)
        assert ok is True
        assert closed["flag"] is True

    asyncio.run(_run())
    assert disconnect_guard.get_diagnostics()["reclaimed_streams"] == 1
    print("test_safe_aclose_closes_generator: PASSED")


def test_safe_aclose_is_exception_safe():
    """上游 aclose() 抛异常时返回 False 且绝不上抛。"""
    _reset_guard()

    class Exploding:
        async def aclose(self):
            raise RuntimeError("boom")

    async def _run():
        assert await disconnect_guard.safe_aclose(Exploding()) is False
        assert await disconnect_guard.safe_aclose(None) is False
        assert await disconnect_guard.safe_aclose(object()) is False

    asyncio.run(_run())
    print("test_safe_aclose_is_exception_safe: PASSED")


# ------------------------------------------------------------------ guard_stream


def test_guard_stream_normal_completion():
    """正常走完：不计断连，但上游流仍被回收关闭。"""
    _reset_guard()
    closed = {"flag": False}

    async def upstream():
        try:
            for c in ["x", "y", "z"]:
                yield c
        finally:
            closed["flag"] = True

    async def _run():
        out = []
        async for item in guard_stream(upstream(), provider_key="qwen"):
            out.append(item)
        assert out == ["x", "y", "z"]

    asyncio.run(_run())
    stats = disconnect_guard.get_diagnostics()
    assert closed["flag"] is True, "上游流必须在正常完成后被显式关闭"
    assert stats["client_disconnects"] == 0
    assert stats["reclaimed_streams"] == 1
    print("test_guard_stream_normal_completion: PASSED")


def test_guard_stream_early_break_records_disconnect():
    """消费者提前 break：计入 client_disconnects，且上游生成器被关闭。"""
    _reset_guard()
    closed = {"flag": False}

    async def upstream():
        try:
            for i in range(50):
                yield i
        finally:
            closed["flag"] = True

    async def _run():
        got = []
        async for item in guard_stream(upstream(), provider_key="doubao"):
            got.append(item)
            if len(got) >= 2:
                break
        assert got == [0, 1]

    asyncio.run(_run())
    stats = disconnect_guard.get_diagnostics()
    assert closed["flag"] is True, "提前断连必须传播到上游并关闭生成器"
    assert stats["client_disconnects"] == 1
    assert stats["per_provider"]["doubao"]["disconnects"] == 1
    print("test_guard_stream_early_break_records_disconnect: PASSED")


def test_guard_stream_cancellation_records_disconnect():
    """任务被 cancel：CancelledError 正常传播，计入断连并关闭上游。"""
    _reset_guard()
    closed = {"flag": False}

    async def upstream():
        try:
            i = 0
            while True:
                yield i
                i += 1
                await asyncio.sleep(0.01)
        finally:
            closed["flag"] = True

    async def _run():
        async def consume():
            async for _ in guard_stream(upstream(), provider_key="kimi"):
                await asyncio.sleep(0.01)

        task = asyncio.create_task(consume())
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
            raise AssertionError("应当抛出 CancelledError")
        except asyncio.CancelledError:
            pass

    asyncio.run(_run())
    stats = disconnect_guard.get_diagnostics()
    assert closed["flag"] is True, "取消必须传播到上游并关闭生成器"
    assert stats["client_disconnects"] == 1
    print("test_guard_stream_cancellation_records_disconnect: PASSED")


def test_guard_stream_on_close_callback_error_swallowed():
    """on_close 回调抛异常时不得影响主流程。"""
    _reset_guard()
    called = {"n": 0}

    async def upstream():
        yield 1
        yield 2

    async def bad_callback():
        called["n"] += 1
        raise RuntimeError("callback exploded")

    async def _run():
        got = []
        async for item in guard_stream(upstream(), provider_key="glm", on_close=bad_callback):
            got.append(item)
            if len(got) == 1:
                break
        assert got == [1]

    asyncio.run(_run())
    assert called["n"] == 1
    assert disconnect_guard.get_diagnostics()["client_disconnects"] == 1
    print("test_guard_stream_on_close_callback_error_swallowed: PASSED")


# ------------------------------------------- 回归：Leader 断连不得拖死 Follower


def test_broadcast_stream_wrapper_finishes_group_on_cancel():
    """回归缺陷 1：Leader 任务被取消后，飞行组必须进入终局，Follower 必须被唤醒。"""
    _reset_guard()

    async def _run():
        group = FlightGroup("k" * 64, debounce_ttl=0.0, stale_timeout=600.0)

        async def slow_upstream():
            yield "first"
            await asyncio.sleep(30)  # 永远不会结束，等待被取消
            yield "never"

        async def leader():
            async for _ in broadcast_stream_wrapper(slow_upstream(), group):
                pass

        leader_task = asyncio.create_task(leader())
        await asyncio.sleep(0.05)

        # Follower 挂载并消费
        follower_chunks = []

        async def follower():
            async for chunk in stream_follower_gen(group):
                follower_chunks.append(chunk)

        follower_task = asyncio.create_task(follower())
        await asyncio.sleep(0.05)

        # 模拟 Leader 客户端断连：取消 Leader 任务
        leader_task.cancel()
        try:
            await leader_task
        except asyncio.CancelledError:
            pass

        # 飞行组必须已是终局
        assert group.leader_done is True, "Leader 取消后 flight group 必须进入终局"
        assert group.error is not None, "被中止的飞行组必须携带错误信息"

        # Follower 必须被及时唤醒（收到错误传播），而不是永久挂起
        try:
            await asyncio.wait_for(follower_task, timeout=2.0)
            raise AssertionError("Follower 应当收到 Leader 中止的错误传播")
        except RuntimeError as e:
            assert "中止" in str(e) or "作废" in str(e), str(e)
        assert follower_chunks == ["first"], follower_chunks

    asyncio.run(_run())
    print("test_broadcast_stream_wrapper_finishes_group_on_cancel: PASSED")


def test_broadcast_stream_wrapper_aclose_finishes_group():
    """Leader 生成器被 aclose() 提前关闭时，不得抛 RuntimeError 且组必须终局。"""
    _reset_guard()

    async def _run():
        group = FlightGroup("m" * 64, debounce_ttl=0.0, stale_timeout=600.0)

        async def upstream():
            yield "a"
            yield "b"
            yield "c"

        wrapper = broadcast_stream_wrapper(upstream(), group)
        assert await wrapper.__anext__() == "a"
        # 提前关闭，触发 GeneratorExit 路径
        await wrapper.aclose()
        # 让 finally 中调度的 finish 任务得以执行
        await asyncio.sleep(0.05)

        assert group.leader_done is True, "提前关闭后飞行组必须进入终局"
        assert group.done_event.is_set() is True

    asyncio.run(_run())
    print("test_broadcast_stream_wrapper_aclose_finishes_group: PASSED")


def test_stale_flight_eviction_wakes_followers():
    """回归缺陷 1 的泄漏面：陈旧飞行组必须被回收，且其 Follower 必须被唤醒。"""
    _reset_guard()
    manager = SingleFlightManager(debounce_ttl=0.0)

    async def _run():
        stale = FlightGroup("s" * 64, debounce_ttl=0.0, stale_timeout=10.0)
        # 回拨创建时间，使其成为陈旧组
        stale.created_at -= 3600.0
        manager._flights[stale.key] = stale

        # 先挂一个 Follower 在其上
        history, queue, is_done = await stale.add_subscriber()
        assert is_done is False
        assert queue is not None

        follower_chunks = []

        async def follower():
            async for chunk in stream_follower_gen(stale):
                follower_chunks.append(chunk)

        follower_task = asyncio.create_task(follower())
        await asyncio.sleep(0.05)

        # 触发一次清理（get_or_create 内部会调用 _cleanup_expired_locked）
        decision = await manager.get_or_create("s" * 64)
        assert decision.is_leader is True, "陈旧组不得复用于新请求"
        assert manager._flights.get("s" * 64) is not stale, "陈旧组必须被移除并换新组"

        # Follower 必须被唤醒并结束，而不是永久挂起
        try:
            await asyncio.wait_for(follower_task, timeout=2.0)
            raise AssertionError("Follower 应当收到陈旧飞行组作废的错误传播")
        except RuntimeError as e:
            assert "作废" in str(e) or "中断" in str(e), str(e)

    asyncio.run(_run())
    assert disconnect_guard.get_diagnostics()["orphaned_flights_recovered"] >= 1
    print("test_stale_flight_eviction_wakes_followers: PASSED")


def test_follower_idle_timeout_bounded():
    """Follower 在 Leader 长期无响应时必须有界退出，绝不永久挂起。"""
    _reset_guard()

    async def _run():
        group = FlightGroup("t" * 64, debounce_ttl=0.0, stale_timeout=600.0)
        group.follower_idle_timeout = 0.2

        async def follower():
            async for _ in stream_follower_gen(group):
                pass

        task = asyncio.create_task(follower())
        try:
            await asyncio.wait_for(task, timeout=3.0)
            raise AssertionError("应当因空闲超时抛出异常")
        except RuntimeError as e:
            assert "无响应" in str(e)
        except asyncio.TimeoutError:
            raise AssertionError("Follower 空闲超时未生效，仍然挂起")

    asyncio.run(_run())
    print("test_follower_idle_timeout_bounded: PASSED")


# ------------------------------------------------------------------ 遥测与开关


def test_disconnect_guard_diagnostics():
    """校验诊断指标键集合与数值累加。"""
    guard = DisconnectGuard(stale_flight_timeout=123.0)
    guard.record_client_disconnect("qwen", estimated_tokens=500)
    guard.record_client_disconnect("qwen")
    guard.record_upstream_abort("qwen", reason="test")
    guard.record_orphaned_flight_recovered()

    diag = guard.get_diagnostics()
    for key in (
        "enabled", "client_disconnects", "upstream_aborts", "reclaimed_streams",
        "reclaimed_estimated_tokens", "orphaned_flights_recovered",
        "stale_flight_timeout_seconds", "per_provider",
    ):
        assert key in diag, key

    assert diag["client_disconnects"] == 2
    assert diag["upstream_aborts"] == 1
    assert diag["reclaimed_estimated_tokens"] == 500
    assert diag["orphaned_flights_recovered"] == 1
    assert diag["stale_flight_timeout_seconds"] == 123.0
    assert diag["per_provider"]["qwen"] == {"disconnects": 2, "aborts": 1}
    assert diag["enabled"] is True
    print("test_disconnect_guard_diagnostics: PASSED")


def test_disconnect_guard_disabled_env():
    """GATEWAY_DISABLE_DISCONNECT_GUARD=1 时 enabled 为 False。"""
    os.environ["GATEWAY_DISABLE_DISCONNECT_GUARD"] = "1"
    try:
        assert disconnect_guard.is_disabled() is True
        assert disconnect_guard.get_diagnostics()["enabled"] is False
    finally:
        os.environ.pop("GATEWAY_DISABLE_DISCONNECT_GUARD", None)
    assert disconnect_guard.is_disabled() is False
    print("test_disconnect_guard_disabled_env: PASSED")


def run_all():
    test_safe_aclose_closes_generator()
    test_safe_aclose_is_exception_safe()
    test_guard_stream_normal_completion()
    test_guard_stream_early_break_records_disconnect()
    test_guard_stream_cancellation_records_disconnect()
    test_guard_stream_on_close_callback_error_swallowed()
    test_broadcast_stream_wrapper_finishes_group_on_cancel()
    test_broadcast_stream_wrapper_aclose_finishes_group()
    test_stale_flight_eviction_wakes_followers()
    test_follower_idle_timeout_bounded()
    test_disconnect_guard_diagnostics()
    test_disconnect_guard_disabled_env()
    print("\n✅ All disconnect propagation & upstream reclamation tests (OPT-009) passed!")


if __name__ == "__main__":
    run_all()
