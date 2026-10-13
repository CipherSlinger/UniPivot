"""请求生命周期终局保障与并发配额确定性回收引擎 (OPT-010) 测试套件

验证：
1. terminate_aborted_group：飞行组中断终局兜底（幂等 / 已终局空操作 / 唤醒 Follower）
2. 非流式 Leader 在首个上游响应阶段被 CancelledError（客户端断连语义）中断时：
   - 飞行组必须进入终局，绝不残留永不终结的死组
   - load_balancer 并发配额必须被确定性释放（回归：Responses 端点曾永久泄漏槽位）
3. 三大协议（OpenAI Chat / Anthropic Messages / Responses）端点级覆盖
4. 回归：正常成功的非流式请求不受影响（组正常终局、结果正确、不误判为中断）

全部离线运行，无任何外部网络请求。

运行: .venv/bin/python tests/test_lifecycle.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["GATEWAY_DISABLE_AUTO_REFRESH"] = "1"
os.environ["TESTING"] = "1"
os.environ["GATEWAY_DISABLE_PACING"] = "1"
os.environ["QWEN_AUTH_TOKEN"] = "fake_qwen"
os.environ["DEEPSEEK_AUTH_TOKEN"] = "fake_ds"
os.environ["DOUBAO_AUTH_TOKEN"] = "fake_doubao"
os.environ["KIMI_AUTH_TOKEN"] = "fake_kimi"
os.environ["GLM_AUTH_TOKEN"] = "fake_glm"

import server as srv  # noqa: E402
from providers.anthropic_compat import AnthropicMessagesRequest  # noqa: E402
from providers.disconnect_guard import disconnect_guard  # noqa: E402
from providers.responses_compat import ResponsesRequest  # noqa: E402
from providers.singleflight import (  # noqa: E402
    FlightGroup,
    stream_follower_gen,
    terminate_aborted_group,
)


# ---------------------------------------------------------------- 测试夹具


def _reset() -> None:
    srv.singleflight_manager.clear()
    disconnect_guard.clear()


def _patch_slow_upstream():
    """替换首个上游调用为永不返回的挂起，用于在调用中被取消。"""
    original = srv.failover_router.execute_chat

    async def _slow(*_args, **_kwargs):
        await asyncio.sleep(30)
        return None, None, None  # pragma: no cover - 永远到不了

    srv.failover_router.execute_chat = _slow
    return original


def _patch_fast_upstream():
    """替换首个上游调用为立即返回的确定性流，用于验证正常路径。"""
    original = srv.failover_router.execute_chat

    class _DummyProvider:
        async def delete_conversation(self, *_a, **_k):  # pragma: no cover
            return None

    async def _stream():
        for chunk in ("你好", "，", "世界"):
            yield chunk, {"conversation_id": "conv_lifecycle_010"}

    async def _fast(*_args, **_kwargs):
        return _stream(), None, _DummyProvider()

    srv.failover_router.execute_chat = _fast
    return original


def _non_terminal_groups() -> list:
    return [g for g in srv.singleflight_manager._flights.values() if not g.leader_done]


async def _cancel_in_flight(coro, provider_key: str = "qwen", timeout: float = 5.0) -> bool:
    """启动端点协程，待其进入上游调用（并发配额已占用）后取消，模拟客户端断连。

    返回是否如期观察到 CancelledError。
    """
    task = asyncio.create_task(coro)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if srv.load_balancer.get_metrics(provider_key).active_concurrency >= 1:
            break
        if task.done():
            break
        await asyncio.sleep(0.01)

    task.cancel()
    cancelled = False
    try:
        await task
    except asyncio.CancelledError:
        cancelled = True
    # 让 finally 中调度的 group.finish 独立任务得以执行
    await asyncio.sleep(0.05)
    return cancelled


def _chat_request(**kwargs):
    payload = {
        "model": "qwen",
        "stream": False,
        "messages": [srv.ChatMessage(role="user", content="生命周期终局化回归")],
    }
    payload.update(kwargs)
    return srv.ChatCompletionRequest(**payload)


def _anthropic_request(**kwargs):
    payload = {
        "model": "claude-3-7-sonnet",
        "stream": False,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "生命周期终局化回归"}],
    }
    payload.update(kwargs)
    return AnthropicMessagesRequest(**payload)


def _responses_request(**kwargs):
    payload = {"model": "gpt-4o", "stream": False, "input": "生命周期终局化回归"}
    payload.update(kwargs)
    return ResponsesRequest(**payload)


# ---------------------------------------------------------------- 单元：终局兜底


def test_terminate_aborted_group_finalizes_group():
    """中断兜底必须把组置为终局并携带错误。"""
    _reset()

    async def _run():
        group = FlightGroup("a" * 64, debounce_ttl=0.0, stale_timeout=600.0)
        assert group.leader_done is False
        terminate_aborted_group(group)
        await asyncio.sleep(0)
        assert group.leader_done is True, "中断兜底后组必须终局"
        assert group.error is not None, "被作废的组必须携带错误信息"
        assert group.done_event.is_set() is True

    asyncio.run(_run())
    assert disconnect_guard.get_diagnostics()["orphaned_flights_recovered"] == 1
    print("test_terminate_aborted_group_finalizes_group: PASSED")


def test_terminate_aborted_group_is_idempotent_and_noop_after_finish():
    """已终局的组（如正常完成）绝不能被兜底覆盖，且重复调用不产生副作用。"""
    _reset()

    async def _run():
        group = FlightGroup("b" * 64, debounce_ttl=0.0, stale_timeout=600.0)
        await group.finish(result={"ok": True}, headers={"x-a": "1"})

        before = disconnect_guard.get_diagnostics()["orphaned_flights_recovered"]
        terminate_aborted_group(group)
        terminate_aborted_group(group)
        terminate_aborted_group(None)
        await asyncio.sleep(0)

        assert group.error is None, "正常完成的组不得被兜底注入错误"
        assert group.completed_result == {"ok": True}, "正常结果不得被覆盖"
        assert group.completed_headers == {"x-a": "1"}
        assert disconnect_guard.get_diagnostics()["orphaned_flights_recovered"] == before, \
            "已终局组不得计入悬挂回收"

    asyncio.run(_run())
    print("test_terminate_aborted_group_is_idempotent_and_noop_after_finish: PASSED")


def test_terminate_aborted_group_wakes_follower():
    """终端兜底必须唤醒已挂载的 Follower，而不是让其永久挂起。"""
    _reset()

    async def _run():
        group = FlightGroup("c" * 64, debounce_ttl=0.0, stale_timeout=600.0)
        received: list = []

        async def follower():
            async for chunk in stream_follower_gen(group):
                received.append(chunk)

        task = asyncio.create_task(follower())
        await asyncio.sleep(0.05)
        assert len(group.subscribers) == 1, "Follower 应已挂载"

        terminate_aborted_group(group)

        try:
            await asyncio.wait_for(task, timeout=2.0)
            raise AssertionError("Follower 应当收到中断传播的异常")
        except RuntimeError as e:
            assert "中断" in str(e) or "作废" in str(e), str(e)
        assert received == []

    asyncio.run(_run())
    print("test_terminate_aborted_group_wakes_follower: PASSED")


def test_terminate_aborted_group_without_running_loop():
    """无运行事件循环的同步上下文也必须安全终局（绝不抛异常、绝不留死组）。"""
    _reset()
    group = FlightGroup("d" * 64, debounce_ttl=0.0, stale_timeout=600.0)
    terminate_aborted_group(group)
    assert group.leader_done is True
    assert group.error is not None
    assert group.done_event.is_set() is True
    print("test_terminate_aborted_group_without_running_loop: PASSED")


# ---------------------------------------------------------------- 端点级：中断回归


def test_chat_nonstream_leader_abort_finalizes_group():
    """回归：OpenAI Chat 非流式 Leader 中断后不得残留死组、不得泄漏并发配额。"""
    _reset()
    original = _patch_slow_upstream()
    try:
        cancelled = asyncio.run(_cancel_in_flight(srv.chat_completions(_chat_request())))
    finally:
        srv.failover_router.execute_chat = original

    assert cancelled is True, "端点应如期被取消"
    assert disconnect_guard.get_diagnostics()["orphaned_flights_recovered"] >= 1, \
        "中断的 Leader 必须被识别并作废"
    assert _non_terminal_groups() == [], "不得残留永不终结的飞行组"
    assert srv.load_balancer.get_metrics("qwen").active_concurrency == 0, \
        "中断后并发配额必须被确定性回收"
    print("test_chat_nonstream_leader_abort_finalizes_group: PASSED")


def test_anthropic_nonstream_leader_abort_finalizes_group():
    """回归：Anthropic Messages 非流式 Leader 中断后同样必须终局化并回收配额。"""
    _reset()
    original = _patch_slow_upstream()
    try:
        cancelled = asyncio.run(_cancel_in_flight(srv.anthropic_messages(_anthropic_request())))
    finally:
        srv.failover_router.execute_chat = original

    assert cancelled is True
    assert disconnect_guard.get_diagnostics()["orphaned_flights_recovered"] >= 1
    assert _non_terminal_groups() == []
    assert srv.load_balancer.get_metrics("qwen").active_concurrency == 0
    print("test_anthropic_nonstream_leader_abort_finalizes_group: PASSED")


def test_responses_abort_releases_concurrency_slot():
    """回归缺陷：Responses 端点在首个上游响应阶段中断曾永久泄漏并发槽位。"""
    _reset()
    original = _patch_slow_upstream()
    try:
        cancelled = asyncio.run(
            _cancel_in_flight(srv.responses_endpoint(_responses_request(stream=False)))
        )
    finally:
        srv.failover_router.execute_chat = original

    assert cancelled is True
    assert srv.load_balancer.get_metrics("qwen").active_concurrency == 0, \
        "Responses 端点中断后并发槽位必须被回收（绝不永久泄漏）"
    assert disconnect_guard.get_diagnostics()["orphaned_flights_recovered"] >= 1
    assert _non_terminal_groups() == []
    print("test_responses_abort_releases_concurrency_slot: PASSED")


def test_responses_stream_abort_releases_concurrency_slot():
    """Responses 流式请求在首个上游响应前中断时同样必须回收并发配额。"""
    _reset()
    original = _patch_slow_upstream()
    try:
        cancelled = asyncio.run(
            _cancel_in_flight(srv.responses_endpoint(_responses_request(stream=True)))
        )
    finally:
        srv.failover_router.execute_chat = original

    assert cancelled is True
    assert srv.load_balancer.get_metrics("qwen").active_concurrency == 0
    print("test_responses_stream_abort_releases_concurrency_slot: PASSED")


# ---------------------------------------------------------------- 回归：正常路径不受影响


def test_successful_nonstream_flow_is_unaffected():
    """正常成功的非流式请求：组正常终局、结果正确、配额正常释放、不误判中断。"""
    _reset()
    before = disconnect_guard.get_diagnostics()["orphaned_flights_recovered"]
    original = _patch_fast_upstream()
    try:
        resp = asyncio.run(srv.chat_completions(_chat_request()))
    finally:
        srv.failover_router.execute_chat = original

    assert resp.status_code == 200, getattr(resp, "body", resp)
    assert disconnect_guard.get_diagnostics()["orphaned_flights_recovered"] == before, \
        "正常路径绝不能被误判为中断并计入悬挂回收"
    assert srv.load_balancer.get_metrics("qwen").active_concurrency == 0
    groups = list(srv.singleflight_manager._flights.values())
    assert groups and all(g.leader_done for g in groups), "正常结束后组必须已终局"
    assert all(g.error is None for g in groups), "正常结束的组不得携带错误"
    print("test_successful_nonstream_flow_is_unaffected: PASSED")


def run_all():
    test_terminate_aborted_group_finalizes_group()
    test_terminate_aborted_group_is_idempotent_and_noop_after_finish()
    test_terminate_aborted_group_wakes_follower()
    test_terminate_aborted_group_without_running_loop()
    test_chat_nonstream_leader_abort_finalizes_group()
    test_anthropic_nonstream_leader_abort_finalizes_group()
    test_responses_abort_releases_concurrency_slot()
    test_responses_stream_abort_releases_concurrency_slot()
    test_successful_nonstream_flow_is_unaffected()
    print("\n✅ All request lifecycle finalization & quota reclamation tests (OPT-010) passed!")


if __name__ == "__main__":
    run_all()
