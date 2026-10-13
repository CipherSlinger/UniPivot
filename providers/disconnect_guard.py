"""客户端断连传播与上游资源回收引擎 (Client Disconnect Propagation & Upstream Resource Reclamation Engine)

实现 [P0] OPT-009：
1. 断连传播 (Disconnect Propagation):
   - 当客户端（Claude Code CLI / Codex CLI / 网页端）中途放弃流式响应时，
     立即将中止信号传播至上游异步生成器，杜绝「客户端已走、上游仍在生成」的空耗。
2. 上游资源即时回收 (Upstream Stream Reclamation):
   - 显式 aclose() 上游流，令 provider 侧 `async with client.stream(...)` 立即退出，
     释放 HTTP/2 连接槽位与上游并发配额，避免风控敏感节点的配额被僵尸流占用。
3. 悬挂飞行组回收 (Orphaned Flight Recovery):
   - 配合 providers/singleflight.py，回收 Leader 断连后遗留的飞行组，
     唤醒其上永久等待的 Follower，防止连接与内存无界泄漏。
4. 遥测与响应头透传:
   - x-disconnect-guard: enabled | disabled
   - 暴露在 GET /v1/diagnostics 的 disconnect_guard 段供仪表盘与自动化探针审计。
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, Optional

logger = logging.getLogger("gateway.disconnect_guard")


class DisconnectGuard:
    """客户端断连传播与上游资源回收守卫。"""

    def __init__(self, stale_flight_timeout: Optional[float] = None) -> None:
        if stale_flight_timeout is None:
            stale_flight_timeout = float(os.getenv("GATEWAY_STALE_FLIGHT_TIMEOUT", "600.0"))
        self.stale_flight_timeout = stale_flight_timeout
        self._lock = threading.Lock()
        self._per_provider: Dict[str, Dict[str, int]] = {}
        self._stats = {
            "client_disconnects": 0,
            "upstream_aborts": 0,
            "reclaimed_streams": 0,
            "reclaimed_estimated_tokens": 0,
            "orphaned_flights_recovered": 0,
        }

    def is_disabled(self) -> bool:
        """判断是否禁用断连回收守卫。"""
        return os.getenv("GATEWAY_DISABLE_DISCONNECT_GUARD", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )

    def _provider_bucket_locked(self, provider_key: str) -> Dict[str, int]:
        """获取（或惰性创建）指定 Provider 的分项计数器（内部方法，需持有锁）。"""
        key = (provider_key or "unknown").lower()
        bucket = self._per_provider.get(key)
        if bucket is None:
            bucket = {"disconnects": 0, "aborts": 0}
            self._per_provider[key] = bucket
        return bucket

    async def safe_aclose(self, agen: Any) -> bool:
        """安全关闭一个异步生成器。

        无论底层 aclose() 抛出何种异常（含 BaseException），均吞掉并记录，
        保证回收动作本身绝不会破坏调用方的正常控制流。
        成功关闭返回 True。
        """
        if agen is None or not hasattr(agen, "aclose"):
            return False
        try:
            await agen.aclose()
        except BaseException as exc:  # noqa: BLE001 - 回收路径必须绝对安全
            logger.warning("上游流安全关闭失败 (已忽略): %s: %s", type(exc).__name__, exc)
            return False
        with self._lock:
            self._stats["reclaimed_streams"] += 1
        return True

    def record_client_disconnect(self, provider_key: str, estimated_tokens: int = 0) -> None:
        """记录一次客户端断连事件。"""
        with self._lock:
            self._stats["client_disconnects"] += 1
            if estimated_tokens > 0:
                self._stats["reclaimed_estimated_tokens"] += estimated_tokens
            self._provider_bucket_locked(provider_key)["disconnects"] += 1

    def record_upstream_abort(self, provider_key: str, reason: str = "") -> None:
        """记录一次上游中止（主动 aclose 或异常中止）。"""
        with self._lock:
            self._stats["upstream_aborts"] += 1
            self._provider_bucket_locked(provider_key)["aborts"] += 1
        if reason:
            logger.info("上游流中止回收 [%s]: %s", provider_key or "unknown", reason)

    def record_orphaned_flight_recovered(self) -> None:
        """记录一次悬挂飞行组回收。"""
        with self._lock:
            self._stats["orphaned_flights_recovered"] += 1

    def get_diagnostics(self) -> dict:
        """获取断连回收与上游资源治理统计指标。"""
        with self._lock:
            return {
                "enabled": not self.is_disabled(),
                "client_disconnects": self._stats["client_disconnects"],
                "upstream_aborts": self._stats["upstream_aborts"],
                "reclaimed_streams": self._stats["reclaimed_streams"],
                "reclaimed_estimated_tokens": self._stats["reclaimed_estimated_tokens"],
                "orphaned_flights_recovered": self._stats["orphaned_flights_recovered"],
                "stale_flight_timeout_seconds": self.stale_flight_timeout,
                "per_provider": {
                    k: dict(v) for k, v in self._per_provider.items()
                },
            }

    def clear(self) -> None:
        """重置全部计数器（测试用）。"""
        with self._lock:
            self._per_provider = {}
            self._stats = {
                "client_disconnects": 0,
                "upstream_aborts": 0,
                "reclaimed_streams": 0,
                "reclaimed_estimated_tokens": 0,
                "orphaned_flights_recovered": 0,
            }


# 全局单例
disconnect_guard = DisconnectGuard()


async def guard_stream(
    gen: AsyncGenerator[Any, None],
    provider_key: str = "",
    on_close: Optional[Callable[[], Awaitable[None]]] = None,
) -> AsyncGenerator[Any, None]:
    """包装上游异步生成器：无论消费者正常结束、异常、还是提前断连，
    都保证显式 aclose() 上游流并回收资源，同时记录遥测。

    - 正常迭代完毕 -> 不计入 client_disconnects，但仍回收（aclose）上游流。
    - 消费者异常   -> 异常照常向上传播，finally 中仍执行回收。
    - 提前断连/取消 -> 计入 client_disconnects，并调用 on_close 回调。
    """
    if gen is None:
        return

    completed = False
    try:
        async for item in gen:
            yield item
        completed = True
    finally:
        # 1. 显式关闭上游流，令 provider 侧 httpx 连接立即释放
        await disconnect_guard.safe_aclose(gen)

        # 2. 提前中止时记录断连遥测并触发回调
        if not completed:
            disconnect_guard.record_client_disconnect(provider_key)
            logger.info("客户端断连，已回收上游流 [%s]", provider_key or "unknown")
            if on_close is not None:
                try:
                    await on_close()
                except Exception as cb_exc:  # noqa: BLE001 - 回调异常不得掩盖原始异常
                    logger.warning("断连回调执行失败 (已忽略): %s", cb_exc)


def reap_orphaned_flight(group: Any, now: Optional[float] = None) -> None:
    """回收一个 Leader 已消失的悬挂飞行组。

    以「组内所有等待者被唤醒 + 组标记为终局」的方式作废该组，
    避免 Follower 在 queue.get() 上永久挂起。
    """
    err = RuntimeError("Leader 流已中断，飞行组作废（客户端断连）")
    disconnect_guard.record_orphaned_flight_recovered()
    finish = getattr(group, "finish", None)
    if finish is None:
        return
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # 无运行中的事件循环：退化为同步标记终局，绝不让组停留在非终局状态
        group.leader_done = True
        group.error = err
        group.done_event.set()
        return
    try:
        task = asyncio.ensure_future(finish(error=err))
    except BaseException:  # noqa: BLE001 - 调度失败也必须保证组终局
        group.leader_done = True
        group.error = err
        group.done_event.set()
        return
    task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
