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

    async def safe_aclose_detached(self, agen: Any) -> bool:
        """[C2] 在**当前任务可能已被取消**的情况下可靠地关闭上游流。

        断连时的收尾代码正运行于被取消的任务中，就地 `await agen.aclose()` 会被
        立即二次取消：实测 provider 生成器的 `finally` 一次都没有执行，上游 HTTP
        连接与配额随之泄漏。这里把关闭动作调度成独立任务再 shield 等待——取消只会
        打断等待，关闭任务本身仍会跑完。

        对**从未启动**的生成器，`aclose()` 是空操作；调用方（`guard_stream`）在
        关闭前已由 `ManagedStream.close()` 推进一帧使其进入运行态，因此内部
        `finally` 能真正执行。
        """
        if agen is None or not hasattr(agen, "aclose"):
            return False
        try:
            task = asyncio.ensure_future(agen.aclose())
        except Exception as exc:  # noqa: BLE001
            logger.warning("上游流安全关闭失败 (已忽略): %s: %s", type(exc).__name__, exc)
            return False
        interrupted = False
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            # 当前任务被取消：关闭任务继续在后台完成
            interrupted = True
        except BaseException as exc:  # noqa: BLE001 - 回收路径必须绝对安全
            # 当前生成器自身正在被关闭时（事件循环收尾 / GeneratorExit），
            # 等待点会立刻以 GeneratorExit 结束——这描述的是**等待方**的处境，
            # 不是关闭任务的结局。实测此时关闭任务已正常完成、上游 finally 已执行，
            # 若据此记为失败，一次成功回收会被计成一次失败，计数器长期少计。
            interrupted = True
            logger.debug("关闭等待被中断（关闭任务本身不受影响）: %s", type(exc).__name__)
        # 结论只看任务本身：它才代表「上游流是否真的关掉了」。
        if task.done() and not task.cancelled():
            outcome = task.exception()
            if outcome is not None:
                logger.warning(
                    "上游流安全关闭失败 (已忽略): %s: %s", type(outcome).__name__, outcome
                )
                return False
            with self._lock:
                self._stats["reclaimed_streams"] += 1
            return True
        if not interrupted:
            return False
        # 等待被打断且任务尚未完成：关闭仍在后台进行，交给 done_callback 记账，
        # 这样「确实关掉了」的流不会被永久漏计。
        task.add_done_callback(self._record_reclaim_outcome)
        return False

    def _record_reclaim_outcome(self, task: "asyncio.Future[Any]") -> None:
        """关闭任务在等待被中断后才得出结论时的记账入口。

        等待被打断不代表关闭失败：任务仍然会在后台跑完。这里按任务**真实的**
        结局记账——成功则计入 reclaimed_streams，失败则记录一次告警。回收路径
        必须绝对安全，因此回调自身不得抛出。
        """
        try:
            if task.cancelled():
                return
            outcome = task.exception()
        except BaseException:  # noqa: BLE001 - 回收路径必须绝对安全
            return
        if outcome is not None:
            logger.warning(
                "上游流安全关闭失败 (已忽略): %s: %s", type(outcome).__name__, outcome
            )
            return
        with self._lock:
            self._stats["reclaimed_streams"] += 1

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
        # 1. 先做**同步**回收（槽位/租约），它不受当前任务取消状态影响
        release_sync = getattr(gen, "release_sync", None)
        if callable(release_sync):
            try:
                release_sync()
            except Exception as exc:  # noqa: BLE001
                logger.warning("同步资源回收失败 (已忽略): %s", exc)

        # 2. 显式关闭上游流，令 provider 侧 httpx 连接立即释放。
        #    [C2] 客户端断连时本生成器正运行在被取消的任务里：就地 await 会被
        #    立即二次取消，上游流根本关不掉（实测断连路径上 provider 的 finally
        #    从未执行，连接与上游配额随之泄漏）。调度独立任务并 shield 等待。
        await disconnect_guard.safe_aclose_detached(gen)

        # 3. 提前中止时记录断连遥测并触发回调
        if not completed:
            disconnect_guard.record_client_disconnect(provider_key)
            logger.info("客户端断连，已回收上游流 [%s]", provider_key or "unknown")
            if on_close is not None:
                try:
                    await asyncio.shield(asyncio.ensure_future(on_close()))
                except BaseException as cb_exc:  # noqa: BLE001 - 回调异常不得掩盖原始异常
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
