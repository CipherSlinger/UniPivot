"""智能并发语义去重与流式请求合并引擎 (SingleFlight & Stream Fanout Engine)

实现 [P0] OPT-008：
1. 请求指纹计算 (Deterministic Fingerprint Hashing):
   - 基于 provider、model、messages 内容、tools、temperature、thinking 与 stream 模式生成确定性 SHA-256 签名。
2. 流式单飞多路广播 (Streaming SingleFlight Multi-Subscriber Fanout):
   - 首个请求作为 Leader 发起上游调用，并维护历史重放缓存 (Replay Buffer)；
   - 执行期间并发到达的相同指纹请求作为 Follower 挂载，无缝重放已产生的历史帧，并实时订阅后续广播帧；
   - 彻底杜绝并发重复请求造成的上游并发超限、429 频控熔断与 WAF 拦截风暴；
   - 异常安全隔离：某个 Follower 断开连接不影响 Leader 与其他 Follower。
3. 非流式单飞聚合 (Non-Streaming Coalescing):
   - Follower 挂载等待 Leader 完成，直接共享上游响应 JSON 结果。
4. 遥测响应头与指标诊断:
   - x-singleflight: leader | follower
   - x-singleflight-followers: <N>
   - x-singleflight-key: <hash[:12]>
   - 暴露在 GET /v1/diagnostics 供仪表盘与自动化探针审计。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

logger = logging.getLogger("gateway.singleflight")

_SENTINEL = object()


@dataclass
class SingleFlightDecision:
    """单飞调度决策。"""
    is_leader: bool
    flight_key: str
    short_key: str
    group: FlightGroup


class FlightGroup:
    """单个飞行中请求的协同广播组。"""

    def __init__(self, key: str, debounce_ttl: float = 2.0, stale_timeout: Optional[float] = None):
        self.key = key
        self.short_key = key[:12]
        self.created_at = time.time()
        self.debounce_ttl = debounce_ttl
        if stale_timeout is None:
            stale_timeout = float(os.getenv("GATEWAY_STALE_FLIGHT_TIMEOUT", "600.0"))
        self.stale_timeout = stale_timeout
        self.follower_idle_timeout = float(
            os.getenv("GATEWAY_FOLLOWER_IDLE_TIMEOUT", str(stale_timeout))
        )
        self.history: List[Any] = []
        self.subscribers: List[asyncio.Queue] = []
        self._lock = asyncio.Lock()
        self.done_event = asyncio.Event()
        self.completed_result: Optional[Any] = None
        self.completed_headers: Optional[Dict[str, str]] = None
        self.error: Optional[Exception] = None
        self.leader_done = False
        self.follower_count = 0
        self.finished_at: Optional[float] = None

    async def add_subscriber(self) -> Tuple[List[Any], Optional[asyncio.Queue], bool]:
        """Follower 挂载订阅。

        返回: (history_snapshot, queue, is_already_done)
        """
        async with self._lock:
            history_snapshot = list(self.history)
            if self.leader_done:
                return history_snapshot, None, True
            q = asyncio.Queue()
            self.subscribers.append(q)
            return history_snapshot, q, False

    async def remove_subscriber(self, q: asyncio.Queue) -> None:
        """移除已退出的订阅者。"""
        async with self._lock:
            if q in self.subscribers:
                self.subscribers.remove(q)

    async def broadcast_chunk(self, item: Any) -> None:
        """Leader 向所有订阅者分发增量数据帧，并存入历史重放缓冲区。"""
        async with self._lock:
            self.history.append(item)
            for q in list(self.subscribers):
                try:
                    q.put_nowait(item)
                except Exception:
                    pass

    async def finish(
        self,
        result: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
        error: Optional[Exception] = None,
    ) -> None:
        """Leader 完成流式分发或非流式计算。"""
        async with self._lock:
            self.leader_done = True
            self.finished_at = time.time()
            self.completed_result = result
            self.completed_headers = headers
            self.error = error
            for q in list(self.subscribers):
                try:
                    q.put_nowait(error if error else _SENTINEL)
                except Exception:
                    pass
            self.done_event.set()

    def is_stale(self, now: float) -> bool:
        """Leader 长时间未终结的飞行组视为陈旧（客户端断连泄漏），应立即作废。"""
        if self.leader_done:
            return False
        return (now - self.created_at) >= self.stale_timeout

    def is_expired(self, now: float) -> bool:
        """检查该飞行组是否已过防抖保留期。"""
        # 陈旧组（Leader 断连后永不终结）必须优先作废，否则永不清理且拖死后续 Follower
        if self.is_stale(now):
            return True
        if not self.leader_done or self.finished_at is None:
            return False
        # 异常失败的飞行组不进入保留期，立即过期以防错误扩散
        if self.error is not None:
            return True
        return (now - self.finished_at) >= self.debounce_ttl


class SingleFlightManager:
    """智能并发请求去重与单飞分发管理器。"""

    def __init__(self, debounce_ttl: Optional[float] = None):
        if debounce_ttl is None:
            self.debounce_ttl = float(os.getenv("GATEWAY_SINGLEFLIGHT_DEBOUNCE_TTL", "0.0"))
        else:
            self.debounce_ttl = debounce_ttl
        self._flights: Dict[str, FlightGroup] = {}
        self._lock = asyncio.Lock()
        self._stats = {
            "total_flights": 0,
            "leader_executions": 0,
            "follower_shares": 0,
        }

    def is_disabled(self) -> bool:
        """判断是否禁用 SingleFlight。"""
        return os.getenv("GATEWAY_DISABLE_SINGLEFLIGHT", "").strip().lower() in ("1", "true", "yes")

    @staticmethod
    def compute_key(
        provider: str,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        thinking: Optional[bool] = None,
        stream: bool = False,
        protocol: str = "openai_chat",
    ) -> str:
        """计算请求特征指纹 (SHA-256)。"""
        payload = {
            "protocol": protocol,
            "provider": provider,
            "model": model,
            "messages": messages,
            "tools": tools or [],
            "temperature": round(temperature or 0.7, 2),
            "thinking": bool(thinking),
            "stream": stream,
        }
        raw_str = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw_str.encode("utf-8")).hexdigest()

    async def get_or_create(self, key: str) -> SingleFlightDecision:
        """获取现有活跃飞行组或创建新组（决定 Leader / Follower 角色）。"""
        now = time.time()
        async with self._lock:
            self._cleanup_expired_locked(now)

            if key in self._flights:
                group = self._flights[key]
                # 异常失败的飞行组、或已完成且防抖期已满的飞行组，绝不复用于新请求
                if group.leader_done and (group.error or group.is_expired(now)):
                    self._flights.pop(key, None)
                else:
                    group.follower_count += 1
                    self._stats["follower_shares"] += 1
                    return SingleFlightDecision(
                        is_leader=False,
                        flight_key=key,
                        short_key=group.short_key,
                        group=group,
                    )

            group = FlightGroup(key, debounce_ttl=self.debounce_ttl)
            self._flights[key] = group
            self._stats["total_flights"] += 1
            self._stats["leader_executions"] += 1
            return SingleFlightDecision(
                is_leader=True,
                flight_key=key,
                short_key=group.short_key,
                group=group,
            )

    async def cleanup(self, key: str) -> None:
        """清理已完成的飞行组（支持被动或主动清理）。"""
        async with self._lock:
            self._flights.pop(key, None)

    def _cleanup_expired_locked(self, now: float) -> None:
        """清理超时的已完成组与 Leader 断连遗留的陈旧组（需在持有 _lock 时调用）。"""
        from .disconnect_guard import reap_orphaned_flight

        expired = [k for k, g in self._flights.items() if g.is_expired(now)]
        for k in expired:
            group = self._flights.pop(k, None)
            if group is None:
                continue
            # 陈旧组：Leader 已消失但组内可能仍挂着 Follower，必须唤醒后再作废，
            # 否则 Follower 将在 queue.get() 上永久挂起。
            if group.is_stale(now):
                reap_orphaned_flight(group, now)
                logger.warning(
                    "回收陈旧飞行组 [%s]：Leader 长时间未终结 (存活 %.1fs)，已唤醒 %d 个等待节点",
                    group.short_key, now - group.created_at, len(group.subscribers),
                )

    def clear(self) -> None:
        """重置管理器状态（测试用）。"""
        self._flights.clear()
        self._stats = {
            "total_flights": 0,
            "leader_executions": 0,
            "follower_shares": 0,
        }

    def get_diagnostics(self) -> Dict[str, Any]:
        """获取 SingleFlight 运行指标统计。"""
        now = time.time()
        active = sum(1 for g in self._flights.values() if not g.leader_done)
        completed = sum(1 for g in self._flights.values() if g.leader_done)
        total_reqs = self._stats["leader_executions"] + self._stats["follower_shares"]
        saved_ratio = round(self._stats["follower_shares"] / total_reqs, 4) if total_reqs > 0 else 0.0

        return {
            "enabled": not self.is_disabled(),
            "active_flights": active,
            "cached_completed_flights": completed,
            "total_flights": self._stats["total_flights"],
            "leader_executions": self._stats["leader_executions"],
            "follower_shares": self._stats["follower_shares"],
            "saved_ratio": saved_ratio,
            "debounce_ttl_seconds": self.debounce_ttl,
        }


# 全局单例
singleflight_manager = SingleFlightManager()


async def broadcast_stream_wrapper(
    gen: AsyncGenerator[Any, None],
    group: FlightGroup,
) -> AsyncGenerator[Any, None]:
    """包装 Leader 流式生成器，透明向所有 Follower 广播数据帧。

    无论正常结束、上游异常、还是 Leader 客户端断连（CancelledError /
    GeneratorExit），都必须让飞行组进入终局，否则已挂载的 Follower 会在
    stream_follower_gen 的 queue.get() 上永久挂起。
    """
    finished = False
    try:
        async for chunk in gen:
            await group.broadcast_chunk(chunk)
            yield chunk
        await group.finish()
        finished = True
    except GeneratorExit:
        # 生成器被 aclose() 提前关闭。此处绝不可 await（会触发
        # "async generator ignored GeneratorExit"），统一交给 finally 调度。
        raise
    except BaseException as exc:  # noqa: BLE001 - 必须覆盖 CancelledError
        # CancelledError 等非 Exception 需归一化为可向 Follower 传递的异常
        if isinstance(exc, Exception):
            err: BaseException = exc
        else:
            err = RuntimeError(f"Leader 流被中止: {type(exc).__name__}")
        await group.finish(error=err)
        finished = True
        raise
    finally:
        if not finished:
            # 兜底：以独立任务标记终局，绝不阻塞生成器关闭流程。
            try:
                asyncio.get_running_loop().create_task(
                    group.finish(error=RuntimeError("Leader 流提前关闭，飞行组已作废"))
                )
            except RuntimeError:
                # 无运行中的事件循环，退化为同步标记终局
                group.leader_done = True
                group.error = RuntimeError("Leader 流提前关闭，飞行组已作废")
                group.done_event.set()


def terminate_aborted_group(group: Optional[FlightGroup]) -> None:
    """请求中断（客户端断连 CancelledError / 其它 BaseException）时的飞行组终局兜底。

    非流式 Leader 在首个上游响应或聚合上游流期间被中断时，既有的
    ``except ProviderError`` / ``except Exception`` 分支不会执行——因为
    ``CancelledError`` 继承自 ``BaseException`` 而非 ``Exception``。若不在
    ``finally`` 中补齐终局，飞行组将永远停留在 ``leader_done=False``，导致：

    1. 已挂载的 Follower 在 ``done_event`` 上阻塞至超时（默认 TIMEOUT=600s），
       期间客户端只能干等；
    2. 该指纹对应的死组长期驻留 ``_flights``，使后续同指纹请求持续错误地
       挂载到一个永不终结的组上。

    本函数以独立任务补齐终局，**绝不 await**：调用点通常正处于任务取消状态，
    任何 await 都会被立即二次取消。组已终局时为空操作（幂等）。
    """
    if group is None or group.leader_done:
        return
    from .disconnect_guard import reap_orphaned_flight

    reap_orphaned_flight(group)


async def stream_follower_gen(group: FlightGroup) -> AsyncGenerator[Any, None]:
    """为 Follower 客户端消费回放与实时广播队列。"""
    history_snapshot, queue, is_already_done = await group.add_subscriber()

    # 1. 回放历史已产生的数据帧
    for chunk in history_snapshot:
        yield chunk

    if is_already_done:
        if group.error:
            raise group.error
        return

    # 2. 消费后续实时广播的数据帧
    try:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=group.follower_idle_timeout)
            except asyncio.TimeoutError:
                # Leader 断连且未被及时回收时的最终兜底：绝不让 Follower 永久挂起
                raise RuntimeError(
                    f"Leader 流无响应超过 {group.follower_idle_timeout:.0f}s，飞行组已作废"
                )
            if item is _SENTINEL:
                break
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        await group.remove_subscriber(queue)
