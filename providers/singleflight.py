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

# [M4] 全进程累计的「因消费过慢被摘除」跟随者数。放在模块级而不是实例上，
# 是因为多个端点/多个管理器实例都会广播，而运维关心的是「这条路是否在丢客户端」。
_DETACHED_STATS: Dict[str, int] = {"followers_detached": 0}


def _bump_detached() -> None:
    _DETACHED_STATS["followers_detached"] += 1


def get_follower_backpressure_stats() -> Dict[str, int]:
    """[M4] 背压遥测：被摘除的慢跟随者总数。"""
    return dict(_DETACHED_STATS)


def follower_backlog_limit() -> int:
    """[M4] 单个跟随者允许积压的帧数上限（默认 256）。唯一一份定义。"""
    try:
        return max(1, int(os.getenv("GATEWAY_SINGLEFLIGHT_BACKLOG", "256")))
    except ValueError:
        return 256


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
        # [M4] 跟随者背压状态。见 `broadcast_chunk` 的注释：
        # 「一帧都不消费的跟随者」必须有明确结局，否则它的队列会被无限灌满。
        self._followers_detached = 0
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
            # 有新的跟随者挂上来，说明上一轮背压已经结束（那几位已经退出队列），
            # 重置摘除标记，避免一次慢客户端永久关闭本组的跟随能力。
            self._followers_detached = 0
            return history_snapshot, q, False

    def _follower_backlog_limit(self) -> int:
        """[M4] 背压上限（委托给模块级唯一判据）。"""
        return follower_backlog_limit()

    async def remove_subscriber(self, q: asyncio.Queue) -> None:
        """移除已退出的订阅者。"""
        async with self._lock:
            if q in self.subscribers:
                self.subscribers.remove(q)

    async def broadcast_chunk(self, item: Any) -> None:
        """Leader 向所有订阅者分发增量数据帧，并存入历史重放缓冲区。

        [M4] 跟随者必须有**背压下限**，否则「一帧都不消费的客户端」会被无声地
        喂到内存耗尽：队列是无界的（`asyncio.Queue()` 默认 maxsize=0），而这里
        `put_nowait` 永不失败，异常分支实际是死代码。实测：Leader 广播 20000 帧
        512 字节后，单个不消费的跟随者队列里躺着 20000 帧 —— 约 10 MB，且完全
        没有任何上限；正常的速度差（客户端在慢网络上、或 SSE 客户端处理卡顿）
        会比任何人手写慢得多的测试更早地把它堆起来。

        正确的处置不是**阻塞** Leader（那会用一个慢客户端拖住所有跟随者，
        正是 SingleFlight 要消除的问题），而是**摘除**该跟随者并让它知道自己被
        摘除：队列里放一个错误，`stream_follower_gen` 会把它抛给该客户端——
        它拿到的是「本路共享流已不可靠，请重试」的明确答复，而不是永远等下去，
        也不是被静默截断成半截回答。这台机器上只有它一个人，因此保留 Leader 对
        其余跟随者的广播。
        """
        limit = self._follower_backlog_limit()
        async with self._lock:
            self.history.append(item)
            for q in list(self.subscribers):
                if q.qsize() >= limit:
                    self.subscribers.remove(q)
                    self._followers_detached += 1
                    _bump_detached()
                    # 摘除时**清空**已积压的帧：留着它们既占着这笔内存（正是要
                    # 防的东西），又没有任何用处——该跟随者收到的结局是「重试」，
                    # 而不是「读完这半截再重试」。
                    while not q.empty():
                        try:
                            q.get_nowait()
                        except Exception:  # noqa: BLE001
                            break
                    try:
                        q.put_nowait(
                            RuntimeError(
                                f"本路共享流消费过慢（积压超过 {limit} 帧），已被摘除；"
                                f"请重试该请求（x-singleflight: leader 不受影响）"
                            )
                        )
                    except Exception:  # noqa: BLE001 - 摘除失败不影响其余跟随者
                        pass
                    continue
                try:
                    q.put_nowait(item)
                except Exception:  # noqa: BLE001
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
        constraints: Optional[Dict[str, Any]] = None,
        session_key: Optional[str] = None,
        upstream_conv_id: Optional[str] = None,
    ) -> str:
        """计算请求特征指纹 (SHA-256)。

        指纹的正确性判据只有一条：**指纹相同 ⇒ 两条请求的正确答案可以互换**。
        Follower 直接领用 Leader 的答案，所以任何会改变输出的客户端声明都必须
        在指纹里。

        [S3] `constraints` 承载的正是这一类声明（`tool_choice` / `stop` /
        `response_format` / `parallel_tool_calls` …）。它们原先不在指纹里，
        而当时这是**无害**的 —— 因为那时它们被解析之后就没人读，确实不改变
        输出。随后每个字段被逐个修好（允许名单、网关侧截断、进提示词、裁剪
        调用数量），于是它们**开始**改变输出，而指纹没有同步更新：声明
        `tool_choice:"none"` 的请求会领到工具调用，声明 `stop` 的请求会领到
        未截断的文本，全都没有任何错误可看。

        加字段只会让键**更**具体（合并变少），绝不会把两条不该合并的请求并到
        一起 —— 这是唯一安全的方向。`None` 与 `{}` 等价，所以未传约束的调用点
        逐值保持原语义。

        [SINGLEFLIGHT] `session_key` / `upstream_conv_id` 是同一条判据的又一次
        应用，缺陷形状与 [S3] 完全相同：**会话身份决定哪一段历史算作「前文」**。
        已绑定上游会话的请求走续聊分支、只发本轮增量；全新会话的请求展平全量
        历史。两者的 `messages` 可以逐字相同（agent 派生的子 agent 与父 agent
        发同样的指令、声明同样的工具），而正确答案完全不同。实测：会话 A 已绑定、
        会话 B 全新、messages 逐字相同 → 指纹相同 → B 领到 A 的续聊答案，上游
        只被调用一次，B 的会话从未创建，而它的响应头却写着 `bound`。
        一个没有报错的错答案。

        没有会话身份的请求（匿名一次性调用）两者都是 `None`，逐值不变——它们
        之间确实可以互换，仍然合并。
        """
        payload = {
            "protocol": protocol,
            "provider": provider,
            "model": model,
            "messages": messages,
            "tools": tools or [],
            "temperature": round(temperature or 0.7, 2),
            "thinking": bool(thinking),
            "stream": stream,
            # 空字典与缺省必须产生同一个 payload；`sort_keys` 负责顺序无关
            "constraints": constraints or {},
            # 会话身份：决定前文在上游还是在本轮提示词里，因此改变正确答案。
            # 匿名请求两者都是 None，与修复前逐值等价。
            "session_key": session_key or "",
            "upstream_conv_id": upstream_conv_id or "",
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
        # [M4] 模块级背压计数同属「管理器状态」，必须一并重置，否则用例之间串扰。
        _DETACHED_STATS["followers_detached"] = 0

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
            # [M4] 背压：不设上限时，一个不消费的跟随者会被无声地喂到内存耗尽。
            # 被摘除的跟随者会收到明确错误（而不是被静默截断），计数在这里可见。
            "followers_detached": _DETACHED_STATS["followers_detached"],
            "follower_backlog_limit": follower_backlog_limit(),
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
                # [M4] 这里会收到两类错误：Leader 的真实失败（上游异常），以及
                # 「本路消费过慢、已被摘除」的背压通知。两者都必须抛给客户端——
                # 静默结束会让客户端把半截回答当成完整回答，是比报错更糟的结局。
                raise item
            yield item
    finally:
        await group.remove_subscriber(queue)
