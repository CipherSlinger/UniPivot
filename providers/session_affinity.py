"""智能会话粘滞路由与跨模型状态迁移引擎 (Smart Session Affinity & State Handoff Engine)

实现 [P0] OPT-007：
1. 会话粘滞注册表 (Session Affinity Registry):
   - 基于 LRU 与 TTL (默认 30 分钟) 追踪多轮交互的会话归属。
   - 保持后续轮次请求优先分配到已建立对话上下文的后端 Provider 及模型。
2. 跨模型透明状态迁移 (Cross-Model State Handoff):
   - 当原 Provider 触发风控拦截、节点熔断或临时冷却时，检测到粘滞断裂；
   - 自动清洗剥离上游异构 conversation_id（避免将 A 厂私有 ID 传给 B 厂引发报错）；
   - 基于网关全量上下文历史平滑迁移至同环/降级候选节点，维持多轮交互不中断。
3. 遥测与响应头透传:
   - x-session-affinity: bound | handoff | fresh
   - x-session-handoff-from: <from_provider> (仅在发生跨模型迁移时产生)
   - x-session-turns: <turn_count>

[TOOL-LOOP] 客户端的 `conversation_id` 是**线程标识**，不是上游会话标识。
两者原先被当成同一个值：注册表里没有这条线程时，客户端传来的 id 被原样当成
`upstream_conv_id` 交给 provider，而 provider 只要看到非空 `conversation_id`
就走「续聊」分支——只发送最后一条 user 之后的增量内容。首轮请求的增量是空的，
于是上游收到的提示词**只有最后一句用户输入**：注入的系统提示、`## Tools Available`
工具契约、`response_format`、整段历史全部消失。实测同一请求在续聊分支下的提示词是
`'read foo.py'`，在展平分支下是 990 字符。这就是「模型完全不理我给的 tools」的
直接成因，而且只发生在「客户端自己生成了线程 id」的客户端上——症状因此显得时有时无。

现在只有网关**实际观测到**的上游会话 id 才会被回放；未绑定的线程一律以
`upstream_conv_id=None` 开始，由 provider 新建会话。
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

logger = logging.getLogger("gateway.session_affinity")


@dataclass
class AffinityEntry:
    """单个会话的粘滞元数据。"""
    session_id: str
    provider_key: str
    model_name: str
    upstream_conv_id: Optional[str]
    created_at: float
    last_accessed_at: float
    turn_count: int = 1


@dataclass
class AffinityDecision:
    """会话粘滞路由决策。"""
    status: str  # "bound" | "degraded" (原节点短冷却，粘滞保留) | "handoff" (已跨模型迁移) | "fresh" (全新会话)
    provider_key: str
    wire_model: str
    upstream_conv_id: Optional[str]
    is_handoff: bool = False
    original_provider: Optional[str] = None
    turn_count: int = 1


class SessionAffinityManager:
    """智能会话粘滞管理器。"""

    def __init__(self, ttl_seconds: float = 1800.0, max_entries: int = 10000):
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._lock = threading.Lock()
        self._registry: Dict[str, AffinityEntry] = {}
        # 度量计数器
        self._stats = {
            "total_recorded_sessions": 0,
            "affinity_hits": 0,
            "affinity_degraded": 0,
            "affinity_handoffs": 0,
        }

    def record_turn(
        self,
        session_id: Optional[str],
        provider_key: str,
        model_name: str,
        upstream_conv_id: Optional[str] = None,
    ) -> None:
        """记录或更新会话交互轮次与粘滞目标。"""
        if not session_id:
            return

        now = time.time()
        with self._lock:
            self._clean_expired_locked(now)

            if session_id in self._registry:
                entry = self._registry[session_id]
                entry.provider_key = provider_key
                entry.model_name = model_name
                if upstream_conv_id:
                    entry.upstream_conv_id = upstream_conv_id
                entry.last_accessed_at = now
                entry.turn_count += 1
            else:
                if len(self._registry) >= self.max_entries:
                    # 淘汰最久未访问的条目
                    oldest_key = min(self._registry, key=lambda k: self._registry[k].last_accessed_at)
                    self._registry.pop(oldest_key, None)

                self._registry[session_id] = AffinityEntry(
                    session_id=session_id,
                    provider_key=provider_key,
                    model_name=model_name,
                    # 绝不拿客户端线程 id 冒充上游会话 id：未观测到就保持 None，
                    # 让 provider 走「新建会话 + 展平全量历史」分支（见模块 docstring）。
                    upstream_conv_id=upstream_conv_id,
                    created_at=now,
                    last_accessed_at=now,
                    turn_count=1,
                )
                self._stats["total_recorded_sessions"] += 1

    def resolve_affinity(
        self,
        session_id: Optional[str],
        primary_provider: str,
        primary_model: str,
        is_provider_available: Callable[[str], bool],
        can_retain_degraded: Optional[Callable[[str], bool]] = None,
    ) -> AffinityDecision:
        """解析当前会话粘滞关系。

        若存在已绑定的可用 Provider，则锁定该 Provider；
        若原 Provider 处于硬阻断状态，则触发跨模型状态迁移并剥离旧 ID。

        [M3] 「可用」与「值得为它放弃整段上下文」是两个判据，此前被混成一个。
        `is_provider_available` 对**任何**软冷却（哪怕 15 秒的 429 退避）都返回
        False，于是每有一次频控退避，进行中的长程会话就被永久改嫁到另一家厂商：
        上游上下文全部重建、提示词缓存全丢、工具调用契约重新学习——为躲开一个
        数十秒的窗口付出几十倍的代价。这里改为**只判定粘滞是否仍保留得下**：
        硬阻断（租约 / 禁用 / 长效避让）不可保留，短冷却可以保留（见
        `FailoverRouter.is_affinity_retainable`）。调用方传入的仍是同一个谓词，
        由服务端注入更严格的语义，判定口径因此只有一份。
        """
        if not session_id:
            return AffinityDecision(
                status="fresh",
                provider_key=primary_provider,
                wire_model=primary_model,
                upstream_conv_id=None,
                is_handoff=False,
                turn_count=1,
            )

        now = time.time()
        with self._lock:
            self._clean_expired_locked(now)
            entry = self._registry.get(session_id)

            if not entry:
                # 未绑定线程：客户端给的是线程标识，网关从未为它建立过上游会话，
                # 因此**没有**可回放的上游 id。把它当上游 id 传下去会让 provider
                # 走续聊分支，只发最后一条 user 之后的增量——首轮即只剩一句提问，
                # 工具契约与历史整段丢失（[TOOL-LOOP] 见模块 docstring）。
                return AffinityDecision(
                    status="fresh",
                    provider_key=primary_provider,
                    wire_model=primary_model,
                    upstream_conv_id=None,
                    is_handoff=False,
                    turn_count=1,
                )

            entry.last_accessed_at = now

            # 原绑定节点不可用：先区分「一时退避」与「不值得保留」
            if is_provider_available(entry.provider_key):
                self._stats["affinity_hits"] += 1
                return AffinityDecision(
                    status="bound",
                    provider_key=entry.provider_key,
                    wire_model=entry.model_name or primary_model,
                    upstream_conv_id=entry.upstream_conv_id,
                    is_handoff=False,
                    turn_count=entry.turn_count,
                )

            degraded_ok = can_retain_degraded or (lambda _p: False)
            # 原绑定节点一时退避（短冷却），**保留粘滞关系**：不迁移、不清洗上游
            # 会话 id。本次请求仍以常规候选为主选（调用方传入的 primary），由派发
            # 边界按 OPT-021 的「短冷却只降级」语义处置；冷却一结束，同一会话自动
            # 回到原节点、原上游会话，无需重建上下文。
            if degraded_ok(entry.provider_key):
                self._stats["affinity_degraded"] += 1
                return AffinityDecision(
                    status="degraded",
                    provider_key=entry.provider_key,
                    wire_model=entry.model_name or primary_model,
                    upstream_conv_id=entry.upstream_conv_id,
                    is_handoff=False,
                    turn_count=entry.turn_count,
                )

            # 原绑定节点硬阻断（租约 / 禁用 / 长效避让），执行平滑状态迁移
            self._stats["affinity_handoffs"] += 1
            orig_provider = entry.provider_key
            logger.info(
                "会话 [%s] 发生跨模型状态迁移: 原节点 [%s] 不可用 -> 迁移至候选目标 [%s]",
                session_id, orig_provider, primary_provider,
            )
            # 迁移时将绑定的 Provider 更新为新的 Provider，并剥离旧的异构 upstream_conv_id
            entry.provider_key = primary_provider
            entry.model_name = primary_model
            entry.upstream_conv_id = None

            return AffinityDecision(
                status="handoff",
                provider_key=primary_provider,
                wire_model=primary_model,
                upstream_conv_id=None,
                is_handoff=True,
                original_provider=orig_provider,
                turn_count=entry.turn_count,
            )

    def has_upstream_session(self, session_id: Optional[str]) -> bool:
        """该线程是否已经持有**网关观测到的**上游会话 id。

        调用方据此在「把全量历史展平发出去」与「只发本轮增量」之间选择——这是
        [TOOL-LOOP] S1 的收口点：客户端自带的线程 id 永远不算数，只有 provider
        真的返回过 `conversation_id` 并且已 `record_turn` 的线程才算。
        只读，不改动任何状态。
        """
        if not session_id:
            return False
        now = time.time()
        with self._lock:
            self._clean_expired_locked(now)
            entry = self._registry.get(session_id)
            return bool(entry and entry.upstream_conv_id)

    def _clean_expired_locked(self, now: float) -> None:
        """清理已超时的会话缓存（内部私有方法，需持有锁）。"""
        expired_keys = [
            k for k, v in self._registry.items()
            if (now - v.last_accessed_at) > self.ttl_seconds
        ]
        for k in expired_keys:
            self._registry.pop(k, None)

    def clear(self) -> None:
        """重置会话粘滞表。"""
        with self._lock:
            self._registry.clear()
            self._stats = {
                "total_recorded_sessions": 0,
                "affinity_hits": 0,
                "affinity_degraded": 0,
                "affinity_handoffs": 0,
            }

    def get_diagnostics(self) -> dict:
        """获取会话粘滞与跨模型状态迁移统计指标。"""
        now = time.time()
        with self._lock:
            self._clean_expired_locked(now)
            active_count = len(self._registry)
            return {
                "active_sessions": active_count,
                "total_recorded_sessions": self._stats["total_recorded_sessions"],
                "affinity_hits": self._stats["affinity_hits"],
                # [M3] 「降级但保留」是本轮新增的第三种结局，必须可观测：
                # 它如果长期为 0 而 handoffs 很高，说明判据没有真正生效。
                "affinity_degraded": self._stats["affinity_degraded"],
                "affinity_handoffs": self._stats["affinity_handoffs"],
                "ttl_seconds": self.ttl_seconds,
            }


# 全局单例
session_affinity_manager = SessionAffinityManager()
