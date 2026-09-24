"""智能会话粘滞路由与跨模型状态迁移引擎 (Smart Session Affinity & State Handoff Engine)

实现 [P0] OPT-007：
1. 会话粘滞注册表 (Session Affinity Registry):
   - 基于 LRU 与 TTL (默认 30 分钟) 追踪多轮交互的会话归属。
   - 保持后续轮次请求优��分配到已建立对话上下文的后端 Provider 及模型。
2. 跨模型透明状态迁移 (Cross-Model State Handoff):
   - 当原 Provider 触发风控拦截、节点熔断或临时冷却时，检测到粘滞断裂；
   - 自动清洗剥离上游异构 conversation_id（避免将 A 厂私有 ID 传给 B 厂引发报错）；
   - 基于网关全量上下文历史平滑迁移至同环/降级候选节点，维持多轮交互不中断。
3. 遥测与响应头透传:
   - x-session-affinity: bound | handoff | fresh
   - x-session-handoff-from: <from_provider> (仅在发生跨模型迁移时产生)
   - x-session-turns: <turn_count>
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
    status: str  # "bound" (已粘滞) | "handoff" (已跨模型迁移) | "fresh" (全新会话)
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
                    upstream_conv_id=upstream_conv_id or session_id,
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
    ) -> AffinityDecision:
        """解析当前会话粘滞关系。

        若存在已绑定的健康 Provider，则锁定该 Provider；
        若原 Provider 离线或不可用，则触发跨模型状态迁移并剥离旧 ID。
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
                return AffinityDecision(
                    status="fresh",
                    provider_key=primary_provider,
                    wire_model=primary_model,
                    upstream_conv_id=session_id,
                    is_handoff=False,
                    turn_count=1,
                )

            entry.last_accessed_at = now

            # 检查原绑定的 Provider 是否健康可用
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

            # 原绑定节点不可用，执行平滑状态迁移
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
                "affinity_handoffs": self._stats["affinity_handoffs"],
                "ttl_seconds": self.ttl_seconds,
            }


# 全局单例
session_affinity_manager = SessionAffinityManager()
