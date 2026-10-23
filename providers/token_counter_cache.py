"""确定性 Token 计数记忆化缓存 (Deterministic Token Count Memoization Cache)

实现 [P0] OPT-009 的配额保护配套能力：
1. 确定性记忆化 (Deterministic Memoization):
   - `/v1/messages/count_tokens` 是纯函数：相同 messages/system/tools 必然产出相同 token 数。
   - Claude Code CLI 在每次请求前高频调用该端点，而 flatten_messages + _est_tokens
     对超长会话是 O(n) 字符串运算，重复计算纯属浪费。
2. 有界 LRU (Bounded LRU):
   - 基于 OrderedDict 实现 LRU 淘汰，容量由 GATEWAY_TOKEN_CACHE_MAX_ENTRIES 控制（默认 2048），
     杜绝无界内存增长。
3. 遥测指标:
   - 暴露命中/未命中/淘汰计数与命中率，供 GET /v1/diagnostics 审计。
   - GATEWAY_DISABLE_TOKEN_CACHE=1 可整体关闭该优化。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from collections import OrderedDict
from typing import Any, Optional

logger = logging.getLogger("gateway.token_counter_cache")


class TokenCounterCache:
    """确定性 token 计数的有界 LRU 记忆化缓存。"""

    def __init__(self, max_entries: Optional[int] = None) -> None:
        if max_entries is None:
            max_entries = int(os.getenv("GATEWAY_TOKEN_CACHE_MAX_ENTRIES", "2048"))
        self.max_entries = max(1, max_entries)
        self._lock = threading.Lock()
        self._store: "OrderedDict[str, int]" = OrderedDict()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
        }

    def is_disabled(self) -> bool:
        """判断是否禁用 token 计数缓存。"""
        return os.getenv("GATEWAY_DISABLE_TOKEN_CACHE", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )

    def get(self, key: str) -> Optional[int]:
        """读取缓存；命中时将该键移至 LRU 末尾。"""
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                self._stats["hits"] += 1
                return self._store[key]
            self._stats["misses"] += 1
            return None

    def put(self, key: str, value: int) -> None:
        """写入缓存并执行 LRU 淘汰。"""
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = value
            while len(self._store) > self.max_entries:
                self._store.popitem(last=False)
                self._stats["evictions"] += 1

    def get_diagnostics(self) -> dict:
        """获取缓存命中率与容量指标。"""
        with self._lock:
            total = self._stats["hits"] + self._stats["misses"]
            hit_rate = round(self._stats["hits"] / total, 4) if total > 0 else 0.0
            return {
                "enabled": not self.is_disabled(),
                "entries": len(self._store),
                "max_entries": self.max_entries,
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "evictions": self._stats["evictions"],
                "hit_rate": hit_rate,
            }

    def clear(self) -> None:
        """重置缓存与计数器（测试用）。"""
        with self._lock:
            self._store.clear()
            self._stats = {"hits": 0, "misses": 0, "evictions": 0}


# 全局单例
token_counter_cache = TokenCounterCache()


def _normalize(value: Any) -> Any:
    """将 Pydantic 模型 / 嵌套结构归一化为可稳定序列化的纯数据结构。"""
    if value is None:
        return None
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return dump(mode="json")
        except TypeError:
            return dump()
    if isinstance(value, (list, tuple)):
        return [_normalize(v) for v in value]
    if isinstance(value, dict):
        return {k: _normalize(v) for k, v in value.items()}
    return value


def compute_count_key(req: Any) -> str:
    """为 count_tokens 请求计算确定性 SHA-256 指纹。

    仅取参与 token 计算的语义字段，并以 sort_keys 规范化，保证同义请求命中同一键。

    [PROTOCOL] `tool_choice` 也必须进指纹：它会被 `format_tools_system_prompt`
    翻译成工具提示词里的一句约束指令（`tool_choice:"none"` 追加「本回合不要调用
    任何工具」），因此两个 `tool_choice` 不同的请求，送给模型的提示词并不相同。
    漏掉它就会把后到请求的估计直接返回成先到请求的数字——实测（清空缓存后逐个
    测量）`none` = 314、`auto` = 290，先 `none` 后 `auto` 连发时两次都是 314。
    与 SingleFlight `compute_key` 同一条纪律：改变输出的字段必须进指纹，且加入
    字段只会减少共享，绝不会造成错误共享。

    `parallel_allowed`（由 `tool_choice.disable_parallel_tool_use` 派生）不需要单独
    入键：`parallel_calls_disallowed(None, tool_choice)` 是 `tool_choice` 的纯函数，
    上面已经覆盖。派生量另立一项只会制造「同一事实两份键」的漂移机会。
    """
    payload = {
        "messages": _normalize(getattr(req, "messages", None)),
        "system": _normalize(getattr(req, "system", None)),
        "tools": _normalize(getattr(req, "tools", None)),
        "tool_choice": _normalize(getattr(req, "tool_choice", None)),
    }
    try:
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        # 极端不可序列化场景：退化为相对稳定的字符串表示，绝不因此中断请求
        canonical = repr(payload)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
