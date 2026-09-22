"""长上下文 Prompt Cache 预计算与复用管理器 (prompt_cache.py)。

实现:
1. PromptCacheManager 单例管理器，具备线程安全与异步安全并发保障；
2. 两级 LRU / TTL 缓存架构 (L1: 完全 Prompt 精确匹配, L2: 前缀 Prefix 匹配与会话复用)；
3. 基于 compute_prompt_hash(system_prompt, messages_prefix) 的确定性 SHA-256 签名计算；
4. 会话复用 (conversation_id / session_id)：复用已创建的网页端会话，跳过超长 System Prompt 的冷启动耗时；
5. Anthropic Prompt Caching 规范兼容：
   - cache_control 标记识别 ({"type": "ephemeral"})；
   - cache_creation_input_tokens 与 cache_read_input_tokens 精准计算与透传；
6. 统计与度量指标：total_requests, cache_hits, cache_misses, saved_tokens, hit_ratio。
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass, field
import hashlib
import json
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from .base import _est_tokens

DEFAULT_MAX_CAPACITY = int(os.environ.get("PROMPT_CACHE_MAX_ENTRIES", "500"))
DEFAULT_TTL = float(os.environ.get("PROMPT_CACHE_TTL", "1800"))  # 默认 30 分钟 (1800秒)


def _extract_text_robust(content: Any) -> str:
    """提取消息 content 中的纯文本（支持 str, list of blocks, None 等）。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict):
                b_type = p.get("type")
                if b_type == "text":
                    parts.append(p.get("text", ""))
                elif b_type == "tool_result":
                    sub = p.get("content", "")
                    if isinstance(sub, list):
                        sub = _extract_text_robust(sub)
                    parts.append(str(sub))
                elif "text" in p:
                    parts.append(str(p["text"]))
                elif "content" in p:
                    parts.append(_extract_text_robust(p["content"]))
        return "\n".join(parts)
    return str(content)


def compute_prompt_hash(
    system_prompt: Union[str, List[Any], None] = None,
    messages_prefix: Union[List[Dict[str, Any]], Dict[str, Any], str, None] = None,
) -> str:
    """对系统提示词与消息前缀生成稳定的 SHA-256 签名。

    无论输入为纯字符串、Anthropic content blocks 还是消息列表，均提取规范文本后计算指纹。
    """
    hasher = hashlib.sha256()

    # 1. 规范化注入 system prompt
    sys_str = _extract_text_robust(system_prompt).strip()
    hasher.update(sys_str.encode("utf-8"))
    hasher.update(b"\x00")

    # 2. 规范化遍历 messages_prefix
    norm_messages: List[Dict[str, Any]] = []
    if messages_prefix is not None:
        if isinstance(messages_prefix, list):
            for item in messages_prefix:
                if isinstance(item, dict):
                    norm_messages.append(item)
                elif isinstance(item, str):
                    norm_messages.append({"role": "user", "content": item})
        elif isinstance(messages_prefix, dict):
            norm_messages.append(messages_prefix)
        elif isinstance(messages_prefix, str):
            norm_messages.append({"role": "user", "content": messages_prefix})

    for msg in norm_messages:
        role = str(msg.get("role", "")).strip()
        content = msg.get("content")
        txt = _extract_text_robust(content)
        hasher.update(role.encode("utf-8"))
        hasher.update(b":")
        hasher.update(txt.encode("utf-8"))
        if "tool_calls" in msg and msg["tool_calls"]:
            hasher.update(
                json.dumps(msg["tool_calls"], sort_keys=True, ensure_ascii=False).encode("utf-8")
            )
        hasher.update(b"\x01")

    return hasher.hexdigest()


@dataclass
class CacheEntry:
    """Prompt 缓存条目模型。"""

    prompt_hash: str
    system_prompt: str
    session_id: Optional[str] = None
    conversation_id: Optional[str] = None
    provider_key: str = ""
    wire_model: str = ""
    estimated_tokens: int = 0
    created_at: float = field(default_factory=time.time)
    last_accessed_at: float = field(default_factory=time.time)
    hit_count: int = 0
    cache_control: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.conversation_id and not self.session_id:
            self.session_id = self.conversation_id
        elif self.session_id and not self.conversation_id:
            self.conversation_id = self.session_id

    def is_expired(self, ttl: float, now: Optional[float] = None) -> bool:
        """检查条目是否已超过 TTL 过期。"""
        if ttl <= 0:
            return False
        curr = now if now is not None else time.time()
        return (curr - self.last_accessed_at) > ttl

    def to_dict(self) -> Dict[str, Any]:
        """序列化输出供诊断与遥测使用。"""
        return {
            "prompt_hash": self.prompt_hash,
            "system_prompt": (
                self.system_prompt[:100] + "..."
                if len(self.system_prompt) > 100
                else self.system_prompt
            ),
            "session_id": self.session_id,
            "conversation_id": self.conversation_id,
            "provider_key": self.provider_key,
            "wire_model": self.wire_model,
            "estimated_tokens": self.estimated_tokens,
            "created_at": self.created_at,
            "last_accessed_at": self.last_accessed_at,
            "hit_count": self.hit_count,
            "metadata": self.metadata,
        }


def has_cache_control(data: Any) -> bool:
    """递归检查请求对象中是否显式声明了 Anthropic cache_control 标记。"""
    if data is None:
        return False

    if isinstance(data, dict):
        if "cache_control" in data and data["cache_control"]:
            return True
        for v in data.values():
            if has_cache_control(v):
                return True
        return False

    if isinstance(data, list):
        for item in data:
            if has_cache_control(item):
                return True
        return False

    if hasattr(data, "cache_control") and getattr(data, "cache_control", None):
        return True

    # 检查 Pydantic 模型属性
    if hasattr(data, "system"):
        if has_cache_control(getattr(data, "system")):
            return True
    if hasattr(data, "messages"):
        if has_cache_control(getattr(data, "messages")):
            return True
    if hasattr(data, "tools"):
        if has_cache_control(getattr(data, "tools")):
            return True

    return False


class PromptCacheManager:
    """长上下文 Prompt Cache 预计算与复用单例管理器。

    特性:
    - 线程安全 / 协程安全 (threading.RLock + asyncio.Lock 混合防御)
    - 两级 LRU / TTL 缓存架构:
      * L1: Exact Prompt Match (完整 Prompt 精确命中)
      * L2: Prefix Prefix Match (前缀 Prompt 匹配与会话 conversation_id 复用)
    - Anthropic Prompt Caching 规范支持 (cache_creation_input_tokens / cache_read_input_tokens)
    - 运行状态诊断指标输出 (total_requests, cache_hits, cache_misses, saved_tokens, hit_ratio)
    """

    _instance: Optional[PromptCacheManager] = None
    _singleton_lock: threading.Lock = threading.Lock()

    def __init__(
        self,
        max_capacity: int = DEFAULT_MAX_CAPACITY,
        ttl: float = DEFAULT_TTL,
    ):
        self.max_capacity = max(1, max_capacity)
        self.ttl = max(0.0, ttl)

        # L1: 完整 Prompt 缓存 (key -> CacheEntry)
        self._l1_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        # L2: 前缀 Prefix 缓存 (prefix_key -> CacheEntry)
        self._l2_prefix_cache: OrderedDict[str, CacheEntry] = OrderedDict()

        # 线程与协程安全锁
        self._lock = threading.RLock()
        self._async_lock: Optional[asyncio.Lock] = None

        # 度量统计
        self.total_requests: int = 0
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        self.saved_tokens: int = 0
        self.eviction_count: int = 0

    @classmethod
    def get_instance(cls) -> PromptCacheManager:
        """获取全局单例管理器。"""
        if cls._instance is None:
            with cls._singleton_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @property
    def hit_ratio(self) -> float:
        """计算缓存命中率 (0.0 ~ 1.0)。"""
        with self._lock:
            if self.total_requests <= 0:
                return 0.0
            return round(self.cache_hits / self.total_requests, 4)

    def _get_async_lock(self) -> asyncio.Lock:
        """延迟初始化 asyncio.Lock 避免跨 EventLoop 污染。"""
        if self._async_lock is None:
            self._async_lock = asyncio.Lock()
        return self._async_lock

    def _make_key(self, prompt_hash: str, provider_key: str = "", wire_model: str = "") -> str:
        """构造具备 Provider/Model 隔离的复合缓存 Key。"""
        p = provider_key.strip().lower() if provider_key else "*"
        m = wire_model.strip().lower() if wire_model else "*"
        return f"{p}:{m}:{prompt_hash}"

    def purge_expired(self, now: Optional[float] = None) -> int:
        """主动扫描并清理过期条目，返回清理条目数。"""
        curr = now if now is not None else time.time()
        purged = 0

        with self._lock:
            # 清理 L1
            expired_l1 = [k for k, v in self._l1_cache.items() if v.is_expired(self.ttl, curr)]
            for k in expired_l1:
                del self._l1_cache[k]
                purged += 1

            # 清理 L2
            expired_l2 = [k for k, v in self._l2_prefix_cache.items() if v.is_expired(self.ttl, curr)]
            for k in expired_l2:
                del self._l2_prefix_cache[k]
                purged += 1

            if purged > 0:
                self.eviction_count += purged

        return purged

    def get(
        self,
        prompt_hash: str,
        provider_key: str = "",
        wire_model: str = "",
        record_stats: bool = True,
        now: Optional[float] = None,
    ) -> Optional[CacheEntry]:
        """精确根据 prompt_hash 获取缓存条目 (L1 查找)。"""
        key = self._make_key(prompt_hash, provider_key, wire_model)
        wildcard_key = self._make_key(prompt_hash, "", "")
        curr = now if now is not None else time.time()

        with self._lock:
            if record_stats:
                self.total_requests += 1

            entry = self._l1_cache.get(key) or self._l1_cache.get(wildcard_key)

            if entry is not None:
                if entry.is_expired(self.ttl, curr):
                    # 过期移除
                    self._l1_cache.pop(key, None)
                    self._l1_cache.pop(wildcard_key, None)
                    self.eviction_count += 1
                    if record_stats:
                        self.cache_misses += 1
                    return None

                # 命中: 更新 LRU 顺序与命中计数
                entry.last_accessed_at = curr
                entry.hit_count += 1
                self._l1_cache.move_to_end(key if key in self._l1_cache else wildcard_key)

                if record_stats:
                    self.cache_hits += 1
                    self.saved_tokens += entry.estimated_tokens
                return entry

            if record_stats:
                self.cache_misses += 1
            return None

    def match_prefix(
        self,
        system_prompt: Union[str, List[Any], None],
        messages: List[Dict[str, Any]],
        provider_key: str = "",
        wire_model: str = "",
        now: Optional[float] = None,
        record_stats: bool = True,
    ) -> Optional[CacheEntry]:
        """执行两级缓存匹配：先查 L1 精确命中，未命中则查 L2 前缀匹配以复用 conversation_id。"""
        curr = now if now is not None else time.time()

        with self._lock:
            if record_stats:
                self.total_requests += 1

            # 1. 阶段一：尝试 L1 完整匹配
            full_hash = compute_prompt_hash(system_prompt, messages)
            full_key = self._make_key(full_hash, provider_key, wire_model)
            wild_full_key = self._make_key(full_hash, "", "")

            entry = self._l1_cache.get(full_key) or self._l1_cache.get(wild_full_key)
            if entry is not None and not entry.is_expired(self.ttl, curr):
                entry.last_accessed_at = curr
                entry.hit_count += 1
                self._l1_cache.move_to_end(full_key if full_key in self._l1_cache else wild_full_key)
                if record_stats:
                    self.cache_hits += 1
                    self.saved_tokens += entry.estimated_tokens
                return entry

            # 2. 阶段二：尝试 L2 前缀匹配 (System Prompt + 首条用户指令)
            if messages:
                # 截取前缀：第一条 user 消息
                prefix_msgs = messages[:1]
                prefix_hash = compute_prompt_hash(system_prompt, prefix_msgs)
                prefix_key = self._make_key(prefix_hash, provider_key, wire_model)
                wild_prefix_key = self._make_key(prefix_hash, "", "")

                prefix_entry = self._l2_prefix_cache.get(prefix_key) or self._l2_prefix_cache.get(wild_prefix_key)
                if prefix_entry is not None and not prefix_entry.is_expired(self.ttl, curr):
                    prefix_entry.last_accessed_at = curr
                    prefix_entry.hit_count += 1
                    self._l2_prefix_cache.move_to_end(
                        prefix_key if prefix_key in self._l2_prefix_cache else wild_prefix_key
                    )
                    if record_stats:
                        self.cache_hits += 1
                        self.saved_tokens += prefix_entry.estimated_tokens
                    return prefix_entry

            if record_stats:
                self.cache_misses += 1
            return None

    def set(
        self,
        system_prompt: Union[str, List[Any], None],
        messages: Union[List[Dict[str, Any]], Dict[str, Any], None],
        conversation_id: str,
        provider_key: str = "",
        wire_model: str = "",
        estimated_tokens: Optional[int] = None,
        cache_control: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        now: Optional[float] = None,
    ) -> CacheEntry:
        """向 L1 与 L2 写入或更新缓存条目，并执行 LRU 容量驱逐。"""
        curr = now if now is not None else time.time()
        norm_messages: List[Dict[str, Any]] = []
        if messages is not None:
            if isinstance(messages, list):
                norm_messages = [m for m in messages if isinstance(m, dict)]
            elif isinstance(messages, dict):
                norm_messages = [messages]

        sys_txt = _extract_text_robust(system_prompt)
        prompt_hash = compute_prompt_hash(sys_txt, norm_messages)

        if estimated_tokens is None or estimated_tokens <= 0:
            content_concat = sys_txt + "\n" + "\n".join(_extract_text_robust(m.get("content")) for m in norm_messages)
            estimated_tokens = max(1, _est_tokens(content_concat))

        entry = CacheEntry(
            prompt_hash=prompt_hash,
            system_prompt=sys_txt,
            session_id=conversation_id,
            conversation_id=conversation_id,
            provider_key=provider_key,
            wire_model=wire_model,
            estimated_tokens=estimated_tokens,
            created_at=curr,
            last_accessed_at=curr,
            hit_count=0,
            cache_control=cache_control,
            metadata=metadata or {},
        )

        with self._lock:
            # 1. 写入 L1 (完整匹配)
            l1_key = self._make_key(prompt_hash, provider_key, wire_model)
            self._l1_cache[l1_key] = entry
            self._l1_cache.move_to_end(l1_key)

            # L1 LRU 淘汰
            while len(self._l1_cache) > self.max_capacity:
                self._l1_cache.popitem(last=False)
                self.eviction_count += 1

            # 2. 写入 L2 (前缀匹配)
            if norm_messages:
                prefix_hash = compute_prompt_hash(sys_txt, norm_messages[:1])
                l2_key = self._make_key(prefix_hash, provider_key, wire_model)
                self._l2_prefix_cache[l2_key] = entry
                self._l2_prefix_cache.move_to_end(l2_key)

                # L2 LRU 淘汰
                while len(self._l2_prefix_cache) > self.max_capacity:
                    self._l2_prefix_cache.popitem(last=False)
                    self.eviction_count += 1

        return entry

    def is_cached(self, conversation_id: str, now: Optional[float] = None) -> bool:
        """检查指定的 conversation_id 是否正被缓存池活跃引用（防止提前被 upstream 删除）。"""
        if not conversation_id:
            return False
        curr = now if now is not None else time.time()
        cid = str(conversation_id).strip()

        with self._lock:
            for entry in self._l1_cache.values():
                if entry.conversation_id == cid and not entry.is_expired(self.ttl, curr):
                    return True
            for entry in self._l2_prefix_cache.values():
                if entry.conversation_id == cid and not entry.is_expired(self.ttl, curr):
                    return True
        return False

    def calculate_anthropic_usage(
        self,
        req: Any,
        is_hit: bool,
        cached_entry: Optional[CacheEntry] = None,
        total_input_tokens: Optional[int] = None,
    ) -> Tuple[int, int, int]:
        """计算 Anthropic 规范下的 Token 用量分项指标。

        返回:
            (input_tokens, cache_creation_input_tokens, cache_read_input_tokens)
        """
        tot = total_input_tokens or 0
        has_cc = has_cache_control(req)

        # 场景 1: 缓存命中
        if is_hit and cached_entry is not None:
            read_tokens = min(cached_entry.estimated_tokens, tot) if tot > 0 else cached_entry.estimated_tokens
            creation_tokens = 0
            uncached_input = max(0, tot - read_tokens)
            return uncached_input, creation_tokens, read_tokens

        # 场景 2: 缓存未命中，但请求带 cache_control 标记 (写入缓存)
        if has_cc:
            # 估算需要被缓存的前缀 tokens（例如 system prompt + cache_control 块）
            sys_text = ""
            if hasattr(req, "system") and req.system:
                sys_text = _extract_text_robust(req.system)
            elif isinstance(req, dict) and req.get("system"):
                sys_text = _extract_text_robust(req["system"])

            creation_tokens = _est_tokens(sys_text) if sys_text else tot
            if creation_tokens <= 0:
                creation_tokens = tot
            creation_tokens = min(creation_tokens, tot) if tot > 0 else creation_tokens

            read_tokens = 0
            uncached_input = max(0, tot - creation_tokens)
            return uncached_input, creation_tokens, read_tokens

        # 场景 3: 普通未命中且无 cache_control
        return tot, 0, 0

    async def get_async(
        self,
        prompt_hash: str,
        provider_key: str = "",
        wire_model: str = "",
        record_stats: bool = True,
        now: Optional[float] = None,
    ) -> Optional[CacheEntry]:
        """协程安全获取缓存。"""
        async_lock = self._get_async_lock()
        async with async_lock:
            return self.get(
                prompt_hash,
                provider_key=provider_key,
                wire_model=wire_model,
                record_stats=record_stats,
                now=now,
            )

    async def match_prefix_async(
        self,
        system_prompt: Union[str, List[Any], None],
        messages: List[Dict[str, Any]],
        provider_key: str = "",
        wire_model: str = "",
        now: Optional[float] = None,
        record_stats: bool = True,
    ) -> Optional[CacheEntry]:
        """协程安全前缀匹配。"""
        async_lock = self._get_async_lock()
        async with async_lock:
            return self.match_prefix(
                system_prompt=system_prompt,
                messages=messages,
                provider_key=provider_key,
                wire_model=wire_model,
                now=now,
                record_stats=record_stats,
            )

    async def set_async(
        self,
        system_prompt: Union[str, List[Any], None],
        messages: Union[List[Dict[str, Any]], Dict[str, Any], None],
        conversation_id: str,
        provider_key: str = "",
        wire_model: str = "",
        estimated_tokens: Optional[int] = None,
        cache_control: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        now: Optional[float] = None,
    ) -> CacheEntry:
        """协程安全写入缓存。"""
        async_lock = self._get_async_lock()
        async with async_lock:
            return self.set(
                system_prompt=system_prompt,
                messages=messages,
                conversation_id=conversation_id,
                provider_key=provider_key,
                wire_model=wire_model,
                estimated_tokens=estimated_tokens,
                cache_control=cache_control,
                metadata=metadata,
                now=now,
            )

    def get_stats(self) -> Dict[str, Any]:
        """获取当前缓存管理器指标字典。"""
        with self._lock:
            return {
                "total_requests": self.total_requests,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "saved_tokens": self.saved_tokens,
                "hit_ratio": self.hit_ratio,
                "cached_entries": len(self._l1_cache),
                "prefix_entries": len(self._l2_prefix_cache),
                "max_capacity": self.max_capacity,
                "ttl": self.ttl,
                "eviction_count": self.eviction_count,
            }

    def clear(self) -> None:
        """清空缓存与度量计数器。"""
        with self._lock:
            self._l1_cache.clear()
            self._l2_prefix_cache.clear()
            self.total_requests = 0
            self.cache_hits = 0
            self.cache_misses = 0
            self.saved_tokens = 0
            self.eviction_count = 0


# 全局默认单例实例
prompt_cache_manager = PromptCacheManager.get_instance()


def get_prompt_cache_manager() -> PromptCacheManager:
    """获取全局单例管理器。"""
    return PromptCacheManager.get_instance()
