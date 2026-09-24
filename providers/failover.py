"""五大国内模型对等互备路由网格 (providers/failover.py)。

通义千问 (Qwen)、深度求索 (DeepSeek)、豆包 (Doubao)、月之暗面 (Kimi)、智谱清言 (GLM)
处于完全对等地位，对等互备。

核心设计：
1. 三级对等能力环 (Peer Capability Rings)：
   - REASONING (思考环): deepseek-reasoner, kimi-think / kimi-k1, qwen3.7-max, doubao-think, glm-zero-preview
   - FLAGSHIP (旗舰全能环): qwen3.7-max, glm-4-plus, kimi, doubao-pro, deepseek-chat
   - SPEED (极速经济环): qwen3.6-flash, doubao-lite, glm-4-flash

2. 风控与频控元数据 (Provider Risk & Rate Limiting Metadata)：
   - 标注风控与频率敏感度 (HIGH vs LOW)
   - 自适应调用间隔与随机抖动 (Pacing & Jitter)
   - 节点并发控制信号量 (Concurrency Limiting)

3. 动态冷却避让机制 (Dynamic Cooldown Avoidance)：
   - 遭遇 429、WAF 人机验证 (RGV587 / 验证码) 或连续报错时进入冷却期 (30-60s)
   - 优先调度同能力环中的其他健康低风险节点

4. 自动故障转移调度器 (FailoverRouter)：
   - 提供 execute_chat() 包装生成器流与首包安全捕获
   - 标记 x-failover-from 与 x-failover-to
   - 全节点故障时抛出综合异常
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import logging
import os
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

from .base import ProviderError

logger = logging.getLogger("gateway.failover")


class CapabilityRing(str, Enum):
    """三级对等能力环"""
    REASONING = "reasoning"   # 深度思考、代码与复杂数学推理
    FLAGSHIP = "flagship"     # 旗舰全能、大上下文、复杂指令遵循
    SPEED = "speed"           # 极速经济、低延迟、轻量日常任务


class RiskLevel(str, Enum):
    """节点风控与频控风险评级"""
    LOW = "low"         # 频控宽松、WAF 风险低 (如 Doubao, GLM-Flash)
    MEDIUM = "medium"   # 适度频控 (如 GLM-4-Plus)
    HIGH = "high"       # 频控严格、高 WAF/人机验证风险 (如 Qwen, DeepSeek, Kimi)


@dataclass
class ProviderRiskMeta:
    """提供方风控与限流元数据"""
    provider_name: str
    risk_level: RiskLevel
    min_call_interval: float  # 最小调用间隔 (秒)
    jitter_range: Tuple[float, float]  # 随机抖动范围 (秒, 如 (0.05, 0.2))
    max_concurrency: int  # 最大突发并发上限
    default_cooldown: float = 30.0  # 触发 429/风控后的默认冷却时长 (秒)


# 各 Provider 风控元数据配置表
PROVIDER_RISK_SPECS: Dict[str, ProviderRiskMeta] = {
    "qwen": ProviderRiskMeta(
        provider_name="qwen",
        risk_level=RiskLevel.HIGH,
        min_call_interval=0.3,
        jitter_range=(0.05, 0.15),
        max_concurrency=3,
        default_cooldown=45.0,  # 规避 RGV587 WAF 滑块
    ),
    "deepseek": ProviderRiskMeta(
        provider_name="deepseek",
        risk_level=RiskLevel.HIGH,
        min_call_interval=0.4,
        jitter_range=(0.05, 0.2),
        max_concurrency=2,
        default_cooldown=60.0,  # 规避 Cloudflare / 429 严格频控
    ),
    "kimi": ProviderRiskMeta(
        provider_name="kimi",
        risk_level=RiskLevel.HIGH,
        min_call_interval=0.35,
        jitter_range=(0.05, 0.15),
        max_concurrency=2,
        default_cooldown=45.0,  # 规避 token refresh 与会话频控
    ),
    "glm": ProviderRiskMeta(
        provider_name="glm",
        risk_level=RiskLevel.LOW,
        min_call_interval=0.1,
        jitter_range=(0.01, 0.05),
        max_concurrency=5,
        default_cooldown=30.0,
    ),
    "doubao": ProviderRiskMeta(
        provider_name="doubao",
        risk_level=RiskLevel.LOW,
        min_call_interval=0.08,
        jitter_range=(0.01, 0.05),
        max_concurrency=6,
        default_cooldown=30.0,
    ),
}


# 对等能力环模型成员定义 (provider_key, wire_model)
RING_DEFINITIONS: Dict[CapabilityRing, List[Tuple[str, str]]] = {
    CapabilityRing.REASONING: [
        ("deepseek", "deepseek-reasoner"),
        ("kimi", "kimi-explore"),
        ("qwen", "Qwen3.7-Max"),
        ("doubao", "doubao-think"),
        ("glm", "glm-zero-preview"),
    ],
    CapabilityRing.FLAGSHIP: [
        ("qwen", "Qwen3.7-Max"),
        ("glm", "glm-4-plus"),
        ("kimi", "kimi"),
        ("doubao", "doubao-pro"),
        ("deepseek", "deepseek-chat"),
    ],
    CapabilityRing.SPEED: [
        ("qwen", "Qwen3.6-Flash"),
        ("doubao", "doubao-lite"),
        ("glm", "glm-4-flash"),
    ],
}


def classify_capability_ring(provider_key: str, wire_model: str) -> CapabilityRing:
    """根据 provider_key 与 wire_model 推导所属对等能力环"""
    wm_lower = (wire_model or "").lower()
    pk = (provider_key or "").lower()

    # 1. 深度思考推理特征
    if any(k in wm_lower for k in ("reasoner", "r1", "think", "zero", "k1", "explore", "research")):
        return CapabilityRing.REASONING

    # 2. 极速经济特征
    if any(k in wm_lower for k in ("flash", "lite", "fast", "speed", "mini")):
        return CapabilityRing.SPEED

    # 3. 旗舰/通用特征
    if any(k in wm_lower for k in ("max", "plus", "pro", "chat", "kimi", "glm-4")):
        return CapabilityRing.FLAGSHIP

    # 若特定模型名称
    if pk == "deepseek" and "chat" in wm_lower:
        return CapabilityRing.FLAGSHIP
    if pk == "qwen" and "3.6-flash" in wm_lower:
        return CapabilityRing.SPEED

    # 默认归属旗舰环
    return CapabilityRing.FLAGSHIP


def is_risk_interception(exc_or_text: Any) -> Tuple[bool, str]:
    """识别错误或异常文本是否属于平台级风控/WAF/滑块验证拦截/Token失效。
    返回: (is_risk: bool, reason_detail: str)
    """
    if exc_or_text is None:
        return False, ""
    msg = str(exc_or_text).lower()

    # 1. 阿里巴巴 (Tongyi Qianwen / Aliyun WAF)
    if any(k in msg for k in ("rgv587", "fail_sys_user_validate", "aliyun waf", "sec_token")):
        return True, "Alibaba WAF (RGV587 / 人机验证)"
    if "人机" in msg and any(k in msg for k in ("qwen", "千问", "ali", "rgv")):
        return True, "Alibaba WAF (RGV587 / 人机验证)"

    # 2. 字节跳动 (Doubao / Acrawler)
    if any(k in msg for k in ("豆包触发风控", "acrawler", "verify_ticket", "need_verify")):
        return True, "ByteDance Risk Engine (Acrawler / 滑块验证)"
    if "人机验证" in msg and any(k in msg for k in ("豆包", "doubao")):
        return True, "ByteDance Risk Engine (Acrawler / 滑块验证)"

    # 3. DeepSeek / Cloudflare
    if any(k in msg for k in ("authorization failed (invalid token)", "turnstile", "cf_clearance")):
        return True, "DeepSeek Risk / Cloudflare Interception"
    if "deepseek" in msg and any(k in msg for k in ("authorization failed", "403 forbidden", "cloudflare")):
        return True, "DeepSeek Risk / Cloudflare Interception"

    # 4. 月之暗面 (Kimi)
    if any(k in msg for k in ("rate_limit_exceeded", "kimi rate limit")):
        return True, "Moonshot Kimi Rate Limit"

    # 5. 智谱 (GLM)
    if any(k in msg for k in ("智谱 glm 令牌无效或已过期", "glm 令牌无效", "chatglm_token")):
        return True, "Zhipu GLM Auth Expiration"

    # 6. 通用 WAF / 滑块验证码通用匹配
    if any(k in msg for k in ("人机验证", "滑块验证", "captcha", "security check")):
        return True, "Generic WAF / Captcha Challenge"

    return False, ""


class CooldownTracker:
    """动态冷却与避让跟踪器"""

    def __init__(self):
        # 禁用提供方配置 (从环境变量 DISABLED_PROVIDERS 读取，逗号分隔，如 "deepseek,glm")
        disabled_str = os.getenv("DISABLED_PROVIDERS", "")
        self.disabled_providers: Set[str] = {
            p.strip().lower() for p in disabled_str.split(",") if p.strip()
        }

        # provider -> float (冷却结束的时间戳)
        self._cooldowns: Dict[str, float] = {}
        # provider -> 连续失败次数
        self._failure_counts: Dict[str, int] = {}
        # provider -> 上次调用时间戳
        self._last_call_time: Dict[str, float] = {}
        # provider -> asyncio.Semaphore 并发控制
        self._semaphores: Dict[str, asyncio.Semaphore] = {
            pk: asyncio.Semaphore(meta.max_concurrency)
            for pk, meta in PROVIDER_RISK_SPECS.items()
        }
        # 保护锁
        self._lock = asyncio.Lock()

    def is_disabled(self, provider_key: str) -> bool:
        """检查指定 Provider 是否被配置主动禁用"""
        return (provider_key or "").lower() in self.disabled_providers

    def is_in_cooldown(self, provider_key: str) -> bool:
        """检查指定 Provider 是否处于冷却避让期"""
        now = time.time()
        end_time = self._cooldowns.get(provider_key, 0.0)
        return now < end_time

    def get_remaining_cooldown(self, provider_key: str) -> float:
        """获取剩余冷却时间（秒），若未处于冷却期则返回 0.0"""
        now = time.time()
        end_time = self._cooldowns.get(provider_key, 0.0)
        return max(0.0, end_time - now)

    def get_all_cooldowns(self) -> Dict[str, float]:
        """获取所有当前处于冷却���态的节点及剩余冷却时间（秒）"""
        now = time.time()
        res = {}
        for pk, end_time in list(self._cooldowns.items()):
            rem = end_time - now
            if rem > 0:
                res[pk] = round(rem, 2)
        return res

    async def record_failure(
        self,
        provider_key: str,
        status: int,
        error_msg: str = "",
        custom_cooldown: Optional[float] = None,
    ) -> float:
        """记录失败并触发动态冷却避让。
        返回该节点进入的冷却时长。
        """
        async with self._lock:
            cur_fails = self._failure_counts.get(provider_key, 0) + 1
            self._failure_counts[provider_key] = cur_fails

            meta = PROVIDER_RISK_SPECS.get(provider_key)
            base_cd = custom_cooldown or (meta.default_cooldown if meta else 30.0)

            # 429 限流或人机验证 (RGV587 / 验证码) 强制冷却
            is_risk, risk_reason = is_risk_interception(error_msg)
            is_waf_or_rate = is_risk or status == 429 or any(
                k in error_msg.lower() for k in ("rgv587", "captcha", "人机", "验证码", "rate_limit", "频控")
            )
            # 401 令牌过期也需要避让
            is_auth_error = status == 401

            if is_risk or is_waf_or_rate:
                # 针对频控与风控加权冷却 (指数退避: base_cd * (1.2 ** min(cur_fails - 1, 3)))
                cooldown_duration = base_cd * (1.2 ** min(cur_fails - 1, 3))
                if is_risk:
                    logger.warning(
                        f"[风控拦截避让] 节点 [{provider_key}] 命中风控规则: {risk_reason}，"
                        f"触发避让冷却 {cooldown_duration:.1f}s (第 {cur_fails} 次失败)"
                    )
            elif is_auth_error:
                cooldown_duration = max(60.0, base_cd)
            else:
                # 普通网络超时或 5xx
                cooldown_duration = min(base_cd, 15.0 * cur_fails)

            end_time = time.time() + cooldown_duration
            # 如果已有更长的冷却期，保留较长者
            self._cooldowns[provider_key] = max(self._cooldowns.get(provider_key, 0.0), end_time)

            logger.warning(
                f"[对等互备网格] 节点 [{provider_key}] 失败 (HTTP {status})，进入冷却避让期 "
                f"{cooldown_duration:.1f}s (第 {cur_fails} 次连续失败, 剩余冷却: {cooldown_duration:.1f}s)"
            )
            return cooldown_duration

    async def record_success(self, provider_key: str) -> None:
        """调用成功，重置失败计数并清除冷却状态"""
        async with self._lock:
            self._failure_counts[provider_key] = 0
            self._cooldowns.pop(provider_key, None)

    async def apply_pacing_and_acquire(self, provider_key: str) -> None:
        """自适应调用步调（Pacing & Jitter）及并发获取"""
        if os.getenv("GATEWAY_DISABLE_PACING") == "1":
            self._last_call_time[provider_key] = time.time()
            return

        meta = PROVIDER_RISK_SPECS.get(provider_key)
        if not meta:
            return

        # 1. 频控间隔检查与随机抖动 Jitter
        now = time.time()
        last_time = self._last_call_time.get(provider_key, 0.0)
        elapsed = now - last_time

        jitter = random.uniform(meta.jitter_range[0], meta.jitter_range[1])
        target_interval = meta.min_call_interval + jitter

        if elapsed < target_interval:
            wait_time = target_interval - elapsed
            await asyncio.sleep(wait_time)

        self._last_call_time[provider_key] = time.time()

    def get_semaphore(self, provider_key: str) -> asyncio.Semaphore:
        """获取节点的并发控制信号量"""
        if provider_key not in self._semaphores:
            meta = PROVIDER_RISK_SPECS.get(provider_key)
            limit = meta.max_concurrency if meta else 4
            self._semaphores[provider_key] = asyncio.Semaphore(limit)
        return self._semaphores[provider_key]

    def reset(self) -> None:
        """测试或管理员重置所有冷却和计数状态"""
        self._cooldowns.clear()
        self._failure_counts.clear()
        self._last_call_time.clear()
        disabled_str = os.getenv("DISABLED_PROVIDERS", "")
        self.disabled_providers = {
            p.strip().lower() for p in disabled_str.split(",") if p.strip()
        }


class TrafficPacer:
    """网关自适应流量整形器：提供并发信号量管理、随机抖动防风控与等待度量。"""

    def __init__(self, cooldown_tracker: Optional[CooldownTracker] = None):
        self.cooldown_tracker = cooldown_tracker
        self._semaphores: Dict[str, asyncio.Semaphore] = {
            pk: asyncio.Semaphore(meta.max_concurrency)
            for pk, meta in PROVIDER_RISK_SPECS.items()
        }
        self._last_call_times: Dict[str, float] = {}
        self._lock = asyncio.Lock()
        self._total_waits: int = 0
        self._total_delay_ms: float = 0.0

    @asynccontextmanager
    async def acquire(self, provider_key: str) -> AsyncIterator[float]:
        """获取并发许可并在必要时应用拟人随机延迟 Jitter。
        yields: 注入的等待毫秒数 wait_ms (float)。
        """
        pk = (provider_key or "").lower()
        if pk not in self._semaphores:
            meta = PROVIDER_RISK_SPECS.get(pk)
            limit = meta.max_concurrency if meta else 2
            self._semaphores[pk] = asyncio.Semaphore(limit)

        sem = self._semaphores[pk]
        async with sem:
            wait_ms = 0.0
            if os.getenv("GATEWAY_DISABLE_PACING") != "1":
                meta = PROVIDER_RISK_SPECS.get(pk)
                if meta:
                    async with self._lock:
                        now = time.time()
                        last = self._last_call_times.get(pk, 0.0)
                        elapsed = now - last
                        jitter = random.uniform(meta.jitter_range[0], meta.jitter_range[1])
                        target_interval = meta.min_call_interval + jitter
                        if elapsed < target_interval:
                            wait_time = target_interval - elapsed
                            wait_ms = wait_time * 1000.0
                            self._last_call_times[pk] = now + wait_time
                        else:
                            self._last_call_times[pk] = now

                    if wait_ms > 0:
                        await asyncio.sleep(wait_ms / 1000.0)
                        self._total_waits += 1
                        self._total_delay_ms += wait_ms

            yield round(wait_ms, 2)

    def get_stats(self) -> Dict[str, Any]:
        active_semaphores = {}
        providers_info = {}
        for pk, sem in self._semaphores.items():
            meta = PROVIDER_RISK_SPECS.get(pk)
            limit = meta.max_concurrency if meta else 2
            cur_available = getattr(sem, "_value", limit)
            in_use = max(0, limit - cur_available)
            active_semaphores[pk] = in_use
            providers_info[pk] = {
                "max_concurrency": limit,
                "in_use": in_use,
                "min_interval_s": meta.min_call_interval if meta else 0.1,
                "jitter_s": list(meta.jitter_range) if meta else [0.01, 0.05],
                "risk_level": meta.risk_level.value if meta else "medium",
            }
        return {
            "active_semaphores": active_semaphores,
            "providers": providers_info,
            "total_waits": self._total_waits,
            "avg_delay_ms": round(self._total_delay_ms / max(1, self._total_waits), 2),
        }


_traffic_pacer_instance: Optional[TrafficPacer] = None


def get_traffic_pacer() -> TrafficPacer:
    """获取全局单例流量整形器"""
    global _traffic_pacer_instance
    if _traffic_pacer_instance is None:
        _traffic_pacer_instance = TrafficPacer()
    return _traffic_pacer_instance


@dataclass
class FailoverCandidate:
    """后备候选节点"""
    provider_key: str
    wire_model: str
    risk_level: RiskLevel
    in_cooldown: bool
    remaining_cooldown: float
    failure_count: int


@dataclass
class FailoverEvent:
    """故障转移元数据记录"""
    from_provider: str
    from_model: str
    to_provider: str
    to_model: str
    reason: str
    ring: CapabilityRing

    def to_dict(self) -> dict:
        return {
            "from_provider": self.from_provider,
            "from_model": self.from_model,
            "to_provider": self.to_provider,
            "to_model": self.to_model,
            "reason": self.reason,
            "ring": self.ring.value,
        }


class FailoverRouter:
    """五大模型对等互备路由网格调度器"""

    def __init__(
        self,
        provider_factory_map: Optional[Dict[str, Callable]] = None,
        cooldown_tracker: Optional[CooldownTracker] = None,
    ):
        """
        provider_factory_map: provider_key -> 实例化 provider 的无参或单参 lambda
        """
        self.cooldown_tracker = cooldown_tracker or CooldownTracker()
        self._factory_map = provider_factory_map or {}

    def set_provider_factory(self, provider_key: str, factory: Callable) -> None:
        self._factory_map[provider_key] = factory

    def get_fallback_candidates(
        self,
        primary_provider: str,
        primary_model: str,
        excluded_providers: Optional[Set[str]] = None,
    ) -> List[FailoverCandidate]:
        """根据当前主选节点生成同等能力环内的有序候选节点列表。

        排序策略：
        1. 排除当前排除集（已尝试失败过的 Provider 以及被主动禁用的 Provider）
        2. 健康节点排在冷却避让中节点之前
        3. 健康节点中：低风险 (LOW) > 中风险 (MEDIUM) > 高风险 (HIGH)
        4. 同等风险下，按历史失败次数升序排列
        """
        ring = classify_capability_ring(primary_provider, primary_model)
        ring_members = RING_DEFINITIONS.get(ring, [])
        excluded = set(excluded_providers or ()) | {primary_provider}

        candidates: List[FailoverCandidate] = []

        # 遍历环中所有成员
        for p_key, w_model in ring_members:
            if p_key in excluded or self.cooldown_tracker.is_disabled(p_key):
                continue

            meta = PROVIDER_RISK_SPECS.get(p_key)
            risk = meta.risk_level if meta else RiskLevel.MEDIUM
            in_cd = self.cooldown_tracker.is_in_cooldown(p_key)
            rem_cd = self.cooldown_tracker.get_remaining_cooldown(p_key)
            fails = self.cooldown_tracker._failure_counts.get(p_key, 0)

            candidates.append(
                FailoverCandidate(
                    provider_key=p_key,
                    wire_model=w_model,
                    risk_level=risk,
                    in_cooldown=in_cd,
                    remaining_cooldown=rem_cd,
                    failure_count=fails,
                )
            )

        # 排序键：
        # (in_cooldown: 0 或 1, risk_score: LOW=0, MED=1, HIGH=2, failure_count: int, remaining_cd: float)
        def sort_key(c: FailoverCandidate):
            risk_val = 0 if c.risk_level == RiskLevel.LOW else (1 if c.risk_level == RiskLevel.MEDIUM else 2)
            cd_val = 1 if c.in_cooldown else 0
            return (cd_val, risk_val, c.failure_count, c.remaining_cooldown)

        candidates.sort(key=sort_key)
        return candidates

    async def execute_chat(
        self,
        primary_provider_key: str,
        primary_wire_model: str,
        provider_factory: Callable[[str], Any],
        chat_args: dict,
    ) -> Tuple[AsyncIterator[Tuple[str, dict]], Optional[FailoverEvent], Any]:
        """执行带透明故障转移的流式聊天调用。

        - 优先尝试初选节点；
        - 若触发 401, 429, 502, 503, 504, WAF 人机校验或 ProviderError，记录错误与冷却；
        - 从三级能力环中挑选健康候补节点无缝替换；
        - 在首包返回前若出现故障均可透明切换；
        - 首包到达后流式安全输出。
        - 返回 (stream_generator, failover_event, active_provider)。
        """
        ring = classify_capability_ring(primary_provider_key, primary_wire_model)
        tried_providers: Set[str] = set()
        composite_errors: List[str] = []

        curr_provider_key = primary_provider_key
        curr_wire_model = primary_wire_model
        failover_event: Optional[FailoverEvent] = None
        primary_exc: Optional[Exception] = None

        while True:
            tried_providers.add(curr_provider_key)
            sem = self.cooldown_tracker.get_semaphore(curr_provider_key)

            try:
                # 1. 频控规避 (Pacing & Jitter)
                await self.cooldown_tracker.apply_pacing_and_acquire(curr_provider_key)

                # 2. 实例化 Provider
                async with sem:
                    provider = provider_factory(curr_provider_key)

                    # 3. 试探性获取首包 (确保在流交给外部客户端前捕获初始化错误与上游 4xx/5xx)
                    call_kwargs = dict(chat_args)
                    call_kwargs["model"] = curr_wire_model

                    raw_stream = provider.chat(**call_kwargs)

                    # 尝试读取第一个元素（以捕获握手及首包异常）
                    first_item: Optional[Tuple[str, dict]] = None
                    try:
                        first_item = await raw_stream.__anext__()
                    except StopAsyncIteration:
                        first_item = None

                    # 首包读取成功，说明此 Provider 健康可用
                    await self.cooldown_tracker.record_success(curr_provider_key)

                    # 构造包装后的安全生成器
                    async def wrapped_stream() -> AsyncIterator[Tuple[str, dict]]:
                        nonlocal first_item, raw_stream
                        # 输出首个包
                        if first_item is not None:
                            delta, meta = first_item
                            # 如果发生了故障转移，在首包元数据注入标记
                            if failover_event:
                                meta = dict(meta or {})
                                meta["x-failover-from"] = failover_event.from_provider
                                meta["x-failover-to"] = failover_event.to_provider
                                meta["x-failover-model"] = failover_event.to_model
                                meta["x-failover-ring"] = failover_event.ring.value
                            yield delta, meta

                        # 持续迭代剩余数据包
                        try:
                            async for item in raw_stream:
                                yield item
                        except Exception as stream_err:
                            logger.error(
                                f"[对等互备网格] 节点 [{curr_provider_key}] 流传输中途断开: {stream_err}"
                            )
                            # 中途断开按原有规范上抛，不产生半包污染
                            raise stream_err

                    return wrapped_stream(), failover_event, provider

            except (ProviderError, Exception) as exc:
                if primary_exc is None:
                    primary_exc = exc
                status_code = getattr(exc, "status", 502)
                err_msg = str(getattr(exc, "message", str(exc)))
                err_summary = f"[{curr_provider_key}:{curr_wire_model}] -> HTTP {status_code}: {err_msg}"
                composite_errors.append(err_summary)
                logger.warning(f"[对等互备网格] 尝试失败: {err_summary}")

                # 记录失败并进入动态冷却
                await self.cooldown_tracker.record_failure(curr_provider_key, status_code, err_msg)

                # 寻找同能力环内的下一个健康备用节点
                candidates = self.get_fallback_candidates(
                    primary_provider_key,
                    primary_wire_model,
                    excluded_providers=tried_providers,
                )

                if not candidates:
                    # 环内所有节点均已遍历尝试完毕
                    report = (
                        f"对等能力环 [{ring.value}] 内所有节点均已尝试且全部失效。\n"
                        f"故障详情与避让记录:\n" + "\n".join(f" - {e}" for e in composite_errors)
                    )
                    logger.error(f"[对等互备网格] 全部后备节点熔断: {report}")
                    status = getattr(primary_exc, "status", 502) if primary_exc else 502
                    err_type = getattr(primary_exc, "err_type", "upstream_error") if primary_exc else "upstream_error"
                    raise ProviderError(report, status=status, err_type=err_type)

                # 选取最优后备候选
                next_cand = candidates[0]
                failover_event = FailoverEvent(
                    from_provider=primary_provider_key,
                    from_model=primary_wire_model,
                    to_provider=next_cand.provider_key,
                    to_model=next_cand.wire_model,
                    reason=err_msg,
                    ring=ring,
                )
                logger.info(
                    f"[对等互备网格] 自动故障转移: [{curr_provider_key}] -> [{next_cand.provider_key}] "
                    f"(模型: {next_cand.wire_model}, 原因: {err_msg[:60]}...)"
                )
                curr_provider_key = next_cand.provider_key
                curr_wire_model = next_cand.wire_model
