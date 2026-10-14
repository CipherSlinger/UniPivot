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
import contextlib
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
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from .base import ProviderError
from .credential_renewal import get_credential_renewal
from .credential_watch import get_credential_watch
from .quarantine import get_quarantine_store
from .risk_correlation import get_risk_correlation
from .risk_ledger import (
    SEVERITY_AUTH,
    SEVERITY_RATE_LIMIT,
    SEVERITY_WAF,
    _MAX_EVIDENCE_CHARS,
    classify_interception_severity,
    get_risk_ledger,
)

logger = logging.getLogger("gateway.failover")

# [OPT-028] 观察期的**证据积累期**下限（秒）：即便 `GATEWAY_PROBATION_WINDOW=0`
# 也必须保留这么长的记录，否则「零证据仍在门控」的节点永远无法通过后续成功毕业。
_PROBATION_EVIDENCE_FLOOR = 300.0

# [OPT-029] 「凭证确已更新」可解除的避让成因：认证失效（令牌过期，重登即修）与
# 人机验证（RGV587 / 验证码，**只有**人工重登能解除）。两者被解除的依据是同一条
# 可观测事实——凭证文件变了。频控**永不**在内：它本就会自行衰减，重登毫无意义。
_RELEASABLE_BY_RENEWAL = (SEVERITY_AUTH, SEVERITY_WAF)


# ---------------------------------------------------------------- [OPT-014] 风控证据驱动的自适应步调

# 各严重度对步调放大的权重（WAF 人机验证最危险 → 放慢最多）
_SEVERITY_PACING_WEIGHT: Dict[str, float] = {
    SEVERITY_WAF: 1.0,
    SEVERITY_AUTH: 0.8,
    SEVERITY_RATE_LIMIT: 0.6,
}
# provider -> (multiplier, expires_at) 证据缓存，避免每请求都读取台账文件
_pacing_evidence_cache: Dict[str, Tuple[float, float]] = {}


def compute_pacing_multiplier(
    severities: Mapping[str, int], max_multiplier: float = 2.5
) -> float:
    """依据近期拦截严重度分布计算步调放大系数（纯函数，>= 1.0）。

    每单位权重线性放大 25%，并以 ``max_multiplier`` 封顶；无证据时返回 1.0。
    拦截越多、越危险（WAF > auth > ratelimit），上游调用间隔被拉得越长，
    从而在无法绕开风控的前提下主动「降速避让」。
    """
    if not severities:
        return 1.0
    weight = 0.0
    for sev, cnt in severities.items():
        try:
            weight += _SEVERITY_PACING_WEIGHT.get(str(sev), 0.5) * int(cnt)
        except (TypeError, ValueError):
            continue
    ceiling = max(1.0, float(max_multiplier))
    return float(min(max(1.0, 1.0 + 0.25 * weight), ceiling))


def _global_risk_pacing_multiplier(now: Optional[float] = None) -> float:
    """[OPT-024] 全局系统性风控压制倍率（多个不同账号被同时拦截时整体降速）。

    无关联证据时恒为 1.0，对既有步调零影响；查询异常亦返回 1.0，绝不影响请求流程。
    """
    try:
        return get_risk_correlation().cached_global_pacing_multiplier(now)
    except Exception:  # noqa: BLE001
        return 1.0


def get_adaptive_pacing_multiplier(
    provider_key: str, now: Optional[float] = None
) -> float:
    """查询指定 Provider 的自适应步调放大系数（带 TTL 缓存的证据查询）。

    GATEWAY_DISABLE_ADAPTIVE_PACING=1 时恒为 1.0（退化为基础步调）。
    查询失败/台账不可用时保守返回 1.0，绝不影响请求流程。
    """
    if os.getenv("GATEWAY_DISABLE_ADAPTIVE_PACING", "").strip().lower() in ("1", "true", "yes"):
        return 1.0
    pk = (provider_key or "").lower()
    now = now if now is not None else time.time()
    cached = _pacing_evidence_cache.get(pk)
    if cached and cached[1] > now:
        return cached[0]
    try:
        ttl = float(os.getenv("GATEWAY_PACING_EVIDENCE_TTL", "60"))
        hours = float(os.getenv("GATEWAY_PACING_EVIDENCE_HOURS", "6"))
        max_mult = float(os.getenv("GATEWAY_PACING_MAX_MULTIPLIER", "2.5"))
        recent = get_risk_ledger().recent_interceptions(hours=hours, now=now)
        bucket = (recent.get("per_provider") or {}).get(pk) or {}
        mult = compute_pacing_multiplier(bucket.get("severities") or {}, max_mult)
    except Exception:  # noqa: BLE001 - 证据查询失败绝不阻断请求
        mult = 1.0
    _pacing_evidence_cache[pk] = (mult, now + max(0.0, ttl))
    return mult


def clear_pacing_evidence_cache() -> None:
    """清空自适应步调证据缓存（测试用）。"""
    _pacing_evidence_cache.clear()


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
        # 风控拦截台账与冷却状态持久化（best-effort）
        self.risk_ledger = get_risk_ledger()
        # [OPT-020] 运维人工长期封禁租约：折入冷却读数，使所有门控自动遵循
        self.quarantine = get_quarantine_store()
        # [OPT-025] 全环凭证失效检测：整环 401 时快速失败，杜绝"无效凭证风暴"
        self.credential_watch = get_credential_watch()
        # [OPT-026] 凭证续期检测：认证失败时留指纹，续期后据此复位「认证级」避让
        self.renewal = get_credential_renewal()
        # [OPT-026] provider -> 当前冷却的成因严重度（auth/ratelimit/waf），
        # 随冷却一并持久化：只有确认冷却由「认证失效」造成，才允许被续期复位。
        self._cooldown_severity: Dict[str, str] = {}
        # [OPT-027] 「新凭证仍被拒」证据：provider -> 自凭证更新以来，拿着**已更新**的
        # 凭证仍被 401 拒绝的连续次数。凭证换过仍被拒 => 问题不在凭证上（账号级封禁），
        # 继续按「凭证过期」处置只会一次次重新触碰被封的账号。
        self._fresh_credential_rejected: Dict[str, int] = {}
        # [OPT-027] provider -> 自动推断租约生效时所用的凭证指纹。用于判定「自动租约的
        # 推断依据是否已失效」：若之后又出现更新的凭证，说明当时那次「新凭证仍被拒」
        # 只针对旧凭证，自动租约必须可被自动解除并重新验证。
        self._auto_lease_fingerprint: Dict[str, str] = {}
        self._ban_stats: Dict[str, Any] = {
            "rejections_after_renewal": 0,
            "auto_leases_applied": 0,
            "auto_leases_lifted": 0,
        }
        # [OPT-022] 避让解除后的「恢复观察期」(probation) 状态
        # provider -> 冷却 episode 的预计结束时间戳（解除后据此判定观察期）
        self._episode_ends: Dict[str, float] = {}
        # provider -> 观察期内已连续成功次数（达标即提前毕业）
        self._probation_successes: Dict[str, int] = {}
        # provider -> 观察期串行闸门（仅放行单个握手，避免惊群）
        self._probation_locks: Dict[str, asyncio.Lock] = {}
        self._probation_stats: Dict[str, Any] = {
            "entered": 0,
            "graduated": 0,
            "canary_serialized": 0,
            "canary_successes": 0,
        }
        self._restore_persisted_state()

    def _restore_persisted_state(self) -> None:
        """从磁盘恢复冷却状态与连续失败计数，使避让策略跨重启生效。"""
        try:
            state = self.risk_ledger.load_state()
        except Exception:  # noqa: BLE001 - 恢复失败不得影响启动
            return
        if not state:
            return
        now = time.time()
        cooldowns = state.get("cooldowns") or {}
        failures = state.get("failure_counts") or {}
        restored_cd = 0
        if isinstance(cooldowns, dict):
            for pk, end in cooldowns.items():
                if isinstance(end, (int, float)) and end > now:
                    self._cooldowns[str(pk)] = float(end)
                    restored_cd += 1
        if isinstance(failures, dict):
            for pk, n in failures.items():
                if isinstance(n, int) and n > 0:
                    self._failure_counts[str(pk)] = n
        # [OPT-026] 恢复冷却成因：缺失时**不**推断（fail-closed，宁可继续避让），
        # 因此本机制启用前遗留的冷却不会被误判为认证成因而提前释放。
        severity_map = state.get("cooldown_severity") or {}
        if isinstance(severity_map, dict):
            for pk, sev in severity_map.items():
                if isinstance(sev, str) and sev:
                    self._cooldown_severity[str(pk)] = sev
        # [OPT-027] 恢复「新凭证仍被拒」证据与自动租约指纹。二者都必须跨界存活：
        # 一次重启若能抹掉「我们已经试过新凭证」这一事实，就等于给了被封账号一个
        # 重新被反复触碰的入口。
        rejected_map = state.get("fresh_credential_rejected") or {}
        if isinstance(rejected_map, dict):
            for pk, n in rejected_map.items():
                if isinstance(n, int) and n > 0:
                    self._fresh_credential_rejected[str(pk)] = n
        auto_fp_map = state.get("auto_lease_fingerprint") or {}
        if isinstance(auto_fp_map, dict):
            for pk, fp in auto_fp_map.items():
                if isinstance(fp, str) and fp:
                    self._auto_lease_fingerprint[str(pk)] = fp
        # [OPT-022] 恢复 episode 结束时间：据此可在解封瞬间重新启用观察期，
        # 使「避让期间重启」不会绕过惊群保护。仅保留观察期窗口内仍有意义的记录。
        episode_ends = state.get("episode_ends") or {}
        if isinstance(episode_ends, dict):
            # [OPT-028] 保留期限必须覆盖**续期后**的完整视野：否则一次「解封后重启」
            # 会把仍在门控中的观察期记录整体丢弃，节点立即以全并发回到环内。
            retention = self._probation_retention()
            for pk, end in episode_ends.items():
                if isinstance(end, (int, float)) and now < float(end) + retention:
                    self._episode_ends[str(pk)] = float(end)
        if restored_cd:
            logger.warning(
                "[风控避让] 已从磁盘恢复 %d 个节点的跨重启冷却状态: %s",
                restored_cd,
                {k: round(v - now, 1) for k, v in self._cooldowns.items()},
            )

    def _persist_state(self) -> None:
        """持久化冷却状态与失败计数（best-effort，绝不上抛）。"""
        try:
            self.risk_ledger.save_state(
                {
                    "cooldowns": {k: v for k, v in self._cooldowns.items()},
                    "failure_counts": {k: v for k, v in self._failure_counts.items()},
                    # [OPT-022] 观察期判定依赖 episode 结束时间，须随冷却一并持久化，
                    # 否则「避让期间重启」会丢失观察期，导致解封瞬间直接放行并发洪峰。
                    "episode_ends": {k: v for k, v in self._episode_ends.items()},
                    # [OPT-026] 冷却成因同样必须持久化：否则重启后无法判断
                    # 「这次冷却是否由认证失效造成」，复位便失去依据（fail-closed）。
                    "cooldown_severity": {k: v for k, v in self._cooldown_severity.items()},
                    # [OPT-027] 「新凭证仍被拒」的证据与自动租约指纹必须跨界存活：
                    # 重启若能抹掉它，账号就会被重新反复触碰。
                    "fresh_credential_rejected": {k: v for k, v in self._fresh_credential_rejected.items()},
                    "auto_lease_fingerprint": {k: v for k, v in self._auto_lease_fingerprint.items()},
                }
            )
        except Exception:  # noqa: BLE001
            pass

    def is_disabled(self, provider_key: str) -> bool:
        """检查指定 Provider 是否被配置主动禁用"""
        return (provider_key or "").lower() in self.disabled_providers

    def is_in_cooldown(self, provider_key: str) -> bool:
        """检查指定 Provider 是否处于冷却避让期（含运维封禁租约）"""
        return self.get_remaining_cooldown(provider_key) > 0.0

    def get_remaining_cooldown(self, provider_key: str) -> float:
        """获取剩余冷却时间（秒），若未处于冷却期则返回 0.0。

        [OPT-020] 读取端取「自动冷却」与「运维封禁租约」的最大值：
        租约只能延长避让窗口、绝不缩短既有冷却，因此所有冷却门控
        （探针 / 刷新 / 会话回收 / 自愈选举 / 路由选点 / 自适应步调）
        都能自动遵循人工长期封禁，无需逐处接线。
        """
        now = time.time()
        end_time = self._cooldowns.get(provider_key, 0.0)
        remaining = max(0.0, end_time - now)
        try:
            lease = self.quarantine.remaining(provider_key, now)
        except Exception:  # noqa: BLE001 - 租约查询失败绝不阻断既有冷却语义
            lease = 0.0
        return max(remaining, lease)

    def is_quarantined(self, provider_key: str) -> bool:
        """[OPT-021] 检查节点是否处于**运维长期封禁租约**（硬阻断）。

        与 `is_in_cooldown`（软冷却，仅降级排序）语义不同：
        - 软冷却由自动风控触发、时长有界（30s~24h），是可自愈的短暂退避，
          环内全部节点都冷却时「最后一搏」仍可尝试，以免可用性回退；
        - 运维封禁租约表达的是「该账号已被平台封禁」，持续时间以天计，
          任何触碰都不会有正常返回，只会延长封禁。因此必须**硬阻断**：
          既不可作为主选派发，也不可作为故障转移的后备候选，只能快速失败。
        """
        if os.getenv("GATEWAY_DISABLE_QUARANTINE_HARD_STOP", "").strip().lower() in ("1", "true", "yes"):
            return False
        try:
            return self.quarantine.remaining(provider_key) > 0.0
        except Exception:  # noqa: BLE001 - 查询失败时按未封禁处理，绝不误伤正常节点
            return False

    # -------------------------------------------------------- [OPT-022] 恢复观察期

    def _probation_window(self) -> float:
        try:
            return max(0.0, float(os.getenv("GATEWAY_PROBATION_WINDOW", "300")))
        except ValueError:
            return 300.0

    def _probation_min_successes(self) -> int:
        try:
            return max(1, int(os.getenv("GATEWAY_PROBATION_MIN_SUCCESSES", "3")))
        except ValueError:
            return 3

    def _probation_pacing_factor(self) -> float:
        try:
            return max(1.0, float(os.getenv("GATEWAY_PROBATION_PACING_FACTOR", "2.0")))
        except ValueError:
            return 2.0

    def _probation_max_renewals(self) -> int:
        """观察期窗口的续期次数上限。

        [OPT-028] 窗口原先是**硬截止**：到点即放开全并发，无论是否拿到过证据。
        但低流量节点（尤其夜间）在 300 秒里往往连一次握手都没有，于是这个保护
        在最需要它的场景下静默失效——解封瞬间被压制的需求随后以全并发抵达一个
        **仍可能处于封禁中**的账号。改为按窗口续期：每耗尽一个窗口即自动续期
        一次，最多 `GATEWAY_PROBATION_MAX_RENEWALS`（默认 3，即最多门控
        `窗口 × 4`）次，之后按证据决定是否毕业（见 `is_in_probation`）。
        `GATEWAY_PROBATION_MAX_RENEWALS=0` 恢复正常。
        """
        try:
            return max(0, int(os.getenv("GATEWAY_PROBATION_MAX_RENEWALS", "3")))
        except ValueError:
            return 3

    def _probation_horizon(self, last_release: float, now: float) -> float:
        """[OPT-028] 返回观察期的**有效截止时刻**，含依时间自动续期。

        纯函数、无副作用、无需持久化：续期次数由「已流逝的窗口数」推导，
        因此重启、乱序查询都不会改变结论，也不存在读路径写放大的问题。
        """
        window = self._probation_window()
        if window <= 0.0:
            return last_release
        elapsed = max(0.0, now - last_release)
        renewals = min(int(elapsed // window), self._probation_max_renewals())
        return last_release + window * (1 + renewals)

    def _probation_retention(self) -> float:
        """观察期记录的保留时长（须覆盖续期后的完整视野，否则重启即抹掉观察期）。

        [OPT-028] 窗口为 0（「不给时间宽限」）时仍须保留一段**证据积累期**：观察期
        的毕业条件已是证据而非时间，若保留期随窗口一起归零，则零证据仍在门控的节点
        将永远无法通过后续成功毕业（成功握手连记录都不被接受），门控变成死锁。
        """
        window = max(self._probation_window(), _PROBATION_EVIDENCE_FLOOR)
        return window * (1 + self._probation_max_renewals())

    def is_probation_disabled(self) -> bool:
        return os.getenv("GATEWAY_DISABLE_PROBATION", "").strip().lower() in ("1", "true", "yes")

    def _last_release_ts(self, provider_key: str, now: float) -> Optional[float]:
        """返回节点最近一次「避让解除」的时间戳（未解除过或无记录则 None）。

        两个来源取较晚者：
        1. 冷却 episode 的预计结束时间（`record_failure` 时即已排定）；
        2. 运维封禁租约的解除/到期时间（`QuarantineStore` 记录）。
        两者都只在该时刻**已经过去**时才参与判定（否则说明仍在避让中）。
        """
        candidates: List[float] = []
        episode_end = self._episode_ends.get(provider_key)
        if episode_end is not None and episode_end <= now:
            candidates.append(episode_end)
        try:
            released = self.quarantine.last_released_at(provider_key)
        except Exception:  # noqa: BLE001
            released = None
        if isinstance(released, (int, float)) and released <= now:
            candidates.append(float(released))
        return max(candidates) if candidates else None

    def is_in_probation(self, provider_key: str) -> bool:
        """[OPT-022] 节点是否处于「避让解除后的恢复观察期」。

        动机：一次多日的账号级封禁解除后，网关侧往往已积压了大量需求，且此前被
        冷却压制的并发会在同一瞬间释放——若放任其以正常并发（max_concurrency）
        涌入刚刚解封的账号，正是当初触发封禁的流量特征，极易被立刻二次封禁。

        因此避让一旦解除，节点先进入观察期：串行放行单个「探路」请求（握手阶段
        独占），并以放大的步调间隔发送；累计达到 `GATEWAY_PROBATION_MIN_SUCCESSES`
        次连续成功后方可毕业，恢复常规并发与步调。`GATEWAY_DISABLE_PROBATION=1`
        可关闭。

        [OPT-028] 毕业条件必须**基于证据**，而不是「时间到了就算毕业」：窗口只
        作为续期单位（`GATEWAY_PROBATION_MAX_RENEWALS`，默认 3），耗尽续期预算后
        仍要求至少一次成功握手才放行——零证据的节点（自 episode 结束以来从未被
        成功验证过，包括从未被触达的情形）保持串行门控。对从未被触达的节点而言
        继续门控**不产生任何代价**（没有请求到来就无所谓洪峰），而放开则等于把
        一个完全未经验证的账号直接交给全并发，恰是本观察期存在的理由所反对的。
        """
        if self.is_probation_disabled():
            return False
        provider_key = (provider_key or "").strip().lower()
        if not provider_key:
            return False
        now = time.time()
        # 仍在避让中（冷却或封禁）不属观察期，避免语义混淆
        if self.get_remaining_cooldown(provider_key) > 0.0:
            return False
        last_release = self._last_release_ts(provider_key, now)
        if last_release is None:
            return False
        successes = self._probation_successes.get(provider_key, 0)
        if now >= self._probation_horizon(last_release, now):
            # 续期预算耗尽 -> 仅凭证据毕业：有正面证据即放行，零证据继续门控。
            return successes <= 0
        return successes < self._probation_min_successes()

    def probation_lock(self, provider_key: str) -> Optional[asyncio.Lock]:
        """观察期内返回串行闸门（握手阶段独占），否则返回 None。

        闸门仅在观察期生效，用于把「解封瞬间的并发洪峰」压成逐个探路，
        毕业或观察期结束即自动失效，不影响正常并发。
        """
        if not self.is_in_probation(provider_key):
            return None
        lock = self._probation_locks.get(provider_key)
        if lock is None:
            lock = asyncio.Lock()
            self._probation_locks[provider_key] = lock
        self._probation_stats["canary_serialized"] += 1
        return lock

    def probation_pacing_multiplier(self, provider_key: str) -> float:
        """观察期内放大调用间隔的倍率（非观察期恒为 1.0）。"""
        return self._probation_pacing_factor() if self.is_in_probation(provider_key) else 1.0

    def _enter_probation(self, provider_key: str, episode_end: float) -> None:
        """记录冷却 episode 的预计结束时间，并重置观察期成功计数。

        必须在 `record_failure` 中调用（持锁），这样即便进程在避让期间重启，
        观察期判定依然可基于持久化的 episode 结束时间推导出来。
        """
        self._episode_ends[provider_key] = episode_end
        self._probation_successes[provider_key] = 0
        self._probation_stats["entered"] += 1

    def _note_probation_success(self, provider_key: str) -> None:
        """观察期内记录一次成功；达标即毕业（提前结束观察期）。"""
        if self.is_probation_disabled():
            return
        now = time.time()
        if self._last_release_ts(provider_key, now) is None:
            return
        if provider_key not in self._episode_ends:
            return
        # [OPT-028] 成功握手是**证据**，因此在 episode 记录的整个保留期内都应被计入，
        # 而不能像旧语义那样过了窗口就丢弃：否则「窗口耗尽、零证据、仍在门控」的
        # 节点永远无法通过后续成功毕业（续期预算耗尽后的门控必须以证据为出口）。
        if now >= self._episode_ends[provider_key] + self._probation_retention():
            return
        self._probation_successes[provider_key] = self._probation_successes.get(provider_key, 0) + 1
        self._probation_stats["canary_successes"] += 1
        if self._probation_successes[provider_key] >= self._probation_min_successes():
            self._probation_stats["graduated"] += 1
            logger.info(
                "[恢复观察期] 节点 [%s] 已连续成功 %d 次，观察期提前毕业，恢复常规并发与步调",
                provider_key,
                self._probation_successes[provider_key],
            )

    def get_probation_states(self, now: Optional[float] = None) -> Dict[str, Dict[str, Any]]:
        """列出当前处于观察期的节点（供 /v1/diagnostics 审计）。"""
        now = time.time() if now is None else now
        result: Dict[str, Dict[str, Any]] = {}
        for pk in list(self._episode_ends.keys()):
            if not self.is_in_probation(pk):
                continue
            last_release = self._last_release_ts(pk, now)
            release = last_release or now
            window = self._probation_window()
            # [OPT-028] 已续期次数与「尚无任何证据」标记：运维需要能区分
            # 「正在被验证且进展顺利」与「一直没能被验证」这两种观察期。
            renewals = 0
            if window > 0.0:
                renewals = min(
                    int(max(0.0, now - release) // window), self._probation_max_renewals()
                )
            successes = self._probation_successes.get(pk, 0)
            result[pk] = {
                "successes": successes,
                "required_successes": self._probation_min_successes(),
                "renewals": renewals,
                "max_renewals": self._probation_max_renewals(),
                "awaiting_first_handshake": successes <= 0,
                "remaining_seconds": round(
                    max(0.0, self._probation_horizon(release, now) - now), 1
                ),
            }
        return result

    def get_probation_stats(self) -> Dict[str, Any]:
        stats = dict(self._probation_stats)
        # [OPT-028] 派生指标（只读）：续期与「零证据仍被门控」的规模，
        # 使「观察期是否真的在起作用」可被观测，而非只能看到「进入了多少次」。
        try:
            states = self.get_probation_states()
            stats["renewing"] = sum(1 for s in states.values() if s["renewals"] > 0)
            stats["awaiting_first_handshake"] = sum(
                1 for s in states.values() if s["awaiting_first_handshake"]
            )
            stats["budget_exhausted"] = sum(
                1 for s in states.values() if s["renewals"] >= s["max_renewals"]
            )
        except Exception:  # noqa: BLE001 - 派生指标失败绝不阻断诊断
            pass
        return stats

    def get_all_cooldowns(self) -> Dict[str, float]:
        """获取所有当前处于冷却状态的节点及剩余冷却时间（秒）"""
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

            # 严重度分级：auth（令牌/账号级）> ratelimit（频控）> waf（人机验证）
            severity = classify_interception_severity(
                is_risk=is_risk, risk_reason=risk_reason, status=status, error_msg=error_msg
            )
            escalated = False
            if severity == SEVERITY_AUTH or is_auth_error:
                # 认证失效通常需重新登录，且高频重试会加剧上游封禁：
                # 首次维持既有短冷却，连续 ≥3 次后按 2 的幂次长效升级并持久化。
                cooldown_duration = max(60.0, base_cd)
                if cur_fails >= 3:
                    long_base = float(os.getenv("GATEWAY_AUTH_COOLDOWN_BASE", "1800"))
                    long_cap = float(os.getenv("GATEWAY_AUTH_COOLDOWN_MAX", "86400"))
                    cooldown_duration = min(
                        long_cap, long_base * (2 ** min(cur_fails - 3, 6))
                    )
                    escalated = True
                logger.warning(
                    f"[风控拦截避让] 节点 [{provider_key}] 认证失效 (HTTP {status}, {severity})，"
                    f"触发{'长效' if escalated else ''}冷却 {cooldown_duration:.1f}s (第 {cur_fails} 次连续失败)"
                )
            elif is_risk or is_waf_or_rate:
                # 针对频控与风控加权冷却 (指数退避: base_cd * (1.2 ** min(cur_fails - 1, 3)))
                waf_cap = float(os.getenv("GATEWAY_WAF_COOLDOWN_MAX", "900"))
                cooldown_duration = min(waf_cap, base_cd * (1.2 ** min(cur_fails - 1, 3)))
                if severity == SEVERITY_WAF:
                    # [OPT-029] 人机验证与频控必须分开处置——两者的**恢复方式相反**：
                    # 频控（429）是瞬时的、会自行衰减，因此「到点重试」本身就是正确的
                    # 恢复策略，短冷却 + 有上限是对的；
                    # 而人机验证（RGV587 / 验证码）**只有人能解除**（OPT-023 档案对此类
                    # 的结论就是 auto_recovers=False、recoverable_by_relogin=True），
                    # 每一次自动重试都是又一次「自动化尝试」，只会加重账号标记。
                    # 实测旧行为：重复的人机验证拦截在第 4 次即饱和于 ~78s 且**永不再增长**
                    # （1.2**3 封顶 + 900s 上限），于是该节点每 ~78 秒被再次触碰一次，无限循环
                    # ——恰恰是对最需要人介入的那一类，给了最短的耐心。这是缺陷的反转形态，
                    # 与 OPT-025 的整环风暴同源，只是发生在单节点上并自我维持。
                    # 因此重复的人机验证必须按 2 的幂次长效升级（与认证级同构）。
                    waf_min = self._waf_escalation_min_failures()
                    if cur_fails >= waf_min:
                        long_base = float(os.getenv("GATEWAY_WAF_LONG_COOLDOWN_BASE", "3600"))
                        long_cap = float(os.getenv("GATEWAY_WAF_LONG_COOLDOWN_MAX", "86400"))
                        cooldown_duration = min(
                            long_cap, long_base * (2 ** min(cur_fails - waf_min, 6))
                        )
                        escalated = True
                        logger.warning(
                            f"[风控拦截避让] 节点 [{provider_key}] 人机验证重复出现 "
                            f"(第 {cur_fails} 次)：该类拦截**只有人工重登可解除**，"
                            f"自动重试只会加重标记，已升级为长效避让 {cooldown_duration:.1f}s。"
                            f"请执行 `.venv/bin/python login.py {provider_key}` 由人工完成验证"
                        )
                if not escalated:
                    logger.warning(
                        f"[风控拦截避让] 节点 [{provider_key}] 命中风控规则: "
                        f"{risk_reason or error_msg[:80]}，"
                        f"触发避让冷却 {cooldown_duration:.1f}s (第 {cur_fails} 次失败)"
                    )
            else:
                # 普通网络超时或 5xx
                cooldown_duration = min(base_cd, 15.0 * cur_fails)

            end_time = time.time() + cooldown_duration
            # 如果已有更长的冷却期，保留较长者
            self._cooldowns[provider_key] = max(self._cooldowns.get(provider_key, 0.0), end_time)
            # [OPT-026] 记录本次冷却的成因，供凭证续期后判定「是否可复位」
            self._cooldown_severity[provider_key] = severity

            # [OPT-025] 认证失效还需上报全环检测器：若多个**不同**节点在同一窗口内
            # 相继 401，说明整环凭证已失效，故障转移不可能成功——必须快速失败而非
            # 逐个打满五个节点（那正是上游眼中的「自动化凭证尝试」流量特征本身）。
            if severity == SEVERITY_AUTH or is_auth_error:
                try:
                    self.credential_watch.note_auth_failure(provider_key)
                except Exception:  # noqa: BLE001 - 检测器异常绝不影响冷却主流程
                    pass
                # [OPT-026] 留下**此刻**的凭证指纹：日后指纹变化即说明凭证已续期，
                # 由 2 的幂次升级来的 24h 冻结便失去了惩罚对象，应被复位。
                # [OPT-027] 该调用同时回答一个关键问题：本次失败是否发生在一枚
                # **已更新过**的凭证上。凭证已经换过了仍被 401 拒绝，就排除了
                # 「凭证只是过期」这一唯一无害解释——问题不在凭证上。
                renewed_at_failure = False
                try:
                    renewed_at_failure = bool(self.renewal.note_auth_failure(provider_key))
                except Exception:  # noqa: BLE001
                    renewed_at_failure = False
                self._note_renewal_rejection(provider_key, renewed_at_failure)
            elif severity == SEVERITY_WAF:
                # [OPT-029] 人机验证同样要留下凭证基线：人工 `login.py`（完成滑块验证）
                # 成功后令牌必然刷新，指纹变化即「人类已介入」的唯一可观测证据。
                # 没有这条基线，长效避让就没有出口——同一个自锁缺陷换了个成因。
                try:
                    self.renewal.note_interception(provider_key)
                except Exception:  # noqa: BLE001 - 基线记录失败绝不影响冷却主流程
                    pass

            # [OPT-022] 真正的风控/认证级避让才需要「解除后恢复观察期」：
            # 普通 5xx/超时说明是上游故障而非账号被标记，解封瞬间并无惊群风险。
            if is_risk or is_waf_or_rate or is_auth_error:
                self._enter_probation(provider_key, self._cooldowns[provider_key])

            # 台账记录：仅记录真正的拦截事件（风控 / 频控 / 认证失效），
            # 普通 5xx 与超时不入账，避免污染风控画像。
            if is_risk or is_waf_or_rate or is_auth_error:
                try:
                    self.risk_ledger.record_interception(
                        provider=provider_key,
                        status=status,
                        reason=risk_reason or error_msg[:_MAX_EVIDENCE_CHARS],
                        severity=severity,
                        cooldown_seconds=cooldown_duration,
                        evidence=error_msg,
                    )
                except Exception:  # noqa: BLE001 - 台账失败绝不影响请求流程
                    pass

            # 持久化冷却状态与失败计数，使避让跨重启生效
            self._persist_state()

            logger.warning(
                f"[对等互备网格] 节点 [{provider_key}] 失败 (HTTP {status})，进入冷却避让期 "
                f"{cooldown_duration:.1f}s (第 {cur_fails} 次连续失败, 剩余冷却: {cooldown_duration:.1f}s)"
            )
            return cooldown_duration

    async def record_success(self, provider_key: str) -> None:
        """调用成功，重置失败计数并清除冷却状态"""
        async with self._lock:
            # [OPT-022] 观察期内的成功计入毕业进度（须在清除 episode 之前判定）
            self._note_probation_success(provider_key)
            # [OPT-025] 该节点已能用有效凭证成功调用 -> 从全环失效集合中摘除
            try:
                self.credential_watch.note_success(provider_key)
            except Exception:  # noqa: BLE001
                pass
            # [OPT-026] 节点已恢复正常：指纹记录失去意义，清除以免日后误判为「已续期」
            try:
                self.renewal.clear_provider(provider_key)
            except Exception:  # noqa: BLE001
                pass
            self._failure_counts[provider_key] = 0
            self._cooldowns.pop(provider_key, None)
            self._cooldown_severity.pop(provider_key, None)
            # [OPT-027] 节点已用当前凭证成功 -> 「新凭证仍被拒」的证据被证伪，清除。
            self._fresh_credential_rejected.pop(provider_key, None)
            self._auto_lease_fingerprint.pop(provider_key, None)
            self._persist_state()

    # -------------------------------------------------------- [OPT-027] 账号级封禁推断

    def is_ban_inference_disabled(self) -> bool:
        """[OPT-027] 是否关闭「新凭证仍被拒 -> 账号级封禁」推断。

        与租约/台账同约定关闭：本机制的唯一产物是自动租约，若租约被关闭，
        推断出结论也无处安放，继续计数只会制造无意义的日志。
        """
        if os.getenv("GATEWAY_DISABLE_BAN_INFERENCE", "").strip().lower() in ("1", "true", "yes"):
            return True
        for name in ("GATEWAY_DISABLE_QUARANTINE", "GATEWAY_DISABLE_RISK_LEDGER"):
            if os.getenv(name, "").strip().lower() in ("1", "true", "yes"):
                return True
        return False

    def _ban_inference_min_failures(self) -> int:
        try:
            return max(1, int(os.getenv("GATEWAY_BAN_INFERENCE_MIN_FAILURES", "2")))
        except ValueError:
            return 2

    def _auto_quarantine_seconds(self) -> float:
        try:
            return max(60.0, float(os.getenv("GATEWAY_AUTO_QUARANTINE_DURATION", "259200")))
        except ValueError:
            return 259200.0

    # ---------------------------------------------------- [OPT-029] 人机验证长效避让

    def _waf_escalation_min_failures(self) -> int:
        """重复多少次人机验证拦截后升级为长效避让（默认 3）。"""
        try:
            return max(1, int(os.getenv("GATEWAY_WAF_ESCALATION_MIN_FAILURES", "3")))
        except ValueError:
            return 3

    def cooldown_cause(self, provider_key: str) -> str:
        """[OPT-029] 返回该节点**当前冷却的成因**（无冷却时返回空串）。

        公开的只读访问器：档案与运维提示必须能区分「频控短暂退避，无需干预」与
        「人机验证，必须由人工完成验证」，否则运维会照着「无需人工介入」的建议
        去处理一个只有人能解决的节点。
        """
        if self.get_remaining_cooldown(provider_key) <= 0.0:
            return ""
        return self._effective_cooldown_severity(provider_key)

    def _note_renewal_rejection(self, provider_key: str, renewed_at_failure: bool) -> None:
        """[OPT-027] 累加「新凭证仍被拒」证据，达标即推断账号级封禁。

        为什么这条证据是决定性的：401 只有两种解释——凭证过期，或账号级封禁。
        凭证一旦（已证明地）更新过仍被 401 拒绝，第一种解释就被排除，第二种成立。
        而两者的正确处置完全相反：前者要重登（`login.py`），后者重登彻底无效——
        封禁绑定的是账号而不是令牌，重登只是又一次触碰被封账号。
        因此这里不再等待人工判断，直接施加**自动租约**停止一切触碰。
        """
        if self.is_ban_inference_disabled():
            return
        if renewed_at_failure:
            # 出现了一枚**更新**的凭证并被拒：此前针对旧凭证积累的证据就此作废，
            # 新凭证理应获得属于自己的机会，因此重置为 1 而非累加。
            self._fresh_credential_rejected[provider_key] = 1
            self._ban_stats["rejections_after_renewal"] += 1
        elif provider_key in self._fresh_credential_rejected:
            # 已处于「新凭证被拒」状态：本次失败发生在同一枚已更新的凭证上，
            # 是同一份证据的延续，计数必须继续累加而不是归零。
            self._fresh_credential_rejected[provider_key] += 1
        else:
            # 普通凭证过期：不进入封禁推断路径
            return
        self._maybe_apply_ban_lease(provider_key)

    def _maybe_apply_ban_lease(self, provider_key: str) -> bool:
        """[OPT-027] 证据达阈值即施加**自动**租约（仅当当前没有任何租约时）。"""
        count = self._fresh_credential_rejected.get(provider_key, 0)
        if count < self._ban_inference_min_failures():
            return False
        try:
            if self.quarantine.source(provider_key):
                # 已持有租约（自动或运维）：封禁已被避让，无需重复施加。
                # 运维租约的语义更高，绝不被自动标记覆盖或改写。
                return False
        except Exception:  # noqa: BLE001 - 查询失败按「已有租约」处理（保守）
            return False
        try:
            current = self.renewal.current_fingerprint(provider_key)
        except Exception:  # noqa: BLE001
            current = None
        duration = self._auto_quarantine_seconds()
        reason = (
            f"auto:ban-inferred 新凭证仍被拒 {count} 次 -> 账号级封禁推断；"
            "重登已被证伪（封禁绑定账号而非令牌），请勿重试。"
            "可用 DELETE /v1/admin/quarantine 解除，或 GATEWAY_DISABLE_BAN_INFERENCE=1 关闭推断"
        )
        try:
            self.quarantine.set_lease(
                provider_key, duration_seconds=duration, reason=reason, source="auto"
            )
        except Exception:  # noqa: BLE001
            return False
        if current:
            self._auto_lease_fingerprint[provider_key] = current
        self._ban_stats["auto_leases_applied"] += 1
        logger.error(
            "[封禁推断] 节点 [%s] 在**已更新的凭证**上仍连续 %d 次 401 —— 凭证过期这一解释已被排除，"
            "判定为账号级封禁。已施加 %.1f 小时自动租约，停止一切上游触碰（对该类封禁重登无效）。",
            provider_key, count, duration / 3600.0,
        )
        return True

    def _lift_stale_auto_lease(self, provider_key: str) -> bool:
        """[OPT-027] 若该节点的**自动**租约所依据的凭证已被再次更新，则解除它。

        自动租约的正当性完全来自「我们已经试过一次新凭证且仍被拒」。若之后又出现
        更新的凭证（运维执行了第二次重登），这条证据只针对**旧**凭证，自动租约便失去
        依据——必须解除并让新凭证重新接受验证，否则一次错误的推断会永久冻结一个
        其实已经修好的节点。运维租约（人类声明）永不走这条路径。
        """
        try:
            if self.quarantine.source(provider_key) != "auto":
                # 无租约或已被运维接管：自动指纹记录已无意义
                self._auto_lease_fingerprint.pop(provider_key, None)
                self._fresh_credential_rejected.pop(provider_key, None)
                return False
            expected = self._auto_lease_fingerprint.get(provider_key)
            try:
                current = self.renewal.current_fingerprint(provider_key)
            except Exception:  # noqa: BLE001
                current = None
            if not expected or current is None or current == expected:
                return False  # 没有「更新的凭证」证据 -> 保守：维持租约
            if not self.quarantine.release_auto_lease(provider_key, time.time()):
                return False
        except Exception:  # noqa: BLE001 - 任何异常都不改变既有状态（保守）
            return False
        self._auto_lease_fingerprint.pop(provider_key, None)
        self._fresh_credential_rejected.pop(provider_key, None)
        self._ban_stats["auto_leases_lifted"] += 1
        logger.warning(
            "[封禁推断] 节点 [%s] 出现更新的凭证，自动租约的推断依据失效，已解除并重新验证",
            provider_key,
        )
        # 解除自动租约同样是「解封」：进入恢复观察期，避免需求瞬间涌出
        self._enter_probation(provider_key, time.time())
        return True

    def ban_inference_report(self, now: Optional[float] = None) -> Dict[str, Any]:
        """[OPT-027] 「新凭证仍被拒」证据与自动租约的诊断视图。"""
        now = time.time() if now is None else now
        auto_leases: Dict[str, Any] = {}
        for pk, fp in self._auto_lease_fingerprint.items():
            try:
                source = self.quarantine.source(pk)
                remaining = self.quarantine.remaining(pk, now)
            except Exception:  # noqa: BLE001
                source, remaining = "", 0.0
            auto_leases[pk] = {
                "lease_source": source,
                "remaining_hours": round(remaining / 3600.0, 2),
                "fingerprint_at_inference": fp[:12],
            }
        return {
            "enabled": not self.is_ban_inference_disabled(),
            "min_failures": self._ban_inference_min_failures(),
            "auto_lease_hours": round(self._auto_quarantine_seconds() / 3600.0, 2),
            "rejections_after_renewal": {k: v for k, v in self._fresh_credential_rejected.items()},
            "auto_leases": auto_leases,
            "stats": dict(self._ban_stats),
            "hint": (
                "这些节点在**已更新的凭证**上仍被 401 拒绝：凭证过期已被排除，判定为账号级封禁。"
                "重登对该类封禁无效（封禁绑定账号而非令牌），网关已自动停止触碰；"
                "若已确认平台解除封禁，可用 DELETE /v1/admin/quarantine 提前解除自动租约。"
            ) if auto_leases else "",
        }

    # -------------------------------------------------------- [OPT-026] 认证级避让复位

    def _effective_cooldown_severity(self, provider_key: str) -> str:
        """[OPT-026] 返回该节点冷却的成因严重度。

        优先用随冷却一并持久化的记录；记录缺失时（本机制上线之前就已存在的冷却，
        例如真实台账里那五条 24h 冻结）退化为**只读**查询风控台账的最近一次拦截成因。
        没有这个退化，运维照档案建议重登之后仍然无法复位，机制形同虚设。
        台账查询同样以「最近一次拦截」判定，因此成因限定并未放松。
        """
        sev = self._cooldown_severity.get(provider_key)
        if sev:
            return sev
        try:
            recent = self.risk_ledger.recent_interceptions(hours=24.0)
            bucket = (recent.get("per_provider") or {}).get(provider_key) or {}
            return str(bucket.get("last_severity") or "")
        except Exception:  # noqa: BLE001 - 台账不可读则不改变判定（保守）
            return ""

    def release_auth_avoidance_locked(
        self,
        providers,
        reason: str = "credential-renewal",
        causes: Optional[Sequence[str]] = None,
    ) -> List[str]:
        """[OPT-026] 复位由**凭证成因**导致的避让状态（调用方须持 `self._lock`，
        或确保当前无并发访问——启动期单线程调用即属此列）。

        为什么需要它：`record_failure` 对认证失效按 2 的幂次长效升级，真实台账里
        五个节点各 54~87 次失败，全部撞到 24h 上限。这个惩罚针对的是「拿着同一个
        失效凭证反复打」；凭证一旦续期，惩罚就失去了对象——继续冻结只会锁死已修好的
        系统（`_deferred_refresh_after_cooldown` 甚至会把令牌静默刷新一并推迟 24h）。

        严格限定作用域（fail-closed，任何不确定都选择继续避让）：
        1. 持有运维封禁租约的节点**永不**释放——租约表达账号级封禁，重登无效；
        2. 仅当最近一次失败的严重度**属于 `causes`** 时释放——若之后又发生其它类
           拦截，说明是另一类问题，绝不代它释放；
        3. 无实际冷却与失败计数可清者跳过，避免平白引入观察期。
        释放后节点进入恢复观察期 (OPT-022)：串行探路 + 放大步调，防止被压制的
        需求在凭证刚更新时集中涌出。

        [OPT-029] `causes` 使**成因**成为显式参数：默认仅 `auth`，而「凭证确已更新」
        这一触发条件（`release_renewed_credentials`）对 `waf` 同样成立——人工执行
        `login.py` 完成滑块验证后令牌必然刷新，指纹变化即「人类已介入」的证据。
        若不把 `waf` 纳入，人机验证的长效避让就没有出口，会退化成永久冻结。
        `ratelimit` 永远不在可释放之列：它本就是瞬时可自愈的，重登毫无意义。
        """
        allowed = tuple(causes) if causes else (SEVERITY_AUTH,)
        now = time.time()
        released: List[str] = []
        lifted: List[str] = []
        for raw in providers or []:
            provider_key = (raw or "").strip().lower()
            if not provider_key:
                continue
            # 0. [OPT-027] 先处理**自动**租约：若其推断依据（当初那枚凭证）已被再次
            #    更新，租约理由不再成立，必须先行解除，新凭证才能接受验证。
            #    运维租约不在此路径内（`_lift_stale_auto_lease` 自行判定来源）。
            if self._lift_stale_auto_lease(provider_key):
                lifted.append(provider_key)
            # 1. 租约是账号级封禁的表达，与「凭证过期」无关，绝不因续期释放
            try:
                if self.quarantine.remaining(provider_key, now) > 0.0:
                    logger.warning(
                        "[凭证续期] 节点 [%s] 持有运维封禁租约，拒绝复位认证级避让", provider_key
                    )
                    continue
            except Exception:  # noqa: BLE001 - 查询失败按「有租约」处理（保守）
                continue
            # 2. 只有 `causes` 指定的成因才可被复位
            if self._effective_cooldown_severity(provider_key) not in allowed:
                continue
            # 3. 确有待复位状态
            has_cooldown = self._cooldowns.get(provider_key, 0.0) > now
            if not has_cooldown and self._failure_counts.get(provider_key, 0) <= 0:
                continue

            self._cooldowns.pop(provider_key, None)
            self._cooldown_severity.pop(provider_key, None)
            self._failure_counts[provider_key] = 0
            # 注意：**不**清除凭证指纹记录。释放只是「假设凭证已修好」，而不是
            # 「确认修好」——指纹记录必须保留，否则下一次失败的比对基准被抹掉，
            # OPT-027 赖以判定的「本次失败是否发生在一枚更新的凭证上」将永远为假，
            # 封禁推断会彻底失效（这正是修复过程中被测试抓到的一处逻辑反转）。
            try:
                self.credential_watch.note_success(provider_key)
            except Exception:  # noqa: BLE001
                pass
            # 释放即进入恢复观察期：解封瞬间的并发洪峰同样适用于「凭证刚续期」
            self._enter_probation(provider_key, now)
            released.append(provider_key)
            logger.warning(
                "[凭证续期] 节点 [%s] 认证级避让已复位 (原因: %s)：请以串行探路方式重新进入",
                provider_key, reason,
            )
        if released or lifted:
            self._persist_state()
        return released

    async def release_auth_avoidance(
        self,
        providers,
        reason: str = "credential-renewal",
        causes: Optional[Sequence[str]] = None,
    ) -> List[str]:
        """[OPT-026] 持锁版本的凭证成因避让复位（供运行期管理接口调用）。

        [OPT-029] `causes` 默认仅 `auth`；显式点名某节点的管理接口可传入
        `(SEVERITY_AUTH, SEVERITY_WAF)`，因为人工为某个节点完成 H 验证同样是
        「凭证成因」被消除的直接证据。
        """
        async with self._lock:
            return self.release_auth_avoidance_locked(providers, reason, causes)

    def renewed_providers(self) -> List[str]:
        """[OPT-026] 凭证文件确已更新的节点（只读判定）。"""
        try:
            return self.renewal.renewed_providers()
        except Exception:  # noqa: BLE001 - 判定失败绝不阻断任何流程
            return []

    def release_renewed_credentials_now(self, providers=None, reason: str = "credential-renewal") -> List[str]:
        """[OPT-026] 同步复位入口：用于启动期（无并发）或已在锁内的调用方。

        `providers` 为 None 时依据凭证指纹自动判定；显式给出时跳过判定直接复位
        （覆盖环境变量式凭证等无法取指纹的部署）。

        [OPT-029] 触发条件是「凭证**确已更新**」，该事实对 `auth` 与 `waf` 同时成立
        （人工完成滑块验证后令牌同样会刷新），因此两者都可被复位；`ratelimit` 永不。
        """
        targets = list(providers) if providers is not None else self.renewed_providers()
        if not targets:
            return []
        return self.release_auth_avoidance_locked(targets, reason, _RELEASABLE_BY_RENEWAL)

    async def release_renewed_credentials(self, providers=None, reason: str = "credential-renewal") -> List[str]:
        """[OPT-026] 运行期入口：自动判定（或指定）并复位凭证成因的避让。"""
        targets = list(providers) if providers is not None else self.renewed_providers()
        if not targets:
            return []
        async with self._lock:
            return self.release_auth_avoidance_locked(targets, reason, _RELEASABLE_BY_RENEWAL)

    def compute_pacing_interval(self, provider_key: str, jitter: float = 0.0) -> float:
        """计算节点下一次调用前应保持的最小间隔（秒，不含抖动时传入 jitter=0）。

        [OPT-022] 因子乘积：基础间隔 × 证据驱动放大倍率 × 观察期放大倍率。
        [OPT-024] 再乘全局系统性压制倍率：当多个**不同**账号在同一窗口内被拦时，
        根因在我们的客户端足迹而非各账号自身，此时逐节点避让无法治本，必须整体降速。
        无关联证据时该倍率为 1.0，对既有步调零影响。
        抽成纯函数以便确定性测试，避免以计时断言验证步调。
        """
        meta = PROVIDER_RISK_SPECS.get(provider_key)
        if not meta:
            return 0.0
        return (
            meta.min_call_interval
            * get_adaptive_pacing_multiplier(provider_key)
            * self.probation_pacing_multiplier(provider_key)
            * _global_risk_pacing_multiplier()
            + jitter
        )

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
        # [OPT-022] 观察期内放大调用间隔：避免解封瞬间以常规速率涌入而立即二次触发风控
        target_interval = self.compute_pacing_interval(provider_key, jitter)

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
        # [OPT-026] 冷却成因记录一并重置（否则会与后续用例的冷却串扰）
        self._cooldown_severity.clear()
        # [OPT-027] 封禁推断证据一并重置，确保用例之间零串扰
        self._fresh_credential_rejected.clear()
        self._auto_lease_fingerprint.clear()
        self._ban_stats = {
            "rejections_after_renewal": 0,
            "auto_leases_applied": 0,
            "auto_leases_lifted": 0,
        }
        # [OPT-022] 观察期状态一并重置，确保用例之间零串扰
        self._episode_ends.clear()
        self._probation_successes.clear()
        self._probation_locks.clear()
        self._probation_stats = {
            "entered": 0,
            "graduated": 0,
            "canary_serialized": 0,
            "canary_successes": 0,
        }
        disabled_str = os.getenv("DISABLED_PROVIDERS", "")
        self.disabled_providers = {
            p.strip().lower() for p in disabled_str.split(",") if p.strip()
        }
        # 同步清除持久化状态，避免内存已重置而磁盘仍残留旧冷却
        try:
            self.risk_ledger.save_state({"cooldowns": {}, "failure_counts": {}})
        except Exception:  # noqa: BLE001
            pass


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
                    # [OPT-014] 依据近期风控拦截证据自适应放大调用间隔
                    # [OPT-024] 再乘全局系统性压制倍率（无关联证据时为 1.0）
                    mult = get_adaptive_pacing_multiplier(pk) * _global_risk_pacing_multiplier()
                    async with self._lock:
                        now = time.time()
                        last = self._last_call_times.get(pk, 0.0)
                        elapsed = now - last
                        jitter = random.uniform(meta.jitter_range[0], meta.jitter_range[1])
                        target_interval = meta.min_call_interval * mult + jitter
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
                "effective_interval_s": round(
                    (meta.min_call_interval if meta else 0.1)
                    * get_adaptive_pacing_multiplier(pk)
                    * _global_risk_pacing_multiplier(), 4
                ),
                "pacing_multiplier": get_adaptive_pacing_multiplier(pk),
                "global_risk_multiplier": _global_risk_pacing_multiplier(),
                "jitter_s": list(meta.jitter_range) if meta else [0.01, 0.05],
                "risk_level": meta.risk_level.value if meta else "medium",
            }
        return {
            "active_semaphores": active_semaphores,
            "providers": providers_info,
            "adaptive_pacing": os.getenv("GATEWAY_DISABLE_ADAPTIVE_PACING", "").strip().lower()
            not in ("1", "true", "yes"),
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
        include_quarantined: bool = False,
    ) -> List[FailoverCandidate]:
        """根据当前主选节点生成同等能力环内的有序候选节点列表。

        排序策略：
        1. 排除当前排除集（已尝试失败过的 Provider 以及被主动禁用的 Provider）
        2. [OPT-021] 排除处于运维长期封禁租约的节点（硬阻断，除非显式放行）
        3. 健康节点排在冷却避让中节点之前
        4. 健康节点中：低风险 (LOW) > 中风险 (MEDIUM) > 高风险 (HIGH)
        5. 同等风险下，按历史失败次数升序排列

        注意软冷却节点**仍保留在候选中**（仅被沉底）：短时自动冷却属于可自愈退避，
        环内全部冷却时允许最后一搏以保可用性；而运维封禁租约意味着账号级封禁，
        触碰不会有正常返回，必须彻底排除。
        """
        ring = classify_capability_ring(primary_provider, primary_model)
        ring_members = RING_DEFINITIONS.get(ring, [])
        excluded = set(excluded_providers or ()) | {primary_provider}

        candidates: List[FailoverCandidate] = []

        # 遍历环中所有成员
        for p_key, w_model in ring_members:
            if p_key in excluded or self.cooldown_tracker.is_disabled(p_key):
                continue

            # [OPT-021] 被封禁租约锁定的账号绝不作为后备候选
            if not include_quarantined and self.cooldown_tracker.is_quarantined(p_key):
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

            # [OPT-025] 全环凭证失效快速失败：当多个不同节点在同一窗口内相继 401，
            # 说明我们没有可用凭证，故障转移不可能成功。此时继续逐个打满五个节点，
            # 恰恰是上游 WAF 眼中的「自动化凭证尝试」流量特征——风暴本身就是风控触发器。
            # 故立即拒绝派发（零上游调用），并给出重登指引；TTL 后自动放行探路。
            try:
                outage = self.cooldown_tracker.credential_watch.is_outage_active()
            except Exception:  # noqa: BLE001
                outage = False
            if outage:
                failed = self.cooldown_tracker.credential_watch.active_failures()
                self.cooldown_tracker.credential_watch.note_fast_fail()
                logger.error(
                    "[凭证失效快速失败] 整环凭证失效 (%s)，拒绝派发以避免无效凭证风暴",
                    ",".join(failed) or "-",
                )
                raise ProviderError(
                    "全部/多数上游节点的认证凭证已失效 ("
                    + (", ".join(failed) if failed else "unknown")
                    + ")，故障转移不可能成功。为避免持续制造无效凭证请求（上游会将其视为"
                    "自动化凭证尝试并可能加剧风控），本次请求已拒绝派发并且未产生任何上游调用。"
                    "请执行 `.venv/bin/python login.py all` 重新登录全部厂商。"
                    "解除后也可用 DELETE /v1/admin/credential-outage 立即清除该状态。",
                    status=503,
                    err_type="credential_outage",
                )

            # [OPT-021] 硬阻断：运维封禁租约节点在任何情况下都不得被派发真实请求。
            # 这是整条冷却门控体系的最后一道闸口 —— 即便调用方显式指定了被封禁的
            # 模型别名，也必须立即改派或快速失败，绝不能因为「环内只剩它」而破例触碰。
            if self.cooldown_tracker.is_quarantined(curr_provider_key):
                remaining = self.cooldown_tracker.quarantine.remaining(curr_provider_key)
                logger.error(
                    "[封禁硬阻断] 节点 [%s] 处于运维封禁租约 (剩余 %.1f 小时)，拒绝派发上游请求",
                    curr_provider_key,
                    remaining / 3600.0,
                )
                candidates = self.get_fallback_candidates(
                    primary_provider_key,
                    primary_wire_model,
                    excluded_providers=tried_providers,
                )
                if not candidates:
                    raise ProviderError(
                        f"目标节点 [{curr_provider_key}:{curr_wire_model}] 处于运维封禁租约"
                        f"（剩余 {remaining / 3600.0:.1f} 小时），且能力环 [{ring.value}] 内"
                        f"已无未被封禁的可用节点。为避免延长上游封禁，本次请求已拒绝派发 —— "
                        f"请使用 POST /v1/admin/quarantine 解除租约，或改用其他厂商模型。",
                        status=503,
                        err_type="quarantined",
                    )
                next_cand = candidates[0]
                logger.warning(
                    "[封禁硬阻断] 改派: [%s] -> [%s] (模型: %s)",
                    curr_provider_key,
                    next_cand.provider_key,
                    next_cand.wire_model,
                )
                curr_provider_key = next_cand.provider_key
                curr_wire_model = next_cand.wire_model
                continue

            sem = self.cooldown_tracker.get_semaphore(curr_provider_key)

            try:
                # 1. 频控规避 (Pacing & Jitter)
                await self.cooldown_tracker.apply_pacing_and_acquire(curr_provider_key)

                # 2. 实例化 Provider
                # [OPT-022] 观察期内先取串行闸门：把「解封瞬间的并发洪峰」压成逐个
                # 探路请求（闸门只覆盖握手到首包，不占用整段流式输出），且必须在
                # 信号量之前获取，避免多个请求先占住并发槽位再排队等闸门。
                probation_lock = self.cooldown_tracker.probation_lock(curr_provider_key)
                async with contextlib.AsyncExitStack() as stack:
                    if probation_lock is not None:
                        await stack.enter_async_context(probation_lock)
                    await stack.enter_async_context(sem)
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
