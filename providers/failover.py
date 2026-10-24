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
import json
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

from .base import ProviderError, StopSequenceHit
from .credential_renewal import get_credential_renewal
from .credential_watch import get_credential_watch
from .quarantine import get_quarantine_store
from .stream_completion import (
    completion_check_enabled,
    is_incomplete,
    observe as observe_completion,
)
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

# [DAY-ZERO] 「本机没有配置凭证」是配置缺口，与上面两者的**共同点**是：它们的解除
# 依据都是「凭证出现了/变了」这一条可观测事实。区别在于它**没有**可比的旧凭证
# （文件从来不存在），所以它的判定走 `.unconfigured_providers()` 这条独立证据链
# （详见 `CredentialRenewalTracker.note_unconfigured`）。
SEVERITY_UNCONFIGURED = "unconfigured"
_RELEASABLE_BY_CONFIGURED = (SEVERITY_UNCONFIGURED,)
# 凭证事实（变更或首次出现）可以解除的全部成因。
_RELEASABLE_BY_CREDENTIAL_EVENT = _RELEASABLE_BY_RENEWAL + _RELEASABLE_BY_CONFIGURED

# [BUDGET-6] **风控成因**：由上游的拦截给出、且处置方式取决于成因的那些。
# `timeout` / `truncated` 是**上游故障**（本网关自己的观测），不是风控拦截——
# 它们既不该写台账，也不该改写一个仍在生效的风控成因。
_RISK_CAUSES = (SEVERITY_AUTH, SEVERITY_RATE_LIMIT, SEVERITY_WAF)


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


# [BUDGET-3] 一次握手尝试的**最低有意义时长**（秒）。
#
# 低于它，任何真实上游都不可能返回首包（这五家的首包以秒计），于是这次尝试的结果
# 是确定的：超时。确定必败的调用没有任何收益，只有一次白打的上游凭证流量和一次
# 假的超时冷却，与整条避让线「不白白触碰上游」相反。因此它没有第二种用法：
# 份额低于地板时**不派发**，而不是按地板派发。
#
# 取 0.05s 是因为这就是原实现里那个地板值——它此前被当作「本次上限」使用（于是
# 派发了一次必败尝试）。现在同一个数字被移到了正确的判据上：低于它就别打。
_MIN_HANDSHAKE_ATTEMPT_SECONDS = 0.05


class BudgetCappedTimeout(ProviderError):
    """[BUDGET-3] 这次超时是**网关的总预算**裁出来的，不是节点的能力边界。

    为什么必须是一个独立类型而不是一句文案：`execute_chat` 的 `except` 分支会对每一次
    失败 `record_failure`（冷却 + 失败计数），而「份额 0.1s 到了」这个事实只说明
    **我们**只肯等 0.1s，不说明节点有故障——推理模型正常思考几十秒是健康状态
    （`_first_chunk_timeout` 的 docstring 写得很清楚）。把它按节点故障记账，一个健康
    节点会被冻结 60 秒，而冻结它的正是我们自己的预算决定。这正是本项目反复记录的
    那类谎报（「我们决定不打」写成「上游失败了」），只不过这次发生在 try 内部。

    因此它**继续往后故障转移**（这一次尝试确实没拿到首包，客户端等不到东西），但
    不进冷却、不增失败计数。
    """

    budget_capped = True


class BudgetExhaustedRefusal(ProviderError):
    """[BUDGET-3] 预算在**进入尝试之前**就不够——零上游调用，一次都不该发生。

    与 `BudgetCappedTimeout` 的区别是「是否真的打了上游」：这个类型表示我们根本
    没有调用任何节点，因此它既不是节点失败，也不该继续往后故障转移（后面的候选
    同样拿不到份额）。抛出后派发循环立刻终止。
    """

    zero_upstream = True


def _is_budget_capped(exc: BaseException) -> bool:
    return bool(getattr(exc, "budget_capped", False))


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


# ---------------------------------------------------------------- [DAY-ZERO] 未配置凭证

def _provider_is_configured(provider_key: str) -> bool:
    """本机是否配置了该节点的凭证——委托给 `session_store` 的**唯一**判据。

    在本模块留一层薄封装，是为了让判定失败的方向明确落在「保守」一侧：任何异常都
    按「已配置」处理（`session_store.is_configured` 已如此实现），于是真实的 401
    绝不会因为一次读取异常而逃过认证级避让。若反过来把「读不到」当成「未配置」，
    一次磁盘故障就能让整环停止升级认证避让——风控线的失败方向必须朝向「继续避让」。
    """
    try:
        from session_store import is_configured

        return bool(is_configured(provider_key))
    except Exception:  # noqa: BLE001 - 判定不可用时不改变既有语义
        return True


def _unconfigured_cooldown() -> float:
    """未配置凭证的固定短冷却（秒，默认 60，`GATEWAY_UNCONFIGURED_COOLDOWN`）。

    固定而不随次数增长，是有意的：这一档的作用是「别把它当候选反复派发」，不是惩罚。
    未配置不是错误行为，只是一个还没完成的前置步骤；真正让它恢复的信号是凭证出现，
    而不是时间流逝或惩罚累积。
    """
    try:
        return max(1.0, float(os.getenv("GATEWAY_UNCONFIGURED_COOLDOWN", "60")))
    except (TypeError, ValueError):
        return 60.0


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
        # [M1] provider -> 已预约的「下一次可调用时刻」（节奏预约，见
        # apply_pacing_and_acquire）。与 `_last_call_time` 分工不同：前者是
        # **预测**（谁排在什么时刻），后者是**事实**（真正放行在什么时刻）。
        self._next_slot: Dict[str, float] = {}
        # [M1] provider -> 最近一次**实际施加**的节奏等待毫秒数。
        # 节奏裁决唯一化之后，响应头 x-risk-pacing-delay-ms 必须报告这里记录的真实
        # 等待，而不是由调用方另起一套计时自己算一个「大概等了多少」。
        self._last_pacing_wait_ms: Dict[str, float] = {}
        # provider -> asyncio.Semaphore 并发控制
        self._semaphores: Dict[str, asyncio.Semaphore] = {
            pk: asyncio.Semaphore(meta.max_concurrency)
            for pk, meta in PROVIDER_RISK_SPECS.items()
        }
        # 保护锁
        self._lock = asyncio.Lock()
        # [H5] 已落盘状态的指纹：内容未变则不重写。
        # `record_success` 在每个请求的首包成功后被调用，而健康节点的成功**不会改变
        # 任何持久化状态**（冷却为空、失败计数为 0），原实现却无条件整份重写
        # （temp 文件 + fsync + os.replace）。实测：健康节点上 20 次成功 = 20 次落盘，
        # 且每次都在 `_lock` 内阻塞事件循环——这正是「连接 agent 时抖动」的一个来源。
        self._persisted_fingerprint: Optional[str] = None
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
        # [BUDGET-4] provider -> 连续被**总预算裁短**的超时次数。
        #
        # 这是与 `_failure_counts` 分开的一本账，因为它们是两件不同的事实：
        # `_failure_counts` 回答「节点自己坏了多少次」，本表回答「节点多少次没能
        # 在我们给它的那份时间里吐首包」。前者在份额 0.1s 时是诬告（上一轮
        # `BudgetCappedTimeout` 修的就是它），后者不是——它如实描述了节点与
        # **预算份额**的关系，而这个关系正是派发决策需要的。
        #
        # 没有这本账的后果是实测出来的：五节点环、qwen 挂死、预算 5s → 首份 1.0s，
        # 连发四轮全是 1.00s、`qwen_calls` 单调增长、冷却与失败计数恒 0。长会话的
        # agent 每次请求都稳定多等一秒且永不收敛，而且每次都对一个已知不吐字的节点
        # 发起了一次带凭证的真实调用。
        self._budget_capped_streak: Dict[str, int] = {}
        # [BUDGET-4] 累计升级次数，仅用于诊断可观测性。
        self._budget_capped_escalations: Dict[str, int] = {}
        # [OPT-034] 后台触点避让门控的拒绝直方图（按成因）
        self._touch_block_stats: Dict[str, int] = {}
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
        # [OPT-031] 长效避让门控统计：整环长效避让时的零上游快速失败，以及把长效避让
        # 节点改派给健康节点的次数。两者都是「没有触碰被封节点」的证据，必须可观测。
        self._long_avoidance_stats: Dict[str, Any] = {
            "fast_fails": 0,
            "redirects": 0,
            "last_fast_fail_provider": "",
            "last_fast_fail_cause": "",
            "last_redirect": "",
        }
        # [OPT-033] 禁用门控统计：`DISABLED_PROVIDERS` 是运维关于「这个节点不属于本次
        # 部署」的显式声明，此前派发边界并不理会它（只有负载均衡与会话亲和会避让），
        # 于是「绕开负载均衡的那条路」可以把请求送进被显式禁用的节点。本统计即为
        # 「被禁用的节点确实没被触碰」这条声明的可观测证据。
        self._disabled_gate_stats: Dict[str, Any] = {
            "fast_fails": 0,
            "redirects": 0,
            "last_fast_fail_provider": "",
            "last_redirect": "",
        }
        # 闸门拒绝计数：blocked_by -> 次数。使「哪一道闸门在承担拒载」可观测，
        # 而不是只能从日志里翻。
        self._dispatch_gate_stats: Dict[str, int] = {
            "credential_outage": 0,
            "disabled": 0,
            "quarantine": 0,
            "long_avoidance": 0,
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
            # [H5] 磁盘上没有状态 = 空状态与内存一致（「没有需要避让的节点」）。
            # 播种指纹，使进程启动后的第一批健康请求不会只为写一个空文件而 fsync。
            self._persisted_fingerprint = self._snapshot_fingerprint()
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
        # [BUDGET-4] 恢复「连续被预算裁短」的证据与累计升级次数。与上面两本账同理：
        # 能被一次重启清零的连续计数等于不存在。
        for key, target in (
            ("budget_capped_streak", self._budget_capped_streak),
            ("budget_capped_escalations", self._budget_capped_escalations),
        ):
            raw = state.get(key) or {}
            if isinstance(raw, dict):
                for pk, n in raw.items():
                    if isinstance(n, int) and n > 0:
                        target[str(pk)] = n
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
        # [H5] 若恢复后的状态与磁盘上的完全一致（没有过期冷却被丢弃、没有零计数被
        # 剪掉），则磁盘已经持有这份内容——直接播种指纹，使重启后的第一批健康请求
        # 不再把同一份状态原样重写一遍。
        try:
            restored_fingerprint = self._snapshot_fingerprint()
            if json.dumps(state, sort_keys=True, default=str) == restored_fingerprint:
                self._persisted_fingerprint = restored_fingerprint
        except Exception:  # noqa: BLE001 - 播种失败只意味着多写一次，不影响正确性
            pass

    def _snapshot_fingerprint(self) -> str:
        return json.dumps(self._state_snapshot(), sort_keys=True, default=str)

    def _state_snapshot(self) -> Dict[str, Any]:
        """当前需要跨重启存活的风控状态。

        抽取成单一实现，是为了让「写盘内容」与「恢复后判定磁盘是否已持有同一份
        内容」（[H5] 的指纹播种）共用一份定义：两处各写一遍必然漂移，那正是本项目
        反复修复的缺陷类型。
        """
        return {
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
            # [BUDGET-4] 「连续 N 次被预算裁短」同样必须跨界存活：一次重启若能清零
            # 它，就等于给了一个挂死的节点重新被白等 N 次的机会，而这正是本机制的
            # 全部意义所在。只存正数，保持「健康」与「缺席」同构（[H5] 的同一条纪律）。
            "budget_capped_streak": {
                k: v for k, v in self._budget_capped_streak.items() if v > 0
            },
            "budget_capped_escalations": {
                k: v for k, v in self._budget_capped_escalations.items() if v > 0
            },
        }

    def _persist_state(self) -> None:
        """持久化冷却状态与失败计数（best-effort，绝不上抛）。

        [H5] 内容未变时不写盘。本方法在两个**每请求**路径上被调用
        （`record_success` 的首包成功、`record_failure` 的失败），而多数请求既不改
        冷却也不改失败计数——无条件重写意味着把 temp 文件 + fsync + os.replace
        放进了每一次正常请求里，且在 `_lock` 内阻塞事件循环。落盘是「为崩溃保留
        证据」，对一份与磁盘上逐字节相同的状态重复写入没有任何收益。
        """
        try:
            state = self._state_snapshot()
            fingerprint = self._snapshot_fingerprint()
            if fingerprint == self._persisted_fingerprint:
                return
            if self.risk_ledger.save_state(state):
                self._persisted_fingerprint = fingerprint
        except Exception as exc:  # noqa: BLE001
            # [H5] 这里曾经是裸 `pass`。后果不是「少了一次写盘」，而是**整条持久化
            # 链路静默失效且完全不可见**：一次 `NameError`（`json` 未导入）就让冷却
            # 状态再也不落盘，而日志里没有任何一行提示——正是本项目反复记录的
            # 「文档/埋点声称做了、代码其实没做」的缺陷。best-effort 不等于无声失败。
            logger.warning(
                "[风控持久化] 冷却状态写盘失败 (已忽略本次): %s: %s",
                type(exc).__name__,
                exc,
            )

    def is_disabled(self, provider_key: str) -> bool:
        """[OPT-033] 检查指定 Provider 是否被运维显式禁用。

        这是「该节点是否被禁用」的**唯一**判读处：派发闸门、候选过滤、负载均衡、会话亲和
        的可用性检查全部经由本方法。此前 `server._is_provider_healthy` 另行现读一次
        `DISABLED_PROVIDERS` 环境变量，而本类在构造/reset 时把同一份配置快照成集合——
        同一份配置被两处各自解读，进程内改配置时两者会给出不同答案（本项目反复出现的
        缺陷类型）。

        判读 = **程序集合 ∪ 环境变量现读**：
        - 集合承载程序化声明（测试、以及任何不经进程重启的运行时变更），并由 `reset()`
          用当时的环境变量刷新；
        - 环境变量现读保证「改了 `DISABLED_PROVIDERS` 就立刻生效」，与旧行为逐值一致
          （旧 `_is_provider_healthy` 就是现读环境变量的，改成快照会让既有行为退化）。

        两者取并集而不是二选一：禁用是**声明**，多一条声明只会更保守，不会更激进——
        这正是这类运维声明应有的合并方向（对照 OPT-020 租约只取 max() 的同一考虑）。
        """
        key = (provider_key or "").lower()
        if key in self.disabled_providers:
            return True
        raw = os.getenv("DISABLED_PROVIDERS", "")
        if not raw:
            return False
        return key in {p.strip().lower() for p in raw.split(",") if p.strip()}

    def is_in_cooldown(self, provider_key: str) -> bool:
        """检查指定 Provider 是否处于冷却避让期（含运维封禁租约）"""
        return self.get_remaining_cooldown(provider_key) > 0.0

    # ---------------------------------------------------- [OPT-034] 后台触点判据

    def touch_avoidance_reason(self, provider_key: str) -> str:
        """[OPT-034] 后台自维护触点（探针/令牌刷新/会话回收/自愈选举）的**唯一**判据。

        返回 `""` 表示可以触碰上游；否则返回成因，取值 `disabled` / `quarantined` /
        `cooldown` / `unconfigured`（heaviest-wins，顺序即优先级）。

        为什么需要这一条：派发边界（OPT-033）与健康探针（OPT-012）都遵守「运维禁用」，
        但**另外三条自维护通道只问了冷却、没问禁用**——一个被 `DISABLED_PROVIDERS`
        声明出局的节点，只要它恰好没有冷却，网关仍会对它发起真实的令牌刷新、真实的
        临时会话删除、以及自愈选举中的真实上游补全。实测复现：`DISABLED_PROVIDERS=kimi`
        且无冷却时，`_refresh_stale_provider("kimi", …)` 真的调用了 refresher
        （打印「kimi 令牌已超过 6h，尝试静默刷新…」）。这与 OPT-020（租约折入冷却读数，
        使所有门控自动遵循）同属一类问题：一条避让声明只在部分通道生效。

        语义分层（与 OPT-033 的派发闸门同序，但**更严**）：运维禁用是部署声明，排在最前；
        其次是运维/自动租约；最后是自动冷却。与派发闸门的唯一差别是**短软冷却**：派发
        保留「最后一搏」，而后台自维护永远不值得为一次冷却中的节点冒险，故此处一律阻断。

        成因必须如实返回而不是笼统的「冷却中」：禁用的解除方式是改配置，租约的解除方式
        是人工撤租，冷却的解除方式是等待——把三者混成一句话正是本项目反复修复的
        「误导运维」缺陷类型（OPT-026/027/029/031）。

        判定为非空即在此记录一次拒载（`note_touch_block`）：各消费方**只在决定是否触碰
        上游的那一刻**调用本方法，因此直方图读作「被拦下的触碰次数」而非节点数。记录写在
        判据内，是为了让日后新增的消费方自动获得遥测，而不是每加一处就漏记一处。
        """
        if self.is_disabled(provider_key):
            self.note_touch_block("disabled")
            return "disabled"
        # 用本类自己的 is_quarantined（与派发闸门对租约的判定完全同源），
        # 而不是直接问 quarantine store——后者没有这个谓词，写错会被下面的
        # except 静默吞掉，把租约误报成「冷却中」（正是本条要消除的误导）。
        try:
            if self.is_quarantined(provider_key):
                self.note_touch_block("quarantined")
                return "quarantined"
        except Exception:  # noqa: BLE001 - 租约查询失败回退到冷却判读，绝不误放行
            pass
        if self.get_remaining_cooldown(provider_key) > 0.0:
            # [DAY-ZERO] 成因如实：一次「本机没有凭证」不是风控冷却。若笼统报
            # `cooldown`，探针日志就会写「避免加剧上游封禁」，而该账号从未被触碰过——
            # 与 OPT-026/027/029/031 反复修复的误导类型同形，只是成因换成了配置缺口。
            # 禁用/租约仍优先于它（那两者是更重的声明）。
            if not _provider_is_configured(provider_key):
                self.note_touch_block("unconfigured")
                return "unconfigured"
            self.note_touch_block("cooldown")
            return "cooldown"
        return ""

    def note_touch_block(self, cause: str) -> None:
        """[OPT-034] 记一次后台触点被避让声明拦下（按成因分类）。"""
        if not cause:
            return
        self._touch_block_stats[cause] = self._touch_block_stats.get(cause, 0) + 1

    def touch_gate_report(self) -> Dict[str, Any]:
        """[OPT-034] 后台触点避让门控诊断。"""
        return {
            "blocked_by_cause": dict(self._touch_block_stats),
            "causes": ["disabled", "quarantined", "cooldown"],
        }

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

    # ---------------------------------------------------------------- [OPT-031] 长效避让
    @staticmethod
    def _env_float(name: str, default: float) -> float:
        try:
            return float(os.getenv(name, str(default)))
        except ValueError:
            return default

    def long_avoidance_threshold(self) -> float:
        """[OPT-031] 「短冷却」与「长效避让」的分界（秒，默认 1200）。

        这个阈值不是调出来的经验值，而是一条**真空带**，两端都由代码常量给出：
        - 上界（短冷却可达上界）：频控与人机验证的短冷却被 `GATEWAY_WAF_COOLDOWN_MAX`
          （默认 900）封顶，指数也在第 4 次封顶（base * 1.2^3，实测上界 deepseek
          60s -> 103.7s，见 `tests/test_waf_escalation.py`）；
        - 下界（升级下界）：认证级长效底数 `GATEWAY_AUTH_COOLDOWN_BASE`（默认 1800），
          人机验证升级底数更高（3600，见 OPT-029）。

        真空带即 (900, 1800)，默认值 1200 落在正中。**夹逼是强制的**：误配既不能侵入
        短冷却（那会吃掉 OPT-021 为「环内全部冷却」刻意保留的最后一搏），也不能等于
        或超过升级下界（那会让升级后的节点重新被派发，风暴复现）。因此实际取值恒满足
        `短冷却上界 < 阈值 < 升级下界`——两侧都是**严格**不等，各留 1 秒余量，避免
        「读取时刻略晚于写入时刻」把边界判定翻个面（这正是把阈值取成 1800 时的实测
        缺陷：认证升级恰好是 1800，读到的剩余时长已是 1799.99，判定立刻翻回「非长效」）。
        """
        short_cap = self._env_float("GATEWAY_WAF_COOLDOWN_MAX", 900.0)
        escal_floor = self._env_float("GATEWAY_AUTH_COOLDOWN_BASE", 1800.0)
        configured = self._env_float("GATEWAY_LONG_AVOIDANCE_THRESHOLD", 1200.0)
        return min(max(short_cap + 1.0, configured), escal_floor - 1.0)

    def is_long_avoidance_disabled(self) -> bool:
        """[OPT-031] 总开关：禁用后回退到「逐个走遍能力环」的旧行为。"""
        return os.getenv("GATEWAY_DISABLE_LONG_AVOIDANCE_FAST_FAIL", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )

    def is_in_long_avoidance(self, provider_key: str) -> bool:
        """[OPT-031] 该节点是否处于**长效避让**（升级后的长冷却，或任何封禁租约）。

        与 `is_in_cooldown`（软冷却）的差别**只在时长**，但处置相反：
        - 短冷却（频控退避，秒级~两分钟）是可自愈的短暂退避，值得一次最后一搏；
        - 长效避让（升级后的人机验证/认证冷却，小时~天级）意味着上游已经用**连续多次**
          拦截给出了答案，再打一发只是又一次「自动化触碰」的证据，只会延长封禁窗口。

        为什么要单独立一个语义：OPT-021 的硬阻断只覆盖运维租约，OPT-025 的快速失败只
        覆盖整环 401。当环内全部节点都处于**混合成因**的长效避让（有的 waf 升级、有的
        auth 升级）时，每条客户端请求仍会把五个节点逐个打满——这正是 OPT-025 记录的
        「无效凭证风暴」形状，只不过成因不是 401。风暴本身就是风控触发器。
        """
        if self.is_long_avoidance_disabled():
            return False
        # 运维租约无论长短一律硬阻断（与 OPT-021 一致：人工声明优先于时长）
        try:
            if self.quarantine.remaining(provider_key) > 0.0:
                return True
        except Exception:  # noqa: BLE001 - 租约查询失败不改变软冷却判定
            pass
        return self.get_remaining_cooldown(provider_key) >= self.long_avoidance_threshold()

    def is_affinity_retainable(self, provider_key: str) -> bool:
        """[M3] 会话粘滞是否仍可保留在该节点（只读，供会话亲和判定）。

        原判定只看「节点是否可用」，而可用性对**任何**软冷却都回答 False。后果是：
        一次 429 退避（15 秒）就会把一个正在进行长程 agent 会话的客户端整体迁移到
        另一个厂商——上游上下文全部重建、提示词缓存全丢、工具调用契约需要重新学习，
        而这一切只为躲开一次 15 秒的频控。跨厂商迁移的代价是**几十倍**的，为了一个
        秒级窗口付这笔钱是明确的错误取舍。

        语义与派发边界的分层保持一致（OPT-021/031 已确立）：
        - **硬阻断**（运维/自动租约、`DISABLED_PROVIDERS`、长效避让）→ 不可保留：
          这些是「不要碰这个节点」的声明，等待没有意义，必须迁移；
        - **短冷却**（频控退避、一次性人机验证）→ 保留：它数秒到两分钟即自愈，
          而迁移要付出整段上下文的代价。粘滞关系**不动**，冷却一结束就自动回到
          原节点、原上游会话；单次请求仍由派发边界按原语义处置（短冷却只降级、
          保留最后一搏，失败则在本次请求内故障转移），但会话本身不会被永久改嫁。

        只读：本方法不改变任何风控状态。
        """
        try:
            if self.is_disabled(provider_key):
                return False
        except Exception:  # noqa: BLE001 - 判定失败按不可用处理，绝不误留
            return False
        try:
            if self.is_quarantined(provider_key):
                return False
        except Exception:  # noqa: BLE001
            return False
        try:
            if self.is_in_long_avoidance(provider_key):
                return False
        except Exception:  # noqa: BLE001
            return False
        return True

    def ring_long_avoidance_snapshot(self) -> Dict[str, Dict[str, Any]]:
        """[OPT-031] 当前处于长效避让的节点及其成因/剩余时长（只读，供快速失败文案与诊断）。"""
        snapshot: Dict[str, Dict[str, Any]] = {}
        for p_key in PROVIDER_RISK_SPECS:
            if not self.is_in_long_avoidance(p_key):
                continue
            remaining = self.get_remaining_cooldown(p_key)
            snapshot[p_key] = {
                "cause": self.long_avoidance_cause(p_key),
                "remaining_seconds": round(remaining, 1),
                "remaining_hours": round(remaining / 3600.0, 2),
            }
        return snapshot

    def long_avoidance_cause(self, provider_key: str) -> str:
        """[OPT-031] 长效避让的**成因**，用于给运维正确的处置指引。

        与 `cooldown_cause()` 的差别：租约（人工或自动）在这里一律报 `ban`——租约表达的
        是账号级封禁，重登无效、只能等平台解除；若沿用软冷却的成因（例如升级前的那个
        `auth`），运维会照着「执行 login.py 重登」去处理一个重登根本无效的节点，
        正是 OPT-027/OPT-029 反复修正的同一类错误。
        """
        try:
            if self.quarantine.remaining(provider_key) > 0.0:
                return "ban"
        except Exception:  # noqa: BLE001 - 租约查询失败退回软冷却成因判定
            pass
        return self.cooldown_cause(provider_key) or "risk"

    def note_long_avoidance_fast_fail(self, provider_key: str, cause: str) -> None:
        """[OPT-031] 记一次「整环长效避让 → 零上游调用拒绝派发」。"""
        self._long_avoidance_stats["fast_fails"] += 1
        self._long_avoidance_stats["last_fast_fail_provider"] = provider_key
        self._long_avoidance_stats["last_fast_fail_cause"] = cause

    def note_long_avoidance_redirect(self, from_key: str, to_key: str) -> None:
        """[OPT-031] 记一次「长效避让节点被改派到健康节点」（未触碰避让节点）。"""
        self._long_avoidance_stats["redirects"] += 1
        self._long_avoidance_stats["last_redirect"] = f"{from_key}->{to_key}"

    def note_disabled_gate_fast_fail(self, provider_key: str) -> None:
        """[OPT-033] 记一次「环内已无未被禁用节点 → 零上游调用拒绝派发」。"""
        self._disabled_gate_stats["fast_fails"] += 1
        self._disabled_gate_stats["last_fast_fail_provider"] = provider_key

    def note_disabled_gate_redirect(self, from_key: str, to_key: str) -> None:
        """[OPT-033] 记一次「被禁用节点被改派到可用节点」（未触碰被禁用节点）。"""
        self._disabled_gate_stats["redirects"] += 1
        self._disabled_gate_stats["last_redirect"] = f"{from_key}->{to_key}"

    def note_dispatch_gate_block(self, blocked_by: str) -> None:
        """[OPT-033] 记一次派发闸门拒载（按闸门名分类），使「谁在承担拒载」可观测。"""
        if not blocked_by:
            return
        self._dispatch_gate_stats[blocked_by] = self._dispatch_gate_stats.get(blocked_by, 0) + 1

    def disabled_gate_report(self) -> Dict[str, Any]:
        """[OPT-033] 禁用门控与闸门拒载分布诊断。"""
        effective = set(self.disabled_providers)
        raw = os.getenv("DISABLED_PROVIDERS", "")
        if raw:
            effective |= {p.strip().lower() for p in raw.split(",") if p.strip()}
        return {
            "disabled_providers": sorted(effective),
            "gate_blocks": dict(self._dispatch_gate_stats),
            "stats": dict(self._disabled_gate_stats),
        }

    def long_avoidance_report(self) -> Dict[str, Any]:
        """[OPT-031] 长效避让门控诊断。"""
        return {
            "enabled": not self.is_long_avoidance_disabled(),
            "threshold_seconds": self.long_avoidance_threshold(),
            "avoiding": self.ring_long_avoidance_snapshot(),
            "stats": dict(self._long_avoidance_stats),
        }

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
        is_timeout: bool = False,
        is_truncation: bool = False,
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

            # [M2] 超时是被**显式判定**的成因，不靠解析文案去猜：握手与流式空闲
            # 边界由网关自己产生，它知道这次失败是「上游不吐字」。判定必须发生在
            # 分支选择**之前**，否则超时的错误文案会先被风控匹配器认领（分类器对
            # 任何非 401/429 的输入都缺省回 waf），落进 WAF 分支拿到 ≤45 秒的短冷却
            # ——既起不到避让作用，又给节点记下一段它从未有过的风控画像。
            is_timeout = bool(is_timeout) or status == 504

            # [S2] 截断与超时同属**上游故障**而非风控拦截，判定同样必须发生在
            # 分支选择之前：截断走的是 502，而分类器对任何非 401/429 的输入都
            # 缺省回 waf —— 实测一次截断被归因为 `waf`，于是档案会建议运维去
            # 完成一个平台从未出现过的人机验证（与 [M2] 对超时的误判同形）。
            is_truncation = bool(is_truncation)

            # [DAY-ZERO] 「本机没有配置这个节点的凭证」是**配置缺口**，不是上游的 401。
            #
            # `_make_provider` 在本地拿不到凭证时抛出 status=401 的 ProviderError，
            # 而派发循环的 except 只读 `exc.status`——于是这条从未离开本机的结论被当成
            # 上游 HTTP 401 记进了认证级避让（实测三轮之后五个节点全部 1800s，而**上游
            # 调用次数为 0**）。一次 401 与一次「我们没配置」在成因上正相反：
            # 前者的处置是重登，后者是去配置；而认证级是升级最重、解除路径最窄的一档。
            #
            # 判定必须问**唯一判据** `session_store.is_configured()`，而不是嗅探错误文案：
            # 文案会变，而「本机此刻有没有凭证」是一个可查询的事实。同一个判据也用在
            # `_make_provider` 的抛出条件与健康探针上，三处问同一个问题。
            is_unconfigured = not (is_timeout or is_truncation) and not _provider_is_configured(provider_key)
            if is_unconfigured:
                # 未配置就是未配置：它不是风控拦截，也不是上游故障，而是我们自己的
                # 配置缺口。下面所有风控副作用（台账、全环检测、封禁推断、观察期）
                # 都必须跳过——`is_upstream_fault` 是它们的统一闸门，与超时/截断同路。
                is_timeout = False
                is_truncation = False
                is_risk = False
                is_waf_or_rate = False
                is_auth_error = False

            # 「是上游故障，不是账号被标记」只定义一次：风控台账、全环凭证检测、
            # 恢复观察期三处都用它。否则每新增一种上游故障都要在四处各加一个
            # `and not ...`，正是本仓库反复在修的「同一事实散落多处」。
            is_upstream_fault = is_timeout or is_truncation or is_unconfigured

            # 严重度分级：auth（令牌/账号级）> ratelimit（频控）> waf（人机验证）
            severity = classify_interception_severity(
                is_risk=is_risk, risk_reason=risk_reason, status=status, error_msg=error_msg
            )
            escalated = False
            if is_unconfigured:
                # [DAY-ZERO] 配置缺口自成一档，**永远不升级**。
                #
                # 为什么不能借用 auth 档：认证级的解除路径是「凭证指纹变化 -> 自动复位」
                # （OPT-026），而凭证文件从来不存在时压根没有指纹记录可比，那条解除
                # 路径对它**结构上不可见**——于是 30 分钟避让一旦升级上来就没人能解除，
                # 运维照着网关自己的提示跑完 `login.py` 也依然不可用。这正是 OPT-026
                # 记录过的自锁，换了个触发条件。
                #
                # 取一段固定的短冷却（不随次数增长）：它的作用只是「别把它当候选反复
                # 派发」，而不是惩罚——未配置不是错误行为，只是还没配置。真正让它恢复
                # 的信号是「凭证出现了」，由 `release_renewed_credentials*` 处理。
                cooldown_duration = _unconfigured_cooldown()
                severity = "unconfigured"
                # 留下「此刻不存在凭证」的基线：`note_auth_failure` 在文件不存在时
                # 不写任何记录，于是续期判定永远没有比较对象——运维补上凭证之后
                # `renewed_providers()` 仍返回空，节点继续被避让，网关让运维做的事
                # 做完后没有效果。这条基线走**独立**证据链（`kind="unconfigured"`），
                # 绝不喂给 OPT-027 的封禁推断判别式（详见 `note_unconfigured`）。
                try:
                    self.renewal.note_unconfigured(provider_key)
                except Exception:  # noqa: BLE001 - 基线失败绝不影响冷却主流程
                    pass
                logger.warning(
                    f"[未配置避让] 节点 [{provider_key}] 本机没有凭证：冷却 "
                    f"{cooldown_duration:.1f}s（第 {cur_fails} 次）。这是**本地配置缺口**，"
                    f"不是上游风控——不写台账、不触发整环失效检测、不进入恢复观察期；"
                    f"补上凭证后会自动复位"
                )
            elif is_timeout:
                # 上游故障，不是风控：常规 5xx 那套 15s×n 的递增在这里方向相反
                # ——连续超时说明上游**持续**不吐字，越敲越亏；而 45 秒以内的冷却
                # 又落进「短冷却值得最后一搏」的区间，节点会一直被派发下去。
                cooldown_duration = max(30.0, base_cd)
                severity = "timeout"
                logger.warning(
                    f"[超时避让] 节点 [{provider_key}] 上游超时 (HTTP {status})："
                    f"冷却 {cooldown_duration:.1f}s (第 {cur_fails} 次连续失败)。"
                    f"这是上游故障而非风控拦截，不会写入风控台账"
                )
            elif is_truncation:
                # [S2] 回答被截断：上游连上了、说了一半就停了。这是上游故障，
                # 和超时同类处置——记冷却让它不再当主选（否则失败计数永远是 0，
                # 客户端反复看到「回复说到一半停住」），但不进风控台账、不进
                # 恢复观察期（没有封禁解除的惊群可言）。
                cooldown_duration = max(30.0, base_cd)
                severity = "truncated"
                logger.warning(
                    f"[截断避让] 节点 [{provider_key}] 的回答被截断（上游流结束但"
                    f"未出现协议完成标记）：冷却 {cooldown_duration:.1f}s "
                    f"(第 {cur_fails} 次连续失败)。这是上游故障而非风控拦截，"
                    f"不会写入风控台账"
                )
            elif severity == SEVERITY_AUTH or is_auth_error:
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
                # 普通 5xx（超时已在上面单独分支处置）
                cooldown_duration = min(base_cd, 15.0 * cur_fails)

            end_time = time.time() + cooldown_duration
            # 如果已有更长的冷却期，保留较长者
            self._cooldowns[provider_key] = max(self._cooldowns.get(provider_key, 0.0), end_time)
            # [OPT-026] 记录本次冷却的成因，供凭证续期后判定「是否可复位」。
            # [BUDGET-6] 经统一判据写入：超时/截断是上游故障，不得抹掉仍在生效的风控成因。
            self._set_cooldown_cause(provider_key, severity, end_time)

            # [OPT-025] 认证失效还需上报全环检测器：若多个**不同**节点在同一窗口内
            # 相继 401，说明整环凭证已失效，故障转移不可能成功——必须快速失败而非
            # 逐个打满五个节点（那正是上游眼中的「自动化凭证尝试」流量特征本身）。
            if not is_upstream_fault and (severity == SEVERITY_AUTH or is_auth_error):
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
            if not is_upstream_fault and (is_risk or is_waf_or_rate or is_auth_error):
                self._enter_probation(provider_key, self._cooldowns[provider_key])

            # 台账记录：仅记录真正的拦截事件（风控 / 频控 / 认证失效），
            # 普通 5xx 与超时不入账，避免污染风控画像。
            if not is_upstream_fault and (is_risk or is_waf_or_rate or is_auth_error):
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

    # ---------------------------------------------------------------- [BUDGET-4]

    def budget_capped_limit(self) -> int:
        """连续被预算裁短多少次后升级为一次真实的上游超时冷却。

        `0` 表示永不升级（回到「每次请求都白等一份份额」的旧行为，保留为开关）。
        负值按 0 处理——一个无意义的配置不该悄悄变成「每次立即冷却」。
        """
        try:
            return max(0, int(os.getenv("GATEWAY_BUDGET_CAPPED_LIMIT", "3")))
        except (TypeError, ValueError):
            return 3

    def budget_capped_count(self, provider_key: str) -> int:
        """该节点当前连续被预算裁短的次数（只读，供诊断与用例断言）。"""
        return self._budget_capped_streak.get(provider_key, 0)

    def budget_actions(self, provider_key: str) -> int:
        """该节点累计因预算裁短而升级的冷却次数（只读，仅供诊断）。"""
        return self._budget_capped_escalations.get(provider_key, 0)

    def budget_capped_report(self) -> Dict[str, Any]:
        """[BUDGET-4] 预算裁短升级机制的可观测面。

        「这个机制到底有没有在做事」必须是可观测的，而不是靠读日志推断：
        `streaks` 是各节点当前的连续计数，`escalations` 是累计升级次数。
        两者都持久化，故重启后仍能读出真实状态。
        """
        limit = self.budget_capped_limit()
        return {
            "enabled": limit > 0,
            "limit": limit,
            "streaks": {k: v for k, v in self._budget_capped_streak.items() if v > 0},
            "escalations": {
                k: v for k, v in self._budget_capped_escalations.items() if v > 0
            },
            "timeout_cooldown_seconds": self._env_float(
                "GATEWAY_TIMEOUT_COOLDOWN", 60.0
            ),
        }

    async def note_budget_capped(self, provider_key: str) -> float:
        """记一次「本候选没能在我们给它的份额里吐首包」，返回升级出的冷却时长。

        为什么这**不是**上一轮修掉的诬告：诬告是把「份额 0.1s 到了」说成
        「节点坏了」，并据此给它 60s 冷却。本方法说的是另一件事，而且是可观测的
        事实：**连续 N 次，我们把全部可用份额都给了它，它一次都没吐首包**。对一个
        真的挂死的节点，这条事实与「它坏了」在行为上等价（冷却后走健康节点）；
        对一个只是比份额慢的健康节点，它也会在冷却到期后重新被试探，且一旦成功
        `record_success` 立即把连续计数清零。

        单次绝不触发——这正是 `BudgetCappedTimeout` 存在的理由，阈值机制不会把它
        吃掉。也不进风控台账、不进恢复观察期、不写 `waf`/`auth` 成因：这是上游
        健康问题，不是账号被标记。
        """
        async with self._lock:
            limit = self.budget_capped_limit()
            if limit == 0:
                # 机制关闭：不积攒任何证据。留着一个永远无法兑现的连续计数会让
                # `budget_capped_report()` 出现「enabled: false，却有 streaks」的
                # 自相矛盾读数（关闭开关也要能被证明是关闭）。
                self._budget_capped_streak.pop(provider_key, None)
                self._persist_state()
                return 0.0
            streak = self._budget_capped_streak.get(provider_key, 0) + 1
            self._budget_capped_streak[provider_key] = streak
            if streak < limit:
                self._persist_state()
                return 0.0

            # 达到阈值：按**上游超时**计一次失败（成因 timeout）。
            #
            # [BUDGET-5] 连续计数**不清零，而是饱和在阈值上**。第一版把它清零，于是
            # 每个冷却周期都重新发放 `limit` 次完整的份额尝试：请求间隔一旦大于冷却
            # 时长（长程 agent 的两次工具调用之间往往是分钟级），节点就退化回
            # 「每次请求都白等一整份」，只是周期从「每请求」变成了「每冷却窗口」。
            # 实测（份额 1.0s、冷却 1s、轮间 1.2s）：**14/14 轮全部停顿 1 秒**。
            #
            # 饱和之后，冷却到期后的**第一次**尝试就是金丝雀：它同时是唯一的恢复
            # 探测点。若它在份额内吐首包，`record_success` 会把计数清零（节点已自证
            # 恢复）；若它再次被裁短，那就是「刚为这个原因冷却过、立刻复现」——这一次
            # 尝试本身就足够定罪，不需要再攒够 `limit` 次。
            self._budget_capped_streak[provider_key] = limit
            self._budget_capped_escalations[provider_key] = (
                self._budget_capped_escalations.get(provider_key, 0) + 1
            )
            # 与 `FailoverRouter._timeout_cooldown` 同一个配置项：升级后的处置就是
            # 「一次上游超时」，所以时长必须与真正的超时分支一致，否则两条路径会
            # 各自朝不同方向漂移（本仓库反复出现的「同一事实两份定义」）。
            cooldown = self._env_float("GATEWAY_TIMEOUT_COOLDOWN", 60.0)
            end_time = time.time() + cooldown
            self._cooldowns[provider_key] = max(
                self._cooldowns.get(provider_key, 0.0), end_time
            )
            # [BUDGET-6] 经统一判据写入：本方法**不是**风控事件，若节点此刻正处于一段
            # 仍在生效的风控避让（人机验证 / 认证失效），它的成因必须保留——否则人工重登
            # 这个唯一出口会失效，档案也会把 captcha 节点误报成「网络问题」。
            self._set_cooldown_cause(provider_key, "timeout", end_time)
            self._persist_state()
            logger.warning(
                "[握手预算] 节点 [%s] 连续 %d 次未能在分到的预算份额内返回首包，"
                "按上游超时冷却 %.1fs。这不是风控拦截（不写入风控台账）：节点只是"
                "比预算份额慢或挂死，冷却到期后会再探一次。若该节点实际健康且只是"
                "较慢，请调大 GATEWAY_HANDSHAKE_BUDGET，而不是关掉本机制——"
                "关掉之后每个请求都会为它白等一整份份额，且每次都真的触发一次上游调用",
                provider_key,
                limit,
                cooldown,
            )
            return cooldown

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
            # [H5] 用 pop 而不是写 0：`failure_counts` 的语义是「当前处于失败中的
            # 节点」，写 0 会留下一个与「从未失败」等价、却让快照指纹每次都不相同的
            # 条目，使健康节点的成功永远无法被判定为「状态未变」而白白写盘。
            self._failure_counts.pop(provider_key, None)
            self._cooldowns.pop(provider_key, None)
            self._cooldown_severity.pop(provider_key, None)
            # [OPT-027] 节点已用当前凭证成功 -> 「新凭证仍被拒」的证据被证伪，清除。
            self._fresh_credential_rejected.pop(provider_key, None)
            self._auto_lease_fingerprint.pop(provider_key, None)
            # [BUDGET-4] 成功即证明它能在份额内吐首包 -> 「连续被裁短」的连续计数
            # 归零。不清零的话，隔很久之后一次单次被裁短就会立刻触发升级，等于把
            # 一段早已过去的旧账算在现在的节点头上（「连续」必须是连续的）。
            self._budget_capped_streak.pop(provider_key, None)
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

    def _set_cooldown_cause(self, provider_key: str, cause: str, cooldown_end: float) -> None:
        """[BUDGET-6] 记录这次的**冷却成因**——但不得让非风控记账抹掉仍在生效的风控成因。

        `_cooldowns` 与 `_cooldown_severity` 是同一个窗口的两半：前者取 `max()`（保留
        更长的窗口），后者过去是**无条件覆盖**。于是「更长的窗口」留下来了，「那个窗口
        为什么存在」却被抹掉，而 `release_auth_avoidance_locked` 与档案建议都按成因判定。

        实测的两种到达方式（都不是风控事件）：
        - [BUDGET-6] BUDGET-5 的预算裁短升级（`note_budget_capped`）：qwen 带着一次真实的
          人机验证 `(45s, waf)`，被我们自己的预算裁短一次后变成 `(60s, timeout)`，
          `release_auth_avoidance_locked(..., _RELEASABLE_BY_RENEWAL)` 返回 `[]`——
          运维照档案做完滑块验证也解除不了，而 waf 是**唯一只有人能解除**的一类。
        - 同形的既有缺陷：先种 900s 的 waf，再记一次 429 → 窗口仍是 900s，成因变成
          `ratelimit`，重登复位同样失效。

        规则（一条，两个写入方共用，避免「同一事实两份定义」）：
        - 风控成因之间**可以**互相改写——后一次拦截就是更新的证据，OPT-029 的成因限定
          依赖这一点；
        - **非风控**成因只有在它确实延长了窗口（即旧窗口已被取代）时才改写；否则旧窗口
          仍然生效，成因必须跟着旧窗口走。
        """
        prev_end = self._cooldowns.get(provider_key, 0.0)
        # 读「有效成因」而不是原始记录：复位路径（`release_auth_avoidance_locked`）就是
        # 用它判定的，两处必须问同一个问题。记录缺失时它会退化到台账的最近一次拦截，
        # 方向是**保留**风控成因（宁可多留一次人工出口，不可平白丢掉）。
        prev_cause = self._effective_cooldown_severity(provider_key) if prev_end > time.time() else ""
        if (
            cause not in _RISK_CAUSES
            and prev_cause in _RISK_CAUSES
            and prev_end > time.time()
            and prev_end >= cooldown_end
        ):
            # 旧的**风控**窗口仍然生效且没有被这次记账取代：保留它的成因。
            # 只在内存里保留，落盘由调用方统一的 `_persist_state()` 完成。
            return
        self._cooldown_severity[provider_key] = cause

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
            cause_before_release = self._effective_cooldown_severity(provider_key)
            if cause_before_release not in allowed:
                continue
            # 3. 确有待复位状态
            has_cooldown = self._cooldowns.get(provider_key, 0.0) > now
            if not has_cooldown and self._failure_counts.get(provider_key, 0) <= 0:
                continue

            self._cooldowns.pop(provider_key, None)
            self._cooldown_severity.pop(provider_key, None)
            # [H5] 同上：清零用 pop，避免在持久化状态里留下「0 = 从未失败」的噪声条目。
            self._failure_counts.pop(provider_key, None)
            # 注意：**不**清除凭证指纹记录。释放只是「假设凭证已修好」，而不是
            # 「确认修好」——指纹记录必须保留，否则下一次失败的比对基准被抹掉，
            # OPT-027 赖以判定的「本次失败是否发生在一枚更新的凭证上」将永远为假，
            # 封禁推断会彻底失效（这正是修复过程中被测试抓到的一处逻辑反转）。
            try:
                self.credential_watch.note_success(provider_key)
            except Exception:  # noqa: BLE001
                pass
            # [DAY-ZERO] 「首次配置」的复位**不进入**恢复观察期：观察期是为封禁解除
            # 瞬间的需求释放准备的，而一个刚被扫码登录的节点没有任何被压制的历史积压。
            # 给它上串行化 + 放大步调，只是在惩罚一个刚刚完成首次登录的健康节点。
            if cause_before_release == SEVERITY_UNCONFIGURED:
                try:
                    self.renewal.clear_unconfigured(provider_key)
                except Exception:  # noqa: BLE001
                    pass
                released.append(provider_key)
                logger.warning(
                    "[未配置复位] 节点 [%s] 的凭证已出现（原因: %s）：配置缺口已补齐，"
                    "避让解除，节点恢复为可派发候选",
                    provider_key, reason,
                )
                continue
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

    def unconfigured_providers(self) -> List[str]:
        """[DAY-ZERO] 曾经未配置、现已补上凭证的节点（只读判定）。"""
        try:
            return self.renewal.unconfigured_providers()
        except Exception:  # noqa: BLE001 - 判定失败绝不阻断任何流程
            return []

    def _credential_event_providers(self) -> List[str]:
        """「凭证事实变了」的全部节点：续期（换掉了）**或**首次出现（配置补上了）。

        两者合成一个入口，是它们共享同一个触发动作（运维执行了 `login.py`）；
        但**成因仍然分开判定**——`release_auth_avoidance_locked` 按 `causes` 逐个
        比对，所以一条「首次配置」的触发绝不会顺手复位一个 `auth`/`waf` 冷却，
        反之亦然。合并的只是「谁被检查」，不是「谁被复位」。
        """
        merged: List[str] = []
        for pk in list(self.renewed_providers()) + list(self.unconfigured_providers()):
            if pk not in merged:
                merged.append(pk)
        return merged

    def release_renewed_credentials_now(self, providers=None, reason: str = "credential-renewal") -> List[str]:
        """[OPT-026] 同步复位入口：用于启动期（无并发）或已在锁内的调用方。

        `providers` 为 None 时依据凭证指纹自动判定；显式给出时跳过判定直接复位
        （覆盖环境变量式凭证等无法取指纹的部署）。

        [OPT-029] 触发条件是「凭证**确已更新**」，该事实对 `auth` 与 `waf` 同时成立
        （人工完成滑块验证后令牌同样会刷新），因此两者都可被复位；`ratelimit` 永不。
        """
        targets = list(providers) if providers is not None else self._credential_event_providers()
        if not targets:
            return []
        return self.release_auth_avoidance_locked(
            targets, reason, _RELEASABLE_BY_CREDENTIAL_EVENT
        )

    async def release_renewed_credentials(self, providers=None, reason: str = "credential-renewal") -> List[str]:
        """[OPT-026] 运行期入口：自动判定（或指定）并复位凭证成因的避让。"""
        targets = list(providers) if providers is not None else self._credential_event_providers()
        if not targets:
            return []
        async with self._lock:
            return self.release_auth_avoidance_locked(
                targets, reason, _RELEASABLE_BY_CREDENTIAL_EVENT
            )

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
        """自适应调用步调（Pacing & Jitter）：**唯一的节奏裁决点**。

        [M1] 原实现把它写成「读 → 算 → 睡 → 写」四段，每一段都在锁外：
        两个并发调用者读到同一个 `last`、算出同一个 `elapsed`、同时睡完、
        同时写回。于是 `min_call_interval`（qwen 0.3s / deepseek 0.4s /
        kimi 0.35s）在并发下被完全绕过——实测三个并发派发里有两个在同
        一毫秒打到上游（`[0.0, 0.301, 0.301]`）。

        更根本的问题是节奏有**两处**：`server.py` 在派发前也做了一次
        `traffic_pacer.acquire(...)` 的「等待后立即释放」，而真正的上游调用
        发生在后面，所以那次等待只推迟了请求自己，不约束任何并发者。

        现在用**两段式**语义，缺一段都不成立：

        1. **预约**（锁内）：把「下一次可调用时刻」写回 `_next_slot`，并发调用者
           因此各拿到一段互不重叠的时间片，不会挤在同一个空档。
        2. **放行时复核**（锁内循环）：睡醒后在锁内用**真实时刻**再判一次
           `now - 上次实际放行 >= interval`，不满足就继续等到满足为止。

        只做第 1 段是不够的——实测过：事件循环被压住时调用者会**睡过头**
        （0.300s 的槽位在 0.398s 才醒），它自己的槽位没错，但与下一个槽位
        （0.600s）的**实际**间隔被压缩到 0.203s。槽位是预测值，风控看的是真实
        上游调用间隔，所以真正的约束必须落在第 2 段的放行时刻上。睡过头只会
        把后续调用整体后移，永远不会压缩间隔。
        """
        if os.getenv("GATEWAY_DISABLE_PACING") == "1":
            self._last_call_time[provider_key] = time.time()
            return

        meta = PROVIDER_RISK_SPECS.get(provider_key)
        if not meta:
            return

        # [OPT-022] 观察期内放大调用间隔：避免解封瞬间以常规速率涌入而立即二次触发风控
        # 倍率在锁外取（纯函数、带 TTL 缓存），锁内只做「读-算-预约」这一小段。
        jitter = random.uniform(meta.jitter_range[0], meta.jitter_range[1])
        target_interval = self.compute_pacing_interval(provider_key, jitter)

        # ---- 第 1 段：预约一个互不重叠的时间片 ----
        async with self._lock:
            now = time.time()
            slot = max(now, self._next_slot.get(provider_key, 0.0))
            self._next_slot[provider_key] = slot + target_interval
        wait_time = max(0.0, slot - time.time())
        if wait_time > 0:
            await asyncio.sleep(wait_time)

        # ---- 第 2 段：在锁内按**真实时刻**占座，睡过头时补足间隔 ----
        while True:
            async with self._lock:
                now = time.time()
                last_time = self._last_call_time.get(provider_key, 0.0)
                deficit = target_interval - (now - last_time)
                if deficit <= 0:
                    self._last_call_time[provider_key] = now
                    break
            wait_time += deficit
            await asyncio.sleep(deficit)

        # 记录「本节点最近一次实际施加的等待」，供遥测头读取（见 _last_pacing_wait_ms）
        self._last_pacing_wait_ms[provider_key] = wait_time * 1000.0

    def last_pacing_wait_ms(self, provider_key: str) -> float:
        """最近一次节奏裁决实际施加的等待毫秒数（只读，供遥测）。"""
        return float(self._last_pacing_wait_ms.get(provider_key, 0.0))

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
        self._next_slot.clear()
        self._last_pacing_wait_ms.clear()
        # [OPT-026] 冷却成因记录一并重置（否则会与后续用例的冷却串扰）
        self._cooldown_severity.clear()
        # [OPT-027] 封禁推断证据一并重置，确保用例之间零串扰
        self._fresh_credential_rejected.clear()
        self._auto_lease_fingerprint.clear()
        # [BUDGET-4] 预算裁短的连续计数同样不得跨用例串扰
        self._budget_capped_streak.clear()
        self._budget_capped_escalations.clear()
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
        # [OPT-031] 长效避让门控统计一并重置，确保用例之间零串扰
        self._long_avoidance_stats = {
            "fast_fails": 0,
            "redirects": 0,
            "last_fast_fail_provider": "",
            "last_fast_fail_cause": "",
            "last_redirect": "",
        }
        # [OPT-033] 禁用门控与闸门拒载分布一并重置。注意 `disabled_providers` 是**紧随
        # 其后**从环境变量重读的，因此测试里改环境变量后调用 reset() 即可生效；反过来，
        # 直接改 `disabled_providers` 集合的用例必须自己恢复，不能依赖 reset()。
        self._disabled_gate_stats = {
            "fast_fails": 0,
            "redirects": 0,
            "last_fast_fail_provider": "",
            "last_redirect": "",
        }
        self._dispatch_gate_stats = {
            "credential_outage": 0,
            "disabled": 0,
            "quarantine": 0,
            "long_avoidance": 0,
        }
        # [OPT-034] 后台触点拒载直方图同样复位（它不是持久状态，只是本轮观测）
        self._touch_block_stats = {}
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


class ManagedStream:
    """[C2] 带显式回收钩子的上游流。

    `execute_chat` 在**返回之前**就已持有节点并发信号量并打开了上游流，而归还动作
    原本只写在包装生成器的 `finally` 里。异步生成器的收尾代码有一个容易被忽略的
    前提：**它必须先被启动过**。`aclose()` 对一个从未 `__anext__()` 的生成器是
    空操作，不会执行 `finally`——于是「拿到了流、却一帧都没拉就开始关闭」这条路径
    上，信号量与上游连接永久泄漏，节点并发上限被幽灵流占满。

    本类型把回收变成一个可以单独调用的动作，并且**幂等**：无论是流正常走完、
    中途异常、被 `aclose()`，还是调用方从未开始迭代就直接 `close()`，槽位都只
    归还一次。它实现完整的异步迭代协议，对既有消费方完全透明。
    """

    def __init__(self, agen: AsyncIterator[Tuple[str, dict]], close: Optional[Callable[[], None]] = None):
        self._agen = agen
        self._close_hook = close
        self._closed = False
        self._started = False

    def release_sync(self) -> None:
        """只执行同步槽位回收，不关闭上游生成器。

        供**任务已被取消**的收尾路径使用（那里不允许 await）：槽位必须立即归还，
        否则并发上限被幽灵流占满；上游连接由框架侧的 `aclose` 负责关闭。
        """
        if self._close_hook is not None:
            try:
                self._close_hook()
            except Exception:  # noqa: BLE001 - 回收失败不得掩盖原始错误
                logger.warning("[对流回收] 回收节点槽位失败 (已忽略)")

    async def close(self) -> None:
        """幂等收尾：先关闭生成器，再执行回收钩子。

        [C2] 关键细节：若流**从未被启动**，`aclose()` 不会执行生成器内部的任何
        收尾代码（Python 对未启动的异步生成器直接返回）。而「拿到流却一帧没拉就
        放弃」正是要回收的那条路径——于是这里先推进一帧使生成器进入运行态，再
        `aclose()`，其 `finally`（关闭上游 `raw_stream`）才会真正执行。对已经
        启动过的流，这一步会被 `_started` 跳过，行为逐字节不变。
        """
        if self._closed:
            return
        self._closed = True
        try:
            if not self._started:
                try:
                    await self._agen.__anext__()
                except (StopAsyncIteration, Exception):  # noqa: BLE001
                    # 启动即失败/即结束：生成器已自行收尾，继续走关闭流程
                    pass
            aclose = getattr(self._agen, "aclose", None)
            if aclose is not None:
                await aclose()
        finally:
            self.release_sync()

    def __aiter__(self) -> "ManagedStream":
        return self

    async def __anext__(self) -> Tuple[str, dict]:
        if self._closed:
            raise StopAsyncIteration
        self._started = True
        try:
            return await self._agen.__anext__()
        except StopAsyncIteration:
            await self.close()
            raise

    async def aclose(self) -> None:
        await self.close()


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
        include_long_avoidance: bool = False,
    ) -> List[FailoverCandidate]:
        """根据当前主选节点生成同等能力环内的有序候选节点列表。

        排序策略：
        1. 排除当前排除集（已尝试失败过的 Provider 以及被主动禁用的 Provider）
        2. [OPT-021] 排除处于运维长期封禁租约的节点（硬阻断，除非显式放行）
        3. [OPT-031] 排除处于**长效避让**（升级后的长冷却，小时~天级）的节点
        4. 健康节点排在冷却避让中节点之前
        5. 健康节点中：低风险 (LOW) > 中风险 (MEDIUM) > 高风险 (HIGH)
        6. 同等风险下，按历史失败次数升序排列

        注意**短时软冷却**节点仍保留在候选中（仅被沉底）：秒级~两分钟的自动退避属于可
        自愈的短暂冷却，环内全部冷却时允许最后一搏以保可用性；而运维封禁租约意味着账号级
        封禁，长效避让意味着上游已连续多次拦截（见 OPT-029/OPT-031），触碰不会有正常返回，
        只会延长封禁，两者都必须彻底排除而非降级。

        `include_quarantined` / `include_long_avoidance` 是**仅供审计与诊断视图**的两个
        放行开关：审计需要看到「被排除的节点此刻的冷却读数」，因此必须能把它们包含回来。
        派发路径一律不传，所以在真实的派发决策中租约与长效避让恒被排除。
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

            # [OPT-031] 长效避让节点同样绝不作为后备候选（小时~天级的升级冷却，
            # 上游已连续拦截过多次；把它留在候选里等于给风暴留了一条通道）。
            # 短冷却不受影响：它小于阈值，因此这里恒为 False，降级排序语义不变。
            if not include_long_avoidance and self.cooldown_tracker.is_in_long_avoidance(p_key):
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

    # [OPT-033] 派发闸门的**唯一定义**
    def evaluate_dispatch_gate(
        self,
        provider_key: str,
        outage_active: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """回答「此刻这个节点能不能被派发、若不能是被哪一道闸门拦住」。

        这是派发闸门的**唯一**定义处：`execute_chat` 的派发循环与 `audit_dispatch` 的
        预演都只经由本函数取答案。OPT-032 的零漂移是靠**测试**保证的（预演说改派到 X，
        真实派发就必须派发到 X）——那已经比两处代码各自正确前进了一步，但测试只能覆盖
        它想到的场景：实测就漏掉了 `disabled` 这一格，`execute_chat` 当时对
        `DISABLED_PROVIDERS` 毫无门控，而预演却把该节点报成 `blocked_by="disabled"` 并
        预测改派到 glm，真实派发却直接把请求送进了被运维显式禁用的节点。把谓词收束到
        一处后，「预演 == 派发」变成**结构性**成立：只有一份闸门顺序可以偏离。

        与 `get_fallback_candidates` 的分工：本函数回答「这个**已知**节点现在能不能用」，
        候选选择回答「环里还有谁可以用」。两者都读同一组谓词，但前者是硬阻断判据
        （含环级凭证失效），后者只做排序与可用性过滤。

        闸门顺序即语义，且**必须与 `execute_chat` 逐条一致**：

        1. `credential_outage`（环级，OPT-025）——整环无可用凭证时，任何节点都不会被派发，
           故它必须排在逐节点闸门之前，否则会误导运维去处理某一个节点；
        2. `disabled`（运维声明，`DISABLED_PROVIDERS`）——「这个节点不属于本次部署」，
           与租约同属人工声明级，因此同样是硬阻断而非降级。此前它只在负载均衡与会话亲和
           的可用性检查里生效，派发边界本身不予理会，于是「绕开负载均衡那条路」就能把
           请求送进被显式禁用的节点；
        3. `quarantine`（运维/自动租约，OPT-020/021/027）；
        4. `long_avoidance`（升级后长冷却，OPT-031）。

        长短软冷却**刻意不在此列**：短冷却按 OPT-021 的语义保持 demote-only，环内全冷却时
        仍允许最后一搏，可用性优先。
        """
        if outage_active is None:
            try:
                outage_active = bool(self.cooldown_tracker.credential_watch.is_outage_active())
            except Exception:  # noqa: BLE001
                outage_active = False

        gate: Dict[str, Any] = {
            "dispatchable": True,
            "blocked_by": "",
            "remaining_seconds": 0.0,
            "cause": "",
            "long_avoidance": False,
            "disabled": False,
        }

        if outage_active:
            gate["dispatchable"] = False
            gate["blocked_by"] = "credential_outage"
            return gate

        if self.cooldown_tracker.is_disabled(provider_key):
            gate["dispatchable"] = False
            gate["blocked_by"] = "disabled"
            gate["disabled"] = True
            return gate

        if self.cooldown_tracker.is_quarantined(provider_key):
            gate["dispatchable"] = False
            gate["blocked_by"] = "quarantine"
            gate["cause"] = "ban"
            try:
                gate["remaining_seconds"] = float(self.cooldown_tracker.quarantine.remaining(provider_key))
            except Exception:  # noqa: BLE001
                gate["remaining_seconds"] = 0.0
            return gate

        if self.cooldown_tracker.is_in_long_avoidance(provider_key):
            gate["dispatchable"] = False
            gate["blocked_by"] = "long_avoidance"
            gate["long_avoidance"] = True
            gate["cause"] = self.cooldown_tracker.long_avoidance_cause(provider_key)
            try:
                gate["remaining_seconds"] = float(self.cooldown_tracker.get_remaining_cooldown(provider_key))
            except Exception:  # noqa: BLE001
                gate["remaining_seconds"] = 0.0
            return gate

        return gate

    def evaluate_dispatch_gate_full(
        self, provider_key: str, outage_active: Optional[bool] = None
    ) -> Dict[str, Any]:
        """`evaluate_dispatch_gate` 的补充视图：即使某节点已可派发，也一并报告它**是否**
        命中 disabled / 租约 / 长效避让这类「本可阻断但现在被前置闸门盖住」的状态。

        `audit_dispatch` 的逐节点备注用它，使运维在环级凭证失效掩盖了逐节点状态时，
        仍能看到「这个节点其实还在被封禁租约里」，而不是只看到一句环级成因。
        """
        info = self.evaluate_dispatch_gate(provider_key, outage_active=outage_active)
        try:
            info["disabled"] = bool(self.cooldown_tracker.is_disabled(provider_key))
        except Exception:  # noqa: BLE001
            pass
        try:
            info["quarantined"] = bool(self.cooldown_tracker.is_quarantined(provider_key))
        except Exception:  # noqa: BLE001
            pass
        try:
            info["in_long_avoidance"] = bool(self.cooldown_tracker.is_in_long_avoidance(provider_key))
        except Exception:  # noqa: BLE001
            pass
        try:
            cooldown = float(self.cooldown_tracker.get_remaining_cooldown(provider_key))
        except Exception:  # noqa: BLE001
            cooldown = 0.0
        info["remaining_cooldown_seconds"] = round(cooldown, 1)
        return info

    def audit_dispatch(self, provider_key: str, wire_model: str) -> Dict[str, Any]:
        """[OPT-032] 零上游派发预演：此刻这个请求会被谁承接、谁被排除、被**哪一道**闸门排除。

        动机来自一次真实操作：运维声明「deepseek 已被封禁，零误触是硬约束」之后，唯一能
        确认这份声明**真的生效**的办法是手写一段脚本、注入假 Provider 工厂、用 TestClient
        发一次请求，再看被触碰的节点集合。而这个确认动作本身是日常的——每次改租约、改
        环境变量、重启之后都要问一遍「现在到底会不会派发到它」。逐个翻看五个 admin 端点
        （quarantine / risk / correlation / credential-outage / ban-inference）都回答不了它，
        因为问题不是「这个节点处于什么姿态」，而是「这条请求此刻的派发决定是什么」。

        只读且零上游调用：不实例化任何 Provider，只向**派发路径同一组**闸门函数提问，
        因此不存在「文档说会避让、实际仍派发」的漂移——预演与派发读的是同一份状态、同一组
        谓词。这也是它敢被称为「预演」而不是「估算」的唯一理由。

        刻意**不模拟**负载均衡与会话亲和：它们可能改变最终承接节点，而模拟它们就等于在
        这里重写一遍三个协议的派发前管线，那正是本项目反复出现的「文档与实现漂移」缺陷的
        温床。因此字段以 `predicted_` 命名，`scope_note` 显式声明该边界——宁可说清它不管
        什么，也不要给一个看起来更完整、实则会错的承诺。

        闸门顺序与 `execute_chat` 派发循环**逐条一致**（全环凭证失效 → 运维禁用
        `DISABLED_PROVIDERS` → 运维/自动租约 → 长效避让），因为「哪一道闸门先开火」正是
        运维需要知道的信息。**[OPT-033] 该顺序不再靠两处代码各自维护：逐节点状态一律向
        `evaluate_dispatch_gate_full()` 求值，只有一份闸门定义可以偏离。**
        """
        ring = classify_capability_ring(provider_key, wire_model)
        member_keys = {p_key for p_key, _ in RING_DEFINITIONS.get(ring, [])}
        if provider_key not in member_keys:
            # 宁可当场报错，也不要产出一份「看起来完整、实则问错了对象」的预演：
            # 传入模型别名而非 provider key 是最容易犯的错，而它会让 is_quarantined
            # 等谓词对着一个不存在的 key 求值，从而把封禁节点报成可派发——这正是
            # 本预演存在意义所要消灭的那类谎报。
            raise ValueError(
                f"provider_key {provider_key!r} 不属于能力环 {ring.value!r}；"
                f"预演需要已解析的 (provider_key, wire_model)，请先用 resolve_model() 解析模型别名"
            )

        # [OPT-025] 全环凭证失效是**环级**决定：它一旦成立，环内任何节点都不会被派发，
        # 且必须排在逐节点闸门之前报告，否则会误导运维去处理某单个节点。
        try:
            outage_active = bool(self.cooldown_tracker.credential_watch.is_outage_active())
            outage_failed = list(self.cooldown_tracker.credential_watch.active_failures())
        except Exception:  # noqa: BLE001
            outage_active, outage_failed = False, []

        members: List[Dict[str, Any]] = []
        for p_key, w_model in RING_DEFINITIONS.get(ring, []):
            # [OPT-033] 逐节点状态一律向**唯一**的闸门定义处求值，预演不再自带一份
            # if/elif 副本——副本正是 OPT-032 漏掉 `disabled` 那一格的成因。
            info = self.evaluate_dispatch_gate_full(p_key, outage_active=outage_active)
            blocked_by = info["blocked_by"]
            remaining = float(info["remaining_seconds"])

            members.append(
                {
                    "provider": p_key,
                    "wire_model": w_model,
                    "dispatchable": not blocked_by,
                    "blocked_by": blocked_by or None,
                    "blocking_cause": info["cause"] or None,
                    "remaining_seconds": round(remaining, 1) if remaining else 0.0,
                    "remaining_hours": round(remaining / 3600.0, 2) if remaining else 0.0,
                    # 被前置闸门盖住、但客观存在的状态，供运维分辨「环级成因」与「该节点
                    # 自身也尚未恢复」
                    "disabled": bool(info.get("disabled")),
                    "quarantined": bool(info.get("quarantined")),
                    "in_long_avoidance": bool(info.get("in_long_avoidance")),
                    "remaining_cooldown_seconds": info.get("remaining_cooldown_seconds", 0.0),
                }
            )

        dispatchable = [m for m in members if m["dispatchable"]]
        primary = next((m for m in members if m["provider"] == provider_key), None)

        # 改派目标**必须**取自真正的候选选择函数，而不是「环内第一个可派发成员」。
        # 实测教训：两者并不相同——`get_fallback_candidates` 按 (冷却, 风险等级, 失败次数)
        # 排序，所以 glm 可能排在环序更靠前的 qwen 之前；按环序猜会预测出 qwen 而实际派发
        # 到 glm，预演当场变成谎报。本套件的 test_prediction_matches_real_dispatch_redirect
        # 就是据这次实测写的。
        redirect_target = None
        if primary is not None and not primary["dispatchable"]:
            try:
                candidates = self.get_fallback_candidates(provider_key, wire_model)
            except Exception:  # noqa: BLE001
                candidates = []
            if candidates:
                redirect_target = candidates[0].provider_key

        # 预测的错误类型必须与派发路径**真正会抛出**的那一个一致：租约/禁用/长效避让都是
        # 先改派、改派不出去才报错，因此只有整环确实无节点可派时才有 err_type——否则预演
        # 就成了谎报（把一次正常的改派说成会失败）。映射表与 execute_chat 的抛出一一对应。
        _GATE_ERR_TYPE = {
            "credential_outage": "credential_outage",
            "disabled": "disabled",
            "quarantine": "quarantined",
            "long_avoidance": "ring_exhausted",
        }
        if outage_active:
            err_type: Optional[str] = "credential_outage"
        elif not dispatchable and primary is not None and not primary["dispatchable"]:
            err_type = _GATE_ERR_TYPE.get(primary["blocked_by"], "ring_exhausted")
        else:
            err_type = None

        return {
            "provider": provider_key,
            "wire_model": wire_model,
            "ring": ring.value,
            "upstream_calls": 0,
            "predicted_primary_dispatchable": bool(primary and primary["dispatchable"]),
            "predicted_redirect_target": redirect_target,
            "predicted_fail_closed": not dispatchable,
            "predicted_err_type": err_type,
            "dispatchable_count": len(dispatchable),
            "ring_size": len(members),
            "credential_outage": {"active": outage_active, "failed_providers": sorted(outage_failed)},
            "ring_members": members,
            "scope_note": (
                "仅预演**闸门**决定（全环凭证失效 / 运维禁用 DISABLED_PROVIDERS / 运维与自动"
                "租约 / 长效避让），零上游调用；不含负载均衡分流与会话亲和固定，最终承接节点"
                "可能与 predicted_redirect_target 不同。"
            ),
        }

    # ---------------------------------------------------------------- [M2] 超时边界
    def _first_chunk_timeout(self) -> float:
        """[M2] 等待上游**首包**的上限（秒，默认 180）。

        这不是「回复多久算慢」的调参，而是「握手阶段可以占用多少共享资源」的边界：
        在首包到达之前，本函数握着节点并发信号量与观察期串行闸门，而推理模型正常
        思考几十秒是**健康**的（心跳机制存在的原因），所以上限必须明显高于正常
        思考时间、又必须存在——否则一个挂住的上游会把这些共享资源占到 httpx 的
        600 秒读超时为止，把同账号的所有请求一起拖住。
        `GATEWAY_FIRST_CHUNK_TIMEOUT=0` 关闭（回退为裸 await，逐值等同旧行为）。
        """
        return self.cooldown_tracker._env_float("GATEWAY_FIRST_CHUNK_TIMEOUT", 180.0)

    def _stream_idle_timeout(self) -> float:
        """[M2] 上游**两条**数据帧之间的最长间隔（秒，默认 180）。

        为什么需要它：httpx 的 read timeout 是**每个 socket 读**的，每来一个字节
        就重置一次，因此一个每 599 秒吐一字节的上游永远不会触发它；而心跳只会
        持续产出 `: keepalive`，让客户端也永远不会放弃。结果是连接、上游 socket、
        节点并发信号量与负载均衡配额可以被一个半死不活的上游无限期占住。
        这里给「上游还活着吗」一个明确的判据，超时按上游故障处置并计入失败。
        `GATEWAY_STREAM_IDLE_TIMEOUT=0` 关闭。
        """
        return self.cooldown_tracker._env_float("GATEWAY_STREAM_IDLE_TIMEOUT", 180.0)

    def _handshake_budget(self) -> float:
        """整条请求在**握手阶段**可以花掉的总时间（秒，默认 180）。

        [BUDGET] `_first_chunk_timeout()` 是**每个候选**的上限，而派发循环会逐个
        走完能力环里的候选——于是真正的等待是「候选数 × 每次上限」，五个节点就是
        900 秒。这不是调参问题，是**边界挂错了层**：客户端等待的是「这条请求」，
        而不是「第 N 次尝试」，它不知道网关内部还有几个候选；而这段时间里响应头
        都还没发出（`execute_chat` 在构造响应之前被 await，见 H4），心跳没有运行
        的机会，负载均衡配额与节点信号量一直被这条注定失败的请求占着。

        实测（`GATEWAY_FIRST_CHUNK_TIMEOUT=1.0`，五节点全挂）：`failed after 5.0s`。
        乘以默认的 180s 就是 900s，超过实测的客户端容忍上限（Claude Code 420s /
        Codex 600s 仍未放弃，即 harness 先到点）。

        预算**不改变**单次尝试的语义：它与每次上限取 `min()`，所以正常情况（首个
        候选就成功）逐值不变；只有走到多个候选时才由预算封顶。剩余预算不足时不再
        发起注定超时的尝试，直接如实报错——少一次无用上游调用，也少一次无效凭证
        流量，这与整条避让线「不白白触碰上游」的立场一致。
        `GATEWAY_HANDSHAKE_BUDGET=0` 关闭（回退为纯「候选数 × 每次上限」）。
        """
        return self.cooldown_tracker._env_float("GATEWAY_HANDSHAKE_BUDGET", 180.0)

    def _count_untried_candidates(
        self,
        primary_provider: str,
        primary_model: str,
        excluded_providers: Optional[Set[str]] = None,
        current_provider: Optional[str] = None,
    ) -> int:
        """除 `excluded_providers` 与 `current_provider` 之外、此刻仍可被派发的候选数。

        [BUDGET-2] 只用于把总预算**公平分摊**给尚未尝试的候选：`get_fallback_candidates`
        已经排除了已尝试节点、运维禁用、封禁租约与长效避让，所以这里的计数与派发循环
        真正会再走几轮是同一个集合。用作除数而不是用来判决——判决仍只看时钟。

        `current_provider` 必须显式传入：`get_fallback_candidates` **恒排除初选节点**，
        所以「当前节点是否已经记进 `tried_providers`」会让返回值差一（当前节点是初选时
        永远不在返回值里，是后备候选时才可能在里面）。把它作为参数排除掉之后，调用点
        在派发循环的顶部与 `try` 内部都可以用同一个除数 `返回值 + 1`——两处的份额因此
        逐值相同，不会出现「顶部按 1/5 判决、内部按 1/4 计时」这种自相矛盾。
        """
        excluded = set(excluded_providers or ())
        if current_provider:
            excluded.add(current_provider)
        try:
            return len(
                self.get_fallback_candidates(
                    primary_provider,
                    primary_model,
                    excluded_providers=excluded,
                )
            )
        except Exception:  # noqa: BLE001 - 预算分摊失败不应影响派发
            return 0

    def _handshake_share(
        self,
        deadline: float,
        primary_provider: str,
        primary_model: str,
        tried: Set[str],
        current_provider: str,
    ) -> Tuple[float, float]:
        """返回 `(剩余总预算, 本次尝试的公平份额)`（秒）。

        「本次尝试」是剩余候选中的一员，故份额 = 剩余预算 / (后续候选数 + 1)：
        最后一个候选拿走全部剩余，所以总时长的上界仍严格成立。
        """
        remaining_budget = deadline - asyncio.get_running_loop().time()
        later = self._count_untried_candidates(
            primary_provider,
            primary_model,
            excluded_providers=set(tried),
            current_provider=current_provider,
        )
        return remaining_budget, remaining_budget / max(later + 1, 1)

    def _budget_exhausted_error(
        self,
        handshake_budget: float,
        tried_providers: Set[str],
        composite_errors: List[str],
        remaining_budget: float,
        share: float,
    ) -> ProviderError:
        """构造「预算不足以再发起一次有意义的尝试」的拒绝。

        这是**零上游**结论，因此文案必须让运维能自己判断该做什么：试过哪些节点、
        预算多少、为什么剩下的时间连一次尝试都不够。只说「超时」会让人去查网络，
        而实际问题可能是预算配小了（默认 180s 分摊到五节点每份 36s）。
        `retry_after` 取已尝试节点里**最早**可能恢复的冷却窗口——与「环已走完」
        同一条纪律（H7），不编一个常量倒计时。
        """
        detail = "\n".join(f" - {e}" for e in composite_errors) or " - (无)"
        if remaining_budget > 0.0:
            shortfall = (
                f"剩余 {remaining_budget:.2f}s，分摊给当前及后续候选后每次仅 "
                f"{share:.3f}s，低于一次有意义的尝试下限 "
                f"{_MIN_HANDSHAKE_ATTEMPT_SECONDS:.2f}s"
            )
        else:
            shortfall = f"剩余预算 {max(remaining_budget, 0.0):.2f}s 已不足一次尝试"
        logger.error(
            "[握手预算] 总预算 %.1fs 已用尽（已尝试 %d 个节点；%s），停止派发以保证"
            "客户端等待有上界。已记录的失败:\n%s",
            handshake_budget,
            len(tried_providers),
            shortfall,
            detail,
        )
        return BudgetExhaustedRefusal(
            f"握手总预算 {handshake_budget:.0f}s 已用尽（已尝试 "
            f"{len(tried_providers)} 个节点: {', '.join(sorted(tried_providers))}；"
            f"{shortfall}），故不再发起注定失败的上游调用——否则客户端要对每个候选"
            f"白等一次、再拿到同一个结论。这属于上游健康问题（非风控拦截，不会写入"
            f"风控台账）。请检查网络与各节点状态，或用 GATEWAY_HANDSHAKE_BUDGET 调整"
            f"总预算。\n已记录的失败:\n{detail}",
            status=504,
            err_type="upstream_timeout",
            retry_after=min(
                [
                    r
                    for r in (
                        self.cooldown_tracker.get_remaining_cooldown(p)
                        for p in tried_providers
                    )
                    if r and r > 0
                ],
                default=self._timeout_cooldown(),
            ),
        )

    def _timeout_cooldown(self) -> float:
        """[M2] 超时失败后的冷却时长（秒，默认 60）。

        超时与 5xx 的其它成员不同：`record_failure` 的通用分支按
        `min(base_cd, 15 × 连续失败)` 递增，对真实 5xx 是对的（上游可能正在抖），
        但对超时它是**反向**的——连续超时意味着上游**持续**不吐字，此时反复以
        15 秒节奏去敲，每一次都白占一个节点并发槽与 HTTP/2 连接，并把客户端拖在
        故障转移链上。更糟的是通用分支按规则可能落在 45 秒以内，也就是**短冷却**
        区间，于是该节点被判定为「值得最后一搏」继续派发（OPT-021/031 的判据），
        超时节点因此永远得不到真正的退避。
        取 60 秒：明确高于短冷却上限对「一次性 5xx」的取值，同时又远低于长效避让，
        让一次网络抖动不至于把节点冻结数小时。
        `GATEWAY_TIMEOUT_COOLDOWN` 可调。
        """
        return self.cooldown_tracker._env_float("GATEWAY_TIMEOUT_COOLDOWN", 60.0)

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

        # [BUDGET] 握手总预算：整条请求（跨全部候选）在首包之前能花多久。
        # 见 `_handshake_budget()`——单次上限乘以候选数才是缺陷，而候选数由能力环
        # 定义决定，不由客户端决定。0 表示不设总预算（逐值回退旧行为）。
        handshake_budget = self._handshake_budget()
        deadline = (
            asyncio.get_running_loop().time() + handshake_budget
            if handshake_budget > 0
            else None
        )

        while True:
            # [BUDGET] 握手总预算已耗尽 / 不足以完成一次有意义的尝试：不再派发。
            #
            # 这一次判决必须发生在**进入 try 之前**。放在 try 内会让 `except` 分支
            # 给一个从未被调用的节点记一次失败、上一份冷却——把一个「我们决定不打」
            # 写成一个「上游失败了」，正是本项目反复记录的那类谎报。这里直接如实
            # 报告：试过哪些节点、预算是多少。
            #
            # [BUDGET-3] 判据是**本次可用份额**，不是裸的剩余时长。曾在 try 内部用
            # `max(剩余预算, 0.05)` 作为本次上限，于是份额为 0.024s 时仍按 50ms 派发：
            # 对任何真实上游都是必败，只换来一次白打的上游调用、一次假的超时冷却，
            # 而请求结局与「零上游拒绝」一模一样（504 upstream_timeout）。0.05s 于是
            # 从「兜底」变成了「一定会浪费一次上游触碰」——与整条避让线「不白白触碰
            # 上游」的立场相反。份额低于该地板时，与其它零上游拒绝一致：如实报错、
            # 不派发、不记任何节点的失败。
            if deadline is not None:
                remaining_budget, share = self._handshake_share(
                    deadline,
                    primary_provider_key,
                    primary_wire_model,
                    tried_providers,
                    curr_provider_key,
                )
                if remaining_budget <= 0.0 or share < _MIN_HANDSHAKE_ATTEMPT_SECONDS:
                    raise self._budget_exhausted_error(
                        handshake_budget,
                        tried_providers,
                        composite_errors,
                        remaining_budget,
                        share,
                    )

            tried_providers.add(curr_provider_key)

            # [OPT-033] 闸门顺序的**唯一来源**：本循环不再自带一份 if/elif 副本，而是
            # 向 evaluate_dispatch_gate() 问一次「此刻这个节点能不能被派发、被哪一道
            # 闸门拦住」，再按它给出的 blocked_by 选择处置分支。此前循环里手写的那份
            # 副本与 audit_dispatch 里的另一份副本漂移过：`disabled`（运维显式禁用的
            # 节点）在预演里被报成硬阻断、在派发路径上却毫无门控，实测能把请求直接
            # 送进被禁用的节点。收束到一处后，「预演 == 派发」是结构性成立的。
            gate = self.evaluate_dispatch_gate(curr_provider_key)
            if gate["blocked_by"]:
                self.cooldown_tracker.note_dispatch_gate_block(gate["blocked_by"])

            # [OPT-025] 全环凭证失效快速失败：当多个不同节点在同一窗口内相继 401，
            # 说明我们没有可用凭证，故障转移不可能成功。此时继续逐个打满五个节点，
            # 恰恰是上游 WAF 眼中的「自动化凭证尝试」流量特征——风暴本身就是风控触发器。
            # 故立即拒绝派发（零上游调用），并给出重登指引；TTL 后自动放行探路。
            if gate["blocked_by"] == "credential_outage":
                try:
                    failed = self.cooldown_tracker.credential_watch.active_failures()
                except Exception:  # noqa: BLE001
                    failed = []
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
                    # [H6] 真实窗口：快速失败态还剩多久就报多久（TTL 到期即放行探路）。
                    retry_after=self.cooldown_tracker.credential_watch.remaining_seconds(),
                )

            # [OPT-033] 运维禁用硬阻断：`DISABLED_PROVIDERS` 是运维关于「这个节点不属于
            # 本次部署」的**显式声明**，与封禁租约同属人工声明级，因此同样是硬阻断而非
            # 降级。此前它在负载均衡（select_route）与会话亲和（_is_provider_healthy）
            # 的可用性检查里生效，但**派发边界本身不理会**——于是任何绕过负载均衡到达
            # execute_chat 的路径（显式指定的主选、亲和重新解析后的结果）都能把请求送进
            # 被运维禁用的节点，而 OPT-032 的预演却报它是 blocked_by="disabled"。声明与
            # 事实不一致，正是本项目反复记录的那类缺陷。
            if gate["blocked_by"] == "disabled":
                disabled_list = ",".join(sorted(self.cooldown_tracker.disabled_providers)) or "-"
                logger.error(
                    "[禁用硬阻断] 节点 [%s] 已被 DISABLED_PROVIDERS 显式禁用，拒绝派发上游请求",
                    curr_provider_key,
                )
                candidates = self.get_fallback_candidates(
                    primary_provider_key,
                    primary_wire_model,
                    excluded_providers=tried_providers,
                )
                if not candidates:
                    self.cooldown_tracker.note_disabled_gate_fast_fail(curr_provider_key)
                    raise ProviderError(
                        f"目标节点 [{curr_provider_key}:{curr_wire_model}] 已被运维通过"
                        f" DISABLED_PROVIDERS 显式禁用（当前禁用集合: {disabled_list}），"
                        f"且能力环 [{ring.value}] 内已无其他可用节点。为避免派发到被禁用的"
                        f"节点，本次请求已拒绝派发（零上游调用）—— 请移除该节点的禁用声明"
                        f"（环境变量 DISABLED_PROVIDERS）或改用其他厂商模型。",
                        status=503,
                        err_type="disabled",
                        # [H6] 禁用是**运维声明**，没有任何窗口会自己到期——等多久都没用，
                        # 故不报退避时长（由网关层回退到默认值，而不是编一个倒计时）。
                    )
                next_cand = candidates[0]
                self.cooldown_tracker.note_disabled_gate_redirect(
                    curr_provider_key, next_cand.provider_key
                )
                logger.warning(
                    "[禁用硬阻断] 改派: [%s] -> [%s] (模型: %s)",
                    curr_provider_key,
                    next_cand.provider_key,
                    next_cand.wire_model,
                )
                curr_provider_key = next_cand.provider_key
                curr_wire_model = next_cand.wire_model
                continue

            # [OPT-021] 硬阻断：运维封禁租约节点在任何情况下都不得被派发真实请求。
            # 这是整条冷却门控体系的最后一道闸口 —— 即便调用方显式指定了被封禁的
            # 模型别名，也必须立即改派或快速失败，绝不能因为「环内只剩它」而破例触碰。
            if gate["blocked_by"] == "quarantine":
                remaining = gate["remaining_seconds"]
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
                        # [H6] 租约的剩余时长是已知的，照实报。
                        retry_after=remaining,
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

            # [OPT-031] 长效避让硬阻断：处于升级后长冷却（人机验证/认证，小时~天级）的
            # 节点拒绝派发真实请求。与上面 OPT-021 的租约硬阻断是同一处置，成因不同：
            # 租约是人工声明，这里是自动升级出来的长期避让。
            #
            # 为什么必须收在这里：OPT-021 只挡租约、OPT-025 只挡整环 401，两者之间留下了
            # 「整环混合成因的长效避让」这一片空白——五家都已被各自的守护机制判定为
            # 长期不可用，每条客户端请求却仍会把这五个节点逐个打满，产出若干次新的拦截。
            # 这正是 OPT-025 记录的「无效凭证风暴」形状（风暴本身即风控触发器），只是
            # 成因从 401 换成了 waf/auth 升级。而且此刻敲哪个节点都拿不到正常返回。
            #
            # 与「最后一搏」的边界：短软冷却（频控退避，秒级~两分钟）仍按 OPT-021 的
            # demote-only 语义保留最后一搏，可用性不受影响——`is_in_long_avoidance`
            # 的阈值落在短冷却可达上界与升级下界之间的真空带内，短冷却恒定不越界。
            if gate["blocked_by"] == "long_avoidance":
                remaining = gate["remaining_seconds"]
                cause = gate["cause"]
                logger.error(
                    "[长效避让硬阻断] 节点 [%s] 处于长效避让 (成因 %s，剩余 %.1f 小时)，拒绝派发",
                    curr_provider_key,
                    cause,
                    remaining / 3600.0,
                )
                candidates = self.get_fallback_candidates(
                    primary_provider_key,
                    primary_wire_model,
                    excluded_providers=tried_providers,
                )
                if not candidates:
                    avoiding = self.cooldown_tracker.ring_long_avoidance_snapshot()
                    detail = "、".join(
                        f"{k}({v['cause']}，剩余{v['remaining_hours']:.1f}h)" for k, v in sorted(avoiding.items())
                    ) or f"{curr_provider_key}({cause})"
                    self.cooldown_tracker.note_long_avoidance_fast_fail(curr_provider_key, cause)
                    logger.error(
                        "[长效避让硬阻断] 能力环 [%s] 已无可用容量 (%s)，零上游调用拒绝派发",
                        ring.value,
                        detail,
                    )
                    raise ProviderError(
                        f"能力环 [{ring.value}] 内所有节点均处于长效避让，本次请求已拒绝派发"
                        f"（零上游调用），以免继续触碰上游风控而延长封禁。当前状态: {detail}。"
                        f"处置: 成因 auth/waf 表示凭证失效或需完成人机验证，请执行 "
                        f"`.venv/bin/python login.py <provider>`；成因 ban 表示账号级封禁，"
                        f"重登无效，请静待平台解除，切勿重试。",
                        status=503,
                        err_type="ring_exhausted",
                        # [H6] 整环最早可用的时刻 = 当前避让节点里**最短**的剩余时长
                        # （其余节点更晚才恢复，按最短报才不会让客户端在白等最长的
                        # 那一个）。没有快照时退回默认值。
                        retry_after=min(
                            (v["remaining_seconds"] for v in avoiding.values()),
                            default=None,
                        ),
                    )
                next_cand = candidates[0]
                self.cooldown_tracker.note_long_avoidance_redirect(curr_provider_key, next_cand.provider_key)
                logger.warning(
                    "[长效避让硬阻断] 改派: [%s] -> [%s] (模型: %s)",
                    curr_provider_key,
                    next_cand.provider_key,
                    next_cand.wire_model,
                )
                curr_provider_key = next_cand.provider_key
                curr_wire_model = next_cand.wire_model
                continue

            sem = self.cooldown_tracker.get_semaphore(curr_provider_key)
            provider_key = curr_provider_key
            wire_model = curr_wire_model

            try:
                # [BUDGET-3] 在**占用任何资源之前**把本次份额再判一次。
                #
                # 循环顶部已经判过，但顶部与本处之间还隔着频控等待与信号量排队
                # （并发满载时可以到秒级）——份额会在这段时间里继续缩水。若等到
                # 握手那一刻才发现不足，信号量与上游生成器都已在手上，虽能回收却
                # 白花一次排队。这里提前判，零资源占用。
                if deadline is not None:
                    rem, share_now = self._handshake_share(
                        deadline,
                        primary_provider_key,
                        primary_wire_model,
                        tried_providers,
                        curr_provider_key,
                    )
                    if rem <= 0.0 or share_now < _MIN_HANDSHAKE_ATTEMPT_SECONDS:
                        raise self._budget_exhausted_error(
                            handshake_budget, tried_providers, composite_errors, rem, share_now
                        )

                # 1. 频控规避 (Pacing & Jitter)
                await self.cooldown_tracker.apply_pacing_and_acquire(provider_key)

                # 2. 实例化 Provider
                #
                # [STREAM-AUDIT] 两道闸门的**生命周期不同**，必须分别持有：
                #
                #   * 观察期串行闸门 (OPT-022) 只覆盖「握手 → 首包」，文档明确它不占用
                #     整段流式输出，否则观察期内所有会话都会被串行化；
                #   * 节点并发信号量必须覆盖**整段流**。此前两者都挂在同一个
                #     `AsyncExitStack` 上，而本函数是在 `async with` **内部** return 的
                #     —— 返回的那一刻 `__aexit__` 就把信号量还回去了，主体一个字节都
                #     还没读。于是 PROVIDER_RISK_SPECS 里 qwen 3 / deepseek 2 / kimi 2
                #     这些并发上限**只约束了毫秒级的握手**，对真正的并发流完全无效：
                #     多个 agent 会话同时长跑时，同一账号可以同时背着任意多条上游
                #     流式会话，而这正是反爬系统用来识别自动化并发的特征之一。
                #
                # 现在的持有方式是手动 acquire、由返回的生成器在 finally 里 release，
                # 且释放是同步的（`Semaphore.release()`），因此在断连路径
                # (GeneratorExit) 上也安全——那里不允许再 await。
                probation_lock = self.cooldown_tracker.probation_lock(provider_key)
                probation_held = False
                sem_held = False
                sem_handed_off = False
                budget_capped = False
                capped_at = 0.0
                try:
                    if probation_lock is not None:
                        await probation_lock.acquire()
                        probation_held = True
                    await sem.acquire()
                    sem_held = True

                    provider = provider_factory(provider_key)

                    # 3. 试探性获取首包 (确保在流交给外部客户端前捕获初始化错误与上游 4xx/5xx)
                    call_kwargs = dict(chat_args)
                    call_kwargs["model"] = wire_model

                    raw_stream = provider.chat(**call_kwargs)

                    # 尝试读取第一个元素（以捕获握手及首包异常）
                    #
                    # [M2] 这里必须有**上限**。此前是一个裸 await：上游连上了却不
                    # 吐字时，唯一的边界是 httpx 的 600 秒读超时，而这段时间里本函数
                    # 正握着节点并发信号量与观察期串行闸门——一次挂住的握手会把同一
                    # 账号下所有请求堵在观察期闸门后面长达十分钟，客户端早已放弃。
                    # 超时按「上游故障」处置（可故障转移），不是风控成因，因此不会
                    # 给节点上风控冷却，只是普通的失败计数。
                    first_item: Optional[Tuple[str, dict]] = None
                    handshake_timeout = self._first_chunk_timeout()
                    # [BUDGET] 本次尝试只能用「总预算的剩余部分」：整条请求的边界
                    # 必须先于单次尝试的边界。这里只做**收紧**，不判决（判决在循环
                    # 顶部，见 `_handshake_budget` 的说明）。
                    #
                    # [BUDGET-2] 收紧必须按**剩余候选数公平分配**，而不是把剩下的
                    # 全给当前这一个。第一版就是直接把剩余预算当作本次上限：默认
                    # 配置（单次 180s、总预算 180s）下 `min(180, 180)` = 180，
                    # 第一个慢候选就吃光了整条预算，后四个候选一次都不再尝试。
                    # 实测（`claude -p` 对着五节点全挂、单次 30s/总预算 15s 的环）：
                    # 「已尝试 1 个节点」就耗尽 —— 这是**修复前不存在**的可用性回归，
                    # 「第一个节点慢、第二个节点快」这种最常见故障形态反而不自愈了。
                    # 公平份额 = 剩余预算 / 剩余未尝试候选数，且不高于单次上限；
                    # 最后一个候选拿走全部剩余，故总时长上界仍严格成立。
                    #
                    # [BUDGET-3] 份额在这里**必定**已经判过下限（进入 try 时就判了），
                    # 所以不再需要 `max(share, 0.05)` 那个地板——它此前把「不足」变成
                    # 了「按 50ms 派发」，即一次确定必败的上游调用。份额在排队期间
                    # 缩水到地板以下的极端情况，交给下面的超时分支按「预算裁短」处置。
                    if deadline is not None and handshake_timeout > 0:
                        remaining_budget, share = self._handshake_share(
                            deadline,
                            primary_provider_key,
                            primary_wire_model,
                            tried_providers,
                            curr_provider_key,
                        )
                        if share < handshake_timeout:
                            handshake_timeout = max(share, _MIN_HANDSHAKE_ATTEMPT_SECONDS)
                            # [BUDGET-3] 被总预算裁剪过 = 本次上限**不是**节点的能力边界。
                            # 记账时必须区分：否则一个健康但比份额慢的节点会拿到 60s
                            # 冷却（实测：份额 0.1s、节点 0.4s 应答 → qwen 冷却 60s、
                            # 成因 timeout、failure_count 1），被后续请求无谓绕开。
                            budget_capped = True
                            capped_at = handshake_timeout
                    if handshake_timeout > 0:
                        # `wait_for` 的超时会取消被等待的协程——这里正是我们想要的：
                        # 超时即判定上游已死，主动取消拉取以防它作为幽灵任务持有连接。
                        try:
                            first_item = await asyncio.wait_for(
                                raw_stream.__anext__(), timeout=handshake_timeout
                            )
                        except asyncio.TimeoutError:
                            # 保留一位小数：公平分摊后份额可以小于 1s，用 `:.0f`
                            # 会把 0.3s 写成「在 0s 内未返回首包」——报告的数字
                            # 与真实等待不符，正是本项目反复记录的谎报类型。
                            message = (
                                f"节点 [{provider_key}:{wire_model}] 在 "
                                f"{handshake_timeout:.1f}s 内未返回首包（握手超时）"
                            )
                            if budget_capped:
                                message += (
                                    "；本次上限由握手总预算裁剪得出"
                                    "（节点本身未表现出故障，故不计冷却）"
                                )
                                err = BudgetCappedTimeout(
                                    message, status=504, err_type="upstream_timeout"
                                )
                                err.handshake_timeout = capped_at
                                raise err from None
                            raise ProviderError(
                                message, status=504, err_type="upstream_timeout"
                            ) from None
                        except StopAsyncIteration:
                            first_item = None
                    else:
                        try:
                            first_item = await raw_stream.__anext__()
                        except StopAsyncIteration:
                            first_item = None

                    # [STREAM-AUDIT] 上游一帧都没吐 = 空流，不是成功。
                    # 五家的 chat() 结尾都**无条件**产出一帧末位元数据
                    # （`("", {"conversation_id": ...})`），所以「首个元素就是
                    # StopAsyncIteration」只可能意味着上游连接正常建立却没有任何
                    # 内容帧。原实现把它记成 record_success 并返回一个空流，客户端
                    # 收到一个正常的空回复，节点却被标记为健康——一次真实的失败被
                    # 记成了成功，失败计数永远不增长。
                    if first_item is None:
                        raise ProviderError(
                            f"节点 [{provider_key}:{wire_model}] 返回了空流（上游未产出任何内容帧）",
                            status=502,
                            err_type="upstream_error",
                        )

                    # 首包读取成功，说明此 Provider 健康可用
                    await self.cooldown_tracker.record_success(provider_key)

                    # [C2] 槽位回收必须是**可显式调用**的，不能只依赖生成器的
                    # finally：`aclose()` 对一个从未 `__anext__` 过的异步生成器是
                    # 空操作（Python 不会为未启动的生成器运行收尾代码）。而本函数
                    # 在返回之前就已 acquire 了信号量并打开了上游流——调用方若在拉取
                    # 第一帧之前放弃（客户端在响应头到达前断连），槽位会永久泄漏，
                    # 节点并发上限被幽灵流占满，负载均衡随后持续把流量分流走。
                    def _release_sem() -> None:
                        nonlocal sem_held
                        if sem_held:
                            sem_held = False
                            sem.release()

                    # [C3] 关闭本函数打开的那条上游流。
                    #
                    # `aclose()` 对**从未启动**的生成器是空操作，对已耗尽的生成器
                    # 也是空操作，所以这里可以无条件调用：正常跑完的路径逐字节不变。
                    # 危险只在收尾发生在一个**已被取消**的任务里（客户端断连）：
                    # 原地 await 会被立刻二次取消，provider 的 `finally` 一行都执行
                    # 不到。因此派成独立任务再 shield 等待——取消只打断等待，关闭
                    # 任务本身仍会跑完（与 `disconnect_guard.safe_aclose_detached`
                    # 同一条纪律，那边记录了这个坑的实测）。
                    async def _close_raw_stream() -> None:
                        aclose = getattr(raw_stream, "aclose", None)
                        if aclose is None:
                            return
                        try:
                            task = asyncio.ensure_future(aclose())
                        except Exception as exc:  # noqa: BLE001 - 回收必须绝对安全
                            logger.warning("[对流回收] 上游流关闭调度失败 (已忽略): %s", exc)
                            return
                        try:
                            await asyncio.shield(task)
                        except asyncio.CancelledError:
                            # 当前任务被取消：关闭任务继续在后台跑完
                            pass
                        except BaseException as exc:  # noqa: BLE001
                            logger.warning(
                                "[对流回收] 上游流关闭失败 (已忽略): %s: %s",
                                type(exc).__name__, exc,
                            )

                    # [M2] 空闲超时拉取：用 `asyncio.wait` 观察，而不是
                    # `asyncio.wait_for` / `asyncio.timeout()`。
                    #
                    # 后两者靠**取消**实现超时，而取消会被抛进上游生成器体内、把它
                    # 整体关闭（`stream_keepalive` 的 docstring 记录了同一个坑：首个
                    # 沉默之后的数据全部永久丢失）。两者的区别在这里很关键：
                    # 普通超时路径上我们其实**希望**关闭上游（已经判定它死了），
                    # 但取消必须由我们主动、在确认超时之后发起；若由 `wait_for`
                    # 顺手取消，我们就分不清「超时」与「调用方取消」，也没机会先
                    # 把上游流关干净再报错。
                    async def _next_with_idle(stream, idle: float):
                        if idle <= 0:
                            return await stream.__anext__()
                        task = asyncio.ensure_future(stream.__anext__())
                        try:
                            done, _ = await asyncio.wait({task}, timeout=idle)
                        except asyncio.CancelledError:
                            # 调用方取消（客户端断连）：把取消传给上游拉取，
                            # 否则它会作为一个持有上游流的幽灵任务继续挂着。
                            task.cancel()
                            raise
                        if not done:
                            task.cancel()
                            try:
                                await task
                            except BaseException:  # noqa: BLE001 - 取消/收尾异常无关紧要
                                pass
                            raise ProviderError(
                                f"节点 [{provider_key}:{wire_model}] 上游已 "
                                f"{idle:.0f}s 未产出任何数据（流式空闲超时）",
                                status=504,
                                err_type="upstream_timeout",
                            )
                        return task.result()

                    # 构造包装后的安全生成器
                    async def wrapped_stream() -> AsyncIterator[Tuple[str, dict]]:
                        nonlocal first_item, raw_stream
                        idle_timeout = self._stream_idle_timeout()
                        # [S2] 上游完整性三态：True 看到完成标记 / False 协议有标记
                        # 但没看到（截断）/ None 协议没有标记（未知，行为零变化）。
                        completeness: Optional[bool] = None
                        check_completion = completion_check_enabled()
                        try:
                            # 输出首包
                            if first_item is not None:
                                delta, meta = first_item
                                completeness = observe_completion(meta, completeness)
                                # 如果发生了故障转移，在首包元数据注入标记
                                if failover_event:
                                    meta = dict(meta or {})
                                    meta["x-failover-from"] = failover_event.from_provider
                                    meta["x-failover-to"] = failover_event.to_provider
                                    meta["x-failover-model"] = failover_event.to_model
                                    meta["x-failover-ring"] = failover_event.ring.value
                                yield delta, meta

                            # 持续迭代剩余数据包（[M2] 带空闲上限，见上一个函数）
                            while True:
                                try:
                                    item = await _next_with_idle(raw_stream, idle_timeout)
                                except StopAsyncIteration:
                                    # [S2] 「干净耗尽」是唯一**不抛异常**的失败通道。
                                    # 协议有完成标记而我们始终没看到它 → 这段回答
                                    # 是被截断的。原实现在这里直接 break，于是截断
                                    # 与说完走同一条出口：协议处理器补上
                                    # `finish_reason: "stop"` 与 `[DONE]`，客户端
                                    # 收到一个**自称完整**的半截回答，而节点的失败
                                    # 计数保持 0（record_success 记在首包），下一轮
                                    # 它仍是主选 —— 这正是 C3「中途断流」的症状，
                                    # 从不抛异常的那道门进来。
                                    #
                                    # 未知（协议无标记，三家如此）必须照常 break：
                                    # 把未知当截断会让这三家每条正常回答都被误判。
                                    if check_completion and is_incomplete(completeness):
                                        raise ProviderError(
                                            f"节点 [{provider_key}:{wire_model}] 的回答被截断"
                                            f"（上游流结束但未出现协议完成标记）",
                                            status=502,
                                            err_type="upstream_incomplete",
                                        )
                                    break
                                completeness = observe_completion(item[1], completeness)
                                yield item
                        except StopSequenceHit:
                            raise
                        except Exception as stream_err:
                            logger.error(
                                f"[对等互备网格] 节点 [{provider_key}] 流传输中途断开: {stream_err}"
                            )
                            # [STREAM-AUDIT] 中途断流必须记入冷却。
                            # 原实现只打一行日志，节点健康度毫发无损：一个每几次请求
                            # 就吐到一半断掉的节点，失败计数永远是 0，于是下一轮仍然
                            # 被选为主选，客户端反复看到「回复说到一半停住」——这正是
                            # 「稳定性差」的直接来源。风险类拦截（401/429/WAF）由上层
                            # server 侧统一记账，这里只管非风险类，避免同一次失败记两遍。
                            is_risk, _ = is_risk_interception(stream_err)
                            if not is_risk:
                                try:
                                    stream_status = int(getattr(stream_err, "status", 502))
                                    # [S2] 截断必须带着自己的成因进入记账：它走 502，
                                    # 而严重度分类器对非 401/429 缺省回 waf。
                                    is_trunc = (
                                        getattr(stream_err, "err_type", "")
                                        == "upstream_incomplete"
                                    )
                                    await self.cooldown_tracker.record_failure(
                                        provider_key,
                                        stream_status,
                                        str(getattr(stream_err, "message", stream_err)),
                                        # [M2] 流式空闲超时按**超时成因**处置，而不是
                                        # 通用 5xx 的 15s×n（见 `_timeout_cooldown`）。
                                        custom_cooldown=(
                                            self._timeout_cooldown()
                                            if stream_status == 504
                                            else None
                                        ),
                                        is_timeout=(stream_status == 504),
                                        is_truncation=is_trunc,
                                    )
                                except Exception:  # noqa: BLE001 - 记账失败不得掩盖原始错误
                                    pass
                            # 中途断开按原有规范上抛，不产生半包污染
                            raise
                        finally:
                            # [C3] 上游生成器必须**在这里**被显式关闭，不能只归还
                            # 槽位。`raw_stream` 是本函数打开的那条流，它的
                            # `finally` 才是 provider 侧退出 `async with
                            # client.stream(...)`、释放 HTTP/2 连接的地方。
                            #
                            # 此前这里只调 `_release_sem()`，于是：
                            #   * 响应体从未启动 → `ManagedStream.close()` 推进一帧
                            #     让本生成器跑进 `finally`，槽位归还了，可 provider
                            #     生成器仍停在 `yield` 上**永远挂着**；
                            #   * 客户端中途断连 → 同上。
                            # 唯一「关闭」它的时机是 CPython 的引用计数回收——
                            # 那不是一个可以被断言的时机。实测（把流引用一直握在
                            # 手里、禁止 GC）：`opened=1 closed=0`；放手让 GC 跑完
                            # 才变 1/1。真实进程里这条流会带着上游连接一直活到
                            # 垃圾回收，正是「客户端已走、上游仍在生成」的空耗，
                            # 也是连接池被慢慢吃光的来源。
                            #
                            # 必须先关上游、后放槽位：反过来的话，下一个拿到槽位的
                            # 请求会与这条尚未关闭的流并存，节点并发上界当场被突破。
                            # `_close_raw_stream` 自己就是「独立任务 + shield」，
                            # 因此这里在取消路径上也等得到它。
                            try:
                                await _close_raw_stream()
                            except BaseException:  # noqa: BLE001 - 回收不得掩盖原始错误
                                pass
                            _release_sem()

                    sem_handed_off = True
                    return (
                        ManagedStream(wrapped_stream(), close=_release_sem),
                        failover_event,
                        provider,
                    )
                finally:
                    # 观察期闸门只到首包为止：无论成功交接与否，走出握手区就释放。
                    if probation_held:
                        probation_lock.release()
                    # 握手失败 / 空流 / 构造异常：信号量尚未交接给流，就地归还。
                    if sem_held and not sem_handed_off:
                        sem.release()

            except StopSequenceHit:
                # [PROTOCOL] 停用词命中是客户端要求「到此为止」的正常终止，不是上游
                # 故障：若落进下面的兜底分支，会被记成 502 并给健康节点上冷却。
                raise
            except (ProviderError, Exception) as exc:
                # [BUDGET-3] 零上游拒绝：预算在进入尝试之前就不够，一个节点都没碰过。
                # 不能落进下面的记账分支（那会给一个从未被调用的节点上冷却），也不能
                # 继续往后故障转移（后续候选同样没有份额）。直接如实上抛。
                if getattr(exc, "zero_upstream", False):
                    raise
                if primary_exc is None:
                    primary_exc = exc
                status_code = getattr(exc, "status", 502)
                err_msg = str(getattr(exc, "message", str(exc)))
                err_summary = f"[{curr_provider_key}:{curr_wire_model}] -> HTTP {status_code}: {err_msg}"
                composite_errors.append(err_summary)
                logger.warning(f"[对等互备网格] 尝试失败: {err_summary}")

                # 记录失败并进入动态冷却
                # [M2] 超时（握手 / 空闲）用专门的冷却时长，其余维持原判据。
                #
                # [BUDGET-3] 被**我们家**的预算裁出来的超时不在此列：份额 0.1s 到了
                # 只说明我们只肯等 0.1s，不说明节点有故障。按节点失败记账会让一个
                # 健康节点白拿 60s 冷却（实测：0.4s 应答的健康节点 → 冷却 60s、
                # failure_count 1），下一轮请求绕开它——我们自己的预算决定被写成了
                # 上游的罪状，正是本项目反复记录的那类谎报。继续故障转移（这次确实
                # 没拿到首包），但不进冷却、不增失败计数。
                if _is_budget_capped(exc):
                    # 单次不记账（那是诬告），但**连续**被裁短是一条不同的事实：
                    # 「我们在全部可用份额里都没等到首包」。没有这本账，一个挂死的
                    # 节点会在每次请求上重新被白等一整份份额且永不收敛（实测四轮
                    # 全是 1.00s）。升级仍按上游超时处置，不进风控台账。
                    escalated = await self.cooldown_tracker.note_budget_capped(
                        curr_provider_key
                    )
                    logger.warning(
                        "[握手预算] 节点 [%s] 的本次上限被总预算裁剪至 %.1fs（节点本身"
                        "未表现出故障，单次不计冷却%s），继续故障转移",
                        curr_provider_key,
                        getattr(exc, "handshake_timeout", 0.0) or 0.0,
                        f"；已连续 {self.cooldown_tracker.budget_capped_limit()} 次，"
                        f"升级为上游超时冷却 {escalated:.1f}s"
                        if escalated
                        else "",
                    )
                else:
                    is_timeout = status_code == 504
                    await self.cooldown_tracker.record_failure(
                        curr_provider_key,
                        status_code,
                        err_msg,
                        custom_cooldown=self._timeout_cooldown() if is_timeout else None,
                        is_timeout=is_timeout,
                    )

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
                    # [DAY-ZERO] 措辞必须如实：如果这一环的节点**根本没有配置凭证**，
                    # 那它们不是「失效」——「失效」意味着曾经可用、现在坏了，会让运维
                    # 去排查上游与网络，而真实动作是第一次登录。全新克隆的第一次请求
                    # 走的就是这条路径，这是产品的第一天，措辞错了就会把用户带偏。
                    _unconfigured = [
                        pk for pk in tried_providers if not _provider_is_configured(pk)
                    ]
                    if _unconfigured:
                        report += (
                            f"\n注意：其中 {len(_unconfigured)} 个节点（"
                            + "、".join(sorted(_unconfigured))
                            + "）在本机**尚未配置凭证**——它们从未被调用过，"
                            "这不是上游故障，也未被上游风控拦截。请先执行 "
                            "`.venv/bin/python login.py all` 完成登录（或为相应节点配置"
                            "环境变量），配置完成后这些节点会自动恢复为可派发候选。"
                        )
                    # [BUDGET-2] 预算现在会被**公平分摊**给各候选，因此常见结局不再是
                    # 「预算耗尽」而是「每个候选都只分到一小份、各自超时后环内全灭」。
                    # 此时报错若只列失败，运维无法区分「预算太小」和「节点真的全挂」，
                    # 也就不知道该调预算还是去查网络。把预算与份额一并写出（仅在预算
                    # 确实参与了本次判决时，否则逐值保持旧文案）。
                    if deadline is not None:
                        report += (
                            f"\n注：握手总预算 {handshake_budget:.0f}s 已按剩余候选公平"
                            f"分摊（每个候选只有其中一份），因此单次超时时间远短于 "
                            f"GATEWAY_FIRST_CHUNK_TIMEOUT。若节点实际健康但响应较慢，"
                            f"请调大 GATEWAY_HANDSHAKE_BUDGET。"
                        )
                    logger.error(f"[对等互备网格] 全部后备节点熔断: {report}")
                    status = getattr(primary_exc, "status", 502) if primary_exc else 502
                    err_type = getattr(primary_exc, "err_type", "upstream_error") if primary_exc else "upstream_error"
                    # [H7] 环已走完 ⇒ 这是一个**确定性**结论，不是「上游抖了一下」。
                    # 同一个环再走一遍不会得到不同答案，而客户端会立刻重来。
                    # 实测（真实 Claude Code，五节点全挂、同样的 504 与文案，唯一变量
                    # 是响应头）：仅 504 → 240s 内 8 次请求；加 `x-should-retry:false`
                    # → 18s、2 次请求、EXIT=1。8 次 × 5 候选 = 40 次打在已知全挂的环上
                    # 的上游调用，正是风控线一直在消灭的风暴形态。
                    # 退避时长取抛出点真正知道的窗口：这些节点刚被判定超时，各自的
                    # 超时冷却还剩多久是可问的，报最短的那个（最早可能恢复的时刻）。
                    # 只有超时这一种终局在这里是确定性的：5xx/WAF/401 各自的语义由
                    # 它们自己决定，不在此处一并改写。
                    if err_type == "upstream_timeout":
                        raise ProviderError(
                            report,
                            status=status,
                            err_type=err_type,
                            retry_after=min(
                                [
                                    r
                                    for r in (
                                        self.cooldown_tracker.get_remaining_cooldown(p)
                                        for p in tried_providers
                                    )
                                    if r and r > 0
                                ],
                                default=self._timeout_cooldown(),
                            ),
                        )
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
