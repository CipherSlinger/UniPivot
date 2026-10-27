"""风控拦截档案与规避策略研判 (providers/risk_dossier.py)。

实现 [P1] OPT-023：
1. 全量拦截档案 (All-Time Interception Dossier)：
   - `providers/risk_ledger.py` 的 `recent_interceptions()` 是**滑动窗口**（默认 24h），
     适合驱动实时避让，但不适合回答「这家到底被拦过几次、都是什么原因」。
     窗口一滑过去，运维就再也看不到历史——尤其在各家封禁窗口以「天」为单位时，
     24h 窗口几乎必然在封禁解除前就把证据丢光了。
   - 本模块以**全量台账**（受 `GATEWAY_RISK_LEDGER_MAX_LINES` 有界约束）聚合每个节点
     的累计拦截次数、首/末次时间、严重度分布与最近一次的真实上游原文，
     使封禁历史可审计、可复盘。
2. 姿态研判 (Posture Derivation)：
   - 把分散在四处的事实——封禁租约 (`quarantine`)、软冷却 (`CooldownTracker`)、
     恢复观察期 (OPT-022)、风控证据驱动的步调倍率 (OPT-014)——合成**单一姿态**：
     `quarantined` > `cooling` > `probation` > `watch` > `clear`，附「是否可派发」布尔值。
   - 运维只需读一个字段即可判断该节点此刻的真实处境，无需自行推断多处状态。
3. 规避策略研判 (Avoidance Verdict) —— 直接回答「能否绕开，不能绕开则如何避免」：
   - 按严重度分级给出确定性结论：技术手段能否绕开、**重新登录能否恢复**、
     是否会自动恢复、以及应当采取的具体动作。
   - 结论是保守且诚实的：账号级封禁无法通过任何客户端手段绕开，重登亦无效，
     唯一正确策略是零误触等待平台解封；重登只能恢复「令牌/会话失效」与「人机验证」两类。

设计约束：本模块**只读**，不修改任何风控状态（不解除租约、不清冷却、不写台账），
因此可安全暴露在诊断与运维查询路径上，不存在因查询而扰动避让语义的风险。
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

from .failover import PROVIDER_RISK_SPECS, get_adaptive_pacing_multiplier
from .quarantine import QuarantineStore, get_quarantine_store
from .risk_ledger import RiskLedger, get_risk_ledger

logger = logging.getLogger("gateway.risk_dossier")

# ---------------------------------------------------------------- 姿态分级

POSTURE_QUARANTINED = "quarantined"
POSTURE_DISABLED = "disabled"
POSTURE_UNCONFIGURED = "unconfigured"
POSTURE_LONG_AVOIDANCE = "long_avoidance"
POSTURE_COOLING = "cooling"
POSTURE_PROBATION = "probation"
POSTURE_WATCH = "watch"
POSTURE_CLEAR = "clear"

# 姿态由重到轻；派生取**最重**的命中项（fail-closed）。
POSTURE_ORDER: tuple = (
    POSTURE_QUARANTINED,
    POSTURE_DISABLED,
    POSTURE_UNCONFIGURED,
    POSTURE_LONG_AVOIDANCE,
    POSTURE_COOLING,
    POSTURE_PROBATION,
    POSTURE_WATCH,
    POSTURE_CLEAR,
)

POSTURE_LABELS: Dict[str, str] = {
    POSTURE_QUARANTINED: "长期封禁（硬阻断，零派发）",
    POSTURE_DISABLED: "运维声明出局（DISABLED_PROVIDERS，硬阻断，零派发）",
    POSTURE_UNCONFIGURED: "本机未配置凭证（硬阻断，零派发；执行 login.py 后自动恢复）",
    POSTURE_LONG_AVOIDANCE: "长效避让（升级后的长冷却，硬阻断，零派发）",
    POSTURE_COOLING: "避让冷却中（自动，时长有界）",
    POSTURE_PROBATION: "恢复观察期（串行探路 + 步调放大）",
    POSTURE_WATCH: "监视期（近期有拦截证据，步调已放慢）",
    POSTURE_CLEAR: "正常（无近期风控记录）",
}

# 只有硬封禁与长效避让不可派发；短软冷却保留「最后一搏」语义以保可用性（见 OPT-021）。
# [OPT-031] 之所以把「长效避让」单列一档而不是并入 cooling：派发边界的处置相反——
# 短冷却仍会被派发（最后一搏），升级后的长冷却已被硬阻断、零上游调用。若沿用同一个
# 姿态与 dispatchable=True，档案就在派发事实上说了谎，运维会据此误判节点仍可用。
# [OPT-035] 同理把「运维声明出局」单列一档：派发边界（OPT-033）对 DISABLED_PROVIDERS
# 是硬阻断，而档案此前对它给出「正常（可派发）」——档案在派发事实上说谎的第二处。
# 排序上刻意把 quarantined 放在 disabled **之前**（危险者优先）：一个既被封禁又被运维
# 禁用的节点（正是本机 deepseek 的真实状态）若先报「出局」，运维最自然的反应是「那就把
# 禁用解除」，从而把一个**被封禁**的节点重新放回派发路径——静默且有害。先报「封禁」
# 则给出「等待解封」的正确动作，而「出局」作为条目标记仍然可见。
POSTURE_DISPATCHABLE: Dict[str, bool] = {
    POSTURE_QUARANTINED: False,
    POSTURE_DISABLED: False,
    POSTURE_UNCONFIGURED: False,
    POSTURE_LONG_AVOIDANCE: False,
    POSTURE_COOLING: True,
    POSTURE_PROBATION: True,
    POSTURE_WATCH: True,
    POSTURE_CLEAR: True,
}

# 进入 watch 姿态的观察窗口（秒）：最近一次拦截在此时间内即降低置信度。
_WATCH_WINDOW = 3600.0


# ---------------------------------------------------------------- 规避研判

# 严重度 -> 规避策略。这是本模块的核心产出，直接回答
# 「记录每家的风控拦截，看是否有办法绕开，没有办法绕开的话，思考如何避免」。
AVOIDANCE_MATRIX: Dict[str, Dict[str, Any]] = {
    "ratelimit": {
        "label": "频率限制 (429 / Too Many Requests)",
        "technically_bypassable": False,
        "recoverable_by_relogin": False,
        "auto_recovers": True,
        "typical_span": "秒级 ~ 分钟级",
        "strategy": "无法「绕开」，只能「降低触发概率」：保持避让步调（OPT-014 证据驱动放大）"
                    "并等待窗口自然衰减，同时将流量分流至同能力环的其他节点。频率限制是"
                    "瞬时的，不需要也不必重新登录。",
    },
    "waf": {
        "label": "WAF 人机验证 (RGV587 / 验证码)",
        "technically_bypassable": False,
        "recoverable_by_relogin": True,
        "auto_recovers": False,
        "typical_span": "分钟级 ~ 小时级",
        "strategy": "无法用程序绕过（这正是人机验证的设计目的，任何自动破解都属对抗性"
                    "且会加重风控评分）。正确做法是退出自动路径，由人工执行 "
                    "`python login.py <provider>` 完成滑块验证，重登可恢复会话。"
                    "重复出现时网关会升级为长效避让（与认证级同构）——因为持续重试"
                    "只会加重标记，而这一类拦截**没有任何自动恢复路径**；人工完成"
                    "验证后凭证指纹变化，网关自动解除避让并转入串行探路重新验证。",
    },
    "auth": {
        "label": "认证失效 (令牌/会话过期)",
        "technically_bypassable": False,
        "recoverable_by_relogin": True,
        "auto_recovers": False,
        "typical_span": "需人工介入",
        "strategy": "非风控封禁，而是凭证生命周期到期。执行 `python login.py <provider>` "
                    "重新登录取令牌即可恢复；网关亦会尝试静默刷新（OPT-013），"
                    "但刷新本身同样受避让门控约束，避免刷新动作成为封禁窗口的触碰通道。",
    },
    "ban": {
        "label": "账号级封禁 (平台侧封禁窗口)",
        "technically_bypassable": False,
        "recoverable_by_relogin": False,
        "auto_recovers": True,
        "typical_span": "天级",
        "strategy": "**无法绕开，重新登录亦无效**——封禁作用在账号本身而非令牌，换令牌、"
                    "换会话、换指纹都不可能解除平台侧封禁，任何尝试只会延长封禁。唯一"
                    "正确策略是识别、退避并在窗口内**零误触**，等待平台自然解封。"
                    "网关以运维长期封禁租约 (`POST /v1/admin/quarantine`) 表达该窗口，"
                    "并在租约解除后以恢复观察期 (OPT-022) 渐进试探，避免解封瞬间的并发"
                    "洪峰把账号二次打进封禁。",
    },
    # [M2] 上游超时：不是风控拦截，aggregate 严重度分类器对它的缺省回答是 waf，
    # 会把一次「上游挂了」误报成「需要人工完成滑块验证」。处置建议必须指向网络与
    # 上游可用性，而不是让运维去登录一个根本没有拦截他的平台。
    "timeout": {
        "label": "上游超时 (握手或流式空闲超时)",
        "technically_bypassable": False,
        "recoverable_by_relogin": False,
        "auto_recovers": True,
        "typical_span": "分钟级",
        "strategy": "这不是风控拦截，而是上游迟迟不产出数据（握手超过 "
                    "`GATEWAY_FIRST_CHUNK_TIMEOUT`，或两条数据帧间隔超过 "
                    "`GATEWAY_STREAM_IDLE_TIMEOUT`）。网关已按上游故障处置并故障转移。"
                    "**重新登录无效**，也不应破解任何验证码；若频繁出现，应检查上游"
                    "服务状态与本地出网质量，而非调整风控参数。",
    },
    # [S2] 回答被截断：上游连上了、说了一半就停了。与超时同属上游故障。
    "truncated": {
        "label": "回答被截断 (上游流结束但未出现协议完成标记)",
        "technically_bypassable": False,
        "recoverable_by_relogin": False,
        "auto_recovers": True,
        "typical_span": "分钟级",
        "strategy": "这不是风控拦截：上游连接正常建立、吐了一部分内容就结束了，"
                    "且其协议的完成标记（qwen 的 `status == complete`、kimi 的 "
                    "`all_done`）始终没有出现。网关已按上游故障记冷却，使该节点"
                    "不再优先当主选。**重新登录无效**，也不应破解任何验证码；"
                    "若频繁出现，应检查上游服务状态与本地出网质量。注意另外三家"
                    "(deepseek/doubao/glm) 协议里没有完成标记，它们的截断无法被"
                    "判定——那是协议的限制，不是漏判。",
    },
    # [DAY-ZERO] 未配置凭证：本机的配置缺口，连上游都没碰过。
    # 与 `disabled` 同属「不是平台风控」，但方向相反：disabled 是运维**主动**排除，
    # 这里是运维**还没**配置。两者的正确动作因此完全不同（改配置 vs 执行登录）。
    "unconfigured": {
        "label": "本机未配置凭证 (尚未 `login.py`)",
        "technically_bypassable": False,
        "recoverable_by_relogin": True,
        "auto_recovers": True,
        "typical_span": "一次登录动作",
        "strategy": "这不是上游风控，也不是上游故障：本机**根本没有该节点的凭证**，"
                    "网关从未向该厂商发出过任何请求。正确动作是执行 "
                    "`.venv/bin/python login.py <provider>`（或配置对应的 `*_AUTH_TOKEN` "
                    "环境变量）完成第一次登录。网关为该节点保留的是一段**固定短冷却**、"
                    "**不随次数升级**，且凭证一出现就自动解除（不需要重启）——"
                    "因为这一档若借用认证级的升级路径，就会落入「只有人工重登能修好、"
                    "而重登产生的状态又锁死着自己」的自锁。",
    },
    # [OPT-035] 运维声明出局：不是平台的风控，而是运维把节点排除出本次部署。
    # 措辞刻意与「封禁」和「认证失效」都不同，因为它们**都**建议重登，而这里重登
    # 完全无效——这正是本项目反复修复的「误导运维」缺陷类型（OPT-026/027/029/031）。
    "disabled": {
        "label": "运维声明出局 (DISABLED_PROVIDERS)",
        "technically_bypassable": False,
        "recoverable_by_relogin": False,
        "auto_recovers": False,
        "typical_span": "需人工修改配置",
        "strategy": "这不是平台风控，而是运维的部署声明：该节点被 `DISABLED_PROVIDERS` "
                    "显式排除，派发边界对其**硬阻断**（零上游调用）。解除方式只有一处——"
                    "修改 `DISABLED_PROVIDERS` 或程序化解除声明；**重新登录无效**，"
                    "自动避让也不会解除它（它不是冷却，没有会到期的窗口）。"
                    "解除前请先确认该节点本身是否仍处于封禁或人机验证避让，"
                    "否则会把一个仍受风控限制的节点直接放回派发路径。",
    },
}

# 严重度降级映射：租约（人工声明的封禁）在研判中一律视为 ban。
_SEVERITY_TO_VERDICT = {
    "ratelimit": "ratelimit",
    "waf": "waf",
    "auth": "auth",
    # [M2] 超时是上游故障，不是风控；映射到独立的处置建议。
    "timeout": "timeout",
    # [S2] 截断同理：上游故障，重新登录无效。
    "truncated": "truncated",
    # [DAY-ZERO] 未配置凭证：本地配置缺口，不是平台风控。
    # 不加这一行会落到 `_SEVERITY_TO_VERDICT.get(...) or "ratelimit"` 的缺省上，
    # 于是档案会告诉运维「频率限制，无需人工介入，等待窗口衰减即可」——而真实动作
    # 是第一次登录。这是本项目反复修复的「误导运维」缺陷类型
    # （OPT-026/027/029/031/035），换了个成因。
    "unconfigured": "unconfigured",
}

# [M2] 台账**无法表达**的成因：超时从不入账（它不是风控拦截），因此研判不能
# 只在台账里找它，否则「此刻正因超时避让」会被报成台账里的旧成因（通常是 waf），
# 把运维引去登录一个从未拦截过他的平台。只有这类成因才允许覆盖台账多数成因。
# [DAY-ZERO] 未配置同理，而且更彻底：这类节点**从未被上游看见过**，台账里必然
# 一条记录都没有，于是档案的缺省答案会是「无拦截记录」，同样不对。
_LIVE_ONLY_CAUSES = ("timeout", "truncated", "unconfigured")


class RiskDossier:
    """风控拦截全量档案与规避策略研判器（只读）。"""

    def __init__(
        self,
        ledger: Optional[RiskLedger] = None,
        quarantine: Optional[QuarantineStore] = None,
        cooldown_tracker: Any = None,
    ) -> None:
        self.ledger = ledger if ledger is not None else get_risk_ledger()
        self.quarantine = quarantine if quarantine is not None else get_quarantine_store()
        # CooldownTracker 由 server.py 在构造 failover_router 后注入
        # （与 conversation_gc 的 set_resolver / set_cooldown_gate 同一约定）。
        self.cooldown_tracker = cooldown_tracker

    def set_cooldown_tracker(self, tracker: Any) -> None:
        """注入 CooldownTracker（软冷却 / 封禁 / 观察期读数的唯一来源）。"""
        self.cooldown_tracker = tracker

    def is_disabled(self) -> bool:
        """判断是否关闭档案装配（关闭后仍可查询，但姿态退化为仅凭台账推断）。"""
        return os.getenv("GATEWAY_DISABLE_RISK_DOSSIER", "").strip().lower() in ("1", "true", "yes")

    # ------------------------------------------------------------ 内部读取

    def _known_providers(self) -> List[str]:
        """所有需要出现在档案中的节点：静态风险表 ∪ 台账 ∪ 生效租约 ∪ 当前冷却。"""
        keys = set(PROVIDER_RISK_SPECS.keys())
        try:
            for entry in self.ledger._read_tail(self.ledger.max_scan_lines):
                pk = entry.get("provider")
                if pk:
                    keys.add(str(pk).lower())
        except Exception as exc:  # noqa: BLE001 - 只读聚合，绝不因台账异常而中断
            logger.warning("档案读取台账失败 (已忽略): %s: %s", type(exc).__name__, exc)
        try:
            keys.update(str(k).lower() for k in self.quarantine.list_active())
            if self.cooldown_tracker is not None:
                keys.update(str(k).lower() for k in self.cooldown_tracker.get_all_cooldowns())
        except Exception as exc:  # noqa: BLE001
            logger.warning("档案读取冷却状态失败 (已忽略): %s: %s", type(exc).__name__, exc)
        return sorted(k for k in keys if k)

    def _all_time(self, provider_key: str) -> Dict[str, Any]:
        """该节点的全量拦截统计（非滑动窗口）。"""
        empty = {
            "total": 0,
            "first_ts": None,
            "last_ts": None,
            "by_severity": {},
            "last_reason": "",
            "last_evidence": "",
        }
        try:
            entries = self.ledger._read_tail(self.ledger.max_scan_lines)
        except Exception:  # noqa: BLE001
            return empty
        total = 0
        first_ts: Optional[float] = None
        last_ts: Optional[float] = None
        by_severity: Dict[str, int] = {}
        last_reason = ""
        last_evidence = ""
        for entry in entries:
            if str(entry.get("provider") or "").lower() != provider_key:
                continue
            ts = entry.get("ts")
            ts = float(ts) if isinstance(ts, (int, float)) else 0.0
            total += 1
            if first_ts is None or ts < first_ts:
                first_ts = ts
            if last_ts is None or ts >= last_ts:
                last_ts = ts
                last_reason = str(entry.get("reason", ""))
                last_evidence = str(entry.get("evidence", ""))
            sev = str(entry.get("severity") or "unknown")
            by_severity[sev] = by_severity.get(sev, 0) + 1
        return {
            "total": total,
            "first_ts": round(first_ts, 3) if first_ts else None,
            "last_ts": round(last_ts, 3) if last_ts else None,
            "by_severity": by_severity,
            "last_reason": last_reason,
            "last_evidence": last_evidence,
        }

    def _avoidance(self, provider_key: str, stats: Dict[str, Any]) -> Dict[str, Any]:
        """按主要严重度给出规避研判；有租约时一律按账号级封禁研判。"""
        quarantine = self._quarantine_state(provider_key)
        if quarantine["active"]:
            verdict_key = "ban"
            # [OPT-027] 自动租约是**网关推断**的封禁，其唯一依据是「已试过一枚更新的
            # 凭证且仍被拒」。措辞必须区分，否则运维会误以为它是人类声明而不敢重登
            # ——而对自动租约来说，重登恰恰是唯一正确的解除动作。
            if quarantine.get("source") == "auto":
                reason = ("该节点持有**自动推断**的封禁租约（依据：凭证已被证明更新过之后"
                          "仍被 401 拒绝）；重新登录取新令牌即可自动解除并重新验证")
            else:
                reason = "该节点处于运维长期封禁租约，按平台侧账号级封禁研判"
        elif self._is_disabled(provider_key):
            # [OPT-035] 声明出局优先于「按拦截严重度研判」：对一个运维已排除的节点谈
            # 「主要症结是频控/认证」会把运维引向重登，而重登对声明出局完全无效。
            verdict_key = "disabled"
            reason = "该节点被运维声明出局（DISABLED_PROVIDERS），非平台风控"
        elif not self._is_configured(provider_key):
            # [READINESS] 无凭证也必须先于「按拦截严重度研判」：这样的节点**从未被上游
            # 看见过**，台账里必然一条记录都没有，于是缺省答案会是「无拦截记录，可正常
            # 派发」——而它其实连一次请求都发不出去。处置与 disabled 相反：不是改配置，
            # 而是去登录（`recoverable_by_relogin: True`）。
            verdict_key = "unconfigured"
            reason = "本机没有该节点的凭证（部署尚未完成），与平台风控无关"
        else:
            # 取出现次数最多的严重度作为主要症结；无记录则按最轻的 ratelimit。
            by_sev = stats.get("by_severity") or {}
            dominant = max(by_sev.items(), key=lambda kv: kv[1])[0] if by_sev else ""
            # [M2] 超时**不入台账**（它不是风控拦截，见 `record_failure` 的 M2 注释），
            # 因此台账永远无法给出这个成因。一个**此刻**正因为超时避让的节点，若档案
            # 照旧报「主要症结 waf」，运维会照着去登录一个从未拦截过他的平台、甚至去找
            # 验证码——处置方向完全相反。这里只让**台账无法表达**的成因覆盖台账多数成因；
            # 其余情况仍以全量台账研判，避免一次 15 秒的频控退避抹掉整段风控历史。
            live_cause = self._live_cooldown_cause(provider_key)
            source = live_cause if live_cause in _LIVE_ONLY_CAUSES else dominant
            verdict_key = _SEVERITY_TO_VERDICT.get(source) or "ratelimit"
            if source and source != dominant:
                reason = (
                    f"当前冷却成因：{source}"
                    f"（全量台账主要症结：{dominant or '无'}）"
                )
            else:
                reason = f"主要症结：{dominant or '无拦截记录'}"
        matrix = AVOIDANCE_MATRIX[verdict_key]
        return {
            "verdict": verdict_key,
            "reason": reason,
            **matrix,
        }

    def _live_cooldown_cause(self, provider_key: str) -> str:
        """[M2] 该节点**此刻**的冷却成因（无冷却/不可读时返回空串）。

        判读委托给 `CooldownTracker.cooldown_cause()`，即全项目唯一的冷却成因判读者，
        不在档案里另写一份——两处各写一遍必然漂移（本项目的既有缺陷类型）。
        """
        tracker = self.cooldown_tracker
        if tracker is None:
            return ""
        try:
            return str(tracker.cooldown_cause(provider_key) or "")
        except Exception:  # noqa: BLE001 - 判读失败不改变台账研判
            return ""

    def _is_disabled(self, provider_key: str) -> bool:
        """[OPT-035] 该节点是否被运维声明出局（判读委托给唯一判读者，查询失败按未禁用）。"""
        tracker = self.cooldown_tracker
        if tracker is None:
            return False
        try:
            return bool(tracker.is_disabled(provider_key))
        except Exception:  # noqa: BLE001 - 判读失败绝不误报「出局」
            return False

    def _is_configured(self, provider_key: str) -> bool:
        """[READINESS] 本机是否有该节点的凭证（委托给唯一判据；查询失败按「已配置」）。

        失败方向与 `session_store.is_configured` 一致，也是刻意的：一次读取异常绝不能
        让档案把一个**已配置**的节点报成「未配置」，进而给出「去登录」这条无用建议。
        """
        try:
            from session_store import is_configured

            return bool(is_configured(provider_key))
        except Exception:  # noqa: BLE001 - 判读失败视作已配置（保守）
            return True

    def _quarantine_state(self, provider_key: str) -> Dict[str, Any]:
        try:
            remaining = self.quarantine.remaining(provider_key)
        except Exception:  # noqa: BLE001
            remaining = 0.0
        if remaining <= 0.0:
            return {
                "active": False, "remaining_hours": 0.0, "until": None,
                "reason": "", "source": "",
            }
        active = self.quarantine.list_active()
        info = active.get(provider_key, {})
        return {
            "active": True,
            "remaining_seconds": round(remaining, 1),
            "remaining_hours": round(remaining / 3600.0, 2),
            "until": info.get("until"),
            "reason": info.get("reason", ""),
            # [OPT-027] 租约来源："operator"（人类声明，重登亦无效，只能等平台解除）
            # 或 "auto"（网关由「新凭证仍被拒」推断而来，出现更新的凭证即自动解除）。
            # 二者对运维的动作要求不同，档案必须能区分，否则会误导。
            "source": info.get("source", "operator"),
        }

    def _cooldown_state(self, provider_key: str) -> float:
        if self.cooldown_tracker is None:
            return 0.0
        try:
            return float(self.cooldown_tracker.get_remaining_cooldown(provider_key))
        except Exception:  # noqa: BLE001
            return 0.0

    def _cooldown_cause(self, provider_key: str) -> str:
        """[OPT-029] 当前冷却的成因（`auth` / `waf` / `ratelimit` / 空）。

        档案必须能区分成因才能给出**正确**的处置建议：频控是瞬时可自愈的（等待即可），
        认证失效可由重登修复，而人机验证只有人工完成验证才能解除。混为一谈会让运维
        照「无需人工介入」的建议处理一个只有人能解决的节点。
        """
        if self.cooldown_tracker is None:
            return ""
        try:
            return str(self.cooldown_tracker.cooldown_cause(provider_key) or "")
        except Exception:  # noqa: BLE001 - 只读查询失败绝不影响档案装配
            return ""

    def _probation_state(self, provider_key: str) -> Dict[str, Any]:
        if self.cooldown_tracker is None:
            return {"active": False, "successes": 0, "required": 0, "pacing_factor": 1.0}
        try:
            active = bool(self.cooldown_tracker.is_in_probation(provider_key))
            states = self.cooldown_tracker.get_probation_states()
        except Exception:  # noqa: BLE001
            return {"active": False, "successes": 0, "required": 0, "pacing_factor": 1.0}
        info = states.get(provider_key, {})
        return {
            "active": active,
            "successes": int(info.get("successes", 0)),
            "required": int(info.get("required_successes", 0)),
            "pacing_factor": float(info.get("pacing_factor", 1.0)),
        }

    def _pacing_state(self, provider_key: str) -> Dict[str, Any]:
        meta = PROVIDER_RISK_SPECS.get(provider_key)
        base = float(meta.min_call_interval) if meta else 0.0
        try:
            adaptive = float(get_adaptive_pacing_multiplier(provider_key))
        except Exception:  # noqa: BLE001
            adaptive = 1.0
        if self.cooldown_tracker is not None:
            try:
                effective = float(self.cooldown_tracker.compute_pacing_interval(provider_key))
            except Exception:  # noqa: BLE001
                effective = base * adaptive
            try:
                probation = float(self.cooldown_tracker.probation_pacing_multiplier(provider_key))
            except Exception:  # noqa: BLE001
                probation = 1.0
        else:
            effective = base * adaptive
            probation = 1.0
        return {
            "base_interval_s": round(base, 4),
            "adaptive_multiplier": round(adaptive, 4),
            "probation_multiplier": round(probation, 4),
            "effective_interval_s": round(effective, 4),
        }

    def derive_posture(
        self,
        provider_key: str,
        now: Optional[float] = None,
        stats: Optional[Dict[str, Any]] = None,
    ) -> str:
        """派生单一姿态（fail-closed：命中即取最重者）。"""
        now = time.time() if now is None else now
        if self._quarantine_state(provider_key)["active"]:
            return POSTURE_QUARANTINED
        # [OPT-035] 运维声明出局（DISABLED_PROVIDERS）在派发边界是硬阻断（OPT-033），
        # 档案必须与之一致。排在租约**之后**：既被封禁又被禁用的节点先报「封禁」，
        # 避免运维看到「出局」就去解除禁用、把一个被封禁的节点放回派发路径。
        try:
            if self.cooldown_tracker is not None and self.cooldown_tracker.is_disabled(provider_key):
                return POSTURE_DISABLED
        except Exception:  # noqa: BLE001 - 判定失败退回既有姿态，绝不误报可用
            pass
        # [READINESS] 本机没有凭证时派发边界是硬阻断（零上游调用），档案必须与之一致。
        # 排在「禁用」之后、「长效避让」之前：无凭证是配置缺口，比激进避让更根本——
        # 一个既无凭证又处于避让的节点，先要让运维知道该去登录。
        #
        # 这条不是为了补一个姿态枚举：此前全新克隆（session/ 为空）的档案对五个节点
        # 一律报 `posture=clear, dispatchable=True`，而真实派发全部在本地构造处失败。
        # 这正是 OPT-031/035 记录过的同一类谎报，只是成因从「避让/声明」换成了「缺凭证」。
        try:
            from session_store import is_configured

            if not is_configured(provider_key):
                return POSTURE_UNCONFIGURED
        except Exception:  # noqa: BLE001 - 判定失败退回既有姿态（失败方向：不误报阻断）
            pass
        # [OPT-031] 升级后的长效避让（小时~天级）在派发边界已被硬阻断，必须先于
        # 「冷却中」判定，否则档案会给出一个可派发、实则零派发的错误姿态。
        try:
            if self.cooldown_tracker.is_in_long_avoidance(provider_key):
                return POSTURE_LONG_AVOIDANCE
        except Exception:  # noqa: BLE001 - 判定失败退回既有的「冷却中」姿态，绝不误报可用
            pass
        if self._cooldown_state(provider_key) > 0.0:
            return POSTURE_COOLING
        if self._probation_state(provider_key)["active"]:
            return POSTURE_PROBATION
        stats = stats if stats is not None else self._all_time(provider_key)
        last_ts = stats.get("last_ts")
        if isinstance(last_ts, (int, float)) and (now - float(last_ts)) <= _WATCH_WINDOW:
            return POSTURE_WATCH
        pacing = self._pacing_state(provider_key)
        if pacing["adaptive_multiplier"] > 1.0:
            return POSTURE_WATCH
        return POSTURE_CLEAR

    # ------------------------------------------------------------ 推荐动作

    def recommended_actions(self, provider_key: str, posture: str, entry: Dict[str, Any]) -> List[str]:
        """给出可执行的下一步（按优先级排序），直接回应「如何避免」。"""
        if posture == POSTURE_DISABLED:
            return [
                "该节点被运维声明出局（DISABLED_PROVIDERS），派发边界对其硬阻断、零上游调用",
                "如需恢复使用：修改 DISABLED_PROVIDERS 或程序化解除声明后重启网关；重登无效",
                "解除前先确认该节点本身是否仍处于封禁/人机验证避让，避免把受控节点放回派发路径",
            ]
        if posture == POSTURE_UNCONFIGURED:
            # 与 POSTURE_DISABLED 的处置**相反**：一个是「人改了配置」，一个是「人还没
            # 登录」。这是本项目反复记录的「把成因混为一谈会让运维去做无用功」——
            # 对无凭证的节点说「修改 DISABLED_PROVIDERS」是彻底的误导。
            return [
                "本机没有该节点的凭证：派发边界对其硬阻断，零上游调用（这与风控无关）",
                f"如需恢复使用：执行 `.venv/bin/python login.py {provider_key}` 完成扫码登录，"
                f"或为该厂商设置 *_AUTH_TOKEN 环境变量",
                "登录后无需重启：下一次请求即会重新问一次凭证（判据是即时的，不缓存）",
            ]
        if posture == POSTURE_QUARANTINED:
            q = entry["quarantine"]
            hours = q.get("remaining_hours", 0.0)
            # [OPT-027] 自动租约与运维租约的处置**相反**，提示必须分开给：
            # 自动租约的依据是「更新的凭证仍被拒」，因此重登正是解除它的正确动作；
            # 对运维租约而言重登毫无意义且会延长封禁。混用会让运维按其一的建议
            # 去处理另一种，两种情况下都是错的。
            if q.get("source") == "auto":
                return [
                    f"该封禁由**网关自动推断**而来（剩余约 {hours} 小时）：依据是「凭证已被"
                    f"证明更新过之后仍被 401 拒绝」，即凭证过期这一成因已被排除",
                    f"若平台封禁确实已解除，执行 `.venv/bin/python login.py {provider_key}` "
                    f"重新登录取新令牌即可——出现更新的凭证时该租约会**自动解除**并进入恢复观察期",
                    "若确认平台仍在封禁，则无需任何动作：租约到期或平台解封后再次重登即可自愈",
                    "注意：账号级封禁本身重登无效。此处重登之所以有效，是因为它同时充当"
                    "「解除推断」与「重新验证」双重角色，网关不会因此去触碰仍被封的账号",
                    "运维也可在确认解封后直接执行 `DELETE /v1/admin/ban-inference` 提前解除"
                    "（该接口绝不动用运维租约）",
                ]
            return [
                f"零误触等待平台自然解封（租约剩余约 {hours} 小时）；此期间网关已在派发路径硬阻断该节点",
                "**不要**尝试重新登录或更换令牌：账号级封禁作用在账号本身，重登无效且会延长封禁",
                "确认封禁已实际解除后，再执行 `DELETE /v1/admin/quarantine` 并留意解封后的恢复观察期",
            ]
        if posture == POSTURE_LONG_AVOIDANCE:
            rem = entry["cooldown_remaining_s"]
            cause = entry.get("cooldown_cause", "")
            # [OPT-031] 长效避让**不是**「冷却中」的一种说法：它已被派发边界硬阻断
            # （零上游调用），成因是上游连续多次拦截后的时长升级。因此建议必须是
            # 「解除成因」而不是「等待自动门控」——后者会把运维引向数小时的空等。
            if cause == "waf":
                return [
                    f"该节点处于**人机验证升级后的长效避让**（剩余约 {rem / 3600.0:.1f} 小时）："
                    f"重复出现的验证码拦截已按 2 的幂升级，自动重试只会加重账号标记",
                    f"唯一有效的处置是人工执行 `.venv/bin/python login.py {provider_key}` "
                    f"完成滑块验证——登录成功后凭证指纹变化，网关会**自动解除**该避让"
                    f"并转入串行探路重新验证",
                    "在完成人工验证之前严禁以脚本方式重试或破解验证码：属对抗行为且会显著加重风控评分",
                ]
            if cause == "auth":
                return [
                    f"该节点处于**凭证失效升级后的长效避让**（剩余约 {rem / 3600.0:.1f} 小时）："
                    f"多次认证失败已按 2 的幂升级",
                    f"执行 `.venv/bin/python login.py {provider_key}` 重新登录取令牌；"
                    f"凭证更新后网关会自动解除该避让并转入串行探路重新验证",
                    "若重登后仍被拒绝，说明是账号级问题而非凭证问题，网关会据此自动施加租约停止触碰",
                ]
            if cause == "timeout":
                return [
                    f"该节点处于长效避让（剩余约 {rem / 3600.0:.1f} 小时），成因是**上游超时**"
                    f"而非风控拦截，已被派发边界硬阻断（零上游调用）",
                    "**无需重新登录**；请检查上游服务状态与本地出网质量。"
                    "这是软冷却，没有人工解除入口，到期后由网关自行放行",
                ]
            return [
                f"该节点处于长效避让（剩余约 {rem / 3600.0:.1f} 小时，成因：{cause or '未知'}），"
                f"已被派发边界硬阻断（零上游调用）",
                "流量已自动分流至同能力环的其他健康节点；整环都在长效避让时请求会被拒绝而非逐个重试",
            ]
        if posture == POSTURE_COOLING:
            rem = entry["cooldown_remaining_s"]
            cause = entry.get("cooldown_cause", "")
            # [OPT-029] 冷却**不能**一律告知「无需人工介入」：人机验证类拦截只有人工
            # 重登能解除，而重复出现时会升级为长效避让——此时「由自动门控管理」的建议
            # 会把运维引向错误的等待，节点将被冻结数小时而无人处理。
            if cause == "waf":
                return [
                    f"该节点因**人机验证**（RGV587 / 验证码）进入避让，剩余约 {rem:.0f} 秒；"
                    f"重复出现已升级为长效避让，自动重试只会加重账号标记",
                    f"唯一有效的处置是人工执行 `.venv/bin/python login.py {provider_key}` "
                    f"完成滑块验证——登录成功后凭证指纹变化，网关会**自动解除**该避让"
                    f"并转入串行探路重新验证",
                    "在完成人工验证之前严禁以脚本方式重试或破解验证码：属对抗行为且会显著加重风控评分",
                ]
            if cause == "auth":
                return [
                    f"该节点因**凭证失效**进入避让，剩余约 {rem:.0f} 秒",
                    f"执行 `.venv/bin/python login.py {provider_key}` 重新登录取令牌；"
                    f"凭证更新后网关会自动解除避让并转入串行探路重新验证",
                ]
            if cause == "unconfigured":
                # [DAY-ZERO] 固定短冷却、凭证出现即自动解除：这里必须说清楚，
                # 否则运维会把它当成「风控冷却」去等待一个不存在的窗口。
                return [
                    f"该节点的凭证**尚未配置**（剩余约 {rem:.0f} 秒的短冷却，"
                    f"不随次数升级）：网关从未向该厂商发出过任何请求，这不属于风控",
                    f"执行 `.venv/bin/python login.py {provider_key}` 完成首次登录"
                    f"（或配置对应的 `*_AUTH_TOKEN`）；凭证出现后网关会**自动解除**该避让，"
                    f"无需重启",
                ]
            if cause == "timeout":
                return [
                    f"该节点因**上游超时**进入避让，剩余约 {rem:.0f} 秒；网关已按上游故障"
                    f"处置并故障转移",
                    "**无需重新登录**：超时不是风控拦截，重登不会改善上游不吐字的问题；"
                    "若频繁出现请检查上游服务状态与本地出网质量",
                ]
            return [
                f"保持避让：冷却剩余约 {rem:.0f} 秒（成因：{cause or '未知'}），由自动门控管理，无需人工介入",
                "流量已自动分流至同能力环的其他健康节点",
            ]
        if posture == POSTURE_PROBATION:
            p = entry["probation"]
            return [
                f"恢复观察期中：串行探路 + 步调 ×{p['pacing_factor']:.1f}，"
                f"累计 {p['successes']}/{p['required']} 次连续成功即自动毕业",
                "无需人工介入；观察期是解封后防止二次封禁的保护，请勿手动绕过",
            ]
        # 无任何拦截历史的健康节点：直接给出「正常」结论，不套用分级策略话术。
        if posture == POSTURE_CLEAR and int(entry["interceptions"]["total"]) == 0:
            return ["无风控记录，可正常派发"]
        verdict = entry["avoidance"]
        if verdict["verdict"] == "auth":
            return [f"执行 `.venv/bin/python login.py {provider_key}` 重新登录取令牌"]
        if verdict["verdict"] == "waf":
            return [
                f"执行 `.venv/bin/python login.py {provider_key}` 由人工完成人机滑块验证",
                "严禁以脚本方式自动破解验证码：属对抗行为且会显著加重账号风控评分",
            ]
        if verdict["verdict"] == "unconfigured":
            return [
                f"本机尚未配置该节点的凭证：执行 `.venv/bin/python login.py {provider_key}` "
                f"完成首次登录（或配置对应的 `*_AUTH_TOKEN`）",
                "网关从未向该厂商发出过请求，因此没有风控记录可言——这不是拦截，也无需等待冷却",
            ]
        if verdict["verdict"] == "timeout":
            return [
                "上游超时：检查上游服务状态与本地出网质量；**重新登录无效**",
                "网关已按上游故障处置并故障转移，无需人工改配置",
            ]
        if verdict["verdict"] == "ratelimit":
            return [
                "保持自适应放慢的调用步调，等待频率限制窗口自然衰减",
                "如拦截频繁，考虑提高该节点的基础调用间隔或将其移出高优先级环",
            ]
        if posture == POSTURE_WATCH:
            return ["处于监视期：近期有拦截证据，步调已自适应放大，暂无人工动作"]
        return ["无风控记录，可正常派发"]

    # ------------------------------------------------------------ 装配

    def provider_entry(self, provider_key: str, now: Optional[float] = None) -> Dict[str, Any]:
        """装配单个节点的完整档案条目。"""
        provider_key = (provider_key or "").strip().lower()
        now = time.time() if now is None else now
        stats = self._all_time(provider_key)
        posture = self.derive_posture(provider_key, now, stats)
        meta = PROVIDER_RISK_SPECS.get(provider_key)
        entry: Dict[str, Any] = {
            "provider": provider_key,
            "posture": posture,
            "posture_label": POSTURE_LABELS.get(posture, posture),
            "dispatchable": POSTURE_DISPATCHABLE.get(posture, True),
            "risk_level": meta.risk_level.value if meta else "medium",
            "risk_meta": {
                "risk_level": meta.risk_level.value if meta else "medium",
                "max_concurrency": meta.max_concurrency if meta else None,
                "default_cooldown_s": meta.default_cooldown if meta else None,
                "jitter_range": list(meta.jitter_range) if meta else None,
            },
            "cooldown_remaining_s": round(self._cooldown_state(provider_key), 1),
            "cooldown_cause": self._cooldown_cause(provider_key),
            # [OPT-035] 逐节点备注：即便姿态被更重的成因（封禁/长效避让）占据，
            # 「运维已声明出局」这一事实仍必须可见——否则运维会以为解除封禁即可恢复派发。
            "disabled": self._is_disabled(provider_key),
            # [READINESS] 同上：无凭证是「人还没登录」，与「人声明出局」处置相反，
            # 必须能被分别看见。
            "unconfigured": not self._is_configured(provider_key),
            "quarantine": self._quarantine_state(provider_key),
            "probation": self._probation_state(provider_key),
            "pacing": self._pacing_state(provider_key),
            "interceptions": stats,
        }
        entry["avoidance"] = self._avoidance(provider_key, stats)
        entry["recommended_actions"] = self.recommended_actions(provider_key, posture, entry)
        return entry

    def build(self, now: Optional[float] = None) -> Dict[str, Any]:
        """装配全量风控档案（含姿态分布汇总与规避策略矩阵）。"""
        now = time.time() if now is None else now
        providers = self._known_providers()
        entries = {pk: self.provider_entry(pk, now) for pk in providers}

        counts = {p: 0 for p in POSTURE_ORDER}
        all_time_total = 0
        for entry in entries.values():
            counts[entry["posture"]] = counts.get(entry["posture"], 0) + 1
            all_time_total += entry["interceptions"]["total"]

        banned = [pk for pk, e in entries.items() if e["posture"] == POSTURE_QUARANTINED]
        cooling = [pk for pk, e in entries.items() if e["posture"] == POSTURE_COOLING]
        recovering = [pk for pk, e in entries.items() if e["posture"] == POSTURE_PROBATION]

        return {
            "generated_at": round(now, 3),
            "enabled": not self.is_disabled(),
            "summary": {
                "posture_counts": counts,
                "quarantined": banned,
                "cooling": cooling,
                "recovering": recovering,
                "all_time_interceptions": all_time_total,
                "dispatchable_count": sum(1 for e in entries.values() if e["dispatchable"]),
                "total_providers": len(entries),
            },
            "providers": entries,
            "avoidance_matrix": AVOIDANCE_MATRIX,
        }


# 全局单例（CooldownTracker 由 server.py 构造路由后注入）
risk_dossier = RiskDossier()


def get_risk_dossier() -> RiskDossier:
    """获取全局风控档案研判器单例。"""
    return risk_dossier
