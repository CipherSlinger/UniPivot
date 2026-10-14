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
POSTURE_COOLING = "cooling"
POSTURE_PROBATION = "probation"
POSTURE_WATCH = "watch"
POSTURE_CLEAR = "clear"

# 姿态由重到轻；派生取**最重**的命中项（fail-closed）。
POSTURE_ORDER: tuple = (
    POSTURE_QUARANTINED,
    POSTURE_COOLING,
    POSTURE_PROBATION,
    POSTURE_WATCH,
    POSTURE_CLEAR,
)

POSTURE_LABELS: Dict[str, str] = {
    POSTURE_QUARANTINED: "长期封禁（硬阻断，零派发）",
    POSTURE_COOLING: "避让冷却中（自动，时长有界）",
    POSTURE_PROBATION: "恢复观察期（串行探路 + 步调放大）",
    POSTURE_WATCH: "监视期（近期有拦截证据，步调已放慢）",
    POSTURE_CLEAR: "正常（无近期风控记录）",
}

# 只有硬封禁不可派发；软冷却保留「最后一搏」语义以保可用性（见 OPT-021）。
POSTURE_DISPATCHABLE: Dict[str, bool] = {
    POSTURE_QUARANTINED: False,
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
                    "`python login.py <provider>` 完成滑块验证，重登可恢复会话。",
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
}

# 严重度降级映射：租约（人工声明的封禁）在研判中一律视为 ban。
_SEVERITY_TO_VERDICT = {
    "ratelimit": "ratelimit",
    "waf": "waf",
    "auth": "auth",
}


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
        else:
            # 取出现次数最多的严重度作为主要症结；无记录则按最轻的 ratelimit。
            by_sev = stats.get("by_severity") or {}
            dominant = max(by_sev.items(), key=lambda kv: kv[1])[0] if by_sev else ""
            verdict_key = _SEVERITY_TO_VERDICT.get(dominant) or "ratelimit"
            reason = f"主要症结：{dominant or '无拦截记录'}"
        matrix = AVOIDANCE_MATRIX[verdict_key]
        return {
            "verdict": verdict_key,
            "reason": reason,
            **matrix,
        }

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
        if posture == POSTURE_COOLING:
            rem = entry["cooldown_remaining_s"]
            return [
                f"保持避让：冷却剩余约 {rem:.0f} 秒，由自动门控管理，无需人工介入",
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
