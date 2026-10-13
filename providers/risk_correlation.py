"""跨节点风控关联与系统性升级研判 (providers/risk_correlation.py)。

实现 [P1] OPT-024：
1. 单账号视角的盲区 (The Blind Spot of Per-Account Reasoning)：
   OPT-011~OPT-023 的全部避让逻辑都以「单个 Provider 账号」为单位：该账号被拦截，
   就冷却它、封禁它、观察它。但风控拦截存在第二种成因——**我们自己的客户端足迹**
   （请求头指纹、UA、设备参数、出口 IP、时间分布）被平台标记。此时各账号被拦并非
   彼此独立，而是同一根因的多个表现。
   - 二者需要**完全相反**的应对：单账号成因下，「换一个节点继续打」是正确的；
     系统性成因下，换节点毫无用处——新节点会以同样的足迹被立即拦下，只会
     逐个把剩下的账号也送进封禁。
   - 单账号视角永远看不到这一层：它把每个账号的拦截都解释为「这家风控严」，
     于是一个接一个地消耗掉全部五个账号。
2. 关联研判 (Correlation Heuristics)：
   - 以风控台账为唯一输入，在时间窗口内统计**被拦节点的去重个数**，并计算
     任何 `BURST_WINDOW` 子窗口内被同时拦截的节点数（突发度）。
   - 关联性来自「去重节点数」，而非拦截次数：同一账号被拦 100 次仍是单账号问题，
     三个不同账号在 2 分钟内先后被拦则几乎不可能由各自独立的原因导致。
   - 分级：`none` < `localized` < `widespread` < `systemic`。
3. 全局步调治理 (Global Pacing Governor)：
   - `systemic` 时对**所有**节点统一放大调用间隔（与 OPT-014 的单节点证据倍率、
     OPT-022 的观察期倍率三者相乘），从整体上降低对平台的流量压力——这是系统性
     成因下唯一可能奏效的自救手段。
   - `widespread` 时轻量放大并告警，但不自动重压（避免误判导致全局性能损失）。

设计约束：本模块**只读**（不写台账、不改冷却），且默认因子为 1.0——
无关联证据时对既有步调零影响（零回归）。
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from .risk_ledger import RiskLedger, get_risk_ledger

logger = logging.getLogger("gateway.risk_correlation")

# ---------------------------------------------------------------- 升级分级

ESCALATION_NONE = "none"
ESCALATION_LOCALIZED = "localized"
ESCALATION_WIDESPREAD = "widespread"
ESCALATION_SYSTEMIC = "systemic"

ESCALATION_ORDER: tuple = (
    ESCALATION_NONE,
    ESCALATION_LOCALIZED,
    ESCALATION_WIDESPREAD,
    ESCALATION_SYSTEMIC,
)

ESCALATION_LABELS: Dict[str, str] = {
    ESCALATION_NONE: "无风控事件",
    ESCALATION_LOCALIZED: "单账号事件（按账号级避让处理）",
    ESCALATION_WIDESPREAD: "多账号事件（疑似共享足迹，需人工确认）",
    ESCALATION_SYSTEMIC: "系统性事件（客户端足迹被标记，换节点无效）",
}

# 分级 -> 应对策略。核心是让运维/自动策略区分「换节点有用」与「换节点无用」。
ESCALATION_PLAYBOOK: Dict[str, Dict[str, Any]] = {
    ESCALATION_NONE: {
        "per_account_avoidance_helps": True,
        "rotate_node_helps": True,
        "strategy": "无风控事件，正常派发。",
    },
    ESCALATION_LOCALIZED: {
        "per_account_avoidance_helps": True,
        "rotate_node_helps": True,
        "strategy": "仅单个账号被拦，属该账号自身的风控问题：对该节点执行避让（冷却/租约）"
                    "并把流量转移至同能力环的其他节点即可，其他账号不受影响。"
                    "可按 `GET /v1/admin/risk?provider=<key>` 查看该账号的规避研判。",
    },
    ESCALATION_WIDESPREAD: {
        "per_account_avoidance_helps": False,
        "rotate_node_helps": False,
        "strategy": "多个**不同**账号在同一窗口内先后被拦，这是共享足迹（请求头指纹/UA/"
                    "设备参数/出口 IP/时间分布）被平台标记的典型信号，而非各家风控各自变严。"
                    "此时逐账号避让无法治本——轮换节点只会把同样的足迹带到下一个账号。"
                    "应人工确认：出口 IP 是否被标记、是否与其它自动化流量共用设备参数、"
                    "调用时间分布是否存在机器特征。",
    },
    ESCALATION_SYSTEMIC: {
        "per_account_avoidance_helps": False,
        "rotate_node_helps": False,
        "strategy": "短时间内多个不同账号被集中拦截，几乎不可能由各自独立的原因导致——"
                    "根因在我们的客户端足迹。**继续轮换节点会把剩余账号逐个送进封禁**，"
                    "是当前最危险的操作。正确应对：立即整体降低调用量（全局步调已自动放大）、"
                    "暂停高价值账号、优先更换客户端身份特征（UA / 设备参数 / 请求头顺序 / "
                    "出口 IP）后再逐步恢复。",
    },
}


class RiskCorrelationDetector:
    """跨节点风控关联与系统性升级研判器（只读）。"""

    def __init__(self, ledger: Optional[RiskLedger] = None) -> None:
        self.ledger = ledger if ledger is not None else get_risk_ledger()
        self._lock = threading.Lock()
        self._cache_value: Optional[float] = None
        self._cache_ts: float = 0.0

    # ------------------------------------------------------------ 配置

    def is_disabled(self) -> bool:
        return os.getenv("GATEWAY_DISABLE_RISK_CORRELATION", "").strip().lower() in ("1", "true", "yes")

    def _window(self) -> float:
        try:
            return max(60.0, float(os.getenv("GATEWAY_RISK_CORRELATION_WINDOW", "900")))
        except ValueError:
            return 900.0

    def _burst_window(self) -> float:
        try:
            return max(10.0, float(os.getenv("GATEWAY_RISK_CORRELATION_BURST_WINDOW", "120")))
        except ValueError:
            return 120.0

    def _systemic_min(self) -> int:
        try:
            return max(2, int(os.getenv("GATEWAY_RISK_CORRELATION_SYSTEMIC_MIN", "3")))
        except ValueError:
            return 3

    def _widespread_min(self) -> int:
        try:
            return max(2, int(os.getenv("GATEWAY_RISK_CORRELATION_WIDESPREAD_MIN", "3")))
        except ValueError:
            return 3

    def _systemic_factor(self) -> float:
        try:
            return max(1.0, float(os.getenv("GATEWAY_RISK_CORRELATION_SYSTEMIC_FACTOR", "2.0")))
        except ValueError:
            return 2.0

    def _widespread_factor(self) -> float:
        try:
            return max(1.0, float(os.getenv("GATEWAY_RISK_CORRELATION_WIDESPREAD_FACTOR", "1.25")))
        except ValueError:
            return 1.25

    # ------------------------------------------------------------ 关联分析

    def _window_entries(self, now: float) -> List[Tuple[float, str, str]]:
        """返回窗口内的 (ts, provider, severity) 三元组（只读）。"""
        cutoff = now - self._window()
        out: List[Tuple[float, str, str]] = []
        try:
            entries = self.ledger._read_tail(self.ledger.max_scan_lines)
        except Exception as exc:  # noqa: BLE001 - 只读研判，异常一律降级为「无证据」
            logger.warning("关联研判读取台账失败 (已忽略): %s: %s", type(exc).__name__, exc)
            return out
        for entry in entries:
            ts = entry.get("ts")
            if not isinstance(ts, (int, float)) or ts < cutoff:
                continue
            out.append((float(ts), str(entry.get("provider") or "unknown"),
                        str(entry.get("severity") or "unknown")))
        return out

    def _max_burst_distinct(self, rows: List[Tuple[float, str, str]]) -> Tuple[int, List[str]]:
        """滑动子窗口内被同时拦截的**去重节点**最大数及其成员。

        关联性来自去重节点数而非拦截次数：同一账号被拦 100 次仍是单账号问题，
        三个不同账号在同一个 2 分钟窗口内先后被拦则几乎不可能彼此独立。

        用双指针滑动窗口（O(n)）而非嵌套遍历，避免窗口内记录较多时的平方开销。
        """
        if not rows:
            return 0, []
        burst = self._burst_window()
        ordered = sorted(rows, key=lambda r: r[0])
        best = 0
        best_members: List[str] = []
        counts: Dict[str, int] = {}
        left = 0
        for right, (ts, pk, _) in enumerate(ordered):
            counts[pk] = counts.get(pk, 0) + 1
            # 收缩左边界，使窗口内时间跨度不超过 burst
            while ordered[left][0] < ts - burst:
                old = ordered[left][1]
                counts[old] -= 1
                if counts[old] <= 0:
                    del counts[old]
                left += 1
            if len(counts) > best:
                best = len(counts)
                best_members = sorted(counts)
        return best, best_members

    def evaluate(self, now: Optional[float] = None) -> Dict[str, Any]:
        """研判当前的风控关联等级（只读）。"""
        now = time.time() if now is None else now
        window = self._window()
        if self.is_disabled():
            return {
                "level": ESCALATION_NONE,
                "level_label": "关联研判已关闭",
                "window_seconds": window,
                "burst_window_seconds": self._burst_window(),
                "window_total": 0,
                "distinct_providers": [],
                "distinct_provider_count": 0,
                "burst_providers": [],
                "burst_provider_count": 0,
                "by_severity": {},
                "playbook": ESCALATION_PLAYBOOK[ESCALATION_NONE],
                "pacing_multiplier": 1.0,
                "generated_at": round(now, 3),
            }

        rows = self._window_entries(now)
        distinct = sorted({pk for _, pk, _ in rows})
        by_severity: Dict[str, Dict[str, Any]] = {}
        for _, pk, sev in rows:
            bucket = by_severity.setdefault(sev, {"count": 0, "providers": []})
            bucket["count"] += 1
            if pk not in bucket["providers"]:
                bucket["providers"].append(pk)
        for bucket in by_severity.values():
            bucket["providers"] = sorted(bucket["providers"])
            bucket["provider_count"] = len(bucket["providers"])

        burst_count, burst_members = self._max_burst_distinct(rows)

        # 分级：突发度（强信号，驱动自动治理）优先，其次窗口内去重节点数。
        if burst_count >= self._systemic_min():
            level = ESCALATION_SYSTEMIC
        elif len(distinct) >= self._widespread_min():
            level = ESCALATION_WIDESPREAD
        elif len(distinct) >= 1:
            level = ESCALATION_LOCALIZED
        else:
            level = ESCALATION_NONE

        multiplier = 1.0
        if level == ESCALATION_SYSTEMIC:
            multiplier = self._systemic_factor()
        elif level == ESCALATION_WIDESPREAD:
            multiplier = self._widespread_factor()

        result = {
            "level": level,
            "level_label": ESCALATION_LABELS[level],
            "window_seconds": window,
            "burst_window_seconds": self._burst_window(),
            "window_total": len(rows),
            "distinct_providers": distinct,
            "distinct_provider_count": len(distinct),
            "burst_providers": burst_members,
            "burst_provider_count": burst_count,
            "by_severity": by_severity,
            "playbook": ESCALATION_PLAYBOOK[level],
            "pacing_multiplier": multiplier,
            "generated_at": round(now, 3),
        }

        if level in (ESCALATION_SYSTEMIC, ESCALATION_WIDESPREAD):
            logger.warning(
                "[风控关联] 升级为 %s：窗口 %.0fs 内 %d 个不同账号被拦（突发 %.0fs 内 %d 个：%s）。"
                "轮换节点无法治本，已对全局步调施加 ×%.2f 压制。",
                level, window, len(distinct), self._burst_window(), burst_count,
                ",".join(burst_members) or "-", multiplier,
            )
        return result

    def global_pacing_multiplier(self, now: Optional[float] = None) -> float:
        """全局步调压制倍率（无关联证据时为 1.0，零回归）。"""
        try:
            return float(self.evaluate(now)["pacing_multiplier"])
        except Exception:  # noqa: BLE001
            return 1.0

    # 简短缓存：步调路径调用频繁，避免每次派发都全量扫描台账。
    def cached_global_pacing_multiplier(self, now: Optional[float] = None) -> float:
        now = time.time() if now is None else now
        ttl = self._cache_ttl()
        with self._lock:
            if self._cache_value is not None and (now - self._cache_ts) < ttl:
                return self._cache_value
        value = self.global_pacing_multiplier(now)
        with self._lock:
            self._cache_value = value
            self._cache_ts = now
        return value

    def _cache_ttl(self) -> float:
        try:
            return max(0.0, float(os.getenv("GATEWAY_RISK_CORRELATION_CACHE_TTL", "15")))
        except ValueError:
            return 15.0

    def get_diagnostics(self, now: Optional[float] = None) -> Dict[str, Any]:
        return self.evaluate(now)

    def clear_cache_for_tests(self) -> None:
        with self._lock:
            self._cache_value = None
            self._cache_ts = 0.0


# 全局单例
risk_correlation = RiskCorrelationDetector()


def get_risk_correlation() -> RiskCorrelationDetector:
    """获取全局跨节点风控关联研判器单例。"""
    return risk_correlation
