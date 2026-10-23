"""全环凭证失效检测与快速失败 (providers/credential_watch.py)。

实现 [P0] OPT-025：
1. 现场发现的问题 (Observed in the live risk ledger)：
   - 台账实证：2026-10-13 18:25~19:44 的 80 分钟内，五个节点被记录了 335 次拦截，
     其中 `auth` 类占 312 次，且**五个节点在同一秒内同时返回 401**（如 19:36:11、
     19:44:22）。这不是五次独立的令牌过期——那是概率上不可能的事。真实过程是：
     一个客户端请求沿能力环依次故障转移，五个节点全部以 401 拒绝，于是同一个请求
     在 1 秒内产生了 5 条 auth 记录；该模式每 7 分钟重复一次，即每来一个客户端请求
     就重演一次。
   - 结论：整环凭证同时失效，容量为零，故障转移不可能成功。
2. 危害不止于「浪费」(The storm is itself the risk trigger)：
   - 环内无可用凭证时，网关仍会对每个客户端请求依次打满五个节点。这 335 次带无效
     凭证的失败请求，恰好是上游 WAF 眼中**「自动化凭证尝试」的流量特征**——
     也就是说，这个风暴本身就在持续拉高风控评分，可能正是账号被封的成因之一。
   - 因此正确的行为不是「更努力地重试」，而是**立即停止**：环内凭证全失效时重试
     没有任何成功可能，却持续制造风险。这正是 OPT-025 的核心判断。
3. 与既有机制的语义区别 (Distinct from everything else)：
   - 冷却 (`CooldownTracker`)：单节点、时长有界、自动解除，其余节点仍可服务。
   - 封禁租约 (OPT-020)：单节点、运维声明、以天计、任何情况都不派发。
   - 凭证失效 (本模块)：**整环**、无需人工声明、只要凭证更新即立刻恢复。
     它不是「某个节点有问题」，而是「我们没有可用凭证了」，所以处置方式是重登而非避让。
4. 自我保护 (Never wedges)：
   - 状态带 TTL（`GATEWAY_CREDENTIAL_OUTAGE_TTL`，默认 120s）自动失效。TTL 过后放行
     下一次真实请求作为探路；若五个节点再次全部 401，则立即重新进入失效态。
   - 于是探测频率自然退化为「每 TTL 一次」，而非「每个客户端请求一次」，风暴被削平，
     同时又不可能因状态卡死而永久拒绝服务。
   - 任一次成功即摘除该节点；全部节点恢复即退出失效态。

开关：`GATEWAY_DISABLE_CREDENTIAL_WATCH=1` 关闭；`GATEWAY_DISABLE_RISK_LEDGER=1`
一并关闭（与封禁租约同约定，保证纯内存冷却语义的测试套件不受影响）。
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("gateway.credential_watch")


class CredentialOutageWatch:
    """全环凭证失效检测器（只读统计 + 快速失败判定）。"""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        # provider -> 最近一次 auth 失败时间戳
        self._auth_failures: Dict[str, float] = {}
        # 失效态首次被确认的时间戳（用于 TTL 自动失效与探路节流）
        self._outage_since: Optional[float] = None
        self._last_failed: List[str] = []
        self._stats = {
            "auth_failures": 0,
            "outages_entered": 0,
            "fast_fails": 0,
            "recovered_by_success": 0,
            "expired_by_ttl": 0,
        }

    # ------------------------------------------------------------ 配置

    def is_disabled(self) -> bool:
        if os.getenv("GATEWAY_DISABLE_CREDENTIAL_WATCH", "").strip().lower() in ("1", "true", "yes"):
            return True
        # 与封禁租约同约定：关闭持久化风控状态时一并关闭，
        # 使纯内存冷却语义的测试套件（GATEWAY_DISABLE_RISK_LEDGER=1）不受影响。
        return os.getenv("GATEWAY_DISABLE_RISK_LEDGER", "").strip().lower() in ("1", "true", "yes")

    def _window(self) -> float:
        try:
            return max(30.0, float(os.getenv("GATEWAY_CREDENTIAL_OUTAGE_WINDOW", "300")))
        except ValueError:
            return 300.0

    def _min_providers(self) -> int:
        try:
            return max(2, int(os.getenv("GATEWAY_CREDENTIAL_OUTAGE_MIN_PROVIDERS", "3")))
        except ValueError:
            return 3

    def _ttl(self) -> float:
        try:
            return max(10.0, float(os.getenv("GATEWAY_CREDENTIAL_OUTAGE_TTL", "120")))
        except ValueError:
            return 120.0

    # ------------------------------------------------------------ 记录

    def note_auth_failure(self, provider_key: str, now: Optional[float] = None) -> None:
        """记录一次认证失效（401 / 令牌过期）。"""
        if self.is_disabled():
            return
        pk = (provider_key or "").strip().lower()
        if not pk:
            return
        now = time.time() if now is None else now
        with self._lock:
            self._auth_failures[pk] = now
            self._stats["auth_failures"] += 1
            self._prune(now)
            if len(self._auth_failures) >= self._min_providers():
                if self._outage_since is None:
                    self._stats["outages_entered"] += 1
                    logger.error(
                        "[凭证失效] 检测到全环凭证失效：%.0fs 内 %d 个节点认证失败 (%s)。"
                        "故障转移不可能成功——继续重试会被上游视为自动化凭证尝试，"
                        "可能加剧风控。已进入快速失败态，请在 %s 后放行探路。",
                        self._window(), len(self._auth_failures),
                        ",".join(sorted(self._auth_failures)), f"{self._ttl():.0f}s",
                    )
                # 持续失效时刷新起点：使探测节流为「每 TTL 一次」而非每请求一次
                self._outage_since = now
                self._last_failed = sorted(self._auth_failures)

    def note_success(self, provider_key: str, now: Optional[float] = None) -> None:
        """记录一次成功：摘除该节点；全部恢复则退出失效态。"""
        if self.is_disabled():
            return
        pk = (provider_key or "").strip().lower()
        if not pk:
            return
        now = time.time() if now is None else now
        with self._lock:
            if self._auth_failures.pop(pk, None) is not None:
                logger.warning("[凭证失效] 节点 [%s] 已恢复可用（%d 个节点仍待恢复）",
                               pk, len(self._auth_failures))
            if not self._auth_failures:
                if self._outage_since is not None:
                    self._stats["recovered_by_success"] += 1
                    logger.warning("[凭证失效] 全环凭证已恢复，退出快速失败态")
                self._outage_since = None
                self._last_failed = []

    # ------------------------------------------------------------ 判定

    def _prune(self, now: float) -> None:
        """移除窗口外的失败记录（须持锁调用）。"""
        cutoff = now - self._window()
        for pk in [k for k, ts in self._auth_failures.items() if ts < cutoff]:
            del self._auth_failures[pk]

    def is_outage_active(self, now: Optional[float] = None) -> bool:
        """是否处于全环凭证失效态（TTL 到期后自动放行探路）。"""
        if self.is_disabled():
            return False
        now = time.time() if now is None else now
        with self._lock:
            if self._outage_since is None:
                return False
            self._prune(now)
            if len(self._auth_failures) < self._min_providers():
                # 失败证据已自然衰减（例如超窗），退出失效态
                self._outage_since = None
                self._last_failed = []
                return False
            if now - self._outage_since > self._ttl():
                # TTL 到期：放行一次探路。若再度全环 401，note_auth_failure 会立即重建。
                self._stats["expired_by_ttl"] += 1
                logger.info(
                    "[凭证失效] 快速失败态 TTL 到期 (%.0fs)，放行下一次请求作为探路",
                    self._ttl(),
                )
                self._outage_since = None
                return False
            self._last_failed = sorted(self._auth_failures)
            return True

    def note_fast_fail(self) -> None:
        """记录一次因凭证失效而被快速失败拒绝的请求。"""
        with self._lock:
            self._stats["fast_fails"] += 1

    def clear(self, reason: str = "manual") -> None:
        """退出失效态（重新登录成功后或运维手动解除）。"""
        with self._lock:
            had = self._outage_since is not None
            self._auth_failures.clear()
            self._outage_since = None
            self._last_failed = []
            if had:
                logger.warning("[凭证失效] 已解除快速失败态 (原因: %s)", reason)

    # ------------------------------------------------------------ 诊断

    def active_failures(self, now: Optional[float] = None) -> List[str]:
        now = time.time() if now is None else now
        with self._lock:
            self._prune(now)
            return sorted(self._auth_failures)

    def remaining_seconds(self, now: Optional[float] = None) -> float:
        """[H6] 当前快速失败态还剩多久（未处于失效态时为 0）。

        用途只有一个：让拒绝响应里的 `Retry-After` 说出**网关自己知道的**时长，
        而不是一个与真实窗口无关的常量。TTL 到期即放行探路，所以这个值就是客户端
        无需白等的那段窗口。
        """
        now = time.time() if now is None else now
        with self._lock:
            since = self._outage_since
        if since is None:
            return 0.0
        return max(0.0, self._ttl() - (now - since))

    def get_diagnostics(self, now: Optional[float] = None) -> Dict[str, Any]:
        now = time.time() if now is None else now
        with self._lock:
            self._prune(now)
            active = sorted(self._auth_failures)
            since = self._outage_since
            stats = dict(self._stats)
        ttl_remaining = 0.0
        if since is not None:
            ttl_remaining = max(0.0, self._ttl() - (now - since))
        return {
            "enabled": not self.is_disabled(),
            "outage_active": self.is_outage_active(now),
            "failed_providers": active,
            "failed_count": len(active),
            "min_providers": self._min_providers(),
            "window_seconds": self._window(),
            "ttl_seconds": self._ttl(),
            "ttl_remaining_seconds": round(ttl_remaining, 1),
            "stats": stats,
            "hint": (
                "整环凭证已失效，故障转移不可能成功。请执行 "
                "`.venv/bin/python login.py all` 重新登录全部厂商；"
                "在凭证恢复前继续请求会被上游视为自动化凭证尝试，可能加剧风控。"
            ) if active and len(active) >= self._min_providers() else "",
        }

    def clear_all_for_tests(self) -> None:
        with self._lock:
            self._auth_failures.clear()
            self._outage_since = None
            self._last_failed = []
            self._stats = {
                "auth_failures": 0,
                "outages_entered": 0,
                "fast_fails": 0,
                "recovered_by_success": 0,
                "expired_by_ttl": 0,
            }


# 全局单例
credential_watch = CredentialOutageWatch()


def get_credential_watch() -> CredentialOutageWatch:
    """获取全局全环凭证失效检测器单例。"""
    return credential_watch
