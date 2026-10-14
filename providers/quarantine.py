"""运维人工长期封禁租约 (providers/quarantine.py)。

实现 [P1] OPT-020：
1. 长期持有 (Operator Quarantine Lease):
   - 自动风控冷却存在上限（认证类冷却封顶 GATEWAY_AUTH_COOLDOWN_MAX，默认 24h），
     但账号级封禁（如 DeepSeek 已触发的封禁）往往持续数天。冷却上限到期后网关会
     重新触碰仍处于封禁窗口的节点，人为延长封禁。
   - 本模块提供**运维可显式设定**的长期封禁租约：以绝对到期时间持久化落盘
     （POSIX 原子写，目录 0o700 / 文件 0o600），跨进程重启持续生效，直到窗口结束。
2. 统一生效 (Single Source of Truth):
   - 租约被折入 `CooldownTracker.get_remaining_cooldown()` / `is_in_cooldown()`，
     因此**所有**已有冷却门控自动遵循：健康探针 (OPT-012)、令牌刷新 (OPT-013)、
     临时会话回收 (OPT-017)、自愈大脑选举 (OPT-018)、路由选点与自适应步调，
     无需逐处接线，也杜绝遗漏。
3. 配置与运维:
   - `GATEWAY_QUARANTINE="deepseek:72h,glm:30m,kimi:2026-10-20T00:00:00"` 启动即生效，
     支持 s/m/h/d 后缀时长与 ISO-8601 绝对时间。
   - `GET/POST/DELETE /v1/admin/quarantine` 供运维在运行期查看、设定与解除。
   - `GATEWAY_DISABLE_QUARANTINE=1` 关闭租约（不再读写磁盘，环境变量租约亦不生效）。

设计约束：租约只会**延长**避让窗口，绝不缩短既有冷却（读取端取 max），
因此不可能因误操作而提前放行一个正在封禁的节点。

[OPT-027] 租约来源（source）：运维声明的租约（默认）与**推断得来的**自动租约
（`source="auto"`，见 `providers/failover.py` 的「新凭证仍被拒 -> 账号级封禁」推断）
在语义上必须可区分——自动租约的正当性完全来自「我们已经试过一次新凭证且仍被拒」，
因此一旦出现**更新的**凭证，这条证据即告失效，自动租约必须能被自动解除并重新验证；
而运维租约是人类声明，永不被自动解除。任何来源不明的租约一律按运维租约处理（保守）。
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("gateway.quarantine")

_DURATION_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*([smhd]?)$", re.IGNORECASE)
_UNIT_SECONDS = {"": 1.0, "s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}


def parse_quarantine_until(
    spec: str, now: Optional[float] = None
) -> Optional[float]:
    """解析时长或绝对时间表达式为绝对到期时间戳。

    支持 `"72h"` / `"30m"` / `"45"` / `"2d"` 等时长，以及
    `"2026-10-20T00:00:00"` / `"2026-10-20"` 等 ISO-8601 绝对时间（本地时区）。
    无法解析时返回 None。
    """
    text = (spec or "").strip()
    if not text:
        return None
    base = time.time() if now is None else now

    match = _DURATION_RE.match(text)
    if match:
        value = float(match.group(1))
        unit = (match.group(2) or "").lower()
        return base + value * _UNIT_SECONDS[unit]

    # 绝对时间（允许 'T' 或空格分隔，秒/分可省略）
    normalized = text.replace(" ", "T", 1) if " " in text and "T" not in text else text
    for candidate in (normalized, normalized.replace("Z", "+00:00")):
        try:
            dt = datetime.fromisoformat(candidate)
        except ValueError:
            continue
        if dt.tzinfo is None:
            dt = dt.astimezone()
        return dt.timestamp()
    return None


class QuarantineStore:
    """运维人工长期封禁租约的持久化存储。"""

    def __init__(self, directory: Optional[Path] = None) -> None:
        self.directory = self._resolve_directory(directory)
        self.path = self.directory / "quarantine.json"
        self._lock = threading.RLock()
        self._leases: Dict[str, float] = {}
        self._reasons: Dict[str, str] = {}
        # [OPT-027] provider -> 租约来源（"operator" | "auto"）。缺省/未知一律视为
        # 运维租约（保守）：只有确知来源为 auto 的租约才允许被自动解除。
        self._sources: Dict[str, str] = {}
        # [OPT-022] provider -> 最近一次租约解除（显式清除或自然到期）的时间戳。
        # 供 CooldownTracker 在封禁解除后启动「恢复观察期」，避免解封瞬间的并发洪峰
        # 立刻把账号二次打进封禁。仅保留近期记录，避免无界增长。
        self._releases: Dict[str, float] = {}
        self._stats = {"set": 0, "cleared": 0, "expired": 0, "persist_failures": 0}
        self._load()
        self._apply_env()

    # ------------------------------------------------------------ 配置

    def _resolve_directory(self, directory: Optional[Path]) -> Path:
        if directory is not None:
            return Path(directory)
        env_dir = os.getenv("GATEWAY_QUARANTINE_DIR", "").strip()
        if env_dir:
            return Path(env_dir)
        ledger_dir = os.getenv("GATEWAY_RISK_LEDGER_DIR", "").strip()
        if ledger_dir:
            return Path(ledger_dir)
        if os.getenv("TESTING") == "1":
            return Path(tempfile.mkdtemp(prefix="gateway_quarantine_"))
        return Path(__file__).resolve().parent.parent / "session"

    def is_disabled(self) -> bool:
        """判断是否关闭租约（关闭后不读写磁盘，且内存中不保留租约）。"""
        if os.getenv("GATEWAY_DISABLE_QUARANTINE", "").strip().lower() in ("1", "true", "yes"):
            return True
        # 与风控台账同开关：关闭持久化风控状态时不再落盘
        return os.getenv("GATEWAY_DISABLE_RISK_LEDGER", "").strip().lower() in ("1", "true", "yes")

    def _persist_enabled(self) -> bool:
        """是否允许读写磁盘（关闭开关时租约仍可存在于内存，但不落盘）。"""
        return os.getenv("GATEWAY_DISABLE_QUARANTINE", "").strip().lower() not in ("1", "true", "yes")

    # ------------------------------------------------------------ 原子写

    def _persist(self) -> None:
        if not self._persist_enabled():
            return
        with self._lock:
            payload = {
                "leases": {
                    pk: {
                        "until": round(until, 3),
                        "reason": self._reasons.get(pk, ""),
                        "source": self._sources.get(pk, "operator"),
                    }
                    for pk, until in self._leases.items()
                },
                "releases": {pk: round(ts, 3) for pk, ts in self._releases.items()},
                "saved_at": round(time.time(), 3),
            }
        try:
            content = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        except Exception:  # noqa: BLE001
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.chmod(self.path.parent, 0o700)
            except OSError:
                pass
            temp_file = self.path.parent / (
                f".{self.path.name}_{os.getpid()}_{threading.get_ident()}.tmp"
            )
            try:
                fd = os.open(temp_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                with open(fd, "w", encoding="utf-8") as f:
                    f.write(content)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(temp_file, self.path)
            finally:
                if temp_file.exists():
                    try:
                        temp_file.unlink(missing_ok=True)
                    except OSError:
                        pass
        except Exception as exc:  # noqa: BLE001
            self._stats["persist_failures"] += 1
            logger.warning("封禁租约落盘失败 (已忽略): %s: %s", type(exc).__name__, exc)

    def _load(self) -> None:
        if not self._persist_enabled():
            return
        with self._lock:
            try:
                if not self.path.exists():
                    return
                raw = self.path.read_text(encoding="utf-8").strip()
                if not raw:
                    return
                data = json.loads(raw)
                leases = data.get("leases") if isinstance(data, dict) else None
                if not isinstance(leases, dict):
                    return
                now = time.time()
                for pk, entry in leases.items():
                    if not isinstance(entry, dict):
                        continue
                    until = entry.get("until")
                    if not isinstance(until, (int, float)) or until <= now:
                        continue
                    key = str(pk).lower()
                    self._leases[key] = float(until)
                    self._reasons[key] = str(entry.get("reason", ""))
                    # 未知来源按运维租约处理（保守：绝不被自动解除）
                    source = str(entry.get("source", "operator") or "operator").lower()
                    self._sources[key] = "auto" if source == "auto" else "operator"
                # [OPT-022] 恢复租约解除时间（供解封后的恢复观察期判定）
                releases = data.get("releases")
                if isinstance(releases, dict):
                    cutoff = now - self._release_retention()
                    for pk, ts in releases.items():
                        if isinstance(ts, (int, float)) and float(ts) > cutoff:
                            self._releases[str(pk).lower()] = float(ts)
                if self._leases:
                    logger.warning(
                        "[封禁租约] 已从磁盘恢复 %d 个运维封禁租约: %s",
                        len(self._leases),
                        {k: round(v - now, 1) for k, v in self._leases.items()},
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("读取封禁租约失败 (已忽略): %s: %s", type(exc).__name__, exc)

    def _apply_env(self) -> None:
        """应用 GATEWAY_QUARANTINE 环境变量租约（启动即生效，落盘保持跨重启）。"""
        spec = os.getenv("GATEWAY_QUARANTINE", "").strip()
        if not spec:
            return
        applied: Dict[str, float] = {}
        for item in spec.split(","):
            item = item.strip()
            if not item or ":" not in item:
                continue
            provider, _, value = item.partition(":")
            provider = provider.strip().lower()
            until = parse_quarantine_until(value)
            if not provider or until is None:
                logger.warning("[封禁租约] 无法解析环境变量条目 %r，已跳过", item)
                continue
            applied[provider] = until
        with self._lock:
            for provider, until in applied.items():
                if until > self._leases.get(provider, 0.0):
                    self._leases[provider] = until
                    self._reasons.setdefault(provider, "env:GATEWAY_QUARANTINE")
                self._sources[provider] = "operator"
        if applied:
            logger.warning(
                "[封禁租约] 已应用环境变量租约: %s",
                {k: round(v - time.time(), 1) for k, v in applied.items()},
            )
            self._persist()

    # ------------------------------------------------------------ 租约操作

    def _release_retention(self) -> float:
        """租约解除记录的保留时长（秒），供解封后的恢复观察期使用。"""
        try:
            return max(60.0, float(os.getenv("GATEWAY_QUARANTINE_RELEASE_TTL", "86400")))
        except ValueError:
            return 86400.0

    def _note_release(self, provider: str, ts: Optional[float] = None) -> None:
        """记录租约解除时刻（须持锁调用）。"""
        key = (provider or "").strip().lower()
        if not key:
            return
        self._releases[key] = time.time() if ts is None else ts

    def last_released_at(self, provider: str) -> Optional[float]:
        """返回该节点最近一次封禁租约解除的时间戳（无记录或已过保留期则 None）。

        [OPT-022] CooldownTracker 据此在封禁解除后启动「恢复观察期」：
        解封瞬间以串行 + 放大步调试探，避免并发洪峰把刚恢复的账号二次打进封禁。
        查询路径只做只读判定，不写盘（避免读放大）。
        """
        key = (provider or "").strip().lower()
        if not key:
            return None
        # 先触发一次惰性到期清理：确保「租约刚刚自然到期」也能被本方法单独观测到，
        # 而不依赖调用方恰好先查过 remaining()。
        if key in self._leases:
            self.remaining(key)
        ts = self._releases.get(key)
        if ts is None:
            return None
        if time.time() - ts > self._release_retention():
            return None
        return ts

    def set_lease(
        self,
        provider: str,
        duration_seconds: Optional[float] = None,
        until: Optional[float] = None,
        reason: str = "",
        source: str = "operator",
    ) -> float:
        """设定/延长某节点的封禁租约，返回绝对到期时间戳。

        `until` 优先；否则由 `duration_seconds` 计算。租约只延长不缩短：
        新到期时间早于现有租约时不生效。

        [OPT-027] `source` 区分运维声明（"operator"）与推断得来的自动租约（"auto"）。
        来源优先级：运维声明**永远**覆盖自动标记（即便未延长到期时间），
        自动标记绝不覆盖运维声明——否则一次自动推断会让运维的正式声明变成可被自动解除。
        """
        key = (provider or "").strip().lower()
        if not key:
            raise ValueError("provider 不能为空")
        if until is None:
            if duration_seconds is None:
                raise ValueError("必须提供 duration_seconds 或 until")
            until = time.time() + max(0.0, float(duration_seconds))
        normalized = "auto" if str(source).strip().lower() == "auto" else "operator"
        with self._lock:
            if float(until) > self._leases.get(key, 0.0):
                self._leases[key] = float(until)
                if reason:
                    self._reasons[key] = str(reason)[:300]
                self._stats["set"] += 1
            # 来源优先级：运维声明永远覆盖自动标记；自动标记只在**尚无运维声明**时生效。
            # 注意这里必须区分「没有记录」与「记录为 operator」——用默认值 "operator"
            # 会让自动标记永远写不进去（曾因此让自动租约被误判成运维租约）。
            existing = self._sources.get(key)
            if normalized == "operator" or existing in (None, "auto"):
                self._sources[key] = normalized
            effective = self._leases[key]
        logger.warning(
            "[封禁租约] 节点 %s 进入长期封禁避让，剩余 %.1f 小时 (原因: %s)",
            key,
            max(0.0, effective - time.time()) / 3600.0,
            self._reasons.get(key, ""),
        )
        self._persist()
        return effective

    def clear(self, provider: Optional[str] = None) -> int:
        """解除指定节点（或全部）的封禁租约，返回解除数量。"""
        with self._lock:
            if provider is None:
                count = len(self._leases)
                # [OPT-022] 全部解除同样逐节点记录解除时刻（供恢复观察期）
                self._note_release_all(list(self._leases.keys()))
                self._leases.clear()
                self._reasons.clear()
                self._sources.clear()
            else:
                key = provider.strip().lower()
                count = 1 if self._leases.pop(key, None) is not None else 0
                self._reasons.pop(key, None)
                self._sources.pop(key, None)
                if count:
                    self._note_release(key)
            self._stats["cleared"] += count
        if count:
            logger.warning("[封禁租约] 已解除 %d 个封禁租约 (provider=%s)", count, provider or "ALL")
            self._persist()
        return count

    def _note_release_all(self, providers: List[str]) -> None:
        """批量记录解除时刻（须持锁调用）。"""
        for pk in providers:
            self._note_release(pk)

    def source(self, provider: str) -> str:
        """返回租约来源（无租约时返回空串）：`"operator"` 或 `"auto"`。"""
        key = (provider or "").strip().lower()
        if not key:
            return ""
        with self._lock:
            if key not in self._leases:
                return ""
            return self._sources.get(key, "operator")

    def release_auto_lease(self, provider: str, now: Optional[float] = None) -> bool:
        """[OPT-027] 仅当租约来源为 `"auto"` 时解除它，返回是否真的解除。

        这是自动租约唯一的解除入口：运维租约（含来源不明的租约）一律返回 False，
        因此「自动推断」在结构上不可能解除任何人类声明。
        """
        key = (provider or "").strip().lower()
        if not key:
            return False
        with self._lock:
            if key not in self._leases:
                return False
            if self._sources.get(key, "operator") != "auto":
                return False
            until = self._leases.pop(key, 0.0)
            self._reasons.pop(key, None)
            self._sources.pop(key, None)
            self._stats["cleared"] += 1
            # 记录解除时刻，使解封后同样进入恢复观察期（与运维解除一致）
            self._note_release(key, until)
        logger.warning(
            "[封禁租约] 已解除节点 %s 的**自动推断**租约（出现更新的凭证，推断依据失效）", key
        )
        self._persist()
        return True

    def remaining(self, provider: str, now: Optional[float] = None) -> float:
        """返回节点剩余租约秒数（无租约或已到期返回 0.0）。"""
        key = (provider or "").strip().lower()
        if not key:
            return 0.0
        current = time.time() if now is None else now
        with self._lock:
            until = self._leases.get(key)
            if until is None:
                return 0.0
            if until <= current:
                # 惰性清理：到期即移除（不落盘，避免读路径产生写放大）。
                # [OPT-022] 同时记录解除时刻，供解封后的恢复观察期判定。
                self._leases.pop(key, None)
                self._reasons.pop(key, None)
                self._sources.pop(key, None)
                self._note_release(key, until)
                self._stats["expired"] += 1
                return 0.0
            return until - current

    def list_active(self, now: Optional[float] = None) -> Dict[str, Dict[str, Any]]:
        """列出当前生效的租约（provider -> {remaining_hours, until, reason}）。"""
        current = time.time() if now is None else now
        result: Dict[str, Dict[str, Any]] = {}
        for pk in list(self._leases.keys()):
            rem = self.remaining(pk, current)
            if rem > 0:
                result[pk] = {
                    "remaining_seconds": round(rem, 1),
                    "remaining_hours": round(rem / 3600.0, 2),
                    "until": round(self._leases.get(pk, 0.0), 3),
                    "reason": self._reasons.get(pk, ""),
                    "source": self._sources.get(pk, "operator"),
                }
        return result

    # ------------------------------------------------------------ 诊断

    def get_diagnostics(self) -> Dict[str, Any]:
        active = self.list_active()
        return {
            "enabled": not self.is_disabled(),
            "persist_enabled": self._persist_enabled(),
            "path": str(self.path),
            "active_count": len(active),
            "active": active,
            "stats": dict(self._stats),
        }

    def clear_all_for_tests(self) -> None:
        """清空内存租约并删除落盘文件（测试用）。"""
        with self._lock:
            self._leases.clear()
            self._reasons.clear()
            self._releases.clear()
            self._sources.clear()
            self._stats = {"set": 0, "cleared": 0, "expired": 0, "persist_failures": 0}
            try:
                self.path.unlink(missing_ok=True)
            except OSError:
                pass


# 全局单例
quarantine_store = QuarantineStore()


def get_quarantine_store() -> QuarantineStore:
    """获取全局封禁租约存储单例。"""
    return quarantine_store
