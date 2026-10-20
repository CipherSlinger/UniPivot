"""风控拦截持久化台账与冷却状态跨重启恢复引擎 (providers/risk_ledger.py)。

实现 [P0] OPT-011：
1. 冷却状态持久化 (Durable Cooldown State):
   - 将 provider -> 冷却截止时间戳与连续失败计数以 POSIX 原子写
     (os.replace + os.fsync, 目录 0o700 / 文件 0o600) 落盘，
     进程重启后自动恢复。杜绝「重启即重置避让、立刻再次触碰已被风控封禁的
     节点」，避免因高频重试而延长上游封禁窗口。
2. 风控拦截台账 (Interception Ledger):
   - 每一次风控 / WAF 人机验证 / 认证失效拦截都以 JSONL 追加落盘，
     记录时间、节点、HTTP 状态、分级严重度、原因与证据摘要，
     形成可审计、可跨重启检索的长期风控记录。
3. 严重度分级 (Severity Classification):
   - ratelimit: 频率限制 (429 / 频控)，维持既有短冷却调速策略
   - waf:       人机验证 / 滑块 / WAF 拦截
   - auth:      令牌失效 / 认证失败 / 账号级限制
4. 遥测与诊断:
   - GET /v1/diagnostics 暴露 risk_ledger 段（近 N 小时各节点拦截计数、
     当前冷却与长效避让剩余），供仪表盘与自动化探针审计。
   - GATEWAY_DISABLE_RISK_LEDGER=1 可完全关闭持久化（退化为纯内存冷却）。

磁盘写入全部为 best-effort：台账或状态落盘失败绝不阻断、绝不改变请求处理流程。
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Mapping, Optional

logger = logging.getLogger("gateway.risk_ledger")

# 严重度分级常量
SEVERITY_RATE_LIMIT = "ratelimit"
SEVERITY_WAF = "waf"
SEVERITY_AUTH = "auth"

_MAX_EVIDENCE_CHARS = 300


def classify_interception_severity(
    is_risk: bool,
    risk_reason: str = "",
    status: int = 0,
    error_msg: str = "",
) -> str:
    """将一次上游拦截归类为 ratelimit / waf / auth 三级。

    判定优先级：认证失效 > 频率限制 > WAF 人机验证。
    """
    text = f"{error_msg} {risk_reason}".lower()
    if status == 401 or any(
        k in text
        for k in (
            "auth expiration",
            "authorization failed",
            "invalid token",
            "令牌无效",
            "令牌失效",
            "认证失效",
            "未登录",
            "unauthorized",
            "token expired",
        )
    ):
        return SEVERITY_AUTH
    if status == 429 or any(k in text for k in ("rate limit", "rate_limit", "频控", "too many requests")):
        return SEVERITY_RATE_LIMIT
    return SEVERITY_WAF


class RiskLedger:
    """风控拦截台账与冷却状态持久化存储。"""

    def __init__(self, directory: Optional[Path] = None) -> None:
        self.directory = self._resolve_directory(directory)
        self.cooldown_path = self.directory / "risk_cooldowns.json"
        self.ledger_path = self.directory / "risk_ledger.jsonl"
        self.max_lines = int(os.getenv("GATEWAY_RISK_LEDGER_MAX_LINES", "20000"))
        self.max_scan_lines = int(os.getenv("GATEWAY_RISK_LEDGER_SCAN_LINES", "20000"))
        self._lock = threading.RLock()
        self._stats = {
            "interceptions_recorded": 0,
            "cooldown_saves": 0,
            "cooldown_loads": 0,
            "ledger_prunes": 0,
            "write_failures": 0,
        }

    # ------------------------------------------------------------ 配置

    def _resolve_directory(self, directory: Optional[Path]) -> Path:
        if directory is not None:
            return Path(directory)
        env_dir = os.getenv("GATEWAY_RISK_LEDGER_DIR", "").strip()
        if env_dir:
            return Path(env_dir)
        if os.getenv("TESTING") == "1":
            # 测试环境绝不写入真实 session/ 目录，避免污染线上风控状态
            return Path(tempfile.mkdtemp(prefix="gateway_risk_ledger_"))
        return Path(__file__).resolve().parent.parent / "session"

    def is_disabled(self) -> bool:
        """判断是否关闭持久化（关闭后退化为纯内存冷却，行为与历史版本一致）。"""
        return os.getenv("GATEWAY_DISABLE_RISK_LEDGER", "").strip().lower() in ("1", "true", "yes")

    # ------------------------------------------------------------ 原子写

    def _atomic_write(self, path: Path, content: str) -> bool:
        """POSIX 原子写：临时文件 + fsync + os.replace。失败仅记日志，绝不上抛。"""
        with self._lock:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.chmod(path.parent, 0o700)
                except OSError:
                    pass
                temp_file = path.parent / f".{path.name}_{os.getpid()}_{threading.get_ident()}.tmp"
                try:
                    fd = os.open(temp_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                    with open(fd, "w", encoding="utf-8") as f:
                        f.write(content)
                        f.flush()
                        os.fsync(f.fileno())
                    os.replace(temp_file, path)
                finally:
                    if temp_file.exists():
                        try:
                            temp_file.unlink(missing_ok=True)
                        except OSError:
                            pass
                return True
            except Exception as exc:  # noqa: BLE001 - 持久化失败不得影响请求流程
                self._stats["write_failures"] += 1
                logger.warning("风控台账落盘失败 (已忽略): %s: %s", type(exc).__name__, exc)
                return False

    # ------------------------------------------------------------ 冷却状态

    def load_state(self) -> Dict[str, Any]:
        """读取持久化的冷却状态与失败计数。文件缺失/损坏时返回空结构。"""
        if self.is_disabled():
            return {}
        with self._lock:
            try:
                if not self.cooldown_path.exists():
                    return {}
                raw = self.cooldown_path.read_text(encoding="utf-8").strip()
                if not raw:
                    return {}
                data = json.loads(raw)
                if not isinstance(data, dict):
                    return {}
                self._stats["cooldown_loads"] += 1
                return data
            except Exception as exc:  # noqa: BLE001
                logger.warning("读取风控冷却状态失败 (已忽略): %s: %s", type(exc).__name__, exc)
                return {}

    def save_state(self, state: Mapping[str, Any]) -> bool:
        """持久化冷却状态与失败计数（best-effort）。

        返回是否**确实写盘成功**。调用方（[H5] 的 `_persist_state`）用它决定要不要
        记住这次内容的指纹：写失败时绝不能记住，否则下一份相同状态会被误判为
        「已经落盘」而跳过重试——一次磁盘瞬时故障就会让冷却状态直到内容再次变化
        之前都不再持久化。
        """
        if self.is_disabled():
            return False
        try:
            payload = json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True)
        except Exception:  # noqa: BLE001
            return False
        if self._atomic_write(self.cooldown_path, payload):
            self._stats["cooldown_saves"] += 1
            return True
        return False

    # ------------------------------------------------------------ 拦截台账

    def record_interception(
        self,
        provider: str,
        status: int,
        reason: str,
        severity: str,
        cooldown_seconds: float,
        evidence: str = "",
        now: Optional[float] = None,
    ) -> None:
        """追加一条风控拦截记录（JSONL，best-effort）。"""
        if self.is_disabled():
            return
        entry = {
            "ts": round(now if now is not None else time.time(), 3),
            "provider": (provider or "unknown").lower(),
            "status": int(status) if isinstance(status, (int, float)) else 0,
            "severity": severity,
            "reason": reason,
            "cooldown_seconds": round(float(cooldown_seconds), 2),
            "evidence": (evidence or "")[:_MAX_EVIDENCE_CHARS],
        }
        try:
            line = json.dumps(entry, ensure_ascii=False)
        except Exception:  # noqa: BLE001
            return
        with self._lock:
            try:
                self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.ledger_path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
                    f.flush()
                self._stats["interceptions_recorded"] += 1
            except Exception as exc:  # noqa: BLE001
                self._stats["write_failures"] += 1
                logger.warning("风控拦截台账追加失败 (已忽略): %s: %s", type(exc).__name__, exc)
                return
        self._maybe_prune()

    def _maybe_prune(self) -> None:
        """台账行数超过上限时保留后一半，防止无界增长。"""
        if self.max_lines <= 0:
            return
        with self._lock:
            try:
                if not self.ledger_path.exists():
                    return
                with open(self.ledger_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                if len(lines) <= self.max_lines:
                    return
                keep = lines[len(lines) // 2:]
                self._atomic_write(self.ledger_path, "".join(keep))
                self._stats["ledger_prunes"] += 1
                logger.info("风控台账已裁剪：%d -> %d 行", len(lines), len(keep))
            except Exception as exc:  # noqa: BLE001
                logger.warning("风控台账裁剪失败 (已忽略): %s: %s", type(exc).__name__, exc)

    def _read_tail(self, limit: int) -> Deque[Dict[str, Any]]:
        entries: Deque[Dict[str, Any]] = deque(maxlen=max(1, limit))
        with self._lock:
            try:
                if not self.ledger_path.exists():
                    return entries
                with open(self.ledger_path, "r", encoding="utf-8") as f:
                    for raw in f:
                        raw = raw.strip()
                        if not raw:
                            continue
                        try:
                            obj = json.loads(raw)
                        except Exception:  # noqa: BLE001 - 跳过损坏行
                            continue
                        if isinstance(obj, dict):
                            entries.append(obj)
            except Exception as exc:  # noqa: BLE001
                logger.warning("读取风控台账失败 (已忽略): %s: %s", type(exc).__name__, exc)
        return entries

    def recent_interceptions(self, hours: float = 24.0, now: Optional[float] = None) -> Dict[str, Any]:
        """汇总近 N 小时各节点的拦截次数与最近一次拦截详情。"""
        now = now if now is not None else time.time()
        cutoff = now - hours * 3600.0
        per_provider: Dict[str, Dict[str, Any]] = {}
        total = 0
        for entry in self._read_tail(self.max_scan_lines):
            ts = entry.get("ts")
            if not isinstance(ts, (int, float)) or ts < cutoff:
                continue
            total += 1
            pk = str(entry.get("provider") or "unknown")
            bucket = per_provider.setdefault(
                pk,
                {"count": 0, "last_ts": 0.0, "last_reason": "", "last_severity": "", "severities": {}},
            )
            bucket["count"] += 1
            if ts >= bucket["last_ts"]:
                bucket["last_ts"] = ts
                bucket["last_reason"] = entry.get("reason", "")
                bucket["last_severity"] = entry.get("severity", "")
            sev = str(entry.get("severity") or "unknown")
            bucket["severities"][sev] = bucket["severities"].get(sev, 0) + 1
        for bucket in per_provider.values():
            bucket["last_ts"] = round(bucket["last_ts"], 3)
        return {"window_hours": hours, "total": total, "per_provider": per_provider}

    # ------------------------------------------------------------ 诊断

    def get_diagnostics(self, now: Optional[float] = None) -> Dict[str, Any]:
        now = now if now is not None else time.time()
        return {
            "enabled": not self.is_disabled(),
            "directory": str(self.directory),
            "cooldown_path": str(self.cooldown_path),
            "ledger_path": str(self.ledger_path),
            "recent": self.recent_interceptions(hours=24.0, now=now),
            "stats": dict(self._stats),
        }

    def clear(self) -> None:
        """清空持久化状态与台账（测试用）。"""
        with self._lock:
            for path in (self.cooldown_path, self.ledger_path):
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
            self._stats = {
                "interceptions_recorded": 0,
                "cooldown_saves": 0,
                "cooldown_loads": 0,
                "ledger_prunes": 0,
                "write_failures": 0,
            }


# 全局单例
risk_ledger = RiskLedger()


def get_risk_ledger() -> RiskLedger:
    """获取全局风控台账单例。"""
    return risk_ledger
