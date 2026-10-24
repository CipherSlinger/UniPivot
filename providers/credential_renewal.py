"""凭证续期检测与认证级避让复位 (providers/credential_renewal.py)。

实现 [P0] OPT-026：
1. 现场问题的第二层——风暴之后的 24 小时尾巴 (The 24-hour tail after the storm)：
   `record_failure` 对认证失效按 2 的幂次长效升级：连续 ≥3 次即
   `GATEWAY_AUTH_COOLDOWN_BASE`(1800s) × 2ⁿ，封顶 `GATEWAY_AUTH_COOLDOWN_MAX`(86400s)。
   真实台账里五个节点的认证失败次数是 54/54/54/87/60 —— 全部撞到 24 小时上限。
   于是风暴结束后，**五个节点各自被冻结 24 小时**，且该状态经台账持久化、跨重启恢复。
2. 为什么这是错的 (Why the escalation becomes wrong once the cause is gone)：
   长效升级惩罚的是「拿着**同一个失效凭证**反复打」——这确实该罚，因为每次重试都是
   一次「自动化凭证尝试」。但凭证一旦续期，这个惩罚就失去了对象：失败的原因已经
   不存在了。此刻继续冻结节点，惩罚的不再是坏行为，而是**已经修好的系统**：
   - `get_remaining_cooldown()` 仍返回 24h → 健康探针跳过、自愈选举排除、路由降级排序；
   - 更荒谬的是 `server.py:_deferred_refresh_after_cooldown` —— **令牌静默刷新本身被
     冷却门控挡住，最多等 24 小时**。也就是说，「凭证失效」制造的状态，正好锁死了
     唯一能修复它的机制。
   - 反向验证：OPT-023 档案里 `auth` 类的结论是 `recoverable_by_relogin = True`，
     即系统自己告诉运维「重登可恢复」；但重登之后 24 小时内网关依然在避让该节点，
     结论与实际行为自相矛盾。
3. 判定「已续期」(Detecting renewal)：
   - 在记录认证失败的那一刻，对凭证来源（`session/<provider>.json`）取**内容指纹**
     （token + cookie + captured_at 的哈希；不可解析时退化为 mtime_ns:size）。
     该指纹随状态持久化，因此「先重登、再重启」也能被识别。
   - 此后任一时刻比较当前指纹：不同即说明凭证确已更新。指纹只覆盖**凭证内容**
     （token / cookie / user_id / device_id），**不含 `captured_at`**——后者每次
     `Session.save()` 都会刷新，把它算进去会让「重新保存了一次」冒充「已续期」，
     一次什么都没修好的静默刷新就能解除避让（见 `_FINGERPRINT_FIELDS` 的说明）。
4. 复位必须限定「认证成因」(Cause-scoped release)：
   - 仅当该节点**最近一次**失败严重度为 `auth` 时才复位（`cooldown_severity` 随冷却
     一并持久化）。若之后又发生 WAF/频控，说明是另一类问题，绝不代它释放。
   - 持有运维封禁租约的节点**永不**复位：租约表达的是账号级封禁（重登无效），
     与「凭证过期」是两回事。
   - 复位后仍有保护：节点进入恢复观察期 (OPT-022) 的串行探路与放大步调，
     避免被压制的需求在凭证刚更新时集中涌出。
5. 保守默认 (Fail-closed)：
   - 没有指纹记录（本机制启用前就已存在的冷却）→ 不复位，继续避让；
   - 凭证文件不可读 → 不复位；
   - 指纹相同 → 不复位。
   任何不确定都选择「继续避让」，宁可多等，不可误放。

[OPT-027] 本模块还提供「新凭证仍被拒」这一**关键判别证据**：
`note_auth_failure()` 额外返回「本次失败是否发生在一枚**已更新过**的凭证上」。
这排除了「凭证只是过期」这一唯一解释——凭证已经换过了，仍被上游 401 拒绝，
那么问题不在凭证上。该证据由 `CooldownTracker` 用于推断账号级封禁（见 failover.py）。

开关：`GATEWAY_DISABLE_CREDENTIAL_RENEWAL=1` 关闭；`GATEWAY_DISABLE_RISK_LEDGER=1`
一并关闭（与封禁租约 / 凭证失效检测同约定，保证纯内存冷却语义的测试套件不受影响）。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .risk_ledger import get_risk_ledger

logger = logging.getLogger("gateway.credential_renewal")

# 五个对等节点的凭证文件（与 session_store 中的定义一致，但此处只读取、不写入）
_SESSION_DIR = Path(__file__).resolve().parent.parent / "session"
_PROVIDER_SESSION_FILES: Dict[str, Path] = {
    "qwen": _SESSION_DIR / "qwen.json",
    "deepseek": _SESSION_DIR / "deepseek.json",
    "doubao": _SESSION_DIR / "doubao.json",
    "kimi": _SESSION_DIR / "kimi.json",
    "glm": _SESSION_DIR / "glm.json",
}

# 指纹中参与哈希的凭证字段 = **凭证内容**，不含 `captured_at`。
#
# [REFRESH-VERIFY] 为什么不含 `captured_at`：`Session.save()` 每次保存都会刷新它，
# 而"续期"这个判断要回答的是"凭证被换掉了吗"。把保存时间算进去，会让**任何**一次
# 保存都算续期——包括浏览器 profile 里还留着旧令牌、无头刷新把它原样再存一遍的
# 情况（令牌一个字节都没变）。实测后果是链条式的：
#   一次 60s 的 auth 冷却 ——> refreshed_providers() 报告已续期 ——> 冷却复位到 0
# 也就是说，一次什么都没修好的静默刷新，解除了 OPT-026 为"拿着同一个失效凭证
# 反复打"而设的避让，让节点立刻重新进入上游；同时 `note_auth_failure` 会把
# "同一枚凭证 + 一次空刷新"当成"新凭证仍被拒"，进而推断账号级封禁 (OPT-027)。
#
# 字段选择与 `_save_session_from_context` 捕获到的东西一致：真正重登时至少有一项
# 会变（令牌刷新，或安全 Cookie 轮换，或账号/设备身份变化）。
_FINGERPRINT_FIELDS = ("token", "cookie", "user_id", "device_id")


def credential_path(provider_key: str) -> Optional[Path]:
    """返回指定节点的凭证文件路径（未知节点返回 None）。

    独立成模块级函数，便于测试注入临时文件，绝不触碰真实 `session/`。
    """
    return _PROVIDER_SESSION_FILES.get((provider_key or "").strip().lower())


def _fingerprint_of_file(path: Optional[Path]) -> Optional[str]:
    """对凭证文件取内容指纹；不可读/不可解析时退化为 mtime_ns:size。

    只做只读访问：绝不写入、绝不修改凭证文件。
    """
    if path is None:
        return None
    try:
        st = path.stat()
    except OSError:
        return None
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except Exception:  # noqa: BLE001 - 解析失败退化为元数据指纹，绝不上抛
        return f"meta:{st.st_mtime_ns}:{st.st_size}"
    if not isinstance(data, dict):
        return f"meta:{st.st_mtime_ns}:{st.st_size}"
    parts: List[str] = []
    for key in _FINGERPRINT_FIELDS:
        parts.append(str(data.get(key, "")))
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:32]
    return f"sha:{digest}"


def credential_fingerprint(provider_key: str) -> Optional[str]:
    """返回指定节点头部凭证的当前指纹（无凭证/不可读时返回 None）。"""
    return _fingerprint_of_file(credential_path(provider_key))


class CredentialRenewalTracker:
    """凭证续期检测器：记录认证失败时刻的凭证指纹，并据此判定凭证是否已更新。

    本类**只回答一个事实**（「凭证变了吗」），不做任何避让状态的改动——
    真正的复位由 `CooldownTracker.release_auth_avoidance()` 执行，
    从而保持「判定」与「状态变更」分离，前者只读、可安全查询。
    """

    def __init__(self, directory: Optional[Path] = None, ledger: Any = None) -> None:
        self._lock = threading.RLock()
        # 复用风控台账的目录解析：自动遵循 GATEWAY_RISK_LEDGER_DIR 与 TESTING 隔离
        if directory is not None:
            self.directory = Path(directory)
        else:
            base = ledger if ledger is not None else get_risk_ledger()
            self.directory = Path(getattr(base, "directory", _SESSION_DIR))
        self.store_path = self.directory / "credential_renewal.json"
        # provider -> {"fingerprint": str, "at": float}
        self._records: Dict[str, Dict[str, Any]] = {}
        self._stats = {
            "renewals_recorded": 0,
            "renewals_detected": 0,
            "records_cleared": 0,
            # [OPT-027] 发生在「已更新过的凭证」上的认证失败次数——即
            # 「新凭证仍被拒」证据的出现次数（账号级封禁的判决依据）。
            "rejections_after_renewal": 0,
        }
        self._load()

    # ------------------------------------------------------------ 配置

    def is_disabled(self) -> bool:
        if os.getenv("GATEWAY_DISABLE_CREDENTIAL_RENEWAL", "").strip().lower() in ("1", "true", "yes"):
            return True
        # 与封禁租约 / 凭证失效检测同约定：关闭持久化风控状态时一并关闭
        return os.getenv("GATEWAY_DISABLE_RISK_LEDGER", "").strip().lower() in ("1", "true", "yes")

    # ------------------------------------------------------------ 持久化

    def _load(self) -> None:
        try:
            raw = self.store_path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except Exception:  # noqa: BLE001 - 缺失/损坏一律视为无记录（保守：不复位）
            return
        records = data.get("records") if isinstance(data, dict) else None
        if not isinstance(records, dict):
            return
        with self._lock:
            for pk, rec in records.items():
                if isinstance(rec, dict) and isinstance(rec.get("fingerprint"), str):
                    self._records[str(pk)] = {
                        "fingerprint": rec["fingerprint"],
                        "at": float(rec.get("at") or 0.0),
                    }

    def _save_locked(self) -> None:
        """POSIX 原子写（临时文件 + fsync + os.replace），失败仅记日志。"""
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            try:
                os.chmod(self.directory, 0o700)
            except OSError:
                pass
            temp = self.directory / f".{self.store_path.name}_{os.getpid()}_{threading.get_ident()}.tmp"
            try:
                fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                with open(fd, "w", encoding="utf-8") as f:
                    f.write(json.dumps({"records": self._records}, ensure_ascii=False, indent=2))
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(temp, self.store_path)
            finally:
                if temp.exists():
                    try:
                        temp.unlink(missing_ok=True)
                    except OSError:
                        pass
        except Exception as exc:  # noqa: BLE001 - 持久化失败绝不影响请求流程
            logger.warning("凭证续期指纹落盘失败 (已忽略): %s: %s", type(exc).__name__, exc)

    # ------------------------------------------------------------ 记录

    def note_auth_failure(self, provider_key: str, now: Optional[float] = None) -> bool:
        """记录一次认证失败：**此刻**的凭证指纹即为「失效凭证」的指纹。

        之所以在失败时刻取指纹而非在复位时反推：只有这样才能回答
        「与我失败时相比，凭证是否已经被换掉了」。

        [OPT-027] 返回一个额外事实：**本次失败是否发生在一枚已经更新过的凭证上**
        （即与上次记录的指纹不同）。这是「新凭证仍被拒」这一关键证据的唯一来源：
        它排除了「凭证只是过期」这一解释——凭证已经换过了，仍被上游以 401 拒绝，
        那么问题就不在凭证上。判定必须在**覆盖旧记录之前**完成，否则证据每次都自毁。
        """
        if self.is_disabled():
            return False
        pk = (provider_key or "").strip().lower()
        if not pk:
            return False
        fp = credential_fingerprint(pk)
        if fp is None:
            # 凭证文件不可读：不记录，从而永远无法据此复位（保守）
            return False
        renewed_since_last = False
        now = time.time() if now is None else now
        with self._lock:
            previous = self._records.get(pk)
            if previous is not None and previous.get("fingerprint") != fp:
                renewed_since_last = True
            self._records[pk] = {"fingerprint": fp, "at": now}
            self._stats["renewals_recorded"] += 1
            if renewed_since_last:
                self._stats["rejections_after_renewal"] += 1
            self._save_locked()
        return renewed_since_last

    def note_interception(self, provider_key: str, now: Optional[float] = None) -> bool:
        """记录一次**非认证类**拦截时刻的凭证指纹（[OPT-029]，只记基线，不做推断）。

        为什么人机验证也需要基线：被 RGV587 / 验证码拦截的节点，唯一正确的解除方式
        是**人工**执行 `login.py` 完成滑块验证，而这件事的可观测证据恰恰是**凭证文件
        变了**（登录成功后令牌必然刷新）。若不在 WAF 拦截时刻留下基线，日后指纹变化
        就无从比对，「人工已介入」这一事实将永远无法被网关识别——长效避让就会变成
        没有出口的永久冻结，正是 OPT-026 已经修过的那类自锁缺陷，只是换了个成因。

        与 `note_auth_failure` 的关键区别：**不返回**「本次失败是否发生在一枚更新的
        凭证上」，也不计入 `rejections_after_renewal`。那条证据链是给 OPT-027 的
        账号级封禁推断用的，判别式严格限定为「被 401 拒绝」；把人机验证混入其中会
        污染该判别式（在封禁成因完全不同的时候触发自动租约）。
        """
        if self.is_disabled():
            return False
        pk = (provider_key or "").strip().lower()
        if not pk:
            return False
        fp = credential_fingerprint(pk)
        if fp is None:
            return False
        now = time.time() if now is None else now
        with self._lock:
            self._records[pk] = {"fingerprint": fp, "at": now}
            self._stats["renewals_recorded"] += 1
            self._save_locked()
        return True

    def current_fingerprint(self, provider_key: str) -> Optional[str]:
        """返回该节点凭证的当前指纹（供调用方比对历史指纹，只读）。"""
        return credential_fingerprint(provider_key)

    def clear_provider(self, provider_key: str) -> None:
        """清除某节点的指纹记录（节点已恢复可用，无需再判定续期）。"""
        if self.is_disabled():
            return
        pk = (provider_key or "").strip().lower()
        if not pk:
            return
        with self._lock:
            if self._records.pop(pk, None) is not None:
                self._stats["records_cleared"] += 1
                self._save_locked()

    # ------------------------------------------------------------ 判定

    def renewed_providers(self, now: Optional[float] = None) -> List[str]:
        """返回「自上次认证失败以来凭证确已更新」的节点（只读判定）。

        保守规则：无记录、凭证不可读、指纹相同 → 一律不算已续期。
        """
        if self.is_disabled():
            return []
        with self._lock:
            snapshot = {pk: dict(rec) for pk, rec in self._records.items()}
        renewed: List[str] = []
        for pk, rec in snapshot.items():
            current = credential_fingerprint(pk)
            if current is None:
                continue
            if current != rec.get("fingerprint"):
                renewed.append(pk)
        if renewed:
            with self._lock:
                self._stats["renewals_detected"] += len(renewed)
        return sorted(renewed)

    def has_record(self, provider_key: str) -> bool:
        with self._lock:
            return (provider_key or "").strip().lower() in self._records

    def get_diagnostics(self, now: Optional[float] = None) -> Dict[str, Any]:
        now = time.time() if now is None else now
        with self._lock:
            records = {
                pk: {"since_seconds": round(max(0.0, now - float(rec.get("at") or 0.0)), 1)}
                for pk, rec in self._records.items()
            }
            stats = dict(self._stats)
        return {
            "enabled": not self.is_disabled(),
            "tracked_providers": sorted(records),
            "records": records,
            "renewed_providers": self.renewed_providers(now),
            "stats": stats,
            "hint": (
                "检测到凭证文件已更新，这些节点的认证级避让将（或已）被复位："
                "先以串行探路 + 放大步调重新进入（恢复观察期），达标后恢复常规并发。"
                "账号级封禁租约不受影响，仍由运维租约控制。"
            ) if self.renewed_providers(now) else "",
        }

    def clear(self) -> None:
        """清空所有指纹记录（运维手动解除 / 测试）。"""
        with self._lock:
            self._records.clear()
            self._save_locked()

    def clear_all_for_tests(self) -> None:
        with self._lock:
            self._records.clear()
            self._stats = {
                "renewals_recorded": 0,
                "renewals_detected": 0,
                "records_cleared": 0,
                "rejections_after_renewal": 0,
            }


# 全局单例
credential_renewal = CredentialRenewalTracker()


def get_credential_renewal() -> CredentialRenewalTracker:
    """获取全局凭证续期检测器单例。"""
    return credential_renewal
