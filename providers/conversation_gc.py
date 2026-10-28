"""远端临时会话删除的持久化重试队列 (providers/conversation_gc.py)。

实现 [P1] OPT-016：
1. 持久化删除队列 (Durable Deletion Queue):
   - 无痕/临时调用完成后需要删除上游会话（远端防膨胀），历史实现仅
     `asyncio.create_task(delete_conversation(...))` fire-and-forget：
     删除失败或被进程重启打断即永久丢失，导致用户网页端侧边栏持续堆积
     临时会话。本模块将待删除会话以 POSIX 原子写 (os.replace + os.fsync,
     目录 0o700 / 文件 0o600) 落盘，形成跨重启存活的删除队列。
2. 有界重试与退避 (Bounded Retry):
   - 删除失败时累计尝试次数并保留在队列，由后台周期 drain 重试；
     达到 GATEWAY_CONVERSATION_GC_MAX_ATTEMPTS 后放弃并记日志，
     杜绝无界重试与队列无界增长。
3. 遥测与诊断:
   - GET /v1/diagnostics 暴露 conversation_gc 段（待删除数、成功/失败/
     重试/放弃计数），供仪表盘审计远端会话清理健康度。
   - GATEWAY_DISABLE_CONVERSATION_GC=1 可完全关闭（退化为历史 fire-and-forget）。

实现 [P1] OPT-017：风控冷却感知的会话回收割让 (Cooldown-Aware Deletion Gate)
   - 删除临时会话本身也是一次真实上游请求。当节点正处于风控避让冷却期时
     发起删除等同于持续触碰已被封禁的节点，会人为延长封禁窗口——这与健康
     探针（OPT-012）、令牌刷新（OPT-013）属于同一类隐蔽通道。
   - 因此本模块在真正发起删除前查询 CooldownTracker：冷却期内的节点其删除
     动作被**推迟**（记录保留在队列，且不消耗重试次数），冷却解除后由后台
     drain 自然补做。GATEWAY_DISABLE_CONVERSATION_GC_COOLDOWN_GATE=1 可回退。

磁盘写入全部为 best-effort：落盘失败绝不阻断、绝不改变请求处理流程。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger("gateway.conversation_gc")

_MAX_EVIDENCE_CHARS = 200


class ConversationGC:
    """远端临时会话删除的持久化重试队列。"""

    def __init__(self, directory: Optional[Path] = None) -> None:
        self.directory = self._resolve_directory(directory)
        self.queue_path = self.directory / "conversation_gc.json"
        self.max_attempts = int(os.getenv("GATEWAY_CONVERSATION_GC_MAX_ATTEMPTS", "5"))
        self._lock = threading.RLock()
        # key = "<provider>:<conversation_id>" -> entry
        self._pending: Dict[str, Dict[str, Any]] = {}
        self._resolver: Optional[Callable[[str], Any]] = None
        self._cooldown_gate: Optional[Callable[[str], float]] = None
        # [OPT-034] 后台避让成因判据（优先于冷却浮点门控；未注册时退回后者）
        self._avoidance_gate: Optional[Callable[[str], str]] = None
        self._defer_stats: Dict[str, Any] = {"total": 0, "per_provider": {}}
        self._stats = {
            "scheduled": 0,
            "deleted": 0,
            "retried": 0,
            "failed": 0,
            "dropped": 0,
            "persist_failures": 0,
        }
        self._load()

    # ------------------------------------------------------------ 配置

    def _resolve_directory(self, directory: Optional[Path]) -> Path:
        if directory is not None:
            return Path(directory)
        env_dir = os.getenv("GATEWAY_CONVERSATION_GC_DIR", "").strip()
        if env_dir:
            return Path(env_dir)
        if os.getenv("TESTING") == "1":
            return Path(tempfile.mkdtemp(prefix="gateway_conv_gc_"))
        return Path(__file__).resolve().parent.parent / "session"

    def is_disabled(self) -> bool:
        return os.getenv("GATEWAY_DISABLE_CONVERSATION_GC", "").strip().lower() in ("1", "true", "yes")

    def set_resolver(self, resolver: Callable[[str], Any]) -> None:
        """注册 provider 构造器：入参 provider 名，返回可删除会话的 provider 实例。"""
        self._resolver = resolver

    def set_cooldown_gate(self, gate: Callable[[str], float]) -> None:
        """[OPT-017] 注册风控冷却查询回调：入参 provider 名，返回剩余冷却秒数。

        返回 > 0 时表示该节点正处风控避让期，其临时会话删除必须推迟——
        删除同样是一次真实上游请求，冷却期发起只会持续触碰已封禁节点。
        """
        self._cooldown_gate = gate

    def set_avoidance_gate(self, gate: Callable[[str], str]) -> None:
        """[OPT-034] 注册后台避让成因判据：返回 "" 表示可删除，否则返回成因。

        成因来自 `CooldownTracker.touch_avoidance_reason`：`disabled`（运维禁用）、
        `quarantined`（运维/自动租约）、`cooldown`（自动冷却）。注册后本判据**优先于**
        冷却浮点门控，因为只有它能区分「禁用」与「冷却」——两者对运维的含义与解除方式
        完全不同（改配置 vs 等待），笼统报一句「冷却中」正是本项目反复修复的误导类型。
        未注册时退回 `set_cooldown_gate` 的既有行为，保持向后兼容。
        """
        self._avoidance_gate = gate

    def _cooldown_remaining(self, provider_key: str) -> float:
        """查询节点剩余风控冷却秒数（未注册回调或已关闭门控时返回 0）。"""
        if self._gate_disabled():
            return 0.0
        gate = self._cooldown_gate
        if gate is None:
            return 0.0
        try:
            return max(0.0, float(gate(provider_key)))
        except Exception:  # noqa: BLE001 - 门控查询失败绝不阻断回收流程
            return 0.0

    def _gate_disabled(self) -> bool:
        """[OPT-017/034] 会话回收冷却门控的总开关。"""
        return os.getenv("GATEWAY_DISABLE_CONVERSATION_GC_COOLDOWN_GATE", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )

    def _avoidance_reason(self, provider_key: str) -> str:
        """[OPT-034] 查询节点后台避让成因（"" 表示可删除）。

        优先使用成因判据（能区分禁用/租约/冷却）；未注册时用冷却浮点门控折算为
        `cooldown`，与 OPT-017 的既有行为逐值一致。
        """
        if self._gate_disabled():
            return ""
        gate = self._avoidance_gate
        if gate is not None:
            try:
                return str(gate(provider_key) or "")
            except Exception:  # noqa: BLE001 - 判据查询失败绝不阻断回收流程
                return ""
        return "cooldown" if self._cooldown_remaining(provider_key) > 0 else ""

    def _record_defer(self, provider_key: str, remaining: float, cause: str = "cooldown") -> None:
        with self._lock:
            self._defer_stats["total"] += 1
            per = self._defer_stats["per_provider"]
            key = (provider_key or "unknown").lower()
            per[key] = per.get(key, 0) + 1
            by_cause = self._defer_stats.setdefault("per_cause", {})
            by_cause[cause] = by_cause.get(cause, 0) + 1
        _CAUSE_LABEL = {
            "disabled": "已被运维禁用 (DISABLED_PROVIDERS)",
            "quarantined": "处于运维/自动封禁租约期",
            "cooldown": "处于风控避让冷却中",
            # [DAY-ZERO] 配置缺口，不是风控：该账号从未被触碰，说成封禁会误导运维。
            "unconfigured": "本机尚未配置凭证",
            # [TOUCH-CAUSE] 上游故障三类，同样不是风控。说成「避免触碰被封禁的账号」
            # 会让运维去排查一个从未发生的封禁。
            "timeout": "处于上游超时避让中",
            "truncated": "处于上游截断避让中",
            "upstream_error": "处于上游故障避让中",
        }
        # 上游故障的收尾语与风控/禁用不同：这里推迟删除的理由是「别在故障期间再发
        # 一次上游请求」，不是「别碰被封禁的账号」。共用一句收尾语正是本条要消除的
        # 误导——文案也必须跟着成因分支。
        _UPSTREAM_FAULT_TAIL = "（上游故障期间不追加上游请求）"
        is_upstream_fault = cause in ("timeout", "truncated", "upstream_error")
        detail = f" (剩余 {remaining:.0f}s)" if cause == "cooldown" or is_upstream_fault else ""
        tail = (
            _UPSTREAM_FAULT_TAIL
            if is_upstream_fault
            else "，推迟其临时会话删除以避免触碰被封禁/已下线的账号"
        )
        logger.warning(
            "[会话回收] 节点 %s %s%s%s",
            provider_key,
            _CAUSE_LABEL.get(cause, "处于后台避让期"),
            detail,
            tail,
        )

    @staticmethod
    def _key(provider_key: str, conversation_id: str) -> str:
        return f"{(provider_key or 'unknown').lower()}:{conversation_id}"

    # ------------------------------------------------------------ 原子写

    def _atomic_write(self, content: str) -> bool:
        with self._lock:
            try:
                self.queue_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.chmod(self.queue_path.parent, 0o700)
                except OSError:
                    pass
                temp_file = self.queue_path.parent / (
                    f".{self.queue_path.name}_{os.getpid()}_{threading.get_ident()}.tmp"
                )
                try:
                    fd = os.open(temp_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                    with open(fd, "w", encoding="utf-8") as f:
                        f.write(content)
                        f.flush()
                        os.fsync(f.fileno())
                    os.replace(temp_file, self.queue_path)
                finally:
                    if temp_file.exists():
                        try:
                            temp_file.unlink(missing_ok=True)
                        except OSError:
                            pass
                return True
            except Exception as exc:  # noqa: BLE001
                self._stats["persist_failures"] += 1
                logger.warning("会话删除队列落盘失败 (已忽略): %s: %s", type(exc).__name__, exc)
                return False

    def _load(self) -> None:
        if self.is_disabled():
            return
        with self._lock:
            try:
                if not self.queue_path.exists():
                    return
                raw = self.queue_path.read_text(encoding="utf-8").strip()
                if not raw:
                    return
                data = json.loads(raw)
                pending = data.get("pending") if isinstance(data, dict) else None
                if not isinstance(pending, list):
                    return
                for entry in pending:
                    if not isinstance(entry, dict):
                        continue
                    pk = str(entry.get("provider") or "")
                    cid = str(entry.get("conversation_id") or "")
                    if not pk or not cid:
                        continue
                    self._pending[self._key(pk, cid)] = {
                        "provider": pk,
                        "conversation_id": cid,
                        "attempts": int(entry.get("attempts", 0) or 0),
                        "created_at": float(entry.get("created_at", time.time()) or time.time()),
                        "last_error": str(entry.get("last_error", "")),
                    }
                if self._pending:
                    logger.warning("[会话回收] 已从磁盘恢复 %d 个待删除会话", len(self._pending))
            except Exception as exc:  # noqa: BLE001
                logger.warning("读取会话删除队列失败 (已忽略): %s: %s", type(exc).__name__, exc)

    def _persist(self) -> None:
        if self.is_disabled():
            return
        with self._lock:
            payload = {
                "pending": list(self._pending.values()),
                "saved_at": round(time.time(), 3),
            }
        try:
            content = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        except Exception:  # noqa: BLE001
            return
        self._atomic_write(content)

    # ------------------------------------------------------------ 入队 / 调度

    def enqueue(self, provider_key: str, conversation_id: str) -> None:
        """将待删除会话写入持久化队列（幂等，已存在则保留原记录）。"""
        if self.is_disabled() or not conversation_id:
            return
        key = self._key(provider_key, conversation_id)
        with self._lock:
            if key in self._pending:
                return
            self._pending[key] = {
                "provider": (provider_key or "unknown").lower(),
                "conversation_id": conversation_id,
                "attempts": 0,
                "created_at": round(time.time(), 3),
                "last_error": "",
            }
        self._persist()

    def schedule(self, provider: Any, provider_key: str, conversation_id: str) -> None:
        """入队并异步尝试删除；失败则保留在队列等待后台重试。

        无运行中的事件循环（或关闭 GC）时安全降级：关闭 GC 时退化为历史
        fire-and-forget 行为，无循环时仅入队不发起删除（等待 drain）。
        """
        if not conversation_id:
            return
        if self.is_disabled():
            if provider is not None and hasattr(provider, "delete_conversation"):
                try:
                    asyncio.get_running_loop()
                    asyncio.create_task(provider.delete_conversation(conversation_id))
                except RuntimeError:
                    pass
            return
        self.enqueue(provider_key, conversation_id)
        self._stats["scheduled"] += 1
        # [OPT-017] 冷却期节点推迟删除：记录保留在队列，不消耗重试次数
        # [OPT-034] 判据扩为全部后台避让成因（运维禁用 ∪ 租约 ∪ 冷却）
        reason = self._avoidance_reason(provider_key)
        if reason:
            self._record_defer(provider_key, self._cooldown_remaining(provider_key), reason)
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return  # 无事件循环：保留在队列，等待后台 drain
        try:
            asyncio.create_task(self._attempt(provider, provider_key, conversation_id))
        except Exception:  # noqa: BLE001 - 调度失败也必须保留队列记录
            pass

    # ------------------------------------------------------------ 删除尝试

    async def _attempt(self, provider: Any, provider_key: str, conversation_id: str) -> bool:
        """尝试删除单个会话。成功则出队，失败则累计次数（达上限后放弃）。"""
        key = self._key(provider_key, conversation_id)
        if provider is None or not hasattr(provider, "delete_conversation"):
            return False
        error_msg = ""
        try:
            ok = bool(await provider.delete_conversation(conversation_id))
        except Exception as exc:  # noqa: BLE001
            ok = False
            error_msg = f"{type(exc).__name__}: {exc}"

        entry_ref: Optional[Dict[str, Any]] = None
        with self._lock:
            if ok:
                if self._pending.pop(key, None) is not None:
                    self._stats["deleted"] += 1
            else:
                entry = self._pending.get(key)
                if entry is None:
                    return False
                entry["attempts"] = int(entry.get("attempts", 0)) + 1
                entry["last_error"] = (error_msg or "delete returned falsy")[:_MAX_EVIDENCE_CHARS]
                entry["last_attempt"] = round(time.time(), 3)
                self._stats["failed"] += 1
                if entry["attempts"] >= self.max_attempts:
                    self._pending.pop(key, None)
                    self._stats["dropped"] += 1
                    entry_ref = entry
                else:
                    self._stats["retried"] += 1
        if entry_ref is not None:
            logger.warning(
                "[会话回收] 会话 %s/%s 删除重试达上限 (%d 次)，放弃：%s",
                entry_ref.get("provider"),
                entry_ref.get("conversation_id"),
                entry_ref.get("attempts"),
                entry_ref.get("last_error"),
            )
        self._persist()
        return ok

    async def drain_once(self) -> Dict[str, int]:
        """对队列中所有待删除会话各尝试一次（启动时与后台周期调用）。

        处于风控冷却期的节点被跳过（记录保留、不消耗重试次数），
        冷却解除后由后续轮次自然补做。
        """
        if self.is_disabled() or self._resolver is None:
            return {"attempted": 0, "succeeded": 0, "deferred": 0}
        with self._lock:
            items: Tuple[Tuple[str, str], ...] = tuple(
                (entry["provider"], entry["conversation_id"]) for entry in self._pending.values()
            )
        attempted = succeeded = deferred = 0
        for pk, cid in items:
            # [OPT-034] 与 schedule 同一判据：禁用/租约/冷却一律推迟
            reason = self._avoidance_reason(pk)
            if reason:
                self._record_defer(pk, self._cooldown_remaining(pk), reason)
                deferred += 1
                continue
            try:
                provider = self._resolver(pk)
            except Exception:  # noqa: BLE001 - 构造失败视为本轮未成功，保留队列
                provider = None
            attempted += 1
            if await self._attempt(provider, pk, cid):
                succeeded += 1
        return {"attempted": attempted, "succeeded": succeeded, "deferred": deferred}

    # ------------------------------------------------------------ 诊断

    def get_diagnostics(self) -> Dict[str, Any]:
        with self._lock:
            pending = [
                {
                    "provider": e.get("provider"),
                    "conversation_id": e.get("conversation_id"),
                    "attempts": int(e.get("attempts", 0)),
                    "last_error": e.get("last_error", ""),
                }
                for e in self._pending.values()
            ]
            stats = dict(self._stats)
            defer_stats = {
                "total": int(self._defer_stats.get("total", 0)),
                "per_provider": dict(self._defer_stats.get("per_provider", {})),
                # [OPT-034] 按成因拆分：使「谁在推迟」可观测，而不是笼统一句「冷却中」
                "per_cause": dict(self._defer_stats.get("per_cause", {})),
            }
        return {
            "enabled": not self.is_disabled(),
            "directory": str(self.directory),
            "queue_path": str(self.queue_path),
            "max_attempts": self.max_attempts,
            "pending_count": len(pending),
            "pending": pending[:50],
            "stats": stats,
            "cooldown_gate_enabled": (
                self._cooldown_gate is not None
                and os.getenv("GATEWAY_DISABLE_CONVERSATION_GC_COOLDOWN_GATE", "").strip().lower()
                not in ("1", "true", "yes")
            ),
            # [OPT-034] 成因判据启用状态（启用时它优先于冷却浮点门控）
            "avoidance_gate_enabled": self._avoidance_gate is not None and not self._gate_disabled(),
            "deferred_by_cooldown": defer_stats,
        }

    def clear(self) -> None:
        """清空队列与统计（测试用）。"""
        with self._lock:
            self._pending.clear()
            self._defer_stats = {"total": 0, "per_provider": {}, "per_cause": {}}
            self._stats = {
                "scheduled": 0,
                "deleted": 0,
                "retried": 0,
                "failed": 0,
                "dropped": 0,
                "persist_failures": 0,
            }
            try:
                self.queue_path.unlink(missing_ok=True)
            except OSError:
                pass


# 全局单例
conversation_gc = ConversationGC()


def get_conversation_gc() -> ConversationGC:
    """获取全局会话回收队列单例。"""
    return conversation_gc
