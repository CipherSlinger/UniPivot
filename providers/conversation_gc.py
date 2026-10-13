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
        """对队列中所有待删除会话各尝试一次（启动时与后台周期调用）。"""
        if self.is_disabled() or self._resolver is None:
            return {"attempted": 0, "succeeded": 0}
        with self._lock:
            items: Tuple[Tuple[str, str], ...] = tuple(
                (entry["provider"], entry["conversation_id"]) for entry in self._pending.values()
            )
        attempted = succeeded = 0
        for pk, cid in items:
            try:
                provider = self._resolver(pk)
            except Exception:  # noqa: BLE001 - 构造失败视为本轮未成功，保留队列
                provider = None
            attempted += 1
            if await self._attempt(provider, pk, cid):
                succeeded += 1
        return {"attempted": attempted, "succeeded": succeeded}

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
        return {
            "enabled": not self.is_disabled(),
            "directory": str(self.directory),
            "queue_path": str(self.queue_path),
            "max_attempts": self.max_attempts,
            "pending_count": len(pending),
            "pending": pending[:50],
            "stats": stats,
        }

    def clear(self) -> None:
        """清空队列与统计（测试用）。"""
        with self._lock:
            self._pending.clear()
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
