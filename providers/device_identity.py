"""节点设备身份持久化：稳定的设备标识符 (providers/device_identity.py)。

实现 [P0] OPT-030：
1. 现场问题（实测复现）：Kimi 的 `X-Msh-Device-Id` 与 GLM 的 `X-Device-Id` 在
   **每一次上游请求**上都是一个全新的随机值——因为 `_make_provider` 在每次派发
   尝试时才实例化 Provider（`execute_chat` 内 `provider_factory(...)`），而两者的
   构造器写作 `device_id or random.randint(...)` / `device_id or uuid.uuid4().hex`，
   且它们的 session 文件里 `device_id` 为空（登录流程只在豆包分支捕获了 TEA web_id）。
   实测（连续三次构造）：
       kimi dev=7096…  → 7884…  → 7989…        （每次请求都不同）
       glm  dev=a2b3…  → a880…  → ad3a…        （每次请求都不同）
   Qwen 与 Doubao 在**有 cookie** 时是稳定的，但在「令牌来自环境变量、无 cookie」
   的部署形态下（凭证续期扫描接口明确支持这种形态）同样每次请求换设备：
       qwen dev=9276…  → 3c65…  → 5aa4…
       doubao dev=同一毫秒 → 每毫秒不同;  fp=verify_…（每次请求都不同）
2. 为什么这是风控级缺陷（而非卫生问题）：「同一账号 + 同一 Cookie/令牌 + 每次请求
   一个不同的设备标识」是反爬系统能拿到的最强自动化特征之一。一台真实浏览器在
   同一份 profile 下设备标识是**恒定**的；它随请求变化，只可能来自自动化。
   这正是 OPT-024 只能「全局降速」的那种**系统性成因**（根因在我们自己的客户端
   足迹上，换节点与按账号避让都无效），但它同时又是一个**可修复**的根因，而不是
   只能缓解的既成事实——降速治标，身份稳定治本。
3. 修复原则（身份与凭证解耦）：设备标识必须**一次生成、永久复用**。它不是凭证，
   因此**绝不随重登/令牌刷新而改变**（真实设备在重新登录后依然是同一台设备）；
   也绝不按进程、按请求重新生成。落盘位置 `session/device_identity.json`
   （目录 0o700 / 文件 0o600，POSIX 原子写 `os.replace` + `os.fsync`），与凭证文件
   同级但独立。
4. 节点间必须**互不相同**：五家平台若共用同一个设备标识，就制造出一条跨平台的
   关联线索——这正是 OPT-024 警告的那类「共享足迹」。因此按节点分别持久化。
5. 保守默认（fail-closed）：调用方传入的显式 `device_id`（登录捕获的真实值）与
   cookie 中已存在的标识**永远优先**，本模块只接管「原本要随机生成」的那一格；
   存储不可读时退回调用方给出的生成器（退化为旧行为，不抛错）；
   `GATEWAY_DISABLE_DEVICE_IDENTITY=1` 完全退化为修复前的随机行为。

开关：`GATEWAY_DISABLE_DEVICE_IDENTITY=1` 关闭；`GATEWAY_DEVICE_IDENTITY_DIR`
覆盖落盘目录；`TESTING=1` 时使用临时目录（绝不污染开发者真实 session 目录）。
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger("gateway.device_identity")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DIR = _PROJECT_ROOT / "session"


def _resolve_default_dir() -> Path:
    """解析落盘目录：环境覆盖 > TESTING 临时目录 > 项目 session/。"""
    override = os.getenv("GATEWAY_DEVICE_IDENTITY_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    if os.getenv("TESTING", "").strip().lower() in ("1", "true", "yes"):
        return Path(tempfile.mkdtemp(prefix="gateway_device_identity_"))
    return _DEFAULT_DIR


class DeviceIdentityStore:
    """按节点持久化设备标识符：生成一次，之后恒定复用。

    本类**只负责「同一个 key 永远得到同一个字符串」**这一件事，不关心标识的格式
    ——格式由调用方给出的 `generate` 决定（Kimi 要 19 位十进制、GLM/Qwen 要 32 位
    十六进制、Doubao 要毫秒级十进制），从而在不改变既有报文形态的前提下消除抖动。
    """

    def __init__(self, directory: Optional[Path] = None) -> None:
        self._lock = threading.RLock()
        self.directory = Path(directory) if directory is not None else _resolve_default_dir()
        self.store_path = self.directory / "device_identity.json"
        # key -> {"id": str, "created_at": float}
        self._devices: Dict[str, Dict[str, Any]] = {}
        self._loaded = False
        self._stats = {
            "created": 0,          # 首次生成（新身份）
            "reused": 0,           # 复用既有身份（本应随机、实际稳定）
            "revived_from_disk": 0,  # 跨进程/跨重启复用
            "bypassed": 0,         # 关闭开关或因存储不可用而退化
            "persist_failures": 0,
        }
        self._load()

    # ------------------------------------------------------------ 配置

    def is_disabled(self) -> bool:
        return os.getenv("GATEWAY_DISABLE_DEVICE_IDENTITY", "").strip().lower() in ("1", "true", "yes")

    # ------------------------------------------------------------ 持久化

    def _load(self) -> None:
        with self._lock:
            if self._loaded:
                return
            self._loaded = True
            try:
                raw = self.store_path.read_text(encoding="utf-8")
                data = json.loads(raw)
            except Exception:  # noqa: BLE001 - 缺失/损坏一律视为「尚无身份」，重新生成
                return
            devices = data.get("devices") if isinstance(data, dict) else None
            if not isinstance(devices, dict):
                return
            for key, rec in devices.items():
                if isinstance(rec, dict) and str(rec.get("id") or ""):
                    self._devices[str(key)] = {
                        "id": str(rec["id"]),
                        "created_at": float(rec.get("created_at") or 0.0),
                    }
            if self._devices:
                self._stats["revived_from_disk"] = len(self._devices)

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
                    f.write(json.dumps({"devices": self._devices}, ensure_ascii=False, indent=2))
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(temp, self.store_path)
            finally:
                if temp.exists():
                    try:
                        temp.unlink(missing_ok=True)
                    except OSError:
                        pass
        except Exception as exc:  # noqa: BLE001 - 落盘失败绝不影响请求流程
            with self._lock:
                self._stats["persist_failures"] += 1
            logger.warning("设备身份落盘失败 (已忽略): %s: %s", type(exc).__name__, exc)

    # ------------------------------------------------------------ 取用

    def get(self, key: str, generate: Callable[[], str]) -> str:
        """返回该 key 的稳定标识；首次调用时用 `generate()` 生成并永久保存。

        `generate` 只在**首次**被调用——这正是修复点：调用方的随机表达式从此只
        执行一次，之后每次请求都拿到同一个字符串。
        """
        name = (key or "").strip()
        if not name:
            return generate()
        if self.is_disabled():
            with self._lock:
                self._stats["bypassed"] += 1
            return generate()
        with self._lock:
            rec = self._devices.get(name)
            if rec is not None and rec.get("id"):
                self._stats["reused"] += 1
                return str(rec["id"])
        # 生成放在锁外（生成器可能很慢或含随机源），随后在锁内二次确认，
        # 避免两个线程同时首生成时互相覆盖。
        candidate = generate()
        with self._lock:
            rec = self._devices.get(name)
            if rec is not None and rec.get("id"):
                self._stats["reused"] += 1
                return str(rec["id"])
            self._devices[name] = {"id": str(candidate), "created_at": time.time()}
            self._stats["created"] += 1
            self._save_locked()
        logger.info(
            "[OPT-030] 已为该节点生成并持久化稳定设备标识 (%s)：此后每次请求复用，"
            "不再随进程/请求重新生成", name
        )
        return str(candidate)

    def peek(self, key: str) -> Optional[str]:
        """只读查看（不生成）：用于诊断与测试断言。"""
        with self._lock:
            rec = self._devices.get((key or "").strip())
        return str(rec["id"]) if rec else None

    def keys(self) -> list[str]:
        with self._lock:
            return sorted(self._devices)

    # ------------------------------------------------------------ 诊断

    @staticmethod
    def _mask(value: str) -> str:
        """只暴露前缀：设备标识是反爬系统最敏感的字段，诊断输出无需完整明文。"""
        v = value or ""
        return f"{v[:6]}…({len(v)}位)" if len(v) > 8 else ("…" if v else "")

    def get_diagnostics(self) -> Dict[str, Any]:
        now = time.time()
        with self._lock:
            devices = {
                key: {
                    "preview": self._mask(str(rec.get("id") or "")),
                    "age_seconds": round(max(0.0, now - float(rec.get("created_at") or 0.0)), 1),
                }
                for key, rec in self._devices.items()
            }
            stats = dict(self._stats)
        return {
            "enabled": not self.is_disabled(),
            "store_path": str(self.store_path),
            "tracked_nodes": sorted(devices),
            "devices": devices,
            "stats": stats,
            "hint": (
                "设备标识一经生成即永久复用，且**不随重登/令牌刷新改变**（身份与凭证解耦）。"
                "GATEWAY_DISABLE_DEVICE_IDENTITY=1 可退化为修复前的行为。"
            ),
        }

    def clear_for_tests(self, key: Optional[str] = None) -> None:
        with self._lock:
            if key:
                self._devices.pop((key or "").strip(), None)
            else:
                self._devices.clear()
            self._stats = {
                "created": 0,
                "reused": 0,
                "revived_from_disk": 0,
                "bypassed": 0,
                "persist_failures": 0,
            }
            self._save_locked()


# 全局单例
device_identity_store = DeviceIdentityStore()


def stable_device_id(key: str, generate: Callable[[], str]) -> str:
    """获取某 key 的稳定设备标识（首次生成后永久复用）。

    调用方把**原本的随机表达式**作为 `generate` 传入即可，例如
    `stable_device_id("glm", lambda: uuid.uuid4().hex)`——语义上仍是「没有设备标识
    时造一个」，但只造一次。
    """
    return device_identity_store.get(key, generate)


def get_device_identity_store() -> DeviceIdentityStore:
    """获取全局设备身份存储单例。"""
    return device_identity_store
