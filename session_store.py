"""令牌/会话的本地存取与解析。

优先级：环境变量 > session/<provider>.json（login.py 扫码登录自动写入）。
"""

from __future__ import annotations

import json
import os
import threading
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

SESSION_DIR = Path(__file__).resolve().parent / "session"
QWEN_SESSION_FILE = SESSION_DIR / "qwen.json"
DEEPSEEK_SESSION_FILE = SESSION_DIR / "deepseek.json"
DOUBAO_SESSION_FILE = SESSION_DIR / "doubao.json"
KIMI_SESSION_FILE = SESSION_DIR / "kimi.json"
GLM_SESSION_FILE = SESSION_DIR / "glm.json"

# 令牌默认信任时长：超过后服务启动时会尝试静默刷新（复用已登录的浏览器配置）
MAX_AGE = 6 * 3600

_file_io_lock = threading.RLock()


@dataclass
class Session:
    token: str
    cookie: str = ""
    user_agent: str = ""
    user_id: str = ""
    device_id: str = ""
    captured_at: float = field(default_factory=time.time)

    @property
    def age(self) -> float:
        return time.time() - self.captured_at

    def save(self, path: Path) -> None:
        with _file_io_lock:
            path = Path(path)
            parent = path.parent
            parent.mkdir(parents=True, exist_ok=True)
            try:
                os.chmod(parent, 0o700)
            except OSError:
                pass

            temp_file = parent / f".{path.stem}_{os.getpid()}_{threading.get_ident()}.tmp"
            try:
                flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
                fd = os.open(temp_file, flags, 0o600)
                with open(fd, "w", encoding="utf-8") as f:
                    f.write(
                        json.dumps(asdict(self), ensure_ascii=False, indent=2)
                    )
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(temp_file, path)
            finally:
                if temp_file.exists():
                    try:
                        temp_file.unlink(missing_ok=True)
                    except OSError:
                        pass

    @classmethod
    def load(cls, path: Path) -> Optional["Session"]:
        with _file_io_lock:
            path = Path(path)
            if not path.exists():
                return None
            try:
                content = path.read_text(encoding="utf-8").strip()
                if not content:
                    warnings.warn(f"Session 文件为空: {path}", UserWarning, stacklevel=2)
                    return None
                data = json.loads(content)
                if not isinstance(data, dict):
                    warnings.warn(f"Session 文件格式异常(非字典): {path}", UserWarning, stacklevel=2)
                    return None
                valid_keys = {f for f in cls.__annotations__}
                return cls(**{k: v for k, v in data.items() if k in valid_keys})
            except Exception as e:
                warnings.warn(f"读取或解析 Session 文件失败 {path}: {e}", UserWarning, stacklevel=2)
                return None


def _resolve_session(env_keys: list[str], cookie_env: str, session_file: Path) -> Optional[Session]:
    token = next((os.getenv(k, "").strip() for k in env_keys if os.getenv(k, "").strip()), "")
    if token:
        return Session(token=token, cookie=os.getenv(cookie_env, "").strip())
    return Session.load(session_file)


def resolve_qwen() -> Optional[Session]:
    """环境变量优先，其次 session/qwen.json。"""
    return _resolve_session(["QWEN_AUTH_TOKEN"], "QWEN_COOKIE", QWEN_SESSION_FILE)


def resolve_deepseek() -> Optional[Session]:
    """环境变量优先，其次 session/deepseek.json。"""
    return _resolve_session(["DEEPSEEK_AUTH_TOKEN"], "DEEPSEEK_COOKIE", DEEPSEEK_SESSION_FILE)


def resolve_doubao() -> Optional[Session]:
    """环境变量优先，其次 session/doubao.json。"""
    return _resolve_session(["DOUBAO_AUTH_TOKEN", "DOUBAO_SESSION_ID"], "DOUBAO_COOKIE", DOUBAO_SESSION_FILE)


def resolve_kimi() -> Optional[Session]:
    """环境变量优先，其次 session/kimi.json。"""
    return _resolve_session(["KIMI_AUTH_TOKEN", "KIMI_REFRESH_TOKEN"], "KIMI_COOKIE", KIMI_SESSION_FILE)


def resolve_glm() -> Optional[Session]:
    """环境变量优先，其次 session/glm.json。"""
    return _resolve_session(["GLM_AUTH_TOKEN", "GLM_REFRESH_TOKEN", "CHATGLM_AUTH_TOKEN"], "GLM_COOKIE", GLM_SESSION_FILE)


# [DAY-ZERO] 「本机是否配置了该节点的凭证」——全项目**唯一**判据。
#
# 判据只有一条：解析器能否拿到一个带 token 的会话（env 优先，其次 session/<provider>.json）。
# 这与 `server.py:_make_provider` 抛「未检测到 … 令牌」的条件、以及健康探针
# (`healing/monitor.py` 的 `if not sess or not sess.token`) **逐字一致**——三处必须问
# 同一个问题，否则「算不算已配置」会出现第二份定义，而第二份定义必然漂移。
#
# 为什么它必须存在：没有凭证是**纯本地的配置缺口**，不是上游的 401。过去网关把两者
# 混为一谈，于是一个全新克隆的第一次请求就给自己记下了认证级风控避让（详见
# `providers/failover.py:record_failure` 的 [DAY-ZERO] 段与 CLAUDE.md 同名条目）。
_PROVIDER_RESOLVERS = {
    "qwen": resolve_qwen,
    "deepseek": resolve_deepseek,
    "doubao": resolve_doubao,
    "kimi": resolve_kimi,
    "glm": resolve_glm,
}


def is_configured(provider_key: str) -> bool:
    """本机是否配置了该节点的凭证（只读，绝不写盘、绝不联网）。

    未知节点返回 True（不属于本判据的处理范围，交由原有语义决定）。
    解析异常同样返回 True：**判定失败必须退回旧行为**，绝不能因为一次读取异常就把
    一个已配置的节点误判成「未配置」——那会让一次真实的上游 401 逃过认证级避让。
    """
    resolver = _PROVIDER_RESOLVERS.get((provider_key or "").strip().lower())
    if resolver is None:
        return True
    try:
        session = resolver()
    except Exception:  # noqa: BLE001 - 判定失败一律按「已配置」处理（保守）
        return True
    return bool(session and session.token)
