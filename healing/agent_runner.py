"""本地智能体环境发现、自举安装与执行器 (healing/agent_runner.py)。

负责检测本地 Claude Code (claude) 或 Codex (codex) CLI 环境，
在缺失时支持后台静默自举安装，并在自愈任务触发时安全调起智能体子进程。
"""

from __future__ import annotations

import asyncio
import glob
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger("healing.agent_runner")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"


@dataclass
class AgentInfo:
    name: str  # "claude" | "codex"
    path: str  # 可执行文件绝对路径
    version: Optional[str] = None


def _get_version(executable: str) -> Optional[str]:
    """尝试获取 CLI 命令行工具版本。"""
    try:
        res = subprocess.run(
            [executable, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception as e:
        logger.debug("获取工具 %s 版本失败: %s", executable, e)
    return None


def find_agent() -> Optional[AgentInfo]:
    """发现本地可用的智能体命令行工具（优先 claude，其次 codex）。"""
    # 1. 优先从 PATH 中检索 claude
    claude_path = shutil.which("claude")
    if claude_path:
        return AgentInfo(
            name="claude",
            path=str(Path(claude_path).resolve()),
            version=_get_version(claude_path),
        )

    # 2. 检索常见用户和系统路径中的 claude
    home = Path.home()
    claude_candidates = [
        *glob.glob(str(home / ".nvm/versions/node/*/bin/claude")),
        str(home / ".npm-global/bin/claude"),
        str(home / ".local/bin/claude"),
        "/usr/local/bin/claude",
        "/opt/homebrew/bin/claude",
    ]
    for p in claude_candidates:
        cand = Path(p)
        if cand.is_file() and os.access(cand, os.X_OK):
            return AgentInfo(
                name="claude",
                path=str(cand.resolve()),
                version=_get_version(str(cand)),
            )

    # 3. 若无 claude，检索 PATH 中的 codex
    codex_path = shutil.which("codex")
    if codex_path:
        return AgentInfo(
            name="codex",
            path=str(Path(codex_path).resolve()),
            version=_get_version(codex_path),
        )

    # 4. 检索常见路径中的 codex
    codex_candidates = [
        *glob.glob(str(home / ".nvm/versions/node/*/bin/codex")),
        str(home / ".npm-global/bin/codex"),
        str(home / ".local/bin/codex"),
        "/usr/local/bin/codex",
        "/opt/homebrew/bin/codex",
    ]
    for p in codex_candidates:
        cand = Path(p)
        if cand.is_file() and os.access(cand, os.X_OK):
            return AgentInfo(
                name="codex",
                path=str(cand.resolve()),
                version=_get_version(str(cand)),
            )

    return None


def find_agent_cli() -> Optional[AgentInfo]:
    """发现本地可用的智能体命令行工具（find_agent 的标准化别名）。"""
    return find_agent()


async def bootstrap_claude(log_path: Optional[Path] = None) -> Optional[AgentInfo]:
    """若本地既无 claude 又无 codex，尝试通过 npm 在后台自举安装 @anthropic-ai/claude-code。"""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = log_path or (LOGS_DIR / "agent_install.log")

    # 寻找 npm / node
    npm_path = shutil.which("npm")
    if not npm_path:
        home = Path.home()
        candidates = [
            *glob.glob(str(home / ".nvm/versions/node/*/bin/npm")),
            str(home / ".npm-global/bin/npm"),
            "/usr/local/bin/npm",
            "/opt/homebrew/bin/npm",
        ]
        for c in candidates:
            cand = Path(c)
            if cand.is_file() and os.access(cand, os.X_OK):
                npm_path = str(cand)
                break

    if not npm_path:
        msg = "[自举安装] 未检测到 npm 环境，无法自动安装 @anthropic-ai/claude-code，请手动安装 node/npm。"
        logger.warning(msg)
        with open(out_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
        return None

    logger.info("开始通过 %s 安装 @anthropic-ai/claude-code...", npm_path)
    with open(out_file, "a", encoding="utf-8") as f:
        f.write(f"\n--- 开始安装 @anthropic-ai/claude-code ({npm_path}) ---\n")

    try:
        with open(out_file, "a", encoding="utf-8") as f:
            proc = await asyncio.create_subprocess_exec(
                npm_path,
                "install",
                "-g",
                "@anthropic-ai/claude-code",
                stdout=f,
                stderr=asyncio.subprocess.STDOUT,
            )
            ret = await proc.wait()

        if ret == 0:
            logger.info("Claude Code 安装成功，重新检测路径...")
            return find_agent()
        else:
            logger.warning("通过 npm 安装 Claude Code 失败，退出码: %d，详见 %s", ret, out_file)
            return None
    except Exception as e:
        logger.error("自举安装执行异常: %s", e)
        with open(out_file, "a", encoding="utf-8") as f:
            f.write(f"自举安装异常: {e}\n")
        return None


async def ensure_agent(auto_install: bool = True) -> Optional[AgentInfo]:
    """确保获取到一个可用的智能体 CLI。若不存在且 auto_install=True 则自举安装。"""
    agent = find_agent()
    if agent:
        return agent

    if auto_install:
        return await bootstrap_claude()
    return None


async def run_agent(
    agent_info: AgentInfo,
    prompt: str,
    model_alias: str,
    port: int,
    log_file: Path,
    env_overrides: Optional[dict[str, str]] = None,
    cwd: Optional[Path] = None,
) -> int:
    """在后台执行智能体进程，重定向输出至日志文件，返回进程退出码。"""
    log_file.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{port}"
    env["ANTHROPIC_API_KEY"] = "self-healing-agent-key"
    env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
    if env_overrides:
        env.update(env_overrides)

    work_dir = str(cwd or PROJECT_ROOT)

    if agent_info.name == "claude":
        cmd = [agent_info.path, "-p", prompt, "--model", model_alias]
    else:
        # codex CLI 或其他备用智能体
        cmd = [agent_info.path, "exec", prompt]

    logger.info("调起自愈智能体: %s (model=%s, log=%s)", agent_info.name, model_alias, log_file)
    with open(log_file, "a", encoding="utf-8") as out:
        out.write(f"=== 自愈任务启动 [{agent_info.name}] ===\n")
        out.write(f"命令: {' '.join(cmd)}\n")
        out.write(f"工作目录: {work_dir}\n")
        out.write(f"模型后端: {model_alias}\n")
        out.write(f"网关基址: {env['ANTHROPIC_BASE_URL']}\n\n")
        out.flush()

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=out,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
            cwd=work_dir,
        )
        return await proc.wait()
