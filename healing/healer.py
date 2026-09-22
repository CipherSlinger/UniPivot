"""自愈决策器与 Prompt 提示词工程 (healing/healer.py)。

当模型探测器检测到 Provider 失效时：
1. 挑选当前 healthy 的最强模型作为修复大脑；
2. 组装故障上下文与自愈 Prompt（定位代码、协议变动特征、单测目标）；
3. 调起本地 Claude/Codex CLI 执行代码修复；
4. 修复后自动复检并恢复健康。
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from .agent_runner import AgentInfo, ensure_agent, run_agent

if TYPE_CHECKING:
    from .monitor import HealthMonitor

logger = logging.getLogger("healing.healer")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"

# 推荐用于驱动自愈智能体的健康模型优先级
PREFERRED_HEALER_MODELS = {
    "qwen": "qwen3.7-max",
    "deepseek": "deepseek-chat",
    "doubao": "doubao-pro",
    "kimi": "kimi-explore",
    "glm": "glm-4-plus",
}


def pick_healer_model(monitor: HealthMonitor, failing_provider: str) -> Optional[str]:
    """从当前除 failing_provider 之外 healthy 的 Provider 中，选出最强的模型别名。"""
    # 优先顺序：qwen(qwen3.7-max) -> deepseek(deepseek-chat) -> doubao(doubao-pro) -> kimi(kimi-explore) -> glm(glm-4-plus)
    priority = ["qwen", "deepseek", "doubao", "kimi", "glm"]

    for cand in priority:
        if cand == failing_provider:
            continue
        info = monitor.get_status(cand)
        if info and info.get("status") == "healthy":
            return PREFERRED_HEALER_MODELS[cand]

    # 如果没有完全健康的，但有处于 degraded 状态的（只偶尔超时但未彻底 offline），也可以做备选
    for cand in priority:
        if cand == failing_provider:
            continue
        info = monitor.get_status(cand)
        if info and info.get("status") == "degraded":
            return PREFERRED_HEALER_MODELS[cand]

    return None


def build_healing_prompt(
    provider_name: str,
    last_error: Optional[str] = None,
    traceback_info: Optional[str] = None,
) -> str:
    """构建用于驱动 Claude / Codex CLI 的自愈任务提示词。"""
    target_provider_file = f"providers/{provider_name}/provider.py"
    target_test_file = "tests/test_providers.py"

    err_summary = last_error or "模型无法返回有效响应或抛出未知异常"
    tb_summary = traceback_info or "无完整堆栈捕获"

    prompt = f"""你是一个运行在本地网关后端的自愈与修复专家（Self-Healing Agent）。

【故障报警】
网关后台健康探测器检测到上游提供方 [{provider_name}] 已失效，可能由于官方网页端协议更新、Header 校验变动、SSE 帧格式调整或 WAF 响应码更新引起。

【故障详情】
- 异常信息: {err_summary}
- 堆栈/上下文摘要: {tb_summary}

【核心任务目标】
1. 深入分析 `{target_provider_file}` 中的网络请求逻辑、Headers、Cookie 处理及 SSE 流解析代码。
2. 结合官方网页端的常见改动（如请求 Header 补全、Referer/Origin、SSE 数据包结构、状态码如 401/429/5xx 校验），排查代码中的失效点并进行修复。
3. 运行本地单测 `.venv/bin/python {target_test_file}`，确保修复后的代码通过全部测试。
4. 如果单测或实现中需要同步调整针对该 Provider 的 Mock / 解析逻辑，请一并规范更新。
5. 必须保持对外接口兼容性（遵循 ProviderError 规范与 (delta, meta) 生成器契约），严禁破坏网关原有稳定性。

请立即开始检查代码并进行修复，直至单测 `.venv/bin/python {target_test_file}` 完全通过！"""
    return prompt.strip()


class HealingEngine:
    """自愈引擎控制器。负责协调健康监控、提示词生成、智能体调起和状态复原。"""

    def __init__(self, monitor: HealthMonitor, port: int = 8000):
        self.monitor = monitor
        self.port = port
        self._healing_locks: Dict[str, asyncio.Lock] = {}
        self.current_tasks: Dict[str, Dict[str, Any]] = {}

    def is_healing(self, provider_name: str) -> bool:
        return provider_name in self.current_tasks

    def get_healing_status(self) -> Dict[str, Any]:
        """返回当前自愈系统的总体状态和正在运行的任务。"""
        return {
            "active_count": len(self.current_tasks),
            "running_tasks": {
                name: {
                    "started_at": task["started_at"],
                    "healer_model": task["healer_model"],
                    "agent": task["agent"],
                    "log_file": str(task["log_file"]),
                }
                for name, task in self.current_tasks.items()
            },
        }

    async def trigger_healing(
        self,
        provider_name: str,
        reason: Optional[str] = None,
        force: bool = False,
    ) -> bool:
        """异步触发某个 Provider 的自愈流程。

        流程：
        1. 检查是否正在自愈中；
        2. 选取健康可用的模型驱动自愈；
        3. 检测本地智能体 CLI 环境（claude / codex），必要时自举安装；
        4. 设置 provider 状态为 "recovering"；
        5. 在后台调起智能体执行修复任务；
        6. 任务完成后复测，若恢复健康则置为 "healthy"，否则置为 "offline" 并等待退避。
        """
        if provider_name not in self._healing_locks:
            self._healing_locks[provider_name] = asyncio.Lock()

        lock = self._healing_locks[provider_name]
        if lock.locked() and not force:
            logger.info("Provider %s 的自愈任务已在运行中，跳过重复触发", provider_name)
            return False

        async def _healing_worker():
            async with lock:
                # 1. 挑选用于驱动修复的健康模型
                healer_model = pick_healer_model(self.monitor, provider_name)
                if not healer_model:
                    logger.warning(
                        "无法为 %s 启动自愈：没有其他处于 healthy 状态的模型可用作为修复引擎",
                        provider_name,
                    )
                    self.monitor.mark_offline(
                        provider_name,
                        error=f"自愈失败: 缺少健康的备选模型驱动修复 (上一次错误: {reason})",
                    )
                    return False

                # 2. 发���或安装智能体环境
                agent_info = await ensure_agent(auto_install=True)
                if not agent_info:
                    logger.error("无法启动自愈：本地未找到 claude 或 codex，且自动安装未成功")
                    self.monitor.mark_offline(
                        provider_name,
                        error=f"自愈失败: 未找到可用的智能体环境 (上一次错误: {reason})",
                    )
                    return False

                # 3. 标记状态为 recovering
                prev_status = self.monitor.get_status(provider_name)
                last_err = reason or (prev_status.get("last_error") if prev_status else None)
                self.monitor.mark_recovering(provider_name)

                ts = int(time.time())
                log_file = LOGS_DIR / f"healing_{provider_name}_{ts}.log"

                self.current_tasks[provider_name] = {
                    "started_at": ts,
                    "healer_model": healer_model,
                    "agent": agent_info.name,
                    "log_file": log_file,
                }

                prompt = build_healing_prompt(
                    provider_name=provider_name,
                    last_error=last_err,
                )

                logger.info(
                    "启动自愈子进程: provider=%s, agent=%s, healer_model=%s, log=%s",
                    provider_name,
                    agent_info.name,
                    healer_model,
                    log_file,
                )

                try:
                    exit_code = await run_agent(
                        agent_info=agent_info,
                        prompt=prompt,
                        model_alias=healer_model,
                        port=self.port,
                        log_file=log_file,
                    )
                    logger.info("自愈任务子进程执行结束: exit_code=%d", exit_code)
                except Exception as exc:
                    logger.error("自愈任务运行异常: %s", exc)
                    exit_code = -1
                finally:
                    self.current_tasks.pop(provider_name, None)

                # 4. 执行复测与状态更新
                logger.info("自愈任务结束，正在对 %s 执行健康复检...", provider_name)
                # 先清除 recovering 状态，允许 check_provider 正常探测更新
                if self.monitor._states.get(provider_name):
                    self.monitor._states[provider_name].status = "degraded"

                is_healthy, err = await self.monitor.check_provider(provider_name)
                if is_healthy:
                    self.monitor.mark_healthy(provider_name)
                    logger.info("Provider %s 成功通过自愈恢复健康！", provider_name)
                    return True
                else:
                    logger.warning(
                        "Provider %s 修复后复检仍未通过: %s，标记为 offline",
                        provider_name,
                        err,
                    )
                    self.monitor.mark_offline(
                        provider_name,
                        error=f"自愈执行完成但复检未通过: {err}",
                    )
                    return False

        # 以后台任务启动，不阻塞调用方
        asyncio.create_task(_healing_worker())
        return True
