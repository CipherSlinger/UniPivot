"""健康探测与定时审计器 (healing/monitor.py)。

维护所有 Provider 的健康状态：
- status: "healthy" | "degraded" | "recovering" | "offline"
- last_check: float
- last_error: Optional[str]
- consecutive_failures: int

探测逻辑：
- 轻量心跳探针：向各配置好的 Provider 发送最小化试探（如 "hi"），捕获 401（Token 失效）、429（限流/WAF 人机）、5xx 或未知异常。
- 启动时后台初始探测（不阻塞主服务启动）。
- 定时周期探测（默认 300 秒，通过 HEAL_CHECK_INTERVAL 配置）。
- 发现失效时触发自愈引擎。
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import traceback
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx

from providers.base import ProviderError

logger = logging.getLogger("healing.monitor")


@dataclass
class ProviderHealth:
    status: str = "healthy"  # "healthy" | "degraded" | "recovering" | "offline"
    last_check: float = 0.0
    last_error: Optional[str] = None
    consecutive_failures: int = 0
    configured: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HealthMonitor:
    """管理并维护各 Provider 的在线状态与定时探活审计。"""

    def __init__(
        self,
        check_interval: Optional[float] = None,
        max_consecutive_failures: int = 2,
    ):
        # 默认 300 秒探测一次，支持环境变量覆盖
        env_interval = os.getenv("HEAL_CHECK_INTERVAL")
        if check_interval is not None:
            self.check_interval = check_interval
        elif env_interval and env_interval.isdigit():
            self.check_interval = float(env_interval)
        else:
            self.check_interval = 300.0

        self.max_consecutive_failures = max_consecutive_failures
        self.check_count: int = 0
        self.last_check_timestamp: float = 0.0
        self._states: Dict[str, ProviderHealth] = {
            "qwen": ProviderHealth(),
            "deepseek": ProviderHealth(),
            "doubao": ProviderHealth(),
            "kimi": ProviderHealth(),
            "glm": ProviderHealth(),
        }
        self._provider_factories: Dict[str, Callable[[], Any]] = {}
        self._provider_resolvers: Dict[str, Callable[[], Any]] = {}
        self._healer_trigger: Optional[Callable[[str, Optional[str]], Any]] = None
        self._loop_task: Optional[asyncio.Task] = None
        self._running = False
        self._check_locks: Dict[str, asyncio.Lock] = {
            name: asyncio.Lock() for name in self._states
        }

    def register_provider(
        self,
        name: str,
        resolver: Callable[[], Any],
        factory: Callable[[], Any],
    ) -> None:
        """注册 Provider 构造工厂与凭据解析器。"""
        self._provider_resolvers[name] = resolver
        self._provider_factories[name] = factory
        if name not in self._states:
            self._states[name] = ProviderHealth()
            self._check_locks[name] = asyncio.Lock()

    def set_healer_trigger(self, trigger: Callable[[str, Optional[str]], Any]) -> None:
        """设置当探测失效时触发自愈的回调函数。"""
        self._healer_trigger = trigger

    def get_status(self, name: str) -> Optional[Dict[str, Any]]:
        h = self._states.get(name)
        return h.to_dict() if h else None

    def get_all_statuses(self) -> Dict[str, Dict[str, Any]]:
        # 刷新 configured 标志
        for name, resolver in self._provider_resolvers.items():
            s = resolver()
            configured = bool(s and s.token)
            self._states[name].configured = configured
            # 未配置的直接置为 offline（若之前不是 recovering）
            if not configured and self._states[name].status not in ("recovering", "offline"):
                self._states[name].status = "offline"
                self._states[name].last_error = "未配置令牌"

        return {k: v.to_dict() for k, v in self._states.items()}

    def mark_recovering(self, name: str) -> None:
        if name in self._states:
            self._states[name].status = "recovering"

    def mark_offline(self, name: str, error: Optional[str] = None) -> None:
        if name in self._states:
            self._states[name].status = "offline"
            if error:
                self._states[name].last_error = error

    def mark_healthy(self, name: str) -> None:
        if name in self._states:
            self._states[name].status = "healthy"
            self._states[name].consecutive_failures = 0
            self._states[name].last_error = None

    async def probe_provider(self, name: str) -> Tuple[bool, Optional[str]]:
        """轻量心跳探针实现：调用一次最小化 chat 并验证流式产出。"""
        resolver = self._provider_resolvers.get(name)
        factory = self._provider_factories.get(name)
        if not resolver or not factory:
            return False, f"未注册的 Provider: {name}"

        sess = resolver()
        if not sess or not sess.token:
            return False, f"{name} 未配置有效令牌"

        try:
            provider = factory()
            # 最小化试探参数
            probe_messages = [{"role": "user", "content": "hi"}]
            default_models = {
                "qwen": "Qwen3.6-Flash",
                "deepseek": "deepseek-chat",
                "doubao": "doubao-lite",
                "kimi": "kimi",
                "glm": "glm-4-flash",
            }
            model_to_use = default_models.get(name, name)

            # 超时保护（探针专用超时 15 秒）
            probe_stream = provider.chat(
                messages=probe_messages,
                model=model_to_use,
                conversation_id=None,
                max_tokens=2,
            )

            # 接收到任意数据包或正常结束即视为通道健康
            async def _consume_first_chunk():
                async for _delta, _meta in probe_stream:
                    return True
                return True

            await asyncio.wait_for(_consume_first_chunk(), timeout=15.0)
            return True, None
        except asyncio.TimeoutError:
            return False, f"探测超时 (15s): 上游无响应"
        except ProviderError as pe:
            return False, f"上游接口异常 [{pe.status} {pe.err_type}]: {pe.message}"
        except Exception as e:
            tb = traceback.format_exc(limit=2)
            return False, f"未知探测异常 [{type(e).__name__}]: {e}\n{tb}"

    async def check_provider(self, name: str) -> Tuple[bool, Optional[str]]:
        """对单个 Provider 执行探测并更新其健康状态机。"""
        if name not in self._check_locks:
            self._check_locks[name] = asyncio.Lock()

        async with self._check_locks[name]:
            # 如果正在处于自愈中，不强行覆盖为 degraded/offline
            current_state = self._states.get(name)
            if current_state and current_state.status == "recovering":
                logger.info("Provider %s 正在自愈修复中，跳过普通周期探测", name)
                return True, None

            self.check_count += 1
            self.last_check_timestamp = time.time()

            resolver = self._provider_resolvers.get(name)
            if resolver:
                s = resolver()
                configured = bool(s and s.token)
                self._states[name].configured = configured
                if not configured:
                    self._states[name].status = "offline"
                    self._states[name].last_error = "未配置令牌"
                    self._states[name].last_check = time.time()
                    return False, "未配置令牌"

            is_healthy, err = await self.probe_provider(name)
            self._states[name].last_check = time.time()

            if is_healthy:
                self._states[name].status = "healthy"
                self._states[name].consecutive_failures = 0
                self._states[name].last_error = None
                return True, None

            # 探测失败
            self._states[name].consecutive_failures += 1
            self._states[name].last_error = err
            logger.warning(
                "Provider %s 探针异常 (连续失败 %d 次): %s",
                name,
                self._states[name].consecutive_failures,
                err,
            )

            # 状态机跃迁：
            # 1 次失败: degraded
            # >= max_consecutive_failures 次: offline / 触发自愈
            if self._states[name].consecutive_failures >= self.max_consecutive_failures:
                self._states[name].status = "offline"
                # 触发自愈回调
                if self._healer_trigger:
                    try:
                        logger.info("达到失败阈值，向自愈引擎派发任务: %s", name)
                        res = self._healer_trigger(name, err)
                        if asyncio.iscoroutine(res):
                            asyncio.create_task(res)
                    except Exception as he:
                        logger.error("触发自愈异常: %s", he)
            else:
                self._states[name].status = "degraded"

            return False, err

    async def check_all(self) -> Dict[str, Tuple[bool, Optional[str]]]:
        """并发探测所有已注册的 Provider。"""
        results = {}
        tasks = {name: self.check_provider(name) for name in self._provider_resolvers}
        for name, task in tasks.items():
            results[name] = await task
        return results

    async def _audit_loop(self) -> None:
        """后台定时审计循环。"""
        logger.info("健康审计循环已启动，探测间隔: %.1f 秒", self.check_interval)
        try:
            while self._running:
                try:
                    await self.check_all()
                except Exception as e:
                    logger.error("周期审计发生未捕获异常: %s", e)
                await asyncio.sleep(self.check_interval)
        except asyncio.CancelledError:
            logger.info("健康审计循环已取消")

    def start(self, initial_delay: float = 2.0) -> None:
        """启动后台定时探活任务。程序启动时不会阻塞主线程/主事件循环。"""
        if self._running:
            return
        self._running = True

        async def _delayed_start():
            if initial_delay > 0:
                await asyncio.sleep(initial_delay)
            # 初始探测一次
            try:
                await self.check_all()
            except Exception as e:
                logger.error("初始探测异常: %s", e)
            # 进入定时循环
            self._loop_task = asyncio.create_task(self._audit_loop())

        asyncio.create_task(_delayed_start())

    def stop(self) -> None:
        """停止后台审计循环。"""
        self._running = False
        if self._loop_task and not self._loop_task.done():
            self._loop_task.cancel()
