"""健康探测与定时审计器 (healing/monitor.py)。

维护所有 Provider 的健康状态：
- status: "healthy" | "degraded" | "recovering" | "offline" | "cooling"
- last_check: float
- last_error: Optional[str]
- consecutive_failures: int

探测逻辑：
- 轻量心跳探针：向各配置好的 Provider 发送最小化试探（如 "hi"），捕获 401（Token 失效）、429（限流/WAF 人机）、5xx 或未知异常。
- 启动时后台初始探测（不阻塞主服务启动）。
- 定时周期探测（默认 300 秒，通过 HEAL_CHECK_INTERVAL 配置）。
- 发现失效时触发自愈引擎。
- 风控冷却门控 [P0] OPT-012：若节点正处于风控避让冷却期（CooldownTracker），
  则跳过真实网络探针——否则每 300 秒一次的探针会持续触碰已被上游风控封禁的
  节点，人为延长封禁窗口。冷却期内既不探测也不计失败、不触发自愈。
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
from providers.stream_completion import observe

logger = logging.getLogger("healing.monitor")

# [TOUCH-CAUSE] 上游故障成因的探针文案。三种成因都**不是**风控拦截：风控冷却要等窗口
# 或等人工完成验证，上游故障要等上游恢复——把一次 502 说成「规避上游封禁」，运维就会
# 去排查一个从未发生的拦截（与 [DAY-ZERO] 的 `unconfigured` 完全同形）。
_UPSTREAM_FAULT_LABELS = {
    "timeout": "上游超时",
    "truncated": "上游流被截断",
    "upstream_error": "上游故障（5xx）",
}


@dataclass
class ProviderHealth:
    status: str = "healthy"  # "healthy" | "degraded" | "recovering" | "offline" | "cooling"
    last_check: float = 0.0
    last_error: Optional[str] = None
    consecutive_failures: int = 0
    configured: bool = False
    cooldown_remaining: float = 0.0  # 风控避让剩余冷却秒数 (0 表示不在避让期)


    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _schedule_probe_conversation_cleanup(
    provider: Any, provider_key: str, conversation_id: str
) -> None:
    """把探针创建的临时会话交给 ConversationGC（失败也不能影响探测结果）。

    [PROBE-GC] 探针是**网关自己**制造的流量，没有客户端会话可归属，所以它建的
    上游会话必须由网关自己回收。走 GC 而不是直接 `delete_conversation`，是为了
    沿用既有的冷却门（冷却中的节点不该被删除请求再碰一次）与跨重启持久化。
    """
    try:
        from providers.conversation_gc import conversation_gc

        conversation_gc.schedule(provider, provider_key, conversation_id)
    except Exception:  # noqa: BLE001 - 清理失败绝不能让探针报「上游不健康」
        logger.debug("探针会话 %s 的回收调度失败", conversation_id, exc_info=True)


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
        # [OPT-012] 风控冷却门控：返回节点剩余冷却秒数的回调 (>0 表示处于避让期)
        self._cooldown_probe_gate: Optional[Callable[[str], float]] = None
        # [OPT-034] 运维禁用与后台触点避让的注入判据（未注入时退化为环境变量/冷却判读）
        self._disabled_gate: Optional[Callable[[str], bool]] = None
        self._touch_avoidance_gate: Optional[Callable[[str], str]] = None
        self._probe_skip_stats: Dict[str, Any] = {"total": 0, "per_provider": {}}
        # [OPT-018] 自愈大脑选举的冷却避让统计
        self._healer_skip_stats: Dict[str, Any] = {"total": 0, "per_provider": {}}
        self._check_locks: Dict[str, asyncio.Lock] = {
            name: asyncio.Lock() for name in self._states
        }

    def set_disabled_gate(self, gate: Callable[[str], bool]) -> None:
        """[OPT-034] 注入「节点是否被运维禁用」的唯一判读器。

        此前本类自行现读 `DISABLED_PROVIDERS` 环境变量（同一份配置的**第三**份解读：
        CooldownTracker 持有一份并集判读者，本类又现读一次），于是运行时改配置时两者
        给出不同答案。现在改由外部注入的那个判读者回答。
        """
        self._disabled_gate = gate

    def is_disabled(self, name: str) -> bool:
        """检查提供方是否被运维禁用。

        [OPT-034] 判读委托给注入的判据（`CooldownTracker.is_disabled`，程序集合 ∪ 环境
        变量现读）。未注入时退化为现读环境变量，使本类在独立构造时（单元测试、无网关
        上下文的调用者）仍可用。
        """
        gate = self._disabled_gate
        if gate is not None:
            try:
                return bool(gate(name))
            except Exception:  # noqa: BLE001 - 判据查询失败回退到环境变量，绝不误放行
                pass
        disabled_str = os.getenv("DISABLED_PROVIDERS", "")
        if not disabled_str:
            return False
        disabled_set = {p.strip().lower() for p in disabled_str.split(",") if p.strip()}
        return (name or "").lower() in disabled_set

    def set_touch_avoidance_gate(self, gate: Callable[[str], str]) -> None:
        """[OPT-034] 注入后台触点避让判据（`CooldownTracker.touch_avoidance_reason`）。

        自愈选举（`pick_healer_model`）最终会经本网关发出一次**真实上游补全**，因此它也
        属于后台自维护触点，与探针/刷新/会话回收共用同一判据。
        """
        self._touch_avoidance_gate = gate

    def touch_avoidance_reason(self, name: str) -> str:
        """[OPT-034] 查询节点是否处于后台避让期，返回成因（"" 表示可触碰）。

        判定顺序即优先级：运维禁用 → （冷却总开关）→ 注入判据（租约 ∪ 冷却）。

        「运维禁用」**不**受冷却总开关 `GATEWAY_DISABLE_COOLDOWN_GATE` 影响：该开关的
        语义是「回退为无冷却感知」，而禁用是部署声明（本类原有的 `is_disabled` 分支也一
        直不受该开关影响），把它一并交出去等于让一个开关悄悄放开一个被声明出局的节点。
        租约与冷却则仍受总开关约束，关闭时退化为「冷却即可避让」之外不再扩大阻断范围，
        与 OPT-018 的既有行为逐值一致。

        未注入判据时退化为「冷却即避让」（成因 `cooldown`），保证独立构造本类时语义不变。
        """
        if self.is_disabled(name):
            return "disabled"
        if self._cooldown_gate_disabled():
            return ""
        gate = self._touch_avoidance_gate
        if gate is not None:
            try:
                return str(gate(name) or "")
            except Exception:  # noqa: BLE001 - 判据查询失败按「不避让」处理，不影响可用性
                return ""
        return "cooldown" if self._probe_cooldown_remaining(name) > 0 else ""

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

    def set_cooldown_probe_gate(self, gate: Callable[[str], float]) -> None:
        """[OPT-012] 注册风控冷却查询回调：入参 provider 名，返回剩余冷却秒数。

        返回 > 0 时表示该节点正处风控避让期，周期探针必须跳过真实网络调用，
        以免持续触碰已被上游封禁的节点而延长封禁窗口。
        """
        self._cooldown_probe_gate = gate

    def _cooldown_gate_disabled(self) -> bool:
        """[OPT-018] 冷却避让总开关：探针门控、状态叠加与自愈选举共用同一开关。

        `GATEWAY_DISABLE_COOLDOWN_GATE` 为语义化新名，`GATEWAY_DISABLE_PROBE_COOLDOWN_GATE`
        为历史名（向后兼容），任一为真即整体回退为「无冷却感知」。
        """
        for env in ("GATEWAY_DISABLE_COOLDOWN_GATE", "GATEWAY_DISABLE_PROBE_COOLDOWN_GATE"):
            if os.getenv(env, "").strip().lower() in ("1", "true", "yes"):
                return True
        return False

    def _probe_cooldown_remaining(self, name: str) -> float:
        """查询节点剩余风控冷却秒数（未注册回调或已关闭门控时返回 0）。"""
        if self._cooldown_gate_disabled():
            return 0.0
        gate = self._cooldown_probe_gate
        if gate is None:
            return 0.0
        try:
            return max(0.0, float(gate(name)))
        except Exception:  # noqa: BLE001 - 门控查询失败绝不阻断探测流程
            return 0.0

    def get_cooldown_remaining(self, name: str) -> float:
        """[OPT-018] 公开接口：查询节点剩余风控冷却秒数（>0 表示处于避让期）。

        与探针门控共享同一 CooldownTracker 数据源与同一开关，供自愈大脑选举、
        状态叠加等读取侧消费方使用，避免各自实现冷却判断而产生分歧。
        """
        return self._probe_cooldown_remaining(name)

    def record_healer_skip(self, name: str) -> None:
        """[OPT-018] 记录一次因风控冷却而被排除出候选的自愈大脑选举。"""
        self._healer_skip_stats["total"] = int(self._healer_skip_stats.get("total", 0)) + 1
        per = self._healer_skip_stats.setdefault("per_provider", {})
        per[name] = per.get(name, 0) + 1

    def get_healer_skip_stats(self) -> Dict[str, Any]:
        """[OPT-018] 返回因冷却被排除出候选的选举次数统计。"""
        return {
            "total": int(self._healer_skip_stats.get("total", 0)),
            "per_provider": dict(self._healer_skip_stats.get("per_provider", {})),
        }

    def get_probe_skip_stats(self) -> Dict[str, Any]:
        """[OPT-012] 返回被冷却门控跳过的探针统计。"""
        return {
            "total": int(self._probe_skip_stats.get("total", 0)),
            "per_provider": dict(self._probe_skip_stats.get("per_provider", {})),
        }

    def get_status(self, name: str) -> Optional[Dict[str, Any]]:
        if self.is_disabled(name):
            if name not in self._states:
                self._states[name] = ProviderHealth()
            self._states[name].status = "disabled"
            self._states[name].last_error = "节点已被禁用 (DISABLED_PROVIDERS)"
        elif name in self._states and self._states[name].status == "disabled":
            # 动态解禁恢复
            self._states[name].status = "healthy"
            if self._states[name].last_error == "节点已被禁用 (DISABLED_PROVIDERS)":
                self._states[name].last_error = None
        h = self._states.get(name)
        if h is None:
            return None
        info = h.to_dict()
        # [OPT-018] 冷却状态在读取时叠加（不落盘、不粘滞）：
        # ProviderHealth 默认 status="healthy"，进程刚重启或节点尚未被探针覆盖时，
        # 处于风控冷却期的节点会被所有消费方（自愈大脑选举、状态面板等）
        # 误判为健康并加以触碰。此处以实时冷却查询覆盖之。
        if info.get("status") not in ("disabled", "offline"):
            remaining = self._probe_cooldown_remaining(name)
            if remaining > 0:
                info["status"] = "cooling"
                info["cooldown_remaining"] = round(remaining, 1)
        return info

    def get_all_statuses(self) -> Dict[str, Dict[str, Any]]:
        # 刷新 disabled 状态与动态解禁
        for name in list(self._states.keys()):
            if self.is_disabled(name):
                self._states[name].status = "disabled"
                self._states[name].last_error = "节点已被禁用 (DISABLED_PROVIDERS)"
            elif self._states[name].status == "disabled":
                self._states[name].status = "healthy"
                if self._states[name].last_error == "节点已被禁用 (DISABLED_PROVIDERS)":
                    self._states[name].last_error = None

        # 刷新 configured 标志
        for name, resolver in self._provider_resolvers.items():
            if self.is_disabled(name):
                continue
            s = resolver()
            configured = bool(s and s.token)
            self._states[name].configured = configured
            # 未配置的直接置为 offline（若之前不是 recovering）
            if not configured and self._states[name].status not in ("recovering", "offline"):
                self._states[name].status = "offline"
                self._states[name].last_error = "未配置令牌"

        # [OPT-018] 与 get_status 一致，读取时叠加实时冷却状态
        result: Dict[str, Dict[str, Any]] = {}
        for k, v in self._states.items():
            info = v.to_dict()
            if info.get("status") not in ("disabled", "offline"):
                remaining = self._probe_cooldown_remaining(k)
                if remaining > 0:
                    info["status"] = "cooling"
                    info["cooldown_remaining"] = round(remaining, 1)
            result[k] = info
        return result

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

            # [PROBE-GC] 探针以 conversation_id=None 调用，上游会建一个临时会话。
            # 之前这里只读首帧就 return，把 meta 连同 conversation_id 一起丢掉，
            # 于是 300s × 5 节点 = 每天 1440 个「hi」对话永久留在用户侧栏——
            # 既污染账面，又给出「每 60 秒一个 hi」的自动化时序特征。
            #
            # 同时把「读到首帧就算健康」改成「消费到流真正结束」：三家适配器
            # (deepseek/doubao/glm) 的 conversation_id 在**末位**元数据帧里，
            # 只读首帧根本拿不到它；而 has_marker 的那两家顺带给出确切的完整性
            # 判据（`observe`，缺键 = 未知，不会误伤无标记协议）。
            probe_conversation_id: Optional[str] = None
            probe_complete: Optional[bool] = None
            probe_saw_frame = False

            async def _consume_stream():
                nonlocal probe_conversation_id, probe_complete, probe_saw_frame
                async for _delta, meta in probe_stream:
                    probe_saw_frame = True
                    if isinstance(meta, dict):
                        cid = meta.get("conversation_id")
                        if cid:
                            probe_conversation_id = str(cid)
                        probe_complete = observe(meta, probe_complete)
                return True

            try:
                await asyncio.wait_for(_consume_stream(), timeout=15.0)
            except asyncio.TimeoutError:
                # 五个上游都**不支持** max_tokens，提问「hi」可能换来一段长回答，
                # 所以这里的超时不再等同于「上游没响应」：已经收到帧就说明通道
                # 是通的，只是这一轮回答还没结束——判不健康会制造假故障。
                if not probe_saw_frame:
                    raise
                logger.debug("探针超时但已收到数据帧，按健康处理: %s", name)
                if probe_conversation_id:
                    _schedule_probe_conversation_cleanup(
                        provider, name, probe_conversation_id
                    )
                return True, None

            if probe_conversation_id:
                _schedule_probe_conversation_cleanup(provider, name, probe_conversation_id)
            # 只有协议**真有**完成标记且没看到时才判截断；None = 未知，不惩罚。
            if probe_complete is False:
                return False, "探测流未收到完成标记（上游截断）"
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
            # [OPT-034] 一次性向唯一判据求值（运维禁用 ∪ 租约 ∪ 冷却），而不是分别读两次
            avoidance = self.touch_avoidance_reason(name)
            if avoidance == "disabled":
                logger.info("Provider %s 已被禁用 (DISABLED_PROVIDERS)，跳过健康探测", name)
                if name not in self._states:
                    self._states[name] = ProviderHealth()
                self._states[name].status = "disabled"
                self._states[name].last_error = "节点已被禁用 (DISABLED_PROVIDERS)"
                self._states[name].last_check = time.time()
                return False, "节点已被禁用 (DISABLED_PROVIDERS)"

            if name in self._states and self._states[name].status == "disabled":
                self._states[name].status = "healthy"
                if self._states[name].last_error == "节点已被禁用 (DISABLED_PROVIDERS)":
                    self._states[name].last_error = None

            # 如果正在处于自愈中，不强行覆盖为 degraded/offline
            current_state = self._states.get(name)
            if current_state and current_state.status == "recovering":
                logger.info("Provider %s 正在自愈修复中，跳过普通周期探测", name)
                return True, None

            # [OPT-012] 风控冷却门控：避让冷却期内绝不发起真实探针，
            # 既不触碰已封禁节点（避免延长封禁窗口），也不计入失败或触发自愈。
            # [OPT-034] 判据来自唯一的 touch_avoidance_reason，成因如实记录。
            cooldown_remaining = self._probe_cooldown_remaining(name)
            if cooldown_remaining > 0:
                st = self._states[name]
                st.status = "cooling"
                st.last_check = time.time()
                st.cooldown_remaining = round(cooldown_remaining, 1)
                # [DAY-ZERO] 成因是「本机没有凭证」时不得说成风控：该账号从未被
                # 触碰过，把它报成「避免加剧上游封禁」会让运维去排查一个不存在的封禁。
                # [TOUCH-CAUSE] 上游故障（5xx / 超时 / 截断）同理：这三种冷却与风控
                # 的处置方式完全不同（等上游恢复 vs 等窗口/人工介入），把它们读作
                # 「风控避让」会把运维引向一个从未发生的拦截。只区分「是不是风控」，
                # 不合并风控内部的成因（那三者的处置在别处已经分开）。
                upstream_label = _UPSTREAM_FAULT_LABELS.get(avoidance)
                if avoidance == "unconfigured":
                    st.last_error = (
                        f"本机尚未配置凭证 (剩余 {cooldown_remaining:.0f}s)，已跳过真实探针"
                    )
                elif upstream_label:
                    st.last_error = (
                        f"{upstream_label}避让中 (剩余 {cooldown_remaining:.0f}s)，已跳过真实探针"
                    )
                else:
                    st.last_error = (
                        f"风控避让冷却中 (剩余 {cooldown_remaining:.0f}s)，已跳过真实探针以规避上游封禁"
                    )
                self._probe_skip_stats["total"] += 1
                per = self._probe_skip_stats["per_provider"]
                per[name] = per.get(name, 0) + 1
                if upstream_label:
                    logger.warning(
                        "[探针冷却门控] 节点 [%s] %s避让中 (成因 %s, 剩余 %.0fs)，本轮跳过真实探针"
                        "（上游故障，不是风控拦截）",
                        name,
                        upstream_label,
                        avoidance,
                        cooldown_remaining,
                    )
                else:
                    logger.warning(
                        "[探针冷却门控] 节点 [%s] 处于风控避让冷却期 (成因 %s, 剩余 %.0fs)，本轮跳过真实探针，"
                        "避免加剧上游封禁",
                        name,
                        avoidance or "cooldown",
                        cooldown_remaining,
                    )
                return False, st.last_error

            self.check_count += 1
            self.last_check_timestamp = time.time()
            self._states[name].cooldown_remaining = 0.0

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
