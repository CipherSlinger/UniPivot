"""智能会话粘滞路由与跨模型状态迁移引擎 (Smart Session Affinity & State Handoff Engine)

实现 [P0] OPT-007：
1. 会话粘滞注册表 (Session Affinity Registry):
   - 基于 LRU 与 TTL (默认 30 分钟) 追踪多轮交互的会话归属。
   - 保持后续轮次请求优先分配到已建立对话上下文的后端 Provider 及模型。
2. 跨模型透明状态迁移 (Cross-Model State Handoff):
   - 当原 Provider 触发风控拦截、节点熔断或临时冷却时，检测到粘滞断裂；
   - 自动清洗剥离上游异构 conversation_id（避免将 A 厂私有 ID 传给 B 厂引发报错）；
   - 基于网关全量上下文历史平滑迁移至同环/降级候选节点，维持多轮交互不中断。
3. 遥测与响应头透传:
   - x-session-affinity: bound | handoff | fresh
   - x-session-handoff-from: <from_provider> (仅在发生跨模型迁移时产生)
   - x-session-turns: <turn_count>

[TOOL-LOOP] 客户端的 `conversation_id` 是**线程标识**，不是上游会话标识。
两者原先被当成同一个值：注册表里没有这条线程时，客户端传来的 id 被原样当成
`upstream_conv_id` 交给 provider，而 provider 只要看到非空 `conversation_id`
就走「续聊」分支——只发送最后一条 user 之后的增量内容。首轮请求的增量是空的，
于是上游收到的提示词**只有最后一句用户输入**：注入的系统提示、`## Tools Available`
工具契约、`response_format`、整段历史全部消失。实测同一请求在续聊分支下的提示词是
`'read foo.py'`，在展平分支下是 990 字符。这就是「模型完全不理我给的 tools」的
直接成因，而且只发生在「客户端自己生成了线程 id」的客户端上——症状因此显得时有时无。

现在只有网关**实际观测到**的上游会话 id 才会被回放；未绑定的线程一律以
`upstream_conv_id=None` 开始，由 provider 新建会话。

[SESSION-IDENTITY] 会话 id 命名的是「一次会话」，不是「一条对话线」。
Claude Code 2.1.293 的自动模式在每次工具审批时额外发一个 `/v1/messages`：
非流式、`tools: []`、`max_tokens: 2112`、`stop_sequences: ["</block>"]`、
自带一段 139 702 字符的「安全审查员」系统提示，要求回 `<block>yes|no</block>`，
用户消息里包一段 `<transcript>`——而 `metadata.user_id.session_id` 与父会话
**完全相同**（从真实二进制捕获的三份请求体实测：父会话第 1 轮、分类器、父会话
第 2 轮；系统提示 sha 0270404a12 / 2d052caeb7，47 个工具 vs 0 个）。

注册表原先只按会话 id 认线，于是网关把父会话正在使用的上游对话交给了分类器，
父会话的下一轮又继续使用同一条对话。经本网关 + 脚本化 provider 实测（线上提示词
由真实的 `flatten_messages` / `continuation_content` 构造，不是复刻）：

    parent#1   affinity=fresh  wire conv=None  44 763 字符
    classifier affinity=bound  wire conv='up1' 140 302 字符  <-- 被写进父会话
    parent#2   affinity=bound  wire conv='up1'  44 543 字符  <-- 继续被污染的那条线

修复后同三份真实请求体、同一实测脚本：

    parent#1   affinity=fresh  wire conv=None  44 763 字符
    classifier affinity=fresh  wire conv=None 140 320 字符  <-- 自己的对话线
    parent#2   affinity=bound  wire conv='up1'  44 543 字符  <-- 回到自己的线

（分类器第二轮起以自己的会话续聊；它声明 0 个工具，所以响应里没有
`x-session-tool-contract` 头，父会话有。）

两个后果都不是报错：用户的上游对话被永久写入一段 14 万字符的安全审查内容（此后
每一轮的回答都在它之上生成），而分类器被当成「续聊」——它要评判的 transcript 与
那条对话的历史毫无关系。

客户端的会话 id 无法再细分（线路上没有更细的标识），因此修法是 §53/§75 用过的
**保守方向**：在「这是哪一条对话线」的身份里再加一个事实——本请求声明的工具契约。
在 47 个工具下开启的对话不会被一个声明 0 个工具的请求续上；两条线各自持有自己的
上游会话。加宽身份键只可能拆开对话，绝不会把本该分开的两条合并。

诚实的残留：如果客户端在**同一次对话**里交替更换声明的工具集（第 1 轮 47 个、
第 2 轮 0 个、第 3 轮 47 个），它会交替命中两条线，各自缺少另一条的轮次。目前没有
任何实测客户端这样做——Claude Code 两轮都是 47/47（摘要 035695cc210236e5），
Codex 每个会话稳定（§62/§68）。
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from .tool_streamer import tool_names_of

logger = logging.getLogger("gateway.session_affinity")


def tool_contract_key(tools: Any) -> str:
    """把一次请求声明的工具集摘要成一小段稳定字符串，作为对话线身份的一部分。

    空字符串表示「没有声明任何工具」（`tools` 缺省、`[]`、或全是无效声明）。

    [SESSION-IDENTITY] 复用 `tool_names_of`，因为它已经回答了「客户端声明了哪些
    工具」这个问题的全部三种协议形状（OpenAI 的 `function.name`、Anthropic /
    Responses 的顶层 `name`、custom / freeform 的 `{"type":"custom","name":…}`）。
    在这里另写一遍名字提取，自定义工具与 namespace 容器（§62）就会在「提示词里
    能调用的工具」和「对话线身份」两条路径上得到不同的答案——即本仓库反复出现的
    「一个事实两份定义」。

    只摘要**名字集合**，不摘要 schema：参数微调不应把一段对话劈成两条线，而增删
    工具（分类器 0 个 vs 父会话 47 个）必须。
    """
    names = tool_names_of(tools)
    if not names:
        return ""
    joined = "\n".join(sorted(names))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def _line_key(session_id: str, tool_contract: str, requested_model: str = "") -> str:
    """对话线的注册表键。

    没有工具契约、也没有声明模型时**原样返回会话 id**：所有既有调用点（匿名请求、
    纯聊天客户端、直接调用本模块的测试）因此逐字节保持不变。

    [MODEL-LINE] 两个身份组成都是「加宽」：只会把本该分开的对话分开，绝不会把
    两条合并（§53/§75 的方向规则）。`requested_model` 传的是**负载均衡之前**解析
    出的上游模型名（`resolve_model` 的结果），不是客户端原始别名——两个别名解析到
    同一个上游模型的回答是可互换的，不该劈成两条线；而负载均衡的降级是暂时的，
    更不能因此把一段对话拆开（[M3] 的同一条理由）。
    """
    parts = [session_id]
    if tool_contract:
        parts.append(tool_contract)
    if requested_model:
        parts.append(requested_model)
    return session_id if len(parts) == 1 else "\x00".join(parts)


@dataclass
class AffinityEntry:
    """单条对话线的粘滞元数据。"""
    session_id: str
    provider_key: str
    model_name: str
    upstream_conv_id: Optional[str]
    created_at: float
    last_accessed_at: float
    turn_count: int = 1
    tool_contract: str = ""


@dataclass
class AffinityDecision:
    """会话粘滞路由决策。"""
    status: str  # "bound" | "degraded" (原节点短冷却，粘滞保留) | "handoff" (已跨模型迁移) | "fresh" (全新会话)
    provider_key: str
    wire_model: str
    upstream_conv_id: Optional[str]
    is_handoff: bool = False
    original_provider: Optional[str] = None
    turn_count: int = 1
    # 决策自己携带本条线的工具契约与声明的模型，调用方 `record_turn` 时直接用它——
    # 解析与记录于是不可能对「这是哪条线」产生分歧（`_affinity_record` 的同一理由）。
    tool_contract: str = ""
    requested_model: str = ""


class SessionAffinityManager:
    """智能会话粘滞管理器。"""

    def __init__(self, ttl_seconds: float = 1800.0, max_entries: int = 10000):
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._lock = threading.Lock()
        self._registry: Dict[str, AffinityEntry] = {}
        # 度量计数器
        self._stats = {
            "total_recorded_sessions": 0,
            "affinity_hits": 0,
            "affinity_degraded": 0,
            "affinity_handoffs": 0,
        }

    def record_turn(
        self,
        session_id: Optional[str],
        provider_key: str,
        model_name: str,
        upstream_conv_id: Optional[str] = None,
        tool_contract: str = "",
        requested_model: str = "",
    ) -> None:
        """记录或更新一条对话线的交互轮次与粘滞目标。

        `model_name` 是**真实派发**的模型，`requested_model` 是这条线的身份里
        声明的模型（`resolve_affinity` 的同一个值）。两者可以不相等：负载均衡
        可能把一次请求降级到同能力环的另一个模型，此时粘滞关系记的是实际服务的
        那个模型，而身份仍属于客户端声明的那一个。
        """
        if not session_id:
            return

        key = _line_key(session_id, tool_contract, requested_model)
        now = time.time()
        with self._lock:
            self._clean_expired_locked(now)

            if key in self._registry:
                entry = self._registry[key]
                entry.provider_key = provider_key
                entry.model_name = model_name
                if upstream_conv_id:
                    entry.upstream_conv_id = upstream_conv_id
                entry.last_accessed_at = now
                entry.turn_count += 1
            else:
                if len(self._registry) >= self.max_entries:
                    # 淘汰最久未访问的条目
                    oldest_key = min(self._registry, key=lambda k: self._registry[k].last_accessed_at)
                    self._registry.pop(oldest_key, None)

                self._registry[key] = AffinityEntry(
                    session_id=session_id,
                    provider_key=provider_key,
                    model_name=model_name,
                    # 绝不拿客户端线程 id 冒充上游会话 id：未观测到就保持 None，
                    # 让 provider 走「新建会话 + 展平全量历史」分支（见模块 docstring）。
                    upstream_conv_id=upstream_conv_id,
                    created_at=now,
                    last_accessed_at=now,
                    turn_count=1,
                    tool_contract=tool_contract,
                )
                self._stats["total_recorded_sessions"] += 1

    def resolve_affinity(
        self,
        session_id: Optional[str],
        primary_provider: str,
        primary_model: str,
        is_provider_available: Callable[[str], bool],
        can_retain_degraded: Optional[Callable[[str], bool]] = None,
        tool_contract: str = "",
        requested_model: str = "",
    ) -> AffinityDecision:
        """解析当前对话线的粘滞关系。

        `tool_contract` 是对话线身份的一部分（见模块 docstring 的
        [SESSION-IDENTITY]）：同一次会话里声明的工具集不同的请求是两条线，各自
        持有自己的上游会话。缺省空串，既有调用点逐字节不变。

        `requested_model` 是同一身份的第二部分（[MODEL-LINE]）：客户端这次声明的
        **上游模型名**（`resolve_model` 的结果，负载均衡之前）。不传它，绑定分支
        会返回第一轮记下的 `entry.model_name`——客户端中途换模型（`/model`、或在
        同一会话 id 下发出的后台小模型请求）就被静默忽略，实测声明
        `claude-3-7-sonnet`（→ `Qwen3.7-Max`）时上游收到 `Qwen3.6-Flash`。

        若存在已绑定的可用 Provider，则锁定该 Provider；
        若原 Provider 处于硬阻断状态，则触发跨模型状态迁移并剥离旧 ID。

        [M3] 「可用」与「值得为它放弃整段上下文」是两个判据，此前被混成一个。
        `is_provider_available` 对**任何**软冷却（哪怕 15 秒的 429 退避）都返回
        False，于是每有一次频控退避，进行中的长程会话就被永久改嫁到另一家厂商：
        上游上下文全部重建、提示词缓存全丢、工具调用契约重新学习——为躲开一个
        数十秒的窗口付出几十倍的代价。这里改为**只判定粘滞是否仍保留得下**：
        硬阻断（租约 / 禁用 / 长效避让）不可保留，短冷却可以保留（见
        `FailoverRouter.is_affinity_retainable`）。调用方传入的仍是同一个谓词，
        由服务端注入更严格的语义，判定口径因此只有一份。
        """
        if not session_id:
            return AffinityDecision(
                status="fresh",
                provider_key=primary_provider,
                wire_model=primary_model,
                upstream_conv_id=None,
                is_handoff=False,
                turn_count=1,
                tool_contract=tool_contract,
                requested_model=requested_model,
            )

        now = time.time()
        with self._lock:
            self._clean_expired_locked(now)
            entry = self._registry.get(_line_key(session_id, tool_contract, requested_model))

            if not entry:
                # 未绑定线程：客户端给的是线程标识，网关从未为它建立过上游会话，
                # 因此**没有**可回放的上游 id。把它当上游 id 传下去会让 provider
                # 走续聊分支，只发最后一条 user 之后的增量——首轮即只剩一句提问，
                # 工具契约与历史整段丢失（[TOOL-LOOP] 见模块 docstring）。
                return AffinityDecision(
                    status="fresh",
                    provider_key=primary_provider,
                    wire_model=primary_model,
                    upstream_conv_id=None,
                    is_handoff=False,
                    turn_count=1,
                    tool_contract=tool_contract,
                    requested_model=requested_model,
                )

            entry.last_accessed_at = now

            # 原绑定节点不可用：先区分「一时退避」与「不值得保留」
            if is_provider_available(entry.provider_key):
                self._stats["affinity_hits"] += 1
                return AffinityDecision(
                    status="bound",
                    provider_key=entry.provider_key,
                    wire_model=entry.model_name or primary_model,
                    upstream_conv_id=entry.upstream_conv_id,
                    is_handoff=False,
                    turn_count=entry.turn_count,
                    tool_contract=tool_contract,
                    requested_model=requested_model,
                )

            degraded_ok = can_retain_degraded or (lambda _p: False)
            # 原绑定节点一时退避（短冷却），**保留粘滞关系**：不迁移、不清洗上游
            # 会话 id。本次请求仍以常规候选为主选（调用方传入的 primary），由派发
            # 边界按 OPT-021 的「短冷却只降级」语义处置；冷却一结束，同一会话自动
            # 回到原节点、原上游会话，无需重建上下文。
            if degraded_ok(entry.provider_key):
                self._stats["affinity_degraded"] += 1
                return AffinityDecision(
                    status="degraded",
                    provider_key=entry.provider_key,
                    wire_model=entry.model_name or primary_model,
                    upstream_conv_id=entry.upstream_conv_id,
                    is_handoff=False,
                    turn_count=entry.turn_count,
                    tool_contract=tool_contract,
                    requested_model=requested_model,
                )

            # 原绑定节点硬阻断（租约 / 禁用 / 长效避让），执行平滑状态迁移
            self._stats["affinity_handoffs"] += 1
            orig_provider = entry.provider_key
            logger.info(
                "会话 [%s] 发生跨模型状态迁移: 原节点 [%s] 不可用 -> 迁移至候选目标 [%s]",
                session_id, orig_provider, primary_provider,
            )
            # 迁移时将绑定的 Provider 更新为新的 Provider，并剥离旧的异构 upstream_conv_id
            entry.provider_key = primary_provider
            entry.model_name = primary_model
            entry.upstream_conv_id = None

            return AffinityDecision(
                status="handoff",
                provider_key=primary_provider,
                wire_model=primary_model,
                upstream_conv_id=None,
                is_handoff=True,
                original_provider=orig_provider,
                turn_count=entry.turn_count,
                tool_contract=tool_contract,
                requested_model=requested_model,
            )

    def has_upstream_session(self, session_id: Optional[str], tool_contract: str = "",
                             requested_model: str = "") -> bool:
        """该线程是否已经持有**网关观测到的**上游会话 id。

        调用方据此在「把全量历史展平发出去」与「只发本轮增量」之间选择——这是
        [TOOL-LOOP] S1 的收口点：客户端自带的线程 id 永远不算数，只有 provider
        真的返回过 `conversation_id` 并且已 `record_turn` 的线程才算。
        只读，不改动任何状态。

        `tool_contract` / `requested_model` 必须与 `resolve_affinity` 传入的值
        一致：两者共用同一个 `_line_key`，否则调用方会在「本条线确有上游会话」
        时展平全量历史（多花钱），或反过来在一条新线上只发增量（模型看不到
        工具契约）。
        """
        if not session_id:
            return False
        now = time.time()
        with self._lock:
            self._clean_expired_locked(now)
            entry = self._registry.get(_line_key(session_id, tool_contract, requested_model))
            return bool(entry and entry.upstream_conv_id)

    def _clean_expired_locked(self, now: float) -> None:
        """清理已超时的会话缓存（内部私有方法，需持有锁）。"""
        expired_keys = [
            k for k, v in self._registry.items()
            if (now - v.last_accessed_at) > self.ttl_seconds
        ]
        for k in expired_keys:
            self._registry.pop(k, None)

    def clear(self) -> None:
        """重置会话粘滞表。"""
        with self._lock:
            self._registry.clear()
            self._stats = {
                "total_recorded_sessions": 0,
                "affinity_hits": 0,
                "affinity_degraded": 0,
                "affinity_handoffs": 0,
            }

    def get_diagnostics(self) -> dict:
        """获取会话粘滞与跨模型状态迁移统计指标。"""
        now = time.time()
        with self._lock:
            self._clean_expired_locked(now)
            active_count = len(self._registry)
            return {
                # [SESSION-IDENTITY] 两个数必须分开：`active_sessions` 是不同会话
                # id 的数量，`active_lines` 是注册表条数（一次会话可以有多条线，
                # 例如 Claude Code 的父会话与自动模式分类器）。只报前者会让
                # 「分类器是不是真的分到了一条独立线」无法从遥测上看出来；只报
                # 后者会把一对多误读成重复计数。
                "active_sessions": len({e.session_id for e in self._registry.values()}),
                "active_lines": active_count,
                "total_recorded_sessions": self._stats["total_recorded_sessions"],
                "affinity_hits": self._stats["affinity_hits"],
                # [M3] 「降级但保留」是本轮新增的第三种结局，必须可观测：
                # 它如果长期为 0 而 handoffs 很高，说明判据没有真正生效。
                "affinity_degraded": self._stats["affinity_degraded"],
                "affinity_handoffs": self._stats["affinity_handoffs"],
                "ttl_seconds": self.ttl_seconds,
            }


# 全局单例
session_affinity_manager = SessionAffinityManager()
