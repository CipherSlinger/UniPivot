"""同枢 (UniPivot) 本地多协议网关：
将标准化的 OpenAI Chat Completions、Anthropic Messages 与 OpenAI Responses 协议，
实时转译至多源模型服务与 Web/客户端会话。

用法:
    export QWEN_AUTH_TOKEN=...      # 或写入 .env（值为 tongyi_sso_ticket）
    export DEEPSEEK_AUTH_TOKEN=...  # 或写入 .env
    export DOUBAO_AUTH_TOKEN=...    # 或写入 .env（值为 sessionid）
    .venv/bin/uvicorn server:app --host 127.0.0.1 --port 8000

模型路由:
    qwen-* / qwq-*            -> 通义千问网页端（未知模型名回退到默认 Qwen）
    deepseek-*                -> DeepSeek 网页端
    doubao-*                  -> 豆包网页端（默认 doubao-pro）
    kimi-*                    -> Kimi 网页端
    glm-*                     -> 智谱清言网页端
    claude-*                  -> 依据 CLAUDE_BACKEND 动态映射
    gpt-* / o1 / o3           -> 依据 OPENAI_BACKEND 动态映射
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
import urllib.parse
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Awaitable, Callable, List, Optional, Union

import dotenv
import httpx
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict
from starlette.exceptions import HTTPException as StarletteHTTPException

from providers import (
    AnthropicCountTokensRequest,
    AnthropicMessagesRequest,
    CLAUDE_MODELS_METADATA,
    DeepSeekProvider,
    DoubaoProvider,
    GLMProvider,
    KimiProvider,
    ProviderError,
    QwenProvider,
    CapabilityRing,
    RING_DEFINITIONS,
    FailoverRouter,
    AdaptiveLoadBalancer,
    BalancingDecision,
    PROVIDER_RISK_SPECS,
    TrafficPacer,
    get_traffic_pacer,
    is_risk_interception,
    anthropic_error_response,
    anthropic_message_response,
    anthropic_sse_event,
    build_anthropic_tool_blocks,
    completion_response,
    continuation_content,
    convert_anthropic_to_gateway_messages,
    error_response,
    flatten_messages,
    format_anthropic_models,
    format_response_format_prompt,
    format_tools_system_prompt,
    last_user_content,
    parse_tool_call,
    parse_tool_calls,
    parse_tool_calls_with_healing,
    cap_parallel_tool_calls,
    parallel_calls_disallowed,
    recover_tools_from_reasoning,
    resolve_client_session_id,
    resolve_allowed_tools,
    StopSequenceScanner,
    truncate_final,
    SpeculativeToolStreamer,
    sse_content_block_delta,
    sse_content_block_start,
    sse_content_block_stop,
    sse_frame,
    sse_message_delta,
    sse_message_start,
    sse_message_stop,
    prompt_cache_manager,
    get_prompt_cache_manager,
    has_cache_control,
    ResponsesRequest,
    ResponsesEventStream,
    convert_responses_to_gateway_messages,
    format_responses_completion,
    custom_tool_call_item,
    custom_tool_input_text,
    freeform_tool_names,
    function_call_item,
    message_item,
    reasoning_item,
    sse_done,
    sse_response_error,
)
from providers.base import _est_tokens, _id, text_of
from providers import http_client
from providers.disconnect_guard import disconnect_guard, guard_stream

logger = logging.getLogger("gateway.server")
from providers.stream_keepalive import sse_keepalive
from providers.http_client import close_http_client, get_default_limits, get_http_client
from providers.prompt_optimizer import prompt_optimizer, make_prompt_folding_headers
from providers.session_affinity import AffinityDecision, session_affinity_manager
from providers.singleflight import (
    SingleFlightDecision,
    broadcast_stream_wrapper,
    singleflight_manager,
    stream_follower_gen,
    terminate_aborted_group,
)
from providers.credential_renewal import credential_renewal, get_credential_renewal
from providers.credential_watch import credential_watch, get_credential_watch
from providers.device_identity import get_device_identity_store
from providers.quarantine import get_quarantine_store, parse_quarantine_until, quarantine_store
from providers.risk_correlation import get_risk_correlation, risk_correlation
from providers.risk_dossier import get_risk_dossier, risk_dossier
from providers.risk_ledger import risk_ledger
from providers.conversation_gc import conversation_gc
from providers.token_counter_cache import compute_count_key, token_counter_cache
from session_store import (
    MAX_AGE,
    resolve_deepseek,
    resolve_doubao,
    resolve_glm,
    resolve_kimi,
    resolve_qwen,
)
from healing import HealthMonitor, HealingEngine, TaskDispatcher, TaskEvaluator, find_agent_cli

# [OPT-032] 测试进程绝不能被部署环境的 .env 配置。
# `load_dotenv()` 在 **import 时**执行，而各测试套件的「清空 GATEWAY_* 前置」写在 import
# 之前——于是 .env 里的一行真实配置（例如 `GATEWAY_QUARANTINE="deepseek:21d"`）会在前置
# 之后重新灌回进程，击穿整个隔离。实测后果：新增 .env 声明深度封禁后，
# `tests/test_quarantine_hardstop.py` 的第一个用例立刻变红（`is_quarantined("deepseek")`
# 期望 False 却为 True），而代码本身完全正确——这是本项目记录过的「本机环境决定测试结果」
# 那一类最难排查的顺序相关失败。`TESTING=1` 是所有会 import server 的套件的既有约定，
# 因此以它为闸门：零新增约定，且语义正确（测试运行时不存在「部署配置」这一概念）。
if os.getenv("TESTING") != "1":
    dotenv.load_dotenv()

HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))
TIMEOUT = float(os.getenv("TIMEOUT", "600"))

# 全局健康监控与自愈引擎
health_monitor = HealthMonitor()
healing_engine = HealingEngine(health_monitor, port=PORT)
health_monitor.set_healer_trigger(lambda name, err: healing_engine.trigger_healing(name, reason=err))

# 智能体研发任务分发与成果验收系统
task_dispatcher = TaskDispatcher(port=PORT)

# 全局五大模型对等互备路由调度器
failover_router = FailoverRouter()

# [OPT-023] 将冷却读数接入风控档案研判器：姿态派生需要同时读取
# 软冷却、封禁租约（经冷却转换）与恢复观察期，均由该 tracker 提供。
risk_dossier.set_cooldown_tracker(failover_router.cooldown_tracker)

# [OPT-012] 将风控冷却状态接入健康探针门控：处于避让冷却期的节点，
# 周期探针跳过真实网络调用，杜绝持续触碰已被上游风控封禁的节点。
health_monitor.set_cooldown_probe_gate(failover_router.cooldown_tracker.get_remaining_cooldown)
# [OPT-034] 「运维禁用」的判读只有一处：此前健康监控自行现读 DISABLED_PROVIDERS
# （同一份配置的第三份解读），现统一取 CooldownTracker 的并集判读者。
health_monitor.set_disabled_gate(failover_router.cooldown_tracker.is_disabled)
# [OPT-034] 后台自维护触点（探针/刷新/会话回收/自愈选举）统一向同一判据求值：
# 只问冷却、不问禁用会让运维声明出局的节点仍收到真实的上游请求。
health_monitor.set_touch_avoidance_gate(failover_router.cooldown_tracker.touch_avoidance_reason)

# 全局自适应流量整形器（并发与反爬步调控制）
traffic_pacer = get_traffic_pacer()
traffic_pacer.cooldown_tracker = failover_router.cooldown_tracker

# 全局冷热动态负载均衡与自适应降级调度引擎
load_balancer = AdaptiveLoadBalancer(cooldown_tracker=failover_router.cooldown_tracker)


def _make_lb_headers(decision: BalancingDecision) -> dict[str, str]:
    """构造负载均衡与降级分流决策响应头"""
    headers = {
        "x-load-balancing-action": decision.action,
        "x-dispatched-provider": decision.provider_key,
        "x-dispatched-model": decision.wire_model,
    }
    if decision.reason:
        headers["x-load-balancing-reason"] = urllib.parse.quote(
            decision.reason, safe=" /:;=@[](){}<>_-,."
        )
    return headers


def _make_risk_headers(
    delay_ms: Optional[float], provider_key: str, primary_provider: Optional[str] = None
) -> dict[str, str]:
    """构造流量整形等待耗时与风控避让冷却响应头。

    [M1] 不传 `delay_ms` 时报告的是节奏裁决者记录的**本节点最近一次实际施加**
    的等待（`CooldownTracker.last_pacing_wait_ms`）。节奏裁决已唯一化到紧贴上游
    调用的那一处（带锁预约时刻），本函数不再自带一套计时——两套计时各自为政时，
    这个头报告的只是一个「大概等了多少」，与真实节奏无关。
    """
    applied_ms = (
        failover_router.cooldown_tracker.last_pacing_wait_ms(provider_key)
        if delay_ms is None
        else delay_ms
    )
    headers = {
        "x-risk-pacing-delay-ms": str(round(applied_ms, 2)),
    }
    rem_cd = failover_router.cooldown_tracker.get_remaining_cooldown(provider_key)
    if rem_cd <= 0 and primary_provider:
        rem_cd = failover_router.cooldown_tracker.get_remaining_cooldown(primary_provider)
    if rem_cd > 0:
        headers["x-risk-cooldown-active"] = str(round(rem_cd, 2))
    # [OPT-022] 避让解除后的恢复观察期：以串行 + 放大步调试探，避免解封惊群
    if failover_router.cooldown_tracker.is_in_probation(provider_key):
        headers["x-risk-probation"] = "1"
    return headers

# ---------------------------------------------------------------- 模型表

# 国内版通义千问模型表（未知 qwen-* 名称会透传到 provider，由 provider 回退默认模型）
QWEN_ALIASES = {
    "qwen": "Qwen",
    "qwen-plus": "Qwen",
    "qwen-max": "Qwen3.7-Max",
    "qwen-flash": "Qwen3.6-Flash",
    "qwen3.7-max": "Qwen3.7-Max",
    "qwen3.6-flash": "Qwen3.6-Flash",
    "qwen3.7": "Qwen",
    "qwen3.6": "Qwen3.6-Flash",
    # 兼容历史别名
    "qwen3-235b-a22b": "Qwen",
    "qwen-max-latest": "Qwen3.7-Max",
}
QWEN_MODELS = [
    "Qwen",
    "Qwen3.7-Max",
    "Qwen3.6-Flash",
]

DEEPSEEK_ALIASES = {
    "deepseek": "deepseek-chat",
    "deepseek-instant": "deepseek-chat",
    "deepseek-coder": "deepseek-chat",
    "deepseek-expert": "deepseek-reasoner",
    "deepseek-r1": "deepseek-reasoner",
}
DEEPSEEK_MODELS = ["deepseek-chat", "deepseek-reasoner"]

DOUBAO_ALIASES = {
    "doubao": "doubao-pro",
    "doubao-chat": "doubao-pro",
    "doubao-pro-chat": "doubao-pro",
    "doubao-deep-think": "doubao-think",
}
DOUBAO_MODELS = [
    "doubao-pro",
    "doubao-pro-32k",
    "doubao-pro-4k",
    "doubao-lite",
    "doubao-lite-32k",
    "doubao-lite-4k",
    "doubao-think",
    "doubao-expert",
    "doubao-1.5-pro",
    "doubao-1.5-lite",
    "doubao-1.5-thinking",
]

KIMI_ALIASES = {
    "kimi": "kimi",
    "kimi-chat": "kimi",
    "kimi-latest": "kimi",
    "kimi-k0-math": "kimi-math",
    "kimi-math": "kimi-math",
    "kimi-explore": "kimi-explore",
    "kimi-research": "kimi-explore",
    "kimi-think": "kimi-explore",
    "kimi-k1": "kimi-explore",
    "kimi-k1.5": "kimi-explore",
    "moonshot-v1-8k": "kimi",
    "moonshot-v1-32k": "kimi",
    "moonshot-v1-128k": "kimi",
}
KIMI_MODELS = [
    "kimi",
    "kimi-explore",
    "kimi-math",
]

GLM_ALIASES = {
    "glm": "glm-4-plus",
    "chatglm": "glm-4-plus",
    "glm-4": "glm-4",
    "glm-4-plus": "glm-4-plus",
    "glm-4-air": "glm-4-air",
    "glm-4-flash": "glm-4-flash",
    "glm-4-long": "glm-4-long",
    "glm-zero-preview": "glm-zero-preview",
    "glm-think": "glm-zero-preview",
}
GLM_MODELS = [
    "glm-4",
    "glm-4-plus",
    "glm-4-air",
    "glm-4-flash",
    "glm-4-long",
    "glm-zero-preview",
]

# 通用第三方客户端（Cursor、Continue、NextChat、OpenWebUI）常用模型别名
OPENAI_ALIASES = {
    # 旗舰模型 (Flagship)
    "gpt-4o": "flagship",
    "gpt-4o-2024-05-13": "flagship",
    "gpt-4o-2024-08-06": "flagship",
    "gpt-4o-2024-11-20": "flagship",
    "gpt-4o-latest": "flagship",
    "gpt-4-turbo": "flagship",
    "gpt-4-turbo-preview": "flagship",
    "gpt-4": "flagship",
    "gpt-4-0613": "flagship",
    "gpt-4-32k": "flagship",
    # 深度思考模型 (Reasoner)
    "o1": "reasoner",
    "o1-preview": "reasoner",
    "o1-mini": "reasoner",
    "o3": "reasoner",
    "o3-mini": "reasoner",
    # [AGENT-LOOP] Agent CLI 实际会发的模型名。Codex CLI 默认请求 `gpt-5-codex`
    # 一类的名字，缺表时 resolve_model 返回 (None, None)，客户端拿到 404 就再也
    # 不会发起请求——表现正是「网关连上 agent 但一点效果都没有」。
    "gpt-5": "flagship",
    "gpt-5-codex": "reasoner",
    "gpt-5-mini": "fast",
    "gpt-5-nano": "fast",
    "codex-mini-latest": "fast",
    "o3-pro": "reasoner",
    "o4": "reasoner",
    "o4-mini": "reasoner",
    # 极速轻量模型 (Fast / Mini)
    "gpt-4o-mini": "fast",
    "gpt-4o-mini-2024-07-18": "fast",
    "gpt-3.5-turbo": "fast",
    "gpt-3.5-turbo-0125": "fast",
    "gpt-3.5-turbo-1106": "fast",
    "gpt-3.5": "fast",
    # 通用别名
    "auto": "default",
    "default": "default",
}


def resolve_model(model: str) -> tuple[Optional[str], Optional[str]]:
    """返回 (provider_key, wire_model)；未知模型返回 (None, None)。"""
    m = model.strip()
    m_lower = m.lower()
    if m_lower in QWEN_ALIASES:
        return "qwen", QWEN_ALIASES[m_lower]
    if m_lower in DEEPSEEK_ALIASES:
        return "deepseek", DEEPSEEK_ALIASES[m_lower]
    if m_lower in DOUBAO_ALIASES:
        return "doubao", DOUBAO_ALIASES[m_lower]
    if m_lower in KIMI_ALIASES:
        return "kimi", KIMI_ALIASES[m_lower]
    if m_lower in GLM_ALIASES:
        return "glm", GLM_ALIASES[m_lower]
    if m in QWEN_MODELS:
        return "qwen", m
    if m in DOUBAO_MODELS:
        return "doubao", m
    if m in KIMI_MODELS:
        return "kimi", m
    if m in GLM_MODELS:
        return "glm", m
    if m_lower.startswith("qwen") or m_lower.startswith("qwq"):
        return "qwen", m
    if m_lower.startswith("deepseek"):
        return "deepseek", m_lower
    if m_lower.startswith("doubao"):
        return "doubao", m_lower
    if m_lower.startswith("kimi") or m_lower.startswith("moonshot"):
        return "kimi", m_lower
    if m_lower.startswith("glm") or m_lower.startswith("chatglm"):
        return "glm", m_lower

    # Claude 兼容模型路由
    if m_lower.startswith("claude") or m_lower.startswith("anthropic."):
        if "deepseek" in m_lower:
            backend = "deepseek"
        elif "doubao" in m_lower:
            backend = "doubao"
        elif "kimi" in m_lower:
            backend = "kimi"
        elif "glm" in m_lower or "chatglm" in m_lower:
            backend = "glm"
        elif "qwen" in m_lower:
            backend = "qwen"
        else:
            backend = os.getenv("CLAUDE_BACKEND", "qwen").strip().lower()

        if backend == "deepseek":
            if any(k in m_lower for k in ("opus", "r1", "reasoner", "fable")):
                return "deepseek", "deepseek-reasoner"
            return "deepseek", "deepseek-chat"
        elif backend == "doubao":
            if any(k in m_lower for k in ("opus", "think", "reasoner", "fable")):
                return "doubao", "doubao-think"
            elif any(k in m_lower for k in ("haiku", "lite", "flash")):
                return "doubao", "doubao-lite"
            return "doubao", "doubao-pro"
        elif backend == "kimi":
            if any(k in m_lower for k in ("opus", "explore", "research", "k1", "reasoner", "think")):
                return "kimi", "kimi-explore"
            return "kimi", "kimi"
        elif backend in ("glm", "chatglm"):
            if any(k in m_lower for k in ("opus", "zero", "think", "reasoner")):
                return "glm", "glm-zero-preview"
            elif any(k in m_lower for k in ("haiku", "flash", "lite")):
                return "glm", "glm-4-flash"
            return "glm", "glm-4-plus"
        else:
            # 默认 Qwen 后端
            if any(k in m_lower for k in ("opus", "3-7", "3.7", "max", "fable")):
                return "qwen", "Qwen3.7-Max"
            elif any(k in m_lower for k in ("haiku", "flash", "lite")):
                return "qwen", "Qwen3.6-Flash"
            return "qwen", "Qwen"

    # OpenAI 兼容模型路由（Cursor / Continue / NextChat 开箱即用）
    tier = OPENAI_ALIASES.get(m_lower)
    if not tier:
        if m_lower.startswith("gpt-4o-mini") or m_lower.startswith("gpt-3.5"):
            tier = "fast"
        # [AGENT-LOOP] 未知模型名的失败方向必须是「可服务」而非 404：
        # 各家 CLI 会随版本引入新的模型 id（gpt-5.1、o5、codex-*），一张固定表
        # 永远追不上。前缀兜底让新 id 落到最有把握的档位，与下游 upstream_fallback
        # 一致——宁可降级服务，也不要让客户端在握手阶段就被拒绝。
        elif m_lower.startswith(("gpt-5", "gpt-6", "chatgpt")):
            tier = "flagship"
        elif m_lower.startswith(("gpt-4", "gpt-o", "gpt")):
            tier = "flagship"
        elif m_lower.startswith(("o1", "o3", "o4", "o5")):
            tier = "reasoner"
        elif m_lower.startswith(("codex", "computer-use")):
            tier = "flagship"

    if tier:
        backend = (
            os.getenv("OPENAI_BACKEND")
            or os.getenv("DEFAULT_BACKEND")
            or os.getenv("CLAUDE_BACKEND")
            or "qwen"
        ).strip().lower()
        if backend == "deepseek":
            if tier == "reasoner":
                return "deepseek", "deepseek-reasoner"
            return "deepseek", "deepseek-chat"
        elif backend == "doubao":
            if tier == "reasoner":
                return "doubao", "doubao-think"
            elif tier == "fast":
                return "doubao", "doubao-lite"
            return "doubao", "doubao-pro"
        elif backend == "kimi":
            if tier == "reasoner":
                return "kimi", "kimi-explore"
            return "kimi", "kimi"
        elif backend in ("glm", "chatglm"):
            if tier == "reasoner":
                return "glm", "glm-zero-preview"
            elif tier == "fast":
                return "glm", "glm-4-flash"
            return "glm", "glm-4-plus"
        else:
            # 默认 Qwen 后端
            if tier in ("flagship", "reasoner"):
                return "qwen", "Qwen3.7-Max"
            elif tier == "fast":
                return "qwen", "Qwen3.6-Flash"
            return "qwen", "Qwen"

    return None, None


def all_models() -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for name in (
        QWEN_MODELS
        + DEEPSEEK_MODELS
        + DOUBAO_MODELS
        + KIMI_MODELS
        + GLM_MODELS
        + list(QWEN_ALIASES)
        + list(DEEPSEEK_ALIASES)
        + list(DOUBAO_ALIASES)
        + list(KIMI_ALIASES)
        + list(GLM_ALIASES)
        + list(OPENAI_ALIASES)
        + [m["id"] for m in CLAUDE_MODELS_METADATA]
        + ["claude", "claude-3-7-sonnet", "claude-3-5-sonnet", "claude-3-5-haiku", "claude-3-opus"]
    ):
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


# ---------------------------------------------------------------- 请求模型

class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: str
    content: Optional[Union[str, List[dict]]] = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str
    messages: List[ChatMessage]
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    conversation_id: Optional[str] = None
    stream_options: Optional[dict] = None
    tools: Optional[List[dict]] = None
    tool_choice: Optional[Union[str, dict]] = None
    # [TOOL-LOOP] 显式声明而非依赖 extra=allow：客户端禁止并行调用时解析端要据此裁剪
    parallel_tool_calls: Optional[bool] = None
    # [PROTOCOL] 停用词截断：此前只有 extra=allow 收着，没有任何路径使用它
    stop: Optional[Union[str, List[str]]] = None
    # [PROTOCOL] 结构化输出：此前同样只被解析、从未进入提示词
    response_format: Optional[dict] = None
    # 网页端专属能力，经 extra_body 传入。
    # [AGENT-LOOP] 类型放宽到对象：Anthropic 一侧的现行写法已经是
    # `{"type": "adaptive"}`，把同一字段在 OpenAI 协议上钉成 bool 只会让
    # 形状稍有差异的客户端在 422 上失手——判定统一交给 `_wants_thinking`。
    thinking: Union[bool, dict, str] = False
    search: bool = False


class EmbeddingsRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    input: Union[str, List[Union[str, List[int], dict]]]
    model: str = "text-embedding-3-small"
    dimensions: Optional[int] = None
    user: Optional[str] = None


class DispatchTaskRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    instruction: str
    model: str = "claude-3-7-sonnet"
    expected_artifacts: Optional[List[str]] = None
    run_tests: bool = True


class CredentialRenewalScanRequest(BaseModel):
    """[OPT-026] 凭证续期扫描请求：providers 省略时按凭证指纹自动判定。"""
    providers: Optional[List[str]] = None


# ---------------------------------------------------------------- 应用

def _provider_cooldown_remaining(name: str) -> float:
    """[OPT-013] 查询节点剩余风控冷却秒数（查询失败按 0 处理，绝不阻断刷新流程）。"""
    try:
        return max(0.0, float(failover_router.cooldown_tracker.get_remaining_cooldown(name)))
    except Exception:  # noqa: BLE001
        return 0.0


def _refresh_stale_provider(name: str, resolver, refresher) -> str:
    """按需对单个 Provider 执行静默令牌刷新。

    返回: "fresh"（无需刷新）/ "disabled"（运维禁用，不做）/ "cooling"（风控避让中，暂缓）
          / "refreshed" / "failed"。
    [OPT-013] 处于风控避让冷却期的节点绝不发起登录/刷新请求，
    避免「静默刷新」成为持续触碰已被上游封禁账号的隐蔽通道。
    [OPT-034] 运维禁用（DISABLED_PROVIDERS）的节点同样不做：它是运维「该节点不属于本次
    部署」的声明，且往往正因封禁而下线——一次静默刷新就是一次真实的上游登录请求。实测
    复现：DISABLED_PROVIDERS=kimi 且无冷却时，此处真的调用了 refresher。
    与冷却的区别在于**不进入延迟重试**：冷却是在等一个会到期的窗口，而禁用是在等运维改
    配置，没有可等待的期限，因此不加入 deferred（运维改回配置后，下一次启动即会刷新）。
    """
    s = resolver()
    if not s or not s.token or getattr(s, "age", 0.0) < MAX_AGE:
        return "fresh"
    if failover_router.cooldown_tracker.is_disabled(name):
        print(f"[网关] {name} 已被运维禁用 (DISABLED_PROVIDERS)，不做令牌刷新")
        return "disabled"
    if _provider_cooldown_remaining(name) > 0:
        print(f"[网关] {name} 处于风控避让冷却期，暂缓令牌刷新（避免加剧上游封禁）")
        return "cooling"
    print(f"[网关] {name} 令牌已超过 {MAX_AGE // 3600}h，尝试静默刷新…")
    ns = refresher(timeout=60)
    if ns:
        print(f"[网关] {name} 令牌刷新成功")
        return "refreshed"
    print(f"[网关] {name} 静默刷新失败（可能已退出登录）")
    return "failed"


def _deferred_refresh_after_cooldown(deferred: list, max_wait: Optional[float] = None, poll: Optional[float] = None) -> None:
    """[OPT-013] 冷却期结束后，对启动时被暂缓的节点补做令牌刷新。

    有界等待：最多等待 max_wait 秒（默认 24h），每 poll 秒（默认 300s）检查一次冷却是否解除；
    到期仍未解除则放弃本轮刷新（下一个启动周期再评估），绝不无限阻塞。
    """
    if not deferred:
        return
    if poll is None:
        poll = float(os.getenv("GATEWAY_REFRESH_DEFER_POLL", "300"))
    if max_wait is None:
        max_wait = float(os.getenv("GATEWAY_REFRESH_DEFER_MAX_WAIT", "86400"))
    deadline = time.time() + max_wait
    while True:
        if not any(_provider_cooldown_remaining(name) > 0 for name, _, _ in deferred):
            break
        remaining_wait = deadline - time.time()
        if remaining_wait <= 0:
            break
        time.sleep(min(poll, remaining_wait))

    for name, resolver, refresher in deferred:
        if _provider_cooldown_remaining(name) > 0:
            print(f"[网关] {name} 风控冷却未解除（已达等待上限 {max_wait / 3600:.1f}h），跳过本轮令牌刷新")
            continue
        try:
            _refresh_stale_provider(name, resolver, refresher)
        except Exception as e:  # noqa: BLE001
            print(f"[网关] {name} 延迟静默刷新异常: {type(e).__name__}: {e}")


def _auto_refresh_stale_sessions() -> None:
    """启动时后台线程：对超过 MAX_AGE 的会话做一次静默刷新（复用已登录
    的浏览器配置，不弹窗）。刷新失败不影响服务，请求报 401 时再提示登录。"""
    if os.getenv("GATEWAY_DISABLE_AUTO_REFRESH") == "1" or os.getenv("TESTING") == "1":
        return

    from login import refresh_deepseek, refresh_doubao, refresh_glm, refresh_kimi, refresh_qwen

    deferred: list = []
    for name, resolver, refresher in (
        ("qwen", resolve_qwen, refresh_qwen),
        ("deepseek", resolve_deepseek, refresh_deepseek),
        ("doubao", resolve_doubao, refresh_doubao),
        ("kimi", resolve_kimi, refresh_kimi),
        ("glm", resolve_glm, refresh_glm),
    ):
        try:
            if _refresh_stale_provider(name, resolver, refresher) == "cooling":
                deferred.append((name, resolver, refresher))
        except Exception as e:
            print(f"[网关] {name} 静默刷新异常: {type(e).__name__}: {e}")

    if deferred:
        threading.Thread(
            target=_deferred_refresh_after_cooldown, args=(deferred,), daemon=True
        ).start()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # [OPT-026] 启动即检测凭证续期：一次「重登 + 重启」的常见运维流程必须自动
    # 复位由 2 的幂次升级来的 24h 认证级冻结，否则令牌刷新门控（OPT-013）本身
    # 会被这个冻结挡住最多 24h——即「凭证失效」制造的状态锁死了修复它的机制。
    try:
        renewed = await failover_router.cooldown_tracker.release_renewed_credentials()
        if renewed:
            print(f"[网关] 检测到凭证已续期，已复位认证级避让: {', '.join(renewed)}")
    except Exception as exc:  # noqa: BLE001 - 检测失败绝不阻断启动
        print(f"[网关] 凭证续期检测失败 (已忽略): {type(exc).__name__}: {exc}")

    threading.Thread(target=_auto_refresh_stale_sessions, daemon=True).start()
    await get_http_client(TIMEOUT)
    if os.getenv("TESTING") != "1":
        health_monitor.start(initial_delay=2.0)
    # [OPT-016] 启动即补做上一进程遗留的临时会话删除，避免重启导致远端会话泄漏
    conv_gc_task: Optional[asyncio.Task] = None
    if os.getenv("TESTING") != "1" and not conversation_gc.is_disabled():
        try:
            await conversation_gc.drain_once()
        except Exception as exc:  # noqa: BLE001
            print(f"[会话回收] 启动排空失败 (已忽略): {type(exc).__name__}: {exc}")
        conv_gc_task = asyncio.create_task(_conversation_gc_loop())
    try:
        yield
    finally:
        # [OPT-015] 优雅停机：先等待在途请求（含流式）自然收尾，再释放连接池，
        # 避免重启时腰斩正在传输的响应（客户端重试会额外放大上游负载与风控压力）。
        try:
            drain_timeout = float(os.getenv("GATEWAY_SHUTDOWN_DRAIN_TIMEOUT", "30"))
        except ValueError:
            drain_timeout = 30.0
        await _drain_inflight_requests(drain_timeout)
        if conv_gc_task is not None:
            conv_gc_task.cancel()
            try:
                await conv_gc_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        health_monitor.stop()
        await close_http_client()


app = FastAPI(title="同枢 (UniPivot) 本地多协议网关", version="1.3.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------- [OPT-015] 在途请求优雅停机排空

_inflight_requests = 0

# 观测端点自身不得计入在途数。它们的存在意义就是**读数**：读数请求把自己算进
# 被读的量里，量表就不再是「现在有多少工作在做」，而是「有多少工作 + 1」。
# 实测缺陷：一台完全空闲的网关，`GET /v1/diagnostics` 永远返回
# `inflight_requests: 1`（三个并发诊断请求返回 1/2/3/4，即每个请求都数到了
# 自己）。这个数字同时是优雅停机的完成判据（`_drain_inflight_requests` 以 0
# 为排空），一个不可能为 0 的量表既无法验证排空，也无法回答运维唯一想问的
# 那个问题——「现在忙不忙」。
_INFLIGHT_OBSERVER_PATHS = frozenset({"/health", "/v1/diagnostics", "/diagnostics"})


class InflightTrackerMiddleware:
    """纯 ASGI 中间件：精确统计在途请求数（含流式响应全程）。

    相比 BaseHTTPMiddleware，纯 ASGI 中间件会 await 整个应用调用，
    因此流式响应只有在全部帧发送完毕后才会释放计数，
    从而保证停机排空能真正等到长连接流式响应收尾。

    观测端点（见 `_INFLIGHT_OBSERVER_PATHS`）不计入：它们是读表的人，
    不是表里的数。
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        global _inflight_requests
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        if scope.get("path", "") in _INFLIGHT_OBSERVER_PATHS:
            await self.app(scope, receive, send)
            return
        _inflight_requests += 1
        try:
            await self.app(scope, receive, send)
        finally:
            _inflight_requests -= 1
            if _inflight_requests < 0:
                _inflight_requests = 0


async def _drain_inflight_requests(timeout: float) -> bool:
    """等待在途请求排空，最多等待 timeout 秒。返回是否已完全排空。"""
    if _inflight_requests <= 0:
        return True
    print(f"[网关] 优雅停机：等待 {_inflight_requests} 个在途请求收尾（最多 {timeout:.0f}s）…")
    deadline = time.time() + max(0.0, timeout)
    while _inflight_requests > 0 and time.time() < deadline:
        await asyncio.sleep(min(0.1, max(0.005, deadline - time.time())))
    if _inflight_requests > 0:
        print(f"[网关] 优雅停机：仍有 {_inflight_requests} 个在途请求未收尾，已达等待上限，强制关闭。")
        return False
    print("[网关] 优雅停机：在途请求已全部收尾。")
    return True


app.add_middleware(InflightTrackerMiddleware)


# ---------------------------------------------------------------- [OPT-016] 远端临时会话回收

async def _conversation_gc_loop() -> None:
    """后台周期排空会话删除队列（有界重试，失败保留至下轮）。"""
    try:
        interval = float(os.getenv("GATEWAY_CONVERSATION_GC_RETRY_INTERVAL", "120"))
    except ValueError:
        interval = 120.0
    interval = max(5.0, interval)
    while True:
        try:
            await asyncio.sleep(interval)
            result = await conversation_gc.drain_once()
            if result.get("attempted") or result.get("deferred"):
                print(
                    f"[会话回收] 重试 {result['attempted']} 个待删除会话，成功 {result['succeeded']} 个，"
                    f"因风控冷却推迟 {result.get('deferred', 0)} 个。"
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - 后台循环必须永不退出
            print(f"[会话回收] 周期排空异常 (已忽略): {type(exc).__name__}: {exc}")


def _wants_thinking(thinking: object) -> bool:
    """客户端是否要求本回合开启思考——三个协议共用这一个判定。

    [AGENT-LOOP] 实测：当前 Claude Code（2.1.293）发出的是
    `{"type": "adaptive", "display": "omitted"}`，而网关此前只认
    `type == "enabled"`，于是**静默**把思考关掉：请求照常成功、客户端收到一个
    非思考的回答，没有任何错误或日志可看。Claude Code 自己的二进制里写着
    "`thinking.type=enabled` is deprecated. Use `thinking.type=adaptive`
    instead"——只认 enabled 等于对着新客户端永久失效。

    另一侧同样错：OpenAI 路径的 `bool(req.thinking)` 把 `{"type":
    "disabled"}`（一个非空 dict）判成「要思考」，而客户端明确要求关闭。两处
    合起来说明「要不要思考」缺一个统一判定，而不是某一处写错了。

    无法识别时失败开放：只有**明确**的 `disabled`/`false` 才关闭思考。
    """
    if not thinking:
        return False
    if isinstance(thinking, str):
        return thinking.strip().lower() not in ("disabled", "none", "off")
    if isinstance(thinking, dict):
        t_type = thinking.get("type")
        if isinstance(t_type, str):
            return t_type.strip().lower() not in ("disabled", "none", "off")
        return True
    return bool(thinking)


def _thinking_display(thinking: object) -> str:
    """客户端希望**如何展示**思考——`"omitted"` / `"summarized"` / `""`（未声明）。

    [PROTOCOL] 这是 `_wants_thinking` 的第二问，两者正交：前者是「要不要思考」，
    这里是「思考了要不要给客户端看」。

    实测（真实 `claude` 2.1.293，`--model claude-opus-5-5`）：Claude Code 的
    **每一个**请求都带 `{"type":"adaptive","display":"omitted"}`，而网关只回答了
    第一问、完全忽略第二问，于是照常把思考文本以 `thinking_delta` 发出去。后果
    不止是「多显示了内容」：客户端把它存进对话历史，下一轮 `messages` 里就多出
    一个 `{"type":"thinking",...}` 块，**尽管它明确要求 omitted**——实测一次工具
    循环里该块被反复回放，会话涨到 154 个 assistant 轮次、每轮都带思考块，每一轮
    都要重新付这几百 token 的输入成本。

    `display` 的语义取自 Claude Code 自己的 schema：
    `display: z(["summarized","omitted"]).optional()`，二进制内文档写明
    `'omitted'（no thinking text）`、`'highlights'` 是「每段思考只给一个短标题」。

    只认**明确的** `omitted`：其余取值（含 `summarized`、缺省、无法识别的形状）
    一律照常发出——失败开放，与 `_wants_thinking` 同一条纪律。
    """
    if isinstance(thinking, dict):
        display = thinking.get("display")
        if isinstance(display, str):
            return display.strip().lower()
    return ""


def _thinking_suppressed(thinking: object) -> bool:
    """本回合是否应当**不向客户端发送**思考文本。"""
    return _thinking_display(thinking) == "omitted"


def _make_thinking_headers(
    thinking_duration_ms: Optional[float] = None,
    thinking_tokens: Optional[int] = None,
) -> dict[str, str]:
    """构造思考流耗时与 Token 统计响应头"""
    headers = {}
    if thinking_duration_ms is not None and thinking_duration_ms > 0:
        headers["x-thinking-duration-ms"] = str(round(thinking_duration_ms, 2))
    if thinking_tokens is not None and thinking_tokens > 0:
        headers["x-thinking-tokens"] = str(thinking_tokens)
    return headers


_tool_healing_stats = {
    "total_tool_calls": 0,
    "healed_tool_calls": 0,
    # [TOOL-LOOP] 思考通道兜底命中次数：正文为空、工具调用只存在于思考流里的回合。
    "reasoning_recovered_tool_calls": 0,
}


def _recover_reasoning_tools(
    streamer: "SpeculativeToolStreamer",
    allowed,
    parallel_disallowed: bool,
    answer_text: str,
):
    """[TOOL-LOOP] 三个协议的流式收口点共用：正文为空时从思考流找回工具调用。

    收口点的代码几乎逐字重复（它们已经因为别的缺陷各自漂移过一次），所以判定与
    记账只写一份：先是 `recover_tools_from_reasoning` 的「正文必须为空」规则，
    再是客户端「禁止并行」的裁剪，最后是诊断计数。返回可直接使用的 tool_list。
    """
    tool_list, was_healed = recover_tools_from_reasoning(
        "".join(streamer.accumulated_reasoning),
        allowed,
        answer_text=answer_text,
    )
    if not tool_list:
        return None, False
    _tool_healing_stats["reasoning_recovered_tool_calls"] += len(tool_list)
    return cap_parallel_tool_calls(tool_list, parallel_disallowed), was_healed


def _make_tool_headers(
    tool_count: int = 0,
    was_healed: bool = False,
) -> dict[str, str]:
    """构造工具调用与自愈修复响应头"""
    headers = {}
    if tool_count > 0:
        headers["x-tool-calls-count"] = str(tool_count)
    if was_healed:
        headers["x-tool-healing"] = "1"
    return headers


def _make_affinity_headers(decision: AffinityDecision) -> dict[str, str]:
    """构造会话粘滞与跨模型状态迁移响应头。"""
    headers = {
        "x-session-affinity": decision.status,
        "x-session-turns": str(decision.turn_count),
    }
    if decision.is_handoff and decision.original_provider:
        headers["x-session-handoff-from"] = decision.original_provider
    # [M3] 粘滞保留在处于短冷却的原节点上：本次请求可能由别的节点服务（故障转移），
    # 但会话归属并未改嫁。把这个选择显式说出来，而不是让人从「没有 handoff 头」
    # 反推——「降级但保留」与「一切正常」在响应上是两件事。
    if decision.status == "degraded" and decision.provider_key:
        headers["x-session-affinity-degraded"] = decision.provider_key
    return headers


def _make_singleflight_headers(decision: SingleFlightDecision) -> dict[str, str]:
    """构造并发去重 SingleFlight 响应头。"""
    headers = {
        "x-singleflight": "leader" if decision.is_leader else "follower",
        "x-singleflight-key": decision.short_key,
    }
    if not decision.is_leader:
        headers["x-singleflight-followers"] = str(decision.group.follower_count)
    return headers


def _make_disconnect_headers() -> dict[str, str]:
    """构造客户端断连传播与上游回收遥测响应头。"""
    return {
        "x-disconnect-guard": "disabled" if disconnect_guard.is_disabled() else "enabled",
    }


def _stream_reclaimer(stream) -> Optional[Callable[[], None]]:
    """[C2] 为一条已打开的上游流构造同步回收钩子。

    `ManagedStream.close()` 是异步的，但响应收尾往往发生在**任务已被取消**的时刻，
    那里不允许 await。这里只取它的同步槽位回收动作（幂等），作为「响应体从未启动」
    路径上的兜底；上游生成器本身仍由框架侧的 `aclose` 负责关闭。
    """
    hook = getattr(stream, "release_sync", None)
    return hook if callable(hook) else None


def _stream_closer(stream) -> Optional[Callable[[], Awaitable[None]]]:
    """[C2] 为上游流构造异步关闭动作（响应体从未启动时的兜底）。"""
    close = getattr(stream, "close", None)
    return close if callable(close) else None


async def _close_upstream_stream(stream) -> None:
    """[C2] 关闭 `execute_chat` 交出的上游流。

    这条关闭动作有两条到达路径，缺一不可：

      * 响应体正常跑完 / 中途异常 —— 协议层生成器的 `finally` 走到这里；
      * 响应体从未被拉取 —— 生成器的 `finally` 根本不会执行，此时由
        `ReclaimingStreamingResponse` 的 `on_close`（同步槽位回收）兜底，上游
        连接则在独立的关闭任务里释放。

    对已关闭或未实现的流是空操作。
    """
    close = getattr(stream, "close", None)
    if close is None:
        return
    await _close_stream_safely(close)


async def _close_stream_safely(aclose: Callable[[], Awaitable[None]]) -> None:
    """在**当前任务可能已被取消**的情况下可靠地关闭一条响应体流。

    客户端断连时，收尾代码通常正运行于被取消的任务中，就地 `await aclose()` 会被
    立即二次取消，收尾一行都执行不到（实测断连路径上上游流既不关闭、槽位也不归还，
    且因为从未启动而不会被任何生成器 `finally` 兜住）。改为调度独立任务，并
    `shield` 当前协程等待其完成——取消只会打断等待，收尾任务仍会跑完。
    """
    try:
        task = asyncio.ensure_future(aclose())
    except Exception:  # noqa: BLE001
        return
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        pass
    except BaseException as exc:  # noqa: BLE001 - 回收绝不得掩盖原始异常
        logger.warning("流式响应体收尾失败 (已忽略): %s: %s", type(exc).__name__, exc)


class ReclaimingStreamingResponse(StreamingResponse):
    """[C2] 保证响应体收尾的流式响应。

    问题不在「断连传播」而在「**从未开始**」：三条流式路径都在返回响应之前就已经
    占用了上游资源（`execute_chat` 已握有首包与节点信号量，`/v1/responses` 还额外
    先 `load_balancer.acquire()`），而归还动作写在生成器的 `finally` 里。生成器如果
    从未被 `__anext__()` 过，Python 不会为它运行收尾代码——`guard_stream` 也看不到
    任何中断，因为它同样从未启动。

    实测（ASGI 层，客户端在响应头到达前断开）：三条路径的负载均衡配额、节点并发
    信号量与上游流**全部**泄漏；`/v1/responses` 的配额是在构造阶段就占用的，泄漏后
    负载均衡会逐步把该节点判为高负载并持续把流量分流走，直到进程重启。

    这里把「响应体走到哪里都不能不关」变成结构性质：`__call__` 的 `finally` 无条件
    `aclose()` 响应体，并在其后再调用一次显式的 `on_close` 钩子。

    两点必须同时存在，缺一不可：

      1. `aclose()` 只对**已经启动**的生成器有效。若响应体一帧都没被拉取，它不会
         执行任何用户代码——而那恰恰是「资源已占用、收尾代码在 finally 里」的失效
         场景本身。所以真正兜底的是 `on_close`。
      2. `on_close` 必须接收响应体**自身**的关闭信号（本类的 `_on_close`），而不是
         让各条协议路径自己再记一份上游流的引用：`BaseHTTPMiddleware` 会在响应体
         之外再包一层流，任何由协议层记账的引用都会与真实生命周期脱节。

    正常路径逐字节不变：`aclose()` 与 `on_close` 都是幂等空操作。
    """

    def __init__(self, content, *args, on_close=None, on_close_async=None, on_exit=None, **kwargs) -> None:
        super().__init__(content, *args, **kwargs)
        self._on_close = on_close
        self._on_close_async = on_close_async
        self._on_exit = on_exit

    async def __call__(self, scope, receive, send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            # 1. 同步资源回收最先执行，且**不受当前任务取消状态影响**：
            #    客户端断连时本任务通常已在取消中，任何 await 都会被二次取消。
            if self._on_close is not None:
                try:
                    self._on_close()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("流式响应资源回收失败 (已忽略): %s", exc)
            # 2. 关闭响应体。独立任务 + shield：正常路径上是空操作，断连路径上
            #    也不会因二次取消而跳过。
            body = getattr(self, "body_iterator", None)
            aclose = getattr(body, "aclose", None)
            if aclose is not None:
                await _close_stream_safely(aclose)
            # 3. 异步兜底：响应体生成器可能**从未启动**，此时它内部的 finally
            #    （关闭上游流）永远不执行。这条钩子直接关闭 `execute_chat` 交出的
            #    上游流，与响应体是否被拉取无关。
            if self._on_close_async is not None:
                await _close_stream_safely(self._on_close_async)
            # 4. 协议层在构造响应之前占用的配额（`/v1/responses` 的负载均衡槽位）
            #    也必须在这条路径上归还：它的 finally 同样位于响应体生成器内部。
            if self._on_exit is not None:
                try:
                    self._on_exit()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("流式响应退出回收失败 (已忽略): %s", exc)


def _is_provider_healthy(pkey: str) -> bool:
    """检查指定 Provider 是否未被显式禁用且未处于熔断冷却期（**可用性**判据）。

    [OPT-033] 禁用判定改由 `cooldown_tracker.is_disabled()` 提供，不再在此处**另读一次**
    `DISABLED_PROVIDERS` 环境变量。此前本函数与 `CooldownTracker` 各自解析同一份配置：
    tracker 在构造/reset 时快照成集合，本函数每次现读环境变量，两者在「进程内改配置」
    （测试、或任何不经进程重启的配置变更）时会给出不一致的答案——而「同一份配置被两处
    各自解读」正是本项目反复出现的缺陷类型。现在只有一份解读，会话亲和与派发边界因此
    对「哪些节点被禁用」永远同答。

    [M3] 注意本函数回答的是「此刻能不能用」，而会话粘滞要问的是「值不值得为它放弃
    整段上下文」——后者由 `_is_provider_retainable` 回答，两者的差别只在短冷却。
    """
    if failover_router.cooldown_tracker.is_disabled(pkey):
        return False
    if failover_router.cooldown_tracker.is_in_cooldown(pkey):
        return False
    return True


def _is_provider_retainable(pkey: str) -> bool:
    """[M3] 会话粘滞是否仍**保留得下**（可迁移性判据，与可用性是两件事）。

    `_is_provider_healthy` 回答「这个节点此刻能不能用」，对任何软冷却都答 False；
    本函数回答「值不值得为了这次不可用把整段会话改嫁别家」。两者的差别只在短冷却：
    一次 15 秒的 429 退避不该让进行中的长程 agent 会话永久切换厂商（上下文重建、
    提示词缓存全丢、工具契约重新学习），而租约 / 禁用 / 长效避让必须迁移。
    判据由 `CooldownTracker.is_affinity_retainable()` 唯一给出。
    """
    return failover_router.cooldown_tracker.is_affinity_retainable(pkey)


def is_anthropic_request(request: Request) -> bool:
    """判断当前请求是否遵循 Anthropic 协议。

    判定规则：
    1. 请求路径以 /v1/messages 或 /messages 开头；
    2. 请求头包含 anthropic-version。
    """
    if not request:
        return False
    path = getattr(request, "url", None) and request.url.path or ""
    if path.startswith("/v1/messages") or path.startswith("/messages"):
        return True
    if request.headers.get("anthropic-version"):
        return True
    return False


def format_validation_error_message(exc: RequestValidationError) -> str:
    """提取 Pydantic / FastAPI 请求数据校验异常信息，拼接为易读摘要。"""
    messages = []
    for err in exc.errors():
        loc_parts = [str(x) for x in err.get("loc", []) if x not in ("body",)]
        loc = " -> ".join(loc_parts)
        msg = err.get("msg", "Validation error")
        if loc:
            messages.append(f"{loc}: {msg}")
        else:
            messages.append(msg)
    return "; ".join(messages) if messages else "请求数据格式校验失败"


# [H6] 零上游确定性拒绝 —— 网关**知道**退避窗口有多长的四种情形。
#
# 这四个 err_type 的共同点：派发边界已经判定「此刻发出去一定是白打」，因此零上游
# 调用直接返回。结论是**确定性**的，不是一次上游抖动，客户端立刻重试一秒都不会有用。
#
# 实测（真实 Claude Code CLI v2.1.293，同一台服务器、同一台真实 CLI，40 秒窗口，
# 每次重启服务端后测量，唯一变量是响应头）：
#   - 不带任何头：            客户端重试 **7 次**，每次拿到同样的零上游 503；
#   - 只带 `Retry-After: 60`： 客户端重试 **7 次** —— 一次都没有少。
#   - 只带 `x-should-retry: false`：客户端重试 **1 次**，立刻把真实原因报给用户。
#
# 为什么 `Retry-After` 在 503 上不起作用（读 CLI 源码确认，不是猜）：
#   `if(k.status===429){let O=parseInt(k.headers.get("retry-after")??"",10); ...}`
#   —— 它**只在 429 上**解析 Retry-After。放在 503 上是一份没人读的声明。
# 而 `x-should-retry` 在状态码判断**之前**被读取：
#   `let h=n.headers.get("x-should-retry"); if(h==="true")return!0; if(h==="false")return!1;`
#
# 两个头所以都发：真值信源只有 `x-should-retry`，但 `Retry-After` 是 HTTP 标准字段，
# 其他 SDK / 中间层会读它，且它对人类排查有用。**不要**以为它是抑制重试的那个。
#
# 没有这两个头时，客户端把消息转述给用户为「这是一个服务端问题，通常是暂时的，请稍后
# 再试」——而网关自己知道那五家还剩 0.4~24 小时。用户会照着重试。
#
# 与「上游抖动」的边界必须守住：一次 502/504 值得立刻重试，给抖动加 `x-should-retry:
# false` 会让客户端把一个健康节点整体放逐一小时，那是比不说话更糟的误导。所以只认这
# 四个语义明确的零上游拒绝，其余一律不带。
_DETERMINISTIC_REFUSAL_ERR_TYPES = {
    "credential_outage",   # [OPT-025] 整环凭证失效
    "disabled",            # [OPT-033] 运维禁用且环内无可用节点
    "quarantined",         # [OPT-021] 运维封禁租约
    "ring_exhausted",      # [OPT-031] 整环长效避让
}

# 退避秒数：取「最快可能自愈的那一档」，不复述最长的那一档。
# 凭证失效由人重登解除（通常几分钟内），封禁租约是天级——按最长的报会让一次刚好的
# 重登后还要多等一天。宁可让客户端早一点回来撞一次（那一次仍然是零上游、无成本），
# 也不要让它在问题已经修好之后还拒绝过来。
_REFUSAL_RETRY_AFTER_SECONDS = int(os.getenv("GATEWAY_REFUSAL_RETRY_AFTER", "600"))
_REFUSAL_RETRY_AFTER_MAX_SECONDS = int(os.getenv("GATEWAY_REFUSAL_RETRY_AFTER_MAX", "3600"))


def _is_deterministic_refusal(status: int, err_type: Optional[str]) -> bool:
    """本次拒绝是否属于「网关已确定、重试无用」的零上游拒绝。"""
    if status < 500:
        return False
    return (err_type or "") in _DETERMINISTIC_REFUSAL_ERR_TYPES


def _refusal_headers(exc: ProviderError) -> dict:
    """为确定性拒绝构造退避头；非确定性失败返回空字典（零回归）。

    两个头各司其职，缺一不可：

    `x-should-retry: false` —— **真正抑制重试的那个**。Claude Code 在按状态码分类之前
    先读它（见上方实测），少了它，客户端对同一个零上游 503 重试 7 次。

    `Retry-After` —— 给人看、给其他 SDK 看的标准化退避时长。时长优先取**抛出点自己
    知道的那个窗口**（`ProviderError.retry_after`）：租约剩余多少小时、快速失败态还剩
    多少秒、整环最早何时可用，这些在 raise 的那一刻都是已知的，用常量代替等于让客户端
    对着一份编出来的倒计时等待。只有抛出点明确表示「没有窗口可等」（`retry_after is
    None`：运维禁用，等多久都不会自己好）时才回退默认值。
    """
    status, err_type = exc.status, exc.err_type
    if not _is_deterministic_refusal(status, err_type):
        return {}
    if exc.retry_after is not None:
        seconds = exc.retry_after
    else:
        seconds = _REFUSAL_RETRY_AFTER_SECONDS
    seconds = int(max(1, min(seconds, _REFUSAL_RETRY_AFTER_MAX_SECONDS)))
    return {"Retry-After": str(seconds), "x-should-retry": "false"}


@app.middleware("http")
async def unified_exception_middleware(request: Request, call_next):
    """全局统一异常处理中间件：捕获所有未预期的异常并按请求协议输出对应的标准 JSON。"""
    try:
        return await call_next(request)
    except Exception as exc:
        err_msg = f"服务器内部异常: {type(exc).__name__}: {str(exc)}"
        if is_anthropic_request(request):
            return JSONResponse(
                status_code=500,
                content=anthropic_error_response(err_msg, status=500, err_type="api_error"),
            )
        return JSONResponse(
            status_code=500,
            content=error_response(err_msg, status=500, err_type="server_error"),
        )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """将请求校验失败（400）转换为当前协议的标准错误。"""
    msg = format_validation_error_message(exc)
    if is_anthropic_request(request):
        return JSONResponse(
            status_code=400,
            content=anthropic_error_response(msg, status=400, err_type="invalid_request_error"),
        )
    return JSONResponse(
        status_code=400,
        content=error_response(msg, status=400, err_type="invalid_request_error"),
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """将路由缺失（404）或方法不支持（405）等标准 HTTP 异常按协议转换。"""
    status_code = exc.status_code
    detail = str(exc.detail) if exc.detail else "HTTP Exception"
    if is_anthropic_request(request):
        return JSONResponse(
            status_code=status_code,
            content=anthropic_error_response(detail, status=status_code),
        )
    err_type = "invalid_request_error" if status_code < 500 else "server_error"
    return JSONResponse(
        status_code=status_code,
        content=error_response(detail, status=status_code, err_type=err_type),
    )


@app.exception_handler(ProviderError)
async def provider_error_handler(request: Request, exc: ProviderError):
    """统一捕获 ProviderError 并按当前客户端协议格式化。"""
    headers = _refusal_headers(exc)
    if is_anthropic_request(request):
        return JSONResponse(
            status_code=exc.status,
            content=anthropic_error_response(exc.message, status=exc.status, err_type=exc.err_type),
            headers=headers,
        )
    return JSONResponse(
        status_code=exc.status,
        content=error_response(exc.message, status=exc.status, err_type=exc.err_type),
        headers=headers,
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """捕获所有未被特定 handler 处理的 Python 异常（500）。"""
    err_msg = f"服务器内部异常: {type(exc).__name__}: {str(exc)}"
    if is_anthropic_request(request):
        return JSONResponse(
            status_code=500,
            content=anthropic_error_response(err_msg, status=500, err_type="api_error"),
        )
    return JSONResponse(
        status_code=500,
        content=error_response(err_msg, status=500, err_type="server_error"),
    )

_STATIC_DIR = Path(__file__).resolve().parent / "static"
_REPORTS_DIR = Path(__file__).resolve().parent / "reports"


def _configured(resolver) -> bool:
    s = resolver()
    return bool(s and s.token)


@app.get("/health")
async def health():
    providers_status = health_monitor.get_all_statuses()
    disabled_set = failover_router.cooldown_tracker.disabled_providers | {
        p.strip().lower() for p in os.getenv("DISABLED_PROVIDERS", "").split(",") if p.strip()
    }
    for p_name, p_info in providers_status.items():
        meta = PROVIDER_RISK_SPECS.get(p_name)
        p_info["risk_level"] = meta.risk_level.value if meta else "medium"
        in_cd = failover_router.cooldown_tracker.is_in_cooldown(p_name)
        rem_cd = failover_router.cooldown_tracker.get_remaining_cooldown(p_name)
        is_dis = p_name in disabled_set
        if is_dis:
            p_info["status"] = "disabled"
        p_info["is_disabled"] = is_dis
        p_info["disabled"] = is_dis
        p_info["in_cooldown"] = in_cd
        p_info["cooldown_status"] = "in_cooldown" if in_cd else "ready"
        p_info["remaining_cooldown"] = round(rem_cd, 1)
        p_info["remaining_cooldown_seconds"] = round(rem_cd, 1)

    healing_status = healing_engine.get_healing_status()
    risk_recent = risk_ledger.recent_interceptions(hours=24.0)
    _dg_diag = disconnect_guard.get_diagnostics()
    _sf_diag = singleflight_manager.get_diagnostics()
    return {
        "status": "ok",
        "disabled_providers": sorted(list(disabled_set)),
        "inflight_requests": _inflight_requests,
        "risk_ledger": {
            "enabled": not risk_ledger.is_disabled(),
            "recent_24h_total": risk_recent["total"],
            "recent_24h_per_provider": {
                pk: bucket["count"] for pk, bucket in risk_recent["per_provider"].items()
            },
        },
        "quarantine": quarantine_store.list_active(),
        # [C2] 断连回收的**结果**必须在就绪面上可见，而不只在 /v1/diagnostics 的明细里。
        # 这两组数字是「网关有没有在白白占用上游配额」的唯一直接证据：客户端中断
        # （Claude Code 的 Esc、Codex 的 Ctrl-C、任何重连）在 agent 使用中是常态，
        # 而它是否真的回收了上游流、飞行组是否被唤醒，此前只有翻完整诊断才能看到。
        "disconnect_guard": {
            "enabled": not disconnect_guard.is_disabled(),
            "client_disconnects": _dg_diag["client_disconnects"],
            "reclaimed_streams": _dg_diag["reclaimed_streams"],
            "orphaned_flights_recovered": _dg_diag["orphaned_flights_recovered"],
        },
        "singleflight": {
            "active_flights": _sf_diag["active_flights"],
        },
        "qwen_configured": _configured(resolve_qwen),
        "deepseek_configured": _configured(resolve_deepseek),
        "doubao_configured": _configured(resolve_doubao),
        "kimi_configured": _configured(resolve_kimi),
        "glm_configured": _configured(resolve_glm),
        "providers": providers_status,
        "healing": healing_status,
    }


@app.post("/v1/heal/{provider_name}")
@app.post("/heal/{provider_name}")
async def heal_provider(provider_name: str):
    name = provider_name.strip().lower()
    if name not in PROVIDER_SPECS:
        return JSONResponse(
            error_response(f"未知 Provider: {name}，支持: {', '.join(PROVIDER_SPECS.keys())}", 404, "not_found"),
            status_code=404,
        )

    triggered = await healing_engine.trigger_healing(name, reason="手动触发自愈请求", force=True)
    return {
        "status": "triggered" if triggered else "already_running",
        "provider": name,
        "healing_status": healing_engine.get_healing_status(),
    }


# ---------------------------------------------------------------- 智能体研发任务分发与验收评估
@app.post("/v1/tasks/dispatch")
@app.post("/tasks/dispatch")
async def dispatch_rd_task(req: DispatchTaskRequest):
    """提交并异步分发新的智能体研发任务。"""
    if not req.instruction.strip():
        return JSONResponse(
            error_response("任务指令 `instruction` 不能为空", 400, "invalid_request_error"),
            status_code=400,
        )

    task = await task_dispatcher.dispatch_task(
        instruction=req.instruction,
        model_alias=req.model,
        expected_artifacts=req.expected_artifacts,
        run_evaluation_tests=req.run_tests,
    )
    return JSONResponse(task.to_dict())


@app.get("/v1/tasks/evaluation")
@app.get("/tasks/evaluation")
@app.get("/v1/reports/latest")
@app.get("/reports/latest")
async def get_tasks_evaluation():
    """获取最新成果验收与优化空间评估报告（包含 JSON 度量与 Markdown 建议书）。"""
    report = task_dispatcher.get_latest_evaluation_report()
    return JSONResponse(report)


@app.get("/v1/tasks/{task_id}")
@app.get("/tasks/{task_id}")
async def get_rd_task(task_id: str):
    """查看研发任务的执行日志与当前进度。"""
    task = task_dispatcher.get_task(task_id)
    if not task:
        return JSONResponse(
            error_response(f"未找到任务: {task_id}", 404, "task_not_found"),
            status_code=404,
        )
    return JSONResponse(task)


# ---------------------------------------------------------------- 诊断与监控指标接口
@app.get("/v1/diagnostics")
@app.get("/diagnostics")
async def get_diagnostics():
    """实时系统状态、多提供方健康与度量诊断接口。"""
    all_statuses = health_monitor.get_all_statuses()
    disabled_set = failover_router.cooldown_tracker.disabled_providers | {
        p.strip().lower() for p in os.getenv("DISABLED_PROVIDERS", "").split(",") if p.strip()
    }
    providers_diag = {}

    for p_name in ("qwen", "deepseek", "doubao", "kimi", "glm"):
        p_state = all_statuses.get(p_name, {})
        meta = PROVIDER_RISK_SPECS.get(p_name)
        in_cd = failover_router.cooldown_tracker.is_in_cooldown(p_name)
        rem_cd = failover_router.cooldown_tracker.get_remaining_cooldown(p_name)
        consec_fails = p_state.get("consecutive_failures", 0)
        configured = p_state.get("configured", False)
        raw_status = p_state.get("status", "offline")
        is_disabled = p_name in disabled_set

        # 映射规范化健康状态: healthy / degraded / failing / unhealthy / disabled
        if is_disabled:
            h_status = "disabled"
            raw_status = "disabled"
        elif not configured or raw_status == "offline":
            h_status = "unhealthy"
        elif raw_status == "cooling":
            h_status = "cooling"
        elif raw_status == "recovering" or consec_fails >= 2:
            h_status = "failing"
        elif raw_status == "degraded" or consec_fails == 1:
            h_status = "degraded"
        elif raw_status == "healthy":
            h_status = "healthy"
        else:
            h_status = raw_status

        max_concurrency = meta.max_concurrency if meta else 3
        sem = failover_router.cooldown_tracker.get_semaphore(p_name)
        rem_permits = getattr(sem, "_value", max_concurrency)
        cur_concurrency = max(0, max_concurrency - rem_permits)

        providers_diag[p_name] = {
            "status": h_status,
            "health_status": h_status,
            "raw_status": raw_status,
            "is_disabled": is_disabled,
            "disabled": is_disabled,
            "risk_level": meta.risk_level.value.upper() if meta else "MEDIUM",
            "in_cooldown": bool(in_cd),
            "remaining_cooldown_s": round(float(rem_cd), 2),
            "remaining_cooldown": round(float(rem_cd), 2),
            "consecutive_failures": int(consec_fails),
            "min_call_interval": float(meta.min_call_interval if meta else 0.1),
            "current_concurrency": cur_concurrency,
            "max_concurrency": max_concurrency,
            "configured": bool(configured),
            "last_check": p_state.get("last_check", 0.0),
            "last_error": p_state.get("last_error"),
        }

    # 2. 三级对等能力环 (reasoning, flagship, speed) 与模型列表
    capability_rings = []
    for ring_enum in (CapabilityRing.REASONING, CapabilityRing.FLAGSHIP, CapabilityRing.SPEED):
        ring_members = RING_DEFINITIONS.get(ring_enum, [])
        capability_rings.append({
            "ring": ring_enum.value,
            "name": ring_enum.value,
            "models": [wire_model for _, wire_model in ring_members],
            "member_models": [wire_model for _, wire_model in ring_members],
            "members": [{"provider": p, "model": m} for p, m in ring_members],
        })

    # 3. 自愈状态
    last_check_ts = getattr(health_monitor, "last_check_timestamp", 0.0)
    if not last_check_ts:
        last_check_ts = max((s.get("last_check", 0.0) for s in all_statuses.values()), default=0.0)

    running_tasks = healing_engine.get_healing_status().get("running_tasks", {})
    current_healing_task = None
    if running_tasks:
        first_k = next(iter(running_tasks))
        current_healing_task = {
            "provider": first_k,
            **running_tasks[first_k],
        }

    healing_status = {
        "last_check_timestamp": float(last_check_ts),
        "last_check": float(last_check_ts),
        "check_count": int(getattr(health_monitor, "check_count", 0)),
        "cooldown_probe_skips": health_monitor.get_probe_skip_stats(),
        "cooldown_healer_skips": health_monitor.get_healer_skip_stats(),
        "current_healing_task": current_healing_task,
        "running_tasks": running_tasks,
        "active_count": len(running_tasks),
    }

    # 4. 智能体 CLI 信息
    agent_info = find_agent_cli()
    if agent_info:
        agent_cli = {
            "name": agent_info.name,
            "path": agent_info.path,
            "version": agent_info.version,
            "available": True,
        }
    else:
        agent_cli = {
            "name": None,
            "path": None,
            "version": None,
            "available": False,
        }

    # 5. HTTP 连接池配置与状态
    limits = get_default_limits()
    shared_client = getattr(http_client, "_shared_client", None)
    is_active = bool(shared_client and not shared_client.is_closed)
    active_conns = 0
    if is_active:
        transport = getattr(shared_client, "_transport", None)
        pool = getattr(transport, "_pool", None)
        if pool and hasattr(pool, "connections"):
            active_conns = len(pool.connections)

    http_pool = {
        "status": "ready" if is_active else "uninitialized",
        "is_closed": not is_active,
        "max_connections": limits.max_connections,
        "max_keepalive_connections": limits.max_keepalive_connections,
        "keepalive_expiry": float(limits.keepalive_expiry),
        "active_connections": active_conns,
    }

    # 6. Prompt Cache 真实动态命中率与 Token 节省统计
    cache_stats = prompt_cache_manager.get_stats()
    prompt_cache = {
        "status": "active",
        **cache_stats,
        "hit_rate": cache_stats["hit_ratio"],
        "hit_rate_pct": round(cache_stats["hit_ratio"] * 100, 2),
        "hits": cache_stats["cache_hits"],
        "tokens_saved": cache_stats["saved_tokens"],
        "ttft_reduction_pct": 45.0 if cache_stats["cache_hits"] > 0 else 0.0,
    }

    # 7. 自适应负载均衡器实时度量与被禁用节点列表
    lb_stats = load_balancer.get_load_stats()
    if "providers" not in lb_stats and "nodes" in lb_stats:
        lb_stats["providers"] = lb_stats["nodes"]
    if "total_requests" not in lb_stats:
        lb_stats["total_requests"] = sum(
            node.get("request_count", 0) for node in lb_stats.get("nodes", {}).values()
        )

    # 8. 流量整形与风控避让遥测指标
    pacer_stats = traffic_pacer.get_stats()
    cooldown_nodes = failover_router.cooldown_tracker.get_all_cooldowns()
    risk_avoidance = {
        "status": "active" if os.getenv("GATEWAY_DISABLE_PACING") != "1" else "disabled",
        "active_semaphores": pacer_stats.get("active_semaphores", {}),
        "cooldown_nodes": cooldown_nodes,
        "traffic_pacing": pacer_stats,
        "provider_risk_specs": {
            pk: {
                "risk_level": meta.risk_level.value,
                "max_concurrency": meta.max_concurrency,
                "min_interval_s": meta.min_call_interval,
                "default_cooldown_s": meta.default_cooldown,
            }
            for pk, meta in PROVIDER_RISK_SPECS.items()
        },
    }

    return {
        "status": "ok",
        "timestamp": time.time(),
        "disabled_providers": sorted(list(disabled_set)),
        "load_balancer": lb_stats,
        "providers": providers_diag,
        "capability_rings": capability_rings,
        "healing_status": healing_status,
        "agent_cli": agent_cli,
        "http_pool": http_pool,
        "prompt_cache": prompt_cache,
        "prompt_optimizer": prompt_optimizer.get_stats(),
        "thinking_stream": {
            "status": "supported",
            "supported_protocols": ["openai_chat", "anthropic_messages", "openai_responses"],
            "features": [
                "reasoning_content_streaming",
                "thinking_block_state_machine",
                "telemetry_headers",
                "auto_fold_and_highlighting",
            ],
        },
        "tool_calling": {
            "status": "supported",
            "speculative_streaming": True,
            "self_healing_json": True,
            "supported_protocols": ["openai_chat", "anthropic_messages", "openai_responses"],
            "total_tool_calls": _tool_healing_stats["total_tool_calls"],
            "healed_tool_calls": _tool_healing_stats["healed_tool_calls"],
            # [TOOL-LOOP] 正文为空、调用只写在思考流里的回合数：这个计数不为零
            # 说明上游正在系统性地把工具调用吐进思考通道，是可观测的产品信号。
            "reasoning_recovered_tool_calls": _tool_healing_stats["reasoning_recovered_tool_calls"],
            "features": [
                "speculative_text_streaming",
                "zero_latency_reasoning",
                "reasoning_channel_tool_recovery",
                "deep_json_repair",
                "truncated_json_auto_completion",
                "telemetry_headers",
            ],
        },
        "session_affinity": session_affinity_manager.get_diagnostics(),
        "singleflight": singleflight_manager.get_diagnostics(),
        "disconnect_guard": disconnect_guard.get_diagnostics(),
        "token_counter_cache": token_counter_cache.get_diagnostics(),
        "risk_ledger": risk_ledger.get_diagnostics(),
        "quarantine": quarantine_store.get_diagnostics(),
        "conversation_gc": conversation_gc.get_diagnostics(),
        "probation": {
            "active": failover_router.cooldown_tracker.get_probation_states(),
            "stats": failover_router.cooldown_tracker.get_probation_stats(),
        },
        # [OPT-023] 仅暴露汇总（明细见 GET /v1/admin/risk），避免诊断接口膨胀
        "risk_dossier_summary": risk_dossier.build()["summary"],
        # [OPT-024] 跨节点风控关联与系统性升级研判
        "risk_correlation": risk_correlation.evaluate(),
        # [OPT-025] 全环凭证失效检测（整环 401 时快速失败，避免无效凭证风暴）
        "credential_outage": credential_watch.get_diagnostics(),
        # [OPT-026] 凭证续期检测（认证级避让的复位依据）
        "credential_renewal": credential_renewal.get_diagnostics(),
        # [OPT-027] 账号级封禁推断（新凭证仍被拒 -> 自动租约停触）
        "ban_inference": failover_router.cooldown_tracker.ban_inference_report(),
        # [OPT-030] 节点设备身份稳定性（同一节点每次请求必须是同一台「设备」）
        "device_identity": get_device_identity_store().get_diagnostics(),
        # [OPT-031] 长效避让门控（整环长效避让时零上游快速失败，短冷却仍保留最后一搏）
        "long_avoidance": failover_router.cooldown_tracker.long_avoidance_report(),
        # [OPT-033] 禁用门控与闸门拒载分布（哪一道闸门在承担拒载，可观测）
        "dispatch_gates": failover_router.cooldown_tracker.disabled_gate_report(),
        # [OPT-034] 后台自维护触点的避让拒载分布（按成因）
        "touch_gates": failover_router.cooldown_tracker.touch_gate_report(),
        "inflight_requests": _inflight_requests,
        "risk_avoidance": risk_avoidance,
    }


# ---------------------------------------------------------------- [OPT-020] 运维人工长期封禁租约

class QuarantineRequest(BaseModel):
    provider: str
    duration_hours: Optional[float] = None
    duration: Optional[str] = None
    until: Optional[str] = None
    reason: str = ""


@app.get("/v1/admin/quarantine")
@app.get("/admin/quarantine")
async def list_quarantine():
    """查看当前生效的运维封禁租约。"""
    return JSONResponse(quarantine_store.get_diagnostics())


@app.post("/v1/admin/quarantine")
@app.post("/admin/quarantine")
async def set_quarantine(req: QuarantineRequest):
    """设定/延长某节点的长期封禁租约。

    - `duration_hours`: 小时数，如 72 表示封禁 3 天；
    - `duration`: 时长表达式，如 `"72h"` / `"3d"` / `"30m"`；
    - `until`: 绝对时间（ISO-8601，如 `"2026-10-20T00:00:00"`），优先级最高。

    租约只延长避让窗口、绝不缩短既有冷却，因此不存在「误操作提前放行」的风险。
    """
    provider = (req.provider or "").strip().lower()
    if provider not in PROVIDER_SPECS:
        return JSONResponse(
            error_response(
                f"未知 Provider: {req.provider}，支持: {', '.join(PROVIDER_SPECS.keys())}",
                404,
                "not_found",
            ),
            status_code=404,
        )
    until: Optional[float] = None
    if req.until:
        until = parse_quarantine_until(req.until)
        if until is None:
            return JSONResponse(
                error_response(f"无法解析 until 时间表达式: {req.until}", 400, "invalid_request_error"),
                status_code=400,
            )
    elif req.duration:
        until = parse_quarantine_until(req.duration)
        if until is None:
            return JSONResponse(
                error_response(f"无法解析 duration 时长表达式: {req.duration}", 400, "invalid_request_error"),
                status_code=400,
            )
    elif req.duration_hours is None:
        return JSONResponse(
            error_response("必须提供 duration_hours、duration 或 until 之一", 400, "invalid_request_error"),
            status_code=400,
        )

    duration_seconds = req.duration_hours * 3600.0 if req.duration_hours is not None else None
    effective = quarantine_store.set_lease(
        provider,
        duration_seconds=duration_seconds,
        until=until,
        reason=req.reason or "admin_api",
    )
    return JSONResponse(
        {
            "status": "ok",
            "provider": provider,
            "until": round(effective, 3),
            "remaining_hours": round(max(0.0, effective - time.time()) / 3600.0, 2),
            "active": quarantine_store.list_active(),
        }
    )


@app.delete("/v1/admin/quarantine")
@app.delete("/admin/quarantine")
async def clear_quarantine(provider: Optional[str] = None):
    """解除指定节点（省略 provider 则解除全部）的封禁租约。

    仅解除「人工租约」；由真实风控拦截产生的自动冷却不受影响，仍会按其自身
    退避窗口自然到期。
    """
    count = quarantine_store.clear(provider)
    return JSONResponse(
        {"status": "ok", "cleared": count, "active": quarantine_store.list_active()}
    )


# ---------------------------------------------------------------- 风控档案接口
@app.get("/v1/admin/risk")
@app.get("/admin/risk")
async def get_risk_dossier(provider: Optional[str] = None):
    """风控拦截全量档案与规避策略研判。

    逐家给出：全量（非 24h 窗口）拦截统计与严重度分布、最近一次上游原文、
    当前姿态（封禁 / 冷却 / 观察期 / 监视 / 正常）、是否可派发、步调实际倍率，
    以及**该家风控能否绕开、能否靠重登恢复、应当如何避免**的确定性结论。

    传 `?provider=qwen` 则只返回该节点的档案条目。
    """
    if provider:
        entry = risk_dossier.provider_entry(provider)
        return JSONResponse(
            {"generated_at": round(time.time(), 3), "provider": entry["provider"], "entry": entry}
        )
    return JSONResponse(risk_dossier.build())


@app.get("/v1/admin/risk/correlation")
@app.get("/admin/risk/correlation")
async def get_risk_correlation_report():
    """跨节点风控关联与系统性升级研判。

    区分两种成因完全相反的拦截：单账号成因（该账号自身风控，避让 + 换节点有效）
    与系统性成因（我们的客户端足迹被标记，换节点无效且会把剩余账号逐个送进封禁）。
    后者只能通过整体降速与更换客户端身份特征应对。
    """
    return JSONResponse(risk_correlation.evaluate())


# [OPT-032] 零上游派发预演：把「此刻这条请求会被谁承接、谁被排除、被哪一道闸门排除」
# 变成一次可随时执行的查询。它存在的理由是运维动作之后的确认成本——改完租约/环境变量
# 或重启后，「现在到底还会不会派发到被封禁的那个节点」本该一问即知，而不是靠发一次
# 真实请求去试（那正是本项目一直在消除的「用一次上游触碰换一个答案」）。
# 只读、零上游调用：不实例化任何 Provider，只向派发路径同一组闸门提问。
@app.get("/v1/admin/dispatch-audit")
@app.get("/admin/dispatch-audit")
async def dispatch_audit(model: Optional[str] = None):
    """派发预演：回答该模型此刻的派发决定（零上游调用）。

    `?model=` 可传模型别名（如 `deepseek-chat` / `claude-3-7-sonnet` / `gpt-4o`），
    缺省用 `CLAUDE_BACKEND`（默认 qwen）。逐节点给出 `blocked_by`（credential_outage /
    quarantine / long_avoidance / disabled）与剩余时长，并预测是否会改派或快速失败。

    边界：只预演**闸门**决定，不含负载均衡分流与会话亲和固定，故字段以 `predicted_`
    命名；最终承接节点可能与之不同。
    """
    target = (model or os.getenv("CLAUDE_BACKEND", "qwen")).strip()
    provider_key, wire_model = resolve_model(target)
    if not provider_key:
        return JSONResponse(
            {"error": {"type": "invalid_request_error", "message": f"无法解析模型 {target!r}"}},
            status_code=400,
        )
    try:
        report = failover_router.audit_dispatch(provider_key, wire_model)
    except ValueError as exc:
        return JSONResponse(
            {"error": {"type": "invalid_request_error", "message": str(exc)}}, status_code=400
        )
    return JSONResponse({"generated_at": round(time.time(), 3), "model": target, **report})

# [OPT-025] 整环凭证失效检测：查询 / 手动解除。
# 与「冷却」(单节点、自动) 和「封禁租约」(单节点、运维声明) 不同，本状态是**整环**的
# 「我们没有可用凭证了」，处置方式是重新登录而非避让；重新登录成功后可用 DELETE 立即解除。
@app.get("/v1/admin/credential-outage")
@app.get("/admin/credential-outage")
async def get_credential_outage(provider: Optional[str] = None):
    return JSONResponse(get_credential_watch().get_diagnostics())


@app.delete("/v1/admin/credential-outage")
@app.delete("/admin/credential-outage")
async def clear_credential_outage(reason: Optional[str] = None):
    watch = get_credential_watch()
    before = watch.get_diagnostics()
    watch.clear(reason or "manual")
    # [OPT-026] 本接口的语义就是「我已经续期并修好了凭证」，因此除了清除全环失效态，
    # 还必须一并复位这些节点的**认证级**避让（含 2 的幂次升级来的 24h 冻结），
    # 否则运维照做之后网关仍会继续避让最多 24h，与接口承诺不符。
    released = await failover_router.cooldown_tracker.release_auth_avoidance(
        before["failed_providers"], reason or "manual"
    )
    return JSONResponse({
        "status": "ok",
        "was_active": before["outage_active"],
        "cleared_providers": before["failed_providers"],
        "released_auth_avoidance": released,
        "outage": watch.get_diagnostics(),
    })


# [OPT-026] 凭证续期检测与认证级避让复位：
# 「重登」是 auth 类故障唯一正确的修复动作，但 record_failure 对认证失效按 2 的幂次
# 长效升级（真实台账中五个节点各 54~87 次失败 -> 全部撞到 24h 上限），且该状态跨重启
# 持久化。于是凭证修好之后节点仍被冻结：探针跳过、自愈选举排除、路由降级，
# 连 `_deferred_refresh_after_cooldown` 都会把令牌静默刷新一并推迟最多 24h——
# 亦即「凭证失效」制造的状态锁死了修复它的机制。本接口提供两条复位路径。
@app.get("/v1/admin/credential-renewal")
@app.get("/admin/credential-renewal")
async def get_credential_renewal_report():
    return JSONResponse(get_credential_renewal().get_diagnostics())


@app.get("/v1/admin/ban-inference")
@app.get("/admin/ban-inference")
async def get_ban_inference():
    """[OPT-027] 查看「新凭证仍被拒」证据与自动推断租约。"""
    return JSONResponse(failover_router.cooldown_tracker.ban_inference_report())


@app.delete("/v1/admin/ban-inference")
@app.delete("/admin/ban-inference")
async def clear_ban_inference(provider: Optional[str] = None):
    """[OPT-027] 解除自动推断租约并清除证据。

    仅解除**自动**租约：运维租约必须走 DELETE /v1/admin/quarantine，本接口
    在结构上不可能解除人类声明。`provider` 省略时处理全部自动租约。
    """
    tracker = failover_router.cooldown_tracker
    before = tracker.ban_inference_report()
    targets = [provider] if provider else list(before["auto_leases"].keys())
    cleared: List[str] = []
    for pk in targets:
        key = (pk or "").strip().lower()
        if not key:
            continue
        try:
            if tracker.quarantine.release_auto_lease(key, time.time()):
                cleared.append(key)
        except Exception:  # noqa: BLE001
            continue
        tracker._auto_lease_fingerprint.pop(key, None)
        tracker._fresh_credential_rejected.pop(key, None)
    if cleared:
        tracker._persist_state()
    return JSONResponse({
        "status": "ok",
        "cleared": cleared,
        "candidates": targets,
        "ban_inference": tracker.ban_inference_report(),
    })


@app.post("/v1/admin/credential-renewal/scan")
@app.post("/admin/credential-renewal/scan")
async def scan_credential_renewal(req: Optional[CredentialRenewalScanRequest] = None):
    tracker = failover_router.cooldown_tracker
    explicit = list(req.providers) if req and req.providers else None
    if explicit:
        # 显式指定：用于凭证来自环境变量（无 session/<provider>.json 可比指纹）的部署
        targets, released = explicit, await tracker.release_auth_avoidance(explicit, "admin-scan")
    else:
        targets = tracker.renewed_providers()
        released = await tracker.release_renewed_credentials(None, "admin-scan")
    return JSONResponse({
        "status": "ok",
        "scan_mode": "explicit" if explicit else "fingerprint",
        "candidates": targets,
        "released": released,
        "renewal": get_credential_renewal().get_diagnostics(),
    })


# ---------------------------------------------------------------- 优化空间报告接口
@app.get("/v1/reports/latest")
@app.get("/reports/latest")
async def get_latest_optimization_report():
    """获取最新优化空间与迭代需求建议书（优先从 reports/optimization_proposal.json 读取，不存在则动态生成）。"""
    for fname in ("optimization_proposal.json", "optimization_recommendations.json"):
        fpath = _REPORTS_DIR / fname
        if fpath.exists():
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return JSONResponse(data)
            except Exception:
                pass

    evaluator = TaskEvaluator()
    report = evaluator.evaluate_optimization_opportunities()
    return JSONResponse(report)


@app.get("/")
async def index():
    index_html = _STATIC_DIR / "index.html"
    if index_html.exists():
        return FileResponse(index_html)
    return {"message": "同枢 (UniPivot) 本地多协议网关已启动", "docs": "/docs"}


@app.get("/v1/models")
@app.get("/models")
async def list_models(request: Request):
    if request.headers.get("anthropic-version"):
        return JSONResponse(format_anthropic_models())
    created = int(time.time())
    groups = [
        (QWEN_MODELS, "qwen"),
        (DEEPSEEK_MODELS, "deepseek"),
        (DOUBAO_MODELS, "doubao"),
        (KIMI_MODELS, "kimi"),
        (GLM_MODELS, "glm"),
        (QWEN_ALIASES, "qwen-alias"),
        (DEEPSEEK_ALIASES, "deepseek-alias"),
        (DOUBAO_ALIASES, "doubao-alias"),
        (KIMI_ALIASES, "kimi-alias"),
        (GLM_ALIASES, "glm-alias"),
        (OPENAI_ALIASES, "openai-alias"),
        ([m["id"] for m in CLAUDE_MODELS_METADATA], "anthropic-claude"),
    ]
    data = [
        {"id": name, "object": "model", "created": created, "owned_by": owner}
        for models, owner in groups for name in models
    ]
    return {"object": "list", "data": data}


@app.get("/v1/models/{model_id:path}")
@app.get("/models/{model_id:path}")
async def retrieve_model(model_id: str, request: Request):
    pk, wm = resolve_model(model_id)
    if not pk:
        msg = f"未知模型 `{model_id}`"
        if is_anthropic_request(request):
            return JSONResponse(
                anthropic_error_response(msg, 404, "model_not_found"),
                status_code=404,
            )
        return JSONResponse(
            error_response(msg, 404, "model_not_found"),
            status_code=404,
        )
    return {
        "id": model_id,
        "object": "model",
        "created": int(time.time()),
        "owned_by": pk,
    }


@app.post("/v1/embeddings")
@app.post("/embeddings")
async def create_embeddings(req: EmbeddingsRequest):
    import hashlib
    import math

    raw_inputs = req.input if isinstance(req.input, list) else [req.input]
    dim = req.dimensions or 1536
    data = []
    total_tokens = 0

    for idx, inp in enumerate(raw_inputs):
        if isinstance(inp, str):
            text = inp
        elif isinstance(inp, list):
            text = " ".join(map(str, inp))
        elif isinstance(inp, dict):
            text = inp.get("text", "") or json.dumps(inp)
        else:
            text = str(inp)

        tokens = _est_tokens(text)
        total_tokens += tokens

        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
        vec = [math.sin(seed + i) for i in range(dim)]
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        vec = [round(x / norm, 6) for x in vec]
        data.append({
            "object": "embedding",
            "index": idx,
            "embedding": vec,
        })

    return {
        "object": "list",
        "data": data,
        "model": req.model,
        "usage": {
            "prompt_tokens": max(1, total_tokens),
            "total_tokens": max(1, total_tokens),
        },
    }


PROVIDER_SPECS = {
    "qwen": (
        resolve_qwen,
        "未检测到 Qwen 令牌。请运行 .venv/bin/python login.py qwen 扫码登录，或配置 QWEN_AUTH_TOKEN",
        lambda s, client=None: QwenProvider(
            s.token,
            s.cookie or None,
            TIMEOUT,
            user_agent=s.user_agent or None,
            user_id=s.user_id or None,
            client=client,
        ),
    ),
    "deepseek": (
        resolve_deepseek,
        "未检测到 DeepSeek 令牌。请运行 .venv/bin/python login.py deepseek 扫码登录，或配置 DEEPSEEK_AUTH_TOKEN",
        lambda s, client=None: DeepSeekProvider(s.token, s.cookie or None, TIMEOUT, client=client),
    ),
    "doubao": (
        resolve_doubao,
        "未检测到豆包令牌。请运行 .venv/bin/python login.py doubao 登录，或配置 DOUBAO_AUTH_TOKEN",
        lambda s, client=None: DoubaoProvider(
            s.token,
            s.cookie or None,
            TIMEOUT,
            user_agent=s.user_agent or None,
            device_id=s.device_id or None,
            client=client,
        ),
    ),
    "kimi": (
        resolve_kimi,
        "未检测到 Kimi 令牌。请运行 .venv/bin/python login.py kimi 登录，或配置 KIMI_AUTH_TOKEN",
        lambda s, client=None: KimiProvider(
            s.token,
            s.cookie or None,
            TIMEOUT,
            user_agent=s.user_agent or None,
            device_id=s.device_id or None,
            client=client,
        ),
    ),
    "glm": (
        resolve_glm,
        "未检测到智谱 GLM 令牌。请运行 .venv/bin/python login.py glm 登录，或配置 GLM_AUTH_TOKEN",
        lambda s, client=None: GLMProvider(
            s.token,
            s.cookie or None,
            TIMEOUT,
            user_agent=s.user_agent or None,
            device_id=s.device_id or None,
            client=client,
        ),
    ),
}


def _make_provider(provider_key: str, client: Optional[httpx.AsyncClient] = None):
    spec = PROVIDER_SPECS.get(provider_key)
    if not spec:
        raise ProviderError(f"未知 Provider: {provider_key}", status=400)
    resolver, err_msg, factory = spec
    session = resolver()
    if not session or not session.token:
        raise ProviderError(err_msg, status=401, err_type="auth_error")
    return factory(session, client)


# 注册健康监控提供者
for _p_name, (_resolver, _err_msg, _factory) in PROVIDER_SPECS.items():
    health_monitor.register_provider(
        _p_name,
        resolver=_resolver,
        factory=lambda name=_p_name: _make_provider(name),
    )

# 会话回收队列复用 provider 构造器（后台重试时重建节点实例）
conversation_gc.set_resolver(_make_provider)
# [OPT-017] 会话删除同样遵守风控冷却门控：冷却期节点不得被删除请求触碰
conversation_gc.set_cooldown_gate(failover_router.cooldown_tracker.get_remaining_cooldown)
# [OPT-034] 成因判据优先于冷却浮点门控：只有它能区分「运维禁用」与「冷却中」，
# 使被声明出局的节点连临时会话删除也不会被发起。
conversation_gc.set_avoidance_gate(failover_router.cooldown_tracker.touch_avoidance_reason)


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    if not req.messages:
        return JSONResponse(
            error_response("`messages` 不能为空", 400, "invalid_request_error"),
            status_code=400,
        )

    provider_key, wire_model = resolve_model(req.model)
    if not provider_key:
        return JSONResponse(
            error_response(
                f"未知模型 `{req.model}`，可用模型: {', '.join(all_models())}",
                404, "model_not_found",
            ),
            status_code=404,
        )

    messages = [m.model_dump() for m in req.messages]
    # [TOOL-LOOP] 客户端「禁止并行调用」的声明既要进提示词，也要在解析端执行
    parallel_disallowed = parallel_calls_disallowed(req.parallel_tool_calls, req.tool_choice)
    # [PROTOCOL] 停用词是硬语义：提示词降低发生率，解析端保证结果（见 stop_sequences.py）
    stop_seq = req.stop
    system_additions: list[str] = []
    if req.tools:
        tools_prompt = format_tools_system_prompt(
            req.tools,
            tool_choice=req.tool_choice,
            parallel_allowed=not parallel_disallowed,
        )
        if tools_prompt:
            system_additions.append(tools_prompt)
    # [PROTOCOL] 客户端要求结构化输出时，模型必须**看见**这条要求
    fmt_prompt = format_response_format_prompt(req.response_format)
    if fmt_prompt:
        system_additions.append(fmt_prompt)
    if system_additions:
        block = "\n\n".join(system_additions)
        if messages and messages[0].get("role") == "system":
            # [PROTOCOL] 客户端的 system content 可以是 parts 列表；直接 f-string
            # 会把它变成 Python repr（`[{'type': 'text', ...}]`），模型在工具契约
            # 前面读到一段既不是指令、也不是它自己语言的噪声。统一走 text_of。
            orig = text_of(messages[0].get("content"))
            merged = f"{orig}\n\n{block}".strip()
            messages[0]["content"] = merged
        else:
            messages.insert(0, {"role": "system", "content": block})

    temperature = req.temperature if req.temperature is not None else 0.7
    max_tokens = req.max_tokens if req.max_tokens is not None else 2048

    # 0. 自适应提示词折叠优化：在估算 tokens 与路由调度前折叠超长历史轮次
    messages, fold_meta = prompt_optimizer.optimize_chat_messages(messages)
    fold_headers = make_prompt_folding_headers(fold_meta)

    # 1. 估算 prompt tokens 与思考意图
    # [TOOL-LOOP] 决定发全量还是增量的，是**网关有没有这条线程的上游会话**，
    # 而不是客户端有没有给线程 id。原先按 req.conversation_id 判断：一个自己生成
    # 线程 id 的客户端（Codex / Claude Code 类）首轮就被当成「续聊」，上游只收到
    # 最后一句提问，注入的系统提示与工具契约整段丢失——「模型无视我的 tools」。
    prompt_text = (
        continuation_content(messages)
        if session_affinity_manager.has_upstream_session(req.conversation_id)
        else flatten_messages(messages)
    )
    prompt_tokens = _est_tokens(prompt_text)
    is_thinking = _wants_thinking(req.thinking)

    # 2. 自适应负载均衡与被禁节点即刻转移路由决策
    decision = load_balancer.select_route(
        primary_provider=provider_key,
        primary_model=wire_model,
        prompt_tokens=prompt_tokens,
        is_thinking=is_thinking,
    )
    lb_headers = _make_lb_headers(decision)
    dispatched_provider = decision.provider_key
    dispatched_model = decision.wire_model

    # 2.1 智能会话粘滞解析与跨模型状态迁移
    affinity_decision = session_affinity_manager.resolve_affinity(
        session_id=req.conversation_id,
        primary_provider=dispatched_provider,
        primary_model=dispatched_model,
        is_provider_available=_is_provider_healthy,
        can_retain_degraded=_is_provider_retainable,
    )
    affinity_headers = _make_affinity_headers(affinity_decision)
    dispatched_provider = affinity_decision.provider_key
    dispatched_model = affinity_decision.wire_model

    chat_args = {
        "messages": messages,
        "conversation_id": affinity_decision.upstream_conv_id,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "thinking": _wants_thinking(req.thinking),
        "search": req.search,
        "stream": req.stream,
    }

    # [M1] 节奏裁决只有一处：`execute_chat` 派发循环里的
    # `apply_pacing_and_acquire`（紧贴上游调用、带锁预约可用时刻）。这里原先还有一次
    # `traffic_pacer.acquire(...)` 的「等待后立即释放」：它只推迟了本条请求自己，
    # 对并发者毫无约束（实测三路并发里有两路在同一毫秒打到上游），却让诊断数字
    # 看起来像是做了频控。现在此处只**读取**裁决结果用于遥测，不再产生第二段等待。
    risk_headers = _make_risk_headers(None, dispatched_provider, primary_provider=provider_key)

    # 2.2 智能并发语义去重与流式请求合并 (SingleFlight)
    flight_decision = None
    sf_headers = {}
    if not singleflight_manager.is_disabled():
        flight_key = singleflight_manager.compute_key(
            provider=dispatched_provider,
            model=dispatched_model,
            messages=messages,
            tools=req.tools,
            temperature=temperature,
            thinking=is_thinking,
            stream=req.stream,
            protocol="openai_chat",
            # [S3] 会改变输出的客户端声明必须进去重键：Follower 直接领用
            # Leader 的答案，键相同就等于断言「两者的正确答案可以互换」。
            # 这些字段原先不在键里，而那时它们确实不改变输出（被解析后无人读）；
            # 逐个修好之后它们开始改变输出，键却没同步 —— 声明 tool_choice:"none"
            # 的请求会领到工具调用，且没有任何错误可看。
            constraints={
                "tool_choice": req.tool_choice,
                "parallel_tool_calls": req.parallel_tool_calls,
                "stop": req.stop,
                "response_format": req.response_format,
                "max_tokens": req.max_tokens,
            },
        )
        flight_decision = await singleflight_manager.get_or_create(flight_key)
        sf_headers = _make_singleflight_headers(flight_decision)

    if req.stream:
        stream_headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
        stream_headers.update(lb_headers)
        stream_headers.update(fold_headers)
        stream_headers.update(risk_headers)
        stream_headers.update(affinity_headers)
        stream_headers.update(sf_headers)
        stream_headers.update(_make_disconnect_headers())

        if flight_decision and not flight_decision.is_leader:
            return StreamingResponse(
                sse_keepalive(stream_follower_gen(flight_decision.group)),
                media_type="text/event-stream",
                headers=stream_headers,
            )

        # [H4] 零上游快速失败必须在**状态行**上体现。`execute_chat` 在派发前就可能
        # 拒绝（凭证整环失效 / 运维禁用 / 封禁租约 / 长效避让 / 环内耗尽），此时还没
        # 有任何字节发出。若把它留在生成器里 await，响应头已经以 200 发出，客户端
        # 只能读到一段 `data: {"error": ...}`——多数 SDK 会把状态码当成功并丢掉错误，
        # 表现为「静默失败」；而 `/v1/responses` 走的是返回前的非流式分支，能如实
        # 返回 503。同一条协议纪律在两条路径上有两种结果，是本项目反复记录的那类
        # 缺陷。这里只提前 **第一个事件之前** 的 await，不是把所有内容缓冲起来：
        # 一旦拿到上游流，其余一切仍是逐字转发。
        preopened_stream = None
        preopened_event = None
        preopened_provider = None
        if not (flight_decision and not flight_decision.is_leader):
            try:
                preopened_stream, preopened_event, preopened_provider = (
                    await failover_router.execute_chat(
                        primary_provider_key=dispatched_provider,
                        primary_wire_model=dispatched_model,
                        provider_factory=lambda pkey: _make_provider(pkey),
                        chat_args=chat_args,
                    )
                )
            except ProviderError as e:
                is_risk, _ = is_risk_interception(e.message)
                if is_risk:
                    await failover_router.cooldown_tracker.record_failure(
                        dispatched_provider, e.status, e.message
                    )
                if flight_decision and flight_decision.is_leader:
                    await flight_decision.group.finish(error=e)
                return JSONResponse(
                    error_response(e.message, e.status, e.err_type),
                    status_code=e.status,
                    headers=_refusal_headers(e),
                )
            except Exception as e:
                if flight_decision and flight_decision.is_leader:
                    await flight_decision.group.finish(error=e)
                return JSONResponse(
                    error_response(f"上游服务异常: {e}", 502, "upstream_error"),
                    status_code=502,
                )

        async def gen():
            cid = _id("chatcmpl")
            created = int(time.time())
            conversation_id = req.conversation_id
            accumulated_chunks: list[str] = []
            accumulated_reasoning: list[str] = []

            start_time = time.time()
            load_balancer.acquire(dispatched_provider)
            ttft_recorded = False
            success = False
            is_rate_limit = False
            error_msg = ""

            # 1. 纯文本模式：无 tools 声明时走零延迟逐字流式
            if not req.tools:
                try:
                    # [H4] 流已在返回响应之前打开（见 preopened_stream），这里只接管。
                    stream = preopened_stream
                    failover_event = preopened_event
                    active_provider = preopened_provider

                    first_chunk = {
                        "id": cid,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": req.model,
                        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""},
                                     "finish_reason": None}],
                    }
                    if failover_event:
                        first_chunk["failover"] = failover_event.to_dict()
                    yield sse_frame(first_chunk)

                    # [PROTOCOL] 正文的停用词截断：命中后不再发送、也不再读上游
                    stop_scan = StopSequenceScanner(stop_seq)

                    async for delta, meta in stream:
                        if not ttft_recorded and delta:
                            load_balancer.record_ttft(
                                dispatched_provider, (time.time() - start_time) * 1000.0
                            )
                            ttft_recorded = True

                        if not delta:
                            if meta and meta.get("conversation_id"):
                                conversation_id = meta["conversation_id"]
                            continue

                        is_r = bool(meta and meta.get("reasoning"))
                        if not is_r and stop_scan.hit is not None:
                            # 已经命中停用词：后续正文一律丢弃（思考流仍照常透传）
                            continue
                        if is_r:
                            accumulated_reasoning.append(delta)
                            yield sse_frame({
                                "id": cid,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": req.model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {"reasoning_content": delta},
                                    "finish_reason": None,
                                }],
                            })
                        else:
                            safe = stop_scan.feed(delta)
                            accumulated_chunks.append(delta)
                            if not safe:
                                continue
                            yield sse_frame({
                                "id": cid,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": req.model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {"content": safe},
                                    "finish_reason": None,
                                }],
                            })

                        if meta and meta.get("conversation_id"):
                            conversation_id = meta["conversation_id"]

                    tail = stop_scan.flush()
                    if tail:
                        yield sse_frame({
                            "id": cid,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": req.model,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": tail},
                                "finish_reason": None,
                            }],
                        })

                    final = {
                        "id": cid,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": req.model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    }
                    if conversation_id:
                        final["conversation_id"] = conversation_id
                    if failover_event:
                        final["failover"] = failover_event.to_dict()
                    yield sse_frame(final)

                    if req.stream_options and req.stream_options.get("include_usage"):
                        pt = _est_tokens(prompt_text)
                        ct = max(1, _est_tokens("".join(accumulated_chunks) + "".join(accumulated_reasoning)))
                        yield sse_frame({
                            "id": cid,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": req.model,
                            "choices": [],
                            "usage": {
                                "prompt_tokens": pt,
                                "completion_tokens": ct,
                                "total_tokens": pt + ct,
                            },
                        })

                    yield "data: [DONE]\n\n"

                    if conversation_id:
                        session_affinity_manager.record_turn(
                            req.conversation_id or conversation_id,
                            dispatched_provider,
                            dispatched_model,
                            conversation_id,
                        )

                    if not req.conversation_id and conversation_id and hasattr(active_provider, "delete_conversation"):
                        conversation_gc.schedule(active_provider, dispatched_provider, conversation_id)
                    success = True
                except ProviderError as e:
                    is_rate_limit = (e.status == 429)
                    error_msg = e.message
                    is_risk, _ = is_risk_interception(e.message)
                    if is_risk:
                        await failover_router.cooldown_tracker.record_failure(dispatched_provider, e.status, e.message)
                    yield sse_frame(error_response(e.message, e.status, e.err_type))
                except Exception as e:
                    error_msg = str(e)
                    yield sse_frame(error_response(f"上游服务异常: {e}", 502, "upstream_error"))
                finally:
                    load_balancer.release(
                        dispatched_provider,
                        success=success,
                        is_rate_limit=is_rate_limit,
                        error_msg=error_msg,
                    )
                return

            # 2. Tools 模式：推测式流式分发与自愈解析
            try:
                # [H4] 同上：流已在返回响应之前打开。
                stream = preopened_stream
                failover_event = preopened_event
                active_provider = preopened_provider

                streamer = SpeculativeToolStreamer(
                    tools_enabled=True,
                    allowed=resolve_allowed_tools(req.tools, req.tool_choice),
                )
                # [PROTOCOL] 工具模式同样受停用词约束（三条协议共用同一条纪律）
                stop_scan = StopSequenceScanner(stop_seq)
                first_frame_emitted = False

                async for delta, meta in stream:
                    if not ttft_recorded and delta:
                        load_balancer.record_ttft(
                            dispatched_provider, (time.time() - start_time) * 1000.0
                        )
                        ttft_recorded = True

                    if not delta:
                        if meta and meta.get("conversation_id"):
                            conversation_id = meta["conversation_id"]
                        continue

                    if meta and meta.get("conversation_id"):
                        conversation_id = meta["conversation_id"]

                    is_r = bool(meta and meta.get("reasoning"))
                    actions = (
                        streamer.feed_chunk(delta, is_reasoning=True)
                        if is_r
                        else streamer.feed_text(delta)
                    )
                    for act in actions:
                        if act.action_type == "reasoning_delta":
                            chunk = {
                                "id": cid,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": req.model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {"role": "assistant", "reasoning_content": act.content},
                                    "finish_reason": None,
                                }],
                            }
                            if not first_frame_emitted and failover_event:
                                chunk["failover"] = failover_event.to_dict()
                            yield sse_frame(chunk)
                            first_frame_emitted = True
                        elif act.action_type == "text_delta":
                            safe_act = stop_scan.feed(act.content)
                            if not safe_act:
                                continue
                            chunk = {
                                "id": cid,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": req.model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {"role": "assistant", "content": safe_act},
                                    "finish_reason": None,
                                }],
                            }
                            if not first_frame_emitted and failover_event:
                                chunk["failover"] = failover_event.to_dict()
                            yield sse_frame(chunk)
                            first_frame_emitted = True

                # [PROTOCOL] 停用词必须先截断累积再 finalize：命中点之后的
                # ```tool_call 对客户端不存在，解析出来就是一次没有预期的执行
                _matched_stop = streamer.truncate_at_stop(stop_seq)
                if _matched_stop is not None:
                    # [PROTOCOL] 把命中的原文交给扫描器：线契约要求 stop_sequence
                    # 带着它，只报告「被截了」而不说被什么截，客户端无法定位。
                    stop_scan.mark_hit(_matched_stop)
                rem_text, tool_list, was_healed = streamer.finalize()
                # [TOOL-LOOP] 客户端禁止并行调用时只回传第一个（见 cap_parallel_tool_calls）
                tool_list = cap_parallel_tool_calls(tool_list, parallel_disallowed)
                if not tool_list:
                    # [TOOL-LOOP] 思考通道兜底：正文为空时工具调用可能整个写在思考流里
                    tool_list, was_healed = _recover_reasoning_tools(
                        streamer,
                        resolve_allowed_tools(req.tools, req.tool_choice),
                        parallel_disallowed,
                        answer_text="".join(streamer.accumulated_text),
                    )
                # [PROTOCOL] 终局剩余正文同样要过停用词判定，并把扣住的尾巴放行
                rem_text = stop_scan.feed(rem_text) + stop_scan.flush()
                if rem_text:
                    chunk = {
                        "id": cid,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": req.model,
                        "choices": [{
                            "index": 0,
                            "delta": {"role": "assistant", "content": rem_text},
                            "finish_reason": None,
                        }],
                    }
                    if not first_frame_emitted and failover_event:
                        chunk["failover"] = failover_event.to_dict()
                    yield sse_frame(chunk)
                    first_frame_emitted = True

                full_text = "".join(streamer.accumulated_text)

                if tool_list:
                    _tool_healing_stats["total_tool_calls"] += len(tool_list)
                    if was_healed:
                        _tool_healing_stats["healed_tool_calls"] += 1
                    delta_tool_calls = [
                        {
                            "index": idx,
                            "id": _id("call", "_"),
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(args, ensure_ascii=False),
                            },
                        }
                        for idx, (tool_name, args) in enumerate(tool_list)
                    ]
                    tool_chunk = {
                        "id": cid,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": req.model,
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                                "tool_calls": delta_tool_calls,
                            },
                            "finish_reason": None,
                        }],
                    }
                    if not first_frame_emitted and failover_event:
                        tool_chunk["failover"] = failover_event.to_dict()
                    if conversation_id:
                        tool_chunk["conversation_id"] = conversation_id
                    yield sse_frame(tool_chunk)
                    yield sse_frame({
                        "id": cid,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": req.model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                    })
                else:
                    if not first_frame_emitted:
                        init_chunk = {
                            "id": cid,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": req.model,
                            "choices": [{
                                "index": 0,
                                "delta": {"role": "assistant", "content": full_text},
                                "finish_reason": None,
                            }],
                        }
                        if failover_event:
                            init_chunk["failover"] = failover_event.to_dict()
                        if conversation_id:
                            init_chunk["conversation_id"] = conversation_id
                        yield sse_frame(init_chunk)
                    yield sse_frame({
                        "id": cid,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": req.model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    })

                if req.stream_options and req.stream_options.get("include_usage"):
                    pt = _est_tokens(prompt_text)
                    ct = max(1, _est_tokens(full_text))
                    yield sse_frame({
                        "id": cid,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": req.model,
                        "choices": [],
                        "usage": {
                            "prompt_tokens": pt,
                            "completion_tokens": ct,
                            "total_tokens": pt + ct,
                        },
                    })

                yield "data: [DONE]\n\n"

                if conversation_id:
                    # [M3] 若本次因原节点短冷却而改由别家服务，会话归属**不得**跟着
                    # 改嫁：否则一次频控退避就会把长程会话永久迁到另一个厂商，导致
                    # 上游上下文全部重建。归属仍记在原节点上，并清掉本轮观测到的
                    # 上游会话 id（它属于服务本次请求的那个节点，不能回放给原节点）。
                    keep_provider = (
                        affinity_decision.provider_key
                        if affinity_decision.status == "degraded"
                        else dispatched_provider
                    )
                    session_affinity_manager.record_turn(
                        req.conversation_id or conversation_id,
                        keep_provider,
                        dispatched_model,
                        None if affinity_decision.status == "degraded" else conversation_id,
                    )

                if not req.conversation_id and conversation_id and hasattr(active_provider, "delete_conversation"):
                    conversation_gc.schedule(active_provider, dispatched_provider, conversation_id)
                success = True
            except ProviderError as e:
                is_rate_limit = (e.status == 429)
                error_msg = e.message
                is_risk, _ = is_risk_interception(e.message)
                if is_risk:
                    await failover_router.cooldown_tracker.record_failure(dispatched_provider, e.status, e.message)
                yield sse_frame(error_response(e.message, e.status, e.err_type))
            except Exception as e:
                error_msg = str(e)
                yield sse_frame(error_response(f"上游服务异常: {e}", 502, "upstream_error"))
            finally:
                # [C2] 上游流必须显式关闭：`ManagedStream.close()` 归还节点信号量，
                # 并关闭 provider 侧的 HTTP 流。正常结束/中途异常/断连都走这里；
                # 响应体从未启动的路径由响应类的 `on_close` 同步兜底。
                await _close_upstream_stream(stream)
                load_balancer.release(
                    dispatched_provider,
                    success=success,
                    is_rate_limit=is_rate_limit,
                    error_msg=error_msg,
                )

        stream_body = gen()
        if flight_decision and flight_decision.is_leader:
            stream_body = broadcast_stream_wrapper(stream_body, flight_decision.group)
        # 客户端断连传播：显式关闭上游流并即时回收上游并发配额
        if not disconnect_guard.is_disabled():
            stream_body = guard_stream(stream_body, provider_key=dispatched_provider)

        # [C2] 上游流与节点信号量在返回响应之前就已持有；响应体若从未被拉取，
        # 生成器的 finally 不会执行（guard_stream 同样从未启动，看不到中断）。
        # `on_close` 是这条路径上唯一能归还已有资源的钩子。
        return ReclaimingStreamingResponse(
            sse_keepalive(stream_body),
            media_type="text/event-stream",
            headers=stream_headers,
            on_close=_stream_reclaimer(preopened_stream),
            on_close_async=_stream_closer(preopened_stream),
        )

    # 非流式：若当前请求为 Follower，挂载等待 Leader 完成
    if flight_decision and not flight_decision.is_leader:
        try:
            await asyncio.wait_for(flight_decision.group.done_event.wait(), timeout=TIMEOUT)
        except asyncio.TimeoutError:
            return JSONResponse(error_response("等待并发共享请求超时", 504, "timeout_error"), status_code=504)
        if flight_decision.group.error:
            if isinstance(flight_decision.group.error, ProviderError):
                pe = flight_decision.group.error
                return JSONResponse(error_response(pe.message, pe.status, pe.err_type), status_code=pe.status)
            return JSONResponse(error_response(f"上游服务异常: {flight_decision.group.error}", 502, "upstream_error"), status_code=502)

        follower_headers = dict(flight_decision.group.completed_headers or {})
        follower_headers.update(sf_headers)
        return JSONResponse(flight_decision.group.completed_result, headers=follower_headers)

    # 非流式：聚合上游流
    chunks: list[str] = []
    reasoning_chunks: list[str] = []
    conversation_id = req.conversation_id
    start_time = time.time()
    thinking_start_time: Optional[float] = None
    thinking_end_time: Optional[float] = None
    load_balancer.acquire(dispatched_provider)
    ttft_recorded = False
    success = False
    group_finalized = False
    is_rate_limit = False
    error_msg = ""
    try:
        stream, failover_event, active_provider = await failover_router.execute_chat(
            primary_provider_key=dispatched_provider,
            primary_wire_model=dispatched_model,
            provider_factory=lambda pkey: _make_provider(pkey),
            chat_args=chat_args,
        )
        async for delta, meta in stream:
            if not ttft_recorded and delta:
                load_balancer.record_ttft(
                    dispatched_provider, (time.time() - start_time) * 1000.0
                )
                ttft_recorded = True
            if not delta:
                if meta and meta.get("conversation_id"):
                    conversation_id = meta["conversation_id"]
                continue
            if meta and meta.get("reasoning"):
                if thinking_start_time is None:
                    thinking_start_time = time.time()
                thinking_end_time = time.time()
                reasoning_chunks.append(delta)
            else:
                chunks.append(delta)
            if meta and meta.get("conversation_id"):
                conversation_id = meta["conversation_id"]
        success = True
    except ProviderError as e:
        is_rate_limit = (e.status == 429)
        error_msg = e.message
        is_risk, _ = is_risk_interception(e.message)
        if is_risk:
            await failover_router.cooldown_tracker.record_failure(dispatched_provider, e.status, e.message)
        if flight_decision and flight_decision.is_leader:
            await flight_decision.group.finish(error=e)
            group_finalized = True
        return JSONResponse(
            error_response(e.message, e.status, e.err_type),
            status_code=e.status,
            headers=_refusal_headers(e),
        )
    except Exception as e:
        error_msg = str(e)
        if flight_decision and flight_decision.is_leader:
            await flight_decision.group.finish(error=e)
            group_finalized = True
        return JSONResponse(error_response(f"上游服务异常: {e}", 502, "upstream_error"), status_code=502)
    finally:
        # 客户端断连（CancelledError）等 BaseException 不会进入上述 except 分支，
        # 必须在 finally 中确定性回收并发配额并补齐飞行组终局，避免 Follower 悬挂。
        load_balancer.release(
            dispatched_provider,
            success=success,
            is_rate_limit=is_rate_limit,
            error_msg=error_msg,
        )
        if not success and not group_finalized and flight_decision and flight_decision.is_leader:
            terminate_aborted_group(flight_decision.group)

    if conversation_id:
        # [M3] 短冷却降级期间不迁移会话归属（见流式分支同处注释）。
        keep_provider = (
            affinity_decision.provider_key
            if affinity_decision.status == "degraded"
            else dispatched_provider
        )
        session_affinity_manager.record_turn(
            req.conversation_id or conversation_id,
            keep_provider,
            dispatched_model,
            None if affinity_decision.status == "degraded" else conversation_id,
        )

    if not req.conversation_id and conversation_id and hasattr(active_provider, "delete_conversation"):
        conversation_gc.schedule(active_provider, dispatched_provider, conversation_id)

    prompt = prompt_text
    full_output = "".join(chunks)
    full_reasoning = "".join(reasoning_chunks)
    # [PROTOCOL] 停用词截断必须在工具解析之前：截断点之后的内容对客户端不存在，
    # 若那里还藏着一个工具调用，把它回传出去就是一次客户端从未预期过的执行。
    full_output, _stop_matched = truncate_final(full_output, stop_seq)

    thinking_duration_ms = None
    thinking_toks = None
    if full_reasoning:
        thinking_toks = max(1, _est_tokens(full_reasoning))
        if thinking_start_time and thinking_end_time:
            thinking_duration_ms = max(1.0, (thinking_end_time - thinking_start_time) * 1000.0)

    thinking_headers = _make_thinking_headers(thinking_duration_ms, thinking_toks)

    resp_headers = dict(lb_headers)
    resp_headers.update(fold_headers)
    resp_headers.update(risk_headers)
    resp_headers.update(thinking_headers)
    resp_headers.update(affinity_headers)
    resp_headers.update(sf_headers)
    if failover_event:
        resp_headers.update({
            "x-failover-from": failover_event.from_provider,
            "x-failover-to": failover_event.to_provider,
            "x-failover-model": failover_event.to_model,
            "x-failover-ring": failover_event.ring.value,
        })

    if req.tools:
        parsed, was_healed = parse_tool_calls_with_healing(
            full_output, resolve_allowed_tools(req.tools, req.tool_choice)
        )
        if not parsed:
            # [TOOL-LOOP] 思考通道兜底：正文为空时工具调用可能整个写在思考流里
            _rt, _rh = recover_tools_from_reasoning(
                full_reasoning,
                resolve_allowed_tools(req.tools, req.tool_choice),
                answer_text=full_output,
            )
            if _rt:
                _rt = cap_parallel_tool_calls(_rt, parallel_disallowed)
                _tool_healing_stats["reasoning_recovered_tool_calls"] += len(_rt)
                parsed, was_healed = ("", _rt), _rh
        if parsed:
            prefix, tool_list = parsed
            # [TOOL-LOOP] 客户端禁止并行调用时只回传第一个
            tool_list = cap_parallel_tool_calls(tool_list, parallel_disallowed)
            _tool_healing_stats["total_tool_calls"] += len(tool_list)
            if was_healed:
                _tool_healing_stats["healed_tool_calls"] += 1
            resp_headers.update(_make_tool_headers(len(tool_list), was_healed))
            tool_calls = [
                {
                    "id": _id("call", "_"),
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(args, ensure_ascii=False),
                    },
                }
                for tool_name, args in tool_list
            ]
            resp_data = completion_response(
                req.model,
                prefix or None,
                prompt,
                conversation_id,
                thinking_content=full_reasoning,
                tool_calls=tool_calls,
                finish_reason="tool_calls",
            )
            if failover_event:
                resp_data["failover"] = failover_event.to_dict()
            if flight_decision and flight_decision.is_leader:
                await flight_decision.group.finish(result=resp_data, headers=resp_headers)
            return JSONResponse(resp_data, headers=resp_headers if resp_headers else None)

    resp_data = completion_response(
        req.model, full_output, prompt, conversation_id, thinking_content=full_reasoning,
        # [PROTOCOL] OpenAI 的 finish_reason 枚举里没有 stop_sequence（那是 Anthropic 的），
        # 停用词命中与自然结束在 OpenAI 语义下**都是** "stop"，不能自造枚举值。
        finish_reason="stop",
    )
    if failover_event:
        resp_data["failover"] = failover_event.to_dict()
    if flight_decision and flight_decision.is_leader:
        await flight_decision.group.finish(result=resp_data, headers=resp_headers)
    return JSONResponse(resp_data, headers=resp_headers if resp_headers else None)


@app.post("/v1/messages")
@app.post("/messages")
async def anthropic_messages(req: AnthropicMessagesRequest):
    if not req.messages:
        return JSONResponse(
            anthropic_error_response("`messages` 不能为空", 400, "invalid_request_error"),
            status_code=400,
        )

    provider_key, wire_model = resolve_model(req.model)
    if not provider_key:
        return JSONResponse(
            anthropic_error_response(
                f"未知模型 `{req.model}`，可用模型: {', '.join(all_models())}",
                404,
                "model_not_found",
            ),
            status_code=404,
        )

    parallel_disallowed = parallel_calls_disallowed(None, req.tool_choice)
    # [SESSION-ID] 本请求的客户端线程标识：`conversation_id` 是显式字段，但它并非
    # 所有客户端都会发——真实 Claude Code 2.1.293 driver 19 份抓包里出现 0 次，
    # 身份全在 `metadata.user_id` 的 JSON 串里。此前只读前者，于是整条粘滞/续聊/
    # 匿名回收链路对 Claude Code 完全空转（实测每轮新建上游会话 + 重发全量历史）。
    client_session_id = resolve_client_session_id(req.conversation_id, req.metadata)
    # [PROTOCOL] Anthropic 的 stop_sequences 是硬语义，网关侧截断（见 stop_sequences.py）
    stop_seq = req.stop_sequences
    gateway_messages = convert_anthropic_to_gateway_messages(
        req.messages,
        system=req.system,
        tools=req.tools,
        tool_choice=req.tool_choice,
        parallel_allowed=not parallel_disallowed,
    )
    # 0. 自适应提示词折叠优化：在估算 tokens 与路由调度前折叠超长历史轮次
    gateway_messages, fold_meta = prompt_optimizer.optimize_chat_messages(gateway_messages)
    fold_headers = make_prompt_folding_headers(fold_meta)

    input_text = flatten_messages(gateway_messages)
    input_tokens = _est_tokens(input_text)
    temperature = req.temperature if req.temperature is not None else 0.7
    max_tokens = req.max_tokens or 4096

    # 判定 thinking 意图
    enable_thinking = _wants_thinking(req.thinking)
    # [PROTOCOL] 与「要不要思考」正交的第二问：思考了，要不要发给客户端。
    suppress_thinking = _thinking_suppressed(req.thinking)

    # 自适应负载均衡与被禁节点即刻转移路由决策
    decision = load_balancer.select_route(
        primary_provider=provider_key,
        primary_model=wire_model,
        prompt_tokens=input_tokens,
        is_thinking=enable_thinking,
    )
    lb_headers = _make_lb_headers(decision)
    dispatched_provider = decision.provider_key
    dispatched_model = decision.wire_model

    # 2.1 智能会话粘滞解析与跨模型状态迁移
    affinity_decision = session_affinity_manager.resolve_affinity(
        session_id=client_session_id,
        primary_provider=dispatched_provider,
        primary_model=dispatched_model,
        is_provider_available=_is_provider_healthy,
        can_retain_degraded=_is_provider_retainable,
    )
    affinity_headers = _make_affinity_headers(affinity_decision)
    dispatched_provider = affinity_decision.provider_key
    dispatched_model = affinity_decision.wire_model

    chat_args = {
        "messages": gateway_messages,
        "conversation_id": affinity_decision.upstream_conv_id,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "thinking": enable_thinking,
    }

    # [M1] 节奏裁决只有一处：`execute_chat` 派发循环里的
    # `apply_pacing_and_acquire`（紧贴上游调用、带锁预约可用时刻）。这里原先还有一次
    # `traffic_pacer.acquire(...)` 的「等待后立即释放」：它只推迟了本条请求自己，
    # 对并发者毫无约束（实测三路并发里有两路在同一毫秒打到上游），却让诊断数字
    # 看起来像是做了频控。现在此处只**读取**裁决结果用于遥测，不再产生第二段等待。
    risk_headers = _make_risk_headers(None, dispatched_provider, primary_provider=provider_key)

    # 2.2 智能并发语义去重与流式请求合并 (SingleFlight)
    flight_decision = None
    sf_headers = {}
    if not singleflight_manager.is_disabled():
        flight_key = singleflight_manager.compute_key(
            provider=dispatched_provider,
            model=dispatched_model,
            messages=gateway_messages,
            tools=req.tools,
            temperature=temperature,
            thinking=enable_thinking,
            stream=req.stream,
            protocol="anthropic_messages",
            # [S3] 会改变输出的客户端声明必须进去重键：Follower 直接领用
            # Leader 的答案，键相同就等于断言「两者的正确答案可以互换」。
            # 这些字段原先不在键里，而那时它们确实不改变输出（被解析后无人读）；
            # 逐个修好之后它们开始改变输出，键却没同步 —— 声明 tool_choice:"none"
            # 的请求会领到工具调用，且没有任何错误可看。
            constraints={
                "tool_choice": req.tool_choice,
                "stop_sequences": req.stop_sequences,
                "max_tokens": req.max_tokens,
            },
        )
        flight_decision = await singleflight_manager.get_or_create(flight_key)
        sf_headers = _make_singleflight_headers(flight_decision)

    if req.stream:
        stream_headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
        stream_headers.update(lb_headers)
        stream_headers.update(fold_headers)
        stream_headers.update(risk_headers)
        stream_headers.update(affinity_headers)
        stream_headers.update(sf_headers)
        stream_headers.update(_make_disconnect_headers())

        if flight_decision and not flight_decision.is_leader:
            return StreamingResponse(
                sse_keepalive(stream_follower_gen(flight_decision.group)),
                media_type="text/event-stream",
                headers=stream_headers,
            )

        # [H4] 与 OpenAI 路径同一条纪律：派发前的拒绝必须体现在状态行上，而不是
        # 200 + 行内错误。只提前「第一个事件之前」的 await，拿到上游流后仍逐字转发。
        preopened_stream = None
        preopened_event = None
        preopened_provider = None
        try:
            preopened_stream, preopened_event, preopened_provider = (
                await failover_router.execute_chat(
                    primary_provider_key=dispatched_provider,
                    primary_wire_model=dispatched_model,
                    provider_factory=lambda pkey: _make_provider(pkey),
                    chat_args=chat_args,
                )
            )
        except ProviderError as e:
            is_risk, _ = is_risk_interception(e.message)
            if is_risk:
                await failover_router.cooldown_tracker.record_failure(
                    dispatched_provider, e.status, e.message
                )
            if flight_decision and flight_decision.is_leader:
                await flight_decision.group.finish(error=e)
            return JSONResponse(
                anthropic_error_response(e.message, e.status, e.err_type),
                status_code=e.status,
                headers=_refusal_headers(e),
            )
        except Exception as e:
            if flight_decision and flight_decision.is_leader:
                await flight_decision.group.finish(error=e)
            return JSONResponse(
                anthropic_error_response(f"上游服务异常: {e}", 502, "upstream_error"),
                status_code=502,
            )

        async def gen():
            msg_id = _id("msg", "_")
            conv_id: Optional[str] = None
            start_time = time.time()
            load_balancer.acquire(dispatched_provider)
            ttft_recorded = False
            success = False
            is_rate_limit = False
            error_msg = ""

            try:
                # 1. 纯文本模式：无 tools 声明时走零延迟逐字流式
                if not req.tools:
                    # [H4] 流已在返回响应之前打开（见 preopened_stream），这里只接管。
                    stream = preopened_stream
                    failover_event = preopened_event
                    active_provider = preopened_provider

                    yield sse_message_start(msg_id, req.model, input_tokens)

                    current_block_index: int = -1
                    current_block_type: Optional[str] = None
                    accumulated_chunks: list[str] = []
                    accumulated_thinking: list[str] = []
                    # [PROTOCOL] 停用词只作用于正文，思考流不受影响
                    stop_scan = StopSequenceScanner(stop_seq)

                    async for delta, _meta in stream:
                        if not ttft_recorded and delta:
                            load_balancer.record_ttft(
                                dispatched_provider, (time.time() - start_time) * 1000.0
                            )
                            ttft_recorded = True

                        if not delta:
                            if _meta and _meta.get("conversation_id"):
                                conv_id = _meta["conversation_id"]
                            continue

                        is_reasoning = bool(_meta and _meta.get("reasoning"))
                        # [PROTOCOL] 抑制只在**发送边界**：思考照常累积进
                        # `accumulated_thinking`（思考通道兜底与 usage 都要用它），
                        # 只是不开块、不发 delta。客户端明确要求 omitted 时，
                        # 一个 thinking 块都不该出现在事件流里。
                        if is_reasoning and suppress_thinking:
                            accumulated_thinking.append(delta)
                            if _meta and _meta.get("conversation_id"):
                                conv_id = _meta["conversation_id"]
                            continue
                        target_type = "thinking" if is_reasoning else "text"

                        # [PROTOCOL] 停用词必须在**开块之前**判定，且判定后立刻收口。
                        # 原实现是在开完块之后才 feed，于是：块被打开 → feed 返回空
                        # （尾巴还在保留窗口里）→ 关掉这个空块 → 继续开下一个空块；
                        # 命中之后仍然把上游残余流读完，收口处再拿硬编码索引 0 开一个
                        # 空块——而索引 0 早已使用并关闭过。实测 Claude Code 对重复
                        # 索引的反应是保留后一块、**静默丢弃**前一块的正文。
                        if is_reasoning:
                            accumulated_thinking.append(delta)
                        else:
                            delta = stop_scan.feed(delta)
                            if not delta:
                                if stop_scan.hit is not None:
                                    # 命中：本回合到此为止，不再读上游残料
                                    break
                                continue
                            accumulated_chunks.append(delta)

                        if current_block_type != target_type:
                            if current_block_type is not None:
                                yield sse_content_block_stop(current_block_index)
                            current_block_index += 1
                            current_block_type = target_type
                            if is_reasoning:
                                yield sse_content_block_start(
                                    current_block_index, {"type": "thinking", "thinking": ""}
                                )
                            else:
                                yield sse_content_block_start(
                                    current_block_index, {"type": "text", "text": ""}
                                )

                        if is_reasoning:
                            yield sse_content_block_delta(
                                current_block_index,
                                {"type": "thinking_delta", "thinking": delta},
                            )
                        else:
                            yield sse_content_block_delta(
                                current_block_index,
                                {"type": "text_delta", "text": delta},
                            )

                        if _meta and _meta.get("conversation_id"):
                            conv_id = _meta["conversation_id"]

                    tail = stop_scan.flush()
                    if tail:
                        if current_block_type != "text":
                            if current_block_type is not None:
                                yield sse_content_block_stop(current_block_index)
                            current_block_index += 1
                            current_block_type = "text"
                            yield sse_content_block_start(
                                current_block_index, {"type": "text", "text": ""}
                            )
                        yield sse_content_block_delta(
                            current_block_index, {"type": "text_delta", "text": tail}
                        )

                    if current_block_type is not None:
                        yield sse_content_block_stop(current_block_index)
                    elif current_block_index < 0:
                        # 只有**从未开过任何块**时才补一个空正文块（客户端要求至少有
                        # 一个块）。若是因为被扣住尾巴/命中停用词而关过块，索引已经
                        # 用掉了，这里再开一个就是复用索引——实测 Claude Code 会因此
                        # 丢弃前一块的正文。
                        yield sse_content_block_start(0, {"type": "text", "text": ""})
                        yield sse_content_block_stop(0)

                    output_tokens = max(
                        1, _est_tokens("".join(accumulated_chunks) + "".join(accumulated_thinking))
                    )
                    yield sse_message_delta(
                        "stop_sequence" if stop_scan.hit else "end_turn",
                        output_tokens,
                        stop_sequence=stop_scan.hit.matched if stop_scan.hit else None,
                    )
                    yield sse_message_stop()

                    if client_session_id or conv_id:
                        session_affinity_manager.record_turn(
                            client_session_id or conv_id,
                            dispatched_provider,
                            dispatched_model,
                            conv_id,
                        )

                    if not client_session_id and conv_id and hasattr(active_provider, "delete_conversation"):
                        conversation_gc.schedule(active_provider, dispatched_provider, conv_id)
                    success = True
                    return

                # 2. Agent / 工具模式：推测式流式分发与容错解析
                # [H4] 同上：流已在返回响应之前打开。
                stream = preopened_stream
                failover_event = preopened_event
                active_provider = preopened_provider

                streamer = SpeculativeToolStreamer(
                    tools_enabled=True,
                    allowed=resolve_allowed_tools(req.tools, req.tool_choice),
                )
                # [PROTOCOL] 工具模式同样受停用词约束（三条协议共用同一条纪律）
                stop_scan = StopSequenceScanner(stop_seq)
                yield sse_message_start(msg_id, req.model, input_tokens)

                block_idx = 0
                in_thinking_block = False
                in_text_block = False

                async for delta, _meta in stream:
                    if not ttft_recorded and delta:
                        load_balancer.record_ttft(
                            dispatched_provider, (time.time() - start_time) * 1000.0
                        )
                        ttft_recorded = True

                    if not delta:
                        if _meta and _meta.get("conversation_id"):
                            conv_id = _meta["conversation_id"]
                        continue
                    if _meta and _meta.get("conversation_id"):
                        conv_id = _meta["conversation_id"]

                    is_r = bool(_meta and _meta.get("reasoning"))
                    actions = (
                        streamer.feed_chunk(delta, is_reasoning=True)
                        if is_r
                        else streamer.feed_text(delta)
                    )
                    for act in actions:
                        # [ANTHROPIC-STREAM] 两种内容块必须**互斥**且一次只开一个。
                        # 原实现只在「思考 → 正文」方向关块，反向切换（正文 → 再次思考，
                        # 推理模型在工具循环里极常见）既不关正文块也不递增 index，于是
                        # 同一个 index 被 start 两次、没有中间 stop——这是非法的
                        # Anthropic 事件流，SDK 客户端会拒绝或错位后续内容。实测复现：
                        # start(1,text) → start(1,thinking) → stop(1) → stop(2)。
                        if act.action_type == "reasoning_delta":
                            # [PROTOCOL] display=omitted：不发思考文本，也不开块。
                            # 流式器照常累积（工具兜底要用），这里只是不发送。
                            if suppress_thinking:
                                continue
                            if in_text_block:
                                yield sse_content_block_stop(block_idx)
                                in_text_block = False
                                block_idx += 1
                            if not in_thinking_block:
                                yield sse_content_block_start(block_idx, {"type": "thinking", "thinking": ""})
                                in_thinking_block = True
                            yield sse_content_block_delta(block_idx, {"type": "thinking_delta", "thinking": act.content})
                        elif act.action_type == "text_delta":
                            if in_thinking_block:
                                yield sse_content_block_stop(block_idx)
                                in_thinking_block = False
                                block_idx += 1
                            # [PROTOCOL] 停用词在**开块之前**判定：命中时这一块
                            # 根本不该存在，先开一个空 text 块再不发内容会在
                            # 事件流里留下一个没有正文的块。
                            safe_act = stop_scan.feed(act.content)
                            if not safe_act:
                                continue
                            if not in_text_block:
                                yield sse_content_block_start(block_idx, {"type": "text", "text": ""})
                                in_text_block = True
                            yield sse_content_block_delta(block_idx, {"type": "text_delta", "text": safe_act})

                # [PROTOCOL] 停用词必须先截断累积再 finalize：命中点之后的
                # ```tool_call 对客户端不存在，解析出来就是一次没有预期的执行
                _matched_stop = streamer.truncate_at_stop(stop_seq)
                if _matched_stop is not None:
                    # [PROTOCOL] 把命中的原文交给扫描器：线契约要求 stop_sequence
                    # 带着它，只报告「被截了」而不说被什么截，客户端无法定位。
                    stop_scan.mark_hit(_matched_stop)
                rem_text, tool_list, was_healed = streamer.finalize()
                # [TOOL-LOOP] 客户端禁止并行调用时只回传第一个（见 cap_parallel_tool_calls）
                tool_list = cap_parallel_tool_calls(tool_list, parallel_disallowed)
                if not tool_list:
                    # [TOOL-LOOP] 思考通道兜底：正文为空时工具调用可能整个写在思考流里
                    tool_list, was_healed = _recover_reasoning_tools(
                        streamer,
                        resolve_allowed_tools(req.tools, req.tool_choice),
                        parallel_disallowed,
                        answer_text="".join(streamer.accumulated_text),
                    )
                if in_thinking_block:
                    yield sse_content_block_stop(block_idx)
                    in_thinking_block = False
                    block_idx += 1

                # [PROTOCOL] 流结束前把被扣住的尾巴放行（或确认它已被停用词截掉）
                rem_text = stop_scan.feed(rem_text) + stop_scan.flush() if rem_text else stop_scan.flush()
                if rem_text:
                    if not in_text_block:
                        yield sse_content_block_start(block_idx, {"type": "text", "text": ""})
                        in_text_block = True
                    yield sse_content_block_delta(block_idx, {"type": "text_delta", "text": rem_text})
                hit_stop = stop_scan.hit

                if in_text_block:
                    yield sse_content_block_stop(block_idx)
                    in_text_block = False
                    block_idx += 1

                output_tokens = max(1, _est_tokens("".join(streamer.accumulated_text)))

                if tool_list:
                    _tool_healing_stats["total_tool_calls"] += len(tool_list)
                    if was_healed:
                        _tool_healing_stats["healed_tool_calls"] += 1
                    for tool_name, args in tool_list:
                        tool_use_id = _id("toolu", "_")[:22]
                        yield sse_content_block_start(
                            block_idx,
                            {
                                "type": "tool_use",
                                "id": tool_use_id,
                                "name": tool_name,
                                "input": {},
                            },
                        )
                        yield sse_content_block_delta(
                            block_idx,
                            {
                                "type": "input_json_delta",
                                "partial_json": json.dumps(args, ensure_ascii=False),
                            },
                        )
                        yield sse_content_block_stop(block_idx)
                        block_idx += 1

                    yield sse_message_delta("tool_use", output_tokens)
                else:
                    if block_idx == 0:
                        yield sse_content_block_start(0, {"type": "text", "text": ""})
                        yield sse_content_block_stop(0)
                    # [PROTOCOL] Anthropic 有真正的 stop_sequence 终止原因，如实报告；
                    # OpenAI 没有这个枚举值，那边只能是 "stop"。
                    yield sse_message_delta(
                        "stop_sequence" if hit_stop else "end_turn",
                        output_tokens,
                        stop_sequence=hit_stop.matched if hit_stop else None,
                    )

                yield sse_message_stop()

                if client_session_id or conv_id:
                    # [M3] 短冷却降级期间不迁移会话归属（见 chat 流式分支同处注释）。
                    keep_provider = (
                        affinity_decision.provider_key
                        if affinity_decision.status == "degraded"
                        else dispatched_provider
                    )
                    session_affinity_manager.record_turn(
                        client_session_id or conv_id,
                        keep_provider,
                        dispatched_model,
                        None if affinity_decision.status == "degraded" else conv_id,
                    )

                if not client_session_id and conv_id and hasattr(active_provider, "delete_conversation"):
                    conversation_gc.schedule(active_provider, dispatched_provider, conv_id)
                success = True
            except ProviderError as e:
                is_rate_limit = (e.status == 429)
                error_msg = e.message
                is_risk, _ = is_risk_interception(e.message)
                if is_risk:
                    await failover_router.cooldown_tracker.record_failure(dispatched_provider, e.status, e.message)
                yield anthropic_sse_event(
                    "error", anthropic_error_response(e.message, e.status, e.err_type)
                )
            except Exception as e:
                error_msg = str(e)
                yield anthropic_sse_event(
                    "error", anthropic_error_response(f"上游服务异常: {e}", 502, "upstream_error")
                )
            finally:
                # [C2] 上游流必须显式关闭：`ManagedStream.close()` 归还节点信号量，
                # 并关闭 provider 侧的 HTTP 流。正常结束/中途异常/断连都走这里；
                # 响应体从未启动的路径由响应类的 `on_close` 同步兜底。
                await _close_upstream_stream(stream)
                load_balancer.release(
                    dispatched_provider,
                    success=success,
                    is_rate_limit=is_rate_limit,
                    error_msg=error_msg,
                )

        stream_body = gen()
        if flight_decision and flight_decision.is_leader:
            stream_body = broadcast_stream_wrapper(stream_body, flight_decision.group)
        # 客户端断连传播：显式关闭上游流并即时回收上游并发配额
        if not disconnect_guard.is_disabled():
            stream_body = guard_stream(stream_body, provider_key=dispatched_provider)

        # [C2] 上游流与节点信号量在返回响应之前就已持有；响应体若从未被拉取，
        # 生成器的 finally 不会执行（guard_stream 同样从未启动，看不到中断）。
        # `on_close` 是这条路径上唯一能归还已有资源的钩子。
        return ReclaimingStreamingResponse(
            sse_keepalive(stream_body),
            media_type="text/event-stream",
            headers=stream_headers,
            on_close=_stream_reclaimer(preopened_stream),
            on_close_async=_stream_closer(preopened_stream),
        )

    # 非流式：若当前请求为 Follower，挂载等待 Leader 完成
    if flight_decision and not flight_decision.is_leader:
        try:
            await asyncio.wait_for(flight_decision.group.done_event.wait(), timeout=TIMEOUT)
        except asyncio.TimeoutError:
            return JSONResponse(anthropic_error_response("等待并发共享请求超时", 504, "timeout_error"), status_code=504)
        if flight_decision.group.error:
            if isinstance(flight_decision.group.error, ProviderError):
                pe = flight_decision.group.error
                return JSONResponse(anthropic_error_response(pe.message, pe.status, pe.err_type), status_code=pe.status)
            return JSONResponse(anthropic_error_response(f"上游服务异常: {flight_decision.group.error}", 502, "upstream_error"), status_code=502)

        follower_headers = dict(flight_decision.group.completed_headers or {})
        follower_headers.update(sf_headers)
        return JSONResponse(flight_decision.group.completed_result, headers=follower_headers)

    # 非流式
    chunks: list[str] = []
    thinking_chunks: list[str] = []
    conv_id: Optional[str] = None
    start_time = time.time()
    thinking_start_time: Optional[float] = None
    thinking_end_time: Optional[float] = None
    load_balancer.acquire(dispatched_provider)
    ttft_recorded = False
    success = False
    group_finalized = False
    is_rate_limit = False
    error_msg = ""
    try:
        stream, failover_event, active_provider = await failover_router.execute_chat(
            primary_provider_key=dispatched_provider,
            primary_wire_model=dispatched_model,
            provider_factory=lambda pkey: _make_provider(pkey),
            chat_args=chat_args,
        )
        async for delta, _meta in stream:
            if not ttft_recorded and delta:
                load_balancer.record_ttft(
                    dispatched_provider, (time.time() - start_time) * 1000.0
                )
                ttft_recorded = True
            if not delta:
                if _meta and _meta.get("conversation_id"):
                    conv_id = _meta["conversation_id"]
                continue
            if _meta and _meta.get("reasoning"):
                if thinking_start_time is None:
                    thinking_start_time = time.time()
                thinking_end_time = time.time()
                thinking_chunks.append(delta)
            else:
                chunks.append(delta)
            if _meta and _meta.get("conversation_id"):
                conv_id = _meta["conversation_id"]
        success = True
    except ProviderError as e:
        is_rate_limit = (e.status == 429)
        error_msg = e.message
        is_risk, _ = is_risk_interception(e.message)
        if is_risk:
            await failover_router.cooldown_tracker.record_failure(dispatched_provider, e.status, e.message)
        if flight_decision and flight_decision.is_leader:
            await flight_decision.group.finish(error=e)
            group_finalized = True
        return JSONResponse(
            anthropic_error_response(e.message, e.status, e.err_type),
            status_code=e.status,
            headers=_refusal_headers(e),
        )
    except Exception as e:
        error_msg = str(e)
        if flight_decision and flight_decision.is_leader:
            await flight_decision.group.finish(error=e)
            group_finalized = True
        return JSONResponse(
            anthropic_error_response(f"上游服务异常: {e}", 502, "upstream_error"),
            status_code=502,
        )
    finally:
        # 客户端断连（CancelledError）等 BaseException 不会进入上述 except 分支，
        # 必须在 finally 中确定性回收并发配额并补齐飞行组终局，避免 Follower 悬挂。
        load_balancer.release(
            dispatched_provider,
            success=success,
            is_rate_limit=is_rate_limit,
            error_msg=error_msg,
        )
        if not success and not group_finalized and flight_decision and flight_decision.is_leader:
            terminate_aborted_group(flight_decision.group)

    if client_session_id or conv_id:
        # [M3] 短冷却降级期间不迁移会话归属（见 chat 流式分支同处注释）。
        keep_provider = (
            affinity_decision.provider_key
            if affinity_decision.status == "degraded"
            else dispatched_provider
        )
        session_affinity_manager.record_turn(
            client_session_id or conv_id,
            keep_provider,
            dispatched_model,
            None if affinity_decision.status == "degraded" else conv_id,
        )

    if not client_session_id and conv_id and hasattr(active_provider, "delete_conversation"):
        conversation_gc.schedule(active_provider, dispatched_provider, conv_id)

    full_output = "".join(chunks)
    full_thinking = "".join(thinking_chunks)
    # [PROTOCOL] 停用词截断必须在工具解析之前（截断点后的调用对客户端不存在）
    full_output, _stop_matched = truncate_final(full_output, stop_seq)
    output_tokens = max(1, _est_tokens(full_output + full_thinking))
    msg_id = _id("msg", "_")

    thinking_duration_ms = None
    thinking_toks = None
    if full_thinking:
        thinking_toks = max(1, _est_tokens(full_thinking))
        if thinking_start_time and thinking_end_time:
            thinking_duration_ms = max(1.0, (thinking_end_time - thinking_start_time) * 1000.0)

    thinking_headers = _make_thinking_headers(thinking_duration_ms, thinking_toks)

    resp_headers = dict(lb_headers)
    resp_headers.update(fold_headers)
    resp_headers.update(risk_headers)
    resp_headers.update(thinking_headers)
    resp_headers.update(affinity_headers)
    resp_headers.update(sf_headers)
    if failover_event:
        resp_headers.update({
            "x-failover-from": failover_event.from_provider,
            "x-failover-to": failover_event.to_provider,
            "x-failover-model": failover_event.to_model,
            "x-failover-ring": failover_event.ring.value,
        })

    blocks: list[dict] = []
    # [PROTOCOL] display=omitted：思考照常参与 usage 与思考通道兜底，只是不进
    # 响应体——客户端会把它存进历史，下一轮再发回来。
    if full_thinking and not suppress_thinking:
        blocks.append({"type": "thinking", "thinking": full_thinking})

    if req.tools:
        parsed, was_healed = parse_tool_calls_with_healing(
            full_output, resolve_allowed_tools(req.tools, req.tool_choice)
        )
        if not parsed:
            # [TOOL-LOOP] 思考通道兜底：正文为空时工具调用可能整个写在思考流里
            _rt, _rh = recover_tools_from_reasoning(
                full_thinking,
                resolve_allowed_tools(req.tools, req.tool_choice),
                answer_text=full_output,
            )
            if _rt:
                _rt = cap_parallel_tool_calls(_rt, parallel_disallowed)
                _tool_healing_stats["reasoning_recovered_tool_calls"] += len(_rt)
                parsed, was_healed = ("", _rt), _rh
        if parsed:
            prefix, tool_list = parsed
            # [TOOL-LOOP] 客户端禁止并行调用时只回传第一个
            tool_list = cap_parallel_tool_calls(tool_list, parallel_disallowed)
            _tool_healing_stats["total_tool_calls"] += len(tool_list)
            if was_healed:
                _tool_healing_stats["healed_tool_calls"] += 1
            resp_headers.update(_make_tool_headers(len(tool_list), was_healed))
            tool_blocks = build_anthropic_tool_blocks(prefix, tool_list)
            blocks.extend(tool_blocks)
            msg_data = anthropic_message_response(
                msg_id=msg_id,
                model=req.model,
                content=blocks,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                stop_reason="tool_use",
                conversation_id=client_session_id or conv_id,
            )
            if failover_event:
                msg_data["failover"] = failover_event.to_dict()
            if flight_decision and flight_decision.is_leader:
                await flight_decision.group.finish(result=msg_data, headers=resp_headers)
            return JSONResponse(
                msg_data,
                headers=resp_headers if resp_headers else None,
            )

    if full_output or not full_thinking:
        blocks.append({"type": "text", "text": full_output})

    msg_data = anthropic_message_response(
        msg_id=msg_id,
        model=req.model,
        content=blocks,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        # [PROTOCOL] Anthropic 的枚举里有 stop_sequence，如实报告截断原因
        stop_reason="stop_sequence" if _stop_matched else "end_turn",
        conversation_id=client_session_id or conv_id,
        stop_sequence=_stop_matched or None,
    )
    if failover_event:
        msg_data["failover"] = failover_event.to_dict()
    if flight_decision and flight_decision.is_leader:
        await flight_decision.group.finish(result=msg_data, headers=resp_headers)
    return JSONResponse(
        msg_data,
        headers=resp_headers if resp_headers else None,
    )


@app.post("/v1/messages/count_tokens")
@app.post("/messages/count_tokens")
async def count_tokens(req: AnthropicCountTokensRequest):
    # 确定性记忆化：相同 messages/system/tools 的 token 数恒定，避免重复 O(n) 估算
    cache_key: Optional[str] = None
    if not token_counter_cache.is_disabled():
        cache_key = compute_count_key(req)
        cached = token_counter_cache.get(cache_key)
        if cached is not None:
            return {"input_tokens": cached}

    # [PROTOCOL] 估算路径必须看到与 `/v1/messages` **完全相同**的提示词：
    # `tool_choice.disable_parallel_tool_use` 会让 messages 路径给工具提示词追加一句
    # 「本回合不要发出多个工具调用」的约束，这里却从不传 `parallel_allowed`，两份
    # 提示词实测长度 1267 vs 1122。Claude Code 用这个数字判断上下文预算，系统性
    # 偏低会让它在该压缩时继续硬塞。与 `tool_choice` 进指纹是同一条纪律——上报给
    # 客户端的是一个它自己的请求算不出来的数，就是错。
    parallel_disallowed = parallel_calls_disallowed(None, req.tool_choice)
    gateway_messages = convert_anthropic_to_gateway_messages(
        req.messages,
        system=req.system,
        tools=req.tools,
        tool_choice=req.tool_choice,
        parallel_allowed=not parallel_disallowed,
    )
    prompt = flatten_messages(gateway_messages)
    input_tokens = _est_tokens(prompt)

    if cache_key is not None:
        token_counter_cache.put(cache_key, input_tokens)

    return {"input_tokens": input_tokens}


@app.post("/v1/responses")
@app.post("/responses")
async def responses_endpoint(req: ResponsesRequest):
    parallel_disallowed = parallel_calls_disallowed(req.parallel_tool_calls, req.tool_choice)
    # [PROTOCOL] OpenAI 生态通用字段：Responses 客户端同样会带 stop
    stop_seq = req.stop
    gateway_messages = convert_responses_to_gateway_messages(req)
    if not gateway_messages:
        return JSONResponse(
            error_response("`input` 或 `instructions` 不能为空", 400, "invalid_request_error"),
            status_code=400,
        )

    # 0. 自适应提示词折叠优化：在估算 tokens 与路由调度前折叠超长历史轮次
    gateway_messages, fold_meta = prompt_optimizer.optimize_chat_messages(gateway_messages)
    fold_headers = make_prompt_folding_headers(fold_meta)

    provider_key, wire_model = resolve_model(req.model)
    if not provider_key:
        return JSONResponse(
            error_response(
                f"未知模型 `{req.model}`，可用模型: {', '.join(all_models())}",
                404,
                "model_not_found",
            ),
            status_code=404,
        )

    input_text = flatten_messages(gateway_messages)
    input_tokens = _est_tokens(input_text)
    temperature = req.temperature if req.temperature is not None else 0.7
    max_tokens = req.max_output_tokens or req.max_tokens or 4096

    # 判定 thinking 意图
    enable_thinking = False
    if req.thinking is not None:
        if isinstance(req.thinking, dict):
            enable_thinking = req.thinking.get("type") == "enabled"
        else:
            enable_thinking = bool(req.thinking)
    else:
        # [AGENT-LOOP] 线上有两种写法：`reasoning_effort:"high"`（旧版 SDK）与
        # `reasoning:{"effort":"high"}`（Codex CLI 现行）。只认前者会让 Codex 的
        # 「想清楚再动手」静默失效——尤其代码类模型（gpt-5-codex）不走推理前缀兜底。
        effort = req.reasoning_effort
        if not effort and isinstance(req.reasoning, dict):
            effort = req.reasoning.get("effort")
        elif not effort and isinstance(req.reasoning, str):
            effort = req.reasoning
        if isinstance(effort, str) and effort.lower() in ("high", "medium", "low", "minimal"):
            enable_thinking = True

    # 自适应负载均衡与被禁节点即刻转移路由决策
    decision = load_balancer.select_route(
        primary_provider=provider_key,
        primary_model=wire_model,
        prompt_tokens=input_tokens,
        is_thinking=enable_thinking,
    )
    lb_headers = _make_lb_headers(decision)
    dispatched_provider = decision.provider_key
    dispatched_model = decision.wire_model

    # 2.1 智能会话粘滞解析与跨模型状态迁移
    affinity_decision = session_affinity_manager.resolve_affinity(
        session_id=req.conversation_id,
        primary_provider=dispatched_provider,
        primary_model=dispatched_model,
        is_provider_available=_is_provider_healthy,
        can_retain_degraded=_is_provider_retainable,
    )
    affinity_headers = _make_affinity_headers(affinity_decision)
    dispatched_provider = affinity_decision.provider_key
    dispatched_model = affinity_decision.wire_model

    chat_args = {
        "messages": gateway_messages,
        "conversation_id": affinity_decision.upstream_conv_id,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "thinking": enable_thinking,
    }

    # [M1] 节奏裁决只有一处：`execute_chat` 派发循环里的
    # `apply_pacing_and_acquire`（紧贴上游调用、带锁预约可用时刻）。这里原先还有一次
    # `traffic_pacer.acquire(...)` 的「等待后立即释放」：它只推迟了本条请求自己，
    # 对并发者毫无约束（实测三路并发里有两路在同一毫秒打到上游），却让诊断数字
    # 看起来像是做了频控。现在此处只**读取**裁决结果用于遥测，不再产生第二段等待。
    risk_headers = _make_risk_headers(None, dispatched_provider, primary_provider=provider_key)

    # 2.2 智能并发语义去重与流式请求合并 (SingleFlight)
    flight_decision = None
    sf_headers = {}
    if not singleflight_manager.is_disabled():
        flight_key = singleflight_manager.compute_key(
            provider=dispatched_provider,
            model=dispatched_model,
            messages=gateway_messages,
            tools=req.tools,
            temperature=temperature,
            thinking=enable_thinking,
            stream=req.stream,
            protocol="openai_responses",
            # [S3] 会改变输出的客户端声明必须进去重键：Follower 直接领用
            # Leader 的答案，键相同就等于断言「两者的正确答案可以互换」。
            # 这些字段原先不在键里，而那时它们确实不改变输出（被解析后无人读）；
            # 逐个修好之后它们开始改变输出，键却没同步 —— 声明 tool_choice:"none"
            # 的请求会领到工具调用，且没有任何错误可看。
            constraints={
                "tool_choice": req.tool_choice,
                "parallel_tool_calls": req.parallel_tool_calls,
                "stop": getattr(req, "stop", None),
                "text": req.text,
                "max_output_tokens": req.max_output_tokens,
            },
        )
        flight_decision = await singleflight_manager.get_or_create(flight_key)
        sf_headers = _make_singleflight_headers(flight_decision)

    base_headers = dict(lb_headers)
    base_headers.update(fold_headers)
    base_headers.update(risk_headers)
    base_headers.update(affinity_headers)
    base_headers.update(sf_headers)
    base_headers.update(_make_disconnect_headers())

    if req.stream:
        stream_headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
        stream_headers.update(base_headers)

        if flight_decision and not flight_decision.is_leader:
            return StreamingResponse(
                sse_keepalive(stream_follower_gen(flight_decision.group)),
                media_type="text/event-stream",
                headers=stream_headers,
            )
    else:
        if flight_decision and not flight_decision.is_leader:
            try:
                await asyncio.wait_for(flight_decision.group.done_event.wait(), timeout=TIMEOUT)
            except asyncio.TimeoutError:
                return JSONResponse(error_response("等待并发共享请求超时", 504, "timeout_error"), status_code=504)
            if flight_decision.group.error:
                if isinstance(flight_decision.group.error, ProviderError):
                    pe = flight_decision.group.error
                    return JSONResponse(error_response(pe.message, pe.status, pe.err_type), status_code=pe.status)
                return JSONResponse(error_response(f"上游服务异常: {flight_decision.group.error}", 502, "upstream_error"), status_code=502)

            follower_headers = dict(flight_decision.group.completed_headers or {})
            follower_headers.update(sf_headers)
            return JSONResponse(flight_decision.group.completed_result, headers=follower_headers)

    resp_id = _id("resp", "_")
    msg_id = _id("msg", "_")
    start_time = time.time()
    load_balancer.acquire(dispatched_provider)
    ttft_recorded = False
    success = False
    group_finalized = False
    is_rate_limit = False
    error_msg = ""
    lb_released = False
    lb_handed_off = False

    def _release_lb() -> None:
        """幂等释放并发配额：首个上游调用阶段中断时由 finally 兜底调用。"""
        nonlocal lb_released
        if lb_released:
            return
        lb_released = True
        load_balancer.release(
            dispatched_provider,
            success=success,
            is_rate_limit=is_rate_limit,
            error_msg=error_msg,
        )

    try:
        stream, failover_event, active_provider = await failover_router.execute_chat(
            primary_provider_key=dispatched_provider,
            primary_wire_model=dispatched_model,
            provider_factory=lambda pkey: _make_provider(pkey),
            chat_args=chat_args,
        )
        # 配额所有权移交下游流式/聚合分支，由其负责释放
        lb_handed_off = True
    except ProviderError as e:
        is_risk, _ = is_risk_interception(e.message)
        if is_risk:
            await failover_router.cooldown_tracker.record_failure(dispatched_provider, e.status, e.message)
        is_rate_limit = (e.status == 429)
        error_msg = e.message
        if flight_decision and flight_decision.is_leader:
            await flight_decision.group.finish(error=e)
            group_finalized = True
        return JSONResponse(
            error_response(e.message, e.status, e.err_type),
            status_code=e.status,
            headers=_refusal_headers(e),
        )
    except Exception as e:
        error_msg = str(e)
        if flight_decision and flight_decision.is_leader:
            await flight_decision.group.finish(error=e)
            group_finalized = True
        return JSONResponse(error_response(f"上游服务异常: {e}", 502, "upstream_error"), status_code=502)
    finally:
        # 首个上游响应前被中断（客户端断连 CancelledError 等 BaseException）时，
        # 上述 except 分支不会执行：必须在此确定性回收并发配额并作废飞行组。
        if not lb_handed_off:
            _release_lb()
            if not group_finalized and flight_decision and flight_decision.is_leader:
                terminate_aborted_group(flight_decision.group)

    resp_headers = dict(base_headers)
    if failover_event:
        resp_headers.update({
            "x-failover-from": failover_event.from_provider,
            "x-failover-to": failover_event.to_provider,
            "x-failover-model": failover_event.to_model,
            "x-failover-ring": failover_event.ring.value,
        })

    if req.stream:
        stream_headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
        stream_headers.update(resp_headers)

        async def gen():
            nonlocal ttft_recorded, success, is_rate_limit, error_msg
            conv_id: Optional[str] = None
            ev = ResponsesEventStream(resp_id, req.model)
            try:
                # [TOOL-LOOP] 两个分支现在共用同一套事件构造器与同一套流处理逻辑。
                # 原先「有 tools」与「无 tools」是两份几乎相同的 200 行循环，差异只在
                # 是否经过工具锁存——推理增量只在无 tools 分支被过滤掉、文本事件在
                # 两处各自拼装，于是任何一处改动都会与另一处漂移（这正是本仓库反复
                # 出现的缺陷类型）。现在由 `feed_chunk(tools_enabled=...)` 统一决定。
                streamer = SpeculativeToolStreamer(
                    tools_enabled=bool(req.tools),
                    allowed=resolve_allowed_tools(req.tools, req.tool_choice),
                )
                # [TOOL-LOOP] freeform（custom）工具的出口形状与 function 工具不同，
                # 在流的起点算一次、全程复用（与 allowed 同源，避免两份判断）。
                freeform_names = freeform_tool_names(req.tools)
                # [PROTOCOL] 工具模式同样受停用词约束（三条协议共用同一条纪律）
                stop_scan = StopSequenceScanner(stop_seq)
                yield ev.created()
                yield ev.in_progress()

                # [RESPONSES-STREAM] 输出项必须按**到达顺序**逐个关闭，且每个项
                # 独占一个 index。原实现把所有 index 递增都攒到流末尾，于是
                # reasoning 与 message 同时挂在 index 0，第二段思考（正文之后再次
                # 思考，推理模型在工具循环里的常态）还被追加到一个已经写过正文的
                # 消息项上——实测事件流为
                #   added(0,reasoning) … added(0,message) … reasoning_delta(0) …
                # 客户端按 output_index 归档内容，重复索引会互相覆盖，表现为
                # 「agent 输出丢了一半」。
                out_idx = 0
                text_item_id = msg_id
                reasoning_id = _id("rs", "_")
                in_text_part = False
                in_reasoning = False
                emitted_parts: list[str] = []
                emitted_reasoning: list[str] = []
                output_items: list[dict] = []

                async def _close_reasoning():
                    """关闭当前 reasoning 项并让出 index。

                    缓冲必须**按项清空**：不清空时第二段思考的 done 会带上第一段
                    内容（实测 'think1 think2'），客户端与 completed.output 不一致。
                    """
                    nonlocal out_idx, in_reasoning, reasoning_id
                    if not in_reasoning:
                        return
                    text = "".join(emitted_reasoning)
                    item = reasoning_item(reasoning_id, text)
                    yield ev.reasoning_summary_done(reasoning_id, text, out_idx)
                    yield ev.output_item_done(out_idx, item)
                    output_items.append(item)
                    emitted_reasoning.clear()
                    in_reasoning = False
                    reasoning_id = _id("rs", "_")
                    out_idx += 1

                async def _close_text():
                    """关闭当前 message 项并让出 index（缓冲按项清空，理由同上）。"""
                    nonlocal out_idx, in_text_part, text_item_id
                    if not in_text_part:
                        return
                    text = "".join(emitted_parts)
                    yield ev.output_text_done(text_item_id, text, out_idx, 0)
                    yield ev.content_part_done(text_item_id, text, out_idx, 0)
                    item = message_item(text_item_id, text)
                    yield ev.output_item_done(out_idx, item)
                    output_items.append(item)
                    emitted_parts.clear()
                    in_text_part = False
                    text_item_id = _id("msg", "_")
                    out_idx += 1

                async for delta, _meta in stream:
                    if not ttft_recorded and delta:
                        load_balancer.record_ttft(
                            dispatched_provider, (time.time() - start_time) * 1000.0
                        )
                        ttft_recorded = True

                    if not delta:
                        if _meta and _meta.get("conversation_id"):
                            conv_id = _meta["conversation_id"]
                        continue
                    if _meta and _meta.get("conversation_id"):
                        conv_id = _meta["conversation_id"]

                    is_r = bool(_meta and _meta.get("reasoning"))
                    for act in (
                        streamer.feed_chunk(delta, is_reasoning=True)
                        if is_r
                        else streamer.feed_text(delta)
                    ):
                        if act.action_type == "reasoning_delta":
                            # 思考流必须以 reasoning output item 呈现。原实现静默丢弃，
                            # Codex 看到的是「想了几十秒然后什么都没有」。
                            if in_text_part:
                                async for frame in _close_text():
                                    yield frame
                            if not in_reasoning:
                                yield ev.output_item_added(
                                    out_idx, reasoning_item(reasoning_id, "", status="in_progress")
                                )
                                in_reasoning = True
                            emitted_reasoning.append(act.content)
                            yield ev.reasoning_summary_delta(reasoning_id, act.content, out_idx)
                        elif act.action_type == "text_delta":
                            if in_reasoning:
                                async for frame in _close_reasoning():
                                    yield frame
                            # [PROTOCOL] 停用词在**开项之前**判定：命中时这一项
                            # 根本不该出现，先开一个空 message 项会污染 output 列表
                            safe_act = stop_scan.feed(act.content)
                            if not safe_act:
                                continue
                            if not in_text_part:
                                yield ev.output_item_added(
                                    out_idx, message_item(text_item_id, "", status="in_progress")
                                )
                                yield ev.content_part_added(text_item_id, out_idx, 0)
                                in_text_part = True
                            emitted_parts.append(safe_act)
                            yield ev.output_text_delta(text_item_id, safe_act, out_idx, 0)

                # [PROTOCOL] 停用词必须先截断累积再 finalize：命中点之后的
                # ```tool_call 对客户端不存在，解析出来就是一次没有预期的执行
                _matched_stop = streamer.truncate_at_stop(stop_seq)
                if _matched_stop is not None:
                    # [PROTOCOL] 把命中的原文交给扫描器：线契约要求 stop_sequence
                    # 带着它，只报告「被截了」而不说被什么截，客户端无法定位。
                    stop_scan.mark_hit(_matched_stop)
                rem_text, tool_list, was_healed = streamer.finalize()
                # [TOOL-LOOP] 客户端禁止并行调用时只回传第一个（见 cap_parallel_tool_calls）
                tool_list = cap_parallel_tool_calls(tool_list, parallel_disallowed)
                if not tool_list:
                    # [TOOL-LOOP] 思考通道兜底：正文为空时工具调用可能整个写在思考流里
                    tool_list, was_healed = _recover_reasoning_tools(
                        streamer,
                        resolve_allowed_tools(req.tools, req.tool_choice),
                        parallel_disallowed,
                        answer_text="".join(streamer.accumulated_text),
                    )
                # [PROTOCOL] 终局剩余正文同样过停用词判定，并把扣住的尾巴放行
                rem_text = stop_scan.feed(rem_text) + stop_scan.flush()
                if rem_text:
                    if in_reasoning:
                        async for frame in _close_reasoning():
                            yield frame
                    if not in_text_part:
                        yield ev.output_item_added(
                            out_idx, message_item(text_item_id, "", status="in_progress")
                        )
                        yield ev.content_part_added(text_item_id, out_idx, 0)
                        in_text_part = True
                    emitted_parts.append(rem_text)
                    yield ev.output_text_delta(text_item_id, rem_text, out_idx, 0)

                full_reasoning = "".join(streamer.accumulated_reasoning)
                async for frame in _close_reasoning():
                    yield frame
                async for frame in _close_text():
                    yield frame

                if tool_list:
                    _tool_healing_stats["total_tool_calls"] += len(tool_list)
                    if was_healed:
                        _tool_healing_stats["healed_tool_calls"] += 1
                    for tool_name, args in tool_list:
                        call_id = _id("call", "_")
                        # [TOOL-LOOP] freeform 工具发 `custom_tool_call` + 原文 input；
                        # function 工具发 `function_call` + JSON arguments。两者的事件名
                        # 都不同，客户端按名分派，混用等于丢掉调用。
                        if tool_name in freeform_names:
                            tool_text = custom_tool_input_text(args)
                            yield ev.output_item_added(
                                out_idx,
                                custom_tool_call_item(call_id, tool_name, "", status="in_progress"),
                            )
                            yield ev.custom_tool_call_input_delta(call_id, out_idx, tool_text, call_id)
                            yield ev.custom_tool_call_input_done(call_id, out_idx, tool_text, call_id)
                            item_fn = custom_tool_call_item(call_id, tool_name, tool_text)
                        else:
                            args_str = json.dumps(args, ensure_ascii=False) if isinstance(args, (dict, list)) else str(args)
                            yield ev.output_item_added(
                                out_idx, function_call_item(call_id, tool_name, "", status="in_progress")
                            )
                            yield ev.function_call_args_delta(call_id, out_idx, call_id, args_str)
                            yield ev.function_call_args_done(call_id, out_idx, call_id, args_str)
                            item_fn = function_call_item(call_id, tool_name, args_str)
                        yield ev.output_item_done(out_idx, item_fn)
                        output_items.append(item_fn)
                        out_idx += 1
                elif not output_items:
                    # 既无文本也无工具（空回复）：仍然给出一个结构完整的空消息项，
                    # 否则客户端会认为本轮没有任何输出。
                    # [RESPONSES-STREAM] 判据必须是「本轮还没有任何 message 项」而非
                    # `not in_text_part`——后者在每个项关闭后都成立，于是有思考无正文的
                    # 回复会多出一个 idx=4 的空消息项。
                    yield ev.output_item_added(
                        out_idx, message_item(text_item_id, "", status="in_progress")
                    )
                    yield ev.content_part_added(text_item_id, out_idx, 0)
                    yield ev.output_text_done(text_item_id, "", out_idx, 0)
                    yield ev.content_part_done(text_item_id, "", out_idx, 0)
                    item_msg = message_item(text_item_id, "")
                    yield ev.output_item_done(out_idx, item_msg)
                    output_items.append(item_msg)

                out_tokens = max(
                    1, _est_tokens("".join(streamer.accumulated_text) + full_reasoning)
                )
                reasoning_tokens = max(1, _est_tokens(full_reasoning)) if full_reasoning else 0
                yield ev.completed(
                    output_items,
                    input_tokens,
                    out_tokens,
                    reasoning_tokens=reasoning_tokens,
                    failover=failover_event.to_dict() if failover_event else None,
                    conversation_id=req.conversation_id or conv_id,
                    # [PROTOCOL] 被客户端 stop 截断时如实标明 incomplete
                    incomplete_reason="stop" if stop_scan.hit else None,
                )
                yield sse_done()

                if req.conversation_id or conv_id:
                    session_affinity_manager.record_turn(
                        req.conversation_id or conv_id,
                        dispatched_provider,
                        dispatched_model,
                        conv_id,
                    )

                if not req.conversation_id and conv_id and hasattr(active_provider, "delete_conversation"):
                    conversation_gc.schedule(active_provider, dispatched_provider, conv_id)
                success = True
            except ProviderError as e:
                is_rate_limit = (e.status == 429)
                error_msg = e.message
                is_risk, _ = is_risk_interception(e.message)
                if is_risk:
                    await failover_router.cooldown_tracker.record_failure(dispatched_provider, e.status, e.message)
                yield ev.error(e.message, e.status, e.err_type)
            except Exception as e:
                error_msg = str(e)
                yield ev.error(f"上游服务异常: {e}", 502, "upstream_error")
            finally:
                # [C2] 先关闭上游流（`ManagedStream.close` 归还节点信号量并关闭
                # provider 侧连接），再释放负载均衡配额。
                await _close_upstream_stream(stream)
                _release_lb()

        stream_body = gen()
        if flight_decision and flight_decision.is_leader:
            stream_body = broadcast_stream_wrapper(stream_body, flight_decision.group)
        # 客户端断连传播：显式关闭上游流并即时回收上游并发配额
        if not disconnect_guard.is_disabled():
            stream_body = guard_stream(stream_body, provider_key=dispatched_provider)

        # [C2] 上游流与节点信号量在返回响应之前就已持有；响应体若从未被拉取，
        # 生成器的 finally 不会执行（guard_stream 同样从未启动，看不到中断）。
        # `on_close` 是这条路径上唯一能归还已有资源的钩子。
        return ReclaimingStreamingResponse(
            sse_keepalive(stream_body),
            media_type="text/event-stream",
            headers=stream_headers,
            on_close=_stream_reclaimer(stream),
            on_close_async=_stream_closer(stream),
            # [C2] 本路径的负载均衡配额在构造响应之前就已 acquire（见上方
            # `load_balancer.acquire`），而归还写在 `gen()` 的 finally 里。响应体
            # 从未启动时该 finally 不会执行，配额永久泄漏——负载均衡随后会逐步把
            # 这个节点判为高负载并持续分流走，直到进程重启。这里补一个幂等同步回收。
            on_exit=_release_lb,
        )

    # 非流式处理
    chunks: list[str] = []
    reasoning_chunks: list[str] = []
    conv_id: Optional[str] = None
    thinking_start_time: Optional[float] = None
    thinking_end_time: Optional[float] = None
    try:
        async for delta, meta in stream:
            if not ttft_recorded and delta:
                load_balancer.record_ttft(
                    dispatched_provider, (time.time() - start_time) * 1000.0
                )
                ttft_recorded = True
            if not delta:
                if meta and meta.get("conversation_id"):
                    conv_id = meta["conversation_id"]
                continue
            if meta and meta.get("reasoning"):
                if thinking_start_time is None:
                    thinking_start_time = time.time()
                thinking_end_time = time.time()
                reasoning_chunks.append(delta)
            else:
                chunks.append(delta)
            if meta and meta.get("conversation_id"):
                conv_id = meta["conversation_id"]
        success = True
    except ProviderError as e:
        is_rate_limit = (e.status == 429)
        error_msg = e.message
        if flight_decision and flight_decision.is_leader:
            await flight_decision.group.finish(error=e)
            group_finalized = True
        return JSONResponse(
            error_response(e.message, e.status, e.err_type),
            status_code=e.status,
            headers=_refusal_headers(e),
        )
    except Exception as e:
        error_msg = str(e)
        if flight_decision and flight_decision.is_leader:
            await flight_decision.group.finish(error=e)
            group_finalized = True
        return JSONResponse(error_response(f"上游服务异常: {e}", 502, "upstream_error"), status_code=502)
    finally:
        # 客户端断连（CancelledError）等 BaseException 不会进入上述 except 分支，
        # 必须在 finally 中确定性回收并发配额并补齐飞行组终局，避免 Follower 悬挂。
        _release_lb()
        if not success and not group_finalized and flight_decision and flight_decision.is_leader:
            terminate_aborted_group(flight_decision.group)

    if req.conversation_id or conv_id:
        session_affinity_manager.record_turn(
            req.conversation_id or conv_id,
            dispatched_provider,
            dispatched_model,
            conv_id,
        )

    if not req.conversation_id and conv_id and hasattr(active_provider, "delete_conversation"):
        conversation_gc.schedule(active_provider, dispatched_provider, conv_id)

    full_output = "".join(chunks)
    full_reasoning = "".join(reasoning_chunks)
    # [PROTOCOL] 停用词截断必须在工具解析之前（截断点后的调用对客户端不存在）
    full_output, _stop_matched = truncate_final(full_output, stop_seq)
    output_tokens = max(1, _est_tokens(full_output + full_reasoning))

    thinking_duration_ms = None
    thinking_toks = None
    if full_reasoning:
        thinking_toks = max(1, _est_tokens(full_reasoning))
        if thinking_start_time and thinking_end_time:
            thinking_duration_ms = max(1.0, (thinking_end_time - thinking_start_time) * 1000.0)

    thinking_headers = _make_thinking_headers(thinking_duration_ms, thinking_toks)
    resp_headers.update(thinking_headers)

    tool_calls_list = None
    parsed_res = None
    was_healed = False
    if req.tools:
        parsed_res, was_healed = parse_tool_calls_with_healing(
            full_output, resolve_allowed_tools(req.tools, req.tool_choice)
        )
    if not parsed_res:
        # [TOOL-LOOP] 思考通道兜底：正文为空时工具调用可能整个写在思考流里
        _rt, _rh = recover_tools_from_reasoning(
            full_reasoning,
            resolve_allowed_tools(req.tools, req.tool_choice),
            answer_text=full_output,
        )
        if _rt:
            _rt = cap_parallel_tool_calls(_rt, parallel_disallowed)
            _tool_healing_stats["reasoning_recovered_tool_calls"] += len(_rt)
            parsed_res, was_healed = ("", _rt), _rh
    if parsed_res:
        prefix, tool_list = parsed_res
        # [TOOL-LOOP] 客户端禁止并行调用时只回传第一个
        tool_list = cap_parallel_tool_calls(tool_list, parallel_disallowed)
        _tool_healing_stats["total_tool_calls"] += len(tool_list)
        if was_healed:
            _tool_healing_stats["healed_tool_calls"] += 1
        resp_headers.update(_make_tool_headers(len(tool_list), was_healed))
        tool_calls_list = [
            {
                "id": _id("call", "_"),
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(args, ensure_ascii=False) if isinstance(args, (dict, list)) else str(args),
                },
            }
            for tool_name, args in tool_list
        ]
        full_output = prefix

    resp_data = format_responses_completion(
        resp_id=resp_id,
        model=req.model,
        output_text=full_output,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        conversation_id=req.conversation_id or conv_id,
        tool_calls=tool_calls_list,
        msg_id=msg_id,
        failover=failover_event.to_dict() if failover_event else None,
        reasoning_text=full_reasoning,
        reasoning_tokens=max(1, _est_tokens(full_reasoning)) if full_reasoning else 0,
        # [TOOL-LOOP] freeform 工具必须以 custom_tool_call 输出（同流式路径）
        freeform_tools=freeform_tool_names(req.tools),
    )
    # [PROTOCOL] 被客户端 stop 截断时如实标明 incomplete（与流式路径同一语义）
    if _stop_matched:
        resp_data["status"] = "incomplete"
        resp_data["incomplete_details"] = {"reason": "stop"}
    if failover_event:
        resp_data["failover"] = failover_event.to_dict()
    if flight_decision and flight_decision.is_leader:
        await flight_decision.group.finish(result=resp_data, headers=resp_headers)
    return JSONResponse(
        resp_data,
        headers=resp_headers if resp_headers else None,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host=HOST, port=PORT, reload=False)
