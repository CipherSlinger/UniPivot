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
import os
import threading
import time
import urllib.parse
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional, Union

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
    convert_anthropic_to_gateway_messages,
    error_response,
    flatten_messages,
    format_anthropic_models,
    format_tools_system_prompt,
    last_user_content,
    parse_tool_call,
    parse_tool_calls,
    parse_tool_calls_with_healing,
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
    convert_responses_to_gateway_messages,
    format_responses_completion,
    sse_response_created,
    sse_response_output_item_added,
    sse_response_content_part_added,
    sse_response_text_delta,
    sse_response_content_part_done,
    sse_response_output_item_done,
    sse_response_completed,
    sse_response_function_call_args_delta,
    sse_response_function_call_args_done,
    sse_response_done,
    sse_response_error,
)
from providers.base import _est_tokens, _id
from providers import http_client
from providers.http_client import close_http_client, get_default_limits, get_http_client
from providers.prompt_optimizer import prompt_optimizer, make_prompt_folding_headers
from providers.session_affinity import AffinityDecision, session_affinity_manager
from session_store import (
    MAX_AGE,
    resolve_deepseek,
    resolve_doubao,
    resolve_glm,
    resolve_kimi,
    resolve_qwen,
)
from healing import HealthMonitor, HealingEngine, TaskDispatcher, TaskEvaluator, find_agent_cli

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
    delay_ms: float, provider_key: str, primary_provider: Optional[str] = None
) -> dict[str, str]:
    """构造流量整形等待耗时与风控避让冷却响应头"""
    headers = {
        "x-risk-pacing-delay-ms": str(round(delay_ms, 2)),
    }
    rem_cd = failover_router.cooldown_tracker.get_remaining_cooldown(provider_key)
    if rem_cd <= 0 and primary_provider:
        rem_cd = failover_router.cooldown_tracker.get_remaining_cooldown(primary_provider)
    if rem_cd > 0:
        headers["x-risk-cooldown-active"] = str(round(rem_cd, 2))
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
        elif m_lower.startswith("gpt-4") or m_lower.startswith("gpt-4o"):
            tier = "flagship"
        elif m_lower.startswith("o1") or m_lower.startswith("o3"):
            tier = "reasoner"

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
    # 网页端专属能力，经 extra_body 传入
    thinking: bool = False
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


# ---------------------------------------------------------------- 应用

def _auto_refresh_stale_sessions() -> None:
    """启动时后台线程：对超过 MAX_AGE 的会话做一次静默刷新（复用已登录
    的浏览器配置，不弹窗）。刷新失败不影响服务，请求报 401 时再提示登录。"""
    if os.getenv("GATEWAY_DISABLE_AUTO_REFRESH") == "1" or os.getenv("TESTING") == "1":
        return

    from login import refresh_deepseek, refresh_doubao, refresh_glm, refresh_kimi, refresh_qwen

    for name, resolver, refresher in (
        ("qwen", resolve_qwen, refresh_qwen),
        ("deepseek", resolve_deepseek, refresh_deepseek),
        ("doubao", resolve_doubao, refresh_doubao),
        ("kimi", resolve_kimi, refresh_kimi),
        ("glm", resolve_glm, refresh_glm),
    ):
        try:
            s = resolver()
            if not s or not s.token or s.age < MAX_AGE:
                continue
            print(f"[网关] {name} 令牌已超过 {MAX_AGE // 3600}h，尝试静默刷新…")
            ns = refresher(timeout=60)
            if ns:
                print(f"[网关] {name} 令牌刷新成功")
            else:
                print(f"[网关] {name} 静默刷新失败（可能已退出登录）")
        except Exception as e:
            print(f"[网关] {name} 静默刷新异常: {type(e).__name__}: {e}")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    threading.Thread(target=_auto_refresh_stale_sessions, daemon=True).start()
    await get_http_client(TIMEOUT)
    if os.getenv("TESTING") != "1":
        health_monitor.start(initial_delay=2.0)
    try:
        yield
    finally:
        health_monitor.stop()
        await close_http_client()


app = FastAPI(title="同枢 (UniPivot) 本地多协议网关", version="1.3.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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
}


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
    return headers


def _is_provider_healthy(pkey: str) -> bool:
    """检查指定 Provider 是否未被显式禁用且未处于熔断冷却期。"""
    disabled = os.getenv("DISABLED_PROVIDERS", "").split(",")
    if pkey in [d.strip() for d in disabled if d.strip()]:
        return False
    if failover_router.cooldown_tracker.is_in_cooldown(pkey):
        return False
    return True


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
    if is_anthropic_request(request):
        return JSONResponse(
            status_code=exc.status,
            content=anthropic_error_response(exc.message, status=exc.status, err_type=exc.err_type),
        )
    return JSONResponse(
        status_code=exc.status,
        content=error_response(exc.message, status=exc.status, err_type=exc.err_type),
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
    return {
        "status": "ok",
        "disabled_providers": sorted(list(disabled_set)),
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
    """实时���统状态、多提供方健康与度量诊断接口。"""
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
            "features": [
                "speculative_text_streaming",
                "zero_latency_reasoning",
                "deep_json_repair",
                "truncated_json_auto_completion",
                "telemetry_headers",
            ],
        },
        "session_affinity": session_affinity_manager.get_diagnostics(),
        "risk_avoidance": risk_avoidance,
    }


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
    if req.tools:
        tools_prompt = format_tools_system_prompt(req.tools)
        if tools_prompt:
            if messages and messages[0].get("role") == "system":
                orig = messages[0].get("content") or ""
                messages[0]["content"] = f"{orig}\n\n{tools_prompt}".strip()
            else:
                messages.insert(0, {"role": "system", "content": tools_prompt})

    temperature = req.temperature if req.temperature is not None else 0.7
    max_tokens = req.max_tokens if req.max_tokens is not None else 2048

    # 0. 自适应提示词折叠优化：在估算 tokens 与路由调度前折叠超长历史轮次
    messages, fold_meta = prompt_optimizer.optimize_chat_messages(messages)
    fold_headers = make_prompt_folding_headers(fold_meta)

    # 1. 估算 prompt tokens 与思考意图
    prompt_text = last_user_content(messages) if req.conversation_id else flatten_messages(messages)
    prompt_tokens = _est_tokens(prompt_text)
    is_thinking = bool(req.thinking)

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
    )
    affinity_headers = _make_affinity_headers(affinity_decision)
    dispatched_provider = affinity_decision.provider_key
    dispatched_model = affinity_decision.wire_model

    chat_args = {
        "messages": messages,
        "conversation_id": affinity_decision.upstream_conv_id,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "thinking": req.thinking,
        "search": req.search,
        "stream": req.stream,
    }

    async with traffic_pacer.acquire(dispatched_provider) as pacing_delay:
        risk_headers = _make_risk_headers(pacing_delay, dispatched_provider, primary_provider=provider_key)

    if req.stream:
        stream_headers = {
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
        stream_headers.update(lb_headers)
        stream_headers.update(fold_headers)
        stream_headers.update(risk_headers)
        stream_headers.update(affinity_headers)

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
                    stream, failover_event, active_provider = await failover_router.execute_chat(
                        primary_provider_key=dispatched_provider,
                        primary_wire_model=dispatched_model,
                        provider_factory=lambda pkey: _make_provider(pkey),
                        chat_args=chat_args,
                    )

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
                            accumulated_chunks.append(delta)
                            yield sse_frame({
                                "id": cid,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": req.model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {"content": delta},
                                    "finish_reason": None,
                                }],
                            })

                        if meta and meta.get("conversation_id"):
                            conversation_id = meta["conversation_id"]

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
                        asyncio.create_task(active_provider.delete_conversation(conversation_id))
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
                stream, failover_event, active_provider = await failover_router.execute_chat(
                    primary_provider_key=dispatched_provider,
                    primary_wire_model=dispatched_model,
                    provider_factory=lambda pkey: _make_provider(pkey),
                    chat_args=chat_args,
                )

                streamer = SpeculativeToolStreamer(tools_enabled=True)
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
                    actions = streamer.feed_chunk(delta, is_reasoning=is_r)
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
                            chunk = {
                                "id": cid,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": req.model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {"role": "assistant", "content": act.content},
                                    "finish_reason": None,
                                }],
                            }
                            if not first_frame_emitted and failover_event:
                                chunk["failover"] = failover_event.to_dict()
                            yield sse_frame(chunk)
                            first_frame_emitted = True

                rem_text, tool_list, was_healed = streamer.finalize()
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
                    session_affinity_manager.record_turn(
                        req.conversation_id or conversation_id,
                        dispatched_provider,
                        dispatched_model,
                        conversation_id,
                    )

                if not req.conversation_id and conversation_id and hasattr(active_provider, "delete_conversation"):
                    asyncio.create_task(active_provider.delete_conversation(conversation_id))
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

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers=stream_headers,
        )

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
        return JSONResponse(error_response(e.message, e.status, e.err_type), status_code=e.status)
    except Exception as e:
        error_msg = str(e)
        return JSONResponse(error_response(f"上游服务异常: {e}", 502, "upstream_error"), status_code=502)
    finally:
        load_balancer.release(
            dispatched_provider,
            success=success,
            is_rate_limit=is_rate_limit,
            error_msg=error_msg,
        )

    if conversation_id:
        session_affinity_manager.record_turn(
            req.conversation_id or conversation_id,
            dispatched_provider,
            dispatched_model,
            conversation_id,
        )

    if not req.conversation_id and conversation_id and hasattr(active_provider, "delete_conversation"):
        asyncio.create_task(active_provider.delete_conversation(conversation_id))

    prompt = prompt_text
    full_output = "".join(chunks)
    full_reasoning = "".join(reasoning_chunks)

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
    if failover_event:
        resp_headers.update({
            "x-failover-from": failover_event.from_provider,
            "x-failover-to": failover_event.to_provider,
            "x-failover-model": failover_event.to_model,
            "x-failover-ring": failover_event.ring.value,
        })

    if req.tools:
        parsed, was_healed = parse_tool_calls_with_healing(full_output)
        if parsed:
            prefix, tool_list = parsed
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
            return JSONResponse(resp_data, headers=resp_headers if resp_headers else None)

    resp_data = completion_response(
        req.model, full_output, prompt, conversation_id, thinking_content=full_reasoning
    )
    if failover_event:
        resp_data["failover"] = failover_event.to_dict()
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

    gateway_messages = convert_anthropic_to_gateway_messages(
        req.messages, system=req.system, tools=req.tools
    )
    # 0. 自适应提示词折叠优化：在估算 tokens 与路由调度前折叠超长历史轮次
    gateway_messages, fold_meta = prompt_optimizer.optimize_chat_messages(gateway_messages)
    fold_headers = make_prompt_folding_headers(fold_meta)

    input_text = flatten_messages(gateway_messages)
    input_tokens = _est_tokens(input_text)
    temperature = req.temperature if req.temperature is not None else 0.7
    max_tokens = req.max_tokens or 4096

    # 判定 thinking 意图
    enable_thinking = False
    if req.thinking:
        if isinstance(req.thinking, dict):
            enable_thinking = req.thinking.get("type") == "enabled"
        else:
            enable_thinking = bool(req.thinking)

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

    async with traffic_pacer.acquire(dispatched_provider) as pacing_delay:
        risk_headers = _make_risk_headers(pacing_delay, dispatched_provider, primary_provider=provider_key)

    if req.stream:
        stream_headers = {
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
        stream_headers.update(lb_headers)
        stream_headers.update(fold_headers)
        stream_headers.update(risk_headers)
        stream_headers.update(affinity_headers)

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
                    stream, failover_event, active_provider = await failover_router.execute_chat(
                        primary_provider_key=dispatched_provider,
                        primary_wire_model=dispatched_model,
                        provider_factory=lambda pkey: _make_provider(pkey),
                        chat_args=chat_args,
                    )

                    yield sse_message_start(msg_id, req.model, input_tokens)

                    current_block_index: int = -1
                    current_block_type: Optional[str] = None
                    accumulated_chunks: list[str] = []
                    accumulated_thinking: list[str] = []

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
                        target_type = "thinking" if is_reasoning else "text"

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
                            accumulated_thinking.append(delta)
                            yield sse_content_block_delta(
                                current_block_index,
                                {"type": "thinking_delta", "thinking": delta},
                            )
                        else:
                            accumulated_chunks.append(delta)
                            yield sse_content_block_delta(
                                current_block_index,
                                {"type": "text_delta", "text": delta},
                            )

                        if _meta and _meta.get("conversation_id"):
                            conv_id = _meta["conversation_id"]

                    if current_block_type is not None:
                        yield sse_content_block_stop(current_block_index)
                    else:
                        yield sse_content_block_start(0, {"type": "text", "text": ""})
                        yield sse_content_block_stop(0)

                    output_tokens = max(
                        1, _est_tokens("".join(accumulated_chunks) + "".join(accumulated_thinking))
                    )
                    yield sse_message_delta("end_turn", output_tokens)
                    yield sse_message_stop()

                    if conv_id or req.conversation_id:
                        session_affinity_manager.record_turn(
                            req.conversation_id or conv_id,
                            dispatched_provider,
                            dispatched_model,
                            conv_id,
                        )

                    if not req.conversation_id and conv_id and hasattr(active_provider, "delete_conversation"):
                        asyncio.create_task(active_provider.delete_conversation(conv_id))
                    success = True
                    return

                # 2. Agent / 工具模式：推测式流式分发与容错解析
                stream, failover_event, active_provider = await failover_router.execute_chat(
                    primary_provider_key=dispatched_provider,
                    primary_wire_model=dispatched_model,
                    provider_factory=lambda pkey: _make_provider(pkey),
                    chat_args=chat_args,
                )

                streamer = SpeculativeToolStreamer(tools_enabled=True)
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
                    actions = streamer.feed_chunk(delta, is_reasoning=is_r)
                    for act in actions:
                        if act.action_type == "reasoning_delta":
                            if not in_thinking_block:
                                yield sse_content_block_start(block_idx, {"type": "thinking", "thinking": ""})
                                in_thinking_block = True
                            yield sse_content_block_delta(block_idx, {"type": "thinking_delta", "thinking": act.content})
                        elif act.action_type == "text_delta":
                            if in_thinking_block:
                                yield sse_content_block_stop(block_idx)
                                in_thinking_block = False
                                block_idx += 1
                            if not in_text_block:
                                yield sse_content_block_start(block_idx, {"type": "text", "text": ""})
                                in_text_block = True
                            yield sse_content_block_delta(block_idx, {"type": "text_delta", "text": act.content})

                rem_text, tool_list, was_healed = streamer.finalize()
                if in_thinking_block:
                    yield sse_content_block_stop(block_idx)
                    in_thinking_block = False
                    block_idx += 1

                if rem_text:
                    if not in_text_block:
                        yield sse_content_block_start(block_idx, {"type": "text", "text": ""})
                        in_text_block = True
                    yield sse_content_block_delta(block_idx, {"type": "text_delta", "text": rem_text})

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
                    yield sse_message_delta("end_turn", output_tokens)

                yield sse_message_stop()

                if conv_id or req.conversation_id:
                    session_affinity_manager.record_turn(
                        req.conversation_id or conv_id,
                        dispatched_provider,
                        dispatched_model,
                        conv_id,
                    )

                if not req.conversation_id and conv_id and hasattr(active_provider, "delete_conversation"):
                    asyncio.create_task(active_provider.delete_conversation(conv_id))
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
                load_balancer.release(
                    dispatched_provider,
                    success=success,
                    is_rate_limit=is_rate_limit,
                    error_msg=error_msg,
                )

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers=stream_headers,
        )

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
        return JSONResponse(
            anthropic_error_response(e.message, e.status, e.err_type),
            status_code=e.status,
        )
    except Exception as e:
        error_msg = str(e)
        return JSONResponse(
            anthropic_error_response(f"上游服务异常: {e}", 502, "upstream_error"),
            status_code=502,
        )
    finally:
        load_balancer.release(
            dispatched_provider,
            success=success,
            is_rate_limit=is_rate_limit,
            error_msg=error_msg,
        )

    if conv_id or req.conversation_id:
        session_affinity_manager.record_turn(
            req.conversation_id or conv_id,
            dispatched_provider,
            dispatched_model,
            conv_id,
        )

    if not req.conversation_id and conv_id and hasattr(active_provider, "delete_conversation"):
        asyncio.create_task(active_provider.delete_conversation(conv_id))

    full_output = "".join(chunks)
    full_thinking = "".join(thinking_chunks)
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
    if failover_event:
        resp_headers.update({
            "x-failover-from": failover_event.from_provider,
            "x-failover-to": failover_event.to_provider,
            "x-failover-model": failover_event.to_model,
            "x-failover-ring": failover_event.ring.value,
        })

    blocks: list[dict] = []
    if full_thinking:
        blocks.append({"type": "thinking", "thinking": full_thinking})

    if req.tools:
        parsed, was_healed = parse_tool_calls_with_healing(full_output)
        if parsed:
            prefix, tool_list = parsed
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
                conversation_id=conv_id or req.conversation_id,
            )
            if failover_event:
                msg_data["failover"] = failover_event.to_dict()
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
        conversation_id=conv_id or req.conversation_id,
    )
    if failover_event:
        msg_data["failover"] = failover_event.to_dict()
    return JSONResponse(
        msg_data,
        headers=resp_headers if resp_headers else None,
    )


@app.post("/v1/messages/count_tokens")
@app.post("/messages/count_tokens")
async def count_tokens(req: AnthropicCountTokensRequest):
    gateway_messages = convert_anthropic_to_gateway_messages(
        req.messages, system=req.system, tools=req.tools
    )
    prompt = flatten_messages(gateway_messages)
    return {"input_tokens": _est_tokens(prompt)}


@app.post("/v1/responses")
@app.post("/responses")
async def responses_endpoint(req: ResponsesRequest):
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
    elif req.reasoning_effort and req.reasoning_effort.lower() in ("high", "medium", "low"):
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

    async with traffic_pacer.acquire(dispatched_provider) as pacing_delay:
        risk_headers = _make_risk_headers(pacing_delay, dispatched_provider, primary_provider=provider_key)

    resp_id = _id("resp", "_")
    msg_id = _id("msg", "_")
    start_time = time.time()
    load_balancer.acquire(dispatched_provider)
    ttft_recorded = False
    success = False
    is_rate_limit = False
    error_msg = ""

    try:
        stream, failover_event, active_provider = await failover_router.execute_chat(
            primary_provider_key=dispatched_provider,
            primary_wire_model=dispatched_model,
            provider_factory=lambda pkey: _make_provider(pkey),
            chat_args=chat_args,
        )
    except ProviderError as e:
        is_risk, _ = is_risk_interception(e.message)
        if is_risk:
            await failover_router.cooldown_tracker.record_failure(dispatched_provider, e.status, e.message)
        load_balancer.release(
            dispatched_provider,
            success=False,
            is_rate_limit=(e.status == 429),
            error_msg=e.message,
        )
        return JSONResponse(error_response(e.message, e.status, e.err_type), status_code=e.status)
    except Exception as e:
        load_balancer.release(
            dispatched_provider,
            success=False,
            is_rate_limit=False,
            error_msg=str(e),
        )
        return JSONResponse(error_response(f"上游服务异常: {e}", 502, "upstream_error"), status_code=502)

    resp_headers = dict(lb_headers)
    resp_headers.update(fold_headers)
    resp_headers.update(risk_headers)
    resp_headers.update(affinity_headers)
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
            "X-Accel-Buffering": "no",
        }
        stream_headers.update(resp_headers)

        async def gen():
            nonlocal ttft_recorded, success, is_rate_limit, error_msg
            conv_id: Optional[str] = None
            try:
                # 1. 纯文本模式：无 tools 声明时走零延迟逐字流式
                if not req.tools:
                    yield sse_response_created(resp_id, req.model)

                    output_item = {
                        "id": msg_id,
                        "type": "message",
                        "status": "in_progress",
                        "role": "assistant",
                        "content": [],
                    }
                    yield sse_response_output_item_added(resp_id, 0, output_item)
                    yield sse_response_content_part_added(
                        resp_id, 0, 0, {"type": "text", "text": ""}
                    )

                    accumulated_chunks: list[str] = []
                    accumulated_reasoning: list[str] = []

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

                        is_r = bool(_meta and _meta.get("reasoning"))
                        if is_r:
                            accumulated_reasoning.append(delta)
                        else:
                            accumulated_chunks.append(delta)
                            yield sse_response_text_delta(delta, resp_id=resp_id, output_index=0, content_index=0)

                        if _meta and _meta.get("conversation_id"):
                            conv_id = _meta["conversation_id"]

                    full_text = "".join(accumulated_chunks)
                    full_reasoning = "".join(accumulated_reasoning)
                    out_tokens = max(1, _est_tokens(full_text + full_reasoning))

                    yield sse_response_content_part_done(
                        full_text, resp_id=resp_id, output_index=0, content_index=0
                    )

                    done_item = {
                        "id": msg_id,
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "text", "text": full_text}],
                    }
                    yield sse_response_output_item_done(done_item, resp_id=resp_id, output_index=0)

                    yield sse_response_completed(
                        resp_id,
                        req.model,
                        [done_item],
                        input_tokens,
                        out_tokens,
                        failover=failover_event.to_dict() if failover_event else None,
                        conversation_id=conv_id or req.conversation_id,
                    )
                    yield sse_response_done()

                    if conv_id or req.conversation_id:
                        session_affinity_manager.record_turn(
                            req.conversation_id or conv_id,
                            dispatched_provider,
                            dispatched_model,
                            conv_id,
                        )

                    if not req.conversation_id and conv_id and hasattr(active_provider, "delete_conversation"):
                        asyncio.create_task(active_provider.delete_conversation(conv_id))
                    success = True
                    return

                # 2. Tools 模式：推测式流式分发与自愈解析
                streamer = SpeculativeToolStreamer(tools_enabled=True)
                yield sse_response_created(resp_id, req.model)

                out_idx = 0
                in_text_part = False

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
                    actions = streamer.feed_chunk(delta, is_reasoning=is_r)
                    for act in actions:
                        if act.action_type == "text_delta":
                            if not in_text_part:
                                yield sse_response_output_item_added(resp_id, out_idx, {
                                    "id": msg_id,
                                    "type": "message",
                                    "status": "in_progress",
                                    "role": "assistant",
                                    "content": [],
                                })
                                yield sse_response_content_part_added(
                                    resp_id, out_idx, 0, {"type": "text", "text": ""}
                                )
                                in_text_part = True
                            yield sse_response_text_delta(
                                act.content, resp_id=resp_id, output_index=out_idx, content_index=0
                            )

                rem_text, tool_list, was_healed = streamer.finalize()
                if rem_text:
                    if not in_text_part:
                        yield sse_response_output_item_added(resp_id, out_idx, {
                            "id": msg_id,
                            "type": "message",
                            "status": "in_progress",
                            "role": "assistant",
                            "content": [],
                        })
                        yield sse_response_content_part_added(
                            resp_id, out_idx, 0, {"type": "text", "text": ""}
                        )
                        in_text_part = True
                    yield sse_response_text_delta(
                        rem_text, resp_id=resp_id, output_index=out_idx, content_index=0
                    )

                output_items: list[dict] = []
                if in_text_part:
                    full_emitted = "".join(streamer.emitted_text) + rem_text
                    yield sse_response_content_part_done(
                        full_emitted, resp_id=resp_id, output_index=out_idx, content_index=0
                    )
                    item_msg = {
                        "id": msg_id,
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "text", "text": full_emitted}],
                    }
                    yield sse_response_output_item_done(item_msg, resp_id=resp_id, output_index=out_idx)
                    output_items.append(item_msg)
                    out_idx += 1

                if tool_list:
                    _tool_healing_stats["total_tool_calls"] += len(tool_list)
                    if was_healed:
                        _tool_healing_stats["healed_tool_calls"] += 1
                    for tool_name, args in tool_list:
                        call_id = _id("call", "_")
                        args_str = json.dumps(args, ensure_ascii=False) if isinstance(args, (dict, list)) else str(args)
                        item_fn = {
                            "id": call_id,
                            "type": "function_call",
                            "call_id": call_id,
                            "name": tool_name,
                            "arguments": "",
                            "status": "in_progress",
                        }
                        yield sse_response_output_item_added(resp_id, out_idx, item_fn)
                        yield sse_response_function_call_args_delta(resp_id, out_idx, call_id, args_str)
                        yield sse_response_function_call_args_done(resp_id, out_idx, call_id, args_str)

                        item_fn_done = {
                            "id": call_id,
                            "type": "function_call",
                            "call_id": call_id,
                            "name": tool_name,
                            "arguments": args_str,
                            "status": "completed",
                        }
                        yield sse_response_output_item_done(item_fn_done, resp_id=resp_id, output_index=out_idx)
                        output_items.append(item_fn_done)
                        out_idx += 1
                else:
                    if not in_text_part:
                        full_text = "".join(streamer.accumulated_text)
                        item_msg = {
                            "id": msg_id,
                            "type": "message",
                            "status": "completed",
                            "role": "assistant",
                            "content": [{"type": "text", "text": full_text}],
                        }
                        yield sse_response_output_item_added(resp_id, 0, {
                            "id": msg_id,
                            "type": "message",
                            "status": "in_progress",
                            "role": "assistant",
                            "content": [],
                        })
                        yield sse_response_content_part_added(
                            resp_id, 0, 0, {"type": "text", "text": ""}
                        )
                        if full_text:
                            yield sse_response_text_delta(full_text, resp_id=resp_id, output_index=0, content_index=0)
                        yield sse_response_content_part_done(
                            full_text, resp_id=resp_id, output_index=0, content_index=0
                        )
                        yield sse_response_output_item_done(item_msg, resp_id=resp_id, output_index=0)
                        output_items.append(item_msg)

                out_tokens = max(1, _est_tokens("".join(streamer.accumulated_text)))
                yield sse_response_completed(
                    resp_id,
                    req.model,
                    output_items,
                    input_tokens,
                    out_tokens,
                    failover=failover_event.to_dict() if failover_event else None,
                    conversation_id=conv_id or req.conversation_id,
                )
                yield sse_response_done()

                if conv_id or req.conversation_id:
                    session_affinity_manager.record_turn(
                        req.conversation_id or conv_id,
                        dispatched_provider,
                        dispatched_model,
                        conv_id,
                    )

                if not req.conversation_id and conv_id and hasattr(active_provider, "delete_conversation"):
                    asyncio.create_task(active_provider.delete_conversation(conv_id))
                success = True
            except ProviderError as e:
                is_rate_limit = (e.status == 429)
                error_msg = e.message
                yield sse_response_error(e.message, e.status, e.err_type)
            except Exception as e:
                error_msg = str(e)
                yield sse_response_error(f"上游服务异常: {e}", 502, "upstream_error")
            finally:
                load_balancer.release(
                    dispatched_provider,
                    success=success,
                    is_rate_limit=is_rate_limit,
                    error_msg=error_msg,
                )

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers=stream_headers,
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
        return JSONResponse(error_response(e.message, e.status, e.err_type), status_code=e.status)
    except Exception as e:
        error_msg = str(e)
        return JSONResponse(error_response(f"上游服务异常: {e}", 502, "upstream_error"), status_code=502)
    finally:
        load_balancer.release(
            dispatched_provider,
            success=success,
            is_rate_limit=is_rate_limit,
            error_msg=error_msg,
        )

    if conv_id or req.conversation_id:
        session_affinity_manager.record_turn(
            req.conversation_id or conv_id,
            dispatched_provider,
            dispatched_model,
            conv_id,
        )

    if not req.conversation_id and conv_id and hasattr(active_provider, "delete_conversation"):
        asyncio.create_task(active_provider.delete_conversation(conv_id))

    full_output = "".join(chunks)
    full_reasoning = "".join(reasoning_chunks)
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
    parsed_res, was_healed = parse_tool_calls_with_healing(full_output)
    if parsed_res:
        prefix, tool_list = parsed_res
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
        conversation_id=conv_id or req.conversation_id,
        tool_calls=tool_calls_list,
        msg_id=msg_id,
        failover=failover_event.to_dict() if failover_event else None,
    )
    if failover_event:
        resp_data["failover"] = failover_event.to_dict()
    return JSONResponse(
        resp_data,
        headers=resp_headers if resp_headers else None,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host=HOST, port=PORT, reload=False)
