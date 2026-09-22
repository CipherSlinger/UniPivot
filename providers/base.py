"""OpenAI 兼容格式与通用工具：消息展平、响应/流式帧构造、错误类型。"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator, List, Optional, Union

import httpx

from .http_client import get_http_client

_ROLE_LABELS = {"system": "System", "user": "User", "assistant": "Assistant"}


@asynccontextmanager
async def ensure_async_client(
    client: Optional[httpx.AsyncClient] = None,
    default_timeout: float = 600.0,
) -> AsyncIterator[httpx.AsyncClient]:
    """若提供了未关闭的 client 则直接复用（不关闭）；否则获取全局共享的 AsyncClient 单例。"""
    if client is not None and not client.is_closed:
        yield client
    else:
        shared = await get_http_client(default_timeout)
        yield shared


class ProviderError(Exception):
    """上游（网页端）调用失败，携带给客户端看的错误信息。"""

    def __init__(self, message: str, status: int = 502, err_type: str = "upstream_error"):
        super().__init__(message)
        self.status = status
        self.err_type = err_type
        self.message = message


def text_of(content: Union[str, List[dict], None]) -> str:
    """从 OpenAI 消息 content（字符串或多部分列表）中提取纯文本。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for p in content:
        if isinstance(p, dict) and p.get("type") == "text":
            parts.append(p.get("text", ""))
    return "\n".join(parts)


def flatten_messages(messages: List[dict]) -> str:
    """把一段 OpenAI 历史消息展平成网页端可理解的一段提示词。

    单条 user 消息原样发送；多轮/system 消息按 "角色: 内容" 序列化。
    若末尾为 assistant 消息（prefill 模式），保留其内容作为补全前缀，不重复添加 Assistant 标签。
    """
    if len(messages) == 1 and messages[0].get("role") == "user":
        return text_of(messages[0].get("content"))

    lines = []
    for m in messages:
        label = _ROLE_LABELS.get(m.get("role", "user"), str(m.get("role", "user")).capitalize())
        content = text_of(m.get("content"))
        if content:
            lines.append(f"{label}: {content}")
    if messages and messages[-1].get("role") != "assistant":
        lines.append("Assistant:")
    return "\n\n".join(lines)


def last_user_content(messages: List[dict]) -> str:
    """取最后一条 user 消息作为“本轮新消息”（续聊会话时使用）。"""
    for m in reversed(messages):
        if m["role"] == "user":
            return text_of(m.get("content"))
    return text_of(messages[-1].get("content"))


def _now() -> int:
    return int(time.time())


def _id(prefix: str = "chatcmpl", sep: str = "-") -> str:
    return f"{prefix}{sep}" + uuid.uuid4().hex


def _est_tokens(text: str) -> int:
    """估算中英文与代码混合文本的 token 数。

    中文/CJK 字符约 1.5 token/字；英文单词与符号约 3.5 字符/token。
    """
    if not text:
        return 1
    cjk_count = sum(
        1 for c in text
        if "一" <= c <= "鿿" or "　" <= c <= "〿" or "＀" <= c <= "￯"
    )
    other_count = len(text) - cjk_count
    tokens = int(cjk_count * 1.5 + other_count / 3.5)
    return max(1, tokens)


def parse_cookie_header(cookie: str) -> dict[str, str]:
    """把 Cookie 字符串解析成字典。"""
    cookies: dict[str, str] = {}
    if not cookie:
        return cookies
    for item in cookie.split(";"):
        if "=" in item:
            k, v = item.strip().split("=", 1)
            cookies[k.strip()] = v.strip()
    return cookies


def sse_frame(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def completion_response(
    model: str, content: Optional[str], prompt: str,
    conversation_id: Optional[str] = None,
    thinking_content: str = "",
    tool_calls: Optional[List[dict]] = None,
    finish_reason: str = "stop",
) -> dict:
    """构造非流式的 OpenAI chat.completion 对象。"""
    pt = _est_tokens(prompt)
    ct = _est_tokens((content or "") + thinking_content)
    if tool_calls:
        ct += sum(_est_tokens(json.dumps(tc, ensure_ascii=False)) for tc in tool_calls)
    msg: dict = {"role": "assistant", "content": content}
    if thinking_content:
        msg["reasoning_content"] = thinking_content
        if not content:
            msg["content"] = f"<think>\n{thinking_content}\n</think>"
    if tool_calls:
        msg["tool_calls"] = tool_calls
    resp = {
        "id": _id(),
        "object": "chat.completion",
        "created": _now(),
        "model": model,
        "choices": [{"index": 0, "message": msg, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "total_tokens": pt + ct,
        },
    }
    if conversation_id:
        resp["conversation_id"] = conversation_id
    return resp


def error_response(
    message: str,
    status: int = 502,
    err_type: str = "upstream_error",
    param: Optional[str] = None,
    code: Optional[Union[str, int]] = None,
) -> dict:
    err: dict = {"message": message, "type": err_type}
    if param is not None:
        err["param"] = param
    if code is not None:
        err["code"] = code
    return {"error": err}
