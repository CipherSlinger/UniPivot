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


def _tool_call_id_map(messages: List[dict]) -> dict:
    """收集 assistant.tool_calls 的 id → 工具名映射。

    OpenAI 的 `role:"tool"` 消息只带 `tool_call_id` 而不带工具名，若不做映射，
    上游只能看到一段无名的工具输出，无法判断它对应用途。
    """
    mapping: dict = {}
    for m in messages:
        if not isinstance(m, dict) or m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            call_id = tc.get("id")
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
            name = fn.get("name") or tc.get("name") or ""
            if call_id and name:
                mapping[call_id] = name
    return mapping


def _serialize_tool_calls(tool_calls) -> str:
    """把 assistant.tool_calls 序列化成上游可识别的 ```tool_call 块。

    [TOOL-LOOP] 这是工具循环能闭合的一半：上游是纯文本网页接口，模型只能通过
    它自己被告知的 ```tool_call 契约来「回忆」自己发过哪些调用。原先这里完全
    忽略 tool_calls，导致客户端回传的 assistant 工具调用在提示词里整段消失，
    模型看到一个自己从未发起过的工具结果，因而无法接续（实测见
    tests/test_tool_roundtrip.py）。
    """
    if not isinstance(tool_calls, list):
        return ""
    blocks = []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
        name = fn.get("name") or tc.get("name") or ""
        if not name:
            continue
        args = fn.get("arguments", tc.get("arguments", {}))
        if isinstance(args, str):
            # OpenAI 线上格式里 arguments 是 JSON 字符串；原样塞进块会变成
            # 被二次编码的字符串，模型看到的是转义后的文本而不是参数对象。
            try:
                args = json.loads(args) if args.strip() else {}
            except Exception:  # noqa: BLE001 - 参数不合法时降级为空对象，绝不丢调用本身
                args = {}
        elif not isinstance(args, dict):
            args = {}
        blocks.append(
            "```tool_call\n"
            + json.dumps({"name": name, "arguments": args}, ensure_ascii=False)
            + "\n```"
        )
    return "\n\n".join(blocks)


def _message_line(m: dict, id_map: dict) -> Optional[str]:
    """把单条消息渲染成上游可读的一行；无任何可表达内容时返回 None。"""
    role = str(m.get("role") or "user").lower()
    content = text_of(m.get("content"))

    if role == "assistant":
        calls = _serialize_tool_calls(m.get("tool_calls"))
        if calls:
            content = f"{content}\n\n{calls}" if content.strip() else calls
        label = "Assistant"
    elif role == "tool":
        name = m.get("name") or id_map.get(m.get("tool_call_id") or "", "")
        label = f"Tool ({name})" if name else "Tool"
        # 工具执行成功但无输出时也要让模型知道它跑过了，否则会重复调用。
        content = content or "(empty result)"
    else:
        label = _ROLE_LABELS.get(role, role.capitalize())

    if not content:
        return None
    return f"{label}: {content}"


def flatten_messages(messages: List[dict]) -> str:
    """把一段 OpenAI 历史消息展平成网页端可理解的一段提示词。

    单条 user 消息原样发送；多轮/system 消息按 "角色: 内容" 序列化。
    assistant 携带的 `tool_calls` 与 `role:"tool"` 的工具结果都会被如实展开
    （[TOOL-LOOP]），使客户端回传的 agent 循环在上游保持完整。
    若末尾为 assistant 消息（prefill 模式），保留其内容作为补全前缀，不重复添加 Assistant 标签。
    """
    if len(messages) == 1 and messages[0].get("role") == "user":
        return text_of(messages[0].get("content"))

    id_map = _tool_call_id_map(messages)
    lines = [ln for ln in (_message_line(m, id_map) for m in messages) if ln]
    if messages and messages[-1].get("role") != "assistant":
        lines.append("Assistant:")
    return "\n\n".join(lines)


def continuation_content(messages: List[dict]) -> str:
    """续聊模式下本轮真正要发送的新内容。

    续聊（`conversation_id` 非空）时上游会话自己保存历史，因此只发送「新的一轮
    输入」，不能整段重发。**但一轮 agent 工具循环里，最后一条消息不是 user 而是
    tool**：assistant 发起调用 → 客户端执行 → 回传 tool 结果，此时工具结果*就是*
    本轮要送达的新内容。

    [TOOL-LOOP] 原实现无条件取最后一条 `role=="user"`，于是工具结果被整个跳过、
    把上一轮的旧提问重复发出去（实测：返回 'read foo.py' 而不是工具输出），模型
    永远接不上工具调用。现改为取「最后一条 user 之后」的全部消息：没有工具循环时
    行为与原先逐值相同（零回归），有工具循环时如实送达工具调用与结果。
    """
    if not messages:
        return ""

    id_map = _tool_call_id_map(messages)
    last_user_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if isinstance(m, dict) and str(m.get("role") or "").lower() == "user":
            last_user_idx = i
            break

    if last_user_idx < 0:
        # 没有 user（如纯 prefill 的 system+assistant）：整段作为续写前缀
        tail = messages
    else:
        tail = messages[last_user_idx + 1:]

    if not tail:
        return text_of(messages[last_user_idx].get("content"))

    lines = [ln for ln in (_message_line(m, id_map) for m in tail) if ln]
    if not lines:
        # 尾部全部不可表达（例如 assistant 空内容）：退回上一条 user，避免发出空提示
        return text_of(messages[last_user_idx].get("content")) if last_user_idx >= 0 else ""
    return "\n\n".join(lines)


def last_user_content(messages: List[dict]) -> str:
    """取最后一条 user 消息（保留给仍需旧语义的调用方）。

    [TOOL-LOOP] 网关的上游路径已全部改用 `continuation_content()`：本函数在一轮
    agent 工具循环里（最后一条是 `role:"tool"`）会返回上一轮的旧提问。保留它只为
    不破坏既有导入面，**不应**再用作上游提示词来源。
    """
    for m in reversed(messages):
        if m.get("role") == "user":
            return text_of(m.get("content"))
    return text_of(messages[-1].get("content")) if messages else ""


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
