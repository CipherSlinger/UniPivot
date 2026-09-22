"""OpenAI Responses API (/v1/responses) 协议兼容层。

支持 Codex CLI 及新版 OpenAI SDK Responses 协议标准：
- 请求转译（ResponsesRequest -> gateway_messages）
- 响应格式化（Response 对象、Message Output Item、Function Call Output Item）
- SSE 事件流生成（response.created -> output_item.added -> content_part.added -> text.delta -> content_part.done -> output_item.done -> response.completed -> [DONE]）
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, ConfigDict

from .anthropic_compat import format_tools_system_prompt
from .base import _est_tokens, _id, error_response


class ResponsesRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    input: Optional[Union[str, List[Any]]] = None
    instructions: Optional[Union[str, List[dict]]] = None
    stream: bool = False
    temperature: Optional[float] = None
    tools: Optional[List[dict]] = None
    max_output_tokens: Optional[int] = None
    max_tokens: Optional[int] = None
    conversation_id: Optional[str] = None
    reasoning_effort: Optional[str] = None
    thinking: Optional[Union[dict, bool]] = None


def extract_responses_content(content: Any) -> str:
    """从 Responses input content（字符串、列表或块结构）中提取文本内容。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        b_type = content.get("type", "")
        if b_type in ("text", "input_text"):
            return content.get("text", "")
        if "text" in content:
            return str(content["text"])
        return json.dumps(content, ensure_ascii=False)
    if not isinstance(content, list):
        return str(content)

    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            b_type = block.get("type", "")
            if b_type in ("text", "input_text"):
                t = block.get("text", "")
                if t:
                    parts.append(t)
            elif b_type == "thinking":
                th = block.get("thinking", "")
                if th:
                    parts.append(f"<thinking>\n{th}\n</thinking>")
            elif b_type in ("tool_result", "function_call_output"):
                sub = block.get("output", "") or block.get("content", "")
                if isinstance(sub, list):
                    sub = extract_responses_content(sub)
                elif isinstance(sub, dict):
                    sub = json.dumps(sub, ensure_ascii=False)
                c_id = block.get("call_id") or block.get("tool_use_id", "")
                tag = f"[Tool Result for {c_id}]" if c_id else "[Tool Result]"
                parts.append(f"{tag}:\n{sub or ''}")
            elif b_type in ("tool_use", "function_call"):
                name = block.get("name") or (
                    block.get("function", {}).get("name")
                    if isinstance(block.get("function"), dict)
                    else ""
                )
                args = block.get("arguments") or block.get("input") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        pass
                parts.append(
                    f"```tool_call\n{json.dumps({'name': name, 'arguments': args}, ensure_ascii=False)}\n```"
                )
            elif "text" in block and block["text"]:
                parts.append(str(block["text"]))
    return "\n".join(parts)


def convert_responses_to_gateway_messages(req: ResponsesRequest) -> List[dict]:
    """将 instructions, tools 以及多样化格式的 input 转换为网关底层标准消息列表。"""
    system_parts: list[str] = []

    if req.instructions:
        if isinstance(req.instructions, str):
            ins = req.instructions.strip()
            if ins:
                system_parts.append(ins)
        elif isinstance(req.instructions, list):
            ins = extract_responses_content(req.instructions).strip()
            if ins:
                system_parts.append(ins)
        else:
            ins = str(req.instructions).strip()
            if ins:
                system_parts.append(ins)

    if req.tools:
        tools_prompt = format_tools_system_prompt(req.tools)
        if tools_prompt:
            system_parts.append(tools_prompt)

    messages: list[dict] = []
    if system_parts:
        messages.append({"role": "system", "content": "\n\n".join(system_parts)})

    if req.input is None:
        return messages

    if isinstance(req.input, str):
        if req.input.strip():
            messages.append({"role": "user", "content": req.input})
        return messages

    if isinstance(req.input, list):
        for item in req.input:
            if isinstance(item, str):
                if item.strip():
                    messages.append({"role": "user", "content": item})
            elif isinstance(item, dict):
                item_type = item.get("type")
                if item_type == "message":
                    role = item.get("role", "user")
                    c_text = extract_responses_content(item.get("content"))
                    messages.append({"role": role, "content": c_text})
                elif item_type == "function_call":
                    name = item.get("name") or (
                        item.get("function", {}).get("name")
                        if isinstance(item.get("function"), dict)
                        else ""
                    )
                    args = item.get("arguments") or item.get("input") or {}
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            pass
                    call_text = f"```tool_call\n{json.dumps({'name': name, 'arguments': args}, ensure_ascii=False)}\n```"
                    messages.append({"role": "assistant", "content": call_text})
                elif item_type == "function_call_output":
                    c_id = item.get("call_id", "")
                    out = item.get("output", "")
                    if isinstance(out, (dict, list)):
                        out = json.dumps(out, ensure_ascii=False)
                    tag = f"[Tool Result for {c_id}]" if c_id else "[Tool Result]"
                    messages.append({"role": "user", "content": f"{tag}:\n{out}"})
                elif "role" in item:
                    role = item.get("role", "user")
                    c_text = extract_responses_content(item.get("content"))
                    if role == "tool":
                        c_id = item.get("tool_call_id") or item.get("call_id", "")
                        tag = f"[Tool Result for {c_id}]" if c_id else "[Tool Result]"
                        messages.append({"role": "user", "content": f"{tag}:\n{c_text}"})
                    else:
                        if "tool_calls" in item and isinstance(item["tool_calls"], list):
                            tc_parts = []
                            for tc in item["tool_calls"]:
                                fn = tc.get("function", {})
                                fn_name = fn.get("name") or tc.get("name")
                                fn_args = fn.get("arguments") or tc.get("arguments") or {}
                                if isinstance(fn_args, str):
                                    try:
                                        fn_args = json.loads(fn_args)
                                    except Exception:
                                        pass
                                tc_parts.append(
                                    f"```tool_call\n{json.dumps({'name': fn_name, 'arguments': fn_args}, ensure_ascii=False)}\n```"
                                )
                            if tc_parts:
                                c_text = (c_text + "\n" + "\n".join(tc_parts)).strip()
                        messages.append({"role": role, "content": c_text})
                else:
                    c_text = extract_responses_content(item)
                    if c_text:
                        messages.append({"role": "user", "content": c_text})

    return messages


def format_responses_completion(
    resp_id: str,
    model: str,
    output_text: str,
    input_tokens: int,
    output_tokens: int,
    status: str = "completed",
    created_at: Optional[int] = None,
    conversation_id: Optional[str] = None,
    tool_calls: Optional[List[dict]] = None,
    msg_id: Optional[str] = None,
    failover: Optional[dict] = None,
) -> dict:
    """构造标准的 Responses API JSON 对象。"""
    created = created_at or int(time.time())
    m_id = msg_id or _id("msg", "_")

    output_items: list[dict] = []
    if (output_text and output_text.strip()) or not tool_calls:
        output_items.append({
            "id": m_id,
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": output_text or "",
                }
            ],
        })

    if tool_calls:
        for tc in tool_calls:
            c_id = tc.get("id") or tc.get("call_id") or _id("call", "_")
            fn = tc.get("function", {}) if isinstance(tc.get("function"), dict) else {}
            fn_name = tc.get("name") or fn.get("name", "")
            fn_args = tc.get("arguments") if "arguments" in tc else fn.get("arguments", "")
            if isinstance(fn_args, (dict, list)):
                fn_args = json.dumps(fn_args, ensure_ascii=False)
            elif fn_args is None:
                fn_args = "{}"
            output_items.append({
                "id": c_id,
                "type": "function_call",
                "call_id": c_id,
                "name": fn_name,
                "arguments": str(fn_args),
                "status": "completed",
            })

    resp = {
        "id": resp_id,
        "object": "response",
        "created_at": created,
        "status": status,
        "model": model,
        "output": output_items,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }
    if conversation_id:
        resp["conversation_id"] = conversation_id
    if failover:
        resp["failover"] = failover
    return resp


def sse_response_created(resp_id: str, model: str, created_at: Optional[int] = None) -> str:
    created = created_at or int(time.time())
    data = {
        "response": {
            "id": resp_id,
            "object": "response",
            "created_at": created,
            "status": "in_progress",
            "model": model,
            "output": [],
        }
    }
    return f"event: response.created\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def sse_response_output_item_added(
    resp_id: str,
    output_index: int,
    item: dict,
) -> str:
    data = {
        "response_id": resp_id,
        "output_index": output_index,
        "item": item,
    }
    return f"event: response.output_item.added\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def sse_response_content_part_added(
    resp_id: str,
    output_index: int,
    content_index: int,
    part: Optional[dict] = None,
) -> str:
    data = {
        "response_id": resp_id,
        "output_index": output_index,
        "content_index": content_index,
        "part": part or {"type": "text", "text": ""},
    }
    return f"event: response.content_part.added\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def sse_response_text_delta(
    delta: str,
    resp_id: Optional[str] = None,
    output_index: int = 0,
    content_index: int = 0,
) -> str:
    data: dict[str, Any] = {"delta": delta}
    if resp_id:
        data["response_id"] = resp_id
        data["output_index"] = output_index
        data["content_index"] = content_index
    return f"event: response.text.delta\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def sse_response_content_part_done(
    full_text: str,
    resp_id: Optional[str] = None,
    output_index: int = 0,
    content_index: int = 0,
) -> str:
    data: dict[str, Any] = {
        "part": {
            "type": "text",
            "text": full_text,
        }
    }
    if resp_id:
        data["response_id"] = resp_id
        data["output_index"] = output_index
        data["content_index"] = content_index
    return f"event: response.content_part.done\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def sse_response_output_item_done(
    item: dict,
    resp_id: Optional[str] = None,
    output_index: int = 0,
) -> str:
    data: dict[str, Any] = {
        "item": item,
    }
    if resp_id:
        data["response_id"] = resp_id
        data["output_index"] = output_index
    return f"event: response.output_item.done\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def sse_response_completed(
    resp_id: str,
    model: str,
    output: list,
    input_tokens: int,
    output_tokens: int,
    created_at: Optional[int] = None,
    failover: Optional[dict] = None,
    conversation_id: Optional[str] = None,
) -> str:
    created = created_at or int(time.time())
    resp_obj: dict[str, Any] = {
        "id": resp_id,
        "object": "response",
        "created_at": created,
        "status": "completed",
        "model": model,
        "output": output,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }
    if conversation_id:
        resp_obj["conversation_id"] = conversation_id
    if failover:
        resp_obj["failover"] = failover

    data = {
        "response": resp_obj,
    }
    if failover:
        data["failover"] = failover
    return f"event: response.completed\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def sse_response_function_call_args_delta(
    resp_id: str,
    output_index: int,
    call_id: str,
    delta: str,
) -> str:
    data = {
        "response_id": resp_id,
        "output_index": output_index,
        "call_id": call_id,
        "delta": delta,
    }
    return f"event: response.function_call_arguments.delta\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def sse_response_function_call_args_done(
    resp_id: str,
    output_index: int,
    call_id: str,
    arguments: str,
) -> str:
    data = {
        "response_id": resp_id,
        "output_index": output_index,
        "call_id": call_id,
        "arguments": arguments,
    }
    return f"event: response.function_call_arguments.done\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def sse_response_done() -> str:
    return "data: [DONE]\n\n"


def sse_response_error(message: str, status: int = 502, err_type: str = "upstream_error") -> str:
    data = error_response(message, status, err_type)
    return f"event: error\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
