"""Anthropic Messages API (/v1/messages) 协议兼容层。

负责 Anthropic 请求格式、Content Block、错误响应与 SSE 事件流的标准化转译。
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, ConfigDict, Field
from .base import _id

CLAUDE_MODELS_METADATA = [
    {"id": "claude-3-7-sonnet-20250219", "display_name": "Claude 3.7 Sonnet", "created_at": "2025-02-19T00:00:00Z"},
    {"id": "claude-3-7-sonnet-latest", "display_name": "Claude 3.7 Sonnet Latest", "created_at": "2025-02-19T00:00:00Z"},
    {"id": "claude-3-5-sonnet-20241022", "display_name": "Claude 3.5 Sonnet", "created_at": "2024-10-22T00:00:00Z"},
    {"id": "claude-3-5-sonnet-latest", "display_name": "Claude 3.5 Sonnet Latest", "created_at": "2024-10-22T00:00:00Z"},
    {"id": "claude-3-5-haiku-20241022", "display_name": "Claude 3.5 Haiku", "created_at": "2024-10-22T00:00:00Z"},
    {"id": "claude-3-5-haiku-latest", "display_name": "Claude 3.5 Haiku Latest", "created_at": "2024-10-22T00:00:00Z"},
    {"id": "claude-3-opus-20240229", "display_name": "Claude 3 Opus", "created_at": "2024-02-29T00:00:00Z"},
    {"id": "claude-3-opus-latest", "display_name": "Claude 3 Opus Latest", "created_at": "2024-02-29T00:00:00Z"},
    {"id": "claude-opus-5", "display_name": "Claude Opus 5", "created_at": "2026-03-01T00:00:00Z"},
    {"id": "claude-sonnet-5", "display_name": "Claude Sonnet 5", "created_at": "2026-03-01T00:00:00Z"},
    {"id": "claude-fable-5-1", "display_name": "Claude Fable 5.1", "created_at": "2026-05-01T00:00:00Z"},
    {"id": "claude-haiku-4-5-20251001", "display_name": "Claude Haiku 4.5", "created_at": "2025-10-01T00:00:00Z"},
]


class AnthropicMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: str
    content: Union[str, List[Union[dict, str]]]
    cache_control: Optional[dict] = None


class AnthropicMessagesRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str
    messages: List[AnthropicMessage]
    max_tokens: int = 4096
    system: Optional[Union[str, List[Union[dict, str]]]] = None
    stream: bool = False
    temperature: Optional[float] = None
    tools: Optional[List[dict]] = None
    thinking: Optional[Union[dict, bool]] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    metadata: Optional[dict] = None
    stop_sequences: Optional[List[str]] = None
    conversation_id: Optional[str] = None
    cache_control: Optional[dict] = None


class AnthropicCountTokensRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str
    messages: List[AnthropicMessage]
    system: Optional[Union[str, List[dict]]] = None
    tools: Optional[List[dict]] = None
    thinking: Optional[Union[dict, bool]] = None


def extract_anthropic_content(content: Union[str, List[Union[dict, str]], None]) -> str:
    """从 Anthropic content（字符串或 content block 列表）中提取纯文本。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            b_type = block.get("type")
            if b_type == "text":
                t = block.get("text", "")
                if t:
                    parts.append(t)
            elif b_type == "thinking":
                th = block.get("thinking", "")
                if th:
                    parts.append(f"<thinking>\n{th}\n</thinking>")
            elif b_type == "tool_result":
                sub = block.get("content", "")
                if isinstance(sub, list):
                    sub = extract_anthropic_content(sub)
                t_id = block.get("tool_use_id", "")
                is_err = block.get("is_error", False)
                tag = f"[Tool Error for {t_id}]" if is_err else f"[Tool Result for {t_id}]"
                parts.append(f"{tag}:\n{sub or ''}")
            elif b_type == "tool_use":
                name = block.get("name", "")
                inp = block.get("input", {})
                parts.append(
                    f"```tool_call\n{json.dumps({'name': name, 'arguments': inp}, ensure_ascii=False)}\n```"
                )
            elif b_type == "image":
                source = block.get("source", {})
                media_type = source.get("media_type", "image")
                parts.append(f"[Image: {media_type}]")
            elif "text" in block and block["text"]:
                parts.append(str(block["text"]))
    return "\n".join(parts)


def format_tools_system_prompt(tools: List[dict]) -> str:
    """将 tools 声明（支持 Anthropic 规范与 OpenAI 函数规范）转换为供大模型理解的紧凑工具使用说明。"""
    if not tools:
        return ""
    lines = [
        "## Tools Available\n"
        "You have access to the following tools to accomplish tasks:",
    ]
    for t in tools:
        if not isinstance(t, dict):
            continue
        if t.get("type") == "function" and "function" in t and isinstance(t["function"], dict):
            fn = t["function"]
            name = fn.get("name", "")
            desc = (fn.get("description", "") or "").strip().split("\n")[0][:200]
            schema = fn.get("parameters", {}) or {}
        else:
            name = t.get("name", "")
            desc = (t.get("description", "") or "").strip().split("\n")[0][:200]
            schema = t.get("input_schema", {}) or t.get("parameters", {}) or {}

        props = schema.get("properties", {}) if isinstance(schema, dict) else {}
        req_fields = schema.get("required", []) if isinstance(schema, dict) else []

        args_str_list = []
        if isinstance(props, dict):
            for prop_name, prop_val in props.items():
                prop_type = prop_val.get("type", "any") if isinstance(prop_val, dict) else "any"
                is_req = prop_name in req_fields
                args_str_list.append(f"{prop_name}: {prop_type}{' (required)' if is_req else ''}")

        args_summary = ", ".join(args_str_list) if args_str_list else "None"
        lines.append(f"- `{name}({args_summary})`: {desc}")

    lines.append(
        "\n## Tool Calling Rules\n"
        "1. When you need to read files, write files, run commands, or use any tool, you MUST output a tool call block formatted EXACTLY as:\n"
        "```tool_call\n"
        '{"name": "tool_name", "arguments": {"param1": "value1"}}\n'
        "```\n"
        "2. Only call ONE tool at a time.\n"
        "3. After outputting a tool call, stop and wait for the tool execution result.\n"
        "4. When you receive a `[Tool Result for ...]`, use the result to continue the task."
    )
    return "\n".join(lines)


_BLOCK_TOOL_PATTERN = re.compile(
    r"```+[a-zA-Z0-9_.-]*\s*\n?(.*?)(?:\n?```+|$)",
    re.DOTALL,
)
_XML_TOOL_PATTERN = re.compile(
    r"<(?:tool_call|function_call)(?:\s+[^>]*)?>([\s\S]*?)</(?:tool_call|function_call)>",
    re.DOTALL,
)


def extract_balanced_json_objects(text: str) -> List[Tuple[int, int, str]]:
    """从文本中提取所有括号平衡的最外层 JSON 对象。
    返回 [(start_idx, end_idx, json_str), ...]。
    正确处理嵌套括号、字符串字面量和反斜杠转义，防止嵌套对象被过早截断。
    """
    results: List[Tuple[int, int, str]] = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] == "{":
            start = i
            depth = 0
            in_string = False
            escape = False
            j = i
            while j < n:
                c = text[j]
                if escape:
                    escape = False
                elif c == "\\" and in_string:
                    escape = True
                elif c == '"':
                    in_string = not in_string
                elif not in_string:
                    if c == "{":
                        depth += 1
                    elif c == "}":
                        depth -= 1
                        if depth == 0:
                            results.append((start, j + 1, text[start : j + 1]))
                            i = j
                            break
                j += 1
        i += 1
    return results


def _clean_trailing_commas(s: str) -> str:
    """循环剥离 JSON 对象与数组中尾部多余的逗号：{"a": 1,} -> {"a": 1}。"""
    pattern = re.compile(r",\s*([\]}])")
    prev = ""
    curr = s
    while prev != curr:
        prev = curr
        curr = pattern.sub(r"\1", curr)
    return curr


def _safe_json_loads(s: str) -> Optional[Any]:
    """鲁棒的 JSON 解析器，容忍末尾逗号、未转义控制字符、单引号 Python 字面量及弱格式化。"""
    if not s or not isinstance(s, str):
        return None
    s_clean = s.strip()
    if not s_clean:
        return None

    # 1. 直接解析
    try:
        return json.loads(s_clean, strict=False)
    except Exception:
        pass

    # 2. 递归剥离多余尾部逗号
    try:
        no_trailing = _clean_trailing_commas(s_clean)
        return json.loads(no_trailing, strict=False)
    except Exception:
        pass

    # 3. 尝试作为安全 Python 字面量解析（支持单引号、True/False/None）
    try:
        return ast.literal_eval(s_clean)
    except Exception:
        pass

    # 4. 尝试从混合文本中提取最外层平衡括号对象
    balanced = extract_balanced_json_objects(s_clean)
    for _, _, cand in balanced:
        try:
            return json.loads(cand, strict=False)
        except Exception:
            try:
                no_trailing = _clean_trailing_commas(cand)
                return json.loads(no_trailing, strict=False)
            except Exception:
                try:
                    return ast.literal_eval(cand)
                except Exception:
                    pass

    return None


def _normalize_tool_args(raw_args: Any) -> dict:
    """归一化工具参数为字典。处理字符串、None、空值或复杂嵌套。"""
    if raw_args is None:
        return {}
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str):
        s = raw_args.strip()
        if not s or s in ("{}", "null", "None"):
            return {}
        parsed = _safe_json_loads(s)
        if isinstance(parsed, dict):
            return parsed
        return {"input": s}
    return {}


def _extract_single_tool(item: Any) -> Optional[Tuple[str, dict]]:
    """从单个字典对象中提取工具名称与归一化参数字典。"""
    if not isinstance(item, dict):
        return None

    name = (
        item.get("name")
        or item.get("tool_name")
        or item.get("tool")
        or (item.get("function", {}).get("name") if isinstance(item.get("function"), dict) else None)
        or item.get("action")
    )
    if not name or not isinstance(name, str):
        return None

    raw_args = (
        item.get("arguments")
        if "arguments" in item
        else item.get("args")
        if "args" in item
        else item.get("input")
        if "input" in item
        else item.get("parameters")
        if "parameters" in item
        else (item.get("function", {}).get("arguments") if isinstance(item.get("function"), dict) else None)
        if "function" in item
        else item.get("action_input")
        if "action_input" in item
        else {}
    )
    args = _normalize_tool_args(raw_args)
    return name.strip(), args


def _extract_tools_from_data(data: Any) -> List[Tuple[str, dict]]:
    """从解析后的数据（可能为 dict、list、或嵌套列表）中提取一个或多个工具调用。"""
    tools: List[Tuple[str, dict]] = []
    if isinstance(data, list):
        for item in data:
            tool = _extract_single_tool(item)
            if tool:
                tools.append(tool)
    elif isinstance(data, dict):
        if "tool_calls" in data and isinstance(data["tool_calls"], list):
            for item in data["tool_calls"]:
                tool = _extract_single_tool(item)
                if tool:
                    tools.append(tool)
        elif "tools" in data and isinstance(data["tools"], list):
            for item in data["tools"]:
                tool = _extract_single_tool(item)
                if tool:
                    tools.append(tool)
        else:
            tool = _extract_single_tool(data)
            if tool:
                tools.append(tool)
    return tools


def parse_tool_calls(text: str) -> Optional[Tuple[str, List[Tuple[str, dict]]]]:
    """从模型输出文本中提取所有工具调用（支持多工具调用、列表调用与边界容错）。

    返回 (prefix_text, [(tool_name, arguments), ...]) 或 None。
    """
    if not text or not isinstance(text, str):
        return None

    # 1. 优先匹配代码块 ```tool_call / ```json ...
    first_block_start: Optional[int] = None
    all_tools: List[Tuple[str, dict]] = []
    for match in _BLOCK_TOOL_PATTERN.finditer(text):
        content = match.group(1).strip()
        if not content:
            continue
        parsed = _safe_json_loads(content)
        if parsed is not None:
            tools = _extract_tools_from_data(parsed)
            if tools:
                if first_block_start is None:
                    first_block_start = match.start()
                all_tools.extend(tools)
    if all_tools and first_block_start is not None:
        return text[:first_block_start].strip(), all_tools

    # 2. 匹配 XML 标签 <tool_call>...</tool_call>
    first_xml_start: Optional[int] = None
    all_tools = []
    for match in _XML_TOOL_PATTERN.finditer(text):
        content = match.group(1).strip()
        if not content:
            continue
        parsed = _safe_json_loads(content)
        if parsed is not None:
            tools = _extract_tools_from_data(parsed)
            if tools:
                if first_xml_start is None:
                    first_xml_start = match.start()
                all_tools.extend(tools)
    if all_tools and first_xml_start is not None:
        return text[:first_xml_start].strip(), all_tools

    # 3. 匹配文本中的平衡 JSON 块（正确处理嵌套对象）
    first_json_start: Optional[int] = None
    all_tools = []
    for start, _, cand in extract_balanced_json_objects(text):
        if any(k in cand for k in ('"name"', '"tool_name"', '"tool"', '"function"', '"action"')):
            parsed = _safe_json_loads(cand)
            if parsed is not None:
                tools = _extract_tools_from_data(parsed)
                if tools:
                    if first_json_start is None:
                        first_json_start = start
                    all_tools.extend(tools)
    if all_tools and first_json_start is not None:
        return text[:first_json_start].strip(), all_tools

    return None


def parse_tool_call(text: str) -> Optional[Tuple[str, str, dict]]:
    """向后兼容的单工具解析函数。返回 (prefix_text, tool_name, arguments) 或 None。"""
    res = parse_tool_calls(text)
    if res and res[1]:
        prefix, tools = res
        return prefix, tools[0][0], tools[0][1]
    return None


def build_anthropic_tool_blocks(
    prefix: Optional[str],
    tool_name_or_calls: Union[str, List[Tuple[str, dict]]],
    args: Optional[dict] = None,
    tool_use_id: Optional[str] = None,
) -> List[dict]:
    """构建 Anthropic 规范的 content blocks（可选的 text 前缀 + 1个或多个 tool_use block）。"""
    blocks: List[dict] = []
    if prefix:
        blocks.append({"type": "text", "text": prefix})

    if isinstance(tool_name_or_calls, list):
        for item in tool_name_or_calls:
            t_name, t_args = item
            blocks.append({
                "type": "tool_use",
                "id": _id("toolu", "_")[:22],
                "name": t_name,
                "input": t_args if isinstance(t_args, dict) else {},
            })
    else:
        blocks.append({
            "type": "tool_use",
            "id": tool_use_id or (_id("toolu", "_")[:22]),
            "name": tool_name_or_calls,
            "input": args if isinstance(args, dict) else {},
        })
    return blocks


def convert_anthropic_to_gateway_messages(
    messages: List[AnthropicMessage],
    system: Optional[Union[str, List[dict]]] = None,
    tools: Optional[List[dict]] = None,
) -> List[dict]:
    """将 Anthropic messages 与顶层 system 提示转为标准网关 messages 列表。"""
    sys_parts: list[str] = []
    if system is not None:
        sys_text = extract_anthropic_content(system)
        if sys_text.strip():
            sys_parts.append(sys_text.strip())

    if tools:
        tools_prompt = format_tools_system_prompt(tools)
        if tools_prompt:
            sys_parts.append(tools_prompt)

    # 提取 messages 中的系统消息与对话消息（合并系统提示，避免在 user 后产生迷航 system 块）
    other_msgs: list[dict] = []
    for m in messages:
        text = extract_anthropic_content(m.content)
        if m.role == "system":
            if text.strip():
                sys_parts.append(text.strip())
        else:
            other_msgs.append({"role": m.role, "content": text})

    result: list[dict] = []
    if sys_parts:
        result.append({"role": "system", "content": "\n\n".join(sys_parts)})

    # 如果有工具，且最后一条消息是 user 消息，在末尾追加强有力的工具调用指引
    if tools and other_msgs:
        for idx in range(len(other_msgs) - 1, -1, -1):
            if other_msgs[idx]["role"] == "user":
                tool_hint = (
                    "\n\n[Important Tool Calling Instruction]\n"
                    "If you need to read/write/edit files or run terminal commands, you MUST call the tool directly using:\n"
                    "```tool_call\n"
                    '{"name": "tool_name", "arguments": {"param1": "value1"}}\n'
                    "```\n"
                    "Do NOT refuse or say you lack tools/permissions. Call the tool now!"
                )
                other_msgs[idx]["content"] = other_msgs[idx]["content"] + tool_hint
                break

    result.extend(other_msgs)
    return result


_STATUS_ERROR_MAP = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    429: "rate_limit_error",
    529: "overloaded_error",
}
_ERR_TYPE_MAP = {
    "auth_error": "authentication_error",
    "authentication_error": "authentication_error",
    "model_not_found": "not_found_error",
    "not_found_error": "not_found_error",
    "invalid_request_error": "invalid_request_error",
}


def map_anthropic_error_type(status: int, err_type: Optional[str] = None) -> str:
    """根据 HTTP 状态码与异常类型映射至 Anthropic 标准错误类型枚举。"""
    if err_type and err_type in _ERR_TYPE_MAP:
        return _ERR_TYPE_MAP[err_type]
    return _STATUS_ERROR_MAP.get(status, "api_error")


def anthropic_error_response(message: str, status: int = 500, err_type: Optional[str] = None) -> dict:
    """构建 Anthropic 规范的错误响应字典。"""
    return {
        "type": "error",
        "error": {
            "type": map_anthropic_error_type(status, err_type),
            "message": message,
        },
    }


def anthropic_sse_event(event: str, data: dict) -> str:
    """序列化 Anthropic SSE 事件帧：

    event: <event_name>\n
    data: <json_string>\n\n
    """
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def sse_message_start(
    msg_id: str,
    model: str,
    input_tokens: int,
    cache_creation_input_tokens: Optional[int] = None,
    cache_read_input_tokens: Optional[int] = None,
) -> str:
    usage: dict[str, Any] = {
        "input_tokens": input_tokens,
        "output_tokens": 1,
    }
    if cache_creation_input_tokens is not None:
        usage["cache_creation_input_tokens"] = cache_creation_input_tokens
    if cache_read_input_tokens is not None:
        usage["cache_read_input_tokens"] = cache_read_input_tokens

    return anthropic_sse_event(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": usage,
            },
        },
    )


def sse_content_block_start(index: int, content_block: dict) -> str:
    return anthropic_sse_event(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": index,
            "content_block": content_block,
        },
    )


def sse_content_block_delta(index: int, delta: dict) -> str:
    return anthropic_sse_event(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": index,
            "delta": delta,
        },
    )


def sse_content_block_stop(index: int) -> str:
    return anthropic_sse_event(
        "content_block_stop",
        {
            "type": "content_block_stop",
            "index": index,
        },
    )


def sse_message_delta(
    stop_reason: str,
    output_tokens: int,
    cache_creation_input_tokens: Optional[int] = None,
    cache_read_input_tokens: Optional[int] = None,
) -> str:
    usage: dict[str, Any] = {
        "output_tokens": output_tokens,
    }
    if cache_creation_input_tokens is not None:
        usage["cache_creation_input_tokens"] = cache_creation_input_tokens
    if cache_read_input_tokens is not None:
        usage["cache_read_input_tokens"] = cache_read_input_tokens

    return anthropic_sse_event(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {
                "stop_reason": stop_reason,
                "stop_sequence": None,
            },
            "usage": usage,
        },
    )


def sse_message_stop() -> str:
    return anthropic_sse_event("message_stop", {"type": "message_stop"})


def anthropic_message_response(
    msg_id: str,
    model: str,
    content: Union[str, List[dict]],
    input_tokens: int,
    output_tokens: int,
    stop_reason: str = "end_turn",
    conversation_id: Optional[str] = None,
    cache_creation_input_tokens: Optional[int] = None,
    cache_read_input_tokens: Optional[int] = None,
) -> dict:
    """构建非流式 Anthropic Message 响应对象。"""
    blocks = (
        [{"type": "text", "text": content}]
        if isinstance(content, str)
        else content
    )
    usage: dict[str, Any] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }
    if cache_creation_input_tokens is not None:
        usage["cache_creation_input_tokens"] = cache_creation_input_tokens
    if cache_read_input_tokens is not None:
        usage["cache_read_input_tokens"] = cache_read_input_tokens

    resp = {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": usage,
    }
    if conversation_id:
        resp["conversation_id"] = conversation_id
    return resp


def format_anthropic_models() -> dict:
    """返回 Anthropic 格式的 models 列表。"""
    return {
        "data": [
            {
                "type": "model",
                "id": m["id"],
                "display_name": m["display_name"],
                "created_at": m["created_at"],
            }
            for m in CLAUDE_MODELS_METADATA
        ],
        "has_more": False,
        "first_id": CLAUDE_MODELS_METADATA[0]["id"] if CLAUDE_MODELS_METADATA else None,
        "last_id": CLAUDE_MODELS_METADATA[-1]["id"] if CLAUDE_MODELS_METADATA else None,
    }
