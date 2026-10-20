"""OpenAI Responses API (/v1/responses) 协议兼容层。

支持 Codex CLI 及新版 OpenAI SDK Responses 协议标准：
- 请求转译（ResponsesRequest -> gateway_messages）
- 响应格式化（Response 对象、Message Output Item、Function Call Output Item）
- SSE 事件流生成

[TOOL-LOOP] 事件流必须与 OpenAI Responses 的**线上形状**逐字段一致，而不是「大致
相似」：Codex CLI 这类客户端按事件名分派、按 `item_id` 关联增量、按
`sequence_number` 判断顺序，任何一处不同都会让客户端丢弃整段输出（表现为「agent
连上了但不说话」或「工具调用没跑起来」）。本模块因此用 `ResponsesEventStream`
作为**唯一**的事件构造入口：

  response.created
    -> response.in_progress
    -> response.output_item.added            (item: message | function_call | reasoning)
    -> response.content_part.added           (part: {"type": "output_text"})
    -> response.output_text.delta  ...       (逐字增量)
    -> response.output_text.done
    -> response.content_part.done
    -> response.output_item.done
    -> response.function_call_arguments.delta / .done   (工具调用)
    -> response.completed
    -> [DONE]

历史缺陷（实测）：
  1. 文本增量事件名是 `response.text.delta`，现行规范是
     `response.output_text.delta`——按名分派的客户端收不到任何文本。
  2. 所有事件都缺 `type` 字段，而 SDK 正是按 body 里的 `type`（而非 SSE 的
     `event:` 行）分派。
  3. `content_part.added` / 增量事件缺 `item_id`，客户端无法把增量挂到 item 上。
  4. 消息内容块类型写成 `text`，规范是 `output_text`；`annotations` 缺失。
  5. 没有任何 `sequence_number`，且思考流（reasoning）被整段丢弃。
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, ConfigDict

from .anthropic_compat import (
    FREEFORM_TOOL_INPUT_KEY,
    _FREEFORM_TOOL_DECL_TYPES,
    format_response_format_prompt,
    format_tools_system_prompt,
)
from .base import (
    FILE_PLACEHOLDER_TEMPLATE,
    IMAGE_PLACEHOLDER,
    _est_tokens,
    _id,
    error_response,
)


class ResponsesRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    input: Optional[Union[str, List[Any]]] = None
    instructions: Optional[Union[str, List[dict]]] = None
    stream: bool = False
    temperature: Optional[float] = None
    tools: Optional[List[dict]] = None
    tool_choice: Optional[Union[str, dict]] = None
    parallel_tool_calls: Optional[bool] = None
    # [PROTOCOL] OpenAI 生态通用字段：Responses 客户端同样会带 stop
    stop: Optional[Union[str, List[str]]] = None
    # [PROTOCOL] Responses 的结构化输出在 `text.format` 里（旧客户端用 response_format）
    text: Optional[dict] = None
    response_format: Optional[dict] = None
    max_output_tokens: Optional[int] = None
    max_tokens: Optional[int] = None
    conversation_id: Optional[str] = None
    reasoning_effort: Optional[str] = None
    # [AGENT-LOOP] Codex CLI 发的是 `{"reasoning": {"effort": "high", ...}}` 对象，
    # 而不是 `reasoning_effort` 字符串。此前只认后者，模型名又恰好落在
    # `gpt-5-codex` 这样的非推理前缀上，于是「想清楚再动手」的意图被静默丢弃。
    reasoning: Optional[Union[dict, str]] = None
    thinking: Optional[Union[dict, bool]] = None


def _responses_text_format(req: "ResponsesRequest") -> Optional[dict]:
    """从 Responses 请求里取出结构化输出约束（`text.format` 优先，兼容 `response_format`）。

    Responses 把 OpenAI 的 `response_format` 搬到了 `text.format`：
    `{"text": {"format": {"type": "json_schema", "name": ..., "schema": ...}}}`。
    这里把它改写成 `format_response_format_prompt` 认识的老形状再复用，避免同一套
    翻译逻辑存在两份（本仓库反复出现的漂移缺陷类型）。
    """
    if isinstance(req.text, dict):
        fmt = req.text.get("format")
        if isinstance(fmt, dict):
            f_type = fmt.get("type")
            if f_type == "json_schema":
                return {
                    "type": "json_schema",
                    "json_schema": {
                        "name": fmt.get("name"),
                        "schema": fmt.get("schema"),
                    },
                }
            if isinstance(f_type, str):
                return {"type": f_type}
    if isinstance(req.response_format, dict):
        return req.response_format
    return None


def extract_responses_content(content: Any) -> str:
    """从 Responses input content（字符串、列表或块结构）中提取文本内容。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        b_type = content.get("type", "")
        if b_type in ("text", "input_text", "output_text"):
            t = content.get("text")
            return t if isinstance(t, str) else ""
        if "text" in content:
            return str(content["text"]) if content["text"] is not None else ""
        return json.dumps(content, ensure_ascii=False)
    if not isinstance(content, list):
        return str(content)

    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            b_type = block.get("type", "")
            if b_type in ("text", "input_text", "output_text"):
                t = block.get("text")
                if isinstance(t, str) and t:
                    parts.append(t)
            elif b_type in ("input_image", "image_url", "image"):
                # [PROTOCOL] 上游是纯文本网页接口，图片无法送达；静默丢弃会让模型
                # 对着一个它从未看到的附件作答（详见 providers/base.py 的说明）。
                parts.append(IMAGE_PLACEHOLDER)
            elif b_type in ("input_file", "file", "document"):
                name = block.get("filename") or block.get("file_id") or ""
                parts.append(FILE_PLACEHOLDER_TEMPLATE.format(name=name or "unnamed"))
            elif b_type == "thinking":
                th = block.get("thinking", "")
                if th:
                    parts.append(f"<thinking>\n{th}\n</thinking>")
            elif b_type in ("summary_text", "reasoning_text"):
                t = block.get("text", "")
                if t:
                    parts.append(f"<thinking>\n{t}\n</thinking>")
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
                # [TOOL-LOOP] 带上调用 id：客户端回传的历史里，工具结果只带
                # call_id，若调用侧不写 id，模型看到的就是一段无主的输出。
                payload: dict[str, Any] = {"name": name, "arguments": args}
                c_id = block.get("call_id") or block.get("id")
                if c_id:
                    payload["id"] = c_id
                parts.append(f"```tool_call\n{json.dumps(payload, ensure_ascii=False)}\n```")
            elif "text" in block and block["text"]:
                parts.append(str(block["text"]))
    return "\n".join(parts)


def _reasoning_item_text(item: dict) -> str:
    """从 Responses 的 reasoning output item 中取出可回放的思考文本。

    Codex 的 summary 形状是 `[{"type": "summary_text", "text": ...}]`，部分客户端
    用 `content: [{"type": "reasoning_text", "text": ...}]`；`encrypted_content`
    是密文，上游无法理解，忽略。
    """
    parts: list[str] = []
    for key in ("summary", "content"):
        blocks = item.get(key)
        if isinstance(blocks, str):
            if blocks.strip():
                parts.append(blocks)
        elif isinstance(blocks, list):
            for b in blocks:
                if isinstance(b, dict) and isinstance(b.get("text"), str) and b["text"].strip():
                    parts.append(b["text"])
                elif isinstance(b, str) and b.strip():
                    parts.append(b)
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
        tools_prompt = format_tools_system_prompt(
            req.tools,
            tool_choice=req.tool_choice,
            parallel_allowed=req.parallel_tool_calls is not False,
        )
        if tools_prompt:
            system_parts.append(tools_prompt)
    # [PROTOCOL] `text.format` 是 Responses 的 response_format，同样是必须被模型看见的约束
    fmt_prompt = format_response_format_prompt(_responses_text_format(req))
    if fmt_prompt:
        system_parts.append(fmt_prompt)

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
                elif item_type == "reasoning":
                    # [AGENT-LOOP] Codex CLI 会在下一轮把上一轮的**全部** output items
                    # 原样回放，其中包含 `type:"reasoning"`。此前它落进最后的兜底分支，
                    # 被 `extract_responses_content` 序列化成一段 JSON 当成 **user 输入**
                    # 发给上游（实测：user 消息内容为
                    # `{"type": "reasoning", "id": "rs_1", "summary": [...]}`）——模型
                    # 看到一段自己不认识的 JSON，上下文被污染且每轮白烧 token。
                    # 思考回放归入 assistant 的 `<thinking>` 块，空思考直接跳过。
                    think = _reasoning_item_text(item)
                    if think.strip():
                        messages.append({
                            "role": "assistant",
                            "content": f"<thinking>\n{think.strip()}\n</thinking>",
                        })
                elif item_type in ("item_reference", "compaction"):
                    # 存储型引用（本网关无 responses 存储，无法解析）与压缩占位：
                    # 它们**不是内容**，跳过远好于把原始 JSON 当成用户输入。
                    continue
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
                    payload: dict[str, Any] = {"name": name, "arguments": args}
                    c_id = item.get("call_id") or item.get("id")
                    if c_id:
                        payload["id"] = c_id
                    call_text = f"```tool_call\n{json.dumps(payload, ensure_ascii=False)}\n```"
                    messages.append({"role": "assistant", "content": call_text})
                elif item_type in ("custom_tool_call", "local_shell_call", "shell_call"):
                    # [TOOL-LOOP] Codex CLI 的 `apply_patch` 等 freeform 工具回放的是
                    # `custom_tool_call`，参数是**纯文本** `input`（不是 JSON
                    # `arguments`）。原先这一支没有分支，落进最后的兜底，被序列化成
                    # 一段原始 JSON 当成 **user 输入**：模型看到自己从未写过的补丁
                    # JSON，既认不出这是自己的调用，也接不上随后的执行结果。
                    # 现在按网关契约回放成 ```tool_call + {"input": "<原文>"}，与
                    # 提示词里为 freeform 合成的 schema 严格一致。
                    name = item.get("name") or item.get("action", {}).get("type") or ""
                    if not name and item_type in ("local_shell_call", "shell_call"):
                        name = "local_shell"
                    raw_input = item.get("input")
                    if raw_input is None and isinstance(item.get("action"), dict):
                        raw_input = item["action"]
                    if isinstance(raw_input, str):
                        args_obj: dict = {FREEFORM_TOOL_INPUT_KEY: raw_input}
                    elif isinstance(raw_input, dict):
                        args_obj = raw_input
                    elif raw_input is None:
                        args_obj = {}
                    else:
                        args_obj = {FREEFORM_TOOL_INPUT_KEY: json.dumps(raw_input, ensure_ascii=False)}
                    payload: dict[str, Any] = {"name": name, "arguments": args_obj}
                    c_id = item.get("call_id") or item.get("id")
                    if c_id:
                        payload["id"] = c_id
                    messages.append({
                        "role": "assistant",
                        "content": f"```tool_call\n{json.dumps(payload, ensure_ascii=False)}\n```",
                    })
                elif item_type in ("custom_tool_call_output", "local_shell_call_output"):
                    # freeform 调用的执行结果：`output` 同样可以是字符串或结构。
                    c_id = item.get("call_id", "")
                    out = item.get("output", "")
                    if isinstance(out, (dict, list)):
                        out = json.dumps(out, ensure_ascii=False)
                    tag = f"[Tool Result for {c_id}]" if c_id else "[Tool Result]"
                    messages.append({"role": "user", "content": f"{tag}:\n{out}"})
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
                                tc_payload: dict[str, Any] = {"name": fn_name, "arguments": fn_args}
                                tc_id = tc.get("id")
                                if tc_id:
                                    tc_payload["id"] = tc_id
                                tc_parts.append(
                                    f"```tool_call\n{json.dumps(tc_payload, ensure_ascii=False)}\n```"
                                )
                            if tc_parts:
                                c_text = (c_text + "\n" + "\n".join(tc_parts)).strip()
                        messages.append({"role": role, "content": c_text})
                else:
                    c_text = extract_responses_content(item)
                    if c_text:
                        messages.append({"role": "user", "content": c_text})

    return messages


# ---------------------------------------------------------------- 响应对象

def _usage(input_tokens: int, output_tokens: int, reasoning_tokens: int = 0) -> dict:
    """Responses 规范的 usage：除总数外还带 *_tokens_details 子对象。"""
    return {
        "input_tokens": input_tokens,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": output_tokens,
        "output_tokens_details": {"reasoning_tokens": reasoning_tokens},
        "total_tokens": input_tokens + output_tokens,
    }


def message_item(
    msg_id: str,
    text: str = "",
    *,
    status: str = "completed",
) -> dict:
    """构造 assistant message 输出项（内容块类型为规范的 `output_text`）。"""
    return {
        "id": msg_id,
        "type": "message",
        "status": status,
        "role": "assistant",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def reasoning_item(rs_id: str, text: str = "", *, status: str = "completed") -> dict:
    """构造 reasoning 输出项；思考内容以 summary 呈现（Codex 的展示方式）。"""
    return {
        "id": rs_id,
        "type": "reasoning",
        "status": status,
        "summary": [{"type": "summary_text", "text": text}] if text else [],
    }


def function_call_item(
    call_id: str,
    name: str,
    arguments: str = "",
    *,
    status: str = "completed",
) -> dict:
    return {
        "id": call_id,
        "type": "function_call",
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
        "status": status,
    }


# [TOOL-LOOP] Responsese 的 custom / freeform 工具族（Codex CLI 的 `apply_patch`
# 是其中最常用的一个）。它们的输出项 `type` 是 `custom_tool_call`、参数是**一段
# 自由文本** `input`，而不是 `function_call` + JSON `arguments`。网关若把两者一并
# 当成 `function_call` 回传，客户端按 `type` 分派时找不到自己声明的工具，本轮调用
# 被静默丢弃——这正是「Codex 接上网关后不会改文件」的形态。
_CUSTOM_TOOL_ITEM_TYPE = "custom_tool_call"
_CUSTOM_TOOL_OUTPUT_ITEM_TYPE = "custom_tool_call_output"


def custom_tool_call_item(
    call_id: str,
    name: str,
    tool_input: str = "",
    *,
    status: str = "completed",
) -> dict:
    """构造 `custom_tool_call` 输出项（freeform 工具的唯一正确出口形状）。"""
    return {
        "id": call_id,
        "type": _CUSTOM_TOOL_ITEM_TYPE,
        "call_id": call_id,
        "name": name,
        "input": tool_input,
        "status": status,
    }


def freeform_tool_names(tools: Any) -> set:
    """客户端声明的 freeform（custom）工具名集合。

    只认 `{"type": "custom", "name": ...}` —— 这是 Responses 的 freeform 声明形状。
    `local_shell` / `shell` 也走自由文本出口（OpenAI 的 `local_shell_call` 项同样
    不是 `function_call`），但它们没有客户端声明的 `name`，由调用方按 `type` 单独处理。
    """
    names: set = set()
    if not isinstance(tools, list):
        return names
    for t in tools:
        if not isinstance(t, dict):
            continue
        if t.get("type") in _FREEFORM_TOOL_DECL_TYPES:
            name = t.get("name")
            if isinstance(name, str) and name.strip():
                names.add(name.strip())
    return names


def custom_tool_input_text(args: Any) -> str:
    """把网关契约里的参数对象还原成 freeform 工具的原始文本。

    [TOOL-LOOP] 约定是 `{"input": "<原文>"}`（见 `freeform_tool_input_schema`）。
    模型偶尔会把原文直接写成字符串参数，`_normalize_tool_args` 会折成同样的
    `{"input": ...}`；再偶尔会把补丁本身写成一个整串 arguments。三种形状都要
    还原成原文，而不是把一段 JSON 包在字符串里发给客户端（`apply_patch` 收到
    带引号和 `\\n` 的包装文本会直接解析失败）。
    """
    if isinstance(args, str):
        # [TOOL-LOOP] 非流式路径上 `arguments` 是 JSON 字符串（`{"input": "..."}`），
        # 直接原样返回会把一段带引号和 `\n` 转义的包装文本交给 apply_patch，它解析
        # 失败后整轮改动丢失。能解析成对象就按对象契约解包；裸补丁本身不是合法
        # JSON，解析失败时才是真正的「原文整串」。
        stripped = args.strip()
        if stripped.startswith("{"):
            try:
                parsed = json.loads(stripped)
            except (ValueError, TypeError):
                parsed = None
            if isinstance(parsed, dict):
                return custom_tool_input_text(parsed)
        return args
    if isinstance(args, dict):
        if FREEFORM_TOOL_INPUT_KEY in args:
            val = args[FREEFORM_TOOL_INPUT_KEY]
            if isinstance(val, str):
                return val
            return json.dumps(val, ensure_ascii=False) if val is not None else ""
        if len(args) == 1:
            only = next(iter(args.values()))
            if isinstance(only, str):
                return only
        return json.dumps(args, ensure_ascii=False)
    if args is None:
        return ""
    return str(args)


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
    reasoning_text: str = "",
    reasoning_id: Optional[str] = None,
    reasoning_tokens: int = 0,
    freeform_tools: Optional[set] = None,
) -> dict:
    """构造标准的 Responses API JSON 对象。

    `freeform_tools` 是客户端声明的 custom（freeform）工具名集合：命中它们的调用
    必须以 `custom_tool_call` 输出（参数是纯文本 `input`），而不是 `function_call`。
    """
    created = created_at or int(time.time())
    m_id = msg_id or _id("msg", "_")

    output_items: list[dict] = []
    if reasoning_text:
        output_items.append(reasoning_item(reasoning_id or _id("rs", "_"), reasoning_text))
    if (output_text and output_text.strip()) or not tool_calls:
        output_items.append(message_item(m_id, output_text or ""))

    if tool_calls:
        for tc in tool_calls:
            c_id = tc.get("id") or tc.get("call_id") or _id("call", "_")
            fn = tc.get("function", {}) if isinstance(tc.get("function"), dict) else {}
            fn_name = tc.get("name") or fn.get("name", "")
            fn_args = tc.get("arguments") if "arguments" in tc else fn.get("arguments", "")
            # [TOOL-LOOP] freeform 工具走 `custom_tool_call`：客户端按 `type` 分派，
            # 把 apply_patch 报成 function_call 会让它找不到工具、静默丢弃本轮调用。
            if freeform_tools and fn_name in freeform_tools:
                output_items.append(
                    custom_tool_call_item(c_id, fn_name, custom_tool_input_text(fn_args))
                )
                continue
            if isinstance(fn_args, (dict, list)):
                fn_args = json.dumps(fn_args, ensure_ascii=False)
            elif fn_args is None:
                fn_args = "{}"
            output_items.append(function_call_item(c_id, fn_name, str(fn_args)))

    resp = {
        "id": resp_id,
        "object": "response",
        "created_at": created,
        "status": status,
        "model": model,
        "output": output_items,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "usage": _usage(input_tokens, output_tokens, reasoning_tokens),
    }
    if conversation_id:
        resp["conversation_id"] = conversation_id
    if failover:
        resp["failover"] = failover
    return resp


# ---------------------------------------------------------------- SSE 事件流

class ResponsesEventStream:
    """单个 /v1/responses 请求的事件构造器（唯一实现）。

    为什么是类而不是一堆自由函数：`sequence_number` 必须单调递增，`item_id` 必须
    在 added / delta / done 之间保持一致。这些是**跨事件的状态**，用自由函数实现
    就只能由调用方各自维护一份，最终必然漂移（本仓库反复出现的缺陷类型）。
    """

    def __init__(self, resp_id: str, model: str, created_at: Optional[int] = None):
        self.resp_id = resp_id
        self.model = model
        self.created_at = created_at or int(time.time())
        self.seq = 0

    # -- 内部 ---------------------------------------------------------------
    def _frame(self, event: str, data: dict) -> str:
        data["type"] = event
        data["sequence_number"] = self.seq
        self.seq += 1
        return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    def _response_shell(self, status: str, output: Optional[list] = None) -> dict:
        return {
            "id": self.resp_id,
            "object": "response",
            "created_at": self.created_at,
            "status": status,
            "model": self.model,
            "output": output or [],
        }

    # -- 生命周期 -----------------------------------------------------------
    def created(self) -> str:
        return self._frame("response.created", {"response": self._response_shell("in_progress")})

    def in_progress(self) -> str:
        return self._frame("response.in_progress", {"response": self._response_shell("in_progress")})

    def output_item_added(self, output_index: int, item: dict) -> str:
        return self._frame(
            "response.output_item.added", {"output_index": output_index, "item": item}
        )

    def content_part_added(
        self,
        item_id: str,
        output_index: int,
        content_index: int,
        part: Optional[dict] = None,
    ) -> str:
        return self._frame(
            "response.content_part.added",
            {
                "item_id": item_id,
                "output_index": output_index,
                "content_index": content_index,
                "part": part or {"type": "output_text", "text": "", "annotations": []},
            },
        )

    def output_text_delta(
        self,
        item_id: str,
        delta: str,
        output_index: int = 0,
        content_index: int = 0,
    ) -> str:
        return self._frame(
            "response.output_text.delta",
            {
                "item_id": item_id,
                "output_index": output_index,
                "content_index": content_index,
                "delta": delta,
                "logprobs": [],
            },
        )

    def output_text_done(
        self,
        item_id: str,
        text: str,
        output_index: int = 0,
        content_index: int = 0,
    ) -> str:
        return self._frame(
            "response.output_text.done",
            {
                "item_id": item_id,
                "output_index": output_index,
                "content_index": content_index,
                "text": text,
                "logprobs": [],
            },
        )

    def content_part_done(
        self,
        item_id: str,
        full_text: str,
        output_index: int = 0,
        content_index: int = 0,
    ) -> str:
        return self._frame(
            "response.content_part.done",
            {
                "item_id": item_id,
                "output_index": output_index,
                "content_index": content_index,
                "part": {"type": "output_text", "text": full_text, "annotations": []},
            },
        )

    def output_item_done(self, output_index: int, item: dict) -> str:
        return self._frame(
            "response.output_item.done", {"output_index": output_index, "item": item}
        )

    # -- 思考流 -------------------------------------------------------------
    def reasoning_summary_delta(
        self, item_id: str, delta: str, output_index: int = 0, summary_index: int = 0
    ) -> str:
        return self._frame(
            "response.reasoning_summary_text.delta",
            {
                "item_id": item_id,
                "output_index": output_index,
                "summary_index": summary_index,
                "delta": delta,
            },
        )

    def reasoning_summary_done(
        self, item_id: str, text: str, output_index: int = 0, summary_index: int = 0
    ) -> str:
        return self._frame(
            "response.reasoning_summary_text.done",
            {
                "item_id": item_id,
                "output_index": output_index,
                "summary_index": summary_index,
                "text": text,
            },
        )

    # -- 工具调用 -----------------------------------------------------------
    def function_call_args_delta(
        self, item_id: str, output_index: int, call_id: str, delta: str
    ) -> str:
        return self._frame(
            "response.function_call_arguments.delta",
            {
                "item_id": item_id,
                "output_index": output_index,
                "call_id": call_id,
                "delta": delta,
            },
        )

    def function_call_args_done(
        self, item_id: str, output_index: int, call_id: str, arguments: str
    ) -> str:
        return self._frame(
            "response.function_call_arguments.done",
            {
                "item_id": item_id,
                "output_index": output_index,
                "call_id": call_id,
                "arguments": arguments,
            },
        )

    # -- freeform 工具族（custom_tool_call）----------------------------------
    def custom_tool_call_input_delta(
        self, item_id: str, output_index: int, delta: str, call_id: str = ""
    ) -> str:
        """`custom_tool_call_input.delta`：freeform 工具的原文增量。

        [TOOL-LOOP] freeform 工具没有 JSON arguments，它的事件名与 `function_call`
        完全不同。把 apply_patch 的补丁当成 `function_call_arguments.delta` 发出去，
        客户端会尝试按 JSON 增量解析一段补丁文本并失败，本轮调用丢失。
        """
        return self._frame(
            "response.custom_tool_call_input.delta",
            {
                "item_id": item_id,
                "output_index": output_index,
                "call_id": call_id,
                "delta": delta,
            },
        )

    def custom_tool_call_input_done(
        self, item_id: str, output_index: int, tool_input: str, call_id: str = ""
    ) -> str:
        return self._frame(
            "response.custom_tool_call_input.done",
            {
                "item_id": item_id,
                "output_index": output_index,
                "call_id": call_id,
                "input": tool_input,
            },
        )

    # -- 终局 ---------------------------------------------------------------
    def completed(
        self,
        output: list,
        input_tokens: int,
        output_tokens: int,
        *,
        reasoning_tokens: int = 0,
        failover: Optional[dict] = None,
        conversation_id: Optional[str] = None,
        incomplete_reason: Optional[str] = None,
    ) -> str:
        resp_obj = self._response_shell("completed", output)
        if incomplete_reason:
            # [PROTOCOL] Responses 的 status 枚举里没有 "stop_sequence"：被 client 的
            # stop 截断属于 "incomplete" + reason "stop"，与"自然说完"必须可区分，
            # 否则客户端无法判断这一轮是否被网关截短了。
            resp_obj["status"] = "incomplete"
            resp_obj["incomplete_details"] = {"reason": incomplete_reason}
        resp_obj["usage"] = _usage(input_tokens, output_tokens, reasoning_tokens)
        if conversation_id:
            resp_obj["conversation_id"] = conversation_id
        if failover:
            resp_obj["failover"] = failover
        data: dict[str, Any] = {"response": resp_obj}
        if failover:
            data["failover"] = failover
        return self._frame("response.completed", data)

    def error(self, message: str, status: int = 502, err_type: str = "upstream_error") -> str:
        """Responses 规范的错误事件：`type:error` + code/message/param。"""
        err = error_response(message, status, err_type)["error"]
        return self._frame(
            "error",
            {
                "code": err.get("type") or err_type,
                "message": err.get("message") or message,
                "param": None,
            },
        )


def sse_done() -> str:
    """Responses 流的终结帧。"""
    return "data: [DONE]\n\n"


# ---------------------------------------------------------------- 兼容层
# 以下自由函数保留给仍按「单帧」方式调用的外部代码；网关自身只使用
# `ResponsesEventStream`，避免第二份实现与线上形状漂移。

def _one(response: str) -> str:
    return response


def sse_response_created(resp_id: str, model: str, created_at: Optional[int] = None) -> str:
    return ResponsesEventStream(resp_id, model, created_at).created()


def sse_response_output_item_added(resp_id: str, output_index: int, item: dict) -> str:
    return ResponsesEventStream(resp_id, "").output_item_added(output_index, item)


def sse_response_content_part_added(
    resp_id: str,
    output_index: int,
    content_index: int,
    part: Optional[dict] = None,
    item_id: Optional[str] = None,
) -> str:
    return ResponsesEventStream(resp_id, "").content_part_added(
        item_id or f"msg_{resp_id}", output_index, content_index, part
    )


def sse_response_text_delta(
    delta: str,
    resp_id: Optional[str] = None,
    output_index: int = 0,
    content_index: int = 0,
    item_id: Optional[str] = None,
) -> str:
    """文本增量。名字沿用旧名，线上事件名是 `response.output_text.delta`。"""
    return ResponsesEventStream(resp_id or "", "").output_text_delta(
        item_id or f"msg_{resp_id}", delta, output_index, content_index
    )


def sse_response_content_part_done(
    full_text: str,
    resp_id: Optional[str] = None,
    output_index: int = 0,
    content_index: int = 0,
    item_id: Optional[str] = None,
) -> str:
    return ResponsesEventStream(resp_id or "", "").content_part_done(
        item_id or f"msg_{resp_id}", full_text, output_index, content_index
    )


def sse_response_output_item_done(
    item: dict,
    resp_id: Optional[str] = None,
    output_index: int = 0,
) -> str:
    return ResponsesEventStream(resp_id or "", "").output_item_done(output_index, item)


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
    return ResponsesEventStream(resp_id, model, created_at).completed(
        output,
        input_tokens,
        output_tokens,
        failover=failover,
        conversation_id=conversation_id,
    )


def sse_response_function_call_args_delta(
    resp_id: str,
    output_index: int,
    call_id: str,
    delta: str,
) -> str:
    return ResponsesEventStream(resp_id, "").function_call_args_delta(
        call_id, output_index, call_id, delta
    )


def sse_response_function_call_args_done(
    resp_id: str,
    output_index: int,
    call_id: str,
    arguments: str,
) -> str:
    return ResponsesEventStream(resp_id, "").function_call_args_done(
        call_id, output_index, call_id, arguments
    )


def sse_response_done() -> str:
    return sse_done()


def sse_response_error(message: str, status: int = 502, err_type: str = "upstream_error") -> str:
    return ResponsesEventStream("", "").error(message, status, err_type)