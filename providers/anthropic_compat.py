"""Anthropic Messages API (/v1/messages) 协议兼容层。

负责 Anthropic 请求格式、Content Block、错误响应与 SSE 事件流的标准化转译。
"""

from __future__ import annotations

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
    tool_choice: Optional[Union[str, dict]] = None
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
    # [TOOL-SCHEMA] Anthropic 的 count_tokens 请求与 messages 请求同构，同样可以
    # 携带 tool_choice；估算路径必须看到与真实请求相同的工具提示词，否则算出的
    # token 数在工具循环里系统性偏低，客户端会据此误判上下文预算。
    tool_choice: Optional[Union[str, dict]] = None


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
                # [TOOL-LOOP] 带上调用 id，使多轮回程的 `[Tool Result for <id>]`
                # 能对上是哪一次调用；无 id 时**完全不写该键**，保持既有精确 JSON 断言。
                payload: dict = {}
                t_id = block.get("id")
                if t_id:
                    payload["id"] = t_id
                payload["name"] = name
                payload["arguments"] = inp
                parts.append(
                    f"```tool_call\n{json.dumps(payload, ensure_ascii=False)}\n```"
                )
            elif b_type == "image":
                source = block.get("source", {})
                media_type = source.get("media_type", "image")
                parts.append(f"[Image: {media_type}]")
            elif "text" in block and block["text"]:
                parts.append(str(block["text"]))
    return "\n".join(parts)


# [TOOL-SCHEMA] 工具声明提示词的预算与裁剪常量。
# 上游五家都没有原生工具调用，这份提示词是模型能看到的**唯一** schema，
# 所以保真度就是产品本身；但上下文窗口有限，故保留确定性裁剪与整体预算。
_TOOL_SCHEMA_MAX_DESCRIPTION = 300      # 单个参数/字段 description 的截断长度
_TOOL_SCHEMA_TOTAL_BUDGET = 12000       # 所有工具 schema 渲染文本的总预算（字符）
_TOOL_SCHEMA_DROP_KEYS = ("$schema", "title", "definitions", "$defs")
# 字面量键：其值是**数据**而非子 schema，必须原样保留、不得递归裁剪。
_TOOL_SCHEMA_LITERAL_KEYS = ("default", "const", "enum", "examples")


def _schema_refs(node: Any) -> set:
    """递归收集 schema 中被 `$ref` 引用的定义名，用于决定 definitions/$defs 是否保留。"""
    refs: set = set()
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and "/" in ref:
            refs.add(ref.rsplit("/", 1)[-1])
        for v in node.values():
            refs |= _schema_refs(v)
    elif isinstance(node, list):
        for v in node:
            refs |= _schema_refs(v)
    return refs


def _prune_schema(node: Any, *, keep_defs: bool = False) -> Any:
    """按确定性规则裁剪 JSON Schema，去掉对模型无信息量的键，保持源顺序。

    保留：type/properties/required/enum/items/default/anyOf/oneOf/format/
    minimum/maximum/pattern/description（超长截断）、非 true 的 additionalProperties、
    以及被 `$ref` 实际引用的 definitions/$defs（由 `keep_defs` 控制）。
    """
    if isinstance(node, list):
        return [_prune_schema(v, keep_defs=keep_defs) for v in node]
    if not isinstance(node, dict):
        return node

    out: Dict[str, Any] = {}
    for key, value in node.items():  # 保持源顺序：json.loads 的 dict 顺序即报文顺序
        if key in _TOOL_SCHEMA_DROP_KEYS:
            # definitions/$defs 只在被引用时保留（keep_defs 由顶层判定）
            if key in ("definitions", "$defs") and keep_defs and isinstance(value, dict):
                out[key] = {k: _prune_schema(v, keep_defs=keep_defs) for k, v in value.items()}
            continue
        if key == "additionalProperties":
            # additionalProperties=true 是默认语义，纯噪声；false / schema 有信息量。
            if value is True:
                continue
            out[key] = _prune_schema(value, keep_defs=keep_defs) if isinstance(value, dict) else value
            continue
        if key == "description":
            text = value if isinstance(value, str) else str(value)
            if len(text) > _TOOL_SCHEMA_MAX_DESCRIPTION:
                text = text[:_TOOL_SCHEMA_MAX_DESCRIPTION]
            out[key] = text
            continue
        if key == "properties" and isinstance(value, dict):
            if not value:
                continue
            out[key] = {k: _prune_schema(v, keep_defs=keep_defs) for k, v in value.items()}
            continue
        if key == "required" and isinstance(value, list) and not value:
            continue
        if key in _TOOL_SCHEMA_LITERAL_KEYS:
            # default/const/enum/examples 是数据：原样保留（如 enum 里的对象值）。
            out[key] = value
            continue
        if key in ("properties", "patternProperties") and isinstance(value, dict):
            out[key] = {k: _prune_schema(v, keep_defs=keep_defs) for k, v in value.items()}
            continue
        if key in ("anyOf", "oneOf", "allOf", "prefixItems") and isinstance(value, list):
            out[key] = [_prune_schema(v, keep_defs=keep_defs) for v in value]
            continue
        if key in ("items", "not", "contains", "propertyNames", "if", "then", "else"):
            out[key] = _prune_schema(value, keep_defs=keep_defs)
            continue
        if isinstance(value, dict):
            # 其余形如子 schema 的容器（如 dependentSchemas）也递归裁剪；
            # 若裁剪后为空则丢弃，避免输出 `"unknown":{}` 这类噪声。
            sub = _prune_schema(value, keep_defs=keep_defs)
            if sub != {}:
                out[key] = sub
            continue
        # 标量/数组类约束（format/minimum/pattern/exclusiveMinimum 等）全部原样保留：
        # 未知键宁可保留也不丢信息——提示词是模型唯一能看到的 schema。
        out[key] = value
    return out


def _render_schema_json(schema: Any) -> str:
    """将输入 schema 渲染为紧凑、确定的 JSON（保序、不转义中文）。"""
    if not isinstance(schema, dict):
        return "{}"
    refs = _schema_refs(schema)
    prune_defs = bool(refs)
    pruned = _prune_schema(schema, keep_defs=prune_defs)
    if not isinstance(pruned, dict):
        pruned = {}
    # `{"type":"object"}` 这类无属性 schema 不携带任何信息，避免噪声。
    if pruned in ({"type": "object"}, {}):
        return "{}"
    return json.dumps(pruned, separators=(",", ":"), ensure_ascii=False)


def _compact_args_summary(schema: Any) -> str:
    """旧的展平摘要（`param: type (required)`），仅用于超预算时的降级渲染。"""
    props = schema.get("properties", {}) if isinstance(schema, dict) else {}
    req_fields = schema.get("required", []) if isinstance(schema, dict) else []
    args_str_list = []
    if isinstance(props, dict):
        for prop_name, prop_val in props.items():
            prop_type = prop_val.get("type", "any") if isinstance(prop_val, dict) else "any"
            is_req = prop_name in req_fields
            args_str_list.append(f"{prop_name}: {prop_type}{' (required)' if is_req else ''}")
    return ", ".join(args_str_list) if args_str_list else "None"


def _tool_choice_directive(tool_choice: Any) -> str:
    """把客户端的 tool_choice 约束翻译为一句简短指令；无法识别时返回空串（失败开放）。"""
    if tool_choice is None:
        return ""
    if isinstance(tool_choice, str):
        choice = tool_choice.strip().lower()
        if choice == "none":
            return "Do not call any tool in this turn; answer in plain text."
        if choice in ("required", "any"):
            return "You MUST call at least one tool in this turn."
        # "auto" 及未知字符串：默认行为，不追加指令
        return ""
    if isinstance(tool_choice, dict):
        c_type = tool_choice.get("type")
        if c_type == "any":
            return "You MUST call at least one tool in this turn."
        name = None
        if c_type == "tool":
            name = tool_choice.get("name")
        elif isinstance(tool_choice.get("function"), dict):
            name = tool_choice["function"].get("name")
        if isinstance(name, str) and name.strip():
            return f"You MUST call the tool `{name.strip()}` in this turn, with arguments matching its schema."
    return ""


# [TOOL-LOOP] Responses 的 **custom / freeform** 工具（Codex CLI 的 `apply_patch`
# 就是这种）没有 JSON schema，参数是一段自由文本补丁。上游五家网页端都只能看到
# 提示词里的工具声明，所以这里为它们合成一个**显而易见**的 schema：模型按这个
# 形状写 ```tool_call，网关再把它还原成客户端的 freeform 形状。
#
# 为什么必须合成 schema 而不能只声明名字：`_extract_tools_from_data` 的强校验要求
# 调用带显式参数键，模型若按提示词写出裸字符串（`{"name":"apply_patch","arguments":"*** Begin Patch…"}`），
# 字符串会被 `_normalize_tool_args` 折成 `{"input": ...}`——这正好是我们要的形状，
# 于是「模型怎么理解 freeform」不再是隐式约定。网关契约里没有 freeform 概念：
# 参数永远是对象，所以统一约定成 `{"input": "<原文>"}`，再在出口处还原。
FREEFORM_TOOL_INPUT_KEY = "input"


def freeform_tool_input_schema() -> dict:
    """freeform（custom）工具的合成 schema：唯一字段是放原文的 `input`。"""
    return {
        "type": "object",
        "properties": {
            FREEFORM_TOOL_INPUT_KEY: {
                "type": "string",
                "description": "The complete raw text of this freeform tool's input, verbatim.",
            }
        },
        "required": [FREEFORM_TOOL_INPUT_KEY],
    }


# Responses 里「参数是自由文本」的工具声明类型。OpenAI 给 custom 工具的流式事件
# 与输出项都是 `custom_tool_call*` 这一族；local_shell / shell 同属「没有 function
# 包装」的工具，回放时同样不能当成未知 JSON 丢给模型（见 responses_compat）。
_FREEFORM_TOOL_DECL_TYPES = frozenset({"custom"})


def format_tools_system_prompt(
    tools: List[dict],
    *,
    tool_choice: Any = None,
    parallel_allowed: Any = None,
) -> str:
    """将 tools 声明（支持 Anthropic 规范与 OpenAI 函数规范）转换为供大模型理解的工具使用说明。

    [TOOL-SCHEMA] 每个工具以紧凑 JSON 渲染完整输入 schema（嵌套对象、数组、enum、
    参数描述、required/additionalProperties 等全部保留），只在信息量为零处裁剪：
    丢弃 `$schema`/`title`/`additionalProperties: true`/未被引用的 `definitions`/`$defs`，
    截断超过 300 字符的 description、丢弃空 `properties`/`required`。整体渲染文本超过
    {budget} 字符时，溢出的工具退回旧的 `param: type (required)` 摘要，前 N 个保持保真。

    `tool_choice` 为仅关键字参数，用于追加一条客户端约束指令（none/required/指定工具）；
    无法识别的形状不追加任何内容，绝不抛错。
    """
    if not tools:
        return ""
    lines = [
        "## Tools Available\n"
        "You have access to the following tools to accomplish tasks:",
    ]

    rendered: List[str] = []
    schema_budget_used = 0
    budget_exhausted = False

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
            # [TOOL-LOOP] Responses 的 custom / freeform 声明（`{"type":"custom",
            # "name":"apply_patch"}`）**没有 schema**，原先落到这里就是一个空
            # `{}`，模型被告知「这个工具不接受任何参数」。它于是要么不调用它，要么
            # 把整段补丁写进 `arguments` 字符串——而 `apply_patch` 需要的正是那段
            # 原文。合成 schema 让「怎么传原文」变成显式契约，而不是靠模型猜。
            if not schema and t.get("type") in _FREEFORM_TOOL_DECL_TYPES:
                schema = freeform_tool_input_schema()
                if not desc:
                    desc = "Freeform tool: pass its complete raw input text in the `input` field."

        schema_json = _render_schema_json(schema)
        if budget_exhausted or schema_budget_used + len(schema_json) > _TOOL_SCHEMA_TOTAL_BUDGET:
            # 预算耗尽后的溢出工具：降级为旧摘要。降级是粘性的——一旦触发，
            # 后续工具一律走摘要，保证「前 N 个保真」的确定性切分。
            budget_exhausted = True
            rendered.append(f"- `{name}({_compact_args_summary(schema)})`: {desc}")
        else:
            schema_budget_used += len(schema_json)
            rendered.append(f"- `{name}`: {desc}\n  Args: {schema_json}")

    lines.extend(rendered)

    rules = (
        "\n## Tool Calling Rules\n"
        "1. When you need to read files, write files, run commands, or use any tool, you MUST output a tool call block formatted EXACTLY as:\n"
        "```tool_call\n"
        '{"name": "tool_name", "arguments": {"param1": "value1"}}\n'
        "```\n"
        "2. Only call ONE tool at a time (one tool per fenced block).\n"
        "3. Arguments MUST match the declared schema for that tool (`Args:` JSON above) — "
        "use the exact field names, types, nesting and enum values it declares.\n"
        "4. After outputting a tool call, stop and wait for the tool execution result.\n"
        "5. When you receive a `[Tool Result for <id>]` block, continue the task using that result.\n"
        "6. Never invent tool names that are not listed above; the available tools are exactly those declared in this prompt.\n"
        "7. Do NOT refuse or claim you lack tools/permissions: the tools above are real and callable."
    )
    lines.append(rules)

    directive = _tool_choice_directive(tool_choice)
    if directive:
        lines.append(f"\n## Tool Choice Constraint\n{directive}")

    # [TOOL-LOOP] 客户端明确禁止并行调用时，提示词里必须**说出**这条约束。
    # 解析端仍会裁剪（`cap_parallel_tool_calls`），但先让模型少写一个被丢掉的
    # 调用，比让它写完再被裁掉更省 token，也更少误伤。
    if parallel_allowed is False:
        lines.append(
            "\n## Parallel Tool Calls\n"
            "Do NOT emit more than one tool call in a turn. Call one tool, stop, "
            "and wait for its result before calling the next one."
        )

    return "\n".join(lines)


def format_response_format_prompt(response_format: Any) -> str:
    """[PROTOCOL] 把客户端的 `response_format` 翻译成一条上游能看懂的输出约束。

    网页端模型没有原生结构化输出，`response_format` 此前只是被解析进模型然后丢弃：
    客户端要求 JSON，网关照旧把模型的大白话回给它，客户端按 JSON 去解，整轮报废。
    与 `tool_choice` / `stop` 同一条纪律——约束必须让模型**看见**，能在解析端兜的
    只有「它到底是不是合法 JSON」，而那一步不能替模型决定它该说什么。

    只处理两种有明确语义的取值（`json_object` 与 `json_schema`）；`{"type": "text"}`
    与任何无法识别的形状都不追加内容（失败开放，零回归）。
    """
    if not isinstance(response_format, dict):
        return ""
    rf_type = response_format.get("type")
    if rf_type == "json_object":
        return (
            "## Output Format\n"
            "Respond with a single valid JSON object and nothing else. "
            "Do not wrap it in a markdown code fence, and do not add prose before or after it."
        )
    if rf_type == "json_schema":
        schema = response_format.get("json_schema")
        inner = schema.get("schema") if isinstance(schema, dict) else None
        name = schema.get("name") if isinstance(schema, dict) else None
        if inner is None:
            return (
                "## Output Format\n"
                "Respond with a single valid JSON object and nothing else. "
                "Do not wrap it in a markdown code fence, and do not add prose before or after it."
            )
        label = f" matching the \"{name}\" schema" if isinstance(name, str) and name else ""
        try:
            body = json.dumps(inner, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError):
            return ""
        return (
            "## Output Format\n"
            f"Respond with a single valid JSON object{label} and nothing else. "
            "Do not wrap it in a markdown code fence, and do not add prose before or after it.\n"
            f"Schema: {body}"
        )
    return ""


# [TOOL-LOOP] 围栏语言标签必须落在**可能承载工具调用**的集合内。
# 原正则是 `[a-zA-Z0-9_.-]*`，不锚定标签，于是 ```python / ```ts / ```typescript
# 全都命中，任何带 name 键的普通代码块都会被伪造成工具调用。
#
# 保留 `json` 是有意为之：真实模型经常把工具调用写成 ```json，禁掉它会误伤大量
# 正常调用（本仓库的既有用例即来自实测）。所以防伪造**不靠猜格式**，而靠两道
# 校验——`_extract_single_tool_strict` 要求契约形状，`allowed` 要求名字确实是客户端
# 声明过的工具。格式白名单只负责挡掉 `python`/`ts` 这类显然不相关的代码块。
_BLOCK_TOOL_PATTERN = re.compile(
    r"```+\s*(?:tool_call|tool|function|json|json5)\s*\n(.*?)(?:\n?```+|$)",
    re.DOTALL | re.IGNORECASE,
)
_XML_TOOL_PATTERN = re.compile(
    r"<(?:tool_call|function_call)(?:\s+[^>]*)?>([\s\S]*?)(?:</(?:tool_call|function_call)>|$)",
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

    # 优先使用自愈式容错 JSON 修复引擎
    from .json_repair import safe_loads_with_repair
    parsed, _ = safe_loads_with_repair(s_clean)
    if parsed is not None:
        return parsed

    # 尝试从混合文本中提取最外层平衡括号对象
    balanced = extract_balanced_json_objects(s_clean)
    for _, _, cand in balanced:
        p, _ = safe_loads_with_repair(cand)
        if p is not None:
            return p

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


def _extract_single_tool_strict(item: Any) -> Optional[Tuple[str, dict]]:
    """严格提取：只有**工具调用契约**形状的对象才算工具调用。

    [TOOL-LOOP] 与 `_extract_single_tool` 的区别是「拒绝」而不是「尽力识别」。
    宽松版把任何带 `name` 键的对象都当成调用，参数则用「除保留键之外的所有键」
    兜底，于是普通数据（`{"name": "Alice", "age": 30}`）会被伪造成 `Alice(age=30)`。
    对 agent 来说这是灾难性的：客户端收到一个它从未声明过、也无法执行的工具调用，
    整个回合报废。

    严格版要求满足其一：
      - `{"name": ..., "arguments"|"args"|"parameters"|...: ...}`（网关契约 / OpenAI 形状）
      - `{"function": {"name": ..., "arguments": ...}}`（OpenAI tool_calls 形状）
    并且**不再有兜底参数推断**：没有显式参数键时按空参数处理，而不是把其余键全收。
    """
    if not isinstance(item, dict):
        return None

    fn = item.get("function") if isinstance(item.get("function"), dict) else None
    if fn is not None:
        name = fn.get("name")
        raw_args = fn.get("arguments", fn.get("parameters", {}))
    else:
        name = item.get("name") or item.get("tool_name")
        marker = None
        for key in ("arguments", "args", "parameters", "param", "action_input", "input", "params"):
            if key in item:
                marker = key
                break
        raw_args = item.get(marker) if marker else None
        if marker is None:
            # 没有任何参数键：只有「name + 空对象」这种极窄形状才认，
            # 其余（如 {"name":"Alice","age":30}）一律不是工具调用。
            if set(item.keys()) - {"name", "tool_name", "type", "id"}:
                return None
            raw_args = {}

    if not name or not isinstance(name, str):
        return None
    # 工具名必须是标识符形状：排除 "Alice Smith"、URL、句子等自由文本。
    stripped = name.strip()
    if not stripped or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.:-]{0,127}", stripped):
        return None

    args = _normalize_tool_args(raw_args)
    return stripped, args


def _extract_single_tool(item: Any) -> Optional[Tuple[str, dict]]:
    """从单个字典对象中提取工具名称与归一化参数字典（宽松版，保留向后兼容）。"""
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
        else item.get("params")
        if "params" in item
        else item.get("param")
        if "param" in item
        else item.get("input")
        if "input" in item
        else item.get("parameters")
        if "parameters" in item
        else (item.get("function", {}).get("arguments") if isinstance(item.get("function"), dict) else None)
        if "function" in item
        else item.get("action_input")
        if "action_input" in item
        else {k: v for k, v in item.items() if k not in ("name", "tool_name", "tool", "action", "type")}
    )
    args = _normalize_tool_args(raw_args)
    return name.strip(), args


def _extract_tools_from_data(
    data: Any,
    allowed: Optional[set] = None,
    *,
    strict: bool = True,
) -> List[Tuple[str, dict]]:
    """从解析后的数据（可能为 dict、list、或嵌套列表）中提取一个或多个工具调用。

    `allowed` 为客户端声明的工具名集合；给出时**只接受**集合内的名字。这是
    伪工具调用的最后一道闸：即便模型写出了契约形状，只要名字不是客户端声明过的，
    客户端也只会拒绝整个回合。
    """
    extractor = _extract_single_tool_strict if strict else _extract_single_tool
    tools: List[Tuple[str, dict]] = []
    if isinstance(data, list):
        for item in data:
            tool = extractor(item)
            if tool:
                tools.append(tool)
    elif isinstance(data, dict):
        if "tool_calls" in data and isinstance(data["tool_calls"], list):
            for item in data["tool_calls"]:
                tool = extractor(item)
                if tool:
                    tools.append(tool)
        elif "tools" in data and isinstance(data["tools"], list):
            for item in data["tools"]:
                tool = extractor(item)
                if tool:
                    tools.append(tool)
        else:
            tool = extractor(data)
            if tool:
                tools.append(tool)

    if allowed is not None:
        tools = [(n, a) for (n, a) in tools if n in allowed]
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
    tool_choice: Any = None,
    parallel_allowed: Optional[bool] = None,
) -> List[dict]:
    """将 Anthropic messages 与顶层 system 提示转为标准网关 messages 列表。"""
    sys_parts: list[str] = []
    if system is not None:
        sys_text = extract_anthropic_content(system)
        if sys_text.strip():
            sys_parts.append(sys_text.strip())

    if tools:
        tools_prompt = format_tools_system_prompt(
            tools, tool_choice=tool_choice, parallel_allowed=parallel_allowed
        )
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

    # [TOOL-LOOP] 原先这里会在最后一条 user 消息尾部再追加一段硬编码的
    # `[Important Tool Calling Instruction]`。它与 `format_tools_system_prompt`
    # 的规则段是同一套指令的第二份拷贝，必然随规则演进漂移（改一处、漏一处，
    # 模型就同时看到两条互相矛盾的契约）。现已删除：系统提示词是**唯一**来源，
    # 防拒答句（Do NOT refuse / lack tools）已并入其 Tool Calling Rules 第 7 条。

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
    # [OPT-021] 节点处于运维封禁租约、本次拒绝派发：Anthropic 枚举中最贴近的语义是
    # 「服务当前无法承接该请求，请稍后重试或改用其他模型」，即 overloaded_error。
    "quarantined": "overloaded_error",
    # [OPT-025] 整环凭证失效、请求被快速失败拒绝：Anthropic 枚举中没有「凭证故障」这一类，
    # 语义最贴近的是 authentication_error（本网关确实无法用当前凭证完成认证），
    # 故归入该枚举，便于客户端识别为「需重新认证/联系运维」而非「上游过载」。
    "credential_outage": "authentication_error",
    # [OPT-031] 能力环内全部节点处于长效避让（升级后的人机验证/认证冷却），
    # 本次请求零上游调用拒绝派发：与 quarantined 同源（服务当下无法承接该请求），
    # 故映射到 overloaded_error，客户端可按「稍后重试/改用其他模型」处理。
    "ring_exhausted": "overloaded_error",
    # [OPT-033] 节点被运维通过 DISABLED_PROVIDERS 显式禁用、且环内已无其他可用节点，
    # 本次请求零上游调用拒绝派发：与 quarantined 同属「服务当下无法承接该请求」，
    # 故同样映射到 overloaded_error。
    "disabled": "overloaded_error",
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
    stop_sequence: Optional[str] = None,
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
                # [PROTOCOL] stop_reason="stop_sequence" 时本字段必须是被命中的那个
                # 字符串；写死 None 是线上契约不允许的组合，客户端靠它区分
                # 「被哪个停用词截断」来决定下一段如何续写。
                "stop_sequence": stop_sequence,
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
    stop_sequence: Optional[str] = None,
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
        # [PROTOCOL] 与流式同一契约：stop_sequence 终止原因必须携带命中的字符串
        "stop_sequence": stop_sequence,
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
