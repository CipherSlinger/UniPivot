"""Anthropic Messages API (/v1/messages) 协议兼容层。

负责 Anthropic 请求格式、Content Block、错误响应与 SSE 事件流的标准化转译。
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, ConfigDict, Field, field_validator
from .base import (
    FILE_PLACEHOLDER_TEMPLATE,
    IMAGE_PLACEHOLDER,
    _id,
    content_omitted,
)

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


class AnthropicRequestBody(BaseModel):
    """`/v1/messages` 与 `/v1/messages/count_tokens` 共享的请求体定义。

    [PROTOCOL] 这两个端点描述的是**同一个** Anthropic 请求体，原先却各有一份
    模型，且 `system` 的类型不同（`List[dict]` vs `List[Union[dict, str]]`）。
    于是 `"system": ["you are terse"]` 在 messages 上合法、在 count_tokens 上被
    400 拒绝——而 Claude Code 每次请求前都用 count_tokens 做上下文预算判断，
    400 只会表现为「agent 不动了」。同一份报文有第二份解释，必然漂移；这里只留
    一份，端点各自的特有字段放在子类里。
    """

    model_config = ConfigDict(extra="allow")
    model: str
    messages: List[AnthropicMessage]
    system: Optional[Union[str, List[Union[dict, str]]]] = None
    tools: Optional[List[dict]] = None
    # [PROTOCOL] 与 `_wants_thinking()` 的接受范围保持一致：`"adaptive"` 裸字符串
    # 是 `{"type":"adaptive"}` 的另一种序列化，同一份声明在三条路径上必须得到
    # 同一个结果，否则客户端换个序列化方式就会撞上 400。
    thinking: Optional[Union[dict, bool, str]] = None
    # [TOOL-SCHEMA] Anthropic 的 count_tokens 请求与 messages 请求同构，同样可以
    # 携带 tool_choice；估算路径必须看到与真实请求相同的工具提示词，否则算出的
    # token 数在工具循环里系统性偏低，客户端会据此误判上下文预算。
    tool_choice: Optional[Union[str, dict]] = None

    @field_validator("tools")
    @classmethod
    def _flatten_namespaces(cls, tools: Optional[List[dict]]):
        """`namespace` 容器展平（对普通声明逐值不变，见 `flatten_namespace_tools`）。

        Anthropic 协议本身没有 `namespace`，但「声明 → 提示词 / 白名单」这条链是
        三个协议共用的；把展平放在请求模型上，保证任何入口的 `tools` 都只有一种
        解释——否则同一条声明在提示词里展平、在白名单里没展平，模型写出的子工具
        调用会被当作未声明工具丢弃（症状与修复前完全一样，只是更难追）。
        """
        return flatten_declared_tools(tools)


class AnthropicMessagesRequest(AnthropicRequestBody):
    max_tokens: int = 4096
    stream: bool = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    metadata: Optional[dict] = None
    stop_sequences: Optional[List[str]] = None
    conversation_id: Optional[str] = None
    cache_control: Optional[dict] = None


class AnthropicCountTokensRequest(AnthropicRequestBody):
    """count_tokens 与 messages 同构；独有字段为空（估算只读 messages/system/tools）。"""


def resolve_client_session_id(
    conversation_id: Optional[str], metadata: Optional[dict]
) -> Optional[str]:
    """本请求的**客户端线程标识**——粘滞、续聊判定、匿名会话回收都以它为键。

    [SESSION-ID] Claude Code 从不发送 `conversation_id`：实测 19 份真实抓包
    （`/tmp/e2e/claude_bodies.jsonl`、`/tmp/e2e/anth/rr_bodies.jsonl`、
    `/tmp/e2e/think_replayed.jsonl`、`/tmp/e2e/sigprobe/client_bodies.jsonl`）
    里该键出现 **0 次**，会话身份全部放在 `metadata.user_id` 这个 JSON 字符串里：
    `{"device_id":"29c47c8b…","account_uuid":"","session_id":"<uuid>"}`，
    其中 `device_id` 跨会话恒定、`session_id` 每个会话一个。

    网关此前只读 `req.conversation_id`，于是对 Claude Code 而言这条身份
    **一个字符都没被读过**，后果全部可测（同一份抓包驱动 5 轮）：
    上游收到的 `conversation_id` 恒为 `None`（每轮新建会话）、每轮重发全量展平
    历史（20852/21643/22434/23225/24016 字符，合计 112,170，而带上身份只需
    20852+4×10 = 20,892）、每轮把本该续用的上游会话当成匿名临时会话排进删除队列。

    `conversation_id` 优先——它是显式字段，语义无歧义；`metadata` 是兜底，
    因为其它客户端不一定会填。任何解析失败都退回 `None`（失败开放：宁可新建
    一次会话，也不要把一个坏值当成上游会话 id 回放出去）。
    """
    if conversation_id:
        return conversation_id
    if not isinstance(metadata, dict):
        return None
    user_id = metadata.get("user_id")
    if isinstance(user_id, dict):
        ident = user_id
    elif isinstance(user_id, str) and user_id.strip():
        try:
            ident = json.loads(user_id)
        except (ValueError, TypeError):
            return None
    else:
        return None
    if not isinstance(ident, dict):
        return None
    session_id = ident.get("session_id")
    if isinstance(session_id, str) and session_id.strip():
        return session_id.strip()
    return None


def resolve_responses_session_id(
    conversation_id: Optional[str], client_metadata: Any
) -> Optional[str]:
    """`/v1/responses` 的**客户端线程标识**——粘滞/续聊/匿名回收都以它为键。

    [SESSION-ID] 真实 Codex 0.145.0 从不发 `conversation_id`：三份真实请求体里该键
    出现 **0 次**，身份全在 `client_metadata` 的**扁平**字典里（不是 Anthropic 那种
    JSON 串）：
        {"session_id": <会话>, "thread_id": <线程>, "x-codex-window-id": …}
    子 agent 的请求还带 `x-openai-subagent: collab_spawn` 与
    `x-openai-parent-thread-id`，且 **thread_id 与父线程不同、session_id 相同**。

    因此判据必须是 `thread_id`，不是 `session_id`：会话键的语义是「一条独立对话线」，
    而父子 agent 是两个线程、两条对话线。若按 session_id 取键，子 agent 会把父线程
    的上游上下文当成自己的，两边互相污染——而父子并发正是真实 Codex 的常态。

    此前 `/v1/responses` 只读 `req.conversation_id`（Codex 永不为它赋值），于是对
    Codex 整条链路空转：每轮都是新会话（重发全量历史），每轮还把上一轮的上游会话
    当作匿名临时会话排队删除。零报错，只是每轮都更慢更贵。

    取值顺序：显式 `conversation_id` > `thread_id` > `session_id`。解析失败一律退回
    `None`（失败开放：宁可新建一次会话，也不要把一个坏值当成上游会话 id 回放）。
    """
    if conversation_id:
        return conversation_id
    if not isinstance(client_metadata, dict):
        return None
    for key in ("thread_id", "session_id"):
        value = client_metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


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
                sub = block.get("content") or ""
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
            elif b_type in ("image", "document"):
                # [PROTOCOL] 上游是纯文本网页接口，图片/文档无法送达。占位符必须
                # 明说「未送达」，否则模型会把一个它从未看到的内容当作已读并作答。
                source = block.get("source") if isinstance(block.get("source"), dict) else {}
                media_type = source.get("media_type", "")
                if b_type == "document":
                    parts.append(f"{FILE_PLACEHOLDER_TEMPLATE.format(name=media_type or 'document')}")
                else:
                    parts.append(IMAGE_PLACEHOLDER)
            elif "text" in block and block["text"]:
                parts.append(str(block["text"]))
            else:
                # 未知块类型既不是文本也不是已知结构。旧实现让它**静默消失**——
                # 客户端明明发了内容，模型却完全不知道它存在，于是把「没看到」
                # 当成「没有」，据此作答。与 Responses 侧同一类缺陷（原始 JSON
                # 灌进用户通道），这里的诚实做法是留下一条带类型的占位说明。
                parts.append(content_omitted(b_type))
    return "\n".join(parts)


# [TOOL-SCHEMA] 工具声明提示词的预算与裁剪常量。
# 上游五家都没有原生工具调用，这份提示词是模型能看到的**唯一** schema，
# 所以保真度就是产品本身；但上下文窗口有限，故保留确定性裁剪与整体预算。
_TOOL_SCHEMA_MAX_DESCRIPTION = 300      # 单个参数/字段 description 的截断长度
_TOOL_SCHEMA_TOTAL_BUDGET = 12000       # 所有工具 schema 渲染文本的总预算（字符）
_TOOL_SCHEMA_DROP_KEYS = ("$schema", "title", "definitions", "$defs")

# [TOOL-SCHEMA] 预算不够时**怎么**降级：先压描述文本，永远不丢结构。
#
# 旧实现按**声明位置**整条降级：预算一满，后面每个工具都退回
# `name(param: type (required))`，参数描述、enum、嵌套结构、数组元素类型全部消失。
# 真实 Claude Code 抓包（20 个工具，schema 合计 14700 字符 > 12000 预算）因此精确地
# 把最后四个工具（WebFetch/WebSearch/Workflow/Write）挤成摘要 —— 于是 `Write` 的
# `file_path` 只剩 `file_path: string (required)`，那句「**必须是绝对路径，不能是
# 相对路径**」整句消失。`Write` 是最常用的 agent 工具，而网关**不做任何参数校验**
# （全仓没有 schema 校验器），所以这句提示词就是唯一的契约。
#
# 更根本的问题是**任意性**：把同一个数组反过来，保真的变成 Write/WebFetch，降级的
# 变成 Bash/Agent —— 一个工具能不能拿到自己的契约，不能由它在数组里的下标决定。
#
# 现在改为**统一的描述压缩阶梯**：所有工具走同一档，档位由「让整体刚好装进预算」的
# 最小压缩量决定。结构性信息（type/required/enum/items/properties/nested）在任何
# 档位都完整保留，被压缩的只有**自然语言描述**，而且**必填参数的描述拿到更高额度**：
# 必填参数是模型必须填的，它的语义约束（绝对路径、格式、范围）是「填错就失败」的
# 信息；可选参数的描述更多是「什么时候用它」的偏好。
#
# 实测（真实 20 工具抓包）：opt=300 -> 14700（超），opt=100 -> 11586（装得下），
# 于是整份抓包**全部 20 个工具都保住完整结构**，且必填参数描述保留原文。
_TOOL_SCHEMA_DESCRIPTION_LADDER = (
    # (必填参数描述上限, 其他描述上限)；None 表示不压缩（用 _TOOL_SCHEMA_MAX_DESCRIPTION）
    (None, _TOOL_SCHEMA_MAX_DESCRIPTION),   # 先试完全不压（常规情形）
    (None, 160),
    (None, 120),
    (None, 100),
    (None, 80),
    (160, 40),
    (80, 0),
    (0, 0),                                  # 只留结构，描述全压掉
)

# [TOOL-SCHEMA] 工具**自身**描述的额度阶梯（与上表的「参数描述」是两个不同的字段）。
#
# 旧实现只有一条规则：`description.strip().split("\n")[0][:200]` —— 取首行、截 200 字。
# 但真实声明的首行是**标题**而不是内容：`# SendMessage`、`# Edit`，真正的前提条件
# 写在第二行往后。实测真实 20 工具抓包（`tests/fixtures/claude_code_20_tools.json`）：
#
#   SendMessage  4259 -> 13      Skill  1416 -> 15      Bash 1096 -> 47
#   Edit 里那句「You must Read the file in this conversation before editing, or the
#   call will fail」**整句消失**；Read 的「file_path must be an absolute path」同理。
#
# 也就是说，模型被要求遵守的前提条件，恰恰是被固定截断丢掉的那部分——而模型没有
# 任何其他途径知道它们（网关不做参数校验，提示词就是唯一契约）。总量并不大：
# 首行截断合计 2078 字符，完整描述（压掉换行空白）28022 字符，13.5 倍；而这是
# **整份系统提示词**里信息密度最高的一段。
#
# 与参数描述一样按预算分档，而不是按位置：整份提示词"刚好装得下"的最轻一档。
_TOOL_DESC_BUDGET = 20000


def _normalize_tool_description(text: str) -> str:
    """把多行描述压成一段：`- ` 列表项之间用 `; `，其余空白并成单个空格。

    保留全部句子，只去掉排版空白。首行标题（`# SendMessage`）也保留——它是工具
    自述的名字，放在最前面读起来仍然通顺。
    """
    parts = [seg.strip() for seg in str(text or "").split("\n")]
    segs = [p for p in parts if p]
    out = "; ".join(segs)
    return " ".join(out.split())


def _cap_tool_description(text: str, cap: Optional[int]) -> str:
    """按额度截断，并**显式标注**截断（`…`）——静默截断会让模型以为描述到此为止。"""
    norm = _normalize_tool_description(text)
    if cap is None or len(norm) <= cap:
        return norm
    dropped = len(norm) - cap
    return norm[:cap].rstrip() + f" …(+{dropped} chars)"


# (工具描述上限)；None = 不截断（只做空白归一）。按需逐档收紧，保证有界。
_TOOL_DESC_LADDER = (None, 2000, 800, 400, 160, 40)
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


def _cap_descriptions(
    schema: Any,
    *,
    required_cap: Optional[int],
    other_cap: Optional[int],
) -> Any:
    """按「必填参数 / 其他」两档压缩 schema 里的 description 文本。

    只动 `description` 字段本身，**绝不**删键、删结构或删 enum：type / required /
    items / properties / enum / 嵌套对象在任何档位下都原样保留。`None` 表示该档
    不压缩（沿用 `_TOOL_SCHEMA_MAX_DESCRIPTION` 的既有截断）。

    必填与否是**在每一层 properties 上**判定的：嵌套对象里的必填字段同样享受
    更高额度 —— 判断依据始终是「这个字段对模型是不是必填」，而不是它位于第几层。
    """
    if not isinstance(schema, dict):
        return schema

    def walk(node: Any, this_cap: Optional[int]) -> Any:
        if isinstance(node, list):
            return [walk(v, this_cap) for v in node]
        if not isinstance(node, dict):
            return node
        required = {
            r for r in (node.get("required") or []) if isinstance(r, str)
        } if isinstance(node.get("required"), list) else set()
        out: Dict[str, Any] = {}
        for key, value in node.items():
            if key == "description":
                text = value if isinstance(value, str) else str(value)
                if this_cap is not None and len(text) > this_cap:
                    text = text[:this_cap]
                # 压到空串的 description 是纯噪声（`"description":""`），直接丢弃：
                # `_prune_schema` 保留 description 是有信息量时才有意义。
                if text:
                    out[key] = text
                continue
            if key == "properties" and isinstance(value, dict):
                out[key] = {
                    prop: walk(
                        sub,
                        required_cap if prop in required else other_cap,
                    )
                    for prop, sub in value.items()
                }
                continue
            if key == "required":
                out[key] = value
                continue
            out[key] = walk(value, this_cap)
        return out

    return walk(schema, other_cap)


def _render_schema_json_capped(
    schema: Any,
    *,
    required_cap: Optional[int],
    other_cap: Optional[int],
) -> str:
    """先按两档压缩描述，再走既有的确定性裁剪/渲染流程。"""
    if not isinstance(schema, dict):
        return "{}"
    return _render_schema_json(
        _cap_descriptions(
            schema, required_cap=required_cap, other_cap=other_cap
        )
    )


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
        else:
            # Responses 扁平形状（`{"type":"function","name":...}` / custom 的
            # `{"type":"custom","name":...}`）；与 `resolve_allowed_tools` 同一判定。
            name = tool_choice.get("name")
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

# 无名内置工具说明行的条数上限（真实 Codex 请求只有 `tool_search` / `web_search`
# 两个；上限存在只是为了不让客户端用任意多个无名条目撑大提示词）。
_NAMELESS_TOOL_NOTE_LIMIT = 8

# [TOOL-LOOP] Responses 的 `namespace` 声明：**一个容器，不是工具**。
#
# 实测（codex-cli 0.145.0 的原始 `/v1/responses` 请求体，10 个工具里的第 6 个）：
#
#   {"type": "namespace", "name": "multi_agent_v1",
#    "description": "Tools for spawning and managing sub-agents.",
#    "tools": [{"type": "function", "name": "close_agent", "parameters": {...}},
#              ... resume_agent / send_input / spawn_agent / wait_agent ...]}
#
# 旧实现把它当普通声明走通用分支：取顶层 `name`/`description`，找不到
# `parameters`，于是渲染成 `- \`multi_agent_v1\`: …` + `Args: {}`。两个错误同时发生：
# 五个子工具的名字与**全部参数 schema** 消失（模型无从调用），而它被告知可调用的
# 名字是 `multi_agent_v1` —— 一个真实客户端会拒绝的名字。
#
# [TOOL-LOOP] 客户端到底怎么路由，实测（codex exec 0.145.0，同一份 mock 上游，只改
# 调用项的字段形状；判据是**工具有没有真的执行**，不是模型自己怎么措辞）：
#   `exec_command` 无 namespace                  -> 真正执行（`hi`）
#   `spawn_agent`  无 namespace                  -> `unsupported call: spawn_agent`
#   `spawn_agent`  + 真实 namespace              -> `collab: SpawnAgent` -> `{"agent_id":…}`
#   `spawn_agent`  + `"bogus_ns"`                -> `unsupported call: bogus_nsspawn_agent`
#   `exec_command` + namespace                   -> `unsupported call: multi_agent_v1exec_command`
#
# 也就是说查表键是 `namespace + name` **无分隔符拼接**，两个方向都要对：属于某个
# namespace 的工具，出口侧必须把 namespace 一起带上；普通工具**绝不能**带。所以
#   - **声明侧**（提示词与解析白名单）：子工具按裸名 + 完整 schema，namespace 本身
#     不可调用，只作分组说明。展平只有一份定义（`flatten_namespace_tools`），请求
#     模型与提示词渲染都调它 —— 提示词决定「模型想调什么」，白名单决定「什么能被
#     回传」，两者必须来自同一份展平结果。
#   - **出口侧**：`namespace_of_tools()` 是「这个名字属于哪个 namespace」的唯一判据，
#     由同一份展平结果构造，恢复调用项上的 `namespace` 字段。
_NAMESPACE_TOOL_DECL_TYPE = "namespace"

# 展平后子工具携带的分组标记（网关内部键，前缀 `_` 表示不来自客户端）。
# 它只用于渲染「这些工具属于哪个 namespace」的说明行；客户端永远看不到工具列表。
_NAMESPACE_GROUP_KEY = "_namespace"

# 嵌套 namespace 的展开深度上限：真实抓包只有一层，上限只为防御自引用报文。
_NAMESPACE_MAX_DEPTH = 4
_NAMESPACE_GROUP_NOTE_LIMIT = 8


def flatten_declared_tools(tools: Any) -> Any:
    """请求体里 `tools` 字段的统一入口：`None` 原样返回，其余走 namespace 展平。

    两个请求模型（`AnthropicRequestBody` / `ResponsesRequest`）共用这一份，
    避免「同一份声明有两种解释」——提示词已经展平、白名单没有展平的话，模型写出的
    子工具调用会被当作未声明工具丢弃，症状与修复前一样。
    """
    if tools is None:
        return None
    return flatten_namespace_tools(tools)


def flatten_namespace_tools(tools: Any, _depth: int = 0) -> list:
    """把 `namespace` 容器展平成其可调用的子工具（裸名 + 原样 schema）。

    不修改调用方的 dict（请求体属于客户端，展平结果是网关内部视图）；返回的每个
    子工具都带上 `_namespace` 标记，供提示词渲染分组说明。非 `namespace` 的声明
    原样保留，顺序不变——普通工具不该因为这条规则被重排。
    """
    if not isinstance(tools, list):
        return tools if isinstance(tools, list) else []

    out: list = []
    for t in tools:
        if not (isinstance(t, dict) and t.get("type") == _NAMESPACE_TOOL_DECL_TYPE):
            out.append(t)
            continue

        sub = t.get("tools")
        if not isinstance(sub, list) or not sub or _depth >= _NAMESPACE_MAX_DEPTH:
            # 空的 / 过深的 namespace 没有可调用的子工具。它本身不可调用，也没有
            # 任何 schema 可以交给模型，静默丢弃是唯一不产生错误契约的选择。
            continue

        ns_name = t.get("name")
        ns_name = ns_name.strip() if isinstance(ns_name, str) else ""
        for child in flatten_namespace_tools(sub, _depth + 1):
            if isinstance(child, dict):
                child = dict(child)
                if ns_name and not child.get(_NAMESPACE_GROUP_KEY):
                    child[_NAMESPACE_GROUP_KEY] = ns_name
            out.append(child)
    return out


def namespace_of_tools(tools: Any) -> dict:
    """`工具名 -> 它所属的 namespace`（出口侧恢复 `namespace` 字段的唯一判据）。

    [TOOL-LOOP] 客户端查表用的键是 `namespace + name` 无分隔符拼接，所以出口侧的
    调用项**必须**带回 namespace，而普通工具必须不带（实测：给 `exec_command` 加上
    它，客户端拼成 `multi_agent_v1exec_command` 直接 `unsupported call`）。这不是
    第二份「哪些工具在哪个 namespace」的知识：输入就是 `flatten_namespace_tools`
    的产物，读的是它写下的同一个 `_namespace` 标记，所以提示词、白名单与出口三者
    永远出自同一份展平结果。

    名字重复时以先出现者为准：客户端若把同一个名字声明在多个 namespace 下，它自己
    的查表也无法区分，保守地选第一个至少保证行为稳定、可复现。
    """
    mapping: dict = {}
    if not isinstance(tools, list):
        return mapping
    for t in flatten_namespace_tools(tools):
        if not isinstance(t, dict):
            continue
        ns = t.get(_NAMESPACE_GROUP_KEY)
        if not (isinstance(ns, str) and ns):
            continue
        fn = t.get("function") if isinstance(t.get("function"), dict) else t
        name = fn.get("name") if isinstance(fn, dict) else None
        if isinstance(name, str) and name.strip():
            mapping.setdefault(name.strip(), ns)
    return mapping


def _namespace_group_notes(tools: Any) -> List[str]:
    """按声明顺序汇总「哪些工具属于哪个 namespace」的说明行。

    namespace 本身**不是**可调用的名字（真实 Codex 会回 `unsupported call`），
    但它携带的分组信息是客户端明明白白声明过的，不该静默消失。说明行必须写明
    「不可调用」，否则模型会去调它——那正是修复前的错误输出形态。
    """
    groups: List[tuple] = []
    for t in tools if isinstance(tools, list) else []:
        if not isinstance(t, dict):
            continue
        ns = t.get(_NAMESPACE_GROUP_KEY)
        if not (isinstance(ns, str) and ns):
            continue
        name = t.get("function", {}).get("name") if isinstance(t.get("function"), dict) else t.get("name")
        if isinstance(name, str) and name.strip():
            for g_name, members in groups:
                if g_name == ns:
                    members.append(name.strip())
                    break
            else:
                groups.append((ns, [name.strip()]))

    notes = []
    for ns, members in groups[:_NAMESPACE_GROUP_NOTE_LIMIT]:
        listed = ", ".join(f"`{m}`" for m in members)
        notes.append(
            f"- `{ns}` (namespace, **not callable**): its tools {listed} are declared "
            "above under their own names — call those directly, never the namespace."
        )
    return notes


def _nameless_tool_note(tool: dict, desc: str) -> str:
    """给「没有 name、因此不可调用」的内置工具生成一句说明。

    [TOOL-LOOP] 真实 Codex CLI 请求里会出现 `{"type":"tool_search", ...}` 与
    `{"type":"web_search", ...}` 这类声明：它们由客户端在本地处理，没有 `name`，
    模型写不出调用名。渲染成可调用条目就是空名噪声；静默丢弃又会让模型完全不知道
    客户端具备该能力。这里按 `type` 给一句准确的说明，未知类型则只说它由客户端处理。
    """
    t = (tool.get("type") or "").strip() if isinstance(tool.get("type"), str) else ""
    if t == "tool_search":
        return (
            "- Tool discovery (`tool_search`, executed by the client): deferred tools "
            "outside this list can be exposed to you by the client itself; you cannot "
            "invoke the search directly."
        )
    if t == "web_search":
        return (
            "- Web search (`web_search`, executed by the client): available to the client "
            "but not as a callable tool in this turn."
        )
    if not t:
        return ""
    detail = f" — {desc}" if desc else ""
    return f"- Client-handled tool (`{t}`): not callable from this turn{detail}"


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
    {budget} 字符时**不按位置降级**，而是对所有工具统一走
    `_TOOL_SCHEMA_DESCRIPTION_LADDER`：取「刚好装进预算」的最轻一档压缩**描述文本**，
    结构（type/required/enum/items/嵌套）在任何档位都完整保留，且必填参数的描述
    拿到更高额度。工具能否拿到自己的契约因此与它在数组里的下标无关。

    `tool_choice` 为仅关键字参数，用于追加一条客户端约束指令（none/required/指定工具）；
    无法识别的形状不追加任何内容，绝不抛错。
    """
    if not tools:
        return ""
    # [TOOL-LOOP] `namespace` 是容器不是工具：先展平，子工具按裸名、带完整 schema
    # 进入可调用列表。在这里（而不是只在上游请求模型里）展平，是为了让**任何**
    # 把原始声明直接喂进来的调用方也得到正确的提示词——提示词是模型唯一的契约。
    tools = flatten_namespace_tools(tools)
    lines = [
        "## Tools Available\n"
        "You have access to the following tools to accomplish tasks:",
    ]

    rendered: List[str] = []
    nameless_notes: List[str] = []
    # namespace 的分组说明与「无名内置工具」是两件事：后者根本没有名字，前者有名字
    # 但**不可调用**（真实 Codex 对 `multi_agent_v1` 回 `unsupported call`）。放进同一
    # 个「没有可调用名字」的标题下会把原因说错，而模型的下一步动作正是由原因决定的。
    namespace_notes: List[str] = _namespace_group_notes(tools)

    # 第一遍：规范化每条声明（取名字/描述/schema，合成 freeform schema，
    # 抽出不可调用的无名条目）。先全量规范化再定档，是因为「档位」必须由**整份
    # 工具集**的成本决定，而不是边遍历边判断 —— 按位置决定就是旧缺陷本身。
    callable_tools: List[Tuple[str, str, dict]] = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        if t.get("type") == "function" and "function" in t and isinstance(t["function"], dict):
            fn = t["function"]
            name = fn.get("name", "")
            raw_desc = fn.get("description", "") or ""
            schema = fn.get("parameters", {}) or {}
        else:
            name = t.get("name", "")
            raw_desc = t.get("description", "") or ""
            schema = t.get("input_schema", {}) or t.get("parameters", {}) or {}
            # [TOOL-LOOP] Responses 的 custom / freeform 声明（`{"type":"custom",
            # "name":"apply_patch"}`）**没有 schema**，原先落到这里就是一个空
            # `{}`，模型被告知「这个工具不接受任何参数」。它于是要么不调用它，要么
            # 把整段补丁写进 `arguments` 字符串——而 `apply_patch` 需要的正是那段
            # 原文。合成 schema 让「怎么传原文」变成显式契约，而不是靠模型猜。
            if not schema and t.get("type") in _FREEFORM_TOOL_DECL_TYPES:
                schema = freeform_tool_input_schema()
                if not raw_desc:
                    raw_desc = "Freeform tool: pass its complete raw input text in the `input` field."

        # [TOOL-LOOP] 没有名字的声明**不可能被调用**：模型写不出调用名，
        # `tool_names_of` 也不会把它放进白名单。真实 Codex 请求里恰好有两个这样的
        # 内置工具——`{"type":"tool_search", ...}` 与 `{"type":"web_search", ...}`
        # 都没有 `name` 字段。旧代码无条件 `t.get("name", "")`，于是提示词末尾多出
        # `- \`\`: ` + `Args: {}` 的空名条目：一个模型能「调用」却无名可写的工具，
        # 纯噪声，还诱导模型编造工具名（规则 6 明确禁止）。
        # 处理与「可调用性」一致：不进可调用列表；但把已知的内置工具降级成一句说明，
        # 让模型知道它存在（例如 tool_search 会让下一轮看到更多工具），而不是
        # 静默吞掉客户端明明白白声明过的东西。
        if not isinstance(name, str) or not name.strip():
            # 说明行同样要受预算约束：客户端可以声明任意多个无名条目，而这段文本
            # 是要送进上下文的。超出上限后只计数，不再追加。
            note = _nameless_tool_note(t, _normalize_tool_description(raw_desc))
            if note and len(nameless_notes) < _NAMELESS_TOOL_NOTE_LIMIT:
                nameless_notes.append(note)
            continue
        callable_tools.append((name.strip(), raw_desc, schema))

    # 第二遍：选档。取「整体刚好装进预算」的最轻压缩档；所有工具**同一档**渲染。
    # 一档都装不下（>=40 个巨型工具）时才退回紧凑摘要兜底，保证提示词仍有界。
    tier: Optional[Tuple[Optional[int], Optional[int]]] = None
    for required_cap, other_cap in _TOOL_SCHEMA_DESCRIPTION_LADDER:
        costs = [
            len(_render_schema_json_capped(
                schema, required_cap=required_cap, other_cap=other_cap))
            for _, _, schema in callable_tools
        ]
        if sum(costs) <= _TOOL_SCHEMA_TOTAL_BUDGET:
            tier = (required_cap, other_cap)
            break

    # 工具**自身**描述选档：与 schema 参数描述同一条纪律——按整份提示词的预算
    # 取「刚好装得下」的最轻一档，而不是按位置拍一个常数。旧实现固定「首行 + 200
    # 字」，对真实抓包等于丢掉 13.5 倍的前提条件，见 _TOOL_DESC_LADDER 的实测。
    desc_cap: Optional[int] = _TOOL_DESC_LADDER[0]
    for cand in _TOOL_DESC_LADDER:
        desc_total = sum(
            len(_cap_tool_description(d, cand)) for _, d, _ in callable_tools
        )
        if desc_total <= _TOOL_DESC_BUDGET:
            desc_cap = cand
            break

    if tier is None:
        # 病态集合：结构保真也装不下，退回紧凑摘要（每个工具仍保留名字与类型骨架）。
        for name, desc, schema in callable_tools:
            summary_desc = _cap_tool_description(desc, desc_cap)
            rendered.append(f"- `{name}({_compact_args_summary(schema)})`: {summary_desc}")
    else:
        required_cap, other_cap = tier
        for name, desc, schema in callable_tools:
            schema_json = _render_schema_json_capped(
                schema, required_cap=required_cap, other_cap=other_cap)
            desc = _cap_tool_description(desc, desc_cap)
            rendered.append(f"- `{name}`: {desc}\n  Args: {schema_json}")

    lines.extend(rendered)
    if namespace_notes:
        lines.append(
            "\n## Tool Namespaces (not callable)\n"
            "The client grouped some of the tools above into a namespace. The namespace "
            "name itself is NOT a tool: call the member tools directly by their own names."
        )
        lines.extend(namespace_notes)
    if nameless_notes:
        lines.append(
            "\n## Client-Side Internal Tools (not directly callable)\n"
            "The client declared the following tools without a callable name. They are "
            "handled by the client itself; do not attempt to call them."
        )
        lines.extend(nameless_notes)

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


#: JSON Schema 顶层类型里能写成自然语言的那几种。
_JSON_SCHEMA_TYPES = frozenset(
    {"object", "array", "string", "number", "integer", "boolean", "null"}
)

#: 顶层类型 -> 提示词里的名词。`json_schema` 声明的是 array 时，提示词若仍然说
#: "a single valid JSON object"，模型读到的就是一份与紧跟在后面的 `Schema:` 相矛盾
#: 的契约（实测：`{"type":"array"}` 的请求，提示词让模型输出 object）。提示词是这些
#: 网页端上游**唯一**能看到的契约，说错形状等于让它必然违约。
_TOP_LEVEL_PHRASES = {
    "object": "JSON object",
    "array": "JSON array",
    "string": "JSON string",
    "number": "JSON number",
    "integer": "JSON integer",
    "boolean": "JSON true/false value",
    "null": "JSON null",
}


def response_format_top_level_type(response_format: Any) -> Optional[str]:
    """客户端在 `json_schema.schema.type` 里声明的顶层类型（没声明/不认识时 `None`）。

    提示词与解析端校验读的是同一处定义：两边各写一份「顶层该是什么」，第一处改动
    就会让它们悄悄对不上（本仓库反复记录的形态）。
    """
    if not isinstance(response_format, dict):
        return None
    schema = response_format.get("json_schema")
    inner = schema.get("schema") if isinstance(schema, dict) else None
    if not isinstance(inner, dict):
        return None
    declared = inner.get("type")
    return declared if isinstance(declared, str) and declared in _JSON_SCHEMA_TYPES else None


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
        noun = _TOP_LEVEL_PHRASES.get(
            response_format_top_level_type(response_format) or "", "JSON object"
        )
        try:
            body = json.dumps(inner, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError):
            return ""
        return (
            "## Output Format\n"
            f"Respond with a single valid {noun}{label} and nothing else. "
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
# [TOOL-LOOP] 收尾围栏必须是**惰性**的，而且不能把下一块的开头吞掉。
#
# 原收尾是 `\n?```+`——贪婪。面对两个背靠背写出的工具块（真实 codex 抓包里的
# 字节形状，见 CLAUDE.md 第 68 节）：
#
#     ```tool_call\n{...}\n``````tool_call\n{...}\n```
#
# 第一块的收尾会把后一块的**开**头那三个反引号一起吃掉（匹配到 6 个），于是
# `finditer` 从吞掉的位置之后继续扫描，第二块再也不匹配：模型发起的两个调用
# 只有一个到达客户端，另一个连同它的调用一起消失，且没有任何报错。
#
# 修复方式是把收尾从"吃掉所有能吃的反引号"改成"**惰性**吃掉，但不在后跟一个
# 语言标记时停手"：`\`{3,}?` 优先只吃三颗，负向前瞻 `(?!\s*LANG(?![A-Za-z0-9_]))`
# 保证停手点不是一个新块的开头。于是相邻块被正确切开，而四个反引号的围栏、
# 语言别名、未闭合的截断块、正文里的行内反引号全部逐值不变（见
# `tests/test_parallel_tool_calls.py::test_fence_shapes_are_unchanged`）。
#
# `(?![A-Za-z0-9_])` 是必要的：`tool_calling` / `json5x` 不是语言标记，不能被
# 当成新块的开始。
_TOOL_FENCE_LANG = r"(?:tool_call|tool|function|json|json5)"
_BLOCK_TOOL_PATTERN = re.compile(
    r"```+\s*" + _TOOL_FENCE_LANG + r"\s*\n(.*?)"
    r"(?:\n?`{3,}?(?!\s*" + _TOOL_FENCE_LANG + r"(?![A-Za-z0-9_]))|$)",
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


# 惰性：只吃掉收尾所需的三个反引号。贪心版本在「正文含围栏 + 两个块背靠背」时会把
# 下一个块的开头一起吃掉——正是第 68 节那个「吞掉相邻块」的缺陷换了个位置复发。
_TRAILING_FENCE = re.compile(r"\s*`{3,}?")


def _balanced_json_at(text: str, start: int) -> Optional[int]:
    """返回从 `start` 起（跳过空白）开始的平衡 JSON 对象的**结束下标**；没有则 None。

    `extract_balanced_json_objects` 按出现顺序返回对象，所以第一个结果的起点不是
    0 就意味着这里根本没有对象开始。
    """
    objs = extract_balanced_json_objects(text[start:])
    if objs and objs[0][0] == 0:
        return start + objs[0][1]
    return None


def iter_tool_blocks(text: str):
    """逐个产出代码块工具调用：`(start, end, body, closed)`。

    **块边界必须由 JSON 结构决定，而不是由字节形状决定。** `_BLOCK_TOOL_PATTERN`
    的收尾是「任意一个不被语言标记跟随的三反引号」，它对正文一无所知；当模型写出

        ```tool_call
        {"name": "Write", "arguments": {"content": "```python\\nprint(1)\\n```"}}
        ```

    时，收尾匹配到的正是 JSON **字符串字面量内部**那对反引号：参数在第一个代码围栏
    处被截断，剩下的 `..."}}` 加上真正的块收尾被当成正文丢掉。`write` 一个 README、
    一份配置、任何内含代码围栏的文件都会触发，而且是静默的（自愈解析器甚至把截断后
    的 JSON 修好，报告「已修复」）。

    因此：正文若以 `{` 开头，则用括号平衡扫描求出对象的真实终点；当它比正则匹配到的
    收尾更靠后时，说明那个收尾落在字符串里，把正文延长到对象结束，并吃掉紧随其后
    的围栏。`closed=False` 用于「正文是 JSON 但对象还没闭合」——通常是缓冲区停在
    半个对象上；流式锁存据此**不得**把裸 JSON 当正文放出去。
    """
    pos = 0
    n = len(text)
    while pos < n:
        # 模块级名字在调用时查找：既有用例通过替换 `_BLOCK_TOOL_PATTERN` 来验证
        # 「解析行为只由这一份定义决定」，这里必须继续走那一个名字。
        match = _BLOCK_TOOL_PATTERN.search(text, pos)
        if not match:
            return
        body_start, body_end = match.span(1)
        body = match.group(1)
        obj_end = _balanced_json_at(text, body_start)

        if obj_end is not None and obj_end > body_end:
            # 收尾落在 JSON 字符串内部（正文里有代码围栏），延长到对象真实终点。
            fence = _TRAILING_FENCE.match(text, obj_end)
            end = fence.end() if fence else obj_end
            yield match.start(), end, text[body_start:obj_end], True
            pos = end
            continue

        if obj_end is None and body.lstrip().startswith("{"):
            # 以 JSON 对象开头却还没闭合：半个对象（流式边界），或模型自己截断了。
            # 解析器仍按原样交给自愈引擎处理，只把 `closed` 标成 False。
            yield match.start(), match.end(), body, False
            pos = match.end()
            continue

        closed = match.end() < n or bool(re.search(r"`{3,}\s*$", match.group(0)))
        yield match.start(), match.end(), body, closed
        pos = match.end()


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
    for start, _end, content, _closed in iter_tool_blocks(text):
        content = content.strip()
        if not content:
            continue
        parsed = _safe_json_loads(content)
        if parsed is not None:
            tools = _extract_tools_from_data(parsed)
            if tools:
                if first_block_start is None:
                    first_block_start = start
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
    # [READINESS] 本机没有任何该厂商的凭证（全新克隆 / 尚未 login）：与
    # `credential_outage` 同族——网关确实无法用当前凭证完成认证，区别只是成因在
    # 部署侧而非上游侧。归入 authentication_error，客户端因此会提示「需重新认证」
    # 而不是「上游过载，稍后重试」，而后者正是对这件事最无用的建议。
    "unconfigured": "authentication_error",
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
