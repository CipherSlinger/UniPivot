"""上下文长度管理与历史消息自适应折叠优化器 (prompt_optimizer.py)。

提供:
- fold_history: 估算对话上下文 Token 数，超出限制时自适应折叠中间轮次中的超长工具执行输出与代码块，
  并在极端情况下将老旧历史轮次压缩为紧凑摘要提示，同时严格保护系统提示词 (index 0)、初始任务指令与最近 N 轮对话，
  并确保工具调用状态机 (tool calling) 与角色平衡。
- compute_prompt_hash: 对系统提示词与消息前缀生成确定性摘要哈希 (SHA-256)，用于缓存跟踪、遥测与去重。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from .base import _est_tokens, text_of

DEFAULT_MAX_TOKENS = 24000
DEFAULT_PRESERVE_RECENT_TURNS = 4
OVERSIZED_TOOL_RESULT_CHARS = 300
OVERSIZED_CODE_BLOCK_CHARS = 400

SUMMARY_MARKER = "[前序对话历史已由网关自适应折叠，保留核心结论与上下文]"
FOLDED_TOOL_RESULT_TEMPLATE = "[已折叠历史执行结果: 原 {orig_len} 字符, 状态正常]"

# 匹配 Anthropic 风格或文本中的工具执行结果标签
_TOOL_RESULT_RE = re.compile(
    r"(\[(?:Tool Result|Tool Error)(?: for [^\]\n]+)?\]:\s*\n)([\s\S]*?)(?=(?:\n\[(?:Tool Result|Tool Error)|\Z))"
)

# 匹配代码块 ```... \n <code> \n ```
_CODE_BLOCK_RE = re.compile(
    r"(```+[a-zA-Z0-9_.-]*\s*\n)([\s\S]+?)(\n```+)"
)


def _extract_text_robust(content: Any) -> str:
    """提取消息 content 中的纯文本（支持 str, list of blocks, None 等）。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict):
                if p.get("type") == "text":
                    parts.append(p.get("text", ""))
                elif p.get("type") == "tool_result":
                    sub = p.get("content", "")
                    if isinstance(sub, list):
                        sub = _extract_text_robust(sub)
                    parts.append(str(sub))
                elif "content" in p:
                    parts.append(_extract_text_robust(p["content"]))
                elif "text" in p:
                    parts.append(str(p["text"]))
        return "\n".join(parts)
    return str(content)


def estimate_message_tokens(msg: Dict[str, Any]) -> int:
    """估算单条消息消耗的 Token 数。"""
    content = msg.get("content")
    txt = _extract_text_robust(content)
    tokens = _est_tokens(txt)
    if "tool_calls" in msg and msg["tool_calls"]:
        tokens += sum(_est_tokens(json.dumps(tc, ensure_ascii=False)) for tc in msg["tool_calls"])
    return max(1, tokens)


def estimate_history_tokens(messages: List[Dict[str, Any]]) -> int:
    """估算完整消息列表消耗的 Token 总数。"""
    return sum(estimate_message_tokens(m) for m in messages)


def _is_tool_result_message(msg: Dict[str, Any]) -> bool:
    """判断一条消息是否是工具执行结果消息。"""
    if msg.get("role") == "tool":
        return True
    content = msg.get("content")
    if isinstance(content, str) and ("[Tool Result" in content or "[Tool Error" in content):
        return True
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                return True
    return False


def _has_tool_call(msg: Dict[str, Any]) -> bool:
    """判断一条消息是否包含工具调用（tool_calls 或 Anthropic tool_use 代码块）。"""
    if msg.get("role") != "assistant":
        return False
    if msg.get("tool_calls"):
        return True
    content = msg.get("content")
    if isinstance(content, str):
        if "```tool_call" in content or "<tool_call>" in content or "<function_call>" in content:
            return True
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                return True
    return False


def _fold_text_content(text: str) -> Tuple[str, bool]:
    """折叠文本内容中超长的工具结果标签与超长代码块。"""
    if not text or len(text) < OVERSIZED_TOOL_RESULT_CHARS:
        return text, False

    folded = False

    # 1. 折叠 [Tool Result for xxx]:\n ... 或 [Tool Error for xxx]:\n ...
    def _replace_tool_result(match: re.Match) -> str:
        nonlocal folded
        prefix = match.group(1)
        body = match.group(2)
        if len(body.strip()) > OVERSIZED_TOOL_RESULT_CHARS:
            folded = True
            marker = FOLDED_TOOL_RESULT_TEMPLATE.format(orig_len=len(body))
            return f"{prefix}{marker}"
        return match.group(0)

    new_text = _TOOL_RESULT_RE.sub(_replace_tool_result, text)

    # 2. 折叠长代码块 ```... \n <code> \n ```
    def _replace_code_block(match: re.Match) -> str:
        nonlocal folded
        prefix = match.group(1)
        body = match.group(2)
        suffix = match.group(3)
        if len(body) > OVERSIZED_CODE_BLOCK_CHARS:
            folded = True
            marker = FOLDED_TOOL_RESULT_TEMPLATE.format(orig_len=len(body))
            return f"{prefix}{marker}{suffix}"
        return match.group(0)

    new_text = _CODE_BLOCK_RE.sub(_replace_code_block, new_text)

    return new_text, folded


def fold_single_message(msg: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """对单条中间轮次消息执行超长工具结果或长代码块截断。

    保留原有 role、tool_call_id 与元数据，确保工具调用平衡。
    """
    role = msg.get("role")
    content = msg.get("content")

    # 1. OpenAI 规范的 tool 角色消息
    if role == "tool":
        text = _extract_text_robust(content)
        if len(text) > OVERSIZED_TOOL_RESULT_CHARS:
            new_msg = dict(msg)
            new_msg["content"] = FOLDED_TOOL_RESULT_TEMPLATE.format(orig_len=len(text))
            return new_msg, True
        return msg, False

    # 2. Anthropic 多块结构中的 tool_result
    if isinstance(content, list):
        new_blocks = []
        folded = False
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                sub_content = block.get("content", "")
                sub_str = _extract_text_robust(sub_content)
                if len(sub_str) > OVERSIZED_TOOL_RESULT_CHARS:
                    new_b = dict(block)
                    new_b["content"] = FOLDED_TOOL_RESULT_TEMPLATE.format(orig_len=len(sub_str))
                    new_blocks.append(new_b)
                    folded = True
                    continue
            new_blocks.append(block)
        if folded:
            new_msg = dict(msg)
            new_msg["content"] = new_blocks
            return new_msg, True
        return msg, False

    # 3. 纯字符串文本
    if isinstance(content, str):
        new_text, folded = _fold_text_content(content)
        if folded:
            new_msg = dict(msg)
            new_msg["content"] = new_text
            return new_msg, True

    return msg, False


def fold_history(
    messages: List[Dict[str, Any]],
    max_tokens: int = DEFAULT_MAX_TOKENS,
    preserve_recent_turns: int = DEFAULT_PRESERVE_RECENT_TURNS,
) -> List[Dict[str, Any]]:
    """自适应折叠对话历史，避免超长工具执行与多轮会话导致上游 Web 模型超时或 400 载荷超限。

    策略:
    1. 若未启用优化（GATEWAY_DISABLE_PROMPT_FOLDING=1）或总 Token 数未超过 max_tokens，原样返回。
    2. 若超出限制:
       - 始终保留 index 0 处的系统提示词 (以及开头连续的所有 system 消息)。
       - 始终保留首条用户消息 (初始任务需求与指令)。
       - 始终保留最近 preserve_recent_turns 轮对话，并维护 tool_call 与 tool_result 配对平衡。
       - 对中间轮次进行阶段一折叠: 截断超长工具执行结果与长代码块，替换为 concise marker:
         `[已折叠历史执行结果: 原 {orig_len} 字符, 状态正常]`。
       - 若折叠后依然超过 max_tokens，执行阶段二折叠: 将老旧中间对话轮次收敛压缩为统一摘要轮次:
         `{"role": "user", "content": "[前序对话历史已由网关自适应折叠，保留核心结论与上下文]"}`。
    """
    if os.environ.get("GATEWAY_DISABLE_PROMPT_FOLDING", "").strip().lower() in ("1", "true", "yes"):
        return messages

    if not messages or len(messages) <= 1:
        return messages

    total_tokens = estimate_history_tokens(messages)
    if total_tokens <= max_tokens:
        return messages

    # 1. 提取头部系统提示词 (index 0 及开头连续的 system 消息)
    head_indices: List[int] = []
    for i, m in enumerate(messages):
        if m.get("role") == "system":
            head_indices.append(i)
        else:
            break

    # 2. 提取第一条用户消息 (初始任务指令)
    first_user_idx: Optional[int] = None
    for i in range(len(head_indices), len(messages)):
        if messages[i].get("role") == "user":
            first_user_idx = i
            break

    if first_user_idx is not None:
        head_indices.append(first_user_idx)

    head_end = (max(head_indices) + 1) if head_indices else 0

    # 3. 确定保留的尾部最近轮次 (preserve_recent_turns)
    tail_start = max(head_end, len(messages) - max(1, preserve_recent_turns))

    # 维护工具调用配对平衡 (tool calling balance):
    # 若 tail_start 恰好切割在 tool_result 处，或者前一条是发起了 tool_calls 的 assistant 消息，
    # 向前推移 tail_start 确保 tool_use 与 tool_result 作为一个完整事务被同时保留在 tail 中。
    while tail_start > head_end:
        curr_msg = messages[tail_start]
        prev_msg = messages[tail_start - 1]
        if _is_tool_result_message(curr_msg):
            tail_start -= 1
            continue
        if _has_tool_call(prev_msg):
            tail_start -= 1
            continue
        break

    # 如果中间轮次为空，说明所有消息均属于 Head 或 Tail
    if head_end >= tail_start:
        return messages

    intermediate_msgs = messages[head_end:tail_start]

    # 阶段一: 折叠中间轮次中的超长工具执行输出与长代码块
    folded_intermediate: List[Dict[str, Any]] = []
    for m in intermediate_msgs:
        folded_m, _ = fold_single_message(m)
        folded_intermediate.append(folded_m)

    candidate_messages = messages[:head_end] + folded_intermediate + messages[tail_start:]
    cand_tokens = estimate_history_tokens(candidate_messages)

    if cand_tokens <= max_tokens:
        return candidate_messages

    # 阶段二: 若折叠超长工具输出后依然超载，收敛压缩中间老旧对话为紧凑摘要提示
    summary_msg: Dict[str, Any] = {
        "role": "user",
        "content": SUMMARY_MARKER,
    }

    return messages[:head_end] + [summary_msg] + messages[tail_start:]


def compute_prompt_hash(system_prompt: str, messages: List[Dict[str, Any]]) -> str:
    """对系统提示词与消息列表计算稳定的 SHA-256 指纹哈希。

    用于会话遥测、端点缓存追踪与前缀命中分析。
    """
    hasher = hashlib.sha256()

    # 注入 system prompt
    sys_str = (system_prompt or "").strip()
    hasher.update(sys_str.encode("utf-8"))
    hasher.update(b"\x00")

    # 规范化遍历 messages
    for msg in (messages or []):
        role = str(msg.get("role", "")).strip()
        content = msg.get("content")
        txt = _extract_text_robust(content)
        hasher.update(role.encode("utf-8"))
        hasher.update(b":")
        hasher.update(txt.encode("utf-8"))
        if "tool_calls" in msg and msg["tool_calls"]:
            hasher.update(
                json.dumps(msg["tool_calls"], sort_keys=True, ensure_ascii=False).encode("utf-8")
            )
        hasher.update(b"\x01")

    return hasher.hexdigest()
