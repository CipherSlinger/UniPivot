"""推测式流式工具调用解析器 (Speculative Streaming Tool Parser)

支持零延迟思考流透传、前序自然语言文本即时流式输出、工具调用标记推测式拦截
与自愈式 JSON 参数提取。
"""

from __future__ import annotations

import re
from typing import Any, List, Optional, Tuple, NamedTuple
from .json_repair import safe_loads_with_repair, strip_markdown_fence
from .anthropic_compat import (
    extract_balanced_json_objects,
    _extract_tools_from_data,
    _BLOCK_TOOL_PATTERN,
    _XML_TOOL_PATTERN,
)


class StreamAction(NamedTuple):
    action_type: str  # "reasoning_delta" | "text_delta" | "tool_detected" | "flush_text"
    content: str = ""
    tool_list: Optional[List[Tuple[str, dict]]] = None
    was_healed: bool = False


# 工具调用的潜在触发前缀模式（用于推测式拦截）
_TRIGGER_PATTERNS = [
    re.compile(r"```+(?:tool_call|json|function|action)", re.IGNORECASE),
    re.compile(r"<(?:tool_call|function_call|action)", re.IGNORECASE),
    re.compile(r'\{\s*"(?:name|tool_name|tool|function|action)"\s*:', re.IGNORECASE),
]

_PARTIAL_TRIGGER_PREFIXES = [
    "```",
    "```t", "```to", "```too", "```tool", "```tool_", "```tool_c", "```tool_ca", "```tool_cal", "```tool_call",
    "```j", "```js", "```jso", "```json",
    "<", "<t", "<to", "<too", "<tool", "<tool_", "<tool_c", "<tool_ca", "<tool_cal", "<tool_call",
    "<f", "<fu", "<fun", "<func", "<funct", "<functi", "<functio", "<function",
    "{", '{"', '{"n', '{"na', '{"nam', '{"name',
]


def parse_tool_calls_with_healing(text: str) -> Tuple[Optional[Tuple[str, List[Tuple[str, dict]]]], bool]:
    """从文本中提取所有工具调用，集成自愈式 JSON 容错引擎。

    返回 ((prefix_text, [(tool_name, arguments), ...]), was_healed)。
    """
    if not text or not isinstance(text, str):
        return None, False

    was_healed_any = False

    # 1. 优先匹配代码块 ```tool_call / ```json ...
    first_block_start: Optional[int] = None
    all_tools: List[Tuple[str, dict]] = []
    for match in _BLOCK_TOOL_PATTERN.finditer(text):
        content = match.group(1).strip()
        if not content:
            continue
        parsed, healed = safe_loads_with_repair(content)
        if parsed is not None:
            tools = _extract_tools_from_data(parsed)
            if tools:
                if healed:
                    was_healed_any = True
                if first_block_start is None:
                    first_block_start = match.start()
                all_tools.extend(tools)
    if all_tools and first_block_start is not None:
        return (text[:first_block_start].strip(), all_tools), was_healed_any

    # 2. 匹配 XML 标签 <tool_call>...</tool_call>
    first_xml_start: Optional[int] = None
    all_tools = []
    for match in _XML_TOOL_PATTERN.finditer(text):
        content = match.group(1).strip()
        if not content:
            continue
        parsed, healed = safe_loads_with_repair(content)
        if parsed is not None:
            tools = _extract_tools_from_data(parsed)
            if tools:
                if healed:
                    was_healed_any = True
                if first_xml_start is None:
                    first_xml_start = match.start()
                all_tools.extend(tools)
    if all_tools and first_xml_start is not None:
        return (text[:first_xml_start].strip(), all_tools), was_healed_any

    # 3. 匹配文本中的平衡 JSON 块
    first_json_start: Optional[int] = None
    all_tools = []
    for start, _, cand in extract_balanced_json_objects(text):
        if any(k in cand for k in ('"name"', '"tool_name"', '"tool"', '"function"', '"action"')):
            parsed, healed = safe_loads_with_repair(cand)
            if parsed is not None:
                tools = _extract_tools_from_data(parsed)
                if tools:
                    if healed:
                        was_healed_any = True
                    if first_json_start is None:
                        first_json_start = start
                    all_tools.extend(tools)
    if all_tools and first_json_start is not None:
        return (text[:first_json_start].strip(), all_tools), was_healed_any

    # 4. 终极自愈降级：如果全文整体是一个破损或截断的 JSON 工具对象
    s_clean = text.strip()
    if s_clean.startswith("{") or s_clean.startswith("```"):
        parsed, healed = safe_loads_with_repair(s_clean)
        if parsed is not None:
            tools = _extract_tools_from_data(parsed)
            if tools:
                return ("", tools), True

    return None, False


class SpeculativeToolStreamer:
    """推测式流式工具分发器。

    状态机:
    - STREAMING_REASONING: 思考过程，零延迟直出
    - STREAMING_TEXT: 自然语言文本，维持滑动窗口安全推测发射
    - TOOL_BUFFERING: 侦测到工具标记后，转入缓冲累积与参数自愈解析
    """

    LOOKAHEAD_WINDOW = 32  # 滑动推测窗口大小

    def __init__(self, tools_enabled: bool = True):
        self.tools_enabled = tools_enabled
        self.state = "STREAMING_TEXT"
        self.text_window: str = ""
        self.accumulated_text: List[str] = []
        self.accumulated_tool_buffer: List[str] = []
        self.emitted_text: List[str] = []
        self.tool_detected = False

    def _matches_any_trigger(self, s: str) -> bool:
        """检查字符串是否包含确定的工具调用起始标记。"""
        return any(pat.search(s) for pat in _TRIGGER_PATTERNS)

    def _is_potential_prefix(self, s: str) -> bool:
        """检查字符串末尾是否可能是某个工具触发词的前缀。"""
        s_tail = s[-16:] if len(s) > 16 else s
        for prefix in _PARTIAL_TRIGGER_PREFIXES:
            if s_tail.endswith(prefix):
                return True
        return False

    def feed_chunk(self, delta: str, is_reasoning: bool = False) -> List[StreamAction]:
        """注入一个新的流式数据块，返回即时应发射的动作列表。"""
        actions: List[StreamAction] = []

        if not delta:
            return actions

        # 1. 思考流内容永远零延迟即时发射
        if is_reasoning:
            actions.append(StreamAction("reasoning_delta", content=delta))
            return actions

        # 记录全量文本（用于最终工具解析与 Token 统计）
        self.accumulated_text.append(delta)

        # 2. 如果未开启 tools，直接作为文本流直出
        if not self.tools_enabled:
            actions.append(StreamAction("text_delta", content=delta))
            return actions

        # 3. 已经在工具缓冲阶段
        if self.tool_detected:
            self.accumulated_tool_buffer.append(delta)
            return actions

        # 4. 处于推测式文本流状态
        self.text_window += delta

        # 检查当前窗口内是否出现了确定性的工具调用标记
        if self._matches_any_trigger(self.text_window):
            self.tool_detected = True
            # 找到工具标记的最早位置
            earliest_idx = len(self.text_window)
            for pat in _TRIGGER_PATTERNS:
                m = pat.search(self.text_window)
                if m and m.start() < earliest_idx:
                    earliest_idx = m.start()

            prefix_to_emit = self.text_window[:earliest_idx]
            tool_part = self.text_window[earliest_idx:]

            if prefix_to_emit:
                actions.append(StreamAction("text_delta", content=prefix_to_emit))
                self.emitted_text.append(prefix_to_emit)

            actions.append(StreamAction("tool_detected"))
            self.accumulated_tool_buffer.append(tool_part)
            self.text_window = ""
            return actions

        # 检查末尾是否处于潜在触发词前缀（例如刚刚输出了 "```t"）
        if self._is_potential_prefix(self.text_window):
            # 维持安全前缀推测缓冲，暂不全部发射
            safe_len = max(0, len(self.text_window) - self.LOOKAHEAD_WINDOW)
            if safe_len > 0:
                safe_chunk = self.text_window[:safe_len]
                self.text_window = self.text_window[safe_len:]
                actions.append(StreamAction("text_delta", content=safe_chunk))
                self.emitted_text.append(safe_chunk)
            return actions

        # 无触发词危险，窗口如果超过安全长度，向客户端发射
        if len(self.text_window) > self.LOOKAHEAD_WINDOW:
            emit_len = len(self.text_window) - self.LOOKAHEAD_WINDOW
            safe_chunk = self.text_window[:emit_len]
            self.text_window = self.text_window[emit_len:]
            actions.append(StreamAction("text_delta", content=safe_chunk))
            self.emitted_text.append(safe_chunk)

        return actions

    def finalize(self) -> Tuple[str, Optional[List[Tuple[str, dict]]], bool]:
        """流结束时调用。

        返回 (未发射的剩余文本前缀, 工具调用列表或 None, 是否触发了自愈修复)。
        """
        full_text = "".join(self.accumulated_text)
        tool_buffer_str = "".join(self.accumulated_tool_buffer)
        parsed_res, was_healed = parse_tool_calls_with_healing(full_text)

        if parsed_res:
            prefix, tool_list = parsed_res
            already_emitted = "".join(self.emitted_text)
            # 计算剩余未发射的前缀
            remaining_prefix = ""
            if prefix and prefix.startswith(already_emitted):
                remaining_prefix = prefix[len(already_emitted):]
            elif not already_emitted:
                remaining_prefix = prefix
            return remaining_prefix, tool_list, was_healed

        # 如果没有成功解析出 tool_calls，将剩余的窗口内容作为普通文本输出
        remaining_text = self.text_window + tool_buffer_str
        self.text_window = ""
        self.accumulated_tool_buffer.clear()
        return remaining_text, None, False
