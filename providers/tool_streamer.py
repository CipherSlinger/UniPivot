"""推测式流式工具调用解析器 (Speculative Streaming Tool Parser)

支持零延迟思考流透传、前序自然语言文本即时流式输出、工具调用标记推测式拦截
与自愈式 JSON 参数提取。
"""

from __future__ import annotations

import re
from typing import Any, List, Optional, Tuple, NamedTuple
from .json_repair import safe_loads_with_repair, strip_markdown_fence
from .stop_sequences import truncate_final
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


# [TOOL-LOOP] 触发词必须**只**包含工具调用专用标记。
# 原先 `json` 也在其中：模型只要在回答里写一个 ```json 数据块，流式分发器就永久
# 切进工具缓冲，此后所有文本被吞掉、最后乱序一次性吐出。这与解析端「围栏必须
# 锚定 tool_call」是同一条纪律——普通代码块不是工具调用。
_TRIGGER_PATTERNS = [
    re.compile(r"```+\s*(?:tool_call|tool|function|json|json5)\s*\n", re.IGNORECASE),
    re.compile(r"<(?:tool_call|function_call)>", re.IGNORECASE),
    # 裸 JSON 的起始形态；命中后若校验不通过会被 `_release_buffer()` 放行成文本
    re.compile(r'\{\s*"(?:name|tool_name|tool|function)"\s*:', re.IGNORECASE),
]

_PARTIAL_TRIGGER_PREFIXES = [
    "```",
    "```t", "```to", "```too", "```tool", "```tool_", "```tool_c", "```tool_ca", "```tool_cal", "```tool_call",
    "```f", "```fu", "```fun", "```func", "```funct", "```functi", "```functio", "```function",
    "<", "<t", "<to", "<too", "<tool", "<tool_", "<tool_c", "<tool_ca", "<tool_cal", "<tool_call",
    "<f", "<fu", "<fun", "<func", "<funct", "<functi", "<functio", "<function",
]


def tool_names_of(tools: Any) -> Optional[set]:
    """从客户端声明的 `tools` 里取出所有工具名（兼容 OpenAI / Anthropic 两种形状）。

    返回 None 表示「调用方没有声明工具」——此时不做白名单裁剪，保持既有解析行为。
    """
    if not tools:
        return None
    names = set()
    for t in tools:
        if not isinstance(t, dict):
            continue
        fn = t.get("function") if isinstance(t.get("function"), dict) else None
        name = fn.get("name") if fn else t.get("name")
        if isinstance(name, str) and name.strip():
            names.add(name.strip())
    return names or None


def resolve_allowed_tools(tools: Any, tool_choice: Any = None) -> Optional[set]:
    """结合 `tool_choice` 计算本回合**允许回传**的工具名集合。

    [TOOL-LOOP] `tool_choice` 原先只被翻译成提示词里的一句话（"Do not call any
    tool in this turn"），解析端却仍然按「全部声明工具」的白名单放行。于是模型
    一旦没听指令（网页端模型对约束的遵从度本就不稳定），`tool_choice:"none"` 的
    请求照样收到一个 `tool_calls` / `tool_use` / `function_call`——客户端按契约
    根本不会去执行它，整个回合就此报废。实测复现（三个协议同样）：
    `tool_choice:"none"` + 上游回 ```` ```tool_call ```` → 客户端收到 finish_reason
    `tool_calls`。约束必须落在**解析**上，提示词只是降低发生概率，不能当作保证。

    返回语义与 `tool_names_of` 对齐：
      - `None`：不做裁剪（未声明工具，或 tool_choice 形状无法识别——失败开放）
      - 空集合：本回合不允许任何工具调用（`tool_choice: "none"`）
      - 非空集合：只接受集合内的名字（点名某个工具时即为该工具）
    """
    names = tool_names_of(tools)
    if names is None:
        return None

    if tool_choice is None:
        return names

    if isinstance(tool_choice, str):
        # "auto" / "required" / "any" / 未知字符串：维持全部声明工具
        return set() if tool_choice.strip().lower() == "none" else names

    if isinstance(tool_choice, dict):
        c_type = tool_choice.get("type")
        if c_type == "none":
            return set()
        chosen = None
        if c_type == "tool":
            chosen = tool_choice.get("name")
        elif isinstance(tool_choice.get("function"), dict):
            # OpenAI: {"type": "function", "function": {"name": ...}}
            chosen = tool_choice["function"].get("name")
        if isinstance(chosen, str) and chosen.strip():
            # 点名的工具若根本没声明（客户端自相矛盾），按无法识别处理：
            # 收紧白名单只会把一个可用的回合变成空回合，失败开放更安全。
            chosen = chosen.strip()
            return {chosen} if chosen in names else names

    return names


def parallel_calls_disallowed(
    parallel_tool_calls: Any = None,
    tool_choice: Any = None,
) -> bool:
    """判断客户端是否**禁止**本回合返回多个并行工具调用。

    [TOOL-LOOP] 与 `tool_choice` 同属「客户端约束被静默丢弃」的缺陷类：三个协议
    都允许客户端声明禁止并行调用（OpenAI/Responses 的 `parallel_tool_calls:false`、
    Anthropic 的 `tool_choice.disable_parallel_tool_use:true`），而网关此前完全不看
    这个字段——实测上游回两个 ```tool_call 块时，三个协议都原样回传两个调用。
    对无法并行执行/无法合并结果的 agent 客户端，第二个调用是一次它没有预期、
    也不会执行的调用，整个回合的 tool 配对就此错位。

    语义与 `resolve_allowed_tools` 对齐：只有**明确声明**禁止时才收紧；未知形状
    或未声明一律按允许处理（失败开放，零回归）。
    """
    if parallel_tool_calls is False:
        return True
    if isinstance(tool_choice, dict) and tool_choice.get("disable_parallel_tool_use") is True:
        return True
    return False


def cap_parallel_tool_calls(tool_list: Optional[List[Tuple[str, dict]]], disallowed: bool):
    """客户端禁止并行调用时，只保留**第一个**调用，其余丢弃。

    保留第一个而不是全部丢弃：第一个调用是模型当前最想做的事，客户端执行完它
    之后发起的下一轮请求会把剩余工作继续推进；而丢弃全部只会得到一个「什么都没
    发生」的空回合。
    """
    if disallowed and tool_list and len(tool_list) > 1:
        return tool_list[:1]
    return tool_list


def recover_tools_from_reasoning(
    reasoning_text: str,
    allowed: Optional[set] = None,
    answer_text: str = "",
) -> Tuple[Optional[List[Tuple[str, dict]]], bool]:
    """[TOOL-LOOP] 思考通道兜底：正文为空时，从思考流里找回被吞掉的工具调用。

    推理模型把「该做什么」写在思考通道里，网页端的上游又把思考与正文分开推送。
    实测：模型把整段 ```tool_call 写进 `reasoning_content`、正文一个字都没有时，
    网关按「正文里没有工具调用」处理，客户端收到的是 `finish_reason: stop`、没有
    `tool_calls`、正文是被补成 `…` 的空壳——对 agent 客户端而言这一轮什么都没发生，
    它既没有答案可读，也没有调用可执行，循环就此停摆。

    **只在正文为空时才兜底**，这是本条的关键取舍：若模型已经给出了正文，那段正文
    就是它的答案，思考里出现的工具块更可能是「我本可以这样调用」的举例；把它翻成
    真实调用会让一次正常的回答变成一次没人预期的工具执行。正文为空则不存在这个
    歧义——模型没有给出任何面向用户的输出，那一轮唯一可执行的内容就在思考流里。

    返回 (tool_list 或 None, was_healed)。前缀文本**故意不返回**：思考流已经逐字
    发给客户端了，再当正文发一遍就是重复。
    """
    if not reasoning_text or (answer_text or "").strip():
        return None, False
    parsed, was_healed = parse_tool_calls_with_healing(reasoning_text, allowed)
    if not parsed:
        return None, False
    _, tool_list = parsed
    if not tool_list:
        return None, False
    return tool_list, was_healed


def parse_tool_calls_with_healing(
    text: str,
    allowed: Optional[set] = None,
) -> Tuple[Optional[Tuple[str, List[Tuple[str, dict]]]], bool]:
    """从文本中提取所有工具调用，集成自愈式 JSON 容错引擎。

    返回 ((prefix_text, [(tool_name, arguments), ...]), was_healed)。

    `allowed` 是客户端声明的工具名集合。给出时只接受其中的名字——模型臆造的工具
    名对客户端毫无意义，只会让整个回合被拒绝（见 `_extract_single_tool_strict`）。
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
            tools = _extract_tools_from_data(parsed, allowed)
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
            tools = _extract_tools_from_data(parsed, allowed)
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
                tools = _extract_tools_from_data(parsed, allowed)
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
            tools = _extract_tools_from_data(parsed, allowed)
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
    # [TOOL-LOOP] 工具缓冲的硬上限：超过它仍未闭合，说明这不是工具调用，
    # 必须放弃锁存、把内容当普通文本放出去，而不是继续吞掉整段回答。
    MAX_TOOL_BUFFER = 65536

    def __init__(self, tools_enabled: bool = True, allowed: Optional[set] = None):
        self.tools_enabled = tools_enabled
        self.allowed = allowed
        self.state = "STREAMING_TEXT"
        self.text_window: str = ""
        self.accumulated_text: List[str] = []
        self.accumulated_tool_buffer: List[str] = []
        # [TOOL-LOOP] 思考流单独累积：三个协议都要在终局回报 reasoning
        # （Responses 的 reasoning output item、Anthropic 的 thinking 统计），
        # 而思考增量不进 `accumulated_text`——它既不该参与工具解析，也不该计入正文。
        self.accumulated_reasoning: List[str] = []
        self.emitted_text: List[str] = []
        self.tool_detected = False

    def _buffer_len(self) -> int:
        return sum(len(x) for x in self.accumulated_tool_buffer)

    def _release_buffer(self) -> List[StreamAction]:
        """放弃工具缓冲，把已累积内容当作普通文本放出（解除锁存）。

        [TOOL-LOOP] 原实现 `tool_detected` 一旦置位永不复位：一次误触发之后整个流
        被吞进缓冲，直到 finalize 才一次性吐出，客户端看到的是「说到一半停住」。
        这里给出两条脱身路径：围栏已闭合却解析不出工具调用，或缓冲区超过硬上限。
        """
        self.tool_detected = False
        text = "".join(self.accumulated_tool_buffer)
        self.accumulated_tool_buffer.clear()
        self.state = "STREAMING_TEXT"
        if not text:
            return []
        self.emitted_text.append(text)
        return [StreamAction("text_delta", content=text)]

    def _should_unlatch(self) -> bool:
        """判断锁存的工具缓冲是否已被证伪：闭合了却不是工具调用，或已经太大。"""
        buf = "".join(self.accumulated_tool_buffer)
        if len(buf) > self.MAX_TOOL_BUFFER:
            return True
        closed = buf.rstrip().endswith("```") or re.search(r"</(?:tool_call|function_call)>\s*$", buf)
        if not closed:
            return False
        parsed, _ = parse_tool_calls_with_healing(buf, self.allowed)
        return not parsed

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
            self.accumulated_reasoning.append(delta)
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
            # [TOOL-LOOP] 锁存必须可被证伪，否则一次误触发会吞掉整段回答
            if self._should_unlatch():
                actions.extend(self._release_buffer())
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

            self.accumulated_tool_buffer.append(tool_part)
            self.text_window = ""
            # [TOOL-LOOP] 触发即校验：若这一块已经闭合却不是合法工具调用，
            # 立刻解除锁存并放行。否则要等下一个 chunk 才发现，而流可能就此结束
            # （或整段回答已经在缓冲里，再也不会流出）。
            if self._should_unlatch():
                actions.extend(self._release_buffer())
                return actions

            actions.append(StreamAction("tool_detected"))
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

    def feed_text(self, delta: str) -> List[StreamAction]:
        """把一段**已知是正文**的文本送入状态机。

        [TOOL-LOOP] 未声明 tools 时，流仍然是逐字直出的，但终局统计（token 数、
        正文全文）必须完整。原先无 tools 分支自己维护 `accumulated_chunks`，
        与 tools 分支的 `accumulated_text` 是两份账；这里让两条路径共用同一份
        记账，差异只在是否锁存工具调用。
        """
        if not self.tools_enabled:
            if not delta:
                return []
            self.accumulated_text.append(delta)
            self.emitted_text.append(delta)
            return [StreamAction("text_delta", content=delta)]
        return self.feed_chunk(delta)

    def truncate_at_stop(self, stop: Any) -> bool:
        """[PROTOCOL] 客户端停用词命中后，把内部累积截断到命中点之前。

        为什么必须在 `finalize()` **之前**做：命中点之后的正文对客户端不存在，而
        `finalize()` 是对**全部**累积文本做工具解析。若不停用，一个写在停用词之后
        的 ```tool_call 会被解析出来并作为真实调用回传——客户端明确要求「到此为止」，
        却收到了一次它没有预期的执行。同时清空工具缓冲与文本窗口，否则 `finalize()`
        会把这段残料当作普通文本重新吐出来。

        返回是否命中了停用词。
        """
        joined = "".join(self.accumulated_text)
        kept, matched = truncate_final(joined, stop)
        if matched is None:
            return False
        self.accumulated_text = [kept] if kept else []
        self.accumulated_tool_buffer.clear()
        self.tool_detected = False
        self.text_window = ""
        return True

    def finalize(self) -> Tuple[str, Optional[List[Tuple[str, dict]]], bool]:
        """流结束时调用。

        返回 (未发射的剩余文本前缀, 工具调用列表或 None, 是否触发了自愈修复)。
        """
        full_text = "".join(self.accumulated_text)
        tool_buffer_str = "".join(self.accumulated_tool_buffer)
        # [TOOL-LOOP] 终局解析必须沿用同一份白名单：否则一个已被证伪并放行的
        # 未声明工具名会在流结束时被重新"识别"出来，变成一次客户端无法执行的
        # 工具调用，同时把已放行的尾巴作为已发射内容吞掉。
        parsed_res, was_healed = parse_tool_calls_with_healing(full_text, self.allowed)

        if parsed_res:
            prefix, tool_list = parsed_res
            already_emitted = "".join(self.emitted_text)
            # 计算剩余未发射的前缀
            remaining_prefix = ""
            if prefix:
                if prefix.startswith(already_emitted):
                    remaining_prefix = prefix[len(already_emitted):]
                else:
                    # 解析出的前缀与已发射文本不一致（例如边界空白被 strip 掉）：
                    # 从公共前缀处续接，宁可重复少量字符也绝不吞掉正文
                    common = 0
                    for a, b in zip(already_emitted, prefix):
                        if a != b:
                            break
                        common += 1
                    remaining_prefix = prefix[common:]
            return remaining_prefix, tool_list, was_healed

        # 如果没有成功解析出 tool_calls，将剩余的窗口内容作为普通文本输出
        remaining_text = self.text_window + tool_buffer_str
        self.text_window = ""
        self.accumulated_tool_buffer.clear()
        return remaining_text, None, False
