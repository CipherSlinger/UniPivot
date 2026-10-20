"""客户端停用词截断 (PROTOCOL)

OpenAI 的 `stop`（字符串或列表）与 Anthropic 的 `stop_sequences` 都是客户端的
**请求级终止条件**：模型一旦产出其中任一字符串，输出必须在它之前结束。网关此前
三个协议都把这个字段解析进模型、然后完全不使用——实测 `stop: ["END"]` 时上游
吐出的 `END` 之后的全部内容原样回给了客户端。这不是「少一个可选特性」：agent
客户端用停用词划分分段输出，多出来的尾部会让它把下一段内容当成同一段的延续。

为什么在**网关侧**截断，而不是塞进提示词：
提示词只能降低概率，而停用词是硬语义——一个说得不够准的模型会漏掉它，客户端
随后拿到它明确要求截掉的内容。与 `tool_choice` 同一条纪律：提示词负责降低发生率，
解析端负责保证结果。

为什么流式路径要**多留一个尾巴**：
停用词可能被切在两个 chunk 之间（上游吐 "EN" 然后 "D"）。已发出的字节收不回来，
所以最后一个 `len(最长停用词)-1` 个字符必须留在手里，等下一块到了再判定。
这个延迟对客户端不可见：它只在极短的窗口内被扣住，而且最终一定会被发出或截断。
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from .base import StopSequenceHit


def normalize_stop(stop: Any) -> Optional[str]:
    """把客户端声明的停用词归一成「最长的一个」（OpenAI `stop` / Anthropic `stop_sequences`）。

    OpenAI 允许字符串或字符串列表，Anthropic 只允许列表；空字符串与空白项忽略
    （所有文本都包含空串，按它截断等于把输出全部丢光，任何客户端都不会真的想要）。
    多个停用词同时存在时，取最长的一个作为流式判定的保留长度：较短的那个一定
    在它自身长度内就能判定，用最长者做上界只会多留几个字符，不会漏判。
    """
    if stop is None:
        return None
    items: List[str] = []
    if isinstance(stop, str):
        items = [stop]
    elif isinstance(stop, (list, tuple)):
        items = [s for s in stop if isinstance(s, str)]
    items = [s for s in items if s]
    if not items:
        return None
    return max(items, key=len)


def all_stops(stop: Any) -> List[str]:
    """取出全部非空停用词（判定必须在**每一个**上做，而不只是最长的那个）。"""
    if stop is None:
        return []
    if isinstance(stop, str):
        return [stop] if stop else []
    if isinstance(stop, (list, tuple)):
        return [s for s in stop if isinstance(s, str) and s]
    return []


class StopSequenceScanner:
    """流式停用词判定器：逐块喂入已完成文本，返回「现在可以安全发出的部分」。

    用法：每个 chunk 调 `feed(delta)`，把返回的文本发给客户端；流结束时调
    `flush()` 拿到被扣住的尾巴。`hit` 被置位后调用方应停止继续发送并结束回合。

    判定顺序是「先查命中、再扣尾巴」：若本块里出现了停用词，返回的文本就是命中点
    之前的全部内容，多留的尾巴不再需要（回合已经结束）。
    """

    def __init__(self, stop: Any = None):
        self.stops = all_stops(stop)
        self.window: str = ""
        self.hit: Optional[StopSequenceHit] = None
        self._keep = (len(normalize_stop(stop)) - 1) if self.stops else 0

    @property
    def active(self) -> bool:
        return bool(self.stops)

    def feed(self, delta: str) -> str:
        """喂入一个增量，返回本块中**确信安全**、可以立即发给客户端的文本。

        未配置停用词时必须**原样透传**：调用方把返回值当作「本块可发出的文本」，
        若在这里返回空串，没有声明 `stop` 的普通请求会被整段吞掉——实测这一处
        让三个协议的流式正文全部变成空串。
        """
        if not self.active:
            return delta
        if self.hit is not None:
            return ""
        if not delta:
            return ""
        self.window += delta

        # 1. 命中判定：找最早出现的停用词
        earliest_idx = -1
        earliest_len = 0
        for s in self.stops:
            idx = self.window.find(s)
            if idx != -1 and (earliest_idx == -1 or idx < earliest_idx):
                earliest_idx, earliest_len = idx, len(s)
        if earliest_idx != -1:
            text = self.window[:earliest_idx]
            matched = self.window[earliest_idx:earliest_idx + earliest_len]
            self.hit = StopSequenceHit(text, matched)
            self.window = ""
            return text

        # 2. 未命中：扣住可能被切断的尾巴
        if self._keep <= 0:
            out, self.window = self.window, ""
            return out
        if len(self.window) <= self._keep:
            return ""
        out = self.window[:-self._keep]
        self.window = self.window[-self._keep:]
        return out

    def mark_hit(self, matched: str = "") -> None:
        """[PROTOCOL] 由工具流式路径回报「停用词已在累积文本中命中」。

        工具模式的正文走 `SpeculativeToolStreamer`，判定发生在 `truncate_at_stop()`
        里（必须在 `finalize()` 解析工具之前）。命中事实仍要记到这个扫描器上，
        否则收口处报告终止原因时会说成「自然结束」。
        """
        if self.hit is None:
            self.hit = StopSequenceHit(self.window, matched)

    def flush(self) -> str:
        """流结束：把仍被扣住的尾巴交出来（未命中时它就是正常结尾）。"""
        if self.hit is not None:
            return ""
        out, self.window = self.window, ""
        return out


def truncate_final(text: str, stop: Any) -> Tuple[str, Optional[str]]:
    """非流式路径：对完整文本做一次截断，返回 (保留的文本, 命中的停用词)。"""
    stops = all_stops(stop)
    if not text or not stops:
        return text, None
    best_idx = -1
    best = ""
    for s in stops:
        idx = text.find(s)
        if idx != -1 and (best_idx == -1 or idx < best_idx):
            best_idx, best = idx, s
    if best_idx == -1:
        return text, None
    return text[:best_idx], best
