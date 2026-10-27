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

from typing import Any, AsyncIterator, List, Optional, Tuple

from .base import StopSequenceHit, _est_tokens


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


# --------------------------------------------------------------- 输出 token 上界


def _largest_prefix_within(text: str, limit: int) -> int:
    """返回最长的 n，使 `_est_tokens(text[:n]) <= limit`。

    `_est_tokens` 对前缀是单调不减的（它只是把字符分成 CJK 与其它两类再相加），
    所以二分成立。`_est_tokens("")` 返回 1，因此 n=0 在 `limit >= 1` 时总是合法，
    二分不会落在无效区间上。
    """
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _est_tokens(text[:mid]) <= limit:
            lo = mid
        else:
            hi = mid - 1
    return lo


class TokenBudget:
    """客户端声明的输出 token 上界（`max_tokens` / `max_output_tokens`）。

    与停用词同源：两者都是**客户端划定的输出终点**，都必须在网关侧执行，因为
    提示词只能降低概率、不能兑现契约，而这五个上游数不清 token。区别在于停用词
    命中是一个字符串，token 上界是一个数——所以它需要按估算值反推截断点。

    只在 `limit` 显式给出时生效。Anthropic 的 `max_tokens` 有 4096 默认值，
    「没声明」与「声明了 4096」在字段值上无法区分，判据必须由调用方用
    `model_fields_set` 提供；执行一个隐式默认值等于截断一个从来没人限定过的回答。

    只统计**可见正文**：思考通道不进预算（它不占客户端的输出配额，而且第 58 节
    的规则要求思考仍然参与 `output_tokens` 统计与工具救援）。
    """

    def __init__(self, limit: Optional[int]) -> None:
        # 非正数视为未声明：客户端不会用 0 表达「别说话」，而 0 会让二分退化。
        self.limit = limit if isinstance(limit, int) and limit > 0 else None
        self._text = ""
        self.hit = False

    @property
    def active(self) -> bool:
        return self.limit is not None

    @property
    def text(self) -> str:
        """已经被放行的可见正文（截断之后不再增长）。"""
        return self._text

    def feed(self, delta: str) -> str:
        """喂入一块可见正文，返回真正可以继续下发的部分。"""
        if not self.active or not delta:
            return delta
        if self.hit:
            return ""
        candidate = self._text + delta
        if _est_tokens(candidate) <= self.limit:
            self._text = candidate
            return delta
        before = len(self._text)
        cut = _largest_prefix_within(candidate, self.limit)
        self._text = candidate[:cut]
        self.hit = True
        return candidate[before:cut]


async def apply_token_budget(stream: AsyncIterator, budget: TokenBudget) -> AsyncIterator:
    """把上游流包一层：只放行预算之内的可见正文，其余原样透传。

    三条义务，每一条漏掉都会造成一个不同的、无声的故障：

    1. **元数据帧原样放行**（`delta == ""` 且带 `{"conversation_id": …}`）。
       五家上游的会话 id 都在尾帧里，吞掉它等于让 affinity 与匿名会话回收全部失效。
    2. **思考通道不受预算约束**（`meta["reasoning"]`）。思考不占客户端的输出配额；
       把它计进预算会让一个正常的长思考被误判成超限，而第 58 节的工具救援与
       `output_tokens` 统计都依赖思考仍然完整到达。
    3. **截断之后继续把上游抽干**。上游流的 `finally` 里是并发信号量与连接的归还，
       提前 `break` 会把它们留在原地；而且客户端需要的尾帧（会话 id）还没到。
       所以 `hit` 之后仍然逐帧拉取，只是不再下发任何内容。
    """
    async for delta, meta in stream:
        if meta and meta.get("reasoning"):
            yield delta, meta
            continue
        if not delta:
            yield delta, meta
            continue
        kept = budget.feed(delta)
        if kept:
            yield kept, meta
        # kept 为空时这一帧被整块丢弃——但循环继续，见义务 3。

