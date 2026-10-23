"""SSE 心跳保活 (STREAM-KEEPALIVE)

问题不在网关，也不在客户端，而在两者之间的**沉默**：上游五家网页接口在
「收到请求」到「吐出第一个字」之间可能思考几十秒（推理模型尤其如此）。这段时间
网关没有任何字节可发，而 Codex CLI / Claude Code 这类客户端对「连接建立后长时间
无字节」有很短的判定窗口——它们会判定连接已死并重连或放弃，用户看到的是
「网关连上了，但 agent 不说话」，网关日志里却一切正常（请求还在跑）。

此前所有 `StreamingResponse` 都是裸生成器：上游沉默多久，客户端就沉默多久。
本模块提供 `sse_keepalive()`，在拉取下一帧「迟迟不返回」时插入 SSE 注释帧
`: keepalive`。

为什么用注释行而不是空 data 帧：
- 以 `:` 开头的行按 SSE 规范是注释，任何合规客户端都必须忽略，因此不会污染
  任何协议的事件流（OpenAI/Anthropic/Responses 三种都无法误解析它）。
- 它仍然是真实字节，会立刻刷新 TCP/TLS 缓冲，从而真正重置客户端的空闲计时器。

为什么**不能**用 `asyncio.wait_for(iterator.__anext__(), ...)`：
实测该写法会在超时时取消 `__anext__`，而取消会被抛进异步生成器体内、把它整体
关闭——首个沉默之后的上游数据全部永久丢失（第一版实现把 'a' 之后的 'b'/'c' 静默丢弃，
由 tests/test_stream_keepalive.py 的首条用例抓到）。因此这里把拉取放进一个
**不会被取消**的后台任务，
用 `asyncio.wait` 只观察它是否完成；超时只是多产出一帧，拉取任务继续挂着。
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncGenerator, AsyncIterable, List, TypeVar

from .base import get_stream_keepalive_interval

T = TypeVar("T")

logger = logging.getLogger("gateway.stream_keepalive")

KEEPALIVE_FRAME = ": keepalive\n\n"


async def sse_keepalive(
    source: AsyncIterable[T],
    interval: float | None = None,
) -> AsyncGenerator[T | str, None]:
    """在 `source` 的沉默期插入 SSE 注释帧。

    参数:
        source: 任意异步可迭代对象（上游增量流、飞行组回放等）。
        interval: 空闲多少秒后发一帧；None 时读 `GATEWAY_STREAM_KEEPALIVE`，
                  该值为 0 或负数时完全关闭（零回归，逐值等同旧行为）。

    产出:
        `source` 的每一项原样产出；沉默期额外产出 `KEEPALIVE_FRAME` 字符串。

    异常:
        `source` 抛出的异常原样上抛——心跳绝不吞掉上游错误，否则失败转移和
        错误映射会把一次真正的失败伪装成「一直在想」。
    """
    if interval is None:
        interval = get_stream_keepalive_interval()
    if interval <= 0:
        async for item in source:
            yield item
        return

    iterator = source.__aiter__()
    pending: "asyncio.Future | None" = None
    try:
        while True:
            if pending is None:
                pending = asyncio.ensure_future(iterator.__anext__())

            done, _ = await asyncio.wait({pending}, timeout=interval)
            if not done:
                # 上游仍在思考：把沉默变成可观测的字节，让客户端计时器复位。
                # 注意拉取任务**保持挂起**，不能取消，否则上游生成器会被关闭。
                yield KEEPALIVE_FRAME
                continue

            task, pending = pending, None
            try:
                item = task.result()
            except StopAsyncIteration:
                return
            yield item
    finally:
        # 客户端断连 / 上游出错时，取消仍在等待的拉取，避免泄漏一个持有上游
        # 流的幽灵任务（与 disconnect_guard 的资源回收语义一致）。
        if pending is not None:
            # 无论它是否已完成，都要**取回结局**。源恰好在我们离开时自行结束时，
            # 这个任务已经完成且带着 `StopAsyncIteration`，取消它只是空操作，
            # 而没人读取的异常会在回收时变成
            # `Task exception was never retrieved`（实测网关日志里确实出现过）。
            # 本项目所有资源泄漏都以「一条没人看的告警」的形式出现，这类噪声
            # 正是真泄漏的藏身处，必须在源头消掉。
            if not pending.done():
                pending.cancel()
            try:
                await pending
            except (asyncio.CancelledError, StopAsyncIteration):
                pass
            except asyncio.InvalidStateError:
                # 已完成的任务无法再被取消（cancel 在 await 之前抛）：结局已取，
                # 无需再做什么。
                pass
            except Exception as exc:  # noqa: BLE001 - 收尾路径必须绝对安全
                # 拉取因上游异常结束：该异常已由主循环上抛过，这里只是取回结局。
                logger.debug("挂起拉取的结局: %s: %s", type(exc).__name__, exc)
        # [C2] 取消拉取并不等于关闭源：若拉取任务在取消前刚好完成，或源本身
        # 从未被启动，取消是空操作，源生成器的 finally（上游连接、并发槽位）
        # 永远不会执行。这里显式关闭——对已结束的生成器是空操作，绝不多做。
        #
        # 用**独立任务**而不是就地 await：客户端断连时调用方通常已处于取消状态，
        # 任何 await 都会立即被二次取消，收尾逻辑一行都跑不到（实测断连路径上
        # 通道既不关闭也被计为「从未启动」，静默泄漏）。调度一个任务则不受当前
        # 任务取消状态影响。
        aclose = getattr(iterator, "aclose", None)
        if aclose is not None:
            task = asyncio.ensure_future(aclose())
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                # 当前任务被取消：收尾任务继续在后台完成，不阻塞关闭流程。
                pass
            except (StopAsyncIteration, Exception):  # noqa: BLE001
                pass