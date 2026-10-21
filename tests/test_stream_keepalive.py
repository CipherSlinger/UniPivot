"""SSE 心跳保活测试 (STREAM-KEEPALIVE)

动机来自「网关连接 agent 效果差」：上游网页接口在两个字节之间可能沉默几十秒
（推理模型尤甚），而 Codex CLI / Claude Code 对「连接建立后无字节」的容忍很短。
在此之前所有 `StreamingResponse` 都是裸生成器，沉默多久客户端就沉默多久，表现为
「连上了但 agent 不说话」，而网关日志一切正常。

本套件钉住的契约：
1. 沉默期必须产出 SSE 注释帧 `: keepalive`（合规客户端会忽略它）
2. 帧之间的间隔由间隔参数决定，且**额外**产出不改变原有项的顺序与内容
3. `GATEWAY_STREAM_KEEPALIVE=0` 时逐值等同于裸生成器（零回归）
4. 上游异常必须原样上抛，绝不能被心跳吞成「一直在想」
5. 上游立即结束时不产出任何心跳
6. 三个协议的 StreamingResponse 都已接上心跳（结构性断言）
7. 端到端：一个慢速 provider 的流式请求仍能完整送达全部内容

运行: .venv/bin/python tests/test_stream_keepalive.py
"""

from __future__ import annotations

import asyncio
import inspect
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["TESTING"] = "1"
os.environ["GATEWAY_DISABLE_CREDENTIAL_WATCH"] = "1"
os.environ["GATEWAY_DISABLE_AUTO_REFRESH"] = "1"
os.environ.pop("GATEWAY_STREAM_KEEPALIVE", None)

import server as srv  # noqa: E402
from providers.base import get_stream_keepalive_interval  # noqa: E402
from providers.stream_keepalive import KEEPALIVE_FRAME, sse_keepalive  # noqa: E402


async def _collect(agen):
    return [item async for item in agen]


async def _slow_source(delays, values):
    for d, v in zip(delays, values):
        await asyncio.sleep(d)
        yield v


def test_keepalive_frames_during_silence():
    """沉默期必须产出注释帧，且原生项一个不少、顺序不变。"""
    async def _run():
        return await _collect(
            sse_keepalive(_slow_source([0.0, 0.25, 0.0], ["a", "b", "c"]), interval=0.1)
        )

    out = asyncio.run(_run())
    assert out.count(KEEPALIVE_FRAME) >= 2, f"沉默期没有心跳: {out}"
    assert [x for x in out if x != KEEPALIVE_FRAME] == ["a", "b", "c"], out
    assert KEEPALIVE_FRAME == ": keepalive\n\n"
    print("test_keepalive_frames_during_silence: PASSED")


def test_keepalive_is_ignorable_sse_comment():
    """帧必须是以冒号开头的 SSE 注释行，任何合规客户端都会忽略。"""
    for line in KEEPALIVE_FRAME.splitlines():
        if line:
            assert line.startswith(":"), f"非注释行会污染事件流: {line!r}"
    print("test_keepalive_is_ignorable_sse_comment: PASSED")


def test_keepalive_zero_interval_is_passthrough():
    """关闭开关时逐值等同于裸生成器（零回归）。"""
    os.environ["GATEWAY_STREAM_KEEPALIVE"] = "0"
    try:
        assert get_stream_keepalive_interval() == 0.0

        async def _run():
            return await _collect(
                sse_keepalive(_slow_source([0.0, 0.2], ["x", "y"]), interval=None)
            )

        assert asyncio.run(_run()) == ["x", "y"]
    finally:
        os.environ.pop("GATEWAY_STREAM_KEEPALIVE", None)
    print("test_keepalive_zero_interval_is_passthrough: PASSED")


def test_keepalive_propagates_upstream_error():
    """上游异常必须原样上抛，不能被心跳吞成「一直在想」。"""
    class Boom(Exception):
        pass

    async def _boom():
        yield "first"
        await asyncio.sleep(0.05)
        raise Boom("upstream died")

    async def _run():
        got = []
        try:
            async for item in sse_keepalive(_boom(), interval=0.2):
                got.append(item)
        except Boom:
            return got
        raise AssertionError("异常被吞掉了")

    got = asyncio.run(_run())
    assert got == ["first"], got
    print("test_keepalive_propagates_upstream_error: PASSED")


def test_keepalive_no_frames_for_fast_stream():
    """上游即时结束时不得产出任何心跳。"""
    async def _fast():
        for v in ("1", "2", "3"):
            yield v

    out = asyncio.run(_collect(sse_keepalive(_fast(), interval=5.0)))
    assert out == ["1", "2", "3"], out
    print("test_keepalive_no_frames_for_fast_stream: PASSED")


def test_all_streaming_responses_are_wrapped():
    """结构性断言：三协议的全部流式出口都必须经过心跳。

    逐个手写接入是本案最容易漏的地方（6 个出口，任一漏装就回到「沉默即被杀」），
    因此直接把「每个 StreamingResponse 的第一个参数都包着 sse_keepalive」钉成断言。
    """
    src = Path(srv.__file__).read_text(encoding="utf-8")
    # 只数**返回出口**，不数类定义行（`class ReclaimingStreamingResponse(...)`）:
    # 类名本身含 "StreamingResponse(" 子串，直接数会把定义当成第八个出口。
    total = src.count("return StreamingResponse(") + src.count(
        "return ReclaimingStreamingResponse("
    )
    wrapped = src.count("sse_keepalive(stream_body)") + src.count(
        "sse_keepalive(stream_follower_gen("
    )
    assert total == 6, f"流式出口数量变化，请确认新增出口也接上心跳（当前 {total}）"
    assert wrapped == total, f"存在未接入心跳的流式出口: {wrapped}/{total}"
    print("test_all_streaming_responses_are_wrapped: PASSED")


def test_keepalive_interval_reads_env():
    os.environ["GATEWAY_STREAM_KEEPALIVE"] = "2.5"
    try:
        assert get_stream_keepalive_interval() == 2.5
    finally:
        os.environ.pop("GATEWAY_STREAM_KEEPALIVE", None)
    assert get_stream_keepalive_interval() == 15.0, "默认值应为 15s"
    os.environ["GATEWAY_STREAM_KEEPALIVE"] = "not-a-number"
    try:
        assert get_stream_keepalive_interval() == 15.0, "非法配置必须降级为默认值而不是崩溃"
    finally:
        os.environ.pop("GATEWAY_STREAM_KEEPALIVE", None)
    print("test_keepalive_interval_reads_env: PASSED")


def test_end_to_end_slow_stream_delivers_everything():
    """端到端：慢速上游的流式请求内容完整，心跳不污染协议事件。"""
    from fastapi.testclient import TestClient

    class SlowProvider:
        async def chat(self, *a, **kw):
            for chunk in ("Hello", " ", "world"):
                await asyncio.sleep(0.12)
                yield chunk, {}

    os.environ["GATEWAY_STREAM_KEEPALIVE"] = "0.05"
    default_factory = srv._make_provider
    srv._make_provider = lambda key: SlowProvider()
    try:
        client = TestClient(srv.app)
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={"model": "qwen3.7-max", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        ) as r:
            raw = "".join(r.iter_text())
    finally:
        srv._make_provider = default_factory
        os.environ.pop("GATEWAY_STREAM_KEEPALIVE", None)

    assert ": keepalive" in raw, "慢速流式请求没有发出心跳"
    text = ""
    import json as _json
    for block in raw.split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data: ") and line[6:].strip() != "[DONE]":
                text += _json.loads(line[6:])["choices"][0]["delta"].get("content") or ""
    assert text == "Hello world", f"心跳污染或丢失了正文: {text!r}"
    print("test_end_to_end_slow_stream_delivers_everything: PASSED")


def run_all():
    tests = [
        test_keepalive_frames_during_silence,
        test_keepalive_is_ignorable_sse_comment,
        test_keepalive_zero_interval_is_passthrough,
        test_keepalive_propagates_upstream_error,
        test_keepalive_no_frames_for_fast_stream,
        test_all_streaming_responses_are_wrapped,
        test_keepalive_interval_reads_env,
        test_end_to_end_slow_stream_delivers_everything,
    ]
    for fn in tests:
        fn()
    print(f"\n✅ All stream keepalive tests passed! ({len(tests)} cases)")


if __name__ == "__main__":
    run_all()