r"""SingleFlight 跟随者看到的结局，必须与 Leader 相同 (SINGLEFLIGHT)

SingleFlight 把同一个指纹的并发请求合并成一次上游调用：Leader 真的去调上游，
每个 Follower 领用 Leader 的答案。当 Leader 在**派发前**被拒绝时（凭证整环失效 /
运维禁用 / 封禁租约 / 长效避让 / 部署尚未配置凭证），`execute_chat` 在**任何上游
调用之前**就抛出 `ProviderError`，`server.py` 用 `group.finish(error=e)` 让飞行组
终局——而此时**历史缓冲是空的**，因为 Leader 一帧都还没产生。

三条流式跟随者路径此前都是 `return StreamingResponse(sse_keepalive(
stream_follower_gen(group)))`：状态行在拉第一帧之前就已发出（200），随后
`stream_follower_gen` 把这个异常抛出来，响应体一个字节都没有，连接干净地断开。

实测（真实 uvicorn + TCP，同一份请求，唯一变量是「我是 Leader 还是 Follower」）：

    Leader    HTTP 429，JSON 错误体（上游给的真实原因）
    Follower  HTTP 200，响应体为空，无终局事件、无 [DONE]、无任何错误

在 `/v1/responses` 上这正好复现 §57 的症状 `stream closed before response.completed`，
而真实原因（配额 / 上下文 / 风控）全部丢失；三种协议都一样。同一条协议纪律在
Leader 与 Follower 上给出两种结果，是本仓库反复记录的那类缺陷（§45 的同形）。

`/v1/responses` 的**非流式**跟随者早就做对了：它 `await group.done_event`，
把 Leader 的错误原样变成状态行。本套件要求流式路径也做到这一点，并且为
「等不到结局」（`GATEWAY_FOLLOWER_OPEN_TIMEOUT` 到期，fail-open 回旧行为）
留下**行内终局失败帧**，而不是静默的空 200。

运行: .venv/bin/python tests/test_follower_failure_visibility.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["TESTING"] = "1"
os.environ["GATEWAY_DISABLE_CREDENTIAL_WATCH"] = "1"
os.environ["GATEWAY_DISABLE_AUTO_REFRESH"] = "1"
os.environ["GATEWAY_DISABLE_PROMPT_FOLDING"] = "1"
os.environ["GATEWAY_DISABLE_PACING"] = "1"
os.environ["GATEWAY_DISABLE_RISK_LEDGER"] = "1"
#: 刻意**不**设置任何 `*_AUTH_TOKEN`：这台机器的 `session/` 是空的，于是派发闸门
#: 给出真实的「未配置凭证」零上游拒绝（§78），错误体、状态码、退避头都是产品的
#: 真实输出，而不是测试编出来的。
for _k in ("QWEN_AUTH_TOKEN", "DEEPSEEK_AUTH_TOKEN", "DOUBAO_AUTH_TOKEN",
           "KIMI_AUTH_TOKEN", "GLM_AUTH_TOKEN"):
    os.environ.pop(_k, None)

import httpx  # noqa: E402

import server as srv  # noqa: E402
from providers.base import ProviderError  # noqa: E402
from providers.singleflight import FlightGroup, stream_follower_gen  # noqa: E402

BODIES = {
    "openai_chat": ("/v1/chat/completions",
                    {"model": "qwen", "stream": True,
                     "messages": [{"role": "user", "content": "同一个问题"}]}),
    "anthropic": ("/v1/messages",
                  {"model": "claude-3-7-sonnet", "max_tokens": 64, "stream": True,
                   "messages": [{"role": "user", "content": "同一个问题"}]}),
    "responses": ("/v1/responses",
                  {"model": "gpt-4o", "stream": True, "input": "同一个���题"}),
}


# --------------------------------------------------------------- harness

_GUARD_HITS: list[str] = []


def _guarded_factory(key, *a, **kw):
    _GUARD_HITS.append(key)
    raise AssertionError("测试发出了未安装假 provider 的上游请求")


_DEFAULT_FACTORY = srv._make_provider
srv._make_provider = _guarded_factory


def _slow_dispatch(delay: float = 0.35):
    """把 Leader 的派发推迟一小段，只为**确定地**造出并发窗口。

    缺陷只在「跟随者已挂上、Leader 尚未终局」的窗口里可达；真实的死环天然提供
    这个窗口（握手预算 180s，五个候选逐个数十秒），测试不能靠碰运气去撞它。
    被推迟的只是时钟：真正的拒绝仍是产品自己做的零上游拒绝。
    """
    real = srv.failover_router.execute_chat

    async def slow(*a, **kw):
        await asyncio.sleep(delay)
        return await real(*a, **kw)

    srv.failover_router.execute_chat = slow

    def restore():
        srv.failover_router.execute_chat = real

    return restore


def _drive_pair(path: str, body: dict, stagger: float = 0.05):
    """同一个请求发两次：先发的成为 Leader，后发的成为 Follower。"""

    async def run():
        transport = httpx.ASGITransport(app=srv.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t",
                                     timeout=30) as c:
            async def one():
                frames: list[str] = []
                try:
                    async with c.stream("POST", path, json=body) as r:
                        status, headers = r.status_code, dict(r.headers)
                        async for line in r.aiter_lines():
                            if line.strip():
                                frames.append(line)
                        return {"status": status, "headers": headers,
                                "frames": frames, "error": None}
                except Exception as e:  # noqa: BLE001 - 客户端看到的真实症状
                    return {"status": None, "headers": {}, "frames": frames,
                            "error": f"{type(e).__name__}: {e}"}

            leader_task = asyncio.create_task(one())
            await asyncio.sleep(stagger)
            follower = await one()
            leader = await leader_task
            return leader, follower

    return asyncio.run(run())


def _fresh_flights():
    srv.singleflight_manager.clear()
    srv.failover_router.cooldown_tracker.reset()


# ------------------------------------------------- status line, per protocol

def _assert_refusal_on_the_status_line(protocol: str):
    path, body = BODIES[protocol]
    _fresh_flights()
    restore = _slow_dispatch()
    try:
        leader, follower = _drive_pair(path, body)
    finally:
        restore()

    assert leader["status"] == 503, (
        f"[{protocol}] Leader 没有拿到零上游拒绝: {leader}"
    )
    assert follower["status"] is not None, (
        f"[{protocol}] 跟随者连响应头都没拿到（实测异常 {follower['error']!r}）——"
        "Leader 的拒绝让跟随者的响应体在零帧处抛出，真实客户端只看到 200 + 空响应体"
    )
    assert follower["status"] == leader["status"], (
        f"[{protocol}] 同一条确定性拒绝在 Leader 上是 {leader['status']}，"
        f"在跟随者上却是 {follower['status']}——两种结局，客户端无法据此判断该不该重试"
    )
    assert follower["headers"].get("x-singleflight") == "follower", (
        f"[{protocol}] 第二个请求没有成为跟随者，用例没有测到它: {follower['headers']}"
    )
    assert follower["headers"].get("x-should-retry") == "false", (
        f"[{protocol}] 跟随者拿到的确定性拒绝没有带退避信号（§54）：客户端会对着一个"
        f"需要人工处置的环原地重试: {follower['headers']}"
    )
    assert follower["frames"], (
        f"[{protocol}] 跟随者的错误响应体是空的"
    )
    leader_msg = json.loads("\n".join(leader["frames"]))
    follower_msg = json.loads("\n".join(follower["frames"]))
    assert follower_msg == leader_msg, (
        f"[{protocol}] 跟随者拿到的错误体与 Leader 不同:\n"
        f"  leader   = {leader_msg}\n  follower = {follower_msg}"
    )
    return leader, follower


def test_openai_follower_sees_the_refusal_on_the_status_line():
    _assert_refusal_on_the_status_line("openai_chat")
    print("test_openai_follower_sees_the_refusal_on_the_status_line: PASSED")


def test_anthropic_follower_sees_the_refusal_on_the_status_line():
    _assert_refusal_on_the_status_line("anthropic")
    print("test_anthropic_follower_sees_the_refusal_on_the_status_line: PASSED")


def test_responses_follower_sees_the_refusal_on_the_status_line():
    _assert_refusal_on_the_status_line("responses")
    print("test_responses_follower_sees_the_refusal_on_the_status_line: PASSED")


# ------------------------------------------------- in-band net (fail-open)

def test_in_band_failure_frame_when_the_status_line_cannot_wait():
    """等不到结局时必须发**协议的终局失败帧**，而不是静默的空 200。

    这条路径是 fail-open 的一半：`GATEWAY_FOLLOWER_OPEN_TIMEOUT=0` 让跟随者立刻以
    200 开流（旧行为）。Leader 随后拒绝，跟随者已经没有任何办法改状态行了，
    唯一诚实的信号就是协议自己的终止事件——OpenAI 的 `data: {"error"}`（且**不**发
    `[DONE]`）、Anthropic 的 `event: error`、Responses 的 `response.failed`（§57：
    Codex 会整条丢弃裸 `event: error`）。
    """
    old = os.environ.get("GATEWAY_FOLLOWER_OPEN_TIMEOUT")
    os.environ["GATEWAY_FOLLOWER_OPEN_TIMEOUT"] = "0"
    try:
        for protocol, (path, body) in BODIES.items():
            _fresh_flights()
            restore = _slow_dispatch()
            try:
                leader, follower = _drive_pair(path, body)
            finally:
                restore()
            assert leader["status"] == 503, f"[{protocol}] {leader}"
            assert follower["status"] == 200, (
                f"[{protocol}] 打开等待开关后跟随者应当立刻开流，实测 {follower['status']}"
            )
            text = "\n".join(follower["frames"])
            if protocol == "openai_chat":
                assert '"error"' in text, (
                    f"[{protocol}] 跟随者的空流没有行内失败帧，客户端只看到一次静默断开: {text!r}"
                )
                assert "[DONE]" not in text, (
                    f"[{protocol}] 失败的流却发了 [DONE]，把自己伪装成了正常说完的答案"
                )
            elif protocol == "anthropic":
                assert "event: error" in text, (
                    f"[{protocol}] 跟随者的流里没有 Anthropic 的错误事件: {text!r}"
                )
                assert "event: message_stop" not in text, (
                    f"[{protocol}] 失败的流却发了 message_stop，客户端会当成正常结束"
                )
            else:
                assert "response.failed" in text, (
                    f"[{protocol}] Codex 会丢弃裸 `event: error`，跟随者的失败原因必须走"
                    f" `response.failed`（§57）: {text!r}"
                )
                assert "response.completed" not in text, (
                    f"[{protocol}] 失败的流却发了 response.completed"
                )
    finally:
        if old is None:
            os.environ.pop("GATEWAY_FOLLOWER_OPEN_TIMEOUT", None)
        else:
            os.environ["GATEWAY_FOLLOWER_OPEN_TIMEOUT"] = old
    print("test_in_band_failure_frame_when_the_status_line_cannot_wait: PASSED")


def test_mid_stream_group_error_becomes_a_terminal_frame():
    """Leader 中途失败：跟随者已经收到部分内容，也必须以终局失败帧收尾。

    这里直接驱动共享层（`stream_follower_gen`）：一帧已广播、随后飞行组以错误
    终局。旧行为是把异常抛给客户端——它先看到的是一段内容，然后是**干净地断开**，
    与「正常说完」无法区分（§52 的同一条纪律）。
    """
    group = FlightGroup("unit-mid-stream")

    async def run():
        await group.broadcast_chunk("data: {\"choices\":[{\"delta\":{\"content\":\"半\"}}]}\n\n")
        await group.finish(error=ProviderError("上游中途断开", 502, "upstream_error"))
        out = []
        async for frame in stream_follower_gen(
            group, render_error=srv._follower_error_frames("openai_chat")
        ):
            out.append(frame)
        return out

    frames = asyncio.run(run())
    text = "".join(frames)
    assert "半" in text, f"跟随者没有收到已经广播出去的那一帧: {frames}"
    assert '"error"' in text, (
        f"中途失败后跟随者只拿到半截内容就断开了，没有任何失败帧: {frames}"
    )
    assert "[DONE]" not in text, "失败的流发了 [DONE]"
    print("test_mid_stream_group_error_becomes_a_terminal_frame: PASSED")


# ------------------------------------------------------------- zero regression

class ScriptedProvider:
    """正常出字的脚本化上游：验证等待首个事件没有破坏健康路径。"""

    async def chat(self, messages, model, **kw):
        for chunk in ["你", "好", "。"]:
            await asyncio.sleep(0.05)
            yield chunk, {"conversation_id": "conv_follower"}


def test_a_healthy_follower_still_receives_every_frame():
    path, body = BODIES["openai_chat"]
    _fresh_flights()
    # 这条用例要证明「等待首个事件」没有改变**健康**路径，所以必须真的走到派发：
    # 给本用例安上假凭证（其余用例刻意不配，用真实的零上游拒绝）。
    for key in ("QWEN_AUTH_TOKEN", "DEEPSEEK_AUTH_TOKEN", "DOUBAO_AUTH_TOKEN"):
        os.environ[key] = "fake_follower_test"
    srv._make_provider = lambda key, *a, **kw: ScriptedProvider()
    restore = _slow_dispatch()
    try:
        leader, follower = _drive_pair(path, body)
    finally:
        restore()
        srv._make_provider = _guarded_factory
        for key in ("QWEN_AUTH_TOKEN", "DEEPSEEK_AUTH_TOKEN", "DOUBAO_AUTH_TOKEN"):
            os.environ.pop(key, None)

    assert leader["status"] == 200 and follower["status"] == 200, (leader, follower)
    assert follower["headers"].get("x-singleflight") == "follower", follower["headers"]
    assert follower["frames"] == leader["frames"], (
        "健康的跟随者收到的帧与 Leader 不同——等待首个事件不能改变正常路径:\n"
        f"  leader   = {leader['frames']}\n  follower = {follower['frames']}"
    )
    assert any("[DONE]" in f for f in follower["frames"]), (
        f"健康的跟随者没有收到 [DONE]: {follower['frames']}"
    )
    print("test_a_healthy_follower_still_receives_every_frame: PASSED")


def run_all():
    tests = [
        test_openai_follower_sees_the_refusal_on_the_status_line,
        test_anthropic_follower_sees_the_refusal_on_the_status_line,
        test_responses_follower_sees_the_refusal_on_the_status_line,
        test_in_band_failure_frame_when_the_status_line_cannot_wait,
        test_mid_stream_group_error_becomes_a_terminal_frame,
        test_a_healthy_follower_still_receives_every_frame,
    ]
    for fn in tests:
        fn()
    srv._make_provider = _DEFAULT_FACTORY
    # 故障转移会把绊线的断言包成 502/503；「没碰上游」的直接证据是计数器。
    assert not _GUARD_HITS, (
        f"有 {len(_GUARD_HITS)} 次上游调用未经脚本化 provider: {sorted(set(_GUARD_HITS))}"
    )
    print(f"\n✅ All follower failure visibility tests passed! ({len(tests)} cases)")


if __name__ == "__main__":
    run_all()
