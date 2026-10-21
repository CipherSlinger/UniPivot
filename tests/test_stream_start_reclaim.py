"""流式响应体「从未开始」时的资源回收测试 (tests/test_stream_start_reclaim.py)。

[C2] 三条流式路径都有一个共同的失效模式：**上游资源在响应体生成器开始执行之前
就已经被占用**，而回收动作写在生成器的 `finally` 里。生成器如果从未被 `__anext__()`
过，`finally` 就永远不会执行——Python 不会为一个从未启动的生成器运行收尾代码。

实测（ASGI 层直接调用，客户端在响应头到达前断开）：
  - `execute_chat` 已经拿到上游首包（`provider.chat` 的 finally 未执行），
    节点并发信号量仍被持有；
  - `/v1/responses` 的 `load_balancer.acquire()` 在返回 `StreamingResponse` **之前**
    就调用了，配额永久泄漏，路由会把该节点逐步判为高负载并持续分流走。

这不是「断连」问题：`guard_stream` 只在生成器已经启动时才能观察到中断。这里要
保证的是**构造阶段就持有资源时，任何未启动路径都必须回收**。覆盖：

  1. 响应体从未被迭代 —— 资源仍必须归还（LB 配额、上游流、信号量、飞行组）；
  2. 正常完整流式 —— 不得重复回收（幂等），也不得改变既有行为；
  3. 上游中途失败 —— 仍按原语义在流内报错并回收一次。

运行: .venv/bin/python tests/test_stream_start_reclaim.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["TESTING"] = "1"
os.environ["GATEWAY_DISABLE_AUTO_REFRESH"] = "1"
os.environ["GATEWAY_DISABLE_CREDENTIAL_WATCH"] = "1"
os.environ["GATEWAY_DISABLE_PROBING"] = "1"
os.environ["GATEWAY_DISABLE_PACING"] = "1"
os.environ["GATEWAY_DISABLE_SINGLEFLIGHT"] = "1"
os.environ["QWEN_AUTH_TOKEN"] = "fake_qwen"
os.environ["DEEPSEEK_AUTH_TOKEN"] = "fake_ds"
os.environ["DOUBAO_AUTH_TOKEN"] = "fake_doubao"
os.environ["KIMI_AUTH_TOKEN"] = "fake_kimi"
os.environ["GLM_AUTH_TOKEN"] = "fake_glm"

import server as srv  # noqa: E402


class _TrackedProvider:
    """记录上游流是否被真正关闭的最小 Provider。"""

    def __init__(self, chunks=("hello ", "world "), fail_after: int | None = None):
        self.opened = 0
        self.closed = 0
        self.chunks = chunks
        self.fail_after = fail_after

    async def chat(self, *args, **kwargs):
        self.opened += 1
        try:
            for idx, chunk in enumerate(self.chunks):
                if self.fail_after is not None and idx >= self.fail_after:
                    raise srv.ProviderError("上游中途失败", status=502, err_type="upstream_error")
                yield chunk, {"conversation_id": "conv_c2"}
            yield "", {"conversation_id": "conv_c2"}
        finally:
            self.closed += 1

    async def delete_conversation(self, *args, **kwargs):
        return None


def _scope(path: str) -> dict:
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.4"},
        "http_version": "1.1",
        "method": "POST",
        "headers": [(b"content-type", b"application/json")],
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "client": ("127.0.0.1", 1),
        "server": ("127.0.0.1", 8000),
        "scheme": "http",
    }


async def _drive(provider, path: str, body: bytes, *, send_frames: int = 0, idle_timeout: float = 0.3):
    """以一个「在 send 上失败」的客户端驱动整个 ASGI 栈。

    `send_frames=0` 表示客户端在响应头到达前就已消失（生成器一帧都不会被拉取）；
    大于 0 表示先成功接收若干帧再断开。
    """
    srv._make_provider = lambda key, client=None: provider
    sent = {"count": 0, "types": []}

    async def receive():
        if not sent["types"]:
            return {"type": "http.request", "body": body, "more_body": False}
        await asyncio.sleep(idle_timeout)
        return {"type": "http.disconnect"}

    async def send(message):
        sent["types"].append(message["type"])
        sent["count"] += 1
        if sent["count"] > send_frames:
            raise OSError("client gone")

    try:
        await asyncio.wait_for(srv.app(_scope(path), receive, send), timeout=10)
    except BaseException:
        pass
    # 给收尾任务（独立 create_task / 生成器回收）留出调度机会
    for _ in range(50):
        await asyncio.sleep(0.01)
    return sent


def _sem_value(provider_key: str) -> int:
    return srv.failover_router.cooldown_tracker.get_semaphore(provider_key)._value


OPENAI_BODY = b'{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}],"stream":true}'
ANTHROPIC_BODY = (
    b'{"model":"claude-3-5-sonnet","max_tokens":16,'
    b'"messages":[{"role":"user","content":"hi"}],"stream":true}'
)
RESPONSES_BODY = b'{"model":"gpt-4o","input":"hi","stream":true}'


def _reset_counters() -> None:
    for key in ("qwen", "glm"):
        metrics = srv.load_balancer.get_metrics(key)
        while metrics.active_concurrency > 0:
            metrics.release(success=False)


def test_bodies_never_started_still_release_everything():
    """响应体一帧都没被拉取时，配额/上游流/信号量必须全部归还。"""
    cases = [
        ("/v1/chat/completions", OPENAI_BODY),
        ("/v1/messages", ANTHROPIC_BODY),
        ("/v1/responses", RESPONSES_BODY),
    ]
    async def _run():
        leaks = []
        for path, body in cases:
            _reset_counters()
            provider = _TrackedProvider()
            before_sem = _sem_value("qwen")
            await _drive(provider, path, body, send_frames=0)
            lb = srv.load_balancer.get_metrics("qwen").active_concurrency
            sem = _sem_value("qwen")
            if lb != 0:
                leaks.append(f"{path}: LB 配额泄漏 {lb}")
            if sem != before_sem:
                leaks.append(f"{path}: 节点信号量泄漏 {before_sem} -> {sem}")
            if provider.opened and provider.closed != provider.opened:
                leaks.append(f"{path}: 上游流未关闭 {provider.opened} 开 / {provider.closed} 关")
        return leaks

    leaks = asyncio.run(_run())
    assert not leaks, "\n".join(leaks)
    print("test_bodies_never_started_still_release_everything: PASSED")


def test_complete_stream_releases_exactly_once():
    """正常跑完的流不得被重复回收（重复释放会把配额计数打到负数）。"""
    async def _run():
        _reset_counters()
        provider = _TrackedProvider()
        sent = await _drive(provider, "/v1/chat/completions", OPENAI_BODY, send_frames=10_000)
        return sent, srv.load_balancer.get_metrics("qwen").active_concurrency, provider

    sent, lb, provider = asyncio.run(_run())
    assert "http.response.start" in sent["types"]
    assert lb == 0, f"完整流结束后配额未归零: {lb}"
    assert provider.closed == provider.opened == 1
    print("test_complete_stream_releases_exactly_once: PASSED")


def test_upstream_failure_mid_stream_still_errors_in_band():
    """上游中途失败：仍在流内报错（不因回收改动变成异常抛出），并回收一次。"""
    async def _run():
        _reset_counters()
        provider = _TrackedProvider(fail_after=1)
        return await _drive(provider, "/v1/chat/completions", OPENAI_BODY, send_frames=10_000)

    sent = asyncio.run(_run())
    assert "http.response.start" in sent["types"], "上游已经吐过内容，状态行无法再改"
    assert srv.load_balancer.get_metrics("qwen").active_concurrency == 0
    print("test_upstream_failure_mid_stream_still_errors_in_band: PASSED")


def test_managed_stream_closes_never_started_source():
    """回收原语：`ManagedStream.close()` 必须收尾一条**从未被迭代**的源流。

    这是整条防线真正的承重点。Python 对从未启动的异步生成器 `aclose()` 是空操作，
    其 `finally` 一次都不会执行——这意味着**任何包装层**（`sse_keepalive` /
    `guard_stream`）都无法靠自身 `aclose` 把关闭传播下去，它们同样从未启动。
    因此关闭必须直达最内层、并由它先推进一帧再关闭。若本用例失败，上面三条
    「响应体从未拉取」的端到端用例只是碰巧通过。
    """
    from providers import ManagedStream

    closed = {"n": 0}
    released = {"n": 0}

    async def source():
        try:
            yield "a"
        finally:
            closed["n"] += 1

    async def _run():
        managed = ManagedStream(source(), close=lambda: released.__setitem__("n", released["n"] + 1))
        await managed.close()
        await managed.close()  # 幂等：重复关闭不得再次执行源流的 finally

    asyncio.run(_run())
    assert closed["n"] == 1, (
        "从未被迭代的源流在 close() 时没有执行 finally——上游连接与并发配额随之泄漏"
    )
    assert released["n"] == 1, "槽位回收钩子必须恰好执行一次"
    print("test_managed_stream_closes_never_started_source: PASSED")


def run_all():
    test_bodies_never_started_still_release_everything()
    test_complete_stream_releases_exactly_once()
    test_upstream_failure_mid_stream_still_errors_in_band()
    test_managed_stream_closes_never_started_source()
    print("\n✅ All stream start-reclaim tests passed!")


if __name__ == "__main__":
    run_all()