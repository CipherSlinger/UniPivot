"""流式请求的派发前拒绝必须体现在状态行上 (tests/test_stream_preflight.py)。

[H4] `execute_chat` 在**第一次上游调用之前**就可能拒绝请求：凭证整环失效
（OPT-025）、运维禁用（OPT-033）、封禁租约（OPT-021）、能力环耗尽（OPT-031）。
这些都是「零上游调用」的确定性结论，没有任何内容需要流式传输。

过去两条流式路径把这次 await 留在响应生成器里，于是响应头已经以 **200** 发出，
客户端读到的是 `data: {"error": ...}`：

  - 多数 SDK 先看状态码，200 即当成功，错误体被丢掉或解析失败 → **静默失败**；
  - 客户端以为请求成功并继续 agent 循环，下一轮再撞一次墙。

而 `/v1/responses` 走的是返回前的非流式分支，一直如实返回 503——同一条协议纪律在
不同路径上有两种结果，正是本项目反复记录的那类缺陷。

本套件锁住：三条协议在「派发前拒绝」时都返回真实状态码与错误类型；而**已经拿到
上游流之后**的失败仍然走流内错误事件（不能因为在头部之前提前 await 就把后续内容
也缓冲起来——那样会牺牲流式本身）。

运行: .venv/bin/python tests/test_stream_preflight.py
"""

from __future__ import annotations

import json
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

from fastapi.testclient import TestClient  # noqa: E402

import server as srv  # noqa: E402
from providers.base import ProviderError  # noqa: E402

OPENAI_BODY = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}], "stream": True}
ANTHROPIC_BODY = {
    "model": "claude-3-5-sonnet",
    "max_tokens": 16,
    "messages": [{"role": "user", "content": "hi"}],
    "stream": True,
}


def _rejecting_router(status: int, err_type: str, message: str):
    async def execute_chat(**kwargs):
        raise ProviderError(message, status=status, err_type=err_type)

    return execute_chat


def test_openai_stream_reports_pre_dispatch_rejection_in_status():
    original = srv.failover_router.execute_chat
    srv.failover_router.execute_chat = _rejecting_router(
        503, "ring_exhausted", "能力环内全部节点停止派发（零上游调用）"
    )
    try:
        with TestClient(srv.app).stream(
            "POST", "/v1/chat/completions", json=OPENAI_BODY
        ) as r:
            assert r.status_code == 503, (
                f"流式请求的派发前拒绝被报成了 HTTP {r.status_code}：客户端会把 200 当成功，"
                f"错误被静默丢弃"
            )
            body = "".join(r.iter_text())
    finally:
        srv.failover_router.execute_chat = original
    assert json.loads(body)["error"]["type"] == "ring_exhausted"
    print("test_openai_stream_reports_pre_dispatch_rejection_in_status: PASSED")


def test_anthropic_stream_reports_pre_dispatch_rejection_in_status():
    original = srv.failover_router.execute_chat
    srv.failover_router.execute_chat = _rejecting_router(
        503, "quarantined", "节点处于封禁租约（零上游调用）"
    )
    try:
        with TestClient(srv.app).stream(
            "POST", "/v1/messages", json=ANTHROPIC_BODY
        ) as r:
            assert r.status_code == 503, r.text
            body = "".join(r.iter_text())
    finally:
        srv.failover_router.execute_chat = original
    payload = json.loads(body)
    assert payload["type"] == "error"
    # Anthropic 枚举里没有「封禁」，最贴近的语义是「服务当下无法承接」
    assert payload["error"]["type"] == "overloaded_error"
    print("test_anthropic_stream_reports_pre_dispatch_rejection_in_status: PASSED")


def test_all_protocols_agree_on_the_same_rejection():
    original = srv.failover_router.execute_chat
    srv.failover_router.execute_chat = _rejecting_router(
        503, "credential_outage", "全部凭证失效"
    )
    try:
        client = TestClient(srv.app)
        statuses = {
            "openai": client.post("/v1/chat/completions", json=OPENAI_BODY).status_code,
            "anthropic": client.post("/v1/messages", json=ANTHROPIC_BODY).status_code,
            "responses": client.post("/v1/responses", json={
                "model": "gpt-4o", "input": "hi", "stream": True,
            }).status_code,
        }
    finally:
        srv.failover_router.execute_chat = original
    assert set(statuses.values()) == {503}, (
        f"同一次拒绝在三条协议上报出了不同状态码: {statuses}"
    )
    print("test_all_protocols_agree_on_the_same_rejection: PASSED")


def test_failure_after_stream_opened_still_uses_in_band_error():
    """拿到上游流之后的失败必须留在流内：提前 await 不能变成「缓冲整个响应」。"""

    class LateFailProvider:
        async def chat(self, messages, model, conversation_id=None, **kw):
            yield "已经开始的正文", {}
            raise ProviderError("上游中途失败", status=502, err_type="upstream_error")

    srv._make_provider = lambda key, client=None: LateFailProvider()
    with TestClient(srv.app).stream(
        "POST", "/v1/chat/completions", json=OPENAI_BODY
    ) as r:
        assert r.status_code == 200, "上游已经输出过内容，此时无法再改状态行"
        body = "".join(r.iter_text())
    assert "[DONE]" not in body
    errors = []
    for line in body.splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            try:
                frame = json.loads(line[6:])
            except ValueError:
                continue
            if "error" in frame:
                errors.append(frame["error"])
    assert errors, f"流内失败没有作为事件发出:\n{body[:600]}"
    print("test_failure_after_stream_opened_still_uses_in_band_error: PASSED")


def run_all():
    test_openai_stream_reports_pre_dispatch_rejection_in_status()
    test_anthropic_stream_reports_pre_dispatch_rejection_in_status()
    test_all_protocols_agree_on_the_same_rejection()
    test_failure_after_stream_opened_still_uses_in_band_error()
    print("\n✅ All stream pre-flight tests passed!")


if __name__ == "__main__":
    run_all()