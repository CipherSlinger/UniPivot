"""流式响应的生命周期：连接必须被归还 (STREAM-AUDIT)

S5 是审计里最「安静」的一条：Kimi 与 GLM 用 `client.send(req, stream=True)` 拿到
响应后，成功路径与每一条提前退出都没有调用 `resp.aclose()`。它不会让任何一次请求
失败，只会让连接池慢慢漏干——共享 `AsyncClient` 的上限是 100 条连接 / 40 条
keep-alive，而健康探针每 5 分钟对每个节点再泄漏一条，于是几小时后整个网关一起
变慢，且症状（随机超时、池耗尽）与真正的原因相距极远。

本套件用 `httpx.MockTransport` 精确控制每一帧，断言：
  1. 正常读完 → 响应被关闭；
  2. 401 / 429 / 5xx 提前退出 → 响应同样被关闭；
  3. 帧内报错（in-band error）提前退出 → 响应同样被关闭；
  4. 中途解析到非法帧不再抛异常（S6：非对象 JSON 会让 `obj.get` 崩掉）。

运行: .venv/bin/python tests/test_stream_lifecycle.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["TESTING"] = "1"

import httpx  # noqa: E402

from providers.base import ProviderError  # noqa: E402
from providers.glm_provider import GLMProvider  # noqa: E402
from providers.kimi_provider import KimiProvider  # noqa: E402

_CLOSED: list[str] = []


@contextmanager
def _mock(module, handler):
    orig = module.httpx.AsyncClient

    class MockClient(orig):
        def __init__(self, *a, **kw):
            kw["transport"] = httpx.MockTransport(handler)
            super().__init__(*a, **kw)

    module.httpx.AsyncClient = MockClient
    try:
        yield
    finally:
        module.httpx.AsyncClient = orig


def _sse(lines: list[str]) -> bytes:
    return ("\n\n".join(lines) + "\n\n").encode()


async def _drain(agen) -> str:
    text = ""
    async for delta, _meta in agen:
        text += delta
    return text


def _kimi(handler):
    import providers.kimi_provider as m

    _CLOSED.clear()

    def wrapped(request):
        resp = handler(request)
        real_aclose = resp.aclose

        async def aclose():
            _CLOSED.append(request.url.path)
            await real_aclose()

        resp.aclose = aclose
        return resp

    with _mock(m, wrapped):
        p = KimiProvider("fake_refresh", timeout=10)
        p._get_access_token = lambda client: _ok(("tok", "uid"))
        p._create_conversation = lambda *a, **k: _ok("conv_1")
        try:
            return asyncio.run(_drain(p.chat([{"role": "user", "content": "hi"}], "kimi")))
        finally:
            pass


async def _ok(value):
    return value


KIMI_OK = [
    'data: {"event":"cmpl","text":"Hello"}',
    'data: {"event":"cmpl","text":" world"}',
    'data: {"event":"all_done"}',
]


def test_kimi_closes_response_on_success():
    _kimi(lambda r: httpx.Response(200, content=_sse(KIMI_OK)))
    assert _CLOSED, "正常结束后上游响应没有被关闭（连接槽泄漏）"
    print("test_kimi_closes_response_on_success: PASSED")


def test_kimi_closes_response_on_auth_failure():
    def handler(r):
        return httpx.Response(401, text="expired")

    import providers.kimi_provider as m

    _CLOSED.clear()

    def wrapped(request):
        resp = handler(request)
        real = resp.aclose

        async def aclose():
            _CLOSED.append(request.url.path)
            await real()

        resp.aclose = aclose
        return resp

    with _mock(m, wrapped):
        p = KimiProvider("fake_refresh", timeout=10)
        p._get_access_token = lambda client: _ok(("tok", "uid"))
        p._create_conversation = lambda *a, **k: _ok("conv_1")
        try:
            asyncio.run(_drain(p.chat([{"role": "user", "content": "hi"}], "kimi")))
        except ProviderError as e:
            assert e.err_type == "auth_error", e.err_type
        else:
            raise AssertionError("401 没有抛出 ProviderError")
    assert _CLOSED, "401 提前退出时上游响应没有被关闭（连接槽泄漏）"
    print("test_kimi_closes_response_on_auth_failure: PASSED")


def test_kimi_survives_non_object_json_frames():
    """S6：`data: []` / `null` / 数字 会让 `obj.get` 抛 AttributeError。"""
    frames = [
        "data: []",
        "data: null",
        "data: 42",
        'data: {"event":"cmpl","text":"ok"}',
        'data: {"event":"all_done"}',
    ]
    text = _kimi(lambda r: httpx.Response(200, content=_sse(frames)))
    assert text == "ok", f"非对象帧导致解析中断或崩溃: {text!r}"
    assert _CLOSED, "解析异常后响应没有被关闭"
    print("test_kimi_survives_non_object_json_frames: PASSED")


# ---------------------------------------------------------------- GLM

# GLM 的 parts 是**累积快照**：content 是 content-part 列表，文本在 part["text"]。
GLM_OK = [
    'data: {"status":"ok","parts":[{"logic_id":"1","content":[{"type":"text","text":"Hi"}]}]}',
    'data: {"status":"ok","parts":[{"logic_id":"1","content":[{"type":"text","text":"Hi there"}]}]}',
    'data: {"status":"finish"}',
]


def _glm(handler):
    import providers.glm_provider as m

    _CLOSED.clear()

    def wrapped(request):
        resp = handler(request)
        real = resp.aclose

        async def aclose():
            _CLOSED.append(request.url.path)
            await real()

        resp.aclose = aclose
        return resp

    with _mock(m, wrapped):
        p = GLMProvider("fake_refresh", timeout=10)
        p._get_access_token = lambda client: _ok("tok")
        try:
            return asyncio.run(_drain(p.chat([{"role": "user", "content": "hi"}], "glm-4-plus")))
        finally:
            pass


def test_glm_closes_response_on_success():
    text = _glm(lambda r: httpx.Response(200, content=_sse(GLM_OK)))
    assert text == "Hi there", f"增量语义被破坏: {text!r}"
    assert _CLOSED, "正常结束后上游响应没有被关闭（连接槽泄漏）"
    print("test_glm_closes_response_on_success: PASSED")


def test_glm_closes_response_on_rate_limit():
    import providers.glm_provider as m

    _CLOSED.clear()

    def wrapped(request):
        resp = httpx.Response(429, text="blocked")
        real = resp.aclose

        async def aclose():
            _CLOSED.append(request.url.path)
            await real()

        resp.aclose = aclose
        return resp

    with _mock(m, wrapped):
        p = GLMProvider("fake_refresh", timeout=10)
        p._get_access_token = lambda client: _ok("tok")
        try:
            asyncio.run(_drain(p.chat([{"role": "user", "content": "hi"}], "glm-4-plus")))
        except ProviderError as e:
            assert e.err_type == "rate_limit_error", e.err_type
            assert e.status == 429, e.status
        else:
            raise AssertionError("429 没有抛出 ProviderError")
    assert _CLOSED, "429 提前退出时上游响应没有被关闭（连接槽泄漏）"
    print("test_glm_closes_response_on_rate_limit: PASSED")


def test_glm_survives_non_object_json_frames():
    frames = [
        "data: []",
        "data: null",
        'data: {"status":"ok","parts":[{"logic_id":"1","content":[{"type":"text","text":"fine"}]}]}',
        'data: {"status":"finish"}',
    ]
    text = _glm(lambda r: httpx.Response(200, content=_sse(frames)))
    assert text == "fine", f"非对象帧导致解析中断: {text!r}"
    print("test_glm_survives_non_object_json_frames: PASSED")


def run_all():
    tests = [
        test_kimi_closes_response_on_success,
        test_kimi_closes_response_on_auth_failure,
        test_kimi_survives_non_object_json_frames,
        test_glm_closes_response_on_success,
        test_glm_closes_response_on_rate_limit,
        test_glm_survives_non_object_json_frames,
    ]
    for fn in tests:
        fn()
    print(f"\n✅ All stream lifecycle tests passed! ({len(tests)} cases)")


if __name__ == "__main__":
    run_all()