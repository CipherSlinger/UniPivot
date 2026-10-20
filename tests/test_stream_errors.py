"""上游流式响应的错误分类 (STREAM-AUDIT)

动机来自流式解析审计的 S1：Qwen 与 DeepSeek 都在 `async with client.stream(...)`
**内部**调用 `resp.raise_for_status()`，异常抛出时 `async with` 已经关闭了响应，
外层处理器再读 `e.response.text` 只会得到 `httpx.StreamClosed` / `ResponseNotRead`。
后果不是「少一条日志」，而是：

  1. 一个「读不到内容」的异常盖住了原始状态码，风控分类彻底失效；
  2. 403（Cloudflare / WAF）与 429（限流/人机验证）被塌成通用 502，
     风控账本因此无法分级 —— 对 DeepSeek（本部署里已被封禁的那一家）尤其致命：
     它的 403 正是避让逻辑存在的输入；
  3. 401 由各 provider 手写的分支勉强保住，但错误体同样丢失。

本套件用 `httpx.MockTransport` 完整离线地断言：错误响应体在流关闭**之前**被读出、
原始状态码与 err_type 在 401/403/429 上被如实保留、5xx 仍是 502。

运行: .venv/bin/python tests/test_stream_errors.py
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
from providers.deepseek_provider import DeepSeekProvider  # noqa: E402
from providers.qwen_provider import QwenProvider  # noqa: E402


@contextmanager
def _mock(module, handler):
    """把某个 provider 模块里的 httpx.AsyncClient 换成挂 MockTransport 的子类。"""
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


async def _drain(agen):
    async for _ in agen:  # noqa: B007
        pass


def _error_for(provider, err_type):
    """跑一次 chat，返回抛出的 ProviderError（必须抛出）。"""
    try:
        asyncio.run(_drain(provider.chat([{"role": "user", "content": "hi"}], "qwen")))
    except ProviderError as e:
        assert e.err_type == err_type, f"期望 err_type={err_type}，实际 {e.err_type}: {e.message}"
        return e
    raise AssertionError(f"上游返回错误却当成正常完成（期望 {err_type}）")


# ---------------------------------------------------------------- Qwen

def test_qwen_waf_403_keeps_status_and_body():
    def handler(request):
        return httpx.Response(
            403, text='{"ret":["FAIL_SYS_USER_VALIDATE::RGV587_ERROR::SM"]}'
        )

    import providers.qwen_provider as m

    with _mock(m, handler):
        err = _error_for(QwenProvider("ticket", timeout=10), "waf_blocked")
    assert err.status == 403, f"403 被塌成 {err.status}，风控账本无法分级"
    assert "RGV587" in err.message, f"错误体在流关闭前没被读到: {err.message}"
    print("test_qwen_waf_403_keeps_status_and_body: PASSED")


def test_qwen_auth_401_keeps_status():
    import providers.qwen_provider as m

    with _mock(m, lambda r: httpx.Response(401, text='{"error":"expired"}')):
        err = _error_for(QwenProvider("ticket", timeout=10), "auth_error")
    assert err.status == 401, err.status
    assert "login.py qwen" in err.message, err.message
    print("test_qwen_auth_401_keeps_status: PASSED")


def test_qwen_rate_limit_429_keeps_status():
    import providers.qwen_provider as m

    with _mock(m, lambda r: httpx.Response(429, text="too many requests")):
        err = _error_for(QwenProvider("ticket", timeout=10), "rate_limit_error")
    assert err.status == 429, err.status
    assert "too many requests" in err.message, err.message
    print("test_qwen_rate_limit_429_keeps_status: PASSED")


def test_qwen_5xx_stays_generic_502():
    import providers.qwen_provider as m

    with _mock(m, lambda r: httpx.Response(503, text="upstream down")):
        err = _error_for(QwenProvider("ticket", timeout=10), "upstream_error")
    assert err.status == 502, err.status
    assert "upstream down" in err.message, err.message
    print("test_qwen_5xx_stays_generic_502: PASSED")


# ---------------------------------------------------------------- DeepSeek

def _deepseek() -> DeepSeekProvider:
    p = DeepSeekProvider(token="fake_ds", timeout=10)
    # 跳过 PoW 挑战：本套件只关心完成请求的错误分类
    async def _no_pow(client):
        return "pow-header"

    p._pow_header = _no_pow
    return p


def test_deepseek_cloudflare_403_keeps_status_and_body():
    """本部署里被封禁的那一家：403 是避让逻辑唯一的输入，绝不能变成 502。"""

    def handler(request):
        return httpx.Response(
            403, text="<html><title>Attention Required! | Cloudflare</title></html>"
        )

    import providers.deepseek_provider as m

    with _mock(m, handler):
        err = _error_for(_deepseek(), "waf_blocked")
    assert err.status == 403, f"Cloudflare 403 被塌成 {err.status}"
    assert "Cloudflare" in err.message, f"错误体在流关闭前没被读到: {err.message}"
    print("test_deepseek_cloudflare_403_keeps_status_and_body: PASSED")


def test_deepseek_auth_401_keeps_status():
    import providers.deepseek_provider as m

    async def _no_session(client):
        return "sess_1"

    p = _deepseek()
    p._create_chat_session = _no_session
    with _mock(m, lambda r: httpx.Response(401, text='{"code":40003}')):
        err = _error_for(p, "auth_error")
    assert err.status == 401, err.status
    print("test_deepseek_auth_401_keeps_status: PASSED")


def test_deepseek_5xx_stays_generic_502():
    import providers.deepseek_provider as m

    async def _no_session(client):
        return "sess_1"

    p = _deepseek()
    p._create_chat_session = _no_session
    with _mock(m, lambda r: httpx.Response(502, text="bad gateway")):
        err = _error_for(p, "upstream_error")
    assert err.status == 502, err.status
    assert "bad gateway" in err.message, err.message
    print("test_deepseek_5xx_stays_generic_502: PASSED")


def test_error_body_is_read_inside_the_stream_context():
    """直接锁住失败形态本身：错误体可读，且不是 StreamClosed。"""
    import providers.qwen_provider as m

    def handler(request):
        return httpx.Response(403, text="BODY-IS-READABLE")

    with _mock(m, handler):
        err = _error_for(QwenProvider("ticket", timeout=10), "waf_blocked")
    assert "BODY-IS-READABLE" in err.message, (
        f"错误体没有被读到（StreamClosed 覆盖了它）: {err.message}"
    )
    assert "StreamClosed" not in err.message and "ResponseNotRead" not in err.message, err.message
    print("test_error_body_is_read_inside_the_stream_context: PASSED")


def run_all():
    tests = [
        test_qwen_waf_403_keeps_status_and_body,
        test_qwen_auth_401_keeps_status,
        test_qwen_rate_limit_429_keeps_status,
        test_qwen_5xx_stays_generic_502,
        test_deepseek_cloudflare_403_keeps_status_and_body,
        test_deepseek_auth_401_keeps_status,
        test_deepseek_5xx_stays_generic_502,
        test_error_body_is_read_inside_the_stream_context,
    ]
    for fn in tests:
        fn()
    print(f"\n✅ All stream error-classification tests passed! ({len(tests)} cases)")


if __name__ == "__main__":
    run_all()