"""在途请求优雅停机排空 (OPT-015) 测试套件

验证：
1. 无在途请求时排空立即返回
2. 有在途请求时等待其收尾后返回
3. 超时仍未收尾时返回 False（有界等待，绝不无限阻塞）
4. InflightTrackerMiddleware 正确统计在途请求数
5. 中间件覆盖流式响应全程（发送完毕后计数才归零）
6. 非 HTTP scope (lifespan/websocket) 不影响计数
7. /health 与 /v1/diagnostics 暴露 inflight_requests

运行: .venv/bin/python tests/test_graceful_drain.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["GATEWAY_DISABLE_AUTO_REFRESH"] = "1"
os.environ["TESTING"] = "1"
os.environ["QWEN_AUTH_TOKEN"] = "fake_qwen"
os.environ["DEEPSEEK_AUTH_TOKEN"] = "fake_ds"
os.environ["DOUBAO_AUTH_TOKEN"] = "fake_doubao"
os.environ["KIMI_AUTH_TOKEN"] = "fake_kimi"
os.environ["GLM_AUTH_TOKEN"] = "fake_glm"

from fastapi.testclient import TestClient  # noqa: E402

import server as srv  # noqa: E402


def _reset():
    srv._inflight_requests = 0


def test_drain_when_idle_returns_immediately():
    _reset()
    assert asyncio.run(srv._drain_inflight_requests(5.0)) is True
    print("test_drain_when_idle_returns_immediately: PASSED")


def test_drain_waits_for_completion():
    _reset()
    srv._inflight_requests = 1

    async def _release():
        await asyncio.sleep(0.05)
        srv._inflight_requests = 0

    async def _run():
        return await asyncio.gather(srv._drain_inflight_requests(2.0), _release())

    drained, _ = asyncio.run(_run())
    assert drained is True, "在途请求收尾后应排空成功"
    _reset()
    print("test_drain_waits_for_completion: PASSED")


def test_drain_times_out_when_busy():
    _reset()
    srv._inflight_requests = 1
    assert asyncio.run(srv._drain_inflight_requests(0.1)) is False
    _reset()
    print("test_drain_times_out_when_busy: PASSED")


def test_middleware_counts_inflight():
    _reset()
    seen = []

    async def _send(message):
        return None

    async def _app(scope, receive, send):
        seen.append(srv._inflight_requests)
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok", "more_body": False})

    mw = srv.InflightTrackerMiddleware(_app)
    asyncio.run(mw({"type": "http"}, None, _send))
    assert seen == [1], f"应用执行期间在途数应为 1，实际 {seen}"
    assert srv._inflight_requests == 0, "请求结束后在途数必须归零"
    print("test_middleware_counts_inflight: PASSED")


def test_middleware_spans_streaming_response():
    _reset()
    during = []

    async def _send(message):
        return None

    async def _stream_app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        for _ in range(4):
            during.append(srv._inflight_requests)
            await send({"type": "http.response.body", "body": b"x", "more_body": True})
            await asyncio.sleep(0.01)
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    mw = srv.InflightTrackerMiddleware(_stream_app)
    asyncio.run(mw({"type": "http"}, None, _send))
    assert during == [1, 1, 1, 1], f"流式全程在途数应恒为 1，实际 {during}"
    assert srv._inflight_requests == 0
    print("test_middleware_spans_streaming_response: PASSED")


def test_middleware_ignores_non_http_scope():
    _reset()
    calls = []

    async def _app(scope, receive, send):
        calls.append(scope.get("type"))

    mw = srv.InflightTrackerMiddleware(_app)
    asyncio.run(mw({"type": "lifespan"}, None, None))
    asyncio.run(mw({"type": "websocket"}, None, None))
    assert calls == ["lifespan", "websocket"]
    assert srv._inflight_requests == 0
    print("test_middleware_ignores_non_http_scope: PASSED")


def test_health_and_diagnostics_expose_inflight():
    _reset()
    client = TestClient(srv.app)
    h = client.get("/health").json()
    assert "inflight_requests" in h
    d = client.get("/v1/diagnostics").json()
    assert "inflight_requests" in d
    print("test_health_and_diagnostics_expose_inflight: PASSED")


def test_observer_endpoints_do_not_count_themselves():
    """[OPT-015] 一台空闲网关的在途数必须是 0，而不是「1」或「N」。

    实测缺陷：诊断请求把自己计进了被读的量里，空闲网关永远报 1；并发 n 个
    诊断请求就报 n。这同时破坏了优雅停机的判据——`_drain_inflight_requests`
    以「计数归零」认定排空，而计数永远不会归零，于是每次停机都白等满
    GATEWAY_SHUTDOWN_DRAIN_TIMEOUT。
    """
    _reset()

    async def _send(message):
        return None

    async def _app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"{}", "more_body": False})

    mw = srv.InflightTrackerMiddleware(_app)
    for path in ("/health", "/v1/diagnostics", "/diagnostics"):
        asyncio.run(mw({"type": "http", "path": path}, None, _send))
        assert srv._inflight_requests == 0, f"{path} 不得计入在途数"

    # 普通业务路径照常计数，且观测端点**不改变**它读到的值
    seen = []

    async def _work(scope, receive, send):
        seen.append(srv._inflight_requests)
        await _app(scope, receive, send)

    mw_work = srv.InflightTrackerMiddleware(_work)
    asyncio.run(mw_work({"type": "http", "path": "/v1/messages"}, None, _send))
    assert seen == [1], f"业务请求在途时应读到 1，实际 {seen}"
    assert srv._inflight_requests == 0

    # 端到端：空闲网关读诊断为 0
    client = TestClient(srv.app)
    assert client.get("/v1/diagnostics").json()["inflight_requests"] == 0
    assert client.get("/health").json()["inflight_requests"] == 0
    print("test_observer_endpoints_do_not_count_themselves: PASSED")


def run_all():
    test_drain_when_idle_returns_immediately()
    test_drain_waits_for_completion()
    test_drain_times_out_when_busy()
    test_middleware_counts_inflight()
    test_middleware_spans_streaming_response()
    test_middleware_ignores_non_http_scope()
    test_health_and_diagnostics_expose_inflight()
    test_observer_endpoints_do_not_count_themselves()
    print("\n✅ All graceful in-flight drain tests (OPT-015) passed!")


if __name__ == "__main__":
    run_all()
