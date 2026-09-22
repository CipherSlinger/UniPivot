"""全局 HTTP 客户端连接池管理与单例维护。"""

from __future__ import annotations

import asyncio
from typing import Optional

import httpx

try:
    import h2  # noqa: F401
    _HTTP2_SUPPORTED = True
except ImportError:
    _HTTP2_SUPPORTED = False

_ORIGINAL_ASYNC_CLIENT = httpx.AsyncClient
_shared_client: Optional[httpx.AsyncClient] = None
_client_lock = asyncio.Lock()


def get_default_limits() -> httpx.Limits:
    """返回默认的 HTTP 连接池限制：最大连接数 100，保活连接数 40，保活过期 30s。"""
    return httpx.Limits(
        max_connections=100,
        max_keepalive_connections=40,
        keepalive_expiry=30.0,
    )


def create_async_client(timeout: float = 600.0) -> httpx.AsyncClient:
    """创建配置了连接池与 HTTP/2 的 AsyncClient 实例。"""
    return httpx.AsyncClient(
        timeout=httpx.Timeout(timeout, connect=15.0, pool=15.0),
        limits=get_default_limits(),
        http2=_HTTP2_SUPPORTED,
        follow_redirects=True,
    )


async def get_http_client(timeout: float = 600.0) -> httpx.AsyncClient:
    """获取全局共享的 AsyncClient 单例（延迟初始化，协程安全）。"""
    global _shared_client

    # 单元测试兼容：若在测试环境中 mock 了 httpx.AsyncClient，则返回 mock 实例
    if httpx.AsyncClient is not _ORIGINAL_ASYNC_CLIENT:
        return create_async_client(timeout)

    if _shared_client is not None and not _shared_client.is_closed:
        return _shared_client

    async with _client_lock:
        if _shared_client is None or _shared_client.is_closed:
            _shared_client = create_async_client(timeout)
        return _shared_client


async def close_http_client() -> None:
    """优雅关闭全局 AsyncClient 单例连接池。"""
    global _shared_client
    async with _client_lock:
        if _shared_client is not None:
            if not _shared_client.is_closed:
                await _shared_client.aclose()
            _shared_client = None
