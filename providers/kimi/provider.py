"""Kimi（Moonshot，kimi.moonshot.cn）网页端客户端。

协议要点:
  * 鉴权: 抓取 refresh_token（长效）经 /api/auth/token/refresh 换取短效 access_token，
          请求头携带 Authorization: Bearer <access_token>。
  * 预创建会话: POST https://kimi.moonshot.cn/api/chat
          body: {"name": "未命名会话", "is_example": false, "kimiplus_id": "kimi", "enter_method": "new_chat"}
          返回: {"id": "<conv_id>"}
  * 补全流: POST https://kimi.moonshot.cn/api/chat/{conv_id}/completion/stream
          SSE 流，支持思考（k1 / think）、搜索（search）、数学（math）以及模型切分。
  * 删会话: DELETE https://kimi.moonshot.cn/api/chat/{conv_id}
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from typing import AsyncIterator, Dict, List, Optional, Tuple

import httpx

from ..base import (
    ProviderError,
    classify_stream_status,
    ensure_async_client,
    flatten_messages,
    continuation_content,
    parse_cookie_header,
    read_error_body,
)
from ..device_identity import stable_device_id
from ..stream_completion import COMPLETION_KEY
from ..http_client import get_http_client


def _new_kimi_device_id() -> str:
    """Kimi 设备标识的格式：19 位十进制（首位 7）。仅在首次生成时调用一次。"""
    return str(random.randint(7000000000000000000, 7999999999999999999))


def _new_kimi_session_id() -> str:
    """Kimi 会话标识的格式：19 位十进制（首位 1）。仅在首次生成时调用一次。"""
    return str(random.randint(1700000000000000000, 1799999999999999999))

BASE_URL = "https://kimi.moonshot.cn"
REFRESH_ENDPOINT = "/api/auth/token/refresh"
CHAT_ENDPOINT = "/api/chat"

# 模型映射
MODEL_MAP = {
    "kimi": "kimi",
    "kimi-chat": "kimi",
    "kimi-latest": "kimi",
    "kimi-k0-math": "kimi-math",
    "kimi-math": "kimi-math",
    "kimi-explore": "kimi-explore",
    "kimi-research": "kimi-explore",
    "kimi-think": "kimi-explore",
    "kimi-k1": "kimi-k1",
    "moonshot-v1-8k": "kimi",
    "moonshot-v1-32k": "kimi",
    "moonshot-v1-128k": "kimi",
}

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

# access_token 缓存池: refreshToken -> {"access_token": str, "expires_at": float, "user_id": str}
_token_cache: Dict[str, dict] = {}
_token_lock = asyncio.Lock()


class KimiProvider:
    def __init__(
        self,
        token: str,
        cookie: Optional[str] = None,
        timeout: float = 600.0,
        user_agent: Optional[str] = None,
        device_id: Optional[str] = None,
        client: Optional[httpx.AsyncClient] = None,
    ):
        """token 为 refresh_token 或 access_token。"""
        if not token:
            raise ProviderError("缺少 Kimi 令牌（refresh_token）", status=401, err_type="auth_error")
        self.refresh_token = token.strip()
        self.cookie = cookie or ""
        self.timeout = timeout
        self.client = client
        if not user_agent or "HeadlessChrome" in user_agent:
            self.user_agent = UA
        else:
            self.user_agent = user_agent

        # [OPT-030] 设备标识必须**跨进程/跨请求恒定**：Provider 是在每次派发尝试时
        # 才实例化的（failover.execute_chat 内 provider_factory），而 session 文件里
        # 的 device_id 为空，因此原先的 `random.randint(...)` 意味着**每一次上游请求
        # 都自称一台新设备**——「同一账号 + 同一令牌 + 每次请求换设备」是反爬系统能
        # 拿到的最强自动化特征。改为生成一次并持久化复用（不随重登改变）。
        self.device_id = device_id or stable_device_id("kimi", _new_kimi_device_id)
        # 会话标识同理：每次请求换会话比换设备更荒谬，同样固化。
        self.session_id = stable_device_id("kimi/session", _new_kimi_session_id)
        self._cookies = parse_cookie_header(self.cookie)

    def _fake_headers(self, access_token: Optional[str] = None) -> dict[str, str]:
        headers = {
            "Accept": "*/*",
            "Accept-Language": "zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "Origin": BASE_URL,
            "R-Timezone": "Asia/Shanghai",
            "User-Agent": self.user_agent,
            "X-Msh-Device-Id": self.device_id,
            "X-Msh-Platform": "web",
            "X-Msh-Session-Id": self.session_id,
        }
        if self.cookie:
            headers["Cookie"] = self.cookie
        if access_token:
            headers["Authorization"] = f"Bearer {access_token}"
        return headers

    async def _get_access_token(self, client: httpx.AsyncClient) -> Tuple[str, str]:
        """获取有效的 access_token 和 user_id。"""
        now = time.time()
        cached = _token_cache.get(self.refresh_token)
        if cached and cached.get("expires_at", 0) > now + 30:
            return cached["access_token"], cached.get("user_id", "")

        async with _token_lock:
            # 双重检查
            cached = _token_cache.get(self.refresh_token)
            if cached and cached.get("expires_at", 0) > now + 30:
                return cached["access_token"], cached.get("user_id", "")

            # 1. 尝试以 refresh_token 刷新 access_token
            refresh_url = f"{BASE_URL}{REFRESH_ENDPOINT}"
            headers = self._fake_headers()
            headers["Authorization"] = f"Bearer {self.refresh_token}"

            access_token = self.refresh_token
            try:
                resp = await client.get(refresh_url, headers=headers, timeout=15.0)
                if resp.status_code == 200:
                    data = resp.json()
                    access_token = data.get("access_token", self.refresh_token)
                elif resp.status_code in (401, 403):
                    # 若 refresh 失败但传入的可能直接是有效的 access_token，则继续向下探测
                    access_token = self.refresh_token
                else:
                    access_token = self.refresh_token
            except Exception:
                access_token = self.refresh_token

            # 2. 获取 user_id 验证有效性
            user_url = f"{BASE_URL}/api/user"
            u_headers = self._fake_headers(access_token)
            user_id = ""
            try:
                u_resp = await client.get(user_url, headers=u_headers, timeout=15.0)
                if u_resp.status_code == 401:
                    raise ProviderError(
                        "Kimi 令牌无效或已过期，请运行 .venv/bin/python login.py kimi 重新登录",
                        status=401,
                        err_type="auth_error",
                    )
                if u_resp.status_code == 200:
                    u_data = u_resp.json()
                    user_id = str(u_data.get("id", ""))
            except ProviderError:
                raise
            except Exception as e:
                # 若无法获取 user 信息，不阻塞流程
                user_id = ""

            _token_cache[self.refresh_token] = {
                "access_token": access_token,
                "expires_at": now + 240,  # 4分钟有效期
                "user_id": user_id,
            }
            return access_token, user_id

    async def _create_conversation(
        self, client: httpx.AsyncClient, access_token: str, user_id: str, model_id: str
    ) -> str:
        headers = self._fake_headers(access_token)
        if user_id:
            headers["X-Traffic-Id"] = user_id
        headers["Content-Type"] = "application/json"

        payload = {
            "name": "未命名会话",
            "is_example": False,
            "kimiplus_id": model_id if len(model_id) == 20 else "kimi",
            "enter_method": "new_chat",
        }
        resp = await client.post(f"{BASE_URL}{CHAT_ENDPOINT}", headers=headers, json=payload, timeout=15.0)
        if resp.status_code == 401:
            raise ProviderError(
                "Kimi 令牌无效或已过期，请运行 .venv/bin/python login.py kimi 重新登录",
                status=401,
                err_type="auth_error",
            )
        if resp.status_code == 429:
            raise ProviderError("Kimi 触发限流，请稍后重试", status=429, err_type="rate_limit_error")
        if resp.status_code != 200:
            raise ProviderError(f"Kimi 创建会话失败: HTTP {resp.status_code} {resp.text[:200]}", status=502)

        data = resp.json()
        conv_id = data.get("id")
        if not conv_id:
            raise ProviderError(f"Kimi 返回会话异常: {data}", status=502)
        return conv_id

    async def chat(
        self,
        messages: List[dict],
        model: str,
        *,
        conversation_id: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        thinking: bool = False,
        search: bool = False,
        stream: bool = True,
    ) -> AsyncIterator[Tuple[str, dict]]:
        wire_model = MODEL_MAP.get(model.lower(), "kimi")
        is_math = "math" in wire_model or "math" in model.lower()
        is_explore = "explore" in wire_model or "research" in wire_model or thinking
        is_k1 = "k1" in wire_model or "explore" in wire_model or thinking

        async with ensure_async_client(self.client, self.timeout) as client:
            access_token, user_id = await self._get_access_token(client)

            created_new_conv = False
            conv_id = conversation_id
            if not conv_id:
                conv_id = await self._create_conversation(client, access_token, user_id, wire_model)
                created_new_conv = True

            prompt = continuation_content(messages) if not created_new_conv else flatten_messages(messages)

            send_messages = [{"role": "user", "content": prompt}]
            stream_url = f"{BASE_URL}/api/chat/{conv_id}/completion/stream"

            headers = self._fake_headers(access_token)
            if user_id:
                headers["X-Traffic-Id"] = user_id
            headers["Content-Type"] = "application/json"
            headers["Referer"] = f"{BASE_URL}/chat/{conv_id}"

            payload = {
                "kimiplus_id": "kimi",
                "messages": send_messages,
                "refs": [],
                "refs_file": [],
                "model": "k1" if is_k1 else "kimi",
                "scene_labels": [],
                "use_math": is_math,
                "use_research": is_explore,
                "use_search": search,
                "extend": {"sidebar": True},
            }

            full_content = ""
            in_thinking = False
            # [S2] 是否看到了协议的完成标记（`event == "all_done"`）。原先这个
            # 标记只被用来 break，证据在函数边界上消失，于是「被截断」与
            # 「说完了」在上层完全无法区分。
            saw_complete = False

            try:
                req = client.build_request("POST", stream_url, headers=headers, json=payload)
                resp = await client.send(req, stream=True)

                # [STREAM-AUDIT] `client.send(..., stream=True)` 拿到的是未关闭的
                # 响应：这里每一个提前退出都必须 `aclose()`，否则连接槽被永久占住。
                # 健康探针每 5 分钟对每个节点跑一次，泄漏是复发的，共享连接池
                # （max 100 conns / 40 keep-alive）耗尽后整个网关一起变慢。
                try:
                    if resp.status_code == 401:
                        _token_cache.pop(self.refresh_token, None)
                        raise ProviderError(
                            "Kimi 票据已过期，请运行 .venv/bin/python login.py kimi 重新登录",
                            status=401,
                            err_type="auth_error",
                        )
                    if resp.status_code == 429:
                        raise ProviderError("Kimi 网页端限流或人机验证阻断", status=429, err_type="rate_limit_error")
                    if resp.status_code != 200:
                        detail = (await read_error_body(resp))[:300]
                        err_type, keep_status = classify_stream_status(resp.status_code)
                        raise ProviderError(
                            f"Kimi 上游请求失败: HTTP {resp.status_code}"
                            + (f" {detail}" if detail else ""),
                            status=resp.status_code if keep_status else 502,
                            err_type=err_type,
                        )

                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        if line.startswith("data:"):
                            data_str = line[5:].strip()
                        else:
                            data_str = line.strip()

                        if not data_str or data_str == "[DONE]":
                            continue

                        try:
                            obj = json.loads(data_str)
                        except Exception:
                            continue
                        # [STREAM-AUDIT] `data: []` / `null` / 数字 都是合法 JSON，
                        # 但对它们调 `.get()` 会抛 AttributeError，把一次上游帧异常
                        # 变成一次 502（并顺带跳过 finally 之外的所有记账）。
                        if not isinstance(obj, dict):
                            continue

                        event = obj.get("event")
                        text_chunk = obj.get("text", "")

                        # 错误处理
                        if event == "error":
                            err_msg = obj.get("message") or "内容违规或请求受限"
                            raise ProviderError(f"Kimi 接口报错: {err_msg}", status=502)

                        # 思考过程 (k1 事件)
                        if event == "k1":
                            full_content += text_chunk
                            yield text_chunk, {"reasoning": True}
                            continue

                        # 正式回答完成 (cmpl 事件)
                        if event == "cmpl":
                            full_content += text_chunk
                            yield text_chunk, {}
                            continue

                        if event == "all_done":
                            saw_complete = True
                            break
                finally:
                    await resp.aclose()

            except ProviderError:
                raise
            except httpx.HTTPError as e:
                raise ProviderError(f"Kimi 网络连接异常: {e}", status=502)

            if not full_content:
                raise ProviderError(
                    "Kimi 未返回有效内容，可能是令牌失效或风控限制，请运行 .venv/bin/python login.py kimi 重新登录",
                    status=502,
                )

            # [S2] Kimi 协议**有**完成标记，所以能给出确定的完整性判据
            # （deepseek/doubao/glm 协议里没有标记，必须省略此键 = 未知）。
            yield "", {"conversation_id": conv_id, COMPLETION_KEY: saw_complete}

    async def delete_conversation(self, conversation_id: str) -> bool:
        if not conversation_id:
            return True
        try:
            async with ensure_async_client(self.client, self.timeout) as client:
                access_token, user_id = await self._get_access_token(client)
                headers = self._fake_headers(access_token)
                if user_id:
                    headers["X-Traffic-Id"] = user_id
                resp = await client.delete(f"{BASE_URL}/api/chat/{conversation_id}", headers=headers, timeout=10.0)
                return resp.status_code in (200, 204, 404)
        except Exception:
            return False
