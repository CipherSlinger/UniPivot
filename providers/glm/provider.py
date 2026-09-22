"""智谱清言（ChatGLM，chatglm.cn）网页端客户端。

协议要点:
  * 鉴权: 抓取 refresh_token（或完整 Cookie），通过 POST /chatglm/user-api/user/refresh
          获取 access_token。签名计算：timestamp, nonce, X-Sign (MD5)。
  * 会话与补全流: POST https://chatglm.cn/chatglm/backend-api/assistant/stream
          支持 assistant_id / chat_model 选择（glm-4, glm-4-plus, glm-zero-preview 等）。
          SSE 流以 parts/logic_id 组织增量内容，同时支持深度思考 reasoning 增量与文本增量。
  * 删会话: POST https://chatglm.cn/chatglm/backend-api/assistant/conversation/delete
          body: {"conversation_id": "<conv_id>"}
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import time
import uuid
from typing import AsyncIterator, Dict, List, Optional, Tuple

import httpx

from ..base import (
    ProviderError,
    ensure_async_client,
    flatten_messages,
    last_user_content,
    parse_cookie_header,
)
from ..http_client import get_http_client

BASE_URL = "https://chatglm.cn/chatglm"
REFRESH_URL = f"{BASE_URL}/user-api/user/refresh"
STREAM_URL = f"{BASE_URL}/backend-api/assistant/stream"
DELETE_URL = f"{BASE_URL}/backend-api/assistant/conversation/delete"

DEFAULT_ASSISTANT_ID = "65940acff94777010aa6b796"
SIGN_SECRET = "8a1317a7468aa3ad86e997d08f3f31cb"

# 模型映射
MODEL_MAP = {
    "glm": ("glm-4", DEFAULT_ASSISTANT_ID),
    "chatglm": ("glm-4", DEFAULT_ASSISTANT_ID),
    "glm-4": ("glm-4", DEFAULT_ASSISTANT_ID),
    "glm-4-plus": ("glm-4-plus", DEFAULT_ASSISTANT_ID),
    "glm-4-air": ("glm-4-air", DEFAULT_ASSISTANT_ID),
    "glm-4-flash": ("glm-4-flash", DEFAULT_ASSISTANT_ID),
    "glm-4-long": ("glm-4-long", DEFAULT_ASSISTANT_ID),
    "glm-zero-preview": ("glm-zero-preview", DEFAULT_ASSISTANT_ID),
    "glm-think": ("glm-zero-preview", DEFAULT_ASSISTANT_ID),
}

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
)

# access_token 缓存池: refresh_token -> {"access_token": str, "expires_at": float}
_token_cache: Dict[str, dict] = {}
_token_lock = asyncio.Lock()


def build_glm_sign() -> Tuple[str, str, str]:
    """生成智谱 API 签名元组: (timestamp, nonce, sign)。"""
    now = str(int(time.time() * 1000))
    digits = [int(char) for char in now]
    checksum = (sum(digits) - digits[-2]) % 10
    timestamp = now[:-2] + str(checksum) + now[-1]
    nonce = uuid.uuid4().hex
    sign = hashlib.md5(f"{timestamp}-{nonce}-{SIGN_SECRET}".encode("utf-8")).hexdigest()
    return timestamp, nonce, sign


class GLMProvider:
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
            raise ProviderError("缺少智谱 GLM 令牌（refresh_token）", status=401, err_type="auth_error")
        self.refresh_token = token.strip()
        self.cookie = cookie or ""
        self.timeout = timeout
        self.client = client
        if not user_agent or "HeadlessChrome" in user_agent:
            self.user_agent = UA
        else:
            self.user_agent = user_agent

        self.device_id = device_id or uuid.uuid4().hex
        self._cookies = parse_cookie_header(self.cookie)

    def _headers(self, access_token: Optional[str] = None) -> dict[str, str]:
        timestamp, nonce, sign = build_glm_sign()
        req_id = f"{self.device_id[:8]}-{int(time.time() * 1000)}-1"

        headers = {
            "Accept": "text/event-stream",
            "Accept-Language": "zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7",
            "App-Name": "chatglm",
            "Cache-Control": "no-cache",
            "Content-Type": "application/json",
            "Origin": "https://chatglm.cn",
            "Pragma": "no-cache",
            "Sec-Ch-Ua": '"Chromium";v="130", "Google Chrome";v="130", "Not?A_Brand";v="99"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"macOS"',
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "User-Agent": self.user_agent,
            "X-App-Fr": "browser_extension",
            "X-App-Platform": "pc",
            "X-App-Version": "0.0.1",
            "X-Device-Id": self.device_id,
            "X-Lang": "zh",
            "X-Nonce": nonce,
            "X-Request-Id": req_id,
            "X-Sign": sign,
            "X-Timestamp": timestamp,
        }
        if self.cookie:
            headers["Cookie"] = self.cookie
        if access_token:
            headers["Authorization"] = f"Bearer {access_token}"
        return headers

    async def _get_access_token(self, client: httpx.AsyncClient) -> str:
        """从 refresh_token 换取 access_token。"""
        now = time.time()
        cached = _token_cache.get(self.refresh_token)
        if cached and cached.get("expires_at", 0) > now + 30:
            return cached["access_token"]

        async with _token_lock:
            cached = _token_cache.get(self.refresh_token)
            if cached and cached.get("expires_at", 0) > now + 30:
                return cached["access_token"]

            headers = self._headers()
            headers["Authorization"] = f"Bearer {self.refresh_token}"

            access_token = self.refresh_token
            try:
                resp = await client.post(REFRESH_URL, headers=headers, json={}, timeout=15.0)
                if resp.status_code == 200:
                    payload = resp.json()
                    res = payload.get("result", {})
                    token = res.get("access_token") or res.get("accessToken")
                    if token:
                        access_token = token
                elif resp.status_code == 401:
                    raise ProviderError(
                        "智谱 GLM 令牌无效或已过期，请运行 .venv/bin/python login.py glm 重新登录",
                        status=401,
                        err_type="auth_error",
                    )
            except ProviderError:
                raise
            except Exception:
                access_token = self.refresh_token

            _token_cache[self.refresh_token] = {
                "access_token": access_token,
                "expires_at": now + 1800,  # 30分钟
            }
            return access_token

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
        wire_model, assistant_id = MODEL_MAP.get(model.lower(), ("glm-4", DEFAULT_ASSISTANT_ID))
        if thinking or "think" in model.lower() or "zero" in model.lower():
            chat_mode = "zero"
        else:
            chat_mode = "chat"

        async with ensure_async_client(self.client, self.timeout) as client:
            access_token = await self._get_access_token(client)

            prompt = last_user_content(messages) if conversation_id else flatten_messages(messages)

            converted_messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}],
                }
            ]

            payload = {
                "assistant_id": assistant_id,
                "conversation_id": conversation_id or "",
                "project_id": "",
                "chat_type": "user_chat",
                "messages": converted_messages,
                "meta_data": {
                    "channel": "",
                    "chat_mode": chat_mode,
                    "draft_id": "",
                    "if_plus_model": "plus" in wire_model or "zero" in wire_model,
                    "input_question_type": "xxxx",
                    "is_networking": search,
                    "is_test": False,
                    "platform": "pc",
                    "quote_log_id": "",
                    "cogview": {"rm_label_watermark": False},
                    "chat_model": wire_model,
                },
            }

            headers = self._headers(access_token)

            conv_id = conversation_id or ""
            full_content = ""
            in_thinking = False

            # 跟踪每个 logic_id 已经派发的文本长度
            part_text_sent: Dict[str, int] = {}
            part_reasoning_sent: Dict[str, int] = {}

            try:
                req = client.build_request("POST", STREAM_URL, headers=headers, json=payload)
                resp = await client.send(req, stream=True)

                if resp.status_code == 401:
                    _token_cache.pop(self.refresh_token, None)
                    raise ProviderError(
                        "智谱 GLM 票据无效或已过期，请运行 .venv/bin/python login.py glm 重新登录",
                        status=401,
                        err_type="auth_error",
                    )
                if resp.status_code == 429:
                    raise ProviderError("智谱 GLM 限流或风控阻断", status=429, err_type="rate_limit_error")
                if resp.status_code != 200:
                    await resp.aread()
                    raise ProviderError(f"智谱 GLM 上游请求失败: HTTP {resp.status_code} {resp.text[:300]}", status=502)

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

                    if not conv_id and obj.get("conversation_id"):
                        conv_id = str(obj["conversation_id"])

                    # parts 增量解析
                    parts = obj.get("parts", [])
                    for part in parts:
                        if not isinstance(part, dict):
                            continue
                        logic_id = str(part.get("logic_id", "default"))
                        content_list = part.get("content", [])

                        cur_text = ""
                        cur_reasoning = ""
                        for c in content_list:
                            if not isinstance(c, dict):
                                continue
                            c_type = c.get("type")
                            if c_type == "text":
                                cur_text += str(c.get("text", ""))
                            elif c_type == "reasoning":
                                cur_reasoning += str(c.get("text", ""))

                        # 推送思考增量
                        if cur_reasoning:
                            sent = part_reasoning_sent.get(logic_id, 0)
                            if len(cur_reasoning) > sent:
                                delta = cur_reasoning[sent:]
                                part_reasoning_sent[logic_id] = len(cur_reasoning)
                                yield delta, {"reasoning": True}

                        # 推送正文增量
                        if cur_text:
                            sent = part_text_sent.get(logic_id, 0)
                            if len(cur_text) > sent:
                                delta = cur_text[sent:]
                                part_text_sent[logic_id] = len(cur_text)
                                yield delta, {}
                                full_content += delta

            except ProviderError:
                raise
            except httpx.HTTPError as e:
                raise ProviderError(f"智谱 GLM 网络连接异常: {e}", status=502)

            if not full_content and not part_text_sent:
                raise ProviderError(
                    "智谱 GLM 未返回有效内容，请运行 .venv/bin/python login.py glm 重新登录",
                    status=502,
                )

            yield "", {"conversation_id": conv_id}

    async def delete_conversation(self, conversation_id: str) -> bool:
        if not conversation_id:
            return True
        try:
            async with ensure_async_client(self.client, self.timeout) as client:
                access_token = await self._get_access_token(client)
                headers = self._headers(access_token)
                headers["Content-Type"] = "application/json"
                resp = await client.post(
                    DELETE_URL,
                    headers=headers,
                    json={"conversation_id": conversation_id},
                    timeout=10.0,
                )
                return resp.status_code in (200, 204, 404)
        except Exception:
            return False
