"""chat.deepseek.com 网页端客户端。

协议要点（依据 sums001/Deepseek-API 开源实现核对，MIT）:
  * 鉴权: Authorization: Bearer <userToken> + 浏览器 Cookie
  * 建会话: POST /api/v0/chat_session/create  ->  biz_data.chat_session.id
  * PoW:   POST /api/v0/chat/create_pow_challenge {"target_path": ...}
           用官方 wasm 求解后得到 x-ds-pow-response 头
  * 补全:  POST /api/v0/chat/completion  body 见下，返回 SSE 流
  * 删会话: POST /api/v0/chat_session/delete {"chat_session_id": "<chat_session_id>"}
  * SSE:   快照帧 {"v":{"response":{"fragments":[...]}}} 之后是
           {"p":"response/fragments/-1/content","o":"APPEND","v":"增量"} 追加帧
  * 多轮:   conversation_id = "<chat_session_id>:<message_id>"
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import AsyncIterator, Dict, List, Optional, Tuple

import httpx

from ..base import ProviderError, ensure_async_client, flatten_messages, last_user_content
from .pow import DeepSeekPow

BASE = "https://chat.deepseek.com"
SESSION_CREATE = "/api/v0/chat_session/create"
SESSION_DELETE = "/api/v0/chat_session/delete"
POW_CHALLENGE = "/api/v0/chat/create_pow_challenge"
COMPLETION_PATH = "/api/v0/chat/completion"

# OpenAI 模型名 -> 网页端 model_type
MODEL_TYPE_MAP = {
    "deepseek-chat": "default",
    "deepseek-reasoner": "expert",
}

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36"
)

# wasmtime Store 不可重入：全局锁串行化 PoW 求解
_pow_lock = asyncio.Lock()
_pow_solver: Optional[DeepSeekPow] = None


def _get_pow() -> DeepSeekPow:
    global _pow_solver
    if _pow_solver is None:
        _pow_solver = DeepSeekPow()
    return _pow_solver


class DeepSeekProvider:
    def __init__(
        self,
        token: str,
        cookie: Optional[str] = None,
        timeout: float = 300.0,
        client: Optional[httpx.AsyncClient] = None,
    ):
        if not token:
            raise ProviderError("缺少 DEEPSEEK_AUTH_TOKEN", status=401, err_type="auth_error")
        self.token = token
        self.cookie = cookie
        self.timeout = timeout
        self.client = client

    def _headers(self) -> dict:
        tz_offset = -time.timezone  # 本地时区相对 UTC 的秒数（东为正）
        headers = {
            "authorization": f"Bearer {self.token}",
            "accept": "*/*",
            "content-type": "application/json",
            "user-agent": UA,
            "origin": BASE,
            "referer": f"{BASE}/",
            "x-app-version": "2.0.0",
            "x-client-version": "2.0.0",
            "x-client-platform": "web",
            "x-client-locale": "zh_CN",
            "x-client-bundle-id": "com.deepseek.chat",
            "x-client-timezone-offset": str(tz_offset),
        }
        if self.cookie:
            headers["cookie"] = self.cookie
        return headers

    @staticmethod
    def _biz(data: dict) -> dict:
        if data.get("code") != 0:
            raise ProviderError(f"DeepSeek 接口错误: {data.get('msg') or data}", status=502)
        biz = data.get("data", {}).get("biz_data")
        if biz is None:
            raise ProviderError(f"DeepSeek 响应结构异常: {data}", status=502)
        return biz

    async def _create_chat_session(self, client: httpx.AsyncClient) -> str:
        r = await client.post(BASE + SESSION_CREATE, json={}, headers=self._headers())
        if r.status_code == 401:
            raise ProviderError("DeepSeek token 无效或已过期，请重新获取", status=401, err_type="auth_error")
        r.raise_for_status()
        biz = self._biz(r.json())
        return biz["chat_session"]["id"]

    async def _pow_header(self, client: httpx.AsyncClient) -> str:
        r = await client.post(
            BASE + POW_CHALLENGE,
            json={"target_path": COMPLETION_PATH},
            headers=self._headers(),
        )
        r.raise_for_status()
        challenge = self._biz(r.json())["challenge"]
        async with _pow_lock:
            return await asyncio.to_thread(_get_pow().make_header, challenge)

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
        """对网页端发起补全，产出 (增量文本, meta) 序列。

        meta 仅在流末尾携带 {"conversation_id": "<chat_session_id>:<message_id>"}。
        conversation_id 为空时按“新会话”处理：展平全部消息并新建会话；
        传入时续聊：只发送最后一条 user 消息。

        temperature / max_tokens 为兼容 OpenAI 协议而接收，DeepSeek 网页端
        不支持这两个参数，直接忽略。
        """
        async with ensure_async_client(self.client, self.timeout) as client:
            session_id: Optional[str] = None
            parent_id: Optional[int] = None
            model_type: Optional[str] = None

            if conversation_id:
                session_id, _, msg = conversation_id.partition(":")
                parent_id = int(msg) if msg.isdigit() else None
                prompt = last_user_content(messages)
                pow_header = await self._pow_header(client)
            else:
                model_type = MODEL_TYPE_MAP.get(model, "default")
                prompt = flatten_messages(messages)
                session_id, pow_header = await asyncio.gather(
                    self._create_chat_session(client),
                    self._pow_header(client),
                )

            body: Dict = {
                "chat_session_id": session_id,
                "parent_message_id": parent_id,
                "prompt": prompt,
                "ref_file_ids": [],
                "thinking_enabled": thinking,
                "search_enabled": search,
                "action": None,
                "preempt": False,
            }
            if model_type is not None:
                body["model_type"] = model_type

            try:
                async with client.stream(
                    "POST",
                    BASE + COMPLETION_PATH,
                    json=body,
                    headers={**self._headers(), "x-ds-pow-response": pow_header},
                ) as resp:
                    if resp.status_code == 401:
                        raise ProviderError(
                            "DeepSeek token 无效或已过期，请重新获取",
                            status=401, err_type="auth_error",
                        )
                    if resp.status_code == 429:
                        raise ProviderError("DeepSeek 网页端限流（429），请稍后再试", status=429)
                    resp.raise_for_status()

                    state: Dict = {}
                    async for item in self._parse_sse(resp, state):
                        yield item
            except httpx.HTTPStatusError as e:
                raise ProviderError(
                    f"DeepSeek 上游请求失败: {e.response.status_code} {e.response.text[:300]}",
                    status=502,
                )
            except httpx.HTTPError as e:
                raise ProviderError(f"DeepSeek 网络错误: {e}", status=502)

            message_id = state.get("message_id")
            cid = f"{session_id}:{message_id}" if message_id is not None else f"{session_id}:"
            yield "", {"conversation_id": cid}

    async def _parse_sse(
        self, resp: httpx.Response, state: Dict
    ) -> AsyncIterator[Tuple[str, dict]]:
        """解析 DeepSeek 的 SSE 流：快照帧 + 追加帧。

        支持解析 THINK 思考片段（附带 reasoning: True 元数据）与 RESPONSE 回复片段。
        同时在 state["message_id"] 记录 assistant message_id 用于构造续聊 conversation_id。
        """
        active_path: Optional[str] = None
        fragment_types: Dict[int, str] = {}
        current_is_reasoning = False

        async for line in resp.aiter_lines():
            if not line or not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue

            v = obj.get("v")

            # 快照帧: {"v": {"response": {...}}}
            if isinstance(v, dict) and "response" in v:
                response = v["response"]
                if isinstance(response, dict):
                    mid = response.get("message_id", response.get("id"))
                    if isinstance(mid, int):
                        state["message_id"] = mid
                    for idx, frag in enumerate(response.get("fragments", [])):
                        ft = (frag.get("type") or "").upper()
                        fragment_types[idx] = ft
                        content = frag.get("content")
                        if content:
                            if ft in ("THINK", "THOUGHT"):
                                yield content, {"reasoning": True}
                            elif ft == "RESPONSE":
                                yield content, {}
                continue

            # 追加帧或包含路径的帧
            if "p" in obj:
                active_path = obj["p"]
                if active_path and active_path.endswith("message_id") and isinstance(v, int):
                    state["message_id"] = v

                # 新 fragment 追加: {"p": "response/fragments", "o": "APPEND", "v": {"type": "RESPONSE", ...}}
                if active_path == "response/fragments" and obj.get("o") == "APPEND" and isinstance(v, dict):
                    new_idx = len(fragment_types)
                    ft = (v.get("type") or "").upper()
                    fragment_types[new_idx] = ft
                    if v.get("content"):
                        if ft in ("THINK", "THOUGHT"):
                            yield v["content"], {"reasoning": True}
                        elif ft == "RESPONSE":
                            yield v["content"], {}
                    continue

                # 识别当前 content 路径归属的 fragment
                if active_path and active_path.endswith("content"):
                    m = re.search(r"fragments/(-?\d+)", active_path)
                    frag_idx = int(m.group(1)) if m else None
                    if frag_idx == -1 or frag_idx is None:
                        idx = max(fragment_types.keys()) if fragment_types else 0
                    else:
                        idx = frag_idx

                    ft = fragment_types.get(idx)
                    if ft in ("THINK", "THOUGHT"):
                        current_is_reasoning = True
                    elif ft == "RESPONSE":
                        current_is_reasoning = False
                    else:
                        # 兼容：如果尚未登记类型，且路径为 0 或含 think，则判定为思考
                        if idx == 0 and ("fragments/0/content" in active_path or "think" in active_path.lower()):
                            current_is_reasoning = True
                            fragment_types[idx] = "THINK"
                        else:
                            current_is_reasoning = False
                            fragment_types[idx] = "RESPONSE"

                    if obj.get("o") == "APPEND" and isinstance(v, str):
                        meta = {"reasoning": True} if current_is_reasoning else {}
                        yield v, meta
                    continue

            # 无路径的裸追加
            if isinstance(v, str) and active_path and active_path.endswith("content"):
                meta = {"reasoning": True} if current_is_reasoning else {}
                yield v, meta

    async def delete_conversation(self, conversation_id: str) -> bool:
        """删除 DeepSeek 远端会话。

        conversation_id 可以是 "<chat_session_id>:<message_id>" 或直接是 "<chat_session_id>"。
        删除成功或会话已不存在返回 True，遇到网络/鉴权异常返回 False。
        """
        if not conversation_id:
            return True

        session_id, _, _ = conversation_id.partition(":")
        session_id = session_id.strip()
        if not session_id:
            return True

        try:
            async with ensure_async_client(self.client, self.timeout) as client:
                r = await client.post(
                    BASE + SESSION_DELETE,
                    json={"chat_session_id": session_id},
                    headers=self._headers(),
                )
                if r.status_code == 200:
                    data = r.json()
                    return data.get("code") == 0
                return False
        except Exception:
            return False

