"""通义千问（国内版 www.qianwen.com）网页端客户端。

协议要点:
  * 鉴权: Cookie 中的 `tongyi_sso_ticket`（<100 字符）或
          `login_aliyunid_ticket`（>=100 字符）。
          扫码登录脚本会自动捕获该票据并存入 session/qwen.json。
  * 补全: POST https://chat2.qianwen.com/api/v2/chat
          单端点 SSE 流式；会话记忆通过 body 的 session_id 续传。
  * 消息: 展平后作为 messages[0].content 发送。
  * SSE:  data 帧为 JSON，`data.messages[].content` 是累计全文（按长度取增量）；
          `data.status=="complete"` 表示结束；
          conversation_id 为 session_id。
"""

from __future__ import annotations

import json
import time
import uuid
from typing import AsyncIterator, List, Optional, Tuple

import httpx

from ..base import (
    ProviderError,
    ensure_async_client,
    flatten_messages,
    last_user_content,
    parse_cookie_header,
)

BASE = "https://chat2.qianwen.com"
CHAT_ENDPOINT = "/api/v2/chat"

# 国内版网页端当前模型
SUPPORTED_MODELS = {
    "Qwen",            # Qwen3.7-千问（默认）
    "Qwen3.7-Max",     # Qwen3.7-Max
    "Qwen3.6-Flash",   # Qwen3.6-Flash
}
DEFAULT_MODEL = "Qwen"

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
)


def _is_valid_conv_part(s: str) -> bool:
    return len(s) == 32 and all(c in "0123456789abcdef" for c in s.lower())


def _split_thought_and_content(raw: str) -> Tuple[str, str]:
    """将包含 <thought> 或 <think> 的累计文本分离为 (思考全文, 回复全文)。"""
    if not raw:
        return "", ""
    tag_name = None
    tag_start = -1
    for tag in ("<thought>", "<think>"):
        pos = raw.find(tag)
        if pos != -1 and (tag_start == -1 or pos < tag_start):
            tag_start = pos
            tag_name = tag[1:-1]

    if tag_start == -1 or tag_name is None:
        return "", raw

    open_tag = f"<{tag_name}>"
    close_tag = f"</{tag_name}>"
    content_after_open = raw[tag_start + len(open_tag):]
    close_idx = content_after_open.find(close_tag)

    if close_idx == -1:
        th = content_after_open
        for l in range(len(close_tag) - 1, 0, -1):
            if th.endswith(close_tag[:l]):
                th = th[:-l]
                break
        return th, raw[:tag_start]
    else:
        th = content_after_open[:close_idx]
        ans = raw[:tag_start] + content_after_open[close_idx + len(close_tag):]
        return th, ans


class QwenProvider:
    def __init__(
        self,
        token: str,
        cookie: Optional[str] = None,
        timeout: float = 600.0,
        user_agent: Optional[str] = None,
        user_id: Optional[str] = None,
        client: Optional[httpx.AsyncClient] = None,
    ):
        """token 即 tongyi_sso_ticket（或 login_aliyunid_ticket）的取值。"""
        if not token:
            raise ProviderError("缺少 Qwen 令牌（tongyi_sso_ticket）", status=401, err_type="auth_error")
        self.ticket = token.strip()
        self.cookie = cookie or ""
        self.timeout = timeout
        self.user_id = user_id or ""
        self.client = client
        if not user_agent or "HeadlessChrome" in user_agent:
            self.user_agent = UA
        else:
            self.user_agent = user_agent
        self._init_cookies()

    def _init_cookies(self) -> None:
        if self.cookie:
            self._cookie_str = self.cookie
        else:
            name = "login_aliyunid_ticket" if len(self.ticket) > 100 else "tongyi_sso_ticket"
            self._cookie_str = (
                f"{name}={self.ticket}; "
                "aliyun_choice=intl; _samesite_flag_=true; "
                f"_device_id={uuid.uuid4().hex}; "
                f"t={uuid.uuid4().hex}"
            )
        self._cookies = parse_cookie_header(self._cookie_str)

    def _headers(self) -> dict:
        device_id = self._cookies.get("_device_id", "") or uuid.uuid4().hex
        b_user_id = self._cookies.get("b-user-id", "")
        uid = self.user_id or b_user_id
        xsrf = self._cookies.get("XSRF-TOKEN", "")
        return {
            "User-Agent": self.user_agent,
            "x-platform": "pc_tongyi",
            "prod_id": "tongyi",
            "x-user-id": uid,
            "x-device-id": device_id,
            "x-xsrf-token": xsrf,
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream, */*",
            "Referer": "https://www.qianwen.com/",
            "Origin": "https://www.qianwen.com",
            "Cookie": self._cookie_str,
        }

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
        """对通义网页端发起补全，产出 (增量文本, meta) 序列。

        meta 仅在流末尾携带 {"conversation_id": <session_id>}。
        conversation_id 为空时按“新会话”处理；传入时续聊。
        """
        if model not in SUPPORTED_MODELS:
            model = DEFAULT_MODEL

        if conversation_id and not _is_valid_conv_part(conversation_id):
            conversation_id = None

        session_id = conversation_id or uuid.uuid4().hex
        prompt = last_user_content(messages) if conversation_id else flatten_messages(messages)
        conv_id = session_id
        acc_text = ""
        acc_thought = ""
        max_attempts = 2

        for attempt in range(max_attempts):
            acc_text = ""
            acc_thought = ""
            waf_detected = False
            cookies = self._cookies
            b_user_id = cookies.get("b-user-id", "") or cookies.get("_device_id", "") or uuid.uuid4().hex
            nonce = uuid.uuid4().hex[:11]
            ts = int(time.time() * 1000)

            url = (
                f"{BASE}{CHAT_ENDPOINT}?"
                f"biz_id=ai_qwen&chat_client=h5&device=pc&fr=pc&pr=qwen&"
                f"ut={b_user_id}&la=zh-CN&tz=Asia%2FShanghai&wv=4.7.17&ve=4.7.17&nonce={nonce}&timestamp={ts}"
            )

            payload = {
                "req_id": uuid.uuid4().hex,
                "parent_req_id": "0",
                "messages": [
                    {
                        "mime_type": "text/plain",
                        "content": prompt,
                        "meta_data": {"ori_query": prompt},
                        "status": "complete",
                    }
                ],
                "scene": "chat",
                "sub_scene": "",
                "scene_param": "continue_turn" if conversation_id else "first_turn",
                "session_id": session_id,
                "biz_id": "ai_qwen",
                "topic_id": "",
                "model": model,
                "from": "default",
                "protocol_version": "v2",
                "messages_merge": False,
                "chat_client": "h5",
                "deep_search": "normal" if search else None,
                "temporary": False if conversation_id else True,
                "params_extra": {"supports_cowork": "false"},
                "chat_mode": "think" if thinking else "quick",
            }

            try:
                async with ensure_async_client(self.client, self.timeout) as client:
                    async with client.stream(
                        "POST", url, headers=self._headers(), json=payload
                    ) as resp:
                        if resp.status_code == 401:
                            raise ProviderError(
                                "通义千问票据无效或已过期，请运行 .venv/bin/python login.py qwen 重新扫码登录",
                                status=401, err_type="auth_error",
                            )
                        if resp.status_code == 429:
                            raise ProviderError("通义千问网页端限流（429），请稍后再试", status=429)
                        resp.raise_for_status()

                        async for line in resp.aiter_lines():
                            if not line:
                                continue
                            if line.startswith("data:"):
                                data = line[5:].strip()
                            else:
                                data = line.strip()

                            if not data or data == "[DONE]":
                                continue
                            try:
                                obj = json.loads(data)
                            except json.JSONDecodeError:
                                continue

                            if not isinstance(obj, dict):
                                continue

                            # 检查 WAF 人机验证 / 限流
                            ret = obj.get("ret")
                            if isinstance(ret, list) and any(
                                "RGV587" in str(x) or "FAIL_SYS_USER_VALIDATE" in str(x) for x in ret
                            ):
                                waf_detected = True
                                break

                            err_code = obj.get("error_code")
                            if err_code and err_code != 0:
                                if "LOGIN" in str(err_code).upper():
                                    raise ProviderError(
                                        "通义千问票据无效或已过期，请运行 .venv/bin/python login.py qwen 重新扫码登录",
                                        status=401, err_type="auth_error",
                                    )
                                raise ProviderError(
                                    f"通义千问返回错误: {err_code} {obj.get('error_msg') or ''}".strip(),
                                    status=502,
                                )

                            comm = obj.get("communication", {})
                            if comm.get("sessionid"):
                                conv_id = comm["sessionid"]

                            msgs = obj.get("data", {}).get("messages", [])
                            raw_thought = ""
                            raw_content = ""
                            for m in msgs:
                                m_type = str(m.get("type") or m.get("role") or "").lower()
                                th_val = m.get("thought_process") or m.get("thought") or m.get("reasoning_content")
                                if th_val and isinstance(th_val, str):
                                    raw_thought = th_val
                                elif m_type == "thought" and m.get("content") and isinstance(m.get("content"), str):
                                    raw_thought = m["content"]
                                elif m.get("content") and isinstance(m.get("content"), str):
                                    raw_content = m["content"]

                            if not raw_thought and raw_content:
                                th_part, ans_part = _split_thought_and_content(raw_content)
                            else:
                                th_part = raw_thought
                                ans_part = raw_content

                            if th_part and len(th_part) > len(acc_thought):
                                th_chunk = th_part[len(acc_thought):]
                                acc_thought = th_part
                                yield th_chunk, {"reasoning": True}

                            if ans_part and len(ans_part) > len(acc_text):
                                text_chunk = ans_part[len(acc_text):]
                                acc_text = ans_part
                                yield text_chunk, {}

                            if obj.get("data", {}).get("status") == "complete":
                                break

                if waf_detected:
                    if attempt == 0:
                        # 尝试静默刷新一次安全 Cookie 并重试
                        try:
                            import asyncio
                            from login import refresh_qwen
                            refreshed = await asyncio.to_thread(refresh_qwen, timeout=25)
                            if refreshed and refreshed.cookie:
                                self.cookie = refreshed.cookie
                                self.ticket = refreshed.token
                                if refreshed.user_id:
                                    self.user_id = refreshed.user_id
                                if refreshed.user_agent and "HeadlessChrome" not in refreshed.user_agent:
                                    self.user_agent = refreshed.user_agent
                                self._init_cookies()
                                continue
                        except Exception:
                            pass

                    raise ProviderError(
                        "通义千问触发人机验证/限流（RGV587），请运行 .venv/bin/python login.py qwen 打开浏览器完成人机校验",
                        status=429,
                        err_type="rate_limit_error",
                    )

                if not acc_text and not acc_thought:
                    raise ProviderError(
                        "通义千问未返回有效回复内容，可能是票据失效或触发了风控，请运行 .venv/bin/python login.py qwen 重新登录",
                        status=502,
                    )
                break
            except ProviderError:
                raise
            except httpx.HTTPStatusError as e:
                await e.response.aread()
                raise ProviderError(
                    f"通义千问上游请求失败: {e.response.status_code} {e.response.text[:300]}", status=502
                )
            except httpx.HTTPError as e:
                raise ProviderError(f"通义���问网络错误: {e}", status=502)

        yield "", {"conversation_id": conv_id}

    async def delete_conversation(self, conversation_id: str) -> bool:
        """删除通义千问远端会话。

        conversation_id 为 session_id。
        成功返回 True，失败或无会话返回 False（不抛异常崩溃）。
        """
        if not conversation_id:
            return True
        cid = str(conversation_id).strip()
        if not cid:
            return True

        cookies = self._cookies
        b_user_id = cookies.get("b-user-id", "") or cookies.get("_device_id", "") or uuid.uuid4().hex
        nonce = uuid.uuid4().hex[:11]
        ts = int(time.time() * 1000)
        url = (
            f"{BASE}/api/v2/session/delete?"
            f"biz_id=ai_qwen&chat_client=h5&device=pc&fr=pc&pr=qwen&"
            f"ut={b_user_id}&la=zh-CN&tz=Asia%2FShanghai&wv=4.7.17&ve=4.7.17&nonce={nonce}&timestamp={ts}"
        )
        payload = {
            "session_id": cid,
            "biz_id": "ai_qwen",
        }
        try:
            async with ensure_async_client(self.client, self.timeout) as client:
                r = await client.post(url, headers=self._headers(), json=payload)
                if r.status_code == 200:
                    try:
                        data = r.json()
                        return data.get("success", False) or data.get("code") == 0 or data.get("ret") == ["SUCCESS"]
                    except Exception:
                        return False
                return False
        except Exception:
            return False

