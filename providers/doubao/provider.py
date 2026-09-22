"""豆包（Doubao，www.doubao.com）网页端 / 桌面端客户端。

协议要点:
  * 鉴权: Cookie 中的 `sessionid`（或 `sessionid_ss`）及 `passport_csrf_token`。
          扫码/账号登录脚本（login.py doubao）会自动捕获 Cookie 并存入 session/doubao.json。
  * 补全: POST https://www.doubao.com/chat/completion
          通过模拟桌面客户端（client_platform: pc_client）直连 SSE 流式端点；
          支持模型选择（Pro / Lite / Think / Expert）及会话记忆（conversation_id 续聊）。
  * SSE:  SSE_ACK 帧回传 conversation_id；
          STREAM_CHUNK / chunk_delta 帧增量吐出文本片段；
          710012001 报登录失效，710022002/710022004 报风控/人机验证。
  * 删会话: POST https://www.doubao.com/im/conversation/batch_del_user_conv
          body: {"conversation_id": ["<id>"], "delete_all": false, "conversation_type": 1}
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple
from urllib.parse import urlencode

import httpx

from ..base import (
    ProviderError,
    ensure_async_client,
    flatten_messages,
    last_user_content,
    parse_cookie_header,
)

BASE_URL = "https://www.doubao.com"
COMPLETION_ENDPOINT = "/chat/completion"
BATCH_DELETE_ENDPOINT = "/im/conversation/batch_del_user_conv"

DEFAULT_BOT_ID = "7338286299411103781"  # 豆包 Pro
LITE_BOT_ID = "7234781073513644036"     # 豆包 Lite

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
)

RISK_ERROR_CODES = {710022002, 710022004}
AUTH_ERROR_CODES = {710012000, 710012001}


class DoubaoProvider:
    def __init__(
        self,
        token: str,
        cookie: Optional[str] = None,
        timeout: float = 600.0,
        user_agent: Optional[str] = None,
        device_id: Optional[str] = None,
        client: Optional[httpx.AsyncClient] = None,
    ):
        """token 为 sessionid 或 sessionid_ss 取值；cookie 为完整 Cookie 串。"""
        if not token:
            raise ProviderError("缺少豆包令牌（sessionid）", status=401, err_type="auth_error")
        self.token = token.strip()
        self.cookie = cookie or ""
        self.timeout = timeout
        self.client = client
        if not user_agent or "HeadlessChrome" in user_agent:
            self.user_agent = UA
        else:
            self.user_agent = user_agent

        # 确定 Cookie 与解析缓存
        self._cookie_str = self.cookie or (
            f"sessionid={self.token}; "
            f"sessionid_ss={self.token}; "
            f"ttwid=1; "
            f"passport_csrf_token={uuid.uuid4().hex}"
        )
        self._cookies: Dict[str, str] = parse_cookie_header(self._cookie_str)
        self.device_id = (
            device_id
            or self._cookies.get("device_id")
            or str(int(time.time() * 1000))
        )
        self._csrf_token = (
            self._cookies.get("passport_csrf_token")
            or self._cookies.get("passport_csrf_token_default")
            or ""
        )
        self._fp = self._cookies.get("s_v_web_id") or f"verify_{uuid.uuid4().hex[:32]}"

    def _security_params(self) -> Dict[str, str]:
        web_id = self._cookies.get("web_id") or self._cookies.get("tea_uuid") or self.device_id
        ms_token = self._cookies.get("msToken") or ""

        params = {
            "aid": "582478",
            "real_aid": "582478",
            "device_id": self.device_id,
            "tea_uuid": self.device_id,
            "web_id": web_id,
            "device_platform": "web",
            "language": "zh",
            "region": "CN",
            "sys_region": "CN",
            "pkg_type": "release_version",
            "version_code": "20800",
            "pc_version": "2.1.7",
            "chromium_version": "148.0.7816.0",
            "client_platform": "pc_client",
            "runtime": "web",
            "runtime_version": "3.5.4",
            "samantha_web": "1",
            "use-olympus-account": "1",
            "fp": self._fp,
            "web_tab_id": str(uuid.uuid4()),
        }
        if ms_token:
            params["msToken"] = ms_token
        return params

    def _headers(self) -> Dict[str, str]:
        return {
            "User-Agent": self.user_agent,
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
            "Origin": BASE_URL,
            "Referer": f"{BASE_URL}/chat",
            "x-tt-passport-csrf-token": self._csrf_token,
            "Cookie": self._cookie_str,
        }

    def _handle_error_code(self, code: int, default_msg: str = "") -> None:
        if code in AUTH_ERROR_CODES:
            raise ProviderError(
                "豆包登录已过期，请运行 .venv/bin/python login.py doubao 重新登录",
                status=401,
                err_type="auth_error",
            )
        if code in RISK_ERROR_CODES:
            raise ProviderError(
                "豆包触发风控/人机验证，请运行 .venv/bin/python login.py doubao 打开浏览器完成人机校验",
                status=429,
                err_type="rate_limit_error",
            )
        if code != 0:
            raise ProviderError(f"豆包返回错误: {code} {default_msg}".strip(), status=502)

    def _resolve_bot_and_think(self, model: str, thinking: bool) -> Tuple[str, int]:
        m_lower = model.lower()
        if "lite" in m_lower:
            bot_id = LITE_BOT_ID
        else:
            bot_id = DEFAULT_BOT_ID

        need_deep_think = 0
        if thinking or "think" in m_lower:
            need_deep_think = 1
        elif "expert" in m_lower:
            need_deep_think = 3

        return bot_id, need_deep_think

    def _build_payload(
        self,
        prompt: str,
        bot_id: str,
        need_deep_think: int,
        conversation_id: Optional[str],
    ) -> Dict[str, Any]:
        cid = conversation_id or ""

        return {
            "client_meta": {
                "local_conversation_id": f"local_{uuid.uuid4().hex[:16]}",
                "conversation_id": cid,
                "bot_id": bot_id,
                "last_section_id": "",
                "last_message_index": 0,
            },
            "messages": [
                {
                    "local_message_id": str(uuid.uuid4()),
                    "content_block": [
                        {
                            "block_type": 10000,
                            "content": {
                                "text_block": {
                                    "text": prompt,
                                    "icon_url": "",
                                    "icon_url_dark": "",
                                    "summary": "",
                                },
                                "pc_event_block": "",
                            },
                            "block_id": str(uuid.uuid4()),
                            "parent_id": "",
                            "meta_info": [],
                            "append_fields": [],
                            "is_finish": True,
                            "patch_type": 2,
                        }
                    ],
                    "message_status": 0,
                }
            ],
            "option": {
                "send_message_scene": "",
                "create_time_ms": 0,
                "collect_id": "",
                "is_audio": False,
                "answer_with_suggest": True,
                "tts_switch": False,
                "need_deep_think": need_deep_think,
                "click_clear_context": False,
                "from_suggest": False,
                "is_regen": False,
                "is_replace": False,
                "disable_sse_cache": False,
                "select_text_action": "",
                "resend_for_regen": False,
                "scene_type": 0,
                "unique_key": str(uuid.uuid4()),
                "start_seq": 0,
                "need_create_conversation": not bool(cid),
                "conversation_init_option": {"need_ack_conversation": True},
                "regen_query_id": [],
                "edit_query_id": [],
                "regen_instruction": "",
                "no_replace_for_regen": False,
                "message_from": 0,
                "shared_app_name": "",
                "action_bar_skill_id": 0,
                "sse_recv_event_options": {"support_chunk_delta": True},
                "is_ai_playground": False,
            },
            "chat_ability": {},
            "ext": {
                "use_deep_think": str(need_deep_think),
                "fp": self._fp,
                "use_submit_pipeline": "1",
                "commerce_credit_config_enable": "0",
                "sub_conv_firstmet_type": "1",
            },
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
        """对豆包网页端发起补全，产出 (增量文本, meta) 序列。

        meta 在流末尾携带 {"conversation_id": <cid>}。
        conversation_id 为空时按“新会话”处理；传入时续聊。
        """
        bot_id, need_deep_think = self._resolve_bot_and_think(model, thinking)
        prompt = last_user_content(messages) if conversation_id else flatten_messages(messages)
        payload = self._build_payload(prompt, bot_id, need_deep_think, conversation_id)
        params = self._security_params()
        url = f"{BASE_URL}{COMPLETION_ENDPOINT}?{urlencode(params)}"

        conv_id = conversation_id or ""
        accumulated_text = ""
        accumulated_blocks: Dict[str, str] = {}
        current_event_name = "message"

        def _extract_block(cb: dict) -> Tuple[str, bool]:
            nonlocal accumulated_text
            if not isinstance(cb, dict):
                return "", False

            b_type = cb.get("block_type")
            content = cb.get("content", {})
            if not isinstance(content, dict):
                content = {}

            is_reasoning = False
            text = ""

            if "thought_block" in content or "think_block" in content:
                is_reasoning = True
                tb = content.get("thought_block") or content.get("think_block") or {}
                text = tb.get("text", "") if isinstance(tb, dict) else str(tb)
            elif b_type is not None and b_type != 10000:
                is_reasoning = True
                tb = content.get("text_block") or content.get("thought_block") or {}
                text = tb.get("text", "") if isinstance(tb, dict) else str(tb)
                if not text and "text" in content:
                    text = str(content["text"])
            elif b_type == 10000 or "text_block" in content:
                tb = content.get("text_block", {})
                text = tb.get("text", "") if isinstance(tb, dict) else str(tb)

            if not text and "text" in cb:
                text = str(cb["text"])

            if bool(cb.get("is_thought") or cb.get("thought")):
                is_reasoning = True

            if text:
                block_id = str(cb.get("block_id") or ("thought" if is_reasoning else "text"))
                prev = accumulated_blocks.get(block_id, "")
                if len(text) > len(prev):
                    delta = text[len(prev):]
                    accumulated_blocks[block_id] = text
                    accumulated_text += delta
                    return delta, is_reasoning
            return "", False

        try:
            async with ensure_async_client(self.client, self.timeout) as client:
                async with client.stream(
                    "POST",
                    url,
                    headers=self._headers(),
                    json=payload,
                ) as resp:
                    if resp.status_code == 401:
                        raise ProviderError(
                            "豆包凭据无效或已过期，请运行 .venv/bin/python login.py doubao 重新扫码登录",
                            status=401,
                            err_type="auth_error",
                        )
                    if resp.status_code == 429:
                        raise ProviderError("豆包网页端限流���429），请稍后再试", status=429)

                    # 检查是否为 JSON 报错页面（非 SSE 流）
                    content_type = resp.headers.get("content-type", "")
                    if "text/event-stream" not in content_type:
                        body_bytes = await resp.aread()
                        body_text = body_bytes.decode("utf-8", errors="replace")
                        try:
                            err_obj = json.loads(body_text)
                            if isinstance(err_obj, dict):
                                code = err_obj.get("code", 0)
                                msg = err_obj.get("msg") or err_obj.get("message", body_text[:200])
                                self._handle_error_code(code, msg)
                        except json.JSONDecodeError:
                            pass
                        raise ProviderError(f"豆包返回异常响应 ({resp.status_code}): {body_text[:200]}", status=502)

                    async for line in resp.aiter_lines():
                        if not line:
                            current_event_name = "message"
                            continue
                        if line.startswith("event:"):
                            current_event_name = line[6:].strip() or "message"
                            continue
                        if line.startswith("data:"):
                            data = line[5:].strip()
                        else:
                            data = line.strip()

                        if not data or data == "[DONE]":
                            continue

                        # 网关级认证错误
                        if current_event_name == "gateway-error":
                            try:
                                err_obj = json.loads(data)
                                code = err_obj.get("code", "")
                                msg = err_obj.get("message", data)
                            except Exception:
                                code, msg = "", data
                            raise ProviderError(
                                f"豆包网关验证错误: {code} {msg}，请运行 .venv/bin/python login.py doubao 重新登录",
                                status=401,
                                err_type="auth_error",
                            )

                        try:
                            obj = json.loads(data)
                        except json.JSONDecodeError:
                            continue

                        if not isinstance(obj, dict):
                            continue

                        # 捕获 conversation_id
                        if current_event_name == "SSE_ACK" or "ack_client_meta" in obj:
                            ack = obj.get("ack_client_meta", {})
                            if isinstance(ack, dict):
                                cid = str(ack.get("conversation_id", "")).strip()
                                if cid and cid != "0":
                                    conv_id = cid
                            continue

                        # 检查错误码
                        if current_event_name == "STREAM_ERROR" or "error_code" in obj:
                            err_code = int(obj.get("error_code", 0))
                            err_msg = str(obj.get("error_msg", ""))
                            self._handle_error_code(err_code, err_msg)

                        # 1. 独立思考/文本增量字段 (chunk_delta 或根级)
                        target_obj = obj.get("chunk_delta") if isinstance(obj.get("chunk_delta"), dict) else obj

                        if "thought" in target_obj and isinstance(target_obj.get("thought"), str) and "error_code" not in obj:
                            delta = target_obj["thought"]
                            if delta:
                                accumulated_text += delta
                                yield delta, {"reasoning": True}
                            continue

                        if "reasoning_content" in target_obj and isinstance(target_obj.get("reasoning_content"), str) and "error_code" not in obj:
                            delta = target_obj["reasoning_content"]
                            if delta:
                                accumulated_text += delta
                                yield delta, {"reasoning": True}
                            continue

                        if "text" in target_obj and isinstance(target_obj.get("text"), str) and "error_code" not in obj:
                            delta = target_obj["text"]
                            if delta:
                                accumulated_text += delta
                                is_r = bool(target_obj.get("is_thought") or target_obj.get("is_reasoning") or target_obj.get("thought_delta"))
                                meta = {"reasoning": True} if is_r else {}
                                yield delta, meta
                            continue

                        content_val = target_obj.get("content")
                        if isinstance(content_val, str) and content_val and "error_code" not in obj:
                            accumulated_text += content_val
                            is_r = bool(target_obj.get("is_thought") or target_obj.get("is_reasoning") or target_obj.get("thought_delta"))
                            meta = {"reasoning": True} if is_r else {}
                            yield content_val, meta
                            continue

                        # 2. patch_op 增量片段
                        for patch in obj.get("patch_op", []):
                            pv = patch.get("patch_value", {})
                            patch_path = str(patch.get("patch_path") or patch.get("path") or "")
                            is_patch_reasoning = "thought" in patch_path.lower() or "think" in patch_path.lower()
                            if isinstance(pv, dict):
                                c_str = pv.get("content", "")
                                if isinstance(c_str, str) and c_str:
                                    try:
                                        c_obj = json.loads(c_str)
                                        t = c_obj.get("text", "") or c_obj.get("thought", "")
                                        c_reasoning = is_patch_reasoning or bool(c_obj.get("is_thought") or c_obj.get("thought"))
                                        if t:
                                            accumulated_text += t
                                            meta = {"reasoning": True} if c_reasoning else {}
                                            yield t, meta
                                    except Exception:
                                        pass
                                for cb in pv.get("content_block", []):
                                    delta, is_r = _extract_block(cb)
                                    if delta:
                                        meta = {"reasoning": True} if (is_r or is_patch_reasoning) else {}
                                        yield delta, meta

                        # 3. 根级 content_block
                        for cb in obj.get("content", {}).get("content_block", []):
                            delta, is_r = _extract_block(cb)
                            if delta:
                                meta = {"reasoning": True} if is_r else {}
                                yield delta, meta

            if not accumulated_text:
                raise ProviderError(
                    "豆包未返回有效回复内容，可能是票据失效或触发了风控，请运行 .venv/bin/python login.py doubao 重新登录",
                    status=502,
                )

            # 流结束时输出 conversation_id
            meta = {}
            if conv_id:
                meta["conversation_id"] = conv_id
            yield "", meta

        except ProviderError:
            raise
        except httpx.HTTPStatusError as e:
            await e.response.aread()
            raise ProviderError(
                f"豆包上游请求失败: {e.response.status_code} {e.response.text[:300]}", status=502
            )
        except Exception as e:
            raise ProviderError(f"豆包连接异常: {type(e).__name__}: {e}", status=502)

    async def delete_conversation(self, conversation_id: str) -> bool:
        """删除豆包远端会话。

        conversation_id 为会话 ID（支持前缀截断或纯 ID）。
        删除成功或会话已不存在返回 True，遇到网络/鉴权/业务异常返回 False（不抛错崩溃）。
        """
        if not conversation_id:
            return True

        cid = str(conversation_id).strip()
        if not cid:
            return True

        params = self._security_params()
        url = f"{BASE_URL}{BATCH_DELETE_ENDPOINT}?{urlencode(params)}"
        headers = self._headers()
        headers["Accept"] = "application/json, text/plain, */*"

        payload = {
            "conversation_id": [cid],
            "delete_all": False,
            "conversation_type": 1,
        }

        try:
            async with ensure_async_client(self.client, self.timeout) as client:
                resp = await client.post(url, headers=headers, json=payload)
                if resp.status_code == 200:
                    try:
                        data = resp.json()
                        return data.get("code") == 0
                    except Exception:
                        return False
                return False
        except Exception:
            return False

