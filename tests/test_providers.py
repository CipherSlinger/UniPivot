"""Provider 请求构造与 SSE 解析的离线验证（无需真实 token）。

运行: .venv/bin/python tests/test_providers.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from providers.deepseek_provider import DeepSeekProvider
from providers.qwen_provider import QwenProvider


async def collect(agen):
    text = ""
    meta = {}
    async for delta, m in agen:
        text += delta
        meta.update(m)
    return text, meta


# ---------------------------------------------------------------- Qwen（国内版通义）

_SID = "a" * 32

QWEN_SSE = f"""\
data: {{"communication":{{"sessionid":"{_SID}"}},"data":{{"status":"streaming","messages":[{{"content":"你好"}}]}}}}

data: {{"communication":{{"sessionid":"{_SID}"}},"data":{{"status":"streaming","messages":[{{"content":"你好，世界"}}]}}}}

data: {{"communication":{{"sessionid":"{_SID}"}},"data":{{"status":"complete","messages":[{{"content":"你好，世界"}}]}}}}

data: [DONE]
"""


def test_qwen_payload_and_sse():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, text=QWEN_SSE)

    import providers.qwen_provider as m

    orig_client = m.httpx.AsyncClient

    class MockClient(orig_client):
        def __init__(self, *a, **kw):
            kw["transport"] = httpx.MockTransport(handler)
            super().__init__(*a, **kw)

    m.httpx.AsyncClient = MockClient
    try:
        provider = QwenProvider("ticket_abc", timeout=10)
        # 传入未知模型名 -> 应回退到国内版默认模型 Qwen
        text, meta = asyncio.run(
            collect(provider.chat([{"role": "user", "content": "你好"}], "unknown-model"))
        )
    finally:
        m.httpx.AsyncClient = orig_client

    assert text == "你好，世界", repr(text)
    assert meta["conversation_id"] == _SID, meta
    body = captured["body"]
    assert body["scene"] == "chat"
    assert body["scene_param"] == "first_turn"
    assert body["model"] == "Qwen", body["model"]  # 未知模型回退默认
    assert body["messages"][0]["content"] == "你好"
    assert body["deep_search"] is None
    assert "chat2.qianwen.com/api/v2/chat" in captured["url"]
    cookie = captured["headers"]["cookie"]
    assert "tongyi_sso_ticket=ticket_abc" in cookie, cookie
    assert captured["headers"]["origin"] == "https://www.qianwen.com"
    assert captured["headers"]["x-platform"] == "pc_tongyi"
    print("[PASS] qwen(国内通义): 单端点 payload -> SSE 解析 -> conversation_id 正确")


def test_qwen_search_and_continuation():
    """search 参数映射 deep_search；conversation_id 续聊使用已有 session_id。"""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, text=QWEN_SSE)

    import providers.qwen_provider as m

    orig_client = m.httpx.AsyncClient

    class MockClient(orig_client):
        def __init__(self, *a, **kw):
            kw["transport"] = httpx.MockTransport(handler)
            super().__init__(*a, **kw)

    m.httpx.AsyncClient = MockClient
    try:
        provider = QwenProvider("ticket_abc", timeout=10)
        asyncio.run(collect(
            provider.chat(
                [
                    {"role": "user", "content": "第一轮"},
                    {"role": "assistant", "content": "回答一"},
                    {"role": "user", "content": "第二轮"},
                ],
                "Qwen3.7-Max",
                conversation_id=_SID,
                search=True,
                thinking=True,
            )
        ))
    finally:
        m.httpx.AsyncClient = orig_client

    body = captured["body"]
    assert body["model"] == "Qwen3.7-Max"  # 支持列表内模型原样透传
    assert body["session_id"] == _SID
    assert body["scene_param"] == "continue_turn"
    assert body["deep_search"] == "normal"
    assert body["chat_mode"] == "think"
    content = body["messages"][0]["content"]
    assert content == "第二轮", content
    print("[PASS] qwen(国内通义): search/think 映射 + 续聊 session_id 传递")


def test_qwen_not_login_error():
    """票据无效时上游返回 error_code 包含 LOGIN -> 401 ProviderError。"""
    import providers.qwen_provider as m
    from providers.base import ProviderError

    sse = (
        'data: {"error_code":"USER_LOGIN_TIMEOUT","error_msg":"请先登录"}\n\n'
        "data: [DONE]\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=sse)

    orig_client = m.httpx.AsyncClient

    class MockClient(orig_client):
        def __init__(self, *a, **kw):
            kw["transport"] = httpx.MockTransport(handler)
            super().__init__(*a, **kw)

    m.httpx.AsyncClient = MockClient
    try:
        provider = QwenProvider("expired_ticket", timeout=10)
        try:
            asyncio.run(collect(provider.chat([{"role": "user", "content": "hi"}], "Qwen")))
            raise AssertionError("应当抛出 401 ProviderError")
        except ProviderError as e:
            assert e.status == 401, e.status
            assert "login.py" in e.message
    finally:
        m.httpx.AsyncClient = orig_client
    print("[PASS] qwen(国内通义): 票据失效 -> 401 错误提示指向 login.py")


def test_qwen_waf_error():
    """触发 WAF 人机验证时抛出 429 ProviderError。"""
    import providers.qwen_provider as m
    from providers.base import ProviderError
    import login

    orig_refresh = getattr(login, "refresh_qwen", None)
    login.refresh_qwen = lambda *a, **kw: None

    sse = (
        'data: {"ret":["FAIL_SYS_USER_VALIDATE::RGV587_ERROR"]}\n\n'
        "data: [DONE]\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=sse)

    orig_client = m.httpx.AsyncClient

    class MockClient(orig_client):
        def __init__(self, *a, **kw):
            kw["transport"] = httpx.MockTransport(handler)
            super().__init__(*a, **kw)

    m.httpx.AsyncClient = MockClient
    try:
        provider = QwenProvider("ticket_abc", timeout=10)
        try:
            asyncio.run(collect(provider.chat([{"role": "user", "content": "hi"}], "Qwen")))
            raise AssertionError("应当抛出 429 ProviderError")
        except ProviderError as e:
            assert e.status == 429, e.status
            assert "RGV587" in e.message
    finally:
        m.httpx.AsyncClient = orig_client
        if orig_refresh is not None:
            login.refresh_qwen = orig_refresh
    print("[PASS] qwen(国内通义): WAF 人机验证 -> 429 错误提示")


def test_qwen_delete_conversation():
    """测试删除通义千问会话接口（POST /api/v2/session/delete）。"""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        path = request.url.path
        if path.endswith("/session/delete"):
            return httpx.Response(200, json={"success": True, "code": 0, "data": {}})
        return httpx.Response(404)

    import providers.qwen_provider as m

    orig_client = m.httpx.AsyncClient

    class MockClient(orig_client):
        def __init__(self, *a, **kw):
            kw["transport"] = httpx.MockTransport(handler)
            super().__init__(*a, **kw)

    m.httpx.AsyncClient = MockClient
    try:
        provider = QwenProvider("ticket_abc", timeout=10)
        res = asyncio.run(provider.delete_conversation(_SID))
        assert res is True
        assert "/api/v2/session/delete" in captured["url"]
        assert captured["body"] == {"session_id": _SID, "biz_id": "ai_qwen"}

        # 空值安全处理
        assert asyncio.run(provider.delete_conversation("")) is True
        assert asyncio.run(provider.delete_conversation(None)) is True
    finally:
        m.httpx.AsyncClient = orig_client

    print("[PASS] qwen(国内通义): delete_conversation 会话删除与异常优雅处理")



# ---------------------------------------------------------------- DeepSeek

DS_SSE = """\
data: {"v":{"response":{"id":42,"message_id":42,"fragments":[{"type":"RESPONSE","content":"深度"}]}}}

data: {"p":"response/fragments/-1/content","o":"APPEND","v":"求索"}

data: {"v":"！"}
"""


def test_deepseek_pow_and_sse():
    captured = {}

    class FakePow:
        def make_header(self, challenge):
            captured["challenge"] = challenge
            return "fake-pow-header"

    import providers.deepseek_provider as ds_mod

    ds_mod._get_pow = lambda: FakePow()

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        path = request.url.path
        if path.endswith("/chat_session/create"):
            return httpx.Response(
                200, json={"code": 0, "data": {"biz_data": {"chat_session": {"id": "sess_9"}}}}
            )
        if path.endswith("/create_pow_challenge"):
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "data": {
                        "biz_data": {
                            "challenge": {
                                "algorithm": "DeepSeekHashV1",
                                "challenge": "ch",
                                "salt": "salt",
                                "difficulty": 1.0,
                                "expire_at": 9999999999,
                                "signature": "sig",
                                "target_path": "/api/v0/chat/completion",
                            }
                        }
                    },
                },
            )
        return httpx.Response(200, text=DS_SSE)

    import providers.deepseek_provider as m

    orig_client = m.httpx.AsyncClient

    class MockClient(orig_client):
        def __init__(self, *a, **kw):
            kw["transport"] = httpx.MockTransport(handler)
            super().__init__(*a, **kw)

    m.httpx.AsyncClient = MockClient
    try:
        provider = DeepSeekProvider("tok_ds", timeout=10)
        text, meta = asyncio.run(collect(provider.chat([{"role": "user", "content": "你好"}], "deepseek-chat")))
    finally:
        m.httpx.AsyncClient = orig_client

    assert text == "深度求索！", repr(text)
    assert meta["conversation_id"] == "sess_9:42", meta
    assert captured["headers"]["x-ds-pow-response"] == "fake-pow-header"
    assert captured["headers"]["authorization"] == "Bearer tok_ds"
    assert captured["body"]["model_type"] == "default"
    assert captured["body"]["chat_session_id"] == "sess_9"
    assert captured["body"]["prompt"] == "你好"
    print("[PASS] deepseek: 会话创建 -> PoW 头 -> SSE 解析 -> conversation_id 正确")


def test_deepseek_delete_conversation():
    """测试删除 DeepSeek 会话接口（POST /api/v0/chat_session/delete）。"""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        path = request.url.path
        if path.endswith("/chat_session/delete"):
            return httpx.Response(
                200, json={"code": 0, "msg": "", "data": {"biz_code": 0, "biz_msg": "", "biz_data": None}}
            )
        return httpx.Response(404)

    import providers.deepseek_provider as m

    orig_client = m.httpx.AsyncClient

    class MockClient(orig_client):
        def __init__(self, *a, **kw):
            kw["transport"] = httpx.MockTransport(handler)
            super().__init__(*a, **kw)

    m.httpx.AsyncClient = MockClient
    try:
        provider = DeepSeekProvider("tok_ds", timeout=10)
        # 测试完整 conversation_id 带 message_id
        res1 = asyncio.run(provider.delete_conversation("sess_123:456"))
        assert res1 is True
        assert captured["url"] == "https://chat.deepseek.com/api/v0/chat_session/delete"
        assert captured["headers"]["authorization"] == "Bearer tok_ds"
        assert captured["body"] == {"chat_session_id": "sess_123"}

        # 测试只有 session_id
        res2 = asyncio.run(provider.delete_conversation("sess_789"))
        assert res2 is True
        assert captured["body"] == {"chat_session_id": "sess_789"}

        # 测试空 conversation_id
        assert asyncio.run(provider.delete_conversation("")) is True
        assert asyncio.run(provider.delete_conversation(":")) is True
    finally:
        m.httpx.AsyncClient = orig_client

    print("[PASS] deepseek: 删除会话 delete_conversation 正确提取 session_id 并发送请求")


def test_deepseek_pow_wasm_loads():
    """真实 wasm 能被 wasmtime 加载（不求解挑战，仅实例化）。"""
    from pow.deepseek_pow import DeepSeekPow

    solver = DeepSeekPow()
    assert solver is not None
    print("[PASS] deepseek PoW: 官方 wasm 在 wasmtime 中加载成功")


# ---------------------------------------------------------------- Doubao（豆包）

DOUBAO_CID = "conv_doubao_123"

DOUBAO_SSE = f"""\
event: SSE_ACK
data: {{"ack_client_meta":{{"conversation_id":"{DOUBAO_CID}"}}}}

data: {{"text":"豆包"}}

data: {{"text":"为你服务！"}}

data: [DONE]
"""


def test_doubao_payload_and_sse():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, text=DOUBAO_SSE)

    from providers.doubao import DoubaoProvider
    import providers.doubao.provider as m

    orig_client = m.httpx.AsyncClient

    class MockClient(orig_client):
        def __init__(self, *a, **kw):
            kw["transport"] = httpx.MockTransport(handler)
            super().__init__(*a, **kw)

    m.httpx.AsyncClient = MockClient
    try:
        provider = DoubaoProvider("sess_tok_123", timeout=10)
        text, meta = asyncio.run(
            collect(provider.chat([{"role": "user", "content": "你好"}], "doubao"))
        )
    finally:
        m.httpx.AsyncClient = orig_client

    assert text == "豆包为你服务！", repr(text)
    assert meta["conversation_id"] == DOUBAO_CID, meta
    assert "chat/completion" in captured["url"]
    assert "client_platform=pc_client" in captured["url"]
    assert "sessionid=sess_tok_123" in captured["headers"]["cookie"]
    assert captured["headers"]["origin"] == "https://www.doubao.com"

    body = captured["body"]
    assert body["client_meta"]["bot_id"] == "7338286299411103781"  # Pro 默认
    assert body["option"]["need_create_conversation"] is True
    assert body["option"]["need_deep_think"] == 0
    assert body["messages"][0]["content_block"][0]["content"]["text_block"]["text"] == "你好"
    print("[PASS] doubao: payload -> SSE_ACK -> chunk_delta 解析 -> conversation_id 正确")


def test_doubao_thinking_and_continuation():
    """doubao-think 启用深度思考；doubao-lite 路由至 Lite Bot；续聊传递 conversation_id。"""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, text=DOUBAO_SSE)

    from providers.doubao import DoubaoProvider
    import providers.doubao.provider as m

    orig_client = m.httpx.AsyncClient

    class MockClient(orig_client):
        def __init__(self, *a, **kw):
            kw["transport"] = httpx.MockTransport(handler)
            super().__init__(*a, **kw)

    m.httpx.AsyncClient = MockClient
    try:
        # 测试 1: doubao-think
        provider = DoubaoProvider("sess_tok_123", timeout=10)
        asyncio.run(
            collect(
                provider.chat(
                    [{"role": "user", "content": "深思问题"}],
                    "doubao-think",
                    conversation_id=DOUBAO_CID,
                )
            )
        )
        body = captured["body"]
        assert body["option"]["need_deep_think"] == 1
        assert body["ext"]["use_deep_think"] == "1"
        assert body["client_meta"]["conversation_id"] == DOUBAO_CID
        assert body["option"]["need_create_conversation"] is False

        # 测试 2: doubao-lite
        asyncio.run(
            collect(
                provider.chat(
                    [{"role": "user", "content": "轻量问题"}],
                    "doubao-lite",
                )
            )
        )
        body2 = captured["body"]
        assert body2["client_meta"]["bot_id"] == "7234781073513644036"
    finally:
        m.httpx.AsyncClient = orig_client

    print("[PASS] doubao: think 深度思考 + lite Bot ID + 续聊 conversation_id 传递")


def test_doubao_auth_error():
    """上游返回登录已过期（710012001）转为 401 ProviderError 并提示 login.py doubao。"""
    from providers.base import ProviderError
    from providers.doubao import DoubaoProvider
    import providers.doubao.provider as m

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Type": "application/json"},
            text='{"code": 710012001, "msg": "登录已过期，请重新登录"}',
        )

    orig_client = m.httpx.AsyncClient

    class MockClient(orig_client):
        def __init__(self, *a, **kw):
            kw["transport"] = httpx.MockTransport(handler)
            super().__init__(*a, **kw)

    m.httpx.AsyncClient = MockClient
    try:
        provider = DoubaoProvider("expired_tok", timeout=10)
        try:
            asyncio.run(collect(provider.chat([{"role": "user", "content": "hi"}], "doubao")))
            raise AssertionError("应当抛出 401 ProviderError")
        except ProviderError as e:
            assert e.status == 401, e.status
            assert "login.py doubao" in e.message
    finally:
        m.httpx.AsyncClient = orig_client

    print("[PASS] doubao: 登录过期 (710012001) -> 401 错误提示指向 login.py doubao")


def test_doubao_risk_error():
    """上游返回风控验证（710022004）转为 429 ProviderError。"""
    from providers.base import ProviderError
    from providers.doubao import DoubaoProvider
    import providers.doubao.provider as m

    sse = (
        'event: STREAM_ERROR\n'
        'data: {"error_code": 710022004, "error_msg": "captcha required"}\n\n'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, text=sse)

    orig_client = m.httpx.AsyncClient

    class MockClient(orig_client):
        def __init__(self, *a, **kw):
            kw["transport"] = httpx.MockTransport(handler)
            super().__init__(*a, **kw)

    m.httpx.AsyncClient = MockClient
    try:
        provider = DoubaoProvider("sess_tok", timeout=10)
        try:
            asyncio.run(collect(provider.chat([{"role": "user", "content": "hi"}], "doubao")))
            raise AssertionError("应当抛出 429 ProviderError")
        except ProviderError as e:
            assert e.status == 429, e.status
            assert "风控/人机验证" in e.message
    finally:
        m.httpx.AsyncClient = orig_client

    print("[PASS] doubao: 风控人机验证 (710022004) -> 429 错误提示")


def test_doubao_delete_conversation():
    """测试删除豆包会话接口（POST /im/conversation/batch_del_user_conv）。"""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        path = request.url.path
        if path.endswith("/im/conversation/batch_del_user_conv"):
            return httpx.Response(200, json={"code": 0, "msg": "success", "data": {}})
        return httpx.Response(404)

    from providers.doubao import DoubaoProvider
    import providers.doubao.provider as m

    orig_client = m.httpx.AsyncClient

    class MockClient(orig_client):
        def __init__(self, *a, **kw):
            kw["transport"] = httpx.MockTransport(handler)
            super().__init__(*a, **kw)

    m.httpx.AsyncClient = MockClient
    try:
        provider = DoubaoProvider("sess_tok_123", timeout=10)
        # 正常会话删除
        res = asyncio.run(provider.delete_conversation("conv_db_456"))
        assert res is True
        assert "/im/conversation/batch_del_user_conv" in captured["url"]
        assert captured["headers"]["x-tt-passport-csrf-token"] == provider._csrf_token
        assert captured["body"] == {
            "conversation_id": ["conv_db_456"],
            "delete_all": False,
            "conversation_type": 1,
        }

        # 空值安全处理
        assert asyncio.run(provider.delete_conversation("")) is True
        assert asyncio.run(provider.delete_conversation(None)) is True
    finally:
        m.httpx.AsyncClient = orig_client

    print("[PASS] doubao: delete_conversation 会话删除与异常优雅处理")


def test_parse_tool_calls_edge_cases():
    from providers.anthropic_compat import parse_tool_call, parse_tool_calls

    # 1. 多个代码块
    multi_code_block = (
        "准备读取和写入：\n"
        "```tool_call\n"
        '{"name": "Read", "arguments": {"file_path": "/tmp/a.txt"}}\n'
        "```\n"
        "接下来写入：\n"
        "```tool_call\n"
        '{"name": "Write", "arguments": {"file_path": "/tmp/b.txt", "content": "ok"}}\n'
        "```"
    )
    res = parse_tool_calls(multi_code_block)
    assert res is not None
    prefix, tools = res
    assert prefix == "准备读取和写入："
    assert len(tools) == 2
    assert tools[0] == ("Read", {"file_path": "/tmp/a.txt"})
    assert tools[1] == ("Write", {"file_path": "/tmp/b.txt", "content": "ok"})

    # 兼容单工具 parse_tool_call
    single_res = parse_tool_call(multi_code_block)
    assert single_res == ("准备读取和写入：", "Read", {"file_path": "/tmp/a.txt"})

    # 2. 单个代码块中包含工具调用列表
    list_code_block = (
        "执行命令：\n```tool_call\n"
        '[\n  {"name": "cmd1", "arguments": {"a": 1}},\n  {"name": "cmd2", "arguments": {"b": 2}}\n]\n'
        "```"
    )
    res_list = parse_tool_calls(list_code_block)
    assert res_list is not None
    assert len(res_list[1]) == 2
    assert res_list[1][0] == ("cmd1", {"a": 1})
    assert res_list[1][1] == ("cmd2", {"b": 2})

    # 3. 参数为空字典、空字符串、null、未提供 arguments 时的鲁棒性
    assert parse_tool_calls('```tool_call\n{"name": "t1", "arguments": {}}\n```')[1][0] == ("t1", {})
    assert parse_tool_calls('```tool_call\n{"name": "t2", "arguments": ""}\n```')[1][0] == ("t2", {})
    assert parse_tool_calls('```tool_call\n{"name": "t3", "arguments": "{}"}\n```')[1][0] == ("t3", {})
    assert parse_tool_calls('```tool_call\n{"name": "t4", "arguments": null}\n```')[1][0] == ("t4", {})
    assert parse_tool_calls('```tool_call\n{"name": "t5"}\n```')[1][0] == ("t5", {})

    # 4. 复杂嵌套 JSON、尾部多余逗号与 Python 单引号字面量
    nested_trailing = (
        "```tool_call\n"
        '{\n'
        '  "name": "deep_tool",\n'
        '  "arguments": {\n'
        '    "nested": {"arr": [1, 2, ], "active": true, },\n'
        '    "text": "带有 \\"引号\\" 和 \\n 换行",\n'
        '  },\n'
        '}\n'
        "```"
    )
    res_nested = parse_tool_calls(nested_trailing)
    assert res_nested is not None
    args_nested = res_nested[1][0][1]
    assert args_nested["nested"]["arr"] == [1, 2]
    assert args_nested["nested"]["active"] is True
    assert "引号" in args_nested["text"]

    # 5. Python 单引号字典字面量
    py_dict = "```tool_call\n{'name': 'py_tool', 'arguments': {'key': 'val', 'flag': True}}\n```"
    res_py = parse_tool_calls(py_dict)
    assert res_py is not None
    assert res_py[1][0] == ("py_tool", {"key": "val", "flag": True})

    # 6. 多 XML 标签
    xml_text = (
        "<tool_call>\n"
        '{"name": "xml_tool_1", "arguments": {"x": 10}}\n'
        "</tool_call>\n"
        "<tool_call>\n"
        '{"name": "xml_tool_2", "arguments": {"y": 20}}\n'
        "</tool_call>"
    )
    res_xml = parse_tool_calls(xml_text)
    assert res_xml is not None
    assert len(res_xml[1]) == 2
    assert res_xml[1][0] == ("xml_tool_1", {"x": 10})
    assert res_xml[1][1] == ("xml_tool_2", {"y": 20})

    print("[PASS] anthropic_compat: parse_tool_calls 多工具、空参数与复杂嵌套边界测试")


def test_http_client_connection_pool_and_ensure_client():
    """验证全局 HTTP 客户端连接池管理、单例生命周期与 ensure_async_client 行为。"""
    from providers.base import ensure_async_client
    from providers.deepseek import DeepSeekProvider
    from providers.doubao import DoubaoProvider
    from providers.http_client import (
        close_http_client,
        create_async_client,
        get_default_limits,
        get_http_client,
    )
    from providers.qwen import QwenProvider

    # 1. 验证默认连接池限制参数
    limits = get_default_limits()
    assert limits.max_connections == 100
    assert limits.max_keepalive_connections == 40
    assert limits.keepalive_expiry == 30.0

    # 2. 验证 create_async_client 超时设置与重定向
    client_inst = create_async_client(timeout=120.0)
    assert client_inst.timeout.connect == 15.0
    assert client_inst.timeout.pool == 15.0
    assert client_inst.timeout.read == 120.0
    assert client_inst.follow_redirects is True
    asyncio.run(client_inst.aclose())

    # 3. 验证 get_http_client 单例与 close_http_client 生命周期
    async def verify_singleton():
        c1 = await get_http_client(timeout=300.0)
        c2 = await get_http_client(timeout=300.0)
        assert c1 is c2, "get_http_client 应当返回同一共享单例"
        assert not c1.is_closed

        # 4. 验证 ensure_async_client 复用逻辑
        # 4.1 当传入显式未关闭 client 时，不应关闭且原样 yield
        custom_client = create_async_client(timeout=50.0)
        async with ensure_async_client(custom_client) as yield_client:
            assert yield_client is custom_client
        assert not custom_client.is_closed, "ensure_async_client 不应关闭外部传入的 client"
        await custom_client.aclose()

        # 4.2 当 client 为 None 时，yield 共享单例且不关闭单例
        async with ensure_async_client(None) as yield_singleton:
            assert yield_singleton is c1
        assert not c1.is_closed, "ensure_async_client 不应关闭全局单例 client"

        # 4.3 当传入已关闭 client 时，应回退并 yield 共享单例
        async with ensure_async_client(custom_client) as yield_fallback:
            assert yield_fallback is c1

        # 5. 验证 close_http_client 优雅关闭单例
        await close_http_client()
        assert c1.is_closed, "close_http_client 应当关闭单例"

        # 再次获取时自动新建单例
        c3 = await get_http_client()
        assert c3 is not c1
        assert not c3.is_closed
        await close_http_client()
        assert c3.is_closed

    asyncio.run(verify_singleton())

    # 6. 验证 Providers 的 __init__ 支持可选 client 传参
    test_cli = create_async_client()
    try:
        q_prov = QwenProvider("ticket", client=test_cli)
        assert q_prov.client is test_cli

        ds_prov = DeepSeekProvider("token", client=test_cli)
        assert ds_prov.client is test_cli

        db_prov = DoubaoProvider("token", client=test_cli)
        assert db_prov.client is test_cli
    finally:
        asyncio.run(test_cli.aclose())

    print("[PASS] http_client: 连接池配置、单例生命周期、ensure_async_client 与 provider client 注入验证")


def test_reasoning_sse_parsing_all_providers():
    """验证 DeepSeek、Doubao 和 Qwen 三大 Provider 对思考片段的解析与 reasoning 元数据标记。"""
    # 1. DeepSeek 思考帧解析
    ds_sse = """\
data: {"v":{"response":{"id":100,"message_id":100,"fragments":[{"type":"THINK","content":"正在分析问题"},{"type":"RESPONSE","content":"最终答案"}]}}}

data: {"p":"response/fragments/0/content","o":"APPEND","v":"，深入推演"}

data: {"p":"response/fragments/1/content","o":"APPEND","v":"如下"}
"""
    class FakePow:
        def make_header(self, challenge):
            return "fake-pow"

    import providers.deepseek_provider as ds_mod
    orig_pow = ds_mod._get_pow
    ds_mod._get_pow = lambda: FakePow()

    def ds_handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/chat_session/create"):
            return httpx.Response(200, json={"code": 0, "data": {"biz_data": {"chat_session": {"id": "sess_ds"}}}})
        if path.endswith("/create_pow_challenge"):
            return httpx.Response(200, json={"code": 0, "data": {"biz_data": {"challenge": {}}}})
        return httpx.Response(200, text=ds_sse)

    orig_client = ds_mod.httpx.AsyncClient
    class MockClientDS(orig_client):
        def __init__(self, *a, **kw):
            kw["transport"] = httpx.MockTransport(ds_handler)
            super().__init__(*a, **kw)

    ds_mod.httpx.AsyncClient = MockClientDS
    try:
        ds_provider = DeepSeekProvider("fake_token", timeout=10)
        chunks = []
        async def run_ds():
            async for delta, meta in ds_provider.chat([{"role": "user", "content": "hi"}], "deepseek-reasoner"):
                chunks.append((delta, meta))
        asyncio.run(run_ds())
    finally:
        ds_mod.httpx.AsyncClient = orig_client
        ds_mod._get_pow = orig_pow

    reasoning_ds = "".join(c[0] for c in chunks if c[1].get("reasoning"))
    content_ds = "".join(c[0] for c in chunks if not c[1].get("reasoning") and c[0])
    assert reasoning_ds == "正在分析问题，深入推演", reasoning_ds
    assert content_ds == "最终答案如下", content_ds

    # 2. Doubao 思考帧解析
    db_sse = """\
event: STREAM_CHUNK
data: {"chunk_delta":{"thought":"开始思考..."}}

event: STREAM_CHUNK
data: {"chunk_delta":{"thought":"梳理逻辑..."}}

event: STREAM_CHUNK
data: {"chunk_delta":{"content":"这是输出结果"}}
"""
    import providers.doubao.provider as db_mod
    def db_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, text=db_sse)

    class MockClientDB(orig_client):
        def __init__(self, *a, **kw):
            kw["transport"] = httpx.MockTransport(db_handler)
            super().__init__(*a, **kw)

    db_mod.httpx.AsyncClient = MockClientDB
    try:
        from providers.doubao import DoubaoProvider
        db_provider = DoubaoProvider("fake_sess", timeout=10)
        chunks_db = []
        async def run_db():
            async for delta, meta in db_provider.chat([{"role": "user", "content": "hi"}], "doubao-think"):
                chunks_db.append((delta, meta))
        asyncio.run(run_db())
    finally:
        db_mod.httpx.AsyncClient = orig_client

    reasoning_db = "".join(c[0] for c in chunks_db if c[1].get("reasoning"))
    content_db = "".join(c[0] for c in chunks_db if not c[1].get("reasoning") and c[0])
    assert reasoning_db == "开始思考...梳理逻辑...", reasoning_db
    assert content_db == "这是输出结果", content_db

    # 3. Qwen 思考与回复分离解析
    qwen_sse = """\
data: {"communication":{"sessionid":"sess_qw"},"data":{"status":"streaming","messages":[{"content":"<thought>逐步推导：第一步"}]}}

data: {"communication":{"sessionid":"sess_qw"},"data":{"status":"streaming","messages":[{"content":"<thought>逐步推导：第一步；第二步</thought>最终答案"}]}}

data: {"communication":{"sessionid":"sess_qw"},"data":{"status":"complete","messages":[{"content":"<thought>逐步推导：第一步；第二步</thought>最终答案！"}]}}
"""
    import providers.qwen.provider as qw_mod
    def qw_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=qwen_sse)

    class MockClientQW(orig_client):
        def __init__(self, *a, **kw):
            kw["transport"] = httpx.MockTransport(qw_handler)
            super().__init__(*a, **kw)

    qw_mod.httpx.AsyncClient = MockClientQW
    try:
        qw_provider = QwenProvider("fake_ticket", timeout=10)
        chunks_qw = []
        async def run_qw():
            async for delta, meta in qw_provider.chat([{"role": "user", "content": "hi"}], "Qwen3.7-Max"):
                chunks_qw.append((delta, meta))
        asyncio.run(run_qw())
    finally:
        qw_mod.httpx.AsyncClient = orig_client

    reasoning_qw = "".join(c[0] for c in chunks_qw if c[1].get("reasoning"))
    content_qw = "".join(c[0] for c in chunks_qw if not c[1].get("reasoning") and c[0])
    assert reasoning_qw == "逐步推导：第一步；第二步", reasoning_qw
    assert content_qw == "最终答案！", content_qw

    print("[PASS] providers: DeepSeek, Doubao, Qwen 思考过程提取与 reasoning 元数据产出验证")


def test_kimi_payload_and_sse():
    """测试 Kimi 网页端协议：Token Refresh、请求构建、cmpl 与 k1 思考流解析、会话删除。"""
    from providers.kimi import KimiProvider
    import providers.kimi.provider as kimi_mod

    kimi_sse = """\
data: {"event":"cmpl","text":"你好！我是"}

data: {"event":"k1","text":"正在深度思考：第一步..."}

data: {"event":"k1","text":"第二步分析完成。"}

data: {"event":"cmpl","text":" Kimi 智能助手。"}

data: {"event":"all_done"}
"""
    captured = {}

    def kimi_handler(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        captured["last_url"] = url_str
        captured["headers"] = dict(request.headers)

        if url_str.endswith("/api/auth/token/refresh"):
            return httpx.Response(
                200,
                json={
                    "access_token": "mock_kimi_access_token",
                    "refresh_token": "mock_kimi_refresh_token",
                    "user_id": "user_kimi_123",
                },
            )
        if url_str.endswith("/api/chat"):
            body = json.loads(request.content)
            captured["create_chat_body"] = body
            return httpx.Response(200, json={"id": "chat_kimi_456"})
        if "/completion/stream" in url_str:
            body = json.loads(request.content)
            captured["stream_body"] = body
            return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, text=kimi_sse)
        if url_str.endswith("/api/chat/chat_kimi_456"):
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(404)

    orig_client = kimi_mod.httpx.AsyncClient

    class MockClientKimi(orig_client):
        def __init__(self, *a, **kw):
            kw["transport"] = httpx.MockTransport(kimi_handler)
            super().__init__(*a, **kw)

    kimi_mod.httpx.AsyncClient = MockClientKimi
    # 清空缓存
    kimi_mod._token_cache.clear()

    try:
        provider = KimiProvider("mock_kimi_refresh_token", timeout=10)
        chunks = []

        async def run_kimi():
            async for delta, meta in provider.chat(
                [{"role": "user", "content": "你好 Kimi"}],
                model="kimi-explore",
            ):
                chunks.append((delta, meta))

        asyncio.run(run_kimi())

        reasoning = "".join(c[0] for c in chunks if c[1].get("reasoning"))
        content = "".join(c[0] for c in chunks if not c[1].get("reasoning") and c[0])
        conv_id = [c[1].get("conversation_id") for c in chunks if c[1].get("conversation_id")][0]

        assert reasoning == "正在深度思考：第一步...第二步分析完成。", reasoning
        assert content == "你好！我是 Kimi 智能助手。", content
        assert conv_id == "chat_kimi_456"

        assert captured["stream_body"]["model"] == "k1"
        assert captured["headers"]["authorization"] == "Bearer mock_kimi_access_token"

        # 测试 delete_conversation
        del_res = asyncio.run(provider.delete_conversation("chat_kimi_456"))
        assert del_res is True
    finally:
        kimi_mod.httpx.AsyncClient = orig_client

    print("[PASS] kimi: Token 刷新 -> 会话创建 -> k1 思考与 cmpl 文本解析 -> 会话删除验证通过")


def test_glm_payload_and_sse():
    """测试 智谱清言 GLM 网页端协议：Token Refresh、MD5 动态签名、parts 增量思考/文本解析、会话删除。"""
    from providers.glm import GLMProvider, build_glm_sign
    import providers.glm.provider as glm_mod

    # 验证 MD5 签名算法
    ts, nonce, sign = build_glm_sign()
    assert len(ts) >= 10
    assert len(nonce) == 32
    assert len(sign) == 32

    glm_sse = """\
data: {"conversation_id":"glm_conv_789","parts":[{"logic_id":"part_0","content":[{"type":"reasoning","text":"GLM 正在思考架构设计..."}]}]}

data: {"conversation_id":"glm_conv_789","parts":[{"logic_id":"part_0","content":[{"type":"reasoning","text":"GLM 正在思考架构设计...完成！"},{"type":"text","text":"这是 GLM-4-Plus"}]}]}

data: {"conversation_id":"glm_conv_789","parts":[{"logic_id":"part_0","content":[{"type":"text","text":"这是 GLM-4-Plus 的正式解答。"}]}]}

data: [DONE]
"""
    captured = {}

    def glm_handler(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        captured["last_url"] = url_str
        captured["headers"] = dict(request.headers)

        if url_str.endswith("/user-api/user/refresh"):
            return httpx.Response(
                200,
                json={
                    "result": {
                        "accessToken": "mock_glm_access_token",
                    }
                },
            )
        if url_str.endswith("/backend-api/assistant/stream"):
            body = json.loads(request.content)
            captured["stream_body"] = body
            return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, text=glm_sse)
        if "/user-api/assistant/conversation/delete" in url_str:
            return httpx.Response(200, json={"code": 0, "status": "success"})
        return httpx.Response(404)

    orig_client = glm_mod.httpx.AsyncClient

    class MockClientGLM(orig_client):
        def __init__(self, *a, **kw):
            kw["transport"] = httpx.MockTransport(glm_handler)
            super().__init__(*a, **kw)

    glm_mod.httpx.AsyncClient = MockClientGLM
    glm_mod._token_cache.clear()

    try:
        provider = GLMProvider("mock_glm_refresh_token", timeout=10)
        chunks = []

        async def run_glm():
            async for delta, meta in provider.chat(
                [{"role": "user", "content": "你好 GLM"}],
                model="glm-4-plus",
            ):
                chunks.append((delta, meta))

        asyncio.run(run_glm())

        reasoning = "".join(c[0] for c in chunks if c[1].get("reasoning"))
        content = "".join(c[0] for c in chunks if not c[1].get("reasoning") and c[0])
        conv_id = [c[1].get("conversation_id") for c in chunks if c[1].get("conversation_id")][0]

        assert reasoning == "GLM 正在思考架构设计...完成！", reasoning
        assert content == "这是 GLM-4-Plus 的正式解答。", content
        assert conv_id == "glm_conv_789"
        assert captured["stream_body"]["assistant_id"] == "65940acff94777010aa6b796"
        assert captured["headers"]["authorization"] == "Bearer mock_glm_access_token"

        # 测试 delete_conversation
        del_res = asyncio.run(provider.delete_conversation("glm_conv_789"))
        assert del_res is True
    finally:
        glm_mod.httpx.AsyncClient = orig_client

    print("[PASS] glm: 动态签名 -> 令牌刷新 -> parts 思考/正文流式解析 -> 会话删除验证通过")


if __name__ == "__main__":
    test_qwen_payload_and_sse()
    test_qwen_search_and_continuation()
    test_qwen_not_login_error()
    test_qwen_waf_error()
    test_qwen_delete_conversation()
    test_deepseek_pow_and_sse()
    test_deepseek_delete_conversation()
    test_deepseek_pow_wasm_loads()
    test_doubao_payload_and_sse()
    test_doubao_thinking_and_continuation()
    test_doubao_auth_error()
    test_doubao_risk_error()
    test_doubao_delete_conversation()
    test_parse_tool_calls_edge_cases()
    test_http_client_connection_pool_and_ensure_client()
    test_reasoning_sse_parsing_all_providers()
    test_kimi_payload_and_sse()
    test_glm_payload_and_sse()
    print("\n全部通过")
