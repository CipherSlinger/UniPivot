"""`previous_response_id` 不得被静默忽略 (RESPONSES-PROTOCOL)

[PROTOCOL] 与 H4 同一类的「声明了、解析了、然后无人使用」。

Responses 协议允许客户端只发**增量**，把前文交给服务端：
`{"previous_response_id": "resp_...", "input": "<本轮新问题>"}`。这个字段此前
不在 `ResponsesRequest` 的声明里，靠 `extra="allow"` 收进 `model_extra`，
**没有任何代码读它**。实测（脚本化 provider，`TESTING=1`，零真实上游调用）：

    body: {"model":"gpt-4o","store":true,"previous_response_id":"resp_abc123",
           "input":[{"type":"message","role":"user","content":"what did I ask before?"}]}
    -> HTTP 200，且上游收到的提示词只有一句
       [{"role": "user", "content": "what did I ask before?"}]

模型拿不到任何上文，于是对着一句没有前因的问题自信作答——一次**没有报错的错答案**，
而客户端的契约恰好是「前文由服务端给」。

修复不能是「静默新建一次会话」：那正是上面测到的行为。网关本来就不持久化
response 对象，所以任何 `previous_response_id` 都是**必然解析不了**的。诚实的
做法是照 OpenAI 自己的线上契约拒绝——`404` + `code: previous_response_not_found`，
这恰好是真实客户端（openai-python、各家网关）已经写好恢复逻辑的那个 code：
收到它就用完整历史重发一次。静默忽略则让他们永远走不到那条恢复路径。

本套件同时钉住零回归：不带该字段的请求行为逐值不变。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["TESTING"] = "1"
os.environ["GATEWAY_DISABLE_CREDENTIAL_WATCH"] = "1"
os.environ["GATEWAY_DISABLE_AUTO_REFRESH"] = "1"
os.environ["GATEWAY_DISABLE_SINGLEFLIGHT"] = "1"
os.environ["QWEN_AUTH_TOKEN"] = "fake_qwen"
os.environ["DEEPSEEK_AUTH_TOKEN"] = "fake_ds"
os.environ["DOUBAO_AUTH_TOKEN"] = "fake_doubao"
os.environ["KIMI_AUTH_TOKEN"] = "fake_kimi"
os.environ["GLM_AUTH_TOKEN"] = "fake_glm"

from fastapi.testclient import TestClient  # noqa: E402

import server as srv  # noqa: E402

_DEFAULT_FACTORY = srv._make_provider
CALLS: list[dict] = []


class CountingProvider:
    """记录调用次数：被调用过，就证明拒绝没有发生在派发之前。"""

    async def chat(self, messages, model, **kw):
        CALLS.append({"model": model, "messages": messages})
        yield "ok", {}
        yield "", {"conversation_id": None}

    async def delete_conversation(self, *a, **kw):
        return None


def _install():
    CALLS.clear()
    srv._make_provider = lambda key, *a, **kw: CountingProvider()


def _restore():
    srv._make_provider = _DEFAULT_FACTORY


def test_previous_response_id_is_refused_not_ignored():
    """**靶点**：修复前 HTTP 200，上游只收到一句裸问题。"""
    _install()
    try:
        for stream in (False, True):
            CALLS.clear()
            r = TestClient(srv.app).post("/v1/responses", json={
                "model": "gpt-4o",
                "store": True,
                "stream": stream,
                "previous_response_id": "resp_abc123",
                "input": [{"type": "message", "role": "user",
                           "content": "what did I ask before?"}],
            })
            where = f"stream={stream}"
            assert r.status_code == 404, (
                f"{where}: `previous_response_id` 被静默忽略了（HTTP {r.status_code}）。"
                "网关不持久化 response，这个 id 必然解析不了；模型只收到一句裸问题，"
                "却会照常作答——一个没有报错的错答案。"
            )
            assert r.headers.get("content-type", "").startswith("application/json"), (
                f"{where}: 拒绝落在 {r.headers.get('content-type')!r} 上，"
                "流式客户端会把它当成一条 SSE 流去解析"
            )
            err = (r.json().get("error") or {})
            assert err.get("code") == "previous_response_not_found", (
                f"{where}: error.code 不是业界恢复逻辑认的那个值，实际 {err.get('code')!r}。"
                "真实客户端正是靠这个 code 决定「改用完整历史重发」。"
            )
            assert err.get("type") == "invalid_request_error", err
            msg = err.get("message", "")
            assert "previous_response_id" in msg, (
                f"{where}: 错误信息没有点名字段，用户无从修起: {msg!r}"
            )
            assert not CALLS, (
                f"{where}: 拒绝之前已经发生过 {len(CALLS)} 次上游调用——"
                "确定性拒绝必须是零上游的（H4）"
            )
    finally:
        _restore()


def test_requests_without_previous_response_id_are_unchanged():
    """零回归：不声明该字段时，行为与原先逐值相同（上游真的被调用）。"""
    _install()
    try:
        r = TestClient(srv.app).post("/v1/responses", json={
            "model": "gpt-4o",
            "input": [{"type": "message", "role": "user", "content": "hi"}],
        })
        assert r.status_code == 200, r.text
        assert len(CALLS) == 1, f"正常请求没有被派发到上游: {r.text}"
        assert r.json()["output"][0]["type"] == "message"
    finally:
        _restore()


def run_all():
    test_previous_response_id_is_refused_not_ignored()
    test_requests_without_previous_response_id_are_unchanged()
    print("\n✅ All previous_response_id tests passed! (2 cases)")


if __name__ == "__main__":
    run_all()