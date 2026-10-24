"""去重指纹必须覆盖**会话身份** (SINGLEFLIGHT / SESSION-CONTINUITY)

## 缺陷：前缀不同的两条请求，指纹相同

SingleFlight 的判据只有一条 —— 指纹相同 ⇒ 两条请求的正确答案可以互换。
`compute_key` 覆盖了 provider / model / messages / tools / temperature /
thinking / stream / constraints，**唯独没有会话身份**。而会话身份决定的是
*哪一段历史被算作「前文」*：

  * 已绑定上游会话的请求，provider 走续聊分支，只发本轮增量，前文在上游；
  * 全新会话的请求，provider 展平全量历史，前文在提示词里。

两条请求的 `messages` 可以逐字相同（同样的指令、同样的工具声明），而正确答案
完全不同：前者是「接着聊」，后者是「从头答」。

实测（本套件第一个用例，真实 ASGI 路径，脚本化 provider，`TESTING=1`）：
会话 A 已绑定上游 `up-1`，会话 B 全新，两者发出的 `messages` 逐字相同 →
指纹相同 → B 成为 Follower，**领到 A 的续聊答案**（`CONT-ANSWER`），而上游
只被调用了一次、只创建了一个会话；B 的注册表条目始终为空，可它的响应头却写着
`x-session-affinity: bound`。

对 agent 客户端这是最常见的形态：Codex / Claude Code 派出的子 agent 与父 agent
发同样的指令文本、声明同样的工具，指纹自然相同——子 agent 被喂了父 agent 的
续聊答案，而它自己的会话从未被创建。一个没有报错的错答案。

## 修复方向：把会话身份放进指纹

与 S3 同理，加字段只会让键**更**具体（合并变少），绝不会把两条不该合并的请求
合并起来。没有会话身份的请求（匿名一次性调用）保持逐值不变——它们之间确实
可以互换。

运行: .venv/bin/python tests/test_singleflight_constraints.py 之后
      .venv/bin/python tests/test_session_continuity.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["TESTING"] = "1"
os.environ["GATEWAY_DISABLE_AUTO_REFRESH"] = "1"
os.environ["GATEWAY_DISABLE_CREDENTIAL_WATCH"] = "1"
os.environ["GATEWAY_DISABLE_PROBING"] = "1"
os.environ["GATEWAY_DISABLE_PACING"] = "1"
os.environ["GATEWAY_DISABLE_RISK_LEDGER"] = "1"
os.environ["QWEN_AUTH_TOKEN"] = "fake_qwen"
os.environ["DEEPSEEK_AUTH_TOKEN"] = "fake_ds"
os.environ["DOUBAO_AUTH_TOKEN"] = "fake_doubao"
os.environ["KIMI_AUTH_TOKEN"] = "fake_kimi"
os.environ["GLM_AUTH_TOKEN"] = "fake_glm"

import httpx  # noqa: E402

import server as srv  # noqa: E402

_DEFAULT_FACTORY = srv._make_provider
UPSTREAM: list[dict] = []

MSGS = [{"role": "user", "content": "deploy to prod"}]


class _Provider:
    def __init__(self, key: str):
        self.key = key

    async def chat(self, messages, model, **kw):
        from providers.base import continuation_content, flatten_messages

        cid = kw.get("conversation_id")
        prompt = continuation_content(messages) if cid else flatten_messages(messages)
        UPSTREAM.append({"cid": cid, "prompt": prompt})
        await asyncio.sleep(0.4)  # 让两条请求真正重叠
        yield ("CONT" if cid else "FRESH"), {}
        yield "", {"conversation_id": cid or "up-new"}

    async def delete_conversation(self, *a, **kw):
        return None


def _install():
    UPSTREAM.clear()
    srv._make_provider = lambda key, *a, **kw: _Provider(key)


def _restore():
    srv._make_provider = _DEFAULT_FACTORY
    srv.session_affinity_manager.clear()


def test_bound_and_fresh_sessions_are_not_deduplicated():
    """**靶点**：修复前 B（全新会话）领到 A（已绑定）的续聊答案，上游只被调一次。"""
    _install()
    try:
        async def run():
            transport = httpx.ASGITransport(app=srv.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
                body = {"model": "qwen", "stream": False, "messages": MSGS}
                # 第 1 轮：会话 A 建立并绑定到上游会话。
                r1 = await c.post("/v1/chat/completions", json={**body, "conversation_id": "A"})
                assert r1.status_code == 200, r1.text
                UPSTREAM.clear()

                async def go(conv):
                    return await c.post("/v1/chat/completions",
                                        json={**body, "conversation_id": conv})

                return await asyncio.gather(go("A"), go("B"))

        ra, rb = asyncio.run(run())
    finally:
        _restore()

    assert ra.status_code == 200 and rb.status_code == 200
    a_body = ra.json()["choices"][0]["message"]["content"]
    b_body = rb.json()["choices"][0]["message"]["content"]

    assert len(UPSTREAM) == 2, (
        f"已绑定会话 A 与全新会话 B 的请求被合并成了一次上游调用（实测 {len(UPSTREAM)} 次）——"
        "两条请求的 messages 相同，但正确答案不可互换：A 是续聊，B 是全新对话。"
        f"B 领到的是 {b_body!r}，而它自己的会话从未被创建。"
    )
    assert any(u["cid"] is None for u in UPSTREAM), (
        f"全新会话 B 没有被展平成一次完整对话：上游收到的都是续聊请求 {UPSTREAM}"
    )
    assert a_body == "CONT", f"A 应当仍然是续聊答案，实测 {a_body!r}"
    assert b_body == "FRESH", (
        f"全新会话 B 领到了续聊答案 {b_body!r}——它问的是一个它从未参与过的会话"
    )


def test_anonymous_callers_are_still_deduplicated():
    """零回归：都没有会话身份的请求之间确实可以互换，必须仍然合并。"""
    _install()
    try:
        async def run():
            transport = httpx.ASGITransport(app=srv.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
                return await c.post("/v1/chat/completions",
                                    json={"model": "qwen", "stream": False, "messages": MSGS})

        asyncio.run(run())
        UPSTREAM.clear()

        async def run2():
            transport = httpx.ASGITransport(app=srv.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
                return await asyncio.gather(*[
                    c.post("/v1/chat/completions",
                           json={"model": "qwen", "stream": False, "messages": MSGS})
                    for _ in range(2)
                ])

        rs = asyncio.run(run2())
    finally:
        _restore()

    assert all(r.status_code == 200 for r in rs)
    assert len(UPSTREAM) == 1, (
        f"两条完全没有会话身份、且逐字相同的请求没有被合并（实测 {len(UPSTREAM)} 次上游调用）——"
        "把键做得太具体，使本来可以互换的请求也不再共享，是比原缺陷更糟的回归"
    )


def run_all():
    test_bound_and_fresh_sessions_are_not_deduplicated()
    test_anonymous_callers_are_still_deduplicated()
    print("\n✅ All session-identity dedup tests passed! (2 cases)")


if __name__ == "__main__":
    run_all()