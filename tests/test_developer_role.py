"""`developer` 是 OpenAI 官方文档里 `system` 的替代名，网关把它当成了未知角色
(tests/test_developer_role.py)。

## 缺陷

Responses / Chat Completions 的 `developer` 角色是 OpenAI 为 o 系列模型引入的、
**取代 `system` 的**同义角色（SDK 的 `EasyInputMessage` 直接支持它）。网关只认识
`"system"`，于是这一个字符串把「常驻契约」变成了第三种东西——一个上游从未见过的
角色名。实测（真实 HTTP 路径，脚本化 provider，`TESTING=1`，零上游调用）：

    首轮  prompt = 'Developer: STANDING-RULE-42: always answer in French.\\n\\nUser: hello\\n\\nAssistant:'

    绑定会话次轮，上游**真正**收到的（`continuation_content`）：
          'and now?'                      ← 整条 standing rule 消失

三处后果、全部无报错：

  * **角色名不可理解** —— 五个上游都是网页聊天，它们只认 system / user / assistant。
    `Developer:` 是一个模型没有任何先验的说话人标签，指令因此失去「高于用户」的地位。
  * **续聊时整段消失** —— `continuation_content` 只把 `role=="system"` 当常驻契约上提，
    `developer` 不在其中，绑定会话第二轮起客户端自己的规则就再也到不了模型。这与
    [SESSION-CONTINUITY]（第 76 节）修的是同一个事实，只是换了一个角色名。
  * **折叠时不保护** —— `prompt_optimizer` 的头部保护区只认 `system`，超长历史折叠时
    开头的 `developer` 契约可以被折叠掉。

对 agent 客户端这正是最要命的形态：首轮还能用，第二轮起「模型突然不听话了」，
而 HTTP 200、内容看起来完全正常。

## 正确语义

"这条消息是常驻契约"是一个**判断**，不是两处各写一遍的字符串比较。`providers.base`
给出唯一实现 `is_contract_role()`，`_ROLE_LABELS`、`continuation_content`、
`prompt_optimizer` 的头部保护区、以及 `server.py` 合并工具契约时的落点判断全部读它。
`developer` 在语义上就是 system（优先级高于 user），因此渲染成 `System:` 是如实的。

运行: .venv/bin/python tests/test_developer_role.py
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
os.environ["GATEWAY_DISABLE_SINGLEFLIGHT"] = "1"
for _pk in ("QWEN", "DEEPSEEK", "DOUBAO", "KIMI", "GLM"):
    os.environ[f"{_pk}_AUTH_TOKEN"] = f"test-token-{_pk.lower()}"

import httpx  # noqa: E402

import server as srv  # noqa: E402
from providers.base import continuation_content, flatten_messages  # noqa: E402

_DEFAULT_FACTORY = srv._make_provider
SEEN: list[dict] = []

RULE = "STANDING-RULE-42: always answer in French."


class _RecordingProvider:
    """把上游**真正收到**的提示词记下来（算法与 provider 内部逐字相同）。"""

    def __init__(self, key: str):
        self.key = key

    async def chat(self, messages, model, **kw):
        cid = kw.get("conversation_id")
        prompt = continuation_content(messages) if cid else flatten_messages(messages)
        SEEN.append({"cid": cid, "prompt": prompt, "messages": messages})
        yield "ok", {}
        yield "", {"conversation_id": "dev-conv-1"}

    async def delete_conversation(self, *a, **kw):
        return None


def _install():
    SEEN.clear()
    srv._make_provider = lambda key, *a, **kw: _RecordingProvider(key)


def _restore():
    srv._make_provider = _DEFAULT_FACTORY
    srv.session_affinity_manager.clear()


def _post(path: str, payload: dict) -> httpx.Response:
    async def run():
        transport = httpx.ASGITransport(app=srv.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            return await c.post(path, json=payload)

    return asyncio.run(run())


# ---------------------------------------------------------------- 1. 提示词层


def test_developer_message_renders_as_the_contract_role():
    """**靶点**：`developer` 是 system 的同义角色，必须渲染成上游认得的 `System:`。

    修复前实测: 'Developer: STANDING-RULE-42: always answer in French.\\n\\nUser: hello…'
    """
    msgs = [
        {"role": "developer", "content": RULE},
        {"role": "user", "content": "hello"},
    ]
    prompt = flatten_messages(msgs)

    assert prompt.startswith("System: "), (
        f"`developer` 被渲染成了上游从未见过的角色名，指令失去「高于用户」的地位。"
        f"实测提示词开头: {prompt.splitlines()[0]!r}"
    )
    assert RULE in prompt, f"契约本身不能丢: {prompt!r}"


def test_continuation_keeps_the_developer_contract():
    """**靶点**：绑定会话的第二轮，上游必须仍然收到 developer 契约。

    修复前实测: continuation_content(...) == 'and now?'
    """
    msgs = [
        {"role": "developer", "content": RULE},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "bonjour"},
        {"role": "user", "content": "and now?"},
    ]
    prompt = continuation_content(msgs)

    assert RULE in prompt, (
        f"续聊增量里没有 developer 契约，模型从第二轮起就读不到常驻指令了。实测: {prompt!r}"
    )
    assert "and now?" in prompt, f"本轮增量不能丢: {prompt!r}"
    assert "hello" not in prompt, f"续聊不该重发历史轮次: {prompt!r}"


def test_a_trailing_developer_message_is_turn_content_not_contract():
    """反面：最后一条 user **之后**的 developer 消息属于本轮内容，不按契约上提。

    常驻契约的判据是「最后一条 user 之前的常驻声明」，位置错了它就变成重复发送。
    """
    msgs = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "developer", "content": "ONE-OFF-NOTE"},
    ]
    prompt = continuation_content(msgs)
    assert prompt.count("ONE-OFF-NOTE") == 1, (
        f"尾部消息既被上提又被逐条渲染，契约发了两遍: {prompt!r}"
    )


# ---------------------------------------------------------------- 2. 真实 HTTP 路径


def test_bound_session_turn_two_carries_the_developer_rule():
    """端到端：两轮绑定会话，第二轮上游收到的提示词里必须还有这条规则。

    断言在「上游真正收到的提示词」上，而不是网关自己的响应对象上——后者可以
    完全正常，而模型根本没被告知规则（第 76 节的同一纪律）。
    """
    _install()
    try:
        turn1 = _post("/v1/chat/completions", {
            "model": "gpt-4o",
            "messages": [
                {"role": "developer", "content": RULE},
                {"role": "user", "content": "hello"},
            ],
            "conversation_id": "dev-sess-1",
        })
        assert turn1.status_code == 200, turn1.text[:300]
        turn2 = _post("/v1/chat/completions", {
            "model": "gpt-4o",
            "messages": [
                {"role": "developer", "content": RULE},
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "bonjour"},
                {"role": "user", "content": "and now?"},
            ],
            "conversation_id": "dev-sess-1",
        })
        assert turn2.status_code == 200, turn2.text[:300]
    finally:
        _restore()

    assert len(SEEN) >= 2, f"两轮请求没有都到达上游: {len(SEEN)}"
    first, second = SEEN[0], SEEN[1]
    assert RULE in first["prompt"], (
        f"首轮提示词里就没有 developer 契约: {first['prompt']!r}"
    )
    assert second["cid"], f"第二轮没有被当成续聊（上游会话没接上）: {second}"
    assert RULE in second["prompt"], (
        f"绑定会话第二轮，上游**真正**收到的提示词里没有 developer 契约——"
        f"模型从这一轮起就不知道客户端定下的规则了。实测: {second['prompt']!r}"
    )


def test_responses_developer_item_reaches_upstream_as_contract():
    """Responses 协议：`input` 里带 role=developer 的 message 项同样要按契约送达。"""
    _install()
    try:
        resp = _post("/v1/responses", {
            "model": "gpt-4o",
            "input": [
                {"type": "message", "role": "developer",
                 "content": [{"type": "input_text", "text": RULE}]},
                {"type": "message", "role": "user",
                 "content": [{"type": "input_text", "text": "hi"}]},
            ],
            "stream": False,
        })
        assert resp.status_code == 200, resp.text[:300]
    finally:
        _restore()

    assert SEEN, "请求没有到达上游"
    prompt = SEEN[0]["prompt"]
    assert prompt.startswith("System: "), (
        f"Responses 的 developer 项被渲染成了未知角色: {prompt.splitlines()[0]!r}"
    )
    assert RULE in prompt, f"developer 契约没有送达: {prompt!r}"


# ---------------------------------------------------------------- 3. 折叠时保护


def test_prompt_folding_protects_a_leading_developer_message():
    """超长历史折叠时，开头的 developer 契约属于「头部保护区」，不能被折叠掉。

    **这一条是回归护栏，不是 RED-first 用例**，如实记录：实测修复前它也是绿的，
    因为头部保护区取的是 `messages[:第一个 user + 1]` 这一整段**连续前缀**，开头的
    developer 消息按位置就已经在保护区内了。折叠路径因此**不构成**一条独立的
    `developer` 缺陷，`prompt_optimizer` 改用 `is_contract_role()` 只是让判据与
    其余三个读点一致，在这个形状上逐值无差异。保留这条用例是因为它断言的性质
    （契约在折叠后仍在）真实存在，将来若有人收窄头部保护区，它会变红。
    """
    from providers.prompt_optimizer import PromptOptimizer

    body = "x" * 1200
    msgs = [
        {"role": "developer", "content": RULE},
        {"role": "user", "content": "do the thing"},
    ]
    for i in range(12):
        msgs.append({"role": "assistant", "content": f"step {i} " + body})
        msgs.append({"role": "user", "content": f"continue {i}"})

    optimizer = PromptOptimizer()
    folded, meta = optimizer.optimize_chat_messages(msgs, max_tokens=900, record=False)
    assert meta.get("applied"), f"前置条件不成立：这个规模没有触发折叠（{meta}）"
    text = "\n".join(str(m.get("content")) for m in folded)
    assert RULE in text, (
        "折叠把头部的 developer 契约当成了可折叠的历史内容——"
        "模型在超长会话里丢掉了客户端定下的规则，且没有任何报错"
    )


def test_system_role_behaviour_is_unchanged():
    """零回归：`system` 的三种既有行为逐值不变。"""
    msgs = [
        {"role": "system", "content": RULE},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "bonjour"},
        {"role": "user", "content": "and now?"},
    ]
    assert flatten_messages(msgs).startswith("System: ")
    assert RULE in continuation_content(msgs)
    assert continuation_content([
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！"},
        {"role": "user", "content": "今天天气如何"},
    ]) == "今天天气如何"


def run_all():
    test_developer_message_renders_as_the_contract_role()
    test_continuation_keeps_the_developer_contract()
    test_a_trailing_developer_message_is_turn_content_not_contract()
    test_bound_session_turn_two_carries_the_developer_rule()
    test_responses_developer_item_reaches_upstream_as_contract()
    test_prompt_folding_protects_a_leading_developer_message()
    test_system_role_behaviour_is_unchanged()
    print("\n✅ All developer-role tests passed! (7 cases)")


if __name__ == "__main__":
    run_all()
