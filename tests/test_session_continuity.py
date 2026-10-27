"""故障转移不得把原节点的上游会话 id 交给另一家厂商 (SESSION-CONTINUITY)

[TOOL-LOOP] 会话粘滞修好了一半，另一半从来没有被测过。

`resolve_affinity` 的 handoff 分支会剥离异构上游 id（OPT-007/M3），但**透明故障
转移**是第二条把请求送到别家节点的路径，它根本不经过 `resolve_affinity`：
`server.py` 把 `affinity_decision.upstream_conv_id` 一次性写进 `chat_args`，
`providers/failover.py` 的派发循环用 `call_kwargs = dict(chat_args)` 把它**逐字**
交给每一个候选。

实测（本套件第一个用例，脚本化 provider，`TESTING=1`，零真实上游调用）：
绑定在 glm 的会话，glm 主选 429 → 改派 doubao，doubao 收到的
`conversation_id` 是 `"glm-conv"`。

承重的不是那串字符，而是它在上游触发的**分支**：五家的 chat() 都是
`prompt = continuation_content(messages) if conversation_id else flatten_messages(messages)`
—— 非空 id 即「续聊」，只发最后一轮增量。于是：

  * 目标节点拿到的是一句没有前因的碎片，却按续聊作答 → 没有报错的错答案；
  * 该 id 属于另一家的会话空间，目标节点既不认识也无法续上；
  * 接着 `record_turn` 把目标节点返回的**它自己的** id 记回原绑定节点名下
    （实测 registry: `provider_key='glm', upstream_conv_id='doubao-conv'`），
    原节点恢复后每一轮都被喂一个外来 id —— 一次转移永久污染粘滞关系。

正确语义：上游会话 id 是**节点私有**的。改派意味着换了一个会话空间，本轮的
正确行为是「展平全量历史、新开上游会话」，与 handoff 分支的处置相同——那条规则
早已存在，只是故障转移是第二个入口。

运行: .venv/bin/python tests/test_session_continuity.py
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
os.environ["GATEWAY_DISABLE_SINGLEFLIGHT"] = "1"
os.environ["GATEWAY_DISABLE_RISK_LEDGER"] = "1"
os.environ["QWEN_AUTH_TOKEN"] = "fake_qwen"
os.environ["DEEPSEEK_AUTH_TOKEN"] = "fake_ds"
os.environ["DOUBAO_AUTH_TOKEN"] = "fake_doubao"
os.environ["KIMI_AUTH_TOKEN"] = "fake_kimi"
os.environ["GLM_AUTH_TOKEN"] = "fake_glm"

import httpx  # noqa: E402

import server as srv  # noqa: E402

_DEFAULT_FACTORY = srv._make_provider
SEEN: list[dict] = []

HISTORY = [
    {"role": "user", "content": "my deploy plan is prod-east-1"},
    {"role": "assistant", "content": "noted"},
    {"role": "user", "content": "what was the plan?"},
]


class _RecordingProvider:
    """记录上游**真正收到**的 conversation_id 与提示词长度。

    这两项合起来才是判据：只看 id 会漏掉「接收方忽略了它」的节点
    （qwen 会丢弃非法 id），只看提示词长度会漏掉「id 恰好无效」的巧合。
    """

    def __init__(self, key: str, fail: bool = False):
        self.key = key
        self.fail = fail

    async def chat(self, messages, model, **kw):
        import providers.base as base
        cid = kw.get("conversation_id")
        prompt = base.continuation_content(messages) if cid else base.flatten_messages(messages)
        SEEN.append({"p": self.key, "cid": cid, "prompt_len": len(prompt)})
        if self.fail:
            raise srv.ProviderError("rate limited", status=429, err_type="rate_limit_error")
        yield "ok", {}
        yield "", {"conversation_id": f"{self.key}-conv"}

    async def delete_conversation(self, *a, **kw):
        return None


def _install(failing: set[str] | None = None):
    SEEN.clear()
    failing = failing or set()
    srv._make_provider = lambda key, *a, **kw: _RecordingProvider(key, fail=key in failing)


def _restore():
    srv._make_provider = _DEFAULT_FACTORY
    srv.session_affinity_manager.clear()
    tracker = srv.failover_router.cooldown_tracker
    tracker._cooldowns.clear()
    tracker._cooldown_severity.clear()
    tracker._failure_counts.clear()


def _post(payload: dict) -> httpx.Response:
    async def run():
        transport = httpx.ASGITransport(app=srv.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            return await c.post("/v1/chat/completions", json=payload)

    return asyncio.run(run())


def test_failover_does_not_hand_a_foreign_upstream_id_to_another_vendor():
    """**靶点**：修复前 doubao 收到 glm 的 `glm-conv`，并按续聊只发增量。"""
    _install()
    try:
        # 第 1 轮：glm 健康，会话绑定到 glm 的上游会话。
        r1 = _post({"model": "glm-4-plus", "conversation_id": "sess-1",
                    "stream": False, "messages": HISTORY[:1]})
        assert r1.status_code == 200, r1.text
        assert r1.headers.get("x-session-affinity") == "fresh"
        assert srv.session_affinity_manager.has_upstream_session(
            "sess-1", requested_model=srv.resolve_model("glm-4-plus")[1])
        SEEN.clear()

        # 第 2 轮：glm 429 → 透明故障转移。同一个会话，全量历史。
        _install(failing={"glm"})
        r2 = _post({"model": "glm-4-plus", "conversation_id": "sess-1",
                    "stream": False, "messages": HISTORY})
        assert r2.status_code == 200, r2.text
        assert r2.headers.get("x-failover-to"), "本轮应当发生故障转移"
        assert len(SEEN) >= 2, f"故障转移没有发生: {SEEN}"

        primary, fallback = SEEN[0], SEEN[-1]
        assert primary["p"] == "glm"
        assert fallback["p"] != "glm"

        assert fallback["cid"] is None, (
            f"改派到 [{fallback['p']}] 时仍带着原节点的上游会话 id "
            f"{fallback['cid']!r} —— 目标节点既不认识这个 id（它属于另一家的会话"
            f"空间），又会因为非空而走「续聊」分支，只收到最后一轮增量。"
            f"实测提示词 {fallback['prompt_len']} 字符。"
        )
        assert fallback["prompt_len"] > primary["prompt_len"], (
            "改派后应当展平全量历史（新开上游会话），而不仅仅是换了个节点继续发增量："
            f"主选 {primary['prompt_len']} 字符 / 改派 {fallback['prompt_len']} 字符"
        )
    finally:
        _restore()


def _line(session_id: str, model_alias: str):
    """对话线的注册表键：会话 id + 工具契约 + **解析后**的上游模型。

    注册表按这条线的完整身份寻址（`providers/session_affinity._line_key`），
    这里用同一个函数、同一个解析结果（`resolve_model`），而不是手拼键——
    两份定义会在加入身份组成部分的那一天悄悄分叉。
    """
    from providers.session_affinity import _line_key

    return _line_key(session_id, "", srv.resolve_model(model_alias)[1])


def test_failover_does_not_poison_the_session_binding():
    """**靶点**：修复前 registry 变成 `glm -> doubao-conv`，原节点被永久喂外来 id。"""
    _install()
    try:
        _post({"model": "glm-4-plus", "conversation_id": "sess-2",
               "stream": False, "messages": HISTORY[:1]})
        bound_before = srv.session_affinity_manager._registry.get(_line("sess-2", "glm-4-plus"))
        assert bound_before is not None and bound_before.provider_key == "glm"

        _install(failing={"glm"})
        r2 = _post({"model": "glm-4-plus", "conversation_id": "sess-2",
                    "stream": False, "messages": HISTORY})
        assert r2.status_code == 200, r2.text
        serving = r2.headers.get("x-failover-to")
        assert serving, "本轮应当发生故障转移"

        entry = srv.session_affinity_manager._registry.get(_line("sess-2", "glm-4-plus"))
        assert entry is not None
        assert entry.provider_key == "glm", (
            f"绑定节点在一次短冷却改派后被改成了 {entry.provider_key!r}"
        )
        assert entry.upstream_conv_id != f"{serving}-conv", (
            f"原绑定节点 [glm] 的记录里被写进了改派节点 [{serving}] 自己的上游会话 id "
            f"{entry.upstream_conv_id!r} —— 那属于另一个会话空间。glm 恢复后，"
            "同一会话每一轮都会被喂一个 glm 不认识的外来 id，且再也不会被纠正。"
        )

        # 第 3 轮：glm 恢复。它会拿到什么？
        _install()
        r3 = _post({"model": "glm-4-plus", "conversation_id": "sess-2",
                    "stream": False, "messages": HISTORY})
        assert r3.status_code == 200, r3.text
        assert SEEN and SEEN[0]["p"] == "glm"
        assert SEEN[0]["cid"] != f"{serving}-conv", (
            f"原绑定节点 [glm] 被喂了 [{serving}] 的上游会话 id "
            f"{SEEN[0]['cid']!r}；它只可能是那一家的会话，glm 无法续上，"
            "而网关会一直这样发下去。"
        )
    finally:
        _restore()


def run_all():
    test_failover_does_not_hand_a_foreign_upstream_id_to_another_vendor()
    test_failover_does_not_poison_the_session_binding()
    print("\n✅ All session-continuity tests passed! (2 cases)")


if __name__ == "__main__":
    run_all()