"""去重指纹必须覆盖 `search`——它改变上游请求体，也就改变正确答案 (SINGLEFLIGHT)

## 判据只有一条

SingleFlight 把指纹相同的两条请求合并成一次上游调用，Leader 真打、Follower
直接领用 Leader 的答案。所以指纹的正确性判据只有一句：

    **指纹相同 ⇒ 两条请求的正确答案可以互换**

`compute_key` 覆盖了 provider / model / messages / tools / temperature /
thinking / stream / constraints / 会话身份，**唯独没有 `search`**，而
`search` 是会被送进上游请求体的：四个 provider 都读它

    qwen    `"deep_search": "normal" if search else None`
    deepseek `"search_enabled": search`
    glm     `"is_networking": search`
    kimi    `"use_search": search`

——开不开联网检索，同一条问题的正确答案当然不同。

## 实测（真实 ASGI 路径，脚本化 provider，`TESTING=1`，零真实上游调用）

两条请求逐字相同，只有 `search` 一个为 True、一个为 False，并发发出：

    statuses: 200 200
    upstream calls: 1 [{'search': True}]
    A search=True  -> SEARCH-ON-ANSWER | sf: leader
    B search=False -> SEARCH-ON-ANSWER | sf: follower

B 明确声明了不要检索，却领到了 A 的检索答案——**上游只被调用一次**，B 的
`search=False` 从未被任何上游看到。一个没有报错的错答案，与 §53（约束字段）、
§75（会话身份）是同一类缺陷的第三次出现。

## 修复方向

把 `search` 放进指纹。与 §53/§75 同理：加字段只会让键**更**具体（合并变少），
绝不会把两条不该合并的请求合并起来。不传 `search` 的调用点（Anthropic /
Responses 两个协议没有这个字段）逐值保持不变。

运行: .venv/bin/python tests/test_singleflight_session_identity.py 之后
      .venv/bin/python tests/test_singleflight_search.py
"""

from __future__ import annotations

import asyncio
import inspect
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["TESTING"] = "1"
for _k in (
    "GATEWAY_DISABLE_CREDENTIAL_WATCH",
    "GATEWAY_DISABLE_AUTO_REFRESH",
    "GATEWAY_DISABLE_PROMPT_FOLDING",
    "GATEWAY_DISABLE_CONVERSATION_GC",
    "GATEWAY_DISABLE_ADAPTIVE_PACING",
    "GATEWAY_DISABLE_PROBING",
    "GATEWAY_DISABLE_BAN_INFERENCE",
    "GATEWAY_DISABLE_STREAM_COMPLETION_CHECK",
    "GATEWAY_DISABLE_LONG_AVOIDANCE_FAST_FAIL",
    "GATEWAY_DISABLE_RISK_LEDGER",
):
    os.environ[_k] = "1"
for _k in ("QWEN", "DEEPSEEK", "DOUBAO", "KIMI", "GLM"):
    os.environ[f"{_k}_AUTH_TOKEN"] = "fake"

import httpx  # noqa: E402

import server as srv  # noqa: E402
from providers.singleflight import SingleFlightManager  # noqa: E402

_GUARD_HITS: list[str] = []


def _guarded_factory(*args, **kwargs):
    _GUARD_HITS.append(str(args[0]) if args else "?")
    raise AssertionError("测试发出了未安装假 provider 的上游请求")


srv._make_provider = _guarded_factory

CALLS: list[dict] = []


class _Provider:
    """把 `search` 的取值直接写进答案，答案本身就证明上游收到了什么。"""

    async def chat(self, messages, model, **kw):
        CALLS.append({"search": kw.get("search")})
        # 把 flight 撑开，让第二条请求有机会成为 Follower——合并要求并发。
        await asyncio.sleep(0.35)
        yield ("SEARCH-ON-ANSWER" if kw.get("search") else "NO-SEARCH-ANSWER"), {}
        yield "", {"conversation_id": "sf-search-conv"}

    async def delete_conversation(self, *args, **kwargs):
        return True


_BODY = {
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "what is today's date?"}],
    "stream": False,
}

_MSGS = _BODY["messages"]


async def _drive(bodies: list[dict]) -> list[httpx.Response]:
    CALLS.clear()
    srv._make_provider = lambda *a, **kw: _Provider()
    try:
        transport = httpx.ASGITransport(app=srv.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            return list(
                await asyncio.gather(
                    *[c.post("/v1/chat/completions", json=b) for b in bodies]
                )
            )
    finally:
        srv._make_provider = _guarded_factory


# ------------------------------------------- 1. 指纹本身必须区分 search

def test_search_changes_the_dedup_key():
    """两条请求只差 `search`，指纹必须不同。"""
    common = dict(
        provider="qwen", model="Qwen", messages=_MSGS, stream=False,
        protocol="openai_chat",
    )
    k_off = SingleFlightManager.compute_key(search=False, **common)
    k_on = SingleFlightManager.compute_key(search=True, **common)
    assert k_off != k_on, (
        "去重指纹没有覆盖 `search`：声明不要联网检索的请求会领到检索答案。"
        f"两种取值算出同一个键 {k_off[:16]}…"
    )
    print("test_search_changes_the_dedup_key: PASSED")


# --------------------------- 2. 真实路径：只差 search 的请求不得被合并

def test_search_differing_requests_are_not_merged():
    """实测修复前：上游只被调用一次，`search=False` 的那条领到了检索答案。"""
    r_on, r_off = asyncio.run(
        _drive([dict(_BODY, search=True), dict(_BODY, search=False)])
    )
    assert r_on.status_code == 200 and r_off.status_code == 200, (
        r_on.text + r_off.text
    )
    body_on = r_on.json()["choices"][0]["message"]["content"]
    body_off = r_off.json()["choices"][0]["message"]["content"]

    assert len(CALLS) == 2, (
        "两条只差 `search` 的请求被合并成了一次上游调用（实测 "
        f"{len(CALLS)} 次，上游收到 {CALLS}）——其中一条的声明从未到达上游"
    )
    assert {c["search"] for c in CALLS} == {True, False}, (
        f"上游收到的 search 取值不完整: {CALLS}"
    )
    assert "SEARCH-ON-ANSWER" in body_on and "NO-SEARCH-ANSWER" in body_off, (
        "答案与各自的声明不符: "
        f"search=True -> {body_on!r}, search=False -> {body_off!r}"
    )
    print("test_search_differing_requests_are_not_merged: PASSED")


# ------------------------------------------------------------ 3. 零回归

def test_identical_search_requests_still_dedup():
    """两条**真正**相同的请求仍必须合并成一次上游调用。

    一个什么都不合并的键，是比原缺陷更糟的回归（§53 的这条规则已用过两次）。
    """
    r1, r2 = asyncio.run(_drive([dict(_BODY, search=True), dict(_BODY, search=True)]))
    assert r1.status_code == r2.status_code == 200, r1.text + r2.text
    assert len(CALLS) == 1, (
        f"两条完全相同的请求没有被合并（实测 {len(CALLS)} 次上游调用）——"
        "去重键变得过窄"
    )
    roles = {r1.headers.get("x-singleflight"), r2.headers.get("x-singleflight")}
    assert roles == {"leader", "follower"}, f"合并角色不对: {roles}"
    print("test_identical_search_requests_still_dedup: PASSED")


# --------------------------------- 4. 承载 search 的调用点必须真的传进去

def test_the_chat_call_site_passes_search_into_the_key():
    """只加参数不改调用点，是一条能通过全部单元测试的假修复（§53/§75）。

    `search` 只存在于 OpenAI chat 协议上，所以断言「恰好一个调用点传了它」，
    而不是「每个调用点都传」——后者会把另外两个协议钉在一个它们没有的字段上。
    """
    src = (Path(__file__).resolve().parent.parent / "server.py").read_text(encoding="utf-8")
    sig = inspect.signature(SingleFlightManager.compute_key)
    assert "search" in sig.parameters, (
        "compute_key 仍然不接受 `search`——调用点无从传入"
    )
    hits = re.findall(r"^\s*search=req\.search,\s*$", src, re.MULTILINE)
    assert len(hits) == 1, (
        "应当恰好有一处把 `req.search` 传进去重键（OpenAI chat 协议），"
        f"实测 {len(hits)} 处"
    )
    print("test_the_chat_call_site_passes_search_into_the_key: PASSED")


def run_all():
    test_search_changes_the_dedup_key()
    test_search_differing_requests_are_not_merged()
    test_identical_search_requests_still_dedup()
    test_the_chat_call_site_passes_search_into_the_key()
    assert not _GUARD_HITS, f"有请求绕过了脚本化 provider: {_GUARD_HITS}"
    print("\n✅ All singleflight-search tests passed! (4 cases)")


if __name__ == "__main__":
    sys.exit(run_all())