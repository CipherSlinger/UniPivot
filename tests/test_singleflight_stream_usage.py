"""`stream_options.include_usage` 是**每条请求**的声明，不是每个飞行组的 (SINGLEFLIGHT)

## 判据只有一条

SingleFlight 把指纹相同的两条请求合并成一次上游调用，Leader 真打，Follower
直接领用 Leader 的**帧**。所以指纹的正确性判据只有一句：

    **指纹相同 ⇒ 两条请求的正确答案可以互换**

`compute_key` 覆盖了 provider / model / messages / tools / temperature /
thinking / stream / constraints / 会话身份 / `search`，**唯独没有
`stream_options`**。而 `usage` 帧是按这条请求自己的声明产出的：

    server.py:2807  if req.stream_options and req.stream_options.get("include_usage"):
    server.py:3079  if req.stream_options and req.stream_options.get("include_usage"):

两处都在 **Leader 的生成器**里，且都读 `req.` —— Leader 自己的声明。Follower
在 `server.py:2617` 就短路返回 `stream_follower_gen(group)`，那一路只回放
Leader 广播的帧，对 `stream_options` 一无所知。于是 `usage` 帧变成了「飞行组
的产物」，而它本该是「每条请求的产物」。

## 实测（真实 ASGI 路径，脚本化 provider，`TESTING=1`，零真实上游调用）

两个方向都错，而且错的方向相反：

    A include_usage=False | sf: leader   | usage frame: False
    B include_usage=True  | sf: follower | usage frame: False     ← 要了，没拿到
    upstream calls: 1

    --- leader=ON, follower=OFF
      first  sf=leader   usage_frame=True
      second sf=follower usage_frame=True                          ← 没要，却拿到了
    --- leader=OFF, follower=ON
      first  sf=leader   usage_frame=False
      second sf=follower usage_frame=False                         ← 要了，没拿到
    upstream calls: 2

第一种形态最要命：一个按 `usage` 做 token 计费/预算的客户端，声明了
`include_usage=True` 却拿到 `choices: []` 之外的什么都没有——它的计数器永远
是 0，且没有任何错误可看。第二种形态则相反，客户端收到一帧它没有要求的
`usage`。

## 修复方向

把 `stream_options` 放进指纹。与 §53（约束字段）、§75（会话身份）、§90
（`search`）完全同一条判据：**「要不要这帧 usage」本来就让两条请求的正确答案
不可互换**，所以它们本就不该被合并。加字段只会让键**更**具体（合并变少），
绝不会把两条不该合并的请求合并起来。`stream_options` 只存在于 OpenAI chat
协议上，另外两个调用点（Anthropic / Responses）不传，逐值保持不变。

运行: .venv/bin/python tests/test_singleflight_search.py 之后
      .venv/bin/python tests/test_singleflight_stream_usage.py
"""

from __future__ import annotations

import asyncio
import inspect
import json
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

CALLS: list[int] = []


class _Provider:
    async def chat(self, messages, model, **kw):
        CALLS.append(1)
        # 把 flight 撑开，让第二条请求有机会成为 Follower——合并要求并发。
        await asyncio.sleep(0.4)
        yield "hello world", {}
        yield "", {"conversation_id": "sf-usage-conv"}

    async def delete_conversation(self, *args, **kwargs):
        return True


_BODY = {
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "hi"}],
    "stream": True,
}

_MSGS = _BODY["messages"]


def _usage_frames(text: str) -> list[dict]:
    """取出真正带 `usage` 字段的 SSE 数据帧（不做子串匹配）。"""
    out = []
    for line in text.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[len("data: "):]
        if payload.strip() == "[DONE]":
            continue
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("usage"):
            out.append(obj)
    return out


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


def _by_role(responses: list[httpx.Response]) -> dict[str, httpx.Response]:
    out = {}
    for r in responses:
        out.setdefault(r.headers.get("x-singleflight", "?"), r)
    return out


def _usage_matches_declaration(asked_first: bool) -> None:
    """核心不变量：**每条响应的 usage 帧存在与否，等于那条请求自己的声明**。

    修复前这条不变量在两个方向上都破：飞行组里 `usage` 是 Leader 一人的声明，
    Follower 只能领到 Leader 的那一份。修复后两条声明不同的请求不再合并
    （`stream_options` 进去重键），各自向自己要的那一份，不变量成立。
    """
    asked = dict(_BODY, stream_options={"include_usage": True})
    plain = dict(_BODY)
    bodies = [asked, plain] if asked_first else [plain, asked]
    wanted = [True, False] if asked_first else [False, True]

    res = asyncio.run(_drive(bodies))
    for r in res:
        assert r.status_code == 200, r.text

    for i, (r, want) in enumerate(zip(res, wanted)):
        got = _usage_frames(r.text)
        if want:
            assert got, (
                f"第 {i} 条请求声明了 include_usage=True，却没有收到 usage 帧"
                "（实测修复前 follower 的 usage 帧 = False）——它自己的声明从未被兑现"
            )
            usage = got[0]["usage"]
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                assert key in usage, f"usage 帧缺少 {key}: {usage}"
        else:
            assert not got, (
                f"第 {i} 条请求没有声明 include_usage，却收到了 usage 帧"
                f"（实测修复前 follower 的 usage 帧 = True）: {got}"
            )

    # 诚实记录代价：`stream_options` 不同的两条请求**不再合并**。这与 §53/§75/§90
    # 是同一个取舍——`usage` 帧是请求自己的属性，合并就等于把一条请求的声明
    # 施加给另一条。声明相同的请求仍然合并（见零回归用例）。
    assert len(CALLS) == 2, (
        "两条 stream_options 不同的请求被合并成了一次上游调用"
        f"（实测 {len(CALLS)} 次）——其中一条的声明无法被兑现"
    )


# ------------------------------------------- 1. 指纹本身必须区分 stream_options

def test_stream_options_changes_the_dedup_key():
    """两条请求只差 `include_usage`，指纹必须不同。"""
    common = dict(
        provider="qwen", model="Qwen", messages=_MSGS, stream=True,
        protocol="openai_chat",
    )
    k_off = SingleFlightManager.compute_key(stream_options=None, **common)
    k_on = SingleFlightManager.compute_key(
        stream_options={"include_usage": True}, **common
    )
    assert k_off != k_on, (
        "去重指纹没有覆盖 `stream_options`：`usage` 帧是每条请求自己的声明，"
        "合并后 Follower 只能领到 Leader 的那一份。"
        f"两种取值算出同一个键 {k_off[:16]}…"
    )
    print("test_stream_options_changes_the_dedup_key: PASSED")


# ------------------- 2. 没声明 include_usage 的客户端不得收到 usage 帧

def test_follower_that_did_not_ask_gets_no_usage_frame():
    """实测修复前：Leader 要了 usage，Follower 没要，却照样收到。

    `usage` 是一个 `choices: []` 的额外帧。一个严格按契约解析的 SDK 会把它
    当作这条请求的输出，而它属于**另一条**请求的声明。
    """
    _usage_matches_declaration(asked_first=True)
    print("test_follower_that_did_not_ask_gets_no_usage_frame: PASSED")


# --------------------- 3. 声明了 include_usage 的客户端必须收到 usage 帧

def test_follower_that_asked_gets_a_usage_frame():
    """实测修复前：Leader 没要 usage，Follower 明确要了，却什么都没拿到。

    这是更严重的方向：一个按 `usage` 做 token 预算的客户端，计数器永远是 0，
    而且没有任何错误可看。与上一个用例是同一不变量的另一极，方向对调。
    """
    _usage_matches_declaration(asked_first=False)
    print("test_follower_that_asked_gets_a_usage_frame: PASSED")


# ------------------------------------------------------------ 4. 零回归

def test_identical_stream_options_still_dedup():
    """两条**真正**相同的请求仍必须合并成一次上游调用。

    一个什么都不合并的键，是比原缺陷更糟的回归（§53 的这条规则已用过三次）。
    """
    res = asyncio.run(_drive([
        dict(_BODY, stream_options={"include_usage": True}),
        dict(_BODY, stream_options={"include_usage": True}),
    ]))
    for r in res:
        assert r.status_code == 200, r.text
    assert len(CALLS) == 1, (
        f"两条完全相同的请求没有被合并（实测 {len(CALLS)} 次上游调用）——"
        "去重键变得过窄"
    )
    by_role = _by_role(res)
    assert set(by_role) == {"leader", "follower"}, f"合并角色不对: {list(by_role)}"
    # 两条都要了 usage，所以两条都必须拿到，且数值一致（指纹相同即答案可互换）
    l_usage, f_usage = _usage_frames(by_role["leader"].text), _usage_frames(
        by_role["follower"].text
    )
    assert l_usage and f_usage, "两条都声明了 include_usage，却有人没收到 usage 帧"
    assert l_usage[0]["usage"] == f_usage[0]["usage"], (
        f"同一条飞行组的两份 usage 不一致: {l_usage[0]['usage']} vs {f_usage[0]['usage']}"
    )
    print("test_identical_stream_options_still_dedup: PASSED")


# ----------------- 5. 承载 stream_options 的调用点必须真的传进去

def test_the_chat_call_site_passes_stream_options_into_the_key():
    """只加参数不改调用点，是一条能通过全部单元测试的假修复（§53/§75/§90）。

    `stream_options` 只存在于 OpenAI chat 协议上，所以断言「恰好一个调用点传了
    它」，而不是「每个调用点都传」——后者会把另外两个协议钉在一个它们没有的
    字段上。
    """
    src = (Path(__file__).resolve().parent.parent / "server.py").read_text(
        encoding="utf-8"
    )
    sig = inspect.signature(SingleFlightManager.compute_key)
    assert "stream_options" in sig.parameters, (
        "compute_key 仍然不接受 `stream_options`——调用点无从传入"
    )
    hits = re.findall(r"^\s*stream_options=req\.stream_options,\s*$", src, re.MULTILINE)
    assert len(hits) == 1, (
        "应当恰好有一处把 `req.stream_options` 传进去重键（OpenAI chat 协议），"
        f"实测 {len(hits)} 处"
    )
    print("test_the_chat_call_site_passes_stream_options_into_the_key: PASSED")


def run_all():
    test_stream_options_changes_the_dedup_key()
    test_follower_that_did_not_ask_gets_no_usage_frame()
    test_follower_that_asked_gets_a_usage_frame()
    test_identical_stream_options_still_dedup()
    test_the_chat_call_site_passes_stream_options_into_the_key()
    assert not _GUARD_HITS, f"有请求绕过了脚本化 provider: {_GUARD_HITS}"
    print("\n✅ All singleflight-stream-usage tests passed! (5 cases)")


if __name__ == "__main__":
    sys.exit(run_all())