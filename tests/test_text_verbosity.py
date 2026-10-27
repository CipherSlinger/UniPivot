"""`text.verbosity` 是真实 Codex 每条请求都发的输出约束，网关解析后从未使用 (PROTOCOL)

## 实测：真实客户端真的发，网关真的丢

用真实的 `codex` 0.145.0 驱动网关（`$CODEX_HOME/config.toml` 声明自定义 provider
指向本机），把每一条入站请求体原样记录下来，两次请求的字段集完全一致：

    /v1/responses keys = ['client_metadata', 'include', 'input', 'instructions',
                          'model', 'parallel_tool_calls', 'prompt_cache_key',
                          'reasoning', 'store', 'stream', 'text', 'tool_choice', 'tools']
    text = {"verbosity": "low"}          ← 2/2 条请求都带它

而 `grep -rn verbosity` 在整个 Python 源里**零命中**：`responses_text_format`
只看 `text.format`，`text` 里其余字段无人读取。这与 §34b/§35/§73 是同一类缺陷
（解析了、然后没人用），只是这次的字段由**真实 Codex** 每轮都发。

实测当前行为（真实 ASGI 路径，脚本化 provider，`TESTING=1`，零真实上游调用）：

    verbosity=low          status=200 promptlen=2  verbosity_in_prompt=False
        'hi'
    verbosity=high         status=200 promptlen=2  verbosity_in_prompt=False
        'hi'
    verbosity=medium       status=200 promptlen=2  verbosity_in_prompt=False
        'hi'
    absent                 status=200 promptlen=2  verbosity_in_prompt=False
        'hi'
    format=json_object     status=502 promptlen=186

注意最后一行：`text.format` 是**同一份 `text` 字典**里的兄弟字段，它工作正常
（186 字符的输出约束被注入提示词，502 只是脚本化 provider 回的不是 JSON）。所以
不是「`text` 整个没读」，而是 `text` 里**只有 `format` 一个键被读**，其余键
连同一段 `## Output Format` 约束的机会都没有。

## 为什么这不是「可选优化」

`verbosity` 是客户端对**答案长度**的明确声明，而五个上游全是网页端聊天模型——
它们的默认答复风格是「先总结、再展开、最后再来一段客套」，正是 `low` 想要压掉的
东西。Codex 每一轮都发 `low`，说明它按「简洁答复」来规划上下文预算；网关把这条
声明丢掉，模型就按自己的默认风格长篇输出，客户端随即为每一轮多付 token，并可能
在上下文预算上提前撞墙。没有任何错误可看，HTTP 200，内容正常。

这与 §58（不制造能力）不冲突：`verbosity` 不是上游的采样参数，它本来就是一条
**提示词级指令**（"be terse" / "be thorough"），和已经生效的 `response_format`、
`thinking` 指令完全同类。真正会违反 §58 的做法是把它翻译成上游没有的采样字段。

## 修复方向

一条纯函数 `responses_text_verbosity(req)`，取值白名单 `low` / `medium` / `high`，
产出与 `format_response_format_prompt` 同形的 `## Output Verbosity` 段；在
`convert_responses_to_gateway_messages` 里紧跟 `fmt_prompt` 注入（提示词只有
一个来源，两条约束并列而不互相覆盖）。无法识别的取值返回空串（失败开放，
零回归）。

去重键无需改动：`server.py` 的 Responses 调用点已经把整个 `req.text` 放进
`constraints`，所以 `verbosity` 已经在键里——这一点由用例 5 结构性钉住，因为
「修了提示词却忘了键」正是 §53 记录的缺陷。

运行: .venv/bin/python tests/test_text_verbosity.py
"""

from __future__ import annotations

import asyncio
import inspect
import os
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
    "GATEWAY_DISABLE_SINGLEFLIGHT",
):
    os.environ[_k] = "1"
for _k in ("QWEN", "DEEPSEEK", "DOUBAO", "KIMI", "GLM"):
    os.environ[f"{_k}_AUTH_TOKEN"] = "fake"

import httpx  # noqa: E402

import server as srv  # noqa: E402
from providers.base import flatten_messages  # noqa: E402

_GUARD_HITS: list[str] = []


def _guarded_factory(*args, **kwargs):
    _GUARD_HITS.append(str(args[0]) if args else "?")
    raise AssertionError("测试发出了未安装假 provider 的上游请求")


srv._make_provider = _guarded_factory

CAPTURED: dict = {}


class _Provider:
    async def chat(self, messages, model, **kw):
        CAPTURED["messages"] = list(messages)
        CAPTURED["kw"] = kw
        yield "ok", {}
        yield "", {"conversation_id": "verbosity-conv"}

    async def delete_conversation(self, *args, **kwargs):
        return True


async def _post(text_envelope) -> tuple[httpx.Response, str]:
    """驱动真实 /v1/responses 路径，返回 (响应, 上游真正收到的提示词)。"""
    srv._make_provider = lambda *a, **kw: _Provider()
    try:
        body = {
            "model": "gpt-5.5",
            "stream": False,
            "input": [{"type": "message", "role": "user", "content": "hi"}],
        }
        if text_envelope is not None:
            body["text"] = text_envelope
        transport = httpx.ASGITransport(app=srv.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            r = await c.post("/v1/responses", json=body)
        return r, flatten_messages(CAPTURED["messages"])
    finally:
        srv._make_provider = _guarded_factory


# ------------------------- 1. 真实 Codex 的形状必须真的到达模型

def test_verbosity_low_reaches_the_prompt():
    """实测修复前：`text={"verbosity":"low"}` → promptlen=2、提示词恰为 `'hi'`。

    这是真实 `codex exec` 2/2 条请求携带的形状。声明「简洁答复」却让模型按默认
    风格长篇输出，客户端为每一轮多付 token，且没有任何错误可看。
    """
    r, prompt = asyncio.run(_post({"verbosity": "low"}))
    assert r.status_code == 200, r.text
    assert prompt.strip() != "hi", (
        "`text.verbosity` 完全没有进入提示词（实测修复前 promptlen=2、内容恰为 "
        f"'hi'）——真实 Codex 每条请求都发它:\n{prompt!r}"
    )
    lowered = prompt.lower()
    assert "concision" in lowered or "concise" in lowered or "brief" in lowered, (
        f"提示词里没有「简洁」这条指令:\n{prompt}"
    )
    print("test_verbosity_low_reaches_the_prompt: PASSED")


# ------------------------- 2. 非默认取值必须产出**不同**的指令

def test_verbosity_high_differs_from_low():
    """`high` 与 `low` 若产出同一段提示词，等于声明没有被兑现。

    实测修复前两者逐字节相同（都是 `'hi'`）——「解析了」并不等于「生效了」。
    """
    _, low_prompt = asyncio.run(_post({"verbosity": "low"}))
    _, high_prompt = asyncio.run(_post({"verbosity": "high"}))
    assert low_prompt != high_prompt, (
        "`verbosity` 的 low 与 high 产出了逐字节相同的提示词（实测修复前两者"
        f"都是 'hi'）——客户端的声明没有被区分:\n{low_prompt!r}"
    )
    assert "thorough" in high_prompt.lower() or "detail" in high_prompt.lower(), (
        f"high 没有产出「详尽」指令:\n{high_prompt}"
    )
    print("test_verbosity_high_differs_from_low: PASSED")


# ------------------- 3. 与 `text.format` 并存，两条约束都要在

def test_verbosity_coexists_with_format():
    """`text` 是一个字典，两个键必须各自生效，互不覆盖。

    修复前 `format` 生效（186 字符）、`verbosity` 丢失——同一份字典里一个键被读、
    另一个被丢，正是「只加了一个 key 的 if」这种修法的典型遗留。
    """
    r, prompt = asyncio.run(_post({
        "format": {"type": "json_object"},
        "verbosity": "low",
    }))
    # 脚本化 provider 回的不是 JSON，所以这里只断言**提示词**，不看状态码：
    # §80 的结构化输出契约会正确地拒绝非 JSON 答案。
    assert "## Output Format" in prompt, f"`text.format` 的约束丢了:\n{prompt}"
    assert "oncision" in prompt or "oncise" in prompt or "rief" in prompt, (
        f"`text.verbosity` 的约束丢了——它与 format 在同一个字典里:\n{prompt}"
    )
    print("test_verbosity_coexists_with_format: PASSED")


# ------------------------------ 4. 未声明 / 无法识别时必须逐字节零回归

def test_absent_or_unknown_verbosity_is_a_no_op():
    """失败开放：没声明、或值是网关不认识的东西，提示词必须逐字节不变。

    `verbosity` 的取值是客户端定义的词表；一个未来的取值（例如 `"minimal"` 的
    表亲）不能因为网关不认识就凭空注入一条错误的长度指令。

    **这条用例在修复前的树上也是绿的**，它是一条守卫而不是 RED-first 的缺陷
    复现（§59 / §69 / §79 的纪律：修复前就绿的用例必须如实标注，不能伪装成
    回归证明）。它存在是因为最容易写错的修法是「`if verbosity:` 就注入」，
    那样一个未来取值会凭空改变模型行为。
    """
    _, absent_prompt = asyncio.run(_post(None))
    assert absent_prompt.strip() == "hi", (
        f"没声明 `text.verbosity` 时提示词被改动了（应当逐字节不变）:\n{absent_prompt!r}"
    )
    _, unknown_prompt = asyncio.run(_post({"verbosity": "banana"}))
    assert unknown_prompt == absent_prompt, (
        "无法识别的 `verbosity` 取值改变了提示词——不认识的值必须当作没声明:\n"
        f"absent={absent_prompt!r}\nunknown={unknown_prompt!r}"
    )
    _, empty_prompt = asyncio.run(_post({"verbosity": ""}))
    assert empty_prompt == absent_prompt, (
        f"空字符串 `verbosity` 改变了提示词:\n{empty_prompt!r}"
    )
    print("test_absent_or_unknown_verbosity_is_a_no_op: PASSED")


# ------- 5. 它会改变输出，所以必须已经在去重键里（§53 的判据）

def test_verbosity_is_covered_by_the_dedup_key():
    """修了提示词却忘了指纹，就是 §53 记录的缺陷重演。

    Responses 的调用点把整个 `req.text` 放进 `constraints`，`verbosity` 因此
    已经在键里。这条用例是结构性的：它钉住「恰好一处把 `req.text` 传进去重键」，
    否则某次重构把 `text` 换成 `text.format` 时，声明 low 的请求就会领到 high
    的答案，且没有任何错误可看。

    **这条用例在修复前的树上也是绿的**——`req.text` 整个进键本来就已经成立，
    修复只加了提示词那一侧。标注为守卫，是因为「修了提示词却忘了键」是 §53
    记录过的真实缺陷形态。
    """
    sig = inspect.signature(
        __import__("providers.singleflight", fromlist=["SingleFlightManager"])
        .SingleFlightManager.compute_key
    )
    assert "constraints" in sig.parameters, "compute_key 不再接受 constraints"

    src = (Path(__file__).resolve().parent.parent / "server.py").read_text(encoding="utf-8")
    assert '"text": req.text,' in src, (
        "Responses 去重键没有把整个 `req.text` 放进 constraints——`verbosity` "
        "会因此不在键里，声明 low 的请求可能领到 high 的答案"
    )
    print("test_verbosity_is_covered_by_the_dedup_key: PASSED")


def run_all():
    test_verbosity_low_reaches_the_prompt()
    test_verbosity_high_differs_from_low()
    test_verbosity_coexists_with_format()
    test_absent_or_unknown_verbosity_is_a_no_op()
    test_verbosity_is_covered_by_the_dedup_key()
    assert not _GUARD_HITS, f"有请求绕过了脚本化 provider: {_GUARD_HITS}"
    print("\n✅ All text-verbosity tests passed! (5 cases)")


if __name__ == "__main__":
    sys.exit(run_all())