"""旧版函数 API 被解析后从未兑现 (TOOL-LOOP)

## 症状：三条请求形状，工具全都不存在

OpenAI 的函数调用有两代写法。现行的是 `tools` / `tool_choice` / `role:"tool"`，
网关完整支持；上一代是

    `functions`      —— 工具声明（少了 `{"type": "function", "function": …}` 外壳）
    `function_call`  —— 强制/禁止调用的指令（"auto" / "none" / {"name": …}）
    `role:"function"`—— 工具结果的回放消息
    assistant.function_call —— 工具调用本身的回放

`ChatCompletionRequest` 用 `extra="allow"` 把它们收进 `model_extra` 之后，
**没有任何代码读过**。实测（真实 ASGI 路径，脚本化 provider，`TESTING=1`）：

    1 functions+function_call      status=200 promptlen=2
        tools_section: False | mentions Bash: False
        prompt: 'go'
    2 role:function history        status=200 promptlen=41
        tools_section: False | mentions Bash: False
        prompt: 'User: go\n\nFunction: a.py b.py\n\nAssistant:'
    3 role:function only           status=200 promptlen=41
        tools_section: False | mentions Bash: False
        prompt: 'User: go\n\nFunction: a.py b.py\n\nAssistant:'

三件事同时坏掉：

(a) **声明消失**——提示词里没有 `## Tools Available`，模型从未被告知它有一个
    叫 `Bash` 的工具，因此永远也不会调用它。这是「工具没跑起来」最彻底的一种。
(b) **结果变成第三人称独白**——`role:"function"` 落进 `_ROLE_LABELS` 的兜底
    `role.capitalize()`，渲染成 `Function: a.py b.py`；工具名 `<name>` 被丢弃，
    契约第 5 条承诺的 `[Tool Result for <id>]` 标记一次都没出现。模型看到一段
    不知是谁说的、关于什么的文本。
(c) **调用本身消失**——assistant 的 `function_call` 没被 `_serialize_tool_calls`
    读取（它只认 `tool_calls`），于是历史里既没有调用也没有结果，只有一段
    无主的输出。

三条都是「解析了、然后没人用」（§34b/§35/§73 反复记录的那一类），且都没有
任何错误可看：HTTP 200，正常内容。

## 为什么这不是「废弃协议，可以不支持」

`functions` 至今仍出现在 OpenAI SDK 的兼容层、若干代理网关，以及大量手写的
接入脚本里。它被接受（200）却不生效，正是本项目最反对的那种失败形态：客户端
**无法**从响应中看出自己的工具从未存在过。

## 修复方向

归一化，不是第二套实现。`ChatCompletionRequest` 的 validator 把三个旧字段
翻译成现行字段，之后整条流水线（提示词、白名单、解析、结果标记）读的都是同一份
`req.tools` / `req.tool_choice`，无需任何一处特殊判断：

    functions            -> tools           （补 `{"type":"function","function":…}` 外壳）
    function_call        -> tool_choice     （"auto"/"none"/{"name"} 三者同义）
    assistant.function_call -> assistant.tool_calls
    role:"function"      -> role:"tool"     （`name` 保留，供无 id 时回退）

现行写法原样优先：客户端同时发了 `tools` 和 `functions` 时以 `tools` 为准（两代
声明同时出现说明它是有意为之，现行写法是更新的意图）。`tools` 缺席时才由
`functions` 补齐——这正是「只加不减」的安全方向。

运行: .venv/bin/python tests/test_singleflight_stream_usage.py 之后
      .venv/bin/python tests/test_legacy_function_api.py
"""

from __future__ import annotations

import asyncio
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
        yield "", {"conversation_id": "legacy-conv"}

    async def delete_conversation(self, *args, **kwargs):
        return True


_LEGACY_FUNCS = [
    {
        "name": "Bash",
        "description": "Run a shell command",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    }
]


async def _post(body: dict) -> tuple[httpx.Response, str]:
    """驱动真实 HTTP 路径，返回 (响应, 上游真正收到的提示词)。"""
    srv._make_provider = lambda *a, **kw: _Provider()
    try:
        transport = httpx.ASGITransport(app=srv.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            r = await c.post("/v1/chat/completions", json={"model": "gpt-4o", **body})
        prompt = flatten_messages(CAPTURED["messages"])
        return r, prompt
    finally:
        srv._make_provider = _guarded_factory


# ------------------------------- 1. `functions` 声明必须真的到达模型

def test_legacy_functions_reach_the_prompt():
    """实测修复前：`promptlen=2`、`tools_section: False`、`mentions Bash: False`。

    模型从未被告知它有一个叫 Bash 的工具，所以它永远不会调用它——这是「工具
    没跑起来」里最彻底的一种，而且 HTTP 200，没有任何错误可看。
    """
    r, prompt = asyncio.run(_post({
        "messages": [{"role": "user", "content": "list files"}],
        "functions": _LEGACY_FUNCS,
        "function_call": "auto",
    }))
    assert r.status_code == 200, r.text
    assert "## Tools Available" in prompt, (
        "旧版 `functions` 声明没有进入提示词（实测修复前 promptlen=2、"
        f"tools_section=False），模型不知道这个工具存在:\n{prompt!r}"
    )
    assert "Bash" in prompt, f"提示词里没有工具名 Bash:\n{prompt}"
    assert "command" in prompt, (
        f"工具参数 schema 丢失，模型无从知道该传什么:\n{prompt}"
    )
    print("test_legacy_functions_reach_the_prompt: PASSED")


# ------------------- 2. `function_call` 是 tool_choice 的同义词

def test_legacy_function_call_maps_to_tool_choice():
    """`function_call:"none"` 必须像 `tool_choice:"none"` 一样禁止调用。

    实测修复前它被 `extra="allow"` 收着，白名单仍是「全部声明工具」，于是
    声明了 `none` 的请求照样可能拿到工具调用——§34b 的缺陷在旧字段上重演。
    """
    r, prompt = asyncio.run(_post({
        "messages": [{"role": "user", "content": "just answer"}],
        "functions": _LEGACY_FUNCS,
        "function_call": "none",
    }))
    assert r.status_code == 200, r.text
    directive = [ln for ln in prompt.splitlines() if "Do not call" in ln or "不要调用" in ln]
    assert directive, (
        "旧版 `function_call:\"none\"` 没有变成任何禁止调用的指令——"
        f"它与 `tool_choice:\"none\"` 同义，必须同样生效:\n{prompt}"
    )
    print("test_legacy_function_call_maps_to_tool_choice: PASSED")


# -------- 3. `role:"function"` 的结果必须带上调用 id 与契约标记

def test_legacy_function_role_renders_as_a_tool_result():
    """实测修复前：`User: go\\n\\nFunction: a.py b.py\\n\\nAssistant:`

    工具名被丢弃、渲染成第三人称独白、契约第 5 条承诺的
    `[Tool Result for <id>]` 一次都没出现。
    """
    r, prompt = asyncio.run(_post({
        "messages": [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": None,
                "function_call": {"name": "Bash", "arguments": '{"command":"ls"}'},
            },
            {"role": "function", "name": "Bash", "content": "a.py b.py"},
        ],
        "functions": _LEGACY_FUNCS,
    }))
    assert r.status_code == 200, r.text
    assert "Function:" not in prompt, (
        "`role:\"function\"` 仍然被渲染成第三人称独白 `Function:`（实测修复前"
        f"就是这样），而不是工具结果:\n{prompt}"
    )
    assert "[Tool Result" in prompt, (
        f"工具结果没有使用契约承诺的标记:\n{prompt}"
    )
    assert "a.py b.py" in prompt, f"工具结果的内容丢失了:\n{prompt}"
    print("test_legacy_function_role_renders_as_a_tool_result: PASSED")


# ------------- 4. assistant.function_call 回放必须与 tool_calls 同形

def test_legacy_assistant_function_call_replays_as_a_tool_call():
    """实测修复前：历史里既没有调用也没有结果，只有一段无主的输出。

    这与 §43/§45 是同一个缺陷（assistant 的工具调用整段消失），只是字段名
    换成了上一代。
    """
    r, prompt = asyncio.run(_post({
        "messages": [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": None,
                "function_call": {"name": "Bash", "arguments": '{"command":"ls"}'},
            },
            {"role": "function", "name": "Bash", "content": "a.py b.py"},
        ],
        "functions": _LEGACY_FUNCS,
    }))
    assert r.status_code == 200, r.text
    assert '"name": "Bash"' in prompt or '"name":"Bash"' in prompt, (
        f"assistant 的 `function_call` 没有作为调用进入历史:\n{prompt}"
    )
    assert '"command"' in prompt, (
        f"调用的参数丢失，模型看不到自己当初传了什么:\n{prompt}"
    )
    print("test_legacy_assistant_function_call_replays_as_a_tool_call: PASSED")


# ------- 5. 端到端：旧版声明必须走通整条工具循环，而不只是提示词好看

def test_legacy_declaration_closes_the_whole_loop():
    """提示词里有声明还不够——模型真的调用之后，客户端必须拿回 `tool_calls`。

    这条用例的价值在于它走的是**同一条**流水线：没有为旧字段新增任何解析
    分支，所以提示词对了、白名单对了、调用被接收、客户端收到现行形状，四件事
    一次全部成立。若当初选了「另写一条解析路径」的修法，这里会立刻暴露。
    """
    class _CallingProvider:
        async def chat(self, messages, model, **kw):
            yield (
                "I will list the files.\n\n"
                '```tool_call\n{"name": "Bash", "arguments": {"command": "ls"}}\n```'
            ), {}
            yield "", {"conversation_id": "legacy-loop-conv"}

        async def delete_conversation(self, *args, **kwargs):
            return True

    async def _drive():
        srv._make_provider = lambda *a, **kw: _CallingProvider()
        try:
            transport = httpx.ASGITransport(app=srv.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
                return await c.post("/v1/chat/completions", json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "list files"}],
                    "functions": _LEGACY_FUNCS,
                    "function_call": "auto",
                })
        finally:
            srv._make_provider = _guarded_factory

    r = asyncio.run(_drive())
    assert r.status_code == 200, r.text
    msg = r.json()["choices"][0]["message"]
    calls = msg.get("tool_calls") or []
    assert calls, (
        "旧版 `functions` 声明虽已进入提示词，模型也写出了 tool_call 块，但客户端"
        f"没有收到工具调用——白名单或解析没有看到这份声明。实测 message={msg}"
    )
    assert calls[0]["function"]["name"] == "Bash", calls
    print("test_legacy_declaration_closes_the_whole_loop: PASSED")


# ------------------------------------------------------------ 6. 零回归

def test_modern_tools_win_and_are_unchanged():
    """现行写法必须逐值不变，且与旧字段同时出现时以现行写法为准。

    两代声明同时出现说明客户端是有意为之，现行写法是更新的意图。把旧字段
    悄悄合并进来会让一个现代客户端的工具集莫名变多。
    """
    modern = [{
        "type": "function",
        "function": {
            "name": "Read",
            "description": "Read a file",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
        },
    }]
    r, prompt = asyncio.run(_post({
        "messages": [{"role": "user", "content": "read it"}],
        "tools": modern,
        "functions": _LEGACY_FUNCS,
    }))
    assert r.status_code == 200, r.text
    assert "Read" in prompt, f"现行 `tools` 声明丢了:\n{prompt}"
    assert "Bash" not in prompt, (
        "同时声明 `tools` 与 `functions` 时，旧字段被合并进来了——现行写法"
        f"应当优先，否则现代客户端的工具集凭空变多:\n{prompt}"
    )
    print("test_modern_tools_win_and_are_unchanged: PASSED")


def run_all():
    test_legacy_functions_reach_the_prompt()
    test_legacy_function_call_maps_to_tool_choice()
    test_legacy_function_role_renders_as_a_tool_result()
    test_legacy_assistant_function_call_replays_as_a_tool_call()
    test_legacy_declaration_closes_the_whole_loop()
    test_modern_tools_win_and_are_unchanged()
    assert not _GUARD_HITS, f"有请求绕过了脚本化 provider: {_GUARD_HITS}"
    print("\n✅ All legacy-function-api tests passed! (6 cases)")


if __name__ == "__main__":
    sys.exit(run_all())