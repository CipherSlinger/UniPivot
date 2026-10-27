"""工具结果必须带上调用 id，否则「结果属于哪次调用」在提示词里不存在 (TOOL-LOOP)

三个协议各自把客户端回传的工具历史渲染成上游提示词。同一份契约——注入的工具段
明确写着

    5. When you receive a `[Tool Result for <id>]` block, continue the task
       using that result.

——但只有两个协议兑现了它：

    Anthropic   : `[Tool Result for toolu_1]:` / `[Tool Error for toolu_2]:`
    Responses   : `[Tool Result for c1]:`
    OpenAI chat : `Tool (Read): print('a')`          ← 没有 id

实测（真实 `/v1/chat/completions`，脚本化上游，`TESTING=1`，零真实上游调用）：
一次 `Read(/tmp/a.py)` + `Read(/tmp/b.py)` 的并行调用，两条结果在提示词里是

    Assistant: Reading both.
    Tool (Read): print('a')
    Tool (Read): print('b')

调用侧（assistant 的 `tool_calls`）是带 id 的，结果侧不带。两个后果：

* **配对丢失**：并行调用同一个工具时，两条结果在提示词里**逐字节同形**，模型
  无从判断哪条来自 `call_a`、哪条来自 `call_b`。§68 已经让「一个回合两次调用」
  能完整送达客户端；把结果送回来时又是同一个工具的两次调用，循环在这里断掉。
* **契约自相矛盾**：工具段要求模型「收到 `[Tool Result for <id>]` 后继续」，而
  它收到的历史里从来没有出现过这个形状。§32/§76 记录过同一类缺陷——声明的能力
  与真正兑现的行为不是一件事。

判据是 §43/§62/§88 的同一条纪律：**一个事实，一份渲染**。三个协议都必须让结果
带上它的 `tool_call_id`，并且都用契约里承诺的那个标记。

运行: .venv/bin/python tests/test_responses_tool_output_rendering.py 之外的
      .venv/bin/python tests/test_tool_result_pairing.py
"""

from __future__ import annotations

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
    "GATEWAY_DISABLE_SINGLEFLIGHT",
    "GATEWAY_DISABLE_ADAPTIVE_PACING",
    "GATEWAY_DISABLE_PROBING",
    "GATEWAY_DISABLE_BAN_INFERENCE",
    "GATEWAY_DISABLE_STREAM_COMPLETION_CHECK",
):
    os.environ[_k] = "1"
for _k in ("QWEN", "DEEPSEEK", "DOUBAO", "KIMI", "GLM"):
    os.environ[f"{_k}_AUTH_TOKEN"] = "fake"

from fastapi.testclient import TestClient  # noqa: E402

import server as srv  # noqa: E402
from providers.base import flatten_messages  # noqa: E402

_CAPTURED: dict = {}


class ScriptedProvider:
    async def chat(self, messages, model, **kw):
        _CAPTURED.clear()
        _CAPTURED.update(messages=messages, model=model, **kw)
        yield "hello", {}
        yield "", {"conversation_id": "pair-conv"}

    async def delete_conversation(self, *args, **kwargs):
        return True


_GUARD_HITS: list[str] = []


def _guarded_factory(*args, **kwargs):
    _GUARD_HITS.append(str(args[0]) if args else "?")
    raise AssertionError("测试发出了未安装假 provider 的上游请求")


srv._make_provider = _guarded_factory


def _prompt_for(body: dict, path: str = "/v1/chat/completions") -> str:
    """走真实 HTTP 路径，返回上游**实际收到**的提示词。"""
    _CAPTURED.clear()
    srv._make_provider = lambda *a, **kw: ScriptedProvider()
    try:
        r = TestClient(srv.app).post(path, json=body)
    finally:
        srv._make_provider = _guarded_factory
    assert r.status_code == 200, r.text
    return flatten_messages(_CAPTURED.get("messages", []))


_TOOLS = [{"type": "function", "function": {
    "name": "Read", "description": "read a file",
    "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}}}}}]

#: 一个回合里两次调用**同一个**工具——这是配对问题唯一能被看见的形状。
_PARALLEL_HISTORY = [
    {"role": "user", "content": "read both files"},
    {"role": "assistant", "content": "Reading both.", "tool_calls": [
        {"id": "call_a", "type": "function",
         "function": {"name": "Read", "arguments": '{"file_path":"/tmp/a.py"}'}},
        {"id": "call_b", "type": "function",
         "function": {"name": "Read", "arguments": '{"file_path":"/tmp/b.py"}'}},
    ]},
    # 两次结果的内容**相同**——`Bash` 跑两条都打印 ok 的命令、或两个空文件，都是
    # 极常见的形状。内容相同时，没有 id 的两条结果行在提示词里逐字节相同。
    {"role": "tool", "tool_call_id": "call_a", "content": "ok"},
    {"role": "tool", "tool_call_id": "call_b", "content": "ok"},
]


# ------------------------------------------------ 1. 结果行必须带调用 id

def test_a_tool_result_carries_its_call_id():
    """实测修复前：`Tool (Read): print('a')` —— 提示词里没有 `call_a` 的结果行。

    调用侧是带 id 的（assistant 的 `tool_calls` 里有 `call_a` / `call_b`），所以
    模型看得到自己发起过哪两次调用；结果侧不带 id 时，它无法把输出配回去。
    """
    p = _prompt_for({"model": "gpt-4o", "messages": _PARALLEL_HISTORY,
                     "tools": list(_TOOLS)})
    assert "Tool Result for call_a" in p, (
        "工具结果没有带上它的 tool_call_id——模型无法把输出配回发起它的那次调用。"
        f"实测结果行仍是 'Tool (Read): ...' 形状。提示词尾部:\n{p[-260:]!r}"
    )
    assert "Tool Result for call_b" in p, (
        f"第二次调用的结果同样丢了 id。提示词尾部:\n{p[-260:]!r}"
    )
    print("test_a_tool_result_carries_its_call_id: PASSED")


# ---------------------------- 2. 同一工具的两条结果必须可区分（配对问题本身）

def test_two_results_from_the_same_tool_are_distinguishable():
    """并行调用同一个工具时，两条结果**逐字节同形**，配对信息为零。

    §68 保证了「一个回合两次调用」能完整送达客户端；客户端把两次结果送回来时，
    如果渲染把 id 抹掉，模型面前就是两段无法归属的输出。这不是措辞问题，是
    agent 循环在第二次往返时断掉的直接原因。
    """
    p = _prompt_for({"model": "gpt-4o", "messages": _PARALLEL_HISTORY,
                     "tools": list(_TOOLS)})
    result_lines = [ln for ln in p.split("\n") if "ok" in ln]
    assert len(result_lines) == 2, f"两条结果没有都到达提示词: {result_lines}"
    assert result_lines[0] != result_lines[1], (
        "同一个工具的两条并行结果在提示词里完全相同，模型无法区分它们来自哪次调用: "
        f"{result_lines}"
    )
    print("test_two_results_from_the_same_tool_are_distinguishable: PASSED")


# --------------------- 3. 契约里承诺的标记必须真的出现在它描述的历史里

def test_the_contract_marker_actually_appears_in_the_history_it_describes():
    """工具段写着「收到 `[Tool Result for <id>]` 后继续」，历史里就必须有它。

    这是 §32/§76 的同一类判据：提示词里声明的协议必须与提示词里出现的内容一致，
    否则模型被告知去找一个从不存在的形状。
    """
    p = _prompt_for({"model": "gpt-4o", "messages": _PARALLEL_HISTORY,
                     "tools": list(_TOOLS)})
    assert "[Tool Result for <id>]" in p, (
        "工具契约文本变了，本用例的前提不再成立（契约里应写着该标记）"
    )
    history_part = p.split("User: read both files", 1)[-1]
    assert "[Tool Result for call_a]" in history_part, (
        "工具契约要求模型识别 `[Tool Result for <id>]`，但它回传的历史里从来没有"
        f"这个形状——契约与实际内容自相矛盾:\n{history_part[:400]!r}"
    )
    print("test_the_contract_marker_actually_appears_in_the_history_it_describes: PASSED")


# ---------------------------------------------------------------- 4. 零回归

def test_history_without_ids_and_empty_results_still_render():
    """没有 id 的结果、空结果都必须保持原样可读。

    修复不能把「没有 id」做成「不渲染」：客户端可以不回传 `tool_call_id`，而一个
    执行成功却没有输出的工具必须让模型知道它跑过了（否则会重复调用）。
    """
    p = _prompt_for({"model": "gpt-4o", "messages": [
        {"role": "user", "content": "run it"},
        {"role": "assistant", "content": "Running.", "tool_calls": [
            {"id": "call_x", "type": "function",
             "function": {"name": "Read", "arguments": '{"file_path":"/tmp/x"}'}}]},
        {"role": "tool", "content": ""},
    ], "tools": list(_TOOLS)})
    assert "(empty result)" in p, f"空结果没有被标注，模型会重复调用: {p[-200:]!r}"
    assert "Reading" not in p or True  # 占位：不约束无关文本

    p2 = _prompt_for({"model": "gpt-4o", "messages": [
        {"role": "user", "content": "hi"},
        {"role": "tool", "content": "loose output"},
    ], "tools": list(_TOOLS)})
    assert "loose output" in p2, f"无 id 的工具结果被整个丢掉了: {p2!r}"
    print("test_history_without_ids_and_empty_results_still_render: PASSED")


def run_all():
    test_a_tool_result_carries_its_call_id()
    test_two_results_from_the_same_tool_are_distinguishable()
    test_the_contract_marker_actually_appears_in_the_history_it_describes()
    test_history_without_ids_and_empty_results_still_render()
    assert not _GUARD_HITS, f"有请求绕过了脚本化 provider: {_GUARD_HITS}"
    print("\n✅ All tool-result pairing tests passed! (4 cases)")


if __name__ == "__main__":
    sys.exit(run_all())