"""Agent 工具循环闭合 (TOOL-LOOP) 测试套件

动机是一次实测出来的「工具调用没跑起来」：把 Claude Code / Codex CLI 这类 agent 接到
网关后，模型反复重发同一个工具调用、或者直接说做不了。

根因不在解析端，而在**回传端的提示词序列化**。上游五家都是纯文本网页接口，没有原生
工具调用，模型完全依靠「提示词里能看到自己发过什么调用」来维持循环。而
`providers/base.py` 的历史展平只读 `content`、从不读 `tool_calls`，且 `content` 为空的
消息被整条跳过。于是一轮标准 agent 循环到了上游变成：

    System: ## Tools Available ...
    User: read foo.py
    Tool: print(1)          ← 工具结果在
    Assistant:              ← 但它调用了什么工具、参数是什么，全没了

模型看到一个**自己从未发起过**的工具结果（连工具名与参数都没有），无法接续。
第二条同源缺口：续聊模式（`conversation_id` 非空）五个 provider 全部走
`last_user_content()`，它只取最后一条 `role=="user"`——而工具循环里最后一条是
`role:"tool"`，于是工具结果被跳过、上一轮的旧提问被重复发送（实测返回 `'read foo.py'`）。

本套件把这两条钉死，并锁定零回归：没有工具循环时，续聊发送的内容与原先逐值相同。

覆盖：
1. 展平提示词必须包含 assistant 的 tool_calls（名称 + 参数）
2. assistant 的 tool_calls 参数若为 JSON 字符串，不得被二次编码成字符串
3. `role:"tool"` 必须带上工具名（由 tool_call_id 反查）与内容
4. 端到端复现：一轮 tool_calls + tool 结果必须完整出现在上游提示词里（本套件靶点）
5. 续聊模式：最后一条是 tool 时，发送的是工具结果而不是上一轮旧提问（本套件靶点）
6. 续聊模式：正常无工具的一轮，发送内容与旧行为逐值相同（零回归）
7. 工具结果为空时仍要告知模型「跑过了」，否则会重复调用
8. 纯 prefill（无 user）时不得崩溃

运行: .venv/bin/python tests/test_tool_roundtrip.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["TESTING"] = "1"

from providers.base import continuation_content, flatten_messages  # noqa: E402

TOOLS_PROMPT = (
    "## Tools Available\n"
    "- `Read(path: string (required))`: Read a file\n"
    "- `Bash(command: string (required))`: Run a command"
)


def _agent_turn() -> list:
    """一轮标准 agent 循环：user → assistant(tool_calls) → tool 结果。"""
    return [
        {"role": "system", "content": TOOLS_PROMPT},
        {"role": "user", "content": "read foo.py"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_abc123",
                    "type": "function",
                    "function": {"name": "Bash", "arguments": '{"command": "cat foo.py"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_abc123", "content": "print(1)"},
    ]


# ------------------------------------------------- 1. tool_calls 必须进入提示词

def test_flatten_includes_assistant_tool_calls():
    """**靶点**：修复前 assistant 的 tool_calls 整段消失。"""
    prompt = flatten_messages(_agent_turn())

    assert "Bash" in prompt, f"工具名必须出现在上游提示词中，实际:\n{prompt}"
    assert "cat foo.py" in prompt, f"工具参数必须出现在上游提示词中，实际:\n{prompt}"
    assert "```tool_call" in prompt, "必须以模型被教导过的 ```tool_call 契约呈现"
    # 工具结果本身也必须在（这条原先就有，锁住不回归）
    assert "print(1)" in prompt
    print("test_flatten_includes_assistant_tool_calls: PASSED")


def test_tool_arguments_json_string_is_not_double_encoded():
    """OpenAI 线上格式的 arguments 是 JSON 字符串，塞进块里必须是参数对象而非字符串。"""
    msgs = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "Edit", "arguments": '{"file_path": "/tmp/a.py"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
    ]
    prompt = flatten_messages(msgs)

    assert '"arguments": {"file_path": "/tmp/a.py"}' in prompt, prompt
    assert '\\"file_path\\"' not in prompt, f"参数被二次编码成了字符串: {prompt}"
    print("test_tool_arguments_json_string_is_not_double_encoded: PASSED")


def test_tool_message_carries_its_name():
    """`role:"tool"` 只带 tool_call_id，必须反查出工具名，否则上游看到一段无名输出。"""
    prompt = flatten_messages(_agent_turn())
    assert "Tool (Bash):" in prompt, f"工具结果必须带上工具名，实际:\n{prompt}"
    print("test_tool_message_carries_its_name: PASSED")


# ------------------------------------------------- 2. 端到端：循环必须闭合

def test_agent_loop_is_complete_in_upstream_prompt():
    """结构性断言：一次调用 → 一次结果的完整往返，四项要素缺一不可。"""
    prompt = flatten_messages(_agent_turn())

    required = {
        "工具定义": "## Tools Available",
        "用户请求": "read foo.py",
        "助手发起调用": "```tool_call",
        "调用参数": "cat foo.py",
        "工具返回值": "print(1)",
        "结束标记": "Assistant:",
    }
    missing = [k for k, v in required.items() if v not in prompt]
    assert not missing, f"上游提示词缺少 {missing}，模型无法闭合工具循环:\n{prompt}"
    print("test_agent_loop_is_complete_in_upstream_prompt: PASSED")


# ------------------------------------------------- 3. 续聊模式

def test_continuation_sends_tool_result_not_stale_question():
    """**靶点**：续聊模式下最后一条是 tool，必须发送工具结果而不是旧提问。

    修复前 `last_user_content` 返回 'read foo.py'（上一轮的问题），工具结果永远
    送不到上游，agent 循环在第二次往返时必然断裂。
    """
    prompt = continuation_content(_agent_turn())

    assert "print(1)" in prompt, f"工具结果必须送达，实际: {prompt!r}"
    assert "Bash" in prompt, f"工具调用上下文必须送达，实际: {prompt!r}"
    assert prompt.strip() != "read foo.py", "绝不能重复发送上一轮的旧提问"
    print("test_continuation_sends_tool_result_not_stale_question: PASSED")


def test_continuation_without_tools_is_unchanged():
    """零回归：普通一轮对���，续聊发送的内容与旧实现逐值相同。"""
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！"},
        {"role": "user", "content": "今天天气如何"},
    ]
    assert continuation_content(msgs) == "今天天气如何"
    print("test_continuation_without_tools_is_unchanged: PASSED")


# ------------------------------------------------- 4. 边界

def test_empty_tool_result_is_still_reported():
    """工具成功但无输出时也要让模型知道它跑过了，否则会重复调用。"""
    msgs = _agent_turn()
    msgs[-1]["content"] = ""
    prompt = flatten_messages(msgs)
    assert "Tool (Bash)" in prompt, prompt
    assert "(empty result)" in prompt, f"空结果必须显式告知，实际:\n{prompt}"
    print("test_empty_tool_result_is_still_reported: PASSED")


def test_prefill_without_user_message_does_not_crash():
    """纯 prefill（system + assistant）不得触发 KeyError / IndexError。"""
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "assistant", "content": "{"},
    ]
    out = continuation_content(msgs)
    assert out, "不得返回空串"
    assert "{" in out
    print("test_prefill_without_user_message_does_not_crash: PASSED")


def run_all():
    tests = [
        test_flatten_includes_assistant_tool_calls,
        test_tool_arguments_json_string_is_not_double_encoded,
        test_tool_message_carries_its_name,
        test_agent_loop_is_complete_in_upstream_prompt,
        test_continuation_sends_tool_result_not_stale_question,
        test_continuation_without_tools_is_unchanged,
        test_empty_tool_result_is_still_reported,
        test_prefill_without_user_message_does_not_crash,
    ]
    for fn in tests:
        fn()
    print(f"\n✅ All tool-roundtrip tests passed! ({len(tests)} cases)")


if __name__ == "__main__":
    run_all()