"""工具声明渲染保真度 (TOOL-SCHEMA) 测试套件

动机：上游五家都没有原生工具调用，工具完全靠注入系统提示词来模拟，调用则从模型
的散文里用正则捞回来。于是 `format_tools_system_prompt` 生成的这段文本是模型
**唯一**能看到的 schema —— 它丢了什么，模型就永远不知道什么。

旧实现把每个工具的 JSON Schema 展平成 `param: type (required)`：

    - `Edit(file_path: string (required), old_string: string (required), ...)`: ...

嵌套对象、对象数组、`enum`、参数级 description、`additionalProperties` 全部消失。
对 Claude Code 的 `Edit`/`Bash`/`Write` 来说，模型因此会发出错误的参数形状
（该给对象的地方给字符串、凭空发明 enum 值），解析再成功下游也执行不了。

覆盖：
1. 嵌套 object schema 必须原样保留（`"enum":[1,2,3]`、`"depth"`）
2. required vs optional 必须可辨认（required 列表出现）
3. Anthropic 形状（`input_schema`）与 OpenAI 形状同等保真
4. `tool_choice` 约束：none 追加禁调用指令、指定工具点名、未知形状静默且不抛错
5. 超长 description 必须截断
6. `[Important Tool Calling Instruction]` 第二份拷贝已删除
7. 超大工具集优雅降级、不抛错、总长有界
8. tool_use 的 id 进入 ```tool_call 块；无 id 时 JSON 不含 id 键

运行: .venv/bin/python tests/test_tool_schema.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["TESTING"] = "1"

from providers.anthropic_compat import (  # noqa: E402
    AnthropicMessage,
    convert_anthropic_to_gateway_messages,
    extract_anthropic_content,
    format_tools_system_prompt,
)


def _nested_tool() -> dict:
    """嵌套 object + 整数 enum 的工具声明（Edit 一类参数的真实形状）。"""
    return {
        "type": "function",
        "function": {
            "name": "Edit",
            "description": "Edit a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to the file"},
                    "options": {
                        "type": "object",
                        "properties": {
                            "depth": {"type": "integer", "enum": [1, 2, 3]},
                        },
                        "required": ["depth"],
                    },
                },
                "required": ["file_path"],
            },
        },
    }


# ------------------------------------------------- 1. 嵌套 schema 必须存活

def test_nested_object_schema_survives():
    """**靶点**：旧实现只打印 `options: object`，嵌套字段与 enum 全部丢失。"""
    text = format_tools_system_prompt([_nested_tool()])

    assert '"enum":[1,2,3]' in text, f"enum 丢失:\n{text}"
    assert '"depth"' in text, f"嵌套字段名丢失:\n{text}"
    assert '"properties"' in text, f"嵌套结构被展平:\n{text}"
    print("test_nested_object_schema_survives: PASSED")


# ------------------------------------------------- 2. required 可辨认

def test_required_vs_optional_is_expressed():
    text = format_tools_system_prompt([_nested_tool()])

    assert '"required":["file_path"]' in text, f"required 列表丢失:\n{text}"
    # 可选字段不得被标成 required
    assert '"required":["file_path","options"]' not in text
    print("test_required_vs_optional_is_expressed: PASSED")


# ------------------------------------------------- 3. Anthropic 形状

def test_anthropic_input_schema_shape():
    tools = [{
        "name": "Bash",
        "description": "Run a shell command",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout": {"type": "integer", "default": 120},
            },
            "required": ["command"],
        },
    }]
    text = format_tools_system_prompt(tools)

    assert "`Bash`" in text
    assert '"command"' in text
    assert '"default":120' in text, f"default 丢失:\n{text}"
    assert '"required":["command"]' in text
    print("test_anthropic_input_schema_shape: PASSED")


# ------------------------------------------------- 4. tool_choice 约束

def test_tool_choice_directives():
    tools = [{"type": "function", "function": {"name": "Bash", "parameters": {"type": "object"}}}]

    none_text = format_tools_system_prompt(tools, tool_choice="none")
    assert "Do not call any tool in this turn" in none_text, none_text
    # 工具列表仍需保留（客户端可能同时下发 tools 与 none）
    assert "`Bash`" in none_text

    auto_text = format_tools_system_prompt(tools, tool_choice="auto")
    assert auto_text == format_tools_system_prompt(tools), "auto 不得追加指令"

    req_text = format_tools_system_prompt(tools, tool_choice="required")
    assert "MUST call at least one tool" in req_text, req_text
    any_text = format_tools_system_prompt(tools, tool_choice={"type": "any"})
    assert "MUST call at least one tool" in any_text, any_text

    named = format_tools_system_prompt(tools, tool_choice={"type": "tool", "name": "Bash"})
    assert "MUST call the tool `Bash`" in named, named
    fn_named = format_tools_system_prompt(
        tools, tool_choice={"type": "function", "function": {"name": "Bash"}}
    )
    assert "MUST call the tool `Bash`" in fn_named, fn_named

    # 未识别的形状：不得追加任何指令，更不能抛错
    for junk in ({}, {"type": "weird"}, ["not", "a", "choice"], 42):
        out = format_tools_system_prompt(tools, tool_choice=junk)
        assert out == format_tools_system_prompt(tools), f"未知 tool_choice 改变了输出: {junk!r}"
    print("test_tool_choice_directives: PASSED")


# ------------------------------------------------- 5. description 截断

def test_long_description_is_truncated():
    long_desc = "x" * 1000
    tools = [{
        "type": "function",
        "function": {
            "name": "Huge",
            "description": "short tool desc",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "string", "description": long_desc}},
            },
        },
    }]
    text = format_tools_system_prompt(tools)

    assert long_desc not in text, "1000 字符的 description 未被截断"
    assert "x" * 300 in text, "截断应保留前 300 字符"
    assert "x" * 301 not in text, "截断长度必须恰为 300"
    print("test_long_description_is_truncated: PASSED")


# ------------------------------------------------- 6. 第二份指令拷贝已删除

def test_duplicate_instruction_block_is_gone():
    tools = [_nested_tool()]
    msgs = [AnthropicMessage(role="user", content="edit the file")]
    out = convert_anthropic_to_gateway_messages(msgs, system="You are a coding agent.", tools=tools)
    joined = "\n".join(m["content"] for m in out)

    assert "[Important Tool Calling Instruction]" not in joined, joined
    # 防拒答语义仍在（已并入唯一规则段）
    assert "Do NOT refuse" in joined, "防拒答句必须保留在统一规则段里"
    print("test_duplicate_instruction_block_is_gone: PASSED")


# ------------------------------------------------- 7. 超大工具集降级

def test_oversized_tool_set_degrades_gracefully():
    # 每个工具 schema 约 1.7k 字符：前几个在 12k 预算内保真，其余溢出降级。
    big_schema = {
        "type": "object",
        "properties": {
            f"field_{i}": {"type": "string", "description": "y" * 280}
            for i in range(6)
        },
        "required": ["field_0"],
    }
    tools = [
        {"type": "function", "function": {"name": f"Tool{i}", "parameters": big_schema}}
        for i in range(30)
    ]
    text = format_tools_system_prompt(tools)

    # 不抛错 + 有界：规则段与降级摘要都有固定规模，整体不应随工具数线性爆炸。
    assert len(text) < 30000, f"渲染文本失控: {len(text)} 字符"
    # 前若干个工具保真（Args: JSON），溢出的工具退回旧摘要形状。
    assert "Args: {" in text
    assert "field_0: string (required)" in text, "溢出工具未降级为紧凑摘要"
    # 每个工具名都仍在（降级不等于丢工具）
    for i in range(30):
        assert f"Tool{i}" in text
    print("test_oversized_tool_set_degrades_gracefully: PASSED")


# ------------------------------------------------- 8. tool_use id 往返

def test_tool_use_id_is_carried_into_transcript():
    with_id = [
        {"type": "tool_use", "id": "toolu_abc", "name": "Bash", "input": {"command": "ls"}},
        {"type": "tool_result", "tool_use_id": "toolu_abc", "content": "a.txt"},
    ]
    text = extract_anthropic_content(with_id)
    assert "```tool_call" in text
    block = text.split("```tool_call\n", 1)[1].split("\n```", 1)[0]
    payload = json.loads(block)
    assert payload["id"] == "toolu_abc", payload
    assert payload["name"] == "Bash"
    assert payload["arguments"] == {"command": "ls"}
    assert "[Tool Result for toolu_abc]" in text

    # 无 id：JSON 中不得出现 id 键（既有精确断言依赖这一点）
    without_id = [{"type": "tool_use", "name": "Bash", "input": {"command": "ls"}}]
    text2 = extract_anthropic_content(without_id)
    block2 = text2.split("```tool_call\n", 1)[1].split("\n```", 1)[0]
    assert "id" not in json.loads(block2), block2

    # is_error 仍走 [Tool Error for ...]
    err = [{"type": "tool_result", "tool_use_id": "toolu_abc", "content": "boom", "is_error": True}]
    assert "[Tool Error for toolu_abc]" in extract_anthropic_content(err)
    print("test_tool_use_id_is_carried_into_transcript: PASSED")


def run_all():
    tests = [
        test_nested_object_schema_survives,
        test_required_vs_optional_is_expressed,
        test_anthropic_input_schema_shape,
        test_tool_choice_directives,
        test_long_description_is_truncated,
        test_duplicate_instruction_block_is_gone,
        test_oversized_tool_set_degrades_gracefully,
        test_tool_use_id_is_carried_into_transcript,
    ]
    for fn in tests:
        fn()
    print(f"\n✅ All tool-schema tests passed! ({len(tests)} cases)")


if __name__ == "__main__":
    run_all()