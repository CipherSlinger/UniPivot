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
from providers.tool_streamer import resolve_allowed_tools  # noqa: E402


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


def test_responses_flat_tool_choice_is_enforced():
    """Responses 的**扁平**形状既是工具声明的形状，也是 tool_choice 的形状。

    [TOOL-LOOP] 两处实现都只认 OpenAI 的嵌套形状
    `{"type":"function","function":{"name":...}}` 与 Anthropic 的
    `{"type":"tool","name":...}`，于是 Responses 原生写法
    `{"type":"function","name":"read_file"}`（顶层 name，无 function 包装）
    在解析白名单里**失败开放**成「全部声明工具」，在提示词里也**不追加任何指令**：

        Responses flat {type:function, name}  allow={'read_file','write_file'}  directive=''

    客户端点名了一个工具，网关既没告诉模型去调它，也没有在解析端挡住别的调用。
    对 Codex 这类按 Responses 契约发请求的客户端，这个约束等于不存在。

    custom（freeform）的扁平形状同理：`{"type":"custom","name":"apply_patch"}`。
    """
    tools = [
        {"type": "function", "name": "read_file", "parameters": {"type": "object"}},
        {"type": "function", "name": "write_file", "parameters": {"type": "object"}},
        {"type": "custom", "name": "apply_patch"},
    ]

    # 解析端：点名即收窄到该工具
    flat_fn = {"type": "function", "name": "read_file"}
    assert resolve_allowed_tools(tools, flat_fn) == {"read_file"}, (
        f"Responses 扁平 tool_choice 未收窄白名单: {resolve_allowed_tools(tools, flat_fn)}"
    )
    flat_custom = {"type": "custom", "name": "apply_patch"}
    assert resolve_allowed_tools(tools, flat_custom) == {"apply_patch"}, (
        f"Responses 扁平 custom 形状未收窄白名单: {resolve_allowed_tools(tools, flat_custom)}"
    )

    # 提示词端：点名必须出现
    assert "MUST call the tool `read_file`" in format_tools_system_prompt(tools, tool_choice=flat_fn)
    assert "MUST call the tool `apply_patch`" in format_tools_system_prompt(tools, tool_choice=flat_custom)

    # 回归边界：**没有** name 的 `{"type":"function"}` 仍是无法识别 -> 失败开放，
    # 不得被误当成「点名了某个空名字」而清空白名单。
    assert resolve_allowed_tools(tools, {"type": "function"}) == {
        "read_file", "write_file", "apply_patch"
    }
    assert format_tools_system_prompt(tools, tool_choice={"type": "function"}) == \
        format_tools_system_prompt(tools)
    print("test_responses_flat_tool_choice_is_enforced: PASSED")


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
    # 每个工具 schema 很大（6 个 280 字符的描述字段）。旧实现从第 N 个开始整条
    # 降级为紧凑摘要；现在改为全局压缩描述文本，**结构全部保留**。
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

    # 不抛错 + 有界：整体不应随工具数线性爆炸。
    assert len(text) < 30000, f"渲染文本失控: {len(text)} 字符"
    # 30 个这种工具在「描述压到 0 字」这一档可以完整保住结构，因此不该再出现
    # 紧凑摘要；结构（required 列表 + 字段名）必须活着。
    assert "field_0: string (required)" not in text, (
        "该集合已可在保结构的前提下装进预算，不应退回紧凑摘要"
    )
    assert '"required":["field_0"]' in text, "结构被压掉了"
    assert '"field_5"' in text, "字段名被压掉了"
    # 每个工具名都仍在（降级不等于丢工具）
    for i in range(30):
        assert f"Tool{i}" in text
    print("test_oversized_tool_set_degrades_gracefully: PASSED")


# ------------------------------------------------- 9. 预算耗尽时的保真度（按位置降级）

# [TOOL-SCHEMA] 真实 Claude Code 抓包的 20 个工具声明，原样存放在
# `tests/fixtures/claude_code_20_tools.json`（取自 `/tmp/e2e/claude_bodies.jsonl`
# 第一行的 `tools` 数组）。用真实抓包而不是手写的近似：手写的最小 schema 只有
# 6.0k 字符，根本撑不爆 12k 预算，于是「按位置降级」这个缺陷在测试里不可见 ——
# 而真实声明的 schema 合计 14.7k，精确地把后 4 个工具挤到降级分支上。
_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


def _claude_code_tools() -> list:
    with open(_FIXTURE_DIR / "claude_code_20_tools.json", encoding="utf-8") as f:
        return json.load(f)


# 真实抓包里被旧实现按位置降级的四个工具。
_POSITIONALLY_DEGRADED = ("WebFetch", "WebSearch", "Workflow", "Write")


def _degraded_tools(text: str, tools: list) -> list:
    """返回渲染成紧凑摘要（`- `Name(...)``）的工具名 —— 即丢掉参数描述的工具。"""
    names = []
    for t in tools:
        head = f"- `{t['name']}("
        if any(line.startswith(head) for line in text.split("\n")):
            names.append(t["name"])
    return names


def test_schema_budget_does_not_silently_drop_required_param_semantics():
    """**靶点（修复前实测红）**：真实 Claude Code 的 20 工具集里，`Write` 被降级成
    `Write(file_path: string (required), content: string (required))`，于是
    `file_path` 的语义要求「**必须是绝对路径，不能是相对路径**」整句消失。

    实测（修复前）：

        prompt_len = 15569
        DEGRADED   = ['WebFetch', 'WebSearch', 'Workflow', 'Write']
        `Write(file_path: string (required), content: string (required))`: ...

    `Write` 是最高频的 agent 工具之一，而它唯一的语义约束恰恰在被丢掉的那句话里：
    模型给出 `foo.py` 这类相对路径，文件就写到了与用户意图不同的位置。网关不做
    参数校验（全仓无任何 schema 校验），所以这段提示词是**唯一**的契约。

    还有一个更根本的判据：**哪些工具被降级完全取决于声明顺序**。把同一个数组
    反过来，保真的变成 `Write`/`WebFetch`，降级的变成 `Bash`/`Agent` —— 一个工具
    能不能拿到自己的契约，不能由它在数组里的下标决定。
    """
    tools = _claude_code_tools()
    text = format_tools_system_prompt(tools, parallel_allowed=True)

    degraded = _degraded_tools(text, tools)
    # 只在 **Write 自己那一块**里找：`NotebookEdit.notebook_path` 的描述同样含有
    # 「(must be absolute, not relative)」，整段搜索会在修复前为错误的原因通过。
    lines = text.splitlines()
    write_idx = next((i for i, l in enumerate(lines) if l.startswith("- `Write")), None)
    assert write_idx is not None, "Write 声明消失了"
    write_block = "\n".join(lines[write_idx:write_idx + 2])
    assert "must be absolute, not relative" in write_block, (
        f"必填参数 file_path 的语义要求被预算降级吃掉了（被降级的工具: {degraded}）:\n"
        f"{write_block[:1200]}"
    )

    # 顺序无关：同一个集合，换个声明顺序，每个工具拿到的保真度必须相同。
    reversed_tools = list(reversed(tools))
    rev_degraded = _degraded_tools(format_tools_system_prompt(reversed_tools), reversed_tools)
    assert set(degraded) == set(rev_degraded), (
        f"降级是**按位置**的：原序降级 {sorted(degraded)}，反序降级 {sorted(rev_degraded)}。"
        "工具能否拿到自己的契约由数组下标决定，这是任意的。"
    )
    print("test_schema_budget_does_not_silently_drop_required_param_semantics: PASSED")


def test_degradation_keeps_structure_enums_and_required_list():
    """降级可以压缩**描述文本**，但不能丢**结构**（type/required/enum/嵌套）。

    旧降级 `_compact_args_summary` 只留 `param: type (required)`：`enum` 值全部
    消失、嵌套对象的字段名消失、数组元素类型消失。`Agent` 的 `model` 只有
    `sonnet/opus/haiku/fable` 四个合法值，降级后模型只能猜。结构是「能不能正确
    构造调用」的信息，描述只是「什么时候用它」的提示 —— 预算是有限的，先丢前者
    的优先级是反的。
    """
    reversed_tools = list(reversed(_claude_code_tools()))  # 这一序下 Agent/Bash 会被降级
    text = format_tools_system_prompt(reversed_tools)

    for name in ("Agent", "Bash"):
        assert any(line.startswith(f"- `{name}`") or line.startswith(f"- `{name}(")
                   for line in text.split("\n")), f"{name} 从提示词中消失"
    # Agent 的 enum 与 required 结构必须在（哪怕描述被压到 0 字）
    assert '"enum":["sonnet","opus","haiku","fable"]' in text, (
        f"降级后 Agent.model 的合法取值丢失:\n{text[:2000]}"
    )
    assert '"required":["description","prompt"]' in text, "降级后 required 列表丢失"
    print("test_degradation_keeps_structure_enums_and_required_list: PASSED")


def test_oversized_tool_set_still_degrades_to_a_bounded_prompt():
    """真正的极端情形（40+ 个巨型工具）仍然必须有界 —— 分层降级不是「无限展开」。

    真实抓包（14700 字符）必须完整保真于预算内；但一个 60 工具、每个 6 个 280 字符
    描述字段的病态集合（约 60k 字符）不可能全保真，此时退回紧凑摘要，整体仍有界。
    """
    big_schema = {
        "type": "object",
        "properties": {f"field_{i}": {"type": "string", "description": "y" * 280} for i in range(6)},
        "required": ["field_0"],
    }
    many = [{"type": "function", "function": {"name": f"Tool{i}", "parameters": big_schema}}
            for i in range(60)]
    text = format_tools_system_prompt(many)

    assert len(text) < 30000, f"渲染文本失控: {len(text)} 字符"
    for i in range(60):
        assert f"Tool{i}" in text, f"工具 Tool{i} 被整条丢弃"


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
        test_responses_flat_tool_choice_is_enforced,
        test_long_description_is_truncated,
        test_duplicate_instruction_block_is_gone,
        test_oversized_tool_set_degrades_gracefully,
        test_schema_budget_does_not_silently_drop_required_param_semantics,
        test_degradation_keeps_structure_enums_and_required_list,
        test_oversized_tool_set_still_degrades_to_a_bounded_prompt,
        test_tool_use_id_is_carried_into_transcript,
    ]
    for fn in tests:
        fn()
    print(f"\n✅ All tool-schema tests passed! ({len(tests)} cases)")


if __name__ == "__main__":
    run_all()