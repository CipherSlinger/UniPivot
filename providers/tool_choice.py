r"""客户端 `tool_choice` 约束的**唯一**判定 (TOOL-LOOP)

§34b 把这个字段拆成两个消费者：解析端的**白名单**（保证结果）与提示词里的
**一句话**（降低发生概率）——「提示词降低发生率，解析端保证结果」。两者读同一个
客户端字段，却曾是两处独立实现，于是两种形状上判得不一样，而两处都不会报错：

  A. `{"type": "none"}` —— Anthropic 合法的「本回合不要调用工具」写法。
     白名单认（空集合），提示词不认，模型从未被告知别调工具。
  B. 点名一个**根本没声明**的工具（`{"type": "tool", "name": "Nope"}`）。
     白名单按「无法识别」失败开放，提示词却仍写 `You MUST call the tool Nope`
     ——要求模型调用一个它连 schema 都没见过的工具，而解析端一定会挡掉它。

实测（脚本化上游，零真实上游调用）：`none` → 指令在提示词里；
`{"type":"none"}` → 不在；`{"type":"tool","name":"Nope"}` → 上游提示词里带着
`You MUST call the tool \`Nope\``，而白名单只有 `{'Read'}`。

修法不是给提示词补一个分支（那仍然是两处判断），而是让**两个消费者读同一个
判定**：本模块返回 `(allowed, directive)` 一次算完，`resolve_allowed_tools`
取前者，`format_tools_system_prompt` 取后者。

失败开放的方向与 `resolve_allowed_tools` 原有语义一致：只有客户端**明确声明**
时才收紧。客户端自相矛盾（点名未声明的工具）时两侧一起退回默认行为——收紧
白名单会把一个可用回合变成空回合，而对模型下一条它注定无法执行的命令更糟。
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

NO_TOOL_DIRECTIVE = "Do not call any tool in this turn; answer in plain text."
MUST_CALL_DIRECTIVE = "You MUST call at least one tool in this turn."


def tool_names_of(tools: Any) -> Optional[set]:
    """从客户端声明的 `tools` 里取出所有工具名（兼容 OpenAI / Anthropic / Responses 形状）。

    返回 None 表示「调用方没有声明工具」——此时不做白名单裁剪，保持既有解析行为。

    [TOOL-LOOP] Responses 的 custom（freeform）工具声明是
    `{"type": "custom", "name": "apply_patch"}`：没有 `function` 包装，但顶层有
    `name`，旧实现靠 `t.get("name")` 恰好覆盖到它。这里显式写出来，是因为白名单
    决定了「模型写出的调用能不能被回传」——漏掉一个声明过的名字，等于该工具永远
    不可调用，而症状是客户端侧的「工具没跑起来」，很难追到这一行。
    """
    if not tools:
        return None
    names = set()
    for t in tools:
        if not isinstance(t, dict):
            continue
        fn = t.get("function") if isinstance(t.get("function"), dict) else None
        name = fn.get("name") if fn else t.get("name")
        if isinstance(name, str) and name.strip():
            names.add(name.strip())
    return names or None


def _chosen_name(tool_choice: dict) -> Optional[str]:
    """从 dict 形状里取出被点名的工具（三种声明方言，取法与白名单同源）。

    - Anthropic：`{"type": "tool", "name": ...}`
    - OpenAI：`{"type": "function", "function": {"name": ...}}`
    - Responses 扁平：`{"type": "function" | "custom", "name": ...}`
    """
    c_type = tool_choice.get("type")
    if c_type == "tool":
        name = tool_choice.get("name")
    elif isinstance(tool_choice.get("function"), dict):
        name = tool_choice["function"].get("name")
    else:
        name = tool_choice.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return None


def interpret_tool_choice(
    tools: Any, tool_choice: Any = None
) -> Tuple[Optional[set], str]:
    """一次判定，两个消费者：`(允许回传的工具名集合, 提示词指令)`。

    集合语义（与 `tool_names_of` 对齐）：
      - `None`：不做裁剪（未声明工具，或形状无法识别——失败开放）
      - 空集合：本回合不允许任何工具调用
      - 非空集合：只接受集合内的名字

    指令只降低发生概率，绝不替代集合；两者由同一次判断产出，因此不可能互相矛盾。
    没有声明任何工具时指令为空——此时上游连工具契约都没收到，再说「必须调用」
    只会自相矛盾。
    """
    names = tool_names_of(tools)
    if names is None:
        return None, ""

    if tool_choice is None:
        return names, ""

    if isinstance(tool_choice, str):
        choice = tool_choice.strip().lower()
        if choice == "none":
            return set(), NO_TOOL_DIRECTIVE
        if choice in ("required", "any"):
            return names, MUST_CALL_DIRECTIVE
        # "auto" 及未知字符串：默认行为，不追加指令
        return names, ""

    if isinstance(tool_choice, dict):
        c_type = tool_choice.get("type")
        if c_type == "none":
            return set(), NO_TOOL_DIRECTIVE
        if c_type == "any":
            return names, MUST_CALL_DIRECTIVE
        chosen = _chosen_name(tool_choice)
        if chosen:
            if chosen in names:
                return {chosen}, (
                    f"You MUST call the tool `{chosen}` in this turn, "
                    "with arguments matching its schema."
                )
            # 客户端自相矛盾：点名了一个未声明的工具。两侧一起失败开放，而不是
            # 一边放行一边命令模型调用一个它看不到 schema 的工具。
            return names, ""
        return names, ""

    return names, ""
