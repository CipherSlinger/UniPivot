---
name: tool-loop
description: Tool-calling correctness ownership — prompt schema fidelity, parsing, anti-fabrication, stop-sequence truncation, tool_choice enforcement. Use when working on tool_streamer.py, base.py flattening, or any code that turns model text into executable tool calls.
---

# 模块：工具调用闭环

## 你拥有的文件
- `providers/tool_streamer.py`
- `providers/base.py` 中 `flatten_messages` / `text_of` / 工具提示词渲染
- `providers/stop_sequences.py`
- 对应测试：`tests/test_tool_*.py`、`tests/test_agent_loop.py`、`tests/test_stop_sequences.py`、`tests/test_tool_schema.py`

## 你必须守住的不变量
1. **工具提示词是上游唯一能看到的 schema** —— 嵌套对象/数组/枚举/required 必须保真。
2. `tool_choice` / `parallel_tool_calls` 必须在**解析层**强制，不能只靠提示词。
3. 反伪造三层：格式白名单 → 严格形状 → 已声明名字白名单。被拒的调用要作为**文本**返回，不能静默吞掉。
4. 停用词截断**先于**工具解析：截断点之后的 ````tool_call```` 绝不能被执行。
5. 工具结果必须带回 `id`/`name`，且历史重放时 assistant 的 `tool_calls` 不能丢。

## 工作方法（硬性）
- 断言必须打在**上游真正收到的 prompt** 上，不是网关自己的响应对象。
- 先写红，修复后跑全套。

## 已被证伪，不要再提
- **巨型工具结果不被折叠**：位于 `preserve_recent_turns` 窗口内的内容按设计不折叠 —— 折叠当前轮的工具结果会毁掉模型正在看的数据。

## 待办队列（PM 维护，从上往下做）
1. 空名工具条目 `- ``: ` 的来源（与 wire-responses 联合排查，可能是同一条）。
2. 当 `apply_patch` 这类 freeform 工具的声明被渲染成 `Args: {}` 时，模型看到的是"这个工具不接受参数"。