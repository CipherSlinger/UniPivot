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
- **空名工具条目 `- ``: ` 是名字损坏**：不是。真实 Codex 0.145.0 声明 `{"type":"tool_search"}` 与 `{"type":"web_search"}`，两者**本来就没有 `name`**。已改为「客户端处理、不可调用」的说明行。
- **`apply_patch` 渲染成 `Args: {}`**：不可达。真实抓包里它走 freeform 分支，得到 `{"input": "<raw text>"}` 契约；`apply_patch_tool_type` 在 0.145.0 只接受 `freeform`。端到端实测文件真的被创建。
- **「工具 schema 超预算时按位置降级是可接受的取舍」**：**证伪，是真实缺陷**（本轮已修）。实测真实 Claude Code 抓包（20 工具，schema 合计 14700 > 12000 预算）：`WebFetch/WebSearch/Workflow/Write` 被整条降级成 `Write(file_path: string (required), content: string (required))`，`file_path` 那句「must be absolute, not relative」**整句消失**。这是最高频的写文件工具，而**网关不做任何参数校验**（`grep jsonschema/validate` 全仓零命中），所以提示词是唯一契约。更根本的判据是**任意性**：把同一数组反过来，保真的变成 `Write/WebFetch`，降级的变成 `Bash/Agent` —— 哪个工具能拿到自己的契约由数组下标决定，没有任何依据。
- **「升预算（12000 → 更大）是正确修法」**：证伪。升预算是**把悬崖推远一格**：真实抓包 20 工具就要 14700，再多几个工具照样溢出，且溢出时仍然是「整条丢结构」。且预算存在的理由是上下文有限，不该用调参掩盖降级策略错误。

## 本轮已关闭（[TOOL-SCHEMA] 预算降级保真度）
- **缺陷**：`format_tools_system_prompt` 超预算时按**声明位置**整条降级为 `_compact_args_summary`（`param: type (required)`），丢掉参数描述、`enum`、嵌套对象字段名、数组元素类型。实测 16 个 FULL / 4 个 COMPACT，切点固定在 index 16。
- **修法**：改为**统一的描述压缩阶梯** `_TOOL_SCHEMA_DESCRIPTION_LADDER`（`(必填参数描述上限, 其他描述上限)`，从「完全不压」到 `(0,0)`）。先规范化全部声明，再按**整份工具集的总成本**选「刚好装进预算」的最轻一档，所有工具**同一档**渲染。结构（type/required/enum/items/properties/nested）在任何档位都完整保留；只有自然语言描述被压缩，且**必填参数的描述拿更高额度**（必填参数的语义是「填错就失败」，可选参数更多是「什么时候用它」）。
- **实测效果**：真实 20 工具抓包在 `(None, 100)` 档装下，**20/20 全部保住完整 schema**，`Write.file_path` 的「must be absolute, not relative」原文保留，`Agent.model` 的 4 个 enum 值保留。提示词总长 15569 → 15057（压掉的是可选参数的长篇描述）。
- **兜底不变**：>=40 个巨型工具仍一档都装不下时，退回紧凑摘要，提示词仍有界（60 工具 8449 字符）。
- **测试**：`tests/test_tool_schema.py` 新增 3 例（真实抓包 fixture `tests/fixtures/claude_code_20_tools.json` + 结构性/顺序无关断言 + 病态集合仍有界），`tests/test_agent_loop.py` 新增 1 例**端到端**（断言 `/v1/messages` 上游真正收到的提示词里含 `Write` 的绝对路径要求）。修复前实测红：`必填参数 file_path 的语义要求被预算降级吃掉了（被降级的工具: ['WebFetch','WebSearch','Workflow','Write']）`。
- **坑（值得记住）**：第一版端到端断言用「整段提示词里找 must be absolute」—— **修复前也是 True**，因为 `NotebookEdit.notebook_path` 的描述里有同一句话，用例在为错误的原因通过。必须断言**该工具自己那一块**。手写的最小 fixture（6.0k 字符）也撑不爆 12k 预算，必须用真实抓包。

## 待办队列（PM 维护，从上往下做）
1. 等待新缺陷（本模块已知问题均已修复并有回归测试）。