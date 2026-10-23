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

## 本轮已关闭（[TOOL-LOOP] `namespace` 工具声明 → 子工具被整段吞掉）

- **实测抓包**（`codex exec` 0.145.0 原始 `/v1/responses` 请求体，10 个工具里的第 6 个）：
  ```json
  {"type":"namespace","name":"multi_agent_v1",
   "description":"Tools for spawning and managing sub-agents.",
   "tools":[{"type":"function","name":"close_agent","parameters":{...}},
            ... resume_agent / send_input / spawn_agent / wait_agent ...]}
  ```
  它**不是工具，是容器**：五个带完整 schema 的 function 子工具。抓包存为
  `tests/fixtures/codex_multi_agent_v1_namespace.json`。
- **缺陷**：`format_tools_system_prompt` 把它当普通声明走通用分支，取顶层
  `name`/`description`、找不到 `parameters`，实测渲染成
  ```
  - `multi_agent_v1`: Tools for spawning and managing sub-agents.
    Args: {}
  ```
  两个错误叠加：(a) 五个子工具的名字与**全部参数 schema** 消失，模型根本不知道
  它们存在；(b) 被告知可调用的名字是 `multi_agent_v1`，而真实客户端**拒绝**它。
  这与 `apply_patch` 被告知 `Args: {}` 属同一缺陷类，只是多了一个错名。
- **客户端路由实测**（同一 mock 上游，只改回传 call item 的形状）：
  | 回传的 call | 真实 Codex 结果 |
  |---|---|
  | `name="multi_agent_v1"` | `unsupported call: multi_agent_v1`（重试数百次） |
  | `name="spawn_agent"`（裸名） | `collab: SpawnAgent` → `agent thread limit reached`（**已路由**） |
  | `close_agent`/`resume_agent`/`send_input`/`wait_agent` 裸名 | 已路由，只有假 agent id 报错 |
  | 裸名 + `"namespace":"multi_agent_v1"` | `unsupported call: multi_agent_v1spawn_agent`（客户端把 namespace 与 name **无分隔符拼接**再查表） |
  | `multi_agent_v1.spawn_agent` / `::` / `__` | 全部 `unsupported call` |
  结论：唯一正确形状是**裸名 + 完整 schema**。
- **修法**：`flatten_namespace_tools()`（`providers/anthropic_compat.py`）是展平的
  唯一实现，两处调用：(1) `format_tools_system_prompt` 入口（任何调用方都能拿到
  正确提示词）；(2) `AnthropicRequestBody` / `ResponsesRequest` 的
  `tools` field_validator（**解析白名单**必须看到同一份展平结果）。只修提示词会让
  模型写出 `spawn_agent` 然后被 `resolve_allowed_tools` 当作未声明工具丢弃——症状与
  修复前一样，只是更难追。namespace 自己的名字**不进**可调用列表，改为单独一节
  `## Tool Namespaces (not callable)` 说明分组，与「无名内置工具」分开（原因不同：
  后者没有名字，前者有名字但不可调用）。
- **零回归**：非 `namespace` 声明原样透传、顺序不变；展平只发生在请求模型入口，
  不修改客户端报文。真实抓包 10 工具 → 13 个可调用工具，`(None, 300)` 档装下
  （12154 < 12000+ 预算），`spawn_agent` 的可选参数与 `send_input.items` 的嵌套字段
  全部保真。
- **测试**：`tests/test_responses_custom_tools.py` 新增 4 例（真实 fixture + 提示词
  保真/不可调用说明/白名单/端到端往返）。修复前实测红，报错原文：
  `namespace 仍被渲染成一个可调用工具，且参数是空的 Args: {}——模型会去调
  \`multi_agent_v1\`，而真实 Codex 回 \`unsupported call: multi_agent_v1\``。
- **遗留（需 PM 决策，不在本模块可改范围）**：OpenAI chat 协议的
  `ChatCompletionRequest` 定义在 `server.py`，它的 `req.tools` 没有经过展平，���以
  那条路径上提示词正确、白名单仍只有 `multi_agent_v1`。真实 Codex 只走
  `/v1/responses`，因此当前不可达；修法是在 `server.py` 的该字段上加同一个
  validator（或让 `resolve_allowed_tools` 内部展平，但那会把展平散到第二处）。

## 本轮已关闭（[TOOL-SCHEMA] 预算降级保真度）
- **缺陷**：`format_tools_system_prompt` 超预算时按**声明位置**整条降级为 `_compact_args_summary`（`param: type (required)`），丢掉参数描述、`enum`、嵌套对象字段名、数组元素类型。实测 16 个 FULL / 4 个 COMPACT，切点固定在 index 16。
- **修法**：改为**统一的描述压缩阶梯** `_TOOL_SCHEMA_DESCRIPTION_LADDER`（`(必填参数描述上限, 其他描述上限)`，从「完全不压」到 `(0,0)`）。先规范化全部声明，再按**整份工具集的总成本**选「刚好装进预算」的最轻一档，所有工具**同一档**渲染。结构（type/required/enum/items/properties/nested）在任何档位都完整保留；只有自然语言描述被压缩，且**必填参数的描述拿更高额度**（必填参数的语义是「填错就失败」，可选参数更多是「什么时候用它」）。
- **实测效果**：真实 20 工具抓包在 `(None, 100)` 档装下，**20/20 全部保住完整 schema**，`Write.file_path` 的「must be absolute, not relative」原文保留，`Agent.model` 的 4 个 enum 值保留。提示词总长 15569 → 15057（压掉的是可选参数的长篇描述）。
- **兜底不变**：>=40 个巨型工具仍一档都装不下时，退回紧凑摘要，提示词仍有界（60 工具 8449 字符）。
- **测试**：`tests/test_tool_schema.py` 新增 3 例（真实抓包 fixture `tests/fixtures/claude_code_20_tools.json` + 结构性/顺序无关断言 + 病态集合仍有界），`tests/test_agent_loop.py` 新增 1 例**端到端**（断言 `/v1/messages` 上游真正收到的提示词里含 `Write` 的绝对路径要求）。修复前实测红：`必填参数 file_path 的语义要求被预算降级吃掉了（被降级的工具: ['WebFetch','WebSearch','Workflow','Write']）`。
- **坑（值得记住）**：第一版端到端断言用「整段提示词里找 must be absolute」—— **修复前也是 True**，因为 `NotebookEdit.notebook_path` 的描述里有同一句话，用例在为错误的原因通过。必须断言**该工具自己那一块**。手写的最小 fixture（6.0k 字符）也撑不爆 12k 预算，必须用真实抓包。

## 待办队列（PM 维护，从上往下做）
1. 等待新缺陷（本模块已知问题均已修复并有回归测试）。