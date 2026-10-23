---
name: wire-responses
description: OpenAI Responses wire contract ownership — what Codex CLI actually dispatches on. Use when working on /v1/responses, responses_compat.py, or Responses streaming/non-streaming paths in server.py.
---

# 模块：Responses 线协议 / Codex CLI 接入

## 你拥有的文件
- `providers/responses_compat.py`
- `server.py` 中 `/v1/responses` 的流式与非流式分支
- 对应测试：`tests/test_responses*.py`、`tests/test_agent_loop.py`

## 你必须守住的不变量
1. `output_index` 唯一、递增，每个 item 一个索引；`item_id` 在 added/delta/done 之间恒定。
2. `sequence_number` 无空洞；`type` 字段与事件名一致。
3. `output_text.delta`（不是旧版 `text.delta`）；`output_text` content part 带 `annotations`。
4. `function_call` 用 `function_call_arguments.*`；**custom/freeform 工具**用 `custom_tool_call_input.*`，且 payload 是**裸文本**不是 JSON 字符串。
5. 工具/推理 item 的生命周期：先关当前 item 再开下一个；reasoning 与 message 绝不能共用一个 index。

## 工作方法（硬性）
- 真实 `codex` 二进制（v0.145.0）在 `CODEX_HOME=/tmp/e2e/codexhome*`，`wire_api = "responses"`。
- **凡是"Codex 期望什么"的判断，必须跑真二进制验证**，不能只读源码猜。
- **先写红**，再修，再跑全套。

## 已被证伪，不要再提
- **`response.incomplete` 终止事件**：真实 Codex 实测 → 5 次重连后硬失败。当前 `response.completed` 是正确选择。
- **`max_tokens`**：上游不支持长度控制，见 wire-anthropic 同一条。

## 待办队列（PM 维护，从上往下做）
1. **实锤可疑**：真实 Codex 的原始请求里，`apply_patch` 是一个 **freeform custom 工具**，但网关把它渲染成了普通 function 工具。
2. 工具声明列表末尾出现空名条目 `- ``: `（`Args: {}`）—— 实测截图可见，来源待查。
3. 跑通真实 Codex 的 `apply_patch` 端到端（文件必须真的被创建）。