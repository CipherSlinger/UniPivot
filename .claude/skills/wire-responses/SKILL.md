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
- **`apply_patch` 被渲染成普通 function 工具**：已修（commit 87b1230）。真实抓包里它是 `{"type":"custom","name":"apply_patch"}`，网关渲染出 `{"input": ...}` freeform 契约，端到端实测文件真被创建。
- **`response.failed` 能减少重试**：不能。实测三种终局事件形状的重试次数都是 6。它修复的是**失败原因能否到达用户**（`stream closed before response.completed` → `upstream quota exceeded`）。
- **`gpt-5-codex` 的抓包能代表 Codex 的工具声明**：不能。该模型不在 Codex 自己的 catalog 里，会退回无 `apply_patch_tool_type` 的元数据，于是**根本不声明 `apply_patch`**。要看完整声明必须用 catalog 内的模型。

## 工作方法补充
- `codex` 的 `base_url` 来自 `CODEX_HOME/config.toml` 的 `[model_providers.*]`，**`OPENAI_BASE_URL` 环境变量被忽略**。测不同端口必须复制一份 config 改 `base_url`，否则所有测量都打在同一个旧端口上（曾因此得出「两种形状没区别」的错误结论）。

## 待办队列（PM 维护，从上往下做）
1. `response.failed` 的 `error.code` 是否该用 OpenAI 标准枚举（`rate_limit_exceeded` 等）而非内部 `err_type`，影响客户端重试分级 —— 需要真实 Codex 验证。