---
name: wire-anthropic
description: Anthropic Messages wire contract ownership — what Claude Code CLI actually dispatches on. Use when working on /v1/messages, /v1/messages/count_tokens, anthropic_compat.py, or the Anthropic streaming/non-streaming paths in server.py.
---

# 模块：Anthropic 线协议 / Claude Code 接入

## 你拥有的文件
- `providers/anthropic_compat.py`
- `server.py` 中 `/v1/messages`、`/v1/messages/count_tokens`、Anthropic 流式与非流式分支
- 对应测试：`tests/test_anthropic_stream.py`、`tests/test_server.py` 的 Anthropic 部分

## 你必须守住的不变量
1. `content_block` 索引唯一、递增、成对 start/stop；每种类型只在自己块内发 delta。
2. `message_start` 第一，`message_stop` 最后；terminal `usage.output_tokens` 必须存在。
3. `stop_reason` 只能是 Anthropic 的枚举；`stop_sequence` 停用词命中时必须**带上命中的原文**。
4. `tool_use` 块：`id` / `name` / `input:{}` + `input_json_delta.partial_json` 累加。
5. 客户端声明 `tool_choice` / `parallel_tool_calls` / `stop_sequences` 必须在**解析层**生效，不能只写在提示词里。

## 工作方法（硬性）
- **先测量，再下结论**。真实 `claude` CLI 的探针在 `/tmp/e2e/`（见 `claudehome/`）。任何"客户端会怎样"的判断都要有实测或客户端源码依据。
- **先写红**：新测试必须能在修复前跑出**预测的那个症状**，并在测试里写清楚。
- 修复后跑全套：`for f in tests/test_*.py; do .venv/bin/python $f; done`
- 提交信息带 `Co-Authored-By: Claude Code <noreply@anthropic.com>`。

## 已被证伪，不要再提
- **`max_tokens` 强制截断**：五个上游网页接口都不暴露长度控制，网关无从执行。接受并转发但无法强制 = 正确行为，不是缺陷。
- **Responses 的 `response.incomplete` 终止事件**：用真实 Codex 二进制实测过，改成 incomplete 会导致 5 次重连后硬失败。现状是对的。

## 待办队列（PM 维护，从上往下做）
1. `count_tokens` 拒绝 `system: ["..."]`（字符串列表）而 `/v1/messages` 接受 —— **实测 400**。
2. thinking 块是否带 `signature` / `signature_delta`（未验证，先查真实 Claude Code 是否要求）。
3. 工具路径的 `stop_reason` 是否会在停用词命中时被 `tool_use` 覆盖（读代码判断，不必修）。
4. `cache_control` 被解析后丢弃，`cache_*_input_tokens` 永远为 0 —— 判断是否值得填。