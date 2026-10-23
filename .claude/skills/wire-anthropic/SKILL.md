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
- **`count_tokens` 只是便利端点**：不是。Claude Code 每次请求前用它判断上下文预算；数字错了表现为 agent 压缩得太晚或太早，且没有任何报错。
- **两个端点各有一份请求体模型**：不行。同一份报文两种解释必然漂移——实测 `system: ["be terse"]` 在 messages 上 200、在 count_tokens 上 400。已合并为 `AnthropicRequestBody` + 子类。
- **`Retry-After` 能抑制 Claude Code 对 503 的重试**：不能，它只在 429 上被解析。见 risk-avoidance。
- **`thinking.display` 只是个显示偏好，忽略无妨**：不是。实测真实 `claude` 2.1.293 每轮都发 `display:"omitted"`，网关照发思考文本后，客户端把它存进历史、**下一轮再发回来**——一次工具循环长到 154 个 assistant 轮次、每轮都带 thinking 块并重复付输入 token。已修（commit 6568c5a）：抑制只发生在**发送边界**，思考仍参与工具兜底与 usage。
- **`output_config.effort` 需要被读取**：目前无处落地。五个上游的 thinking 都是布尔量，没有粒度可言；读它只能造假粒度。保持忽略是有意的，不是遗漏。

## 待办队列（PM 维护，从上往下做）
1. thinking 块是否带 `signature` / `signature_delta`（未验证，先查真实 Claude Code 是否要求）。
2. 工具路径的 `stop_reason` 是否会在停用词命中时被 `tool_use` 覆盖（读代码判断，不必修）。
3. `cache_control` 被解析后丢弃，`cache_*_input_tokens` 永远为 0 —— 判断是否值得填。
4. Responses 路径的 `enable_thinking` 仍是内联的 `req.thinking.get("type") == "enabled"`，没有走 `_wants_thinking`；若 Codex 或第三方客户端发 `adaptive`，那里会静默关掉思考。先确认是否有客户端这么发，再决定是否统一。