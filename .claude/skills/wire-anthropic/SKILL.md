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

## Claude Code 2.1.293 真实发包的完整键集（19 份抓包实测）
| 键 | 出现 | 网关是否读 | 结论 |
|---|---|---|---|
| `model` / `messages` / `tools` / `max_tokens` / `stream` | 19 | 是 | — |
| `system` | 18 | 是 | 可能是 parts list，必须走 `text_of` |
| `metadata` | 18 | 是（1f8abcf 起） | **会话身份唯一来源** |
| `thinking` | 15 | 是 | 详见 6568c5a |
| `context_management` | 15 | 否 | 正确忽略（见已被证伪） |
| `output_config` | 15 | 否 | 正确忽略（`effort` 无粒度可落） |
| `safeguards` | 6（且只在首个请求） | 否 | 正确忽略（见已被证伪） |
| `conversation_id` | **0** | — | 客户端从不发它 |

## 工作方法（硬性）
- **先测量，再下结论**。真实 `claude` CLI 的探针在 `/tmp/e2e/`（见 `claudehome/`）。任何"客户端会怎样"的判断都要有实测或客户端源码依据。
- **先写红**：新测试必须能在修复前跑出**预测的那个症状**，并在测试里写清楚。
- **探针跑法**：`CLAUDE_CONFIG_DIR=/tmp/e2e/claudehome ANTHROPIC_BASE_URL=http://127.0.0.1:<port> ANTHROPIC_API_KEY=test-key ANTHROPIC_DEFAULT_HAIKU_MODEL=claude-3-5-haiku-20241022 CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 timeout 120 claude -p "..." --model claude-opus-5-5 < /dev/null`。三条实测踩过的坑：(1) `pkill -f` 会杀掉自己所在的进程组并返回 144，**启动探针服务必须单独一次 Bash 调用**，否则 `nohup` 那半句根本没执行、端口空着、`curl` 返回 000；(2) 只有 thinking-capable 的模型名（`claude-opus-5-5`、`deepseek-v4.1-flash`）才会发 `thinking`，`claude-3-5-sonnet` 一个都不发；(3) 用"提示词里出现 `[Tool Result for <id>`"判断轮次时，`<id>` 必须是**真实 id**——网关自己的工具契约文本里就写着 `[Tool Result for <id>]` 这个字面量，会误判。
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
- **`context_management` 需要网关执行**：不需要，因为客户端只发得出「保留」这一种语义。二进制的唯一构造点是 `txt(e)`：`if(hasThinking) s.push({type:"clear_thinking_20251015",keep:"all"})`——`keep:"all"` 是**最大化缓存命中**的选择，含义就是「历史 thinking 全部保留」。`clear_tool_uses_20250919` 那条走的是 `f7?.edit`，由 `Ais()` 规划，而它的开关 `$r()` 默认 `"off"`（`Wn(yd,"off")`，键 `tengu_zany_pike`）。19 份真实抓包里 `context_management` 只出现过**一种值**（`{"edits":[{"keep":"all","type":"clear_thinking_20251015"}]}`）。实测网关展平历史时**确实保留了** thinking 块（探针：`SECRET-REASONING-AAA` 出现在上游提示词里），即已满足 `keep:"all"`。客户端对 `applied_edits` 的消费者只有两个，且都只认 `type==="clear_tool_uses_20250919"`（`Gss` 累加 `cleared_input_tokens`、`wt`/`Cis` 判断 `cleared_tool_uses>0`）；二进制里 `cleared_thinking` 出现 **0 次**，所以伪造一条 thinking 清理结果是纯粹的噪声。将来若客户端真的发 `clear_tool_uses_*`（`edit` 非空），那时也**不能**在网关侧删历史：上游是无状态展平，删了就等于替客户端丢掉它仍会重发的轮次。
- **`safeguards` 需要网关回应**：不需要，而且回应反而有害。它是 auto 模式把本机分类上下文（`permission_mode` / `platform` / `cwd` / `home_dir` / `rule_roots`）交给**服务端分类器**的载体，配套的响应字段是 `safeguard_results`。网关忽略它的后果由真实二进制实测，是「优雅降级」而不是「失能」：`--permission-mode auto` 下，首轮之后客户端**不再发** `safeguards`（实测 14 次请求里第 1 次带、后 13 次都不带），并打印 `[server-classifier] a completed response carried no classification result (no safeguard_results); assuming something on the path to the API dropped it, so auto mode classifies locally for the rest of this session`；工具循环照常闭合（`touch /tmp/e2e/SG5_AUTO_TOOL_RAN` 这个 marker 文件被真实创建，即 Bash 工具确实执行了），14 轮也跑过了 `Sw=10` 的「连续 10 次无裁决就停」阈值。顺手记一笔：每条探针都会打印的 auto 模式计费提示来自 `gln()`←`gatewayHost`←`Qvr()`←`pln(co()?.url)`，判定依据是**主机名不是 `*.anthropic.com`**，与 `safeguards` 无关——它是「你走了第三方网关」这句话，任何本地网关都会看到，不代表字段被处理错了。
- **`metadata.user_id` 可以只存不用**：不行，它是 Claude Code **唯一**的会话身份（见下面的 [SESSION-ID]）。

## 已修的实现事实（改动时不要倒退）
- **[SESSION-ID] `metadata.user_id.session_id` 是 Claude Code 的会话身份**（commit 1f8abcf）。真实 2.1.293 的 19 份抓包里 `conversation_id` 出现 **0 次**，身份全在 `metadata.user_id` 这个 JSON 字符串里：`{"device_id":<跨会话恒定>,"account_uuid":"","session_id":<每会话一个 uuid>}`。此前只读 `req.conversation_id`，于是粘滞/续聊/匿名回收整条链路对真实客户端**全部空转**。实测（scripted provider，零上游调用）5 轮：上游 `conversation_id` 恒为 `None`（每轮新建会话），提示词 `20852/21643/22434/23225/24016` = 112,170 字符，而带上身份只需 `20852 + 4×10 = 20,892`（5.4 倍，且随轮数线性放大）；6 轮探针 `deletes=6`（每个活跃会话都被当成匿名临时会话排进删除队列）。修复后真实二进制 14 轮：`conversation_gc.scheduled` 14→0、`affinity_hits` 0→13、`active_sessions` 14→1。判定函数是 `resolve_client_session_id()`：`conversation_id` 优先，`metadata` 兜底，任何解析失败退回 `None`（失败开放——宁可新建一次会话，也不要把坏值当上游会话 id 回放）。

## 待办队列（PM 维护，从上往下做）
1. thinking 块是否带 `signature` / `signature_delta`（未验证，先查真实 Claude Code 是否要求）。
2. 工具路径的 `stop_reason` 是否会在停用词命中时被 `tool_use` 覆盖（读代码判断，不必修）。
3. `cache_control` 被解析后丢弃，`cache_*_input_tokens` 永远为 0 —— 判断是否值得填。
4. Responses 路径的 `enable_thinking` 仍是内联的 `req.thinking.get("type") == "enabled"`，没有走 `_wants_thinking`；若 Codex 或第三方客户端发 `adaptive`，那里会静默关掉思考。先确认是否有客户端这么发，再决定是否统一。