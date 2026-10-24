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

### 长会话 soak 配方（真实二进制，零上游调用）
`/tmp/e2e/soak_server.py` 是可复用的常驻 harness：scripted provider + ASGI `Capture` 中间件，同时落三份 JSONL（`soak_requests.jsonl` 客户端原文、`soak_sse.jsonl` 网关返回的每一帧、`soak_upstream.jsonl` 真正送进上游的展平提示词）。环境变量 `SOAK_TURNS`（默认 15）/ `SOAK_TOOLS`（每轮并行调用数，默认 3）/ `SOAK_LOG_DIR` / `SOAK_PAD`（给工具输出灌填充，用来逼出折叠）/ `SOAK_FAIL_AT`（第 N 轮中途抛上游错误）。

```bash
SOAK_LOG_DIR=/tmp/e2e/soak/logs SOAK_TURNS=15 SOAK_TOOLS=3 TESTING=1 \
  GATEWAY_DISABLE_CREDENTIAL_WATCH=1 GATEWAY_DISABLE_AUTO_REFRESH=1 GATEWAY_DISABLE_PROBING=1 \
  GATEWAY_DISABLE_PACING=1 GATEWAY_DISABLE_SINGLEFLIGHT=1 \
  QWEN_AUTH_TOKEN=fake DEEPSEEK_AUTH_TOKEN=fake DOUBAO_AUTH_TOKEN=fake KIMI_AUTH_TOKEN=fake GLM_AUTH_TOKEN=fake \
  nohup .venv/bin/python -m uvicorn soak_server:app --host 127.0.0.1 --port 83xx --app-dir /tmp/e2e > /tmp/e2e/soak/server.log 2>&1 &
# 单独一次调用再启 claude，理由见上面第 (1) 条
```

**harness 自身的三个坑（都实测过，写测试前先看）**：
1. **轮次计数器必须是模块级的**。`_make_provider` 每次**派发尝试**都会被调用一次，实例属性会随每个请求重置 → 永远发 turn 0（实测 308 次上游调用）。也不能从提示词里数 `[Tool Result for …]`：折叠之后中间历史被替换成一行标记，计数会**倒退**（实测回落到 6 后无限重发 turn 2，309 次请求，EXIT=124）。
2. **工具名必须从注入的契约里读**（`^- \`name\`: …`），不要硬编码 `Bash`。Claude Code 声明的是 `Bash`，Codex 声明的是 `exec_command`；硬编码会让「声明名白名单」把调用退回成文本，看起来像「网关不发工具调用」。
3. **相邻 ```tool_call 围栏之间必须留一个 `\n`**（真实模型的输出本来就有）。`_BLOCK_TOOL_PATTERN` 的尾部 `(?:\n?```+|$)` 会把第 N 块的收尾围栏和第 N+1 块的开头围栏一起吃成 ` `````` `，中间那一块被整块吞掉：实测 `"".join(...)` → 2 次调用（丢了 `echo 1`），`"\n".join(...)` → 3 次。**这是 harness 的坑，不是产品缺陷**（真实模型不会把两个围栏贴在一起，且丢失的调用文本也一并消失，也不会变成错误调用）。

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
- **[COUNT-TOKENS-FOLD] 估算路径必须走**同一份**折叠裁决**。`/v1/messages` 在算 `input_tokens` 之前先调 `optimize_chat_messages`，超过 `GATEWAY_MAX_CONTEXT_TOKENS`（默认 24000）就把中间历史压成一行标记；`/v1/messages/count_tokens` 却直接展平原始历史，从不过滤。于是超过预算之后它报的是一个**网关已经丢弃的**上下文。实测（同一份请求回放，14 轮工具往返；以及真实 `claude` 15 轮 soak 的 16 份请求体逐份回放）：

  | turn | client_msgs | count_tokens | 真实 `message_start.input_tokens` |
  |---|---|---|---|
  | 3 | 11 | 21539 | 21539 |
  | 4 | 14 | **25820** | **21855** |
  | 5 | 17 | **30102** | **17378** |
  | 15 | 47 | **72926** | **17528** |

  第 15 轮偏 **+55400 tokens（4.2 倍）**，且差距随轮次单调放大（折叠生效前两者恒等，这正是「同一份事实两处实现」的典型指纹）。修法是把裁决做成一份、给估算路径加 `record=False`：`optimize_chat_messages(..., record=False)` 只计算不记账——估算不是派发，把它的「节省」计入折叠遥测等于声称省下了从未发生的 token。测试 `test_count_tokens_sees_the_folded_history_messages_actually_sends` 同时钉住两半（修复前红：`count_tokens=32891 vs message_start=19484（偏 13407, 1.69 倍）`；并用 `get_stats()` 前后相等钉住不记账）。

### 真实二进制 soak 结果（2026-10，clean bill of health）
四条臂，全部零上游调用（`_make_provider` 换 scripted provider），端口 8361/8362/8363/8371/8381/8391：

| 臂 | 配置 | 结果 |
|---|---|---|
| 基线 15 轮 | `SOAK_TURNS=15 SOAK_TOOLS=3` | 16 次 `/v1/messages` / 16 次上游调用，turn 0→15，`[Tool Result for …]` 计数 `0,3,6,…,45`，客户端历史 `tool_use=45 / tool_result=45` 完全配对，`EXIT=0`。24 份响应逐帧校验：`message_start` 第一、索引唯一且 `(0,text),(1,tool_use),(2,tool_use)`、`stop_reason` 在工具轮为 `tool_use`、末轮 `end_turn`、`message_stop` 最后，零重复/零孤儿 stop/零在未开块上的 delta。提示词 30398→38833 字符（≈528/轮，与对话增长成比例，无第三种膨胀源） |
| 折叠压力 | `SOAK_PAD=2400×`，同样的 15 轮 | 16 轮完成，`x-prompt-folding-applied: true`（original 72815 → final 17504 tokens）。折叠后**工具配对仍然完整**（`calls=6 / results=6`，无孤儿），`## Tools Available` 在 920 条消息下仍只出现 1 次（工具契约不重复叠加），SSE 生命周期零问题。折叠边界落在保留窗口之前，被折叠的是真实中间轮次 |
| 中途上游失败 | `SOAK_FAIL_AT=3` | 按 H4：HTTP 200 + `message_start` + 带内 `event: error {"type":"api_error","message":"upstream quota exceeded"}`，**不发 `[DONE]`**；客户端打印 `API Error: 502 upstream quota exceeded…` 并 `EXIT=1`；节点冷却 15.0s。客户端约 10 次指数退避重试（502 非确定性拒绝，这是正确行为） |
| 双 agent 并发 | 两个 `CLAUDE_CONFIG_DIR`、`SOAK_TURNS=8 SOAK_TOOLS=2` | 10 份请求体、10 份 SSE，**零**未配对工具块、**零**生命周期违规，两个 agent 都跑到 `SOAK-COMPLETE` |

**确定性拒绝头的真实二进制复核**（`DISABLED_PROVIDERS` 全环禁用 + provider 工厂 tripwire）：真实 `claude -p` 只发 **1 次**请求即放弃（`EXIT=1`），把网关给的中文原因完整打印给用户；线上响应头确认 `retry-after: 600` + `x-should-retry: false`；tripwire 文件为空 = **零上游调用**。这条把 risk-avoidance 模块的「`Retry-After` 只在 429 被解析、真正抑制重试的是 `x-should-retry`」结论在 Anthropic 路径上复验了一遍。

**已经查过、确认不是缺陷的两个疑点**：`count_tokens` 在真实 soak 里出现 **0 次**（`Counter({'/v1/messages': 30})`），所以「Claude Code 每次请求前都调它」这个前提对这个二进制/这条路径**不成立**——但数字错了依然是错的（见上），因为其它 SDK 客户端会调。以及 `<total_tokens>…</total_tokens>` 这类会话中途的 `role:"system"` 消息是 Claude Code **自己**发的（第 1 份 1 条 → 第 15 份 15 条），网关只是把它并进头部系统提示词，不是网关的膨胀源。

## 待办队列（PM 维护，从上往下做）

**上一轮已关闭（都做了实测，别再翻案）**：
- ~~thinking 是否带 `signature`~~ —— **不必带，带了反而造假**。实测：把抑制临时关掉，让网关真的发 thinking 块，真实 Claude Code 2.1.293 下一轮**原样回放**该块，形状是 `{"type":"thinking","thinking":"Let me check that.\n","signature":""}` —— 它自己把 `signature` 填成**空字符串**，且整条循环（4 轮，工具照常执行）跑到 `SOAK-COMPLETE`、`EXIT=0`。也就是说：真实上游凡是会给签名，客户端就带着签名的原值回放；我们（无签名的第三方上游）发空签名，客户端接受。我们**没有任何办法**产出一个真实签名（签名是上游对自己 token 的加密背书），伪造一个非空值只会让下一个真正带签名的上游校验失败。现状「不发 `signature_delta`、块里不带 `signature` 字段」是正确的。
- ~~停用词命中时 `stop_reason` 会不会被 `tool_use` 覆盖~~ —— **会，但现状是对的**。实测（scripted provider 先吐 ```` ```tool_call ```` 再吐 `STOP-HERE`）：`message_delta` 是 `{"stop_reason":"tool_use","stop_sequence":null}`。语义正确：停用词只截断**正文**，截断点之后的调用根本不存在，而截断点之前的调用确实成立；客户端拿到 `tool_use` 就会去执行，这是它该做的。反过来报 `stop_sequence` 会让客户端以为「模型只是被截断了」，从而丢掉一个合法的调用。同一份纪律已经由 `test_tool_call_after_terminator_is_not_executed` 从另一侧钉住（截断点**之后**的调用绝不执行）。
- ~~`cache_control` / `cache_*_input_tokens` 值不值得填~~ —— **不值得，而且不能填**。真实的 16/16 与 31/34 份请求体都带 `cache_control`，但五个上游都是网页接口，没有任何一个暴露 prompt cache 的读写计数；网关若报一个自己算出来的 `cache_read_input_tokens`，就是在编造一个客户端会拿去算钱的数字。真实 Anthropic 的这两个字段是服务端事实的回执，不是估算。保持不填（字段缺省即「无缓存数据」，Anthropic 契约允许）。

1. **[下一项] `count_tokens` 与真实请求的**其余**分叉源**：折叠已经统一（[COUNT-TOKENS-FOLD]），但两者仍各自计算：`/v1/messages` 用 `_est_tokens(flatten_messages(folded))`，`count_tokens` 用同一函数——看上去一致，可一旦 `/v1/messages` 之后又追加了什么（例如工具契约的并行约束段、`response_format` 段在两条路径上的调用顺序不同），两边就会再次漂移。**做法**：写一个属性测试，随机生成 N 份请求体（有无 tools / tool_choice 各形状 / stop_sequences / thinking / 超预算历史），对每份断言 `count_tokens(...) == message_start.usage.input_tokens`。这是一条**性质**断言，能自动抓住下一个「只改了一条路径」的回归，比再来一次人工比对更值钱。

2. **`usage.output_tokens` 的口径**：网关现在是 `max(1, _est_tokens(正文 + 思考))`，而思考被 `display:"omitted"` 抑制时仍然计入。真实 Claude Code 用 `output_tokens` 做预算。**要测的是**：这个估算与客户端自己算的差多少、是否会在长思考回合导致客户端提前压缩。做法：真实二进制 + 强制不抑制思考 + 一个长思考脚本，对比 `x-thinking-tokens` 与客户端下一轮的 `messages` 增长。

3. **流式路径的 `input_tokens` 在 failover 之后是否仍然正确**：`sse_message_start(..., input_tokens)` 在派发**之前**就算好并作为第一个事件发出；若首个候选在首包前失败、切到另一个候选，`input_tokens` 不会重算（仍是被放弃那次的值）。目前两者只差 provider 侧的工具契约长度，通常很小，但没有人量过差值。**做法**：scripted 双候选，第一个抛 401，第二个成功，断言 `message_start.usage.input_tokens` 与「第二个候选真正收到的提示词」的估算一致（或明确记录偏差可接受）。