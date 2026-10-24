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

## 测量脚手架（可复用，别再重造）
- `/tmp/e2e/wr/mock.py`：FastAPI mock，`POST /v1/responses` 打原始 body 到
  `bodies.jsonl`，由 `/tmp/e2e/wr/ctl.json` 决定 mode/code/message/status/headers，
  **换一次测量只重启客户端，不重启服务端**（少一个变量）。
- `/tmp/e2e/wr/runone.sh` / `runone2.sh` / `runhdr2.sh`：改 ctl → 跑 `codex exec
  --skip-git-repo-check` → 报告 `requests` / `elapsed` / `exit`。
  `runone2.sh` 多一个参数可以在 `error` 事件上放一个**不同于** `response.failed` 的 code。
- `/tmp/e2e/wr/gw_server.py`：把 `server.app` 起起来并把 `_make_provider` 换成
  scripted provider，**唯一节点**（`get_fallback_candidates` 被架空），这样测的
  是网关自己的失败类别而不是 mock 仿出来的形状。
- `codex` 的 `base_url` 来自 `CODEX_HOME/config.toml`（见下方「已被证伪」）。

## 已测定的结论：`response.failed.error.code` 会被 Codex 分流

**结论：不是 display-only。** 真实 Codex 0.145.0 对这个字段有分支，但**只有 5 个值**
会改变行为，而这 5 个值网关一个都发不出来，所以**不需要加任何枚举映射**。

测量（唯一变量 = `error.code`，其余（status 200、先吐 `partial `、message 固定）完全不变）：

| `error.code` | 请求数 | 用户看到 |
|---|---|---|
| `server_error` | 6 | `stream disconnected before completion: <msg>` |
| `rate_limit_exceeded` | 6 | 同上 |
| `invalid_request_error` / `model_not_found` / `content_filter` / `unknown_*` | 6 | 同上 |
| `upstream_error` / `auth_error` / `waf_blocked` | 6 | 同上 |
| `rate_limit_error` / `upstream_timeout` / `credential_outage` / `quarantined` / `ring_exhausted` / `disabled` / `upstream_incomplete` | 6 | 同上（网关所有真实 err_type 全在这类） |
| `quota_exceeded` / `billing_error` / `insufficient_credits` / `permission_denied` | 6 | 同上（**看起来更标准的名字没用**） |
| `insufficient_quota` | **1** | `Quota exceeded. Check your plan and billing details.` |
| `usage_not_included` | **1** | `To use Codex with your ChatGPT plan, upgrade to Plus: …` |
| `context_length_exceeded` | **1** | `Codex ran out of room in the model's context window. …` |
| `invalid_prompt` | **1** | `<msg>` 原样显示，不重试 |
| `cyber_policy` | **1** | `<msg>` 原样显示，不重试 |

补充事实（都已实测）：
- **只读 `response.failed`，不读 `error` 事件**：把 `error` 事件的 code 设成
  `insufficient_quota`、`response.failed` 设成 `server_error` → 仍然 6 次重试；
  反过来 → 1 次。所以「两个事件各写一个 code」是有害的双写。
- **message 是纯展示**：把 message 写成 "context window exceeded" 而 code 是
  `server_error` → 仍然 6 次重试，只是文案变了。分支完全由 code 决定。
- **HTTP 状态码路径上 code 被忽略**，重试与否完全由状态码决定（实测，同一空 body）：

  | 状态码 | 请求数 |
  |---|---|
  | 400 | 1 |
  | 429 | 1 |
  | 401 / 402 / 403 / 404 / 408 / 409 / 422 / 425 | 6 |
  | 500 / 501 / 502 / 503 / 504 / 505 / 507 / 508 / 511 | 30 |

  注意 400 不重试但用户看到的是**裸 JSON body**（`ERROR: {"error":{...}}`），
  不是可读原因；429 显示 `exceeded retry limit, last status: 429`；500 显示
  Codex 自带文案 `We're currently experiencing high demand…`。所以「换个状态码」
  都不是免费的改进（见下方「确定性拒绝」一节与已关闭的待办 1）。
- **`x-should-retry` 是 Claude Code 的头，Codex 完全不看**：504 + 该头仍 30 次请求。
  （H7 的 7→1 是 `claude -p` 的测量，不能套到 codex 上。）二进制里连这个字符串
  都不存在。
- **`Retry-After` 对 Codex 无效**（在非 429 上）：503 + `Retry-After: 600` /
  `1` / 不带 → 三者都是 30 次请求、~25s（时间也完全一样，没有被拉长或缩短）。
  二进制里 `retry-after` 只出现在 AWS SDK 与 HTTP 头名单里，没有读取逻辑。
  429 上它的效果也被 1 次上限掩盖（三种取值都是 1 次）。
- **状态行决定重试次数，body 只在 2xx 时被解析**（本轮最重要的更正，见下节）。

**为什么不该映射**：网关的真实失败类别（风控拦截 / 凭证失效 / 整环耗尽 / 运维禁用）
在 OpenAI 枚举里没有对应值。映射只能编，而编出来的值要么落回重试分支（等于没改），
要么被客户端当成另一个原因显示。message 本身是完整可读的上游原因，实测每个分支都会
显示它（3 个自带固定文案的 code 除外）。

## 已修的实现事实：确定性零上游拒绝（Codex 侧无解，保持 503 + JSON）

**结论：当前形状（503 + `application/json` + `Retry-After` + `x-should-retry: false`）
就是诚实的最优解。** 用真实 codex-cli 0.145.0 对 loopback mock（`/tmp/e2e/wr2/`）
与服务端计数测得，**没有任何**形状能同时做到「1 次停下」+「说出真实原因」：

| 服务端形状 | 请求数 | 用户看到 |
|---|---|---|
| **503 + JSON 错误体（当前）** | **30** | 真实原因（25s 后） |
| 503 + SSE `response.failed`（一次配置失误下实测的就是这条） | 30 | `event: response.created`（流被当错误体打印） |
| 503 + SSE + `code:invalid_prompt` | 30 | 同上（code 被忽略） |
| 409 + SSE 流 | 6 | 状态码 + body 片段 |
| 400 + JSON | 1 | **裸 JSON body** |
| 400 + SSE 流 | 1 | 裸 SSE body |
| 429 + JSON | 1 | `exceeded retry limit, last status: 429` |
| 200 + SSE `response.failed` | 6 | 真实原因 |
| 200 + SSE `code:invalid_prompt` | 1 | 真实原因（但 code 是编的） |
| 503 + `Retry-After`(600/1/无) / `x-should-retry:false` | 30 | 无差别 |

三条判据，缺一不可：

1. **状态码是唯一的重试开关，body 只在 2xx 时被解析。** 同一份带
   `insufficient_quota` / `invalid_prompt` 的 `response.failed` 流：挂 200 → 1~6 次；
   挂 503 → 30 次且用户看到 `event: response.created`。所以「用 SSE 改善 5xx」不成立。
2. **能 1 次停下的 4xx 都在撒谎。** 400 = 「你的请求格式错了」（实际用户的请求完全
   合法，是网关没凭证）；429 = 「你请求太快」（重试再快也没用）。而且两者都不给
   用户看原因。
3. **能 1 次停下的 code 都是编的。** 只有 `insufficient_quota` /
   `usage_not_included` / `context_length_exceeded` / `invalid_prompt` /
   `cyber_policy` 会终止重试，语义分别是额度/套餐/上下文/提示词违规。

**重试预算是客户端配置，不是协议。** 实测 `[model_providers.<id>]` 下
`request_max_retries = 0` + `stream_max_retries = 0` → 503 从 30 次降到 **1 次**，
且用户仍看到真实原因。**写在顶层（模型根）被忽略**（仍 30 次）；只写
`stream_max_retries=0` → 5 次 / 3.6s，只写 `request_max_retries=0` → 6 次 / 7.3s
（两个旋钮分别管一层）。这是用户在客户端侧该做的决定，网关不该也不能替它做。

用例：`tests/test_responses_protocol.py::test_zero_upstream_refusal_keeps_a_json_error_on_the_status_line`
（锁 503 + JSON + 真实 `error.type` + 不伪造终止型 code + 真实 `Retry-After`；
已验证把响应改成 400 会变红，即非空洞）。
服务端实测（真实 `server.app`，`DISABLED_PROVIDERS` 全禁用）：codex → **30 次**、
`gw.log` 中 `POST /v1/responses` 30 条、**零上游调用**（35 条「禁用硬阻断」日志）。

## 已被证伪，不要再提
- **「零上游 503 能让 Codex 少重试」的任何 header/body 组合**：见上表。唯一有效的
  是客户端 `request_max_retries`/`stream_max_retries`（provider 级）。
- **`error.code` 该换成 OpenAI 标准枚举**（原 TODO）——见上表：`rate_limit_exceeded`
  与内部 err_type 行为完全一致；只有 5 个与网关失败类别不对应的值会改变行为。
- **`error_response` 做过 OpenAI 风格 code 映射**：没有。它只产出 `message`/`type`，
  从不写 `code`。`failed()` 里的 `err.get("code") or err.get("type") or err_type`
  的 `err.get("code")` 恒为假。已加用例钉住。
- **`response.incomplete` 终止事件**：真实 Codex 实测 → 5 次重连后硬失败。当前 `response.completed` 是正确选择。
- **`max_tokens`**：上游不支持长度控制，见 wire-anthropic 同一条。
- **`apply_patch` 被渲染成普通 function 工具**：已修（commit 87b1230）。真实抓包里它是 `{"type":"custom","name":"apply_patch"}`，网关渲染出 `{"input": ...}` freeform 契约，端到端实测文件真被创建。
- **`response.failed` 能减少重试**：不能。实测三种终局事件形状的重试次数都是 6。它修复的是**失败原因能否到达用户**（`stream closed before response.completed` → `upstream quota exceeded`）。
- **`gpt-5-codex` 的抓包能代表 Codex 的工具声明**：不能。该模型不在 Codex 自己的 catalog 里，会退回无 `apply_patch_tool_type` 的元数据，于是**根本不声明 `apply_patch`**。要看完整声明必须用 catalog 内的模型。
- **`x-should-retry: false` 能抑制 Codex 重试**：不能。那是 Claude Code 的头（实测 codex 504 + 该头仍 30 次请求、503 + `retry-after` + 该头仍 30 次）。

## 工作方法补充
- `codex` 的 `base_url` 来自 `CODEX_HOME/config.toml` 的 `[model_providers.*]`，**`OPENAI_BASE_URL` 环境变量被忽略**。测不同端口必须复制一份 config 改 `base_url`，否则所有测量都打在同一个旧端口上（曾因此得出「两种形状没区别」的错误结论）。
- `codex exec` 不在 git 仓库里跑要加 `--skip-git-repo-check`，否则只会打印一行提示就退出（`EXIT=0`，看起来像成功，实际一个请求都没发）。
- 请求数要从**服务端**数（mock 的 `bodies.jsonl`），不要从客户端 stdout 数。
- **测量"带状态码的 SSE"时，`StreamingResponse` 必须显式传 `status_code=`，否则它
  默认 200。** 本轮第一版 harness 漏了这一点：所有"503 + SSE"样本实际是
  "200 + SSE"，于是得出「SSE 形状能把 5xx 从 30 次降到 6 次」的错误结论——那 6 次
  是 200 分支的数字。**症状**：`grep` 到的用户输出里 body 是 `event: response.created`，
  说明客户端把整个流当错误体打印了，真 503 才会这样。修正后重测才得到正确结论
  （503 + SSE 仍是 30 次）。**先检查状态行，再相信请求数。**
- **判断"这个改动有没有让客户端停下"，要看三样东西而不是只看请求数**：
  请求数（服务端）、耗时、以及对用户说的最后一句话（`runs/<tag>.out` 末尾）。
  400 的请求数是 1，但用户看到的是一坨裸 JSON，不能算"更好的结果"。
- 当前 harness（干净、端口 8300-8349 内）：`/tmp/e2e/wr2/`
  - `mock.py` + `ctl.json`：一个端口，改 `ctl.json` 换形状，**不重启 mock**；
    支持 `http_error` / `sse_failed` / `sse_error` / `sse_created` / `ok`，
    可分别设 `status`、`code`、`error_type`、`headers`（SSE 也遵守 `status`）。
  - `run1.sh <tag> <ctl-json>`：默认 codex home（`codexhome/` → 8301）。
  - `run_nr.sh <tag> <ctl-json> [codex-home]`：换 codex home，用来测客户端旋钮
    （`ch_noretry/` 顶层旋钮＝被忽略、`ch_p0/` provider 级旋钮＝生效、
    `ch_req0/`、`ch_stream0/` 分别只关一层）。
  - `gw_server.py` + `rungw.sh`：真实 `server.app`（8302，全环 `DISABLED_PROVIDERS`），
    测的网关自己的拒绝路径，`gw.log` 计服务端请求数。
  - 一律 `127.0.0.1`；`CODEX_HOME` 的 `base_url` 指向本机端口，绝不外发。

## 待办队列（PM 维护，从上往下做）
1. **[新，未做] `error.code` 的 5 个终止值里，哪些在网关侧真的可达？**
   本轮确认 `insufficient_quota` / `usage_not_included` / `context_length_exceeded`
   在五个网页端上游都没有对应来源，`invalid_prompt` / `cyber_policy` 则可能对应
   上游的真实拒绝（内容违规）。若上游真的返回内容违规，映射成 `cyber_policy`
   是**唯一有实测收益**的映射（1 次请求 + 原样文案），且语义诚实。需要先抓一次
   真实的上游内容拒绝报文才能判断——**没有证据前不做**。
2. **[新，未做] `response.failed` 在 200 上仍有 6 次重连；能否用 `error` 事件之外的
   机制让"流中途失败"更快收敛？** 已测：`response.failed` 的 code 5 个终止值之外
   一律 6 次；`response.incomplete` 更差（5 次硬失败）。目前没有已知更好的形状。