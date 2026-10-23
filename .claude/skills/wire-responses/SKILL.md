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
  | 401 / 402 / 403 / 404 / 408 / 409 / 422 | 6 |
  | 500 / 501 / 502 / 503 / 504 | 30 |

  注意 400 不重试但用户看到的是**裸 JSON body**（`ERROR: {"error":{...}}`），
  不是可读原因；429 显示 `exceeded retry limit, last status: 429`；500 显示
  Codex 自带文案 `We're currently experiencing high demand…`。所以「换个状态码」
  都不是免费的改进（见待办 1）。
- **`x-should-retry` 是 Claude Code 的头，Codex 完全不看**：504 + 该头仍 30 次请求。
  （H7 的 7→1 是 `claude -p` 的测量，不能套到 codex 上。）

**为什么不该映射**：网关的真实失败类别（风控拦截 / 凭证失效 / 整环耗尽 / 运维禁用）
在 OpenAI 枚举里没有对应值。映射只能编，而编出来的值要么落回重试分支（等于没改），
要么被客户端当成另一个原因显示。message 本身是完整可读的上游原因，实测每个分支都会
显示它（3 个自带固定文案的 code 除外）。

## 已被证伪，不要再提
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

## 待办队列（PM 维护，从上往下做）
1. **[新，未做] 真实网关的零上游 503 仍被 Codex 重试 30 次。** 实测：
   `DISABLED_PROVIDERS` 全禁用 → 网关 503 + `x-should-retry: false` +
   `retry-after: 600` → codex 仍发 **30 次**请求（25s）。原因：Codex 不读
   `x-should-retry`，而对 503 有自己的重试策略。这不是 Responses 线协议的缺陷，
   而是「零上游确定性拒绝如何让 Codex 停下」的开放问题——可考虑：
   (a) 用 4xx（Codex 对 400 系列不重试，实测 1 次）；
   (b) 用 `response.failed` + `insufficient_quota` 之类会终止的 code（**语义会撒谎**，
       不推荐）；
   (c) 接受现状并记录。需要进一步测量 4xx 是否会让 Codex 显示正确原因。
2. **[新，未做] 网关是否可能撞上 `context_length_exceeded` 这个真有用的 code。**
   五个上游都是网页端，没有长度控制的错误码；`prompt_optimizer` 在超预算时折叠
   而不是报错。若某天上游真的返回「上下文超限」，把它映射成
   `context_length_exceeded` 是**唯一有实测收益**的映射（1 次请求 + 明确文案）。
   目前没有证据表明该路径可达，故不做。