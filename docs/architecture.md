# 本地多模型转译网关系统架构与工程规范

本文档为本地多模型转译网关（Local Multi-Protocol Model Gateway）的系统架构设计方案与工程实现规范，详细阐述多协议转译、对等互备网格、高可用韧性调度、自愈热修复大脑及各模块目录职责。

---

## 一、系统整体架构设计

网关作为一个部署在本地的无状态高性能反向代理服务，下接国内五大主流大模型网页端 Web 逆向接口，上承主流 AI 客户端开发工具与第三方生态。核心职责在于消除协议差异、隐藏逆向细节、统一错误语义，并提供对等互备容灾与自愈能力。

### 1.1 总体架构拓扑图

```
+--------------------------------------------------------------------------------------------------+
|                                           客户端接入层                                            |
|                                                                                                  |
|   +-------------------+  +-------------------+  +-------------------+  +---------------------+   |
|   |  Claude Code CLI  |  |     Codex CLI     |  | NextChat / Chatbox|  | Codex Light Web UI  |   |
|   | (Anthropic SDK)   |  | (OpenAI Responses)|  | (Chat Completions)|  | (static/index.html) |   |
|   +---------+---------+  +---------+---------+  +---------+---------+  +----------+----------+   |
+-------------|----------------------|----------------------|-----------------------|--------------+
              |                      |                      |                       |
              | POST /v1/messages    | POST /v1/responses   | POST /v1/chat/compl.  | HTTP / SSE
              +----------------------+----------+-----------+-----------------------+
                                                |
                                                v
+--------------------------------------------------------------------------------------------------+
|                                          协议转译与兼容层                                          |
|                                                                                                  |
|   +------------------------------------------------------------------------------------------+   |
|   | 统一路由分发与模型别名解析 (server.py: resolve_model / all_models)                          |   |
|   | 异常捕获与统一协议错误转译 (server.py: unified_exception_middleware / provider_error)         |   |
|   +------------------------------------------------------------------------------------------+   |
|   | Anthropic Messages 兼容引擎           | OpenAI Responses 兼容引擎   | OpenAI Chat Completions 引擎  |
|   | (providers/anthropic_compat.py)      | (providers/responses_compat)| (providers/base.py)           |
|   | - 状态机: msg_start -> delta -> stop  | - Codex 事件流与 ToolCall   | - chunk / reasoning_content   |
|   | - Tool Call 括号深度平衡与 XML 解析    | - content_part / output_item| - stream_options usage 统计   |
+---|---------------------------------------+-----------------------------+------------------------|-+
    |                                                                                              |
    v                                                                                              v
+--------------------------------------------------------------------------------------------------+
|                                         核心调度与高可用韧性层                                     |
|                                                                                                  |
|   +------------------------------------------------------------------------------------------+   |
|   | 全连通对等互备路由调度器 (providers/failover.py: FailoverRouter)                           |   |
|   |   - 三级对等能力环: 思考环 (REASONING) | 旗舰环 (FLAGSHIP) | 极速环 (SPEED)                 |   |
|   |   - 首包前透明故障转移 (Zero Half-chunk Leakage) & x-failover 元数据链路追踪                 |   |
|   +------------------------------------------------------------------------------------------+   |
|   | 动态冷却避让与自适应频控 (providers/failover.py: CooldownTracker)                            |   |
|   |   - 429 / WAF (RGV587) / 验证码智能冷却退避 (指数避让 30s~120s)                              |   |
|   |   - 调用步调平滑 (Pacing) & 随机抖动 (Jitter) & 节点并发信号量 (Semaphore)                    |   |
|   +------------------------------------------------------------------------------------------+   |
|   | 全局 HTTP/2 高性能连接池 (providers/http_client.py: get_http_client)                        |   |
|   |   - 单例共享、长连接复用 (Max 100 conns, 40 keep-alive, 30s expiry)、h2 多路复用           |   |
|   +------------------------------------------------------------------------------------------+   |
|   | 原子凭证存取与权限隔离 (session_store.py)                                                  |   |
|   |   - 优先级: 环境变量 > session/<provider>.json                                            |   |
|   |   - 跨进程 POSIX 原子重命名 (os.replace + fsync) 与 0o700/0o600 严格权限保护                |   |
|   +------------------------------------------------------------------------------------------+   |
|   | 自愈与自动热修复系统 (healing/)                                                           |   |
|   |   - HealthMonitor (monitor.py): 周期探活、连续失败计数、降级/熔断状态机                      |   |
|   |   - HealingEngine (healer.py): 选举健康模型生成修复 Prompt、调度 CLI 智能体自动化修复代码      |   |
|   |   - AgentRunner (agent_runner.py): 本地 Claude / Codex CLI 发现、自举安装与沙箱子进程运行     |   |
+---|----------------------------------------------------------------------------------------------+
    |
    v
+--------------------------------------------------------------------------------------------------+
|                                           Provider 适配层                                        |
|                                                                                                  |
|   +------------------+  +------------------+  +------------------+  +------------------+  +------+
|   | 通义千问 Qwen    |  |  DeepSeek        |  |  豆包 Doubao     |  | 月之暗面 Kimi    |  | GLM  |
|   | (providers/qwen) |  | (providers/deep) |  | (providers/doub) |  | (providers/kimi) |  | (glm)|
|   | - v2 接口 payload|  | - WebAssembly    |  | - PC 客户端模拟  |  | - 长效 refresh   |  | - MD5|
|   | - RGV587 WAF 识别|  |   PoW 引擎求解   |  |   绕过 a_bogus   |  |   换取 access    |  |   动态|
|   | - 临时无痕会话   |  | - 补丁式增量 SSE |  | - SSE_ACK 追踪   |  | - 独立会话预创建 |  |   签名|
+---+------------------+--+------------------+--+------------------+--+------------------+--+------+
    |                      |                      |                      |                      |
    v                      v                      v                      v                      v
 阿里通义主站           DeepSeek 官网           字节豆包官网           月之暗面 Kimi 平台      智谱清言平台
 www.qianwen.com        chat.deepseek.com       www.doubao.com         kimi.moonshot.cn       chatglm.cn
```

---

## 二、核心分层与系统机制详解

### 2.1 客户端层与三协议转译机制

网关原生支持三种客户端调用标准，通过 `server.py` 实现统一分发：

1. **OpenAI Chat Completions 协议 (`/v1/chat/completions`)**：
   - **请求模型解析**：接受 OpenAI 格式的 `messages`、`tools`、`stream`、`temperature` 等参数。
   - **无延迟逐字流式**：在未声明 `tools` 时，以最快速度透传 SSE `chat.completion.chunk` 增量，包含深度思考内容字段 `reasoning_content`。
   - **Tool Calling 动态解析**：利用 `providers/anthropic_compat.py` 内置的高性能括号平衡解析器，在上游输出完整内容后精准提取 JSON 函数入参并构造 `finish_reason: "tool_calls"`。
   - **Token 计量规范**：支持 `stream_options: {"include_usage": true}`，在流结束前吐出符合规范的 usage 统计块。

2. **Anthropic Messages 协议 (`/v1/messages`, `/v1/messages/count_tokens`)**：
   - **严格时序状态机**：流式推送严格遵循 Anthropic 标准时序：
     `message_start` -> `content_block_start` -> `content_block_delta` -> `content_block_stop` -> `message_delta` -> `message_stop`。
   - **思考与文本流分离**：当上游模型（如 DeepSeek-R1、Doubao-Think、Qwen-Think）输出思维链时，自动开启 `type: "thinking"` 块，并在思维结束后无缝平移到 `type: "text"` 块。
   - **双向模型列表兼容**：`GET /v1/models` 依据请求头中是否存在 `anthropic-version` 自动在 OpenAI List 模型格式与 Anthropic 模型数组格式间切换。

3. **OpenAI Responses 协议 (`/v1/responses`)**：
   - **面向新一代 Agent 研发**：专为 Codex CLI、OpenAI Agent SDK 以及新一代推理客户端设计。
   - **Responses SSE 事件序列**：严格按顺序分发 `response.created` -> `response.output_item.added` -> `response.content_part.added` -> `response.text.delta` -> `response.content_part.done` -> `response.output_item.done` -> `response.completed` -> `data: [DONE]`。
   - **Function Calling 流式细粒度更新**：支持 `response.function_call_arguments.delta` 与 `done` 事件。

---

### 2.2 核心调度与高可用韧性层

#### 0. 在途请求优雅停机排空 (`server.py:InflightTrackerMiddleware`)

- **纯 ASGI 计数**：`InflightTrackerMiddleware` 精确统计在途请求数，因其 `await self.app(...)` 仅在响应体（含流式）全部发送完毕后才返回，故流式全程计数保持不降。
- **关停排空**：`lifespan` 关停阶段先有界等待在途请求收尾（`GATEWAY_SHUTDOWN_DRAIN_TIMEOUT`，默认 30s），再停止健康探针与释放连接池，杜绝重启腰斩半传输的流式响应（客户端重试会成倍放大上游负载与风控压力）。
- **遥测**：`inflight_requests` 经 `/health` 与 `/v1/diagnostics` 暴露。

为应对逆向工程固有的上游网络不稳定、WAF 人机滑块风控、429 频率受限等挑战，系统在调度层实现了多重韧性屏障：

#### 1. 全连通对等能力互备环 (`providers/failover.py`)

五大国内模型（通义千问、DeepSeek、豆包、Kimi、智谱 GLM）处于完全对等地位，划分为三级能力环：

| 对等能力环 | 核心能力特征 | 包含模型成员（按优先级与风险度动态排队） |
|---|---|---|
| **思考环 (REASONING)** | 深度思维链、复杂代码架构与复杂数学演算 | `deepseek-reasoner`, `kimi-explore`, `Qwen3.7-Max`, `doubao-think`, `glm-zero-preview` |
| **旗舰环 (FLAGSHIP)** | 旗舰全能、百万上下文、复杂多轮指令遵循 | `Qwen3.7-Max`, `glm-4-plus`, `kimi`, `doubao-pro`, `deepseek-chat` |
| **极速环 (SPEED)** | 毫秒级低延迟响应、日常补全与轻量任务 | `Qwen3.6-Flash`, `doubao-lite`, `glm-4-flash` |

**透明故障转移保证**：
- 在流式响应的首个数据包（First Token）向客户端下发前，执行试探性握手。
- 若初选节点抛出 401（未授权）、429（限流）、5xx、WAF 拦截（如阿里 `RGV587`、字节人机）或网络异常，调度器无缝捕获并迅速调度同能力环中的下一个健康候选节点。
- 绝不产生半包脏数据输出（Zero Half-chunk Leakage），并在元数据中打上 `x-failover-from`、`x-failover-to` 链路标记。

#### 2. 自适应风控避让与频控抖动 (`CooldownTracker`)

针对逆向接口频率特征，配置动态限流元数据：

```python
PROVIDER_RISK_SPECS = {
    "qwen":     RiskLevel.HIGH,   min_interval=0.30s, jitter=(0.05, 0.15), max_concurrency=3, cooldown=45s
    "deepseek": RiskLevel.HIGH,   min_interval=0.40s, jitter=(0.05, 0.20), max_concurrency=2, cooldown=60s
    "kimi":     RiskLevel.HIGH,   min_interval=0.35s, jitter=(0.05, 0.15), max_concurrency=2, cooldown=45s
    "glm":      RiskLevel.LOW,    min_interval=0.10s, jitter=(0.01, 0.05), max_concurrency=5, cooldown=30s
    "doubao":   RiskLevel.LOW,    min_interval=0.08s, jitter=(0.01, 0.05), max_concurrency=6, cooldown=30s
}
```

- **Pacing & Jitter**：在发出请求前，自动平滑计算调用间隙并施加正态随机抖动，破坏固定调用频率指纹。
- **证据驱动自适应放大 [OPT-014]**：调用间隔依据该节点近期风控拦截证据自适应放大——`waf`（人机验证）> `auth` > `ratelimit` 加权，每单位权重线性放慢 25%，封顶 `GATEWAY_PACING_MAX_MULTIPLIER`(默认 2.5x)。刚触发过验证码/429 的节点会被主动「降速避让」，直至其 6h 证据窗口自然衰减。证据查询带 60s TTL 缓存，`GATEWAY_DISABLE_ADAPTIVE_PACING=1` 可退化为固定间隔。
- **并发控制**：每个 Provider 配备独立的 `asyncio.Semaphore`，防止瞬时并发冲垮逆向会话。
- **动态冷却退避**：一旦捕获到 WAF 拦截或 429 报错，该节点自动进入动态隔离冷却期，后续请求自动避开该节点。

#### 3. 全局 HTTP/2 共享连接池 (`providers/http_client.py`)

- **单例协程安全管理**：采用延迟初始化与 `asyncio.Lock` 单例防护。
- **连接复用指标**：配置 `httpx.Limits(max_connections=100, max_keepalive_connections=40, keepalive_expiry=30.0)`。
- **协议升级**：自动探测并启用 `http2=True`，支持在单个 TCP 连接上复用多个并发流，显著减少 TLS 握手耗时。

#### 4. 原子凭证存储与安全隔离 (`session_store.py`)

- **存储隔离**：`session/` 目录强制应用 `0o700` 目录权限，生成的凭证文件强制应用 `0o600` 读写权限，杜绝本地其他用户读取。
- **POSIX 原子写入**：写入凭证时，先写入带有 PID 与线程 ID 的临时文件，执行 `flush` 与 `os.fsync` 后，通过 `os.replace` 原子替换目标文件，杜绝写入中断导致的文件损坏。
- **解析优先级**：
  `环境变量 (QWEN_AUTH_TOKEN 等) > 本地凭证文件 (session/*.json) > 空`

#### 5. 风控拦截持久化台账与冷却状态跨重启恢复 (`providers/risk_ledger.py`)

- **冷却状态持久化**：将各 Provider 的冷却截止时间戳与连续失败计数以 POSIX 原子写（`os.replace` + `os.fsync`，目录 `0o700` / 文件 `0o600`）落盘，进程重启后自动恢复避让，杜绝「重启即重置避让、立刻再次触碰已被上游风控封禁的节点」从而人为延长封禁窗口。
- **拦截台账 (JSONL)**：每一次风控 / WAF 人机验证 / 认证失效拦截均以 JSONL 追加落盘（时间、节点、HTTP 状态、严重度、原因、证据摘要），形成可跨重启审计的长期风控画像，超过 `GATEWAY_RISK_LEDGER_MAX_LINES` 自动裁剪保留后一半。
- **严重度分级**：`auth > ratelimit > waf` 三级。认证失效仅在连续 ≥3 次后按 2 的幂次升级为长效冷却（`GATEWAY_AUTH_COOLDOWN_BASE` × 2ⁿ，封顶 `GATEWAY_AUTH_COOLDOWN_MAX=86400s`），首次失败仍维持既有短冷却，避免误锁刚完成重登的健康节点。
- **遥测**：`GET /v1/diagnostics` 暴露 `risk_ledger` 段，`GET /health` 输出 24h 汇总；`GATEWAY_DISABLE_RISK_LEDGER=1` 可无损退化为纯内存冷却。

#### 6. 风控感知健康探针门控 (`healing/monitor.py`)

- **冷却即静默**：`HealthMonitor` 每次周期探针前先查询 `CooldownTracker` 剩余冷却；处于避让冷却期的节点直接标记为 `cooling` 并**跳过真实网络探针**，杜绝「每 300s 一次探针持续触碰已封禁节点」而人为延长封禁窗口。
- **风控与故障解耦**：冷却期探针既不累计 `consecutive_failures` 也不触发自愈，避免把风控封禁误判为节点故障而派发无效热修复风暴。
- **遥测**：`/v1/diagnostics` 的 `healing_status.cooldown_probe_skips` 输出各节点被门控跳过的探针计数；`GATEWAY_DISABLE_PROBE_COOLDOWN_GATE=1` 可无损回退。

#### 6.1 读取时冷却叠加与自愈大脑选举避让 (`healing/monitor.py:get_status` / `healing/healer.py:pick_healer_model`)

- **问题**：`ProviderHealth.status` 默认即为 `healthy`。在首个探针周期完成前（或进程刚重启、冷却状态已由 `risk_ledger` 恢复但尚未探针覆盖时），处于风控避让冷却期的节点依然对外报告 `healthy`。自愈大脑选举 `pick_healer_model` 仅按 `status == "healthy"` 过滤，会把**已被封禁的节点选为修复引擎**，进而经本网关发出一次真实上游补全请求——这正是整条冷却门控体系要杜绝的隐蔽触碰通道，且优先级表第二位即为 DeepSeek。
- **读取时叠加**：`get_status()` / `get_all_statuses()` 在返回前叠加实时冷却查询结果（`status` 覆盖为 `cooling` 并附带 `cooldown_remaining`）。叠加**不落盘、不粘滞**，冷却解除后自动消失，因此对所有状态消费方（自愈选举、状态面板、诊断接口）一次性生效。
- **选举避让**：`pick_healer_model` 先对全部候选取一次冷却快照，两轮遍历共享同一视图（避免重复统计），冷却节点一律排除并计入 `cooldown_healer_skips`；当除故障节点外**已无可用候选**时返回 `None`，由上层「缺少健康备选模型」分支安全推迟自愈，而非强行触碰封禁节点。
- **总开关**：`GATEWAY_DISABLE_COOLDOWN_GATE=1` 统一关闭探针门控、状态叠加与选举避让（`GATEWAY_DISABLE_PROBE_COOLDOWN_GATE` 保留为向后兼容别名）。

#### 7. 启动期令牌静默刷新的风控冷却门控 (`server.py:_auto_refresh_stale_sessions`)

- **冷却即暂缓**：启动时的静默令牌刷新先查询 `CooldownTracker`，处于风控避让冷却期的节点**暂缓刷新**——对已被上游封禁的账号发起登录/刷新请求同样会被风控记录并加剧封禁。
- **有界延迟补刷**：被暂缓的节点交由后台线程在冷却解除后补做刷新（每 `GATEWAY_REFRESH_DEFER_POLL`(默认 300s) 检查一次，最多等待 `GATEWAY_REFRESH_DEFER_MAX_WAIT`(默认 24h)），超时放弃本轮，**绝不无限阻塞**，避免「跳过即永久放弃」的功能回退。

#### 8. 远端临时会话删除的持久化重试队列 (`providers/conversation_gc.py`)

- **远端防膨胀**：无痕调用（调用方未提供 `conversation_id`）会在上游网页端留下临时会话，网关在响应结束后删除之。历史实现仅 `asyncio.create_task(provider.delete_conversation(cid))` 即发即忘：删除失败或进程重启即永久丢失该动作，用户侧边栏持续堆积临时会话。
- **持久化队列**：待删除会话以 POSIX 原子写（`os.replace` + `os.fsync`，目录 `0o700` / 文件 `0o600`）落盘，进程重启后自动恢复，并由启动排空 + 周期后台 drain 重试。
- **有界重试**：失败累计尝试次数，达到 `GATEWAY_CONVERSATION_GC_MAX_ATTEMPTS`（默认 5）后放弃并记日志，杜绝无界重试与队列无界增长。
- **冷却门控 [OPT-017]**：删除同样是一次真实上游请求，对已封禁节点发起删除等同持续触碰该账号。处于风控避让冷却期的节点其删除动作被**推迟**（记录保留在队列且**不消耗重试次数**），冷却解除后由后台 drain 自然补做；门控查询异常时绝不阻断回收（零回归默认）。
- **遥测**：`GET /v1/diagnostics` 暴露 `conversation_gc` 段（待删除数、成功/失败/重试/放弃计数、`deferred_by_cooldown` 冷却推迟计数）；`GATEWAY_DISABLE_CONVERSATION_GC=1` 可无损退化为历史即发即忘行为，`GATEWAY_DISABLE_CONVERSATION_GC_COOLDOWN_GATE=1` 可单独回退冷却门控。

#### 9. 运维长期封禁租约 (`providers/quarantine.py`)

- **问题**：自动风控冷却存在**上限**（认证类封顶 `GATEWAY_AUTH_COOLDOWN_MAX`，默认 24h），而账号级封禁（如 DeepSeek 已触发的封禁）往往持续**数天**。上限到期后网关会重新触碰仍处于封禁窗口的节点，人为延长封禁。
- **租约**：`QuarantineStore` 提供**运维可显式设定**的长期封禁租约，以绝对到期时间持久化落盘（POSIX 原子写，目录 `0o700` / 文件 `0o600`），进程重启后自动恢复，到期惰性失效。
- **只延长不缩短**：租约读取端取 `max(自动冷却, 人工租约)`，且 `set_lease` 拒绝比现有租约更早的到期时间——误设的短租约绝不可能提前放行一个正在封禁的节点。
- **单一事实来源**：租约折入 `CooldownTracker.get_remaining_cooldown()` / `is_in_cooldown()`，因此**所有**既有门控自动遵循，无需逐处接线：健康探针 (OPT-012)、令牌刷新 (OPT-013)、临时会话回收 (OPT-017)、自愈大脑选举 (OPT-018)、路由选点与自适应步调 (OPT-014)。
- **配置与运维**：`GATEWAY_QUARANTINE="deepseek:72h,glm:30m,kimi:2026-10-20T00:00:00"` 启动即生效（支持 `s/m/h/d` 后缀与 ISO-8601 绝对时间）；`GET/POST/DELETE /v1/admin/quarantine` 供运行期查看、设定与解除；`GATEWAY_DISABLE_QUARANTINE=1` 关闭；`GATEWAY_QUARANTINE_DIR` 覆盖落盘位置。
- **遥测**：`/health` 暴露生效中的租约（`quarantine`），`/v1/diagnostics` 暴露完整租约诊断（生效数、明细、统计）。解除人工租约**不会**连带清除真实风控产生的自动冷却。

#### 10. 封禁租约的派发硬阻断 (`providers/failover.py:is_quarantined` / `execute_chat`)

- **问题（实证）**：OPT-020 把租约折入**冷却读数**，因而门控（探针/刷新/会话回收/自愈选举）全部避让；但**派发路径**并未门控——`get_fallback_candidates` 仅把冷却节点**沉底而非排除**，`execute_chat` 更是无条件派发主选节点。实测证明：处于 72h 封禁租约的 `deepseek` 与 `doubao` 在故障转移中**仍被派发真实上游请求**，且该通道随失败次数放大——这是整条风控体系中最宽的隐蔽触碰通道。
- **语义区分（关键设计）**：软冷却与硬封禁必须分开处理。
  - **软冷却**：由自动风控触发、时长有界（30s~`GATEWAY_AUTH_COOLDOWN_MAX`），是可自愈的短暂退避——环内全部冷却时允许「最后一搏」，以保可用性。
  - **硬封禁**：运维长期租约，表达「该账号已被平台封禁」，持续以天计，任何触碰都不可能有正常返回，只会延长封禁——**任何情况下都不派发**。
- **三道闸口**：① 负载均衡对封禁主选执行同环分流（且绝不分流到封禁节点）；② `get_fallback_candidates` 彻底排除封禁节点；③ `execute_chat` 在派发前逐轮硬校验，命中即改派或失败。
- **快速失败**：环内全部封禁时返回 `503 / err_type="quarantined"`，**零上游调用**，并在错误信息中给出解除指引（`POST /v1/admin/quarantine`）；Anthropic 协议映射为 `overloaded_error`。
- **开关**：`GATEWAY_DISABLE_QUARANTINE_HARD_STOP=1` 退回「仅降级」语义。

#### 11. 避让解除后的恢复观察期 (`providers/failover.py:is_in_probation` / `compute_pacing_interval`)

- **问题**：避让解决的是「不要再触碰被封禁的节点」，但没有解决「不要再把账号打进封禁」。一次多日的封禁解除时，数天来被冷却压制的需求会在同一瞬间释放——这正是当初触发封禁的流量特征，会让账号**立即二次封禁**。避让做得越好，积压越集中，解封瞬间的洪峰越危险。
- **观察期（金丝雀放行）**：避让 episode 结束后的首次派发（风控冷却到期、或封禁租约自然到期/被运维显式解除）使该节点进入**恢复观察期**（`GATEWAY_PROBATION_WINDOW`，默认 300s）。窗口内：
  - **串行探路**：同一时刻至多 **1 个握手**在途，把并发洪峰压成逐个试探；
  - **步调放大**：调用间隔乘以 `GATEWAY_PROBATION_PACING_FACTOR`（默认 2.0）。
- **毕业（[OPT-028] 证据驱动，而非时间驱动）**：累计 `GATEWAY_PROBATION_MIN_SUCCESSES`（默认 3）次**连续成功**即毕业，恢复常规并发与步调。窗口**只是续期单位**：耗尽一个窗口即自动续期一次（最多 `GATEWAY_PROBATION_MAX_RENEWALS`，默认 3，即最多门控 `窗口 × 4`），续期期间串行门控与放大步调**继续有效**；续期预算耗尽后仍按**证据**裁决——有正面证据（至少一次成功握手）即毕业，零证据则保持门控。续期次数由「已流逝窗口数」推导（纯函数、无副作用、无需持久化），因此重启、乱序查询都不改变结论。
- **为什么必须改**：窗口原先是**硬截止**，到点即放开全并发，无论是否拿到过证据。低流量节点（尤其夜间）在 300 秒里往往连一次握手都没有，于是保护恰好在最需要它的场景下静默失效——随后抵达的需求是**全并发**打向一个**仍可能处于封禁中**的账号，而这正是当初触发封禁的形态。对「从未被触达」的节点而言继续门控**不产生任何代价**（没有请求到来就无所谓洪峰），放开却等于把一个完全未经验证的账号交给全并发。`GATEWAY_PROBATION_MAX_RENEWALS=0` 恢复正常。零证据被持续门控的节点仍可通过**后来取得的成功**毕业，门控不会死锁（`GATEWAY_PROBATION_WINDOW=0` 仅表示「不给时间宽限」，完整关闭请用 `GATEWAY_DISABLE_PROBATION=1`）。
- **闸门只覆盖握手到首包**，不占用整段流式输出；且**在并发信号量之前**获取，避免多个请求先占住并发槽位再排队等闸门（否则观察期形同虚设）。
- **组合而非替换**：放大倍率作用在 `compute_pacing_interval()` 内，与 OPT-014 的证据驱动放大倍率**相乘**（`基础间隔 × 证据倍率 × 观察期倍率`），并抽成纯函数以便确定性测试。
- **零回归**：普通 5xx/超时属上游故障而非账号被标记，解封瞬间不存在惊群风险，**不触发**观察期。
- **跨重启生效**：episode 结束时间随风控台账持久化（`episode_ends`），保留期覆盖**续期后**的完整视野，重启即绕过观察期的路径被堵死。
- **开关与遥测**：`GATEWAY_DISABLE_PROBATION=1` 关闭；响应头 `x-risk-probation: 1`，`/v1/diagnostics` 暴露 `probation.active` 与 `probation.stats`（含派生的 `renewing` / `awaiting_first_handshake` / `budget_exhausted`，使「观察期是否真的在起作用」可被观测）。

#### 12. 风控拦截全量档案与规避策略研判 (`providers/risk_dossier.py`)

- **问题**：`RiskLedger.recent_interceptions()` 是**滑动窗口**（默认 24h），适合驱动实时避让，却不适合回答运维最关心的问题——「这家到底被拦过几次、都是什么原因、能不能绕开、重登有没有用」。各家封禁窗口以**天**计，24h 窗口几乎必然在封禁解除前就把证据丢光，而证据一旦滑出窗口就再也无法复盘。
- **全量档案**：以**全量台账**（受 `GATEWAY_RISK_LEDGER_MAX_LINES` 有界约束）聚合每个节点的累计拦截次数、首/末次时间、严重度分布，以及**最近一次的真实上游原文**，使封禁历史可审计、可复盘。
- **姿态合成（fail-closed）**：把分散在四处的事实——封禁租约、软冷却、恢复观察期 (OPT-022)、证据驱动的步调倍率 (OPT-014)——合成为**单一姿态**，命中即取最重者：`quarantined` > `cooling` > `probation` > `watch` > `clear`，并附「是否可派发」。运维读一个字段即可判断节点此刻的真实处境。仅硬封禁不可派发；软冷却保持可派发，与 OPT-021 的「最后一搏」语义一致。
- **规避研判（直接回答「能否绕开」）**：按严重度给出确定性结论——`technically_bypassable`（一律为否：任何客户端手段都无法绕开平台侧风控）、`recoverable_by_relogin`、`auto_recovers`、典型持续时长与具体策略。
  - `auth`（令牌/会话失效）与 `waf`（人机验证 RGV587）→ **重登可恢复**（`python login.py <provider>`）；
  - `ratelimit` → 瞬时自动恢复，重登无意义；
  - `ban`（账号级封禁）→ **重登无效**：封禁作用在账号本身而非令牌，换令牌/换会话/换指纹均不可能解除，任何尝试只会延长封禁，唯一策略是零误触等待平台自然解封。
  - 持有租约的节点**一律**按 `ban` 研判，不受台账严重度影响。
- **只读性**：查询档案**绝不**改动任何风控状态（不解除租约、不清冷却、不写台账），因此可安全暴露在诊断路径上。
- **接口与开关**：`GET /v1/admin/risk`（支持 `?provider=qwen` 单节点视图）返回完整档案与策略矩阵；`/v1/diagnostics` 仅暴露 `risk_dossier_summary` 汇总，避免诊断接口膨胀。`GATEWAY_DISABLE_RISK_DOSSIER=1` 标记为关闭。
#### 13. 跨节点风控关联与系统性升级研判 (`providers/risk_correlation.py`)

- **问题（单账号视角的盲区）**：OPT-011~OPT-023 的全部避让逻辑都以「单个 Provider 账号」为单位——该账号被拦，就冷却它、封禁它、观察它。但风控拦截存在第二种成因：**我们自己的客户端足迹**（请求头指纹、UA、设备参数、出口 IP、时间分布）被平台标记。此时各账号被拦并非彼此独立，而是同一根因的多个表现。
- **两种成因需要完全相反的应对**：
  - **单账号成因**：「换一个节点继续打」是**正确**的——其他账号不受影响。
  - **系统性成因**：换节点**毫无用处**——新节点会以同样的足迹被立即拦下，继续轮换只会逐个把剩余账号也送进封禁，是此时**最危险**的操作。
  - 单账号视角永远看不到这一层：它把每个账号的拦截都解释为「这家风控严」，于是一个接一个地消耗掉全部五个账号。
- **关联性以去重节点数衡量**（核心设计）：同一账号被拦 100 次仍是单账号问题；**多个不同账号**在同一个突发子窗口内被集中拦截，才几乎不可能由各自独立的原因导致。以双指针滑动窗口计算窗口内任意 `GATEWAY_RISK_CORRELATION_BURST_WINDOW`（默认 120s）子窗口的去重节点峰值，复杂度 O(n)。
- **分级（强信号优先）**：突发内去重节点数 ≥ `GATEWAY_RISK_CORRELATION_SYSTEMIC_MIN`（默认 3）→ `systemic`；否则窗口内（`GATEWAY_RISK_CORRELATION_WINDOW`，默认 900s）去重节点数 ≥ `GATEWAY_RISK_CORRELATION_WIDESPREAD_MIN`（默认 3）→ `widespread`；否则 ≥1 → `localized`；无 → `none`。时间集中与时间分散的区别在于前者几乎必然是共享足迹，后者可能是各自独立的原因，因此只轻量放大并提示人工确认。
- **全局步调治理**：`systemic` 对所有节点统一乘以 `GATEWAY_RISK_CORRELATION_SYSTEMIC_FACTOR`（默认 2.0），`widespread` 乘以 `GATEWAY_RISK_CORRELATION_WIDESPREAD_FACTOR`（默认 1.25）。该倍率作为**第四个因子**作用于 `CooldownTracker.compute_pacing_interval()`（基础 × 证据 × 观察期 × 全局）与 `TrafficPacer`，与 OPT-014、OPT-022 相乘而非替换。因步调路径调用频繁，倍率带 TTL 缓存（`GATEWAY_RISK_CORRELATION_CACHE_TTL`，默认 15s）。
- **零回归**：无关联证据时倍率恒为 **1.0**；台账不可用亦降级为 1.0，绝不影响请求流程。研判全程只读，不写台账。
- **策略矩阵**：每一级都显式声明 `rotate_node_helps` 与 `per_account_avoidance_helps`，使「换节点是否有用」成为可查询的确定性结论而非经验判断。
- **接口与开关**：`GET /v1/admin/risk/correlation` 返回完整研判，`/v1/diagnostics` 暴露 `risk_correlation`。`GATEWAY_DISABLE_RISK_CORRELATION=1` 关闭。

#### 14. 全环凭证失效检测与快速失败 (`providers/credential_watch.py`)

- **现场证据（真实台账，纠正了此前的误判）**：`session/risk_ledger.jsonl` 显示 80 分钟内五个节点被记录 335 次拦截，其中 **`auth` 类占 312 次**，`waf` 类极少——也就是说这次故障**不是**风控封禁，而是整环凭证失效。更关键的是：五个节点在**同一秒**内同时返回 401（19:36:11、19:44:22…），每约 7 分钟重复一次。五个节点在同一秒各自独立过期是概率上不可能的事，真实过程是：**一个客户端请求沿能力环依次故障转移，五个节点全部以 401 拒绝**，于是单个请求在 1 秒内产生 5 条 auth 记录。整环容量为零，故障转移不可能成功。
- **风暴本身就是风控触发器（本机制的核心判断）**：环内无可用凭证时，网关仍会为每个客户端请求打满五个节点。80 分钟 335 次带无效凭证的失败请求，恰好是上游 WAF 眼中**「自动化凭证尝试」的流量特征**——这个风暴本身就在持续拉高风控评分，很可能是账号被封的成因之一。因此正确行为不是「更努力地重试」而是**立即停止**。这一点是**单节点冷却无法表达**的：每个节点各自冷却，整环仍然会被走一遍。
- **与既有机制的语义区别**：软冷却=单节点、有时长上限、自动解除，其余节点仍可服务；封禁租约 (OPT-020/021)=单节点、运维声明、以天计、任何情况不派发；凭证失效=**整环**、无需人工声明、凭证更新即刻恢复。它不是「某个节点有问题」，而是「我们没有可用凭证了」，所以处置方式是**重登而非避让**。
- **判定依据是去重节点数**（同 OPT-024，而非失败次数）：`GATEWAY_CREDENTIAL_OUTAGE_MIN_PROVIDERS`（默认 3）个**不同**节点在 `GATEWAY_CREDENTIAL_OUTAGE_WINDOW`（默认 300s）内相继认证失败才进入失效态；单节点失败 100 次不构成整环失效。
- **派发边界快速失败**：检查点位于 `execute_chat` 派发循环的**最前**（在 OPT-021 封禁硬阻断之前、且在**任何 Provider 实例化之前**），因此请求以 `503 / err_type="credential_outage"` 被拒绝且**零上游调用**（Anthropic 协议映射为 `authentication_error`——Anthropic 枚举中无「凭证故障」，此为语义最贴近者）。错误文案直接点名失效节点并给出重登指引。
- **绝不卡死**：状态带 TTL（`GATEWAY_CREDENTIAL_OUTAGE_TTL`，默认 120s），到期后放行**恰好一次**探路；若整环仍在 401，首次失败即立刻重建失效态。于是探测频率由「每个客户端请求一次」退化为「每 TTL 一次」，风暴被削平，同时不可能因状态卡死而永久拒绝服务。任一节点成功（`record_success` → `note_success`）即从失效集合中摘除，容量回到阈值以上**立刻恢复派发**（fail-open）。
- **开关与遥测**：`GATEWAY_DISABLE_CREDENTIAL_WATCH=1` 关闭；`GATEWAY_DISABLE_RISK_LEDGER=1` 一并关闭（与封禁租约同约定，保证纯内存冷却语义的测试套件不受影响）。`/v1/diagnostics` 暴露 `credential_outage`，`GET /v1/admin/credential-outage` 查看状态、`DELETE` 在重登后立即解除。

#### 15. 凭证续期检测与认证级避让复位 (`providers/credential_renewal.py`)

- **风暴之后的 24 小时尾巴（为什么 OPT-025 还不够）**：`record_failure` 对认证失效按 `GATEWAY_AUTH_COOLDOWN_BASE`(1800s) × 2ⁿ 长效升级，封顶 `GATEWAY_AUTH_COOLDOWN_MAX`(86400s)。真实台账里五个节点的认证失败次数是 54/54/54/87/60——**全部撞到 24 小时上限**，而且该状态随台账持久化、跨重启恢复。于是风暴结束后，五个节点**各自被冻结整整一天**。OPT-025 解决了风暴期间的重复派发，但没有解决风暴之后的冻结。
- **惩罚的对象为什么消失了**：长效升级惩罚的是「拿着**同一个失效凭证**反复打」——这确实该罚。但凭证一旦续期，这个行为就不再存在，继续冻结惩罚的已不是坏行为，而是**已经修好的系统**：冷却读数仍为 24h，导致健康探针跳过该节点、自愈大脑选举将其排除、路由降级排序；最直接的问题是 `server.py:_deferred_refresh_after_cooldown`——**令牌静默刷新本身也被冷却门控挡住，最多推迟 24 小时**。也就是说「凭证失效」制造的状态，正好锁死了唯一能自动修复它的机制。反向验证：OPT-023 档案对 `auth` 类的结论是 `recoverable_by_relogin = True`，系统一面告诉运维「重登可恢复」，一面在重登之后继续避让 24 小时，结论与行为自相矛盾。
- **判定「已续期」靠内容指纹**：在记录认证失败的那一刻对凭证来源（`session/<provider>.json`）取指纹——`sha256(token|cookie|captured_at|user_id|device_id)`，不可解析时退化为 `meta:mtime_ns:size`。指纹随冷却一并持久化，因此「先重登、再重启」同样能被识别。`Session.save()` 以 `os.replace` 原子落盘且每次捕获都会刷新 `captured_at`，任何一次真实重登都必然改变指纹。检测器**只回答事实**（凭证变了吗），状态变更由 `CooldownTracker.release_auth_avoidance()` 执行，两者分离使判定路径只读、可安全查询。
- **复位严格限定成因（fail-closed）**：仅当该节点**最近一次**失败严重度为 `auth` 时才复位（成因随冷却持久化）；成因记录缺失时（本机制上线前遗留的冷却，如真实台账里那五条 24h 冻结）退化为**只读**查询风控台账中该节点最近一次拦截的成因，仍保持成因限定。持有**运维封禁租约**的节点永不复位——租约表达账号级封禁，重登无效，与「凭证过期」是两回事；`waf`/`频控` 成因也绝不被代为释放。无记录、凭证不可读、指纹相同，一律判定「未续期」。复位后节点经 OPT-022 恢复观察期重新进入：被压制数日的需求在凭证刚更新时集中释放，正是观察期要防的洪峰。
- **解锁修复链**：复位使 `get_remaining_cooldown()` 归零，从而解除 OPT-013 令牌静默刷新的冷却门控——唯一自动修复路径因此恢复可用。启动时执行一次扫描（`await cooldown_tracker.release_renewed_credentials()`），因为「重登 + 重启」是最常见的运维流程。
- **接口与开关**：`GATEWAY_DISABLE_CREDENTIAL_RENEWAL=1` 关闭；`GATEWAY_DISABLE_RISK_LEDGER=1` 一并关闭（与封禁租约 / 凭证失效检测同约定）。`/v1/diagnostics` 暴露 `credential_renewal`；`GET /v1/admin/credential-renewal` 查看指纹记录与判定结果；`POST /v1/admin/credential-renewal/scan` 触发复位，请求体 `{"providers":[...]}` 用于**凭证来自环境变量**（无文件可比指纹）的部署；`DELETE /v1/admin/credential-outage` 在清除全环失效态的同时**连带复位**这些节点的认证级避让——该接口的语义本就是「我已续期并修好」，否则接口承诺不兑现。


---

#### 16. 账号级封禁推断与自解除自动租约 (`providers/failover.py` / `providers/quarantine.py`)

- **OPT-026 留下的循环**：OPT-026 在发现凭证续期时复位认证级避让——这是对的；但**如果复位后节点依然 401**，网关就会重新派发、再次吃 401、再次按 2 的幂次升到 24h，下一次扫描又复位，循环往复，且每一轮都触碰一次账号。这正是本项目约束所禁止的形态：被封节点必须**零触碰**，而不是被周期性地试探。
- **401 的两种成因，处置完全相反**：(1) 凭证过期 → 处置是重登（`python login.py <provider>`），OPT-026 已复位；(2) 账号级封禁 → 重登**彻底无效**（封禁绑定账号而非令牌），正确处置是零触碰等待平台解除。把 (2) 当成 (1) 处理，就是用重试循环不断重新触发封禁。因此本机制的核心是**判别式**，而网关恰好已经拥有它：**「凭证已被证明地更新过，而节点仍被拒绝」**——这排除了成因 (1)，只剩成因 (2)。这也是运维问题「现在受到风控的 LLM 能不能通过重新登录恢复使用」的机器化形式：对 deepseek 而言答案是否定的。
- **证据按凭证重置**：`CredentialRenewalTracker.note_auth_failure()` 现在额外返回一个事实——本次失败所依据的凭证**是否与它上次记录的不同**；判定必须发生在**覆盖旧记录之前**，否则证据每次调用都会自毁。`CooldownTracker` 以 `_fresh_credential_rejected[provider]` 计数，且每当出现更新的凭证即**重置为 1（而非累加）**：每枚续期过的凭证都获得属于自己的机会，避免把上一枚凭证的失败累加到一枚从未被真正试过的新凭证头上。累计达 `GATEWAY_BAN_INFERENCE_MIN_FAILURES`（默认 2）即施加**自动租约**。
- **与既有四种避让语义的区别**：软冷却=网关对**负载**的推断；运维租约=人类对**策略**的声明；凭证失效=**整环**的容量问题；自动租约=网关对**账号本身**的推断。它在 `QuarantineStore` 中标记 `source="auto"` 并随租约持久化（目录 `0o700` / 文件 `0o600`），重启无法把「自动」洗成「不可解除」；`release_auto_lease()` 在结构上**不可能**解除运维租约，优先级单向（运维声明永远覆盖自动标记，反之绝不；来源缺失一律按运维租约保守处理）。时长 `GATEWAY_AUTO_QUARANTINE_DURATION`（默认 259200s = 72h），短于人工设置的多日租约——推断得到的结论不应拥有与人类声明同等的权威。
- **必然自解除：推断必须可被推翻**：自动租约的全部正当性就是「我们试过一枚新凭证，它仍被拒」；一旦出现**更新的**凭证，这条证据即告失效，自动租约必须被解除、证据清零，节点经 OPT-022 恢复观察期重新验证。没有这一点，一次错误推断会把一个其实已经修好的账号冻结 72 小时。运维租约豁免：永不被自动解除，指纹比对逻辑也不触碰它。
- **零触碰派发**：自动租约复用 OPT-021 的硬阻断（租约折入冷却读数，使探针、刷新、会话回收、自愈选举、路由**无需逐处接线**即全部遵循），因此推断出的封禁在**任何路径**上都不产生上游调用，包括作为故障转移候选时。`GATEWAY_DISABLE_BAN_INFERENCE=1` 关闭推断；与租约/台账开关联动关闭（否则推断结论无处安放）。证据、自动租约指纹与拒斥计数均跨重启存活，重启不构成绕过入口。租约来源（`auto` / `operator`）随租约一并暴露在 OPT-023 风控档案的 `quarantine.source` 字段中，使「推断封禁」与「运维声明封禁」对运维可区分。
- **接口与遥测**：`/v1/diagnostics` 暴露 `ban_inference`（含 `rejections_after_renewal` 与 `auto_leases`），`GET /v1/admin/ban-inference` 查看报告，`DELETE /v1/admin/ban-inference` 供运维在确认平台已解封后**提前**解除自动租约（该接口对运维租约返回空集，绝不误伤）。

---

### 2.3 自愈与自动热修复系统 (`healing/`)

系统内置故障诊断与自动化热修复闭环系统，当上游 Web 协议更新导致服务异常时自动执行自愈：

```
+-------------------+      心跳失败 / 连续报错      +--------------------+
|   上游网页端协议   | ------------------------> |   HealthMonitor    |
|   发生变更失效    |                           | (healing/monitor)  |
+-------------------+                           +---------+----------+
                                                          |
                                                          | 触发告警 (Trigger)
                                                          v
+-------------------+     通过当前健康的最强模型    +--------------------+
|  AgentRunner 执行 | <------------------------ |   HealingEngine    |
| (claude / codex)  |    组装自愈 Prompt 提示词  | (healing/healer)   |
+---------+---------+                           +--------------------+
          |
          | 沙箱执行代码审查、排查逆向字段、修改 providers/* 并运行 pytest 单测
          v
+-------------------+      复测通过恢复 healthy      +--------------------+
|  tests 自动化验证  | ------------------------> | 恢复节点上线服务   |
+-------------------+                           +--------------------+
```

1. **健康探测器 (`HealthMonitor`)**：
   - 维护 Provider 的状态机：`healthy` -> `degraded` -> `recovering` -> `offline`。
   - 周期性发送轻量最小化探针，连续失败达阈值（默认 2 次）后标记故障并自动触发自愈调度。
2. **决策引擎与提示词工程 (`HealingEngine`)**：
   - 选举大脑：从当前剩余的 healthy 节点中选出最强模型（如 `qwen3.7-max` 或 `deepseek-chat`）作为修复驱动大脑。
   - 组装专业自愈 Prompt：注入报错异常信息、网络抓包特征、目标实现文件路径与单测命令。
3. **执行驱动器 (`AgentRunner`)**：
   - 自动发现本地环境中的 `claude` 或 `codex` CLI。若环境缺失，支持通过 npm 进行后台自举安装。
   - 调起非交互式子进程运行修复任务，任务日志记录于 `logs/healing_<provider>_<timestamp>.log`。
   - 任务完成后自动复测，复测通过则无缝切回 `healthy`。

---

## 三、项目目录组织与职责规范

项目遵循正规工程化目录规划，所有模块各司其职，目录结构树如下：

```
/Users/jianye/Desktop/qwen/
├── server.py                   # FastAPI 网关主入口：路由注册、协议分发、生命周期管理
├── login.py                    # 认证与令牌捕获工具：基于 Playwright 驱动系统 Chrome 登录
├── session_store.py            # 凭证存储中心：POSIX 原子锁写入、权限控制与环境变量解析
├── run.sh                      # 本地生产启动脚本
├── requirements.txt            # Python 运行时核心依赖清单
├── CLAUDE.md                   # 针对 Claude Code CLI 的项目指南与操作规程
├── README.md                   # 项目整体介绍、快速上手指南与集成参考
├── static/                     # 前端静态资源
│   ├── index.html              # Codex Light Web UI：极简交互界面、<think> 抽屉、流式渲染
│   └── _shots/                 # 界面截图与视觉展示资源
├── docs/                       # 架构与调研文档
│   ├── architecture.md         # [本文档] 系统整体架构设计与工程开发规范
│   └── provider_capabilities.md# 国内五大模型网页端能力深度全景调研报告
├── providers/                  # 提供方核心实现与协议转译包
│   ├── __init__.py             # 统一导出各 Provider 类与公共工具函数
│   ├── base.py                 # 基础抽象类、消息展平、Token 估算、SSE 帧构建、ProviderError
│   ├── failover.py             # 对等互备路由网格：三级能力环、风控元数据、动态冷却、故障转移、封禁硬阻断、恢复观察期、凭证失效快速失败、认证级避让复位
│   ├── http_client.py          # 全局 HTTP/2 共享连接池单例维护与优雅关闭
│   ├── anthropic_compat.py     # Anthropic Messages API 协议双向映射与 Tool Call 解析器
│   ├── responses_compat.py     # OpenAI Responses API (/v1/responses) 协议转译器
│   ├── risk_ledger.py          # 风控拦截持久化台账：严重度分级、JSONL 审计、冷却状态跨重启恢复
│   ├── conversation_gc.py      # 远端临时会话删除的持久化重试队列（跨重启防泄漏、有界重试）
│   ├── quarantine.py           # 运维长期封禁租约：跨重启持久化、只延长不缩短、来源可区分（运维/自动）、折入全局冷却门控
│   ├── risk_dossier.py         # 风控全量档案与姿态研判：拦截历史可审计、规避策略与重登有效性结论、租约来源可辨
│   ├── risk_correlation.py     # 跨节点风控关联：系统性升级研判与全局步调治理
│   ├── credential_watch.py     # 全环凭证失效检测：去重节点阈值、TTL 探路、零上游调用快速失败
│   ├── credential_renewal.py   # 凭证续期检测：session 凭证内容指纹、认证级避让成因限定复位、封禁租约免疫
│   ├── session_affinity.py     # 会话亲和与跨模型状态移交引擎（LRU/TTL、异构上游会话 ID 剥离）
│   ├── singleflight.py         # 单飞去重与流式扇出引擎（SHA-256 指纹、Leader/Follower 广播）
│   ├── token_counter_cache.py  # Token 计量结果缓存
│   ├── disconnect_guard.py     # 客户端断连守护：上游请求随之取消，杜绝无效负载
│   ├── tool_streamer.py        # 流式工具调用推测解析与 JSON 自愈修复
│   ├── prompt_optimizer.py     # 提示词折叠优化与遥测头
│   ├── json_repair.py          # 截断/畸形 JSON 的自愈修复解析器
│   ├── prompt_cache.py         # 提示词前缀缓存
│   ├── load_balancer.py        # 自适应负载均衡器（结合冷却状态选择节点）
│   ├── qwen/                   # 通义千问国内版逆向适配模块
│   │   ├── __init__.py
│   │   └── provider.py         # v2 API 封装、RGV587 WAF 人机识别、无痕会话管理
│   ├── deepseek/               # DeepSeek 逆向适配模块
│   │   ├── __init__.py
│   │   ├── provider.py         # 会话生命周期、SSE 补丁增量解析、模型映射
│   │   ├── pow.py              # Wasmtime 沙箱内运行官方 sha3_wasm_bg.wasm 求解 PoW
│   │   └── sha3_wasm_bg.wasm   # DeepSeek 官方 PoW WebAssembly 二进制算法包
│   ├── doubao/                 # 豆包逆向适配模块
│   │   ├── __init__.py
│   │   └── provider.py         # 模拟 PC 客户端协议、SSE_ACK 处理、增量文本提取
│   ├── kimi/                   # 月之暗面 Kimi 逆向适配模块
│   │   ├── __init__.py
│   │   └── provider.py         # 长效 refresh_token 换取 access_token、预创建会话、流式解析
│   ├── glm/                    # 智谱清言 GLM-4 逆向适配模块
│   │   ├── __init__.py
│   │   └── provider.py         # 动态 MD5 签名校验、Assistant ID 映射、推理链抽取
│   ├── qwen_provider.py        # 向后兼容导入垫片 (Shim)
│   ├── deepseek_provider.py    # 向后兼容导入垫片 (Shim)
│   ├── doubao_provider.py      # 向后兼容导入垫片 (Shim)
│   ├── kimi_provider.py        # 向后兼容导入垫片 (Shim)
│   └── glm_provider.py         # 向后兼容导入垫片 (Shim)
├── healing/                    # 自愈与自动热修复系统
│   ├── __init__.py             # 包导出定义
│   ├── monitor.py              # 运行健康监控器：周期探活、状态机维护、故障报警
│   ├── healer.py               # 自愈决策引擎：选举健康模型、生成修复 Prompt、任务分发
│   └── agent_runner.py         # 智能体环境发现、自动安装与非交互沙箱执行器
├── scripts/                    # 辅助脚本工具库
│   ├── reverse/                # 逆向工程分析脚本（开发者专用）
│   │   ├── analyze_js.py       # 自动化下载并分析前端 Webpack JS 包
│   │   ├── find_module.py      # 扫描特定功能函数与 API 调用点
│   │   ├── inspect_batch_del.py# 批量会话清理逻辑逆向
│   │   ├── inspect_chat_page.py# 页面动态加载行为监听
│   │   ├── inspect_details.py  # 详细请求体与响应协议嗅探
│   │   ├── inspect_ui.py       # 前端 DOM 结构与验证码容器检查
│   │   ├── inspect_uplink.py   # 上行流量特征与 Header 加解密检查
│   │   ├── inspect_webpack.py  # Webpack Chunk 依赖关系提取
│   │   └── record_chat_req.py  # 实时聊天流量抓包记录
│   └── debug/                  # 调试与联调脚本
│       ├── test_net.py         # 底层网络连通性与代理状态检测
│       ├── test_exact_fetch.py # 精确回放真实浏览器请求头与凭证
│       ├── create_test_conv.py # 针对特定 Provider 快速创建测试会话
│       └── test_api_endpoints.py# 单端点 HTTP 交互快速探测
├── session/                    # 凭证持久化目录（git-ignored，0o700 安全权限）
│   ├── qwen.json               # 通义千问凭证（tongyi_sso_ticket / Cookie / User-Agent）
│   ├── deepseek.json           # DeepSeek 凭证（userToken Bearer / Cookie）
│   ├── doubao.json             # 豆包凭证（sessionid / device_id / Cookie）
│   ├── kimi.json               # Kimi 凭证（refresh_token / Cookie）
│   ├── glm.json                # 智谱 GLM 凭证（chatglm_token / Cookie）
│   └── profiles/               # Playwright Chromium 独立持久化浏览器环境目录
├── tests/                      # 自动化测试套件（完全离线 Mock，无需真实 Token）
│   ├── test_providers.py       # 五大 Provider 协议序列化、SSE 流解析与 PoW 求解测试
│   ├── test_server.py          # 服务端路由、OpenAI/Anthropic 协议转译与错误格式测试
│   ├── test_session.py         # 凭证原子存取、权限掩码、环境变量覆盖优先级测试
│   ├── test_healing.py         # 自愈监控状态机、Prompt 生成、任务调度与单测验证
│   ├── test_responses.py       # OpenAI Responses 协议兼容性、事件流与 ToolCall 测试
│   ├── test_tools.py           # 流式工具调用、推测流式与 JSON 自愈引擎测试
│   ├── test_affinity.py        # 会话亲和、跨模型状态移交、LRU/TTL 与遥测头测试
│   ├── test_singleflight.py    # 单飞指纹、流式广播扇出与去抖合并测试
│   ├── test_risk_ledger.py     # 风控台账严重度分级、原子落盘与跨重启冷却恢复测试
│   ├── test_probe_gate.py      # 冷却感知健康探针门控测试
│   ├── test_refresh_gate.py    # 冷却感知令牌刷新门控与有界延迟补刷测试
│   ├── test_adaptive_pacing.py # 证据驱动自适应步调放大测试
│   ├── test_graceful_drain.py  # 在途请求优雅停机排空测试
│   ├── test_conversation_gc.py # 临时会话删除持久化队列与有界重试测试
│   └── test_quarantine.py      # 运维长期封禁租约：解析、落盘恢复、只延长与全门控遵循测试
│   └── test_quarantine_hardstop.py # 封禁租约的派发硬阻断：零触碰、快速失败与软冷却语义区分测试
│   └── test_probation.py       # 避让解除后的恢复观察期：串行探路闸门、步调放大、证据驱动毕业（窗口续期）与跨重启持久化测试
│   └── test_risk_dossier.py    # 风控全量档案：窗口外证据保留、姿态 fail-closed、只读性与重登恢复结论测试
│   └── test_risk_correlation.py # 跨节点风控关联：去重节点语义、系统性分级、全局步调压制与只读性测试
│   ├── test_credential_watch.py # 全环凭证失效：去重节点阈值、TTL 探路与永不卡死、零上游调用快速失败与双协议错误语义测试
│   └── test_credential_renewal.py # 凭证续期：内容指纹判定、24h 认证级冻结复位、租约/WAF 免疫、解锁刷新门控与跨重启持久化测试
│   └── test_ban_inference.py   # 账号级封禁推断：新凭证仍被拒判别式、普通过期绝不误判、自解除自动租约、运维租约优先与零触碰派发测试
├── examples/                   # 调用示例
│   ├── python_client.py        # Python SDK 调用示例
│   └── curl.sh                 # 常用 curl 请求调用示例
└── logs/                       # 运行时自愈与排错日志（git-ignored）
```

---

## 四、安全与风控避让规范

1. **凭证隔离规范**：
   - 严禁将 `session/*.json` 与 `session/profiles/` 提交到版本控制系统中。
   - 所有会话保存操作必须经过 `Session.save()` 方法，确保权限锁定为 `0o600`，外层目录为 `0o700`。

2. **会话生命周期与远端防膨胀规范**：
   - 当调用方未提供 `conversation_id` 时，网关视其为单次无状态调用。
   - 对于支持无痕模式的厂商（如通义千问），请求体默认注入 `temporary: true`。
   - 流式响应完成后，必须通过 `conversation_gc.schedule(provider, provider_key, cid)` 在后台异步删除会话，避免在用户网页端侧边栏堆积大量临时会话。**严禁**回退为裸 `asyncio.create_task(provider.delete_conversation(cid))`：删除失败或进程重启会永久丢失该删除动作，导致临时会话在上游侧边栏无界堆积。删除动作必须进入 `providers/conversation_gc.py` 的持久化队列，跨重启存活并由后台有界重试。

3. **网络与风控避让规范**：
   - 严禁高并发连续向同一 Provider 刷请求。调用必须受 `CooldownTracker` 信号量与调用间歇（Pacing）约束。
   - 遇到人机验证（RGV587 / 验证码）时，立即对该 Provider 施加冷却，自动切流至备用节点，并在日志中明确提示用户执行 `python login.py <provider>` 进行人机滑块解锁。
   - 冷却避让必须跨进程重启持续生效：所有冷却截止时间与连续失败计数经 `providers/risk_ledger.py` 原子落盘，重启后自动恢复，严禁通过重启「抹除」避让状态而重新触碰已封禁节点。
   - 账号级风控封禁无法通过技术手段绕开：唯一正确策略是识别、退避并在封禁窗口内零误触，等待平台自然解封；重登仅能恢复「令牌/会话失效」一类，对账号级封禁无效。
   - 健康探针同样必须遵守冷却门控：处于避让冷却期的节点禁止发起任何真实上游探针（含自愈复检），杜绝「周期探针」成为持续触碰已封禁节点的隐蔽通道。
   - 启动期静默令牌刷新同样必须遵守冷却门控：处于避让冷却期的节点暂缓刷新，冷却解除后由有界后台线程补刷，杜绝「重启即刷新」成为封禁窗口的隐形延长器。
   - 临时会话删除同样必须遵守冷却门控：处于避让冷却期的节点暂缓删除其临时会话（记录保留在持久化队列且不消耗重试次数），冷却解除后补做，杜绝「会话回收」成为封禁窗口的又一条隐蔽触碰通道。
   - 自愈大脑选举同样必须遵守冷却门控：驱动修复的模型别名最终会经本网关转发为一次真实上游调用，冷却期节点一律不得作为修复引擎；若已无可用候选，必须安全推迟自愈而非强行触碰封禁节点。节点状态在**读取时**叠加实时冷却（而非依赖已被探针覆盖的落盘状态），杜绝「重启后未探针的冷却节点被误判为 healthy」。
   - 自动冷却存在**时长上限**（认证类封顶 `GATEWAY_AUTH_COOLDOWN_MAX`，默认 24h），而账号级封禁可持续数天；因此长于冷却上限的封禁**必须**通过运维长期租约表达：`providers/quarantine.py`（`GATEWAY_QUARANTINE` 环境变量或 `POST /v1/admin/quarantine`），租约经 `CooldownTracker.get_remaining_cooldown()` 折入全局冷却读数，使**所有**既有门控自动遵循。租约只延长不缩短，误操作不可能提前放行封禁节点；解除人工租约**不得**连带清除真实风控产生的自动冷却。
   - **硬封禁必须在派发路径上阻断，而不仅是被门控避让**：折入冷却读数只影响「主动发起的旁路请求」（探针/刷新/回收/自愈），**不覆盖**正常业务派发。因此 `execute_chat` 必须在派发前校验 `is_quarantined`，`get_fallback_candidates` 必须**排除**（而非仅沉底）封禁节点，负载均衡不得把任何请求分流到封禁节点。环内全部封禁时必须**快速失败（503 quarantined）且零上游调用**，严禁以「环内只剩它」为由破例触碰。
   - **必须区分「单账号成因」与「系统性成因」，严禁把后者当作前者处理**：多账号被拦的成因有二——各账号自身风控（按账号避让 + 换节点有效），或我们自己的客户端足迹（请求头指纹/UA/设备参数/出口 IP/时间分布）被平台标记（避让与换节点**均无效**，换节点只会以同样足迹把剩余账号逐个送进封禁）。关联性必须以**去重节点数**而非拦截次数衡量：同一账号被拦百次仍是单账号问题，而多个不同账号在同一突发窗口内被集中拦截则几乎不可能彼此独立。判定为系统性时，必须整体放大全局调用步调并明确提示「换节点无效」；任何面向运维的输出都不得暗示继续轮换节点。
   - **风控拦截必须留下可审计的全量档案，且规避结论不得含糊**：滑动窗口只服务实时避让，封禁历史以「天」计，窗口滑过即不可复盘。因此每家节点的拦截历史、严重度分布与最近一次上游原文必须可从全量台账聚合查询（`GET /v1/admin/risk`）。规避研判必须给出确定性结论而非模棱两可的建议：账号级封禁**不可绕开、重登无效**，只能零误触等待平台解封，严禁在任何面向用户或运维的文案中暗示「重登可解封」；人机验证与令牌失效才可由重登恢复。档案查询必须**只读**，严禁因查询而解除租约或清除冷却。
   - **整环凭证失效必须快速失败，严禁以「重试」制造无效凭证风暴**：当多个**不同**节点在同一窗口内相继返回 401，说明我们没有可用凭证，故障转移不可能成功；此时继续为每个客户端请求逐个打满五个节点，恰恰构成上游 WAF 眼中的「自动化凭证尝试」流量特征——**风暴本身就是风控触发器**。因此派发路径必须在**任何上游调用之前**检测并拒绝（`503 / err_type=credential_outage`，Anthropic 映射为 `authentication_error`），并明确提示执行 `login.py all` 重登。该状态必须带 TTL 自动失效以免永久拒绝服务（到期放行单次探路），且任一节点成功即立即恢复派发。严禁将「整环凭证失效」与「单节点软冷却」「运维封禁租约」混为一谈：前者的处置是重登而非避让，且无需人工声明。
   - **认证级避让必须可被「凭证续期」解除，严禁让修复机制被自己制造的状态锁死**：认证失效按 2 的幂次长效升级（封顶 24h）本意是惩罚「拿着同一个失效凭证反复打」，但凭证一旦续期惩罚对象即消失，继续冻结只会惩罚**已修好的系统**——探针跳过、自愈选举排除、路由降级，甚至把 OPT-013 的令牌静默刷新（唯一自动修复路径）一并推迟 24 小时。因此必须依据**凭证内容指纹**（`session/<provider>.json` 的 token/cookie/captured_at 哈希）在失败时刻留痕、在续期后判定并复位。复位必须**成因限定且 fail-closed**：仅 `auth` 成因可释放，持有运维封禁租约的节点永不释放（账号级封禁重登无效），`waf`/频控成因不得代为释放，无记录/凭证不可读/指纹相同一律判定「未续期」；复位后必须经恢复观察期重新进入。
   - **账号级封禁必须可被「新凭证仍被拒」推断出来，且推断出的租约必须自解除**：401 有两种成因，处置完全相反——凭证过期（正确处置是重登，OPT-026 已复位）与账号级封禁（重登**无效**，只能零触碰等待）。判别式是「**凭证已被证明更新过之后仍被拒绝**」：它排除了前一种解释。若缺少该推断，OPT-026 的复位会与其后的再次 401 形成循环（复位→派发→401→重新升级到 24h→再复位），每轮都触碰一次处于封禁中的账号，正是必须杜绝的行为。推断产出的租约必须标记来源（`source="auto"`）并持久化，与运维声明在结构上可区分：运维声明永远覆盖自动标记、永不被自动解除；推断出的租约在出现**更新**的凭证时必须自动解除（其正当性完全来自那枚被拒的凭证，证据失效则租约失效），并经受观察期重新验证——一次错误推断不得永久冻结一个已经修好的账号。推断结论同样必须走派发硬阻断，被封节点零上游调用。
   - **避让解除后必须经过恢复观察期，严禁解封即满速**：避让只保证「不触碰」，不保证「不再被打进封禁」。多日避让期间被压制的需求会在解封瞬间同时释放，复现当初触发封禁的流量特征。因此任一避让 episode 结束（冷却到期、租约到期或被解除）后，该节点必须先进入恢复观察期：串行放行单个探路握手（并发峰值恒为 1）并放大步调间隔，累计若干次**连续成功**后才毕业恢复常规并发。**毕业条件必须由证据而非时间决定**：时间窗口只能作为续期单位，耗尽即续期（`GATEWAY_PROBATION_MAX_RENEWALS`），严禁以「窗口到点」为由把从未被成功验证过的节点放开到全并发——低流量节点在窗口内连一次握手都没有是常态，那样只会让保护在最需要它的场景下静默失效。观察期状态必须随风控台账持久化且保留期覆盖续期后的完整视野，严禁通过重启绕过。普通 5xx/超时（上游故障，非账号被标记）不得触发观察期。

---

## 五、测试与验证规范

所有新增功能或协议修改，必须通过自动化测试套件的严格验证：

```bash
# 执行全部单元与集成测试（离线运行，无需外网或真实凭据）
.venv/bin/python tests/test_providers.py && \
.venv/bin/python tests/test_server.py && \
.venv/bin/python tests/test_session.py && \
.venv/bin/python tests/test_healing.py && \
.venv/bin/python tests/test_responses.py && \
.venv/bin/python tests/test_tools.py && \
.venv/bin/python tests/test_affinity.py && \
.venv/bin/python tests/test_singleflight.py && \
.venv/bin/python tests/test_risk_ledger.py && \
.venv/bin/python tests/test_probe_gate.py && \
.venv/bin/python tests/test_refresh_gate.py && \
.venv/bin/python tests/test_adaptive_pacing.py && \
.venv/bin/python tests/test_graceful_drain.py && \
.venv/bin/python tests/test_conversation_gc.py && \
.venv/bin/python tests/test_quarantine.py && \
.venv/bin/python tests/test_quarantine_hardstop.py && \
.venv/bin/python tests/test_probation.py && \
.venv/bin/python tests/test_risk_dossier.py && \
.venv/bin/python tests/test_risk_correlation.py && \
.venv/bin/python tests/test_credential_watch.py && \
.venv/bin/python tests/test_credential_renewal.py && \
.venv/bin/python tests/test_ban_inference.py
```

### 测试矩阵要求：
0. **测试隔离与零污染**：所有测试套件必须设置 `TESTING=1`（或显式关闭持久化），使风控台账、拦截 JSONL 与会话回收队列落于临时目录，**严禁污染开发者真实 `session/` 目录**。由于 `CooldownTracker` 会从进程内共享的台账单例恢复冷却状态，未隔离的套件会因用例执行顺序不同而产生非确定性结果——纯内存冷却语义的用例（如 `tests/test_failover.py`）应显式设置 `GATEWAY_DISABLE_RISK_LEDGER=1`，跨重启持久化语义则由 `tests/test_risk_ledger.py` 专测。
1. **协议正确性**：验证 OpenAI Chat Completions、Anthropic Messages、OpenAI Responses 三大协议在流式与非流式下的 JSON 结构与事件时序。
2. **容错与转译**：验证 400、401、404、429、500、502 状态码在不同协议下的标准转译。
3. **思考链完整性**：验证思维链内容能够正确抽取到 `reasoning_content`（OpenAI）与 `type: "thinking"`（Anthropic）。
4. **工具调用健壮性**：验证多工具、嵌套 JSON 参数、代码块转义等边界场景下的函数参数完整解析。
