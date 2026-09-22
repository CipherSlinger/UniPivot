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
- **并发控制**：每个 Provider 配备独立的 `asyncio.Semaphore`，防止瞬时并发冲垮逆向会话。
- **动态冷却退避**：一旦捕获到 WAF 拦截或 429 报错，该节点自动进入动态隔离冷却期，后续请求自动避开该节点。

#### 3. 全局 HTTP/2 共享连接池 (`providers/http_client.py`)

- **单例协程安全管理**：采用延迟初始化与 `asyncio.Lock` 单例防护。
- **连接复用指标**：配置 `httpx.Limits(max_connections=100, max_keepalive_connections=40, keepalive_expiry=30.0)`。
- **协议升级**：自动探测并启用 `http2=True`，支持在单个 TCP 连接上复用多个并发流，显著减少 TLS 握手耗时。

#### 4. 原子凭证存储与安全隔离 (`session_store.py`)

- **存��隔离**：`session/` 目录强制应用 `0o700` 目录权限，生成的凭证文件强制应用 `0o600` 读写权限，杜绝本地其他用户读取。
- **POSIX 原子写入**：写入凭证时，先写入带有 PID 与线程 ID 的临时文件，执行 `flush` 与 `os.fsync` 后，通过 `os.replace` 原子替换目标文件，杜绝写入中断导致的文件损坏。
- **解析优先级**：
  `环境变量 (QWEN_AUTH_TOKEN 等) > 本地凭证文件 (session/*.json) > 空`

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
│   ├── failover.py             # 对等互备路由网格：三级能力环、风控元数据、动态冷却、故障转移
│   ├── http_client.py          # 全局 HTTP/2 共享连接池单例维护与优雅关闭
│   ├── anthropic_compat.py     # Anthropic Messages API 协议双向映射与 Tool Call 解析器
│   ├── responses_compat.py     # OpenAI Responses API (/v1/responses) 协议转译器
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
│   └── test_responses.py       # OpenAI Responses 协议兼容性、事件流与 ToolCall 测试
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
   - 流式响应完成后，必须通过 `asyncio.create_task(provider.delete_conversation(cid))` 在后台异步删除会话，避免在用户网页端侧边栏堆积大量临时会话。

3. **网络与风控避让规范**：
   - 严禁高并发连续向同一 Provider 刷请求。调用必须受 `CooldownTracker` 信号量与调用间歇（Pacing）约束。
   - 遇到人机验证（RGV587 / 验证码）时，立即对该 Provider 施加冷却，自动切流至备用节点，并在日志中明确提示用户执行 `python login.py <provider>` 进行人机滑块解锁。

---

## 五、测试与验证规范

所有新增功能或协议修改，必须通过自动化测试套件的严格验证：

```bash
# 执行全部单元与集成测试（离线运行，无需外网或真实凭据）
.venv/bin/python tests/test_providers.py && \
.venv/bin/python tests/test_server.py && \
.venv/bin/python tests/test_session.py && \
.venv/bin/python tests/test_healing.py && \
.venv/bin/python tests/test_responses.py
```

### 测试矩阵要求：
1. **协议正确性**：验证 OpenAI Chat Completions、Anthropic Messages、OpenAI Responses 三大协议在流式与非流式下的 JSON 结构与事件时序。
2. **容错与转译**：验证 400、401、404、429、500、502 状态码在不同协议下的标准转译。
3. **思考链完整性**：验证思维链内容能够正确抽取到 `reasoning_content`（OpenAI）与 `type: "thinking"`（Anthropic）。
4. **工具调用健壮性**：验证多工具、嵌套 JSON 参数、代码块转义等边界场景下的函数参数完整解析。
