# 同枢 (UniPivot) · 本地多协议大模型网关

<p align="center">
  <b>多源模型统一转译 · 三大协议原生兼容 · 全连通自愈网格 · Codex Light 极简视觉体验</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/Protocols-OpenAI%20%7C%20Anthropic%20%7C%20Responses-success?style=flat-square" alt="Protocols">
  <img src="https://img.shields.io/badge/Architecture-Peer%20Failover%20Grid-orange?style=flat-square" alt="Architecture">
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License">
</p>

---

## 🌟 核心特性

**同枢 (UniPivot)** 是一个运行在本地的高性能、高韧性多协议转译服务。它将标准化的 **OpenAI Chat Completions**、**Anthropic Messages** 以及 **OpenAI Responses** 协议，实时转译为多源主流大模型 Web 端的内部调用。

无需申请官方商业 API Key，无需预付费或绑定信用卡，通过本网关即可直接驱动 **Claude Code CLI**、**Codex CLI**、**NextChat**、**Chatbox**、**Cursor** 等全套开发者生态工具。

```
                       ┌─────────────────────────────────────────────────────────────┐
                       │                       主流客户端与开发工具                      │
                       │   Claude Code CLI  │  Codex CLI  │  Cursor  │  NextChat    │
                       └──────────────────────────────┬──────────────────────────────┘
                                                      │ 标准请求 (OpenAI / Anthropic / Responses)
                                                      ▼
                       ┌─────────────────────────────────────────────────────────────┐
                       │               同枢网关 server.py (127.0.0.1:8000)            │
                       │ ├─ 协议转译层 (Chat Completions / Messages / Responses)      │
                       │ ├─ 全连通对等互备环 (思考环 / 旗舰环 / 极速环)                   │
                       │ ├─ 动态风控避让与冷却 (429/WAF 智能退避 30s~120s)              │
                       │ ├─ 自愈热修复大脑 (HealthMonitor / Healer / AgentRunner)     │
                       │ └─ HTTP/2 复用长连接池 (Max 100 Conns / 40 Keep-Alive)       │
                       └────────┬────────────┬─────────────┬────────────┬────────────┘
                                │            │             │            │            │
                                ▼            ▼             ▼            ▼            ▼
                         通义千问 Qwen   DeepSeek      豆包 Doubao   月之暗面 Kimi  智谱清言 GLM
```

### 1. 多源大模型矩阵
- **通义千问 (Qwen)**：全面对齐国内主站 v2 接口，支持 Qwen3.7-Max 深度思考与临时无痕会话。
- **深度求索 (DeepSeek)**：原生支持 DeepSeek-R1 深度思考推理链，基于 WebAssembly 独立沙箱求解 PoW。
- **字节豆包 (Doubao)**：模拟 PC 桌面客户端协议，免除前端签名限制，原生抽取 `<think>` 推理流。
- **月之暗面 (Kimi)**：长效 refresh_token 自动轮转 access_token，支持独立会话预创建与深度检索模式。
- **智谱清言 (GLM)**：动态 MD5 签名校验与 Assistant ID 映射，完整支持 GLM-4-Plus 与 GLM-Zero-Preview。

### 2. 三大标准协议原生兼容
- **OpenAI Chat Completions (`/v1/chat/completions`)**：流式 SSE 逐字渲染、`reasoning_content` 推理流、`tools` 结构化参数自动平衡解析、`include_usage` 统计。
- **Anthropic Messages (`/v1/messages`, `/v1/messages/count_tokens`)**：无缝对接官方 `claude` CLI（Claude Code），严格实现 `message_start` -> `content_block_delta` -> `message_stop` 状态机。
- **OpenAI Responses (`/v1/responses`)**：无缝对接新一代 `codex` CLI，支持完整的 Responses 事件时序与函数调用增量分发。

### 3. 全连通对等互备网格 (Failover Grid)
- **三级能力环调度**：划分为**思考环 (REASONING)**、**旗舰环 (FLAGSHIP)**、**极速环 (SPEED)**。同环内所有节点对等互备。
- **首包前透明切换**：在首个 Token 到达前捕获上游异常，自动无损平移至环内备用健康节点，杜绝半包脏数据。
- **自适应风控与智能冷却**：遭遇 429、WAF（阿里 RGV587 / 字节人机）时，自动触发指数避让（30s~120s 动态冷却），并在调用链中加入平滑步调与随机抖动（Jitter）。

### 4. 自动化自愈与热修复大脑 (Self-Healing Brain)
- 内置后台健康审计探针，定期巡检五大模型服务状态。
- 上游接口变动时，系统自动选举当前健康的最强模型，组装上下文诊断 Prompt，调起本地 `claude` 或 `codex` 智能体自动排查代码并运行单测，实现代码级自动热修复。

### 5. Codex Light 极简视觉工作台
- 访问 `http://127.0.0.1:8000` 即可开启内置现代化 Web UI，拥有优雅的排版排版与折叠式 `<think>` 思考过程抽屉。

---

## 🚀 快速上手

### 1. 环境准备
```bash
# 克隆仓库并进入目录
git clone <repo_url> && cd qwen

# 创建 Python 虚拟环境并安装核心依赖
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 若本机未安装系统 Chrome，安装 Playwright 浏览器内核
.venv/bin/playwright install chromium
```

### 2. 扫码免密登录（捕获会话凭证）
执行交互式登录命令，系统将调起浏览器窗口供您扫码或账号登录：
```bash
# 登录全部五大服务商
.venv/bin/python login.py all

# 或按需登录单个服务商
.venv/bin/python login.py qwen       # 通义千问
.venv/bin/python login.py deepseek   # DeepSeek
.venv/bin/python login.py doubao     # 豆包
.venv/bin/python login.py kimi       # Kimi
.venv/bin/python login.py glm        # 智谱 GLM

# 静默刷新令牌（复用已有配置，无需弹窗）
.venv/bin/python login.py all --refresh
```

### 3. 启动网关服务
```bash
./run.sh
# 或通过 uvicorn 显式启动
.venv/bin/uvicorn server:app --host 127.0.0.1 --port 8000
```

服务启动后：
- 🌐 **Web 界面**：`http://127.0.0.1:8000`
- 📚 **Swagger 接口文档**：`http://127.0.0.1:8000/docs`
- 🩺 **健康与自愈状态**：`http://127.0.0.1:8000/health`

---

## 💻 开发者生态集成示例

### 1. 接入官方 Claude Code CLI
只需配置三个环境变量，即可使用 Anthropic 官方命令行工具直接驱动底层模型：
```bash
ANTHROPIC_BASE_URL=http://127.0.0.1:8000 \
ANTHROPIC_API_KEY=test-key \
ANTHROPIC_DEFAULT_HAIKU_MODEL=claude-3-5-haiku-20241022 \
CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 \
claude -p "请用 Python 写一个红黑树的完整实现" --model claude-3-7-sonnet
```

### 2. 接入 Codex CLI (Responses API)
```bash
OPENAI_BASE_URL=http://127.0.0.1:8000/v1 \
OPENAI_API_KEY=test-key \
codex -m gpt-4o "优化当前目录下的并发网络连接池逻辑"
```

### 3. OpenAI Python SDK 调用
```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="deepseek-reasoner",  # 或 qwen3.7-max / doubao-think / kimi-explore
    messages=[{"role": "user", "content": "9.11 和 9.8 哪个数字更大？请给出严密证明。"}],
    stream=True,
)

for chunk in response:
    # 提取深度思考过程
    if hasattr(chunk.choices[0].delta, "reasoning_content") and chunk.choices[0].delta.reasoning_content:
        print(chunk.choices[0].delta.reasoning_content, end="", flush=True)
    # 提取最终回答
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

---

## 🗺️ 模型矩阵与路由映射表

### 1. 原生提供方模型与特性

| 提供方 | 原生支持模型 / 别名 | 网页端对应模型与核心能力 |
|---|---|---|
| **通义千问** | `qwen`, `qwen-max`, `qwen3.7-max`, `qwen3.6-flash` | `Qwen` (默认), `Qwen3.7-Max` (推理旗舰), `Qwen3.6-Flash` (超快低延迟) |
| **DeepSeek** | `deepseek-chat`, `deepseek-reasoner`, `deepseek-r1` | `deepseek-chat` (通用旗舰), `deepseek-reasoner` (R1 深度思考推理) |
| **豆包** | `doubao-pro`, `doubao-think`, `doubao-lite`, `doubao-expert` | `doubao-pro` (默认旗舰), `doubao-think` (思维链), `doubao-lite` (极速响应) |
| **月之暗面** | `kimi`, `kimi-explore`, `kimi-math`, `kimi-k1` | `kimi` (标准), `kimi-explore` (深度搜索与推理), `kimi-math` (数学专精) |
| **智谱清言** | `glm-4-plus`, `glm-4-flash`, `glm-zero-preview`, `glm-think` | `glm-4-plus` (旗舰), `glm-4-flash` (高吞吐), `glm-zero-preview` (零度推理) |

### 2. 跨生态别名动态映射

当客户端请求通用别名时，系统依据环境变量（`CLAUDE_BACKEND`、`OPENAI_BACKEND`，默认 `qwen`）自动分流：

| 客户端请求别名 | Qwen (默认) | DeepSeek 后端 | 豆包后端 | Kimi 后端 | GLM 后端 |
|---|---|---|---|---|---|
| `claude-3-7-sonnet` | `Qwen3.7-Max` | `deepseek-reasoner` | `doubao-think` | `kimi-explore` | `glm-zero-preview` |
| `claude-3-5-haiku` | `Qwen3.6-Flash` | `deepseek-chat` | `doubao-lite` | `kimi` | `glm-4-flash` |
| `gpt-4o` (Flagship) | `Qwen3.7-Max` | `deepseek-chat` | `doubao-pro` | `kimi` | `glm-4-plus` |
| `o1` / `o3` (Reasoner)| `Qwen3.7-Max` | `deepseek-reasoner` | `doubao-think` | `kimi-explore` | `glm-zero-preview` |
| `gpt-4o-mini` (Fast) | `Qwen3.6-Flash` | `deepseek-chat` | `doubao-lite` | `kimi` | `glm-4-flash` |

---

## 🛠️ 项目工程组织

```
├── server.py               # 网关主程序（FastAPI 路由分发、生命周期管理）
├── login.py                # 凭证自动化捕获脚本（Playwright Chrome 驱动）
├── session_store.py        # 原子化 POSIX 凭证读���与权限隔离（0o700/0o600）
├── providers/              # 模型适配与转译核心包
│   ├── base.py             # 消息展平、Token 估算、SSE 帧组装
│   ├── failover.py         # 对等互备路由网格、动态冷却避让、三级能力环
│   ├── http_client.py      # HTTP/2 连接池与单例复用
│   ├── anthropic_compat.py # Anthropic Messages 协议状态机与 ToolCall 解析
│   ├── responses_compat.py # OpenAI Responses API 协议转译
│   ├── qwen/               # 通义千问 v2 协议适配
│   ├── deepseek/           # DeepSeek 协议适配与 WebAssembly PoW 引擎
│   ├── doubao/             # 豆包 PC 客户端协议适配
│   ├── kimi/               # Kimi 会话预建与 Token 轮转适配
│   └── glm/                # 智谱清言动态签名与逻辑流适配
├── healing/                # 自愈与热修复系统
│   ├── monitor.py          # Provider 健康探测巡检与故障状态机
│   ├── healer.py           # 选举模型大脑、组装 Prompt 与调度任务
│   └── agent_runner.py     # 智能体 CLI 发现、自举安装与沙箱执行
├── scripts/                # 逆向分析与联调脚本
│   ├── reverse/            # 前端 Webpack 依赖、上行签名等逆向分析
│   └── debug/              # 单端点 HTTP 探测与回放调试
├── static/                 # Codex Light Web UI 前端资源
├── tests/                  # 离线自动化测试套件（覆盖五大 Provider 与三协议）
└── docs/                   # 架构设计与调研文档
    ├── architecture.md     # 完整系统架构设计与工程规范
    └── provider_capabilities.md # 厂商能力深度调研报告
```

---

## 🧪 自动化测试验证

本仓库配备了完全离线的单元测试套件，无需连接真实外网即可验证协议流式状态机、PoW 求解器与自愈逻辑：

```bash
.venv/bin/python tests/test_providers.py && \
.venv/bin/python tests/test_server.py && \
.venv/bin/python tests/test_session.py && \
.venv/bin/python tests/test_healing.py && \
.venv/bin/python tests/test_responses.py
```

---

## 📄 开源许可证

本项目基于 [MIT 许可证](LICENSE) 发布。仅供个人学习、技术逆向研究与自动化开发调试使用，请严格遵守各模型平台的使用条款。
