# CLAUDE.md

This file provides comprehensive guidance to Claude Code (claude.ai/code) and Codex CLI when working with code in this repository.

## Commonly Used Commands

### Environment & Dependencies
```bash
# Set up virtual environment and install dependencies
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# If system Chrome is not installed, install Playwright Chromium
.venv/bin/playwright install chromium
```

### Running the Gateway
```bash
# Start the FastAPI server on 127.0.0.1:8000
./run.sh
# Or directly via uvicorn:
.venv/bin/uvicorn server:app --host 127.0.0.1 --port 8000
```

### Authentication & Token Management (Five Major Models)
```bash
# Interactive login to capture session tokens (opens browser)
.venv/bin/python login.py all        # Log in all providers (qwen, deepseek, doubao, kimi, glm)
.venv/bin/python login.py qwen       # Log in Tongyi Qianwen only (Aliyun SSO / WAF)
.venv/bin/python login.py deepseek   # Log in DeepSeek only (Bearer token / Cloudflare)
.venv/bin/python login.py doubao     # Log in Doubao only (sessionid / device tokens)
.venv/bin/python login.py kimi       # Log in Kimi only (refresh_token)
.venv/bin/python login.py glm        # Log in Zhipu GLM only (chatglm_token)

# Silent token refresh (headless, reuses existing browser profiles)
.venv/bin/python login.py all --refresh
.venv/bin/python login.py qwen --refresh
.venv/bin/python login.py deepseek --refresh
.venv/bin/python login.py doubao --refresh
.venv/bin/python login.py kimi --refresh
.venv/bin/python login.py glm --refresh
```

### Running Tests
Tests are standalone Python scripts using mocked transports (completely offline, no external test runner or network access needed):
```bash
# Run all unit and integration tests
.venv/bin/python tests/test_providers.py && \
.venv/bin/python tests/test_server.py && \
.venv/bin/python tests/test_session.py && \
.venv/bin/python tests/test_healing.py && \
.venv/bin/python tests/test_responses.py && \
.venv/bin/python tests/test_tools.py && \
.venv/bin/python tests/test_affinity.py

# Run individual test files
.venv/bin/python tests/test_providers.py   # Provider protocols, SSE chunk parsing, and PoW wasm loading
.venv/bin/python tests/test_server.py      # Server routing, OpenAI/Anthropic error transformation, and SSE streaming
.venv/bin/python tests/test_session.py     # Session persistence, POSIX atomicity, file permissions, and env priority
.venv/bin/python tests/test_healing.py     # HealthMonitor state machine, HealingEngine, and AgentRunner discovery
.venv/bin/python tests/test_responses.py   # OpenAI Responses protocol (/v1/responses) and Codex compatibility
.venv/bin/python tests/test_tools.py       # Streaming tool calling, speculative streaming, and self-healing JSON engine
.venv/bin/python tests/test_affinity.py    # Session affinity, cross-model state handoff, LRU/TTL, and telemetry headers
```

### Official Claude Code CLI & Codex CLI Integration

#### 1. Official Claude Code CLI Integration
Point the official `claude` CLI (or any Anthropic SDK application) to the local gateway:
```bash
ANTHROPIC_BASE_URL=http://127.0.0.1:8000 \
ANTHROPIC_API_KEY=test-key \
ANTHROPIC_DEFAULT_HAIKU_MODEL=claude-3-5-haiku-20241022 \
CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 \
claude -p "Hello" --model claude-3-7-sonnet
```

#### 2. Codex CLI Integration (OpenAI Responses Protocol)
Point the official `codex` CLI to the local gateway's `/v1` endpoint:
```bash
OPENAI_BASE_URL=http://127.0.0.1:8000/v1 \
OPENAI_API_KEY=test-key \
codex -m gpt-4o "Implement a binary search function in python"
```

---

## Native Models & Alias Mapping Matrix

Incoming model requests are parsed via `server.py:resolve_model()`. The gateway routes models to one of the five backend providers according to specific rules and environment variables (`CLAUDE_BACKEND`, `OPENAI_BACKEND`, default `qwen`):

### 1. Native Provider Models

| Provider | Supported Models / Aliases | Target Web Model / Characteristic |
|---|---|---|
| **通义千问 (Qwen)** | `qwen`, `qwen-plus`, `qwen3.7`, `qwen-max`, `qwen3.7-max`, `qwen-flash`, `qwen3.6-flash`, `qwen3.6` | `Qwen` (default), `Qwen3.7-Max` (reasoning/flagship), `Qwen3.6-Flash` (speed) |
| **DeepSeek** | `deepseek`, `deepseek-chat`, `deepseek-instant`, `deepseek-reasoner`, `deepseek-r1`, `deepseek-expert` | `deepseek-chat` (flagship/general), `deepseek-reasoner` (R1 reasoning) |
| **豆包 (Doubao)** | `doubao`, `doubao-pro`, `doubao-pro-chat`, `doubao-lite`, `doubao-think`, `doubao-deep-think`, `doubao-expert` | `doubao-pro` (default), `doubao-think` (thinking), `doubao-lite` (speed) |
| **月之暗面 (Kimi)** | `kimi`, `kimi-chat`, `kimi-latest`, `kimi-explore`, `kimi-think`, `kimi-math`, `kimi-k1` | `kimi` (standard), `kimi-explore` (deep search/thinking), `kimi-math` |
| **智谱清言 (GLM)** | `glm`, `chatglm`, `glm-4`, `glm-4-plus`, `glm-4-air`, `glm-4-flash`, `glm-4-long`, `glm-zero-preview`, `glm-think` | `glm-4-plus` (flagship), `glm-4-flash` (speed), `glm-zero-preview` (thinking) |

### 2. Claude Model Family Routing (`CLAUDE_BACKEND`, default `qwen`)

| Client Model Name | Qwen (Default) | DeepSeek (`CLAUDE_BACKEND=deepseek`) | 豆包 (`CLAUDE_BACKEND=doubao`) | Kimi (`CLAUDE_BACKEND=kimi`) | GLM (`CLAUDE_BACKEND=glm`) |
|---|---|---|---|---|---|
| `claude-3-7-sonnet` / `claude-3-opus` | `Qwen3.7-Max` | `deepseek-reasoner` | `doubao-think` | `kimi-explore` | `glm-zero-preview` |
| `claude-3-5-sonnet` | `Qwen` | `deepseek-chat` | `doubao-pro` | `kimi` | `glm-4-plus` |
| `claude-3-5-haiku` | `Qwen3.6-Flash` | `deepseek-chat` | `doubao-lite` | `kimi` | `glm-4-flash` |

### 3. OpenAI / Codex Ecosystem Model Routing (`OPENAI_BACKEND`, default `qwen`)

| Tier / Alias | Qwen (Default) | DeepSeek | 豆包 | Kimi | GLM |
|---|---|---|---|---|---|
| **Flagship** (`gpt-4o`, `gpt-4-turbo`, `gpt-4`) | `Qwen3.7-Max` | `deepseek-chat` | `doubao-pro` | `kimi` | `glm-4-plus` |
| **Reasoner** (`o1`, `o1-mini`, `o3`, `o3-mini`) | `Qwen3.7-Max` | `deepseek-reasoner` | `doubao-think` | `kimi-explore` | `glm-zero-preview` |
| **Fast** (`gpt-4o-mini`, `gpt-3.5-turbo`) | `Qwen3.6-Flash` | `deepseek-chat` | `doubao-lite` | `kimi` | `glm-4-flash` |
| **Default** (`auto`, `default`) | `Qwen` | `deepseek-chat` | `doubao-pro` | `kimi` | `glm-4-plus` |

---

## Architecture & Code Structure

The repository is a local multi-protocol API gateway (`server.py`) supporting three standard protocols:
- **OpenAI Chat Completions** (`/v1/chat/completions`, `/chat/completions`)
- **Anthropic Messages** (`/v1/messages`, `/messages`, `/v1/messages/count_tokens`)
- **OpenAI Responses** (`/v1/responses`, `/responses` for Codex CLI)

### Core Subsystems
1. **Peer Capability Rings & Failover Grid (`providers/failover.py`)**:
   - Manages three capability rings: `REASONING`, `FLAGSHIP`, and `SPEED`.
   - Ensures transparent failover before the first token chunk is emitted, preventing half-packet pollution.
   - Dynamic cooldown avoidance for WAF (e.g., RGV587) and 429 rate limits, with exponential backoff (30s~120s).
   - Adaptive pacing, jitter injection, and per-provider concurrency limiting via semaphores.

2. **Self-Healing & Auto-Hotfix Brain (`healing/`)**:
   - `HealthMonitor` (`healing/monitor.py`): Performs non-blocking background probes (interval: `HEAL_CHECK_INTERVAL`, default 300s). Transitions state across `healthy`, `degraded`, `recovering`, and `offline`.
   - `HealingEngine` (`healing/healer.py`): Upon upstream protocol failure, elects the highest-performing healthy model to craft repair prompts and dispatches local CLI repair agents.
   - `AgentRunner` (`healing/agent_runner.py`): Discovers and bootstraps local `claude` or `codex` CLI environments, launching isolated subagents to update code and run tests.

3. **HTTP/2 Connection Pool Management (`providers/http_client.py`)**:
   - Global singleton `AsyncClient` configured with connection limits (max 100 conns, 40 keep-alive, 30s expiry).
   - Multiplexing enabled via `http2=True`.

4. **Atomic Credential Store (`session_store.py`)**:
   - Resolves credentials with priority: `Environment Variables > session/<provider>.json > None`.
   - Thread-safe POSIX atomic write pattern (`os.replace` + `os.fsync`) with strict `0o700` directory and `0o600` file permissions.

5. **Tool / Function Calling & Token Counting**:
   - `anthropic_compat.py`: Balanced bracket parser supporting deeply nested JSON tool calls and XML tool syntax.
   - `responses_compat.py`: Standard OpenAI Responses protocol serializer with `function_call_arguments.delta` / `done`.
   - `base.py`: Token estimation (`_est_tokens`) supporting bilingual CJK weighting.

6. **Smart Session Affinity & State Handoff Engine (`providers/session_affinity.py`)**:
   - Manages LRU and TTL-based conversation tracking to pin multi-turn agent sessions to the same upstream provider.
   - Detects provider degradation, WAF cooldown, or failure and performs seamless cross-model state handoff with heterogeneous upstream conversation ID stripping.
   - Emits `x-session-affinity`, `x-session-handoff-from`, and `x-session-turns` telemetry headers across all protocols.

7. **Web Frontend (`static/index.html`)**:
   - High-aesthetic Codex Light interface with grouped model selector and collapsible `<think>` drawer.
