# 同枢 (UniPivot) · A Local Multi-Protocol LLM Gateway

<p align="center">
  <b>Unified translation across multi-source models · Native compatibility with three protocols · Fully connected self-healing grid · Codex Light minimal visual experience</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/Protocols-OpenAI%20%7C%20Anthropic%20%7C%20Responses-success?style=flat-square" alt="Protocols">
  <img src="https://img.shields.io/badge/Architecture-Peer%20Failover%20Grid-orange?style=flat-square" alt="Architecture">
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License">
</p>

---

## 🌟 Core Features

**同枢 (UniPivot)** is a high-performance, high-resilience multi-protocol translation service that runs locally. It translates the standardized **OpenAI Chat Completions**, **Anthropic Messages**, and **OpenAI Responses** protocols in real time into the internal calls of mainstream multi-source LLM web backends.

No need to apply for official commercial API keys, prepay, or bind a credit card. Through this gateway you can directly drive **Claude Code CLI**, **Codex CLI**, **NextChat**, **Chatbox**, **Cursor**, and the rest of the developer ecosystem toolchain.

```
                       ┌─────────────────────────────────────────────────────────────┐
                       │            Mainstream Clients & Developer Tools             │
                       │   Claude Code CLI  │  Codex CLI  │  Cursor  │  NextChat    │
                       └──────────────────────────────┬──────────────────────────────┘
                                                      │ Standard request (OpenAI / Anthropic / Responses)
                                                      ▼
                       ┌─────────────────────────────────────────────────────────────┐
                       │         UniPivot gateway server.py (127.0.0.1:8000)         │
                       │ ├─ Translation layer (Chat Completions/Messages/Responses)  │
                       │ ├─ Fully connected peer grid (Reasoning/Flagship/Speed)     │
                       │ ├─ Risk-control avoidance (cooling / escalation / lease)    │
                       │ ├─ Self-healing hotfix brain (Monitor/Healer/AgentRunner)   │
                       │ └─ HTTP/2 connection pool (Max 100 Conns / 40 Keep-Alive)   │
                       └────────┬────────────┬─────────────┬────────────┬────────────┘
                                │            │             │            │            │
                                ▼            ▼             ▼            ▼            ▼
                              Qwen       DeepSeek       Doubao        Kimi          GLM
```

### 1. Multi-Source Model Matrix
- **通义千问 (Qwen)**: Fully aligned with the domestic main-site v2 interface, supporting Qwen3.7-Max deep thinking and temporary traceless sessions.
- **深度求索 (DeepSeek)**: Native support for the DeepSeek-R1 deep-thinking reasoning chain, solving PoW in an independent WebAssembly sandbox.
- **字节豆包 (Doubao)**: Emulates the PC desktop client protocol, removing front-end signature restrictions and natively extracting the ` thinking` reasoning stream.
- **月之暗面 (Kimi)**: Long-lived refresh_token automatically rotates the access_token, with support for independent conversation pre-creation and deep-research mode.
- **智谱清言 (GLM)**: Dynamic MD5 signature verification and Assistant ID mapping, with full support for GLM-4-Plus and GLM-Zero-Preview.

### 2. Native Compatibility with Three Standard Protocols
- **OpenAI Chat Completions (`/v1/chat/completions`)**: token-by-token streaming SSE rendering, the `reasoning_content` reasoning stream, automatic balanced parsing of structured `tools` arguments, and `include_usage` statistics.
- **Anthropic Messages (`/v1/messages`, `/v1/messages/count_tokens`)**: seamless integration with the official `claude` CLI (Claude Code), strictly implementing the `message_start` -> `content_block_delta` -> `message_stop` state machine.
- **OpenAI Responses (`/v1/responses`)**: seamless integration with the new-generation `codex` CLI, with full Responses event ordering and incremental function-call dispatch.

### 3. Fully Connected Peer Failover Grid
- **Three-tier capability ring scheduling**: divided into the **Reasoning Ring (REASONING)**, **Flagship Ring (FLAGSHIP)**, and **Speed Ring (SPEED)**. All nodes within a ring are peers that back each other up.
- **Transparent switchover before the first chunk**: upstream anomalies are caught before the first token arrives and shifted losslessly to a healthy backup node in the ring, eliminating half-packet dirty data.
- **Severity-aware avoidance, not a single cooldown number**: the tier determines the patience, because the tiers recover in opposite ways. A 429 is transient and self-healing, so it gets a short exponential backoff (`base × 1.2ⁿ`, capped at `GATEWAY_WAF_COOLDOWN_MAX` = 900s) and "retry when it clears" *is* the correct recovery. A repeated human-verification challenge (Alibaba RGV587 / ByteDance) is the one class **only a human can lift**, so every automated retry is another automated attempt against a flagged account — it escalates by powers of two (`GATEWAY_WAF_LONG_COOLDOWN_BASE` 3600s → capped at 24h), and the release path watches the credential file: completing `login.py` yourself changes it, which the gateway reads as "a human intervened". An operator quarantine lease extends the window with `max()`, so a mistyped lease can never release a banned node early.
- **A ban is never re-poked to "check"**: a long avoidance is a hard block at the dispatch boundary, not a demotion — such a node is excluded from the candidate set, and a ring with no healthy member fails closed with **zero upstream calls** and a message naming each node's cause and remaining time. Confirming a declaration costs no upstream touch at all (`GET /v1/admin/dispatch-audit` is a read-only dry-run that asks the same gate the dispatcher asks).
- Smooth pacing plus random jitter on the call chain, with evidence-driven and probation-aware multipliers layered on top.

### 4. Automated Self-Healing and Hotfix Brain
- A built-in background health-audit probe periodically inspects the status of the five model services.
- When an upstream interface changes, the system automatically elects the currently healthiest and strongest model, assembles a contextual diagnostic prompt, and launches a local `claude` or `codex` agent to investigate the code and run unit tests, achieving code-level automatic hotfixing.

### 5. Codex Light Minimal Visual Workbench
- Visit `http://127.0.0.1:8000` to open the built-in modern Web UI, with elegant typography and a collapsible ` thinking` reasoning drawer.

---

## 🚀 Quick Start

### 1. Environment Preparation
```bash
# Clone the repository and enter the directory
git clone <repo_url> && cd UniPivot

# Create a Python virtual environment and install the core dependencies
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# If system Chrome is not installed, install the Playwright browser engine
.venv/bin/playwright install chromium
```

### 2. Passwordless QR Login (Capturing Session Credentials)
Run the interactive login command; the system will open a browser window for you to scan the QR code or log in with an account:
```bash
# Log in to all five providers
.venv/bin/python login.py all

# Or log in to a single provider as needed
.venv/bin/python login.py qwen       # Qwen
.venv/bin/python login.py deepseek   # DeepSeek
.venv/bin/python login.py doubao     # Doubao
.venv/bin/python login.py kimi       # Kimi
.venv/bin/python login.py glm        # GLM

# Silently refresh tokens (reuse the existing profiles, no popup)
.venv/bin/python login.py all --refresh
```

### 3. Start the Gateway Service
```bash
./run.sh
# Or start it explicitly (note: `python -m uvicorn`, not `.venv/bin/uvicorn`)
.venv/bin/python -m uvicorn server:app --host 127.0.0.1 --port 8000
```

After the service starts:
- 🌐 **Web UI**: `http://127.0.0.1:8000`
- 📚 **Swagger API docs**: `http://127.0.0.1:8000/docs`
- 🩺 **Health and self-healing status**: `http://127.0.0.1:8000/health`

---

## 💻 Developer Ecosystem Integration Examples

### 1. Connecting the Official Claude Code CLI
Four environment variables are enough to drive the underlying models directly with Anthropic's official command-line tool:
```bash
ANTHROPIC_BASE_URL=http://127.0.0.1:8000 \
ANTHROPIC_API_KEY=test-key \
ANTHROPIC_DEFAULT_HAIKU_MODEL=claude-3-5-haiku-20241022 \
CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 \
claude -p "Write a complete implementation of a red-black tree in Python" --model claude-3-7-sonnet
```

### 2. Connecting Codex CLI (Responses API)

Codex **does not read `OPENAI_BASE_URL`**. Measured against the real `codex` 0.145.0 binary: with only that variable set, its `ModelProviderInfo` reports `base_url: None` and the CLI dials `wss://api.openai.com/v1/responses` — your prompts go to OpenAI, not to this gateway. The binary does not even contain the string `OPENAI_BASE_URL`. A custom provider must be declared in `$CODEX_HOME/config.toml`:

```toml
# ~/.codex/config.toml  (or point CODEX_HOME at this directory)
model_provider = "unipivot"
model = "gpt-5.5"
approval_policy = "never"
sandbox_mode = "danger-full-access"

[model_providers.unipivot]
name = "unipivot"
base_url = "http://127.0.0.1:8000/v1"
wire_api = "responses"
requires_openai_auth = false
```

Then simply:
```bash
codex exec "Optimize the concurrent network connection pool logic in the current directory"
```

Verify it is really talking to the gateway before trusting a run — the gateway log will show a `POST /v1/responses`, and `curl -s http://127.0.0.1:8000/v1/diagnostics | head` will show the request. To check end to end including a real two-round tool loop:

```bash
.venv/bin/python scripts/e2e/run_agent_loop.py --client codex
```

### 3. Calling from the OpenAI Python SDK
```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="deepseek-reasoner",  # or qwen3.7-max / doubao-think / kimi-explore
    messages=[{"role": "user", "content": "Which number is larger, 9.11 or 9.8? Give a rigorous proof."}],
    stream=True,
)

for chunk in response:
    # Extract the deep-thinking process
    if hasattr(chunk.choices[0].delta, "reasoning_content") and chunk.choices[0].delta.reasoning_content:
        print(chunk.choices[0].delta.reasoning_content, end="", flush=True)
    # Extract the final answer
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

---

## 🗺️ Model Matrix and Routing Map

### 1. Native Provider Models and Features

| Provider | Natively Supported Models / Aliases | Corresponding Web Model and Core Capabilities |
|---|---|---|
| **Qwen** | `qwen`, `qwen-max`, `qwen3.7-max`, `qwen3.6-flash` | `Qwen` (default), `Qwen3.7-Max` (reasoning flagship), `Qwen3.6-Flash` (ultra-fast, low latency) |
| **DeepSeek** | `deepseek-chat`, `deepseek-reasoner`, `deepseek-r1` | `deepseek-chat` (general-purpose flagship), `deepseek-reasoner` (R1 deep-thinking reasoning) |
| **Doubao** | `doubao-pro`, `doubao-think`, `doubao-lite`, `doubao-expert` | `doubao-pro` (default flagship), `doubao-think` (chain of thought), `doubao-lite` (ultra-fast response) |
| **Kimi** | `kimi`, `kimi-explore`, `kimi-math`, `kimi-k1` | `kimi` (standard), `kimi-explore` (deep search and reasoning), `kimi-math` (mathematics specialist) |
| **GLM** | `glm-4-plus`, `glm-4-flash`, `glm-zero-preview`, `glm-think` | `glm-4-plus` (flagship), `glm-4-flash` (high throughput), `glm-zero-preview` (zero-temperature reasoning) |

### 2. Cross-Ecosystem Dynamic Alias Mapping

When a client requests a generic alias, the system routes automatically according to the environment variables (`CLAUDE_BACKEND`, `OPENAI_BACKEND`, default `qwen`):

| Client-Requested Alias | Qwen (default) | DeepSeek Backend | Doubao Backend | Kimi Backend | GLM Backend |
|---|---|---|---|---|---|
| `claude-3-7-sonnet` | `Qwen3.7-Max` | `deepseek-reasoner` | `doubao-think` | `kimi-explore` | `glm-zero-preview` |
| `claude-3-5-haiku` | `Qwen3.6-Flash` | `deepseek-chat` | `doubao-lite` | `kimi` | `glm-4-flash` |
| `gpt-4o` (Flagship) | `Qwen3.7-Max` | `deepseek-chat` | `doubao-pro` | `kimi` | `glm-4-plus` |
| `o1` / `o3` (Reasoner)| `Qwen3.7-Max` | `deepseek-reasoner` | `doubao-think` | `kimi-explore` | `glm-zero-preview` |
| `gpt-4o-mini` (Fast) | `Qwen3.6-Flash` | `deepseek-chat` | `doubao-lite` | `kimi` | `glm-4-flash` |

---

## 🛠️ Project Organization

```
├── server.py                    # Gateway entry point (FastAPI routing, lifecycle management)
├── login.py                     # Automated credential capture (Playwright Chrome driver)
├── session_store.py             # Atomic POSIX credential storage and permission isolation (0o700/0o600)
├── providers/                   # Model adaptation and translation core package
│   ├── base.py                  # Message flattening, token estimation, SSE frame assembly
│   ├── failover.py              # Peer failover routing grid, dispatch gate, three-tier capability rings
│   ├── risk_ledger.py           # Durable cooldowns + auditable interception ledger (JSONL)
│   ├── risk_dossier.py          # All-time interception record and "can this be bypassed?" verdict
│   ├── risk_correlation.py      # Cross-provider correlation: systemic vs. single-account interception
│   ├── quarantine.py            # Operator ban leases (long-duration, extend-only)
│   ├── credential_watch.py      # Whole-ring credential outage detection and zero-upstream fast-fail
│   ├── credential_renewal.py    # Credential-content fingerprinting and cause-scoped avoidance release
│   ├── device_identity.py       # Stable per-node device identity (constant across requests and restarts)
│   ├── singleflight.py          # In-flight request dedup with streaming fanout and follower backpressure
│   ├── session_affinity.py      # Conversation pinning and cross-model state handoff
│   ├── conversation_gc.py       # Durable retry queue for temporary-conversation deletion
│   ├── tool_streamer.py         # Anti-fabrication tool-call parsing (three-layer defence)
│   ├── stop_sequences.py        # Client-declared stop-sequence truncation (streaming-safe, split-aware)
│   ├── stream_completion.py     # Tri-state completion marker verdict (seen / not-seen / unknown)
│   ├── stream_keepalive.py      # SSE heartbeat during upstream silence
│   ├── prompt_optimizer.py      # History folding that shortens without rewriting
│   ├── http_client.py           # HTTP/2 connection pool and singleton reuse
│   ├── anthropic_compat.py      # Anthropic Messages protocol state machine and ToolCall parsing
│   ├── responses_compat.py      # OpenAI Responses API protocol translation
│   ├── qwen/                    # Qwen v2 protocol adapter
│   ├── deepseek/                # DeepSeek protocol adapter and WebAssembly PoW engine
│   ├── doubao/                  # Doubao PC client protocol adapter
│   ├── kimi/                    # Kimi conversation pre-creation and token rotation adapter
│   └── glm/                     # GLM dynamic signature and logic stream adapter
├── healing/                     # Self-healing and hotfix system
│   ├── monitor.py               # Provider health probe inspection and fault state machine
│   ├── healer.py                # Model brain election, prompt assembly and task dispatch
│   └── agent_runner.py          # Agent CLI discovery, bootstrap installation and sandboxed execution
├── scripts/                     # Reverse-engineering analysis and integration scripts
│   ├── reverse/                 # Frontend Webpack dependencies, upstream signing and other reverse-engineering analysis
│   └── debug/                   # Single-endpoint HTTP probing and replay debugging
├── static/                      # Codex Light Web UI front-end assets
├── tests/                       # Offline automated test suite (covers the five providers and three protocols)
└── docs/                        # Architecture design and research documents
    ├── architecture.md          # Complete system architecture design and engineering specification
    └── provider_capabilities.md # In-depth vendor capability research report
```

---

## 🧪 Automated Test Verification

This repository ships a fully offline unit-test suite in `tests/` — every file a standalone script, each runnable on its own (no test runner, no network). The protocol streaming state machines, the PoW solver, the risk-avoidance ledger, the tool loop and the self-healing logic can all be verified without connecting to the real internet:

```bash
# Run the entire suite
for f in tests/test_*.py; do .venv/bin/python "$f" || { echo "FAILED $f"; break; }; done

# Or run a single module's suite
.venv/bin/python tests/test_responses_protocol.py
.venv/bin/python tests/test_agent_loop.py
.venv/bin/python tests/test_stream_lifecycle.py
```

Every suite sets `TESTING=1`, so the durable stores (risk ledger, cooldowns, quarantine leases, conversation GC queue, device identities) resolve to a `tempfile.mkdtemp()` directory — **no test writes to `session/` and no test issues a real upstream request**.

---

## 📄 Open-Source License

This project is released under the [MIT License](LICENSE). It is intended only for personal learning, technical reverse-engineering research, and automated development debugging; please strictly comply with the terms of service of each model platform.
