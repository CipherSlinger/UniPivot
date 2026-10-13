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
.venv/bin/python tests/test_affinity.py && \
.venv/bin/python tests/test_singleflight.py && \
.venv/bin/python tests/test_disconnect.py && \
.venv/bin/python tests/test_token_cache.py && \
.venv/bin/python tests/test_lifecycle.py && \
.venv/bin/python tests/test_risk_ledger.py && \
.venv/bin/python tests/test_probe_gate.py && \
.venv/bin/python tests/test_refresh_gate.py && \
.venv/bin/python tests/test_adaptive_pacing.py && \
.venv/bin/python tests/test_graceful_drain.py && \
.venv/bin/python tests/test_conversation_gc.py
# Run individual test files
.venv/bin/python tests/test_providers.py   # Provider protocols, SSE chunk parsing, and PoW wasm loading
.venv/bin/python tests/test_server.py      # Server routing, OpenAI/Anthropic error transformation, and SSE streaming
.venv/bin/python tests/test_session.py     # Session persistence, POSIX atomicity, file permissions, and env priority
.venv/bin/python tests/test_healing.py     # HealthMonitor state machine, HealingEngine, and AgentRunner discovery
.venv/bin/python tests/test_responses.py   # OpenAI Responses protocol (/v1/responses) and Codex compatibility
.venv/bin/python tests/test_tools.py       # Streaming tool calling, speculative streaming, and self-healing JSON engine
.venv/bin/python tests/test_affinity.py    # Session affinity, cross-model state handoff, LRU/TTL, and telemetry headers
.venv/bin/python tests/test_singleflight.py # Deterministic SHA-256 fingerprinting, streaming broadcast fanout, and debounce coalescing
.venv/bin/python tests/test_disconnect.py   # Client-disconnect propagation, upstream stream reclamation, and orphaned flight recovery
.venv/bin/python tests/test_token_cache.py  # Deterministic token-count memoization, bounded LRU, and count_tokens endpoint integration
.venv/bin/python tests/test_lifecycle.py    # Request lifecycle finalization, flight-group abort terminality, and load-balancer quota reclamation
.venv/bin/python tests/test_risk_ledger.py  # Risk-control severity tiers, durable cross-restart cooldowns, and the JSONL interception ledger
.venv/bin/python tests/test_probe_gate.py   # Cooldown-aware health probe gating (banned nodes are never re-probed)
.venv/bin/python tests/test_refresh_gate.py # Cooldown-aware token refresh gating with bounded deferred retry
.venv/bin/python tests/test_adaptive_pacing.py # Evidence-driven pacing multiplier from the risk ledger
.venv/bin/python tests/test_graceful_drain.py # In-flight request drain on shutdown (streaming-safe) and inflight telemetry
.venv/bin/python tests/test_conversation_gc.py # Durable retry queue for remote temporary-conversation deletion (survives restarts)
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

7. **SingleFlight & Stream Fanout Engine (`providers/singleflight.py`)**:
   - Computes deterministic cross-protocol SHA-256 fingerprints across wire parameters and payload semantics.
   - Leader/Follower flight groups with streaming multi-subscriber fanout and historical replay buffers.
   - Non-streaming response coalescing with `x-singleflight: leader | follower` telemetry headers and zero duplicate upstream calls.

8. **Disconnect Guard & Upstream Reclamation (`providers/disconnect_guard.py`)**:
   - Propagates client disconnects to the upstream async generator and explicitly closes it, releasing the HTTP/2 connection slot and upstream concurrency quota immediately instead of letting an abandoned stream drain quota.
   - Guarantees a terminal state for the SingleFlight flight group on every exit path, including `CancelledError`, and reclaims flight groups whose leader vanished so followers can never hang.
   - Emits `x-disconnect-guard` telemetry headers across all protocols.

9. **Token Count Cache (`providers/token_counter_cache.py`)**:
   - Memoizes the deterministic `/v1/messages/count_tokens` result behind a bounded LRU, eliminating repeated O(n) estimation on long conversations.

10. **Request Lifecycle Finalization & Quota Reclamation (`providers/singleflight.py:terminate_aborted_group`)**:
    - Guarantees the non-streaming SingleFlight leader path terminates its flight group on every exit, including client aborts (`CancelledError` is a `BaseException`, so `except Exception` never fires) — without it the group stays non-terminal forever, hanging every follower.
    - Makes the `load_balancer` concurrency slot deterministically idempotent-released on abort; the Responses endpoint previously had no `finally`, so an aborted request leaked its slot permanently and the balancer would shed away from that provider until restart.
    - See `tests/test_lifecycle.py`.

11. **Risk-Control Ledger & Durable Cooldown (`providers/risk_ledger.py`)**:
    - Persists each provider's cooldown deadline and consecutive-failure count with a POSIX atomic write (`os.replace` + `os.fsync`, dir `0o700` / file `0o600`), so a restart no longer wipes avoidance and immediately re-pokes a node that upstream has already risk-banned (the ban window is only lengthened by retries).
    - Appends every risk/WAF-captcha/auth interception to an auditable JSONL ledger (time, node, HTTP status, severity tier, reason, evidence excerpt), auto-pruned at `GATEWAY_RISK_LEDGER_MAX_LINES`.
    - Severity tiers `auth > ratelimit > waf`: repeated (≥3) auth failures escalate to a long cooldown (`GATEWAY_AUTH_COOLDOWN_BASE` × 2ⁿ, capped by `GATEWAY_AUTH_COOLDOWN_MAX`), while a first auth failure keeps the short legacy cooldown so a re-login never locks a healthy node for an hour.
    - Exposes `risk_ledger` diagnostics via `/v1/diagnostics` and a 24h summary via `/health`; `GATEWAY_DISABLE_RISK_LEDGER=1` degrades to pure in-memory cooldowns. See `tests/test_risk_ledger.py`.

12. **Cooldown-Aware Health Probe Gate (`healing/monitor.py`)**:
    - `HealthMonitor` consults `CooldownTracker` before every probe: a node in a risk-avoidance cooldown is marked `cooling` and the real network probe is skipped entirely. Without this the 300s audit loop re-poked an upstream-banned node every interval, lengthening the ban window — exactly the failure mode the durable ledger guards against, one layer up.
    - Cooling probes neither increment `consecutive_failures` nor trigger self-healing, so a risk ban is never misread as a node fault; `GATEWAY_DISABLE_PROBE_COOLDOWN_GATE=1` disables the gate and `/v1/diagnostics` exposes `healing_status.cooldown_probe_skips`. See `tests/test_probe_gate.py`.

13. **Cooldown-Aware Token Refresh Gate (`server.py:_auto_refresh_stale_sessions`)**:
    - The startup silent token-refresh pass consults the cooldown tracker: a node in a risk-avoidance cooldown is not refreshed (a login/refresh call to an upstream-banned account would only deepen the ban). Deferred nodes are retried by a bounded background thread (`GATEWAY_REFRESH_DEFER_POLL`, `GATEWAY_REFRESH_DEFER_MAX_WAIT`, default 24h) that refreshes once the cooldown clears and otherwise gives up cleanly. See `tests/test_refresh_gate.py`.

14. **Evidence-Driven Adaptive Pacing (`providers/failover.py:compute_pacing_multiplier`)**:
    - The traffic pacer widens each provider's call interval proportionally to its recent risk-control interceptions (weighted `waf > auth > ratelimit`), so a node that just tripped a captcha or 429 is called more slowly until its 6h evidence window drains. This is the "avoidance" half of the ban problem: when a ban cannot be bypassed, slow down before it happens. Cached per provider (`GATEWAY_PACING_EVIDENCE_TTL`, default 60s); `GATEWAY_DISABLE_ADAPTIVE_PACING=1` reverts to the fixed interval. See `tests/test_adaptive_pacing.py`.

15. **Graceful Shutdown Drain (`server.py:_drain_inflight_requests`)**:
    - A pure-ASGI `InflightTrackerMiddleware` counts in-flight requests across the *entire* response (its `await self.app(...)` only returns after a streaming body finishes, unlike `BaseHTTPMiddleware`). On shutdown the lifespan finally-block waits — bounded by `GATEWAY_SHUTDOWN_DRAIN_TIMEOUT` (default 30s) — for that count to reach zero before stopping the health monitor and closing the HTTP client, so a restart no longer severs a half-sent stream (which would make clients retry and re-load upstream). `inflight_requests` is exposed on `/health` and `/v1/diagnostics`. See `tests/test_graceful_drain.py`.

16. **Durable Conversation Deletion Queue (`providers/conversation_gc.py`)**:
    - Anonymous calls (no client `conversation_id`) leave a temporary session in the user's upstream web sidebar, so the gateway deletes it afterwards. The old `asyncio.create_task(provider.delete_conversation(cid))` was fire-and-forget: a failed delete or a process restart silently leaked the session forever. `ConversationGC` now enqueues each deletion in a POSIX-atomic JSON queue (dir `0o700`, file `0o600`) that survives restarts, retries via a background drain (bounded by `GATEWAY_CONVERSATION_GC_MAX_ATTEMPTS`, default 5, then drops with a warning), and drains once at startup to clean up the previous process's leftovers. `GATEWAY_DISABLE_CONVERSATION_GC=1` degrades to the legacy fire-and-forget behavior. Exposed as `conversation_gc` in `/v1/diagnostics`. See `tests/test_conversation_gc.py`.
    - **[OPT-017] Cooldown-aware deletion gate**: a delete is itself a real upstream request, so firing one at a node in a risk-avoidance cooldown is just another channel that keeps touching a banned account. Deletions for a cooling provider are deferred — the entry stays queued and does **not** consume a retry attempt — and are performed by a later drain once the cooldown clears. Mirrors the probe gate (OPT-012) and the refresh gate (OPT-013). `GATEWAY_DISABLE_CONVERSATION_GC_COOLDOWN_GATE=1` reverts; deferrals are exposed as `deferred_by_cooldown` in `/v1/diagnostics`. A gate-query failure never blocks deletion (zero-regression default).

17. **Web Frontend (`static/index.html`)**:
    - High-aesthetic Codex Light interface with grouped model selector and collapsible `<think>` drawer.
