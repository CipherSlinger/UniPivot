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
.venv/bin/python tests/test_failover.py && \
.venv/bin/python tests/test_disconnect.py && \
.venv/bin/python tests/test_token_cache.py && \
.venv/bin/python tests/test_lifecycle.py && \
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
# Run individual test files
.venv/bin/python tests/test_providers.py   # Provider protocols, SSE chunk parsing, and PoW wasm loading
.venv/bin/python tests/test_server.py      # Server routing, OpenAI/Anthropic error transformation, and SSE streaming
.venv/bin/python tests/test_session.py     # Session persistence, POSIX atomicity, file permissions, and env priority
.venv/bin/python tests/test_healing.py     # HealthMonitor state machine, HealingEngine, and AgentRunner discovery
.venv/bin/python tests/test_responses.py   # OpenAI Responses protocol (/v1/responses) and Codex compatibility
.venv/bin/python tests/test_tools.py       # Streaming tool calling, speculative streaming, and self-healing JSON engine
.venv/bin/python tests/test_affinity.py    # Session affinity, cross-model state handoff, LRU/TTL, and telemetry headers
.venv/bin/python tests/test_singleflight.py # Deterministic SHA-256 fingerprinting, streaming broadcast fanout, and debounce coalescing
.venv/bin/python tests/test_failover.py     # Capability rings, fallback candidate ordering/cooldown demotion, risk metadata, and transparent failover
.venv/bin/python tests/test_disconnect.py   # Client-disconnect propagation, upstream stream reclamation, and orphaned flight recovery
.venv/bin/python tests/test_token_cache.py  # Deterministic token-count memoization, bounded LRU, and count_tokens endpoint integration
.venv/bin/python tests/test_lifecycle.py    # Request lifecycle finalization, flight-group abort terminality, and load-balancer quota reclamation
.venv/bin/python tests/test_risk_ledger.py  # Risk-control severity tiers, durable cross-restart cooldowns, and the JSONL interception ledger
.venv/bin/python tests/test_probe_gate.py   # Cooldown-aware health probe gating + read-time cooling overlay and healer-election avoidance (banned nodes are never re-probed or used for healing)
.venv/bin/python tests/test_refresh_gate.py # Cooldown-aware token refresh gating with bounded deferred retry
.venv/bin/python tests/test_adaptive_pacing.py # Evidence-driven pacing multiplier from the risk ledger
.venv/bin/python tests/test_graceful_drain.py # In-flight request drain on shutdown (streaming-safe) and inflight telemetry
.venv/bin/python tests/test_conversation_gc.py # Durable retry queue for remote temporary-conversation deletion (survives restarts)
.venv/bin/python tests/test_quarantine.py   # Operator long-duration quarantine leases (folded into every cooldown gate + admin API)
.venv/bin/python tests/test_quarantine_hardstop.py # Quarantine hard-stop at the dispatch boundary (quarantined nodes are never dispatched, even as failover fallbacks)
.venv/bin/python tests/test_probation.py    # Post-avoidance recovery probation: serialized canary handshakes, evidence-based graduation (window renewal) and widened pacing after a ban lifts
.venv/bin/python tests/test_risk_dossier.py # All-time interception dossier, posture derivation, and bypass/re-login avoidance verdicts
.venv/bin/python tests/test_risk_correlation.py # Cross-provider risk correlation: systemic escalation detection and the global pacing governor
.venv/bin/python tests/test_credential_watch.py # Whole-ring credential outage: distinct-node threshold, TTL probe, zero-upstream-call fast-fail
.venv/bin/python tests/test_credential_renewal.py # Credential-renewal detection: content fingerprint of session/<provider>.json, cause-scoped release of the 24h auth cooldown, quarantine/WAF immunity, unblocks the refresh gate
.venv/bin/python tests/test_ban_inference.py # Account-level ban inference: "a provably-renewed credential is still rejected" discriminator, self-clearing auto lease, operator-lease precedence, zero-touch dispatch blocking
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
    - **[OPT-018] Read-time cooldown overlay + healer election avoidance**: `ProviderHealth.status` defaults to `healthy`, so before the first probe cycle (or right after a restart) a node sitting in a risk-avoidance cooldown still *reports* healthy — and `pick_healer_model` would elect it as the self-healing brain, dispatching a real upstream completion through a banned account. `get_status()`/`get_all_statuses()` now overlay live cooldown state ("cooling") at read time without persisting it, `pick_healer_model` excludes cooling candidates via `monitor.get_cooldown_remaining()` and records `cooldown_healer_skips`, and when no non-cooling candidate remains it returns `None` so healing is deferred instead of touching the banned node. The single master switch for all cooldown avoidance is `GATEWAY_DISABLE_COOLDOWN_GATE=1` (`GATEWAY_DISABLE_PROBE_COOLDOWN_GATE` remains as a back-compatible alias).

13. **Cooldown-Aware Token Refresh Gate (`server.py:_auto_refresh_stale_sessions`)**:
    - The startup silent token-refresh pass consults the cooldown tracker: a node in a risk-avoidance cooldown is not refreshed (a login/refresh call to an upstream-banned account would only deepen the ban). Deferred nodes are retried by a bounded background thread (`GATEWAY_REFRESH_DEFER_POLL`, `GATEWAY_REFRESH_DEFER_MAX_WAIT`, default 24h) that refreshes once the cooldown clears and otherwise gives up cleanly. See `tests/test_refresh_gate.py`.

14. **Evidence-Driven Adaptive Pacing (`providers/failover.py:compute_pacing_multiplier`)**:
    - The traffic pacer widens each provider's call interval proportionally to its recent risk-control interceptions (weighted `waf > auth > ratelimit`), so a node that just tripped a captcha or 429 is called more slowly until its 6h evidence window drains. This is the "avoidance" half of the ban problem: when a ban cannot be bypassed, slow down before it happens. Cached per provider (`GATEWAY_PACING_EVIDENCE_TTL`, default 60s); `GATEWAY_DISABLE_ADAPTIVE_PACING=1` reverts to the fixed interval. See `tests/test_adaptive_pacing.py`.

15. **Graceful Shutdown Drain (`server.py:_drain_inflight_requests`)**:
    - A pure-ASGI `InflightTrackerMiddleware` counts in-flight requests across the *entire* response (its `await self.app(...)` only returns after a streaming body finishes, unlike `BaseHTTPMiddleware`). On shutdown the lifespan finally-block waits — bounded by `GATEWAY_SHUTDOWN_DRAIN_TIMEOUT` (default 30s) — for that count to reach zero before stopping the health monitor and closing the HTTP client, so a restart no longer severs a half-sent stream (which would make clients retry and re-load upstream). `inflight_requests` is exposed on `/health` and `/v1/diagnostics`. See `tests/test_graceful_drain.py`.

16. **Durable Conversation Deletion Queue (`providers/conversation_gc.py`)**:
    - Anonymous calls (no client `conversation_id`) leave a temporary session in the user's upstream web sidebar, so the gateway deletes it afterwards. The old `asyncio.create_task(provider.delete_conversation(cid))` was fire-and-forget: a failed delete or a process restart silently leaked the session forever. `ConversationGC` now enqueues each deletion in a POSIX-atomic JSON queue (dir `0o700`, file `0o600`) that survives restarts, retries via a background drain (bounded by `GATEWAY_CONVERSATION_GC_MAX_ATTEMPTS`, default 5, then drops with a warning), and drains once at startup to clean up the previous process's leftovers. `GATEWAY_DISABLE_CONVERSATION_GC=1` degrades to the legacy fire-and-forget behavior. Exposed as `conversation_gc` in `/v1/diagnostics`. See `tests/test_conversation_gc.py`.
    - **[OPT-017] Cooldown-aware deletion gate**: a delete is itself a real upstream request, so firing one at a node in a risk-avoidance cooldown is just another channel that keeps touching a banned account. Deletions for a cooling provider are deferred — the entry stays queued and does **not** consume a retry attempt — and are performed by a later drain once the cooldown clears. Mirrors the probe gate (OPT-012) and the refresh gate (OPT-013). `GATEWAY_DISABLE_CONVERSATION_GC_COOLDOWN_GATE=1` reverts; deferrals are exposed as `deferred_by_cooldown` in `/v1/diagnostics`. A gate-query failure never blocks deletion (zero-regression default).

17. **Test Isolation of Durable Stores (`tests/*`)**:
    - The durable stores (risk ledger cooldowns, interception JSONL, conversation GC queue) resolve to a `tempfile.mkdtemp()` directory under `TESTING=1`, but several legacy suites set fake tokens *without* setting `TESTING`, so a test run wrote fake cooldowns into the developer's real `session/` directory — and, because every `CooldownTracker` restores from the shared ledger singleton, later cases in the same process inherited cooldowns recorded by earlier ones, making results order-dependent (`tests/test_failover.py` was red for this reason). Those suites now set `TESTING=1` (isolated temp dir, zero writes to `session/`), and `tests/test_failover.py` additionally sets `GATEWAY_DISABLE_RISK_LEDGER=1` so it tests pure in-memory cooldown semantics as designed — cross-restart durability has its own suite, `tests/test_risk_ledger.py`.

18. **Operator Quarantine Lease (`providers/quarantine.py`)**:
    - Automatic cooldowns are **capped** (`GATEWAY_AUTH_COOLDOWN_MAX`, default 24h), but a real account-level ban lasts days. When the cap expires the gateway would re-touch a still-banned node and deepen the ban. `QuarantineStore` lets an operator set a durable long-duration lease per provider — resolved by absolute expiry timestamp, POSIX-atomic on disk (dir `0o700`, file `0o600`), re-applied on restart, and expiring lazily. Leases only ever **extend** the avoidance window (reads take `max()`), so a mistyped short lease can never release a banned node early.
    - **Single source of truth**: the lease is folded into `CooldownTracker.get_remaining_cooldown()` / `is_in_cooldown()`, so *every* existing gate honors it with zero extra wiring — health probe (OPT-012), token refresh (OPT-013), conversation GC (OPT-017), healer election (OPT-018), router selection and adaptive pacing.
    - Configuration and admin surface: `GATEWAY_QUARANTINE="deepseek:72h,glm:30m,kimi:2026-10-20T00:00:00"` applies at startup (durations `s/m/h/d` or ISO-8601), `GET/POST/DELETE /v1/admin/quarantine` views/sets/clears at runtime, `GATEWAY_DISABLE_QUARANTINE=1` turns the store off, and `GATEWAY_QUARANTINE_DIR` overrides the location. Exposed as `quarantine` on `/health` (active leases) and `/v1/diagnostics`. See `tests/test_quarantine.py`.
    - **[OPT-021] Hard-stop at the dispatch boundary**: folding the lease into the *cooldown reading* gated probes, refresh, conversation GC and healer election — but **not** the dispatch path. `get_fallback_candidates` only *demoted* cooling nodes, and `execute_chat` dispatched the primary unconditionally, so a node under a 72h lease was still sent real upstream requests on every failover (empirically confirmed for both DeepSeek and Doubao). Quarantine is now a **hard block with semantics distinct from soft cooldown**: a soft cooldown (automatic, capped at `GATEWAY_AUTH_COOLDOWN_MAX`) stays demote-only so a ring where everything is briefly cooling can still make a last-resort attempt — availability is preserved; an operator lease (account-level ban, measured in days) is never dispatched at all, at any point. A quarantined primary is redirected, quarantined nodes are excluded from the candidate set, and if the whole ring is quarantined the request **fails closed** with `503 / err_type="quarantined"`, zero upstream calls, and a message telling the operator how to lift the lease (`overloaded_error` in the Anthropic protocol). `GATEWAY_DISABLE_QUARANTINE_HARD_STOP=1` reverts to demote-only. See `tests/test_quarantine_hardstop.py`.

19. **Post-Avoidance Recovery Probation (`providers/failover.py`)**:
    - **[OPT-022] Canary re-entry after an avoidance episode**: avoidance solves *not touching* a banned node but not *not re-triggering* the ban. When a multi-day ban lifts, the demand that was suppressed for days releases at once — the same burst signature that caused the original ban, immediately re-banning the account. On the first dispatch after an avoidance episode ends (risk cooldown expiry, or a quarantine lease expiring or being cleared), the node enters a **probation window** (`GATEWAY_PROBATION_WINDOW`, default 300s) during which at most **one handshake is in flight** and the pacing interval is multiplied (`GATEWAY_PROBATION_PACING_FACTOR`, default 2.0×). It graduates after `GATEWAY_PROBATION_MIN_SUCCESSES` (default 3) consecutive successes; the window is a *renewal unit*, not a licence to resume at full speed (see the [OPT-028] bullet below).
    - The serialization gate covers *handshake to first chunk* only, not the whole streaming response, and is acquired **before** the concurrency semaphore so requests cannot occupy concurrency slots while queued behind the gate. Because plain 5xx/timeouts are upstream faults rather than account flagging, they never enter probation — no thundering herd to protect against, hence zero regression for existing paths. **[OPT-028] Graduation is evidence-based, never time-based**: the window is only a *renewal unit* — exhausting one renews it (up to `GATEWAY_PROBATION_MAX_RENEWALS`, default 3, i.e. at most `window × 4` gated) with the serialization gate and widened pacing still in force, and once the renewal budget is spent the node is graduated only if it holds positive evidence (≥1 successful handshake); with zero evidence it stays gated, because gating a node that has never been successfully verified costs nothing (no successful handshake means no burst to release) while un-gating hands a completely unproven account full concurrency. The same applies to `GATEWAY_PROBATION_WINDOW=0` ("no time allowance", not "release immediately"); `GATEWAY_DISABLE_PROBATION=1` remains the full off switch. A late success always graduates the node, so the gate can never deadlock. See `tests/test_probation.py`.
    - `GATEWAY_PROBATION_PACING_FACTOR` is applied inside `CooldownTracker.compute_pacing_interval()` (pure, order-correct: base × evidence multiplier × probation multiplier) so it composes with the OPT-014 adaptive multiplier instead of replacing it. Probation state is persisted (`episode_ends`) and survives restarts, since a restart would otherwise be a one-step bypass — the retention horizon covers the **renewed** window, so a restart mid-renewal cannot discard the episode. `GATEWAY_DISABLE_PROBATION=1` turns it off entirely. Telemetry: `x-risk-probation: 1` header plus `probation.active` / `probation.stats` on `/v1/diagnostics` (the stats include derived `renewing` / `awaiting_first_handshake` / `budget_exhausted`, so "is the probation actually doing anything?" is observable rather than assumed). See `tests/test_probation.py`.

20. **Risk-Control Dossier & Avoidance Verdict (`providers/risk_dossier.py`)**:
    - **[OPT-023] All-time interception record + the "can it be bypassed?" answer**: `recent_interceptions()` is a *sliding window* (24h default), which is right for driving real-time avoidance but wrong for answering the operational question "how many times has this provider been intercepted, for what reasons, can it be worked around, does re-login help?". A per-provider ban window is measured in *days*, so the 24h window has almost always discarded the evidence before the ban lifts. The dossier aggregates the **whole** (bounded) ledger: total count, first/last timestamps, severity breakdown, and the last real upstream message verbatim.
    - **Posture derivation, fail-closed**: the facts are scattered across four places — quarantine lease, soft cooldown, recovery probation (OPT-022) and the OPT-014 evidence-driven pacing multiplier. They collapse into one posture, heaviest-wins: `quarantined` > `cooling` > `probation` > `watch` > `clear`, each with a `dispatchable` flag. Only a hard lease is non-dispatchable; a soft cooldown stays dispatchable, preserving the last-resort availability semantics of OPT-021.
    - **Avoidance verdict per severity** (`AVOIDANCE_MATRIX`): each tier answers `technically_bypassable` (never true — no client-side action evades a risk control), `recoverable_by_relogin`, `auto_recovers`, typical span, and a concrete strategy. `auth` (expired token/session) and `waf` (captcha/RGV587) *do* recover via `python login.py <provider>`; `ratelimit` is transient and self-healing, so re-login is pointless; an account-level ban (`ban`) is **not** recoverable by re-login — the ban binds the account, not the token — and the only correct strategy is zero-touch waiting for the platform to lift it. Providers holding a lease are always judged as `ban` regardless of what the ledger says.
    - **Read-only**: querying the dossier never mutates risk state (no lease cleared, no cooldown reset, no ledger append), so it is safe to expose on diagnostic paths. `GATEWAY_DISABLE_RISK_DOSSIER=1` marks it disabled. Telemetry: `risk_dossier_summary` on `/v1/diagnostics`, full detail (plus `?provider=<key>` for a single node) on `GET /v1/admin/risk`. See `tests/test_risk_dossier.py`.

21. **Cross-Provider Risk Correlation (`providers/risk_correlation.py`)**:
    - **[OPT-024] The blind spot of per-account reasoning**: every avoidance mechanism so far (OPT-011..OPT-023) is scoped to a *single* provider account — that account was intercepted, so cool it, quarantine it, watch it. But there is a second cause of interception: **our own client footprint** (header fingerprint, UA, device params, egress IP, request timing) being flagged by the platform. The accounts are then not intercepted independently — they are multiple symptoms of one root cause. The two causes demand **opposite** responses: for a single-account cause "switch to another node and keep going" is correct; for a systemic cause it is useless and actively harmful, because the new node gets flagged on the same footprint and you simply burn the remaining accounts one by one. Per-account reasoning can never see this layer — it explains each interception as "this provider's risk control is strict" and hands over all five accounts in sequence.
    - **Correlation is measured in *distinct providers*, not interception counts**: one account intercepted a hundred times is still a single-account problem. The detector scans the ledger (read-only) over `GATEWAY_RISK_CORRELATION_WINDOW` (default 900s) and computes the largest number of distinct providers intercepted inside any `GATEWAY_RISK_CORRELATION_BURST_WINDOW` sub-window (default 120s), using a two-pointer sliding window so it stays O(n).
    - **Grading, strongest signal first**: `systemic` when the burst holds `GATEWAY_RISK_CORRELATION_SYSTEMIC_MIN` (default 3) distinct providers — several different accounts hit within minutes is essentially impossible from independent causes; else `widespread` when `GATEWAY_RISK_CORRELATION_WIDESPREAD_MIN` (default 3) distinct providers appear in the window but not as a burst; else `localized`; else `none`.
    - **Global pacing governor**: `systemic` multiplies every node's interval by `GATEWAY_RISK_CORRELATION_SYSTEMIC_FACTOR` (default 2.0), `widespread` by `GATEWAY_RISK_CORRELATION_WIDESPREAD_FACTOR` (default 1.25). This composes as a fourth factor in `CooldownTracker.compute_pacing_interval` and in `TrafficPacer`, so it multiplies with the OPT-014 evidence factor and the OPT-022 probation factor rather than replacing either. The multiplier is TTL-cached (`GATEWAY_RISK_CORRELATION_CACHE_TTL`, default 15s) because the pacing path is hot, and defaults to **1.0** whenever there is no correlation evidence — zero regression.
    - Each grade carries a playbook stating explicitly whether `rotate_node_helps` and `per_account_avoidance_helps`. Telemetry: `risk_correlation` on `/v1/diagnostics`, full report at `GET /v1/admin/risk/correlation`. `GATEWAY_DISABLE_RISK_CORRELATION=1` disables. See `tests/test_risk_correlation.py`.

22. **Whole-Ring Credential Outage & Fast-Fail (`providers/credential_watch.py`)**:
    - **[OPT-025] Live evidence — the outage was misdiagnosed as a WAF ban**: the risk ledger recorded 335 interceptions over 80 minutes, of which **312 were `auth` (401), not `waf`**, with all five providers returning 401 in the *same second*, repeating every ~7 minutes. Five independent token expiries in the same second is not a coincidence: the real sequence is **one client request failing over through all five nodes, every node rejecting on an expired credential**, so a single request emits five auth records in one second. The ring has zero serving capacity — failover cannot possibly succeed.
    - **The storm is itself the risk trigger**: with no usable credentials the gateway still walked all five nodes for every client request. 335 failed requests carrying invalid credentials in 80 minutes is exactly the *automated credential attempts* signature upstream WAFs score on — the storm was itself inflating the risk score, plausibly contributing to the bans. So the correct behaviour is not "retry harder" but **stop immediately**, which is precisely what a per-node cooldown cannot express: each node cooling independently still lets the whole ring be walked.
    - **Semantics distinct from everything else**: a soft cooldown is per-node, time-capped and self-healing; a quarantine lease (OPT-020/021) is per-node, operator-declared and measured in days. A credential outage is **ring-wide**, needs no operator declaration, and recovers the instant credentials are renewed — it is not "this node has a problem" but "we have no usable credentials", so the remedy is re-login, not avoidance. Detection is by **distinct providers** (as in OPT-024, not by failure count): `GATEWAY_CREDENTIAL_OUTAGE_MIN_PROVIDERS` (default 3) different providers auth-failing within `GATEWAY_CREDENTIAL_OUTAGE_WINDOW` (default 300s) trips it.
    - **Fast-fail at the dispatch boundary**: checked at the top of `execute_chat`'s dispatch loop — *before* the OPT-021 quarantine branch and before any provider is even instantiated — so the request is refused with `503 / err_type="credential_outage"` and **zero upstream calls** (`authentication_error` in the Anthropic protocol, the closest enum, since Anthropic has no credential-outage type). The message names the dead providers and tells the operator to run `.venv/bin/python login.py all`.
    - **Never wedges**: the state carries a TTL (`GATEWAY_CREDENTIAL_OUTAGE_TTL`, default 120s) after which exactly one request is let through as a probe; if the ring is still 401ing, the very first failure re-arms it. Probing therefore degrades from "once per client request" to "once per TTL" — the storm is flattened with no possibility of the state deadlocking into permanent refusal. A single success (`record_success` → `note_success`) drops that provider from the failed set, so capacity returning below the threshold **immediately resumes dispatch** (fail-open). `GATEWAY_DISABLE_CREDENTIAL_WATCH=1` disables it (and `GATEWAY_DISABLE_RISK_LEDGER=1` co-disables, matching the quarantine convention so pure in-memory cooldown suites are unaffected). Telemetry: `credential_outage` on `/v1/diagnostics`, `GET/DELETE /v1/admin/credential-outage` to inspect the state and to lift it immediately after re-login. See `tests/test_credential_watch.py`.

23. **Credential Renewal Detection & Cause-Scoped Avoidance Release (`providers/credential_renewal.py`)**:
    - **[OPT-026] The 24-hour tail after the storm**: `record_failure` escalates auth failures as `GATEWAY_AUTH_COOLDOWN_BASE`(1800s) × 2ⁿ, capped at `GATEWAY_AUTH_COOLDOWN_MAX`(86400s). The real ledger shows 54/54/54/87/60 auth failures per node — **every one hit the 24h cap**, and the state is persisted and restored across restarts. So after the storm the five nodes each stay frozen for a full day.
    - **Why the escalation becomes wrong once the cause is gone**: it punishes *holding one dead credential and retrying with it* — which is correct, since every retry is another "automated credential attempt". But once the credential is renewed the offence no longer exists; continuing to freeze punishes the *already-repaired system*. `get_remaining_cooldown()` still returns 24h → probes skipped, healer election excludes the node, routing demotes it, and worst of all `server.py:_deferred_refresh_after_cooldown` **defers the silent token refresh itself by up to 24h** — i.e. the state the credential failure created locks the only mechanism that can repair it. Reverse check: OPT-023's dossier already answers `auth → recoverable_by_relogin = True`, so the gateway tells the operator "re-login fixes this" and then keeps avoiding the node for 24h after they do it.
    - **Renewal detection by content fingerprint**: on every auth failure the credential source (`session/<provider>.json`) is fingerprinted — `sha256(token|cookie|captured_at|user_id|device_id)`, degrading to `meta:mtime_ns:size` when unparseable. The fingerprint is persisted with the cooldown, so "re-login first, restart second" is still recognised. `Session.save()` writes atomically via `os.replace` and refreshes `captured_at` on every capture, so any real re-login necessarily changes the fingerprint. The tracker only *answers* "did the credential change"; the state mutation lives in `CooldownTracker.release_auth_avoidance()`.
    - **Cause-scoped, fail-closed release**: only a cooldown whose most recent severity is `auth` may be released (the severity is persisted alongside the cooldown; when the record is missing — state predating this mechanism, e.g. the real ledger's five 24h freezes — it degrades to a *read-only* query of the ledger's most recent interception for that node, still cause-scoped). A node holding an operator lease is **never** released (a lease is an account-level ban; re-login is useless). A `waf`/`ratelimit` cause is **never** released on its behalf. No record / unreadable credential / unchanged fingerprint → "not renewed". Release re-enters via the OPT-022 recovery probation (serialized canary + widened pacing), because the demand suppressed for days releases at the moment of renewal with the same burst signature the probation exists for.
    - **Unblocks the repair chain**: releasing zeroes `get_remaining_cooldown()`, so the OPT-013 refresh gate stops deferring the silent token refresh — the one automatic repair path. Startup runs the scan (`await cooldown_tracker.release_renewed_credentials()`), because "re-login + restart" is the common operator flow. `GATEWAY_DISABLE_CREDENTIAL_RENEWAL=1` disables it (and `GATEWAY_DISABLE_RISK_LEDGER=1` co-disables, matching the quarantine convention). Telemetry: `credential_renewal` on `/v1/diagnostics`, `GET /v1/admin/credential-renewal`, `POST /v1/admin/credential-renewal/scan` (body `{"providers":[...]}` for env-var credentials with no file to fingerprint), and `DELETE /v1/admin/credential-outage` now also releases the auth avoidance of the cleared providers. See `tests/test_credential_renewal.py`.

24. **Account-Level Ban Inference & Self-Clearing Auto Lease (`providers/failover.py` / `providers/quarantine.py`)**:
    - **[OPT-027] The loop OPT-026 left open**: OPT-026 releases the auth avoidance the moment a renewed credential appears — correct, but *if the node still 401s* the gateway re-dispatches, takes another 401, re-escalates by 2ⁿ back to 24h, and OPT-026 releases again on the next scan. Each round touches the upstream account once more, and the loop never terminates on its own. That is exactly the failure mode the standing constraint forbids: a banned node must be **zero-touched**, not periodically poked.
    - **401 has two causes whose remedies are opposite**: (a) *credential expiry* → the fix is re-login (`python login.py <provider>`), which OPT-026 already handles; (b) *account-level ban* → re-login is **useless** (the ban binds the account, not the token), and the only correct action is zero-touch waiting for the platform to lift it. Treating (b) as (a) is precisely how a retry loop keeps re-triggering a ban. So the mechanism is worthless without a **discriminator**, and the gateway already possesses it: *"the credential has provably changed and the node is still rejected"* — this **excludes** explanation (a), which leaves (b). This is the machine-readable form of the operator's question "现在受到风控的 LLM 能通过重新登陆恢复使用吗": for DeepSeek the answer is no.
    - **Evidence, reset per credential**: `CredentialRenewalTracker.note_auth_failure()` now returns whether the failure landed on a credential *different from the last one it recorded* — and it must decide this **before** overwriting the record, or the evidence destroys itself on every call. `CooldownTracker` keeps `_fresh_credential_rejected[provider]` as a counter, **reset to 1 (not incremented)** each time a newer credential appears: every renewed credential gets its own chance, so one earlier rejection can never accumulate onto a credential that has never actually been tried. Reaching `GATEWAY_BAN_INFERENCE_MIN_FAILURES` (default 2) applies an **auto lease**.
    - **Semantics distinct from all four existing avoidance kinds**: a soft cooldown is the gateway's own inference about *load*; an operator lease is a human declaration about *policy*; a credential outage is ring-wide about *capacity*; an auto lease is the gateway's own inference about *the account itself*. It is marked `source="auto"` in `QuarantineStore`, which is persisted per provider (dir `0o700` / file `0o600`), so a restart cannot launder an auto lease into an undeletable one, and `release_auto_lease()` structurally **cannot** release an operator lease — precedence runs one way only (an operator declaration always overwrites an auto mark, never the reverse; an unknown/missing source is conservative-`operator`). Duration `GATEWAY_AUTO_QUARANTINE_DURATION` (default 259200s = 72h), shorter than the multi-day manual leases because an inference deserves less authority than a human's.
    - **Self-clearing, because an inference must be revisable**: the whole justification for the lease is "we tried a renewed credential and it was still refused"; a *newer* credential invalidates exactly that evidence, so the auto lease is lifted automatically when the fingerprint moves on, the evidence map is cleared, and the node re-enters via the OPT-022 recovery probation. Without this one wrong inference would freeze a perfectly-repaired account for 72h. Operator leases are exempt: they are never auto-lifted, and the fingerprint-comparison logic never touches them.
    - **Zero-touch dispatch**: the auto lease rides the existing OPT-021 hard-stop (folding the lease into the cooldown reading means every gate — probe, refresh, conversation GC, healer election, routing — honours it with no extra wiring), so an inferred ban produces **zero upstream calls** at any point, including as a failover fallback. `GATEWAY_DISABLE_BAN_INFERENCE=1` disables inference (and it co-disables with quarantine/ledger, whose disable switches would leave the conclusion nowhere to live). Evidence, auto-lease fingerprints and the rejection counts persist across restarts, so a restart is never a bypass. Telemetry: `ban_inference` on `/v1/diagnostics`, `GET /v1/admin/ban-inference` for the report, `DELETE /v1/admin/ban-inference` to lift auto leases early once the operator confirms the platform has lifted the ban. See `tests/test_ban_inference.py`.

25. **Web Frontend (`static/index.html`)**:
    - High-aesthetic Codex Light interface with grouped model selector and collapsible `<think>` drawer.
