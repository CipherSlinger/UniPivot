"""A scripted upstream that drives a REAL two-round agent tool loop.

This is the only instrument in the repository that answers the question the
project exists to answer: **does a real agent CLI actually work through this
gateway?** Every `tests/test_*.py` suite is offline and asserts on the gateway's
own objects; those suites would stay green even if the client rejected every
byte the gateway sent. Here a real `claude` / `codex` binary is the judge.

Round 1: the model asks to run a tool the CLIENT declared.
Round 2: the model sees the tool result and answers.

Every upstream request body is appended to `<E2E_LOG_DIR>/bodies_<port>.jsonl`,
so a run can assert on the prompt the upstream *really* received — the only
prompt that matters (CLAUDE.md sections 32b/39/43/60).

**The harness must never invent a tool name.** The gateway enforces a
declared-name allowlist (anti-fabrication, CLAUDE.md section 32), so a call to a
name the client never declared is correctly returned to the client as *text*.
An earlier version hardcoded Claude Code's `Bash`; Codex declares
`exec_command`, so that run measured the allowlist working rather than the loop
closing. The tool name and its required arguments are therefore read out of the
tool contract the gateway injected into the prompt.

Run (the runner `run_agent_loop.py` does this for you):
    E2E_LOG_DIR=/tmp/e2e TESTING=1 \
      .venv/bin/python -m uvicorn scripted_upstream:app \
      --host 127.0.0.1 --port 8400 --app-dir scripts/e2e
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

# The scripted upstream must satisfy the gateway's own import-time expectations
# without ever touching a real account: fake tokens, no probing, no refresh, no
# pacing, no dedup (dedup would collapse the two rounds into one flight).
os.environ.setdefault("TESTING", "1")
os.environ["GATEWAY_DISABLE_CREDENTIAL_WATCH"] = "1"
os.environ["GATEWAY_DISABLE_AUTO_REFRESH"] = "1"
os.environ["GATEWAY_DISABLE_PROBING"] = "1"
os.environ["GATEWAY_DISABLE_PACING"] = "1"
os.environ["GATEWAY_DISABLE_SINGLEFLIGHT"] = "1"
for _k in ("QWEN", "DEEPSEEK", "DOUBAO", "KIMI", "GLM"):
    os.environ.setdefault(f"{_k}_AUTH_TOKEN", f"fake_{_k.lower()}")

import server as srv  # noqa: E402

LOG_DIR = Path(os.environ.get("E2E_LOG_DIR") or tempfile.mkdtemp(prefix="e2e_"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG = LOG_DIR / f"bodies_{os.environ.get('E2E_PORT', '0')}.jsonl"

# What the tool is asked to do. A shell tool echoes it; a file tool writes it.
_WANT = os.environ.get("E2E_WANT", "echo e2e-ok")

# Directory the per-round markers are written into (only used when E2E_ROUNDS > 1).
_WANT_DIR = os.environ.get("E2E_WANT_DIR", ".")

# Parallel mode: two agents share ONE gateway, so a single global E2E_WANT would
# write one marker file for both and "claude ran 3 rounds" would be satisfiable
# by codex. In this mode the command is derived from the tool the request
# actually used to call (Bash => claude, exec_command => codex), which is the
# only attribution available once both clients' requests are interleaved in the
# same upstream log. Without it the measurement silently attributes the other
# client's rounds to this one.
_PARALLEL = os.environ.get("E2E_PARALLEL") == "1"

# Same-binary mode: two sessions of ONE client type, both routed to the SAME
# upstream account. The tool-name trick cannot separate them (both declare
# `Bash`), so attribution instead uses a **relative** marker path: each session
# runs in its own working directory, so `e2e_marks/ROUND_n` resolves to a
# different directory per session with no coordination at all. That is stronger
# than tagging, because it cannot be forged by the model -- it is the filesystem
# the client's own process is running in.
_SAME = os.environ.get("E2E_SAME") == "1"


def _client_tag(tool_name: str) -> str:
    """Which agent CLI declared this tool. Only used to separate markers."""
    if tool_name == "Bash":
        return "claude"
    if tool_name == "exec_command":
        return "codex"
    return tool_name or "unknown"


def _command_for(tool_name: str, round_no: int) -> str:
    """The command for one tool round, always carrying that round's marker.

    In multi-round mode **every** round writes `ROUND_<n>`, including round one.
    An earlier version emitted the bare base command for round 1, while the
    runner required `ROUND_1..N` — so a perfectly healthy 3-round run reported
    `tool_markers=2/3 ... missing ['ROUND_1']` on both real binaries. The run was
    correct; the harness disagreed with itself about what a round marker is.
    """
    if _SAME:
        # Relative on purpose: the client's own cwd decides where this lands,
        # so two concurrent sessions of the same binary write to two different
        # directories without the upstream knowing which is which.
        return (f"mkdir -p e2e_marks && touch e2e_marks/E2E_MARKER "
                f"&& touch e2e_marks/ROUND_{round_no}")
    if _PARALLEL:
        tag = _client_tag(tool_name)
        d = f"{_WANT_DIR}/{tag}"
        return f"mkdir -p {d} && touch {d}/E2E_MARKER && touch {d}/ROUND_{round_no}"
    if _ROUNDS > 1:
        return f"touch {_WANT} && touch {_WANT_DIR}/ROUND_{round_no}"
    return _WANT

# How many tool rounds to drive before answering. A two-round loop proves the
# closed loop; it does NOT prove the loop *stays* closed. Real agent sessions run
# 5-15 tool calls, and every round re-sends the accumulated history through the
# protocol handlers, so a defect in history replay, item indexing or stop
# truncation can be invisible at round 2 and fatal at round 5.
_ROUNDS = max(1, int(os.environ.get("E2E_ROUNDS", "1")))

# Which declared tool to drive, in preference order. It still has to be *declared
# by the client* — this list only decides among declared names, it never invents
# one. Each client declares a different shell tool: Claude Code `Bash`, Codex
# `exec_command`. Picking the first declared name instead (measured) selects
# `Agent` on Claude Code, which spawns a subagent rather than running the marker
# command, and selects a namespace note on Codex.
_PREFERRED = tuple(
    p for p in os.environ.get(
        "E2E_PREFERRED_TOOLS", "Bash,exec_command,shell,local_shell,Write,Edit"
    ).split(",") if p
)


def _flatten(messages) -> str:
    """Flatten gateway messages **exactly** as the providers do, for probing.

    This used to be a local re-implementation that joined every text fragment
    with no role labels. The providers do not do that: they call
    `providers.base.flatten_messages`, which renders `System: …` / `Developer: …`
    / `User: …` prefixes and appends the tool-call blocks. So every assertion
    made on `prompt_full` (has_tools_section, the declared-tool regex, "did the
    tool result reach upstream") was asserted against a prompt the upstream never
    sees — an instrument that quietly disagrees with the thing it measures.
    Delegating means the probe cannot drift from the product.
    """
    from providers.base import flatten_messages
    return flatten_messages(messages)


def _payload_text(messages, conversation_id) -> str:
    """Characters the provider would REALLY send upstream.

    In continuation mode (`conversation_id` set) the upstream session already
    holds the history, so only the new turn is sent — that is `continuation_content`,
    which is what every provider in this repo does. Measuring `_flatten(messages)`
    instead would report the full history for both modes and make a fix that
    reuses the upstream session look worthless (or, read the other way, make a
    session that is never reused look fine).
    """
    from providers.base import continuation_content
    return continuation_content(messages) if conversation_id else _flatten(messages)


def _newest_tool_result(messages):
    """Text of the newest tool result in the replayed history, or "".

    Two traps, both measured. (a) The gateway's own tool contract contains the
    literal `[Tool Result for <id>]`, so a bare substring test fires on round one
    and the loop never happens (CLAUDE.md section 60) — hence a real id is
    required. (b) A tool result may arrive as `role:"tool"`, as an Anthropic
    `tool_result`, or as a Responses `function_call_output` replayed into a user
    turn; require the rendered *text* rather than the id, because a provider
    renders `role:"tool"` as `Tool (name): …` with no id at all.
    """
    from providers.base import text_of

    for m in reversed(messages):
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or "").lower()
        if role in ("system", "developer"):
            continue          # the contract's example literal lives here
        body = text_of(m.get("content")).strip()
        if role == "tool":
            return body
        if re.search(r"\[Tool Result for [A-Za-z0-9_-]+", body):
            return body
    return ""


def _payload_has_tool_result(messages, conversation_id) -> bool:
    """Does the body this provider puts on the wire carry the newest tool result?

    The question the agent actually depends on: **did the newest tool result the
    client sent back reach the wire?** `prompt_full` / `prompt_tail` are built
    from the replayed history, which is NOT what a continuation request sends, so
    an assertion on them can be satisfied by text that never left the process
    (measured: a round-3 payload containing only the new tool call still shows a
    `[Tool Result for c1]` in the replayed history).
    """
    wanted = _newest_tool_result(messages)
    if not wanted:
        return False
    # Compare on a prefix only: clients append token counters and
    # <system-reminder> blocks after the real output, and those differ per turn.
    return wanted[:80] in _payload_text(messages, conversation_id)


def _declared_tools(prompt: str) -> list[str]:
    return re.findall(r"^- `([A-Za-z0-9_.-]+)`:.*$", prompt, re.M)


def _schema_for(prompt: str, name: str) -> dict | None:
    m = re.search(
        r"^- `" + re.escape(name) + r"`:.*?\n  Args: (\{.*\})$", prompt, re.M
    )
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except Exception:  # noqa: BLE001 - an unparseable schema is simply not a candidate
        return None


def _args_for(schema: dict) -> dict:
    props = schema.get("properties") or {}
    required = list(schema.get("required") or [])
    if not required:
        for cand in ("cmd", "command", "input", "path", "file_path"):
            if cand in props:
                required = [cand]
                break
    args = {}
    for key in required:
        if key in ("cmd", "command", "input"):
            args[key] = _WANT
        elif key in ("file_path", "path"):
            args[key] = os.environ.get("E2E_WANT_PATH", "e2e_marker.txt")
        elif key in ("content", "file_text"):
            args[key] = "e2e-ok"
        else:
            args[key] = "e2e"
    return args


def _pick_declared_tool(prompt: str) -> tuple[str, dict]:
    """Return (name, args) for a tool the CLIENT actually declared.

    Preference order matters: it decides *which declared tool executes
    observably*, never whether a name may be called. Falls back to
    ("no_such_tool", {}) so a missing contract is loud rather than silent.
    """
    declared = _declared_tools(prompt)
    ordered = [n for n in _PREFERRED if n in declared] + [n for n in declared if n not in _PREFERRED]
    for name in ordered:
        schema = _schema_for(prompt, name)
        if schema is None:
            continue
        args = _args_for(schema)
        if args:
            return name, args
    return "no_such_tool", {}


class LoopProvider:
    """Serves round 1 (tool call) and round 2 (final answer) from the same state."""

    def __init__(self, key: str):
        self.key = key

    async def chat(self, messages, model, **kw):
        prompt = _flatten(messages)
        chosen = _pick_declared_tool(prompt)

        with open(LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                # Wall-clock seconds. Without it nothing can prove the two clients'
                # requests actually OVERLAPPED in --parallel mode: two sequential
                # runs would produce byte-identical evidence to a real interleaving
                # and the "concurrent" claim would be an assertion, not a measurement.
                "t": time.time(),
                "provider": self.key,
                "model": model,
                "kw_keys": sorted(kw.keys()),
                # [SESSION-ID] The upstream conversation id the gateway hands the
                # provider is the ONLY observable proof that session affinity
                # really reuses a conversation across turns. Without it a turn-2
                # request that starts a brand-new upstream conversation looks
                # byte-identical to one that continues it.
                "conversation_id": kw.get("conversation_id"),
                # [SESSION-ID] What the provider really puts on the wire: full
                # history on a fresh conversation, only the new turn when the
                # upstream session is reused. This is the cost the agent pays.
                "payload_len": len(_payload_text(messages, kw.get("conversation_id"))),
                # What the model REALLY receives this turn. In continuation mode the
                # provider sends only the new turn, so a tool result that exists in
                # the replayed history but not in the payload was never delivered —
                # `prompt_full` alone cannot tell those apart.
                "payload_tail": _payload_text(messages, kw.get("conversation_id"))[-300:],
                "payload_has_tool_result": _payload_has_tool_result(
                    messages, kw.get("conversation_id")
                ),
                "tools": kw.get("tools"),
                "prompt_len": len(prompt),
                "has_tools_section": "## Tools Available" in prompt,
                "declared_tools": _declared_tools(prompt),
                "chosen_tool": chosen,
                "prompt_tail": prompt[-400:],
                "prompt_full": prompt,
            }, ensure_ascii=False) + "\n")

        # The tool result appears as "[Tool Result for <id>]" in the flattened
        # prompt. It MUST require a real id: the gateway's own tool contract
        # text contains the literal "[Tool Result for <id>]", so a bare
        # substring check fires on round one and the loop never happens
        # (measured; see CLAUDE.md section 60).
        done = len(re.findall(r"\[Tool Result for [A-Za-z0-9_-]+", prompt))
        if done < _ROUNDS:
            name, args = chosen
            # The marker differs per round so the runner can prove THAT round ran,
            # not merely that some round did.
            n = done + 1
            if isinstance(args, dict) and _ROUNDS > 1:
                for k in ("cmd", "command", "input"):
                    if isinstance(args.get(k), str):
                        # The command is rebuilt from the round number, never by
                        # rewriting the base one: the base marker proves the tool
                        # really ran at all, the per-round marker proves *which*
                        # round. An earlier version replaced the base command
                        # (measured) and every multi-round run reported "the
                        # client never executed the tool" while all five round
                        # markers sat on disk.
                        args = {**args, k: _command_for(name, n)}
                        break
            yield "I'll check.\n", {}
            yield (
                "```tool_call\n"
                + json.dumps({"name": name, "arguments": args}, ensure_ascii=False)
                + "\n```"
            ), {"conversation_id": "c1"}
            yield "", {"conversation_id": "c1"}
            return

        yield "The command printed e2e-ok. Done.", {"conversation_id": "c1"}
        yield "", {"conversation_id": "c1"}

    async def delete_conversation(self, *a, **kw):
        return None


def install() -> None:
    """Point every provider construction at the scripted upstream."""
    srv._make_provider = lambda pkey, client=None: LoopProvider(pkey)


install()
app = srv.app