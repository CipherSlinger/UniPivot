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
    """Flatten gateway messages the same way the providers would, for probing."""
    out: list[str] = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            out.append(c)
        elif isinstance(c, list):
            for b in c:
                if isinstance(b, dict):
                    for key in ("text", "content", "thinking"):
                        v = b.get(key)
                        if isinstance(v, str):
                            out.append(v)
                    if isinstance(b.get("content"), list):
                        for inner in b["content"]:
                            if isinstance(inner, dict) and isinstance(inner.get("text"), str):
                                out.append(inner["text"])
    return "\n".join(out)


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
                "provider": self.key,
                "model": model,
                "kw_keys": sorted(kw.keys()),
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
        already_ran = re.search(r"\[Tool Result for [A-Za-z0-9_-]+", prompt) is not None
        if not already_ran:
            name, args = chosen
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