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

import asyncio
import itertools
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

# [E2E-WRITE] Drive a FILE-WRITING tool instead of a shell tool.
#
# Why this is a separate mode rather than "just pick Write": the two clients
# write files through completely different declarations. Claude Code has a
# `Write` tool with `{file_path, content}`; Codex has NO `Write` at all — the
# only way it edits a file is the freeform `apply_patch` custom tool, whose
# argument is a raw patch under `{"input": ...}`. Driving `apply_patch` with a
# shell command (which `_args_for` used to do for any `input` field) produces a
# parse failure the client reports as "the patch is not a patch", so the run
# would measure the transport and prove nothing about editing code.
_WRITE = os.environ.get("E2E_WRITE") == "1"
_WANT_PATH = os.environ.get("E2E_WANT_PATH", "e2e_marker.txt")
_WANT_CONTENT = os.environ.get("E2E_WANT_CONTENT", "e2e-ok\n")

# Seconds to hold a response open, making concurrent requests pile up on purpose.
# Without it "peak in-flight" depends on timing luck: five fast requests may never
# overlap, and a broken semaphore would look identical to a working one.
_HOLD = float(os.environ.get("E2E_HOLD_SECONDS", "0") or 0)

# Emit TWO ```tool_call blocks in one assistant turn, with different arguments.
# Real agents do this constantly (a client that runs calls in parallel), and it is
# the one shape the gateway's per-protocol collection points handle by appending
# to a list — so a defect here drops the second call silently, leaving an agent
# that looks like it "sometimes ignores half the work".
_TWO_CALLS = os.environ.get("E2E_TWO_CALLS") == "1"

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
    """Which agent CLI declared this tool. Only used to separate markers.

    `exec` is Codex's lite-model tool (the JS orchestrator); it is the same
    client as `exec_command`, so it must land in the same marker directory or a
    `--parallel` lite run would write its markers somewhere the runner never
    looks (measured: `bodies=0` for a run that had really executed the tool,
    because the *attribution* rule, not the transport, had gone blind).
    """
    if tool_name == "Bash":
        return "claude"
    if tool_name in ("exec_command", "exec"):
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


def _second_call_marker(tool_name: str) -> str:
    """Where the SECOND call of a two-call turn proves it ran — per mode.

    The marker has to sit in the directory that *that mode's runner* inspects,
    and the three modes each use a different one (same-binary: the client's own
    cwd; two-vendor: `{want_dir}/{tag}`; single: `{want_dir}`). It used to be the
    hardcoded relative string `e2e_marks/SECOND_CALL`, which only resolves in
    same-binary mode: in the other two the directory does not exist, `touch`
    failed, and — because nothing asserted the marker there — a turn that
    executed only the first call still reported PASS (measured, CLAUDE.md
    section 68). A marker nobody checks is not evidence.
    """
    if _SAME:
        return "e2e_marks/SECOND_CALL"
    if _PARALLEL:
        return f"{_WANT_DIR}/{_client_tag(tool_name)}/SECOND_CALL"
    return f"{_WANT_DIR}/SECOND_CALL"

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
#
# [E2E-WRITE] The order must flip in write mode. Both clients declare their shell
# tool *and* a file tool, so leaving shell first means the stub asks `Bash` to run
# the sentence "Write the file … with exactly this content: e2e-ok" — a command
# that fails, or writes nothing, while the run looks like it exercised writing.
# That is the same mistake as the runner's hardcoded `--allowedTools Bash`, one
# layer down: the client was permitted to write, and the upstream never asked it
# to. Codex's file tool is the freeform custom tool `apply_patch` (it declares no
# `Write` at all), hence the two names.
# [RESPONSES-LITE] Codex's lite models (`gpt-5.6-sol` / `-terra` / `-luna`, the
# catalog entries flagged `use_responses_lite: true`) declare NO shell tool: the
# only way they run a command is `exec`, a freeform (custom) tool carrying a
# JavaScript orchestrator whose contract is `await tools.exec_command({cmd})`.
# The other declared names (`wait`, `request_user_input`, the `collaboration`
# sub-tools) cannot execute anything, so without `exec` in this list a lite run
# would fall through to declaration order and ask a non-executing tool to touch
# the marker — "the client never ran the tool" reported for a harness reason.
# It sits last among the shell tools so a client that declares a real shell tool
# still drives that one.
_DEFAULT_PREFERENCE = (
    "Write,apply_patch,Edit,Bash,exec_command,shell,local_shell"
    if _WRITE else
    "Bash,exec_command,shell,local_shell,exec,Write,Edit"
)
_PREFERRED = tuple(
    p for p in os.environ.get("E2E_PREFERRED_TOOLS", _DEFAULT_PREFERENCE).split(",") if p
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


def _exec_input(command: str) -> str:
    """Wrap a shell command in lite `exec`'s JavaScript orchestrator call.

    [RESPONSES-LITE] `exec` is freeform, so its argument is raw text — but not a
    shell command: its description (10,495 chars, captured verbatim in
    `tests/fixtures/codex_responses_lite_additional_tools.json`) declares
    `declare const tools: { exec_command(args: {cmd: string, …}) }`, called as
    `await tools.exec_command(...)`. Handing it `echo e2e-ok` verbatim — what
    `_args_for` did for every freeform `input` — is a JavaScript syntax error:
    the tool IS called, fails inside the client, and the marker never appears,
    which reads exactly like "the gateway dropped the tool call" (§60/§68's
    trap).

    No `return`, deliberately and measured: Codex evaluates `input` as a
    **script body**, so a top-level `return` is a parse error — the first
    version of this wrote `return await …` and the client answered
    `Script error:\nSyntaxError: Illegal return statement`, i.e. the tool ran,
    the command did not. A trailing expression is the script's value.

    The argument is an **object**, also measured: `exec`'s description says
    "Nested tool methods take either a string or an object", but the real
    `exec_command` implementation rejects the string form with
    `Script error:\ntool \`exec_command\` expects a JSON object for arguments`
    — the script ran, the command never did, and the marker stayed absent,
    which reads exactly like "the gateway dropped the call". `{"cmd": …}` is
    the shape the flat models' own `exec_command` declaration uses.
    """
    return "await tools.exec_command(" + json.dumps({"cmd": command}) + ")"


def _with_command(name: str, args: dict, command: str) -> dict:
    """Return `args` with the tool's command-carrying field replaced.

    One place, because the two callers below (per-round markers, second-call
    markers) must produce the *same* wire shape as the first call. Re-deriving
    it at each site is how a lite run ends up sending a bare shell command to a
    JavaScript tool on round two only.
    """
    for key in ("cmd", "command", "input"):
        if isinstance(args.get(key), str):
            value = _exec_input(command) if (key == "input" and name == "exec") else command
            return {**args, key: value}
    return args


def _args_for(schema: dict, name: str = "") -> tuple[dict, str]:
    """Return (wire arguments, the shell command they carry or "").

    The command is returned rather than re-derived by the caller. The first
    version of this read it back out of `args` (`args["input"] == _WANT`), which
    is exactly wrong for lite `exec`: its `input` is the JavaScript wrapper, not
    the command, so the marker rewrite silently stopped happening and every
    multi-round lite run would have reported "round 1 only" for a harness
    reason (measured). One producer, one value.
    """
    props = schema.get("properties") or {}
    required = list(schema.get("required") or [])
    if not required:
        for cand in ("cmd", "command", "input", "path", "file_path"):
            if cand in props:
                required = [cand]
                break
    args = {}
    command = ""
    for key in required:
        if key in ("cmd", "command"):
            args[key] = _WANT
            command = _WANT
        elif key == "input":
            # [E2E-WRITE] A freeform tool's `input` is raw text, not a shell
            # command. Codex's `apply_patch` is the ONE way Codex (non-lite)
            # writes a file (it declares no `Write`), so driving it with a
            # `touch` command tests the transport and proves nothing about
            # editing code.
            if name == "exec":
                args[key] = _exec_input(_WANT)
                command = _WANT
            else:
                args[key] = _patch_text() if _WRITE else _WANT
                command = "" if _WRITE else _WANT
        elif key in ("file_path", "path"):
            args[key] = _WANT_PATH
        elif key in ("content", "file_text"):
            args[key] = _WANT_CONTENT
        else:
            args[key] = "e2e"
    return args, command


def _patch_text() -> str:
    """A minimal `apply_patch` body that CREATES the marker file.

    Delta-only on purpose: a patch that edits a file the harness pre-created
    would pass even if the client never sent our content anywhere, because the
    file already existed. `*** Add File:` can only succeed if the patch really
    arrived, character for character.
    """
    rel = os.path.basename(_WANT_PATH)
    body = "".join(f"+{line}\n" for line in _WANT_CONTENT.splitlines() or [""])
    return f"*** Begin Patch\n*** Add File: {rel}\n{body}*** End Patch\n"


def _pick_declared_tool(prompt: str) -> tuple[str, dict, str]:
    """Return (name, args, command) for a tool the CLIENT actually declared.

    Preference order matters: it decides *which declared tool executes
    observably*, never whether a name may be called. Falls back to
    ("no_such_tool", {}, "") so a missing contract is loud rather than silent.

    `command` is the shell command the chosen tool will run, carried alongside
    the wire arguments because the per-round and second-call markers replace it:
    they must go through `_with_command` rather than patching `args` directly,
    or a lite `exec` call would receive a bare command on rounds 2+ (see
    `_exec_input`). It is `""` when the tool runs no command (`apply_patch`).
    """
    declared = _declared_tools(prompt)
    ordered = [n for n in _PREFERRED if n in declared] + [n for n in declared if n not in _PREFERRED]
    for name in ordered:
        schema = _schema_for(prompt, name)
        if schema is None:
            continue
        args, command = _args_for(schema, name)
        if args:
            return name, args, command
    return "no_such_tool", {}, ""


# Distinct ids for distinct conversations. `itertools.count` is safe here: the
# mint happens synchronously inside one event loop.
_MINTED = itertools.count(1)

# In-flight / peak per ACCOUNT, observed at the upstream itself.
_INFLIGHT: dict = {}
_PEAK: dict = {}


class LoopProvider:
    """Serves round 1 (tool call) and round 2 (final answer) from the same state."""

    def __init__(self, key: str):
        self.key = key

    async def chat(self, messages, model, **kw):
        prompt = _flatten(messages)
        chosen = _pick_declared_tool(prompt)

        # [MOUNT] A real upstream MINTS a conversation id and hands it back; the
        # gateway reuses it on every later turn. This stub used to answer a
        # literal "c1" for every conversation, which made the one property the
        # run exists to measure unobservable: with one constant id, "each
        # session reuses its own upstream conversation" and "all sessions share
        # one conversation" produce byte-identical evidence. Measured in
        # `--sessions 2` mode: 8 bodies from 2 sessions, all `c1`.
        # The counter is module-level on purpose: `LoopProvider` is constructed
        # per request (the gateway's `_make_provider` is a factory), so a
        # per-instance counter would hand every request the same "…_1".
        conv = kw.get("conversation_id") or f"upstream_{self.key}_{next(_MINTED)}"

        # [CONCURRENCY] How many requests this ACCOUNT has in flight at once.
        # This is the quantity an anti-bot system scores ("one account, several
        # parallel sessions"), and the only place it is directly observable: the
        # gateway's own semaphore is what is supposed to bound it, so measuring
        # the gateway's counter would be asking the suspect for its alibi.
        _INFLIGHT[self.key] = _INFLIGHT.get(self.key, 0) + 1
        _PEAK[self.key] = max(_PEAK.get(self.key, 0), _INFLIGHT[self.key])
        try:
            if _HOLD:
                await asyncio.sleep(_HOLD)
            async for item in self._chat(messages, model, conv, prompt, chosen, **kw):
                yield item
        finally:
            _INFLIGHT[self.key] -= 1

    async def _chat(self, messages, model, conv, prompt, chosen, **kw):

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
                # The conversation this request was actually served in: the
                # incoming id when the gateway continued one, a freshly minted
                # one otherwise. `conversation_id` alone cannot answer "do two
                # sessions stay apart?" because a fresh conversation logs None.
                "served_conv": conv,
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
                # [AGENT-LOOP] The reasoning decision the gateway actually made,
                # measured where every upstream decision is measured: on the call
                # itself. `kw_keys` lists that a `thinking` key exists, but not its
                # VALUE, so before this field a run could not tell "the client asked
                # for `xhigh` and the gateway opened thinking" from "the gateway
                # silently ignored the effort and sent `thinking=False`". Qwen maps
                # this straight to `chat_mode: think|quick`, i.e. it is the
                # difference between a reasoning model and a plain one.
                "thinking": bool(kw.get("thinking")),
                # [PROTOCOL] The output budget the client declared, as the gateway
                # forwards it. §96 made the gateway ENFORCE it at the cut, so a
                # real client's number decides whether a normal agent turn is
                # truncated: if Claude Code declared a small budget and the esti-
                # mator over-counted, every tool loop would die at round two with
                # `finish_reason: "length"`. `kw_keys` shows the key exists; only
                # the value can say whether the enforcement is safe on real traffic.
                #
                # This field is only readable *because* the gateway stopped
                # fabricating a value here (`... or 4096` / `... or 2048`): the
                # first version of this instrument reported `budget=4096` for a
                # client that declares nothing at all, i.e. it measured the
                # gateway's own fallback and printed it as the client's words.
                # `None` now means exactly "the client declared no budget".
                "max_tokens": kw.get("max_tokens"),
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
        # A **round** is one user→tool-call→result cycle, not one tool result.
        # With `E2E_TWO_CALLS` each round produces two calls and therefore two
        # results, so counting results made round one look like two completed
        # rounds: the stub answered instead of driving round two, and a healthy
        # run reported `rounds=1/2` (measured, CLAUDE.md section 68). The
        # per-round divisor is what keeps the two notions apart.
        results = len(re.findall(r"\[Tool Result for [A-Za-z0-9_-]+", prompt))
        calls_per_round = 2 if _TWO_CALLS else 1
        done = results // calls_per_round
        if done < _ROUNDS:
            name, args, command = chosen
            # The marker differs per round so the runner can prove THAT round ran,
            # not merely that some round did.
            n = done + 1
            # `_SAME` is included because that mode's runner looks for the marker
            # in `<session>/e2e_marks/`, and this branch is what puts it there:
            # at one round the base `_WANT` is the relative `touch E2E_MARKER`,
            # which the client's cwd resolves to the session root instead, so a
            # healthy `--sessions 2 --rounds 1` reported "never executed a tool
            # (marker absent)" on both sessions (measured, CLAUDE.md section 68).
            # [E2E-WRITE] Never in write mode. `_command_for` builds a shell
            # command, and the argument here is a `content` string or an
            # `apply_patch` body: replacing either with `mkdir -p e2e_marks && …`
            # would write that command as the file's content, which the content
            # assertion then reports as "mangled arguments" — blaming the transport
            # for the harness's own rewrite. Write mode is single-round only and
            # uses a relative path, so each client's cwd already does the
            # attribution this branch exists to provide.
            if (_ROUNDS > 1 or _SAME) and not _WRITE and command:
                # The command is rebuilt from the round number, never by
                # rewriting the base one: the base marker proves the tool
                # really ran at all, the per-round marker proves *which*
                # round. An earlier version replaced the base command
                # (measured) and every multi-round run reported "the
                # client never executed the tool" while all five round
                # markers sat on disk.
                command = _command_for(name, n)
                args = _with_command(name, args, command)
            calls = [(name, args)]
            if _TWO_CALLS and command:
                # The second call must be *distinguishable* in the marker
                # directory, otherwise "both ran" and "one ran twice" look the
                # same on disk. Built through `_with_command` so a lite `exec`
                # call keeps its JavaScript wrapper instead of receiving a bare
                # shell command (which the client would report as a JS syntax
                # error).
                second_marker = _second_call_marker(name)
                calls.append(
                    (name, _with_command(name, args, f"{command} && touch {second_marker}"))
                )
            yield "I'll check.\n", {}
            for cname, cargs in calls:
                yield (
                    "```tool_call\n"
                    + json.dumps({"name": cname, "arguments": cargs}, ensure_ascii=False)
                    + "\n```"
                ), {"conversation_id": conv}
            yield "", {"conversation_id": conv}
            return

        yield "The command printed e2e-ok. Done.", {"conversation_id": conv}
        yield "", {"conversation_id": conv}

    async def delete_conversation(self, *a, **kw):
        return None


def install() -> None:
    """Point every provider construction at the scripted upstream."""
    srv._make_provider = lambda pkey, client=None: LoopProvider(pkey)


install()
app = srv.app


@app.get("/e2e/concurrency")
async def _concurrency():
    """Peak in-flight requests per ACCOUNT, as observed by the upstream.

    Exists because the gateway's own counters cannot answer the question: the
    per-node semaphore is the suspect, not the witness (CLAUDE.md section 36
    measured peak 6 against a qwen account configured for 3 by reading the
    gateway's side and nearly missed it). Zero upstream calls to read.
    """
    return {"peak": dict(_PEAK), "current": dict(_INFLIGHT)}
