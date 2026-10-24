#!/usr/bin/env python3
"""Drive a REAL `claude` / `codex` binary against this gateway and judge the loop.

Why this exists
---------------
Every suite in `tests/` is offline: it asserts on the gateway's own objects and
scripted transports. Those suites stay green even when the client rejects every
byte the gateway sends — the wire contract can be wrong in a way no offline
assertion can see. This runner closes that gap by making the real binary the
judge, which is the only instrument that answers the question the project
exists to answer (CLAUDE.md section 60).

What it proves
--------------
1. The client's request reaches the gateway and the gateway answers on the wire
   (status line, framing) the client accepts.
2. Round 1 causes the client to **really execute a tool** — asserted from
   outside the gateway, on the filesystem, not from the gateway's own claim.
3. Round 2 receives the tool result and answers — the loop closed.
4. The upstream prompt really contained the tool contract and the tool result
   (asserted on the captured bodies, not on the client's stdout).

Design notes that cost real debugging time
------------------------------------------
* `--rounds > 1` **appends** the per-round marker command to the base one. An
  earlier version replaced it, so every multi-round run reported "the client
  never executed the tool" while all five ROUND_* markers sat on disk (measured).
  The base marker proves the tool ran at all; the per-round one proves *which*
  round — both are needed, so the command is `base && touch ROUND_n`.
* Each client gets its own working directory. Sharing one makes "round 3 of
  claude" satisfiable by "round 3 of codex".
* A probe server must be started in its **own** Bash invocation: `pkill -f` in a
  compound command kills its own process group and returns 144, so the `nohup`
  half never runs and the next `curl` gets 000.
* The round detector needs a real id (`[Tool Result for [A-Za-z0-9_-]+`); the
  gateway's tool-contract text itself contains `[Tool Result for <id>]`.
* The scripted upstream never invents a tool name — it reads one out of the
  contract — otherwise the declared-name allowlist correctly returns the call
  as text and the run measures the allowlist, not the loop.

Usage
-----
    .venv/bin/python scripts/e2e/run_agent_loop.py            # both clients present
    .venv/bin/python scripts/e2e/run_agent_loop.py --client claude
    .venv/bin/python scripts/e2e/run_agent_loop.py --rounds 10  # long agent session
    .venv/bin/python scripts/e2e/run_agent_loop.py --parallel --rounds 10  # both at once, one gateway
    .venv/bin/python scripts/e2e/run_agent_loop.py --keep     # keep artifacts for inspection

Exit code 0 means every client that is installed closed the full loop: the client
really ran each round's tool (one marker file per round), and its final answer
came after it had seen every tool result.

A missing binary is *skipped*, never silently counted as a pass.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
APP_DIR = Path(__file__).resolve().parent


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_ready(url: str, timeout: float = 60.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status < 500:
                    return True
        except Exception:  # noqa: BLE001 - not up yet
            time.sleep(0.4)
    return False


def _fetch_peaks(port: int) -> dict:
    """Peak in-flight requests per account, as observed by the upstream itself."""
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/e2e/concurrency", timeout=5
        ) as r:
            return json.load(r).get("peak") or {}
    except Exception:  # noqa: BLE001 - telemetry must never fail a run
        return {}


def _configured_limits() -> dict:
    """max_concurrency per provider, read from the gateway's own spec table.

    Read rather than hardcoded: a test that asserts against a copy of the limit
    keeps passing after the limit changes, which is the failure mode that makes
    concurrency regressions invisible (CLAUDE.md section 36).
    """
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from providers.failover import PROVIDER_RISK_SPECS
        return {k: m.max_concurrency for k, m in PROVIDER_RISK_SPECS.items()}
    except Exception:  # noqa: BLE001
        return {}


def _start_gateway(port: int, log_dir: Path, want: str, rounds: int, want_dir: Path,
                   parallel: bool = False, same: bool = False,
                   extra_env: dict | None = None,
                   two_calls: bool = False) -> subprocess.Popen:
    """Start the scripted upstream.

    `want` has to reach the **server**: it is the model in the scripted upstream
    that decides which command to emit, so setting it only in the client's
    environment (an easy mistake) leaves the server emitting its default and the
    marker is never created even though the whole loop worked.
    """
    env = {
        **os.environ,
        "TESTING": "1",
        "E2E_LOG_DIR": str(log_dir),
        "E2E_PORT": str(port),
        "E2E_WANT": want,
        "E2E_ROUNDS": str(rounds),
        "E2E_WANT_DIR": str(want_dir),
        "E2E_PARALLEL": "1" if parallel else "0",
        "E2E_SAME": "1" if same else "0",
        "E2E_TWO_CALLS": "1" if two_calls else "0",
        "PYTHONUNBUFFERED": "1",
        **(extra_env or {}),
    }
    log = open(log_dir / f"srv_{port}.log", "wb")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "scripted_upstream:app",
         "--host", "127.0.0.1", "--port", str(port), "--app-dir", str(APP_DIR)],
        cwd=str(REPO), env=env, stdout=log, stderr=subprocess.STDOUT,
    )
    proc._e2e_log = log  # type: ignore[attr-defined]
    return proc


def _bodies(log_dir: Path, port: int) -> list[dict]:
    p = log_dir / f"bodies_{port}.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except Exception:  # noqa: BLE001
                pass
    return out


def _marker_seen(marker: Path) -> bool:
    return marker.exists()


# How to tell one client's upstream bodies from the other's when both share a
# gateway: each CLI declares its own shell tool and no other.
_OWN_TOOL = {"claude": "Bash", "codex": "exec_command"}


def _bodies_for(bodies: list[dict], client: str) -> list[dict]:
    tool = _OWN_TOOL.get(client)
    if not tool:
        return bodies
    return [b for b in bodies if tool in (b.get("declared_tools") or [])]


def _failures_for(failures: list[str], owner: str, clients: list[str]) -> list[str]:
    """The failures that make ``owner``'s verdict FAIL.

    Every per-client failure is appended as `f"{name}: ..."`, and the verdict
    used to *recompute* a pass/fail from a hand-picked subset of those same
    facts. The two disagreed the moment a check was added to one and not the
    other: a run that fired `only the FIRST of two tool calls … SECOND_CALL
    marker absent` still printed `codex: PASS` on the line above it (measured,
    CLAUDE.md section 68). A verdict that disagrees with the failure list is
    the "the report lies" defect this repo keeps fixing, so the verdict now
    derives PASS from the list itself and there is one place a check can be
    added.

    A failure prefixed with *another* client's name is that client's alone. A
    failure with no client prefix is ring-wide — "their contexts bleed into
    each other", a group exceeded its concurrency limit — and fails every
    session, because no session can be called healthy while the shared gateway
    is doing that.
    """
    out = []
    for m in failures:
        owners = [c for c in clients if m.startswith(f"{c}: ")]
        if not owners or owner in owners:
            out.append(m)
    return out


def _blind_continuations(bodies: list[dict]) -> list[tuple[int, dict]]:
    """Continuation requests that reached upstream WITHOUT the tool result.

    A continuation is *the request the gateway handed an upstream conversation
    id to* — that, not its position in the log, is what makes it a second turn.
    Slicing `bodies[1:]` assumed one session's requests are contiguous: with N
    sessions sharing one gateway the second session's FIRST request (correctly
    carrying no result) landed in the slice, so a fully healthy run reported
    `round2_saw_result: false` (measured, CLAUDE.md section 68). Round-one
    bodies have no result by definition and are excluded by the same predicate
    that identifies them: no incoming conversation id.
    """
    return [
        (i, b) for i, b in enumerate(bodies)
        if b.get("conversation_id") and not b.get("payload_has_tool_result")
    ]


def _every_continuation_saw_its_result(bodies: list[dict]) -> bool:
    """True when there is at least one continuation and none of them went blind."""
    has_continuation = any(b.get("conversation_id") for b in bodies)
    return has_continuation and not _blind_continuations(bodies)


def _overlap_ms(bodies: list[dict]) -> int:
    """Milliseconds during which BOTH clients had a request in flight.

    The whole point of `--parallel` is interleaving, and a green verdict alone
    cannot distinguish it from two sequential runs — the markers and body counts
    are identical either way. So the claim is measured: each client's upstream
    calls are taken in wall-clock order from the logged `t`, and the span where
    both are active is computed as the union overlap. Zero means the two clients
    never actually ran at the same time and the run proves nothing about sharing.
    """
    spans = {}
    for name in ("claude", "codex"):
        ts = sorted(b["t"] for b in _bodies_for(bodies, name) if isinstance(b.get("t"), (int, float)))
        if ts:
            spans[name] = (ts[0], ts[-1])
    if len(spans) < 2:
        return 0
    (a0, a1), (b0, b1) = spans["claude"], spans["codex"]
    return max(0, int((min(a1, b1) - max(a0, b0)) * 1000))


def _alternation(bodies: list[dict]) -> str:
    """"C"/"X" per upstream call in time order, plus how often the owner changed.

    A real interleaving alternates; two runs that merely happened to share a log
    do not. Measured on a good run: `XCXCXCXCXCXCXCXCXCXCXC`, 21 alternations
    across 22 bodies — every single call switched client, which is what two
    agents driving one gateway actually looks like.
    """
    ordered = sorted(
        (b for b in bodies if isinstance(b.get("t"), (int, float))), key=lambda b: b["t"]
    )
    seq = []
    for b in ordered:
        declared = b.get("declared_tools") or []
        seq.append("C" if "Bash" in declared else ("X" if "exec_command" in declared else "?"))
    switches = sum(1 for a, b in zip(seq, seq[1:]) if a != b)
    return "".join(seq), switches


def _collect(client: str, exit_code: int, bodies: list[dict], work: Path,
             marker: Path, stdout: str) -> dict:
    markers = sorted(p.name for p in work.glob("ROUND_*"))
    base = work / "E2E_MARKER"
    return {
        "client": client,
        "exit": exit_code,
        "bodies": len(bodies),
        "marker": marker.exists() or base.exists(),
        "markers": markers,
        "declared": (bodies[-1]["declared_tools"] if bodies else []),
        "chosen": (bodies[-1]["chosen_tool"] if bodies else None),
        "has_tools_section": bool(bodies and bodies[-1]["has_tools_section"]),
        # [HARNESS] Assert on the payload the upstream REALLY receives, not on
        # `prompt_tail`: in continuation mode the provider sends only the new
        # turn, so a tool result that exists in the replayed history but never
        # made it onto the wire would pass a `prompt_tail` check and fail the
        # agent. The two are different prompts and only one of them is real.
        # EVERY round after the first must carry one — `any()` would let a run
        # that loses the result at round 3 still report PASS.
        #
        # "After the first" is a property of a REQUEST, not of a list index:
        # a continuation is exactly the request the gateway hands an upstream
        # conversation id to. Slicing `bodies[1:]` instead assumed one session's
        # bodies are contiguous, so with N sessions sharing a log the second
        # session's FIRST request (no result yet, correctly) landed in the slice
        # and the whole run reported `round2_saw_result: false` while every real
        # continuation carried its result (measured, CLAUDE.md section 68).
        # The same-binary verdict ignored the field, so it never failed a green
        # run — the parallel verdict did use it, which makes the derivation,
        # not the assertion, the thing to fix.
        "round2_saw_result": _every_continuation_saw_its_result(bodies),
        # [SESSION-ID] The upstream conversation id the gateway hands the provider
        # is the only observable proof that session affinity works for THIS client.
        # Neither real binary ever sends `conversation_id` (Claude Code puts its
        # identity in `metadata.user_id`, Codex in `client_metadata.thread_id`), so
        # before the fix every turn started a brand-new upstream conversation and
        # re-sent the full history. A multi-round run that never reuses one is that
        # defect, measured on the wire instead of inferred from the gateway's state.
        "upstream_convs": [b.get("conversation_id") for b in bodies],
        "session_reused": any(
            b.get("conversation_id") for b in bodies[1:]
        ) if len(bodies) > 1 else False,
        "stdout": stdout[-2000:],
    }


def _claude_argv(port: int, work: Path, log_dir: Path, marker: Path):
    want = f"touch {marker}"
    env = {
        **os.environ,
        "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{port}",
        "ANTHROPIC_API_KEY": "test-key",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": "claude-3-5-haiku-20241022",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "CLAUDE_CONFIG_DIR": str(log_dir / "claudehome"),
        "E2E_WANT": want,
    }
    (log_dir / "claudehome").mkdir(exist_ok=True)
    return [
        shutil.which("claude"), "-p", f"Run `{want}` then tell me what it printed.",
        "--model", "claude-3-7-sonnet", "--allowedTools", "Bash",
    ], env


def run_claude(port: int, work: Path, log_dir: Path, marker: Path) -> dict:
    """Drive `claude -p` and require a real tool execution plus a closed loop."""
    if not shutil.which("claude"):
        return {"client": "claude", "skipped": "claude not on PATH"}
    argv, env = _claude_argv(port, work, log_dir, marker)
    proc = subprocess.run(argv, cwd=str(work), env=env, capture_output=True,
                          text=True, timeout=240, stdin=subprocess.DEVNULL)
    (log_dir / "claude.out").write_text(proc.stdout + "\n---STDERR---\n" + proc.stderr)
    bodies = _bodies_for(_bodies(log_dir, port), "claude")
    return _collect("claude", proc.returncode, bodies, work, marker, proc.stdout)


def _codex_argv(port: int, work: Path, log_dir: Path, marker: Path):
    """Build the codex command + env, or (None, None) when codex is absent."""
    codex = shutil.which("codex")
    if not codex:
        return None, None

    want = f"touch {marker}"
    # `OPENAI_BASE_URL` is NOT honoured by codex 0.145.0 — measured against the
    # real binary: with only that env var set, `ModelProviderInfo` reports
    # `base_url: None` and the CLI connects to `wss://api.openai.com/v1/responses`
    # (the binary contains zero occurrences of the string `OPENAI_BASE_URL`).
    # A custom provider must be declared in `$CODEX_HOME/config.toml`, which is
    # what this writes. Anyone testing the gateway with env vars alone is not
    # testing the gateway at all — they are testing OpenAI.
    codex_home = log_dir / "codexhome"
    codex_home.mkdir(exist_ok=True)
    (codex_home / "config.toml").write_text(
        f'''model_provider = "unipivot"
model = "gpt-5.5"
model_reasoning_effort = "low"
disable_response_storage = true
approval_policy = "never"
sandbox_mode = "danger-full-access"

[model_providers.unipivot]
name = "unipivot"
base_url = "http://127.0.0.1:{port}/v1"
wire_api = "responses"
requires_openai_auth = false

[projects."{work}"]
trust_level = "trusted"
''',
        encoding="utf-8",
    )
    env = {
        **os.environ,
        "CODEX_HOME": str(codex_home),
        "E2E_WANT": want,
    }
    return [
        codex, "exec", "--skip-git-repo-check",
        "--dangerously-bypass-approvals-and-sandbox",
        "-m", "gpt-5.5",
        f"Run `{want}` then tell me what it printed.",
    ], env


def run_codex(port: int, work: Path, log_dir: Path, marker: Path) -> dict:
    """Drive `codex exec` and require a real tool execution plus a closed loop."""
    argv, env = _codex_argv(port, work, log_dir, marker)
    if not argv:
        return {"client": "codex", "skipped": "codex not on PATH"}
    proc = subprocess.run(
        argv, cwd=str(work), env=env, capture_output=True, text=True, timeout=240,
        stdin=subprocess.DEVNULL,
    )
    (log_dir / "codex.out").write_text(proc.stdout + "\n---STDERR---\n" + proc.stderr)
    bodies = _bodies_for(_bodies(log_dir, port), "codex")
    return _collect("codex", proc.returncode, bodies, work, marker, proc.stdout)


def _run_same(log_dir: Path, work: Path, args, results: list, failures: list) -> int:
    """N concurrent sessions of the SAME client against ONE gateway.

    This is the harder shape and the one the risk line exists for: several agent
    sessions of one binary, all routed to the same upstream account. Different
    vendors exercising one gateway is easy; the "one account, several parallel
    sessions" signature is what per-node semaphores and pacing are there to bound,
    and it is the shape an anti-bot system scores as automation.

    Attribution does not use the tool name (both sessions declare `Bash`). Each
    session runs in its own working directory and the scripted upstream emits a
    *relative* marker path, so `e2e_marks/ROUND_n` resolves per session with no
    coordination — the filesystem the client really runs in decides it.
    """
    port = _free_port()
    root = work / "same"
    root.mkdir()
    names = [f"{args.client}-{i + 1}" for i in range(args.sessions)]
    procs = {}
    # Hold each response briefly so concurrent sessions really pile up: a fast
    # stub makes "peak in-flight" a matter of timing luck, and a semaphore that
    # leaks would then be indistinguishable from one that holds.
    gw = _start_gateway(port, log_dir, want="touch E2E_MARKER", rounds=args.rounds,
                        want_dir=root, parallel=True, same=True,
                        extra_env={"E2E_HOLD_SECONDS": "0.4"},
                        two_calls=getattr(args, "two_calls", False))
    try:
        if not _wait_ready(f"http://127.0.0.1:{port}/health"):
            print("[e2e] same: GATEWAY NOT READY")
            return 1
        print(f"[e2e] same: one gateway on :{port}; starting {len(names)} "
              f"{args.client} sessions…")
        for name in names:
            d = root / name
            d.mkdir()
            marker = d / "e2e_marks" / "E2E_MARKER"
            if args.client == "claude":
                argv, env = _claude_argv(port, d, log_dir, marker)
            else:
                argv, env = _codex_argv(port, d, log_dir, marker)
            if not argv or not argv[0]:
                procs[name] = None
                continue
            procs[name] = subprocess.Popen(
                argv, cwd=str(d), env=env, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, stdin=subprocess.DEVNULL,
            )
        outs = {}
        for name, p in procs.items():
            if p is None:
                continue
            outs[name] = p.communicate(timeout=300)[0]
        bodies_all = _bodies(log_dir, port)
        peaks = _fetch_peaks(port)
    finally:
        gw.terminate()
        try:
            gw.wait(timeout=10)
        except subprocess.TimeoutExpired:
            gw.kill()
        getattr(gw, "_e2e_log", None) and gw._e2e_log.close()  # type: ignore[attr-defined]

    for name in names:
        p = procs.get(name)
        if p is None:
            results.append({"client": name, "skipped": f"{args.client} not on PATH"})
            continue
        d = root / name
        marks = d / "e2e_marks"
        res = _collect(name, p.returncode, bodies_all, marks, marks / "E2E_MARKER",
                       outs.get(name, ""))
        res["rounds_expected"] = args.rounds
        # Both sessions declare `Bash`, so upstream bodies CANNOT be attributed
        # per session — only the marker directories can, and they already do.
        # Reporting the shared total as this session's `bodies` would print
        # `bodies=12` for each of two 6-request sessions: a per-session number
        # that is really the sum. Say "shared" rather than invent a split.
        res.pop("bodies", None)
        res["shared_bodies"] = len(bodies_all)
        res["attributable_by"] = "marker-directory"
        results.append(res)
        print(f"[e2e] same {name}: "
              f"{json.dumps({k: v for k, v in res.items() if k not in ('stdout', 'declared')}, ensure_ascii=False)}")
        if not res["marker"]:
            failures.append(f"{name}: never executed a tool (marker absent)")
        if args.rounds > 1 and len(res["markers"]) < args.rounds:
            # At one round no per-round marker is ever written (see the parallel
            # mode's identical guard); `--sessions N --rounds 1` was red on a
            # healthy run for the same reason.
            failures.append(
                f"{name}: only {len(res['markers'])}/{args.rounds} rounds really ran "
                f"while {len(names) - 1} sibling session(s) of the same binary sent requests"
            )
        # Shared log, shared evidence: every session is expected to establish and
        # reuse its own upstream conversation, so the shared log must contain at
        # least one continuation. Reported as a shared fact, never as per-session.
        if args.rounds > 1 and not res.get("session_reused"):
            # Prefixed with the session name on purpose: the fact is read from a
            # shared log, but the verdict below derives this session's PASS from
            # its own failures, so an unprefixed message would print PASS on the
            # line above it.
            failures.append(
                f"{name}: no request in the shared log reused an upstream conversation "
                f"across {args.rounds} rounds (all bodies: {res.get('upstream_convs')})"
            )
        if not res["has_tools_section"]:
            failures.append(f"{name}: its upstream prompt carried no tool contract")
        if getattr(args, "two_calls", False) and not (marks / "SECOND_CALL").exists():
            # The second call in a parallel-tool-use turn never ran. Agents emit
            # these constantly, and a client that executes only the first call
            # looks like it "sometimes skips half the work" with no error.
            failures.append(
                f"{name}: only the FIRST of two tool calls in one turn executed "
                "(SECOND_CALL marker absent)"
            )

    # [SESSION-ID] Two sessions of one binary MUST end up in two different
    # upstream conversations, and each must stay in its own. The scripted
    # upstream used to answer a literal "c1" for every conversation, which made
    # this unobservable: "both sessions reuse their own" and "both share one"
    # produced byte-identical logs (measured: 8 bodies, 2 sessions, all `c1`).
    served = [b.get("served_conv") for b in bodies_all if b.get("served_conv")]
    fresh = [b for b in bodies_all if not b.get("conversation_id")]
    if args.rounds > 1:
        if len(fresh) > len(names):
            failures.append(
                f"{len(names)} sessions started {len(fresh)} upstream conversations "
                "— a session lost its conversation partway through"
            )
        if len(set(served)) < len(names):
            failures.append(
                f"{len(names)} sessions were served by only {len(set(served))} upstream "
                f"conversation(s) {sorted(set(served))} — their contexts bleed into each other"
            )
        # Every continuation must carry the tool result of the turn it continues.
        blind = [i for i, _ in _blind_continuations(bodies_all)]
        if blind:
            failures.append(
                f"{len(blind)} continuation request(s) reached upstream without the tool "
                f"result (body indices {blind}) — the model was asked to continue a loop "
                "whose result it never saw"
            )

    seq, switches = _alternation(bodies_all)
    print("\n================ E2E VERDICT (same binary, N sessions) ================")
    for r in results:
        if "skipped" in r:
            print(f"  {r['client']}: SKIPPED")
            continue
        ok = not _failures_for(failures, r["client"], [x["client"] for x in results])
        print(f"  {r['client']}: {'PASS' if ok else 'FAIL'} "
              f"(rounds={len(r['markers'])}/{r['rounds_expected']})")
    print(f"  {len(bodies_all)} upstream bodies from {len(names)} sessions of one binary, "
          f"{len(set(served))} distinct upstream conversation(s)")
    if peaks:
        # What the ACCOUNT really saw, counted upstream. The per-node semaphore
        # exists to keep this bounded, and the gateway's own counter cannot
        # witness it: section 36 measured peak 6 against a qwen account limited
        # to 3 while the gateway's side looked fine.
        limits = _configured_limits()
        print(f"  peak in-flight per account: {peaks} (configured: "
              f"{ {k: limits.get(k) for k in peaks} })")
        for pk, peak in peaks.items():
            limit = limits.get(pk)
            if limit is not None and peak > limit:
                failures.append(
                    f"{pk}: {peak} concurrent upstream requests against a limit of "
                    f"{limit} — the semaphore did not cover the streamed body"
                )
    for f in failures:
        print(f"  !! {f}")
    if not args.keep:
        shutil.rmtree(log_dir, ignore_errors=True)
        print("  (artifacts removed; re-run with --keep to inspect)")
    else:
        print(f"  artifacts kept at {log_dir}")
    return 1 if failures else 0


def _run_parallel(log_dir: Path, work: Path, args, results: list, failures: list) -> int:
    """Both real binaries at once against ONE gateway.

    This is the setup developers actually run — two agents in two terminals on one
    machine — and nothing else in the repository exercises it end to end. It puts
    the two clients' requests through the same connection pool, the same per-node
    semaphores, the same singleflight manager and the same session-affinity
    registry, interleaved, for ten rounds each.

    Attribution: once both clients' requests share one upstream log, "which client
    ran this round" has to be read from the tool each request actually declared
    (`Bash` => claude, `exec_command` => codex), and the marker paths are derived
    the same way in the scripted upstream. Without that, one client's rounds are
    silently credited to the other.
    """
    port = _free_port()
    root = work / "parallel"
    root.mkdir()
    procs = {}
    marker_paths = {}
    for name in ("claude", "codex"):
        (root / name).mkdir()
        marker_paths[name] = root / name / f"{name.upper()}_E2E_MARKER"
    # Hold each response briefly, for the same reason the same-binary mode does:
    # the run's whole claim is that the two clients' upstream requests really
    # overlapped, and with an instant stub that is timing luck. Measured:
    # `--parallel --rounds 1` produced `0 ms of genuine overlap` and failed the
    # verdict on a completely healthy run, because at one round each client only
    # makes two requests and they missed each other. A hold makes the overlap a
    # property of the setup rather than of the scheduler.
    gw = _start_gateway(port, log_dir, want="touch E2E_MARKER", rounds=args.rounds,
                        want_dir=root, parallel=True,
                        extra_env={"E2E_HOLD_SECONDS": "0.4"},
                        two_calls=getattr(args, "two_calls", False))
    try:
        if not _wait_ready(f"http://127.0.0.1:{port}/health"):
            print("[e2e] parallel: GATEWAY NOT READY")
            return 1
        print(f"[e2e] parallel: one gateway on :{port}; starting both binaries…")
        for name in ("claude", "codex"):
            d = root / name
            if name == "claude":
                argv, env = _claude_argv(port, d, log_dir, marker_paths[name])
            else:
                argv, env = _codex_argv(port, d, log_dir, marker_paths[name])
            if not argv or not argv[0]:
                procs[name] = None
                continue
            procs[name] = subprocess.Popen(
                argv, cwd=str(d), env=env, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, stdin=subprocess.DEVNULL,
            )
        outs = {}
        for name, p in procs.items():
            if p is None:
                continue
            outs[name] = p.communicate(timeout=300)[0]
        bodies_all = _bodies(log_dir, port)
    finally:
        gw.terminate()
        try:
            gw.wait(timeout=10)
        except subprocess.TimeoutExpired:
            gw.kill()
        getattr(gw, "_e2e_log", None) and gw._e2e_log.close()  # type: ignore[attr-defined]

    overlap = _overlap_ms(bodies_all)
    for name in ("claude", "codex"):
        p = procs.get(name)
        if p is None:
            results.append({"client": name, "skipped": f"{name} not on PATH"})
            continue
        bodies = _bodies_for(bodies_all, name)
        res = _collect(name, p.returncode, bodies, root / name,
                       marker_paths[name], outs.get(name, ""))
        res["rounds_expected"] = args.rounds
        res["total_bodies"] = len(bodies_all)
        res["overlap_ms"] = overlap
        results.append(res)
        print(f"[e2e] parallel {name}: "
              f"{json.dumps({k: v for k, v in res.items() if k not in ('stdout', 'declared')}, ensure_ascii=False)}")
        if not res["marker"]:
            failures.append(f"{name}: never executed a tool (marker absent)")
        if len(res["markers"]) < args.rounds:
            # At one round the stub is never asked to write a per-round marker
            # (`_command_for` only rewrites the command when ROUNDS > 1), so this
            # check demanded a file the harness never creates: a healthy
            # `--parallel --rounds 1` printed `0/1 rounds really ran` on both
            # clients (measured, and reproduced on the pre-fix baseline). The
            # single-client mode already treats markers as n/a at one round;
            # this is the same rule, in the mode that lacked it.
            if args.rounds > 1:
                failures.append(
                    f"{name}: {len(res['markers'])}/{args.rounds} rounds really ran with the "
                    f"other agent sending requests at the same time"
                )
        if res["bodies"] < 2:
            failures.append(f"{name}: only {res['bodies']} of its own upstream requests")
        if not res["has_tools_section"]:
            failures.append(f"{name}: its upstream prompt carried no tool contract")
        if not res["round2_saw_result"]:
            failures.append(f"{name}: none of its requests carried a tool result")
        if getattr(args, "two_calls", False) and not (root / name / "SECOND_CALL").exists():
            failures.append(
                f"{name}: only the FIRST of two tool calls in one turn executed "
                "(SECOND_CALL marker absent)"
            )
        if args.rounds > 1 and not res.get("session_reused"):
            failures.append(
                f"{name}: never reused an upstream conversation across "
                f"{args.rounds} rounds while sharing the gateway "
                f"(upstream_convs={res.get('upstream_convs')})"
            )

    # A run where the two binaries never overlapped measures nothing about sharing,
    # and every other check in this function would still pass. Make that loud.
    if overlap <= 0:
        failures.append(
            "the two clients' upstream requests never overlapped in time — this run "
            "proves nothing about sharing one gateway (measurement: 0 ms of overlap)"
        )
    # Alternation is the stronger form of the same claim: a log that is
    # `CCCC…XXXX` overlapped for a while but was not interleaved per turn.
    if _alternation(bodies_all)[1] < 2:
        failures.append(
            "the two clients' requests barely interleaved — the shared-gateway path "
            f"was not exercised per turn (sequence: {_alternation(bodies_all)[0]})"
        )

    print("\n================ E2E VERDICT (parallel) ================")
    for r in results:
        if "skipped" in r:
            print(f"  {r['client']}: SKIPPED")
            continue
        ok = not _failures_for(failures, r["client"], [x["client"] for x in results])
        print(f"  {r['client']}: {'PASS' if ok else 'FAIL'} "
              f"(bodies={r['bodies']}, tool_markers={len(r['markers'])}/{r['rounds_expected']}, "
              f"tool={r['chosen'][0] if r['chosen'] else None})")
    seq, switches = _alternation(bodies_all)
    print(f"  concurrent window: {overlap} ms of genuine overlap "
          f"({len(bodies_all)} upstream bodies total)")
    print(f"  interleaving: {seq} ({switches} client switches)")
    for f in failures:
        print(f"  !! {f}")
    if not args.keep:
        shutil.rmtree(log_dir, ignore_errors=True)
        print("  (artifacts removed; re-run with --keep to inspect)")
    else:
        print(f"  artifacts kept at {log_dir}")
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--client", choices=("claude", "codex", "both"), default="both")
    ap.add_argument("--keep", action="store_true", help="keep artifacts for inspection")
    ap.add_argument("--two-calls", action="store_true",
                    help="emit TWO tool calls in one assistant turn (parallel tool use, "
                         "which real agents do constantly); both must execute")
    ap.add_argument("--rounds", type=int, default=1,
                    help="tool rounds to drive before the model answers (default 1, i.e. a "
                         "two-request loop). Raise it to exercise long agent sessions.")
    ap.add_argument("--parallel", action="store_true",
                    help="drive both clients CONCURRENTLY against one gateway — the real "
                         "developer setup (two agents, one machine). Exercises interleaved "
                         "requests, shared connection pool and shared concurrency budget.")
    ap.add_argument("--sessions", type=int, default=2,
                    help="with --parallel: how many concurrent sessions of ONE binary to "
                         "drive (they all hit the same upstream account). Use --client to "
                         "pick which binary; default 2.")
    args = ap.parse_args()

    log_dir = Path(tempfile.mkdtemp(prefix="unipivot_e2e_"))
    work = log_dir / "work"
    work.mkdir()
    print(f"[e2e] artifacts: {log_dir}")

    clients = ["claude", "codex"] if args.client == "both" else [args.client]
    results = []
    failures = []

    if args.parallel and args.client != "both":
        # N sessions of one binary — the "one account, several parallel sessions"
        # shape. Distinct from --parallel's two *different* vendors.
        return _run_same(log_dir, work, args, results, failures)

    if args.parallel:
        if len(clients) < 2:
            print("[e2e] --parallel needs both clients")
            return 2
        return _run_parallel(log_dir, work, args, results, failures)

    for name in clients:
        port = _free_port()
        # One working directory per client: the per-round markers of the two arms
        # would otherwise be indistinguishable, and "round 3 of claude" could be
        # satisfied by "round 3 of codex".
        client_work = work / name
        client_work.mkdir()
        marker = client_work / f"{name.upper()}_E2E_MARKER"
        proc = _start_gateway(port, log_dir, want=f"touch {marker}",
                              rounds=args.rounds, want_dir=client_work,
                              two_calls=getattr(args, "two_calls", False))
        try:
            if not _wait_ready(f"http://127.0.0.1:{port}/health"):
                failures.append(f"{name}: gateway never became ready on :{port}")
                print(f"[e2e] {name}: GATEWAY NOT READY")
                continue
            print(f"[e2e] {name}: gateway up on :{port}; driving the real binary…")
            res = (
                run_claude(port, client_work, log_dir, marker)
                if name == "claude"
                else run_codex(port, client_work, log_dir, marker)
            )
            res["rounds_expected"] = args.rounds
            # `_collect` already derived `markers` from the working directory; a
            # second derivation here was a second definition of the same fact and
            # printed `tool_markers=0/1` on a single-round run, where no per-round
            # marker is ever written.
            results.append(res)
            print(f"[e2e] {name}: {json.dumps({k: v for k, v in res.items() if k != 'stdout'}, ensure_ascii=False)}")
            if "skipped" in res:
                print(f"[e2e] {name}: SKIPPED ({res['skipped']})")
                continue
            if not res["marker"]:
                failures.append(
                    f"{name}: the client never really executed the tool "
                    f"(marker absent; chosen={res['chosen']}, declared={res['declared'][:6]})"
                )
            if res["bodies"] < 2:
                failures.append(
                    f"{name}: only {res['bodies']} upstream request(s) — a two-round "
                    f"loop needs at least 2; the tool result never came back"
                )
            if args.rounds > 1:
                want_markers = {f"ROUND_{i}" for i in range(1, args.rounds + 1)}
                missing = sorted(want_markers - set(res.get("markers") or []))
                if missing:
                    failures.append(
                        f"{name}: only {len(res.get('markers') or [])}/{args.rounds} tool "
                        f"rounds really executed (missing {missing}) — the loop stopped "
                        f"early; bodies={res['bodies']}"
                    )
            if not res["has_tools_section"]:
                failures.append(f"{name}: the upstream prompt carried no tool contract")
            if getattr(args, "two_calls", False) and not (client_work / "SECOND_CALL").exists():
                # Two calls in one assistant turn, and only the first executed.
                # Agents emit parallel calls constantly, so this is the "it does
                # half the work sometimes" report. Without this check the mode
                # above passed while proving nothing about the second call.
                failures.append(
                    f"{name}: only the FIRST of two tool calls in one turn executed "
                    "(SECOND_CALL marker absent)"
                )
            if not res["round2_saw_result"]:
                failures.append(f"{name}: no captured request carried a tool result")
            if args.rounds > 1 and not res["session_reused"]:
                failures.append(
                    f"{name}: never reused an upstream conversation across "
                    f"{args.rounds} rounds (upstream_convs="
                    f"{res['upstream_convs']}) — session identity is not reaching "
                    f"the gateway, so every turn rebuilds upstream context"
                )
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            getattr(proc, "_e2e_log", None) and proc._e2e_log.close()  # type: ignore[attr-defined]

    print("\n================ E2E VERDICT ================")
    for r in results:
        if "skipped" in r:
            print(f"  {r['client']}: SKIPPED")
        else:
            want_rounds = r.get("rounds_expected", 1)
            markers = len(r.get("markers") or [])
            # PASS is derived from the failure list, never recomputed beside it:
            # a verdict written from a hand-picked subset printed PASS on the
            # line above a fired failure (see `_failures_for`).
            ok = not _failures_for(failures, r["client"], [x["client"] for x in results])
            # At one round no per-round marker exists by construction, so printing
            # `tool_markers=0/1` reads as a failure on a run that is entirely green.
            markers_txt = f"{markers}/{want_rounds}" if want_rounds > 1 else "n/a"
            print(
                f"  {r['client']}: {'PASS' if ok else 'FAIL'} "
                f"(exit={r['exit']}, bodies={r['bodies']}, tool_ran={r['marker']}, "
                f"tool={r['chosen'][0] if r['chosen'] else None}, "
                f"tool_markers={markers_txt}, "
                f"conversation_reused={r.get('session_reused')})"
            )
    for f in failures:
        print(f"  !! {f}")
    if not args.keep:
        shutil.rmtree(log_dir, ignore_errors=True)
        print("  (artifacts removed; re-run with --keep to inspect)")
    else:
        print(f"  artifacts kept at {log_dir}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())