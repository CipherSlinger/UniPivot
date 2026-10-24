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


def _start_gateway(port: int, log_dir: Path, want: str, rounds: int, want_dir: Path) -> subprocess.Popen:
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
        "PYTHONUNBUFFERED": "1",
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


def run_claude(port: int, work: Path, log_dir: Path, marker: Path) -> dict:
    """Drive `claude -p` and require a real tool execution plus a closed loop."""
    claude = shutil.which("claude")
    if not claude:
        return {"client": "claude", "skipped": "claude not on PATH"}

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
    proc = subprocess.run(
        [claude, "-p", f"Run `{want}` then tell me what it printed.",
         "--model", "claude-3-7-sonnet",
         "--allowedTools", "Bash"],
        cwd=str(work), env=env, capture_output=True, text=True, timeout=240,
        stdin=subprocess.DEVNULL,
    )
    (log_dir / "claude.out").write_text(proc.stdout + "\n---STDERR---\n" + proc.stderr)
    bodies = _bodies(log_dir, port)
    return {
        "client": "claude",
        "exit": proc.returncode,
        "bodies": len(bodies),
        "marker": _marker_seen(marker),
        "declared": (bodies[-1]["declared_tools"] if bodies else []),
        "chosen": (bodies[-1]["chosen_tool"] if bodies else None),
        "has_tools_section": bool(bodies and bodies[-1]["has_tools_section"]),
        "round2_saw_result": any(
            "Tool Result for" in (b.get("prompt_tail") or "") for b in bodies
        ),
        "stdout": proc.stdout[-2000:],
    }


def run_codex(port: int, work: Path, log_dir: Path, marker: Path) -> dict:
    """Drive `codex exec` and require a real tool execution plus a closed loop."""
    codex = shutil.which("codex")
    if not codex:
        return {"client": "codex", "skipped": "codex not on PATH"}

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
    proc = subprocess.run(
        [codex, "exec", "--skip-git-repo-check",
         "--dangerously-bypass-approvals-and-sandbox",
         "-m", "gpt-5.5",
         f"Run `{want}` then tell me what it printed."],
        cwd=str(work), env=env, capture_output=True, text=True, timeout=240,
        stdin=subprocess.DEVNULL,
    )
    (log_dir / "codex.out").write_text(proc.stdout + "\n---STDERR---\n" + proc.stderr)
    bodies = _bodies(log_dir, port)
    return {
        "client": "codex",
        "exit": proc.returncode,
        "bodies": len(bodies),
        "marker": _marker_seen(marker),
        "declared": (bodies[-1]["declared_tools"] if bodies else []),
        "chosen": (bodies[-1]["chosen_tool"] if bodies else None),
        "has_tools_section": bool(bodies and bodies[-1]["has_tools_section"]),
        "round2_saw_result": any(
            "Tool Result for" in (b.get("prompt_tail") or "") for b in bodies
        ),
        "stdout": proc.stdout[-2000:],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--client", choices=("claude", "codex", "both"), default="both")
    ap.add_argument("--keep", action="store_true", help="keep artifacts for inspection")
    ap.add_argument("--rounds", type=int, default=1,
                    help="tool rounds to drive before the model answers (default 1, i.e. a "
                         "two-request loop). Raise it to exercise long agent sessions.")
    args = ap.parse_args()

    log_dir = Path(tempfile.mkdtemp(prefix="unipivot_e2e_"))
    work = log_dir / "work"
    work.mkdir()
    print(f"[e2e] artifacts: {log_dir}")

    clients = ["claude", "codex"] if args.client == "both" else [args.client]
    results = []
    failures = []

    for name in clients:
        port = _free_port()
        # One working directory per client: the per-round markers of the two arms
        # would otherwise be indistinguishable, and "round 3 of claude" could be
        # satisfied by "round 3 of codex".
        client_work = work / name
        client_work.mkdir()
        marker = client_work / f"{name.upper()}_E2E_MARKER"
        proc = _start_gateway(port, log_dir, want=f"touch {marker}",
                              rounds=args.rounds, want_dir=client_work)
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
            res["markers"] = sorted(
                p.name for p in client_work.glob("ROUND_*")
            ) if args.rounds > 1 else []
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
            if not res["round2_saw_result"]:
                failures.append(f"{name}: no captured request carried a tool result")
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
            markers_ok = (
                len(r.get("markers") or []) >= r.get("rounds_expected", 1)
                if r.get("rounds_expected", 1) > 1 else True
            )
            ok = (r["marker"] and r["bodies"] >= 2 and r["has_tools_section"]
                  and r["round2_saw_result"] and markers_ok)
            print(
                f"  {r['client']}: {'PASS' if ok else 'FAIL'} "
                f"(exit={r['exit']}, bodies={r['bodies']}, tool_ran={r['marker']}, "
                f"tool={r['chosen'][0] if r['chosen'] else None}, "
                f"tool_markers={len(r.get('markers') or [])}/{r.get('rounds_expected', 1)})"
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