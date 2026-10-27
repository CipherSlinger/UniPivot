"""[E2E-VERDICT] The verdict line must state what the run established, not a
per-mode paraphrase of it.

The e2e runner has three verdict blocks (single client, two clients in parallel,
N sessions of one binary) and each built its own parenthesised field list. The
sequential block was taught, at some point, that at one round **no per-round
marker exists by construction** and printed `tool_markers=n/a`; the other two
kept printing `len(markers)/rounds_expected`. Measured on a green write-mode run:

    claude: PASS (bodies=2, tool_markers=0/1, tool=Write)
    codex:  PASS (bodies=2, tool_markers=0/1, tool=apply_patch)

against per-client data `"written": "e2e-ok", "content_ok": true`. Two separate
lies in one line: a failure-shaped number on a run with nothing wrong, and the
run's strongest evidence (the file's real content) missing entirely — the very
thing `--write` exists to measure. The parallel line also dropped `exit` and
`conversation_reused`, so two of the three modes hid the session-reuse proof
that section 66/67 added for exactly this purpose.

This is the repo's recurring "one fact, two definitions" class, living in the
harness that judges every other claim (CLAUDE.md sections 60/67/68). The fix is
one formatter, `_verdict_line`, used by all three blocks; the cases below pin the
three shapes it must take and the structural rule that no block may re-invent
it.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load_harness():
    """Import the runner as a module without running it (`__main__`-guarded)."""
    path = REPO / "scripts" / "e2e" / "run_agent_loop.py"
    spec = importlib.util.spec_from_file_location("_e2e_runner_for_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)          # type: ignore[union-attr]
    return mod


MOD = _load_harness()


def _line(r: dict, ok: bool = True) -> str:
    """The shared formatter, or a message that names the defect it replaces."""
    fmt = getattr(MOD, "_verdict_line", None)
    assert fmt is not None, (
        "e2e 运行器没有共享的结论行格式函数：三个结论块各自拼装字段，于是并行模式与"
        "N 会话模式打印了幻觉中的 `tool_markers=0/1`，并且漏掉了 `written=` / "
        "`content_ok=` / `exit=` / `conversation_reused=` —— 一个模式里修好过的规则，"
        "另外两个模式从来没学到"
    )
    return fmt(r, ok)


def _result(**kw) -> dict:
    """A per-client result as `_collect()` really returns it."""
    base = {
        "client": "claude",
        "exit": 0,
        "bodies": 2,
        "marker": True,
        "written": None,
        "content_ok": None,
        "markers": [],
        "declared": ["Write"],
        "chosen": ["Write", {"path": "e2e_marker.txt"}],
        "has_tools_section": True,
        "round2_saw_result": False,
        "upstream_convs": [None, "c1"],
        "session_reused": True,
        "rounds_expected": 1,
    }
    base.update(kw)
    return base


def test_write_mode_reports_the_content_not_a_phantom_marker(monkeypatch=None):
    """`--write`, one round: the file's real content is the evidence."""
    import os
    os.environ["E2E_WRITE"] = "1"
    try:
        line = _line(
            _result(written="e2e-ok", content_ok=True,
                    chosen=["Write", {"path": "e2e_marker.txt"}]),
        )
    finally:
        os.environ.pop("E2E_WRITE", None)

    assert "tool_markers=0/1" not in line, (
        f"写模式单轮运行仍然打印了幻觉中的轮次标记计数: {line!r}"
    )
    assert "written='e2e-ok'" in line and "content_ok=True" in line, (
        f"--write 模式最有力的证据（文件真实内容）没有出现在结论行里: {line!r}"
    )
    assert "exit=0" in line, f"结论行缺少 exit（三个模式里有两个曾经漏掉它）: {line!r}"
    print("test_write_mode_reports_the_content_not_a_phantom_marker: PASSED")


def test_single_round_marker_mode_reports_n_a():
    """No `--write`, one round: no per-round marker is ever written."""
    line = _line(_result())
    assert "tool_markers=n/a" in line, (
        f"单轮运行不存在逐轮标记，却打印了一个像失败一样的分数: {line!r}"
    )
    assert "0/1" not in line, line
    assert "written=" not in line, (
        f"非写模式不该报告文件内容（它根本没有被创建）: {line!r}"
    )
    print("test_single_round_marker_mode_reports_n_a: PASSED")


def test_multi_round_marker_mode_reports_the_real_count():
    """Multi-round runs must keep the real `m/n` — the fix must not erase it."""
    line = _line(
        _result(markers=["ROUND_1", "ROUND_2", "ROUND_3"], rounds_expected=3),
    )
    assert "tool_markers=3/3" in line, (
        f"多轮运行丢了真实的逐轮标记计数: {line!r}"
    )
    assert "conversation_reused=True" in line, (
        f"结论行缺少会话复用证据（§66/67 的核心断言）: {line!r}"
    )
    print("test_multi_round_marker_mode_reports_the_real_count: PASSED")


def test_sessions_mode_reports_the_shared_total_honestly():
    """N sessions of one binary: bodies are shared, so the line must say so."""
    r = _result(bodies=None, shared_bodies=8)
    r.pop("bodies")
    line = _line(r)
    assert "shared_bodies=8" in line, (
        f"N 会话模式的结论行把共享的上游请求数当成了单个会话的: {line!r}"
    )
    print("test_sessions_mode_reports_the_shared_total_honestly: PASSED")


def test_every_verdict_block_uses_the_one_definition():
    """Structural: three verdict blocks, one formatter, no hand-built copies.

    A behaviour test can only cover the modes I happened to construct; the defect
    was that a mode I did not look at kept its own copy of the field list. This
    asserts there is exactly one place that formats a verdict body, so the next
    mode has to go through it or turn the suite red.
    """
    src = (REPO / "scripts" / "e2e" / "run_agent_loop.py").read_text(encoding="utf-8")

    definitions = re.findall(r"^def _verdict_line\(", src, re.M)
    assert len(definitions) == 1, f"结论行格式函数有 {len(definitions)} 份定义，应当只有一份"

    calls = re.findall(r"_verdict_line\(", src)
    # 1 definition + 3 call sites (single, parallel, same-binary).
    assert len(calls) == 4, (
        f"只有 {len(calls) - 1} 个模式使用了共享的结论行格式；三个模式都应当使用它"
    )
    for marker in ("E2E VERDICT", "E2E VERDICT (parallel)", "E2E VERDICT (same binary, N sessions)"):
        assert marker in src, marker

    # No block may hand-format the fields the shared definition owns. Scoped to
    # the function body rather than a substring heuristic: the field names also
    # occur in `_collect`'s dict and in prose, and a check that flagged those
    # would be noise the next reader learns to ignore.
    lines = src.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("def _verdict_line("))
    end = next(
        (i for i in range(start + 1, len(lines))
         if lines[i] and not lines[i][0].isspace() and not lines[i].startswith(")")),
        len(lines),
    )
    outside = [
        (i + 1, ln) for i, ln in enumerate(lines)
        if ("tool_markers=" in ln or "content_ok=" in ln or "conversation_reused=" in ln)
        and not ln.lstrip().startswith("#")
        and not (start <= i < end)
    ]
    assert not outside, (
        "结论行字段仍然被某个模式手工拼装（这正是两份定义复发的方式）: "
        + repr(outside[:3])
    )
    print("test_every_verdict_block_uses_the_one_definition: PASSED")


def run_all():
    test_write_mode_reports_the_content_not_a_phantom_marker()
    test_single_round_marker_mode_reports_n_a()
    test_multi_round_marker_mode_reports_the_real_count()
    test_sessions_mode_reports_the_shared_total_honestly()
    test_every_verdict_block_uses_the_one_definition()
    print("\n✅ All e2e verdict tests passed! (5 cases)")


if __name__ == "__main__":
    sys.exit(run_all())
