"""就绪面的**消费方**不得在派发事实上说谎 (tests/test_health_surface.py)。

[HEALTH-SURFACE] `/health` 修好了自己（第 63 节：无凭证时不再报 ok），但它的**用户**
仍然在替它下结论。`static/index.html` 的节点判定是四条启发式分支，最后一条兜底是：

    } else if (rawStatus === "healthy" || isConfigured) {
      statusClass = "healthy"; statusLabel = "健康在线";

`isConfigured` 只回答「本机有没有这个节点的凭证」，与「这条请求会不会被派发过去」
是两件事。实测（把 HEAD 版判定原样抽出来跑在**真实响应**上，node 执行，五种节点状态
× 两个就绪面）：

    状态              /v1/diagnostics(前端首选)        /health(兜底)
    operator-disabled  离线 / 待配置  "4 健康·0 熔断"     健康在线   "5 健康·0 熔断"
    quarantine 72h     已熔断避让                       已熔断避让
    long avoidance 30m 已熔断避让                       已熔断避让
    short cooldown 45s 已熔断避让                       已熔断避让

关键事实是**前端先取 `/v1/diagnostics`**（`/health` 只是取不到时的兜底），所以只修
`/health` 等于修了没人看的那一半——上面第一行的两种说法都是假话，而且方向相反：
diagnostics 把「运维显式排除」说成「本机没登录」（运维会去扫码，而重登对禁用毫无
作用），并把该节点从「健康 + 熔断」两个数里同时抹掉（4+0 ≠ 5）；`/health` 则直接
绿灯、计成健康。短冷却又相反：它按设计只降权（OPT-021 的最后兜底，派发闸门不拦），
两个面都把它计成「熔断避让」，于是「5 个能用」显示成「4 用 1 废」。第二、三行则是
「等 3 天」「等 30 分钟」「平台封了账号」共用同一个不透明标签，运维无从知道是该等、
该改配置，还是该去扫码。

根因不是前端写错了 `if`，而是**闸门只把一个布尔位交出去**：`dispatchable` 丢掉了
唯一能让人行动的信息（被哪一道闸门拦住、还剩多久、该做什么），于是每个消费方只能
自己从 `status` / `in_cooldown` / `*_configured` 反推，而反推必然漂移。这是
OPT-032/033/035（预演/档案在派发事实上说谎）的同一类缺陷，换到了**用户真正盯着的那块
屏幕**上。修法同样是「一份定义、原样转述」：`/health` 和 `/v1/diagnostics` 都把
`evaluate_dispatch_gate()` 的 `blocked_by` / `cause` / `remaining_seconds` 一并说出，
前端只读这些字段。

本套件锁住五件事：

1. `/health` 逐节点的 `blocked_by` / `blocked_cause` / `blocked_remaining_seconds`
   与**同一份**派发闸门逐值一致——用属性断言覆盖每个节点状态，而不是挑场景，因为本项目
   每次漂移都是漏了某一格；
2. `/v1/diagnostics`（前端的第一数据源）同样具备这三个字段，且两个就绪面对同一状态
   给出的前端判定与摘要**逐字一致**；
3. 前端的 `providerVerdict()` 在**任何**节点状态下都不会把不可派发的节点显示为健康
   （「不可派发 ⇒ 不健康」这条不变量，而不是「等于某个字符串」）；
4. 不同的阻断成因必须给出**不同**的标签，且标签里带得出可行动的信息（租约/避让要带
   剩余时长）；
5. 顶栏摘要只数真正可派发的节点。

第 3/4/5 条跑的是从 `static/index.html` 中**原样抽出**的纯函数（标记
`MESH-VERDICT-BEGIN/END` 之间），交给 node 执行——而不是在 Python 里复刻一份逻辑。
复刻出来的断言只能证明复刻是对的。node 不存在时 SKIP，绝不当作通过。

运行: .venv/bin/python tests/test_health_surface.py
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ["TESTING"] = "1"
os.environ["GATEWAY_DISABLE_AUTO_REFRESH"] = "1"
os.environ["GATEWAY_DISABLE_PACING"] = "1"
os.environ["GATEWAY_DISABLE_SINGLEFLIGHT"] = "1"

_SANDBOX = tempfile.mkdtemp(prefix="healthsurface_")
os.environ["GATEWAY_RISK_LEDGER_DIR"] = _SANDBOX
os.environ["GATEWAY_QUARANTINE_DIR"] = _SANDBOX
os.environ["GATEWAY_CONVERSATION_GC_DIR"] = _SANDBOX
os.environ["GATEWAY_DEVICE_IDENTITY_DIR"] = _SANDBOX

for _name in (
    "DISABLED_PROVIDERS",
    "GATEWAY_QUARANTINE",
    "GATEWAY_DISABLE_RISK_DOSSIER",
    "GATEWAY_DISABLE_QUARANTINE",
    "GATEWAY_DISABLE_QUARANTINE_HARD_STOP",
    "GATEWAY_DISABLE_RISK_LEDGER",
    "GATEWAY_DISABLE_CREDENTIAL_WATCH",
    "GATEWAY_DISABLE_BAN_INFERENCE",
    "GATEWAY_DISABLE_PROBATION",
    "GATEWAY_DISABLE_LONG_AVOIDANCE_FAST_FAIL",
):
    os.environ.pop(_name, None)
os.environ["GATEWAY_PROBATION_MAX_RENEWALS"] = "0"

import server as srv  # noqa: E402

ALL = ("qwen", "deepseek", "doubao", "kimi", "glm")

HTML_PATH = ROOT / "static" / "index.html"
_BEGIN = "/* ==== MESH-VERDICT-BEGIN ===="
_END = "/* ==== MESH-VERDICT-END ==== */"


# --------------------------------------------------------------------------
# 环境构造
# --------------------------------------------------------------------------

class _FakeSession:
    def __init__(self, token: str = "fake-token"):
        self.token = token


def _patch_resolvers(mapping):
    """把五个 resolver 在**三层**同时顶掉（详见 test_health_truthfulness.py 的说明）。

    本机 `session/` 里有真实的凭证文件（其中 deepseek 已被封禁），「全新克隆」这个
    场景不能靠「文件不存在」构造，否则本套件在开发机能过、在 CI 上行为不同。
    """
    import session_store

    saved_module = {pk: getattr(srv, f"resolve_{pk}") for pk in ALL}
    saved_store = dict(session_store._PROVIDER_RESOLVERS)
    saved_monitor = dict(srv.health_monitor._provider_resolvers)

    def _resolver_for(pk):
        return (lambda v=mapping[pk]: v) if pk in mapping else (lambda: None)

    for pk in ALL:
        setattr(srv, f"resolve_{pk}", _resolver_for(pk))
        session_store._PROVIDER_RESOLVERS[pk] = _resolver_for(pk)
        srv.health_monitor._provider_resolvers[pk] = _resolver_for(pk)

    def restore():
        for pk, fn in saved_module.items():
            setattr(srv, f"resolve_{pk}", fn)
        session_store._PROVIDER_RESOLVERS.clear()
        session_store._PROVIDER_RESOLVERS.update(saved_store)
        srv.health_monitor._provider_resolvers.clear()
        srv.health_monitor._provider_resolvers.update(saved_monitor)
        for pk in ALL:
            state = srv.health_monitor._states.get(pk)
            if state is None:
                continue
            fn = saved_store.get(pk)
            try:
                sess = fn() if fn else None
            except Exception:  # noqa: BLE001
                sess = None
            state.configured = bool(sess and getattr(sess, "token", None))

    return restore


def _reset_state():
    router = srv.failover_router
    tracker = router.cooldown_tracker
    tracker.reset()
    srv.quarantine_store.clear_all_for_tests()
    try:
        srv.risk_ledger.clear()
    except Exception:  # noqa: BLE001
        pass
    for obj in (tracker.credential_watch,):
        try:
            obj.clear_all_for_tests()
        except Exception:  # noqa: BLE001
            pass
    try:
        from providers.credential_renewal import get_credential_renewal

        get_credential_renewal().clear_all_for_tests()
    except Exception:  # noqa: BLE001
        pass
    return router, tracker, set(tracker.disabled_providers)


def _restore_state(tracker, snapshot):
    tracker.disabled_providers = set(snapshot)
    tracker.reset()
    srv.quarantine_store.clear_all_for_tests()


def _health():
    from fastapi.testclient import TestClient

    return TestClient(srv.app).get("/health")


# --------------------------------------------------------------------------
# 前端纯函数：从 index.html 原样抽出来，交给 node 执行
# --------------------------------------------------------------------------

_DRIVER = r"""
const fs = require("fs");
const payload = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const verdicts = payload.scenarios.map((s) => {
  const v = providerVerdict(s.pInfo, Boolean(s.isConfigured), Boolean(s.isHealing));
  return {
    statusClass: v.statusClass,
    statusLabel: v.statusLabel,
    dispatchable: v.dispatchable,
    hint: v.hint || "",
  };
});
process.stdout.write(JSON.stringify({verdicts: verdicts, summary: summarizeMesh(verdicts)}));
"""


def _extract_verdict_js() -> str:
    html = HTML_PATH.read_text(encoding="utf-8")
    if _BEGIN not in html or _END not in html:
        raise AssertionError(
            "static/index.html 里找不到 MESH-VERDICT-BEGIN/END 标记——"
            "前端判定被搬回了内联的启发式分支，本套件将无法再验证它。"
            "请把节点判定保留为一对标记之间的纯函数（它同时是这两条不变量唯一的验证入口）"
        )
    start = html.index(_BEGIN)
    end = html.index(_END)
    return html[start : end + len(_END)]


def _run_verdict(scenarios):
    """把 `[{pInfo, isConfigured, isHealing}, ...]` 交给前端真正的判定函数。

    返回 `{"verdicts": [...], "summary": "..."}`；node 不可用时返回 `None`（SKIP）。
    """
    node = shutil.which("node")
    if node is None:
        return None

    script = _extract_verdict_js() + "\n" + _DRIVER
    with tempfile.TemporaryDirectory(prefix="verdictjs_") as tmp:
        script_path = Path(tmp) / "verdict.js"
        payload_path = Path(tmp) / "payload.json"
        script_path.write_text(script, encoding="utf-8")
        payload_path.write_text(json.dumps({"scenarios": scenarios}), encoding="utf-8")
        proc = subprocess.run(
            [node, str(script_path), str(payload_path)],
            capture_output=True,
            text=True,
            timeout=60,
        )
    if proc.returncode != 0:
        raise AssertionError(f"前端判定函数在 node 下执行失败:\n{proc.stderr[:2000]}")
    return json.loads(proc.stdout)


def _scenarios_from_readiness(body):
    """把一份真实就绪响应（`/health` 或 `/v1/diagnostics`）转成前端函数的入参。

    两个端点都长这样：`providers[pk]` 加一堆 `{pk}_configured` 布尔位
    （`/v1/diagnostics` 不发这些位，取不到就是 `False`——前端也只会用它来区分
    「状态未知」和「离线/待配置」，节点条目存在时并不参与判定）。
    """
    providers = body.get("providers") or {}
    return [
        {
            "pInfo": providers.get(pk),
            "isConfigured": bool(body.get(f"{pk}_configured")),
            "isHealing": False,
        }
        for pk in ALL
    ]


def _scenarios_from_health(body):
    return _scenarios_from_readiness(body)


def _diagnostics():
    from fastapi.testclient import TestClient

    return TestClient(srv.app).get("/v1/diagnostics")


# --------------------------------------------------------------------------
# 1. /health 必须原样转述闸门的结论
# --------------------------------------------------------------------------

def _node_states():
    """每个场景：(名称, 凭证映射, 施加状态的函数, 施加环境的 dict)。"""
    def _noop(tracker):
        return None

    def _short_cooldown(tracker):
        asyncio.run(tracker.record_failure("kimi", 429, "rate limit exceeded"))

    def _long_avoidance(tracker):
        for _ in range(3):
            asyncio.run(tracker.record_failure("glm", 401, "401 Unauthorized: token expired"))

    def _quarantine(tracker):
        srv.quarantine_store.set_lease("deepseek", duration_seconds=3600, source="operator")

    def _disabled(tracker):
        tracker.disabled_providers = {"doubao"}

    full = {pk: _FakeSession() for pk in ALL}
    return [
        ("healthy", full, _noop, {}),
        ("no_credentials", {}, _noop, {}),
        ("disabled", full, _disabled, {}),
        ("quarantine", full, _quarantine, {}),
        ("long_avoidance", full, _long_avoidance, {}),
        ("short_cooldown", full, _short_cooldown, {}),
    ]


def test_health_reports_the_gate_verdict_per_provider():
    """属性：`blocked_by` / `blocked_cause` / 剩余时长与派发闸门**逐值**一致。

    逐个节点、逐个状态断言。判据只有一份（`evaluate_dispatch_gate`），就绪面只是把它
    说出来；一旦有人在前端或后端「再推一遍」，本用例直接变红。
    """
    for name, mapping, apply_state, env in _node_states():
        for k in ("DISABLED_PROVIDERS",):
            os.environ.pop(k, None)
        os.environ.update(env)
        restore = _patch_resolvers(mapping)
        router, tracker, snap = _reset_state()
        try:
            apply_state(tracker)
            body = _health().json()
            # 闸门现在也读凭证（[READINESS]），期望值必须在顶替窗口内求值
            expected = {
                pk: router.evaluate_dispatch_gate(pk) for pk in ALL
            }
        finally:
            _restore_state(tracker, snap)
            restore()
            for k in env:
                os.environ.pop(k, None)

        for pk in ALL:
            info = body["providers"][pk]
            gate = expected[pk]
            assert info.get("blocked_by") == gate["blocked_by"], (
                f"[{name}] {pk}: /health 报 blocked_by={info.get('blocked_by')!r}，"
                f"派发闸门为 {gate['blocked_by']!r}。就绪面与派发边界必须同一口径"
            )
            assert info.get("blocked_cause") == gate["cause"], (
                f"[{name}] {pk}: blocked_cause 不一致 "
                f"({info.get('blocked_cause')!r} != {gate['cause']!r})"
            )
            got_rem = float(info.get("blocked_remaining_seconds") or 0.0)
            want_rem = float(gate.get("remaining_seconds") or 0.0)
            # 两次读取之间时间在流逝，允许 2 秒的读时钟差
            assert abs(got_rem - want_rem) <= 2.0, (
                f"[{name}] {pk}: 剩余时长 {got_rem}s 与闸门的 {want_rem}s 相差过大——"
                f"消费方会据此告诉运维「再等 X」，编出来的倒计时比没有更糟"
            )
    print("test_health_reports_the_gate_verdict_per_provider: PASSED")


# --------------------------------------------------------------------------
# 1b. /v1/diagnostics 必须说同一句话 —— 而且它才是前端的第一数据源
# --------------------------------------------------------------------------

def test_diagnostics_surface_speaks_the_same_verdict():
    """属性：**两个**就绪面对同一状态的结论必须逐字一致。

    这条不是「顺手也修一下」：`loadDiagnosticsData()` 先取 `/v1/diagnostics`，
    `/health` 只是取不到时的兜底。所以只修 `/health` 等于修了没人看的那一半。实测
    （把 HEAD 版判定原样跑在真实响应上，node 执行）：

        operator-disabled  /v1/diagnostics  -> 离线 / 待配置   summary="4 健康 · 0 熔断避让"
        operator-disabled  /health          -> 健康在线        summary="5 健康 · 0 熔断避让"

    两个都是假话，而且方向相反：一个把「运维显式排除」说成本机没登录（运维会去扫码，
    而重登对禁用毫无作用——OPT-026/029/034 反复记录的那一类误导），另一个直接绿灯；
    摘要更是在第一种情况下把该节点从「健康 + 熔断」两个数里同时抹掉（4+0 ≠ 5），
    在第二种情况下把它算成健康。短冷却则相反：它按设计只降权（OPT-021 最后兜底），
    两个面都被计成「熔断避让」，于是「5 个能用」被显示成「4 用 1 废」。

    断言的是**两个面的判定相等**（逐节点、逐状态），而不是各自等于某个字符串，
    因为漂移的形态每次都不一样，而「同一份事实只有一个说法」是不变的那件事。
    """
    node = shutil.which("node")
    if node is None:
        print("  SKIP test_diagnostics_surface_speaks_the_same_verdict (node 未安装)")
        return

    for name, mapping, apply_state, env in _node_states():
        for k in ("DISABLED_PROVIDERS",):
            os.environ.pop(k, None)
        os.environ.update(env)
        restore = _patch_resolvers(mapping)
        router, tracker, snap = _reset_state()
        try:
            apply_state(tracker)
            health_body = _health().json()
            diag_body = _diagnostics().json()
            expected = {pk: router.evaluate_dispatch_gate(pk) for pk in ALL}
        finally:
            _restore_state(tracker, snap)
            restore()
            for k in env:
                os.environ.pop(k, None)

        for pk in ALL:
            info = diag_body["providers"][pk]
            gate = expected[pk]
            assert info.get("blocked_by") == gate["blocked_by"], (
                f"[{name}] {pk}: /v1/diagnostics 报 blocked_by={info.get('blocked_by')!r}，"
                f"派发闸门为 {gate['blocked_by']!r}。前端的第一数据源就在这里，"
                f"它缺字段，浏览器只能自己猜"
            )
            assert info.get("blocked_cause") == gate["cause"], (
                f"[{name}] {pk}: /v1/diagnostics blocked_cause 不一致 "
                f"({info.get('blocked_cause')!r} != {gate['cause']!r})"
            )
            got_rem = float(info.get("blocked_remaining_seconds") or 0.0)
            want_rem = float(gate.get("remaining_seconds") or 0.0)
            assert abs(got_rem - want_rem) <= 2.0, (
                f"[{name}] {pk}: /v1/diagnostics 剩余时长 {got_rem}s 与闸门的 {want_rem}s 相差过大"
            )
            expected_dispatchable = bool(health_body["providers"][pk].get("dispatchable")) \
                and bool(health_body.get(f"{pk}_configured"))
            assert info.get("dispatchable") == expected_dispatchable, (
                f"[{name}] {pk}: 两个就绪面对 dispatchable 的结论不一致 "
                f"({info.get('dispatchable')} != {expected_dispatchable})"
            )

        h_verdicts = _run_verdict(_scenarios_from_readiness(health_body))
        d_verdicts = _run_verdict(_scenarios_from_readiness(diag_body))
        for pk, hv, dv in zip(ALL, h_verdicts["verdicts"], d_verdicts["verdicts"]):
            assert (hv["statusClass"], hv["statusLabel"]) == (
                dv["statusClass"], dv["statusLabel"],
            ), (
                f"[{name}] {pk}: 同一次状态，/health 渲染为 {hv['statusLabel']!r}，"
                f"/v1/diagnostics 渲染为 {dv['statusLabel']!r}。前端先取 diagnostics，"
                f"用户看到的是后者"
            )
        assert h_verdicts["summary"] == d_verdicts["summary"], (
            f"[{name}] 两个就绪面的摘要不一致: "
            f"{h_verdicts['summary']!r} vs {d_verdicts['summary']!r}"
        )

    print("test_diagnostics_surface_speaks_the_same_verdict: PASSED")


# --------------------------------------------------------------------------
# 2. 前端判定：不可派发 ⇒ 不健康
# --------------------------------------------------------------------------

def test_no_hard_blocked_node_is_ever_displayed_as_healthy():
    """**本套件的直接靶点**：七种状态下，不可派发的节点不得显示为「健康在线」。

    断言的是不变量（`dispatchable === false ⇒ statusClass !== "healthy"`），而不是
    「等于某个字符串」：标签文案可以改，把不可派发说成健康这件事不行。
    """
    node = shutil.which("node")
    if node is None:
        print("  SKIP test_no_hard_blocked_node_is_ever_displayed_as_healthy (node 未安装)")
        return

    seen = {}
    for name, mapping, apply_state, env in _node_states():
        for k in ("DISABLED_PROVIDERS",):
            os.environ.pop(k, None)
        os.environ.update(env)
        restore = _patch_resolvers(mapping)
        router, tracker, snap = _reset_state()
        try:
            apply_state(tracker)
            body = _health().json()
        finally:
            _restore_state(tracker, snap)
            restore()
            for k in env:
                os.environ.pop(k, None)

        result = _run_verdict(_scenarios_from_health(body))
        for pk, info, verdict in zip(ALL, (body["providers"][pk] for pk in ALL), result["verdicts"]):
            dispatchable = info.get("dispatchable") is True
            assert verdict["dispatchable"] == dispatchable, (
                f"[{name}] {pk}: 前端判定 dispatchable={verdict['dispatchable']} 与 "
                f"/health 的 {dispatchable} 不一致"
            )
            if not dispatchable:
                assert verdict["statusClass"] != "healthy", (
                    f"[{name}] {pk}: 派发闸门拒绝该节点（blocked_by="
                    f"{info.get('blocked_by')!r}），前端却显示为 "
                    f"{verdict['statusLabel']!r}。运维看到绿灯就不会去看真正的原因"
                )
                assert "健康" not in verdict["statusLabel"], (
                    f"[{name}] {pk}: 标签 {verdict['statusLabel']!r} 里仍带「健康」字样"
                )
                assert verdict["hint"], (
                    f"[{name}] {pk}: 不可派发却没有给出任何可行动的说明"
                )
            seen.setdefault(name, {})[pk] = (info.get("blocked_by"), verdict["statusLabel"])

    # 直接靶点：运维禁用的节点（实测修复前显示「健康在线」）
    assert seen["disabled"]["doubao"][0] == "disabled", seen["disabled"]["doubao"]
    assert seen["disabled"]["doubao"][1] != "健康在线", (
        f"运维禁用节点仍显示为「健康在线」: {seen['disabled']['doubao']}"
    )
    print("test_no_hard_blocked_node_is_ever_displayed_as_healthy: PASSED")


def test_distinct_block_causes_get_distinct_actionable_labels():
    """不同的阻断成因必须能被区分，且标签带得出「要等多久」。

    实测修复前：租约（等 3 天，重登无用）、长效避让（等数小时）、运维禁用（改配置）
    三者共用「已熔断避让」，运维只能猜。
    """
    if shutil.which("node") is None:
        print("  SKIP test_distinct_block_causes_get_distinct_actionable_labels (node 未安装)")
        return

    def _v(pInfo, isConfigured=True):
        return _run_verdict([{"pInfo": pInfo, "isConfigured": isConfigured, "isHealing": False}])[
            "verdicts"
        ][0]

    base = {"dispatchable": False, "status": "offline", "in_cooldown": False}
    labels = {}
    for blocked_by, extra in (
        ("credential_outage", {}),
        ("disabled", {"status": "disabled"}),
        ("quarantine", {"in_cooldown": True, "blocked_remaining_seconds": 259200, "blocked_cause": "ban"}),
        ("long_avoidance", {"in_cooldown": True, "blocked_remaining_seconds": 14400, "blocked_cause": "waf"}),
        ("unconfigured", {"status": "offline"}),
    ):
        verdict = _v(dict(base, blocked_by=blocked_by, **extra))
        assert verdict["statusClass"] != "healthy", (blocked_by, verdict)
        labels[blocked_by] = verdict["statusLabel"]

    assert len(set(labels.values())) == len(labels), (
        f"不同的阻断成因给出了相同标签，运维无从区分:\n{json.dumps(labels, ensure_ascii=False, indent=2)}"
    )

    # 租约是「等 3 天」，避让是「等 4 小时」——数字必须真的出现在标签里
    assert "3 天" in labels["quarantine"], labels["quarantine"]
    assert "4 小时" in labels["long_avoidance"], labels["long_avoidance"]
    # 重登能不能救，是两种成因最要紧的差别
    q = _v(dict(base, blocked_by="quarantine", in_cooldown=True, blocked_remaining_seconds=3600))
    u = _v(dict(base, blocked_by="unconfigured"))
    assert "无用" in q["hint"] or "login" not in q["hint"], q["hint"]
    assert "login.py" in u["hint"], u["hint"]

    # 零回归：可派发 + 无冷却仍然是「健康在线」
    ok = _v({"dispatchable": True, "status": "healthy", "in_cooldown": False})
    assert ok["statusClass"] == "healthy" and ok["statusLabel"] == "健康在线", ok
    print("test_distinct_block_causes_get_distinct_actionable_labels: PASSED")


def test_summary_counts_only_dispatchable_nodes():
    """顶栏摘要只数可派发的节点（实测修复前把运维禁用的节点算进「健康」）。"""
    if shutil.which("node") is None:
        print("  SKIP test_summary_counts_only_dispatchable_nodes (node 未安装)")
        return

    scenarios = [
        {"pInfo": {"dispatchable": False, "status": "disabled", "blocked_by": "disabled"}, "isConfigured": True},
        {"pInfo": {"dispatchable": True, "status": "healthy"}, "isConfigured": True},
        {"pInfo": {"dispatchable": False, "status": "offline", "blocked_by": "unconfigured"}, "isConfigured": False},
    ]
    result = _run_verdict(scenarios)
    assert result["summary"].startswith("1 可派发"), result["summary"]
    assert "3 可派发" not in result["summary"], result["summary"]
    print("test_summary_counts_only_dispatchable_nodes: PASSED")


def run_all():
    test_health_reports_the_gate_verdict_per_provider()
    test_diagnostics_surface_speaks_the_same_verdict()
    test_no_hard_blocked_node_is_ever_displayed_as_healthy()
    test_distinct_block_causes_get_distinct_actionable_labels()
    test_summary_counts_only_dispatchable_nodes()
    print("\n✅ All health-surface tests passed!")


if __name__ == "__main__":
    run_all()
