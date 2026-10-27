"""就绪面不得报喜：`/health` 说 ok 时，这个部署必须真的能服务 (tests/test_health_truthfulness.py)。

[HEALTH-TRUTH] 每一个新用户看到的第一个界面就是它。`git clone` 之后 `session/` 是空的，
网关照常启动，`GET /health` 返回：

    HTTP 200   {"status": "ok",
                "qwen_configured": false, "deepseek_configured": false,
                "doubao_configured": false, "kimi_configured": false, "glm_configured": false,
                "providers": {每个节点: {"status": "offline"}}}

五个节点全部离线、无一持有凭证，而顶层状态说健康。这不是「少了一个字段」：`/health`
是运维、监控探针和（实测的）agent 接入脚本判断「网关好了没有」的唯一入口，它报 ok
就等于告诉所有人「可以开始发请求了」——随后每个请求都以 401 结束，而运维手上没有
任何一条信息指向真正的原因（缺凭证），因为整份响应里 `status` 是 ok。

同一个缺陷类型本项目记录过多次（OPT-032/033/035：预演说可派发、档案说可派发，而
派发边界拒绝）。`/health` 是**第三个**这样说话的地方，且是唯一一个对外承诺
「现在可以用了」的地方。

本套件锁住四件事：
1. 全环无凭证时，顶层状态不得是 `ok`，且必须给出**成因**而不只是一个降级字样；
2. 有节点可用时仍然报 `ok`（不能修成永远降级——那会让探针彻底失效）；
3. `providers[*].dispatchable` 不得在无凭证的节点上为真（无凭证的节点永远不会被成功派发）；
4. `dispatchable` 与派发闸门**同一口径**（`configured and evaluate_dispatch_gate().dispatchable`），
   用属性断言覆盖每个节点状态，而不是逐个场景列举——本项目的漂移缺陷每次都是漏了某一格。

运行: .venv/bin/python tests/test_health_truthfulness.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["TESTING"] = "1"
os.environ["GATEWAY_DISABLE_AUTO_REFRESH"] = "1"

_SANDBOX = tempfile.mkdtemp(prefix="healthtruth_")
os.environ["GATEWAY_RISK_LEDGER_DIR"] = _SANDBOX
os.environ["GATEWAY_QUARANTINE_DIR"] = _SANDBOX
os.environ["GATEWAY_CONVERSATION_GC_DIR"] = _SANDBOX
os.environ["GATEWAY_DEVICE_IDENTITY_DIR"] = _SANDBOX

PROVIDERS = ("qwen", "deepseek", "doubao", "kimi", "glm")


class _FakeSession:
    """只带 token 的最小 Session：本套件只关心「有没有凭证」这一位。"""

    def __init__(self, token: str = "fake-token"):
        self.token = token


def _patch_resolvers(mapping):
    """把五个 resolver 在**每一层**顶掉，返回还原函数。

    真实机器上 `session/*.json` 是存在的（本机就有一个被封禁的 DeepSeek），所以
    「全新克隆」不能靠「文件不存在」来构造，必须显式顶替——否则这个套件在本机能过、
    在 CI 上行为不同，正是本项目反复记录的「开发机的状态决定测试结果」。

    [READINESS] 必须顶掉**三层**，不是一层。`session_store._PROVIDER_RESOLVERS`
    在 import 时就把五个 resolver 函数抓进了字典，`health_monitor.register_provider()`
    同样抓了一份；只有 `server.py` 里的模块级 `resolve_*` 名字是运行时可替换的。
    早先这个 helper 只顶替 `srv.resolve_*`，而 `_configured()` 现在委托给
    `session_store.is_configured()`——也就是读**第二层**——于是 helper 会静默失效：
    套件仍然「通过」，但它测的是本机磁盘，不是它声称的场景。三层一起顶，
    这个 helper 才对得起它自己的名字。
    """
    import server as srv
    import session_store

    saved_module = {pk: getattr(srv, f"resolve_{pk}") for pk in PROVIDERS}
    saved_store = dict(session_store._PROVIDER_RESOLVERS)
    saved_monitor = dict(srv.health_monitor._provider_resolvers)

    def _resolver_for(pk):
        return (lambda v=mapping[pk]: v) if pk in mapping else (lambda: None)

    for pk in PROVIDERS:
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
        # 探针缓存里的 configured 位是上一次探针的结论，顶替后必须重算
        for pk in PROVIDERS:
            state = srv.health_monitor._states.get(pk)
            if state is None:
                continue
            fn = saved_store.get(pk)
            try:
                sess = fn() if fn else None
            except Exception:  # noqa: BLE001 - 还原不改变被测行为
                sess = None
            state.configured = bool(sess and getattr(sess, "token", None))

    return restore


def _fresh_tracker():
    import server as srv

    tracker = srv.failover_router.cooldown_tracker
    for d in (tracker._cooldowns, tracker._failure_counts, tracker._cooldown_severity, tracker._episode_ends):
        d.clear()
    return tracker


def _health():
    from fastapi.testclient import TestClient
    import server as srv

    return TestClient(srv.app).get("/health")


def test_health_does_not_report_ok_with_no_credentials():
    """全新克隆：五个节点都没有凭证，顶层状态必须是降级且说明原因。"""
    _fresh_tracker()
    restore = _patch_resolvers({})
    try:
        resp = _health()
    finally:
        restore()

    body = resp.json()
    configured = {pk: body.get(f"{pk}_configured") for pk in PROVIDERS}
    assert not any(configured.values()), f"前置条件不成立：测试进程里仍有凭证 {configured}"

    assert body.get("status") != "ok", (
        f"五个节点全部无凭证，/health 仍报 status={body.get('status')!r}。"
        f"这是监控与 agent 接入脚本判断「网关好了没有」的唯一入口，"
        f"报 ok 等于告诉所有人可以开始发请求——随后每个请求都以 401 结束，"
        f"而响应里没有任何一处指向真正的原因（缺凭证）"
    )
    assert body.get("status") == "degraded", (
        f"期望 status='degraded'，实际 {body.get('status')!r}"
    )
    reason = body.get("status_reason") or ""
    assert reason.strip(), (
        "降级状态没有给出成因。只说「degraded」而不说「为什么」/「该做什么」时，"
        "运维仍然不知道要执行 login.py"
    )


def test_health_reports_ok_when_a_provider_can_serve():
    """反面：有可用节点时必须仍然报 ok，否则探针失去意义。"""
    _fresh_tracker()
    restore = _patch_resolvers({"qwen": _FakeSession()})
    try:
        body = _health().json()
    finally:
        restore()

    assert body.get("status") == "ok", (
        f"qwen 持有凭证且无避让，/health 却报 {body.get('status')!r}——"
        f"修成永远降级会让所有监控探针失效"
    )


def test_no_node_is_reported_dispatchable_without_a_credential():
    """属性：`dispatchable` 为真的节点必须真的有凭证。

    无凭证的节点在派发循环里**必然**以本地 `ProviderError` 结束（一个上游请求都没有），
    把它报成可派发，就是 OPT-032/033/035 那个「预演/档案说可以，真实派发却拒绝」的
    同一类缺陷，换到了就绪面上。
    """
    _fresh_tracker()
    restore = _patch_resolvers({"glm": _FakeSession()})
    try:
        body = _health().json()
    finally:
        restore()

    offenders = [
        pk for pk in PROVIDERS
        if body["providers"].get(pk, {}).get("dispatchable") and not body.get(f"{pk}_configured")
    ]
    assert not offenders, (
        f"/health 把没有凭证的节点报成可派发: {offenders}。"
        f"这些节点永远不会被成功派发，运维读到「可派发」只会去查别的地方"
    )


def test_health_dispatchable_agrees_with_the_single_gate():
    """属性：`dispatchable == configured and evaluate_dispatch_gate().dispatchable`。

    逐个节点、逐个状态地断言，而不是挑几个场景：本项目的漂移缺陷每次都是漏了某一格
    （OPT-031 漏了长效避让、OPT-035 漏了运维禁用），一个属性断言让**下一格**漏掉时
    直接变红，而不必等运维来量。
    """
    import server as srv

    scenarios = {
        "healthy": {},
        "no_credentials": {},
        "disabled": {"DISABLED_PROVIDERS": "kimi"},
    }
    for name, env in scenarios.items():
        _fresh_tracker()
        for k in ("DISABLED_PROVIDERS",):
            os.environ.pop(k, None)
        os.environ.update(env)
        mapping = {} if name in ("no_credentials", "disabled") else {pk: _FakeSession() for pk in PROVIDERS}
        restore = _patch_resolvers(mapping)
        try:
            body = _health().json()
            # 闸门现在读凭证（[READINESS]），所以期望值必须在**顶替窗口内**求值：
            # 还原之后再问闸门，问的是开发机上的真实磁盘，而不是本场景。
            expected = {
                pk: bool(body.get(f"{pk}_configured"))
                and bool(srv.failover_router.evaluate_dispatch_gate(pk)["dispatchable"])
                for pk in PROVIDERS
            }
        finally:
            restore()
            for k in env:
                os.environ.pop(k, None)

        for pk in PROVIDERS:
            configured = bool(body.get(f"{pk}_configured"))
            expect = expected[pk]
            got = body["providers"][pk].get("dispatchable")
            assert got == expect, (
                f"[{name}] {pk}: /health 报 dispatchable={got!r}，"
                f"按唯一派发闸门应为 {expect}（configured={configured}）。"
                f"就绪面与派发边界必须同一口径"
            )


def run_all():
    test_health_does_not_report_ok_with_no_credentials()
    test_health_reports_ok_when_a_provider_can_serve()
    test_no_node_is_reported_dispatchable_without_a_credential()
    test_health_dispatchable_agrees_with_the_single_gate()
    print("\n✅ All health-truthfulness tests passed!")


if __name__ == "__main__":
    run_all()