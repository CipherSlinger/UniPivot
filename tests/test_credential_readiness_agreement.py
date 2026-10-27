"""就绪面、预演面、档案面与真实派发必须说的是同一件事 (tests/test_credential_readiness_agreement.py)。

[READINESS] 本机 `session/` 里只有一个 `.gitkeep`（这就是 `git clone` 之后的状态）。网关
照常启动，五个节点一个凭证都没有。修复前实测四个面与真实派发：

    /health            -> degraded, dispatchable_providers=[]          ← 已正确 [HEALTH-TRUTH]
    /v1/diagnostics    -> status: ok，且节点上根本没有 dispatchable 字段
    dispatch-audit     -> predicted_primary_dispatchable: True, 5/5 可派发, err_type=None
    /v1/admin/risk     -> 五个节点 posture=clear, dispatchable=True
    真实 POST /v1/messages
                       -> 401，x-should-retry / Retry-After 均缺失，
                          本地构造了 ['qwen','glm','doubao','kimi','deepseek'] 五个 Provider

三个「预览面」一致地说可以派发，真实派发却说不行。这是 OPT-031/035 记录过的同一类缺陷
（预演/档案在派发事实上说谎），只是这次成因是「这台机器还没登录过」——而这一次的代价
不只是误导运维：`/v1/diagnostics` 报 ok 会让 agent 接入脚本开始发请求，每个请求都要走完
整个环、在本地逐个构造五个 Provider 才失败，并在 `record_failure` 里给每个节点记下一次
60 秒的 `unconfigured` 冷却。

根因不是缺一个 `if`，而是「本机有没有凭证」这一位**没有进入唯一的派发闸门**：
`FailoverRouter.evaluate_dispatch_gate` 当时只有四道（credential_outage → disabled →
quarantine → long_avoidance）。四个预览面各自去问「有没有凭证」（或没问），于是必然漂移。
修复把它做成**第五道闸门**，四个面全部改成向同一处求值。

本套件锁住五件事：
1. 全环无凭证时，四个面**互相一致**，且与真实派发一致（503 + `x-should-retry: false`）；
2. 拒绝是零上游、**零本地构造**的（用 `_make_provider` 绊线证明，不靠推断）；
3. 拒绝不得写入任何风控结论（冷却、台账）——「还没登录」不是「被封禁」；
4. 反面：有一个节点有凭证时，四个面必须仍然说 ok，且该请求真的由它承接
   （只锁第 1 条的话，一个永远拒绝的实现也能通过）；
5. 一致性写成**对每个节点 × 每种状态**的性质，而不是逐个场景列举——本项目的漂移缺陷
   每次都是漏了某一格（OPT-031 漏长效避让、OPT-035 漏运维禁用、本条目漏无凭证）。

运行: .venv/bin/python tests/test_credential_readiness_agreement.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["TESTING"] = "1"
os.environ["GATEWAY_DISABLE_AUTO_REFRESH"] = "1"
os.environ["GATEWAY_DISABLE_PACING"] = "1"
os.environ["GATEWAY_DISABLE_SINGLEFLIGHT"] = "1"
os.environ["GATEWAY_DISABLE_PROBATION"] = "1"
os.environ["GATEWAY_DISABLE_RISK_CORRELATION"] = "1"
os.environ["GATEWAY_DISABLE_CREDENTIAL_WATCH"] = "1"
os.environ["GATEWAY_DISABLE_CREDENTIAL_RENEWAL"] = "1"
os.environ["GATEWAY_DISABLE_DEVICE_IDENTITY"] = "1"
os.environ["GATEWAY_PROBATION_MAX_RENEWALS"] = "0"
for _name in (
    "DISABLED_PROVIDERS",
    "GATEWAY_QUARANTINE",
    "GATEWAY_DISABLE_QUARANTINE",
    "GATEWAY_DISABLE_QUARANTINE_HARD_STOP",
    "GATEWAY_DISABLE_RISK_LEDGER",
    "GATEWAY_DISABLE_LONG_AVOIDANCE_FAST_FAIL",
    "GATEWAY_DISABLE_UNCONFIGURED_GATE",
    "GATEWAY_DISABLE_CREDENTIAL_RENEWAL",
    "GATEWAY_DISABLE_CREDENTIAL_WATCH",
):
    os.environ.pop(_name, None)

# 本套件的第一条性质就是「没有任何凭证」，所以环境变量必须在 import server 之前清掉：
# 任何一个残留的 *_AUTH_TOKEN 都会让 `session_store.is_configured()` 答 True。
for _name in (
    "QWEN_AUTH_TOKEN", "DEEPSEEK_AUTH_TOKEN", "DOUBAO_AUTH_TOKEN", "DOUBAO_SESSION_ID",
    "KIMI_AUTH_TOKEN", "KIMI_REFRESH_TOKEN", "GLM_AUTH_TOKEN", "GLM_REFRESH_TOKEN",
    "CHATGLM_AUTH_TOKEN",
):
    os.environ.pop(_name, None)

_SANDBOX = tempfile.mkdtemp(prefix="readiness_")
os.environ["GATEWAY_RISK_LEDGER_DIR"] = _SANDBOX
os.environ["GATEWAY_QUARANTINE_DIR"] = _SANDBOX
os.environ["GATEWAY_CONVERSATION_GC_DIR"] = _SANDBOX
os.environ["GATEWAY_DEVICE_IDENTITY_DIR"] = _SANDBOX

from fastapi.testclient import TestClient  # noqa: E402

import server as srv  # noqa: E402
import session_store  # noqa: E402
from providers.base import ProviderError  # noqa: E402
from providers.risk_dossier import POSTURE_UNCONFIGURED  # noqa: E402

PROVIDERS = ("qwen", "deepseek", "doubao", "kimi", "glm")

_BODY = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
_ANTHROPIC_BODY = {
    "model": "claude-3-5-sonnet",
    "max_tokens": 16,
    "messages": [{"role": "user", "content": "hi"}],
}


class _FakeSession:
    """只带 token 的最小 Session：本套件只关心「有没有凭证」这一位。"""

    def __init__(self, token: str = "fake-token"):
        self.token = token
        self.cookie = ""
        self.user_agent = ""
        self.user_id = ""
        self.device_id = ""
        self.captured_at = 0.0


def _patch_credentials(mapping):
    """在**每一层**顶替凭证解析器，返回还原函数。

    「有没有凭证」有三份读取者：`session_store._PROVIDER_RESOLVERS`（`is_configured`
    与派发闸门读它）、`server.py` 的模块级 `resolve_*` 名字（`_make_provider` 读它）、
    以及 `health_monitor._provider_resolvers`（探针读它）。它们都是在 import 或注册时
    抓的函数对象，所以顶替 `srv.resolve_qwen` **不会**改变另外两层的答案。

    这正是本套件要对准的缺陷形状：同一个事实有多处读法。要构造一个可信的场景，就得让
    三处一起变——只改一处的接线，测出来的是开发机磁盘，不是用例声明的那个状态。
    """
    saved_store = dict(session_store._PROVIDER_RESOLVERS)
    saved_monitor = dict(srv.health_monitor._provider_resolvers)
    saved_module = {pk: getattr(srv, f"resolve_{pk}") for pk in PROVIDERS}

    def _for(pk):
        return (lambda v=mapping[pk]: v) if pk in mapping else (lambda: None)

    for pk in PROVIDERS:
        session_store._PROVIDER_RESOLVERS[pk] = _for(pk)
        srv.health_monitor._provider_resolvers[pk] = _for(pk)
        setattr(srv, f"resolve_{pk}", _for(pk))

    def restore():
        session_store._PROVIDER_RESOLVERS.clear()
        session_store._PROVIDER_RESOLVERS.update(saved_store)
        srv.health_monitor._provider_resolvers.clear()
        srv.health_monitor._provider_resolvers.update(saved_monitor)
        for pk, fn in saved_module.items():
            setattr(srv, f"resolve_{pk}", fn)

    return restore


def _reset_state():
    """清空冷却 / 失败 / 租约 / 台账，使用例之间互不继承。

    冷却状态是**进程级单例**且会从磁盘恢复，所以不能只 `clear()` 内存——几个用例共用
    一个 `TestClient` 与一个路由器，前一个用例留下的 60 秒冷却会让后一个用例的节点
    带着不属于它的避让。
    """
    router = srv.failover_router
    tracker = router.cooldown_tracker
    for d in (tracker._cooldowns, tracker._failure_counts, tracker._cooldown_severity,
              tracker._episode_ends, tracker._budget_capped_streak):
        d.clear()
    tracker.disabled_providers.clear()
    try:
        srv.quarantine_store.clear_all_for_tests()
    except Exception:  # noqa: BLE001
        pass
    try:
        srv.risk_ledger.clear()
    except Exception:  # noqa: BLE001
        pass
    for obj in (tracker.credential_watch,):
        try:
            obj.clear_all_for_tests()
        except Exception:  # noqa: BLE001
            pass
    return tracker


def _surfaces(client):
    """一次取回四个面（就绪 / 诊断 / 预演 / 档案），后面全部按它们互相断言。"""
    health = client.get("/health").json()
    diagnostics = client.get("/v1/diagnostics").json()
    audit = client.get("/v1/admin/dispatch-audit", params={"model": "gpt-4o"}).json()
    dossier = client.get("/v1/admin/risk").json()
    return health, diagnostics, audit, dossier


class _Tripwire:
    """把 `_make_provider` 换成「记一笔再抛」——证明派发边界**没有**走到本地构造。

    为什么必须用它而不是「看日志里有没有 upstream 报错」：构造 Provider 是纯本地动作，
    失败时不会产生任何上游流量，日志里也看不出「它被构造了几次」。而缺陷的代价恰恰是
    这个次数（每个请求白构造五个），所以只有计数器能证明。
    """

    def __init__(self):
        self.calls = []

    def install(self):
        self.original = srv._make_provider

        def factory(provider_key, client=None):
            self.calls.append(provider_key)
            raise ProviderError(
                f"tripwire: {provider_key}", status=401, err_type="auth_error"
            )

        srv._make_provider = factory
        return self

    def __enter__(self):
        return self.install()

    def __exit__(self, *exc):
        srv._make_provider = self.original
        return False


def test_no_credential_refusal_is_zero_upstream_and_zero_local_construction():
    """全新克隆的第一次请求：确定性拒绝、零上游、**零本地构造**，且带退避信号。

    修复前实测：401，`x-should-retry` / `Retry-After` 均缺失，本地构造了五个 Provider。
    401 还是一个「认证失败」的结论——客户端会去查 key，而真正的原因是这台机器没登录。
    """
    tracker = _reset_state()
    restore = _patch_credentials({})
    tripwire = _Tripwire().install()
    try:
        client = TestClient(srv.app)
        resp = client.post("/v1/messages", json=_ANTHROPIC_BODY,
                           headers={"anthropic-version": "2023-06-01"})
    finally:
        srv._make_provider = tripwire.original
        restore()

    assert not tripwire.calls, (
        f"派发边界仍然构造了 Provider: {tripwire.calls}。没有凭证时每次构造都只是"
        f"一次本地失败，零信息量，却让每个客户端请求白付一次环遍历"
    )
    assert resp.status_code == 503, (
        f"整环无凭证应报 503（确定性拒绝），实际 {resp.status_code}。"
        f"401 会让客户端去查 key，而真正的原因是部署尚未完成"
    )
    assert resp.headers.get("x-should-retry") == "false", (
        f"缺 `x-should-retry: false`：这是 Claude Code 真正读的那个头（它按状态码分类"
        f"之前先读它），少了它客户端会对同一个零上游 503 反复重试。实测头: "
        f"{dict(resp.headers)}"
    )
    assert resp.headers.get("Retry-After"), "确定性拒绝必须给出标准退避时长"

    # 「还没登录」不是风控结论：不得留下冷却，更不得进台账。
    assert not any(tracker.get_remaining_cooldown(pk) > 0 for pk in PROVIDERS), (
        f"拒绝在节点上留下了冷却: "
        f"{ {pk: round(tracker.get_remaining_cooldown(pk), 1) for pk in PROVIDERS} }。"
        f"无凭证不是上游拦截，给节点记避让会让运维以为平台封了账号"
    )
    # 后台触点的成因清单里必须有 `unconfigured`：探针 /  silent refresh / 会话回收 /
    # 自愈选举都会问 `touch_avoidance_reason()`，它答 `unconfigured` 而诊断不列出来，
    # 运维读到的是「成因只有禁用/租约/冷却三种」，于是会去查错方向。
    assert "unconfigured" in tracker.touch_gate_report()["causes"]


def test_all_four_surfaces_agree_on_day_zero():
    """四个面必须互相一致：没有任何节点可派发，且**每一个面都这么说**。

    修复前：/health 说 degraded（对），/v1/diagnostics 说 ok，dispatch-audit 说 5/5 可派发，
    档案说五个 clear。三个面在派发事实上说谎——这是 OPT-031/035 的同一类缺陷，
    只是这次一个面都没被修过。
    """
    _reset_state()
    restore = _patch_credentials({})
    try:
        health, diagnostics, audit, dossier = _surfaces(TestClient(srv.app))
    finally:
        restore()

    # 1. 就绪面
    assert health["status"] == "degraded", health.get("status")
    assert health["dispatchable_providers"] == [], health["dispatchable_providers"]

    # 2. 诊断面：此前它连 `dispatchable` 字段都没有，并且顶层 status 恒为 ok
    per_node = diagnostics["providers"]
    missing = [pk for pk in PROVIDERS if "dispatchable" not in per_node.get(pk, {})]
    assert not missing, (
        f"/v1/diagnostics 的节点缺少 dispatchable 字段: {missing}。"
        f"这个字段是 agent 接入脚本判断「能不能发请求」的依据"
    )
    assert diagnostics["status"] != "ok", (
        f"五个节点全部无凭证，/v1/diagnostics 仍报 status={diagnostics['status']!r}"
    )
    assert diagnostics["dispatchable_providers"] == [], diagnostics["dispatchable_providers"]

    # 3. 预演面：此前 predicted_primary_dispatchable=True、5/5、err_type=None
    assert audit["upstream_calls"] == 0
    assert audit["predicted_primary_dispatchable"] is False, (
        f"预演说主选节点可派发，而真实派发会拒绝它（预测: "
        f"{audit.get('predicted_primary_dispatchable')}）"
    )
    assert audit["predicted_fail_closed"] is True
    assert audit["predicted_err_type"] == "unconfigured", audit.get("predicted_err_type")
    blocked = {m["provider"]: m["blocked_by"] for m in audit["ring_members"]}
    assert set(blocked.values()) == {"unconfigured"}, (
        f"预演把无凭证的节点报成了别的原因: {blocked}"
    )

    # 4. 档案面：此前五个 posture=clear、dispatchable=True
    entries = dossier["providers"]
    wrong = {pk: (e["posture"], e["dispatchable"]) for pk, e in entries.items()
             if e["posture"] != POSTURE_UNCONFIGURED or e["dispatchable"]}
    assert not wrong, (
        f"档案没有把无凭证的节点报成 {POSTURE_UNCONFIGURED!r}/不可派发: {wrong}。"
        f"运维读到 clear 会以为节点是健康的，只是「还没轮到」"
    )
    assert dossier["summary"]["dispatchable_count"] == 0
    assert dossier["summary"]["posture_counts"].get(POSTURE_UNCONFIGURED) == len(PROVIDERS)

    # 5. 四个面与真实派发一致
    resp = TestClient(srv.app).post("/v1/chat/completions", json=_BODY)
    assert resp.status_code == 503, (
        f"三个预览面说不可派发，真实派发却报 {resp.status_code}——预览必须与派发同口径"
    )


def test_one_configured_provider_restores_service_on_every_surface():
    """反面：给一个节点配上凭证，四个面必须同时恢复到「可服务」，且请求真的由它承接。

    只锁「全无凭证时拒绝」的话，一个无条件拒绝的实现也能通过全部用例——那种实现会让
    网关彻底不可用，而套件全绿。
    """
    _reset_state()
    restore = _patch_credentials({"qwen": _FakeSession("fake-qwen")})

    served = []

    class _ScriptedProvider:
        async def chat(self, provider_key=None, *a, **kw):
            served.append(provider_key)
            yield "ok", {"conversation_id": "q_c"}
            yield "", {"conversation_id": "q_c"}

    original_factory = srv._make_provider
    srv._make_provider = lambda key, client=None: _ScriptedProvider()
    try:
        client = TestClient(srv.app)
        health, diagnostics, audit, dossier = _surfaces(client)
        resp = None
        try:
            resp = client.post("/v1/chat/completions", json=_BODY)
        finally:
            pass
    finally:
        srv._make_provider = original_factory
        restore()

    assert health["status"] == "ok", (
        f"qwen 有凭证且无避让，/health 报 {health['status']!r}——修成永远降级会让探针失效"
    )
    assert "qwen" in health["dispatchable_providers"]
    assert diagnostics["status"] == "ok", diagnostics.get("status")
    assert "qwen" in diagnostics["dispatchable_providers"]
    assert audit["predicted_primary_dispatchable"] is True, audit
    assert dossier["providers"]["qwen"]["dispatchable"] is True
    assert dossier["providers"]["qwen"]["posture"] != POSTURE_UNCONFIGURED
    # 其余四个仍然无凭证：它们照旧不可派发，但整体已经可服务。
    assert dossier["providers"]["glm"]["posture"] == POSTURE_UNCONFIGURED

    assert resp is not None and resp.status_code == 200, (
        f"有一个节点有凭证时请求仍被拒绝（{None if resp is None else resp.status_code}）"
    )


def test_dossier_dispatchables_agree_with_the_gate_for_every_node_and_state():
    """性质：逐节点、逐状态下 `档案 dispatchable == 派发闸门 dispatchable`。

    不逐场景列举，是因为本项目的漂移缺陷**每次都是漏了某一格**：OPT-031 漏了长效避让、
    OPT-035 漏了运维禁用、本条目漏了无凭证。一个属性断言让下一格漏掉时直接变红，
    而不必等运维实测发现。
    """
    scenarios = {
        "healthy": ({"qwen": _FakeSession(), "deepseek": _FakeSession(),
                     "doubao": _FakeSession(), "kimi": _FakeSession(),
                     "glm": _FakeSession()}, {}),
        "day_zero": ({}, {}),
        "disabled": ({"qwen": _FakeSession(), "glm": _FakeSession()},
                     {"DISABLED_PROVIDERS": "kimi,deepseek,doubao"}),
        "mixed_disabled_and_unconfigured": ({'qwen': _FakeSession()},
                                            {"DISABLED_PROVIDERS": "glm"}),
    }
    client = TestClient(srv.app)
    for name, (mapping, env) in scenarios.items():
        _reset_state()
        for k in ("DISABLED_PROVIDERS",):
            os.environ.pop(k, None)
        os.environ.update(env)
        restore = _patch_credentials(mapping)
        try:
            dossier = client.get("/v1/admin/risk").json()
            gates = {
                pk: bool(srv.failover_router.evaluate_dispatch_gate(pk)["dispatchable"])
                for pk in PROVIDERS
            }
        finally:
            restore()
            for k in env:
                os.environ.pop(k, None)

        for pk in PROVIDERS:
            got = dossier["providers"][pk]["dispatchable"]
            assert got == gates[pk], (
                f"[{name}] {pk}: 档案报 dispatchable={got!r}，派发闸门为 {gates[pk]!r}。"
                f"档案的职责之一就是回答「这个节点现在能不能被派发」，它与派发边界"
                f"必须是同一个答案（posture={dossier['providers'][pk]['posture']!r}）"
            )


def test_diagnostics_status_is_derived_from_dispatchability():
    """性质：`/v1/diagnostics` 的顶层 status 只能由「真的有节点可派发」推出。

    修复前它恒为 ok——一个全新克隆（五个节点全离线、无凭证）的诊断接口报 ok，
    而 agent 接入脚本正是拿它判断「网关好了没有」。
    """
    client = TestClient(srv.app)
    for mapping in ({}, {"glm": _FakeSession()}):
        _reset_state()
        restore = _patch_credentials(mapping)
        try:
            diagnostics = client.get("/v1/diagnostics").json()
        finally:
            restore()
        should_be_ok = bool(diagnostics["dispatchable_providers"])
        assert (diagnostics["status"] == "ok") == should_be_ok, (
            f"dispatchable_providers={diagnostics['dispatchable_providers']}，"
            f"顶层 status 却是 {diagnostics['status']!r}；"
            f"status_reason={diagnostics.get('status_reason')!r}"
        )
        if not should_be_ok:
            assert diagnostics.get("status_reason", "").strip(), (
                "降级状态必须给出成因（运维需要知道该执行 login.py，而不是去查网络）"
            )


def test_kill_switch_restores_the_old_walk_the_ring_behaviour():
    """`GATEWAY_DISABLE_UNCONFIGURED_GATE=1` 必须逐值退回旧行为。

    一个结论一旦进入派发边界，就必须有一条明确的退回路径——否则任何以「无凭证节点」
    为前置条件的诊断或测试都会失去原来的语义。旧行为是：照旧派发，逐个在本地构造失败。
    """
    _reset_state()
    os.environ["GATEWAY_DISABLE_UNCONFIGURED_GATE"] = "1"
    restore = _patch_credentials({})
    tripwire = _Tripwire().install()
    try:
        client = TestClient(srv.app)
        resp = client.post("/v1/chat/completions", json=_BODY)
        gate = srv.failover_router.evaluate_dispatch_gate("qwen")
    finally:
        srv._make_provider = tripwire.original
        restore()
        os.environ.pop("GATEWAY_DISABLE_UNCONFIGURED_GATE", None)

    assert gate["dispatchable"] is True, (
        f"开关打开后闸门仍然阻断: {gate}。退回路径必须逐值恢复旧语义"
    )
    assert tripwire.calls, "开关打开后应当照旧走到本地构造（旧行为）"
    assert resp.status_code != 503


def test_audit_and_dossier_queries_are_read_only():
    """预演与档案都是只读的：查询一百次也不得改变任何避让状态。

    这两个接口会被监控反复轮询，也会被运维在动作后反复确认；它们一旦写状态，
    「查一下」本身就成了一个上游动作。
    """
    tracker = _reset_state()
    restore = _patch_credentials({})
    try:
        client = TestClient(srv.app)
        for _ in range(5):
            client.get("/v1/admin/dispatch-audit", params={"model": "gpt-4o"})
            client.get("/v1/admin/risk")
            client.get("/v1/health")
    finally:
        restore()

    assert not any(tracker.get_remaining_cooldown(pk) > 0 for pk in PROVIDERS), (
        "只读查询改变了冷却状态"
    )
    report = tracker.touch_gate_report()
    assert not report["blocked_by_cause"], (
        f"只读查询被记成了「被阻断的上游触碰」: {report['blocked_by_cause']}。"
        f"这两个接口会被监控反复轮询，记录它们等于让「查一下」变成一次动作"
    )


def run_all():
    test_no_credential_refusal_is_zero_upstream_and_zero_local_construction()
    test_all_four_surfaces_agree_on_day_zero()
    test_one_configured_provider_restores_service_on_every_surface()
    test_dossier_dispatchables_agree_with_the_gate_for_every_node_and_state()
    test_diagnostics_status_is_derived_from_dispatchability()
    test_kill_switch_restores_the_old_walk_the_ring_behaviour()
    test_audit_and_dossier_queries_are_read_only()
    print("\n✅ All credential-readiness agreement tests passed!")


if __name__ == "__main__":
    run_all()