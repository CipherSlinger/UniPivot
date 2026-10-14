"""零上游派发预演 (OPT-032) 测试套件

动机来自一次真实操作：运维声明「deepseek 已被封禁，零误触是硬约束」之后，唯一能确认
这份声明**真的生效**的办法是手写一段脚本、注入假 Provider 工厂、用 TestClient 发一次
请求，再看被触碰的节点集合。这个确认动作是日常的——每次改租约、改环境变量、重启之后
都要再问一遍「现在到底会不会派发到它」；而逐个翻看五个 admin 端点都回答不了它，因为
问题不是「这个节点处于什么姿态」，而是「这条请求此刻的派发决定是什么」。

本套件验证 `FailoverRouter.audit_dispatch()` 与 `GET /v1/admin/dispatch-audit`：
1. 只读且零上游调用（端点路径下把 Provider 工厂换成会抛错的实现，预演仍必须成功）
2. 逐节点标出 blocked_by（quarantine / long_avoidance / credential_outage / disabled）与剩余时长
3. 租约节点的成因报 ban（重登无效），不得给出「重登即可」的误导
4. 短软冷却**不**被报告为阻断（保住 OPT-021 的「最后一搏」语义）
5. 整环租约 → fail_closed + quarantined；整环长效避让 → ring_exhausted；全环凭证失效 → credential_outage
6. **与派发路径零漂移**：预演说「改派到 X」，真实 execute_chat 就必须改派到 X；
   预演说 fail_closed + err_type，真实抛出的 err_type 就必须一致
7. 传入模型别名而非 provider key → ValueError（宁可当场报错，也不产出问错对象的预演）
8. 端点：模型别名正确分类到能力环，未知模型 400

运行: .venv/bin/python tests/test_dispatch_audit.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["TESTING"] = "1"
os.environ["GATEWAY_DISABLE_AUTO_REFRESH"] = "1"
os.environ["GATEWAY_DISABLE_PACING"] = "1"
os.environ["GATEWAY_DISABLE_SINGLEFLIGHT"] = "1"
for _name in (
    "GATEWAY_DISABLE_LONG_AVOIDANCE_FAST_FAIL",
    "GATEWAY_DISABLE_QUARANTINE",
    "GATEWAY_DISABLE_QUARANTINE_HARD_STOP",
    "GATEWAY_DISABLE_RISK_LEDGER",
    "GATEWAY_DISABLE_CREDENTIAL_WATCH",
    "GATEWAY_DISABLE_BAN_INFERENCE",
    "GATEWAY_DISABLE_PROBATION",
    "GATEWAY_QUARANTINE",
):
    # 关键：GATEWAY_QUARANTINE 必须清掉，否则运维真实 .env 里的 deepseek 租约会渗进
    # 本套件，让「无租约」的前置条件不成立（这正是本项目记录过的跨环境串扰类型）。
    os.environ.pop(_name, None)
os.environ["GATEWAY_PROBATION_MAX_RENEWALS"] = "0"
os.environ["QWEN_AUTH_TOKEN"] = "fake_qwen"
os.environ["DEEPSEEK_AUTH_TOKEN"] = "fake_ds"
os.environ["DOUBAO_AUTH_TOKEN"] = "fake_doubao"
os.environ["KIMI_AUTH_TOKEN"] = "fake_kimi"
os.environ["GLM_AUTH_TOKEN"] = "fake_glm"

from fastapi.testclient import TestClient  # noqa: E402

import server as srv  # noqa: E402
from providers import failover as fov  # noqa: E402

FLAGSHIP_RING = ("qwen", "glm", "kimi", "doubao", "deepseek")
TOUCHED: list[str] = []


def _run(coro):
    return asyncio.run(coro)


def _make_fake():
    def factory(key: str):
        class _Fake:
            async def chat(self, *a, **kw):
                TOUCHED.append(key)
                yield ("hi", {})

            async def delete_conversation(self, *a, **kw):
                return None

        return _Fake()

    return factory


def _clear_shared_evidence(tracker):
    """清空进程级单例证据（凭证失效/续期检测不随 tracker.reset() 复位）。"""
    try:
        tracker.credential_watch.clear_all_for_tests()
    except Exception:  # noqa: BLE001
        pass
    try:
        from providers.credential_renewal import get_credential_renewal

        get_credential_renewal().clear_all_for_tests()
    except Exception:  # noqa: BLE001
        pass


def _reset_router():
    router = srv.failover_router
    tracker = router.cooldown_tracker
    tracker.reset()
    srv.quarantine_store.clear_all_for_tests()
    _clear_shared_evidence(tracker)
    TOUCHED.clear()
    return router, tracker


def _waf(tracker, pk, times=1):
    for _ in range(times):
        _run(tracker.record_failure(pk, 403, "RGV587 人机验证失败，请完成滑块验证"))


def _auth(tracker, pk, times=1):
    for _ in range(times):
        _run(tracker.record_failure(pk, 401, "401 Unauthorized: token expired"))


def _members(report):
    return {m["provider"]: m for m in report["ring_members"]}


# ------------------------------------------------------- 1. 逐节点闸门标记

def test_lease_node_is_marked_with_ban_cause():
    router, tracker = _reset_router()
    try:
        srv.quarantine_store.set_lease("deepseek", duration_seconds=72 * 3600)
        r = router.audit_dispatch("deepseek", "deepseek-chat")

        assert r["upstream_calls"] == 0
        assert r["ring"] == "flagship" and r["ring_size"] == 5
        ds = _members(r)["deepseek"]
        assert ds["dispatchable"] is False
        assert ds["blocked_by"] == "quarantine"
        assert ds["blocking_cause"] == "ban", "租约必须报 ban（重登无效），不得暗示重登可解"
        assert ds["remaining_hours"] > 70

        # 主选被封禁但环内还有健康节点 → 改派，且**不得**报错
        assert r["predicted_primary_dispatchable"] is False
        expected = router.get_fallback_candidates("deepseek", "deepseek-chat")[0].provider_key
        assert r["predicted_redirect_target"] == expected
        assert r["predicted_redirect_target"] != "deepseek"
        assert r["predicted_fail_closed"] is False
        assert r["predicted_err_type"] is None, "能改派就不是失败，报 err_type 即谎报"
        assert r["dispatchable_count"] == 4
    finally:
        _reset_router()
    print("test_lease_node_is_marked_with_ban_cause: PASSED")


def test_long_avoidance_marked_but_short_cooldown_is_not():
    """长效避让必须报阻断；短软冷却**不**得报阻断（OPT-021 的最后一搏语义）。"""
    router, tracker = _reset_router()
    try:
        _waf(tracker, "glm", 3)  # 升级为长效避让
        assert tracker.is_in_long_avoidance("glm") is True
        r = router.audit_dispatch("glm", "glm-4-plus")
        glm = _members(r)["glm"]
        assert glm["blocked_by"] == "long_avoidance"
        assert glm["blocking_cause"] == "waf"
        assert glm["remaining_hours"] >= 1.0

        # 短冷却（429 一次）既不阻断也不进长效避让
        _run(tracker.record_failure("kimi", 429, "rate limit exceeded"))
        assert tracker.is_in_cooldown("kimi") is True
        r2 = router.audit_dispatch("kimi", "kimi")
        kimi = _members(r2)["kimi"]
        assert kimi["dispatchable"] is True, "短冷却必须仍可派发（可用性优先）"
        assert kimi["blocked_by"] is None
        assert _members(r2)["glm"]["blocked_by"] == "long_avoidance"
    finally:
        _reset_router()
    print("test_long_avoidance_marked_but_short_cooldown_is_not: PASSED")


def test_audit_is_read_only():
    """预演绝不得改动任何风控状态（不改租约、不清冷却、不写台账）。"""
    router, tracker = _reset_router()
    try:
        srv.quarantine_store.set_lease("deepseek", duration_seconds=3600)
        _waf(tracker, "glm", 3)
        lease_before = srv.quarantine_store.remaining("deepseek")
        cd_before = tracker.get_remaining_cooldown("glm")

        for _ in range(3):
            router.audit_dispatch("qwen", "Qwen")

        assert srv.quarantine_store.remaining("deepseek") > 0
        assert tracker.is_in_long_avoidance("glm") is True
        assert abs(srv.quarantine_store.remaining("deepseek") - lease_before) < 1.0
        assert tracker.get_remaining_cooldown("glm") <= cd_before
    finally:
        _reset_router()
    print("test_audit_is_read_only: PASSED")


# ------------------------------------------------------- 2. 整环判定

def test_whole_ring_lease_predicts_quarantined_fail_closed():
    router, tracker = _reset_router()
    try:
        for pk in FLAGSHIP_RING:
            srv.quarantine_store.set_lease(pk, duration_seconds=72 * 3600)
        r = router.audit_dispatch("deepseek", "deepseek-chat")
        assert r["dispatchable_count"] == 0
        assert r["predicted_fail_closed"] is True
        assert r["predicted_err_type"] == "quarantined"
        assert r["predicted_redirect_target"] is None
        assert all(m["blocked_by"] == "quarantine" for m in r["ring_members"])
    finally:
        _reset_router()
    print("test_whole_ring_lease_predicts_quarantined_fail_closed: PASSED")


def test_whole_ring_long_avoidance_predicts_ring_exhausted():
    router, tracker = _reset_router()
    try:
        for pk in ("qwen", "glm", "kimi"):
            _waf(tracker, pk, 3)
        for pk in ("doubao", "deepseek"):
            _auth(tracker, pk, 3)
        r = router.audit_dispatch("qwen", "Qwen")
        assert r["dispatchable_count"] == 0
        assert r["predicted_fail_closed"] is True
        assert r["predicted_err_type"] == "ring_exhausted"
        causes = {m["provider"]: m["blocking_cause"] for m in r["ring_members"]}
        assert causes["qwen"] == "waf" and causes["doubao"] == "auth"
    finally:
        _reset_router()
    print("test_whole_ring_long_avoidance_predicts_ring_exhausted: PASSED")


def test_credential_outage_is_reported_as_ring_level():
    """整环凭证失效是**环级**闸门，必须排在逐节点之前，且所有节点都不可派发。"""
    router, tracker = _reset_router()
    try:
        for pk in FLAGSHIP_RING[:3]:
            _auth(tracker, pk, 1)
        assert tracker.credential_watch.is_outage_active() is True

        r = router.audit_dispatch("qwen", "Qwen")
        assert r["credential_outage"]["active"] is True
        assert set(r["credential_outage"]["failed_providers"]) == set(FLAGSHIP_RING[:3])
        assert r["dispatchable_count"] == 0
        assert r["predicted_fail_closed"] is True
        assert r["predicted_err_type"] == "credential_outage"
    finally:
        _reset_router()
    print("test_credential_outage_is_reported_as_ring_level: PASSED")


# ------------------------------------------------------- 3. 与派发路径零漂移

def test_prediction_matches_real_dispatch_redirect():
    """**本套件最重要的一条**：预演说改派到谁，真实派发就必须改派到谁。

    预演一旦与派发路径漂移，它就从「可信的确认手段」退化成「更精致的谎报」——那正是
    本项目反复记录的「文档说会避让、实现仍派发」缺陷类型。
    """
    router, tracker = _reset_router()
    try:
        srv.quarantine_store.set_lease("deepseek", duration_seconds=72 * 3600)
        predicted = router.audit_dispatch("deepseek", "deepseek-chat")["predicted_redirect_target"]
        assert predicted is not None

        stream, event, active = _run(
            router.execute_chat(
                "deepseek", "deepseek-chat", provider_factory=_make_fake(), chat_args={}
            )
        )
        # TOUCHED 记录的是真实进入 chat() 的节点 —— 以观测到的触碰为准，而非从返回对象反推
        assert TOUCHED, "必须真的派发了某个节点"
        assert TOUCHED[-1] == predicted, f"预演说改派到 {predicted}，实际派发到 {TOUCHED[-1]}"
        assert "deepseek" not in TOUCHED, "封禁节点被真实触碰了"
    finally:
        _reset_router()
    print("test_prediction_matches_real_dispatch_redirect: PASSED")


def test_prediction_matches_real_dispatch_error_type():
    """预演说 fail_closed + err_type，真实抛出的 err_type 就必须一致。"""
    from providers.base import ProviderError

    router, tracker = _reset_router()
    try:
        for pk in FLAGSHIP_RING:
            srv.quarantine_store.set_lease(pk, duration_seconds=72 * 3600)
        predicted = router.audit_dispatch("qwen", "Qwen")
        assert predicted["predicted_err_type"] == "quarantined"

        try:
            _run(router.execute_chat("qwen", "Qwen", provider_factory=_make_fake(), chat_args={}))
            raise AssertionError("整环封禁时 execute_chat 必须抛错")
        except ProviderError as exc:
            assert exc.err_type == predicted["predicted_err_type"], (exc.err_type, predicted)
            assert exc.status == 503
        assert TOUCHED == [], f"整环封禁必须零上游触碰，实际 {TOUCHED}"
    finally:
        _reset_router()
    print("test_prediction_matches_real_dispatch_error_type: PASSED")


# ------------------------------------------------------- 4. 用法防错

def test_model_alias_instead_of_provider_key_is_rejected():
    """传模型别名是最容易犯的错，且会静默产出「问错对象」的预演（把封禁报成可派发）。"""
    router, tracker = _reset_router()
    try:
        try:
            router.audit_dispatch("deepseek-chat", "deepseek-chat")
            raise AssertionError("必须以 ValueError 拒绝未解析的模型别名")
        except ValueError as exc:
            assert "resolve_model" in str(exc)
    finally:
        _reset_router()
    print("test_model_alias_instead_of_provider_key_is_rejected: PASSED")


# ------------------------------------------------------- 5. 端点

def test_endpoint_never_instantiates_a_provider():
    """端点路径下把 Provider 工厂换成会抛错的实现，预演仍必须成功 —— 零上游调用的构造性证明。"""
    _reset_router()
    original = srv._make_provider

    def _boom(*a, **kw):
        raise AssertionError("派发预演绝不能实例化 Provider")

    srv._make_provider = _boom
    try:
        c = TestClient(srv.app)
        r = c.get("/v1/admin/dispatch-audit", params={"model": "deepseek-chat"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["upstream_calls"] == 0
        assert body["provider"] == "deepseek" and body["ring"] == "flagship"
        assert body["scope_note"], "必须显式声明「不含负载均衡与会话亲和」的边界"
    finally:
        srv._make_provider = original
        _reset_router()
    print("test_endpoint_never_instantiates_a_provider: PASSED")


def test_endpoint_model_classification_and_errors():
    _reset_router()
    try:
        c = TestClient(srv.app)
        flagship = c.get("/v1/admin/dispatch-audit", params={"model": "claude-3-7-sonnet"}).json()
        assert flagship["ring"] == "flagship" and flagship["ring_size"] == 5

        speed = c.get("/v1/admin/dispatch-audit", params={"model": "gpt-4o-mini"}).json()
        assert speed["ring"] == "speed" and speed["ring_size"] == 3
        assert "deepseek" not in {m["provider"] for m in speed["ring_members"]}, (
            "SPEED 环只有 qwen/doubao/glm，预演不得凭空塞入其他节点"
        )

        default = c.get("/v1/admin/dispatch-audit").json()
        assert default["provider"] == os.getenv("CLAUDE_BACKEND", "qwen")

        bad = c.get("/v1/admin/dispatch-audit", params={"model": "totally-unknown-xyz"})
        assert bad.status_code == 400
        assert bad.json()["error"]["type"] == "invalid_request_error"
    finally:
        _reset_router()
    print("test_endpoint_model_classification_and_errors: PASSED")


def run_all():
    tests = [
        test_lease_node_is_marked_with_ban_cause,
        test_long_avoidance_marked_but_short_cooldown_is_not,
        test_audit_is_read_only,
        test_whole_ring_lease_predicts_quarantined_fail_closed,
        test_whole_ring_long_avoidance_predicts_ring_exhausted,
        test_credential_outage_is_reported_as_ring_level,
        test_prediction_matches_real_dispatch_redirect,
        test_prediction_matches_real_dispatch_error_type,
        test_model_alias_instead_of_provider_key_is_rejected,
        test_endpoint_never_instantiates_a_provider,
        test_endpoint_model_classification_and_errors,
    ]
    for fn in tests:
        fn()
    print(f"\n✅ All dispatch-audit tests (OPT-032) passed! ({len(tests)} cases)")


if __name__ == "__main__":
    run_all()
