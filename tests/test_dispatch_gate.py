"""派发闸门的唯一定义与运维禁用硬阻断 (OPT-033) 测试套件

动机是一条实测出来的漂移：运维用 `DISABLED_PROVIDERS=deepseek` 显式声明「这个节点不属于
本次部署」，负载均衡会分流、会话亲和会避开它、**而派发边界本身毫无门控**——只要请求走到
execute_chat 时主选恰好是它（显式指定、或亲和重新解析的结果），请求就真的被送进了被运维
禁用的节点。与此同时 OPT-032 的零上游预演却报它是 `blocked_by="disabled"` 并预测改派到
glm。**声明与事实不一致**，而预演的全部价值就是「说实话」。

根因不是漏写一个 if，而是**闸门顺序有两份副本**：一份在 execute_chat 的派发循环里，一份在
audit_dispatch 里。OPT-032 的零漂移是靠测试保证的（预演说改派到 X，真实派发就必须派发到
X）——那已比两处代码各自正确前进了一步，但测试只能覆盖它想到的场景，`disabled` 这一格就
漏了。本套件锁定 OPT-033 的收束：闸门顺序只有**一份**定义
（`FailoverRouter.evaluate_dispatch_gate`），预演与派发都只向它求值。

覆盖：
1. 被禁用主选被改派，且**零触碰**被禁用节点（复现并锁死原始漏洞场景）
2. 整环禁用 → 503 / err_type="disabled" 且零上游调用，报文给出解除指引
3. 预演与派发在被禁用节点上零漂移（本轮修复的直接靶点）
4. **结构性**零漂移：篡改唯一闸门定义，派发必须随之改变（若 execute_chat 退回自带副本必红）
5. 闸门顺序共享：环级凭证失效优先于逐节点禁用，但逐节点状态仍作为备注可见
6. 禁用判定只有一份解读：直接改 tracker 集合（不经环境变量）也会让 _is_provider_healthy 生效
7. 可用性零回归：短软冷却仍可派发（OPT-021 的「最后一搏」语义）
8. 端到端：全环禁用 → OpenAI 503 disabled / Anthropic overloaded_error，零上游调用

运行: .venv/bin/python tests/test_dispatch_gate.py
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
    "DISABLED_PROVIDERS",
):
    os.environ.pop(_name, None)
os.environ["GATEWAY_PROBATION_MAX_RENEWALS"] = "0"
os.environ["QWEN_AUTH_TOKEN"] = "fake_qwen"
os.environ["DEEPSEEK_AUTH_TOKEN"] = "fake_ds"
os.environ["DOUBAO_AUTH_TOKEN"] = "fake_doubao"
os.environ["KIMI_AUTH_TOKEN"] = "fake_kimi"
os.environ["GLM_AUTH_TOKEN"] = "fake_glm"

from fastapi.testclient import TestClient  # noqa: E402

import server as srv  # noqa: E402
from providers.base import ProviderError  # noqa: E402

ALL_PROVIDERS = ("qwen", "deepseek", "doubao", "kimi", "glm")
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
    """复位路由器并返回 (router, tracker, 禁用集合快照)。

    禁用集合**直接改集合**（不经环境变量）是本套件的刻意选择：它同时证明「禁用判定只有
    一份解读」——若某处仍各自解析 DISABLED_PROVIDERS 环境变量，直接改集合就对它无效。
    """
    router = srv.failover_router
    tracker = router.cooldown_tracker
    tracker.reset()
    srv.quarantine_store.clear_all_for_tests()
    _clear_shared_evidence(tracker)
    TOUCHED.clear()
    return router, tracker, set(tracker.disabled_providers)


def _restore(tracker, snapshot):
    tracker.disabled_providers = set(snapshot)
    tracker.reset()
    srv.quarantine_store.clear_all_for_tests()
    _clear_shared_evidence(tracker)
    TOUCHED.clear()


def _drain(coro_ctx):
    async def _inner():
        stream, _ev, _p = await coro_ctx
        async for _ in stream:
            pass

    _run(_inner())


# ------------------------------------------------- 1. 被禁用主选：改派 + 零触碰

def test_disabled_primary_is_redirected_with_zero_touch():
    router, tracker, snap = _reset_router()
    try:
        tracker.disabled_providers = {"deepseek"}
        _drain(
            router.execute_chat(
                "deepseek", "deepseek-chat", provider_factory=_make_fake(), chat_args={}
            )
        )
        assert "deepseek" not in TOUCHED, f"被运维禁用的节点被真实触碰了: {TOUCHED}"
        assert TOUCHED, "必须改派到环内其他可用节点，而不是直接失败"
        report = tracker.disabled_gate_report()
        assert report["stats"]["redirects"] == 1
        assert report["stats"]["last_redirect"].startswith("deepseek->")
        assert report["gate_blocks"]["disabled"] == 1, "闸门拒载分布必须记录是谁拦下的"
    finally:
        _restore(tracker, snap)
    print("test_disabled_primary_is_redirected_with_zero_touch: PASSED")


def test_whole_ring_disabled_fails_closed_with_zero_upstream_calls():
    router, tracker, snap = _reset_router()
    try:
        tracker.disabled_providers = set(ALL_PROVIDERS)
        raised = None
        try:
            _drain(
                router.execute_chat(
                    "deepseek", "deepseek-chat", provider_factory=_make_fake(), chat_args={}
                )
            )
        except ProviderError as exc:
            raised = exc
        assert raised is not None, "整环被禁用时必须快速失败，而不是派发"
        assert raised.status == 503
        assert raised.err_type == "disabled"
        assert "DISABLED_PROVIDERS" in raised.message, "必须给出解除禁用声明的指引"
        assert TOUCHED == [], f"必须零上游触碰，实际: {TOUCHED}"
        assert tracker.disabled_gate_report()["stats"]["fast_fails"] == 1
    finally:
        _restore(tracker, snap)
    print("test_whole_ring_disabled_fails_closed_with_zero_upstream_calls: PASSED")


# ------------------------------------------------- 2. 预演与派发零漂移

def test_audit_and_dispatch_agree_on_disabled_primary():
    """**本套件的直接靶点**：修复前预演报「改派到 glm」而真实派发进了 deepseek。"""
    router, tracker, snap = _reset_router()
    try:
        tracker.disabled_providers = {"deepseek"}
        audit = router.audit_dispatch("deepseek", "deepseek-chat")
        member = {m["provider"]: m for m in audit["ring_members"]}["deepseek"]
        assert member["blocked_by"] == "disabled"
        assert member["dispatchable"] is False
        assert member["disabled"] is True, "逐节点备注里必须能看见禁用状态本身"
        assert audit["predicted_fail_closed"] is False, "能改派就不是失败"

        predicted = audit["predicted_redirect_target"]
        assert predicted is not None and predicted != "deepseek"

        _drain(
            router.execute_chat(
                "deepseek", "deepseek-chat", provider_factory=_make_fake(), chat_args={}
            )
        )
        assert TOUCHED[-1] == predicted, f"预演说改派到 {predicted}，实际派发到 {TOUCHED[-1]}"
        assert "deepseek" not in TOUCHED
    finally:
        _restore(tracker, snap)
    print("test_audit_and_dispatch_agree_on_disabled_primary: PASSED")


def test_audit_and_dispatch_agree_on_ring_wide_disabled_err_type():
    router, tracker, snap = _reset_router()
    try:
        tracker.disabled_providers = set(ALL_PROVIDERS)
        audit = router.audit_dispatch("qwen", "Qwen")
        assert audit["predicted_fail_closed"] is True
        assert audit["predicted_err_type"] == "disabled"
        try:
            _drain(router.execute_chat("qwen", "Qwen", provider_factory=_make_fake(), chat_args={}))
            raise AssertionError("整环禁用时 execute_chat 必须抛错")
        except ProviderError as exc:
            assert exc.err_type == audit["predicted_err_type"], (exc.err_type, audit)
        assert TOUCHED == []
    finally:
        _restore(tracker, snap)
    print("test_audit_and_dispatch_agree_on_ring_wide_disabled_err_type: PASSED")


# ------------------------------------------------- 3. 结构性零漂移

def test_dispatch_decision_is_derived_from_the_single_gate_definition():
    """**结构性**证明「闸门只有一份定义」：篡改那唯一一份定义，派发必须随之改变。

    与上一条的区别：上一条验证「此刻预演 == 派发」（可以被两份各自正确的副本同时满足，
    正如修复前 `disabled` 那一格所暴露的）；本条验证「派发没有第二份副本可走」——若
    execute_chat 哪天退回自带的 if/elif 副本，它就会无视被篡改的闸门定义而照常派发，
    本条立即变红。这是把「零漂移」从测试保证升级为结构保证的那一条。
    """
    router, tracker, snap = _reset_router()
    original = router.evaluate_dispatch_gate
    consulted: list[str] = []

    def _rigged(provider_key, outage_active=None):
        consulted.append(provider_key)
        gate = original(provider_key, outage_active=outage_active)
        if provider_key == "qwen":
            # 谎报：qwen 其实健康，但闸门说它被禁用
            gate.update(dispatchable=False, blocked_by="disabled", disabled=True, cause="")
        return gate

    router.evaluate_dispatch_gate = _rigged
    try:
        _drain(router.execute_chat("qwen", "Qwen", provider_factory=_make_fake(), chat_args={}))
        assert "qwen" in consulted, "派发循环必须向唯一闸门定义求值"
        assert "qwen" not in TOUCHED, (
            f"闸门说不可派发，派发路径却照常派发了 —— 说明它自带了一份副本: {TOUCHED}"
        )
        assert TOUCHED, "应当改派到环内其他可用节点"
    finally:
        router.evaluate_dispatch_gate = original
        _restore(tracker, snap)
    print("test_dispatch_decision_is_derived_from_the_single_gate_definition: PASSED")


def test_gate_order_puts_ring_level_outage_before_per_node_disabled():
    """环级凭证失效必须排在逐节点闸门之前（否则会误导运维去处理某一个节点），
    但逐节点状态仍须作为备注可见，不能被环级成因抹掉。"""
    router, tracker, snap = _reset_router()
    try:
        tracker.disabled_providers = {"deepseek"}
        for pk in ("qwen", "doubao", "kimi"):
            _run(tracker.record_failure(pk, 401, "401 Unauthorized: token expired"))
        assert tracker.credential_watch.is_outage_active() is True

        audit = router.audit_dispatch("qwen", "Qwen")
        member = {m["provider"]: m for m in audit["ring_members"]}["deepseek"]
        assert member["blocked_by"] == "credential_outage", "环级成因优先报告"
        assert member["disabled"] is True, "被前置闸门盖住的逐节点状态必须仍可见"

        try:
            _drain(router.execute_chat("qwen", "Qwen", provider_factory=_make_fake(), chat_args={}))
            raise AssertionError("整环凭证失效时必须抛错")
        except ProviderError as exc:
            assert exc.err_type == "credential_outage", exc.err_type
        assert TOUCHED == []
    finally:
        _restore(tracker, snap)
    print("test_gate_order_puts_ring_level_outage_before_per_node_disabled: PASSED")


# ------------------------------------------------- 4. 判定只有一份解读

def test_is_provider_healthy_reads_the_tracker_not_the_env():
    """直接改 tracker 集合（**不动环境变量**）也必须让可用性检查生效。

    修复前 `_is_provider_healthy` 每次现读 DISABLED_PROVIDERS 环境变量，而 CooldownTracker
    在构造/reset 时快照成集合——同一份配置被两处各自解读，进程内改配置时两者给出不同答案。
    """
    router, tracker, snap = _reset_router()
    try:
        assert srv._is_provider_healthy("glm") is True
        tracker.disabled_providers = {"glm"}
        assert srv._is_provider_healthy("glm") is False, "集合里的禁用必须被可用性检查看见"
        assert srv._is_provider_healthy("qwen") is True, "未被禁用的节点不得被误伤"
        assert os.getenv("DISABLED_PROVIDERS", "") == "", "本用例刻意不经环境变量"
    finally:
        _restore(tracker, snap)
    print("test_is_provider_healthy_reads_the_tracker_not_the_env: PASSED")


def test_is_disabled_merges_programmatic_set_and_env():
    """禁用判读必须是**并集**：程序集合 ∪ 环境变量现读。

    两个来源都要算数，且合并方向只能是「更保守」：程序集合承载运行时变更（测试、不经重启
    的配置变更），环境变量承载部署声明。既有套件（tests/test_affinity.py 的换节点用例）
    依赖「改 DISABLED_PROVIDERS 立刻生效」，因此现读环境变量这一旧行为必须保留——本用例
    把它显式锁住，避免日后又退回「只在构造时快照」而悄悄丢掉该能力。
    """
    tracker = srv.failover_router.cooldown_tracker
    snapshot = set(tracker.disabled_providers)
    try:
        tracker.disabled_providers = {"deepseek"}
        assert tracker.is_disabled("deepseek") is True, "程序集合必须生效"
        assert tracker.is_disabled("glm") is False

        os.environ["DISABLED_PROVIDERS"] = "glm"
        try:
            assert tracker.is_disabled("glm") is True, "环境变量现读必须立刻生效"
            assert tracker.is_disabled("deepseek") is True, "集合声明不得被环境变量抹掉"
            assert srv.failover_router.cooldown_tracker.disabled_gate_report()[
                "disabled_providers"
            ] == ["deepseek", "glm"], "诊断必须报告**生效**的并集，而非其中一个来源"
        finally:
            os.environ.pop("DISABLED_PROVIDERS", None)
        assert tracker.is_disabled("glm") is False, "环境变量清空后必须恢复"
    finally:
        tracker.disabled_providers = snapshot
    print("test_is_disabled_merges_programmatic_set_and_env: PASSED")


# ------------------------------------------------- 5. 可用性零回归

def test_short_cooldown_stays_dispatchable_through_the_gate():
    """OPT-021 的「最后一搏」语义不得被新闸门吃掉：短软冷却仍可派发。"""
    router, tracker, snap = _reset_router()
    try:
        _run(tracker.record_failure("kimi", 429, "rate limit exceeded"))
        gate = router.evaluate_dispatch_gate("kimi")
        assert gate["dispatchable"] is True
        assert gate["blocked_by"] == ""
        assert gate["long_avoidance"] is False

        _drain(router.execute_chat("kimi", "kimi", provider_factory=_make_fake(), chat_args={}))
        assert TOUCHED == ["kimi"], f"短冷却必须保留最后一搏，实际: {TOUCHED}"
    finally:
        _restore(tracker, snap)
    print("test_short_cooldown_stays_dispatchable_through_the_gate: PASSED")


# ------------------------------------------------- 6. 端到端

def test_e2e_all_disabled_returns_503_without_upstream_call():
    router, tracker, snap = _reset_router()
    original_make = srv._make_provider
    srv._make_provider = _make_fake()
    try:
        tracker.disabled_providers = set(ALL_PROVIDERS)
        client = TestClient(srv.app)
        r = client.post(
            "/v1/chat/completions",
            json={"model": "deepseek-chat", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 503, r.text
        assert r.json()["error"]["type"] == "disabled", r.text
        assert TOUCHED == [], f"端到端必须零上游触碰，实际: {TOUCHED}"

        r2 = client.post(
            "/v1/messages",
            json={
                "model": "claude-3-5-sonnet",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
            },
            headers={"anthropic-version": "2023-06-01"},
        )
        assert r2.status_code == 503, r2.text
        assert r2.json()["error"]["type"] == "overloaded_error", r2.text
        assert "DISABLED_PROVIDERS" in r2.json()["error"]["message"]
        assert TOUCHED == []
    finally:
        srv._make_provider = original_make
        _restore(tracker, snap)
    print("test_e2e_all_disabled_returns_503_without_upstream_call: PASSED")


def run_all():
    tests = [
        test_disabled_primary_is_redirected_with_zero_touch,
        test_whole_ring_disabled_fails_closed_with_zero_upstream_calls,
        test_audit_and_dispatch_agree_on_disabled_primary,
        test_audit_and_dispatch_agree_on_ring_wide_disabled_err_type,
        test_dispatch_decision_is_derived_from_the_single_gate_definition,
        test_gate_order_puts_ring_level_outage_before_per_node_disabled,
        test_is_provider_healthy_reads_the_tracker_not_the_env,
        test_is_disabled_merges_programmatic_set_and_env,
        test_short_cooldown_stays_dispatchable_through_the_gate,
        test_e2e_all_disabled_returns_503_without_upstream_call,
    ]
    for fn in tests:
        fn()
    print(f"\n✅ All dispatch-gate tests (OPT-033) passed! ({len(tests)} cases)")


if __name__ == "__main__":
    run_all()
