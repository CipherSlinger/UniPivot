"""长效避让的派发硬阻断与整环零上游快速失败 (OPT-031) 测试套件

承 OPT-021（运维租约硬阻断）与 OPT-025（整环凭证失效快速失败）往下推。

实证的空白：派发路径上只有两个闸口——
  * OPT-021：`is_quarantined` —— 只拦**人工/自动租约**；
  * OPT-025：`credential_watch.is_outage_active` —— 只拦**整环 401**。
两者之间留下一整片未覆盖的地带：**整环混合成因的长效避让**。当五个节点都已被
各自的守护机制判定为长期不可用（有人是 waf 升级、有人是 auth 升级，时长小时~天级），
每条客户端请求仍会把这五个节点**逐个打满**，产出若干次新的拦截。这正是 OPT-025
记录过的「无效凭证风暴」形状——风暴本身就是风控触发器——只是成因从 401 换成了
waf/auth 升级，且此刻敲哪个节点都拿不到正常返回。

关键的设计张力（本套件的核心断言）：OPT-021 **刻意**让短软冷却保持 demote-only，
理由写在 `get_fallback_candidates` 的文档里——「环内全部冷却时允许最后一搏，以保
可用性」。因此不能粗暴地改成「全部冷却即失败」。区分二者的**唯一**正当依据是时长：
  * 短冷却（频控退避，秒级~两分钟）值得一次最后一搏；
  * 长效避让（升级后的长冷却，小时~天级）不值得——上游已用**连续多次**拦截作答。
阈值必须落在「短冷却可达上界」与「升级下界」之间的**真空带**内，本套件用
`test_threshold_sits_in_the_gap` 把这条真空带**证明**出来，而不是假设它存在。

本套件验证：
1. 阈值落在真空带内：任何短冷却都不可能越界（OPT-021 语义逐字节不变）
2. 升级后的 waf/auth 避让一律被判定为长效；单次拦截不算
3. 后备候选**排除**长效避让节点，而**保留**短冷却节点（仅沉底）
4. 整环长效避让 → `503 / ring_exhausted`，且**零上游调用**
5. 环内仍有健康节点 → 改派，且长效避让节点的冷却**不被延长**（真正的零触碰）
6. 环内全是短冷却 → 仍按旧语义派出最后一搏（可用性零回归）
7. 开关 `GATEWAY_DISABLE_LONG_AVOIDANCE_FAST_FAIL=1` 逐字回退旧行为
8. 错误类型映射到 Anthropic 的 overloaded_error
9. 档案姿态对长效避让 fail-closed（不可派发），短冷却仍可派发
10. 档案建议按成因分支：waf/auth 指向人工重登，且不得出现「无需人工介入」
11. 租约成因必须报 `ban`（重登无效），不得沿用软冷却的成因误导运维
12. 遥测与诊断暴露开关、阈值、门控统计；reset 后零串扰
13. 审计视图可读回被派发路径排除的节点（放行开关绝不参与真实派发决策）

运行: .venv/bin/python tests/test_long_avoidance.py
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
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
    os.environ.pop(_name, None)
os.environ["GATEWAY_PROBATION_MAX_RENEWALS"] = "0"
os.environ["QWEN_AUTH_TOKEN"] = "fake_qwen"
os.environ["DEEPSEEK_AUTH_TOKEN"] = "fake_ds"
os.environ["DOUBAO_AUTH_TOKEN"] = "fake_doubao"
os.environ["KIMI_AUTH_TOKEN"] = "fake_kimi"
os.environ["GLM_AUTH_TOKEN"] = "fake_glm"

import server as srv  # noqa: E402
from providers.base import ProviderError  # noqa: E402
from providers.quarantine import QuarantineStore  # noqa: E402
from providers.risk_dossier import (  # noqa: E402
    POSTURE_COOLING,
    POSTURE_LONG_AVOIDANCE,
    POSTURE_QUARANTINED,
    RiskDossier,
)
from providers.risk_ledger import RiskLedger  # noqa: E402
from providers import failover as fov  # noqa: E402

# FLAGSHIP 环（主选 qwen/Qwen 所属）为五节点全环；SPEED 环只有三个节点。
FLAGSHIP_RING = ("qwen", "glm", "kimi", "doubao", "deepseek")

TOUCHED: list[str] = []


def _make_fake():
    """假 Provider 工厂：任何进入 chat() 的节点都会被记入 TOUCHED（即「被触碰」）。"""

    def factory(key: str):
        class _Fake:
            async def chat(self, *a, **kw):
                TOUCHED.append(key)
                yield ("hi", {})

            async def delete_conversation(self, *a, **kw):
                return None

        return _Fake()

    return factory


def _run(coro):
    return asyncio.run(coro)


def _clear_shared_evidence(tracker):
    """清空**进程级单例**的风控证据。

    这几处是共享单例（凭证失效检测、凭证续期检测），不随 tracker.reset() 复位。
    不清空会让前面的用例把 auth 证据喂给后面的用例——实测后果：整环用例里的两条
    401 与前序用例的 401 合起来凑够 3 个不同节点，OPT-025 的「整环凭证失效」抢在
    OPT-031 之前触发，err_type 变成 credential_outage。这正是本项目已记录过的
    「跨用例污染」缺陷类型，因此每个用例前后都必须清。
    """
    try:
        tracker.credential_watch.clear_all_for_tests()
    except Exception:  # noqa: BLE001
        pass
    try:
        from providers.credential_renewal import get_credential_renewal

        get_credential_renewal().clear_all_for_tests()
    except Exception:  # noqa: BLE001
        pass


def _tracker() -> fov.CooldownTracker:
    """全隔离的 tracker：独立租约库 + 清空进程级单例证据。"""
    store = QuarantineStore(directory=Path(tempfile.mkdtemp(prefix="opt031_q_")))
    t = fov.CooldownTracker()
    t.quarantine = store
    t.reset()
    _clear_shared_evidence(t)
    return t


def _reset_router():
    """复位共享路由的 tracker（用例前后各调一次，确保零串扰）。"""
    tracker = srv.failover_router.cooldown_tracker
    tracker.reset()
    _clear_shared_evidence(tracker)
    TOUCHED.clear()
    return tracker


def _teardown(tracker):
    tracker.reset()
    _clear_shared_evidence(tracker)
    shutil.rmtree(tracker.quarantine.directory, ignore_errors=True)


def _waf(tracker, pk, times=1):
    """走真实的人机验证升级路径（第 3 次起升级为长效避让）。"""
    for _ in range(times):
        _run(tracker.record_failure(pk, 403, "RGV587 人机验证失败，请完成滑块验证"))


def _auth(tracker, pk, times=1):
    for _ in range(times):
        _run(tracker.record_failure(pk, 401, "401 Unauthorized: token expired"))


def _ratelimit(tracker, pk, times=1):
    for _ in range(times):
        _run(tracker.record_failure(pk, 429, "rate limit exceeded"))


# ------------------------------------------------------------------ 1. 真空带

def test_threshold_sits_in_the_gap():
    """阈值必须落在「短冷却可达上界」与「升级下界」之间的真空带内，且夹逼强制生效。

    这不是一个调出来的经验值：短冷却（频控与人机验证）被 `GATEWAY_WAF_COOLDOWN_MAX`
    （默认 900s）封顶；而升级后的下界是认证级长效底数 `GATEWAY_AUTH_COOLDOWN_BASE`
    （默认 1800s）。阈值必须**严格**落在这段真空里，OPT-021 的「最后一搏」才逐字节不变。
    这里同时证明夹逼两端都挡得住误配：配成 1 不得侵入短冷却，配成天文数字不得越过
    升级下界（否则升级后的节点会重新被派发，风暴复现）。
    """
    tracker = _tracker()
    short_cap = float(os.getenv("GATEWAY_WAF_COOLDOWN_MAX", "900"))
    escal_floor = float(os.getenv("GATEWAY_AUTH_COOLDOWN_BASE", "1800"))
    try:
        threshold = tracker.long_avoidance_threshold()
        assert short_cap < threshold < escal_floor, (
            f"阈值 {threshold} 必须严格落在真空带 ({short_cap}, {escal_floor}) 内"
        )

        # 频控指数封顶后的可达上界也必须落在阈值之下（真实公式推导，不写死数字）
        ratelimit_upper = max(
            meta.default_cooldown * (1.2 ** 3) for meta in fov.PROVIDER_RISK_SPECS.values()
        )
        assert ratelimit_upper < threshold, f"阈值不得低于频控可达上界 {ratelimit_upper}"

        # 夹逼：误配下限
        os.environ["GATEWAY_LONG_AVOIDANCE_THRESHOLD"] = "1"
        try:
            assert tracker.long_avoidance_threshold() == short_cap + 1.0, "误配必须被下限拦住"
        finally:
            os.environ.pop("GATEWAY_LONG_AVOIDANCE_THRESHOLD", None)

        # 夹逼：误配上限（越过升级下界会让风暴复现）
        os.environ["GATEWAY_LONG_AVOIDANCE_THRESHOLD"] = "999999"
        try:
            assert tracker.long_avoidance_threshold() == escal_floor - 1.0, "误配必须被上限拦住"
        finally:
            os.environ.pop("GATEWAY_LONG_AVOIDANCE_THRESHOLD", None)

        os.environ["GATEWAY_LONG_AVOIDANCE_THRESHOLD"] = "not-a-number"
        try:
            assert tracker.long_avoidance_threshold() == 1200.0, "非法值必须退回默认"
        finally:
            os.environ.pop("GATEWAY_LONG_AVOIDANCE_THRESHOLD", None)
    finally:
        _teardown(tracker)
    print("test_threshold_sits_in_the_gap: PASSED")


def test_short_cooldown_is_never_long_avoidance():
    """频控无论被触发多少次，都不得被判定为长效避让（否则可用性会被本优化吃掉）。"""
    tracker = _tracker()
    try:
        _ratelimit(tracker, "glm", 12)
        cd = tracker.get_remaining_cooldown("glm")
        assert 0 < cd < 900.0, f"频控冷却必须仍有界, 实际 {cd}"
        assert tracker.is_in_cooldown("glm") is True
        assert tracker.is_in_long_avoidance("glm") is False, "短冷却绝不能被当成长效避让"
    finally:
        _teardown(tracker)
    print("test_short_cooldown_is_never_long_avoidance: PASSED")


# ------------------------------------------------------------------ 2. 升级判定

def test_escalated_avoidance_is_long_and_single_hit_is_not():
    """单次拦截不算长效；重复升级后的 waf/auth 一律算长效。"""
    tracker = _tracker()
    try:
        _waf(tracker, "qwen")
        assert tracker.is_in_long_avoidance("qwen") is False, "单次人机验证不得升级"

        _waf(tracker, "qwen", 2)
        assert tracker.get_remaining_cooldown("qwen") >= tracker.long_avoidance_threshold()
        assert tracker.is_in_long_avoidance("qwen") is True
        assert tracker.long_avoidance_cause("qwen") == "waf"

        _auth(tracker, "kimi", 3)
        assert tracker.is_in_long_avoidance("kimi") is True
        assert tracker.long_avoidance_cause("kimi") == "auth"

        # 未被触碰的节点不得被误伤
        assert tracker.is_in_long_avoidance("doubao") is False
    finally:
        _teardown(tracker)
    print("test_escalated_avoidance_is_long_and_single_hit_is_not: PASSED")


# ------------------------------------------------------------------ 3. 候选排序

def test_candidates_exclude_long_avoidance_but_keep_short_cooling():
    """长效避让节点必须被**排除**（而非沉底），短冷却节点必须**保留**。"""
    tracker = _tracker()
    router = srv.failover_router
    original = router.cooldown_tracker
    router.cooldown_tracker = tracker
    try:
        _waf(tracker, "deepseek", 3)          # 长效避让
        _ratelimit(tracker, "kimi")           # 短冷却

        cands = router.get_fallback_candidates("qwen", "Qwen")
        keys = [c.provider_key for c in cands]
        assert "deepseek" not in keys, "长效避让节点绝不得出现在后备候选中"
        assert "kimi" in keys, "短冷却节点必须保留（OPT-021 可用性语义）"
        assert cands[0].provider_key != "kimi", "短冷却节点必须被沉底而非置顶"
    finally:
        router.cooldown_tracker = original
        _teardown(tracker)
    print("test_candidates_exclude_long_avoidance_but_keep_short_cooling: PASSED")


# ------------------------------------------------------------------ 4. 整环快速失败

def test_whole_ring_long_avoidance_fails_closed_with_zero_upstream_calls():
    """整环长效避让 → 503 / ring_exhausted，零上游调用，并给出分成因的处置指引。"""
    tracker = _reset_router()
    try:
        # 混合成因：三人机验证升级、两人认证升级
        for pk in ("qwen", "glm", "kimi"):
            _waf(tracker, pk, 3)
        for pk in ("doubao", "deepseek"):
            _auth(tracker, pk, 3)

        raised = None
        try:
            _run(
                srv.failover_router.execute_chat(
                    primary_provider_key="qwen",
                    primary_wire_model="Qwen",
                    provider_factory=_make_fake(),
                    chat_args={"messages": []},
                )
            )
        except ProviderError as exc:
            raised = exc

        assert raised is not None, "整环长效避让时必须快速失败，而非逐个走遍五个节点"
        assert raised.status == 503
        assert raised.err_type == "ring_exhausted"
        assert "长效避让" in raised.message
        assert "login.py" in raised.message, "错误信息必须给出人工处置指引"
        assert TOUCHED == [], f"必须零上游触碰（这正是风暴的成因），实际: {TOUCHED}"

        st = tracker.long_avoidance_report()["stats"]
        assert st["fast_fails"] == 1
        assert st["last_fast_fail_cause"] in ("waf", "auth")
    finally:
        tracker.reset()
    print("test_whole_ring_long_avoidance_fails_closed_with_zero_upstream_calls: PASSED")


def test_speed_ring_has_its_own_boundary():
    """能力环边界按环计算：SPEED 环只有三个节点，三者全避让即快速失败。"""
    tracker = _reset_router()
    try:
        for pk in ("qwen", "doubao", "glm"):
            _waf(tracker, pk, 3)
        raised = None
        try:
            _run(
                srv.failover_router.execute_chat(
                    primary_provider_key="qwen",
                    primary_wire_model="Qwen3.6-Flash",
                    provider_factory=_make_fake(),
                    chat_args={"messages": []},
                )
            )
        except ProviderError as exc:
            raised = exc
        assert raised is not None and raised.err_type == "ring_exhausted"
        assert TOUCHED == []
        # deepseek/kimi 不属于 SPEED 环，因此它们是否健康与本环无关
        assert "deepseek" in raised.message or "doubao" in raised.message
    finally:
        tracker.reset()
    print("test_speed_ring_has_its_own_boundary: PASSED")


# ------------------------------------------------------------------ 5. 改派零触碰

def test_redirect_leaves_avoiding_nodes_untouched():
    """环内仍有健康节点 → 改派到它，且长效避让节点的冷却**不得被延长**。"""
    tracker = _reset_router()
    try:
        avoiding = ("qwen", "glm", "kimi", "doubao")
        for pk in avoiding:
            _waf(tracker, pk, 3)
        before = {pk: tracker.get_remaining_cooldown(pk) for pk in avoiding}

        stream, _event, active = _run(
            srv.failover_router.execute_chat(
                primary_provider_key="qwen",
                primary_wire_model="Qwen",
                provider_factory=_make_fake(),
                chat_args={"messages": []},
            )
        )
        assert TOUCHED == ["deepseek"], f"只允许触碰健康节点, 实际: {TOUCHED}"
        assert getattr(active, "__class__", None) is not None
        assert tracker.long_avoidance_report()["stats"]["redirects"] >= 1

        # 真正的零触碰：避让节点的剩余冷却只允许**衰减**，绝不允许增长
        for pk in avoiding:
            after = tracker.get_remaining_cooldown(pk)
            assert after <= before[pk] + 1e-6, f"[{pk}] 改派路径反而延长了避让: {before[pk]} -> {after}"
            assert tracker.is_in_long_avoidance(pk) is True, "改派不得解除既有长效避让"
    finally:
        tracker.reset()
    print("test_redirect_leaves_avoiding_nodes_untouched: PASSED")


def test_health_monitor_would_have_touched_all_five_before_this_fix():
    """反例：在没有本机制的世界里，同一场景会把五个节点全部打满。

    用 `is_in_long_avoidance` 关闭后的候选列表来证明这一点——候选里重新出现
    全部五个节点，正是风暴的形状。
    """
    tracker = srv.failover_router.cooldown_tracker
    tracker.reset()
    try:
        for pk in FLAGSHIP_RING:
            _waf(tracker, pk, 3)
        os.environ["GATEWAY_DISABLE_LONG_AVOIDANCE_FAST_FAIL"] = "1"
        try:
            keys = [c.provider_key for c in srv.failover_router.get_fallback_candidates("qwen", "Qwen")]
            assert "deepseek" in keys and "kimi" in keys, "关闭开关后应恢复「逐个走遍」的旧行为"
        finally:
            os.environ.pop("GATEWAY_DISABLE_LONG_AVOIDANCE_FAST_FAIL", None)
    finally:
        tracker.reset()
    print("test_health_monitor_would_have_touched_all_five_before_this_fix: PASSED")


# ------------------------------------------------------------------ 6. 可用性零回归

def test_short_cooling_ring_still_makes_last_resort_attempt():
    """环内全是**短**冷却时仍必须派出最后一搏——这是 OPT-021 刻意的可用性保底。"""
    tracker = _reset_router()
    try:
        for pk in FLAGSHIP_RING:
            _ratelimit(tracker, pk, 6)
        for pk in FLAGSHIP_RING:
            assert tracker.is_in_cooldown(pk) is True
            assert tracker.is_in_long_avoidance(pk) is False

        stream, _event, _active = _run(
            srv.failover_router.execute_chat(
                primary_provider_key="qwen",
                primary_wire_model="Qwen",
                provider_factory=_make_fake(),
                chat_args={"messages": []},
            )
        )
        assert TOUCHED == ["qwen"], f"主选短冷却时仍应照常派出最后一搏, 实际: {TOUCHED}"
    finally:
        tracker.reset()
    print("test_short_cooling_ring_still_makes_last_resort_attempt: PASSED")


def test_disable_switch_restores_legacy_behaviour():
    """总开关必须逐字回退到「逐个走遍能力环」的旧行为。"""
    tracker = _reset_router()
    os.environ["GATEWAY_DISABLE_LONG_AVOIDANCE_FAST_FAIL"] = "1"
    try:
        for pk in FLAGSHIP_RING:
            _waf(tracker, pk, 3)
        assert tracker.is_in_long_avoidance("qwen") is False, "关闭后判定应恒为 False"
        raised = None
        try:
            _run(
                srv.failover_router.execute_chat(
                    primary_provider_key="qwen",
                    primary_wire_model="Qwen",
                    provider_factory=_make_fake(),
                    chat_args={"messages": []},
                )
            )
        except ProviderError as exc:
            raised = exc
        assert raised is None, "关闭开关后不得再快速失败"
        assert TOUCHED == ["qwen"], "关闭开关后应恢复旧行为（派出主选）"
    finally:
        os.environ.pop("GATEWAY_DISABLE_LONG_AVOIDANCE_FAST_FAIL", None)
        tracker.reset()
    print("test_disable_switch_restores_legacy_behaviour: PASSED")


# ------------------------------------------------------------------ 7. 协议映射

def test_error_type_maps_to_anthropic_overloaded():
    """Anthropic 协议下 ring_exhausted 必须映射为 overloaded_error（可重试语义）。"""
    from providers.anthropic_compat import map_anthropic_error_type

    assert map_anthropic_error_type(503, "ring_exhausted") == "overloaded_error"
    # 不得与「凭证失效」混同：后者映射为 authentication_error
    assert map_anthropic_error_type(503, "credential_outage") == "authentication_error"
    print("test_error_type_maps_to_anthropic_overloaded: PASSED")


# ------------------------------------------------------------------ 8. 档案姿态

def _isolated_dossier():
    tmp = Path(tempfile.mkdtemp(prefix="opt031_dossier_"))
    ledger = RiskLedger(directory=tmp)
    store = QuarantineStore(directory=tmp)
    tracker = fov.CooldownTracker()
    tracker.quarantine = store
    tracker.risk_ledger = ledger
    tracker.reset()
    dossier = RiskDossier(ledger=ledger, quarantine=store, cooldown_tracker=tracker)
    return dossier, store, tracker, tmp


def test_dossier_posture_is_fail_closed():
    """档案必须对长效避让 fail-closed：不可派发；短冷却仍可派发。"""
    dossier, store, tracker, tmp = _isolated_dossier()
    try:
        assert dossier.derive_posture("qwen") != POSTURE_LONG_AVOIDANCE

        _ratelimit(tracker, "glm", 3)
        assert dossier.derive_posture("glm") == POSTURE_COOLING
        assert dossier.provider_entry("glm")["dispatchable"] is True, "短冷却必须仍可派发"

        _waf(tracker, "qwen", 3)
        assert dossier.derive_posture("qwen") == POSTURE_LONG_AVOIDANCE
        assert dossier.provider_entry("qwen")["dispatchable"] is False, (
            "长效避让已被派发边界硬阻断，档案不得声称仍可派发"
        )

        # 租约优先于长效避让（姿态取最重者）
        store.set_lease("kimi", duration_seconds=3600)
        assert dossier.derive_posture("kimi") == POSTURE_QUARANTINED
        assert dossier.provider_entry("kimi")["dispatchable"] is False
    finally:
        tracker.reset()
        shutil.rmtree(tmp, ignore_errors=True)
    print("test_dossier_posture_is_fail_closed: PASSED")


def test_dossier_advice_branches_on_cause():
    """长效避让的建议必须指向人工重登，且不得出现「无需人工介入」。"""
    dossier, _store, tracker, tmp = _isolated_dossier()
    try:
        _waf(tracker, "qwen", 3)
        entry = dossier.provider_entry("qwen")
        actions = dossier.recommended_actions("qwen", POSTURE_LONG_AVOIDANCE, entry)
        joined = " ".join(actions)
        assert "login.py qwen" in joined, f"人机验证升级必须指向人工重登: {actions}"
        assert "无需人工介入" not in joined, "升级后的长效避让绝不能建议等待"

        _auth(tracker, "kimi", 3)
        entry = dossier.provider_entry("kimi")
        actions = dossier.recommended_actions("kimi", POSTURE_LONG_AVOIDANCE, entry)
        joined = " ".join(actions)
        assert "login.py kimi" in joined
        assert "无需人工介入" not in joined
    finally:
        tracker.reset()
        shutil.rmtree(tmp, ignore_errors=True)
    print("test_dossier_advice_branches_on_cause: PASSED")


# ------------------------------------------------------------------ 9. 成因不误导

def test_lease_cause_is_reported_as_ban():
    """租约（人工/自动）一律报 ban——重登无效，不得沿用软冷却成因误导运维。

    这一段是 OPT-027/OPT-029 反复修正过的同一类错误：把「重登有效」的成因
    贴到「重登无效」的节点上，运维就会照着错的建议去处理。
    """
    tracker = _tracker()
    try:
        # 先留下一条 auth 升级的软冷却成因，再叠加租约
        _auth(tracker, "deepseek", 3)
        assert tracker.cooldown_cause("deepseek") == "auth"
        assert tracker.long_avoidance_cause("deepseek") == "auth"

        tracker.quarantine.set_lease("deepseek", duration_seconds=72 * 3600)
        assert tracker.is_in_long_avoidance("deepseek") is True
        assert tracker.long_avoidance_cause("deepseek") == "ban", (
            "持有租约时长效避让成因必须是 ban（重登无效）"
        )
        # 软冷却成因访问器本身语义不变（它回答的是「这次冷却由什么触发」）
        assert tracker.cooldown_cause("deepseek") == "auth"
    finally:
        _teardown(tracker)
    print("test_lease_cause_is_reported_as_ban: PASSED")


# ------------------------------------------------------------------ 10. 遥测与零串扰

def test_report_shape_and_reset():
    """诊断报告的形状与开关、统计；reset 后必须零串扰。"""
    tracker = _tracker()
    try:
        report = tracker.long_avoidance_report()
        assert report["enabled"] is True
        assert report["threshold_seconds"] == 1200.0
        assert report["avoiding"] == {}
        assert set(report["stats"]) >= {"fast_fails", "redirects"}

        _waf(tracker, "glm", 3)
        snap = tracker.long_avoidance_report()["avoiding"]
        assert snap["glm"]["cause"] == "waf"
        assert snap["glm"]["remaining_hours"] > 0.0

        tracker.note_long_avoidance_fast_fail("glm", "waf")
        tracker.note_long_avoidance_redirect("glm", "qwen")
        stats = tracker.long_avoidance_report()["stats"]
        assert stats["fast_fails"] == 1
        assert stats["redirects"] == 1
        assert stats["last_redirect"] == "glm->qwen"

        tracker.reset()
        stats = tracker.long_avoidance_report()["stats"]
        assert stats["fast_fails"] == 0 and stats["redirects"] == 0, "reset 必须清零统计"
    finally:
        _teardown(tracker)
    print("test_report_shape_and_reset: PASSED")


def test_diagnostics_endpoint_exposes_long_avoidance():
    """诊断接口必须暴露该门控（运维要能看到「为什么整环都不派发」）。"""
    from fastapi.testclient import TestClient

    client = TestClient(srv.app)
    body = client.get("/v1/diagnostics").json()
    assert "long_avoidance" in body
    la = body["long_avoidance"]
    assert la["enabled"] is True
    assert "threshold_seconds" in la and "avoiding" in la and "stats" in la
    print("test_diagnostics_endpoint_exposes_long_avoidance: PASSED")


def test_audit_view_can_include_avoiding_nodes():
    """审计视图必须能读回被派发路径排除的节点：运维需要看到它此刻的冷却读数。

    两个放行开关都只服务于审计/诊断——派发路径一律不传，因此真实派发决策中
    租约与长效避让恒被排除。这条断言同时防止「顺手把审计开关当成派发开关」。
    """
    tracker = _reset_router()
    try:
        _waf(tracker, "glm", 3)
        assert tracker.is_in_long_avoidance("glm") is True

        keys = [c.provider_key for c in srv.failover_router.get_fallback_candidates("qwen", "Qwen")]
        assert "glm" not in keys, "长效避让节点不得出现在派发候选里"

        audit = srv.failover_router.get_fallback_candidates(
            "qwen", "Qwen", include_long_avoidance=True
        )
        audited = {c.provider_key: c for c in audit}
        assert "glm" in audited, "审计视图必须能读回该节点"
        assert audited["glm"].in_cooldown is True
        assert audited["glm"].remaining_cooldown >= tracker.long_avoidance_threshold()
    finally:
        tracker.reset()
    print("test_audit_view_can_include_avoiding_nodes: PASSED")


def test_e2e_whole_ring_long_avoidance_returns_503_without_upstream_call():
    """端到端（真实 ASGI 栈）：整环长效避让 → 503 且零上游调用。

    单元用例只断言 `execute_chat` 抛出的 ProviderError；这里补上「HTTP 层是否
    真的把它变成可读的 503」这一段——与 OPT-021 封禁硬阻断的端到端用例同型。
    最强的证据是 `TOUCHED == []`：在假工厂注入下，任何进入 `chat()` 的节点都会
    被记录，所以「零触碰」不是推断而是观测。
    """
    from fastapi.testclient import TestClient

    tracker = _reset_router()
    original = srv._make_provider
    srv._make_provider = _make_fake()
    try:
        # 混合成因：3 家人机验证升级 + 2 家认证升级，凑成整环长效避让
        for pk in ("qwen", "glm", "kimi"):
            _waf(tracker, pk, 3)
        for pk in ("doubao", "deepseek"):
            _auth(tracker, pk, 3)
        for pk in FLAGSHIP_RING:
            assert tracker.is_in_long_avoidance(pk) is True, f"{pk} 应处于长效避让"

        client = TestClient(srv.app)
        r = client.post(
            "/v1/chat/completions",
            json={"model": "deepseek-chat", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 503, r.text
        err = r.json()["error"]
        assert err["type"] == "ring_exhausted", r.text
        assert "长效避让" in err["message"] and "login.py" in err["message"], err["message"]
        assert TOUCHED == [], f"端到端必须零上游触碰，实际: {TOUCHED}"

        # Anthropic 协议必须映射为 overloaded_error（与 quarantined 同源）
        r2 = client.post(
            "/v1/messages",
            json={"model": "claude-3-5-sonnet", "max_tokens": 16,
                  "messages": [{"role": "user", "content": "hi"}]},
            headers={"anthropic-version": "2023-06-01"},
        )
        assert r2.status_code == 503, r2.text
        assert r2.json()["error"]["type"] == "overloaded_error", r2.text
        assert TOUCHED == []

        stats = tracker.long_avoidance_report()["stats"]
        assert stats["fast_fails"] >= 2, stats
    finally:
        srv._make_provider = original
        tracker.reset()
        _clear_shared_evidence(tracker)
    print("test_e2e_whole_ring_long_avoidance_returns_503_without_upstream_call: PASSED")


# ------------------------------------------------------------------ 运行

def run_all():
    tests = [
        test_threshold_sits_in_the_gap,
        test_short_cooldown_is_never_long_avoidance,
        test_escalated_avoidance_is_long_and_single_hit_is_not,
        test_candidates_exclude_long_avoidance_but_keep_short_cooling,
        test_whole_ring_long_avoidance_fails_closed_with_zero_upstream_calls,
        test_speed_ring_has_its_own_boundary,
        test_redirect_leaves_avoiding_nodes_untouched,
        test_health_monitor_would_have_touched_all_five_before_this_fix,
        test_short_cooling_ring_still_makes_last_resort_attempt,
        test_disable_switch_restores_legacy_behaviour,
        test_error_type_maps_to_anthropic_overloaded,
        test_dossier_posture_is_fail_closed,
        test_dossier_advice_branches_on_cause,
        test_lease_cause_is_reported_as_ban,
        test_report_shape_and_reset,
        test_diagnostics_endpoint_exposes_long_avoidance,
        test_audit_view_can_include_avoiding_nodes,
        test_e2e_whole_ring_long_avoidance_returns_503_without_upstream_call,
    ]
    for fn in tests:
        fn()
    print(f"\n✅ All long-avoidance tests (OPT-031) passed! ({len(tests)} cases)")


if __name__ == "__main__":
    run_all()
