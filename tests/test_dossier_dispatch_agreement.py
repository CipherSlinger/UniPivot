"""风控档案与派发事实的一致性 (OPT-035) 测试套件

动机与 OPT-031 同源、只是换了一格：档案（OPT-023）的职责之一是回答「这个节点现在能不被
派发」，而派发边界才是唯一的事实来源。OPT-031 发现「长效避让」在派发边界已被硬阻断、
档案却报「冷却中（可派发）」，于是补了 `long_avoidance` 一档。**「运维声明出局」
(`DISABLED_PROVIDERS`) 这一格被同样漏掉了**：派发边界对它硬阻断（OPT-033），档案却报
「正常（无近期风控记录）」且 `dispatchable=True`。

实测复现（修复前，`DISABLED_PROVIDERS=qwen`、无租约无冷却）：

    dispatch gate  : {'dispatchable': False, 'blocked_by': 'disabled'}
    dossier posture: clear          ← 档案在派发事实上说了谎

运维据此会误判一个已被声明出局的节点仍可用。本套件锁定修复，并把「档案与派发事实一致」
写成一条**对全部逐节点状态逐一验证**的性质，而不是只验这一个格子——这样下一格再被漏掉时
会直接变红，而不是等运维实测发现。

覆盖：
1. 声明出局的节点：姿态 disabled、不可派发、研判 disabled、且**明确不主张重登**
2. **一致性性质**：逐节点状态下 `档案 dispatchable == 派发闸门 dispatchable`
3. 唯一的**刻意例外**（环级凭证失效）被显式锁住，而不是被隐瞒
4. 排序理由：既被封禁又被声明出局的节点先报「封禁」（危险者优先），
   但「出局」仍作为条目标记可见——避免运维看到「出局」就去解除禁用、
   把一个被封禁的节点放回派发路径
5. 档案是只读的：查询姿态与条目绝不改动避让状态
6. 零回归：健康 / 短冷却 / 长效避让 / 租约各自的既有姿态与可派发性逐值不变

运行: .venv/bin/python tests/test_dossier_dispatch_agreement.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["TESTING"] = "1"
os.environ["GATEWAY_DISABLE_AUTO_REFRESH"] = "1"
os.environ["GATEWAY_DISABLE_PACING"] = "1"
os.environ["GATEWAY_DISABLE_SINGLEFLIGHT"] = "1"
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
os.environ["QWEN_AUTH_TOKEN"] = "fake_qwen"
os.environ["DEEPSEEK_AUTH_TOKEN"] = "fake_ds"
os.environ["DOUBAO_AUTH_TOKEN"] = "fake_doubao"
os.environ["KIMI_AUTH_TOKEN"] = "fake_kimi"
os.environ["GLM_AUTH_TOKEN"] = "fake_glm"

import server as srv  # noqa: E402
from providers.risk_dossier import (  # noqa: E402
    POSTURE_CLEAR,
    POSTURE_COOLING,
    POSTURE_DISABLED,
    POSTURE_DISPATCHABLE,
    POSTURE_LONG_AVOIDANCE,
    POSTURE_ORDER,
    POSTURE_QUARANTINED,
)

ALL = ("qwen", "deepseek", "doubao", "kimi", "glm")


def _run(coro):
    return asyncio.run(coro)


def _reset():
    router = srv.failover_router
    tracker = router.cooldown_tracker
    tracker.reset()
    srv.quarantine_store.clear_all_for_tests()
    # 台账是**进程级单例**：本套件必须比对服务端路由器的派发闸门（与档案共用同一个
    # tracker），因此不能像 test_risk_dossier 那样改用隔离对象，只能清空台账，
    # 否则前一个用例写入的拦截记录会让后一个用例的节点带上前一个用例的 watch 姿态。
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


def _restore(tracker, snapshot):
    tracker.disabled_providers = set(snapshot)
    tracker.reset()
    srv.quarantine_store.clear_all_for_tests()


def _agrees(router, dossier, pk) -> tuple:
    gate = router.evaluate_dispatch_gate(pk)
    entry = dossier.provider_entry(pk)
    return bool(gate["dispatchable"]), bool(entry["dispatchable"]), entry["posture"], gate["blocked_by"]


# ------------------------------------------------- 1. 声明出局的节点

def test_disabled_node_is_reported_as_not_dispatchable():
    """**本套件的直接靶点**：修复前档案报 clear / dispatchable=True。"""
    router, tracker, snap = _reset()
    try:
        tracker.disabled_providers = {"qwen"}
        entry = srv.risk_dossier.provider_entry("qwen")

        assert entry["posture"] == POSTURE_DISABLED, entry["posture"]
        assert entry["dispatchable"] is False
        assert entry["disabled"] is True, "逐节点标记必须能看见「已声明出局」这一事实"
        assert POSTURE_DISPATCHABLE[POSTURE_DISABLED] is False

        verdict = entry["avoidance"]
        assert verdict["verdict"] == "disabled"
        assert verdict["technically_bypassable"] is False
        assert verdict["recoverable_by_relogin"] is False, (
            "声明出局重登无效——主张重登正是本项目反复修复的误导运维缺陷"
        )
        # 措辞级防回归：绝不能把「执行 login.py」当成这个成因的解药
        assert "login.py" not in verdict["strategy"], verdict["strategy"]
        joined = " ".join(entry["recommended_actions"])
        assert "DISABLED_PROVIDERS" in joined
        assert "login.py" not in joined, entry["recommended_actions"]
    finally:
        _restore(tracker, snap)
    print("test_disabled_node_is_reported_as_not_dispatchable: PASSED")


# ------------------------------------------------- 2. 一致性性质

def test_dossier_dispatchability_matches_the_dispatch_gate_for_every_node_state():
    """**结构性性质**：对每一种逐节点状态，档案的可派发性必须等于派发闸门的判定。

    这是把 OPT-031/OPT-033/OPT-035 三次「档案说了谎」的修复收束成一条可回归的性质：
    下一格再被漏掉时，本用例直接变红，而不是等运维实测发现。
    """
    router, tracker, snap = _reset()
    try:
        dossier = srv.risk_dossier
        # 状态 1：健康
        for pk in ALL:
            g, d, posture, blocked = _agrees(router, dossier, pk)
            assert g == d, f"{pk} 健康态档案与派发不一致: gate={g} dossier={d} ({posture})"

        # 状态 2：短软冷却（派发保留「最后一搏」→ 一致为可派发）
        _run(tracker.record_failure("kimi", 429, "rate limit exceeded"))
        g, d, posture, blocked = _agrees(router, dossier, "kimi")
        assert g is True and d is True, f"短冷却双方都应可派发: gate={g} dossier={d} ({posture})"

        # 状态 3：升级后的长效避让（双方都不可派发）
        for _ in range(3):
            _run(tracker.record_failure("glm", 401, "401 Unauthorized: token expired"))
        assert tracker.is_in_long_avoidance("glm") is True, "前置条件：应已升级为长效避让"
        g, d, posture, blocked = _agrees(router, dossier, "glm")
        assert g is False, "长效避让在派发边界是硬阻断"
        assert d is False and posture == POSTURE_LONG_AVOIDANCE, (d, posture)

        # 状态 4：运维封禁租约（双方都不可派发）
        srv.quarantine_store.set_lease("deepseek", duration_seconds=3600, source="operator")
        g, d, posture, blocked = _agrees(router, dossier, "deepseek")
        assert g is False and d is False and posture == POSTURE_QUARANTINED, (g, d, posture)

        # 状态 5：运维声明出局（**本轮修复的格子**）
        tracker.disabled_providers = {"doubao"}
        g, d, posture, blocked = _agrees(router, dossier, "doubao")
        assert g is False and d is False and posture == POSTURE_DISABLED, (g, d, posture)
        assert blocked == "disabled"
    finally:
        _restore(tracker, snap)
    print("test_dossier_dispatchability_matches_the_dispatch_gate_for_every_node_state: PASSED")


def test_ring_wide_credential_outage_is_the_documented_exception():
    """唯一的刻意例外：环级凭证失效是**环级**事实，而档案是**逐节点**姿态。

    把例外显式锁住而不是隐瞒：档案在此报 clear（节点自身确实没有风控记录），
    派发闸门报不可派发（整环没有可用凭证）。承认这个边界并测试它，比让它日后
    被当成 bug 或（更糟）被误当成「档案又说了谎」而乱改更安全。
    """
    router, tracker, snap = _reset()
    try:
        for pk in ("qwen", "doubao", "kimi"):
            _run(tracker.record_failure(pk, 401, "401 Unauthorized: token expired"))
        assert tracker.credential_watch.is_outage_active() is True

        gate = router.evaluate_dispatch_gate("glm")
        entry = srv.risk_dossier.provider_entry("glm")
        assert gate["dispatchable"] is False
        assert gate["blocked_by"] == "credential_outage"
        assert entry["posture"] == POSTURE_CLEAR
        assert entry["dispatchable"] is True
        assert gate["dispatchable"] != entry["dispatchable"], (
            "该例外若被消除，请同步更新本用例与档案的边界说明，而不是让它静默改变"
        )
    finally:
        _restore(tracker, snap)
    print("test_ring_wide_credential_outage_is_the_documented_exception: PASSED")


# ------------------------------------------------- 3. 排序理由

def test_ban_outranks_disable_but_disable_stays_visible():
    """既被封禁又被声明出局的节点（本机 deepseek 的真实状态）必须先报「封禁」。

    若先报「出局」，运维最自然的反应是「那就解除禁用」，从而把一个**被封禁**的节点
    重新放回派发路径——静默且有害。先报「封禁」给出正确动作（等待解封），
    而「出局」作为条目标记仍然可见。
    """
    router, tracker, snap = _reset()
    try:
        srv.quarantine_store.set_lease("deepseek", duration_seconds=3600, source="operator")
        tracker.disabled_providers = {"deepseek"}
        entry = srv.risk_dossier.provider_entry("deepseek")

        assert entry["posture"] == POSTURE_QUARANTINED, entry["posture"]
        assert entry["disabled"] is True, "被更重成因掩盖的「出局」事实必须仍然可见"
        assert entry["avoidance"]["verdict"] == "ban", "研判须为封禁（重登无效）"
        # 危险者优先：封禁必须排在出局之前
        assert POSTURE_ORDER.index(POSTURE_QUARANTINED) < POSTURE_ORDER.index(POSTURE_DISABLED)
    finally:
        _restore(tracker, snap)
    print("test_ban_outranks_disable_but_disable_stays_visible: PASSED")


# ------------------------------------------------- 4. 只读

def test_dossier_query_is_read_only():
    """查询档案绝不改动避让状态（OPT-023 的只读契约）。"""
    router, tracker, snap = _reset()
    try:
        tracker.disabled_providers = {"qwen"}
        _run(tracker.record_failure("kimi", 429, "rate limit exceeded"))
        srv.quarantine_store.set_lease("glm", duration_seconds=3600, source="operator")

        # 只读性质的正确断言是**绝对状态未变**：`get_remaining_cooldown()` /
        # `quarantine.remaining()` 返回的是 `到期时间戳 - now`，会随墙钟毫秒级递减，
        # 直接对它们做相等比较本身就是不稳的（那是测试的缺陷，而非真的被改动）。
        # 因此冷却比**绝对到期时间戳**、租约比 `list_active()` 的 `until`。
        before = {
            "kimi_until": tracker._cooldowns.get("kimi", 0.0),
            "kimi_remaining": tracker.get_remaining_cooldown("kimi"),
            "glm_until": srv.quarantine_store.list_active().get("glm", {}).get("until"),
            "qwen_disabled": tracker.is_disabled("qwen"),
        }
        dossier = srv.risk_dossier
        dossier.build()
        for pk in ALL:
            dossier.provider_entry(pk)
            dossier.derive_posture(pk)

        assert tracker._cooldowns.get("kimi", 0.0) == before["kimi_until"], "查询不得复位冷却"
        # 剩余量只应随时钟递减，绝不能被延长（延长即读路径写了状态）
        assert tracker.get_remaining_cooldown("kimi") <= before["kimi_remaining"] + 1e-6, (
            "查询不得延长冷却"
        )
        assert tracker.get_remaining_cooldown("kimi") > 0, "查询不得解除冷却"
        assert (
            srv.quarantine_store.list_active().get("glm", {}).get("until") == before["glm_until"]
        ), "查询不得改动租约"
        assert tracker.is_disabled("qwen") is before["qwen_disabled"]
        assert tracker.is_in_cooldown("glm") is True, "查询不得解除租约"
    finally:
        _restore(tracker, snap)
    print("test_dossier_query_is_read_only: PASSED")


# ------------------------------------------------- 5. 零回归

def test_healthy_and_cooling_nodes_keep_their_existing_postures():
    router, tracker, snap = _reset()
    try:
        dossier = srv.risk_dossier
        assert dossier.provider_entry("qwen")["posture"] == POSTURE_CLEAR

        _run(tracker.record_failure("kimi", 429, "rate limit exceeded"))
        ent = dossier.provider_entry("kimi")
        assert ent["posture"] == POSTURE_COOLING
        assert ent["dispatchable"] is True, "短冷却必须保留 OPT-021 的最后一搏语义"
        assert ent["disabled"] is False
    finally:
        _restore(tracker, snap)
    print("test_healthy_and_cooling_nodes_keep_their_existing_postures: PASSED")


def run_all():
    tests = [
        test_disabled_node_is_reported_as_not_dispatchable,
        test_dossier_dispatchability_matches_the_dispatch_gate_for_every_node_state,
        test_ring_wide_credential_outage_is_the_documented_exception,
        test_ban_outranks_disable_but_disable_stays_visible,
        test_dossier_query_is_read_only,
        test_healthy_and_cooling_nodes_keep_their_existing_postures,
    ]
    for fn in tests:
        fn()
    print(f"\n✅ All dossier/dispatch agreement tests (OPT-035) passed! ({len(tests)} cases)")


if __name__ == "__main__":
    run_all()
