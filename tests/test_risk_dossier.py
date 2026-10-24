"""风控拦截全量档案与规避策略研判 (OPT-023) 测试套件

动机：风控台账的 `recent_interceptions()` 是**滑动窗口**（默认 24h），适合驱动实时
避让，却无法回答运维最关心的问题——「这家到底被拦过几次、都是什么原因、能不能绕开、
重登有没有用」。各家封禁窗口以「天」计，24h 窗口几乎必然在封禁解除前就把证据丢光。

本套件验证 `providers/risk_dossier.py`：
1. 全量聚合：远超 24h 窗口的历史拦截仍被计入，含首/末次时间与严重度分布
2. 姿态派生 fail-closed：封禁 > 冷却 > 观察期 > 监视 > 正常
3. 可派发性：仅硬封禁不可派发，软冷却保留「最后一搏」语义
4. 规避研判：按主要严重度给出确定性结论，租约一律按账号级封禁研判
5. 重登恢复结论正确：auth/waf 可恢复，ratelimit 与账号级封禁不可恢复
6. 推荐动作可执行且与姿态一致
7. 只读性：查询档案不得改动任何风控状态
8. 开关 GATEWAY_DISABLE_RISK_DOSSIER
9. 遥测：/v1/diagnostics 暴露汇总，GET /v1/admin/risk 暴露明细与单节点视图
10. 台账读取异常时降级而非中断（零回归）

运行: .venv/bin/python tests/test_risk_dossier.py
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["TESTING"] = "1"
os.environ["GATEWAY_DISABLE_AUTO_REFRESH"] = "1"
os.environ["GATEWAY_DISABLE_PACING"] = "1"
os.environ["GATEWAY_PROBATION_WINDOW"] = "300"
os.environ["GATEWAY_PROBATION_MIN_SUCCESSES"] = "3"
os.environ.pop("GATEWAY_DISABLE_RISK_LEDGER", None)
os.environ.pop("GATEWAY_DISABLE_RISK_DOSSIER", None)
os.environ["QWEN_AUTH_TOKEN"] = "fake_qwen"
os.environ["DEEPSEEK_AUTH_TOKEN"] = "fake_ds"
os.environ["DOUBAO_AUTH_TOKEN"] = "fake_doubao"
os.environ["KIMI_AUTH_TOKEN"] = "fake_kimi"
os.environ["GLM_AUTH_TOKEN"] = "fake_glm"

from fastapi.testclient import TestClient  # noqa: E402

import server as srv  # noqa: E402
from providers.failover import CooldownTracker  # noqa: E402
from providers.quarantine import QuarantineStore  # noqa: E402
from providers.risk_dossier import (  # noqa: E402
    AVOIDANCE_MATRIX,
    POSTURE_CLEAR,
    POSTURE_COOLING,
    POSTURE_DISABLED,
    POSTURE_LONG_AVOIDANCE,
    POSTURE_PROBATION,
    POSTURE_QUARANTINED,
    POSTURE_WATCH,
    RiskDossier,
)
from providers.risk_ledger import RiskLedger  # noqa: E402


def _isolated():
    """返回 (dossier, ledger, store, tracker, tmpdir) 的隔离组合。"""
    tmp = Path(tempfile.mkdtemp(prefix="dossier_"))
    ledger = RiskLedger(directory=tmp)
    store = QuarantineStore(directory=tmp)
    tracker = CooldownTracker()
    tracker.quarantine = store
    tracker.risk_ledger = ledger
    tracker.reset()
    dossier = RiskDossier(ledger=ledger, quarantine=store, cooldown_tracker=tracker)
    return dossier, ledger, store, tracker, tmp


def _cleanup(tmp):
    shutil.rmtree(tmp, ignore_errors=True)


def _seed(ledger, provider, severity, count=1, status=429, reason="", evidence="", age_hours=0.0):
    """向台账写入若干条拦截记录，age_hours 控制记录的新旧。"""
    now = time.time() - age_hours * 3600.0
    for i in range(count):
        ledger.record_interception(
            provider=provider,
            status=status,
            reason=reason or severity,
            severity=severity,
            cooldown_seconds=30.0,
            evidence=evidence,
            now=now + i * 0.001,
        )


# ------------------------------------------------------------------ 1. 全量聚合

def test_all_time_aggregation_beyond_window():
    """远超 24h 窗口的拦截必须仍被计入——窗口滑走后证据不得丢失。"""
    dossier, ledger, store, tracker, tmp = _isolated()
    try:
        _seed(ledger, "deepseek", "auth", count=3, status=401, age_hours=24 * 6)
        _seed(ledger, "deepseek", "waf", count=2, status=403, age_hours=24 * 5)

        # 滑动窗口视角：24h 内应为 0（证据"已丢失"）
        window = ledger.recent_interceptions(hours=24.0)
        assert window["per_provider"].get("deepseek", {}).get("count", 0) == 0, (
            "前置条件：24h 滑动窗口内应无记录，以此证明全量聚合的必要性"
        )

        stats = dossier.provider_entry("deepseek")["interceptions"]
        assert stats["total"] == 5, f"全量聚合必须保留窗口外记录，实际 {stats['total']}"
        assert stats["by_severity"] == {"auth": 3, "waf": 2}
        assert stats["first_ts"] is not None and stats["last_ts"] is not None
        assert stats["first_ts"] <= stats["last_ts"]
        assert stats["last_evidence"] == "" or isinstance(stats["last_evidence"], str)
    finally:
        _cleanup(tmp)
    print("test_all_time_aggregation_beyond_window: PASSED")


def test_last_evidence_and_reason_captured():
    dossier, ledger, store, tracker, tmp = _isolated()
    try:
        _seed(ledger, "kimi", "waf", reason="RGV587", evidence="请完成滑块验证", age_hours=200)
        _seed(ledger, "kimi", "ratelimit", reason="too many requests",
              evidence="429 Too Many Requests", age_hours=1)
        stats = dossier.provider_entry("kimi")["interceptions"]
        assert stats["last_reason"] == "too many requests"
        assert "429" in stats["last_evidence"], "必须保留最近一次的真实上游原文"
        assert stats["by_severity"] == {"waf": 1, "ratelimit": 1}
    finally:
        _cleanup(tmp)
    print("test_last_evidence_and_reason_captured: PASSED")


# ------------------------------------------------------------------ 2. 姿态派生

def test_posture_derivation_is_fail_closed():
    dossier, ledger, store, tracker, tmp = _isolated()
    try:
        now = time.time()
        # 正常
        assert dossier.derive_posture("qwen", now) == POSTURE_CLEAR

        # 监视期：近期有拦截证据
        _seed(ledger, "qwen", "ratelimit", age_hours=0.2)
        assert dossier.derive_posture("qwen", now) == POSTURE_WATCH

        # 旧证据不再进入监视期
        _seed(ledger, "glm", "ratelimit", age_hours=5.0)
        assert dossier.derive_posture("glm", now) == POSTURE_CLEAR

        # 观察期优先于监视期
        asyncio.run(tracker.record_failure("glm", 429, "rate_limit", custom_cooldown=0.15))
        time.sleep(0.25)
        assert tracker.is_in_probation("glm") is True
        assert dossier.derive_posture("glm", now) == POSTURE_PROBATION

        # 冷却期优先于观察期
        asyncio.run(tracker.record_failure("kimi", 429, "rate_limit", custom_cooldown=60))
        assert dossier.derive_posture("kimi", now) == POSTURE_COOLING

        # 封禁租约优先于一切
        store.set_lease("doubao", duration_seconds=3600, reason="平台封禁")
        assert dossier.derive_posture("doubao", now) == POSTURE_QUARANTINED
    finally:
        _cleanup(tmp)
    print("test_posture_derivation_is_fail_closed: PASSED")


def test_dispatchability_semantics():
    """只有硬封禁不可派发；软冷却保留最后一搏语义（与 OPT-021 一致）。"""
    dossier, ledger, store, tracker, tmp = _isolated()
    try:
        assert dossier.provider_entry("qwen")["dispatchable"] is True

        asyncio.run(tracker.record_failure("qwen", 429, "rate_limit", custom_cooldown=60))
        cooling = dossier.provider_entry("qwen")
        assert cooling["posture"] == POSTURE_COOLING
        assert cooling["dispatchable"] is True, "软冷却必须仍可派发（可用性优先）"

        store.set_lease("qwen", duration_seconds=3600)
        banned = dossier.provider_entry("qwen")
        assert banned["posture"] == POSTURE_QUARANTINED
        assert banned["dispatchable"] is False, "硬封禁绝不可派发"
    finally:
        _cleanup(tmp)
    print("test_dispatchability_semantics: PASSED")


# ------------------------------------------------------------------ 3. 规避研判

def test_avoidance_verdict_follows_dominant_severity():
    dossier, ledger, store, tracker, tmp = _isolated()
    try:
        _seed(ledger, "qwen", "waf", count=1, age_hours=100)
        _seed(ledger, "qwen", "waf", count=1, age_hours=90)
        _seed(ledger, "qwen", "ratelimit", count=1, age_hours=80)
        v = dossier.provider_entry("qwen")["avoidance"]
        assert v["verdict"] == "waf", "应取出现次数最多的严重度作为主要症结"

        _seed(ledger, "glm", "ratelimit", count=4, age_hours=100)
        assert dossier.provider_entry("glm")["avoidance"]["verdict"] == "ratelimit"

        _seed(ledger, "kimi", "auth", count=2, age_hours=100)
        assert dossier.provider_entry("kimi")["avoidance"]["verdict"] == "auth"

        # 无任何记录 -> 默认按最轻的 ratelimit 研判，不得抛错
        assert dossier.provider_entry("doubao")["avoidance"]["verdict"] == "ratelimit"
    finally:
        _cleanup(tmp)
    print("test_avoidance_verdict_follows_dominant_severity: PASSED")


def test_quarantine_always_judged_as_account_ban():
    """有租约时，无论台账里是什么严重度，一律按账号级封禁研判。"""
    dossier, ledger, store, tracker, tmp = _isolated()
    try:
        _seed(ledger, "deepseek", "ratelimit", count=5, age_hours=100)
        store.set_lease("deepseek", duration_seconds=72 * 3600, reason="平台封禁")
        v = dossier.provider_entry("deepseek")["avoidance"]
        assert v["verdict"] == "ban", "租约必须压过台账严重度，按账号级封禁研判"
        assert v["recoverable_by_relogin"] is False
        assert v["technically_bypassable"] is False
    finally:
        _cleanup(tmp)
    print("test_quarantine_always_judged_as_account_ban: PASSED")


def test_relogin_recovery_answers_are_accurate():
    """直接回答「现在受到风控的 llm，能通过重新登陆恢复使用吗」。"""
    # 令牌/会话失效与人机验证：重登有效
    assert AVOIDANCE_MATRIX["auth"]["recoverable_by_relogin"] is True
    assert AVOIDANCE_MATRIX["waf"]["recoverable_by_relogin"] is True
    # 频率限制：重登无意义（瞬时，自动恢复）
    assert AVOIDANCE_MATRIX["ratelimit"]["recoverable_by_relogin"] is False
    assert AVOIDANCE_MATRIX["ratelimit"]["auto_recovers"] is True
    # 账号级封禁：重登无效，只能等待平台自然解封
    assert AVOIDANCE_MATRIX["ban"]["recoverable_by_relogin"] is False
    assert AVOIDANCE_MATRIX["ban"]["technically_bypassable"] is False
    assert AVOIDANCE_MATRIX["ban"]["auto_recovers"] is True
    # 任何一类都不可用技术手段「绕开」
    assert all(not v["technically_bypassable"] for v in AVOIDANCE_MATRIX.values())
    print("test_relogin_recovery_answers_are_accurate: PASSED")


def test_recommended_actions_match_posture():
    dossier, ledger, store, tracker, tmp = _isolated()
    try:
        # 健康节点：不得套用分级策略话术
        acts = dossier.provider_entry("qwen")["recommended_actions"]
        assert acts == ["无风控记录，可正常派发"], acts

        # 封禁：必须明确告知重登无效
        store.set_lease("deepseek", duration_seconds=72 * 3600)
        acts = dossier.provider_entry("deepseek")["recommended_actions"]
        joined = " ".join(acts)
        assert "重新登录" in joined and "无效" in joined, joined
        assert "login.py" not in joined, "账号级封禁不得建议重登"

        # 认证失效：必须给出具体可执行命令
        _seed(ledger, "kimi", "auth", count=1, age_hours=100)
        acts = dossier.provider_entry("kimi")["recommended_actions"]
        assert any("login.py kimi" in a for a in acts), acts

        # 人机验证：给出命令且明确禁止自动破解
        _seed(ledger, "glm", "waf", count=1, age_hours=100)
        acts = dossier.provider_entry("glm")["recommended_actions"]
        assert any("login.py glm" in a for a in acts), acts
        assert any("严禁" in a for a in acts), acts

        # 观察期：给出进度而非人工动作
        asyncio.run(tracker.record_failure("doubao", 429, "rate_limit", custom_cooldown=0.15))
        time.sleep(0.25)
        acts = dossier.provider_entry("doubao")["recommended_actions"]
        assert any("观察期" in a for a in acts), acts
    finally:
        _cleanup(tmp)
    print("test_recommended_actions_match_posture: PASSED")


# ------------------------------------------------------------------ 4. 只读性

def test_dossier_is_read_only():
    """查询档案绝不得改动风控状态（不得解除租约、清冷却或写台账）。"""
    dossier, ledger, store, tracker, tmp = _isolated()
    try:
        store.set_lease("deepseek", duration_seconds=3600)
        asyncio.run(tracker.record_failure("qwen", 429, "rate_limit", custom_cooldown=60))
        before_lease = store.remaining("deepseek")
        before_cd = tracker.get_remaining_cooldown("qwen")
        before_lines = ledger.ledger_path.read_text(encoding="utf-8") if ledger.ledger_path.exists() else ""

        for _ in range(3):
            dossier.build()

        assert store.remaining("deepseek") > 0
        assert abs(store.remaining("deepseek") - before_lease) < 1.0
        assert tracker.get_remaining_cooldown("qwen") > 0
        assert tracker.get_remaining_cooldown("qwen") <= before_cd
        after_lines = ledger.ledger_path.read_text(encoding="utf-8") if ledger.ledger_path.exists() else ""
        assert after_lines == before_lines, "只读查询不得追加任何台账记录"
    finally:
        _cleanup(tmp)
    print("test_dossier_is_read_only: PASSED")


# ------------------------------------------------------------------ 5. 健壮性

def test_survives_broken_ledger_and_missing_tracker():
    """台账读取异常只读聚合、缺失 tracker 时降级，均不得中断档案装配。"""
    tmp = Path(tempfile.mkdtemp(prefix="dossier_"))
    try:
        ledger = RiskLedger(directory=tmp)
        store = QuarantineStore(directory=tmp)

        class _Boom(RiskLedger):
            def _read_tail(self, limit):  # type: ignore[override]
                raise RuntimeError("boom")

        boom = RiskDossier(ledger=_Boom(directory=tmp), quarantine=store, cooldown_tracker=None)
        d = boom.build()
        assert d["summary"]["total_providers"] >= 5
        assert d["providers"]["qwen"]["interceptions"]["total"] == 0
        assert d["providers"]["qwen"]["posture"] == POSTURE_CLEAR
        assert d["providers"]["qwen"]["pacing"]["effective_interval_s"] >= 0.0

        # 无 tracker：姿态仍可依据租约与台账派生
        store.set_lease("glm", duration_seconds=600)
        d2 = boom.build()
        assert d2["providers"]["glm"]["posture"] == POSTURE_QUARANTINED
        assert d2["providers"]["glm"]["dispatchable"] is False
    finally:
        _cleanup(tmp)
    print("test_survives_broken_ledger_and_missing_tracker: PASSED")


# ------------------------------------------------------------------ 6. 开关

def test_disable_switch():
    dossier, ledger, store, tracker, tmp = _isolated()
    os.environ["GATEWAY_DISABLE_RISK_DOSSIER"] = "1"
    try:
        assert dossier.is_disabled() is True
        assert dossier.build()["enabled"] is False
    finally:
        os.environ.pop("GATEWAY_DISABLE_RISK_DOSSIER", None)
        _cleanup(tmp)
    print("test_disable_switch: PASSED")


# ------------------------------------------------------------------ 7. 端到端

def test_diagnostics_and_admin_endpoint_exposure():
    ledger = srv.risk_ledger
    store = srv.quarantine_store
    tracker = srv.failover_router.cooldown_tracker
    try:
        diag = TestClient(srv.app).get("/v1/diagnostics").json()
        assert "risk_dossier_summary" in diag
        summary = diag["risk_dossier_summary"]
        assert "posture_counts" in summary
        assert set(summary["posture_counts"]) == {
            POSTURE_QUARANTINED, POSTURE_DISABLED, POSTURE_LONG_AVOIDANCE,
            POSTURE_COOLING, POSTURE_PROBATION, POSTURE_WATCH, POSTURE_CLEAR
        }, "姿态词表新增成员时必须同步更新此断言（OPT-035 新增 disabled）"

        r = TestClient(srv.app).get("/v1/admin/risk")
        assert r.status_code == 200, r.text
        body = r.json()
        for key in ("summary", "providers", "avoidance_matrix"):
            assert key in body
        assert set(body["avoidance_matrix"]) == {
            "ratelimit",
            "waf",
            "auth",
            "ban",
            "disabled",
            "timeout",
            "truncated",
            "unconfigured",
        }, (
            "研判矩阵新增成因时必须同步更新此断言"
            "（OPT-035 新增 disabled；[M2] 新增 timeout、[S2] 新增 truncated —— "
            "这两类都是**上游故障**而不是风控拦截，建议必须指向上游健康/本地出口"
            "质量，而不是「去重新登陆」或「去完成人机验证」；"
            "[DAY-ZERO] 新增 unconfigured —— 这是**本机配置缺口**，既不是平台风控、"
            "也不是上游故障：它的解除方式与 auth 一样是 `login.py`（所以重登有效），"
            "而 disabled 的解除方式是改配置（重登无效），两者不得混为一谈）"
        )

        # 单节点视图
        r2 = TestClient(srv.app).get("/v1/admin/risk", params={"provider": "qwen"})
        assert r2.status_code == 200, r2.text
        entry = r2.json()["entry"]
        assert entry["provider"] == "qwen"
        assert "recommended_actions" in entry and "avoidance" in entry

        # 封禁后档案必须反映出来
        store.set_lease("deepseek", duration_seconds=3600)
        try:
            r3 = TestClient(srv.app).get("/v1/admin/risk", params={"provider": "deepseek"})
            e3 = r3.json()["entry"]
            assert e3["posture"] == POSTURE_QUARANTINED
            assert e3["dispatchable"] is False
            assert e3["avoidance"]["verdict"] == "ban"
            assert e3["quarantine"]["remaining_hours"] > 0
        finally:
            store.clear("deepseek")
    finally:
        store.clear_all_for_tests()
        tracker.reset()
        ledger.clear()
    print("test_diagnostics_and_admin_endpoint_exposure: PASSED")


def run_all():
    test_all_time_aggregation_beyond_window()
    test_last_evidence_and_reason_captured()
    test_posture_derivation_is_fail_closed()
    test_dispatchability_semantics()
    test_avoidance_verdict_follows_dominant_severity()
    test_quarantine_always_judged_as_account_ban()
    test_relogin_recovery_answers_are_accurate()
    test_recommended_actions_match_posture()
    test_dossier_is_read_only()
    test_survives_broken_ledger_and_missing_tracker()
    test_disable_switch()
    test_diagnostics_and_admin_endpoint_exposure()
    print("\n✅ All risk-dossier tests (OPT-023) passed!")


if __name__ == "__main__":
    run_all()
