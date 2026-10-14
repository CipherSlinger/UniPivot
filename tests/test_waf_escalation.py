"""人机验证长效避让与人工介入解除 (OPT-029) 测试套件

承 OPT-023 的规避研判往下推。档案对三类的结论是：
  ratelimit  -> auto_recovers=True  ，重登无意义（瞬时、自行衰减）
  auth       -> recoverable_by_relogin=True，重登有效
  waf        -> recoverable_by_relogin=True，但 **auto_recovers=False**
也就是说人机验证是**唯一只有人能解除**的一类。而旧代码把 waf 与 ratelimit 放在
同一个冷却分支里按同一个公式退避：

    cooldown = min(GATEWAY_WAF_COOLDOWN_MAX(900), base * 1.2 ** min(n-1, 3))

实测：重复的人机验证拦截在第 4 次即饱和于 ~78s 且**永不再增长**（指数封顶 3 次
+ 900s 上限），于是该节点每 ~78 秒被再次触碰一次，无限循环——恰恰是对**最需要
人介入**的那一类给了**最短的耐心**，也是对**唯一没有自动恢复路径**的那一类施加了
最频繁的自动重试。这与 OPT-025 的整环风暴同源（重试本身就是风控触发器），只是
发生在单节点上并自我维持，且与本项目「严禁加重其他厂商风控」的约束直接冲突。

本次修复分两半，缺一不可：
1. **升级**：重复的人机验证按 2 的幂次长效避让（与认证级同构）；
2. **出口**：长效避让必须可被**人工介入**解除——而「人工已介入」的唯一可观测
   证据就是**凭证文件变了**（`login.py` 完成滑块验证后令牌必然刷新）。若没有这一
   半，第 1 半就会造出一个没有出口的永久冻结，正是 OPT-026 已经修过的那类自锁
   缺陷换了个成因。因此 WAF 拦截时刻必须留下凭证基线（`note_interception`），
   且复位严格限定成因（`auth` 与 `waf` 可，`ratelimit` 永不）。

本套件验证：
1. **频控零回归**：429 的短冷却 + 上限行为与修复前**逐值一致**
2. 人机验证重复出现即升级，且按 2 的幂次增长、受上限约束
3. 单次人机验证不得被升级（避免误伤偶发验证）
4. 升级后的避让跨重启存活
5. 人工重登（凭证指纹变化）解除后人机验证避让，且经受观察期重新验证
6. **反例**：指纹未变则不得解除——否则等于没有避让
7. 成因限定是双向的：解除 waf 不得解除 ratelimit；解除 ratelimit 更不可能
8. 运维封禁租约优先级更高，永不被该路径解除
9. WAF 基线**不得**污染 OPT-027 的账号级封禁推断（判别式严格限定 401）
10. 档案建议按成因分支：人机验证不得沿用「无需人工介入」
11. 开关与遥测

运行: .venv/bin/python tests/test_waf_escalation.py
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["TESTING"] = "1"
os.environ["GATEWAY_DISABLE_AUTO_REFRESH"] = "1"
os.environ["GATEWAY_DISABLE_PACING"] = "1"
for _name in (
    "GATEWAY_DISABLE_PROBATION",
    "GATEWAY_DISABLE_QUARANTINE",
    "GATEWAY_DISABLE_RISK_LEDGER",
    "GATEWAY_DISABLE_CREDENTIAL_WATCH",
    "GATEWAY_DISABLE_BAN_INFERENCE",
    "GATEWAY_DISABLE_CREDENTIAL_RENEWAL",
):
    os.environ.pop(_name, None)
os.environ["GATEWAY_PROBATION_MAX_RENEWALS"] = "0"
os.environ["QWEN_AUTH_TOKEN"] = "fake_qwen"
os.environ["DEEPSEEK_AUTH_TOKEN"] = "fake_ds"
os.environ["DOUBAO_AUTH_TOKEN"] = "fake_doubao"
os.environ["KIMI_AUTH_TOKEN"] = "fake_kimi"
os.environ["GLM_AUTH_TOKEN"] = "fake_glm"

from providers import failover as fov  # noqa: E402
from providers.credential_renewal import get_credential_renewal  # noqa: E402
from providers.quarantine import QuarantineStore  # noqa: E402
from providers.risk_ledger import SEVERITY_AUTH, SEVERITY_RATE_LIMIT, SEVERITY_WAF  # noqa: E402

import providers.credential_renewal as cred  # noqa: E402

# ------------------------------------------------------------------ 夹具

_TMP = Path(tempfile.mkdtemp(prefix="opt029_"))
_FAKE: dict = {}


def _install_fake_credentials():
    """把凭证路径指向临时文件，绝不触碰真实 `session/`。"""
    files = {pk: _TMP / f"{pk}.json" for pk in ("qwen", "glm", "doubao", "kimi", "deepseek")}
    _FAKE.clear()
    _FAKE.update(files)
    cred.credential_path = lambda pk: files.get((pk or "").strip().lower())


def _original_credential_path(pk):
    return cred._PROVIDER_SESSION_FILES.get((pk or "").strip().lower())


def _write_credential(provider_key: str, token: str, captured_at: float = 1_700_000_000.0):
    path = _FAKE[provider_key]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"token": token, "cookie": f"c_{token}", "captured_at": captured_at,
                    "user_id": "u1", "device_id": "d1"}, ensure_ascii=False),
        encoding="utf-8",
    )


def _tracker() -> fov.CooldownTracker:
    """返回一个全隔离的 tracker（含独立租约存储与清空的所有进程级单例状态）。"""
    store = QuarantineStore(directory=Path(tempfile.mkdtemp(prefix="opt029_q_")))
    t = fov.CooldownTracker()
    t.quarantine = store
    t.reset()
    get_credential_renewal().clear_all_for_tests()
    t.credential_watch.clear_all_for_tests()
    return t


def _teardown(tracker):
    tracker.reset()
    get_credential_renewal().clear_all_for_tests()
    tracker.quarantine.clear_all_for_tests()
    tracker.credential_watch.clear_all_for_tests()


def _waf(tracker, pk, times=1):
    for _ in range(times):
        asyncio.run(tracker.record_failure(pk, 403, "RGV587 人机验证失败，请完成滑块验证"))


def _ratelimit(tracker, pk, times=1):
    for _ in range(times):
        asyncio.run(tracker.record_failure(pk, 429, "rate limit exceeded"))


# ------------------------------------------------------------------ 1. 频控零回归

def test_ratelimit_behaviour_is_unchanged():
    """频控是**瞬时且自愈**的，「到点重试」本身就是正确的恢复策略，因此它的短冷却
    与上限必须与修复前**逐值一致**——本优化只针对人机验证，绝不扩大频控的避让。"""
    tracker = _tracker()
    base = fov.PROVIDER_RISK_SPECS["glm"].default_cooldown
    try:
        got = [_ratelimit_one(tracker, i) for i in range(1, 7)]
        # 公式必须与修复前逐值一致：base * 1.2**min(n-1, 3)，指数在第 4 次封顶
        expected = [base * (1.2 ** min(n - 1, 3)) for n in range(1, 7)]
        assert [round(v, 2) for v in got] == [round(v, 2) for v in expected], got
        assert got[3] == got[5], "指数必须在第 4 次封顶（这正是旧行为的正确部分）"
        assert tracker.cooldown_cause("glm") == SEVERITY_RATE_LIMIT
        # 频控无论重复多少次都不得升级为长效避让
        _ratelimit(tracker, "glm", 10)
        assert tracker.get_remaining_cooldown("glm") < 900.0, "频控不得被升级"
    finally:
        _teardown(tracker)
    print("test_ratelimit_behaviour_is_unchanged: PASSED")


def _ratelimit_one(tracker, n):
    return asyncio.run(tracker.record_failure("glm", 429, "rate limit exceeded"))


# ------------------------------------------------------------------ 2. 升级

def test_single_waf_is_not_escalated():
    """偶发一次人机验证不得升级——升级是对**重复**出现的判定，不是对单次出现。"""
    tracker = _tracker()
    try:
        _waf(tracker, "qwen")
        cd = tracker.get_remaining_cooldown("qwen")
        assert cd < 900.0, f"单次人机验证不得升级, 实际 {cd}"
        assert tracker.cooldown_cause("qwen") == SEVERITY_WAF
    finally:
        _teardown(tracker)
    print("test_single_waf_is_not_escalated: PASSED")


def test_repeated_waf_escalates_exponentially():
    """重复出现的人机验证必须按 2 的幂次升级：该类拦截**只有人能解除**，
    每一次自动重试都是又一次「自动化尝试」，只会加重标记。旧行为饱和于 ~78s
    并无限循环，等于对最需要人介入的一类施加最频繁的自动重试。"""
    tracker = _tracker()
    os.environ["GATEWAY_WAF_ESCALATION_MIN_FAILURES"] = "3"
    try:
        first = asyncio.run(tracker.record_failure("qwen", 403, "RGV587 人机验证失败"))
        second = asyncio.run(tracker.record_failure("qwen", 403, "RGV587 人机验证失败"))
        assert first < 900.0 and second < 900.0, "阈值之前保持短冷却"

        third = asyncio.run(tracker.record_failure("qwen", 403, "RGV587 人机验证失败"))
        assert third == 3600.0, f"达阈值应升级为长效避让, 实际 {third}"
        fourth = asyncio.run(tracker.record_failure("qwen", 403, "RGV587 人机验证失败"))
        assert fourth == 7200.0, f"应按 2 的幂次增长, 实际 {fourth}"

        prev = fourth
        for _ in range(8):
            cur = asyncio.run(tracker.record_failure("qwen", 403, "RGV587 人机验证失败"))
            assert cur >= prev, "冷却必须单调不减"
            assert cur <= 86400.0, f"不得越过高阶上限, 实际 {cur}"
            prev = cur
        assert prev == 86400.0, f"应最终撞到上限, 实际 {prev}"
    finally:
        os.environ["GATEWAY_WAF_ESCALATION_MIN_FAILURES"] = "3"
        _teardown(tracker)
    print("test_repeated_waf_escalates_exponentially: PASSED")


def test_escalation_requires_recognition_of_human_verification():
    """升级只对**人机验证**成立；无法识别为验证码的风控命中保持既有短冷却。"""
    tracker = _tracker()
    try:
        for _ in range(6):
            asyncio.run(tracker.record_failure("kimi", 403, "请求被拒绝"))
        assert tracker.get_remaining_cooldown("kimi") < 900.0, "非验证码类不得升级"
    finally:
        _teardown(tracker)
    print("test_escalation_requires_recognition_of_human_verification: PASSED")


def test_escalated_cooldown_survives_restart():
    """升级后的长效避让必须跨重启存活，否则重启即成为绕过入口。"""
    directory = Path(tempfile.mkdtemp(prefix="opt029_r_"))
    tracker = _tracker()
    ledger = tracker.risk_ledger
    original_dir = ledger.directory
    try:
        ledger.directory = directory
        _waf(tracker, "doubao", 4)
        remaining = tracker.get_remaining_cooldown("doubao")
        assert remaining > 3600.0, remaining
        tracker._persist_state()

        revived = fov.CooldownTracker()
        revived.quarantine = tracker.quarantine
        revived.risk_ledger = ledger
        revived._restore_persisted_state()
        assert revived.get_remaining_cooldown("doubao") > 3600.0, "升级后的避让必须跨界存活"
        assert revived.cooldown_cause("doubao") == SEVERITY_WAF, "成因同样必须跨界存活"
        revived.reset()
    finally:
        ledger.directory = original_dir
        _teardown(tracker)
        shutil.rmtree(directory, ignore_errors=True)
    print("test_escalated_cooldown_survives_restart: PASSED")


# ------------------------------------------------------------------ 3. 人工介入的出口

def test_human_relogin_releases_waf_avoidance():
    """**本机制的另一半，缺了它升级就是永久冻结**：人工执行 `login.py` 完成滑块
    验证后令牌必然刷新，凭证指纹变化即「人类已介入」的唯一可观测证据，网关据此
    自动解除长效避让，并经恢复观察期以串行探路重新验证。"""
    _install_fake_credentials()
    _write_credential("qwen", "A")
    tracker = _tracker()
    try:
        _waf(tracker, "qwen", 4)
        assert tracker.get_remaining_cooldown("qwen") > 3600.0
        assert tracker.cooldown_cause("qwen") == SEVERITY_WAF

        # 人工完成滑块验证 -> 新令牌 -> 指纹变化
        _write_credential("qwen", "B", 1_700_005_000.0)
        assert tracker.renewed_providers() == ["qwen"], "WAF 拦截必须留下凭证基线"
        released = asyncio.run(tracker.release_renewed_credentials())
        assert released == ["qwen"], released

        assert tracker.get_remaining_cooldown("qwen") == 0.0, "人工介入后避让必须解除"
        assert tracker.cooldown_cause("qwen") == ""
        assert tracker.is_in_probation("qwen") is True, "解除后必须经观察期重新验证"
    finally:
        _teardown(tracker)
    print("test_human_relogin_releases_waf_avoidance: PASSED")


def test_waf_avoidance_not_released_without_human_intervention():
    """反例：凭证未变时绝不能解除——否则「升级」等于没有升级。"""
    _install_fake_credentials()
    _write_credential("glm", "A")
    tracker = _tracker()
    try:
        _waf(tracker, "glm", 4)
        assert tracker.renewed_providers() == [], "凭证未变则无人工介入证据"
        assert asyncio.run(tracker.release_renewed_credentials()) == []
        assert tracker.get_remaining_cooldown("glm") > 3600.0, "仍必须处于长效避让"
    finally:
        _teardown(tracker)
    print("test_waf_avoidance_not_released_without_human_intervention: PASSED")


# ------------------------------------------------------------------ 4. 成因限定（双向）

def test_ratelimit_is_never_released_by_relogin():
    """频控是瞬时可自愈的，重登毫无意义——任何情况下都不得被该路径解除。"""
    _install_fake_credentials()
    _write_credential("kimi", "A")
    tracker = _tracker()
    try:
        _ratelimit(tracker, "kimi", 3)
        cd_before = tracker.get_remaining_cooldown("kimi")
        assert cd_before > 0.0
        _write_credential("kimi", "B", 1_700_005_000.0)
        assert asyncio.run(tracker.release_renewed_credentials()) == []
        cd_after = tracker.get_remaining_cooldown("kimi")
        # 只允许自然衰减，绝不出现复位（复位会把剩余量直接打到 0）
        assert cd_after > cd_before - 1.0, f"频控不得被重登解除: {cd_before} -> {cd_after}"
    finally:
        _teardown(tracker)
    print("test_ratelimit_is_never_released_by_relogin: PASSED")


def test_operator_lease_still_wins_over_waf_release():
    """运维封禁租约在语义上高于一切：即使凭证已更新，也不得因它而解除租约本身。"""
    _install_fake_credentials()
    _write_credential("doubao", "A")
    tracker = _tracker()
    try:
        _waf(tracker, "doubao", 4)
        tracker.quarantine.set_lease("doubao", 7200.0, reason="人工封禁")
        _write_credential("doubao", "B", 1_700_005_000.0)
        released = asyncio.run(tracker.release_renewed_credentials())
        assert released == [], "持有运维租约时不得复位"
        assert tracker.is_quarantined("doubao") is True, "运维租约永不被解除"
    finally:
        _teardown(tracker)
    print("test_operator_lease_still_wins_over_waf_release: PASSED")


def test_waf_baseline_does_not_pollute_ban_inference():
    """WAF 的凭证基线**只记基线、不做推断**：OPT-027 的账号级封禁判别式严格限定
    「被 401 拒绝」，把人机验证混进去会在成因完全不同的情况下误判为封禁。"""
    _install_fake_credentials()
    _write_credential("deepseek", "A")
    tracker = _tracker()
    try:
        _waf(tracker, "deepseek")          # 留下基线
        _write_credential("deepseek", "B", 1_700_005_000.0)
        _waf(tracker, "deepseek", 3)       # 新凭证上的人机验证 —— 不是 401
        assert tracker._fresh_credential_rejected == {}, "人机验证不得进入封禁推断路径"
        assert tracker.quarantine.source("deepseek") == "", "不得据此施加自动租约"
        assert tracker.ban_inference_report()["stats"]["auto_leases_applied"] == 0
    finally:
        _teardown(tracker)
    print("test_waf_baseline_does_not_pollute_ban_inference: PASSED")


# ------------------------------------------------------------------ 5. 档案建议

def test_dossier_advice_branches_on_cooldown_cause():
    """档案建议必须按**冷却成因**分支：人机验证不得沿用频控的「无需人工介入」，
    否则运维会照着错误的建议去等待一个只有人能解决的节点（数小时）。"""
    from providers.risk_dossier import RiskDossier

    tracker = _tracker()
    try:
        dossier = RiskDossier(quarantine=tracker.quarantine, cooldown_tracker=tracker)
        _waf(tracker, "qwen", 4)
        entry = dossier.provider_entry("qwen")
        assert entry["cooldown_cause"] == SEVERITY_WAF, entry["cooldown_cause"]
        advice = " ".join(entry["recommended_actions"])
        assert "人机验证" in advice, advice
        assert "login.py qwen" in advice, advice
        assert "无需人工介入" not in advice, "人机验证不得沿用「无需人工介入」"

        # 频控仍应是「无需人工介入」
        tracker.reset()
        _ratelimit(tracker, "glm", 2)
        entry = dossier.provider_entry("glm")
        assert entry["cooldown_cause"] == SEVERITY_RATE_LIMIT
        advice = " ".join(entry["recommended_actions"])
        assert "无需人工介入" in advice, advice
        assert "login.py" not in advice, "频控不得建议重登（毫无意义）"
    finally:
        _teardown(tracker)
    print("test_dossier_advice_branches_on_cooldown_cause: PASSED")


# ------------------------------------------------------------------ 6. 开关与遥测

def test_cooldown_cause_accessor_and_stats():
    """公开的冷成因访问器：无冷却时为空串；有冷却时返回成因。"""
    tracker = _tracker()
    try:
        assert tracker.cooldown_cause("qwen") == "", "无冷却时成因应为空"
        _waf(tracker, "qwen", 1)
        assert tracker.cooldown_cause("qwen") == SEVERITY_WAF
        _waf(tracker, "qwen", 3)
        assert tracker.cooldown_cause("qwen") == SEVERITY_WAF, "升级后成因不变"

        tracker.reset()
        asyncio.run(tracker.record_failure("deepseek", 401, "令牌失效"))
        assert tracker.cooldown_cause("deepseek") == SEVERITY_AUTH
    finally:
        _teardown(tracker)
    print("test_cooldown_cause_accessor_and_stats: PASSED")


def test_escalation_defaults_are_configurable():
    os.environ["GATEWAY_WAF_ESCALATION_MIN_FAILURES"] = "1"
    os.environ["GATEWAY_WAF_LONG_COOLDOWN_BASE"] = "120"
    os.environ["GATEWAY_WAF_LONG_COOLDOWN_MAX"] = "600"
    tracker = _tracker()
    try:
        cd = asyncio.run(tracker.record_failure("qwen", 403, "RGV587 人机验证"))
        assert cd == 120.0, cd
        for _ in range(5):
            cd = asyncio.run(tracker.record_failure("qwen", 403, "RGV587 人机验证"))
        assert cd == 600.0, f"必须受上限约束, 实际 {cd}"
        assert tracker._waf_escalation_min_failures() == 1
    finally:
        os.environ["GATEWAY_WAF_ESCALATION_MIN_FAILURES"] = "3"
        os.environ.pop("GATEWAY_WAF_LONG_COOLDOWN_BASE", None)
        os.environ.pop("GATEWAY_WAF_LONG_COOLDOWN_MAX", None)
        _teardown(tracker)
    print("test_escalation_defaults_are_configurable: PASSED")


def run_all():
    test_ratelimit_behaviour_is_unchanged()
    test_single_waf_is_not_escalated()
    test_repeated_waf_escalates_exponentially()
    test_escalation_requires_recognition_of_human_verification()
    test_escalated_cooldown_survives_restart()
    test_human_relogin_releases_waf_avoidance()
    test_waf_avoidance_not_released_without_human_intervention()
    test_ratelimit_is_never_released_by_relogin()
    test_operator_lease_still_wins_over_waf_release()
    test_waf_baseline_does_not_pollute_ban_inference()
    test_dossier_advice_branches_on_cooldown_cause()
    test_cooldown_cause_accessor_and_stats()
    test_escalation_defaults_are_configurable()
    print("\n✅ All WAF-escalation tests (OPT-029) passed!")


if __name__ == "__main__":
    try:
        run_all()
    finally:
        cred.credential_path = _original_credential_path
