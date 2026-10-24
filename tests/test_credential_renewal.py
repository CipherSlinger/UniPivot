"""凭证续期检测与认证级避让复位 (OPT-026) 测试套件

动机（接 OPT-025 的现场证据继续往下推）：
`session/risk_ledger.jsonl` 里五个节点的认证失败次数是 54/54/54/87/60 —— `record_failure`
对认证失效按 `GATEWAY_AUTH_COOLDOWN_BASE × 2ⁿ` 长效升级（封顶 `GATEWAY_AUTH_COOLDOWN_MAX`
= 86400s），因此**五个节点全部撞到 24 小时上限**，且该状态随台账持久化、跨重启恢复。

这个惩罚本身是对的：它惩罚的是「拿着同一个失效凭证反复打」——每次重试都是一次
「自动化凭证尝试」。但凭证一旦续期，惩罚就失去了对象。此刻继续冻结，惩罚的不再是
坏行为，而是**已经修好的系统**：
  - `get_remaining_cooldown()` 仍返回 24h -> 健康探针跳过、自愈选举排除、路由降级；
  - 尤其荒谬的是 `server.py:_deferred_refresh_after_cooldown` —— **令牌静默刷新本身
    被冷却门控挡住，最多等 24 小时**。也就是说「凭证失效」制造的状态，正好锁死了
    唯一能修复它的机制；
  - 反向验证：OPT-023 档案对 `auth` 类的结论是 `recoverable_by_relogin = True`，
    系统自己告诉运维「重登可恢复」，可重登之后 24 小时内网关依然在避让该节点，
    结论与行为自相矛盾。

本套件验证 `providers/credential_renewal.py` 与 `providers/failover.py` 的接入：
1. 指纹判定：内容哈希（`sha:`），不可解析退化为 `meta:mtime_ns:size`，缺失为 None
2. **保守默认（fail-closed）**：无记录 / 凭证不可读 / 指纹相同 -> 一律判定「未续期」
3. 指纹在**认证失败时刻**捕获（否则无法回答「与我失败时相比是否已被换掉」）
4. 复位严格限定「认证成因」：持有运维封禁租约永不释放；WAF/频控成因永不代释
5. 复位后进入恢复观察期（OPT-022），避免被压制的需求在凭证刚更新时集中涌出
6. **解锁修复链**：复位使 `get_remaining_cooldown()` 归零，从而解除令牌刷新的冷却门控
7. 跨重启：指纹与冷却成因均可从磁盘恢复；成因缺失时退化为只读查询台账（仍成因限定）
8. 开关 `GATEWAY_DISABLE_CREDENTIAL_RENEWAL`；与 `GATEWAY_DISABLE_RISK_LEDGER` 联动关闭
9. 遥测：/v1/diagnostics、GET /v1/admin/credential-renewal、POST .../scan、
   DELETE /v1/admin/credential-outage 连带复位

运行: .venv/bin/python tests/test_credential_renewal.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["TESTING"] = "1"
os.environ["GATEWAY_DISABLE_AUTO_REFRESH"] = "1"
os.environ["GATEWAY_DISABLE_PACING"] = "1"
os.environ.pop("GATEWAY_DISABLE_CREDENTIAL_RENEWAL", None)
os.environ.pop("GATEWAY_DISABLE_RISK_LEDGER", None)
os.environ["QWEN_AUTH_TOKEN"] = "fake_qwen"
os.environ["DEEPSEEK_AUTH_TOKEN"] = "fake_ds"
os.environ["DOUBAO_AUTH_TOKEN"] = "fake_doubao"
os.environ["KIMI_AUTH_TOKEN"] = "fake_kimi"
os.environ["GLM_AUTH_TOKEN"] = "fake_glm"

from fastapi.testclient import TestClient  # noqa: E402

import providers.credential_renewal as cred  # noqa: E402
import providers.failover as fov  # noqa: E402
import server as srv  # noqa: E402
from providers.credential_renewal import (  # noqa: E402
    CredentialRenewalTracker,
    get_credential_renewal,
)
from providers.risk_ledger import SEVERITY_AUTH, SEVERITY_WAF, get_risk_ledger  # noqa: E402


# ------------------------------------------------------------------ 夹具

_TMP = Path(tempfile.mkdtemp(prefix="opt026_"))
_STORE_DIRS: list = []
_FAKE_FILES: dict = {}


def _install_fake_credentials(providers=("qwen", "glm", "doubao", "kimi", "deepseek")) -> None:
    """把 `credential_path` 指向临时目录中的凭证文件，绝不触碰真实 `session/`。"""
    files = {pk: _TMP / f"{pk}.json" for pk in providers}
    _FAKE_FILES.clear()
    _FAKE_FILES.update(files)
    cred.credential_path = lambda pk: files.get((pk or "").strip().lower())


def _original_credential_path(pk):
    return cred._PROVIDER_SESSION_FILES.get((pk or "").strip().lower())


def _write_credential(provider_key: str, token: str, captured_at: float = 1_700_000_000.0):
    path = _FAKE_FILES[provider_key]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"token": token, "cookie": f"c_{token}", "captured_at": captured_at,
             "user_id": "u1", "device_id": "d1"},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def _fresh_dir() -> Path:
    """每个用例独占一个存储目录，确保指纹记录零串扰。"""
    d = Path(tempfile.mkdtemp(prefix="opt026_store_"))
    _STORE_DIRS.append(d)
    return d


def _fresh_tracker(directory: Path | None = None) -> CredentialRenewalTracker:
    """全新检测器实例（默认独立目录）；显式传入同一目录可模拟「重启后读取同一状态」。"""
    return CredentialRenewalTracker(directory=directory or _fresh_dir())


def _reset_all(tracker: fov.CooldownTracker | None = None) -> None:
    get_credential_renewal().clear_all_for_tests()
    if tracker is not None:
        tracker.reset()
    try:
        ledger = get_risk_ledger()
        ledger.clear()
    except Exception:  # noqa: BLE001
        pass


# ------------------------------------------------------------------ 1. 指纹

def test_fingerprint_is_a_content_hash_not_a_save_counter():
    """指纹是**凭证内容**的哈希；重新保存同一份凭证不得改变它。

    这条断言在 [REFRESH-VERIFY] 里被**修正过**，原版的断言恰好是缺陷本身：
    它断言「只改 captured_at 指纹就要变」，理由是「重登必然刷新 captured_at」。
    但 `Session.save()` 每次保存都刷新 captured_at，因此那个断言等价于
    「任何一次保存都算续期」——包括无头刷新把 profile 里的旧令牌原样再存一遍。
    实测后果：一次什么都没修好的静默刷新，把 60s 的 auth 冷却直接复位到 0，
    解除了 OPT-026 为「拿着同一个失效凭证反复打」而设的避让。

    正确的分界是「凭证内容变了」（真正重登：token 刷新 / Cookie 轮换 / 身份变化），
    不是「文件又被写了一次」。详见 tests/test_credential_renewal_discrimination.py。
    """
    _install_fake_credentials()
    _write_credential("qwen", "tok_A", 1_700_000_000.0)
    fp1 = cred.credential_fingerprint("qwen")
    assert fp1 and fp1.startswith("sha:"), fp1

    # 只改 captured_at（模拟 `Session.save()` 又保存了一次，内容没变）
    _write_credential("qwen", "tok_A", 1_700_000_900.0)
    fp2 = cred.credential_fingerprint("qwen")
    assert fp2 == fp1, (
        "内容没变、只是被重新保存，指纹却变了——"
        "任何一次空刷新都会冒充「凭证已续期」，解除 auth 避让"
    )

    # token 变化当然也改变指纹
    _write_credential("qwen", "tok_B", 1_700_000_900.0)
    assert cred.credential_fingerprint("qwen") != fp2
    print("test_fingerprint_is_a_content_hash_not_a_save_counter: PASSED")


def test_fingerprint_meta_fallback_for_unparseable_file():
    """不可解析的凭证文件退化为 `meta:mtime_ns:size`，绝不抛异常。"""
    _install_fake_credentials()
    path = _FAKE_FILES["glm"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not-a-json {", encoding="utf-8")
    fp = cred.credential_fingerprint("glm")
    assert fp and fp.startswith("meta:"), fp

    # 未知节点 / 文件缺失 -> None
    assert cred.credential_fingerprint("nonexistent") is None
    _FAKE_FILES["kimi"].unlink(missing_ok=True)
    assert cred.credential_fingerprint("kimi") is None
    print("test_fingerprint_meta_fallback_for_unparseable_file: PASSED")


# ------------------------------------------------------------------ 2. 保守默认

def test_no_record_is_not_renewed():
    """本机制启用前就已存在的冷却没有指纹记录 -> 判定「未续期」，继续避让。"""
    _install_fake_credentials()
    _write_credential("qwen", "tok_A")
    t = _fresh_tracker()
    t.clear_all_for_tests()
    assert t.has_record("qwen") is False
    assert t.renewed_providers() == []
    print("test_no_record_is_not_renewed: PASSED")


def test_unreadable_credential_is_not_renewed():
    """凭证不可读 -> 既不记录也不判定续期（任何不确定都选择继续避让）。"""
    _install_fake_credentials()
    _FAKE_FILES["doubao"].unlink(missing_ok=True)
    t = _fresh_tracker()
    t.note_auth_failure("doubao")
    assert t.has_record("doubao") is False, "不可读的凭证不应留下指纹记录"
    assert t.renewed_providers() == []
    print("test_unreadable_credential_is_not_renewed: PASSED")


def test_same_fingerprint_is_not_renewed():
    """凭证未变 -> 绝不判定续期（否则每次扫描都会白白释放 24h 冻结）。"""
    _install_fake_credentials()
    _write_credential("qwen", "tok_A")
    t = _fresh_tracker()
    t.note_auth_failure("qwen")
    assert t.renewed_providers() == []
    assert t.get_diagnostics()["renewed_providers"] == []
    print("test_same_fingerprint_is_not_renewed: PASSED")


def test_renewal_detected_only_after_credential_changes():
    _install_fake_credentials()
    _write_credential("qwen", "tok_old", 1_700_000_000.0)
    t = _fresh_tracker()
    t.note_auth_failure("qwen")
    assert t.renewed_providers() == []

    _write_credential("qwen", "tok_new", 1_700_001_000.0)  # 重登
    assert t.renewed_providers() == ["qwen"]
    assert "login" not in t.get_diagnostics()["hint"]  # 提示文案针对的是复位而非重登
    assert t.get_diagnostics()["hint"], "续期后必须给出可执行的提示"
    print("test_renewal_detected_only_after_credential_changes: PASSED")


def test_fingerprint_captured_at_failure_time_not_scan_time():
    """指纹必须在**失败时刻**捕获，否则「失败后又被换掉」会被误读为「失败时就是新的」。"""
    _install_fake_credentials()
    _write_credential("glm", "tok_new", 1_700_002_000.0)  # 已被重登后的新凭证
    t = _fresh_tracker()
    t.note_auth_failure("glm")  # 此刻失败 -> 指纹 = 新凭证
    assert t.renewed_providers() == [], "失败时的凭证即当前凭证，不得判定为已续期"
    print("test_fingerprint_captured_at_failure_time_not_scan_time: PASSED")


def test_clear_provider():
    _install_fake_credentials()
    _write_credential("qwen", "tok_A")
    t = _fresh_tracker()
    t.note_auth_failure("qwen")
    assert t.has_record("qwen") is True
    t.clear_provider("qwen")
    assert t.has_record("qwen") is False
    assert t.renewed_providers() == []
    print("test_clear_provider: PASSED")


def test_records_survive_restart():
    """「先重登、再重启」也必须被识别：指纹随状态落盘，跨实例恢复。"""
    _install_fake_credentials()
    _write_credential("kimi", "tok_old")
    store = _fresh_dir()
    t1 = _fresh_tracker(store)
    t1.note_auth_failure("kimi")

    t2 = _fresh_tracker(store)  # 模拟进程重启
    assert t2.has_record("kimi") is True, "指纹记录必须跨重启存活"
    assert t2.renewed_providers() == []
    _write_credential("kimi", "tok_new", 1_700_003_000.0)
    assert _fresh_tracker(store).renewed_providers() == ["kimi"]
    print("test_records_survive_restart: PASSED")


# ------------------------------------------------------------------ 3. 复位（认证成因限定）

def _tracker() -> fov.CooldownTracker:
    t = fov.CooldownTracker()
    t.reset()
    get_credential_renewal().clear_all_for_tests()
    return t


def _auth_fail(tracker, pk, times=1):
    for _ in range(times):
        asyncio.run(tracker.record_failure(pk, 401, "上游 401: token 无效或已过期"))


def test_release_caps_to_24h_freeze_and_unblocks_refresh_gate():
    """现场复现：认证失败按 2 的幂次升级到 24h 上限 -> 重登 -> 复位 -> 冷却归零。

    这正是 `server.py:_deferred_refresh_after_cooldown` 依赖的读数：冷却归零
    意味着令牌静默刷新（唯一的自动修复机制）不再被挡住 24 小时。
    """
    _install_fake_credentials()
    _write_credential("deepseek", "tok_old")
    tracker = _tracker()
    try:
        _auth_fail(tracker, "deepseek", 9)  # 1800 × 2^6 -> 封顶 86400
        remaining = tracker.get_remaining_cooldown("deepseek")
        assert remaining > 3600.0, f"9 次认证失败应触发长效升级, 实际 {remaining}"
        assert tracker.is_in_cooldown("deepseek") is True
        assert tracker._cooldown_severity.get("deepseek") == SEVERITY_AUTH

        # 尚未续期：绝不释放
        assert asyncio.run(tracker.release_renewed_credentials()) == []
        assert tracker.is_in_cooldown("deepseek") is True

        _write_credential("deepseek", "tok_new", 1_700_004_000.0)  # 运维执行 login.py
        assert tracker.renewed_providers() == ["deepseek"]

        released = asyncio.run(tracker.release_renewed_credentials())
        assert released == ["deepseek"], released
        assert tracker.get_remaining_cooldown("deepseek") == 0.0, "复位后冷却必须归零"
        assert tracker.is_in_cooldown("deepseek") is False
        assert tracker._failure_counts.get("deepseek", 0) == 0
        assert tracker._cooldown_severity.get("deepseek") is None
        # 释放即进入恢复观察期：解封瞬间的洪峰同样适用于「凭证刚续期」
        assert tracker.is_in_probation("deepseek") is True
        # 指纹记录**必须保留**：释放只是「假设凭证已修好」，下一次失败的比对基准
        # 正是它（否则 OPT-027 无法判定「新凭证仍被拒」）。
        assert get_credential_renewal().has_record("deepseek") is True
    finally:
        _reset_all(tracker)
    print("test_release_caps_to_24h_freeze_and_unblocks_refresh_gate: PASSED")


def test_release_never_touches_operator_quarantine_lease():
    """租约表达**账号级封禁**（重登无效），与「凭证过期」是两回事 -> 永不释放。"""
    _install_fake_credentials()
    _write_credential("deepseek", "tok_old")
    tracker = _tracker()
    try:
        _auth_fail(tracker, "deepseek", 9)
        _write_credential("deepseek", "tok_new", 1_700_004_000.0)
        tracker.quarantine.set_lease("deepseek", 72 * 3600.0, reason="账号级封禁")
        try:
            released = asyncio.run(tracker.release_renewed_credentials())
            assert released == [], "持有租约的节点绝不能被续期检测释放"
            assert tracker.is_quarantined("deepseek") is True
            assert tracker.get_remaining_cooldown("deepseek") > 3600.0
        finally:
            tracker.quarantine.clear("deepseek")
    finally:
        _reset_all(tracker)
    print("test_release_never_touches_operator_quarantine_lease: PASSED")


def test_release_never_touches_waf_or_ratelimit_cooldown():
    """若最近一次失败是 WAF/频控，说明是另一类问题 -> 绝不代它释放。"""
    _install_fake_credentials()
    _write_credential("glm", "tok_old")
    tracker = _tracker()
    try:
        asyncio.run(tracker.record_failure("glm", 429, "请求过于频繁，请稍后再试"))
        assert tracker._cooldown_severity.get("glm") == "ratelimit"
        _write_credential("glm", "tok_new", 1_700_004_000.0)
        get_credential_renewal().note_auth_failure("glm")  # 人为留下指纹记录
        assert asyncio.run(tracker.release_renewed_credentials()) == []
        assert tracker.is_in_cooldown("glm") is True

        # WAF 亦然
        tracker.reset()
        os.environ["GATEWAY_DISABLE_RISK_LEDGER"] = ""
        asyncio.run(tracker.record_failure("glm", 403, "RGV587_ERROR::SM 滑动验证"))
        assert tracker._cooldown_severity.get("glm") == SEVERITY_WAF
        assert asyncio.run(tracker.release_auth_avoidance(["glm"])) == []
        assert tracker.is_in_cooldown("glm") is True
    finally:
        _reset_all(tracker)
    print("test_release_never_touches_waf_or_ratelimit_cooldown: PASSED")


def test_release_is_noop_without_actual_state():
    """无冷却、无失败计数时不必平白引入观察期。"""
    _install_fake_credentials()
    _write_credential("kimi", "tok_old")
    tracker = _tracker()
    try:
        get_credential_renewal().note_auth_failure("kimi")
        _write_credential("kimi", "tok_new", 1_700_004_000.0)
        assert asyncio.run(tracker.release_auth_avoidance(["kimi"])) == []
        assert tracker.is_in_probation("kimi") is False
    finally:
        _reset_all(tracker)
    print("test_release_is_noop_without_actual_state: PASSED")


def test_ledger_fallback_releases_pre_existing_state():
    """成因记录缺失（机制上线前的遗留冷却）时，退化为**只读**查询台账最近一次拦截成因。

    没有这个退化，真实场景里「按档案建议重登」之后仍然无法复位，机制形同虚设。
    退化路径同样成因限定：台账最近一次是 waf 就不释放。
    """
    _install_fake_credentials()
    _write_credential("qwen", "tok_new", 1_700_004_000.0)
    tracker = _tracker()
    try:
        ledger = tracker.risk_ledger
        # 手工构造「有冷却、无成因记录」的遗留状态
        tracker._cooldowns["qwen"] = time.time() + 86400.0
        tracker._failure_counts["qwen"] = 54
        assert tracker._cooldown_severity.get("qwen") is None

        ledger.record_interception("qwen", 401, "上游 401", SEVERITY_AUTH, 86400.0)
        assert tracker._effective_cooldown_severity("qwen") == SEVERITY_AUTH
        assert asyncio.run(tracker.release_auth_avoidance(["qwen"])) == ["qwen"]
        assert tracker.is_in_cooldown("qwen") is False

        # 台账最近一次是 WAF -> 不释放
        tracker._cooldowns["qwen"] = time.time() + 86400.0
        tracker._failure_counts["qwen"] = 54
        tracker._cooldown_severity.pop("qwen", None)
        ledger.record_interception("qwen", 403, "RGV587", SEVERITY_WAF, 86400.0)
        assert tracker._effective_cooldown_severity("qwen") == SEVERITY_WAF
        assert asyncio.run(tracker.release_auth_avoidance(["qwen"])) == []
        assert tracker.is_in_cooldown("qwen") is True
    finally:
        _reset_all(tracker)
    print("test_ledger_fallback_releases_pre_existing_state: PASSED")


def test_cooldown_severity_survives_restart():
    """成因记录随冷却一并持久化：重启后不依赖台账也能正确判定成因。"""
    _install_fake_credentials()
    _write_credential("doubao", "tok_old")
    tracker = _tracker()
    try:
        _auth_fail(tracker, "doubao", 9)
        tracker._persist_state()
        revived = fov.CooldownTracker()  # 模拟重启
        assert revived._cooldown_severity.get("doubao") == SEVERITY_AUTH
        assert revived.is_in_cooldown("doubao") is True
        revived.reset()
    finally:
        _reset_all(tracker)
    print("test_cooldown_severity_survives_restart: PASSED")


def test_record_success_clears_fingerprint_record():
    """节点恢复可用后指纹记录失去意义，必须清除，避免日后误判为「已续期」。"""
    _install_fake_credentials()
    _write_credential("glm", "tok_old")
    tracker = _tracker()
    try:
        _auth_fail(tracker, "glm")
        assert get_credential_renewal().has_record("glm") is True
        asyncio.run(tracker.record_success("glm"))
        assert get_credential_renewal().has_record("glm") is False
    finally:
        _reset_all(tracker)
    print("test_record_success_clears_fingerprint_record: PASSED")


# ------------------------------------------------------------------ 4. 开关

def test_disable_switch():
    os.environ["GATEWAY_DISABLE_CREDENTIAL_RENEWAL"] = "1"
    try:
        _install_fake_credentials()
        _write_credential("qwen", "tok_old")
        t = _fresh_tracker()
        t.note_auth_failure("qwen")
        assert t.has_record("qwen") is False, "关闭后不得记录指纹"
        _write_credential("qwen", "tok_new", 1_700_004_000.0)
        assert t.renewed_providers() == []
        assert t.get_diagnostics()["enabled"] is False
    finally:
        os.environ.pop("GATEWAY_DISABLE_CREDENTIAL_RENEWAL", None)
    print("test_disable_switch: PASSED")


def test_co_disabled_with_risk_ledger():
    """与封禁租约 / 凭证失效检测同约定：关闭持久化风控状态时一并关闭。"""
    os.environ["GATEWAY_DISABLE_RISK_LEDGER"] = "1"
    try:
        t = _fresh_tracker()
        assert t.is_disabled() is True
        assert t.renewed_providers() == []
    finally:
        os.environ.pop("GATEWAY_DISABLE_RISK_LEDGER", None)
    print("test_co_disabled_with_risk_ledger: PASSED")


# ------------------------------------------------------------------ 5. 端到端与遥测

def test_admin_endpoints_and_diagnostics():
    _install_fake_credentials()
    _write_credential("qwen", "tok_old")
    tracker = srv.failover_router.cooldown_tracker
    tracker.reset()
    get_credential_renewal().clear_all_for_tests()
    try:
        _auth_fail(tracker, "qwen", 9)
        client = TestClient(srv.app)

        diag = client.get("/v1/diagnostics").json()
        assert "credential_renewal" in diag
        assert diag["credential_renewal"]["enabled"] is True

        report = client.get("/v1/admin/credential-renewal").json()
        assert report["tracked_providers"] == ["qwen"], report
        assert report["renewed_providers"] == []

        # 未续期 -> 扫描是空操作
        scan = client.post("/v1/admin/credential-renewal/scan", json={}).json()
        assert scan["scan_mode"] == "fingerprint"
        assert scan["released"] == []
        assert tracker.is_in_cooldown("qwen") is True

        _write_credential("qwen", "tok_new", 1_700_004_000.0)  # 运维重登
        scan = client.post("/v1/admin/credential-renewal/scan", json={}).json()
        assert scan["released"] == ["qwen"], scan
        assert tracker.is_in_cooldown("qwen") is False

        # 显式模式：用于凭证来自环境变量（无文件可比指纹）的部署
        tracker.reset()
        _auth_fail(tracker, "glm", 9)
        scan = client.post(
            "/v1/admin/credential-renewal/scan", json={"providers": ["glm"]}
        ).json()
        assert scan["scan_mode"] == "explicit"
        assert scan["released"] == ["glm"]
    finally:
        _reset_all(tracker)
    print("test_admin_endpoints_and_diagnostics: PASSED")


def test_clearing_outage_also_releases_auth_avoidance():
    """`DELETE /v1/admin/credential-outage` 的语义就是「我已续期并修好」，
    因此必须在清除全环失效态的同时复位这些节点的认证级避让，否则接口承诺不兑现。"""
    _install_fake_credentials()
    for pk in ("qwen", "glm", "doubao"):
        _write_credential(pk, "tok_old")
    tracker = srv.failover_router.cooldown_tracker
    tracker.reset()
    get_credential_renewal().clear_all_for_tests()
    srv.credential_watch.clear_all_for_tests()
    try:
        client = TestClient(srv.app)
        for pk in ("qwen", "glm", "doubao"):
            _auth_fail(tracker, pk, 9)  # 认证级 24h 冻结 + 触发全环失效态
        assert srv.credential_watch.is_outage_active() is True
        assert tracker.is_in_cooldown("qwen") is True

        body = client.delete("/v1/admin/credential-outage").json()
        assert body["was_active"] is True
        assert sorted(body["released_auth_avoidance"]) == ["doubao", "glm", "qwen"], body
        assert srv.credential_watch.is_outage_active() is False
        for pk in ("qwen", "glm", "doubao"):
            assert tracker.is_in_cooldown(pk) is False, f"{pk} 的认证级避让应已复位"
    finally:
        _reset_all(tracker)
        srv.credential_watch.clear_all_for_tests()
    print("test_clearing_outage_also_releases_auth_avoidance: PASSED")


def run_all():
    test_fingerprint_is_a_content_hash_not_a_save_counter()
    test_fingerprint_meta_fallback_for_unparseable_file()
    test_no_record_is_not_renewed()
    test_unreadable_credential_is_not_renewed()
    test_same_fingerprint_is_not_renewed()
    test_renewal_detected_only_after_credential_changes()
    test_fingerprint_captured_at_failure_time_not_scan_time()
    test_clear_provider()
    test_records_survive_restart()
    test_release_caps_to_24h_freeze_and_unblocks_refresh_gate()
    test_release_never_touches_operator_quarantine_lease()
    test_release_never_touches_waf_or_ratelimit_cooldown()
    test_release_is_noop_without_actual_state()
    test_ledger_fallback_releases_pre_existing_state()
    test_cooldown_severity_survives_restart()
    test_record_success_clears_fingerprint_record()
    test_disable_switch()
    test_co_disabled_with_risk_ledger()
    test_admin_endpoints_and_diagnostics()
    test_clearing_outage_also_releases_auth_avoidance()
    print("\n✅ All credential-renewal tests (OPT-026) passed!")


if __name__ == "__main__":
    try:
        run_all()
    finally:
        # 还原为真实路径解析，避免影响同进程内的其它导入
        cred.credential_path = _original_credential_path
