"""账号级封禁推断与自动租约 (OPT-027) 测试套件

承 OPT-026 继续往下推。OPT-026 解决的是「凭证过期造成的 24h 冻结该被复位」；
但复位之后，如果节点**依然** 401，就会进入一个此前无人注意的循环：
复位 -> 派发 -> 401 -> 重新按 2 的幂次升级 -> 24h -> 复位 -> …… 每轮都触碰一次上游。
而 401 只有两种解释：

  1. 凭证过期  -> 正确处置是重登（`login.py`），OPT-026 已处理好复位；
  2. 账号级封禁 -> 重登**彻底无效**（封禁绑定账号而非令牌），正确处置是零触碰等待。

两种解释的处置完全相反，而「重登是否有效」正是它们的判别式：**凭证（已被证明地）
更新过之后仍被 401 拒绝**，第 1 种解释即被排除，第 2 种成立。这也正是回答
「现在受到风控的 llm 能通过重新登陆恢复使用吗」这个问题的机器化版本——
deepseek 属于第 2 类，重登无效，只能等平台解除。

本套件验证 `providers/failover.py` 与 `providers/quarantine.py` 的接入：
1. **关键安全性质**：普通凭证过期（凭证从未变过）反复 401 **绝不**推断封禁
2. 证据链：失败（凭证 A）-> 重登（B）-> 用 B 再失败 => 证据成立；连续达阈值即推断
3. 每枚更新过的凭证都获得属于自己的机会（证据重置为 1，而非累加到新凭证头上）
4. 推断产出的租约标记为 `source="auto"`，与运维租约在结构上可区分
5. 运维租约优先级更高：不被自动标记覆盖，也**永不**被自动解除
6. 出现**更新**的凭证时自动租约自动解除并重新验证（一次错误推断不会永久冻结好节点）
7. 证据与自动租约指纹跨重启存活（重启不得成为绕过入口）
8. 自动租约生效后禁止派发该节点（端到端零触碰）
9. 开关 `GATEWAY_DISABLE_BAN_INFERENCE`；与租约/台账开关联动关闭
10. 遥测：/v1/diagnostics 的 ban_inference、GET/DELETE /v1/admin/ban-inference

运行: .venv/bin/python tests/test_ban_inference.py
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
for _name in (
    "GATEWAY_DISABLE_BAN_INFERENCE",
    "GATEWAY_DISABLE_QUARANTINE",
    "GATEWAY_DISABLE_RISK_LEDGER",
):
    os.environ.pop(_name, None)
os.environ["QWEN_AUTH_TOKEN"] = "fake_qwen"
os.environ["DEEPSEEK_AUTH_TOKEN"] = "fake_ds"
os.environ["DOUBAO_AUTH_TOKEN"] = "fake_doubao"
os.environ["KIMI_AUTH_TOKEN"] = "fake_kimi"
os.environ["GLM_AUTH_TOKEN"] = "fake_glm"

from fastapi.testclient import TestClient  # noqa: E402

import providers.credential_renewal as cred  # noqa: E402
import providers.failover as fov  # noqa: E402
import server as srv  # noqa: E402
from providers.credential_renewal import get_credential_renewal  # noqa: E402
from providers.quarantine import QuarantineStore  # noqa: E402

# ------------------------------------------------------------------ 夹具

_TMP = Path(tempfile.mkdtemp(prefix="opt027_"))
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
    t = fov.CooldownTracker()
    t.reset()
    get_credential_renewal().clear_all_for_tests()
    t.quarantine.clear_all_for_tests()
    # 本套件会在多个用例里为**不同**节点注入 401，而 OPT-025 的全环失效检测是
    # 进程级单例、按「去重节点数」判定——不清理就会跨用例累积成「整环失效」。
    t.credential_watch.clear_all_for_tests()
    return t


def _fail(tracker, pk, times=1):
    for _ in range(times):
        asyncio.run(tracker.record_failure(pk, 401, "上游 401: token 无效或已过期"))


def _teardown(tracker):
    tracker.reset()
    get_credential_renewal().clear_all_for_tests()
    tracker.quarantine.clear_all_for_tests()
    tracker.credential_watch.clear_all_for_tests()


# ------------------------------------------------------------------ 1. 关键安全性质

def test_plain_expiry_never_infers_a_ban():
    """凭证从未变过时反复 401，**绝不**推断封禁。

    这是本机制最重要的安全性质：一次普通的凭证过期风暴（真实台账里五个节点
    各 54~87 次 401 正是如此）不得被误判为账号级封禁，否则网关会自己把五个
    完好账号全部租出去，容量归零。判别式必须严格依赖「凭证已被证明更新过」。
    """
    _install_fake_credentials()
    _write_credential("qwen", "A")
    tracker = _tracker()
    try:
        _fail(tracker, "qwen", 20)  # 同一枚从未变更的失效凭证，打 20 次
        assert tracker._fresh_credential_rejected == {}, "普通过期不得进入封禁推断路径"
        assert tracker.quarantine.source("qwen") == "", "绝不得施加任何租约"
        assert tracker.ban_inference_report()["stats"]["auto_leases_applied"] == 0
    finally:
        _teardown(tracker)
    print("test_plain_expiry_never_infers_a_ban: PASSED")


def test_first_failure_after_relogin_is_not_yet_enough():
    """重登后的**第一次**失败只是证据的建立，阈值未到前不得施加租约。"""
    _install_fake_credentials()
    _write_credential("deepseek", "A")
    tracker = _tracker()
    try:
        _fail(tracker, "deepseek", 3)  # 过期阶段的失败（不构成证据）
        _write_credential("deepseek", "B", 1_700_005_000.0)  # 运维重登
        _fail(tracker, "deepseek")  # 用新凭证失败 -> 证据成立，计数 1
        assert tracker._fresh_credential_rejected["deepseek"] == 1
        assert tracker.quarantine.source("deepseek") == "", "阈值(2)未到不得施加租约"
    finally:
        _teardown(tracker)
    print("test_first_failure_after_relogin_is_not_yet_enough: PASSED")


# ------------------------------------------------------------------ 2. 推断成立

def test_fresh_credential_rejected_twice_infers_ban():
    """现场复现（deepseek 的形态）：重登 + 重启 -> 复位认证级避让 -> 探路仍 401
    -> 再 401 => 凭证过期已被排除，判定账号级封禁并施加自动租约。"""
    _install_fake_credentials()
    _write_credential("deepseek", "A")
    tracker = _tracker()
    try:
        _fail(tracker, "deepseek", 3)
        _write_credential("deepseek", "B", 1_700_005_000.0)
        # OPT-026：重登后可复位 24h 认证级冻结（这正是 OPT-027 要接住的下半段）
        assert tracker.renewed_providers() == ["deepseek"]
        assert asyncio.run(tracker.release_renewed_credentials()) == ["deepseek"]
        assert tracker.is_in_cooldown("deepseek") is False

        _fail(tracker, "deepseek")  # 探路：新凭证仍被拒（证据 1）
        assert tracker.quarantine.source("deepseek") == ""
        _fail(tracker, "deepseek")  # 第二次（证据 2）-> 达阈值

        assert tracker.quarantine.source("deepseek") == "auto"
        assert tracker.is_quarantined("deepseek") is True
        remaining = tracker.quarantine.remaining("deepseek")
        assert remaining > 3600.0, f"应施加长租约, 实际 {remaining}"
        report = tracker.ban_inference_report()
        assert report["stats"]["auto_leases_applied"] == 1
        assert report["auto_leases"]["deepseek"]["lease_source"] == "auto"
        assert report["hint"], "推断成立必须给出可执行提示"
    finally:
        _teardown(tracker)
    print("test_fresh_credential_rejected_twice_infers_ban: PASSED")


def test_evidence_resets_for_each_new_credential():
    """每枚**更新过**的凭证都获得属于自己的机会：证据重置为 1，不累加到新凭证头上。"""
    _install_fake_credentials()
    _write_credential("glm", "A")
    tracker = _tracker()
    try:
        _fail(tracker, "glm")
        _write_credential("glm", "B", 1_700_005_000.0)
        _fail(tracker, "glm")
        assert tracker._fresh_credential_rejected["glm"] == 1
        # 又换了一枚凭证 C：B 的证据作废，C 重新从 1 开始
        _write_credential("glm", "C", 1_700_006_000.0)
        _fail(tracker, "glm")
        assert tracker._fresh_credential_rejected["glm"] == 1, "新凭证不得继承上一枚的证据"
        assert tracker.quarantine.source("glm") == ""
        # C 再失败一次才达阈值
        _fail(tracker, "glm")
        assert tracker._fresh_credential_rejected["glm"] == 2
        assert tracker.quarantine.source("glm") == "auto"
    finally:
        _teardown(tracker)
    print("test_evidence_resets_for_each_new_credential: PASSED")


def test_threshold_is_configurable_by_env():
    os.environ["GATEWAY_BAN_INFERENCE_MIN_FAILURES"] = "1"
    _install_fake_credentials()
    _write_credential("kimi", "A")
    tracker = _tracker()
    try:
        _fail(tracker, "kimi")
        _write_credential("kimi", "B", 1_700_005_000.0)
        _fail(tracker, "kimi")  # 阈值 1 -> 立即推断
        assert tracker.quarantine.source("kimi") == "auto"
        assert tracker.ban_inference_report()["min_failures"] == 1
    finally:
        _teardown(tracker)
        os.environ["GATEWAY_BAN_INFERENCE_MIN_FAILURES"] = "2"
    print("test_threshold_is_configurable_by_env: PASSED")


# ------------------------------------------------------------------ 3. 与运维租约的关系

def test_operator_lease_takes_precedence_and_is_never_downgraded():
    """运维租约在语义上更高：不被自动标记覆盖，也永不被自动解除。"""
    _install_fake_credentials()
    _write_credential("doubao", "A")
    tracker = _tracker()
    try:
        tracker.quarantine.set_lease("doubao", 3600.0, reason="人工封禁")
        assert tracker.quarantine.source("doubao") == "operator"
        _write_credential("doubao", "B", 1_700_005_000.0)
        _fail(tracker, "doubao", 3)  # 已是 operator 租约，推断不应改写来源
        assert tracker.quarantine.source("doubao") == "operator", "自动标记绝不得覆盖运维声明"
        # 自动解除入口对运维租约必须无效
        assert tracker.quarantine.release_auto_lease("doubao") is False
        assert tracker.is_quarantined("doubao") is True
    finally:
        _teardown(tracker)
    print("test_operator_lease_takes_precedence_and_is_never_downgraded: PASSED")


def test_quarantine_lease_source_survives_reload():
    """租约来源本身必须持久化：重启后仍能区分自动与运维（否则自动租约无法被解除）。"""
    directory = Path(tempfile.mkdtemp(prefix="opt027_q_"))
    store = QuarantineStore(directory=directory)
    store.set_lease("qwen", 7200.0, reason="auto:ban-inferred", source="auto")
    store.set_lease("glm", 7200.0, reason="人工", source="operator")

    revived = QuarantineStore(directory=directory)
    assert revived.source("qwen") == "auto"
    assert revived.source("glm") == "operator"
    assert revived.list_active()["qwen"]["source"] == "auto"
    # 未知/缺失来源一律按运维租约处理（保守）
    (directory / "quarantine.json").write_text(
        json.dumps({"leases": {"kimi": {"until": time.time() + 7200.0, "reason": ""}}}),
        encoding="utf-8",
    )
    assert QuarantineStore(directory=directory).source("kimi") == "operator"
    print("test_quarantine_lease_source_survives_reload: PASSED")


# ------------------------------------------------------------------ 4. 自动解除

def test_auto_lease_lifted_when_a_newer_credential_appears():
    """自动租约的正当性是「已试过一枚新凭证且仍被拒」；再出现更新的凭证时，
    该证据只针对旧凭证，自动租约必须解除并让新凭证重新接受验证——
    一次错误推断绝不允许永久冻结一个其实已经修好的节点。"""
    _install_fake_credentials()
    _write_credential("deepseek", "A")
    tracker = _tracker()
    try:
        _fail(tracker, "deepseek")
        _write_credential("deepseek", "B", 1_700_005_000.0)
        _fail(tracker, "deepseek", 2)
        assert tracker.quarantine.source("deepseek") == "auto"

        _write_credential("deepseek", "C", 1_700_007_000.0)  # 运维再次重登
        assert tracker.renewed_providers() == ["deepseek"]
        released = asyncio.run(tracker.release_renewed_credentials())
        assert released == ["deepseek"], released
        assert tracker.quarantine.source("deepseek") == "", "自动租约应被解除"
        assert tracker.is_quarantined("deepseek") is False
        assert tracker._fresh_credential_rejected == {}, "证据随自动租约一并作废"
        assert tracker.is_in_probation("deepseek") is True
        assert tracker.ban_inference_report()["stats"]["auto_leases_lifted"] == 1

        # 解除后必须真的可以重新派发（新凭证获得完整机会）
        assert tracker.get_remaining_cooldown("deepseek") == 0.0
    finally:
        _teardown(tracker)
    print("test_auto_lease_lifted_when_a_newer_credential_appears: PASSED")


def test_auto_lease_not_lifted_without_a_newer_credential():
    """没有更新的凭证证据时，自动租约必须维持——否则封禁等于没封。"""
    _install_fake_credentials()
    _write_credential("deepseek", "A")
    tracker = _tracker()
    try:
        _fail(tracker, "deepseek")
        _write_credential("deepseek", "B", 1_700_005_000.0)
        _fail(tracker, "deepseek", 2)
        assert tracker.quarantine.source("deepseek") == "auto"
        assert tracker.renewed_providers() == [], "凭证未再变化"
        assert asyncio.run(tracker.release_renewed_credentials()) == []
        assert tracker.is_quarantined("deepseek") is True
    finally:
        _teardown(tracker)
    print("test_auto_lease_not_lifted_without_a_newer_credential: PASSED")


# ------------------------------------------------------------------ 5. 跨重启

def test_evidence_and_auto_fingerprint_survive_restart():
    """重启不得成为绕过入口：证据与自动租约指纹都必须跨界存活。"""
    _install_fake_credentials()
    _write_credential("deepseek", "A")
    tracker = _tracker()
    try:
        _fail(tracker, "deepseek")
        _write_credential("deepseek", "B", 1_700_005_000.0)
        _fail(tracker, "deepseek")
        assert tracker._fresh_credential_rejected["deepseek"] == 1
        tracker._persist_state()

        revived = fov.CooldownTracker()  # 模拟重启（租约由单例存储，此处只验状态）
        assert revived._fresh_credential_rejected.get("deepseek") == 1, "证据必须跨界存活"
        assert revived._auto_lease_fingerprint.get("deepseek") is None  # 尚未施加租约
        # 重启后用同一枚新凭证再失败一次 -> 直接达阈值（证据没有被重启抹掉）
        _fail(revived, "deepseek")
        assert revived.quarantine.source("deepseek") == "auto"
        # 自动租约指纹也随之落盘，供日后判定「依据是否失效」
        assert revived._auto_lease_fingerprint.get("deepseek") is not None
        revived._persist_state()
        assert fov.CooldownTracker()._auto_lease_fingerprint.get("deepseek") is not None
        revived.reset()
    finally:
        _teardown(tracker)
    print("test_evidence_and_auto_fingerprint_survive_restart: PASSED")


# ------------------------------------------------------------------ 6. 开关

def test_disable_switch():
    os.environ["GATEWAY_DISABLE_BAN_INFERENCE"] = "1"
    _install_fake_credentials()
    _write_credential("deepseek", "A")
    tracker = _tracker()
    try:
        _fail(tracker, "deepseek")
        _write_credential("deepseek", "B", 1_700_005_000.0)
        _fail(tracker, "deepseek", 5)
        assert tracker.quarantine.source("deepseek") == "", "关闭后不得推断封禁"
        assert tracker._fresh_credential_rejected == {}
        assert tracker.ban_inference_report()["enabled"] is False
    finally:
        _teardown(tracker)
        os.environ.pop("GATEWAY_DISABLE_BAN_INFERENCE", None)
    print("test_disable_switch: PASSED")


def test_co_disabled_with_quarantine_and_ledger():
    """与租约/台账同约定：它们被关闭时推断出的结论也无处安放，一并关闭。"""
    _install_fake_credentials()
    _write_credential("deepseek", "A")
    for name in ("GATEWAY_DISABLE_QUARANTINE", "GATEWAY_DISABLE_RISK_LEDGER"):
        os.environ[name] = "1"
        try:
            tracker = _tracker()
            assert tracker.is_ban_inference_disabled() is True
            _fail(tracker, "deepseek")
            _write_credential("deepseek", "B", 1_700_005_000.0)
            _fail(tracker, "deepseek", 3)
            assert tracker.quarantine.source("deepseek") == ""
            tracker.reset()
        finally:
            os.environ.pop(name, None)
    print("test_co_disabled_with_quarantine_and_ledger: PASSED")


# ------------------------------------------------------------------ 7. 端到端与遥测

class _KeyedProvider:
    """按厂商键记录调用：用于证明被封节点**零触碰**。"""

    calls: list = []

    def __init__(self, key: str) -> None:
        self.key = key

    async def chat(self, *a, **kw):
        type(self).calls.append(self.key)
        yield "你好", {}


def test_auto_lease_blocks_dispatch_end_to_end():
    """推断出的租约与运维租约一样走 OPT-021 硬阻断：被封节点零上游调用。"""
    _install_fake_credentials()
    _write_credential("deepseek", "A")
    tracker = srv.failover_router.cooldown_tracker
    tracker.reset()
    get_credential_renewal().clear_all_for_tests()
    tracker.quarantine.clear_all_for_tests()
    tracker.credential_watch.clear_all_for_tests()
    original = srv._make_provider
    _KeyedProvider.calls = []
    srv._make_provider = lambda key: _KeyedProvider(key)
    try:
        _fail(tracker, "deepseek")
        _write_credential("deepseek", "B", 1_700_005_000.0)
        _fail(tracker, "deepseek", 2)
        assert tracker.quarantine.source("deepseek") == "auto"

        client = TestClient(srv.app)
        r = client.post(
            "/v1/chat/completions",
            json={"model": "deepseek-reasoner", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 200, r.text
        assert "deepseek" not in _KeyedProvider.calls, (
            f"被封节点不得被触碰, 实际调用: {_KeyedProvider.calls}"
        )
        assert _KeyedProvider.calls, "应故障转移至同环的其它健康节点"
    finally:
        srv._make_provider = original
        tracker.quarantine.clear_all_for_tests()
        tracker.reset()
        get_credential_renewal().clear_all_for_tests()
        tracker.credential_watch.clear_all_for_tests()
    print("test_auto_lease_blocks_dispatch_end_to_end: PASSED")


def test_diagnostics_and_admin_endpoints():
    _install_fake_credentials()
    _write_credential("deepseek", "A")
    tracker = srv.failover_router.cooldown_tracker
    tracker.reset()
    get_credential_renewal().clear_all_for_tests()
    tracker.quarantine.clear_all_for_tests()
    tracker.credential_watch.clear_all_for_tests()
    try:
        client = TestClient(srv.app)
        diag = client.get("/v1/diagnostics").json()
        assert "ban_inference" in diag
        assert diag["ban_inference"]["enabled"] is True
        assert diag["ban_inference"]["auto_leases"] == {}

        _fail(tracker, "deepseek")
        _write_credential("deepseek", "B", 1_700_005_000.0)
        _fail(tracker, "deepseek", 2)

        report = client.get("/v1/admin/ban-inference").json()
        assert "deepseek" in report["auto_leases"]
        assert report["auto_leases"]["deepseek"]["lease_source"] == "auto"
        assert report["rejections_after_renewal"] == {"deepseek": 2}
        assert "重登" in report["hint"]

        # 运维确认平台已解封 -> 提前解除自动租约
        cleared = client.delete("/v1/admin/ban-inference").json()
        assert cleared["cleared"] == ["deepseek"], cleared
        assert tracker.quarantine.source("deepseek") == ""
        assert tracker.ban_inference_report()["auto_leases"] == {}

        # 运维租约绝不被该接口解除
        tracker.quarantine.set_lease("glm", 3600.0, reason="人工")
        cleared = client.delete("/v1/admin/ban-inference").json()
        assert cleared["cleared"] == []
        assert tracker.is_quarantined("glm") is True
    finally:
        tracker.quarantine.clear_all_for_tests()
        tracker.reset()
        get_credential_renewal().clear_all_for_tests()
        tracker.credential_watch.clear_all_for_tests()
    print("test_diagnostics_and_admin_endpoints: PASSED")


def run_all():
    test_plain_expiry_never_infers_a_ban()
    test_first_failure_after_relogin_is_not_yet_enough()
    test_fresh_credential_rejected_twice_infers_ban()
    test_evidence_resets_for_each_new_credential()
    test_threshold_is_configurable_by_env()
    test_operator_lease_takes_precedence_and_is_never_downgraded()
    test_quarantine_lease_source_survives_reload()
    test_auto_lease_lifted_when_a_newer_credential_appears()
    test_auto_lease_not_lifted_without_a_newer_credential()
    test_evidence_and_auto_fingerprint_survive_restart()
    test_disable_switch()
    test_co_disabled_with_quarantine_and_ledger()
    test_auto_lease_blocks_dispatch_end_to_end()
    test_diagnostics_and_admin_endpoints()
    print("\n✅ All ban-inference tests (OPT-027) passed!")


if __name__ == "__main__":
    try:
        run_all()
    finally:
        cred.credential_path = _original_credential_path
