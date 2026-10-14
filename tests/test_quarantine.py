"""运维人工长期封禁租约 (OPT-020) 测试套件

验证：
1. 时长表达式解析（s/m/h/d 与纯数字）与 ISO-8601 绝对时间解析
2. 租约设定后立即生效，且 POSIX 原子落盘（文件 0o600 / 目录 0o700）
3. 跨进程重启从磁盘恢复租约（重启不得放行仍在封禁窗口的节点）
4. 租约只延长不缩短：更短的租约不得覆盖更长的既有租约
5. 到期自动失效（惰性清理），剩余时间归零
6. 解除租约（单个 / 全部）
7. 折入 CooldownTracker：is_in_cooldown / get_remaining_cooldown 取最大值，
   自动冷却与人工租约互不缩短
8. 所有既有门控自动遵循租约（探针 / 会话回收 / 自愈选举）
9. GATEWAY_QUARANTINE 环境变量启动即生效
10. GATEWAY_DISABLE_QUARANTINE=1 关闭租约
11. 服务端接线：诊断暴露 /health 与 /v1/diagnostics，管理端点多/删/查
12. 未知 Provider 与非法表达式返回 4xx，不得静默成功

运行: .venv/bin/python tests/test_quarantine.py
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import stat
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["TESTING"] = "1"
os.environ["GATEWAY_DISABLE_AUTO_REFRESH"] = "1"
os.environ["GATEWAY_DISABLE_PACING"] = "1"
os.environ["QWEN_AUTH_TOKEN"] = "fake_qwen"
os.environ["DEEPSEEK_AUTH_TOKEN"] = "fake_ds"
os.environ["DOUBAO_AUTH_TOKEN"] = "fake_doubao"
os.environ["KIMI_AUTH_TOKEN"] = "fake_kimi"
os.environ["GLM_AUTH_TOKEN"] = "fake_glm"
os.environ.pop("GATEWAY_QUARANTINE", None)
os.environ.pop("GATEWAY_DISABLE_QUARANTINE", None)
os.environ.pop("GATEWAY_DISABLE_RISK_LEDGER", None)

from fastapi.testclient import TestClient  # noqa: E402

import server as srv  # noqa: E402
from providers.quarantine import QuarantineStore, parse_quarantine_until  # noqa: E402


def _fresh(tmp: Path) -> QuarantineStore:
    return QuarantineStore(directory=tmp)


# ------------------------------------------------------------------ 1. 表达式解析

def test_parse_duration_expressions():
    now = 1_000_000.0
    assert parse_quarantine_until("45", now) == now + 45
    assert parse_quarantine_until("30s", now) == now + 30
    assert parse_quarantine_until("30m", now) == now + 1800
    assert parse_quarantine_until("72h", now) == now + 72 * 3600
    assert parse_quarantine_until("3d", now) == now + 3 * 86400
    assert parse_quarantine_until("1.5h", now) == now + 5400
    print("test_parse_duration_expressions: PASSED")


def test_parse_absolute_time_and_invalid():
    parsed = parse_quarantine_until("2026-10-20T00:00:00")
    assert parsed is not None and parsed > 0
    assert parsed == parse_quarantine_until("2026-10-20"), "纯日期等价于当天本地 00:00"
    assert parse_quarantine_until("2026-10-20 06:30:00") > parsed, "空格分隔亦须解析"
    assert parse_quarantine_until("2026-10-20T00:00:00Z") > parsed, "Z 表示 UTC 须解析"
    for bad in ("", "   ", "not-a-time", "abc123"):
        assert parse_quarantine_until(bad) is None, f"{bad!r} 必须解析失败"
    print("test_parse_absolute_time_and_invalid: PASSED")


# ------------------------------------------------------------------ 2/3. 落盘与恢复

def test_lease_persists_atomically_and_recovers():
    tmp = Path(tempfile.mkdtemp(prefix="quar_test_"))
    store = _fresh(tmp)
    store.set_lease("deepseek", duration_seconds=72 * 3600, reason="账号级封禁")

    assert store.path.exists(), "租约必须落盘"
    mode = stat.S_IMODE(store.path.stat().st_mode)
    assert mode == 0o600, f"租约文件权限必须 0o600，实际 {oct(mode)}"
    dirmode = stat.S_IMODE(tmp.stat().st_mode)
    assert dirmode == 0o700, f"租约目录权限必须 0o700，实际 {oct(dirmode)}"

    data = json.loads(store.path.read_text(encoding="utf-8"))
    assert "deepseek" in data["leases"]

    # 模拟进程重启
    revived = QuarantineStore(directory=tmp)
    remaining = revived.remaining("deepseek")
    assert remaining > 71 * 3600, f"重启后租约必须恢复，实际剩余 {remaining}"
    assert revived.remaining("qwen") == 0.0
    assert revived.list_active()["deepseek"]["reason"] == "账号级封禁"
    shutil.rmtree(tmp, ignore_errors=True)
    print("test_lease_persists_atomically_and_recovers: PASSED")


def test_expired_lease_not_restored():
    tmp = Path(tempfile.mkdtemp(prefix="quar_test_"))
    store = _fresh(tmp)
    store.set_lease("glm", until=time.time() - 10)  # 已过期
    revived = QuarantineStore(directory=tmp)
    assert revived.remaining("glm") == 0.0, "过期租约不得在重启后复活"
    shutil.rmtree(tmp, ignore_errors=True)
    print("test_expired_lease_not_restored: PASSED")


def test_corrupt_lease_file_is_ignored():
    tmp = Path(tempfile.mkdtemp(prefix="quar_test_"))
    (tmp / "quarantine.json").write_text("{not-json", encoding="utf-8")
    store = QuarantineStore(directory=tmp)  # 不得抛异常
    assert store.remaining("qwen") == 0.0
    shutil.rmtree(tmp, ignore_errors=True)
    print("test_corrupt_lease_file_is_ignored: PASSED")


# ------------------------------------------------------------------ 4/5/6. 语义

def test_lease_extends_but_never_shortens():
    tmp = Path(tempfile.mkdtemp(prefix="quar_test_"))
    store = _fresh(tmp)
    long_until = store.set_lease("kimi", duration_seconds=72 * 3600)

    # 更短的租约不得缩短既有窗口
    shorter = store.set_lease("kimi", duration_seconds=60)
    assert shorter == long_until, "短租约不得覆盖长租约"
    assert store.remaining("kimi") > 71 * 3600

    # 更长的租约可以延长
    longer = store.set_lease("kimi", until=time.time() + 100 * 3600)
    assert longer > long_until
    assert store.remaining("kimi") > 99 * 3600
    shutil.rmtree(tmp, ignore_errors=True)
    print("test_lease_extends_but_never_shortens: PASSED")


def test_lease_expires_lazily():
    tmp = Path(tempfile.mkdtemp(prefix="quar_test_"))
    store = _fresh(tmp)
    store.set_lease("doubao", until=time.time() + 0.05)
    assert store.remaining("doubao") > 0
    time.sleep(0.08)
    assert store.remaining("doubao") == 0.0, "到期后必须自动失效"
    assert store.list_active() == {}
    assert store.get_diagnostics()["stats"]["expired"] >= 1
    shutil.rmtree(tmp, ignore_errors=True)
    print("test_lease_expires_lazily: PASSED")


def test_clear_lease():
    tmp = Path(tempfile.mkdtemp(prefix="quar_test_"))
    store = _fresh(tmp)
    store.set_lease("qwen", duration_seconds=3600)
    store.set_lease("glm", duration_seconds=3600)

    assert store.clear("qwen") == 1
    assert store.remaining("qwen") == 0.0
    assert store.remaining("glm") > 0
    assert store.clear() == 1, "全部解除应返回剩余数量"
    assert store.list_active() == {}
    assert store.clear() == 0, "空租约解除应为幂等 0"
    shutil.rmtree(tmp, ignore_errors=True)
    print("test_clear_lease: PASSED")


def test_set_lease_requires_provider():
    tmp = Path(tempfile.mkdtemp(prefix="quar_test_"))
    store = _fresh(tmp)
    for bad in ("", "   "):
        try:
            store.set_lease(bad, duration_seconds=10)
        except ValueError:
            pass
        else:
            raise AssertionError("空 provider 必须抛 ValueError")
    shutil.rmtree(tmp, ignore_errors=True)
    print("test_set_lease_requires_provider: PASSED")


# ------------------------------------------------------------------ 7/8. 折入冷却与门控

def test_lease_folds_into_cooldown_tracker():
    tmp = Path(tempfile.mkdtemp(prefix="quar_test_"))
    store = _fresh(tmp)
    from providers.failover import CooldownTracker

    tracker = CooldownTracker()
    tracker.quarantine = store  # 显式注入隔离实例
    try:
        assert tracker.is_in_cooldown("deepseek") is False
        store.set_lease("deepseek", duration_seconds=72 * 3600)

        assert tracker.is_in_cooldown("deepseek") is True, "人工租约必须折入冷却判定"
        assert tracker.get_remaining_cooldown("deepseek") > 71 * 3600
        assert tracker.get_remaining_cooldown("qwen") == 0.0, "未租约节点不得被误伤"

        # 自动冷却与人工租约取最大值：短冷却不得缩短长租约
        asyncio.run(tracker.record_failure("deepseek", 429, "rate limit"))
        assert tracker.get_remaining_cooldown("deepseek") > 71 * 3600
        # 短租约也不得缩短既有的长冷却
        asyncio.run(tracker.record_failure("qwen", 429, "rate limit"))
        cd_before = tracker.get_remaining_cooldown("qwen")
        store.set_lease("qwen", duration_seconds=1)
        assert tracker.get_remaining_cooldown("qwen") >= cd_before - 1.0
    finally:
        tracker.reset()
        shutil.rmtree(tmp, ignore_errors=True)
    print("test_lease_folds_into_cooldown_tracker: PASSED")


def test_existing_gates_honour_lease():
    """租约必须被已有门控自动遵循（探针 / 会话回收 / 自愈选举），无需逐处接线。"""
    store = srv.quarantine_store
    monitor = srv.health_monitor
    tracker = srv.failover_router.cooldown_tracker
    store.clear_all_for_tests()
    tracker.reset()
    try:
        store.set_lease("deepseek", duration_seconds=72 * 3600)

        # 1) 健康探针门控
        from unittest.mock import AsyncMock

        probe = AsyncMock(return_value=(True, None))
        original = monitor.probe_provider
        monitor.probe_provider = probe
        try:
            is_healthy, err = asyncio.run(monitor.check_provider("deepseek"))
            assert probe.call_count == 0, "租约期内绝不能发起真实探针"
            assert is_healthy is False and "冷却" in err
        finally:
            monitor.probe_provider = original

        # 2) 会话回收门控
        gc = srv.conversation_gc
        assert gc._cooldown_remaining("deepseek") > 0, "会话回收必须被租约门控"

        # 3) 自愈大脑选举
        from healing.healer import pick_healer_model

        picked = pick_healer_model(monitor, "qwen")
        assert picked != "deepseek-chat", "被封禁租约的节点绝不能被选为自愈引擎"

        # 4) 路由选点：[OPT-021] 封禁节点被彻底排除出后备候选（硬阻断），
        #    冷却读数仍准确（供审计视图与门控使用）
        cands = srv.failover_router.get_fallback_candidates("glm", "glm-4-plus")
        assert "deepseek" not in [c.provider_key for c in cands], (
            "封禁租约节点绝不得出现在路由后备候选中"
        )
        audit = srv.failover_router.get_fallback_candidates(
            "glm", "glm-4-plus", include_quarantined=True, include_long_avoidance=True
        )
        cd_cands = [c for c in audit if c.in_cooldown]
        assert any(c.provider_key == "deepseek" for c in cd_cands), (
            "审计视图中封禁节点必须仍被标记为冷却"
        )
    finally:
        store.clear_all_for_tests()
        tracker.reset()
    print("test_existing_gates_honour_lease: PASSED")


# ------------------------------------------------------------------ 9/10. 环境变量与开关

def test_env_quarantine_applied_on_startup():
    tmp = Path(tempfile.mkdtemp(prefix="quar_test_"))
    os.environ["GATEWAY_QUARANTINE"] = "deepseek:72h, glm:30m,broken,kimi:2030-01-01T00:00:00"
    try:
        store = _fresh(tmp)
        assert store.remaining("deepseek") > 71 * 3600
        assert store.remaining("glm") > 29 * 60
        assert store.remaining("kimi") > 0
        assert store.remaining("qwen") == 0.0
        # 无效条目被跳过而非整体失败
        assert store.get_diagnostics()["active_count"] == 3
    finally:
        os.environ.pop("GATEWAY_QUARANTINE", None)
        shutil.rmtree(tmp, ignore_errors=True)
    print("test_env_quarantine_applied_on_startup: PASSED")


def test_disable_switch():
    tmp = Path(tempfile.mkdtemp(prefix="quar_test_"))
    os.environ["GATEWAY_DISABLE_QUARANTINE"] = "1"
    try:
        store = _fresh(tmp)
        store.set_lease("qwen", duration_seconds=3600)
        assert not store.path.exists(), "关闭后不得落盘"
        assert store.get_diagnostics()["persist_enabled"] is False
    finally:
        os.environ.pop("GATEWAY_DISABLE_QUARANTINE", None)
        shutil.rmtree(tmp, ignore_errors=True)
    print("test_disable_switch: PASSED")


# ------------------------------------------------------------------ 11/12. 服务端

def test_server_diagnostics_and_admin_endpoints():
    store = srv.quarantine_store
    store.clear_all_for_tests()
    client = TestClient(srv.app)
    try:
        # 诊断暴露
        health = client.get("/health").json()
        assert "quarantine" in health
        diag = client.get("/v1/diagnostics").json()
        assert "quarantine" in diag
        assert diag["quarantine"]["active_count"] == 0

        # 设定：按小时
        r = client.post(
            "/v1/admin/quarantine",
            json={"provider": "deepseek", "duration_hours": 72, "reason": "平台封禁待解"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["provider"] == "deepseek"
        assert body["remaining_hours"] > 71

        # 生效于运行期状态与冷却读数
        assert store.remaining("deepseek") > 71 * 3600
        assert srv.failover_router.cooldown_tracker.is_in_cooldown("deepseek") is True
        assert client.get("/v1/diagnostics").json()["quarantine"]["active_count"] == 1

        # 设定：时长表达式
        r2 = client.post("/v1/admin/quarantine", json={"provider": "glm", "duration": "3d"})
        assert r2.status_code == 200
        assert r2.json()["remaining_hours"] > 71

        # 解除单个
        r3 = client.delete("/v1/admin/quarantine?provider=deepseek")
        assert r3.status_code == 200
        assert r3.json()["cleared"] == 1
        assert store.remaining("deepseek") == 0.0
        assert store.remaining("glm") > 0

        # 非法输入不得静默成功
        assert client.post("/v1/admin/quarantine", json={"provider": "nope", "duration_hours": 1}).status_code == 404
        assert client.post("/v1/admin/quarantine", json={"provider": "kimi", "duration": "???"}).status_code == 400
        assert client.post("/v1/admin/quarantine", json={"provider": "kimi"}).status_code == 400
        assert store.remaining("kimi") == 0.0

        # 全部解除
        r4 = client.delete("/v1/admin/quarantine")
        assert r4.status_code == 200
        assert r4.json()["active"] == {}
    finally:
        store.clear_all_for_tests()
    print("test_server_diagnostics_and_admin_endpoints: PASSED")


def test_server_clear_does_not_undo_automatic_cooldown():
    """解除人工租约不得连带清除真实风控产生的自动冷却。"""
    store = srv.quarantine_store
    tracker = srv.failover_router.cooldown_tracker
    store.clear_all_for_tests()
    tracker.reset()
    try:
        store.set_lease("doubao", duration_seconds=72 * 3600)
        asyncio.run(tracker.record_failure("doubao", 401, "Token expired"))

        client = TestClient(srv.app)
        assert client.delete("/v1/admin/quarantine?provider=doubao").json()["cleared"] == 1
        assert tracker.is_in_cooldown("doubao") is True, (
            "解除人工租约后，真实风控自动冷却仍须继续生效"
        )
    finally:
        store.clear_all_for_tests()
        tracker.reset()
    print("test_server_clear_does_not_undo_automatic_cooldown: PASSED")


def run_all():
    test_parse_duration_expressions()
    test_parse_absolute_time_and_invalid()
    test_lease_persists_atomically_and_recovers()
    test_expired_lease_not_restored()
    test_corrupt_lease_file_is_ignored()
    test_lease_extends_but_never_shortens()
    test_lease_expires_lazily()
    test_clear_lease()
    test_set_lease_requires_provider()
    test_lease_folds_into_cooldown_tracker()
    test_existing_gates_honour_lease()
    test_env_quarantine_applied_on_startup()
    test_disable_switch()
    test_server_diagnostics_and_admin_endpoints()
    test_server_clear_does_not_undo_automatic_cooldown()
    print("\n✅ All operator quarantine lease tests (OPT-020) passed!")


if __name__ == "__main__":
    run_all()
