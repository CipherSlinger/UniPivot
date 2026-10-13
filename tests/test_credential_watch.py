"""全环凭证失效检测与快速失败 (OPT-025) 测试套件

动机（来自真实台账的现场证据）：
`session/risk_ledger.jsonl` 显示 2026-10-13 18:25~19:44 的 80 分钟内五个节点被记录 335 次
拦截，其中 `auth` 类占 312 次，且**五个节点在同一秒内同时返回 401**，该模式每约 7 分钟
重复一次。这不是五次独立的令牌过期（概率上不可能），真实过程是：**一个客户端请求沿能力
环依次故障转移，五个节点全部以 401 拒绝**，于是同一请求在 1 秒内产生 5 条 auth 记录。
也就是说整环凭证同时失效、容量为零，故障转移不可能成功。

危害不止于「浪费」：环内无可用凭证时网关仍会为每个客户端请求打满五个节点，这 335 次
带无效凭证的失败请求恰好是上游 WAF 眼中**「自动化凭证尝试」的流量特征**——风暴本身
就在持续拉高风控评分。因此正确行为不是「更努力地重试」而是**立即停止**。

本套件验证 `providers/credential_watch.py` 与 `providers/failover.py` 的接入：
1. 关联性来自**去重节点数**而非失败次数（同一节点失败 100 次仍是单节点问题）
2. 阈值语义：≥ `GATEWAY_CREDENTIAL_OUTAGE_MIN_PROVIDERS` 个不同节点在窗口内相继 401 才进入失效态
3. 现场复现：单个客户端请求沿环走满五个节点即触发（这正是真实故障的形态）
4. **永不卡死**：TTL 到期自动放行探路；探路失败立即重建失效态；任一节点恢复即恢复派发
5. 零上游调用：失效态下请求被拒绝且不产生任何 Provider 实例化与 chat 调用
6. 端到端错误语义：OpenAI `err_type=credential_outage` / Anthropic `authentication_error`，HTTP 503
7. 开关 `GATEWAY_DISABLE_CREDENTIAL_WATCH`；与 `GATEWAY_DISABLE_RISK_LEDGER` 联动关闭
8. 遥测：/v1/diagnostics、GET/DELETE /v1/admin/credential-outage

运行: .venv/bin/python tests/test_credential_watch.py
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
os.environ.pop("GATEWAY_DISABLE_CREDENTIAL_WATCH", None)
os.environ.pop("GATEWAY_DISABLE_RISK_LEDGER", None)
os.environ["QWEN_AUTH_TOKEN"] = "fake_qwen"
os.environ["DEEPSEEK_AUTH_TOKEN"] = "fake_ds"
os.environ["DOUBAO_AUTH_TOKEN"] = "fake_doubao"
os.environ["KIMI_AUTH_TOKEN"] = "fake_kimi"
os.environ["GLM_AUTH_TOKEN"] = "fake_glm"

from fastapi.testclient import TestClient  # noqa: E402

import providers.failover as fov  # noqa: E402
import server as srv  # noqa: E402
from providers.anthropic_compat import map_anthropic_error_type  # noqa: E402
from providers.credential_watch import CredentialOutageWatch  # noqa: E402

ALL_FIVE = ["deepseek", "doubao", "glm", "kimi", "qwen"]


class _CountingProvider:
    """计数假 Provider：任何实例化或 chat 调用都会被记录，用于证明「零上游调用」。"""

    instances = 0
    chats = 0

    def __init__(self) -> None:
        type(self).instances += 1

    async def chat(self, *a, **kw):
        type(self).chats += 1
        yield "不该被调用", {}


class _EchoProvider:
    async def chat(self, *a, **kw):
        yield "你好", {}


def _fresh() -> CredentialOutageWatch:
    return CredentialOutageWatch()


def _arm(watch, providers=None, now=None) -> None:
    for pk in providers or ["qwen", "glm", "kimi"]:
        watch.note_auth_failure(pk, now)


# ------------------------------------------------------------------ 1. 去重语义

def test_one_provider_never_trips():
    """同一节点反复 401 一百次，仍是「单节点问题」——不得进入全环失效态。"""
    w = _fresh()
    t0 = 1000.0
    for i in range(100):
        w.note_auth_failure("qwen", t0 + i)
    assert w.is_outage_active(t0 + 99) is False
    assert w.active_failures(t0 + 99) == ["qwen"]
    d = w.get_diagnostics(t0 + 99)
    assert d["outage_active"] is False
    assert d["stats"]["auth_failures"] == 100
    assert d["stats"]["outages_entered"] == 0
    assert d["hint"] == "", "未达阈值不应给出重登指引"
    print("test_one_provider_never_trips: PASSED")


def test_two_providers_below_threshold():
    """两个不同节点仍低于默认阈值 (3)，可正常故障转移，不进入失效态。"""
    w = _fresh()
    t0 = 1000.0
    _arm(w, ["qwen", "glm"], t0)
    assert w.is_outage_active(t0) is False
    assert len(w.active_failures(t0)) == 2
    print("test_two_providers_below_threshold: PASSED")


def test_three_distinct_providers_trip():
    w = _fresh()
    t0 = 1000.0
    _arm(w, ["qwen", "glm"], t0)
    assert w.is_outage_active(t0) is False
    w.note_auth_failure("kimi", t0)
    assert w.is_outage_active(t0) is True
    assert w.active_failures(t0) == ["glm", "kimi", "qwen"]
    d = w.get_diagnostics(t0)
    assert d["stats"]["outages_entered"] == 1
    assert "login.py all" in d["hint"], d["hint"]
    print("test_three_distinct_providers_trip: PASSED")


def test_one_client_request_walking_the_ring_is_the_trigger():
    """现场复现：**一个**客户端请求沿能力环走满五个节点，全部 401。

    五个节点在同一秒内被记录，这不是五次独立过期——是同一个请求的故障转移轨迹。
    这正是 OPT-025 要识别的形态，也是「风暴本身就是风控触发器」的来源。
    """
    w = _fresh()
    same_second = 1_700_000_000.0
    for pk in ("qwen", "glm", "doubao", "kimi", "deepseek"):
        w.note_auth_failure(pk, same_second)  # 同一请求，1 秒内走完五跳
    assert w.is_outage_active(same_second + 0.5) is True
    assert w.active_failures(same_second) == ALL_FIVE
    assert w.get_diagnostics(same_second)["failed_count"] == 5
    print("test_one_client_request_walking_the_ring_is_the_trigger: PASSED")


# ------------------------------------------------------------------ 2. 永不卡死

def test_ttl_self_expiry_allows_probe_then_rearms():
    """TTL 到期必须放行探路（绝不因状态卡死而永久拒绝服务）；探路失败立即重建。"""
    w = _fresh()
    t0 = 1000.0
    _arm(w, None, t0)
    assert w.is_outage_active(t0) is True
    assert w.is_outage_active(t0 + 60) is True, "TTL(120s) 内不得放行"
    assert w.is_outage_active(t0 + 121) is False, "TTL 到期必须放行探路"
    assert w.get_diagnostics(t0 + 121)["stats"]["expired_by_ttl"] == 1
    # 探路请求再次 401 -> 立即重建失效态，探测频率退化为「每 TTL 一次」
    w.note_auth_failure("qwen", t0 + 122)
    assert w.is_outage_active(t0 + 122) is True
    print("test_ttl_self_expiry_allows_probe_then_rearms: PASSED")


def test_window_prune_exits_outage():
    """窗口外证据自然衰减后退出失效态（窗口 < TTL 时的另一条退出路径）。"""
    os.environ["GATEWAY_CREDENTIAL_OUTAGE_WINDOW"] = "60"
    os.environ["GATEWAY_CREDENTIAL_OUTAGE_TTL"] = "3600"
    try:
        w = _fresh()
        t0 = 1000.0
        _arm(w, None, t0)
        assert w.is_outage_active(t0 + 59) is True
        assert w.is_outage_active(t0 + 61) is False, "证据超窗后应退出失效态"
        assert w.active_failures(t0 + 61) == []
    finally:
        os.environ["GATEWAY_CREDENTIAL_OUTAGE_WINDOW"] = "300"
        os.environ["GATEWAY_CREDENTIAL_OUTAGE_TTL"] = "120"
    print("test_window_prune_exits_outage: PASSED")


def test_any_recovery_resumes_dispatch():
    """任一节点能用有效凭证成功 -> 容量非零 -> 立即恢复派发（fail-open，绝不卡死）。"""
    w = _fresh()
    t0 = 1000.0
    _arm(w, None, t0)
    assert w.is_outage_active(t0) is True
    w.note_success("glm", t0 + 1)
    assert w.is_outage_active(t0 + 1) is False, "容量恢复后必须立即放行"
    assert w.active_failures(t0 + 1) == ["kimi", "qwen"]

    # 全部恢复则统计计入
    _arm(w, None, t0 + 5)
    assert w.is_outage_active(t0 + 5) is True
    for pk in ("qwen", "glm", "kimi"):
        w.note_success(pk, t0 + 6)
    assert w.is_outage_active(t0 + 6) is False
    assert w.get_diagnostics(t0 + 6)["stats"]["recovered_by_success"] == 1
    print("test_any_recovery_resumes_dispatch: PASSED")


def test_clear_and_fast_fail_counter():
    w = _fresh()
    t0 = 1000.0
    _arm(w, None, t0)
    w.note_fast_fail()
    w.note_fast_fail()
    assert w.get_diagnostics(t0)["stats"]["fast_fails"] == 2
    w.clear("relogin")
    assert w.is_outage_active(t0) is False
    assert w.active_failures(t0) == []
    print("test_clear_and_fast_fail_counter: PASSED")


# ------------------------------------------------------------------ 3. 开关

def test_disable_switch():
    os.environ["GATEWAY_DISABLE_CREDENTIAL_WATCH"] = "1"
    try:
        w = _fresh()
        _arm(w, None, 1000.0)
        assert w.is_outage_active(1000.0) is False, "开关关闭后不得快速失败"
        assert w.get_diagnostics(1000.0)["enabled"] is False
    finally:
        os.environ.pop("GATEWAY_DISABLE_CREDENTIAL_WATCH", None)
    print("test_disable_switch: PASSED")


def test_co_disabled_with_risk_ledger():
    """与封禁租约同约定：关闭持久化风控状态时一并关闭（保护纯内存冷却测试套件）。"""
    os.environ["GATEWAY_DISABLE_RISK_LEDGER"] = "1"
    try:
        w = _fresh()
        assert w.is_disabled() is True
        _arm(w, None, 1000.0)
        assert w.is_outage_active(1000.0) is False
    finally:
        os.environ.pop("GATEWAY_DISABLE_RISK_LEDGER", None)
    print("test_co_disabled_with_risk_ledger: PASSED")


# ------------------------------------------------------------------ 4. 冷却器接入

def test_cooldown_tracker_feeds_the_watch():
    """record_failure(401) 上报、record_success 摘除——接入点必须真实生效。"""
    tracker = fov.CooldownTracker()
    tracker.credential_watch.clear_all_for_tests()
    try:
        for pk in ("qwen", "glm", "doubao"):
            asyncio.run(tracker.record_failure(pk, 401, "上游 401: token 无效或已过期"))
        assert tracker.credential_watch.is_outage_active() is True
        assert tracker.credential_watch.active_failures() == ["doubao", "glm", "qwen"]

        asyncio.run(tracker.record_success("doubao"))
        assert tracker.credential_watch.is_outage_active() is False

        # 非认证类失败不得触发全环判定（否则普通 5xx 风暴会误伤）
        tracker.credential_watch.clear_all_for_tests()
        for pk in ("qwen", "glm", "doubao", "kimi"):
            asyncio.run(tracker.record_failure(pk, 502, "Bad Gateway"))
        assert tracker.credential_watch.is_outage_active() is False
    finally:
        tracker.reset()
        tracker.credential_watch.clear_all_for_tests()
    print("test_cooldown_tracker_feeds_the_watch: PASSED")


# ------------------------------------------------------------------ 5. 端到端

def test_dispatched_request_produces_zero_upstream_calls():
    """失效态下请求被拒绝，且**没有任何** Provider 实例化或 chat 调用。"""
    watch = srv.credential_watch
    watch.clear_all_for_tests()
    original = srv._make_provider
    _CountingProvider.instances = 0
    _CountingProvider.chats = 0
    srv._make_provider = lambda key: _CountingProvider()
    try:
        _arm(watch, ["qwen", "glm", "doubao"], time.time())
        client = TestClient(srv.app)
        r = client.post(
            "/v1/chat/completions",
            json={"model": "qwen-max-latest", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 503, r.text
        body = r.json()
        assert body["error"]["type"] == "credential_outage", body
        assert "login.py all" in body["error"]["message"]
        assert _CountingProvider.instances == 0, "失效态下不得实例化任何 Provider"
        assert _CountingProvider.chats == 0, "失效态下不得产生任何上游调用"
        assert watch.get_diagnostics()["stats"]["fast_fails"] >= 1

        # 恢复后立即恢复派发（绝不卡死）
        watch.note_success("qwen")
        srv._make_provider = lambda key: _EchoProvider()
        r2 = client.post(
            "/v1/chat/completions",
            json={"model": "qwen-max-latest", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r2.status_code == 200, r2.text
    finally:
        srv._make_provider = original
        watch.clear_all_for_tests()
    print("test_dispatched_request_produces_zero_upstream_calls: PASSED")


def test_error_semantics_both_protocols():
    """OpenAI `credential_outage` / Anthropic `authentication_error`，HTTP 503。"""
    watch = srv.credential_watch
    watch.clear_all_for_tests()
    original = srv._make_provider
    srv._make_provider = lambda key: _CountingProvider()
    try:
        _arm(watch, ["qwen", "glm", "doubao"], time.time())
        client = TestClient(srv.app)

        r = client.post(
            "/v1/messages",
            json={"model": "claude-3-5-sonnet", "max_tokens": 16,
                  "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 503, r.text
        body = r.json()
        assert body["type"] == "error"
        assert body["error"]["type"] == "authentication_error", body
        assert "login.py all" in body["error"]["message"]

        # 流式请求：错误以 SSE 帧形式送达，同样携带可机读的 err_type
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={"model": "qwen-max-latest", "messages": [{"role": "user", "content": "hi"}],
                  "stream": True},
        ) as sr:
            text = "".join(sr.iter_text())
        assert "credential_outage" in text, text
        assert "login.py all" in text
    finally:
        srv._make_provider = original
        watch.clear_all_for_tests()
    print("test_error_semantics_both_protocols: PASSED")


def test_anthropic_error_type_mapping():
    assert map_anthropic_error_type(503, "credential_outage") == "authentication_error"
    print("test_anthropic_error_type_mapping: PASSED")


def test_diagnostics_and_admin_endpoints():
    watch = srv.credential_watch
    watch.clear_all_for_tests()
    try:
        client = TestClient(srv.app)
        diag = client.get("/v1/diagnostics").json()
        assert "credential_outage" in diag
        assert diag["credential_outage"]["outage_active"] is False

        r = client.get("/v1/admin/credential-outage")
        assert r.status_code == 200, r.text
        assert r.json()["outage_active"] is False

        _arm(watch, ["qwen", "glm", "doubao"], time.time())
        body = client.get("/v1/admin/credential-outage").json()
        assert body["outage_active"] is True
        assert body["failed_count"] == 3
        assert sorted(body["failed_providers"]) == ["doubao", "glm", "qwen"]
        assert "login.py all" in body["hint"]

        cleared = client.delete("/v1/admin/credential-outage").json()
        assert cleared["status"] == "ok"
        assert cleared["was_active"] is True
        assert sorted(cleared["cleared_providers"]) == ["doubao", "glm", "qwen"]
        assert cleared["outage"]["outage_active"] is False
        assert client.get("/v1/admin/credential-outage").json()["outage_active"] is False
    finally:
        watch.clear_all_for_tests()
    print("test_diagnostics_and_admin_endpoints: PASSED")


def run_all():
    test_one_provider_never_trips()
    test_two_providers_below_threshold()
    test_three_distinct_providers_trip()
    test_one_client_request_walking_the_ring_is_the_trigger()
    test_ttl_self_expiry_allows_probe_then_rearms()
    test_window_prune_exits_outage()
    test_any_recovery_resumes_dispatch()
    test_clear_and_fast_fail_counter()
    test_disable_switch()
    test_co_disabled_with_risk_ledger()
    test_cooldown_tracker_feeds_the_watch()
    test_dispatched_request_produces_zero_upstream_calls()
    test_error_semantics_both_protocols()
    test_anthropic_error_type_mapping()
    test_diagnostics_and_admin_endpoints()
    print("\n✅ All credential-watch tests (OPT-025) passed!")


if __name__ == "__main__":
    run_all()
