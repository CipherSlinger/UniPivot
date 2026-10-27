"""会话粘滞在短冷却期间的保留语义测试 (tests/test_affinity_cooling.py)。

[M3] 问题：会话亲和此前只有两个结局——「原节点可用」就粘滞，「不可用」就**永久
改嫁**到另一家厂商。而「不可用」的判据是可用性谓词，它对**任何**软冷却都答 False，
包括一次 15 秒的 429 频控退避。于是每有一次退避，进行中的长程 agent 会话就被永久
迁走：上游上下文全部重建、提示词缓存全丢、工具调用契约重新学习——为躲开数十秒的
窗口付出几十倍的代价。而这类退避在本项目里恰恰是常态（自适应步调存在的目的就是
在风控之前放慢）。

正确语义与派发边界早已确立的分层一致（OPT-021 / OPT-031）：
  * **硬阻断**（运维/自动租约、`DISABLED_PROVIDERS`、长效避让）→ 必须迁移：
    这些是「不要碰这个节点」的声明，等待没有意义；
  * **短冷却**（频控退避、一次性人机验证）→ 保留粘滞：秒级到两分钟即自愈，
    迁移的代价远高于等待。

覆盖：
  1. 短冷却降级为 `degraded`，粘滞关系不变，不产生 handoff 统计；
  2. 硬阻断（禁用 / 租约 / 长效避让）仍照常 handoff，且剥离异构上游 id；
  3. 冷却过去后会话回到原节点，且**从未**被记录到别家厂商（归属不被降级污染）；
  4. 端到端：短冷却期间响应头为 degraded 且带 `x-session-affinity-degraded`，
     禁用期间仍为 handoff；
  5. 服务端注入的判据是唯一口径（同一节点被两处判定时结论一致）。

运行: .venv/bin/python tests/test_affinity_cooling.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["TESTING"] = "1"
os.environ["GATEWAY_DISABLE_AUTO_REFRESH"] = "1"
os.environ["GATEWAY_DISABLE_CREDENTIAL_WATCH"] = "1"
os.environ["GATEWAY_DISABLE_PROBING"] = "1"
os.environ["GATEWAY_DISABLE_PACING"] = "1"
os.environ["GATEWAY_DISABLE_SINGLEFLIGHT"] = "1"
os.environ["QWEN_AUTH_TOKEN"] = "fake_qwen"
os.environ["DEEPSEEK_AUTH_TOKEN"] = "fake_ds"
os.environ["DOUBAO_AUTH_TOKEN"] = "fake_doubao"
os.environ["KIMI_AUTH_TOKEN"] = "fake_kimi"
os.environ["GLM_AUTH_TOKEN"] = "fake_glm"

from fastapi.testclient import TestClient  # noqa: E402

import server as srv  # noqa: E402
from providers.session_affinity import SessionAffinityManager  # noqa: E402


class _EchoProvider:
    """最小 Provider：返回本节点自有的上游会话 id。"""

    def __init__(self, key: str = "qwen"):
        self.key = key

    async def chat(self, *a, **kw):
        conv_id = kw.get("conversation_id") or f"{self.key}_upstream_conv"
        yield "收到消息", {"conversation_id": conv_id}


def _set_cooldown(provider: str, seconds: float) -> None:
    """直接把节点置入软冷却（绕过风控判定，只测路由语义）。"""
    tracker = srv.failover_router.cooldown_tracker
    tracker._cooldowns[provider] = time.time() + seconds
    tracker._cooldown_severity[provider] = "ratelimit"


def _clear_risk_state() -> None:
    tracker = srv.failover_router.cooldown_tracker
    tracker._cooldowns.clear()
    tracker._cooldown_severity.clear()
    tracker._failure_counts.clear()
    tracker.disabled_providers.clear()
    for lease in list(getattr(tracker.quarantine, "_leases", {}) or {}):
        try:
            tracker.quarantine.clear(lease)
        except Exception:  # noqa: BLE001
            pass


def test_short_cooldown_keeps_affinity():
    """短冷却：粘滞保留，状态为 degraded，不产生 handoff，上游 id 不清洗。"""
    mgr = SessionAffinityManager()
    mgr.record_turn("s1", "qwen", "Qwen", "qwen_upstream_conv")

    dec = mgr.resolve_affinity(
        session_id="s1",
        primary_provider="glm",
        primary_model="glm-4-plus",
        is_provider_available=lambda p: p != "qwen",   # 原节点「不可用」
        can_retain_degraded=lambda p: p == "qwen",     # 但只是短冷却
    )
    assert dec.status == "degraded", dec
    assert dec.is_handoff is False, "短冷却不得计为跨模型迁移"
    assert dec.provider_key == "qwen", "会话归属必须留在原节点"
    assert dec.upstream_conv_id == "qwen_upstream_conv", "短冷却不清洗上游会话 id"
    assert mgr.get_diagnostics()["affinity_handoffs"] == 0
    assert mgr.get_diagnostics()["affinity_degraded"] == 1
    print("test_short_cooldown_keeps_affinity: PASSED")


def test_hard_block_still_hands_off():
    """硬阻断：照常迁移，且必须剥离异构上游 id。"""
    mgr = SessionAffinityManager()
    mgr.record_turn("s2", "qwen", "Qwen", "qwen_upstream_conv")

    dec = mgr.resolve_affinity(
        session_id="s2",
        primary_provider="glm",
        primary_model="glm-4-plus",
        is_provider_available=lambda p: p != "qwen",
        can_retain_degraded=lambda p: False,  # 硬阻断：不可保留
    )
    assert dec.status == "handoff", dec
    assert dec.is_handoff is True
    assert dec.provider_key == "glm"
    assert dec.upstream_conv_id is None, "异构上游 id 必须剥离"
    assert dec.original_provider == "qwen"
    assert mgr.get_diagnostics()["affinity_handoffs"] == 1
    assert mgr.get_diagnostics()["affinity_degraded"] == 0
    print("test_hard_block_still_hands_off: PASSED")


def test_affinity_returns_after_cooldown_without_handoff():
    """冷却结束后会话回到原节点，且归属从未被降级污染。"""
    mgr = SessionAffinityManager()
    mgr.record_turn("s3", "qwen", "Qwen", "qwen_upstream_conv")

    short_cooling = {"on": True}

    def available(p: str) -> bool:
        return p != "qwen" or not short_cooling["on"]

    dec_deg = mgr.resolve_affinity(
        "s3", "glm", "glm-4-plus", available, lambda p: p == "qwen" and short_cooling["on"]
    )
    assert dec_deg.status == "degraded"

    # 本轮由其它节点服务：调用方（server.py）在降级期间把归属仍记在原节点上，
    # 并清掉本轮观测到的、属于别家的上游会话 id。
    mgr.record_turn("s3", dec_deg.provider_key, "Qwen", None)

    short_cooling["on"] = False
    dec_ok = mgr.resolve_affinity("s3", "glm", "glm-4-plus", available, lambda p: False)
    assert dec_ok.status == "bound", dec_ok
    assert dec_ok.provider_key == "qwen", "冷却结束后必须回到原节点"
    assert dec_ok.upstream_conv_id == "qwen_upstream_conv", "原上游会话 id 必须仍然可回放"
    assert mgr.get_diagnostics()["affinity_handoffs"] == 0, "全程不应发生迁移"
    print("test_affinity_returns_after_cooldown_without_handoff: PASSED")


def test_server_predicate_separates_availability_from_retainability():
    """服务端注入的判据必须区分「可用」与「值得保留」。"""
    _clear_risk_state()
    tracker = srv.failover_router.cooldown_tracker

    _set_cooldown("qwen", 15.0)  # 一次 15 秒的频控退避
    try:
        assert srv._is_provider_healthy("qwen") is False, "可用性判据对短冷却仍答 False"
        assert srv._is_provider_retainable("qwen") is True, "短冷却必须仍可保留粘滞"
    finally:
        _clear_risk_state()

    # 禁用是运维声明：不可用，也不可保留
    tracker.disabled_providers.add("glm")
    try:
        assert srv._is_provider_healthy("glm") is False
        assert srv._is_provider_retainable("glm") is False, "禁用必须触发迁移"
    finally:
        _clear_risk_state()

    # 长效避让（升级后的小时级冷却）：同样不可保留
    _set_cooldown("kimi", 7200.0)
    try:
        assert srv._is_provider_retainable("kimi") is False, "长效避让必须触发迁移"
    finally:
        _clear_risk_state()

    # 健康节点：两个判据都答 True，且不产生额外语义
    assert srv._is_provider_healthy("doubao") is True
    assert srv._is_provider_retainable("doubao") is True
    print("test_server_predicate_separates_availability_from_retainability: PASSED")


def test_endpoint_reports_degraded_and_keeps_binding():
    """端到端：短冷却期间响应头为 degraded，归属留在原节点。"""
    _clear_risk_state()
    srv.session_affinity_manager.clear()
    orig_make_provider = srv._make_provider
    srv._make_provider = lambda pkey, client=None: _EchoProvider(pkey)

    try:
        client = TestClient(srv.app)
        session_id = "conv_m3_001"

        r1 = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen",
                "messages": [{"role": "user", "content": "你好"}],
                "conversation_id": session_id,
            },
        )
        assert r1.status_code == 200, r1.text
        assert r1.headers.get("x-session-affinity") == "fresh"

        _set_cooldown("qwen", 15.0)
        r2 = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen",
                "messages": [
                    {"role": "user", "content": "你好"},
                    {"role": "assistant", "content": "收到"},
                    {"role": "user", "content": "继续"},
                ],
                "conversation_id": session_id,
            },
        )
        assert r2.status_code == 200, r2.text
        assert r2.headers.get("x-session-affinity") == "degraded", (
            f"短冷却应报 degraded，实际 {r2.headers.get('x-session-affinity')}"
        )
        assert r2.headers.get("x-session-affinity-degraded") == "qwen"
        assert r2.headers.get("x-session-handoff-from") is None, "短冷却不得报迁移"

        _clear_risk_state()
        r3 = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen",
                "messages": [
                    {"role": "user", "content": "你好"},
                    {"role": "assistant", "content": "收到"},
                    {"role": "user", "content": "再继续"},
                ],
                "conversation_id": session_id,
            },
        )
        assert r3.status_code == 200, r3.text
        assert r3.headers.get("x-session-affinity") == "bound", (
            "冷却结束后必须回到粘滞状态且归属未改嫁"
        )
        assert srv.session_affinity_manager.get_diagnostics()["affinity_handoffs"] == 0

        # 对照：真正禁用时仍必须迁移
        #
        # [SESSION-IDENTITY] 这里必须延续同一条对话线：起手用户轮是线的身份的一部分
        # （同一父会话派生的两个同级子 agent 只能靠它区分，见
        # `tests/test_subagent_affinity_isolation.py`），所以「只发一句新的用户消息」
        # 已经不是真实客户端会发出的形状，它会被判定为新的一条线，从而绕开 handoff。
        srv.failover_router.cooldown_tracker.disabled_providers.add("qwen")
        r4 = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen",
                "messages": [
                    {"role": "user", "content": "你好"},
                    {"role": "assistant", "content": "收到"},
                    {"role": "user", "content": "再继续"},
                    {"role": "assistant", "content": "好的"},
                    {"role": "user", "content": "换节点"},
                ],
                "conversation_id": session_id,
            },
        )
        assert r4.status_code == 200, r4.text
        assert r4.headers.get("x-session-affinity") == "handoff"
        assert r4.headers.get("x-session-handoff-from") == "qwen"
    finally:
        srv._make_provider = orig_make_provider
        srv.session_affinity_manager.clear()
        _clear_risk_state()

    print("test_endpoint_reports_degraded_and_keeps_binding: PASSED")


def run_all():
    test_short_cooldown_keeps_affinity()
    test_hard_block_still_hands_off()
    test_affinity_returns_after_cooldown_without_handoff()
    test_server_predicate_separates_availability_from_retainability()
    test_endpoint_reports_degraded_and_keeps_binding()
    print("\n✅ All affinity cooling tests passed!")


if __name__ == "__main__":
    run_all()