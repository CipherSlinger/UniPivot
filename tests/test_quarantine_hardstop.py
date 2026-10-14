"""运维封禁租约的派发硬阻断 (OPT-021) 测试套件

背景：OPT-020 让运维封禁租约折入 CooldownTracker 冷却读数，因此所有**门控**
（探针/刷新/会话回收/自愈选举）都会自动避让。但**派发路径**并未门控：
`get_fallback_candidates` 仅把冷却节点**沉底**而非排除，`execute_chat` 更是
无条件派发主选节点。实证结果——被封禁 72h 的 deepseek 与 doubao 在故障转移
中**仍被派发真实上游请求**，这是整条风控体系中最宽的隐蔽触碰通道。

本套件验证 OPT-021 引入的语义区分：
- **软冷却**（自动风控触发、有时长上限）→ 仅降级排序，环内全冷却时仍可最后一搏；
- **硬封禁**（运维长期租约、账号级封禁）→ 任何情况下都不派发，快速失败。

覆盖：
1. is_quarantined 语义与开关（GATEWAY_DISABLE_QUARANTINE_HARD_STOP）
2. 后备候选中彻底排除封禁节点；软冷却节点仍保留（可用性回归保护）
3. execute_chat：主选被封禁 → 改派且零触碰
4. execute_chat：故障转移绝不落到封禁节点（复现原实证漏洞场景）
5. execute_chat：环内全部封禁 → 抛 ProviderError(503, quarantined) 且零上游调用
6. 负载均衡：封禁主选被分流，且绝不分流到封禁节点
7. 端到端：五个厂商全部封禁 → /v1/chat/completions 返回 503 quarantined 且零上游调用
8. 端到端：仅封禁主选 → 正常改派并返回 200

运行: .venv/bin/python tests/test_quarantine_hardstop.py
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
os.environ["QWEN_AUTH_TOKEN"] = "fake_qwen"
os.environ["DEEPSEEK_AUTH_TOKEN"] = "fake_ds"
os.environ["DOUBAO_AUTH_TOKEN"] = "fake_doubao"
os.environ["KIMI_AUTH_TOKEN"] = "fake_kimi"
os.environ["GLM_AUTH_TOKEN"] = "fake_glm"
for _k in ("GATEWAY_QUARANTINE", "GATEWAY_DISABLE_QUARANTINE", "GATEWAY_DISABLE_QUARANTINE_HARD_STOP"):
    os.environ.pop(_k, None)

from fastapi.testclient import TestClient  # noqa: E402

import server as srv  # noqa: E402
from providers.base import ProviderError  # noqa: E402
from providers.quarantine import QuarantineStore  # noqa: E402

ALL_PROVIDERS = ("qwen", "deepseek", "doubao", "kimi", "glm")

# 记录被真实派发的节点 key（任何进入 chat() 的节点都算「被触碰」）
TOUCHED: list[str] = []


def _make_fake(behavior: dict | None = None):
    """构造假 Provider 工厂。behavior[key] 指定该节点的行为：'ok' | 'fail'。"""
    behavior = behavior or {}

    def factory(key: str):
        class _Fake:
            async def chat(self, *a, **kw):
                TOUCHED.append(key)
                if behavior.get(key) == "fail":
                    raise ProviderError("boom", status=502)
                yield ("hi", {})

            async def delete_conversation(self, *a, **kw):
                return None

        return _Fake()

    return factory


def _isolated_store() -> QuarantineStore:
    """独立的临时租约库（不污染全局单例）。"""
    tmp = Path(tempfile.mkdtemp(prefix="quar_hs_"))
    return QuarantineStore(directory=tmp)


def _run(coro):
    return asyncio.run(coro)


# ------------------------------------------------------------------ 1. 语义与开关

def test_is_quarantined_semantics():
    tracker = srv.failover_router.cooldown_tracker
    original = tracker.quarantine
    store = _isolated_store()
    tracker.quarantine = store
    try:
        assert tracker.is_quarantined("deepseek") is False
        store.set_lease("deepseek", duration_seconds=72 * 3600)
        assert tracker.is_quarantined("deepseek") is True
        assert tracker.is_quarantined("qwen") is False, "未被租约锁定的节点不得被误伤"

        # 软冷却不等于封禁
        _run(tracker.record_failure("glm", 429, "rate limit"))
        assert tracker.is_in_cooldown("glm") is True
        assert tracker.is_quarantined("glm") is False, "自动软冷却不得升级为硬阻断"

        # 开关：退回旧的「仅降级」语义
        os.environ["GATEWAY_DISABLE_QUARANTINE_HARD_STOP"] = "1"
        try:
            assert tracker.is_quarantined("deepseek") is False
        finally:
            os.environ.pop("GATEWAY_DISABLE_QUARANTINE_HARD_STOP", None)
        assert tracker.is_quarantined("deepseek") is True
    finally:
        tracker.quarantine = original
        tracker.reset()
        shutil.rmtree(store.directory, ignore_errors=True)
    print("test_is_quarantined_semantics: PASSED")


def test_is_quarantined_never_raises():
    """租约查询异常绝不得误伤正常节点（零回归默认）。"""
    tracker = srv.failover_router.cooldown_tracker
    original = tracker.quarantine

    class _Boom:
        def remaining(self, *a, **kw):
            raise RuntimeError("ledger unavailable")

    tracker.quarantine = _Boom()
    try:
        assert tracker.is_quarantined("qwen") is False
    finally:
        tracker.quarantine = original
    print("test_is_quarantined_never_raises: PASSED")


# ------------------------------------------------------------------ 2. 候选过滤

def test_candidates_exclude_quarantined_but_keep_soft_cooling():
    store = srv.quarantine_store
    tracker = srv.failover_router.cooldown_tracker
    store.clear_all_for_tests()
    tracker.reset()
    try:
        store.set_lease("deepseek", duration_seconds=72 * 3600)
        _run(tracker.record_failure("kimi", 429, "rate limit"))  # 软冷却

        cands = srv.failover_router.get_fallback_candidates("qwen", "Qwen")
        keys = [c.provider_key for c in cands]
        assert "deepseek" not in keys, "封禁租约节点绝不得出现在后备候选中"
        assert "kimi" in keys, "软冷却节点必须保留（可用性回归保护）"
        kimi = next(c for c in cands if c.provider_key == "kimi")
        assert kimi.in_cooldown is True
        assert cands[0].provider_key != "kimi", "软冷却节点必须被沉底而非置顶"

        # 显式放行（仅用于诊断/审计视图）
        all_cands = srv.failover_router.get_fallback_candidates(
            "qwen", "Qwen", include_quarantined=True, include_long_avoidance=True
        )
        assert "deepseek" in [c.provider_key for c in all_cands]
    finally:
        store.clear_all_for_tests()
        tracker.reset()
    print("test_candidates_exclude_quarantined_but_keep_soft_cooling: PASSED")


# ------------------------------------------------------------------ 3/4/5. 派发硬阻断

def test_execute_chat_never_dispatches_quarantined_primary():
    store = srv.quarantine_store
    tracker = srv.failover_router.cooldown_tracker
    store.clear_all_for_tests()
    tracker.reset()
    TOUCHED.clear()
    try:
        store.set_lease("deepseek", duration_seconds=72 * 3600)

        stream, event, _ = _run(
            srv.failover_router.execute_chat(
                primary_provider_key="deepseek",
                primary_wire_model="deepseek-chat",
                provider_factory=_make_fake(),
                chat_args={"messages": []},
            )
        )

        async def _drain():
            async for _ in stream:
                pass

        _run(_drain())
        assert "deepseek" not in TOUCHED, f"封禁节点绝不得被派发，实际触碰: {TOUCHED}"
        assert TOUCHED, "必须改派到环内其他健康节点而非直接失败"
    finally:
        store.clear_all_for_tests()
        tracker.reset()
    print("test_execute_chat_never_dispatches_quarantined_primary: PASSED")


def test_failover_never_lands_on_quarantined_node():
    """复现并锁死原始漏洞场景：其他节点失败时，故障转移不得落到封禁节点。"""
    store = srv.quarantine_store
    tracker = srv.failover_router.cooldown_tracker
    store.clear_all_for_tests()
    tracker.reset()
    TOUCHED.clear()
    try:
        store.set_lease("deepseek", duration_seconds=72 * 3600)
        store.set_lease("doubao", duration_seconds=72 * 3600)

        try:
            stream, _ev, _p = _run(
                srv.failover_router.execute_chat(
                    primary_provider_key="qwen",
                    primary_wire_model="Qwen",
                    provider_factory=_make_fake({"qwen": "fail", "glm": "fail", "kimi": "fail"}),
                    chat_args={"messages": []},
                )
            )

            async def _drain():
                async for _ in stream:
                    pass

            _run(_drain())
        except ProviderError:
            pass  # 环内健康节点耗尽后快速失败，符合预期

        leaked = [p for p in TOUCHED if p in ("deepseek", "doubao")]
        assert leaked == [], f"故障转移触碰了封禁节点: {leaked}"
    finally:
        store.clear_all_for_tests()
        tracker.reset()
    print("test_failover_never_lands_on_quarantined_node: PASSED")


def test_all_quarantined_fails_closed_with_zero_upstream_calls():
    store = srv.quarantine_store
    tracker = srv.failover_router.cooldown_tracker
    store.clear_all_for_tests()
    tracker.reset()
    TOUCHED.clear()
    try:
        for pk in ALL_PROVIDERS:
            store.set_lease(pk, duration_seconds=72 * 3600)

        raised = None
        try:
            _run(
                srv.failover_router.execute_chat(
                    primary_provider_key="deepseek",
                    primary_wire_model="deepseek-chat",
                    provider_factory=_make_fake(),
                    chat_args={"messages": []},
                )
            )
        except ProviderError as exc:
            raised = exc

        assert raised is not None, "环内全部封禁时必须快速失败，而非派发"
        assert raised.status == 503
        assert raised.err_type == "quarantined"
        assert "封禁" in raised.message
        assert "admin/quarantine" in raised.message, "错误信息必须给出解除指引"
        assert TOUCHED == [], f"必须零上游触碰，实际: {TOUCHED}"
    finally:
        store.clear_all_for_tests()
        tracker.reset()
    print("test_all_quarantined_fails_closed_with_zero_upstream_calls: PASSED")


# ------------------------------------------------------------------ 6. 负载均衡

def test_load_balancer_redirects_off_quarantined_primary():
    store = srv.quarantine_store
    tracker = srv.failover_router.cooldown_tracker
    store.clear_all_for_tests()
    tracker.reset()
    try:
        store.set_lease("deepseek", duration_seconds=72 * 3600)
        d = srv.load_balancer.select_route("deepseek", "deepseek-chat", prompt_tokens=5000, is_thinking=True)
        assert d.provider_key != "deepseek", f"封禁主选必须被分流，实际 {d.provider_key}"
        assert d.original_provider == "deepseek"

        # 全部封禁时不得分流到任何封禁节点
        for pk in ALL_PROVIDERS:
            store.set_lease(pk, duration_seconds=72 * 3600)
        d2 = srv.load_balancer.select_route("deepseek", "deepseek-chat", prompt_tokens=5000, is_thinking=True)
        assert d2.provider_key == "deepseek", "无可用备选时保持主选交由网格硬阻断（不得分流向封禁节点）"
    finally:
        store.clear_all_for_tests()
        tracker.reset()
    print("test_load_balancer_redirects_off_quarantined_primary: PASSED")


# ------------------------------------------------------------------ 7/8. 端到端

def test_e2e_all_quarantined_returns_503_without_upstream_call():
    store = srv.quarantine_store
    tracker = srv.failover_router.cooldown_tracker
    store.clear_all_for_tests()
    tracker.reset()
    TOUCHED.clear()
    original = srv._make_provider
    srv._make_provider = _make_fake()
    try:
        for pk in ALL_PROVIDERS:
            store.set_lease(pk, duration_seconds=72 * 3600)

        client = TestClient(srv.app)
        r = client.post(
            "/v1/chat/completions",
            json={"model": "deepseek-chat", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 503, r.text
        assert r.json()["error"]["type"] == "quarantined", r.text
        assert TOUCHED == [], f"端到端必须零上游触碰，实际: {TOUCHED}"

        # Anthropic 协议同样映射
        r2 = client.post(
            "/v1/messages",
            json={"model": "claude-3-5-sonnet", "max_tokens": 16,
                  "messages": [{"role": "user", "content": "hi"}]},
            headers={"anthropic-version": "2023-06-01"},
        )
        assert r2.status_code == 503, r2.text
        assert r2.json()["error"]["type"] == "overloaded_error", r2.text
        assert "封禁" in r2.json()["error"]["message"]
        assert TOUCHED == []
    finally:
        srv._make_provider = original
        store.clear_all_for_tests()
        tracker.reset()
    print("test_e2e_all_quarantined_returns_503_without_upstream_call: PASSED")


def test_e2e_quarantined_primary_is_redirected_and_served():
    store = srv.quarantine_store
    tracker = srv.failover_router.cooldown_tracker
    store.clear_all_for_tests()
    tracker.reset()
    TOUCHED.clear()
    original = srv._make_provider
    srv._make_provider = _make_fake()
    try:
        store.set_lease("deepseek", duration_seconds=72 * 3600)

        client = TestClient(srv.app)
        r = client.post(
            "/v1/chat/completions",
            json={"model": "deepseek-chat", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 200, r.text
        assert "deepseek" not in TOUCHED, f"封禁节点被触碰: {TOUCHED}"
        assert TOUCHED, "应当改派并成功返回"
    finally:
        srv._make_provider = original
        store.clear_all_for_tests()
        tracker.reset()
    print("test_e2e_quarantined_primary_is_redirected_and_served: PASSED")


def run_all():
    test_is_quarantined_semantics()
    test_is_quarantined_never_raises()
    test_candidates_exclude_quarantined_but_keep_soft_cooling()
    test_execute_chat_never_dispatches_quarantined_primary()
    test_failover_never_lands_on_quarantined_node()
    test_all_quarantined_fails_closed_with_zero_upstream_calls()
    test_load_balancer_redirects_off_quarantined_primary()
    test_e2e_all_quarantined_returns_503_without_upstream_call()
    test_e2e_quarantined_primary_is_redirected_and_served()
    print("\n✅ All quarantine hard-stop tests (OPT-021) passed!")


if __name__ == "__main__":
    run_all()
