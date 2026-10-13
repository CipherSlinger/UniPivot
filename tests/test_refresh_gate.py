"""启动期令牌静默刷新的风控冷却门控 (OPT-013) 测试套件

验证：
1. 令牌过期（age > MAX_AGE）且节点空闲时正常刷新
2. 令牌未过期 (age < MAX_AGE) 时跳过刷新
3. 处于风控避让冷却期的节点暂缓刷新（绝不发起登录/刷新请求）
4. 冷却解除后，被暂缓的节点由延迟线程补做刷新
5. 冷却超时未解除时放弃本轮刷新（有界等待，绝不无限阻塞）
6. _auto_refresh_stale_sessions 端到端：健康节点刷新、冷却节点被暂缓

运行: .venv/bin/python tests/test_refresh_gate.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["GATEWAY_DISABLE_AUTO_REFRESH"] = "1"
os.environ["TESTING"] = "1"
os.environ["QWEN_AUTH_TOKEN"] = "fake_qwen"
os.environ["DEEPSEEK_AUTH_TOKEN"] = "fake_ds"
os.environ["DOUBAO_AUTH_TOKEN"] = "fake_doubao"
os.environ["KIMI_AUTH_TOKEN"] = "fake_kimi"
os.environ["GLM_AUTH_TOKEN"] = "fake_glm"

import server as srv  # noqa: E402
from session_store import MAX_AGE  # noqa: E402


def _stale_session():
    return SimpleNamespace(token="tok", age=MAX_AGE + 3600)


def _fresh_session():
    return SimpleNamespace(token="tok", age=10.0)


def _patch_cooldown(mapping):
    original = srv._provider_cooldown_remaining
    srv._provider_cooldown_remaining = lambda name: float(mapping.get(name, 0.0))
    return original


def test_stale_provider_refreshed_when_idle():
    """空闲且过期的节点正常刷新。"""
    calls = []

    def refresher(**kw):
        calls.append(kw)
        return True

    orig = _patch_cooldown({})
    try:
        outcome = srv._refresh_stale_provider("qwen", lambda: _stale_session(), refresher)
    finally:
        srv._provider_cooldown_remaining = orig
    assert outcome == "refreshed"
    assert len(calls) == 1
    print("test_stale_provider_refreshed_when_idle: PASSED")


def test_fresh_provider_skipped():
    """未过期令牌无需刷新。"""
    calls = []
    orig = _patch_cooldown({})
    try:
        outcome = srv._refresh_stale_provider(
            "qwen", lambda: _fresh_session(), lambda **kw: calls.append(kw) or True
        )
    finally:
        srv._provider_cooldown_remaining = orig
    assert outcome == "fresh"
    assert calls == []
    print("test_fresh_provider_skipped: PASSED")


def test_cooling_provider_deferred():
    """冷却期内的节点暂缓刷新，绝不发起刷新请求。"""
    calls = []
    orig = _patch_cooldown({"deepseek": 3600.0})
    try:
        outcome = srv._refresh_stale_provider(
            "deepseek", lambda: _stale_session(), lambda **kw: calls.append(kw) or True
        )
    finally:
        srv._provider_cooldown_remaining = orig
    assert outcome == "cooling"
    assert calls == [], "冷却期内绝不能发起令牌刷新（尤其是已封禁账号）"
    print("test_cooling_provider_deferred: PASSED")


def test_deferred_refresh_after_cooldown_clears():
    """冷却解除后由延迟线程补做刷新。"""
    remaining = {"n": 2}

    def fake_remaining(name):
        return 5.0 if remaining["n"] > 0 else 0.0

    calls = []
    orig = srv._provider_cooldown_remaining
    srv._provider_cooldown_remaining = fake_remaining
    try:
        def refresher(**kw):
            calls.append(kw)
            return True

        def ticker():
            time.sleep(0.02)
            remaining["n"] = 0

        import threading
        threading.Thread(target=ticker, daemon=True).start()

        srv._deferred_refresh_after_cooldown(
            [("qwen", lambda: _stale_session(), refresher)], max_wait=2.0, poll=0.01
        )
    finally:
        srv._provider_cooldown_remaining = orig
    assert len(calls) == 1, "冷却解除后必须补做一次刷新"
    print("test_deferred_refresh_after_cooldown_clears: PASSED")


def test_deferred_refresh_gives_up_on_timeout():
    """冷却超时未解除则放弃本轮刷新（有界等待）。"""
    calls = []
    orig = _patch_cooldown({"qwen": 999999.0})
    try:
        srv._deferred_refresh_after_cooldown(
            [("qwen", lambda: _stale_session(), lambda **kw: calls.append(kw) or True)],
            max_wait=0.05,
            poll=0.01,
        )
    finally:
        srv._provider_cooldown_remaining = orig
    assert calls == [], "冷却未解除前绝不能刷新"
    print("test_deferred_refresh_gives_up_on_timeout: PASSED")


def test_auto_refresh_end_to_end_skips_cooling():
    """_auto_refresh_stale_sessions：健康节点刷新、冷却节点被暂缓。"""
    import login

    refreshed = []
    original_resolvers = {n: getattr(srv, f"resolve_{n}") for n in ("qwen", "deepseek", "doubao", "kimi", "glm")}
    original_refreshers = {n: getattr(login, f"refresh_{n}") for n in ("qwen", "deepseek", "doubao", "kimi", "glm")}

    for n in ("qwen", "deepseek", "doubao", "kimi", "glm"):
        setattr(srv, f"resolve_{n}", lambda: _stale_session())

        def make(nm):
            def _r(**kw):
                refreshed.append(nm)
                return True
            return _r

        setattr(login, f"refresh_{n}", make(n))

    orig_cd = _patch_cooldown({"qwen": 7200.0})  # qwen 处于冷却，其余健康
    os.environ.pop("GATEWAY_DISABLE_AUTO_REFRESH", None)
    os.environ.pop("TESTING", None)
    os.environ["GATEWAY_REFRESH_DEFER_MAX_WAIT"] = "0"  # 延迟线程立即有界退出
    os.environ["GATEWAY_REFRESH_DEFER_POLL"] = "0"
    try:
        srv._auto_refresh_stale_sessions()
    finally:
        for n, v in original_resolvers.items():
            setattr(srv, f"resolve_{n}", v)
        for n, v in original_refreshers.items():
            setattr(login, f"refresh_{n}", v)
        srv._provider_cooldown_remaining = orig_cd
        os.environ["GATEWAY_DISABLE_AUTO_REFRESH"] = "1"
        os.environ["TESTING"] = "1"
        os.environ.pop("GATEWAY_REFRESH_DEFER_MAX_WAIT", None)
        os.environ.pop("GATEWAY_REFRESH_DEFER_POLL", None)

    assert "qwen" not in refreshed, "冷却节点绝不能被刷新"
    for n in ("deepseek", "doubao", "kimi", "glm"):
        assert n in refreshed, f"{n} 应被正常刷新"
    print("test_auto_refresh_end_to_end_skips_cooling: PASSED")


def run_all():
    test_stale_provider_refreshed_when_idle()
    test_fresh_provider_skipped()
    test_cooling_provider_deferred()
    test_deferred_refresh_after_cooldown_clears()
    test_deferred_refresh_gives_up_on_timeout()
    test_auto_refresh_end_to_end_skips_cooling()
    print("\n✅ All cooldown-aware token refresh gate tests (OPT-013) passed!")


if __name__ == "__main__":
    run_all()
