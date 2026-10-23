"""确定性 Token 计数记忆化缓存 (OPT-009) 测试套件

验证：
1. LRU 命中/未命中和计数
2. 容量淘汰语义（淘汰最久未使用者）
3. 命中率计算
4. GATEWAY_DISABLE_TOKEN_CACHE 禁用开关
5. 指纹计算 (compute_count_key) 的确定性与区分度
6. /v1/messages/count_tokens 端点集成与零上游调用保证
7. GET /v1/diagnostics 暴露 disconnect_guard 与 token_counter_cache

运行: .venv/bin/python tests/test_token_cache.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["GATEWAY_DISABLE_AUTO_REFRESH"] = "1"
os.environ["TESTING"] = "1"
os.environ["QWEN_AUTH_TOKEN"] = "fake_qwen"
os.environ["DEEPSEEK_AUTH_TOKEN"] = "fake_ds"
os.environ["DOUBAO_AUTH_TOKEN"] = "fake_doubao"
os.environ["KIMI_AUTH_TOKEN"] = "fake_kimi"
os.environ["GLM_AUTH_TOKEN"] = "fake_glm"

from fastapi.testclient import TestClient  # noqa: E402

import server as srv  # noqa: E402
from providers.anthropic_compat import AnthropicCountTokensRequest  # noqa: E402
from providers.token_counter_cache import (  # noqa: E402
    TokenCounterCache,
    compute_count_key,
    token_counter_cache,
)


def test_token_cache_hit_and_miss():
    """LRU 命中与未命中计数。"""
    cache = TokenCounterCache(max_entries=8)
    assert cache.get("absent") is None
    cache.put("k1", 42)
    assert cache.get("k1") == 42

    diag = cache.get_diagnostics()
    assert diag["hits"] == 1
    assert diag["misses"] == 1
    assert diag["entries"] == 1
    assert diag["max_entries"] == 8
    print("test_token_cache_hit_and_miss: PASSED")


def test_token_cache_lru_eviction():
    """超出容量时淘汰最久未使用项。"""
    cache = TokenCounterCache(max_entries=2)
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.get("a") == 1  # a 变为最近使用
    cache.put("c", 3)           # 应淘汰 b

    assert cache.get("a") == 1, "a 刚被访问过，必须保留"
    assert cache.get("b") is None, "b 是最久未使用项，必须被淘汰"
    assert cache.get("c") == 3
    assert cache.get_diagnostics()["evictions"] == 1
    print("test_token_cache_lru_eviction: PASSED")


def test_token_cache_hit_rate():
    """命中率计算正确（四舍五入 4 位）。"""
    cache = TokenCounterCache(max_entries=8)
    cache.put("k", 1)
    cache.get("k")        # hit
    cache.get("k")        # hit
    cache.get("miss1")    # miss
    cache.get("miss2")    # miss
    diag = cache.get_diagnostics()
    assert diag["hits"] == 2
    assert diag["misses"] == 2
    assert diag["hit_rate"] == 0.5
    print("test_token_cache_hit_rate: PASSED")


def test_token_cache_disabled_env():
    """GATEWAY_DISABLE_TOKEN_CACHE=1 时 enabled 为 False。"""
    os.environ["GATEWAY_DISABLE_TOKEN_CACHE"] = "1"
    try:
        assert token_counter_cache.is_disabled() is True
        assert token_counter_cache.get_diagnostics()["enabled"] is False
    finally:
        os.environ.pop("GATEWAY_DISABLE_TOKEN_CACHE", None)
    assert token_counter_cache.is_disabled() is False
    print("test_token_cache_disabled_env: PASSED")


def test_compute_count_key_determinism_and_discrimination():
    """指纹确定性：同输入稳定，异输入区分。"""
    req_a = AnthropicCountTokensRequest(
        model="claude-3-7-sonnet",
        messages=[{"role": "user", "content": "hello"}],
        system="you are helpful",
    )
    req_a_same = AnthropicCountTokensRequest(
        model="claude-3-7-sonnet",
        messages=[{"role": "user", "content": "hello"}],
        system="you are helpful",
    )
    req_diff_content = AnthropicCountTokensRequest(
        model="claude-3-7-sonnet",
        messages=[{"role": "user", "content": "HELLO"}],
        system="you are helpful",
    )
    req_diff_order = AnthropicCountTokensRequest(
        model="claude-3-7-sonnet",
        messages=[
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "hello"},
        ],
        system="you are helpful",
    )

    key_a = compute_count_key(req_a)
    assert key_a == compute_count_key(req_a_same), "相同输入必须产生相同指纹"
    assert key_a != compute_count_key(req_diff_content), "内容不同必须区分"
    assert key_a != compute_count_key(req_diff_order), "消息顺序不同必须区分"
    assert len(key_a) == 64
    print("test_compute_count_key_determinism_and_discrimination: PASSED")


def test_count_tokens_endpoint_memoization_and_no_upstream():
    """端点集成：重复调用命中缓存，且绝不触达上游。"""
    token_counter_cache.clear()

    def exploding_provider(*args, **kwargs):
        raise AssertionError("count_tokens 绝不应触达上游 Provider")

    orig_make_provider = srv._make_provider
    srv._make_provider = exploding_provider

    try:
        client = TestClient(srv.app)
        payload = {
            "model": "claude-3-7-sonnet",
            "messages": [{"role": "user", "content": "记忆化测试 " * 20}],
            "system": "你是助手",
        }

        r1 = client.post("/v1/messages/count_tokens", json=payload)
        assert r1.status_code == 200, r1.text
        tokens_1 = r1.json()["input_tokens"]
        assert tokens_1 > 0

        r2 = client.post("/v1/messages/count_tokens", json=payload)
        assert r2.status_code == 200, r2.text
        assert r2.json()["input_tokens"] == tokens_1

        diag = token_counter_cache.get_diagnostics()
        assert diag["hits"] >= 1, "第二次相同请求必须命中缓存"
        assert diag["entries"] >= 1
    finally:
        srv._make_provider = orig_make_provider
        token_counter_cache.clear()

    print("test_count_tokens_endpoint_memoization_and_no_upstream: PASSED")


def test_diagnostics_exposes_new_sections():
    """GET /v1/diagnostics 暴露 disconnect_guard 与 token_counter_cache。"""
    client = TestClient(srv.app)
    r = client.get("/v1/diagnostics")
    assert r.status_code == 200
    data = r.json()
    assert "disconnect_guard" in data
    assert "token_counter_cache" in data

    dg = data["disconnect_guard"]
    for key in ("enabled", "client_disconnects", "upstream_aborts",
                "reclaimed_streams", "orphaned_flights_recovered", "per_provider"):
        assert key in dg, key

    tc = data["token_counter_cache"]
    for key in ("enabled", "entries", "max_entries", "hits", "misses", "evictions", "hit_rate"):
        assert key in tc, key
    pc = data["prompt_cache"]
    assert pc["status"] in ("active", "not_wired"), pc["status"]
    assert "wired" in pc, sorted(pc)
    print("test_diagnostics_exposes_new_sections: PASSED")


def test_prompt_cache_diagnostics_never_claim_an_unwired_cache():
    """`prompt_cache` 区块不得宣称一个代码里并不存在的缓存。

    [DIAG-HONESTY] 实测：向网关连发 12 个带 `cache_control` 的请求后，这个区块的
    每一个计数器仍然全是 0（前后完全相等），而 `status` 一直写着 `"active"`。
    原因不是缓存没命中，而是 `PromptCacheManager` 在**生产路径上没有任何调用点**
    ——`get`/`set`/`match_prefix` 只被它自己的单测调用，`server.py` 只读了
    `get_stats()`。同时 `ttft_reduction_pct: 45.0` 是一个写死的常数收益。

    一个宣称自己启用了缓存、还把收益写成常数的诊断面板，比没有这个面板更糟：
    运维据此判断长上下文的开销已被缓存吸收，而实际每次请求都在完整重建上下文。
    这与 OPT-031/035 的 dossier/dispatch 分歧是同一类缺陷——系统宣称的事情，
    代码并没有做。

    断言：未接入时必须 `status == "not_wired"`、`wired is False`、且**不伪造**
    收益数字（`ttft_reduction_pct is None`，而不是 0——0 会被读成「测过，没收益」）。
    """
    from providers.prompt_cache import prompt_cache_manager

    prompt_cache_manager.clear()
    client = TestClient(srv.app)
    pc = client.get("/v1/diagnostics").json()["prompt_cache"]

    assert pc["total_requests"] == 0, pc
    assert pc["status"] == "not_wired", (
        f"没有接入任何缓存读写，诊断却报告 status={pc['status']!r}——"
        f"运维会据此认为缓存已经生效"
    )
    assert pc["wired"] is False, pc
    assert pc["ttft_reduction_pct"] is None, (
        f"未接入时仍然报告 ttft 收益 {pc['ttft_reduction_pct']!r}——"
        f"这个数字是写死的常数，不是量出来的"
    )
    print("test_prompt_cache_diagnostics_never_claim_an_unwired_cache: PASSED")


def run_all():
    test_token_cache_hit_and_miss()
    test_token_cache_lru_eviction()
    test_token_cache_hit_rate()
    test_token_cache_disabled_env()
    test_compute_count_key_determinism_and_discrimination()
    test_count_tokens_endpoint_memoization_and_no_upstream()
    test_diagnostics_exposes_new_sections()
    test_prompt_cache_diagnostics_never_claim_an_unwired_cache()
    print("\n✅ All deterministic token count cache tests (OPT-009) passed!")


if __name__ == "__main__":
    run_all()
