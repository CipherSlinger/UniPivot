"""风控证据驱动的自适应步调强化 (OPT-014) 测试套件

验证：
1. compute_pacing_multiplier 纯函数行为（空证据 / WAF 权重 / 封顶 / 未知严重度）
2. get_adaptive_pacing_multiplier 从风控台账读取证据并按节点区分
3. TTL 缓存：窗口内命中缓存，清空后重新采样
4. GATEWAY_DISABLE_ADAPTIVE_PACING=1 退化为 1.0
5. TrafficPacer.acquire 依据放大系数实际拉长注入间隔
6. TrafficPacer.get_stats 暴露 adaptive_pacing / pacing_multiplier / effective_interval_s

运行: .venv/bin/python tests/test_adaptive_pacing.py
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["GATEWAY_DISABLE_AUTO_REFRESH"] = "1"
os.environ["TESTING"] = "1"
os.environ["QWEN_AUTH_TOKEN"] = "fake_qwen"
os.environ["DEEPSEEK_AUTH_TOKEN"] = "fake_ds"
os.environ["DOUBAO_AUTH_TOKEN"] = "fake_doubao"
os.environ["KIMI_AUTH_TOKEN"] = "fake_kimi"
os.environ["GLM_AUTH_TOKEN"] = "fake_glm"

import providers.failover as fov  # noqa: E402
from providers.risk_ledger import (  # noqa: E402
    SEVERITY_AUTH,
    SEVERITY_RATE_LIMIT,
    SEVERITY_WAF,
    RiskLedger,
)


def _fresh_ledger() -> RiskLedger:
    return RiskLedger(Path(tempfile.mkdtemp(prefix="adaptive_pacing_test_")))


def _bind_ledger(ledger: RiskLedger):
    original = fov.get_risk_ledger
    fov.get_risk_ledger = lambda: ledger  # type: ignore[assignment]
    return original


def test_compute_pacing_multiplier_pure():
    """纯函数行为：空证据=1.0，权重线性放大并封顶。"""
    assert fov.compute_pacing_multiplier({}) == 1.0
    assert abs(fov.compute_pacing_multiplier({"waf": 1}) - 1.25) < 1e-9
    assert abs(fov.compute_pacing_multiplier({"ratelimit": 2}) - 1.30) < 1e-9  # 0.6*2=1.2 → 1+0.3
    assert abs(fov.compute_pacing_multiplier({"auth": 1}) - 1.20) < 1e-9  # 0.8 → 1+0.2
    # 未知严重度按 0.5 权重
    assert abs(fov.compute_pacing_multiplier({"mystery": 2}) - 1.25) < 1e-9
    # 封顶
    capped = fov.compute_pacing_multiplier({"waf": 100}, max_multiplier=2.5)
    assert capped == 2.5
    print("test_compute_pacing_multiplier_pure: PASSED")


def test_multiplier_reads_ledger_evidence():
    """从台账读取证据，且按节点区分。"""
    ledger = _fresh_ledger()
    ledger.record_interception("qwen", 403, "rgv587", SEVERITY_WAF, 45.0, now=time.time() - 60)
    orig = _bind_ledger(ledger)
    fov.clear_pacing_evidence_cache()
    try:
        assert abs(fov.get_adaptive_pacing_multiplier("qwen") - 1.25) < 1e-9
        assert fov.get_adaptive_pacing_multiplier("deepseek") == 1.0
    finally:
        fov.get_risk_ledger = orig  # type: ignore[assignment]
        fov.clear_pacing_evidence_cache()
    print("test_multiplier_reads_ledger_evidence: PASSED")


def test_multiplier_ttl_cache():
    """TTL 缓存：窗口内命中缓存，清空后重新采样。"""
    ledger = _fresh_ledger()
    ledger.record_interception("qwen", 403, "rgv587", SEVERITY_WAF, 45.0, now=time.time() - 60)
    orig = _bind_ledger(ledger)
    fov.clear_pacing_evidence_cache()
    os.environ["GATEWAY_PACING_EVIDENCE_TTL"] = "3600"
    try:
        first = fov.get_adaptive_pacing_multiplier("qwen")
        assert abs(first - 1.25) < 1e-9
        # 新增证据，但 TTL 内应命中缓存，值不变
        ledger.record_interception("qwen", 403, "rgv587", SEVERITY_WAF, 45.0, now=time.time() - 30)
        assert fov.get_adaptive_pacing_multiplier("qwen") == first
        # 清空缓存后重新采样，应反映新增证据
        fov.clear_pacing_evidence_cache()
        assert abs(fov.get_adaptive_pacing_multiplier("qwen") - 1.5) < 1e-9
    finally:
        fov.get_risk_ledger = orig  # type: ignore[assignment]
        fov.clear_pacing_evidence_cache()
        os.environ.pop("GATEWAY_PACING_EVIDENCE_TTL", None)
    print("test_multiplier_ttl_cache: PASSED")


def test_disable_gate():
    """GATEWAY_DISABLE_ADAPTIVE_PACING=1 恒为 1.0。"""
    ledger = _fresh_ledger()
    ledger.record_interception("qwen", 403, "rgv587", SEVERITY_WAF, 45.0, now=time.time() - 60)
    orig = _bind_ledger(ledger)
    fov.clear_pacing_evidence_cache()
    os.environ["GATEWAY_DISABLE_ADAPTIVE_PACING"] = "1"
    try:
        assert fov.get_adaptive_pacing_multiplier("qwen") == 1.0
    finally:
        fov.get_risk_ledger = orig  # type: ignore[assignment]
        fov.clear_pacing_evidence_cache()
        os.environ.pop("GATEWAY_DISABLE_ADAPTIVE_PACING", None)
    print("test_disable_gate: PASSED")


def test_traffic_pacer_widens_interval():
    """TrafficPacer.acquire 依据放大系数实际拉长注入间隔。"""
    original_meta = fov.PROVIDER_RISK_SPECS["qwen"]
    patched = dataclasses.replace(original_meta, min_call_interval=0.01, jitter_range=(0.0, 0.0))
    fov.PROVIDER_RISK_SPECS["qwen"] = patched
    original_mult = fov.get_adaptive_pacing_multiplier
    try:
        async def _wait_ms(mult: float) -> float:
            fov.get_adaptive_pacing_multiplier = lambda pk, now=None: mult  # type: ignore[assignment]
            pacer = fov.TrafficPacer()
            pacer._last_call_times["qwen"] = time.time()
            async with pacer.acquire("qwen") as wait_ms:
                return wait_ms

        # 无风控证据 → 基础间隔 0.01s → ~10ms
        base = asyncio.run(_wait_ms(1.0))
        assert 8.0 <= base <= 16.0, f"基础间隔异常: {base}"

        # 4x 放大 → ~40ms
        widened = asyncio.run(_wait_ms(4.0))
        assert widened >= 30.0, f"自适应放大未生效: {widened}"
        assert widened > base * 2.5, f"放大倍数不足: base={base} widened={widened}"
    finally:
        fov.PROVIDER_RISK_SPECS["qwen"] = original_meta
        fov.get_adaptive_pacing_multiplier = original_mult  # type: ignore[assignment]
    print("test_traffic_pacer_widens_interval: PASSED")


def test_traffic_pacer_stats_exposes_multiplier():
    """get_stats 暴露自适应步调字段。"""
    original_mult = fov.get_adaptive_pacing_multiplier
    fov.get_adaptive_pacing_multiplier = lambda pk, now=None: 2.0  # type: ignore[assignment]
    try:
        stats = fov.TrafficPacer().get_stats()
        assert stats["adaptive_pacing"] is True
        info = stats["providers"]["qwen"]
        assert info["pacing_multiplier"] == 2.0
        assert abs(info["effective_interval_s"] - info["min_interval_s"] * 2.0) < 1e-6
    finally:
        fov.get_adaptive_pacing_multiplier = original_mult  # type: ignore[assignment]
    print("test_traffic_pacer_stats_exposes_multiplier: PASSED")


def run_all():
    test_compute_pacing_multiplier_pure()
    test_multiplier_reads_ledger_evidence()
    test_multiplier_ttl_cache()
    test_disable_gate()
    test_traffic_pacer_widens_interval()
    test_traffic_pacer_stats_exposes_multiplier()
    print("\n✅ All evidence-driven adaptive pacing tests (OPT-014) passed!")


if __name__ == "__main__":
    run_all()
