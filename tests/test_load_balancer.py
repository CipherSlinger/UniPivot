"""冷热模型动态负载均衡与自适应降级分流矩阵单元测试 (tests/test_load_balancer.py)。

测试覆盖:
1. SlidingWindowMetrics 秒级桶滑动窗口时间推移与指标聚合
2. SlidingWindowMetrics 并发原子增减 (acquire/release) 与保底
3. AdaptiveLoadBalancer 高水位与风控判定 (is_high_load)
4. AdaptiveLoadBalancer 轻量请求自适应降级至 SPEED 环 (Query Complexity Degradation)
5. AdaptiveLoadBalancer 高水位同环打散分流 (Same-Ring Shedding)
6. AdaptiveLoadBalancer 全量运行状态导出 (get_load_stats)
"""

import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from providers.failover import (
    CapabilityRing,
    CooldownTracker,
    PROVIDER_RISK_SPECS,
    ProviderRiskMeta,
    RiskLevel,
)
from providers.load_balancer import (
    AdaptiveLoadBalancer,
    BalancingDecision,
    SlidingWindowMetrics,
)


class TestLoadBalancer(unittest.TestCase):
    def setUp(self):
        self.cooldown_tracker = CooldownTracker()
        self.cooldown_tracker.reset()

    def test_sliding_window_metrics_basic_and_time_decay(self):
        metrics = SlidingWindowMetrics("test_provider", window_seconds=60.0)
        base_time = 1000.0

        # t=1000: 3 个并发进入
        metrics.acquire(now=base_time)
        metrics.acquire(now=base_time)
        metrics.acquire(now=base_time)
        self.assertEqual(metrics.active_concurrency, 3)

        # 记录 2 个成功，1 个错误 (含 1 个 rate limit)
        metrics.release(success=True, now=base_time + 1)
        metrics.release(success=True, now=base_time + 2)
        metrics.release(success=False, is_rate_limit=True, now=base_time + 2)
        self.assertEqual(metrics.active_concurrency, 0)

        # 记录 TTFT
        metrics.record_ttft(ttft_ms=200.0, now=base_time + 1)
        metrics.record_ttft(ttft_ms=400.0, now=base_time + 2)

        snap = metrics.get_snapshot(now=base_time + 10)
        self.assertEqual(snap.request_count, 3)
        self.assertEqual(snap.success_count, 2)
        self.assertEqual(snap.error_count, 1)
        self.assertEqual(snap.rate_limit_hits, 1)
        self.assertAlmostEqual(snap.avg_ttft_ms, 300.0, places=1)
        self.assertAlmostEqual(snap.error_rate, 1 / 3, places=2)

        # 时间前进 70 秒，超出 60 秒滑动窗口
        snap_expired = metrics.get_snapshot(now=base_time + 80)
        self.assertEqual(snap_expired.request_count, 0)
        self.assertEqual(snap_expired.success_count, 0)
        self.assertEqual(snap_expired.error_count, 0)

    def test_high_load_detection_by_concurrency_and_rate_limit(self):
        lb = AdaptiveLoadBalancer(
            cooldown_tracker=self.cooldown_tracker,
            high_watermark_ratio=0.8,
        )
        base_time = 1000.0

        # qwen max_concurrency=3, 80% 水位为 2.4 => active=2 时不高水位，active=3 时高水位
        self.assertFalse(lb.is_high_load("qwen", now=base_time)[0])

        lb.acquire("qwen", now=base_time)
        lb.acquire("qwen", now=base_time)
        self.assertFalse(lb.is_high_load("qwen", now=base_time)[0])

        lb.acquire("qwen", now=base_time)
        is_busy, reason = lb.is_high_load("qwen", now=base_time)
        self.assertTrue(is_busy)
        self.assertIn("并发度过高", reason)

        # 释放 2 个请求
        lb.release("qwen", success=True, now=base_time + 1)
        lb.release("qwen", success=True, now=base_time + 1)
        self.assertFalse(lb.is_high_load("qwen", now=base_time + 2)[0])

        # 触发 429 rate limit
        lb.release("qwen", success=False, is_rate_limit=True, now=base_time + 3)
        is_busy, reason = lb.is_high_load("qwen", now=base_time + 3)
        self.assertTrue(is_busy)
        self.assertIn("频控阻断", reason)

    def test_same_ring_high_watermark_shedding(self):
        """当 FLAGSHIP 环中的 Qwen3.7-Max 处于高水位时，长任务打散分流至同环内宽松健康的节点 (如 GLM 或 豆包)"""
        lb = AdaptiveLoadBalancer(
            cooldown_tracker=self.cooldown_tracker,
            high_watermark_ratio=0.7,
        )
        base_time = 1000.0

        # 使 qwen 处于高水位 (3 个并发满载)
        lb.acquire("qwen", now=base_time)
        lb.acquire("qwen", now=base_time)
        lb.acquire("qwen", now=base_time)

        # 正常长请求 (prompt_tokens=1500, 需要旗舰能力)
        decision = lb.select_route(
            primary_provider="qwen",
            primary_model="Qwen3.7-Max",
            prompt_tokens=1500,
            is_thinking=False,
            now=base_time,
        )

        self.assertEqual(decision.action, "shedded")
        self.assertNotEqual(decision.provider_key, "qwen")
        self.assertEqual(decision.ring, CapabilityRing.FLAGSHIP)
        self.assertIn("高负载/风控避让", decision.reason)
        # 应选择同环中的宽松/健康节点 (如 doubao 或 glm)
        self.assertIn(decision.provider_key, ["doubao", "glm", "deepseek", "kimi"])

    def test_lightweight_query_complexity_degradation_to_speed_ring(self):
        """当 FLAGSHIP 环目标繁忙时，短文本无思考请求自适应降级至 SPEED 环的高速经济模型"""
        lb = AdaptiveLoadBalancer(
            cooldown_tracker=self.cooldown_tracker,
            prompt_complexity_threshold=300,
        )
        base_time = 1000.0

        # 使 qwen 处于高并发状态
        lb.acquire("qwen", now=base_time)
        lb.acquire("qwen", now=base_time)
        lb.acquire("qwen", now=base_time)

        # 短轻量请求 (tokens=50, 无思考推理)
        decision = lb.select_route(
            primary_provider="qwen",
            primary_model="Qwen3.7-Max",
            prompt_tokens=50,
            is_thinking=False,
            now=base_time,
        )

        self.assertEqual(decision.action, "degraded")
        self.assertEqual(decision.ring, CapabilityRing.SPEED)
        self.assertIn(
            (decision.provider_key, decision.wire_model),
            [
                ("qwen", "Qwen3.6-Flash"),
                ("doubao", "doubao-lite"),
                ("glm", "glm-4-flash"),
            ],
        )
        self.assertIn("自适应降级至 SPEED 环", decision.reason)

    def test_thinking_query_not_degraded(self):
        """即使是短文本，若用户需要深度思考/推理 (is_thinking=True)，绝不降级至 SPEED 环，而是同环分流"""
        lb = AdaptiveLoadBalancer(
            cooldown_tracker=self.cooldown_tracker,
        )
        base_time = 1000.0

        # deepseek 满载
        lb.acquire("deepseek", now=base_time)
        lb.acquire("deepseek", now=base_time)

        decision = lb.select_route(
            primary_provider="deepseek",
            primary_model="deepseek-reasoner",
            prompt_tokens=50,
            is_thinking=True,  # 显式需要思考
            now=base_time,
        )

        # 不降级到 speed 环，而是留在 REASONING 环分流
        self.assertNotEqual(decision.ring, CapabilityRing.SPEED)
        self.assertEqual(decision.ring, CapabilityRing.REASONING)

    def test_get_load_stats(self):
        lb = AdaptiveLoadBalancer(cooldown_tracker=self.cooldown_tracker)
        stats = lb.get_load_stats()
        self.assertIn("nodes", stats)
        self.assertIn("status", stats)
        self.assertIn("total_active_concurrency", stats)
        self.assertIn("qwen", stats["nodes"])
        self.assertIn("doubao", stats["nodes"])
        self.assertIn("glm", stats["nodes"])


if __name__ == "__main__":
    unittest.main()
