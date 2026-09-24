"""风控拦截避让与自适应流量整形单元测试 (tests/test_risk_avoidance.py)
"""
import asyncio
import os
from pathlib import Path
import sys
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from providers.failover import (
    TrafficPacer,
    is_risk_interception,
    PROVIDER_RISK_SPECS,
    RiskLevel,
    get_traffic_pacer,
)

class TestRiskAvoidance(unittest.TestCase):

    def test_risk_interception_matching(self):
        # 1. 阿里 RGV587
        is_risk, reason = is_risk_interception("Error: RGV587 FAIL_SYS_USER_VALIDATE")
        self.assertTrue(is_risk)
        self.assertIn("Alibaba", reason)

        # 2. 字节盾风控
        is_risk, reason = is_risk_interception("ProviderError: 豆包触发风控/人机验证，请运行 .venv/bin/python login.py doubao")
        self.assertTrue(is_risk)
        self.assertIn("ByteDance", reason)

        # 3. DeepSeek 封禁
        is_risk, reason = is_risk_interception("ProviderError: DeepSeek 接口错误: Authorization Failed (invalid token)")
        self.assertTrue(is_risk)
        self.assertIn("DeepSeek", reason)

        # 4. 常规无关报错
        is_risk, _ = is_risk_interception("KeyError: 'choices'")
        self.assertFalse(is_risk)

    def test_traffic_pacer_concurrency_and_pacing(self):
        async def _run():
            pacer = TrafficPacer()
            # 模拟 qwen 严格 max_concurrency = 1
            t0 = time.time()
            order = []

            async def worker(w_id: int):
                async with pacer.acquire("qwen") as wait_ms:
                    order.append((w_id, time.time() - t0, wait_ms))
                    await asyncio.sleep(0.05)

            await asyncio.gather(worker(1), worker(2))
            self.assertEqual(len(order), 2)
            # 串行执行，第二个 worker 的启动时间必须滞后
            self.assertGreaterEqual(order[1][1], order[0][1] + 0.04)

        asyncio.run(_run())

    def test_traffic_pacer_stats_and_singleton(self):
        pacer = get_traffic_pacer()
        self.assertIs(pacer, get_traffic_pacer())
        stats = pacer.get_stats()
        self.assertIn("active_semaphores", stats)
        self.assertIn("providers", stats)
        self.assertIn("qwen", stats["providers"])


if __name__ == "__main__":
    unittest.main()
