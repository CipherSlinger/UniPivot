"""长上下文 Prompt Cache 预计算与复用管理器单元测试 (tests/test_prompt_cache.py)。

测试覆盖:
1. compute_prompt_hash 稳定性与敏感性 (对 system prompt 与 messages 变动敏感)
2. has_cache_control 递归检测 (识别 Anthropic 规范的 cache_control 标记)
3. PromptCacheManager L1 完整 Prompt 精确匹配写入与命中
4. PromptCacheManager L2 前缀匹配与会话 conversation_id 复用
5. TTL 过期判定与主动清理 (purge_expired)
6. LRU 容量超限淘汰机制 (eviction_count)
7. Anthropic usage 分项计算 (cache_creation_input_tokens / cache_read_input_tokens)
8. 多协程并发读写安全 (asyncio.gather 并发测试)
"""

import asyncio
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from providers.prompt_cache import (
    CacheEntry,
    PromptCacheManager,
    compute_prompt_hash,
    has_cache_control,
)


class TestPromptCache(unittest.TestCase):
    def setUp(self):
        self.cache = PromptCacheManager(max_capacity=5, ttl=60.0)
        self.cache.clear()

    def test_compute_prompt_hash_consistency_and_sensitivity(self):
        sys_prompt = "You are a helpful coding assistant."
        messages = [
            {"role": "user", "content": "How do I implement binary search?"},
            {"role": "assistant", "content": "Here is the code..."},
        ]

        h1 = compute_prompt_hash(sys_prompt, messages)
        h2 = compute_prompt_hash(sys_prompt, messages)
        self.assertEqual(h1, h2, "相同的 system prompt 与 messages 产生的 hash 必须完全一致")
        self.assertEqual(len(h1), 64, "SHA-256 哈希值长度必须为 64")

        # 改变消息内容，hash 必须改变
        messages_diff = [
            {"role": "user", "content": "How do I implement quicksort?"},
            {"role": "assistant", "content": "Here is the code..."},
        ]
        h3 = compute_prompt_hash(sys_prompt, messages_diff)
        self.assertNotEqual(h1, h3, "消息内容改变后 hash 必须不同")

    def test_has_cache_control_detection(self):
        # 1. 普通请求无 cache_control
        plain_req = {
            "model": "claude-3-7-sonnet",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        self.assertFalse(has_cache_control(plain_req))

        # 2. system 字典中带 cache_control
        cached_sys_req = {
            "model": "claude-3-7-sonnet",
            "system": [
                {"type": "text", "text": "System prompt", "cache_control": {"type": "ephemeral"}}
            ],
            "messages": [{"role": "user", "content": "Hello"}],
        }
        self.assertTrue(has_cache_control(cached_sys_req))

        # 3. message 块中带 cache_control
        cached_msg_req = {
            "model": "claude-3-7-sonnet",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Huge context...", "cache_control": {"type": "ephemeral"}}
                    ],
                }
            ],
        }
        self.assertTrue(has_cache_control(cached_msg_req))

    def test_l1_exact_match_and_stats(self):
        sys_p = "System instruction"
        msgs = [{"role": "user", "content": "Initial task"}]

        # 首次查询，应当未命中
        matched_before = self.cache.match_prefix(sys_p, msgs, "qwen", "Qwen3.7-Max")
        self.assertIsNone(matched_before)
        self.assertEqual(self.cache.cache_misses, 1)

        # 写入缓存
        entry = self.cache.set(
            system_prompt=sys_p,
            messages=msgs,
            conversation_id="conv_12345",
            provider_key="qwen",
            wire_model="Qwen3.7-Max",
            estimated_tokens=500,
        )
        self.assertEqual(entry.conversation_id, "conv_12345")
        self.assertEqual(entry.estimated_tokens, 500)

        # 再次精确查询，应当 L1 命中
        matched_after = self.cache.match_prefix(sys_p, msgs, "qwen", "Qwen3.7-Max")
        self.assertIsNotNone(matched_after)
        self.assertEqual(matched_after.conversation_id, "conv_12345")
        self.assertEqual(self.cache.cache_hits, 1)
        self.assertEqual(self.cache.saved_tokens, 500)
        self.assertGreater(self.cache.hit_ratio, 0.0)

    def test_l2_prefix_match_conversation_reuse(self):
        """测试多轮长对话中，后续轮次即使增加了新消息，也能通过前缀命中复用原始会话 conversation_id"""
        sys_p = "System instruction for long project"
        turn1_msgs = [{"role": "user", "content": "Turn 1: build scaffolding"}]

        # 写入第一轮会话
        self.cache.set(
            system_prompt=sys_p,
            messages=turn1_msgs,
            conversation_id="conv_project_abc",
            provider_key="qwen",
            wire_model="Qwen3.7-Max",
            estimated_tokens=800,
        )

        # 第二轮对话：带有更多历史消息，但系统提示与首条消息不变
        turn2_msgs = [
            {"role": "user", "content": "Turn 1: build scaffolding"},
            {"role": "assistant", "content": "Scaffolding built successfully."},
            {"role": "user", "content": "Turn 2: add authentication layer"},
        ]

        # match_prefix 应当在 L2 找到对应前缀并复用会话 ID
        matched = self.cache.match_prefix(sys_p, turn2_msgs, "qwen", "Qwen3.7-Max")
        self.assertIsNotNone(matched)
        self.assertEqual(matched.conversation_id, "conv_project_abc")
        self.assertTrue(self.cache.is_cached("conv_project_abc"))

    def test_ttl_expiration_and_purge(self):
        cache = PromptCacheManager(max_capacity=10, ttl=20.0)
        base_time = 1000.0

        cache.set(
            system_prompt="Sys",
            messages=[{"role": "user", "content": "Hello"}],
            conversation_id="conv_temp",
            now=base_time,
        )

        # 10 秒后未过期 (now=1010, 上次访问更新为 1010)
        self.assertIsNotNone(
            cache.get(compute_prompt_hash("Sys", [{"role": "user", "content": "Hello"}]), now=base_time + 10)
        )

        # 35 秒后过期 (距上次访问 1010 过去了 25s, 超期 ttl=20s)
        self.assertIsNone(
            cache.get(compute_prompt_hash("Sys", [{"role": "user", "content": "Hello"}]), now=base_time + 35)
        )

        # purge_expired 应该清零已过期项
        purged = cache.purge_expired(now=base_time + 40)
        self.assertEqual(len(cache._l1_cache), 0)

    def test_lru_capacity_eviction(self):
        # 最大容量为 3
        cache = PromptCacheManager(max_capacity=3, ttl=3600.0)

        for i in range(5):
            cache.set(
                system_prompt=f"Sys {i}",
                messages=[{"role": "user", "content": f"Msg {i}"}],
                conversation_id=f"conv_{i}",
            )

        # 容量必须被限制在 max_capacity (3)
        self.assertEqual(len(cache._l1_cache), 3)
        self.assertGreaterEqual(cache.eviction_count, 2)

    def test_calculate_anthropic_usage(self):
        # 1. 命中场景
        entry = CacheEntry(
            prompt_hash="dummy_hash",
            system_prompt="Sys",
            conversation_id="cid",
            estimated_tokens=400,
        )
        uncached, creation, read = self.cache.calculate_anthropic_usage(
            req={}, is_hit=True, cached_entry=entry, total_input_tokens=500
        )
        self.assertEqual(read, 400)
        self.assertEqual(creation, 0)
        self.assertEqual(uncached, 100)

        # 2. 未命中但带 cache_control 标记场景
        req_with_cc = {
            "system": "System instructions here with length ~20 words",
            "cache_control": {"type": "ephemeral"},
        }
        uncached2, creation2, read2 = self.cache.calculate_anthropic_usage(
            req=req_with_cc, is_hit=False, cached_entry=None, total_input_tokens=600
        )
        self.assertEqual(read2, 0)
        self.assertGreater(creation2, 0)
        self.assertEqual(uncached2 + creation2, 600)

    def test_async_concurrent_operations(self):
        async def run_async_test():
            cache = PromptCacheManager(max_capacity=100, ttl=300.0)
            cache.clear()

            async def worker(idx: int):
                sys_p = f"Sys conc {idx % 5}"
                msgs = [{"role": "user", "content": f"User msg {idx}"}]
                await cache.set_async(
                    system_prompt=sys_p,
                    messages=msgs,
                    conversation_id=f"conv_{idx}",
                    provider_key="qwen",
                    wire_model="Qwen3.7-Max",
                )
                h = compute_prompt_hash(sys_p, msgs)
                found = await cache.get_async(h, "qwen", "Qwen3.7-Max")
                return found is not None

            tasks = [worker(i) for i in range(20)]
            results = await asyncio.gather(*tasks)
            self.assertTrue(all(results))
            self.assertEqual(cache.total_requests, 20)
            self.assertEqual(cache.cache_hits, 20)

        asyncio.run(run_async_test())


if __name__ == "__main__":
    unittest.main()
