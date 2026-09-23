"""Prompt 优化器与对话历史自适应折叠单元测试 (tests/test_prompt_optimizer.py)。

覆盖:
1. compute_prompt_hash 稳定性与敏感性
2. 短对话未超限时原样直通
3. 超长工具执行输出（OpenAI tool、Anthropic tool_result、[Tool Result for ...]）阶段一折叠
4. 阶段二老旧对话收敛折叠与摘要标记
5. 角色平衡与工具调用配对保护（首尾保护 + tool_call / tool_result 配对）
6. GATEWAY_DISABLE_PROMPT_FOLDING 环境变量规避
"""

import os
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from providers.prompt_optimizer import (
    DEFAULT_MAX_TOKENS,
    FOLDED_TOOL_RESULT_TEMPLATE,
    SUMMARY_MARKER,
    PromptOptimizer,
    compute_prompt_hash,
    estimate_history_tokens,
    fold_history,
    fold_single_message,
    make_prompt_folding_headers,
    prompt_optimizer,
)


class TestPromptOptimizer(unittest.TestCase):
    def test_prompt_optimizer_class_and_headers(self):
        opt = PromptOptimizer(max_tokens=1000, preserve_recent_turns=2)
        messages = [
            {"role": "system", "content": "You are a test assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "What is 1+1?"},
        ]
        optimized, meta = opt.optimize_chat_messages(messages)
        self.assertFalse(meta["applied"])
        self.assertEqual(meta["saved_tokens"], 0)
        self.assertEqual(meta["phase"], 0)
        self.assertEqual(optimized, messages)

        headers = make_prompt_folding_headers(meta)
        self.assertEqual(headers["x-prompt-folding-applied"], "false")
        self.assertEqual(headers["x-prompt-folding-saved-tokens"], "0")
        self.assertIn("x-prompt-folding-original-tokens", headers)
        self.assertIn("x-prompt-folding-final-tokens", headers)

        stats = opt.get_stats()
        self.assertIn("total_inspections", stats)
        self.assertEqual(stats["total_inspections"], 1)
        self.assertEqual(stats["total_folded_requests"], 0)
        self.assertEqual(stats["total_saved_tokens"], 0)
        self.assertTrue(stats["folding_enabled"])

    def test_prompt_optimizer_phase_one_folding_and_stats(self):
        opt = PromptOptimizer(max_tokens=1500, preserve_recent_turns=3)
        huge_tool_output = "Line " * 1500  # ~7500 chars

        messages = [
            {"role": "system", "content": "You are a CLI agent."},
            {"role": "user", "content": "Initial prompt: please inspect the logs."},
            {
                "role": "assistant",
                "content": "Running command...",
                "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "Bash"}}],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": huge_tool_output},
            {"role": "assistant", "content": "I see the logs."},
            {"role": "user", "content": "Recent question 1"},
            {"role": "assistant", "content": "Recent answer 1"},
            {"role": "user", "content": "Latest query: what is the conclusion?"},
        ]

        optimized, meta = opt.optimize_chat_messages(messages)
        self.assertTrue(meta["applied"])
        self.assertEqual(meta["phase"], 1)
        self.assertGreater(meta["saved_tokens"], 0)
        self.assertGreater(meta["original_tokens"], meta["final_tokens"])

        headers = make_prompt_folding_headers(meta)
        self.assertEqual(headers["x-prompt-folding-applied"], "true")
        self.assertEqual(headers["x-prompt-folding-saved-tokens"], str(meta["saved_tokens"]))
        self.assertEqual(headers["x-prompt-folding-original-tokens"], str(meta["original_tokens"]))
        self.assertEqual(headers["x-prompt-folding-final-tokens"], str(meta["final_tokens"]))

        stats = opt.get_stats()
        self.assertEqual(stats["total_inspections"], 1)
        self.assertEqual(stats["total_folded_requests"], 1)
        self.assertEqual(stats["total_saved_tokens"], meta["saved_tokens"])

    def test_prompt_optimizer_singleton_and_reset(self):
        inst1 = PromptOptimizer.get_instance()
        inst2 = PromptOptimizer.get_instance()
        self.assertIs(inst1, inst2)
        self.assertIs(inst1, prompt_optimizer)

        inst1.reset_stats()
        stats = inst1.get_stats()
        self.assertEqual(stats["total_inspections"], 0)
        self.assertEqual(stats["total_folded_requests"], 0)
        self.assertEqual(stats["total_saved_tokens"], 0)

    def test_prompt_optimizer_phase_two_summary(self):
        opt = PromptOptimizer(max_tokens=800, preserve_recent_turns=2)
        messages = [{"role": "system", "content": "Agent System Prompt"}]
        messages.append({"role": "user", "content": "First instruction from user"})

        for i in range(25):
            messages.append({"role": "assistant", "content": f"Answer turn {i}: " + "detailed explanation " * 40})
            messages.append({"role": "user", "content": f"User follow-up {i}: " + "more requirements " * 30})

        messages.append({"role": "assistant", "content": "Final assistant response"})
        messages.append({"role": "user", "content": "Final user instruction"})

        optimized, meta = opt.optimize_chat_messages(messages)
        self.assertTrue(meta["applied"])
        self.assertEqual(meta["phase"], 2)
        self.assertGreater(meta["saved_tokens"], 0)
        has_summary = any(SUMMARY_MARKER in str(m.get("content")) for m in optimized)
        self.assertTrue(has_summary)

    def test_prompt_optimizer_empty_and_disabled(self):
        opt_disabled = PromptOptimizer(disabled=True)
        messages = [{"role": "user", "content": "Line " * 500}]
        optimized, meta = opt_disabled.optimize_chat_messages(messages, max_tokens=10)
        self.assertFalse(meta["applied"])
        self.assertEqual(optimized, messages)

        opt = PromptOptimizer()
        optimized_empty, meta_empty = opt.optimize_chat_messages([])
        self.assertFalse(meta_empty["applied"])
        self.assertEqual(optimized_empty, [])
    def test_compute_prompt_hash(self):
        sys_prompt = "You are a helpful programming assistant."
        messages = [
            {"role": "user", "content": "Hello, write a binary search in Python."},
            {"role": "assistant", "content": "Sure, here is the code..."},
        ]

        h1 = compute_prompt_hash(sys_prompt, messages)
        h2 = compute_prompt_hash(sys_prompt, messages)
        self.assertEqual(h1, h2, "相同输入的哈希指纹必须完全一致")
        self.assertEqual(len(h1), 64, "SHA-256 哈希长度必须为 64")

        # 任意消息变化指纹必然改变
        messages_modified = [
            {"role": "user", "content": "Hello, write a quicksort in Python."},
            {"role": "assistant", "content": "Sure, here is the code..."},
        ]
        h3 = compute_prompt_hash(sys_prompt, messages_modified)
        self.assertNotEqual(h1, h3)

    def test_short_history_untouched(self):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hi there!"},
            {"role": "assistant", "content": "Hello! How can I help you today?"},
            {"role": "user", "content": "What is the capital of France?"},
        ]
        folded = fold_history(messages, max_tokens=1000)
        self.assertEqual(messages, folded, "未超限的历史记录必须原样返回")

    def test_phase_one_folding_oversized_tool_results(self):
        # 构造一条中间轮次中包含巨大工具输出的会话
        huge_tool_output = "Line " * 1500  # ~7500 chars

        messages = [
            {"role": "system", "content": "You are a CLI agent."},
            {"role": "user", "content": "Initial prompt: please inspect the logs."},
            {
                "role": "assistant",
                "content": "Running command...",
                "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "Bash"}}],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": huge_tool_output},
            {"role": "assistant", "content": "I see the logs."},
            {"role": "user", "content": "Recent question 1"},
            {"role": "assistant", "content": "Recent answer 1"},
            {"role": "user", "content": "Latest query: what is the conclusion?"},
        ]

        # 设置限制使得原消息超载，但折叠 tool output 后能够容纳
        folded = fold_history(messages, max_tokens=1500, preserve_recent_turns=3)

        # 验证首部系统提示和初始指令被保留
        self.assertEqual(folded[0]["role"], "system")
        self.assertEqual(folded[1]["content"], "Initial prompt: please inspect the logs.")

        # 验证尾部最近 3 轮未被破坏
        self.assertEqual(folded[-1]["content"], "Latest query: what is the conclusion?")
        self.assertEqual(folded[-2]["content"], "Recent answer 1")
        self.assertEqual(folded[-3]["content"], "Recent question 1")

        # 验证中间 tool 结果被截断折叠
        tool_msg = [m for m in folded if m.get("role") == "tool"][0]
        self.assertIn("已折叠历史执行结果", tool_msg["content"])
        self.assertTrue(len(tool_msg["content"]) < 100)

    def test_phase_two_summary_folding_massive_conversation(self):
        # 构造超长对话（数十轮），要求折叠后生成摘要标记
        messages = [{"role": "system", "content": "Agent System Prompt"}]
        messages.append({"role": "user", "content": "First instruction from user"})

        for i in range(25):
            messages.append({"role": "assistant", "content": f"Answer turn {i}: " + "detailed explanation " * 40})
            messages.append({"role": "user", "content": f"User follow-up {i}: " + "more requirements " * 30})

        messages.append({"role": "assistant", "content": "Final assistant response"})
        messages.append({"role": "user", "content": "Final user instruction"})

        folded = fold_history(messages, max_tokens=800, preserve_recent_turns=2)

        # 验证首部保留
        self.assertEqual(folded[0]["role"], "system")
        self.assertEqual(folded[1]["content"], "First instruction from user")

        # 验证出现了统一的折叠摘要标记
        has_summary = any(SUMMARY_MARKER in str(m.get("content")) for m in folded)
        self.assertTrue(has_summary, "极端超载对话应当插入紧凑摘要标记")

        # 验证尾部保留
        self.assertEqual(folded[-1]["content"], "Final user instruction")
        self.assertEqual(folded[-2]["content"], "Final assistant response")

    def test_anthropic_multiblock_tool_result_folding(self):
        # 测试 Anthropic Messages 协议下的 list of content blocks
        huge_output = "Data chunk " * 800
        msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Here is the tool response:"},
                {"type": "tool_result", "tool_use_id": "tool_abc", "content": huge_output},
            ],
        }
        folded_msg, was_folded = fold_single_message(msg)
        self.assertTrue(was_folded)
        content_blocks = folded_msg["content"]
        self.assertEqual(len(content_blocks), 2)
        self.assertEqual(content_blocks[0]["text"], "Here is the tool response:")
        self.assertIn("已折叠历史执行结果", content_blocks[1]["content"])

    def test_disable_prompt_folding_env(self):
        huge_tool_output = "Repeat " * 2000
        messages = [
            {"role": "system", "content": "Sys"},
            {"role": "user", "content": "Init"},
            {"role": "tool", "content": huge_tool_output},
            {"role": "user", "content": "Now"},
        ]
        os.environ["GATEWAY_DISABLE_PROMPT_FOLDING"] = "1"
        try:
            folded = fold_history(messages, max_tokens=100)
            self.assertEqual(messages, folded, "当环境变量设置禁用时，不得进行任何折叠")
        finally:
            os.environ.pop("GATEWAY_DISABLE_PROMPT_FOLDING", None)


if __name__ == "__main__":
    unittest.main()
