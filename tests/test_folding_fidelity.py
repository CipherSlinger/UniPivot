"""历史折叠保真度测试 (tests/test_folding_fidelity.py)。

[TOOL-LOOP] S1.2：折叠是上下文压缩，不是上下文改写。它唯一被允许做的事是
**变短**；一旦它同时改变了「这次执行成功还是失败」「这段内容的说话人是谁」
「这是一个工具调用还是一段散文」，它就从省 token 变成了投毒——而投毒不会被
任何一方的报错提醒，模型只会安静地按被改写过的历史继续推理。

本套件锁住三条不变量：

1. **失败不得被折叠成成功**。原实现无条件写「状态正常」，一次真实的工具失败被
   改写成「执行正常」，模型据此不再重试、直接基于错误结论往下走。这是
   `[Tool Error]` 标签与 Anthropic `is_error` 两条独立通道分别丢失的信息。
2. **调用不得被折叠成散文**。```` ```tool_call ```` 围栏里装的是调用本身；换成一句
   标记后，模型读到的是一个没有参数的残缺调用，工具循环当场断掉。
3. **说话人不得被改写**。用户消息里贴的超长代码是用户自己的输入，换成
   「已折叠历史执行结果」等于凭空给它加了一个「这是工具输出」的身份；assistant
   的普通代码回答同理。

每条用例都在修复前必然失败（git stash 可复现），修复后通过。
运行: .venv/bin/python tests/test_folding_fidelity.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 折叠器本身不碰磁盘，但它经由 providers/__init__ 引入了惰性单例；保持与其他
# 套件一致的隔离约定，绝不让测试碰到真实的 session/ 目录。
os.environ.setdefault("TESTING", "1")

from providers.prompt_optimizer import (  # noqa: E402
    fold_single_message,
    fold_history,
)

LONG_BODY = "S" * 900
LONG_CODE = "x = 1\n" * 300
TOOL_CALL_BODY = '{"name": "bash", "arguments": {"cmd": "ls -la"}}'


def test_tool_error_block_is_not_folded_as_success():
    """多轮回放里的 `[Tool Error ...]` 段被折叠后，必须仍然读得出「失败了」。"""
    text = (
        "[Tool Error for call_1]:\n"
        + LONG_BODY
        + "\n[Tool Result for call_2]:\n"
        + LONG_BODY
    )
    msg = {"role": "assistant", "content": text}
    folded, was_folded = fold_single_message(msg)
    assert was_folded, "超长工具结果是折叠的目标，必须被折叠"

    content = folded["content"]
    err_marker = content.split("[Tool Result")[0]
    ok_marker = content.split("[Tool Result")[1]

    assert "执行失败" in err_marker, f"失败结果被折叠成了成功语义: {err_marker!r}"
    assert "执行成功" not in err_marker, f"失败结果被改写成了「执行成功」: {err_marker!r}"
    assert "执行成功" in ok_marker, f"成功结果不应被误标为失败: {ok_marker!r}"
    print("test_tool_error_block_is_not_folded_as_success: PASSED")


def test_anthropic_is_error_flag_survives_folding():
    """Anthropic `tool_result.is_error=True` 是**协议字段**，折叠时不得丢失。"""
    msg = {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "tu_1",
                "is_error": True,
                "content": LONG_BODY,
            }
        ],
    }
    folded, was_folded = fold_single_message(msg)
    assert was_folded
    block = folded["content"][0]
    assert "执行失败" in block["content"], (
        f"is_error 的 tool_result 被折叠成了成功语义: {block['content']!r}"
    )
    assert block.get("is_error") is True, "折叠不得丢弃 is_error 协议字段"

    ok_msg = {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "tu_2", "content": LONG_BODY}
        ],
    }
    ok_folded, _ = fold_single_message(ok_msg)
    assert "执行失败" not in ok_folded["content"][0]["content"], (
        "成功结果被误标为失败，同样是投毒，只是方向相反"
    )
    print("test_anthropic_is_error_flag_survives_folding: PASSED")


def test_openai_tool_role_does_not_claim_success():
    """OpenAI `role:"tool"` 没有成功/失败字段：读不出来就不能猜。"""
    msg = {"role": "tool", "tool_call_id": "call_1", "content": LONG_BODY}
    folded, was_folded = fold_single_message(msg)
    assert was_folded
    content = folded["content"]
    # 判据是「不得断言任何状态」，不是「不得出现某个特定词的另一种写法」：
    # 「状态正常」与「执行成功」是同一个断言的两种措辞。
    for claim in ("成功", "正常"):
        assert claim not in content, (
            f"协议里没有成功/失败信号，却断言了「{claim}」: {content!r}"
        )
    assert "已折叠历史执行结果" in content, "既有断言依赖的前缀必须保留"
    print("test_openai_tool_role_does_not_claim_success: PASSED")


def test_tool_call_fence_is_never_folded():
    """```tool_call 围栏是调用本身，折叠后模型只会看到一个没有参数的残缺调用。"""
    # body 撑到阈值之上：真实场景里长参数（整段文件内容）很容易超限。
    text = "Running:\n```tool_call\n" + TOOL_CALL_BODY + LONG_BODY + "\n```"
    msg = {"role": "assistant", "content": text}
    folded, was_folded = fold_single_message(msg)
    assert not was_folded, "```tool_call 围栏不允许被折叠"
    assert folded["content"] == text, "调用围栏必须逐字符原样保留"

    # 经由 _fold_text_content 的路径同样不得动它
    from providers.prompt_optimizer import _fold_text_content

    out, folded_flag = _fold_text_content(text)
    assert not folded_flag and out == text, "调用围栏在任何折叠路径上都必须免疫"

    # 作为对照：普通代码块仍然会被折叠（证明上面不是因为整体没触发）
    plain = "```python\n" + LONG_CODE + "```"
    out2, folded2 = _fold_text_content(plain)
    assert folded2 and out2 != plain
    print("test_tool_call_fence_is_never_folded: PASSED")


def test_user_code_block_is_not_relabelled_as_tool_output():
    """用户消息里贴的代码是用户的输入，不是工具输出。"""
    msg = {"role": "user", "content": "请看一下这段代码：\n```python\n" + LONG_CODE + "```"}
    folded, was_folded = fold_single_message(msg)
    assert not was_folded, (
        "用户消息里的代码块被折叠了——这会给它贴上「历史执行结果」的身份，改写说话人语义"
    )
    assert folded == msg

    # assistant 的普通代码回答同理（它不是工具结果）
    a_msg = {"role": "assistant", "content": "这是实现：\n```python\n" + LONG_CODE + "```"}
    a_folded, a_folded_flag = fold_single_message(a_msg)
    assert not a_folded_flag, "assistant 的普通代码回答被当成了工具结果折叠"
    assert a_folded == a_msg

    print("test_user_code_block_is_not_relabelled_as_tool_output: PASSED")


def test_fold_history_keeps_tool_semantics_end_to_end():
    """端到端：经由 fold_history 的中间轮次折叠，失败语义与调用围栏都还在。"""

    # 把中间轮次撑到明显超过预算：预算要真的超，否则根本不会进入折叠路径，
    # 用例会「通过」得毫无意义。
    big = LONG_BODY * 6
    messages = [
        {"role": "system", "content": "You are a CLI agent."},
        {"role": "user", "content": "Deploy the service."},
        {
            "role": "assistant",
            "content": "Deploying.\n```tool_call\n" + TOOL_CALL_BODY + big + "\n```",
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "[Tool Error for call_1]:\n" + big},
        {"role": "user", "content": "Recent 1"},
        {"role": "assistant", "content": "Answer 1"},
        {"role": "user", "content": "Why did it fail?"},
    ]
    folded = fold_history(messages, max_tokens=3000, preserve_recent_turns=2)
    flat = "\n".join(str(m.get("content")) for m in folded)

    assert "```tool_call" in flat, "折叠把工具调用围栏整段吞掉了"
    tool_msgs = [m for m in folded if m.get("role") == "tool"]
    assert tool_msgs, "工具结果消息不应被删除"
    assert "执行失败" in str(tool_msgs[0]["content"]), (
        f"工具失败在端到端折叠后丢失: {tool_msgs[0]['content']!r}"
    )
    print("test_fold_history_keeps_tool_semantics_end_to_end: PASSED")


def test_summary_marker_does_not_claim_to_have_retained_anything():
    """阶段二把中间轮次整段丢弃：标记必须如实说「丢了多少」，而不是「已保留结论」。

    原文案是「保留核心结论与上下文」，而事实上一个 token 都没保留。模型读到
    「保留」会以为自己继承了前文的结论，于是直接往下推理，甚至引用根本不存在的
    约定——这比什么都不说更糟。
    """
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
    for i in range(20):
        messages.append({"role": "assistant", "content": f"a{i} " + "x" * 200})
        messages.append({"role": "user", "content": f"u{i} " + "y" * 200})
    messages.append({"role": "assistant", "content": "final a"})
    messages.append({"role": "user", "content": "final u"})

    folded = fold_history(messages, max_tokens=200, preserve_recent_turns=2)
    markers = [m for m in folded if "前序对话历史" in str(m.get("content"))]
    assert markers, "阶段二必须留下截断标记"
    marker = str(markers[0]["content"])

    assert "保留核心结论" not in marker, (
        f"标记声称保留了核心结论，而中间轮次被整段丢弃: {marker!r}"
    )
    # 省略条数必须与实际被丢弃的条数一致：读得出「丢了多少」才谈得上可复核。
    omitted = len(messages) - len([m for m in folded if "前序对话历史" not in str(m.get("content"))])
    assert f"{omitted} 条消息" in marker, (
        f"标记未如实给出省略条数（应为 {omitted}）: {marker!r}"
    )
    print("test_summary_marker_does_not_claim_to_have_retained_anything: PASSED")


def run_all():
    test_tool_error_block_is_not_folded_as_success()
    test_anthropic_is_error_flag_survives_folding()
    test_openai_tool_role_does_not_claim_success()
    test_tool_call_fence_is_never_folded()
    test_user_code_block_is_not_relabelled_as_tool_output()
    test_fold_history_keeps_tool_semantics_end_to_end()
    test_summary_marker_does_not_claim_to_have_retained_anything()
    print("\n✅ All prompt folding fidelity tests passed!")


if __name__ == "__main__":
    run_all()