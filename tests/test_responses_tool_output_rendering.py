"""工具结果的两种历史形状必须渲染成同一份文本 (PROTOCOL)

Responses 客户端回传工具执行结果时用两种合法形状，本仓库 §43 已经确认过它们指的是
**同一件事**：

    {"type": "function_call_output", "call_id": "c1", "output": "exit code 0"}
    {"type": "message", "role": "tool", "tool_call_id": "c1",
     "content": [{"type": "input_text", "text": "exit code 0"}]}

两种都渲染成 `[Tool Result for c1]:` 开头的一段用户消息——但**内容**用了两条不同的
渲染路径：`role:"tool"` 走 `extract_responses_content`（认得 `input_text` /
`input_image` / `input_file` 块，附件变成 `[Image omitted: …]` 占位符），而
`function_call_output` / `custom_tool_call_output` / `local_shell_call_output` 三条
分支对 dict/list 一律 `json.dumps` ——把结构**原样倒进提示词**。

同一份逻辑结果，实测（真实转换函数，`TESTING=1`）：

    role:"tool"            101 字符  'exit code 0\\n[Image omitted: …]'
    function_call_output 20133 字符  '[{"type": "input_text", …}, {"type": "input_image",
                                      "image_url": "data:image/png;base64,AAAA…"}]'

两个后果，都属于本项目反复修正的同一类缺陷：

* **成本**：base64 附件被逐字塞进提示词。一张截图 20 KB，而这段历史在后续**每一轮**
  都会被重新发送（§66 的 continuation 只省掉答案侧，历史照样回放），模型为一个它
  根本看不到的附件持续付费。
* **可读性**：模型收到一段它无法解释的 JSON，而不是一句「附件已省略」。§41 已经
  论证过这个区别：占位符让模型能说「我看不到图片」，原始 JSON 只会让它对着一个
  从未收到的东西自信作答。

判据是 §43 / §62 的同一条纪律：**一个事实，一份渲染**。`extract_responses_content`
就是那份渲染，三条分支都该问它，而不是各自判断「是不是 dict」。

运行: .venv/bin/python tests/test_responses_tool_output_rendering.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["TESTING"] = "1"

from providers.responses_compat import (  # noqa: E402
    ResponsesRequest,
    convert_responses_to_gateway_messages,
)

#: 一份真实的 `view_image` 结果：base64 的 PNG 头，长度足以暴露「原样倒进提示词」。
_B64 = "data:image/png;base64," + "A" * 20000


def _render(item: dict) -> str:
    msgs = convert_responses_to_gateway_messages(
        ResponsesRequest(model="gpt-4o", input=[item])
    )
    assert len(msgs) == 1, msgs
    return msgs[0]["content"]


def _logical_result() -> list:
    """同一个工具结果的结构化内容：一段文本 + 一个附件。"""
    return [
        {"type": "input_text", "text": "exit code 0"},
        {"type": "input_image", "image_url": _B64},
    ]


# ------------------------------------------- 1. 附件不得被逐字倒进提示词

def test_a_structured_tool_output_omits_attachments_instead_of_dumping_them():
    """实测修复前：20133 字符，整个 base64 进了提示词。

    附件无法送达上游（五个上游都是纯文本网页接口），所以正确行为不是「原样带上」，
    而是像 §41 的其余路径一样**明说省略**。
    """
    out = _render({"type": "function_call_output", "call_id": "c1",
                   "output": _logical_result()})
    assert "base64," not in out and "AAAA" not in out, (
        "工具结果里的 base64 附件被逐字写进了提示词——这一段在后续每一轮都会被"
        f"重新发送，模型为一个它看不到的附件持续付费（实测 {len(out)} 字符）: "
        f"{out[:120]!r}"
    )
    assert len(out) < 1000, (
        f"结构化工具结果的渲染长度 {len(out)} 字符，说明结构被 JSON 序列化了: "
        f"{out[:160]!r}"
    )
    assert "exit code 0" in out, f"文本部分丢失了: {out!r}"
    assert "[Image omitted" in out, (
        "附件被静默丢弃了——模型会对着一个它从未收到的附件作答（§41 的判据）: "
        f"{out[:200]!r}"
    )
    print("test_a_structured_tool_output_omits_attachments_instead_of_dumping_them: PASSED")


# ------------------------------------ 2. 两种形状必须渲染成完全相同的文本

def test_both_history_shapes_render_identically():
    """同一份逻辑结果，两种合法形状，输出必须逐字节相同。

    §43 已经为「工具调用」定过同一条规则（`message.tool_calls` 与
    `function_call` 必须 byte-identical）；这里是它的**结果**一侧，之前只覆盖了
    一半：调用侧统一了，结果侧分成了两条渲染路径。
    """
    via_output = _render({"type": "function_call_output", "call_id": "c1",
                          "output": _logical_result()})
    via_tool_role = _render({"type": "message", "role": "tool", "tool_call_id": "c1",
                             "content": _logical_result()})
    assert via_output == via_tool_role, (
        "同一个工具结果由两条分支渲染成了两份不同的文本——一个事实两份定义，"
        "正是 §43 / §62 反复记录的缺陷:\n"
        f"  function_call_output: {via_output[:200]!r}\n"
        f"  role:tool          : {via_tool_role[:200]!r}"
    )

    # freeform（custom / local_shell）的输出走第三条分支，同样必须一致。
    for item_type in ("custom_tool_call_output", "local_shell_call_output"):
        via_custom = _render({"type": item_type, "call_id": "c1",
                              "output": _logical_result()})
        assert via_custom == via_output, (
            f"`{item_type}` 的渲染与 `function_call_output` 不一致:\n"
            f"  {item_type}: {via_custom[:200]!r}\n"
            f"  参考         : {via_output[:200]!r}"
        )
    print("test_both_history_shapes_render_identically: PASSED")


# ---------------------------------------------------------------- 3. 零回归

def test_plain_and_structural_outputs_survive_verbatim():
    """字符串结果、以及真正的结构化结果（无 `type` 块标记）必须原样保留。

    修复不能把「不要倒 JSON」做成「不要任何结构」：一个返回
    `{"stdout": …, "exit": 0}` 的工具，它的字段就是结果本身。
    """
    assert _render({"type": "function_call_output", "call_id": "c1",
                    "output": "exit code 0"}).endswith("exit code 0")

    out = _render({"type": "function_call_output", "call_id": "c1",
                   "output": {"stdout": "on branch main", "exit": 0}})
    assert "on branch main" in out and '"exit"' in out, (
        f"普通的结构化结果被改写了，模型读不到真实字段: {out!r}"
    )

    single = _render({"type": "function_call_output", "call_id": "c1",
                      "output": [{"type": "input_text", "text": "a"},
                                 {"type": "input_text", "text": "b"}]})
    assert "a" in single and "b" in single, f"多段文本被合并丢了: {single!r}"
    print("test_plain_and_structural_outputs_survive_verbatim: PASSED")


def run_all():
    test_a_structured_tool_output_omits_attachments_instead_of_dumping_them()
    test_both_history_shapes_render_identically()
    test_plain_and_structural_outputs_survive_verbatim()
    print("\n✅ All responses tool-output rendering tests passed! (3 cases)")


if __name__ == "__main__":
    sys.exit(run_all())