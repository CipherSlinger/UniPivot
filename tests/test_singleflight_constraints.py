"""去重指纹必须覆盖所有**会改变输出**的客户端约束 (S3 / SINGLEFLIGHT)

## 缺陷的来源：这是被修复**引入**的

SingleFlight 把指纹相同的在途请求合并成一次上游调用：Leader 真打，Follower
直接领用 Leader 的答案。所以指纹的正确性判据是唯一的 ——

    指纹相同 ⇒ 两条请求的正确答案必须可以互换。

`compute_key` 覆盖了 provider / model / messages / tools / temperature /
thinking / stream。在这些字段之外，`tool_choice`、`stop`、`response_format`、
`parallel_tool_calls` 当时**确实不改变输出** —— 因为它们当时被解析之后就没人
读了（本仓库记录的「声明了却从未生效」缺陷类）。那时把它们排除在指纹外是
无害的。

随后这些字段被逐个修好：`tool_choice` 落到解析端的允许名单上、`stop` 在网关
侧真截断、`response_format` 进了提示词、`parallel_tool_calls` 裁剪调用数量。
每一处修复都让该字段**开始**改变输出，而指纹没有同步更新。于是：

  * A 声明 `tool_choice: "none"`（要散文），B 声明 `"auto"`（要工具调用）。
    两者指纹相同 → 谁先到谁当 Leader。B 领到散文就永远等不到它的调用，
    agent 循环当场停住；A 领到 `tool_calls` 则是一个它已声明不会执行的调用。
  * A 声明 `stop: ["END"]`，B 没声明 → 其中一方拿到的文本按对方的终止符
    截断（或没被截断）。
  * A 声明 `response_format: json_object`，B 没声明 → 其中一方拿到散文。

症状全是「客户端收到一个按它从未提出的约束算出来的答案」，而且没有任何
错误可看 —— 与本仓库反复修的静默失败同形。

## 修复方向只能是「让键更specific」

把字段加进指纹只会让键**更**具体：合并变少，绝不会把两条不该合并的请求
合并起来。这是安全的方向。反过来（放宽键）永远不可取。

运行: .venv/bin/python tests/test_singleflight_constraints.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["TESTING"] = "1"

from providers.singleflight import SingleFlightManager  # noqa: E402

MGR = SingleFlightManager()
MESSAGES = [{"role": "user", "content": "read foo.py"}]
TOOLS = [{"type": "function", "function": {"name": "read_file", "parameters": {}}}]


def _key(**constraints):
    return MGR.compute_key(
        "qwen",
        "Qwen3.7-Max",
        MESSAGES,
        tools=TOOLS,
        temperature=0.7,
        stream=True,
        protocol="openai_chat",
        constraints=constraints or None,
    )


# ------------------------------------------------------------------ 用例


def test_tool_choice_changes_the_fingerprint():
    """`tool_choice` 现在决定解析端的允许名单 —— 必须进指纹。

    这是四条里后果最重的一条：Follower 领到的若是在另一种 `tool_choice` 下
    算出来的答案，它要么永远等不到自己的工具调用（循环停住），要么收到一个
    它已声明不会执行的调用（这一轮作废）。
    """
    none_key = _key(tool_choice="none")
    auto_key = _key(tool_choice="auto")
    named_key = _key(tool_choice={"type": "function", "function": {"name": "read_file"}})

    assert none_key != auto_key, (
        "tool_choice:'none' 与 'auto' 指纹相同 —— 要散文的请求会领到工具调用，"
        "要工具调用的请求会领到散文"
    )
    assert named_key not in (none_key, auto_key), (
        "指定工具名与 none/auto 指纹相同 —— 被限定到单个工具的请求会领到任意调用"
    )
    print("test_tool_choice_changes_the_fingerprint: PASSED")


def test_stop_changes_the_fingerprint():
    """`stop` 在网关侧**真截断**，所以它改变输出。"""
    no_stop = _key()
    with_stop = _key(stop=["END"])
    other_stop = _key(stop=["STOP"])

    assert no_stop != with_stop, (
        "声明了 stop 与未声明 stop 指纹相同 —— 其中一方会拿到按对方终止符"
        "截断（或未截断）的文本"
    )
    assert with_stop != other_stop, "不同的终止符必须产生不同指纹"
    print("test_stop_changes_the_fingerprint: PASSED")


def test_response_format_changes_the_fingerprint():
    """`response_format` 进了提示词，所以它改变输出。"""
    plain = _key()
    as_json = _key(response_format={"type": "json_object"})
    assert plain != as_json, (
        "要求 JSON 输出与未要求指纹相同 —— 要 JSON 的请求可能领到散文，"
        "随后在客户端侧解析失败"
    )
    print("test_response_format_changes_the_fingerprint: PASSED")


def test_parallel_tool_calls_changes_the_fingerprint():
    """`parallel_tool_calls:false` 把返回裁剪到第一个调用，所以它改变输出。"""
    allowed = _key(parallel_tool_calls=True)
    disallowed = _key(parallel_tool_calls=False)
    assert allowed != disallowed, (
        "禁止并行调用与允许并行调用指纹相同 —— 无法并行执行的客户端会领到"
        "多个调用，工具结果配对随即错位"
    )
    print("test_parallel_tool_calls_changes_the_fingerprint: PASSED")


def test_identical_constraints_still_dedup():
    """**零回归方向**：约束完全一致的两条请求必须仍然合并。

    修复的方向只能是让键更 specific。如果连「完全一样的请求」都不再合并，
    SingleFlight 就整体失效了 —— 那是比原缺陷更大的退化。
    """
    a = _key(tool_choice="auto", stop=["END"], response_format={"type": "json_object"})
    b = _key(tool_choice="auto", stop=["END"], response_format={"type": "json_object"})
    assert a == b, "约束一致的请求必须仍然共享同一次上游调用"

    # 约束键的书写顺序不得影响指纹（dict 顺序不是语义）
    c = MGR.compute_key(
        "qwen", "Qwen3.7-Max", MESSAGES, tools=TOOLS, temperature=0.7, stream=True,
        protocol="openai_chat",
        constraints={"stop": ["END"], "response_format": {"type": "json_object"}, "tool_choice": "auto"},
    )
    assert a == c, "约束字典的书写顺序不得改变指纹"
    print("test_identical_constraints_still_dedup: PASSED")


def test_absent_constraints_equal_empty_constraints():
    """不传 constraints 与传空必须等价（否则三协议的调用点会各自漂移）。"""
    assert MGR.compute_key(
        "qwen", "Qwen3.7-Max", MESSAGES, tools=TOOLS, temperature=0.7, stream=True
    ) == MGR.compute_key(
        "qwen", "Qwen3.7-Max", MESSAGES, tools=TOOLS, temperature=0.7, stream=True,
        constraints={},
    ), "缺省与空约束必须等价"
    print("test_absent_constraints_equal_empty_constraints: PASSED")


def test_all_three_protocols_pass_their_constraints():
    """三个协议的调用点都必须把自己的约束传进来。

    只修 `compute_key` 的签名而不改调用点，是一个会全绿的假修复：单元用例
    直接调 `compute_key`，而真实请求路径仍然把约束丢掉。
    """
    src = (Path(__file__).resolve().parent.parent / "server.py").read_text(encoding="utf-8")
    blocks = src.split("singleflight_manager.compute_key(")
    assert len(blocks) == 4, (
        f"预期三个协议各有一处 compute_key 调用，实测 {len(blocks) - 1} 处"
    )
    for idx, block in enumerate(blocks[1:], start=1):
        head = block[:600]
        assert "constraints=" in head, (
            f"第 {idx} 处 compute_key 调用没有传 constraints —— "
            f"该协议的客户端约束仍然会被去重键忽略"
        )
    print("test_all_three_protocols_pass_their_constraints: PASSED")


def run_all():
    test_tool_choice_changes_the_fingerprint()
    test_stop_changes_the_fingerprint()
    test_response_format_changes_the_fingerprint()
    test_parallel_tool_calls_changes_the_fingerprint()
    test_identical_constraints_still_dedup()
    test_absent_constraints_equal_empty_constraints()
    test_all_three_protocols_pass_their_constraints()
    print("\n✅ All singleflight-constraint tests (S3) passed!")


if __name__ == "__main__":
    run_all()
