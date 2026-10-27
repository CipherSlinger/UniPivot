r"""同级子 agent 是两条对话线，不是一条 (SESSION-IDENTITY)

[SESSION-IDENTITY] §104 让「同一会话里工具集不同的请求」分成了两条线（Claude Code
的自动模式分类器 `tools: []` 与父会话 47 个工具）。这条规则有一个它覆盖不到、
而且**更常见**的第二种形态：同一次会话里**顺序派生的同级子 agent**。

真实 Claude Code 2.1.293 的抓包（5 份 `/v1/messages` 请求体，父会话 3 轮 + 两个子
agent）显示，两个子 agent 的：

    metadata.user_id.session_id   完全相同（都等于父会话）
    声明的工具集                   完全相同（各 45 个，逐字相同）
    解析出的上游模型               完全相同

于是 `_line_key(session_id, tool_contract, requested_model)` 对三者给出**同一个
键**，第二个子 agent 被当成第一个的续聊。实测（脚本化 provider，上游提示词由真实的
`flatten_messages` / `continuation_content` 构造）：

    n=1  cid=None      父会话第 1 轮
    n=2  cid=None      子 agent 1
    n=3  cid='agent-1' 父会话第 2 轮
    n=4  cid='conv-2'  子 agent 2   <-- 拿到的正是子 agent 1 那次调用铸出的会话
    n=5  cid='agent-2' 父会话第 3 轮

后果是两条独立的对话在上游被合并：子 agent 2 的回答建立在子 agent 1 的**任务与
结果**之上，而客户端正在为另一个任务读它。没有任何报错，没有任何日志——和 §66
（Codex 分类器写进父会话）、§104（Claude Code 分类器合并进父会话）同一类。

判据是线路上真实存在、且**逐轮不变**的第二个事实：这条线从哪一句用户输入开始。
实测同一父会话三轮（messages 2 → 5 → 8 条）起始用户轮 sha 完全相同
（`3aced25b395c`），两个子 agent 分别是 `b64e375c72de` / `bf77a9b42571`。

本套件钉住：
1. 两个同级子 agent 各自拿到 `conversation_id=None`（各自新开上游会话）
2. 父会话的第 2、3 轮仍然绑在父会话自己的上游会话上（加宽不得把父线拆掉）
3. `line_opening_key` 的性质：历史增长时稳定、起始轮不同时不同
4. 诚实的残留：起始轮完全相同的两个请求仍然合并（与「同一条请求重发」不可区分，
   而后者必须复用上游会话——否则重试会付一次全新的上游上下文）

运行: .venv/bin/python tests/test_subagent_affinity_isolation.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["TESTING"] = "1"
os.environ["GATEWAY_DISABLE_CREDENTIAL_WATCH"] = "1"
os.environ["GATEWAY_DISABLE_AUTO_REFRESH"] = "1"
os.environ["GATEWAY_DISABLE_PROMPT_FOLDING"] = "1"
os.environ["GATEWAY_DISABLE_PACING"] = "1"
os.environ["GATEWAY_DISABLE_RISK_LEDGER"] = "1"
os.environ["GATEWAY_DISABLE_CONVERSATION_GC"] = "1"
for _pk in ("QWEN", "DEEPSEEK", "DOUBAO", "KIMI", "GLM"):
    os.environ[f"{_pk}_AUTH_TOKEN"] = f"test-token-{_pk.lower()}"

from fastapi.testclient import TestClient  # noqa: E402

import server as srv  # noqa: E402
from providers import session_affinity as _affinity  # noqa: E402
from providers.session_affinity import session_affinity_manager  # noqa: E402

#: 基线上没有这个函数。故意用 `getattr` 而不是 `from ... import`：否则整套会在
#: 导入期就 ImportError，行为用例（第一条）永远跑不起来，也就测不到它真正要说的
#: 那句话——「第二个子 agent 拿到了第一个子 agent 的上游会话」。
line_opening_key = getattr(_affinity, "line_opening_key", None)

#: 两个子 agent 声明的是同一套工具（这是本缺陷的前提，必须逐字相同）。
TOOLS = [
    {
        "name": "Bash",
        "description": "Run a shell command.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
    {
        "name": "Read",
        "description": "Read a file.",
        "input_schema": {
            "type": "object",
            "properties": {"file_path": {"type": "string"}},
            "required": ["file_path"],
        },
    },
]

#: 父会话多一个工具——真实抓包里父会话 47 个、子 agent 各 45 个，所以子 agent 1
#: 靠工具契约就已经分到自己的线（§104）。**故意保留这个差别**：这样本套件测到的
#: 就只是「同级子 agent 彼此合并」这一件事，而不是又测一遍工具契约那条规则。
PARENT_TOOLS = TOOLS + [
    {
        "name": "Agent",
        "description": "Spawn a sub-agent.",
        "input_schema": {
            "type": "object",
            "properties": {"prompt": {"type": "string"}},
            "required": ["prompt"],
        },
    },
]

SESSION = "0b6f1d2c-same-parent-session"


# --------------------------------------------------------------- harness

_GUARD_HITS: list[str] = []


def _guarded_factory(key, *a, **kw):
    _GUARD_HITS.append(key)
    raise AssertionError("测试发出了未安装假 provider 的上游请求")


_DEFAULT_FACTORY = srv._make_provider
srv._make_provider = _guarded_factory


class _ScriptedProvider:
    """记录上游**真正收到**的 conversation_id，并按真实上游的方式铸造新会话。

    关键：全新会话返回一个**新的** id（真实上游就是这样），绝不做
    `kw.get(...) or "固定值"` 的回声——那会让「子 agent 2 拿到了子 agent 1 的
    会话」这件事在断言里看不见。
    """

    _seq = 0

    def __init__(self, key: str):
        self.key = key

    async def chat(self, messages, model, **kw):
        cid = kw.get("conversation_id")
        CALLS.append({"provider": self.key, "cid": cid})
        if cid is None:
            type(self)._seq += 1
            cid = f"conv-{type(self)._seq}"
        yield "ok", {}
        yield "", {"conversation_id": cid}

    async def delete_conversation(self, *a, **kw):
        return None


#: 每次 provider 调用观测到的 (provider, 它收到的 conversation_id)。
CALLS: list[dict] = []


def _post(messages: list[dict], tools: list[dict] | None = None):
    body = {
        "model": "claude-3-7-sonnet",
        "max_tokens": 64,
        "tools": TOOLS if tools is None else tools,
        "messages": messages,
        # 真实 Claude Code 的身份就在这里（`conversation_id` 出现 0 次）。
        "metadata": {"user_id": json.dumps({"session_id": SESSION})},
    }
    return TestClient(srv.app).post("/v1/messages", json=body)


def _fresh_registry():
    session_affinity_manager.clear()
    CALLS.clear()
    _ScriptedProvider._seq = 0
    srv._make_provider = lambda key, *a, **kw: _ScriptedProvider(key)


def _msg(role: str, text: str) -> dict:
    return {"role": role, "content": text}


# ------------------------------------------------------- the measured case

def test_sibling_subagents_do_not_share_an_upstream_conversation():
    """两个同级子 agent 各自新开上游会话，父会话的绑定不被打断。

    序列与真实抓包逐项对应（父 3 轮 + 两个子 agent，顺序派生）。断言的是
    provider **真正收到**的 `conversation_id`，不是网关自己的簿记（§67(a)）。
    """
    _fresh_registry()
    try:
        parent_1 = [_msg("user", "PARENT-TASK: refactor the parser")]
        child_1 = [_msg("user", "CHILD-1: audit module alpha")]
        parent_2 = parent_1 + [
            _msg("assistant", "working"),
            _msg("user", "PARENT-CONTINUE-1"),
        ]
        child_2 = [_msg("user", "CHILD-2: audit module beta")]
        parent_3 = parent_2 + [
            _msg("assistant", "working"),
            _msg("user", "PARENT-CONTINUE-2"),
        ]

        for msgs, tools in (
            (parent_1, PARENT_TOOLS),
            (child_1, TOOLS),
            (parent_2, PARENT_TOOLS),
            (child_2, TOOLS),
            (parent_3, PARENT_TOOLS),
        ):
            resp = _post(msgs, tools)
            assert resp.status_code == 200, resp.text
    finally:
        srv._make_provider = _guarded_factory

    assert len(CALLS) == 5, f"预期 5 次上游调用，实测 {len(CALLS)}: {CALLS}"

    parent_cid = CALLS[2]["cid"]      # 父会话第 2 轮收到的
    parent_cid_3 = CALLS[4]["cid"]    # 父会话第 3 轮收到的
    child_1_cid = CALLS[1]["cid"]     # 子 agent 1 收到的
    child_2_cid = CALLS[3]["cid"]     # 子 agent 2 收到的

    assert CALLS[0]["cid"] is None, "父会话首轮不该带任何上游会话 id"
    assert child_1_cid is None, (
        f"子 agent 1 是它的线首轮，却拿到了上游会话 id {child_1_cid!r}"
    )

    # (1) 缺陷本身：子 agent 2 不得继承子 agent 1 的上游会话。
    assert child_2_cid is None, (
        "第二个子 agent 拿到了上游会话 id "
        f"{child_2_cid!r}——它与第一个子 agent 的 session_id、工具集、模型完全相同，"
        "于是被当成了续聊。实测（修复前）：n=4 cid='conv-2' 子 agent 2。"
    )

    # (2) 加宽不得打断父线：父会话第 2、3 轮绑在父会话自己的上游会话上。
    assert parent_cid is not None, (
        "父会话第 2 轮没有粘住自己的上游会话——新的判据把父线也拆开了"
    )
    assert parent_cid == parent_cid_3, (
        f"父会话第 2、3 轮绑到了不同的上游会话（{parent_cid!r} vs {parent_cid_3!r}），"
        "说明身份判据在历史增长时不稳定"
    )

    print("test_sibling_subagents_do_not_share_an_upstream_conversation: PASSED")


def test_line_opening_key_is_stable_and_discriminative():
    """[性质] 起始用户轮：历史增长时稳定，起点不同时不同。"""
    assert line_opening_key is not None, (
        "`providers.session_affinity.line_opening_key` 不存在——对话线身份仍然只由"
        "「会话 id + 工具集 + 模型」三者决定，同级子 agent 会被合并"
    )
    opening = [_msg("user", "PARENT-TASK: refactor the parser")]

    grown = opening + [
        _msg("assistant", "working"),
        _msg("user", "more"),
        _msg("assistant", "working"),
        _msg("user", "even more"),
    ]
    assert line_opening_key(opening) == line_opening_key(grown), (
        "同一条线历史增长后起始轮摘要变了——粘滞会在第 2 轮就断掉"
    )
    assert line_opening_key(opening) != line_opening_key(
        [_msg("user", "CHILD-1: audit module alpha")]
    ), "起始用户轮不同，摘要却相同——同级子 agent 仍会被合并"

    # 空历史 / 非用户开头：必须退回空串，身份与既有行为逐字节相同。
    assert line_opening_key([]) == ""
    assert line_opening_key([_msg("assistant", "x")]) == ""
    assert line_opening_key(None) == ""

    # 前导 system 不参与判定（它不是「这条线从哪里开始」）。
    with_system = [_msg("system", "you are terse")] + opening
    assert line_opening_key(with_system) == line_opening_key(opening), (
        "前导 system 改变了起始轮摘要——系统提示每轮都可能变，粘滞会因此抖动"
    )

    print("test_line_opening_key_is_stable_and_discriminative: PASSED")


def test_identical_opening_still_shares_the_conversation():
    """[守卫/诚实残留] 起始轮完全相同的两个请求仍然合并。

    这不是遗漏，而是必须：一个客户端对同一条请求重发一次（SDK 自己的重试）在网关上
    与「一个新的子 agent 用了同样的开场」不可区分。选「合并」是因为另一种选择会让
    每一次重试都付一次全新的上游上下文（§53/§75 的加宽方向：只会分开，不会合并）。
    """
    _fresh_registry()
    try:
        msgs = [_msg("user", "RETRY-ME: same opening")]
        first = _post(msgs)
        second = _post(msgs)
        assert first.status_code == 200 and second.status_code == 200
    finally:
        srv._make_provider = _guarded_factory

    assert len(CALLS) == 2
    assert CALLS[0]["cid"] is None, "首轮不该带上游会话 id"
    assert CALLS[1]["cid"] is not None, (
        "起始轮相同的重发没有复用上游会话——一次重试就白付一份全新上下文"
    )
    print("test_identical_opening_still_shares_the_conversation: PASSED")


def run_all():
    tests = [
        test_sibling_subagents_do_not_share_an_upstream_conversation,
        test_line_opening_key_is_stable_and_discriminative,
        test_identical_opening_still_shares_the_conversation,
    ]
    for fn in tests:
        fn()
    srv._make_provider = _DEFAULT_FACTORY
    assert not _GUARD_HITS, (
        f"有 {len(_GUARD_HITS)} 次上游调用未经脚本化 provider: {sorted(set(_GUARD_HITS))}"
    )
    print(f"\n✅ All subagent affinity isolation tests passed! ({len(tests)} cases)")


if __name__ == "__main__":
    run_all()
