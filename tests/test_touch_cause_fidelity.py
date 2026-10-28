"""上游故障不是风控 (TOUCH-CAUSE)

## 缺陷：把「一次 502」说成「为了规避上游封禁」

`CooldownTracker.touch_avoidance_reason()` 是后台自维护触点（健康探针 / 令牌刷新 /
临时会话回收 / 自愈选举）的**唯一**判据。它按优先级返回
`disabled` → `quarantined` → `unconfigured` → `cooldown`，而最后那一支只要
`get_remaining_cooldown() > 0` 就回 `cooldown`——**不问成因**。

于是三种「上游故障」冷却被报成了风控冷却：

    plain 5xx   cooldown=15.0s  cause='upstream_error'  touch_reason='cooldown'
    timeout     cooldown=45.0s  cause='timeout'         touch_reason='cooldown'
    truncated   cooldown=...    cause='truncated'       touch_reason='cooldown'

而 `cooldown` 在三个消费方的文案里分别读作：

  * 探针：`风控避让冷却中 (剩余 15s)，已跳过真实探针以规避上游封禁`，
    日志还写「避免加剧上游封禁」；
  * 自愈选举：`处于风控避让冷却期的节点 …，避免在避让期内触碰该账号`；
  * 会话回收：`处于风控避让冷却中，推迟其临时会话删除以避免触碰被封禁/已下线的账号`。

也就是说，一次**上游返回 502**，在运维读到的三处日志里都变成了「这个账号正在被平台
风控/封禁」。这正是 [DAY-ZERO] 为 `unconfigured` 修过的同一个误导
（「该账号从未被触碰过，说成封禁会让运维去排查一个不存在的封禁」），换了个成因；
也是 [COOLDOWN-CAUSE] 刚把普通 5xx 从 `waf` 里摘出来之后，**同一份成因在另一个判据
里又被折叠回去**——新增成因没有走完所有成因判读者（[DAY-ZERO] 的原话：
「一个新成因的工作没有完成，直到每一个成因判读者都认识它」）。

## 正确语义

「该不该触碰上游」这四个通道的答案对全部冷却都是**否**，这一点不变（后台自维护
永远不值得为冷却中的节点冒险，比派发边界更严）。变的只是**成因**：上游故障就报
上游故障，风控就报风控。禁用靠改配置解除、租约靠人工撤租、风控冷却靠等窗口、
上游故障靠上游恢复——四者混成一句话正是本项目反复修复的误导类型
（OPT-026/027/029/031/034）。

`_RISK_CAUSES` 已经定义了「哪些成因是风控」（[BUDGET-6]）：风控成因照旧报
`cooldown`（文案本来就对，且 `login.py all` 的「冷却就照常登录」分支依赖它），
非风控成因如实返回。判据未知（成因记录与台账都读不到）时仍回 `cooldown`——
宁可多说一次「风控」，也不能因为读不到成因就放行一次上游触碰。

运行: .venv/bin/python tests/test_touch_cause_fidelity.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _key in [k for k in os.environ if k.startswith("GATEWAY_DISABLE_")]:
    os.environ.pop(_key, None)
os.environ.pop("GATEWAY_QUARANTINE", None)
os.environ.pop("DISABLED_PROVIDERS", None)
os.environ["TESTING"] = "1"
os.environ["GATEWAY_DISABLE_RISK_LEDGER"] = "1"  # 纯内存冷却，绝不写真实 session/
os.environ["GATEWAY_DISABLE_PACING"] = "1"
os.environ["GATEWAY_DISABLE_SINGLEFLIGHT"] = "1"
os.environ["QWEN_AUTH_TOKEN"] = "fake_qwen"
os.environ["DEEPSEEK_AUTH_TOKEN"] = "fake_ds"
os.environ["DOUBAO_AUTH_TOKEN"] = "fake_doubao"
os.environ["KIMI_AUTH_TOKEN"] = "fake_kimi"
os.environ["GLM_AUTH_TOKEN"] = "fake_glm"

import providers.failover as fov  # noqa: E402
from healing.monitor import HealthMonitor  # noqa: E402
from providers.conversation_gc import ConversationGC  # noqa: E402

_UPSTREAM_FAULT_CAUSES = ("timeout", "truncated", "upstream_error")
# 判据是**断言的短语**，不是关键词「风控」：修复后正确的日志里恰好写着
# 「上游故障，不是风控拦截」，按子串判会命中这句正确措辞（§62/[COOLDOWN-CAUSE]
# 记录过的同一个坑）。这里只禁止那些**把上游故障说成风控/封禁**的句子。
_RISK_CLAIMS = ("风控避让", "规避上游封禁", "被封禁", "风控冷却")


class _MockSession:
    token = "fake_token"


class _HealthyProvider:
    async def chat(self, *a, **kw):
        yield "pong", {}


def _fresh_tracker() -> "fov.CooldownTracker":
    tr = fov.CooldownTracker()
    tr.reset()
    return tr


def _cool(tr, provider_key: str, kind: str) -> float:
    """按成因把一个节点打进冷却。"""
    if kind == "timeout":
        return asyncio.run(
            tr.record_failure(provider_key, 504, "上游超时", is_timeout=True)
        )
    if kind == "truncated":
        return asyncio.run(
            tr.record_failure(provider_key, 502, "上游流被截断", is_truncation=True)
        )
    if kind == "upstream_error":
        return asyncio.run(tr.record_failure(provider_key, 502, "通义千问上游请求失败: 502"))
    if kind == "waf":
        return asyncio.run(tr.record_failure(provider_key, 403, "RGV587 人机验证"))
    raise AssertionError(kind)


def _make_monitor(tr) -> HealthMonitor:
    monitor = HealthMonitor(check_interval=100.0, max_consecutive_failures=1)
    monitor.register_provider(
        "qwen", resolver=lambda: _MockSession(), factory=lambda: _HealthyProvider()
    )
    monitor.set_cooldown_probe_gate(lambda name: tr.get_remaining_cooldown(name))
    monitor.set_touch_avoidance_gate(tr.touch_avoidance_reason)
    return monitor


def _capture_logs(fn):
    """跑 fn，返回它写下的全部日志文本（用于断言「文案说了什么」）。"""
    records: list[str] = []

    class _H(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record.getMessage())

    handler = _H()
    root = logging.getLogger()
    root.addHandler(handler)
    old_level = root.level
    root.setLevel(logging.DEBUG)
    try:
        fn()
    finally:
        root.removeHandler(handler)
        root.setLevel(old_level)
    return "\n".join(records)


# ---------------------------------------------------------------- 1. 判据成因


def test_an_upstream_fault_cooldown_is_named_an_upstream_fault():
    """**靶点**：三种上游故障冷却的触点成因必须如实返回，不能折叠成 `cooldown`。"""
    tr = _fresh_tracker()
    for kind in _UPSTREAM_FAULT_CAUSES:
        tr.reset()
        cd = _cool(tr, "qwen", kind)
        assert cd > 0, f"{kind} 没有产生冷却，用例前提不成立"
        reason = tr.touch_avoidance_reason("qwen")
        assert reason == kind, (
            f"{kind} 冷却的触点成因被报成了 {reason!r}。`cooldown` 在探针/自愈/会话回收"
            f"三处文案里都读作「风控避让 / 规避上游封禁」，而上游故障与风控的处置方式"
            f"完全不同（等上游恢复 vs 等窗口/人工介入）"
        )


# ---------------------------------------------------------------- 2. 探针文案


def test_the_health_probe_does_not_claim_risk_control_for_an_upstream_fault():
    """**靶点**：探针的 `last_error` 与日志不得把上游故障说成风控。"""
    tr = _fresh_tracker()
    _cool(tr, "qwen", "upstream_error")
    monitor = _make_monitor(tr)

    logs = _capture_logs(lambda: asyncio.run(monitor.check_provider("qwen")))
    st = monitor.get_status("qwen")
    assert st["status"] == "cooling"
    err = st["last_error"] or ""
    for text in (err, logs):
        for word in _RISK_CLAIMS:
            assert word not in text, (
                f"一次普通 502 的探针读数里出现了 {word!r}。原文: {text[:240]}"
            )
    assert "上游" in err, (
        f"探针读数没有说明真实成因是上游故障，运维只能读到一句「冷却中」: {err!r}"
    )


# ---------------------------------------------------------------- 3. 自愈选举


def test_the_healer_labels_an_upstream_fault_honestly():
    """**靶点**：自愈选举的跳过文案里，上游故障不得读作风控。"""
    from healing.healer import pick_healer_model

    tr = _fresh_tracker()
    _cool(tr, "qwen", "timeout")
    monitor = _make_monitor(tr)

    # 选举以 glm 为故障节点、qwen 为避让候选：`_avoiding()` 对每个候选都会记一条跳过日志，
    # 因此这里必定有输出（不能因为「可能没有日志」就退化成只断言标签表——那会漏掉真正
    # 写进日志的那句话，而运维读到的正是日志）。
    logs = _capture_logs(lambda: pick_healer_model(monitor, "glm"))
    assert logs, "自愈选举没有为处于避让期的节点写日志，用例前提不成立"
    for word in _RISK_CLAIMS:
        assert word not in logs, f"自愈选举把上游超时说成了 {word!r}: {logs[:240]}"
    assert "上游" in logs, f"自愈选举没有说出真实成因: {logs[:240]}"


# ---------------------------------------------------------------- 4. 会话回收


def test_conversation_gc_labels_an_upstream_fault_honestly():
    """**靶点**：临时会话回收的推迟文案里，上游故障不得读作风控。"""
    tr = _fresh_tracker()
    _cool(tr, "qwen", "truncated")
    gc = ConversationGC()

    logs = _capture_logs(lambda: gc._record_defer("qwen", tr.get_remaining_cooldown("qwen"), "truncated"))
    for word in _RISK_CLAIMS:
        assert word not in logs, f"会话回收把上游截断说成了 {word!r}: {logs[:240]}"
    assert "上游" in logs, f"会话回收没有说出真实成因: {logs[:240]}"


# ---------------------------------------------------------------- 5. 零回归守卫


def test_a_genuine_waf_cooldown_still_reads_as_risk_control():
    """零回归**守卫**：真正的人机验证冷却照旧报 `cooldown`、照旧说「风控」。

    修复前就是绿的，且必须如此：把范围收窄到「非风控成因」时若顺手把风控也改了，
    `login.py all` 的「冷却 → 照常登录」分支会失效（OPT-029 的长效人机验证避让
    唯一的解除路径就是人工完成滑块后重登），比原缺陷更糟。
    """
    tr = _fresh_tracker()
    _cool(tr, "qwen", "waf")
    assert tr.touch_avoidance_reason("qwen") == "cooldown", (
        "真正的人机验证冷却必须仍报 `cooldown`（风控），否则 login.py 的修复入口会失效"
    )
    monitor = _make_monitor(tr)
    asyncio.run(monitor.check_provider("qwen"))
    err = monitor.get_status("qwen")["last_error"] or ""
    assert "风控" in err, f"真正的人机验证冷却的探针读数丢了「风控」二字: {err!r}"


# ---------------------------------------------------------------- 6. 词汇表


def test_the_documented_cause_vocabulary_lists_every_cause_the_gate_can_return():
    """**靶点**：`touch_gates.causes` 是这份词汇表唯一的公开声明，必须完整。

    它与另外三张标签表是同一事实的读点：若诊断只列四种成因，运维读到一份
    「成因只有这些」的清单，看到第五种时会以为那是别的什么东西。
    """
    tr = _fresh_tracker()
    causes = set(tr.touch_gate_report()["causes"])
    for kind in _UPSTREAM_FAULT_CAUSES:
        assert kind in causes, (
            f"`touch_gates.causes` 缺 {kind!r}，而 `touch_avoidance_reason()` 会返回它："
            f"诊断因此声称一份不完整的成因词汇表"
        )


def run_all():
    test_an_upstream_fault_cooldown_is_named_an_upstream_fault()
    test_the_health_probe_does_not_claim_risk_control_for_an_upstream_fault()
    test_the_healer_labels_an_upstream_fault_honestly()
    test_conversation_gc_labels_an_upstream_fault_honestly()
    test_a_genuine_waf_cooldown_still_reads_as_risk_control()
    test_the_documented_cause_vocabulary_lists_every_cause_the_gate_can_return()
    print("\n✅ All touch-cause fidelity tests passed! (6 cases)")


if __name__ == "__main__":
    run_all()
