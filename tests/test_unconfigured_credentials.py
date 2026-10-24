"""未配置凭证 ≠ 上游拒绝：配置缺口不得被记成风控拦截 (tests/test_unconfigured_credentials.py)。

[DAY-ZERO] 每一个新用户都会先撞上这个状态：`git clone` 之后 `session/` 是空的，
但网关已经可以启动了。此时向它发一个请求，`_make_provider` 发现**本地**根本没有
凭证文件，于是抛出：

    ProviderError("未检测到 Qwen 令牌。…", status=401, err_type="auth_error")

`execute_chat` 的 `except` 分支只读 `exc.status`，于是这条**从未离开本机**的
结论被当成上游 HTTP 401 交给了 `record_failure`。

一次 401 与一次「我们没有配置」在成因上正相反，而 `record_failure` 对 401 的
处置是全项目**最有耐心也最重**的那一档（`auth` 级）：

    第 1 轮 -> 60s     第 2 轮 -> 60s     第 3 轮 -> 1800s（`GATEWAY_AUTH_COOLDOWN_BASE`）

一次 agent CLI 的重试（`claude -p` 实测在 40 秒内会重发 7 次）就足以让五个节点
全部进入 30 分钟认证级避让。而这一切的**上游调用次数是 0**。

实测（修复前，真实 CooldownTracker，隔离目录）：

    cooldown = {'qwen': 1799.9, 'deepseek': 1800.0, 'doubao': 1800.0,
                'kimi': 1800.0, 'glm': 1800.0}
    凭证续期记录: {'tracked_providers': [], 'records': {}}

后两行是这个缺陷真正的危害，且它比「记录了本不该记录的东西」严重得多：

`note_auth_failure()` 在凭证文件**不存在**时返回 `False` 且**不写任何记录**
（`credential_fingerprint()` 返回 None）。而 OPT-026 的复位正是靠「记录里的指纹
变了」来判定续期的——没有记录就没有比较对象，`renewed_providers()` 永远返回空。

于是形成闭环自锁：运维照着网关自己的报错执行 `login.py`，凭证文件出现了、变了，
但 `release_renewed_credentials()` 仍然什么都不复位，节点继续在 30 分钟避让里
——**网关让运维做的事，做完之后没有任何效果**。这正是 OPT-026 记录过的那类
自锁缺陷（凭证失效制造的状态锁死了修复它的机制），换了个触发条件。

本套件锁住四件事：
1. 未配置凭证**不写**风控台账、**不进**恢复观察期、**不上报**全环失效检测器；
2. 它仍要让该节点**不被选为可用**（不能因为「不算风险」就让它被反复派发）；
3. 但它的避让**不得**是 `auth` 级长效档——否则重登修不掉它；
4. 最关键的一条：运维补上凭证之后，该节点必须**立刻恢复可用**（自锁在此断开）。

运行: .venv/bin/python tests/test_unconfigured_credentials.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["TESTING"] = "1"
os.environ["GATEWAY_DISABLE_AUTO_REFRESH"] = "1"

_SANDBOX = tempfile.mkdtemp(prefix="dayzero_")
os.environ["GATEWAY_RISK_LEDGER_DIR"] = _SANDBOX
os.environ["GATEWAY_QUARANTINE_DIR"] = _SANDBOX
os.environ["GATEWAY_CONVERSATION_GC_DIR"] = _SANDBOX
os.environ["GATEWAY_DEVICE_IDENTITY_DIR"] = _SANDBOX

PROVIDERS = ("qwen", "deepseek", "doubao", "kimi", "glm")
_not_configured_msg = "未检测到 Qwen 令牌。请运行 .venv/bin/python login.py qwen 扫码登录，或配置 QWEN_AUTH_TOKEN"

_session_dir = Path(__file__).resolve().parent.parent / "session"


def _fresh_tracker():
    """每个用例都从一个干净的 tracker 开始（避免用例之间互相继承冷却）。"""
    from providers.failover import CooldownTracker

    tracker = CooldownTracker()
    for pk in PROVIDERS:
        tracker._cooldowns.pop(pk, None)
        tracker._failure_counts.pop(pk, None)
        tracker._cooldown_severity.pop(pk, None)
        tracker._episode_ends.pop(pk, None)
    return tracker


def test_unconfigured_is_not_recorded_as_a_risk_interception():
    """[DAY-ZERO] 一次「本机没有凭证」不得写进风控台账。

    台账是运维研判的**事实依据**（OPT-023 档案的全部输入）。把一次本地配置检查
    写进去，等于让档案声称「qwen 今天被上游拦截了 1 次」——而那个账号今天根本
    没有被调用过。这与 [M2] 把超时记成 `waf`、[S2] 把截断记成 `waf` 是同一类谎报，
    只是这次连上游都没碰。

    实测（修复前）：`session/risk_ledger.jsonl` 里出现
    `{"status": 401, "severity": "auth", "reason": "未检测到 Qwen 令牌…"}`。
    """
    tracker = _fresh_tracker()
    before = tracker.risk_ledger.recent_interceptions(hours=1.0)["total"]
    asyncio.run(tracker.record_failure("qwen", 401, _not_configured_msg))
    report = tracker.risk_ledger.recent_interceptions(hours=1.0)
    added = report["per_provider"].get("qwen")

    assert report["total"] == before, (
        f"「本机没有配置 qwen 凭证」被写进了风控台账（新增 {report['total'] - before} 条：{added}）。"
        f"这个账号从未被上游拦截过——台账是档案的唯一输入，谎报会直接误导运维"
    )
    print("test_unconfigured_is_not_recorded_as_a_risk_interception: PASSED")


def test_unconfigured_does_not_record_a_recovery_probation_episode():
    """未配置凭证不该**记录**恢复观察期：没有「解封惊群」可言。

    OPT-022 的观察期是为**封禁解除瞬间的需求释放**准备的。一个从未配置过的节点
    解封时释放的是「用户刚扫码」这一件事，没有需要串行化保护的历史积压。

    断言的是 `_episode_ends`（**记录下来的** episode），而不是 `is_in_probation()`：
    后者在冷却仍在生效时恒为 False（函数体第一句就 `return False`），
    于是「冷却期间查一次」永远通过——那是一个与缺陷无关的断言，正属于
    [M1] 记录过的「断言的不是代码真正保护的那个窗口」。真正的危害在冷却**到期之后**
    才显现：节点被串行化并放大步调，而它刚刚才第一次被扫码登录。
    """
    tracker = _fresh_tracker()
    asyncio.run(tracker.record_failure("qwen", 401, _not_configured_msg))
    assert "qwen" not in tracker._episode_ends, (
        f"未配置凭证记录了一个恢复观察期 episode（_episode_ends={dict(tracker._episode_ends)}）——"
        f"该 episode 会在冷却到期后生效，把一个刚完成首次登录的节点串行化并放大步调，"
        f"而它是为封禁解除的惊群准备的机制"
    )
    print("test_unconfigured_does_not_record_a_recovery_probation_episode: PASSED")


def test_unconfigured_does_not_report_a_ring_wide_credential_outage():
    """五个节点都没配置 ≠ 「整环凭证已失效」。

    OPT-025 的措辞是「我们没有可用凭证了」，它成立的前提是**存在过**可用凭证。
    全新克隆时五个都空，这句话把「你还没登录」说成了一个故障。
    """
    tracker = _fresh_tracker()
    for pk in PROVIDERS:
        asyncio.run(tracker.record_failure(pk, 401, _not_configured_msg))
    watch = tracker.credential_watch
    assert not watch.is_outage_active(), (
        f"五个节点都没有配置凭证，却被判成「整环凭证失效」（failed={watch.active_failures()}）。"
        f"该判定的语义是「有凭证但都失效了」，而这里根本还没有凭证"
    )
    print("test_unconfigured_does_not_report_a_ring_wide_credential_outage: PASSED")


def test_an_unconfigured_node_is_still_not_dispatched():
    """反面：不算风控，也必须让它**不可用**。

    这条防的是「修过头」——如果未配置的节点被当成完全健康，负载均衡与会话亲和就会
    反复把它选为目标，每个客户端的每一次请求都要先为它付出一次构造失败。
    """
    tracker = _fresh_tracker()
    asyncio.run(tracker.record_failure("qwen", 401, _not_configured_msg))
    assert tracker.get_remaining_cooldown("qwen") > 0 or tracker.is_disabled("qwen"), (
        "未配置的节点仍被报成可用——它会成为负载均衡与会话亲和的候选目标，"
        "每个请求都要为它白付一次构造失败"
    )
    print("test_an_unconfigured_node_is_still_not_dispatched: PASSED")


def test_unconfigured_avoidance_does_not_escalate_to_the_auth_tier():
    """未配置不得进入 `auth` 级长效档——否则重登修不掉它。

    这是自锁的**直接成因**：`auth` 级在连续 3 次后升到 `GATEWAY_AUTH_COOLDOWN_BASE`
    （1800s），而 `note_auth_failure()` 对「没有凭证文件」不写记录，于是 OPT-026 的
    续期复位永远找不到这个节点（`renewed_providers()` 只扫描已记录节点）。
    实测（修复前）：三轮之后 cooldown 全部 1800s，补上凭证后复位列表为 `[]`。
    """
    tracker = _fresh_tracker()
    for _ in range(3):
        for pk in PROVIDERS:
            asyncio.run(tracker.record_failure(pk, 401, _not_configured_msg))

    worst = max(tracker.get_remaining_cooldown(pk) for pk in PROVIDERS)
    # 阈值必须落在**升级档与短冷却之间的真空带**里，不能写成 `< 1800`：
    # 升级后的冷却恰好就是 1800，读回来是 1799.9（已扣掉流逝的时间），
    # `< 1800` 会让这个用例在缺陷存在时**通过**——OPT-031 记录过同一个陷阱
    # （把阈值取成 1800 会让判决立即翻转）。这里明确取「远低于升级下界」。
    assert worst < 600.0, (
        f"未配置凭证在 3 轮之后升到了 {worst:.0f}s 的认证级长效避让。"
        f"这一档的解除路径是「凭证指纹变化 -> 自动复位」，而凭证文件**不存在**时"
        f"压根不会写下指纹记录，所以复位永远找不到它 —— 重登修不掉，"
        f"正是 OPT-026 已经修过的那类自锁"
    )
    print("test_unconfigured_avoidance_does_not_escalate_to_the_auth_tier: PASSED")


def test_logging_in_immediately_makes_the_node_usable_again():
    """[DAY-ZERO] 最关键的一条：补上凭证之后，节点必须立刻恢复。

    这是自锁的闭环。真实用户流程是：启动网关 -> 收到「请执行 login.py」-> 执行 ->
    **继续用**。若第三步无效，那么网关给出的每一条处置建议都是空话。

    实测（修复前）：登录后 `release_renewed_credentials()` 返回 `[]`，
    五个节点的 cooldown 仍是 1800s。
    """
    tracker = _fresh_tracker()
    for _ in range(3):
        for pk in PROVIDERS:
            asyncio.run(tracker.record_failure(pk, 401, _not_configured_msg))

    assert tracker.get_remaining_cooldown("qwen") > 0, "前置条件不成立：qwen 应当处于避让中"

    # 模拟运维执行 login.py：凭证文件被创建（内容新鲜，但此前从不存在）
    created = []
    _session_dir.mkdir(parents=True, exist_ok=True)
    for pk in PROVIDERS:
        f = _session_dir / f"{pk}.json"
        if f.exists():
            continue
        f.write_text(json.dumps({"token": f"fresh-{pk}-token", "captured_at": time.time()}))
        created.append(f)
    try:
        released = tracker.release_renewed_credentials_now()
        remaining = {pk: round(tracker.get_remaining_cooldown(pk), 1) for pk in PROVIDERS}
    finally:
        for f in created:
            f.unlink(missing_ok=True)

    assert not any(v > 0 for v in remaining.values()), (
        f"补上凭证后节点仍处于避让（released={released}, cooldown={remaining}）。"
        f"网关让运维执行 `login.py`，执行完之后却仍然不可用 —— "
        f"这正是 DAY-ZERO 自锁：未配置状态没有留下指纹记录，续期检测永远比对不到"
    )
    print("test_logging_in_immediately_makes_the_node_usable_again: PASSED")


def test_end_to_end_a_fresh_clone_pollutes_no_risk_state():
    """[DAY-ZERO] 端到端：走**真实 HTTP 路径**，当天状态不得被污染。

    上面的用例直接驱动 `CooldownTracker`，回答「状态机怎么记」；但那只证明状态机本身
    正确，证明不了 `execute_chat` 的 `except` 分支**真的**把 `_make_provider` 的
    401 交给它——一个只改 `record_failure` 的修复会让单元用例全绿而真实请求照旧污染。

    所以这里驱动完整栈（TestClient -> 路由 -> 派发循环 -> 拒绝），断言的是**状态**，
    不是消息文本：

      * 响应必须是失败状态（200 会让 agent CLI 以为成功）；
      * 风控台账不得新增任何一条（该账号从未被上游拦截）；
      * 不得出现 `auth` 级长效避让（那是重登修不掉的那一档）。

    **断言窗口的取舍**：这里不用「`_make_provider` 调用次数 == 0」来代表零上游——
    构造 Provider 是纯本地操作，一次网络请求都没有，把它当上游触点属于
    [M1] 记录的「断言的不是代码真正保护的那个窗口」。改用 tripwire 顶替工厂并
    原样抛出 `ProviderError`：既保证本用例永远不会真的联网，也不把「是否提前
    判定」这种实现细节写进断言。
    """
    import server as srv
    from fastapi.testclient import TestClient
    from providers.base import ProviderError
    from providers.failover import CooldownTracker

    tracker = _fresh_tracker()
    before = tracker.risk_ledger.recent_interceptions(hours=1.0)["total"]

    savings = {"local_refusals": 0}
    real_make = srv._make_provider
    saved_resolvers = {pk: getattr(srv, f"resolve_{pk}") for pk in PROVIDERS}

    def _local_refusal(key, client=None):
        # 与真实 `_make_provider` 在「无凭证」时的行为逐字一致：纯本地判定，零网络
        savings["local_refusals"] += 1
        _r, err_msg, _f = srv.PROVIDER_SPECS[key]
        raise ProviderError(err_msg, status=401, err_type="auth_error")

    for pk in PROVIDERS:
        setattr(srv, f"resolve_{pk}", lambda: None)
    srv._make_provider = _local_refusal
    try:
        resp = TestClient(srv.app).post(
            "/v1/messages",
            json={
                "model": "claude-3-5-sonnet",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
            },
            headers={"anthropic-version": "2023-06-01"},
        )
    finally:
        srv._make_provider = real_make
        for pk, fn in saved_resolvers.items():
            setattr(srv, f"resolve_{pk}", fn)

    report = tracker.risk_ledger.recent_interceptions(hours=1.0)
    remaining = {pk: round(tracker.get_remaining_cooldown(pk), 1) for pk in PROVIDERS}
    worst = max(remaining.values())

    assert resp.status_code >= 400, (
        f"全新克隆（无任何凭证）返回了 {resp.status_code} —— agent CLI 会把它当成成功"
    )
    assert report["total"] == before, (
        f"一次全新克隆的请求写进了风控台账（新增 {report['total'] - before} 条："
        f"{report['per_provider']}）。这些账号从未被上游拦截，台账却是档案的唯一输入"
    )
    assert worst < 600.0, (
        f"一次全新克隆的请求就让节点进入了 {worst:.0f}s 的避让（{remaining}），"
        f"且该档位是重登修不掉的认证级长效档"
    )
    print(
        f"test_end_to_end_a_fresh_clone_pollutes_no_risk_state: PASSED "
        f"(status={resp.status_code}, local_refusals={savings['local_refusals']}, "
        f"ledger_added=0, worst_cooldown={worst:.1f}s)"
    )


def run_all():
    test_unconfigured_is_not_recorded_as_a_risk_interception()
    test_unconfigured_does_not_record_a_recovery_probation_episode()
    test_unconfigured_does_not_report_a_ring_wide_credential_outage()
    test_an_unconfigured_node_is_still_not_dispatched()
    test_unconfigured_avoidance_does_not_escalate_to_the_auth_tier()
    test_logging_in_immediately_makes_the_node_usable_again()
    test_end_to_end_a_fresh_clone_pollutes_no_risk_state()
    print("\n✅ All day-zero (unconfigured credentials) tests passed!")


if __name__ == "__main__":
    run_all()