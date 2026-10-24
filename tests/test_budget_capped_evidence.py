"""预算裁短的超时：不诬告，但也不能永远免费试探 (tests/test_budget_capped_evidence.py)。

[BUDGET-4] 上一轮把「被总预算裁出来的超时」从节点失败里摘出来了（`BudgetCappedTimeout`），
方向是对的：份额 0.1s 到了只说明**我们只肯等 0.1s**，不说明节点有故障，按节点失败
记一次 60s 冷却是诬告一个健康节点。

但摘出来之后走向了另一个极端：**完全没有记账**。于是一个真的挂住的节点会在每一次
请求上重新被派发一次、白等一整份份额，永远学不到任何东西。实测（五节点环、qwen 挂死、
其余立刻应答、预算 5s → 首份 1.0s、连发四轮）：

    turn 1: 1.00s  served_by=glm  qwen_calls=1  qwen_cooldown=0.0s  qwen_failures=0
    turn 2: 1.00s  served_by=glm  qwen_calls=2  qwen_cooldown=0.0s  qwen_failures=0
    turn 3: 1.00s  served_by=glm  qwen_calls=3  qwen_cooldown=0.0s  qwen_failures=0
    turn 4: 1.00s  served_by=glm  qwen_calls=4  qwen_cooldown=0.0s  qwen_failures=0

每一轮都是 1.00s，`qwen_calls` 单调增长，冷却与失败计数恒为 0 —— 对长会话的 agent
客户端就是「每次请求都稳定多等一秒」，而且永不收敛。这既是延迟缺陷，也是**上游流量**
缺陷：每一次都真的向一个已知不吐字的节点发起了一次带凭证的调用。

正确的语义在两者之间：单次被裁短的超时**不构成节点故障的证据**，但「连续 N 次在你
给它的全部份额里都没能吐首包」构成一条不同的事实——**这个候选比它的份额慢**。这不是
诬告（不写成 `waf`/`auth`、不进风控台账、不进恢复观察期），而是如实按上游超时计一次
失败并冷却，让后续请求直接走健康节点；冷却到期后再探一次，节点若真的恢复了，
`record_success` 会把证据清零。

阈值可配：`GATEWAY_BUDGET_CAPPED_LIMIT`（默认 3）。设成 0 表示「永不升级」，
即回到上一轮的行为（每一次都白等一份份额）——那是本用例要钉住的错误方向。

**本套件走真实 HTTP 路径**（`TestClient` + 脚本化 provider），不是直接调
`execute_chat`。理由是一条实测教训：`execute_chat` 对处在**短软冷却**里的主选节点
仍会派发（那是 OPT-021 刻意保留的「最后一搏」），真正把请求改派到健康节点的是它上游
的负载均衡层。直接在路由器层量「第 4 轮是否还等一秒」，量到的是那一层的语义，不是
客户端经历的语义——用真实的 server 路径量，测的才是 agent 每次请求实际等了多久。

运行: .venv/bin/python tests/test_budget_capped_evidence.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["TESTING"] = "1"
os.environ["GATEWAY_DISABLE_AUTO_REFRESH"] = "1"
os.environ["GATEWAY_DISABLE_PROBING"] = "1"
os.environ["GATEWAY_DISABLE_PACING"] = "1"
os.environ["GATEWAY_DISABLE_SINGLEFLIGHT"] = "1"
os.environ["GATEWAY_DISABLE_PROBATION"] = "1"
os.environ["GATEWAY_DISABLE_RISK_CORRELATION"] = "1"
os.environ["GATEWAY_DISABLE_CREDENTIAL_WATCH"] = "1"
os.environ["GATEWAY_DISABLE_CREDENTIAL_RENEWAL"] = "1"
os.environ["GATEWAY_DISABLE_DEVICE_IDENTITY"] = "1"
for _p in ("QWEN", "DEEPSEEK", "DOUBAO", "KIMI", "GLM"):
    os.environ[f"{_p}_AUTH_TOKEN"] = "fake_budget"

from fastapi.testclient import TestClient  # noqa: E402

import server as srv  # noqa: E402

_BODY = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}


class _HungProvider:
    """连上就永不吐字：预算份额一到就被判超时。"""

    def __init__(self):
        self.calls = 0

    async def chat(self, *a, **kw):
        self.calls += 1
        await asyncio.Event().wait()
        yield "", {}


class _FastProvider:
    def __init__(self, key: str):
        self.key = key
        self.calls = 0

    async def chat(self, *a, **kw):
        self.calls += 1
        yield "ok", {"conversation_id": f"{self.key}_c"}
        yield "", {"conversation_id": f"{self.key}_c"}


class _SlowButHealthy:
    """健康，但比给它的份额慢——只该被裁短，不该被定罪。"""

    def __init__(self, delay: float):
        self.delay = delay
        self.calls = 0

    async def chat(self, *a, **kw):
        self.calls += 1
        await asyncio.sleep(self.delay)
        yield "慢但健康", {"conversation_id": "q"}
        yield "", {"conversation_id": "q"}


def _env(**kv: str) -> None:
    for k, v in kv.items():
        os.environ[k] = v


def _fresh_state():
    """清空全局路由器的冷却/失败/连续计数，使用例之间互不继承。"""
    tracker = srv.failover_router.cooldown_tracker
    tracker._cooldowns.clear()
    tracker._cooldown_severity.clear()
    tracker._failure_counts.clear()
    tracker.disabled_providers.clear()
    tracker.reset()
    return tracker


def _turns(hung, fast: dict, count: int):
    """走真实 HTTP 路径连发 `count` 轮，返回 [(耗时, 本轮实际服务的节点)]。"""

    def factory(key):
        return hung if key == "qwen" else fast[key]

    original = srv._make_provider
    srv._make_provider = factory
    try:
        client = TestClient(srv.app)
        results = []
        for _ in range(count):
            before = {k: p.calls for k, p in fast.items()}
            t0 = time.monotonic()
            r = client.post("/v1/chat/completions", json=_BODY)
            elapsed = time.monotonic() - t0
            assert r.status_code == 200, r.text
            served = next(
                (k for k, p in fast.items() if p.calls > before[k]), "?"
            )
            results.append((elapsed, served))
        return results
    finally:
        srv._make_provider = original


def _fast_set():
    return {k: _FastProvider(k) for k in ("glm", "kimi", "doubao", "deepseek")}


def test_a_consistently_hung_node_stops_being_paid_for_on_every_request():
    """连续被预算裁短的超时最终必须变成一次冷却，而不是每次请求都白等一份。

    修复前实测（见模块 docstring）：每轮都是 1.00s、`qwen_calls` 单调增长、冷 0。
    修复后：前 `GATEWAY_BUDGET_CAPPED_LIMIT`（默认 3）轮如实不记账，第 3 轮达到阈值
    后按上游超时计一次冷却，第 4 轮起负载均衡直接改派健康节点、不再触碰 qwen。
    """
    tracker = _fresh_state()
    _env(
        GATEWAY_FIRST_CHUNK_TIMEOUT="30",
        GATEWAY_HANDSHAKE_BUDGET="5",
        GATEWAY_STREAM_IDLE_TIMEOUT="5",
        GATEWAY_BUDGET_CAPPED_LIMIT="3",
    )
    hung = _HungProvider()
    fast = _fast_set()

    results = _turns(hung, fast, 5)
    later = results[3:]
    slow_later = [f"{elapsed:.2f}s" for elapsed, _ in later if elapsed > 0.5]
    assert not slow_later, (
        f"第 4 轮起仍然在为一个挂死的节点白等预算份额（实测各轮耗时 "
        f"{[round(e, 2) for e, _ in results]}）。连续被预算裁短的超时必须能在某个阈值后"
        f"变成一次冷却，否则长会话的 agent 每次请求都稳定多等一份份额、永不收敛，"
        f"而且每次都真的向已知不吐字的节点发起了一次带凭证的调用。"
        f"qwen_calls={hung.calls} 冷却={tracker.get_remaining_cooldown('qwen'):.1f}s "
        f"失败计数={tracker._failure_counts.get('qwen', 0)}"
    )
    # 前 3 轮必须仍是「不诬告」的：阈值那一轮之前不冷却、不记失败。
    assert hung.calls >= 3, (
        f"挂死的节点只被尝试了 {hung.calls} 次就到了阈值——阈值被过早触发"
    )
    assert hung.calls <= 4, (
        f"挂死的节点被尝试了 {hung.calls} 次（5 轮里）——阈值没有生效，"
        f"冷却窗口内不得再被派发"
    )
    # 每一轮都应有一个**健康**节点真的服务了响应（`"="?"` 表示没有任何假 provider
    # 被调用过，那说明请求走了一条本用例没有预期的路径，结论不可信）。
    assert all(served in fast for _e, served in results), (
        f"有请求没有被任何脚本化 provider 服务：{results}"
    )
    assert tracker.cooldown_cause("qwen") == "timeout", (
        f"升级后的成因必须是 timeout（上游故障），实际 "
        f"{tracker.cooldown_cause('qwen')!r}——若落成 waf/auth，运维会被指引去给一个"
        f"从未拦截过我们的平台做人工验证或重新登录"
    )
    print(
        "test_a_consistently_hung_node_stops_being_paid_for_on_every_request: PASSED",
        f"(turn times {[round(e, 2) for e, _ in results]}, qwen_calls={hung.calls}, "
        f"cooldown={tracker.get_remaining_cooldown('qwen'):.1f}s)",
    )


def test_a_single_capped_timeout_still_does_not_convict_the_node():
    """零回归：**一次**被裁短的超时仍然不得给节点上冷却。

    这是上一轮 `BudgetCappedTimeout` 存在的全部理由。阈值机制不能把它吃掉：
    健康但比份额慢一次的节点必须立刻还能被下一次请求使用。
    """
    tracker = _fresh_state()
    _env(
        GATEWAY_FIRST_CHUNK_TIMEOUT="30",
        GATEWAY_HANDSHAKE_BUDGET="0.3",
        GATEWAY_STREAM_IDLE_TIMEOUT="5",
        GATEWAY_BUDGET_CAPPED_LIMIT="3",
    )
    slow = _SlowButHealthy(0.4)
    fast = _fast_set()

    _turns(slow, fast, 1)
    cooldown = tracker.get_remaining_cooldown("qwen")
    failures = tracker._failure_counts.get("qwen", 0)
    assert cooldown == 0.0 and failures == 0, (
        f"**一次**被预算裁短的超时就把节点记成了失败：冷却 {cooldown:.1f}s、"
        f"失败计数 {failures}。份额只是请求的边界，节点在 0.4s 内本可以正常应答。"
    )
    assert slow.calls == 1, slow.calls
    print(
        "test_a_single_capped_timeout_still_does_not_convict_the_node: PASSED",
        f"(cooldown={cooldown:.1f}s, failures={failures})",
    )


def test_threshold_zero_restores_the_never_escalate_behaviour():
    """`GATEWAY_BUDGET_CAPPED_LIMIT=0` 必须逐值回到上一轮：永不升级、永不冷却。

    关闭开关要能被证明是关闭，而不是「恰好没触发」。
    """
    tracker = _fresh_state()
    _env(
        GATEWAY_FIRST_CHUNK_TIMEOUT="30",
        GATEWAY_HANDSHAKE_BUDGET="5",
        GATEWAY_STREAM_IDLE_TIMEOUT="5",
        GATEWAY_BUDGET_CAPPED_LIMIT="0",
    )
    hung = _HungProvider()
    fast = _fast_set()

    _turns(hung, fast, 4)
    assert tracker.get_remaining_cooldown("qwen") == 0.0, (
        "GATEWAY_BUDGET_CAPPED_LIMIT=0 时仍然给了冷却——关闭开关没有生效"
    )
    assert hung.calls == 4, f"关闭后 qwen 应被尝试 4 次，实际 {hung.calls}"
    print(
        "test_threshold_zero_restores_the_never_escalate_behaviour: PASSED",
        f"(qwen_calls={hung.calls}, cooldown=0)",
    )


def test_recovery_clears_the_accumulated_evidence():
    """节点恢复后证据必须清零，否则它会带着旧账在很长的未来里反复被冷却。

    「连续 N 次」的语义是**连续**：中间成功过一次，计数必须归零。
    """
    tracker = _fresh_state()
    _env(
        GATEWAY_FIRST_CHUNK_TIMEOUT="30",
        GATEWAY_HANDSHAKE_BUDGET="10",
        GATEWAY_STREAM_IDLE_TIMEOUT="5",
        GATEWAY_BUDGET_CAPPED_LIMIT="3",
    )

    async def main():
        await tracker.note_budget_capped("qwen")
        await tracker.note_budget_capped("qwen")
        assert tracker.budget_capped_count("qwen") == 2
        await tracker.record_success("qwen")
        assert tracker.budget_capped_count("qwen") == 0, (
            f"成功之后连续计数没有清零，仍是 {tracker.budget_capped_count('qwen')}——"
            f"下一次单次被裁短就会立刻触发升级，等于把一笔旧账算在现在的节点头上"
        )

    asyncio.run(main())
    print("test_recovery_clears_the_accumulated_evidence: PASSED")


def test_a_node_already_cooled_for_this_is_reconvicted_on_the_first_attempt():
    """冷却到期后的**第一次**被裁短就必须再次冷却，不能又白等三份。

    [BUDGET-4] 第一版把冷却后的连续计数清零，于是每个冷却周期都重新发放
    `GATEWAY_BUDGET_CAPPED_LIMIT`（3）次完整的份额尝试。实测（份额 1.0s、冷却 1s、
    每轮之间等 1.2s 让窗口真正过期）：**14/14 轮全部停顿 1 秒**、qwen 被真实触碰
    14 次。也就是说请求间隔一旦大于冷却时长（长程 agent 的两次工具调用之间往往
    是分钟级），缺陷就退化成「每次请求都白等一份」——正是本套件要消灭的那个形状，
    只是周期从「每请求」变成了「每冷却窗口」。

    正确语义是**恢复观察期**（OPT-022 的同一个思想）：冷却到期后的第一次尝试就是
    金丝雀，它同时是唯一的恢复探测点。它若在份额内吐首包，`record_success` 会把
    升级历史一并清空（节点已自证恢复）；它若再次被裁短，就是「刚为这个原因冷却过，
    立刻复现」——这一次尝试本身就足够定罪，不需要再攒三次。
    """
    tracker = _fresh_state()
    _env(
        GATEWAY_FIRST_CHUNK_TIMEOUT="30",
        GATEWAY_HANDSHAKE_BUDGET="5",
        GATEWAY_STREAM_IDLE_TIMEOUT="5",
        GATEWAY_BUDGET_CAPPED_LIMIT="3",
        GATEWAY_TIMEOUT_COOLDOWN="0.2",
    )
    hung = _HungProvider()
    fast = _fast_set()

    # 前 3 轮：攒够证据、升级冷却。
    _turns(hung, fast, 3)
    assert tracker.budget_actions("qwen") == 1, tracker.budget_actions("qwen")
    calls_after_first_cycle = hung.calls

    # 等到冷却真的过期（真实路径上由负载均衡改派健康节点后，这里直接推进时钟）。
    import time as _t

    while tracker.get_remaining_cooldown("qwen") > 0:
        _t.sleep(0.05)

    # 冷却过期后的第一次尝试：必须只白等**一份**就再次冷却。
    results = _turns(hung, fast, 1)
    assert tracker.budget_actions("qwen") == 2, (
        f"冷却到期后的第一次被裁短没有再次定罪（升级次数仍是 "
        f"{tracker.budget_actions('qwen')}，qwen 又被尝试了 "
        f"{hung.calls - calls_after_first_cycle} 次）。每个冷却周期重新发放 3 次完整"
        f"份额，等价于「只要请求间隔大于冷却时长，每次请求都要白等一份」——"
        f"实测各轮耗时 {[round(e, 2) for e, _ in results]}"
    )
    assert hung.calls - calls_after_first_cycle == 1, (
        f"冷却到期后白等了 {hung.calls - calls_after_first_cycle} 份份额才再次冷却；"
        f"应为 1 份（金丝雀）"
    )

    # 反方向：一次成功必须清空连续计数，否则恢复了的节点下一次单次被裁短
    # 就会被立刻定罪（`budget_actions` 是累计遥测，不随成功清零）。
    async def recover():
        await tracker.record_success("qwen")

    asyncio.run(recover())
    assert tracker.budget_capped_count("qwen") == 0, (
        f"节点成功后连续计数没有清零（仍是 {tracker.budget_capped_count('qwen')}）——"
        f"一个已经自证恢复的节点下一次单次被裁短就会被立刻冷却"
    )
    print(
        "test_a_node_already_cooled_for_this_is_reconvicted_on_the_first_attempt: PASSED",
        f"(canary stalls=1, escalations cleared on success)",
    )


def test_a_non_risk_escalation_never_relabels_a_live_risk_cause():
    """[BUDGET-6] 我们自己的降级决定，绝不能改写一个**仍在生效**的风控成因。

    这是 [BUDGET-3]/[BUDGET-4] 一直没测到的那一格：预算裁短/超时记账都**不是风控
    事件**，但它们和风控事件共用 `_cooldowns` 与 `_cooldown_severity` 两个格子，而
    后者是**无条件覆盖**的，前者却取 `max()`。于是「更长的窗口」被保留了，「更长的
    窗口的成因」被抹掉了。

    实测（probe4，真实 HTTP 路径）：qwen 带着一次真实的人机验证拦截
    `(45.0s, cause='waf')`，环内其余四个节点处于 30s 频控短冷却——那是 OPT-021 刻意
    保留的「最后一搏」区间，所以主选节点照常被派发。它的握手被我们自己的总预算裁短，
    连续计数已饱和在阈值上（BUDGET-5 的金丝雀语义），一次请求之后：

        qwen: (60.0s, cause='timeout')     ← 窗口变长，成因被改写
        release_auth_avoidance_locked(["qwen"], "probe", _RELEASABLE_BY_RENEWAL) -> []

    后果不是「一条日志难看」，而是三个下游同时被误导：
    1. **人工重登出口失效**：waf 是唯一只有人能解除的一类（OPT-023/029），复位路径
       按成因判定，成因变成 timeout 之后运维照档案做完滑块验证也解除不了；
    2. **档案给错建议**：`AVOIDANCE_MATRIX['timeout']` 说的是「重登无效，检查上游服务
       与本地出网」，而这个节点其实是被人机验证挡住的；
    3. **长效避让判读**：timeout 的 60s 远低于 1200s 阈值，一个真正的 waf 长效避让
       会被降级成「短冷却，值得最后一搏」，节点继续被自动重试——正是 OPT-029 要消灭
       的自我维持循环。

    同一形状在 `record_failure` 里**早已存在**（probe5：先种 900s 的 waf，再记一次
    429，窗口仍是 900s 而成因变成 ratelimit，重登复位同样返回 `[]`）。所以本用例断言
    的是**一条对两个写入方都成立的性质**，不是给 `note_budget_capped` 打的补丁：
    非风控记账不得改写仍在生效的风控成因；而风控记账之间可以互相改写（后一次拦截
    就是更新的证据，opt029 的成因限定依赖这一点）。
    """
    tracker = _fresh_state()
    _env(GATEWAY_BUDGET_CAPPED_LIMIT="3", GATEWAY_TIMEOUT_COOLDOWN="60")

    from providers.failover import _RELEASABLE_BY_RENEWAL

    async def main():
        # --- 情形一：BUDGET-5 的预算裁短升级（非风控） -------------------
        await tracker.record_failure("qwen", 403, "RGV587 人机验证 captcha", custom_cooldown=45.0)
        assert tracker.cooldown_cause("qwen") == "waf"
        assert tracker.get_remaining_cooldown("qwen") > 40.0

        # 连续计数饱和 = BUDGET-5 的金丝雀态（一次被裁短即再次定罪）
        tracker._budget_capped_streak["qwen"] = 3
        await tracker.note_budget_capped("qwen")

        assert tracker.get_remaining_cooldown("qwen") >= 55.0, (
            "窗口本身应当延长（取 max 是正确的）——本用例只针对成因"
        )
        assert tracker.cooldown_cause("qwen") == "waf", (
            f"一次预算裁短把仍在生效的人机验证成因改写成了 "
            f"{tracker.cooldown_cause('qwen')!r}。这不是风控事件，不得改写风控成因："
            f"改写后人工重登（唯一出口）将无法解除该避让，档案也会建议运维去检查"
            f"网络而不是完成滑块验证"
        )
        assert tracker.release_auth_avoidance_locked(
            ["qwen"], "probe", _RELEASABLE_BY_RENEWAL
        ) == ["qwen"], (
            "人机验证的避让必须仍可被人工重登解除（这是 waf 的唯一出口）"
        )

        # --- 情形二：record_failure 的超时记账（非风控） -------------------
        tracker.reset()
        await tracker.record_failure("deepseek", 403, "RGV587 人机验证 captcha", custom_cooldown=900.0)
        assert tracker.cooldown_cause("deepseek") == "waf"
        await tracker.record_failure("deepseek", 429, "rate limit 频控")
        assert tracker.get_remaining_cooldown("deepseek") > 800.0, "窗口仍应是 900s 那一段"
        assert tracker.cooldown_cause("deepseek") == "ratelimit", (
            "风控成因之间**可以**互相改写——这是 OPT-029 明确写下的判定（「若之后又发生"
            "其它类拦截，说明是另一类问题，绝不代它释放」），本用例不推翻它，只把它作为"
            "对照，证明规则不是『成因永不改变』。注意这里窗口仍是 900s 那段、成因却已"
            "易主，属于同一类「窗口归属与成因不一致」的遗留问题，已记入队列待判"
        )
        await tracker.record_failure("deepseek", 504, "上游超时", is_timeout=True)
        assert tracker.get_remaining_cooldown("deepseek") > 800.0, "窗口仍应是 900s 那一段"
        assert tracker.cooldown_cause("deepseek") == "ratelimit", (
            f"一次超时记账把仍在生效的频控成因改写成了 "
            f"{tracker.cooldown_cause('deepseek')!r}——超时是上游故障，不是风控事件"
        )

    asyncio.run(main())
    print("test_a_non_risk_escalation_never_relabels_a_live_risk_cause: PASSED")


def test_the_saturated_streak_survives_a_restart():
    """重启不得成为一步绕过：饱和的连续计数必须与冷却一起持久化。

    本仓库对每一份「避免触碰上游」的证据都要求跨重启存活（OPT-011/022/027），
    这条也不例外。否则运维重启一次网关，就把节点在冷却到期后的金丝雀资格
    重新换成了 `limit` 次免费的完整份额尝试。
    """
    tracker = _fresh_state()
    _env(
        GATEWAY_TIMEOUT_COOLDOWN="0.5",
        GATEWAY_BUDGET_CAPPED_LIMIT="3",
    )

    async def main():
        for _ in range(3):
            await tracker.note_budget_capped("qwen")
        assert tracker.budget_capped_count("qwen") == 3, tracker.budget_capped_count("qwen")
        snapshot = tracker._state_snapshot()
        assert snapshot["budget_capped_streak"].get("qwen") == 3, (
            f"达到阈值后连续计数没有写进状态快照：{snapshot['budget_capped_streak']}"
        )

        # 模拟重启：清空内存后按状态快照的同一字段恢复（`_restore_persisted_state`
        # 从磁盘读，这里直接驱动它读取的那个字段，避免用例写进真实 session/ 目录）。
        tracker._budget_capped_streak.clear()
        tracker._budget_capped_escalations.clear()
        for pk, streak in (snapshot.get("budget_capped_streak") or {}).items():
            if isinstance(streak, int) and streak > 0:
                tracker._budget_capped_streak[pk] = streak
        for pk, count in (snapshot.get("budget_capped_escalations") or {}).items():
            if isinstance(count, int) and count > 0:
                tracker._budget_capped_escalations[pk] = count
        assert tracker.budget_capped_count("qwen") == 3, (
            f"重启后连续计数丢失（{tracker.budget_capped_count('qwen')}）——"
            f"节点又拿到了 3 次免费的完整份额尝试，重启成了绕过本机制的捷径"
        )

    asyncio.run(main())
    print("test_the_saturated_streak_survives_a_restart: PASSED")


def test_the_mechanism_is_observable_on_diagnostics():
    """「机制有没有在做事」必须在诊断里看得见，而不是靠翻日志推断。

    `/v1/diagnostics` 上的 `budget_capped` 必须如实反映当前的连续计数、阈值与累计
    升级次数——它的全部价值是让运维能回答「这个节点是不是正在被反复裁短」，
    一个只在 `logger.warning` 里出现的事实在运维界面里等于不存在。
    """
    tracker = _fresh_state()
    _env(
        GATEWAY_BUDGET_CAPPED_LIMIT="3",
        GATEWAY_TIMEOUT_COOLDOWN="60",
    )

    async def main():
        await tracker.note_budget_capped("qwen")
        await tracker.note_budget_capped("qwen")

    asyncio.run(main())
    report = tracker.budget_capped_report()
    assert report["enabled"] is True and report["limit"] == 3, report
    assert report["streaks"].get("qwen") == 2, (
        f"连续计数没有出现在诊断里：{report['streaks']}——运维无法从诊断判断某节点"
        f"是否正在被反复裁短"
    )

    with TestClient(srv.app) as client:
        body = client.get("/v1/diagnostics").json()
    assert "budget_capped" in body, (
        f"/v1/diagnostics 缺少 budget_capped：{sorted(body)[:12]}…"
    )
    assert body["budget_capped"]["limit"] == tracker.budget_capped_limit()
    print("test_the_mechanism_is_observable_on_diagnostics: PASSED")


def run_all():
    test_a_consistently_hung_node_stops_being_paid_for_on_every_request()
    test_a_single_capped_timeout_still_does_not_convict_the_node()
    test_threshold_zero_restores_the_never_escalate_behaviour()
    test_recovery_clears_the_accumulated_evidence()
    test_a_node_already_cooled_for_this_is_reconvicted_on_the_first_attempt()
    test_a_non_risk_escalation_never_relabels_a_live_risk_cause()
    test_the_saturated_streak_survives_a_restart()
    test_the_mechanism_is_observable_on_diagnostics()
    print("\n✅ All budget-capped evidence tests passed! (8 cases)")


if __name__ == "__main__":
    run_all()