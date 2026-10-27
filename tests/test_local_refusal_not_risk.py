"""网关自己的零上游拒绝，不得被记成平台风控拦截 (OPT-036)

避让线（OPT-021…035）的每一环都建立在同一个前提上：`record_failure` 记下的每一条
`waf`，都是**上游真的拦截过我们一次**。账本据此升级冷却、据此刻画档案、据此告诉
运维「请完成滑块验证」。这条前提在五道**零上游**闸口上不成立——
`credential_outage` / `disabled` / `quarantined` / `ring_exhausted` / `unconfigured`
都是网关在**本地**推出来的结论，一次 `provider.chat()` 都没发生。

判据是错误**文案**，而文案是写给运维看的：它必须说明成因与处置，于是在
`ring_exhausted` 里写下了

    处置: 成因 auth/waf 表示凭证失效或需完成人机验证，请执行 `.venv/bin/python login.py <provider>`

以及节点清单 `qwen(waf，剩余3.0h), glm(waf，…), …`。`is_risk_interception` 的第 1 条
规则是 `"人机" in msg and any(k in msg for k in ("qwen", "千问", "ali", "rgv"))`——
这段**自述**同时满足了两个条件，于是九个 `server.py` 调用点（都是
`is_risk_interception(e.message)`）把它读成一次真实的阿里 WAF 拦截。

实测（`record_failure` 的真实路径，`TESTING=1`，零上游调用）：

    qwen 冷却 3600s(waf)  --一次整环拒绝-->  7200s(waf)，账本新增 1 条 waf 条目

三个后果同时发生，且都是本项目反复修正过的缺陷类型：
  * 冷却翻倍——一个从未被触碰的节点被我们自己的报文推入了更深的避让；
  * 账本被污染——档案的「累计拦截次数 / 最近拦截原因」从此描述一件没发生过的事；
  * 建议被误导——档案按 `waf` 成因让运维去完成一个人机验证，而平台根本没拦过。

修复的判据必须是 `err_type`（抛出点唯一知道的事实），不是文案：文案里写着成因是
**给运维的转述**，用转述反推来源，必然把「我们的结论」读成「上游的答复」。

本套件验证：
1. 整环长效避让的 503 **不写账本、不加冷却、不改成因**（端到端，真实 ASGI 栈）
2. 五道零上游闸口产出的真实错误全部被判定为本地拒绝，且声明集合与真实产出**互等**
3. 反向：同一条文案若来自**上游**（不带本地拒绝标记），必须照旧识别——分类器不得被整体钝化
4. 结构性：九个调用点必须传异常本身，而不是 `.message`

运行: .venv/bin/python tests/test_local_refusal_not_risk.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["TESTING"] = "1"
os.environ["GATEWAY_DISABLE_AUTO_REFRESH"] = "1"
os.environ["GATEWAY_DISABLE_PACING"] = "1"
os.environ["GATEWAY_DISABLE_SINGLEFLIGHT"] = "1"
os.environ["GATEWAY_DISABLE_PROBING"] = "1"
os.environ["GATEWAY_DISABLE_BAN_INFERENCE"] = "1"
os.environ["GATEWAY_PROBATION_MAX_RENEWALS"] = "0"
for _name in (
    "GATEWAY_DISABLE_LONG_AVOIDANCE_FAST_FAIL",
    "GATEWAY_DISABLE_QUARANTINE",
    "GATEWAY_DISABLE_QUARANTINE_HARD_STOP",
    "GATEWAY_DISABLE_RISK_LEDGER",
    "GATEWAY_DISABLE_CREDENTIAL_WATCH",
    "GATEWAY_DISABLE_UNCONFIGURED_GATE",
    "GATEWAY_QUARANTINE",
    "DISABLED_PROVIDERS",
):
    os.environ.pop(_name, None)
for _k in ("QWEN", "DEEPSEEK", "DOUBAO", "KIMI", "GLM"):
    os.environ[f"{_k}_AUTH_TOKEN"] = f"fake_{_k.lower()}"

import server as srv  # noqa: E402
import session_store  # noqa: E402
from providers import failover as fov  # noqa: E402
from providers.base import ProviderError  # noqa: E402
from providers.failover import is_risk_interception  # noqa: E402

FLAGSHIP_RING = ("qwen", "glm", "kimi", "doubao", "deepseek")
ALL_PROVIDERS = FLAGSHIP_RING

TOUCHED: list[str] = []


def _make_fake():
    """假 Provider 工厂：任何真进入 `chat()` 的节点都被记录（= 被触碰）。"""

    def factory(key: str, *a, **kw):
        class _Fake:
            async def chat(self, *a, **kw):
                TOUCHED.append(key)
                yield ("hi", {})

            async def delete_conversation(self, *a, **kw):
                return None

        return _Fake()

    return factory


def _run(coro):
    return asyncio.run(coro)


def _clear_shared_evidence(tracker):
    """清空进程级单例证据，避免用例间串扰（OPT-025 / OPT-026 的共享单例）。"""
    try:
        tracker.credential_watch.clear_all_for_tests()
    except Exception:  # noqa: BLE001
        pass
    try:
        from providers.credential_renewal import get_credential_renewal

        get_credential_renewal().clear_all_for_tests()
    except Exception:  # noqa: BLE001
        pass


def _reset_router():
    router = srv.failover_router
    tracker = router.cooldown_tracker
    tracker.reset()
    srv.quarantine_store.clear_all_for_tests()
    _clear_shared_evidence(tracker)
    TOUCHED.clear()
    return router, tracker


def _restore(tracker, disabled_snapshot):
    tracker.disabled_providers = set(disabled_snapshot)
    tracker.reset()
    srv.quarantine_store.clear_all_for_tests()
    _clear_shared_evidence(tracker)
    TOUCHED.clear()


def _waf(tracker, pk, times=1):
    for _ in range(times):
        _run(tracker.record_failure(pk, 403, "RGV587 人机验证失败，请完成滑块验证"))


def _drain(coro_ctx):
    async def _inner():
        stream, _ev, _p = await coro_ctx
        async for _ in stream:
            pass

    _run(_inner())


# ------------------------------------- 1. 端到端：整环拒绝不得污染风控状态

def test_ring_exhausted_refusal_is_not_recorded_as_a_waf_interception():
    """整环长效避让触发的 503 是**我们**的结论，不是平台的拦截。

    修复前实测（本套件复现）：请求本身零上游调用，`qwen` 的冷却却从 3600s 涨到
    7200s、成因 `waf`，账本新增一条 waf 条目——一个从未被触碰的账号被自己的
    报文推入了更深的避让，而档案从此让运维去完成一次没发生过的验证。
    """
    from fastapi.testclient import TestClient

    router, tracker = _reset_router()
    snap = set(tracker.disabled_providers)
    original = srv._make_provider
    srv._make_provider = _make_fake()
    try:
        for pk in FLAGSHIP_RING:
            _waf(tracker, pk, 3)
        for pk in FLAGSHIP_RING:
            assert tracker.is_in_long_avoidance(pk) is True, f"{pk} 应处于长效避让"

        before = {
            pk: (tracker.get_remaining_cooldown(pk), tracker.cooldown_cause(pk))
            for pk in FLAGSHIP_RING
        }
        ledger_before = tracker.risk_ledger.recent_interceptions(hours=1.0)["total"]

        r = TestClient(srv.app).post(
            "/v1/chat/completions",
            json={"model": "deepseek-chat", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 503, r.text
        err = r.json()["error"]
        assert err["type"] == "ring_exhausted", r.text
        # 正是这两点让文案匹配上「阿里 WAF」规则：处置建议里有人机验证，节点清里有 qwen。
        assert "人机验证" in err["message"] and "qwen" in err["message"], err["message"]

        ledger_after = tracker.risk_ledger.recent_interceptions(hours=1.0)["total"]
        assert ledger_after == ledger_before, (
            f"网关自己的零上游拒绝被写进了风控账本（{ledger_before} -> {ledger_after} 条）——"
            f"档案的累计拦截次数从此描述一件没发生过的事。报文: {err['message'][:160]!r}"
        )

        grown = []
        for pk in FLAGSHIP_RING:
            remaining, cause = tracker.get_remaining_cooldown(pk), tracker.cooldown_cause(pk)
            base, base_cause = before[pk]
            if remaining > base + 0.001 or cause != base_cause:
                grown.append(f"{pk}: {base:.1f}s({base_cause}) -> {remaining:.1f}s({cause})")
        assert not grown, (
            "一次零上游拒绝延长了节点的风控冷却——它把「我们决定不派发」记成了"
            f"「上游拦截了我们」: {grown}"
        )
        assert TOUCHED == [], f"整环拒绝必须零上游触碰，实际: {TOUCHED}"
    finally:
        srv._make_provider = original
        _restore(tracker, snap)
    print("test_ring_exhausted_refusal_is_not_recorded_as_a_waf_interception: PASSED")


# --------------------------- 2. 五道闸口的真实错误：集合互等 + 一律不算拦截

def _capture_refusal(kind: str) -> ProviderError:
    """把某道闸口推到「整环拒绝」，返回它**真正抛出**的那个异常。

    每个分支都读真实闸口，而不是手写一条报文——手写的文案只能证明「我编的那个
    形状能过」，而缺陷活在这些报文真实的措辞里。
    """
    router, tracker = _reset_router()
    snap = set(tracker.disabled_providers)
    original_configured = session_store.is_configured
    try:
        if kind == "credential_outage":
            for pk in ("qwen", "glm", "kimi"):
                tracker.credential_watch.note_auth_failure(pk)
        elif kind == "disabled":
            tracker.disabled_providers = set(ALL_PROVIDERS)
        elif kind == "quarantined":
            for pk in ALL_PROVIDERS:
                srv.quarantine_store.set_lease(pk, duration_seconds=3600)
        elif kind == "ring_exhausted":
            for pk in ALL_PROVIDERS:
                _waf(tracker, pk, 3)
        elif kind == "unconfigured":
            session_store.is_configured = lambda pk: False  # noqa: ARG005
        else:  # pragma: no cover
            raise AssertionError(kind)

        try:
            _drain(
                router.execute_chat(
                    "deepseek", "deepseek-chat", provider_factory=_make_fake(), chat_args={}
                )
            )
        except ProviderError as exc:
            assert TOUCHED == [], f"[{kind}] 闸口必须零上游调用，实际触碰: {TOUCHED}"
            return exc
        raise AssertionError(f"[{kind}] 整环拒绝必须抛 ProviderError，而不是派发")
    finally:
        session_store.is_configured = original_configured
        _restore(tracker, snap)


def test_every_zero_upstream_gate_refusal_is_excluded_from_risk_classification():
    """五道闸口的真实产出与声明集合必须**互等**，且一律不算风控拦截。

    互等是两个方向：声明了却产不出来的类型是死声明（下一个读代码的人会以为它
    有用），产出来却没声明的类型则是**下一次误记**——这正是本套件存在的理由。
    """
    try:
        from providers.base import LOCAL_REFUSAL_ERR_TYPES
    except ImportError:
        raise AssertionError(
            "网关没有声明「本地拒绝」的 err_type 集合——判据只能靠文案反推，"
            "而文案里写着成因与处置建议，必然把自己的结论读成上游的答复"
        ) from None

    captured: dict[str, ProviderError] = {}
    for kind in ("credential_outage", "disabled", "quarantined", "ring_exhausted", "unconfigured"):
        exc = _capture_refusal(kind)
        captured[exc.err_type] = exc
        assert getattr(exc, "is_local_refusal", False) is True, (
            f"[{kind}] 零上游闸口的错误没有被标记为本地拒绝（err_type={exc.err_type!r}）"
        )
        verdict = is_risk_interception(exc)
        assert verdict == (False, ""), (
            f"[{kind}] 网关自己的拒绝被判定为平台风控拦截 {verdict}——"
            f"一次零上游调用被记成了上游的一次拦截。报文: {exc.message[:160]!r}"
        )

    assert set(captured) == set(LOCAL_REFUSAL_ERR_TYPES), (
        "零上游闸口的真实产出与 `LOCAL_REFUSAL_ERR_TYPES` 不互等: "
        f"真实产出={sorted(captured)} 声明={sorted(LOCAL_REFUSAL_ERR_TYPES)}"
    )

    # 第六条：预算不足的拒绝与真实超时共用 `upstream_timeout` 这个 err_type，
    # 只能靠类自己的 `zero_upstream` 标记区分。它同样是零上游的本地结论，必须
    # 一并排除——否则「按类型的白名单」在下一个零上游拒绝出现时又会漏。
    exhausted = fov.BudgetExhaustedRefusal(
        "握手总预算已用尽 (qwen)", status=504, err_type="upstream_timeout"
    )
    assert exhausted.err_type == "upstream_timeout", exhausted.err_type
    assert exhausted.is_local_refusal is True, (
        "预算不足的拒绝没有声明为本地拒绝——它与真实超时共用 err_type，"
        "不靠自己的标记就无法区分"
    )
    assert is_risk_interception(exhausted) == (False, "")

    # 反向控制：同一条文案若来自**上游**，必须照旧被识别。否则「不再误记」就变成了
    # 「什么都不记」——那会让真实的 WAF 拦截整体逃过避让线。
    msg = captured["ring_exhausted"].message
    assert is_risk_interception(msg)[0] is True, (
        "同一条文案当作上游错误出现时也必须被识别——分类器不得被整体钝化: "
        f"{msg[:160]!r}"
    )
    print("test_every_zero_upstream_gate_refusal_is_excluded_from_risk_classification: PASSED")


# ------------------------------------------------------------- 3. 零回归

def test_an_upstream_interception_still_classifies():
    """上游真的拦截时，判定与修复前逐字一致。"""
    upstream = ProviderError(
        "RGV587 人机验证失败 qwen", status=403, err_type="upstream_error"
    )
    is_risk, reason = is_risk_interception(upstream)
    assert is_risk is True and reason, (is_risk, reason)
    assert is_risk_interception("rate_limit_exceeded")[0] is True
    assert is_risk_interception("401 Unauthorized: token expired")[0] is False, (
        "401 走认证冷却，不属于风控拦截——这条断言防的是把判据放宽成「任何错误」"
    )
    print("test_an_upstream_interception_still_classifies: PASSED")


def test_server_passes_the_exception_not_its_message():
    """结构性：九个调用点必须把异常本身交给分类器。

    只在 `is_risk_interception` 里加判断是**一半的修复**：调用点仍然先 `.message`
    取字符串，本地拒绝标记根本没机会被看见。这类「签名改了、调用点没改」的缺陷
    通过全部单测而在真实路径上完全无效（§53 / §75 记录过同一形状）。
    """
    src = (Path(__file__).resolve().parent.parent / "server.py").read_text(encoding="utf-8")
    assert "is_risk_interception(e.message)" not in src, (
        "仍有调用点把异常文案当成判据——本地拒绝标记在那里不可见"
    )
    assert src.count("is_risk_interception(e)") == 9, (
        "风控判定调用点的数量变了（期望 9 个）：新增的调用点必须同样传异常本身，"
        f"实际 {src.count('is_risk_interception(e)')} 个"
    )
    print("test_server_passes_the_exception_not_its_message: PASSED")


def run_all():
    test_server_passes_the_exception_not_its_message()
    test_an_upstream_interception_still_classifies()
    test_every_zero_upstream_gate_refusal_is_excluded_from_risk_classification()
    test_ring_exhausted_refusal_is_not_recorded_as_a_waf_interception()
    print("\n✅ All local-refusal risk tests passed! (4 cases)")


if __name__ == "__main__":
    sys.exit(run_all())
