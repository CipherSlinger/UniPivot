"""被截断的回答不得冒充完整回答 (S2 / STREAM-AUDIT)

## 缺陷

上游流在**没有完成标记**的情况下结束（TCP 正常关闭、帧迭代器直接停了），
网关把它与「正常说完」完全同等对待。实测确认的三件事：

  (a) `record_success` 是在**首包**记的（`failover.py` 首包读取处），耗尽路径
      上不再复核，所以节点失败计数保持 0 —— 一个每几次请求就吐到一半的节点
      永远是主选，客户端反复看到「回复说到一半停住」；
  (b) 五家的 `chat()` 结尾都**无条件**产出末位元数据帧，所以截断与说完在
      生成器层面长得一模一样；
  (c) 协议处理器随后无条件补上 `finish_reason: "stop"` 与 `data: [DONE]`。
      客户端收到的是一个**自称完整**的半截回答。

(c) 是对 agent 最致命的一条：半截的 ```tool_call 围栏解析不出任何东西，于是
这一轮既没有答案也没有调用、而且没有任何错误可看 —— 「工具调用没跑起来」
的一个与工具层毫无关系的成因。

## 可检测的范围必须诚实

调查五家协议的结论（与最初的猜测不同，故在此记录）：

  * qwen 读 `data.status == "complete"`；
  * kimi 读 `event == "all_done"`；
  * **deepseek / doubao / glm 根本不存在被读取的完成标记** —— doubao 的
    `is_finish` 只出现在**出站请求体**里，glm 的 `finish` 一次都没出现。

因此只有 qwen / kimi 两家**能**判定截断。对另外三家编一个启发式判据
（「文本不以标点结尾」之类）就是凭空制造证据，正是本仓库反复在删的东西。

判据因此是**三态**而不是布尔：

  * `True`  —— 协议有标记且看到了  → 正常结束；
  * `False` —— 协议有标记且**没**看到 → 截断，按中途断流处置；
  * 缺键    —— 协议没有标记（未知）  → **行为零变化**。

「缺键 = 未知 ≠ 截断」是本套件最重要的一条断言：它是另外三家继续正常工作的
唯一保证，也是 fail-open 惯例（参见 `resolve_allowed_tools` 对无法识别的
`tool_choice` 的处理）。

## 判定成立后的动作

截断就是「中途断流」，只是从**唯一不抛异常的那道门**进来的。所以它复用已有
且已正确的那条路径，不另起一套：节点记冷却（否则 (a) 不变），并上抛，使协议
处理器发带内错误且**不**发 `[DONE]`。不重试、不转移 —— 内容已经到客户端了，
重试只会造成半包污染。

运行: .venv/bin/python tests/test_stream_completion.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["TESTING"] = "1"
os.environ["GATEWAY_DISABLE_RISK_LEDGER"] = "1"
os.environ["GATEWAY_DISABLE_AUTO_REFRESH"] = "1"
os.environ["GATEWAY_DISABLE_PROBING"] = "1"
os.environ["GATEWAY_DISABLE_CREDENTIAL_WATCH"] = "1"
os.environ["GATEWAY_DISABLE_CREDENTIAL_RENEWAL"] = "1"
os.environ["GATEWAY_DISABLE_PROBATION"] = "1"
os.environ["GATEWAY_DISABLE_PACING"] = "1"
os.environ["GATEWAY_DISABLE_SINGLEFLIGHT"] = "1"
os.environ["GATEWAY_DISABLE_DEVICE_IDENTITY"] = "1"
os.environ["GATEWAY_STREAM_KEEPALIVE"] = "0"

from providers.base import ProviderError  # noqa: E402
from providers.failover import FailoverRouter  # noqa: E402

ROUTER = FailoverRouter()

PROVIDERS_ROOT = Path(__file__).resolve().parent.parent / "providers"


# ------------------------------------------------------------------ 工具


class _ScriptedProvider:
    """按脚本产出帧、随后**干净耗尽**（不抛异常）的上游。"""

    def __init__(self, frames: list):
        self.frames = frames

    async def chat(self, *a, **kw):
        for delta, meta in self.frames:
            yield delta, meta

    async def delete_conversation(self, *a, **kw):
        return None


def _fresh() -> FailoverRouter:
    router = FailoverRouter()
    t = router.cooldown_tracker
    t.reset()
    t._cooldowns.clear()
    t._failure_counts.clear()
    t._cooldown_severity.clear()
    t.disabled_providers.clear()
    return router


async def _drain(router, frames: list, provider_key: str = "qwen"):
    """跑完整条流；返回 (收到的帧, 捕获的异常或 None)。"""
    provider = _ScriptedProvider(frames)
    got: list = []
    err = None
    try:
        stream, _ev, _p = await router.execute_chat(
            provider_key, "Qwen", lambda k: provider, {"model": "Qwen"}
        )
        async for delta, meta in stream:
            got.append((delta, meta))
    except Exception as e:  # noqa: BLE001 - 本套件就是要看它抛不抛
        err = e
    return got, err


# ------------------------------------------------------------------ 用例


def test_truncated_stream_is_not_a_success():
    """协议有标记、却始终没看到 → 必须判为失败，而不是正常说完。"""
    router = _fresh()
    frames = [
        ("Here is the ans", {}),
        ("", {"conversation_id": "c1", "upstream_complete": False}),
    ]
    got, err = asyncio.run(_drain(router, frames))

    assert err is not None, (
        "上游在没有完成标记的情况下结束，被当成了正常说完 —— "
        "客户端会收到一个自称完整的半截回答"
    )
    assert isinstance(err, ProviderError), f"应为 ProviderError，实为 {type(err)}"
    assert getattr(err, "err_type", "") == "upstream_incomplete", (
        f"截断必须有自己的 err_type（便于与超时/空流区分），实为 {getattr(err, 'err_type', None)!r}"
    )
    # 已送出的内容不回收（无法回收），但必须以错误收尾
    assert any(d for d, _ in got), "截断前已产出的内容应照常送达"

    # 节点必须被记账：否则它继续当主选，症状反复出现
    assert router.cooldown_tracker.get_remaining_cooldown("qwen") > 0, (
        "截断的节点必须进入冷却，否则失败计数永远是 0、它永远是主选"
    )
    print("test_truncated_stream_is_not_a_success: PASSED")


def test_complete_stream_is_untouched():
    """看到完成标记 → 正常结束，零副作用。"""
    router = _fresh()
    frames = [
        ("All good.", {}),
        ("", {"conversation_id": "c1", "upstream_complete": True}),
    ]
    got, err = asyncio.run(_drain(router, frames))
    assert err is None, f"完整的流不得报错: {err}"
    assert router.cooldown_tracker.get_remaining_cooldown("qwen") == 0, "完整的流不得产生冷却"
    assert len(got) == 2, got
    print("test_complete_stream_is_untouched: PASSED")


def test_absent_marker_is_unknown_not_incomplete():
    """**最重要的一条**：协议没有完成标记时缺键 = 未知，行为零变化。

    deepseek / doubao / glm 三家协议里根本没有可读的完成标记。把「缺键」当成
    「截断」会让这三家每一条正常回答都被判失败并记冷却 —— 一个凭空制造的
    全局故障。三态判据的存在就是为了这条。
    """
    router = _fresh()
    frames = [
        ("Fine answer.", {}),
        ("", {"conversation_id": "c1"}),  # 无 upstream_complete 键
    ]
    got, err = asyncio.run(_drain(router, frames, provider_key="deepseek"))
    assert err is None, (
        f"无完成标记的协议（deepseek/doubao/glm）必须按未知处理、行为零变化，实际抛了 {err}"
    )
    assert router.cooldown_tracker.get_remaining_cooldown("deepseek") == 0, (
        "未知完整性不得产生冷却 —— 否则三家没有标记的上游会被全量误判"
    )
    print("test_absent_marker_is_unknown_not_incomplete: PASSED")


def test_disable_switch_restores_old_behaviour():
    """`GATEWAY_DISABLE_STREAM_COMPLETION_CHECK=1` 必须逐值还原旧行为。"""
    os.environ["GATEWAY_DISABLE_STREAM_COMPLETION_CHECK"] = "1"
    try:
        router = _fresh()
        frames = [
            ("half", {}),
            ("", {"conversation_id": "c1", "upstream_complete": False}),
        ]
        _got, err = asyncio.run(_drain(router, frames))
        assert err is None, f"关闭开关后必须与旧行为逐值一致（不报错），实际 {err}"
        assert router.cooldown_tracker.get_remaining_cooldown("qwen") == 0
    finally:
        os.environ.pop("GATEWAY_DISABLE_STREAM_COMPLETION_CHECK", None)
    print("test_disable_switch_restores_old_behaviour: PASSED")


def test_only_providers_with_a_real_marker_report_completeness():
    """上报完整性的节点集合必须**恰好**是协议里真有标记的那两家。

    两个方向都要锁：
      * qwen / kimi 必须上报 —— 否则这个修复在真实部署里是空转（单元用例
        用的是脚本化上游，照样全绿，而真实上游永远不带这个键）；
      * deepseek / doubao / glm 必须**不**上报 —— 它们协议里没有标记，凭空
        加一个只能靠编造判据，那是在用一个假证据换一个真失败。
    """
    # 判据是「是否使用了 COMPLETION_KEY 这个机制」，而不是字面量 —— provider
    # 侧用的是常量，写死字面量会在常量改名时给出一个假绿。
    reporting = []
    for key in ("qwen", "deepseek", "doubao", "kimi", "glm"):
        src = (PROVIDERS_ROOT / key / "provider.py").read_text(encoding="utf-8")
        if "COMPLETION_KEY" in src:
            reporting.append(key)

    assert set(reporting) == {"qwen", "kimi"}, (
        f"上报完整性的节点应恰为 qwen/kimi（协议里真有完成标记的两家），实为 {reporting}。"
        f"新增一家前请先确认它的协议**真的**有完成标记，而不是编一个启发式。"
    )

    # 标记来源也要钉住：标记被读取的那一行若被删掉，上报就成了常量 True
    qwen_src = (PROVIDERS_ROOT / "qwen" / "provider.py").read_text(encoding="utf-8")
    assert '"complete"' in qwen_src, "qwen 的完成标记判据（status == 'complete'）不见了"
    kimi_src = (PROVIDERS_ROOT / "kimi" / "provider.py").read_text(encoding="utf-8")
    assert '"all_done"' in kimi_src, "kimi 的完成标记判据（all_done）不见了"
    print("test_only_providers_with_a_real_marker_report_completeness: PASSED")


def test_truncation_does_not_enter_the_risk_ledger():
    """截断不是风控拦截：不得污染风控判据，也不得被记两遍。

    与 [M2] 对超时的处置同理 —— 把一次上游截断写进风控台账，会让档案告诉
    运维「去重新登陆」，而平台从未拦过他们。
    """
    router = _fresh()
    frames = [
        ("partial", {}),
        ("", {"conversation_id": "c1", "upstream_complete": False}),
    ]
    _got, err = asyncio.run(_drain(router, frames))
    assert err is not None
    cause = router.cooldown_tracker.cooldown_cause("qwen")
    assert cause not in ("waf", "auth"), (
        f"截断被归因为 {cause!r} —— 这会让档案建议运维去完成一个从未出现的人机验证"
    )
    print("test_truncation_does_not_enter_the_risk_ledger: PASSED")


def run_all():
    test_truncated_stream_is_not_a_success()
    test_complete_stream_is_untouched()
    test_absent_marker_is_unknown_not_incomplete()
    test_disable_switch_restores_old_behaviour()
    test_only_providers_with_a_real_marker_report_completeness()
    test_truncation_does_not_enter_the_risk_ledger()
    print("\n✅ All stream-completion tests (S2) passed!")


if __name__ == "__main__":
    run_all()
