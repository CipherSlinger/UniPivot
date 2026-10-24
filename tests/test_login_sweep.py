"""`login.py all` 必须尊重「重登无效」的节点声明。

[LOGIN-SWEEP] `login.py all` 是网关自己的拒绝信息让运维执行的命令——
`providers/credential_watch.py:230` 与 `providers/failover.py:2766` 都写着
「请执行 `.venv/bin/python login.py all` 重新登录全部厂商」。
它也是 OPT-026/027/029 人工解除避让的唯一入口。

但它**无条件下探五个节点**：`main()` 把 "all" 展开成固定的五个名字，逐个调用
`login_<provider>`，全程不问一句「这个节点现在该不该碰」。

这与网关其余每一个上游触点都不同。`CooldownTracker.touch_avoidance_reason()`
是后台触点（探针 / 令牌刷新 / 会话回收 / 自愈选举）的**唯一**判据（OPT-034），
它明确区分三种成因：禁用的解除方式是改配置，租约的解除方式是人工撤租，
冷却的解除方式是等待——把三者混成一句话是 OPT-026/027/029/031 反复修复的
「误导运维」缺陷。`login.py` 一处都没有问过。

实测（修复前）：持有 `deepseek` 运维租约 + `DISABLED_PROVIDERS=kimi` 时，
`login.py all` 的触点顺序是 `['qwen', 'deepseek', 'doubao', 'kimi', 'glm']`——
**五个全部触碰**，包括持有封禁租约的 deepseek 与被声明出局的 kimi。

而 OPT-023 档案对 `ban` 的结论是逐字写着的：
「**无法绕开，重新登录亦无效**——封禁作用在账号本身而非令牌……任何尝试只会延长封禁，
唯一正确策略是识别、退避并在窗口内**零误触**」。

于是网关陷入自相矛盾：它告诉运维「执行 login.py all」，而这条命令会去碰一个
它自己判定为「任何尝试只会延长封禁」的账号。这正是 OPT-035 那一类缺陷
（文档/建议与代码实际行为不一致），只是发生在**建议**这一侧。

判据必须区分成因，不能一刀切「冷却就不登」：
- `quarantined` / `disabled` -> 跳过（重登对它们**无效**，触碰只有害处）；
- `cooldown` -> **照常登**。冷却恰恰是重登要修复的东西：OPT-029 的 waf 长效
  避让解除路径就是「人工完成滑块 -> 凭证文件变化 -> 网关解除避让」。把冷却也
  跳过，等于把唯一的修复入口关掉，正是 OPT-026 已经修过的自锁缺陷。
- 显式点名（`login.py deepseek`）永远执行：那是人在覆盖判断。
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["TESTING"] = "1"
os.environ["GATEWAY_DISABLE_AUTO_REFRESH"] = "1"

_LEDGER_DIR = tempfile.mkdtemp(prefix="login_sweep_ledger_")
os.environ["GATEWAY_RISK_LEDGER_DIR"] = _LEDGER_DIR

_ALL = ("qwen", "deepseek", "doubao", "kimi", "glm")


def _run_main(argv: list[str], touched: list[str]) -> tuple[int, str]:
    """执行 `login.main()`，五个 login_* 被替换为只记录触点的桩。

    桩必须替换在 `login` 模块的属性上：`main()` 在函数体内构造
    `LOGIN_HANDLERS` 字典并按名字取函数，因此替换模块属性即可生效，
    且不需要真的启动浏览器。
    """
    import contextlib
    import io

    import login

    def _stub(name):
        def _f(headless: bool = False, timeout: int = 300, **kw):
            touched.append(name)
            return object()  # 非 None => main() 视为成功

        return _f

    originals = {}
    for name in _ALL:
        attr = f"login_{name}"
        originals[attr] = getattr(login, attr)
        setattr(login, attr, _stub(name))

    old_argv = sys.argv
    sys.argv = ["login.py", *argv]
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            rc = login.main()
    finally:
        sys.argv = old_argv
        for attr, fn in originals.items():
            setattr(login, attr, fn)
    return rc, buf.getvalue()


def _reset_avoidance() -> None:
    from providers.quarantine import get_quarantine_store

    get_quarantine_store().clear()


def test_all_skips_a_node_under_a_ban_lease():
    """持有运维封禁租约的节点不得被 `login.py all` 触碰。

    实测（修复前）：TOUCHED 包含 'deepseek'，即使它持有 72h 租约——
    而档案对 ban 的结论是「任何尝试只会延长封禁」。
    """
    from providers.quarantine import get_quarantine_store

    _reset_avoidance()
    get_quarantine_store().set_lease("deepseek", 72 * 3600)

    touched: list[str] = []
    rc, out = _run_main(["all"], touched)

    assert "deepseek" not in touched, (
        f"`login.py all` 触碰了持有封禁租约的 deepseek（TOUCHED={touched}）——"
        f"档案对 ban 的结论是「重新登录亦无效，任何尝试只会延长封禁」，"
        f"而网关的拒绝信息恰恰让运维执行这条命令"
    )
    # 其余四个不受影响：跳过必须是**针对性**的，不是把整条命令废掉
    for name in ("qwen", "doubao", "kimi", "glm"):
        assert name in touched, f"{name} 无任何声明，不应被跳过（TOUCHED={touched}）"
    assert "deepseek" in out, "被跳过的节点必须在输出里说明原因，不能静默跳过"
    print("test_all_skips_a_node_under_a_ban_lease: PASSED")


def test_all_skips_a_disabled_node_and_says_why():
    """`DISABLED_PROVIDERS` 声明的节点同样不得被触碰。"""
    from providers.quarantine import get_quarantine_store

    _reset_avoidance()
    old = os.environ.get("DISABLED_PROVIDERS")
    os.environ["DISABLED_PROVIDERS"] = "kimi"
    try:
        touched: list[str] = []
        _, out = _run_main(["all"], touched)
    finally:
        if old is None:
            os.environ.pop("DISABLED_PROVIDERS", None)
        else:
            os.environ["DISABLED_PROVIDERS"] = old
    _reset_avoidance()

    assert "kimi" not in touched, (
        f"`login.py all` 触碰了被 DISABLED_PROVIDERS 声明出局的 kimi（TOUCHED={touched}）"
    )
    assert "kimi" in out, "跳过原因必须可见"
    print("test_all_skips_a_disabled_node_and_says_why: PASSED")


def test_all_still_logs_in_a_cooling_node():
    """冷却中的节点**必须照常登录**——冷却正是重登要修复的东西。

    这是上一条的对照。一刀切「有避让就跳过」会把 OPT-029 的唯一解除路径关掉：
    人机验证的长效避让只有「人工完成滑块 -> 凭证变化 -> 网关解除」这一条出路。
    """
    from providers.failover import CooldownTracker
    from providers.quarantine import get_quarantine_store

    _reset_avoidance()
    tracker = CooldownTracker()
    asyncio.run(tracker.record_failure("glm", 429, "too many requests"))

    reason = tracker.touch_avoidance_reason("glm")
    assert reason == "cooldown", f"前置条件不成立：glm 应为 cooldown，实际 {reason!r}"

    touched: list[str] = []
    _run_main(["all"], touched)
    assert "glm" in touched, (
        f"冷却中的 glm 被跳过了（TOUCHED={touched}）——重登正是解除冷却的入口，"
        f"跳过它等于关掉 OPT-029 唯一的人工修复路径"
    )
    print("test_all_still_logs_in_a_cooling_node: PASSED")


def test_explicit_provider_still_runs_despite_a_lease():
    """显式点名永远执行：那是人在覆盖网关的判断。"""
    from providers.quarantine import get_quarantine_store

    _reset_avoidance()
    get_quarantine_store().set_lease("deepseek", 72 * 3600)

    touched: list[str] = []
    _run_main(["deepseek"], touched)
    _reset_avoidance()

    assert touched == ["deepseek"], (
        f"显式 `login.py deepseek` 被跳过了（TOUCHED={touched}）——"
        f"运维点名即覆盖，网关不得代替人做这个决定"
    )
    print("test_explicit_provider_still_runs_despite_a_lease: PASSED")


def run_all():
    test_all_skips_a_node_under_a_ban_lease()
    test_all_skips_a_disabled_node_and_says_why()
    test_all_still_logs_in_a_cooling_node()
    test_explicit_provider_still_runs_despite_a_lease()
    print("\n✅ All login sweep avoidance tests passed!")


if __name__ == "__main__":
    run_all()