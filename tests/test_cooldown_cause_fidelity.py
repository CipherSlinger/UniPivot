"""普通 5xx 不是人机验证 (COOLDOWN-CAUSE)

## 缺陷：网关最常见的失败，被记成了一次「人机验证」

`record_failure` 先用 `classify_interception_severity` 给本次失败定级，而那个分类器
对**任何非 401/429 的输入**缺省回 `waf`：

    def classify_interception_severity(...):
        if status == 401 or ...: return SEVERITY_AUTH
        if status == 429 or ...: return SEVERITY_RATE_LIMIT
        return SEVERITY_WAF          # ← 其余一切，包括一次普通的 502

分支链在每一个风控/认证/超时/截断/未配置分支里都显式写了 `severity`，**唯独最后那个
「普通 5xx」分支没写**，于是它继承了分类器的缺省值：`waf`。

实测（真实 `CooldownTracker`，凭证就位，`TESTING=1`，零真实上游调用）：

    [对等互备网络] 节点 [qwen] 失败 (HTTP 502)，进入冷却避让期 15.0s
    cooldown: 15.0 | cause: waf
    posture: cooling | cooldown_cause: waf
    relogin: False | auto_recovers: True

三条下游消费者同时被误导：

  * **档案的处置建议**：`cooldown_cause` 报 `waf`，而 `waf` 在 `AVOIDANCE_MATRIX` 里
    的结论是「只有人工完成滑块验证能解除」。运维会去登录一个从未拦截过他的平台、
    甚至去找验证码——与 [M2]（超时被记成 waf）、[S2]（截断被记成 waf）、
    [DAY-ZERO]（未配置被记成 auth）是同一类错误，换了个成因。
  * **成因限定的解除路径**：`release_auth_avoidance_locked` 按成因决定能否解除。
    被标成 `waf` 的窗口需要「凭证指纹变化」才能解除，而它真正需要的是「等 15 秒」。
  * **长效避让判据**：成因参与 `is_in_long_avoidance()` / `long_avoidance_cause()`
    的措辞与阈值判断（[OPT-031]）。
  * 另外，`elif severity == SEVERITY_WAF: self.renewal.note_interception(...)` 这一
    支**没有**经过「这真的是一次拦截吗」的统一判据：只要 `severity` 落在 waf（普通
    502 恰好就是），就会为它写一条「人类已介入」的凭证基线。台账与观察期都用了
    `not is_upstream_fault and (is_risk or is_waf_or_rate or is_auth_error)` 这个闸门，
    唯独这条基线写入漏了——同一判据的第四份抄写。

## 正确语义

「普通 5xx」是**上游故障**，与超时、截断同类：冷却该记（节点确实没给出首句），
但它不是风控拦截，因此：

  * 成因记为 `upstream_error`（非风控成因，见 [BUDGET-6]：非风控成因永远不能抹掉
    仍在生效的风控成因）；
  * 不写「人类已介入」的凭证基线；
  * 档案的结论指向上游可用性 / 本地出网质量，并明确「重新登录无效」；
  * 真正的 WAF（403 + RGV587）逐值不变：成因 `waf`、写基线、进台账。

运行: .venv/bin/python tests/test_cooldown_cause_fidelity.py
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

# 干净的起点：这些开关会关掉本套件**正在观测**的证据链（台账 / 凭证续期），
# 因此必须逐条 pop，而不是照抄其它套件的 prelude。
for _key in [k for k in os.environ if k.startswith("GATEWAY_DISABLE_")]:
    os.environ.pop(_key, None)
os.environ.pop("GATEWAY_QUARANTINE", None)
os.environ.pop("DISABLED_PROVIDERS", None)
os.environ["TESTING"] = "1"
os.environ["GATEWAY_DISABLE_AUTO_REFRESH"] = "1"
os.environ["GATEWAY_DISABLE_PROBING"] = "1"
os.environ["GATEWAY_DISABLE_PACING"] = "1"
os.environ["GATEWAY_DISABLE_SINGLEFLIGHT"] = "1"
# 假凭证：`is_configured` 为真，才能走到「上游故障」分支而不是 [DAY-ZERO] 未配置分支。
os.environ["QWEN_AUTH_TOKEN"] = "fake_qwen"
os.environ["DEEPSEEK_AUTH_TOKEN"] = "fake_ds"
os.environ["DOUBAO_AUTH_TOKEN"] = "fake_doubao"
os.environ["KIMI_AUTH_TOKEN"] = "fake_kimi"
os.environ["GLM_AUTH_TOKEN"] = "fake_glm"

import providers.failover as fov  # noqa: E402
import providers.credential_renewal as credmod  # noqa: E402
from providers.credential_renewal import get_credential_renewal  # noqa: E402
from providers.risk_dossier import RiskDossier  # noqa: E402
from providers.risk_ledger import get_risk_ledger  # noqa: E402

# 本套件**绝不**碰真实的 session/：凭证指纹改由临时文件提供。
_REAL_CREDENTIAL_PATH = credmod.credential_path
_TMP_DIR = Path(tempfile.mkdtemp(prefix="gateway_cause_fidelity_"))
_CRED_FILE = _TMP_DIR / "qwen.json"


def _install_credential_file(content: dict | None = None) -> None:
    """让**每个** provider 都有一枚可读的凭证文件（全部指向临时文件）。

    这是本套件第二、第四个用例成立的前提：`note_interception` 在指纹为 None 时
    直接返回 False，所以若没有这枚文件，写基线的那条缺陷分支**根本不会被执行**，
    「没有基线」的断言会在修复前也成立——一条恒真的断言，测不到任何东西。
    （第一版只给 qwen 装了文件，第四个用例于是在 glm 上直接短路，暴露的正是这个
    前提：基线用例必须先把文件摆好。）
    """
    _CRED_FILE.write_text(
        json.dumps(content if content is not None else {"token": "tok-1", "captured_at": 1}),
        encoding="utf-8",
    )
    credmod.credential_path = lambda pk: _CRED_FILE


def _restore_credential_path() -> None:
    credmod.credential_path = _REAL_CREDENTIAL_PATH


def _fresh_tracker() -> "fov.CooldownTracker":
    """一个与磁盘状态完全隔离的 tracker（台账 / 续期记录都先清空）。"""
    get_risk_ledger().clear()
    get_credential_renewal().clear_all_for_tests()
    tr = fov.CooldownTracker()
    tr.reset()
    return tr


def _record_failure(tr: "fov.CooldownTracker", provider_key: str, status: int, msg: str) -> float:
    return asyncio.run(tr.record_failure(provider_key, status, msg))


# ---------------------------------------------------------------- 1. 成因


def test_a_plain_upstream_5xx_is_named_an_upstream_fault():
    """**靶点**：普通 5xx 的冷却成因必须是 `upstream_error`，不是 `waf`。"""
    tr = _fresh_tracker()
    cd = _record_failure(tr, "qwen", 502, "通义千问上游请求失败: 502")

    cause = tr.cooldown_cause("qwen")
    assert abs(cd - 15.0) < 1e-6, (
        f"普通 5xx 的冷却时长应仍是 min(base=45, 15*n)=15s（零回归），实测 {cd}"
    )
    assert cause == "upstream_error", (
        f"一次普通 502 被记成了风控成因 {cause!r}。分类器对任何非 401/429 的输入"
        f"缺省回 waf，而「普通 5xx」分支没有显式写成因，于是继承了这个缺省值——"
        f"档案会据此建议运维去完成一个从未发生的人机验证"
    )


# ---------------------------------------------------------------- 2. 凭证基线


def test_a_plain_5xx_does_not_claim_a_human_verification_happened():
    """**靶点**：普通 5xx 不得写「人类已介入」的凭证基线。

    这条基线是整个 OPT-029 解除链的判据：指纹变化 = 有人完成了滑块。为一次
    上游 502 写基线，等于把「网关自己遇到的一次网络故障」登记成「人工干预证据」。
    """
    _install_credential_file()
    try:
        tr = _fresh_tracker()
        _record_failure(tr, "qwen", 502, "通义千问上游请求失败: 502")
        has_baseline = tr.renewal.has_record("qwen")
    finally:
        _restore_credential_path()

    assert not has_baseline, (
        "一次普通 502 写入了人机验证的凭证基线。台账与观察期都经过"
        "「这真的是一次拦截吗」的统一闸门，唯独这条基线写入漏了同一个判据"
    )


# ---------------------------------------------------------------- 3. 档案结论


def test_the_dossier_does_not_send_the_operator_to_a_captcha():
    """**靶点**：档案不得把一次上游 502 报成需要人工处理的风控。"""
    tr = _fresh_tracker()
    _record_failure(tr, "qwen", 502, "通义千问上游请求失败: 502")

    dossier = RiskDossier(
        ledger=tr.risk_ledger, quarantine=tr.quarantine, cooldown_tracker=tr
    )
    entry = dossier.provider_entry("qwen")

    assert entry["cooldown_cause"] == "upstream_error", (
        f"档案读到的成因是 {entry['cooldown_cause']!r}，应与 tracker 的判读一致"
    )
    verdict = entry["avoidance"]["verdict"]
    assert verdict == "upstream_error", (
        f"档案的处置结论是 {verdict!r}。一次上游 502 不是风控拦截：结论必须指向"
        f"上游可用性 / 本地出网质量，而不是让人去登录一个从未拦截他的平台"
    )
    assert entry["avoidance"]["recoverable_by_relogin"] is False, (
        "上游 502 重新登录无效，档案却说可重登恢复"
    )
    # 判据放在**机器可读的字段**上，而不是搜文案里的关键词：第一版这里写的是
    # `"验证码" not in strategy`，结果被修复后**正确**的那句话自己触发了
    # （「……也不应去完成任何验证码」）——与 §62 记录过的「按子串判定会命中
    # 恰恰正确的措辞」是同一个坑。文案只断言那句必须存在的结论。
    assert "重新登录无效" in entry["avoidance"]["strategy"], (
        f"档案没有明确告诉运维「重新登录无效」，运维会去登录一个没有拦截他的平台: "
        f"{entry['avoidance']['strategy'][:160]}"
    )


# ---------------------------------------------------------------- 4. 零回归


def test_a_genuine_waf_interception_is_unchanged():
    """零回归**守卫**：真正的人机验证仍然记 `waf`、写基线、进台账。

    这条在修复前就是绿的，且必须如此——它是那个「把范围收窄到只改普通 5xx」的
    守卫，不是 RED-first 的靶点（§59/§79 的诚实要求）。它存在的理由是本修复的
    失败方向：把成因改成 `upstream_error` 时若顺手动了判断条件，真正的人机验证
    会**一起**被降级——那会让 OPT-029 的长效避让失去唯一的解除判据（人工完成
    滑块后网关认不出来），比原缺陷更糟。
    """
    _install_credential_file()
    try:
        tr = _fresh_tracker()
        _record_failure(
            tr, "glm", 403, "智谱触发人机验证（RGV587），请运行 login.py glm 完成滑块"
        )
        cause = tr.cooldown_cause("glm")
        has_baseline = tr.renewal.has_record("glm")
        ledger_total = dossier_total = RiskDossier(
            ledger=tr.risk_ledger, quarantine=tr.quarantine, cooldown_tracker=tr
        ).provider_entry("glm")["interceptions"]["total"]
    finally:
        _restore_credential_path()

    assert cause == "waf", f"真正的人机验证成因发生了回归: {cause!r}"
    assert has_baseline, (
        "真正的人机验证必须留下凭证基线——没有它，人工完成滑块后网关就无法识别"
        "「人类已介入」，长效避让将成为没有出口的永久冻结（OPT-029）"
    )
    assert ledger_total == 1, (
        f"真正的人机验证必须进风控台账，实测 {ledger_total} 条记录"
    )
    assert dossier_total == 1

    # 429 频控同理：仍是 ratelimit，且不是 waf。
    tr2 = _fresh_tracker()
    _record_failure(tr2, "doubao", 429, "豆包上游限流 rate_limit 频控")
    assert tr2.cooldown_cause("doubao") == "ratelimit", (
        f"429 频控成因发生了回归: {tr2.cooldown_cause('doubao')!r}"
    )


# ---------------------------------------------------------------- 5. [BUDGET-6] 方向


def test_a_non_risk_booking_never_relabels_a_live_risk_cause():
    """普通 5xx 是**我们观测到的上游故障**，不得抹掉仍在生效的风控成因。

    [BUDGET-6] 已规定：非风控成因只有在自己真的延长了窗口时才���覆盖成因记录。
    这条用例从「另一个方向」验证同一个根因——修复前普通 5xx 的成因是 `waf`
    （风控成因），于是它无条件覆盖掉一个仍在生效的 `ratelimit` 窗口。
    """
    tr = _fresh_tracker()
    # 先种一段仍在生效的频控冷却，远长于普通 5xx 的 15s
    tr._cooldowns["qwen"] = time.time() + 600.0
    tr._cooldown_severity["qwen"] = "ratelimit"

    cd = _record_failure(tr, "qwen", 502, "通义千问上游请求失败: 502")

    assert tr.cooldown_cause("qwen") == "ratelimit", (
        f"一次普通 5xx（{cd:.1f}s）把仍在生效的频控成因改成了 "
        f"{tr.cooldown_cause('qwen')!r}。频控与认证/人机验证的解除路径完全不同，"
        f"成因被改写会让档案与解除链按错误的成因行动"
    )


def run_all():
    test_a_plain_upstream_5xx_is_named_an_upstream_fault()
    test_a_plain_5xx_does_not_claim_a_human_verification_happened()
    test_the_dossier_does_not_send_the_operator_to_a_captcha()
    test_a_genuine_waf_interception_is_unchanged()
    test_a_non_risk_booking_never_relabels_a_live_risk_cause()
    print("\n✅ All cooldown-cause fidelity tests passed! (5 cases)")


if __name__ == "__main__":
    run_all()
