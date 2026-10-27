"""推理强度是「要思考」的声明，不是四个白名单词 (AGENT-LOOP)

`/v1/responses` 判定 thinking 意图时，对 `reasoning.effort` 只认一个字面量集合：

    if isinstance(effort, str) and effort.lower() in ("high", "medium", "low", "minimal"):
        enable_thinking = True

而 Codex CLI 的**真实**目录（`codex debug models`，0.145.0）声明的档位是

    gpt-5.6-sol   : low / medium / high / xhigh / max / ultra
    gpt-5.6-terra : low / medium / high / xhigh / max / ultra
    gpt-5.6-luna  : low / medium / high / xhigh / max
    gpt-5.5       : low / medium / high / xhigh
    gpt-5.4       : low / medium / high / xhigh
    codex-auto-review: low / medium / high / xhigh

于是**越想要模型想清楚，越会把思考整个关掉**：`-c model_reasoning_effort=xhigh`
（以及 max / ultra）落在白名单外，`enable_thinking` 保持 False，请求照常 200、
日志什么都没有，模型以非推理模式作答。这条路径的模型名兜底（§34「不认识的
`gpt-*` 按前缀选档」）明确写了「固定表永远追不上厂商发布」，同一个理由在这里
同样成立——`xhigh` / `max` / `ultra` 就是这样长出来的。

判据应当是**语义**而不是词表：`none` / `off` / `disabled` 明确关闭，其余任何
非空的强度声明一律开启（失败开放）。厂商再加一档时，网关只会「多想一点」，
而不会「静默不想」——后者是 §34、§76 反复修正过的同一类缺陷：声明的能力被
丢掉，且没有任何可观察的错误。

同一字段在 `/v1/chat/completions` 上更彻底：请求模型根本没有声明
`reasoning_effort`，它落在 `extra="allow"` 里，没有任何代码读过（§35 的
「解析了然后没人用」）。本套件把两条路径的判定锁成同一个函数，并用**属性**断言
（对每个取值，两个协议给出相同结论），因为「一处改了另一处没改」正是本项目
最常见的缺陷形状。

运行: .venv/bin/python tests/test_reasoning_effort.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["TESTING"] = "1"
os.environ["GATEWAY_DISABLE_CREDENTIAL_WATCH"] = "1"
os.environ["GATEWAY_DISABLE_AUTO_REFRESH"] = "1"
os.environ["GATEWAY_DISABLE_PROMPT_FOLDING"] = "1"
os.environ["GATEWAY_DISABLE_CONVERSATION_GC"] = "1"
os.environ["GATEWAY_DISABLE_SINGLEFLIGHT"] = "1"
os.environ["GATEWAY_DISABLE_ADAPTIVE_PACING"] = "1"
os.environ["GATEWAY_DISABLE_PROBING"] = "1"
os.environ["GATEWAY_DISABLE_BAN_INFERENCE"] = "1"
for _k in ("QWEN", "DEEPSEEK", "DOUBAO", "KIMI", "GLM"):
    os.environ[f"{_k}_AUTH_TOKEN"] = f"fake_{_k.lower()}"

from fastapi.testclient import TestClient  # noqa: E402

import server as srv  # noqa: E402

#: `codex debug models` 在当前二进制里真实声明的全部档位（0.145.0）。
CATALOG_EFFORTS = ("low", "medium", "high", "xhigh", "max", "ultra")
#: 旧版 SDK 的 `reasoning_effort` 还会发这个（gpt-5.1 起）。
LEGACY_EFFORTS = ("minimal",)
#: 明确表示「不要思考」的取值。
OFF_EFFORTS = ("none", "off", "disabled")


class ScriptedProvider:
    """记录上游**实际收到**的调用参数（`thinking` 就是 qwen 的 `chat_mode` 依据）。"""

    last_kw: dict = {}

    async def chat(self, messages, model, **kw):
        ScriptedProvider.last_kw = dict(kw)
        yield "ok", {}
        yield "", {"conversation_id": "eff-conv"}

    async def delete_conversation(self, *args, **kwargs):
        return True


_GUARD_HITS: list[str] = []


def _guarded_factory(*args, **kwargs):
    _GUARD_HITS.append(str(args[0]) if args else "?")
    raise AssertionError(
        "测试发出了未安装假 provider 的上游请求（所有请求必须经 _run(fn)）"
    )


srv._make_provider = _guarded_factory


def _run(fn):
    ScriptedProvider.last_kw = {}
    srv._make_provider = lambda *a, **kw: ScriptedProvider()
    try:
        return fn()
    finally:
        srv._make_provider = _guarded_factory


def _client():
    return TestClient(srv.app)


def _responses_thinking(body_extra: dict) -> bool:
    """走真实的 `/v1/responses` 路径，返回 provider 实际收到的 `thinking`。"""

    body = {"model": "gpt-5.5", "input": "hi"}
    body.update(body_extra)
    result = _run(lambda: _client().post("/v1/responses", json=body))
    assert result.status_code == 200, result.text
    return bool(ScriptedProvider.last_kw.get("thinking"))


def _chat_thinking(body_extra: dict) -> bool:
    """走真实的 `/v1/chat/completions` 路径，返回 provider 实际收到的 `thinking`。"""

    body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
    body.update(body_extra)
    result = _run(lambda: _client().post("/v1/chat/completions", json=body))
    assert result.status_code == 200, result.text
    return bool(ScriptedProvider.last_kw.get("thinking"))


# ------------------------------------------------ 1. 目录里的每一档都必须开启思考

def test_every_catalog_reasoning_effort_enables_thinking():
    """`xhigh` / `max` / `ultra` 是**更深**的推理档，不是「不认识的值」。

    修复前实测（真实 `/v1/responses` 路径）：`low` / `medium` / `high` 为 True，
    `xhigh` / `max` / `ultra` 为 **False**——用户把强度调到最高，思考反而消失。
    """
    missing = []
    for effort in CATALOG_EFFORTS + LEGACY_EFFORTS:
        got = _responses_thinking({"reasoning": {"effort": effort}})
        if not got:
            missing.append(effort)
    assert not missing, (
        "Codex 目录里声明的推理档位没有开启思考，模型被静默降级为非推理模式: "
        f"{missing}（实测修复前 xhigh/max/ultra 全部为 False）"
    )
    # 旧版 SDK 的扁平写法同样必须生效（§34b 已经记录过这条形状）。
    assert _responses_thinking({"reasoning_effort": "xhigh"}) is True, (
        "`reasoning_effort:'xhigh'`（扁平写法）没有开启思考"
    )
    print("test_every_catalog_reasoning_effort_enables_thinking: PASSED")


# ------------------------------------------- 2. 明确关闭的取值不得被「失败开放」误开

def test_explicit_off_efforts_do_not_enable_thinking():
    """失败开放只对**强度**成立；`none` / `off` / `disabled` 是明确的关闭。"""
    for effort in OFF_EFFORTS:
        assert _responses_thinking({"reasoning": {"effort": effort}}) is False, (
            f"`effort={effort!r}` 明确要求关闭思考，网关却开启了"
        )
    # 空值与非字符串形状：没有强度声明，不得凭空开启。
    for shape in ({"effort": ""}, {"effort": 7}, {"effort": None}, {}):
        assert _responses_thinking({"reasoning": shape}) is False, (
            f"`reasoning={shape!r}` 没有表达任何思考意图，网关却开启了思考"
        )
    # 客户端显式写了 thinking=disabled 时，它是更具体的声明，强度不得翻案。
    assert (
        _responses_thinking(
            {"thinking": {"type": "disabled"}, "reasoning": {"effort": "high"}}
        )
        is False
    ), "显式的 `thinking.type=disabled` 被 `reasoning.effort` 翻成了开启"
    print("test_explicit_off_efforts_do_not_enable_thinking: PASSED")


# ------------------------- 3. 两个协议的判定必须一致（属性断言，不是逐例枚举）

def test_both_protocols_agree_on_every_effort_value():
    """同一份声明在两个协议上必须得到同一个结论。

    逐例断言只能覆盖我想到的取值；「一处改了另一处没改」是本项目反复出现的
    缺陷形状（§53 / §75 / §79），所以这里对**每个**取值比较两个协议的结果。
    修复前 `/v1/chat/completions` 根本不读这个字段：即便 `reasoning_effort` 是
    `high`，provider 收到的 `thinking` 也是 False。
    """
    verdicts = {}
    for effort in CATALOG_EFFORTS + LEGACY_EFFORTS + OFF_EFFORTS:
        via_responses = _responses_thinking({"reasoning": {"effort": effort}})
        via_chat = _chat_thinking({"reasoning_effort": effort})
        verdicts[effort] = (via_responses, via_chat)
        assert via_responses == via_chat, (
            f"`effort={effort!r}` 在两个协议上判定不一致: "
            f"/v1/responses={via_responses} /v1/chat/completions={via_chat}"
        )
    # 反向控制：没有声明时两条路径都必须保持关闭（零回归）。
    assert _responses_thinking({}) is False
    assert _chat_thinking({}) is False
    print("test_both_protocols_agree_on_every_effort_value: PASSED")


# ------------------------------------------------- 4. 不认识的强度也必须失败开放

def test_an_unknown_effort_fails_open():
    """厂商再加一档时，网关只能「多想」，不能「静默不想」。

    `xhigh` / `max` / `ultra` 就是这么长出来的：一个固定白名单在厂商发布新档位
    的那一刻开始静默失效，而且没有任何可观察的错误（§34 对模型名兜底写过同一句）。
    """
    for effort in ("turbo", "extra-ultra", "HIGHEST"):
        assert _responses_thinking({"reasoning": {"effort": effort}}) is True, (
            f"无法识别的强度 `{effort!r}` 让思考被静默关闭——"
            "固定白名单在厂商发布新档位时会永久失效"
        )
    print("test_an_unknown_effort_fails_open: PASSED")


# --------------------------- 5. 验收仪器必须能把这一档送上真实上游并看见结果

def _load_harness():
    """把 e2e 运行器当模块导入（`__main__` 守卫，不需要真实二进制）。"""
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts" / "e2e" / "run_agent_loop.py"
    spec = importlib.util.spec_from_file_location("_e2e_runner_effort", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)          # type: ignore[union-attr]
    return mod


def test_the_harness_can_put_any_effort_on_the_wire():
    """仪器必须**能**看见这个缺陷，否则「全绿」只说明它没在看。

    `run_agent_loop.py` 把 `model_reasoning_effort = "low"` 写死了——而 `low` 恰好
    是旧白名单里的四个值之一。也就是说：这个缺陷从上线那天起就不可能被任何一次
    真实二进制验收发现，因为唯一被演练的取值是还能工作的那个。所以验收必须
    （a）按 `UNIPIVOT_E2E_CODEX_EFFORT` 写入配置，（b）在上游调用记录里带上
    `thinking` 的真实取值，（c）在「声明了强度、但上游一次都没开启思考」时判定失败。
    """
    mod = _load_harness()
    for knob in ("_codex_effort", "_effort_failure"):
        assert hasattr(mod, knob), f"验收运行器缺少 `{knob}`——强度无法被测量"

    # (c) 判定本身：三个方向都要成立，否则一个只会绿的检查等于没有检查。
    fail = mod._effort_failure(
        {"effort": "xhigh", "thinking": False}, "codex"
    )
    assert fail and "xhigh" in fail, (
        "声明了 xhigh、上游却收到 thinking=False 的运行被判成了通过——"
        "这正是本缺陷在真实二进制上唯一的可观察形态"
    )
    assert mod._effort_failure({"effort": "xhigh", "thinking": True}, "codex") is None
    assert mod._effort_failure({"effort": None, "thinking": False}, "claude") is None, (
        "对根本没有声明强度的客户端套用 Codex 的判据，等于给同一件事造了第二个定义"
    )
    assert mod._effort_failure({"effort": "none", "thinking": False}, "codex") is None

    # (a) 配置真的写进去了（否则 (b)(c) 只是在检查一个不存在的场景）。
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        env = {**os.environ, "UNIPIVOT_E2E_CODEX_EFFORT": "xhigh"}
        code = (
            "import importlib.util,sys\n"
            f"spec=importlib.util.spec_from_file_location('m', {str(Path(mod.__file__))!r})\n"
            "m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
            f"argv,env=m._codex_argv(1, __import__('pathlib').Path({str(tmpdir)!r}), "
            f"__import__('pathlib').Path({str(tmpdir)!r}), __import__('pathlib').Path({str(tmpdir / 'm')!r}))\n"
            "print('SKIP' if not argv else 'OK')\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code], env=env, capture_output=True, text=True
        )
        if "OK" in proc.stdout:
            cfg = (tmpdir / "codexhome" / "config.toml").read_text(encoding="utf-8")
            assert 'model_reasoning_effort = "xhigh"' in cfg, (
                "验收工具没有把请求的强度写进 Codex 配置："
                f"{cfg.splitlines()[:4]}"
            )
        else:                                  # pragma: no cover - 本机无 codex
            print("  (codex 不在 PATH，仅校验判据)")

    print("test_the_harness_can_put_any_effort_on_the_wire: PASSED")


def run_all():
    test_every_catalog_reasoning_effort_enables_thinking()
    test_explicit_off_efforts_do_not_enable_thinking()
    test_both_protocols_agree_on_every_effort_value()
    test_an_unknown_effort_fails_open()
    test_the_harness_can_put_any_effort_on_the_wire()
    assert not _GUARD_HITS, f"有请求绕过了脚本化 provider: {_GUARD_HITS}"
    print("\n✅ All reasoning-effort tests passed! (5 cases)")


if __name__ == "__main__":
    sys.exit(run_all())
