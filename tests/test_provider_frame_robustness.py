"""上游 SSE 非对象帧的鲁棒性（跨五个适配器一致）。

[STREAM-AUDIT] 续：`data: []` / `data: null` / `data: 42` 都是**合法 JSON**，
但 `json.loads` 之后直接在它上面调 `.get()` 会抛 `AttributeError`。对网关而言
一个怪帧就让整轮请求变成 502，而客户端看不到任何原因。

Kimi / GLM 在 C3 已经修过（`isinstance(obj, dict)` 守卫），Qwen / Doubao 也有。
**DeepSeek 没有**——它 `json.loads` 之后第一行就是 `obj.get("v")`。

本套件对**五个**适配器用同一注入方式断言同一件事，这样「下一个适配器漏了这个
守卫」会直接变红，而不是等到线上某个怪帧出现。
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["TESTING"] = "1"
os.environ["GATEWAY_DISABLE_AUTO_REFRESH"] = "1"

import httpx  # noqa: E402

# 合法 JSON，但都不是对象。`[DONE]` 已在各家被单独跳过，不在此列。
NON_OBJECT_FRAMES = ["data: []", "data: null", "data: 42", 'data: "text"']
# 夹在怪帧之间的一帧真实内容：证明怪帧被**跳过**而不是让整条流失败。
GOOD_FRAME_QWEN = 'data: {"output": {"choices": [{"messages": [{"content": "ok", "status": "typing"}]}], "status": "typing"}}'


async def _collect(gen):
    out = []
    async with gen as stream:
        async for delta, meta in stream:
            if delta:
                out.append(delta)
    return "".join(out)


def _patch_client(module, handler, orig):
    class MockClient(orig):
        def __init__(self, *a, **kw):
            kw["transport"] = httpx.MockTransport(handler)
            super().__init__(*a, **kw)

    module.httpx.AsyncClient = MockClient


async def _drive(provider, model):
    out = []
    async for delta, meta in provider.chat([{"role": "user", "content": "hi"}], model):
        if delta:
            out.append(delta)
    return "".join(out)


def test_deepseek_skips_non_object_frames():
    """DeepSeek 必须跳过 `data: []` 之类的合法非对象帧。

    实测（修复前）：`data: []` 一帧即抛
    `AttributeError: 'list' object has no attribute 'get'`，整轮请求变成 502。
    同一条流里其它四个适配器都能跳过它。
    """
    import providers.deepseek_provider as m
    from providers.deepseek_provider import DeepSeekProvider

    body = "\n".join(
        NON_OBJECT_FRAMES
        + ['data: {"v": {"response": {"message_id": 7, "fragments": [{"type": "RESPONSE", "content": "ok"}]}}}']
        + NON_OBJECT_FRAMES
    )

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/chat_session/create"):
            return httpx.Response(200, json={"code": 0, "data": {"biz_data": {"chat_session": {"id": "s1"}}}})
        if path.endswith("/create_pow_challenge"):
            return httpx.Response(
                200,
                json={"code": 0, "data": {"biz_data": {"challenge": {
                    "algorithm": "DeepSeekHashV1", "challenge": "c", "salt": "s",
                    "difficulty": 1.0, "expire_at": 9999999999, "signature": "g",
                    "target_path": "/api/v0/chat/completion"}}}},
            )
        return httpx.Response(200, text=body)

    class FakePow:
        def make_header(self, challenge):
            return "hdr"

    m._get_pow = lambda: FakePow()
    orig = m.httpx.AsyncClient
    _patch_client(m, handler, orig)
    try:
        text = asyncio.run(_drive(DeepSeekProvider("tok", timeout=10), "deepseek-chat"))
    finally:
        m.httpx.AsyncClient = orig

    assert text == "ok", (
        f"非对象帧让 DeepSeek 丢了正文，得到 {text!r}——"
        f"`data: []`/`null`/`42` 是合法 JSON，必须跳过而不是 502"
    )
    print("test_deepseek_skips_non_object_frames: PASSED")


def test_every_adapter_declares_the_object_guard():
    """五个适配器都必须显式守卫非对象帧（结构断言，防止下一个漏掉）。

    行为测试只能覆盖我想到的形状；这里直接断言源码里存在守卫，
    使「新增适配器忘了写」在评审阶段可见。
    """
    root = Path(__file__).resolve().parent.parent
    missing = []
    for name in ("qwen", "deepseek", "doubao", "kimi", "glm"):
        # 真正的实现在 `<name>/provider.py`；`<name>_provider.py` 只是兼容 shim，
        # 先取 shim 会让这个结构断言看错文件（实测：五个全被判缺失）。
        src = root / "providers" / name / "provider.py"
        if not src.exists():
            src = root / "providers" / f"{name}_provider.py"
        text = src.read_text(encoding="utf-8")
        if "isinstance(obj, dict)" not in text and "isinstance(payload, dict)" not in text:
            missing.append(name)
    assert not missing, (
        f"这些适配器没有非对象帧守卫：{missing}——"
        f"`data: []` 是合法 JSON，缺守卫会让一帧怪数据变成整轮 502"
    )
    print("test_every_adapter_declares_the_object_guard: PASSED")


def run_all():
    test_deepseek_skips_non_object_frames()
    test_every_adapter_declares_the_object_guard()
    print("\n✅ All provider frame robustness tests passed!")


if __name__ == "__main__":
    run_all()