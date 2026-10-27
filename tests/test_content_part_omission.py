"""未被识别的内联内容块在 OpenAI 两条协议上**静默消失**，而 Anthropic 侧早已公告 (PROTOCOL)

## 实测：同一份内容，三个协议两个答案

真实 ASGI 路径（脚本化 provider，`TESTING=1`，零真实上游调用）。`prompt` 是
`flatten_messages` 真正送给上游的那段文本：

    chat  audio only      200  prompt=''
    chat  text+audio      200  prompt='transcribe this'
    chat  unknown type    200  prompt=''
    chat  refusal part    200  prompt='User: go\\n\\nUser: why\\n\\nAssistant:'
    chat  image ctrl      200  prompt="[Image omitted: this gateway's upstream ...]"
    chat  nested file     200  prompt='[File omitted: unnamed]'
    resp  audio only      200  prompt=''
    resp  text+audio      200  prompt='transcribe this'
    resp  unknown type    200  prompt=''
    anth  audio block     200  prompt='transcribe this\\n[Content omitted: audio — …]'
    anth  unknown ctrl    200  prompt='[Content omitted: container_upload — …]'

四种可测的伤害，全部没有错误码：

1. **`input_audio` 是 OpenAI 两条协议都真实存在的内联内容类型**（Chat Completions
   与 Responses 都是 `{"type":"input_audio","input_audio":{…}}`）。用户发
   `[text "transcribe this", audio]`，模型收到的是 `'transcribe this'` —— 一段
   要求它转写音频的指令，而音频**一个字都没到**。这正是 §41 记录的那条判据：
   「显式占位符是『我无法查看』与幻觉之间的区别」。
2. **助手的 `refusal` 内容块整轮消失**。`[{"type":"refusal","refusal":"I cannot
   help with that."}]` 实测让提示词变成 `User: go\\n\\nUser: why` —— 两个连续的
   user 轮。模型看到自己从未拒绝过，用户看到的是它推翻了一个不存在的立场。
3. **任何未知类型都退化成空字符串**，连「有东西没渲染」都不说。§71 已经为
   Responses 的**条目**层与 Anthropic 的**块**层确立过规则：「沉默比占位符更糟，
   因为看不到内容的模型会断定内容不存在，然后据此作答」。只有 OpenAI 的**内容块**
   这一层没跟上。
4. **文件占位符丢了文件名**：Chat Completions 把名字嵌在 `file` 子字典里
   （`{"type":"file","file":{"filename":"a.py"}}`），取名只读顶层，于是一个客户端
   明明命名过的附件被说成 `unnamed`。占位符的全部理由就是「告诉模型该重发哪一个
   文件」，名字丢了这句话就说不出去了。

根因不是三个缺失的 `elif`，而是**同一个事实有两套默认值**：`extract_anthropic_content`
的兜底分支公告，`base.text_of` 与 `responses_compat.extract_responses_content`
的兜底是「丢弃」。所以修复把措辞收进 `providers/base.py` 的
`CONTENT_OMITTED_TEMPLATE` / `AUDIO_PLACEHOLDER`，三个提取器共用——与 §68 把围栏
语言表抽成 `_TOOL_FENCE_LANG` 是同一条纪律：文案只有一份，调用点不可能各自漂移。
结构性用例断言那个字面量在全仓库只出现一次。

## 零回归

已知类型的渲染逐字节不变（用例 6）：图片占位符、纯文本、以及
`{"type":"text","text":""}` 仍然不产生任何文本。最容易写错的修法是把「兜底公告」
放到已知分支**前面**——那样两种本来正确的渲染会被一条 `[Content omitted: …]`
取代。Anthropic 侧本来就正确，它在本套件里是**对照**，不是修复对象。
用例 7 把「兜底公告」做成结构性断言，而不是再枚举几个类型——行为用例只能覆盖
我想到的形状。

用例 6（已知渲染零回归）在修复前的树上**也是绿的**，所以它在自己的 docstring 里
标明是守卫，不冒充 RED-first 的缺陷复现（§59/§69/§79 的纪律）。其余 6 条各自带着
自己实测到的症状变红。

运行: .venv/bin/python tests/test_content_part_omission.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["TESTING"] = "1"
for _k in (
    "GATEWAY_DISABLE_CREDENTIAL_WATCH",
    "GATEWAY_DISABLE_AUTO_REFRESH",
    "GATEWAY_DISABLE_PROMPT_FOLDING",
    "GATEWAY_DISABLE_CONVERSATION_GC",
    "GATEWAY_DISABLE_ADAPTIVE_PACING",
    "GATEWAY_DISABLE_PROBING",
    "GATEWAY_DISABLE_BAN_INFERENCE",
    "GATEWAY_DISABLE_STREAM_COMPLETION_CHECK",
    "GATEWAY_DISABLE_LONG_AVOIDANCE_FAST_FAIL",
    "GATEWAY_DISABLE_RISK_LEDGER",
    "GATEWAY_DISABLE_SINGLEFLIGHT",
):
    os.environ[_k] = "1"
for _k in ("QWEN", "DEEPSEEK", "DOUBAO", "KIMI", "GLM"):
    os.environ[f"{_k}_AUTH_TOKEN"] = "fake"

import httpx  # noqa: E402

import server as srv  # noqa: E402
from providers.base import flatten_messages  # noqa: E402

try:  # 修复前的树上这两个名字还不存在：用期望值兜底，好让失败信息是**实测行为**
    from providers.base import AUDIO_PLACEHOLDER, CONTENT_OMITTED_TEMPLATE
except ImportError:  # pragma: no cover - 仅在基线树上走到
    AUDIO_PLACEHOLDER = (
        "[Audio omitted: this gateway's upstream providers accept text only]"
    )
    CONTENT_OMITTED_TEMPLATE = "[Content omitted: {kind} — this gateway cannot render it]"

_GUARD_HITS: list[str] = []


def _guarded_factory(*args, **kwargs):
    _GUARD_HITS.append(str(args[0]) if args else "?")
    raise AssertionError("测试发出了未安装假 provider 的上游请求")


srv._make_provider = _guarded_factory

PROMPTS: list[str] = []


class _Provider:
    async def chat(self, messages, model, **kw):
        # 断言的对象是**上游真正收到的提示词**（§41/§43 的纪律），所以在这里
        # 就把网关消息展平成上游会看到的那段文本。
        PROMPTS.append(flatten_messages(list(messages)))
        yield "answer", {}
        yield "", {"conversation_id": "part-conv"}

    async def delete_conversation(self, *args, **kwargs):
        return True


AUDIO = {"type": "input_audio", "input_audio": {"data": "QUJD", "format": "wav"}}


async def _post(path: str, body: dict) -> tuple[int, str]:
    """驱动真实端点，返回 (状态码, 上游收到的提示词)。"""
    PROMPTS.clear()
    srv._make_provider = lambda *a, **kw: _Provider()
    try:
        transport = httpx.ASGITransport(app=srv.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            r = await c.post(path, json=body)
        return r.status_code, (PROMPTS[0] if PROMPTS else "")
    finally:
        srv._make_provider = _guarded_factory


# ------------------- 1. Chat Completions：音频必须被公告

def test_chat_audio_is_announced_not_silently_dropped():
    """实测修复前：`[text "transcribe this", audio]` 的上游提示词是 `'transcribe this'`。

    模型收到的是「转写这段音频」的指令，而音频一个字节都没到——它会照做，
    或者编造一段转写结果。§41 的判据在这里逐字适用。
    """
    status, prompt = asyncio.run(_post("/v1/chat/completions", {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "transcribe this"}, AUDIO]}],
    }))
    assert status == 200, f"状态码 {status}"
    assert "transcribe this" in prompt, f"文字部分丢了: {prompt!r}"
    assert AUDIO_PLACEHOLDER in prompt, (
        f"音频没有留下任何痕迹，模型会对着它从未收到的音频作答（实测修复前 "
        f"prompt={prompt!r}）"
    )
    print("test_chat_audio_is_announced_not_silently_dropped: PASSED")


# ------------------- 2. Responses：同一条规则

def test_responses_audio_is_announced_not_silently_dropped():
    """实测修复前：Responses 侧同样丢成 `'transcribe this'`。"""
    status, prompt = asyncio.run(_post("/v1/responses", {
        "model": "gpt-4o",
        "input": [{"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": "transcribe this"}, AUDIO]}],
    }))
    assert status == 200, f"状态码 {status}"
    assert "transcribe this" in prompt, f"文字部分丢了: {prompt!r}"
    assert AUDIO_PLACEHOLDER in prompt, (
        f"Responses 侧的音频被静默丢弃（实测修复前 prompt={prompt!r}）"
    )
    print("test_responses_audio_is_announced_not_silently_dropped: PASSED")


# ------------------- 3. 未知类型必须带类型公告，而不是消失

def test_unknown_part_types_are_announced_on_both_openai_protocols():
    """实测修复前：`{"type":"video", …}` 在两条协议上都得到 `prompt=''`。

    「沉默」是三种可能里最糟的一种：模型看不到内容，会断定内容不存在。
    """
    for path, body in (
        ("/v1/chat/completions", {"model": "gpt-4o", "messages": [
            {"role": "user", "content": [{"type": "video", "video": {"url": "u"}}]}]}),
        ("/v1/responses", {"model": "gpt-4o", "input": [
            {"type": "message", "role": "user",
             "content": [{"type": "video", "video": {"url": "u"}}]}]}),
    ):
        status, prompt = asyncio.run(_post(path, body))
        assert status == 200, f"{path} 状态码 {status}"
        want = CONTENT_OMITTED_TEMPLATE.format(kind="video")
        assert want in prompt, (
            f"{path} 上没有类型的公告：未知块被静默丢弃（实测修复前 prompt={prompt!r}）"
        )
    print("test_unknown_part_types_are_announced_on_both_openai_protocols: PASSED")


# ------------------- 4. 助手的 refusal 块是它自己的一轮发言

def test_assistant_refusal_part_is_replayed_as_assistant_turn():
    """实测修复前：`[{"type":"refusal","refusal":"I cannot help with that."}]`
    让提示词变成 `User: go\\n\\nUser: why` —— 助手的拒绝整轮消失，历史读起来像
    用户对着空气自我追问。
    """
    status, prompt = asyncio.run(_post("/v1/chat/completions", {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": [
                {"type": "refusal", "refusal": "I cannot help with that."}]},
            {"role": "user", "content": "why"},
        ],
    }))
    assert status == 200, f"状态码 {status}"
    assert "I cannot help with that." in prompt, (
        f"助手的拒绝内容块消失了——模型不知道自己拒绝过（实测修复前 "
        f"prompt={prompt!r}）"
    )
    assert "Assistant:" in prompt, f"整轮助手发言都没了: {prompt!r}"
    print("test_assistant_refusal_part_is_replayed_as_assistant_turn: PASSED")


# ------------------- 5. 已知渲染逐字节不变（守卫）

def test_known_part_rendering_is_value_for_value_unchanged():
    """**这条用例在修复前的树上也是绿的**——它是守卫，不是缺陷复现。

    最容易写错的修法是把「兜底公告」放到已知分支**前面**，那样图片/文件/纯文本
    都会被一条 `[Content omitted: image_url …]` 取代，等于用一次修复换掉两种
    已经正确的渲染。空文本块也必须继续不产生任何字（它此前不会，因为
    `isinstance(text, str) and text` 这一层判空）。
    """
    chat, prompt = asyncio.run(_post("/v1/chat/completions", {
        "model": "gpt-4o", "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "http://x/a.png"}},
            {"type": "text", "text": "done"},
            {"type": "text", "text": ""},
        ]}]}))
    assert chat == 200
    assert prompt == (
        "[Image omitted: this gateway's upstream providers accept text only]\ndone"
    ), f"已知类型（图片/文本）的渲染被改写了: {prompt!r}"

    _resp, rprompt = asyncio.run(_post("/v1/responses", {
        "model": "gpt-4o", "input": [{"type": "message", "role": "user", "content": [
            {"type": "input_image", "image_url": "http://x/a.png"},
            {"type": "input_text", "text": "done"},
        ]}]}))
    assert rprompt == (
        "[Image omitted: this gateway's upstream providers accept text only]\ndone"
    ), f"Responses 侧已知类型的渲染被改写了: {rprompt!r}"
    print("test_known_part_rendering_is_value_for_value_unchanged: PASSED")


# ------------------- 6. 文件占位符必须带上客户端真的写了的文件名

def test_file_placeholder_carries_the_filename_clients_actually_send():
    """实测修复前：Chat Completions 的 `file` 部分渲染成 `[File omitted: unnamed]`。

    OpenAI 的 Chat Completions 把文件名嵌在 `file` 子字典里
    （`{"type":"file","file":{"filename":"a.py"}}`），而取名只读了顶层，
    于是客户端**明明命名过**的附件被说成 unnamed。占位符存在的全部理由是
    「告诉模型该重发哪一个文件」——丢掉名字，这句话就说不出去了（§41）。
    Responses 的 `input_file` 是顶层形状，本来就对，所以它在这里是**对照**。
    """
    for path, body in (
        ("/v1/chat/completions", {"model": "gpt-4o", "messages": [
            {"role": "user", "content": [
                {"type": "file", "file": {"filename": "a.py"}}]}]}),
        ("/v1/responses", {"model": "gpt-4o", "input": [
            {"type": "message", "role": "user", "content": [
                {"type": "input_file", "filename": "a.py"}]}]}),
    ):
        status, prompt = asyncio.run(_post(path, body))
        assert status == 200, f"{path} 状态码 {status}"
        assert "[File omitted: a.py]" in prompt, (
            f"{path} 上文件名丢了（实测修复前 Chat 侧是 '[File omitted: unnamed]'）: "
            f"{prompt!r}"
        )
    print("test_file_placeholder_carries_the_filename_clients_actually_send: PASSED")


# ------------------- 6. 措辞只能有一份定义（结构性）

def test_the_omission_wording_has_exactly_one_definition():
    """§71 的规则要落到**每一个**兜底分支上，靠枚举类型是做不到的。

    行为用例只能覆盖我想得到的块类型；「文案全仓库只有一份」才是让下一个新增
    提取器无法退回静默丢弃的那个属性（§68 的 `_TOOL_FENCE_LANG` 同理）。
    """
    root = Path(__file__).resolve().parent.parent
    hits = []
    for p in root.glob("**/*.py"):
        if any(part in {".venv", "tests", "__pycache__"} for part in p.parts):
            continue
        if "[Content omitted:" in p.read_text(encoding="utf-8"):
            hits.append(p.relative_to(root).as_posix())
    assert hits == ["providers/base.py"], (
        f"公告措辞应当只在 providers/base.py 定义一次，实测出现在 {hits}"
    )
    print("test_the_omission_wording_has_exactly_one_definition: PASSED")


def run_all():
    test_chat_audio_is_announced_not_silently_dropped()
    test_responses_audio_is_announced_not_silently_dropped()
    test_unknown_part_types_are_announced_on_both_openai_protocols()
    test_assistant_refusal_part_is_replayed_as_assistant_turn()
    test_file_placeholder_carries_the_filename_clients_actually_send()
    test_known_part_rendering_is_value_for_value_unchanged()
    test_the_omission_wording_has_exactly_one_definition()
    assert not _GUARD_HITS, f"有请求绕过了脚本化 provider: {_GUARD_HITS}"
    print("\n✅ All content-part omission tests passed! (7 cases)")


if __name__ == "__main__":
    sys.exit(run_all())
