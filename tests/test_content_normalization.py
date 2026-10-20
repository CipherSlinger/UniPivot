"""多模态内容与空字段的入站归一化测试 (tests/test_content_normalization.py)。

[PROTOCOL] 三种客户端**合法**发送、而网关处理错误的 content 形状：

1. `{"type": "image_url"}` / `{"type": "input_image"}` / Anthropic 的 `image`：
   上游五家都是纯文本网页接口，图片**送不到**——但过去它们被静默丢弃。客户端发
   `[text, image]`，模型只看到 text，然后对着一个它从未看到的附件正常作答。用户
   得到的是一个自信的幻觉，而且没有任何报错能提示问题出在网关。
2. `{"type": "text", "text": null}`：Pydantic 允许、客户端常见（可选字段被显式写
   成 null），`"\\n".join` 直接抛 `TypeError`，整个请求 500。
3. OpenAI `system` 的 content 写成 parts 列表：`server.py` 用 f-string 拼接，得到
   的是 Python repr（`[{'type': 'text', ...}]`），模型在工具契约前面读到一段既不是
   指令也不是它自己语言的噪声。

本套件锁住：附件**以显式占位符**送达模型（模型知道自己缺什么，可以说「我看不到
图片」而不是编造），null 字段**不崩溃且不写进提示词**，system parts 以纯文本合并。

运行: .venv/bin/python tests/test_content_normalization.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["TESTING"] = "1"
os.environ["GATEWAY_DISABLE_AUTO_REFRESH"] = "1"
os.environ["GATEWAY_DISABLE_CREDENTIAL_WATCH"] = "1"
os.environ["GATEWAY_DISABLE_PROBING"] = "1"
os.environ["GATEWAY_DISABLE_PACING"] = "1"
os.environ["QWEN_AUTH_TOKEN"] = "fake_qwen"
os.environ["DEEPSEEK_AUTH_TOKEN"] = "fake_ds"
os.environ["DOUBAO_AUTH_TOKEN"] = "fake_doubao"
os.environ["KIMI_AUTH_TOKEN"] = "fake_kimi"
os.environ["GLM_AUTH_TOKEN"] = "fake_glm"

from fastapi.testclient import TestClient  # noqa: E402

import server as srv  # noqa: E402
from providers.anthropic_compat import extract_anthropic_content  # noqa: E402
from providers.base import IMAGE_PLACEHOLDER, flatten_messages, text_of  # noqa: E402
from providers.responses_compat import extract_responses_content  # noqa: E402

PNG = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="


class CapturingProvider:
    """记录上游真正收到的 messages 的脚本化 provider。"""

    def __init__(self):
        self.messages = None

    async def chat(self, messages, model, conversation_id=None, **kw):
        self.messages = messages
        yield "ok", {}
        yield "", {"conversation_id": None}


def _upstream_messages(payload: dict) -> list:
    provider = CapturingProvider()
    srv._make_provider = lambda key, client=None: provider
    client = TestClient(srv.app)
    r = client.post("/v1/chat/completions", json=payload)
    assert r.status_code == 200, r.text
    assert provider.messages is not None, "请求没有到达上游"
    return provider.messages


def test_openai_image_is_announced_not_silently_dropped():
    content = [
        {"type": "text", "text": "这张图里是什么？"},
        {"type": "image_url", "image_url": {"url": PNG}},
    ]
    text = text_of(content)
    assert "这张图里是什么？" in text
    assert IMAGE_PLACEHOLDER in text, (
        f"图片被静默丢弃：模型只会看到文字，然后编造图片内容。实际: {text!r}"
    )

    # 端到端：占位符必须真的到达上游提示词，而不只是本地函数正确
    msgs = _upstream_messages({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": content}],
    })
    assert IMAGE_PLACEHOLDER in flatten_messages(msgs), (
        "图片没有送达上游提示词"
    )
    print("test_openai_image_is_announced_not_silently_dropped: PASSED")


def test_openai_file_is_announced_with_its_name():
    text = text_of([
        {"type": "text", "text": "读一下"},
        {"type": "input_file", "file_id": "file-1", "filename": "main.py"},
    ])
    assert "main.py" in text, f"文件名丢了，模型不知道缺的是哪个文件: {text!r}"
    assert "读一下" in text
    print("test_openai_file_is_announced_with_its_name: PASSED")


def test_null_text_part_does_not_crash_and_is_not_echoed():
    # 过去：TypeError: sequence item 0: expected str instance, NoneType found → 500
    assert text_of([{"type": "text", "text": None}, {"type": "text", "text": "hi"}]) == "hi"
    assert text_of([{"type": "text", "text": None}]) == ""
    # 端到端：一个把可选字段写成 null 的客户端不该收到 500
    msgs = _upstream_messages({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": None},
            {"type": "text", "text": "你好"},
        ]}],
    })
    prompt = flatten_messages(msgs)
    assert "你好" in prompt
    assert "None" not in prompt, f"null 被写成了字符串 'None': {prompt!r}"
    print("test_null_text_part_does_not_crash_and_is_not_echoed: PASSED")


def test_system_parts_list_is_merged_as_text():
    msgs = _upstream_messages({
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "回答要简洁"}]},
            {"role": "user", "content": "hi"},
        ],
        "tools": [{"type": "function", "function": {
            "name": "bash", "parameters": {"type": "object", "properties": {}}}}],
    })
    system = msgs[0]
    assert system["role"] == "system"
    assert isinstance(system["content"], str), (
        f"system content 仍是 {type(system['content']).__name__}，下游会字符串化成 repr"
    )
    assert "回答要简洁" in system["content"]
    assert "{'type'" not in system["content"], (
        f"system content 里出现了 Python repr（客户端发言被当成噪声送入上游）: {system['content'][:200]!r}"
    )
    # 工具契约必须仍在同一条 system 消息里（合并不得丢掉新增内容）
    assert "## Tools Available" in system["content"]
    print("test_system_parts_list_is_merged_as_text: PASSED")


def test_anthropic_image_and_document_are_announced():
    text = extract_anthropic_content([
        {"type": "text", "text": "看图"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAA"}},
        {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "AAA"}},
    ])
    assert "看图" in text
    assert IMAGE_PLACEHOLDER in text
    assert "application/pdf" in text, f"文档附件没有告诉模型缺的是哪种文件: {text!r}"
    print("test_anthropic_image_and_document_are_announced: PASSED")


def test_responses_image_and_file_are_announced():
    text = extract_responses_content([
        {"type": "input_text", "text": "看图"},
        {"type": "input_image", "image_url": PNG},
        {"type": "input_file", "file_id": "file-1", "filename": "notes.txt"},
    ])
    assert "看图" in text
    assert IMAGE_PLACEHOLDER in text
    assert "notes.txt" in text, f"文件名丢了: {text!r}"
    # null 文本块不得崩溃
    assert extract_responses_content([{"type": "input_text", "text": None}]) == ""
    print("test_responses_image_and_file_are_announced: PASSED")


def run_all():
    test_openai_image_is_announced_not_silently_dropped()
    test_openai_file_is_announced_with_its_name()
    test_null_text_part_does_not_crash_and_is_not_echoed()
    test_system_parts_list_is_merged_as_text()
    test_anthropic_image_and_document_are_announced()
    test_responses_image_and_file_are_announced()
    print("\n✅ All content-normalization tests passed!")


if __name__ == "__main__":
    run_all()