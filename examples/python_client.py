"""OpenAI SDK 调用本地网关示例。

依赖: pip install openai
运行: .venv/bin/python examples/python_client.py
"""

from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="not-needed")

MODEL = "qwen-max-latest"  # 换成 deepseek-reasoner / qwq-32b 等

# --- 流式对话 ---
print(f"== 流式 {MODEL} ==")
stream = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "用一句话介绍你自己"}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
print("\n")

# --- 多轮续聊（conversation_id）---
print("== 多轮续聊 ==")
resp = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "我的幸运数字是 7，记住它"}],
)
print("AI:", resp.choices[0].message.content)
cid = getattr(resp, "conversation_id", None)
print("conversation_id:", cid)

resp2 = client.chat.completions.create(
    model=MODEL,
    extra_body={"conversation_id": cid},
    messages=[{"role": "user", "content": "我的幸运数字是多少？"}],
)
print("AI:", resp2.choices[0].message.content)

# --- 网页端专属能力（经 extra_body）---
print("\n== 深度思考 + 联网搜索（DeepSeek）==")
resp3 = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "今天 A 股有什么值得关注的消息？"}],
    extra_body={"thinking": True, "search": True},
)
print("AI:", resp3.choices[0].message.content)
