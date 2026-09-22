#!/usr/bin/env bash
# 本地网关调用示例：非流式 / 流式 / 多轮续聊
set -euo pipefail
BASE="${BASE:-http://127.0.0.1:8000/v1}"

echo "== 1) 列出模型 =="
curl -s "$BASE/models" | python3 -m json.tool | head -20

echo
echo "== 2) 非流式：Qwen =="
curl -s "$BASE/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen-max-latest","messages":[{"role":"user","content":"用一句话介绍你自己"}]}'

echo
echo "== 3) 流式：DeepSeek =="
curl -sN "$BASE/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-chat","messages":[{"role":"user","content":"1+1=?"}],"stream":true}'
