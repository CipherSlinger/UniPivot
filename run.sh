#!/usr/bin/env bash
# 一键启动本地网关
#
# 解释器选择与启动方式的两条约束：
#
# 1) 不直接执行 `.venv/bin/uvicorn`。虚拟环境一旦被**移动或复制**（本机的 .venv 就是
#    从别处搬过来的），venv 生成的控制台脚本里固化的是**创建时**的绝对解释器路径，
#    于是出现 `bad interpreter: No such file or directory`。改用 `python -m uvicorn`：
#    由所选解释器自己加载模块，不经过 shebang，因此虚拟环境与系统 Python 都能用。
# 2) 解释器按「项目 venv 优先、系统 Python 兜底」的顺序解析，两者都能跑起来；都不存在时
#    给出明确指引而不是让 shell 抛一句难懂的报错。
set -euo pipefail
cd "$(dirname "$0")"

# 部署配置（run.sh 会 source .env，因此封禁租约等声明在启动时即生效）
[ -f .env ] && set -a && . ./.env && set +a

# 解析解释器：venv 优先（依赖装在 venv 里），系统 Python 兜底。
# 只认「真的能执行」的解释器：venv 存在但解释器是断链时自动退到系统 Python。
PY=""
for cand in ".venv/bin/python" ".venv/bin/python3" "python3" "python"; do
  if command -v "$cand" >/dev/null 2>&1 && "$cand" -c "import uvicorn" >/dev/null 2>&1; then
    PY="$cand"
    break
  fi
done

if [ -z "$PY" ]; then
  echo "未找到可用的 Python 解释器（要求在 .venv/bin/python 或系统 python3 中可 import uvicorn）。" >&2
  echo "请先准备环境：" >&2
  echo "  python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt" >&2
  echo "  .venv/bin/python -m playwright install chromium   # 若系统未装 Chrome" >&2
  exit 1
fi

echo "[unipivot] 使用解释器: $($PY -c 'import sys; print(sys.executable)')"
exec "$PY" -m uvicorn server:app --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}" --reload
