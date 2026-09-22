#!/usr/bin/env bash
# 一键启动本地网关
set -euo pipefail
cd "$(dirname "$0")"

[ -f .env ] && set -a && . ./.env && set +a

exec .venv/bin/uvicorn server:app --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}" --reload
