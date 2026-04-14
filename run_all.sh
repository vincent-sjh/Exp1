#!/usr/bin/env bash
# 用 run_experiments.py 的默认参数依次跑：金字塔重建、融合、接缝缩放、物体移除
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
exec "$ROOT/.venv/bin/python" "$ROOT/run_experiments.py" all
