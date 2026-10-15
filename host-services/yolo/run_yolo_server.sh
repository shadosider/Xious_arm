#!/usr/bin/env bash
# SPDX-License-Identifier: MulanPSL-2.0
# 启动宿主侧 YOLO 检测服务（:8770）。venv 不在就先建一次。
#
# 容器侧不用装 torch：检测请求由容器内经 --net host 的 127.0.0.1 转发到这里。
set -euo pipefail

HS="$(cd "$(dirname "$0")" && pwd)"        # host-services/yolo
VENV="${YOLO_VENV:-$HS/.venv}"
PY="$VENV/bin/python"
MODEL="${YOLO_MODEL:-$HS/weights/best.pt}"

[[ -x "$PY" ]] || { echo "[run] venv 不在 → 先建一次"; bash "$HS/setup_venv.sh"; }

echo "[run] YOLO 检测服务 → http://${YOLO_HOST:-127.0.0.1}:${YOLO_PORT:-8770}"
echo "[run]   model=$MODEL"
exec "$PY" "$HS/yolo_detect_server.py" \
    --host "${YOLO_HOST:-127.0.0.1}" --port "${YOLO_PORT:-8770}" --model "$MODEL"
