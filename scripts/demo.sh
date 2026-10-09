#!/bin/bash
# Видео с отметками для одного ролика:  bash scripts/demo.sh <путь к ролику> [camera_id]
# Результат: private/renders/demo/<clip_id>.mp4 (поза берётся из кэша, если ролик уже обрабатывался)

if [ $# -lt 1 ]; then echo "Использование: bash scripts/demo.sh <путь к ролику> [camera_id]"; exit 1; fi
VIDEO="$1"; CAM="${2:-demo}"
source "$(dirname "$0")/_env.sh"
python -m smokedet.run --input "$VIDEO" --out-dir results/demo --render-dir private/renders/demo --camera-id "$CAM"
