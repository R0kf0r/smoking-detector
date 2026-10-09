#!/bin/bash
# Просмотрщик результатов (macOS):  bash viewer/start.sh
# Откроется http://127.0.0.1:8765 — выберите или перетащите ролик, нажмите «Обработать ролик».
# Работает только на этом компьютере; остановить — Ctrl+C в этом окне Терминала.

source "$(dirname "$0")/../scripts/_env.sh"
PORT="${PORT:-8765}"
( sleep 2; command -v open >/dev/null 2>&1 && open "http://127.0.0.1:$PORT" ) >/dev/null 2>&1 &
python viewer/app.py --port "$PORT" "$@"
