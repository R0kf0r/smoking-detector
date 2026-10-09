#!/bin/bash
# Скрытый набор:  bash scripts/hidden.sh <папка с роликами> <camera_id> [доп. флаги smokedet.run]
# Пример:         bash scripts/hidden.sh private/hidden cam01
# Разные камеры:  bash scripts/hidden.sh private/hidden "" --camera-map labels/cameras.csv
# После прогона:  results/hidden/events.csv, events.json, timing.csv, manifest.json, REPORT.md
#                 и видео с отметками для самого уверенного события (private/renders/hidden/)

if [ $# -lt 1 ]; then
  echo "Использование: bash scripts/hidden.sh <папка с роликами> <camera_id> [флаги]"; exit 1
fi
IN="$1"; CAM="${2:-}"; shift; [ $# -gt 0 ] && shift
SCRIPTS="$(cd "$(dirname "$0")" && pwd)"
if ! git -C "$SCRIPTS/.." rev-parse -q --verify refs/tags/freeze-v1 >/dev/null 2>&1; then
  echo "== Тега freeze-v1 ещё нет — сначала заморозка"
  bash "$SCRIPTS/freeze.sh" || exit 1
fi
source "$SCRIPTS/_env.sh"

ARGS=(--input "$IN" --out-dir results/hidden)
[ -n "$CAM" ] && ARGS+=(--camera-id "$CAM")
python -m smokedet.run "${ARGS[@]}" "$@"
python -m smokedet.report --run-dir results/hidden > /dev/null

echo "== Видео с отметками для самого уверенного события (поза и VLM из кэша — только рендер)"
if python -m smokedet.demo --run-dir results/hidden --input "$IN" "$@" > results/hidden/demo.log 2>&1; then
  tail -n 1 results/hidden/demo.log
else
  echo "   не получилось (см. results/hidden/demo.log) — для сдачи есть ролик эталона из Гатчины"
fi
echo
echo "Готово: results/hidden/events.csv, events.json, timing.csv, manifest.json, REPORT.md"
echo "Дальше: bash scripts/pack.sh Фебенчукова_Смирнова_Полухин_Попова"
