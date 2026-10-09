#!/bin/bash
# Исправление проверки VLM на эталоне одной командой (из папки проекта):  bash scripts/vlm_fix.sh
# 1) выбирает промпт VLM на кадрах эталона (≈ 6 мин), 2) заново проверяет все ролики эталона с этим
# промптом (поза из кэша), 3) подбирает пороги и пишет отчёт. Видео остаются на этом компьютере.

if pgrep -f "smokedet.run" >/dev/null 2>&1; then
  echo "Сейчас уже идёт прогон smokedet.run. Остановите его в том окне Терминала (Ctrl+C) и запустите снова."
  exit 1
fi
source "$(dirname "$0")/_env.sh"

echo "== 1/3 Выбор промпта VLM"
python -m smokedet.vlm_probe --run-dir results/reference_novlm --gt labels/reference.csv --apply || exit 1

echo "== 2/3 Проверка эталона с новым промптом"
python -m smokedet.run --input private/reference --out-dir results/reference

echo "== 3/3 Подбор порогов и отчёт"
python -m smokedet.calibrate --run-dir results/reference --gt labels/reference.csv --clip-level
python -m smokedet.report --run-dir results/reference --gt labels/reference.csv > /dev/null
echo
echo "Готово. Отчёты: results/vlm_probe.md, results/reference/calibration/report.md, results/reference/REPORT.md"
