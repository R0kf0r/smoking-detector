#!/bin/bash
# Эталон одной командой (из папки проекта):  bash scripts/reference.sh
# Тесты → опись и грубая разметка → прогон без VLM → прогон с VLM (поза из кэша) → подбор порогов.
# Видео остаются на этом компьютере; в results/ пишутся только числа.

source "$(dirname "$0")/_env.sh"

echo "== Тесты"
python -m pytest -q || { echo "Тесты не прошли — пришлите этот вывод"; exit 1; }

if [ ! -f labels/reference.csv ]; then
  echo "== Опись роликов и грубая разметка по папкам positive/negative"
  python -m smokedet.inventory private/reference --labels-template labels/reference.csv --guess-labels || exit 1
fi

echo "== Прогон 1 из 2: без VLM (поза и трекинг — самая долгая часть)"
python -m smokedet.run --input private/reference --out-dir results/reference_novlm --no-vlm "$@"

echo "== Прогон 2 из 2: с VLM (поза берётся из кэша; первый раз скачивается модель ~3 ГБ)"
python -m smokedet.run --input private/reference --out-dir results/reference "$@"

echo "== Подбор порогов"
python -m smokedet.calibrate --run-dir results/reference_novlm --gt labels/reference.csv --clip-level > /dev/null
python -m smokedet.calibrate --run-dir results/reference --gt labels/reference.csv --clip-level

echo
echo "Готово. Отчёты: results/reference/calibration/report.md, results/reference_novlm/calibration/report.md"
