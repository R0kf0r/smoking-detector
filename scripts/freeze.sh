#!/bin/bash
# Заморозка перед скрытым набором (один раз, 2–3 минуты):  bash scripts/freeze.sh
# Тесты → эталон с итоговым конфигом (поза и ответы VLM из кэша) → отчёты → git-коммит и тег freeze-v1.
# Если тег уже есть, ничего не меняет.

source "$(dirname "$0")/_env.sh"

if git rev-parse -q --verify refs/tags/freeze-v1 >/dev/null; then
  echo "Тег freeze-v1 уже есть (коммит $(git rev-list -n 1 freeze-v1 | cut -c1-8)): конфиг заморожен."
  exit 0
fi

echo "== Тесты"
python -m pytest -q || { echo "Тесты не прошли — пришлите этот вывод"; exit 1; }

echo "== Эталон с итоговым конфигом (поза и ответы VLM — из кэша)"
python -m smokedet.run --input private/reference --out-dir results/reference --camera-id reference > /dev/null \
  || { echo "Прогон эталона не прошёл — пришлите results/reference/run.log"; exit 1; }
python -m smokedet.report --run-dir results/reference --gt labels/reference.csv > /dev/null
python -m smokedet.calibrate --run-dir results/reference --gt labels/reference.csv --clip-level > /dev/null
sed -n '7,11p' results/reference/calibration/report.md

echo "== git: коммит и тег freeze-v1"
GITID=()
[ -z "$(git config user.name)" ] && GITID+=(-c "user.name=Vladislav Polukhin")
[ -z "$(git config user.email)" ] && GITID+=(-c "user.email=vladislav.v.polukhin@gmail.com")
git add -A
git "${GITID[@]}" commit -q -m "freeze-v1: итоговый конфиг и результаты эталона до скрытого набора" || exit 1
git "${GITID[@]}" tag -a freeze-v1 -m "Пороги, правила и промпт зафиксированы до скрытого набора" || exit 1
echo
git log -1 --date=format:"%d.%m %H:%M" --format="Готово: коммит %h от %ad, тег freeze-v1"
echo "Вечером: bash scripts/hidden.sh private/hidden <camera_id>"
