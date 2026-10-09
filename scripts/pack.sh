#!/bin/bash
# Сборка материалов к сдаче в ~/hackathon_submit/<имя>/ (не в iCloud):
#   bash scripts/pack.sh <Фамилии_через_подчёркивание> [видео_с_отметками.mp4] [исходное_видео.mp4]
# Пример:
#   bash scripts/pack.sh Фебенчукова_Смирнова_Полухин_Попова
# Без видео в аргументах берёт ролик скрытого набора из results/hidden/demo.txt (его делает hidden.sh)
# и ролик эталона из Гатчины (sm_6). Презентация и отчёт — из private/submit/.

if [ $# -lt 1 ]; then
  echo "Использование: bash scripts/pack.sh <Фамилии_через_подчёркивание> [видео_с_отметками] [исходное_видео]"
  exit 1
fi
NAME="$1"; RENDER="${2:-}"; SOURCE="${3:-}"
cd "$(dirname "$0")/.." || exit 1
OUT="$HOME/hackathon_submit/$NAME"
mkdir -p "$OUT"

echo "== Код с весами моделей, конфигом и README (без видео объекта и кэша)"
rm -f "$OUT/${NAME}_code.zip"
zip -qr "$OUT/${NAME}_code.zip" . -x "private/*" "results/cache/*" "*/__pycache__/*" ".pytest_cache/*" \
    ".git/*" "*.DS_Store" "results/*/signals/*" "results/*/debug/*" "results/*/demo/*"

echo "== Фиксация: коммит с тегом freeze-v1"
if git rev-parse -q --verify refs/tags/freeze-v1 >/dev/null; then
  {
    git for-each-ref refs/tags/freeze-v1 --format="тег freeze-v1 поставлен: %(taggerdate:iso)"
    git show -s --date=iso --format="коммит: %H%nдата коммита: %ad%nавтор: %an%n%s" "freeze-v1^{commit}"
    echo
    echo "Изменения кода и конфига после тега (пусто — не меняли):"
    git diff --stat freeze-v1 -- configs smokedet scripts requirements.txt
  } > "$OUT/${NAME}_freeze.txt"
else
  echo "   тега freeze-v1 нет — запустите bash scripts/freeze.sh"
fi

echo "== Предсказания на скрытом наборе, журнал времени, manifest"
for f in events.csv events.json timing.csv manifest.json; do
  if [ -f "results/hidden/$f" ]; then cp "results/hidden/$f" "$OUT/${NAME}_$f"; else echo "   нет results/hidden/$f"; fi
done

echo "== Отчёты"
[ -f results/hidden/REPORT.md ] && cp results/hidden/REPORT.md "$OUT/${NAME}_report_hidden.md"
[ -f results/reference/REPORT.md ] && cp results/reference/REPORT.md "$OUT/${NAME}_report_reference.md"
[ -f results/reference/calibration/report.md ] && cp results/reference/calibration/report.md "$OUT/${NAME}_calibration.md"

echo "== Презентация и отчёт"
if [ -f private/submit/presentation.pptx ]; then cp private/submit/presentation.pptx "$OUT/${NAME}.pptx"
else echo "   нет private/submit/presentation.pptx"; fi
[ -f private/submit/presentation.pdf ] && cp private/submit/presentation.pdf "$OUT/${NAME}.pdf"
if [ -f private/submit/report.docx ]; then cp private/submit/report.docx "$OUT/${NAME}_report.docx"
else echo "   нет private/submit/report.docx"; fi

echo "== Видео: исходное и с отметками"
copy_pair() {
  [ -f "$1" ] && cp "$1" "$OUT/${NAME}_marked_$(basename "$1")"
  [ -f "$2" ] && cp "$2" "$OUT/${NAME}_source_$(basename "$2")"
  return 0
}
if [ -n "$RENDER" ]; then
  copy_pair "$RENDER" "$SOURCE"
elif [ -f results/hidden/demo.txt ]; then
  copy_pair "$(sed -n 1p results/hidden/demo.txt)" "$(sed -n 2p results/hidden/demo.txt)"
fi
copy_pair private/renders/final/sm_6.mp4 private/reference/positive/sm_6.mp4

echo
ls -la "$OUT"
du -sh "$OUT"
echo "Готово: $OUT — загрузите содержимое в папку организаторов."
