#!/bin/bash
# Публикация на GitHub одной командой (из папки проекта):  bash tools/publish.sh
# 1) коммит: просмотрщик, замер ресурсов, LICENSE (AGPL-3.0) и README для GitHub —
#    код, конфиг и скрипты конвейера после тега freeze-v1 не меняются;
# 2) push ветки main и тега freeze-v1 в https://github.com/R0kf0r/smoking-detector
# Видео, кадры, разметка и private/ в репозиторий не попадают.

cd "$(dirname "$0")/.." || exit 1
URL="https://github.com/R0kf0r/smoking-detector.git"

git rev-parse -q --verify refs/tags/freeze-v1 >/dev/null || { echo "Нет тега freeze-v1 — сначала bash scripts/freeze.sh"; exit 1; }
if [ -f tools/README_github.md ]; then
  cp tools/README_github.md README.md && rm -f tools/README_github.md
fi
git add viewer tools LICENSE README.md
if ! git diff --cached --quiet; then
  git commit -q -F - <<'MSG' || exit 1
Просмотрщик, замер ресурсов, лицензия AGPL-3.0, README для GitHub

Код, конфиг и скрипты конвейера не менялись после тега freeze-v1.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01RVQpkU3dVspmji3pmbebm8
MSG
fi

echo "== Проверка перед публикацией"
CHANGED="$(git diff --stat freeze-v1 HEAD -- configs smokedet scripts requirements.txt)"
[ -n "$CHANGED" ] && { echo "Код или конфиг изменились после freeze-v1:"; echo "$CHANGED"; exit 1; }
BAD="$(git ls-files | grep -Ei '^private/|^labels/.*\.csv$|\.(mp4|mov|avi|mkv|wmv|ts|dav|jpg|jpeg)$')"
[ -n "$BAD" ] && { echo "В репозитории есть файлы, которые нельзя публиковать:"; echo "$BAD"; exit 1; }
echo "   код после freeze-v1 не менялся, видео и разметки в репозитории нет; файлов: $(git ls-files | wc -l | tr -d ' ')"

if git remote get-url origin >/dev/null 2>&1; then git remote set-url origin "$URL"; else git remote add origin "$URL"; fi
echo "== Отправка на GitHub (если попросит логин — имя R0kf0r и токен GitHub вместо пароля)"
git push -u origin main && git push origin freeze-v1 || {
  echo; echo "Не получилось отправить. Можно из PyCharm: Git → Push (Cmd+Shift+K), отметить «Push tags»."; exit 1; }
echo; echo "Готово: https://github.com/R0kf0r/smoking-detector"
