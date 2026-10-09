# Общая часть скриптов: находит conda, при необходимости создаёт окружение smoke и ставит зависимости.
# Не запускается сам по себе: его подключают reference.sh, hidden.sh и demo.sh.

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

find_conda() {
  if command -v conda >/dev/null 2>&1; then return 0; fi
  for d in "$HOME/miniforge3" "$HOME/anaconda3" "$HOME/miniconda3" "/opt/anaconda3" "/opt/miniconda3" \
           "/opt/homebrew/anaconda3" "/opt/homebrew/Caskroom/miniforge/base" "/usr/local/anaconda3"; do
    if [ -x "$d/bin/conda" ]; then export PATH="$d/bin:$PATH"; return 0; fi
  done
  return 1
}

find_conda || { echo "Не найден conda. Поставьте Miniforge: https://conda-forge.org/download"; exit 1; }
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | awk '{print $1}' | grep -qx smoke; then
  echo "== Создаю окружение smoke (Python 3.12): 5–10 минут"
  conda create -n smoke -c conda-forge --override-channels python=3.12 -y || exit 1
fi
conda activate smoke || exit 1
command -v ffmpeg >/dev/null 2>&1 || conda install -c conda-forge --override-channels ffmpeg -y || exit 1
echo "== Проверяю зависимости"
pip install -q -r requirements.txt || exit 1

# Mac не засыпает, пока работает скрипт
if command -v caffeinate >/dev/null 2>&1; then caffeinate -i -w $$ & fi
