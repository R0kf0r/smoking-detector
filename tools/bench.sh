#!/bin/bash
# Замер ресурсов конвейера (macOS):  bash tools/bench.sh [ролик]
# Время по этапам, загрузка CPU и GPU, память. 1–3 минуты; результаты в results/ не трогает.
source "$(dirname "$0")/../scripts/_env.sh"
python tools/bench.py "$@"
