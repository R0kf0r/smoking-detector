# smoking-detector — стартовый набор (v5)

Детектор поведения, похожего на курение, для хакатона «Видеофор» (Гатчина, 8–10.10.2026).
Пока здесь только среда и документы. Код напишет Claude Code по шагам из `docs/PROMPTS.md`.
Подробная пошаговая инструкция по запуску — в документе «Детектор курения: пошаговый запуск».

Приоритеты: корректность → устойчивость к условиям → стоимость → внешний вид.

Сроки:
- **9.10, ночь и утро** — среда, ядро системы, разметка и настройка на сценарных роликах организаторов;
- **9.10, день** — съёмка на объекте; фиксация настроек — строго до прогона скрытого набора;
- **9.10, вечер** — прогон скрытого набора и загрузка материалов **до 23:59**;
- **10.10** — защита (5 + 5 минут).

## Что где лежит

| Путь | Зачем |
|---|---|
| `CLAUDE.md` | правила и приоритеты проекта; Claude Code читает его в начале каждой сессии |
| `.claude/settings.json` | запрет Claude Code читать `private/` |
| `requirements.txt` | базовые библиотеки Python |
| `requirements-extra.txt` | интерфейс, запасная модель позы, клиент Ollama — ставятся позже, по необходимости |
| `docs/SPEC.md` | требования, переписанные из презентации |
| `docs/ACCURACY.md` | план точности: ошибки по сценариям P01–N08, архитектура, подбор порогов, сезоны, удешевление |
| `docs/DESIGN.md` | внешний вид: цвета, шрифт, видео с отметками, интерфейс, графики |
| `docs/design_reference.png` | синтетический макет: как должны выглядеть отметки на видео и «след жеста» |
| `static/` | шрифт Golos Text (OFL) для интерфейса и текста на видео |
| `docs/research.md` | подходы, инструменты, лицензии, данные, статьи |
| `docs/PROMPTS.md` | промпты для Claude Code по шагам |
| `docs/manifest_notes.md` | заметки для manifest.json и отчёта |
| `docs/DEFENSE.md` | защита 10.10: план доклада на 5 минут, вопросы жюри |
| `labels/` | разметка эталона и скрытого набора (только время и метки) |
| `data/public/` | публичные видео (HMDB51, Pexels) и журнал источников `SOURCES.md` |
| `private/` | видео с объекта: в облако не отправлять, в git не класть |
| `results/` | предсказания, метрики, отчёты (только числа) |

## Запуск среды (macOS 14 или новее, Apple Silicon)

1. Проверить `git --version` и `conda --version`. Нет git — `xcode-select --install`. Нет conda — поставить Miniforge с conda-forge.org/download.
2. Распаковать архив в `~/PycharmProjects/` и сделать первый коммит. Презентацию организаторов в папку не кладём: на слайдах есть кадры камер объекта, а требования из неё уже в `docs/SPEC.md`.
   `cd ~/PycharmProjects/smoking-detector`, `git init`, `git add -A`, `git commit -m "Стартовый набор"`.
3. Окружение:
   `conda create -n smoke python=3.12 -y`, `conda activate smoke`,
   `conda install -c conda-forge ffmpeg -y`, `pip install -r requirements.txt`.
4. Проверка:
   `python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"` → версия и `True`;
   `yolo pose predict model=yolo26s-pose.pt source=https://ultralytics.com/images/bus.jpg device=mps` → найдены люди;
   `python -m mlx_vlm.generate --model mlx-community/Qwen3-VL-4B-Instruct-4bit --image https://ultralytics.com/images/bus.jpg --prompt "How many people are in the photo? Answer with one number." --max-tokens 20 --temperature 0` → число (первый раз скачивает около 3 ГБ).
5. В приложении Claude открыть вкладку **Code** → **Local** → **Select folder** → эта папка, режим **Plan**, начать с промпта 1 из `docs/PROMPTS.md`.
