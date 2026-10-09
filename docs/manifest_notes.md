# Заметки для manifest.json и отчёта

Организаторы требуют указать источники данных, лицензии, версии моделей, пороги, промпты, состав дообучения и оборудование. Заполняем по ходу работы.

## Оборудование

- Компьютер: MacBook Pro, Apple Silicon (модель чипа и объём памяти — заполнить)
- ОС: macOS (версия — заполнить)
- Ускоритель: mps

## Модели

| Модель | Версия / файл | Источник | Лицензия | Для чего |
|---|---|---|---|---|
| YOLO26-pose | `yolo26s-pose.pt` (уточнить) | Ultralytics | AGPL-3.0 | люди, 17 ключевых точек, трекинг |
| Qwen3-VL | `mlx-community/Qwen3-VL-4B-Instruct-4bit` через mlx-vlm или `qwen3-vl:…` через Ollama (уточнить) | Qwen / Ollama | Apache-2.0 | проверка кандидатов |

## Шрифты

| Шрифт | Файлы | Источник | Лицензия |
|---|---|---|---|
| Golos Text | `static/GolosText-Regular.ttf`, `static/GolosText-SemiBold.ttf` | github.com/googlefonts/golos-text | SIL OFL 1.1 (`static/OFL-GolosText.txt`) |

## Библиотеки

(версии ultralytics, torch, opencv, ollama — заполнит Claude Code на шаге 2)

## Данные

См. `data/public/SOURCES.md`. Дообучение: (нет / что именно).

## Пороги и правила

См. `configs/default.yaml` на момент тега `freeze-v1`.

## Промпты VLM

(вставить точный текст промпта)
