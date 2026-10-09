# smoking-detector (v6)

Детектор поведения, похожего на курение, для хакатона «Видеофор» (Гатчина, 8–10.10.2026).
По видео со стационарных уличных камер находит эпизоды «рука ко рту → пауза 0,5–5 с → рука от рта»
и выдаёт события в формате организаторов (SPEC §2).

Приоритеты: корректность → устойчивость к условиям → стоимость → внешний вид.

## Одна команда запуска

```bash
conda activate smoke
python -m smokedet.run --input private/hidden --out-dir results/hidden --camera-id <камера>
```

Результат в `results/hidden/`: `events.json` и `events.csv` (события), `timing.csv` (журнал времени),
`manifest.json` (пороги, правила, модели, версии, промпт, оборудование), `debug/` и `signals/` (числа для разбора).

## Как это работает

1. **Кандидаты.** YOLO26-pose находит людей и 17 точек скелета, BoT-SORT с ReID даёт постоянные ID,
   обрывки треков склеиваются. Сигнал d(t) — расстояние «кисть — рот» в длинах торса.
   Автомат с гистерезисом (пороги входа и выхода — в `configs/default.yaml`, подобраны на эталоне)
   находит циклы с паузой у рта 0,5–5 с; затяжки, между которыми рука с сигаретой остаётся у лица,
   разделяются «относительным выходом» (подъём d над минимумом паузы).
2. **Проверка.** Кадры паузы у рта (кроп лица и кисти) проверяет локальная VLM Qwen3-VL-4B (mlx-vlm)
   и детектор предметов COCO. Сигарета — прямой признак; телефон, бутылка, банка, стакан, еда, леденец,
   зубочистка, ручка — вето; обе руки у лица без прямого признака — вето. Вейп и «пустая рука» в итоговом
   конфиге не считаются ни признаком, ни вето: бутылку у рта VLM иногда называет вейпом, а сигарету на уличной
   камере часто не видно (переключатели `vlm.vape_is_smoke` и `vlm.hand_veto`).
3. **События** строго по SPEC §3–4: два цикла за 20 с или один цикл + прямой признак; склейка циклов
   одного человека при паузе ≤ 15 с; разные люди — разные события; человек ниже 80 px — не выдаём.
4. **Кэш.** Тяжёлый прогон (рамки, ID, точки, ответы VLM) сохраняется в `results/cache/`,
   поэтому пороги перебираются за минуты (`smokedet.calibrate`, leave-one-clip-out).

## Скрипты (macOS, из папки проекта)

| Что | Команда |
|---|---|
| Эталон целиком: тесты, опись, два прогона (без VLM и с VLM), подбор порогов | `bash scripts/reference.sh` |
| Заморозка до скрытого набора: тесты, эталон с итоговым конфигом, git-тег `freeze-v1` | `bash scripts/freeze.sh` |
| Скрытый набор + видео с отметками для самого уверенного события | `bash scripts/hidden.sh private/hidden <camera_id>` |
| Видео с отметками для одного ролика | `bash scripts/demo.sh <путь к ролику> [camera_id]` |
| Собрать материалы к сдаче в `~/hackathon_submit/<имя>/` | `bash scripts/pack.sh <Фамилии> [видео с отметками] [исходное видео]` |

Скрипты сами находят conda, при необходимости создают окружение `smoke`, ставят зависимости и не дают Mac уснуть.

## Команды

| Что | Команда |
|---|---|
| Тесты | `python -m pytest -q` |
| Опись роликов и заготовка разметки | `python -m smokedet.inventory private/reference --labels-template labels/reference.csv --guess-labels` |
| Прогон эталона с видео и метриками | `python -m smokedet.run --input private/reference --out-dir results/reference --render-dir private/renders/reference --gt labels/reference.csv` |
| Подбор порогов (leave-one-clip-out) | `python -m smokedet.calibrate --run-dir results/reference --gt labels/reference.csv` (с `--apply` — записать в конфиг) |
| Отчёт по прогону (данные, время, результат, разбор ошибок) | `python -m smokedet.report --run-dir results/reference --gt labels/reference.csv` |
| Метрики отдельно | `python -m smokedet.evaluate --pred results/reference/events.json --gt labels/reference.csv --durations results/reference/timing.csv --out results/reference/metrics.md` |
| Скрытый набор | `python -m smokedet.run --input private/hidden --out-dir results/hidden --camera-id <камера>` |
| Видео с отметками для одного клипа | `python -m smokedet.run --input private/hidden/<клип>.mp4 --out-dir results/demo --render-dir private/renders/demo --camera-id <камера>` |
| Видео с отметками для самого уверенного события прогона (поза и VLM из кэша) | `python -m smokedet.demo --run-dir results/hidden --input private/hidden` |

Полезные флаги `smokedet.run`:

- `--preset fast` — быстрее (yolo26n-pose, 10 к/с), `--preset accurate` — точнее (yolo26m-pose);
- `--no-vlm` — без VLM (если она не скачана или не хватает времени), детектор предметов остаётся;
- `--roi <файл или папка>` — маска зоны интереса: PNG (зона белая на чёрном) или JSON с полигоном;
- `--camera-map cameras.csv` — свой `camera_id` для каждого клипа (колонки `clip_id,camera_id`);
- `--set cycles.enter_threshold=0.18` — переопределить любой параметр конфига;
- `--render-only-events` — видео только для клипов, где есть события.

Все пороги — в `configs/default.yaml`. Итоговые значения — середина плато F1 на эталоне
(`results/reference/calibration/report.md`). Фиксация — `bash scripts/freeze.sh` (коммит и тег `freeze-v1`),
после неё пороги не меняем.

## Результат на эталоне (итоговый конфиг)

Разметка эталона — по роликам, поэтому и оценка по роликам: позитивный ролик найден, если в нём есть событие;
каждое событие в негативном ролике — ложная тревога. Два ролика с белой палочкой у рта помечены IGNORE.

| Найдено позитивных | Ложные тревоги | Ложных в час | Precision | Recall | F1 | F1, leave-one-clip-out |
|---|---|---|---|---|---|---|
| 15 из 19 | 10 на 12,6 мин негативов | 47,6 | 0,60 | 0,79 | 0,68 | 0,57 |

Скорость на MacBook Pro M1 Pro (32 ГБ): поза и трекинг 65 с, VLM 28 с — всего 92 с на минуту видео.

## Что где лежит

| Путь | Зачем |
|---|---|
| `smokedet/` | код: `run.py` (запуск), `pose.py`, `tracks.py`, `signal.py`, `cycles.py`, `verify.py`, `events.py`, `evaluate.py`, `render.py`, `calibrate.py`, `vlm_probe.py`, `report.py`, `demo.py`, `inventory.py`, `roi.py`, `video.py` |
| `configs/default.yaml` | все пороги и параметры |
| `tests/` | тесты оценщика, автомата циклов, правил событий, склейки треков, рендера |
| `models/` | веса YOLO (скачиваются при первом запуске из релизов Ultralytics) |
| `docs/` | требования (SPEC), план точности, дизайн, защита |
| `labels/` | разметка (только время и метки) |
| `private/` | видео с объекта и видео с отметками — в git не попадает |
| `results/` | события, метрики, журнал времени (только числа) |

## Среда (macOS 14+, Apple Silicon)

```bash
conda create -n smoke -c conda-forge --override-channels python=3.12 -y
conda activate smoke
conda install -c conda-forge --override-channels ffmpeg -y
pip install -r requirements.txt
```

Лицензии: Ultralytics YOLO26 — AGPL-3.0; Qwen3-VL — Apache-2.0; шрифт Golos Text — SIL OFL 1.1.
