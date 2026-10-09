"""Отчёт по прогону (только числа): данные, время обработки, результат, разбор ошибок.

    python -m smokedet.report --run-dir results/reference --gt labels/reference.csv [--out results/reference/REPORT.md]

Разметка может быть грубой («весь клип позитивный / негативный»): тогда считаем найденные
позитивные клипы и ложные тревоги на негативных. Если в разметке есть интервалы событий,
дополнительно считаем метрики по протоколу SPEC §6.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .evaluate import evaluate, load_durations, load_gt, to_markdown


def _f(x, digits=2) -> str:
    if x is None:
        return "—"
    return f"{x:.{digits}f}".replace(".", ",")


def why_missed(dbg: dict, sig_path: Path) -> str:
    """Почему в позитивном клипе нет события — по числам из debug и сигналов."""
    tracks = dbg.get("tracks", [])
    if not tracks:
        return "людей не найдено"
    if not any(t.get("has_signal") for t in tracks):
        return "люди мельче порога видимости (80 px)"
    cycles = dbg.get("cycles", [])
    if not cycles:
        d_min = None
        if sig_path.is_file():
            vals = []
            with open(sig_path, encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    try:
                        vals.append(float(r["d"]))
                    except (ValueError, KeyError):
                        pass
            vals = [v for v in vals if not np.isnan(v)]
            d_min = min(vals) if vals else None
        rej = dbg.get("rejected", [])
        reasons = sorted({r["reason"] for r in rej})
        extra = f"; отброшены паузы: {', '.join(reasons)}" if reasons else ""
        if d_min is None:
            return "нет точек лица или кистей (человек спиной, закрыт, мелкий)" + extra
        return f"кисть не подходит ко рту достаточно близко (минимум d = {_f(d_min)}){extra}"
    vetoed = [c for c in cycles if c["verdict"] == "veto"]
    valid = [c for c in cycles if c["verdict"] != "veto"]
    if vetoed and len(valid) < 2:
        reasons = sorted({c["reason"] for c in vetoed})
        return f"циклы отброшены проверкой ({'; '.join(reasons)}), осталось {len(valid)}"
    if len(valid) == 1:
        return "один цикл без прямого признака (VLM не увидела сигарету)"
    return "циклы есть, но не складываются в событие (дальше 20 с друг от друга или у разных людей)"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m smokedet.report", description="Отчёт по прогону")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--gt", help="разметка (labels/reference.csv)")
    ap.add_argument("--out", help="куда записать (по умолчанию <run-dir>/REPORT.md)")
    args = ap.parse_args(argv)
    run = Path(args.run_dir)
    out = Path(args.out) if args.out else run / "REPORT.md"

    timing = list(csv.DictReader(open(run / "timing.csv", encoding="utf-8"))) if (run / "timing.csv").is_file() else []
    events = json.loads((run / "events.json").read_text(encoding="utf-8")) if (run / "events.json").is_file() else []
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8")) if (run / "manifest.json").is_file() else {}
    debug = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted((run / "debug").glob("*.json"))}

    ok = [t for t in timing if t.get("status") == "ok"]
    video_s = sum(float(t["duration_s"] or 0) for t in ok)
    proc_s = sum(float(t["processing_s"] or 0) for t in ok)
    hw = manifest.get("hardware", {})
    cfgv = manifest.get("config", {}).get("values", {})
    lines = [f"# Отчёт: {run.name}", ""]
    lines += ["## Данные и время обработки", "",
              f"- Клипов: {len(timing)} (с ошибкой: {len(timing) - len(ok)}), видео {_f(video_s / 60, 1)} мин.",
              f"- Обработка: {_f(proc_s / 60, 1)} мин, то есть {_f(proc_s / video_s * 60 if video_s else None, 1)} с "
              f"на минуту видео.",
              f"- Оборудование: {hw.get('chip', '—')}, память {hw.get('memory_gb', '—')} ГБ, ускоритель "
              f"{hw.get('device', '—')}.",
              f"- Модель позы: {cfgv.get('pose', {}).get('model', '—')}, вход {cfgv.get('pose', {}).get('imgsz', '—')} px, "
              f"{cfgv.get('video', {}).get('target_fps', '—')} кадров/с; VLM: "
              f"{'да' if manifest.get('vlm', {}).get('used_in_run') else 'нет'}.",
              f"- Событий: {len(events)}.", ""]

    if args.gt:
        gts = load_gt(args.gt)
        clips = set(debug)
        gts = [g for g in gts if g.clip_id in clips]
        coarse = all(g.start == 0 for g in gts if g.label == "POSITIVE")
        pos_clips = sorted({g.clip_id for g in gts if g.label == "POSITIVE"})
        neg_clips = sorted({g.clip_id for g in gts if g.label == "NEGATIVE"} - set(pos_clips))
        ev_by_clip: dict[str, list] = {}
        for e in events:
            ev_by_clip.setdefault(e["clip_id"], []).append(e)
        found = [c for c in pos_clips if ev_by_clip.get(c)]
        fp_events = sum(len(ev_by_clip.get(c, [])) for c in neg_clips)
        neg_h = sum(float(t["duration_s"] or 0) for t in ok if t["clip_id"] in neg_clips) / 3600
        lines += ["## Результат на уровне клипов", "",
                  f"- Позитивных клипов найдено: **{len(found)} из {len(pos_clips)}** "
                  f"(recall по клипам {_f(len(found) / len(pos_clips) if pos_clips else None)}).",
                  f"- Ложных тревог на негативных клипах: **{fp_events}** на {_f(neg_h * 60, 1)} мин видео "
                  f"({_f(fp_events / neg_h if neg_h else None, 1)} в час); негативных клипов с тревогой: "
                  f"{sum(1 for c in neg_clips if ev_by_clip.get(c))} из {len(neg_clips)}.", ""]
        if not coarse:
            m = evaluate(events, gts, load_durations(run / "timing.csv"))
            lines += [to_markdown(m, title="Метрики по протоколу SPEC §6").replace("# Метрики", "## Метрики"), ""]
        lines += ["## Разбор пропусков (позитивные клипы без события)", ""]
        missed = [c for c in pos_clips if not ev_by_clip.get(c)]
        if not missed:
            lines.append("Пропусков нет.")
        for c in missed:
            lines.append(f"- `{c}`: {why_missed(debug[c], run / 'signals' / f'{c}.csv')}")
        lines += ["", "## Разбор ложных тревог (события в негативных клипах)", ""]
        if not fp_events:
            lines.append("Ложных тревог нет.")
        for c in neg_clips:
            for e in ev_by_clip.get(c, []):
                cyc = [x for x in debug[c].get("events", []) if x["event_id"] == e["event_id"]]
                votes = {}
                for x in (cyc[0]["cycles"] if cyc else []):
                    for k, v in x.get("votes", {}).items():
                        votes[k] = votes.get(k, 0) + v
                vs = ", ".join(f"{k} {v}" for k, v in sorted(votes.items(), key=lambda kv: -kv[1])) or "нет"
                lines.append(f"- `{c}` {_f(e['start_sec'], 1)}–{_f(e['end_sec'], 1)} с, ID {e['person_track_id']}, "
                             f"уверенность {_f(e['confidence'])}, циклов {cyc[0]['n_cycles'] if cyc else '—'}; "
                             f"ответы проверки: {vs}")
        lines.append("")
    lines += ["## Время по клипам", "", "| Клип | Видео, с | Обработка, с | с/мин | Треков | Циклов | Событий |",
              "|---|---|---|---|---|---|---|"]
    for t in timing:
        lines.append(f"| {t['clip_id']} | {t.get('duration_s', '')} | {t.get('processing_s', '')} | "
                     f"{t.get('sec_per_min', '')} | {t.get('n_tracks', '')} | {t.get('n_cycles', '')} | "
                     f"{t.get('n_events', '')} |")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
