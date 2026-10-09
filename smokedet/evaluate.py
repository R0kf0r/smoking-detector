"""Оценщик по протоколу организаторов (SPEC §6).

Пара «предсказание — эталон» образуется, если одновременно:
  * совпадают clip_id (и camera_id, когда он указан с обеих сторон);
  * интервалы пересекаются не меньше чем на 1 с после расширения эталона на 3 с в обе стороны;
  * человек тот же: IoU рамок ≥ 0,30, если у эталона есть рамка; иначе совпадение
    person_track_id, если он заполнен в эталоне; иначе проверка человека пропускается.
Сопоставление один к одному, жадно по наибольшему пересечению.
Предсказание не считается FP, если не меньше половины его длительности лежит в IGNORE
и оно не пересекает положительный эталон.

Запуск:
    python -m smokedet.evaluate --pred results/reference/events.json --gt labels/reference.csv \
        [--durations results/reference/timing.csv] [--out results/reference/metrics.md]
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import median


@dataclass
class GT:
    clip_id: str
    camera_id: str
    label: str           # POSITIVE | NEGATIVE | IGNORE
    start: float
    end: float
    scenario: str = ""
    person_track_id: int | None = None
    box: tuple | None = None
    comment: str = ""


def _num(value) -> float | None:
    if value is None:
        return None
    s = str(value).strip().replace(",", ".")
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    return float(s)


def _int(value) -> int | None:
    v = _num(value)
    return None if v is None else int(round(v))


def load_gt(path: str | Path) -> list[GT]:
    rows: list[GT] = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        delim = ";" if sample.count(";") > sample.count(",") else ","
        for r in csv.DictReader(f, delimiter=delim):
            r = {(k or "").strip(): (v or "").strip() for k, v in r.items()}
            label = r.get("label", "").upper()
            start, end = _num(r.get("start_sec")), _num(r.get("end_sec"))
            if label not in {"POSITIVE", "NEGATIVE", "IGNORE"} or start is None or end is None:
                continue
            box = None
            if all(_num(r.get(k)) is not None for k in ("x1", "y1", "x2", "y2")):
                box = tuple(_num(r[k]) for k in ("x1", "y1", "x2", "y2"))
            rows.append(GT(
                clip_id=r.get("clip_id", ""), camera_id=r.get("camera_id", ""), label=label,
                start=min(start, end), end=max(start, end), scenario=r.get("scenario", ""),
                person_track_id=_int(r.get("person_track_id")), box=box, comment=r.get("comment", ""),
            ))
    return rows


def load_predictions(path: str | Path) -> list[dict]:
    path = Path(path)
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("events", [])
        return list(data)
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_durations(path: str | Path | None) -> dict[str, float]:
    """clip_id → длительность, с. Берётся из timing.csv или inventory.csv (колонка duration_s)."""
    if not path:
        return {}
    out: dict[str, float] = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            d = _num(r.get("duration_s"))
            if r.get("clip_id") and d is not None:
                out[r["clip_id"]] = d
    return out


def overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def iou(a, b) -> float:
    ix = overlap(a[0], a[2], b[0], b[2])
    iy = overlap(a[1], a[3], b[1], b[3])
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _union_len(intervals: list[tuple[float, float]], lo: float, hi: float) -> float:
    clipped = sorted((max(lo, a), min(hi, b)) for a, b in intervals if min(hi, b) > max(lo, a))
    total, cur0, cur1 = 0.0, None, None
    for a, b in clipped:
        if cur1 is None or a > cur1:
            if cur1 is not None:
                total += cur1 - cur0
            cur0, cur1 = a, b
        else:
            cur1 = max(cur1, b)
    if cur1 is not None:
        total += cur1 - cur0
    return total


def _pred(p: dict) -> dict:
    box = None
    if all(_num(p.get(k)) is not None for k in ("x1", "y1", "x2", "y2")):
        box = tuple(_num(p[k]) for k in ("x1", "y1", "x2", "y2"))
    return {
        "clip_id": str(p.get("clip_id", "")), "camera_id": str(p.get("camera_id", "") or ""),
        "start": _num(p.get("start_sec")) or 0.0, "end": _num(p.get("end_sec")) or 0.0,
        "track": _int(p.get("person_track_id")), "box": box,
        "confidence": _num(p.get("confidence")) or 0.0, "event_id": p.get("event_id", ""),
    }


def evaluate(predictions: list[dict], gts: list[GT], durations: dict[str, float] | None = None,
             extend_s: float = 3.0, min_overlap_s: float = 1.0, iou_thr: float = 0.30) -> dict:
    preds = [_pred(p) for p in predictions]
    positives = [g for g in gts if g.label == "POSITIVE"]

    cands = []
    for pi, p in enumerate(preds):
        for gi, g in enumerate(positives):
            if p["clip_id"] != g.clip_id:
                continue
            if p["camera_id"] and g.camera_id and p["camera_id"] != g.camera_id:
                continue
            ov = overlap(p["start"], p["end"], g.start - extend_s, g.end + extend_s)
            if ov < min_overlap_s - 1e-9:
                continue
            if g.box is not None and p["box"] is not None:
                if iou(p["box"], g.box) < iou_thr:
                    continue
            elif g.person_track_id is not None and p["track"] is not None:
                if p["track"] != g.person_track_id:
                    continue
            cands.append((ov, pi, gi))
    cands.sort(key=lambda x: -x[0])
    used_p, used_g, pairs = set(), set(), []
    for ov, pi, gi in cands:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        pairs.append((pi, gi, ov))

    fps, ignored = [], []
    for pi, p in enumerate(preds):
        if pi in used_p:
            continue
        dur = max(1e-9, p["end"] - p["start"])
        ign = [(g.start, g.end) for g in gts if g.label == "IGNORE" and g.clip_id == p["clip_id"]]
        hits_pos = any(overlap(p["start"], p["end"], g.start, g.end) > 0
                       for g in positives if g.clip_id == p["clip_id"])
        if _union_len(ign, p["start"], p["end"]) >= 0.5 * dur and not hits_pos:
            ignored.append(pi)
        else:
            fps.append(pi)
    fns = [gi for gi in range(len(positives)) if gi not in used_g]

    tp, fp, fn = len(pairs), len(fps), len(fns)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else None
    latencies = [max(0.0, preds[pi]["start"] - positives[gi].start) for pi, gi, _ in pairs]

    fp_per_hour = None
    if durations:
        neg_seconds = 0.0
        for clip, dur in durations.items():
            pos = [(g.start, g.end) for g in positives if g.clip_id == clip]
            neg_seconds += max(0.0, dur - _union_len(pos, 0.0, dur))
        if neg_seconds > 0:
            fp_per_hour = fp / (neg_seconds / 3600.0)

    # разбивка по сценариям
    scen: dict[str, dict] = {}
    for gi, g in enumerate(positives):
        s = scen.setdefault(g.scenario or "—", {"TP": 0, "FN": 0, "FP": 0})
        s["TP" if gi in used_g else "FN"] += 1
    negs = [g for g in gts if g.label == "NEGATIVE"]
    for pi in fps:
        p = preds[pi]
        owner = None
        for g in negs:
            if g.clip_id == p["clip_id"] and overlap(p["start"], p["end"], g.start - extend_s, g.end + extend_s) > 0:
                owner = g.scenario or "—"
                break
        scen.setdefault(owner or "фон", {"TP": 0, "FN": 0, "FP": 0})["FP"] += 1

    errors = []
    for gi in fns:
        g = positives[gi]
        errors.append({"type": "FN", "clip_id": g.clip_id, "start": g.start, "end": g.end,
                       "scenario": g.scenario, "track": g.person_track_id, "comment": g.comment})
    for pi in fps:
        p = preds[pi]
        errors.append({"type": "FP", "clip_id": p["clip_id"], "start": p["start"], "end": p["end"],
                       "scenario": "", "track": p["track"], "confidence": p["confidence"],
                       "event_id": p["event_id"]})
    errors.sort(key=lambda e: (e["clip_id"], e["start"]))

    return {
        "TP": tp, "FP": fp, "FN": fn, "ignored_predictions": len(ignored),
        "precision": precision, "recall": recall, "f1": f1,
        "fp_per_hour": fp_per_hour,
        "median_latency_s": median(latencies) if latencies else None,
        "by_scenario": dict(sorted(scen.items())),
        "pairs": [{"event_id": preds[pi]["event_id"], "clip_id": positives[gi].clip_id,
                   "gt_start": positives[gi].start, "pred_start": preds[pi]["start"],
                   "overlap_s": round(ov, 3)} for pi, gi, ov in pairs],
        "errors": errors,
    }


def _fmt(x, digits: int = 3) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def to_markdown(m: dict, title: str = "Метрики") -> str:
    lines = [f"# {title}", "",
             "| TP | FP | FN | Precision | Recall | F1 | FP в час | Задержка, с |",
             "|---|---|---|---|---|---|---|---|",
             f"| {m['TP']} | {m['FP']} | {m['FN']} | {_fmt(m['precision'])} | {_fmt(m['recall'])} | "
             f"{_fmt(m['f1'])} | {_fmt(m['fp_per_hour'], 2)} | {_fmt(m['median_latency_s'], 2)} |",
             "", "## По сценариям", "", "| Сценарий | TP | FN | FP |", "|---|---|---|---|"]
    for s, v in m["by_scenario"].items():
        lines.append(f"| {s} | {v['TP']} | {v['FN']} | {v['FP']} |")
    lines += ["", "## Ошибки", ""]
    if not m["errors"]:
        lines.append("Ошибок нет.")
    for e in m["errors"]:
        extra = f", уверенность {e['confidence']:.2f}" if e.get("confidence") is not None and e["type"] == "FP" else ""
        lines.append(f"- {e['type']}: {e['clip_id']} {e['start']:.1f}–{e['end']:.1f} с, трек {e.get('track')}"
                     f"{', ' + e['scenario'] if e.get('scenario') else ''}{extra}")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Оценка по протоколу SPEC §6")
    ap.add_argument("--pred", required=True, help="events.json или .csv")
    ap.add_argument("--gt", required=True, help="labels/reference.csv")
    ap.add_argument("--durations", help="timing.csv или inventory.csv с колонкой duration_s")
    ap.add_argument("--out", help="куда сохранить отчёт .md (рядом сохранится .json)")
    args = ap.parse_args()
    m = evaluate(load_predictions(args.pred), load_gt(args.gt), load_durations(args.durations))
    md = to_markdown(m)
    print(md)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        out.with_suffix(".json").write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
