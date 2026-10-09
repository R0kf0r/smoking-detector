"""Опись роликов (только числа: длительность, частота, разрешение, кодек) и заготовка разметки.

    python -m smokedet.inventory private/reference --out results/inventory_reference.csv \\
        --labels-template labels/reference_template.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .video import list_videos, probe

LABEL_FIELDS = ["clip_id", "camera_id", "label", "start_sec", "end_sec", "scenario", "person_hint",
                "person_track_id", "x1", "y1", "x2", "y2", "comment"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m smokedet.inventory", description="Опись роликов")
    ap.add_argument("folder", help="папка с роликами (ищет во вложенных)")
    ap.add_argument("--out", help="CSV описи (по умолчанию results/inventory_<папка>.csv)")
    ap.add_argument("--labels-template", help="создать заготовку разметки (по строке на клип)")
    ap.add_argument("--guess-labels", action="store_true",
                    help="в заготовке: папка со словом pos/позит → POSITIVE на весь клип, neg/негат → NEGATIVE")
    args = ap.parse_args(argv)

    folder = Path(args.folder)
    videos = list_videos(folder)
    if not videos:
        print(f"Роликов не найдено: {folder}")
        return 1
    rows = []
    for v in videos:
        try:
            info = probe(v)
        except Exception as exc:          # битый файл не должен останавливать опись
            info = {"width": 0, "height": 0, "fps": 0, "duration_s": 0, "codec": f"ошибка: {exc}"[:60]}
        rows.append({"clip_id": v.stem, "folder": v.relative_to(folder).parent.as_posix(), "file": v.name,
                     "duration_s": info["duration_s"], "fps": info["fps"], "width": info["width"],
                     "height": info["height"], "codec": info["codec"],
                     "size_mb": round(v.stat().st_size / 2**20, 1)})
    out = Path(args.out) if args.out else Path("results") / f"inventory_{folder.name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    total = sum(float(r["duration_s"] or 0) for r in rows)
    w = max(len(r["clip_id"]) for r in rows)
    print(f"{'clip_id':<{w}}  папка        длит., с  к/с    размер      кодек")
    for r in rows:
        print(f"{r['clip_id']:<{w}}  {r['folder'][:11]:<11}  {r['duration_s']:>8}  {r['fps']:>5}  "
              f"{r['width']}×{r['height']:<5}  {r['codec']}")
    print(f"\nВсего {len(rows)} роликов, {total / 60:.1f} мин. Опись: {out}")
    if args.labels_template:
        lt = Path(args.labels_template)
        lt.parent.mkdir(parents=True, exist_ok=True)
        with open(lt, "w", encoding="utf-8", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=LABEL_FIELDS)
            wr.writeheader()
            for r in rows:
                row = {"clip_id": r["clip_id"], "comment": r["folder"]}
                path = (r["folder"] + "/" + r["file"]).lower()
                if args.guess_labels:
                    label = "POSITIVE" if ("pos" in path or "позит" in path) else (
                        "NEGATIVE" if ("neg" in path or "негат" in path) else "")
                    if label:
                        row.update(label=label, start_sec=0, end_sec=r["duration_s"],
                                   comment=f"{r['folder']}: весь клип, уточнить границы")
                wr.writerow(row)
        print(f"Заготовка разметки: {lt} (заполните label, start_sec, end_sec, по строке на событие)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
