"""Видео с отметками для одного ролика прогона: берём событие с наибольшей уверенностью.

    python -m smokedet.demo --run-dir results/hidden --input private/hidden [флаги smokedet.run]

Поза и ответы VLM берутся из кэша прогона, остаётся только рендер. Видео пишется в
private/renders/<имя прогона>/<clip_id>.mp4, пути к нему и к исходному ролику — в <run-dir>/demo.txt
(их подхватывает scripts/pack.sh). Флаги прогона (--preset, --set, --roi) нужно передать те же,
иначе кэш не совпадёт.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from . import run as run_mod
from .video import list_videos


def pick_event(run_dir: Path) -> dict | None:
    """Самое уверенное событие; при равной уверенности — из ролика покороче (быстрее рендер)."""
    path = run_dir / "events.json"
    events = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
    if not events:
        return None
    dur: dict[str, float] = {}
    tpath = run_dir / "timing.csv"
    if tpath.is_file():
        with open(tpath, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                try:
                    dur[r["clip_id"]] = float(r.get("duration_s") or 0)
                except ValueError:
                    pass
    return max(events, key=lambda e: (float(e["confidence"]), -dur.get(e["clip_id"], 1e9)))


def find_video(src: Path, clip_id: str, run_dir: Path) -> Path | None:
    if src.is_file():
        return src
    name = None
    tpath = run_dir / "timing.csv"
    if tpath.is_file():
        with open(tpath, encoding="utf-8") as f:
            name = next((r["file"] for r in csv.DictReader(f) if r["clip_id"] == clip_id), None)
    for v in list_videos(src):
        if (name is None or v.name == name) and clip_id in (v.stem, f"{v.parent.name}__{v.stem}"):
            return v
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m smokedet.demo", description=__doc__.split("\n")[0])
    ap.add_argument("--run-dir", default="results/hidden")
    ap.add_argument("--input", required=True, help="папка с роликами прогона")
    ap.add_argument("--render-dir", help="куда писать видео (по умолчанию private/renders/<имя прогона>)")
    args, passthrough = ap.parse_known_args(argv)

    run_dir = Path(args.run_dir)
    ev = pick_event(run_dir)
    if ev is None:
        print("Событий нет — видео с отметками не делаем")
        return 0
    clip = ev["clip_id"]
    video = find_video(Path(args.input), clip, run_dir)
    if video is None:
        print(f"Не нашёл исходный ролик для {clip} в {args.input}")
        return 1
    render_dir = Path(args.render_dir or Path("private") / "renders" / run_dir.name)
    print(f"Видео с отметками: {clip}, событие {ev['start_sec']}–{ev['end_sec']} с, уверенность {ev['confidence']}")
    rc = run_mod.main(["--input", str(video), "--clip-id", clip, "--camera-id", ev["camera_id"],
                       "--out-dir", str(run_dir / "demo"), "--render-dir", str(render_dir), *passthrough])
    out = render_dir / f"{clip}.mp4"
    if not out.is_file():
        print("Рендер не получился — см. сообщения выше")
        return rc or 1
    (run_dir / "demo.txt").write_text(f"{out}\n{video}\n", encoding="utf-8")
    print(f"Готово: {out} (исходный ролик: {video})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
