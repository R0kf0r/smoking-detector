"""Выбор промпта VLM на эталоне: какой вариант лучше отличает позитивные ролики от негативных.

    python -m smokedet.vlm_probe --run-dir results/reference_novlm --gt labels/reference.csv --apply

Берём кадры из середины пауз у рта найденных циклов (по несколько на ролик), задаём модели каждый
вариант промпта из verify.PROMPTS и считаем по роликам долю ответов «сигарета/вейп» и долю вето.
Оценка варианта = (доля «сигарета» в позитивных − в негативных) + 0,5·(доля вето в негативных −
в позитивных). --apply записывает лучший вариант в configs/default.yaml (vlm.prompt).
Наружу — только числа и короткие ответы модели; кадры никуда не сохраняются.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .calibrate import apply_to_config
from .config import DEFAULT_CONFIG, load_config
from .evaluate import load_gt
from .verify import PROMPTS, SMOKE, VETO, Request, VLMJudge, crop_box, extract_crops
from .video import list_videos

VETO_ALL = VETO | {"hand"}


def sample(run_dir: Path, gts, videos: dict[str, Path], per_clip: int) -> list[dict]:
    """По ролику: до per_clip середин пауз у рта, равномерно по найденным циклам."""
    label_of = {}
    for g in gts:
        if g.label in ("POSITIVE", "NEGATIVE"):
            label_of.setdefault(g.clip_id, g.label)
    out = []
    for dbg_path in sorted((run_dir / "debug").glob("*.json")):
        dbg = json.loads(dbg_path.read_text(encoding="utf-8"))
        clip = dbg["clip_id"]
        if clip not in label_of or not dbg.get("cycles") or dbg.get("file") not in videos:
            continue
        sig_path = run_dir / "signals" / f"{clip}.csv"
        if not sig_path.is_file():
            continue
        sig = pd.read_csv(sig_path)
        cyc = dbg["cycles"]
        idx = np.unique(np.linspace(0, len(cyc) - 1, min(per_clip, len(cyc))).round().astype(int))
        reqs = []
        w, h = int(dbg["video"]["width"]), int(dbg["video"]["height"])
        for i in idx:
            c = cyc[int(i)]
            df = sig[sig.track_id == c["track_id"]].reset_index(drop=True)
            if df.empty:
                continue
            row = df.iloc[int((df.t - c["t_peak"]).abs().idxmin())]
            reqs.append(Request(int(c["track_id"]), float(c["t_peak"]), crop_box(row, w, h)))
        if reqs:
            out.append({"clip_id": clip, "label": label_of[clip], "video": videos[dbg["file"]], "reqs": reqs})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m smokedet.vlm_probe", description=__doc__.split("\n")[0])
    ap.add_argument("--run-dir", default="results/reference_novlm", help="прогон эталона (debug/ и signals/)")
    ap.add_argument("--gt", default="labels/reference.csv")
    ap.add_argument("--input", default="private/reference", help="папка с роликами эталона")
    ap.add_argument("--per-clip", type=int, default=5)
    ap.add_argument("--prompts", default="list_neutral,describe,yesno", help="варианты через запятую")
    ap.add_argument("--config")
    ap.add_argument("--apply", action="store_true", help="записать лучший вариант в конфиг")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    videos = {v.name: v for v in list_videos(args.input)}
    clips = sample(Path(args.run_dir), load_gt(args.gt), videos, args.per_clip)
    n_img = sum(len(c["reqs"]) for c in clips)
    print(f"Роликов с циклами: {len(clips)} (позитивных {sum(c['label'] == 'POSITIVE' for c in clips)}), "
          f"кадров: {n_img}")
    t0 = time.time()
    for c in clips:
        c["images"] = [im for _, im in extract_crops(c["video"], c["reqs"], int(cfg["vlm"]["crop_size"]))]
    print(f"Кропы готовы за {time.time() - t0:.0f} с; загружаю VLM")
    judge = VLMJudge(cfg)

    names = [p.strip() for p in args.prompts.split(",") if p.strip() in PROMPTS]
    rows, summary = [], []
    for name in names:
        spec = PROMPTS[name]
        t1 = time.time()
        rates = {"POSITIVE": {"smoke": [], "veto": []}, "NEGATIVE": {"smoke": [], "veto": []}}
        for c in clips:
            labels, raw = judge.ask(c["images"], spec)
            for r, lab, txt in zip(c["reqs"], labels, raw):
                rows.append({"prompt": name, "clip_id": c["clip_id"], "label": c["label"], "track_id": r.track_id,
                             "t": round(r.t, 2), "answer": lab, "raw": txt})
            if labels:
                rates[c["label"]]["smoke"].append(np.mean([x in SMOKE for x in labels]))
                rates[c["label"]]["veto"].append(np.mean([x in VETO_ALL for x in labels]))
        sec = time.time() - t1

        def m(lbl, key):
            v = rates[lbl][key]
            return float(np.mean(v)) if v else 0.0
        score = (m("POSITIVE", "smoke") - m("NEGATIVE", "smoke")) + 0.5 * (m("NEGATIVE", "veto") - m("POSITIVE", "veto"))
        summary.append({"prompt": name, "score": round(score, 3),
                        "smoke_pos": round(m("POSITIVE", "smoke"), 3), "smoke_neg": round(m("NEGATIVE", "smoke"), 3),
                        "veto_pos": round(m("POSITIVE", "veto"), 3), "veto_neg": round(m("NEGATIVE", "veto"), 3),
                        "sec_per_image": round(sec / max(1, n_img), 2)})
        print(f"  {name}: оценка {score:.3f}, {sec:.0f} с")

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "vlm_probe.csv", "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["prompt"])
        wr.writeheader()
        wr.writerows(rows)
    best = max(summary, key=lambda s: s["score"]) if summary else None
    md = ["# Выбор промпта VLM", "",
          "Доли ответов по роликам (среднее по роликам): «сигарета/вейп» и вето (предмет или пустая рука).", "",
          "| Промпт | Оценка | Сигарета в позитивных | Сигарета в негативных | Вето в позитивных | Вето в негативных | с/кадр |",
          "|---|---|---|---|---|---|---|"]
    for s_ in summary:
        md.append(f"| {s_['prompt']} | {s_['score']} | {s_['smoke_pos']} | {s_['smoke_neg']} | {s_['veto_pos']} | "
                  f"{s_['veto_neg']} | {s_['sec_per_image']} |")
    if best:
        md += ["", f"Лучший вариант: **{best['prompt']}**."]
    (out_dir / "vlm_probe.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    if args.apply and best:
        path = Path(args.config) if args.config else DEFAULT_CONFIG
        apply_to_config(path, {"vlm.prompt": best["prompt"]})
        print(f"Записано в {path}: vlm.prompt = {best['prompt']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
