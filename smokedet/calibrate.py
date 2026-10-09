"""Подбор порогов на эталоне по кэшу, без повторного прогона моделей (минуты, а не часы).

    python -m smokedet.calibrate --run-dir results/reference --gt labels/reference.csv [--clip-level] [--apply]

Перебор:
  * автомат циклов: порог входа × зазор гистерезиса (выход = вход + зазор) × относительный выход;
  * проверка: min_votes VLM × вето «пустая рука» (если в кэше есть ответы VLM) × вето «обе руки у лица».
Честная оценка — leave-one-clip-out: пороги подбираются на всех клипах, кроме одного,
и проверяются на нём; так видно, насколько результат переносится на новые ролики.
Рабочая точка — середина плато: максимум F1, сглаженной по соседним порогам, а не одиночный пик.
--apply записывает выбранные значения в configs/default.yaml (комментарии сохраняются).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG, deep_merge, load_config
from .evaluate import evaluate, load_durations, load_gt
from .roi import load_roi
from .run import compute_signals, detect_all, event_rows, finish, make_tracks
from .tracks import Detections

ENTERS = [round(float(x), 3) for x in np.arange(0.20, 0.3601, 0.02)]
GAPS = [0.04, 0.08, 0.12, 0.16]
RELS = [0.0, 0.08]


def load_contexts(run_dir: Path, cfg: dict, log=print) -> list[dict]:
    ctxs = []
    for dbg_path in sorted((run_dir / "debug").glob("*.json")):
        dbg = json.loads(dbg_path.read_text(encoding="utf-8"))
        pose_cache = Path(dbg.get("pose_cache") or "")
        if not pose_cache.is_file():
            # кэш мог быть записан из другой копии проекта: ищем по имени в этом проекте
            alt = Path("results/cache/pose") / pose_cache.name
            if alt.is_file():
                pose_cache = alt
            else:
                log(f"  нет кэша позы для {dbg['clip_id']}: {pose_cache}")
                continue
        dets = Detections.load(pose_cache)
        tracks = make_tracks(dets, cfg)
        signals = compute_signals(tracks, cfg)
        ver = None
        if dbg.get("verify_cache"):
            vpath = Path(dbg["verify_cache"])
            if not vpath.is_file():
                vpath = Path("results/cache/verify") / vpath.name
            if vpath.is_file():
                ver = json.loads(vpath.read_text(encoding="utf-8"))
        w, h = int(dbg["video"]["width"]), int(dbg["video"]["height"])
        roi = load_roi(dbg["roi_file"], w, h) if dbg.get("roi_file") else None
        ctxs.append({"clip_id": dbg["clip_id"], "camera_id": dbg.get("camera_id", ""), "tracks": tracks,
                     "signals": signals, "ver": ver, "roi": roi})
    return ctxs


def counts_for(ctx: dict, res: dict, gts: list, clip_level: bool) -> tuple[int, int, int]:
    g = [x for x in gts if x.clip_id == ctx["clip_id"]]
    if clip_level:
        # грубая разметка «весь клип позитивный / негативный»: позитивный клип найден, если есть
        # хотя бы одно событие; каждое событие в негативном клипе — ложная тревога
        labels = {x.label for x in g}
        n = len(res["events"])
        if "POSITIVE" in labels:
            return (1, 0, 0) if n else (0, 0, 1)
        if "NEGATIVE" in labels:
            return (0, n, 0)
        return (0, 0, 0)
    # camera_id эталона и прогона могут называться по-разному — сравниваем по clip_id
    rows = event_rows(res["events"], "", ctx["clip_id"])
    m = evaluate(rows, g, None)
    return m["TP"], m["FP"], m["FN"]


def f1(tp: int, fp: int, fn: int) -> float:
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def _smooth(grid: np.ndarray) -> np.ndarray:
    """Среднее по окрестности 3×3 (плато важнее одиночного пика)."""
    out = np.zeros_like(grid)
    n_i, n_j = grid.shape
    for i in range(n_i):
        for j in range(n_j):
            out[i, j] = float(np.mean(grid[max(0, i - 1):i + 2, max(0, j - 1):j + 2]))
    return out


def apply_to_config(path: Path, values: dict) -> None:
    """Меняет значения в YAML построчно, не трогая комментарии и порядок."""
    text = path.read_text(encoding="utf-8")

    def fmt(v):
        if isinstance(v, bool):
            return "true" if v else "false"
        return f"{v:g}" if isinstance(v, float) else str(v)

    for key, val in values.items():
        section, name = key.split(".")
        pattern = re.compile(rf"(^{section}:\n(?:[ \t].*\n|\n)*?[ \t]+{name}:[ \t]*)([^\s#]+)", re.M)
        text, n = pattern.subn(lambda m: m.group(1) + fmt(val), text, count=1)
        if n != 1:
            raise RuntimeError(f"не нашёл {key} в {path}")
    path.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m smokedet.calibrate", description=__doc__.split("\n")[0])
    ap.add_argument("--run-dir", required=True, help="папка прогона run.py по эталону (с debug/ и timing.csv)")
    ap.add_argument("--gt", required=True, help="разметка эталона labels/reference.csv")
    ap.add_argument("--config", help="конфиг (по умолчанию configs/default.yaml)")
    ap.add_argument("--apply", action="store_true", help="записать выбранные значения в конфиг")
    ap.add_argument("--clip-level", action="store_true",
                    help="разметка на уровне клипов (позитивный клип найден / в негативном ложные тревоги)")
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir)
    cfg = load_config(args.config)
    gts_all = load_gt(args.gt)
    durations = load_durations(run_dir / "timing.csv") if (run_dir / "timing.csv").is_file() else {}
    ctxs = load_contexts(run_dir, cfg)
    if not ctxs:
        print("Нет клипов с кэшем: сначала запустите smokedet.run на эталоне")
        return 1
    clip_ids = [c["clip_id"] for c in ctxs]
    gts = [g for g in gts_all if g.clip_id in set(clip_ids)]
    n_pos = sum(1 for g in gts if g.label == "POSITIVE")
    print(f"Клипов: {len(ctxs)}, позитивных меток: {n_pos}")
    if not gts:
        print("В разметке нет строк для этих клипов (сравните clip_id)")
        return 1

    vlm_used = any((c["ver"] or {}).get("vlm_used") for c in ctxs)
    votes = [1, 2, 3] if vlm_used else [int(cfg["vlm"]["min_votes"])]
    hand_vetoes = [True, False] if vlm_used else [bool(cfg["vlm"].get("hand_veto", True))]
    bhs = [True, False]
    singles = [True, False]
    vapes = [True, False] if vlm_used else [bool(cfg["vlm"].get("vape_is_smoke", True))]
    cur = (float(cfg["cycles"]["enter_threshold"]),
           round(float(cfg["cycles"]["exit_threshold"]) - float(cfg["cycles"]["enter_threshold"]), 3),
           float(cfg["cycles"].get("relative_exit", 0.0)), int(cfg["vlm"]["min_votes"]),
           bool(cfg["vlm"].get("hand_veto", True)), bool(cfg.get("kinematics", {}).get("both_hands_veto", True)),
           bool(cfg["events"].get("allow_single_with_object", True)), bool(cfg["vlm"].get("vape_is_smoke", True)))

    cyc_combos = [(e, g, r) for r in RELS for e in ENTERS for g in GAPS]
    if cur[:3] not in cyc_combos:
        cyc_combos.append(cur[:3])
    ver_combos = [(v, hv, b, sg, vp) for v in votes for hv in hand_vetoes for b in bhs for sg in singles
                  for vp in vapes]
    if cur[3:] not in ver_combos:
        ver_combos.append(cur[3:])

    counts: dict[tuple, dict[str, tuple]] = {}
    for k, (e, g, r) in enumerate(cyc_combos, 1):
        c_cfg = deep_merge(cfg, {"cycles": {"enter_threshold": e, "exit_threshold": round(e + g, 3),
                                            "relative_exit": r}})
        detected = {c["clip_id"]: detect_all(c["signals"], c_cfg) for c in ctxs}
        for v, hv, b, sg, vp in ver_combos:
            v_cfg = deep_merge(c_cfg, {"vlm": {"min_votes": v, "hand_veto": hv, "vape_is_smoke": vp},
                                       "kinematics": {"both_hands_veto": b},
                                       "events": {"allow_single_with_object": sg}})
            per_clip = {}
            for c in ctxs:
                cyc, rej = detected[c["clip_id"]]
                fresh = {tid: [dataclasses.replace(x, votes={}) for x in lst] for tid, lst in cyc.items()}
                res = finish(c["tracks"], fresh, rej, c["ver"], v_cfg, c["roi"])
                per_clip[c["clip_id"]] = counts_for(c, res, gts, args.clip_level)
            counts[(e, g, r, v, hv, b, sg, vp)] = per_clip
        if k % 20 == 0:
            print(f"  {k}/{len(cyc_combos)}")

    def pooled(combo, exclude=None):
        tp = fp = fn = 0
        for cid, (a, b_, c_) in counts[combo].items():
            if cid == exclude:
                continue
            tp, fp, fn = tp + a, fp + b_, fn + c_
        return tp, fp, fn

    # сглаживание F1 по сетке (вход × зазор) при остальных параметрах фиксированных
    smooth_of: dict[tuple, float] = {}
    for r in RELS:
        for vc in ver_combos:
            grid = np.array([[f1(*pooled((e, g, r, *vc))) if (e, g, r, *vc) in counts else 0.0
                              for g in GAPS] for e in ENTERS])
            sm = _smooth(grid)
            for i, e in enumerate(ENTERS):
                for j, g in enumerate(GAPS):
                    smooth_of[(e, g, r, *vc)] = float(sm[i, j])

    def rank(combo, exclude=None):
        tp, fp, fn = pooled(combo, exclude)
        closeness = (-abs(combo[0] - cur[0]) - abs(combo[1] - cur[1])
                     - 0.5 * sum(combo[k] != cur[k] for k in (2, 3, 4, 5, 6, 7)))
        main_score = smooth_of.get(combo, f1(tp, fp, fn)) if exclude is None else f1(tp, fp, fn)
        return (main_score, f1(tp, fp, fn), -fp, closeness)

    keys = list(counts)
    best = max(keys, key=rank)

    def yn(x):
        return "да" if x else "нет"

    loco = {"TP": 0, "FP": 0, "FN": 0}
    loco_rows = []
    for cid in clip_ids:
        ch = max(keys, key=lambda c: rank(c, exclude=cid))
        a, b_, c_ = counts[ch][cid]
        loco["TP"] += a
        loco["FP"] += b_
        loco["FN"] += c_
        loco_rows.append(f"| {cid} | {ch[0]} | {round(ch[0] + ch[1], 3)} | {ch[2]} | {ch[3]} | {yn(ch[4])} | "
                         f"{yn(ch[5])} | {yn(ch[6])} | {yn(ch[7])} | {a} | {b_} | {c_} |")

    rows = []
    for combo in counts:
        tp, fp, fn = pooled(combo)
        e, g, r, v, hv, b, sg, vp = combo
        rows.append({"enter_threshold": e, "exit_threshold": round(e + g, 3), "gap": g, "relative_exit": r,
                     "min_votes": v, "hand_veto": hv, "both_hands_veto": b, "single_with_object": sg,
                     "vape_is_smoke": vp, "TP": tp, "FP": fp, "FN": fn,
                     "precision": round(tp / (tp + fp), 4) if tp + fp else None,
                     "recall": round(tp / (tp + fn), 4) if tp + fn else None,
                     "f1": round(f1(tp, fp, fn), 4), "f1_smooth": round(smooth_of.get(combo, f1(tp, fp, fn)), 4)})
    grid_df = pd.DataFrame(rows).sort_values(["both_hands_veto", "hand_veto", "min_votes", "relative_exit",
                                              "enter_threshold", "gap"])
    out_dir = run_dir / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    grid_df.to_csv(out_dir / "grid.csv", index=False)

    neg_clips = {g.clip_id for g in gts if g.label == "NEGATIVE"} - {g.clip_id for g in gts if g.label == "POSITIVE"}
    neg_hours = sum(durations.get(c, 0.0) for c in neg_clips) / 3600.0
    per_clip = []
    for c in ctxs:
        a, b_, c_ = counts[cur][c["clip_id"]]
        bb = counts[best][c["clip_id"]]
        kind = "позитив" if any(g.clip_id == c["clip_id"] and g.label == "POSITIVE" for g in gts) else "негатив"
        per_clip.append(f"| {c['clip_id']} | {kind} | {a} | {b_} | {c_} | {bb[0]} | {bb[1]} | {bb[2]} |")
    mode = ("на уровне клипов: позитивный клип найден, если в нём есть событие; каждое событие в негативном "
            "клипе — ложная тревога" if args.clip_level else "по событиям, протокол SPEC §6")
    cur_c, best_c = pooled(cur), pooled(best)

    def line(name, combo, c3):
        return (f"| {name} | {combo[0]} | {round(combo[0] + combo[1], 3)} | {combo[2]} | {combo[3]} | "
                f"{yn(combo[4])} | {yn(combo[5])} | {yn(combo[6])} | {yn(combo[7])} | {c3[0]} | {c3[1]} | "
                f"{c3[2]} | {f1(*c3):.3f} |")

    fph = (f"{cur_c[1] / neg_hours:.1f} (текущий конфиг), {best_c[1] / neg_hours:.1f} (середина плато)"
           if neg_hours and args.clip_level else "—")
    md = [
        "# Подбор порогов на эталоне", "",
        f"Клипов: {len(ctxs)}; позитивных меток: {n_pos}; VLM в кэше: {'да' if vlm_used else 'нет'}; "
        f"оценка {mode}.", "",
        f"Ложных тревог в час на негативных клипах: {fph}.", "",
        "| Вариант | вход | выход | отн. выход | min_votes | вето «пустая рука» | вето «обе руки» | 1 цикл + признак "
        "| вейп = признак | TP | FP | FN | F1 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        line("текущий конфиг", cur, cur_c),
        line("середина плато", best, best_c),
        f"| leave-one-clip-out (честная оценка) | — | — | — | — | — | — | — | — | {loco['TP']} | {loco['FP']} | "
        f"{loco['FN']} | {f1(loco['TP'], loco['FP'], loco['FN']):.3f} |",
        "", "## Выбор параметров без одного клипа", "",
        "| Отложенный клип | вход | выход | отн. выход | min_votes | «пустая рука» | «обе руки» | 1 цикл + признак | "
        "вейп | TP | FP | FN |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
        *loco_rows, "",
        "## По клипам", "",
        "| Клип | Тип | TP (тек.) | FP (тек.) | FN (тек.) | TP (плато) | FP (плато) | FN (плато) |",
        "|---|---|---|---|---|---|---|---|", *per_clip, "",
        "Полная сетка — `grid.csv` (для графика «порог → precision, recall, F1»).",
    ]
    (out_dir / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    if args.apply:
        path = Path(args.config) if args.config else DEFAULT_CONFIG
        vals = {"cycles.enter_threshold": float(best[0]), "cycles.exit_threshold": float(round(best[0] + best[1], 3)),
                "cycles.relative_exit": float(best[2]), "kinematics.both_hands_veto": bool(best[5]),
                "events.allow_single_with_object": bool(best[6])}
        if vlm_used:
            vals["vlm.min_votes"] = int(best[3])
            vals["vlm.hand_veto"] = bool(best[4])
            vals["vlm.vape_is_smoke"] = bool(best[7])
        apply_to_config(path, vals)
        print(f"\nЗаписано в {path}: {vals}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
