"""Одна команда запуска: ролики → события (JSON и CSV), журнал времени, manifest.json,
отладочные числа и, по желанию, видео с отметками.

Примеры:
  python -m smokedet.run --input private/reference --out-dir results/reference \\
      --render-dir private/renders/reference --gt labels/reference.csv
  python -m smokedet.run --input private/hidden --out-dir results/hidden --camera-id cam01
  python -m smokedet.run --input clip.mp4 --out-dir results/one --camera-id cam01 --clip-id clip01

Выход в --out-dir:
  events.json, events.csv     события строго в формате SPEC §2
  timing.csv                  журнал времени обработки по клипам
  manifest.json               пороги, правила, модели, версии, промпт, оборудование
  debug/<clip_id>.json        циклы, отклонённые паузы, голоса VLM, отброшенные события
  signals/<clip_id>.csv       сигнал кисть — рот по кадрам (только числа)
  metrics.md, metrics.json    если передан --gt
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("YOLO_OFFLINE", "1")      # без телеметрии Ultralytics

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import __version__
from .config import PROJECT_ROOT, config_hash, deep_merge, load_config
from .cycles import detect_cycles
from .events import Event, build_events
from .pose import resolve_device, resolve_weights, run_pose_tracking
from .roi import find_roi_file, inside_fraction, load_roi
from .signal import track_signal
from .tracks import Detections, Track, build_tracks, stitch_ids
from .verify import VLM_LABELS, apply_votes, prompt_of, run_verification
from .video import list_videos, probe

EVENT_FIELDS = ["camera_id", "clip_id", "event_id", "start_sec", "end_sec", "confidence", "label",
                "person_track_id", "peak_sec", "x1", "y1", "x2", "y2"]
TIMING_FIELDS = ["clip_id", "camera_id", "file", "status", "duration_s", "fps", "width", "height",
                 "frames_processed", "pose_s", "verify_s", "rules_s", "processing_s", "sec_per_min",
                 "render_s", "device", "pose_model", "vlm_used", "n_tracks", "n_cycles", "n_events",
                 "from_cache", "error"]


class Log:
    def __init__(self, path: Path | None = None):
        self.f = None
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.f = open(path, "a", encoding="utf-8")

    def __call__(self, msg: str) -> None:
        line = f"[{datetime.now():%H:%M:%S}] {msg}"
        print(line, flush=True)
        if self.f:
            self.f.write(line + "\n")
            self.f.flush()


# ---------------------------------------------------------------- анализ (общий с calibrate.py)

def pose_tag(cfg: dict) -> str:
    return config_hash({"video": cfg["video"], "pose": cfg["pose"], "tracker": cfg["tracker"]})[:8]


VLM_REQUEST_KEYS = ("enabled", "backend", "model", "prompt", "scan_threshold", "sample_hz",
                    "max_frames_per_segment", "max_requests_per_clip", "crop_size")


def verify_tag(cfg: dict) -> str:
    """Имя кэша проверки: только то, что меняет запросы к моделям и их ответы (не пороги решения)."""
    vlm = {k: cfg["vlm"].get(k) for k in VLM_REQUEST_KEYS}
    return config_hash({"pose": pose_tag(cfg), "geometry": cfg["geometry"], "signal": cfg["signal"],
                        "vlm": vlm, "object_veto": cfg["object_veto"],
                        "kpt": cfg["pose"]["kpt_conf_min"]})[:8]


def make_tracks(dets: Detections, cfg: dict) -> dict[int, Track]:
    tr = cfg["tracker"]
    mapping = stitch_ids(dets, float(tr["stitch_max_gap_s"]), float(tr["stitch_max_center_shift"]))
    return build_tracks(dets, mapping)


def compute_signals(tracks: dict[int, Track], cfg: dict) -> dict[int, pd.DataFrame]:
    """Сигнал считаем для треков, где человек хотя бы раз достаточно крупный."""
    min_h = float(cfg["geometry"]["min_person_height_px"])
    out = {}
    for tid, tr in tracks.items():
        if len(tr.t) < 3 or float(np.nanmax(tr.heights())) < 0.9 * min_h:
            continue
        out[tid] = track_signal(tr, cfg)
    return out


def finalize_events(events: list[Event], tracks: dict[int, Track], cfg: dict,
                    roi_mask: np.ndarray | None) -> tuple[list[Event], list[tuple[Event, str]]]:
    """Рамка в момент peak_sec, фильтры видимости (рост ≥ 80 px), зоны интереса и уверенности."""
    min_h = float(cfg["geometry"]["min_person_height_px"])
    min_conf = float(cfg["events"].get("output_min_confidence", 0.0))
    min_inside = float(cfg.get("roi", {}).get("min_inside_frac", 0.5))
    kept, dropped = [], []
    for ev in events:
        tr = tracks[ev.track_id]
        sel = (tr.t >= ev.start) & (tr.t <= ev.end)
        boxes = tr.box[sel] if sel.any() else tr.box[[int(np.argmin(np.abs(tr.t - ev.peak)))]]
        ev.median_height = float(np.median(boxes[:, 3] - boxes[:, 1]))
        ev.box = tuple(float(v) for v in tr.box_at(ev.peak))
        reason = ""
        if ev.median_height < min_h:
            reason = f"рост {ev.median_height:.0f} px меньше {min_h:.0f} px"
        elif roi_mask is not None and inside_fraction(roi_mask, boxes) < min_inside:
            reason = "вне зоны интереса"
        elif ev.confidence < min_conf:
            reason = f"уверенность {ev.confidence:.2f} ниже порога {min_conf:.2f}"
        if reason:
            dropped.append((ev, reason))
        else:
            kept.append(ev)
    return kept, dropped


def detect_all(signals: dict[int, pd.DataFrame], cfg: dict) -> tuple[dict, list]:
    """Циклы всех треков (зависят только от сигнала и порогов автомата)."""
    cycles_by_track, rejected = {}, []
    for tid, sig in signals.items():
        cyc, rej = detect_cycles(sig["t"].to_numpy(), sig["d"].to_numpy(), cfg, tid,
                                 sig["dl"].to_numpy(), sig["dr"].to_numpy())
        cycles_by_track[tid] = cyc
        rejected += rej
    return cycles_by_track, rejected


def finish(tracks: dict[int, Track], cycles_by_track: dict, rejected: list, verification: dict | None,
           cfg: dict, roi_mask: np.ndarray | None) -> dict:
    """Голоса проверки → правила событий → фильтры. Циклы изменяются на месте (вердикты)."""
    apply_votes(cycles_by_track, verification, cfg)
    events = build_events(cycles_by_track, cfg)
    kept, dropped = finalize_events(events, tracks, cfg, roi_mask)
    return {"cycles": cycles_by_track, "rejected": rejected, "events": kept, "dropped": dropped}


def analyze(tracks: dict[int, Track], signals: dict[int, pd.DataFrame], verification: dict | None,
            cfg: dict, roi_mask: np.ndarray | None) -> dict:
    cycles_by_track, rejected = detect_all(signals, cfg)
    return finish(tracks, cycles_by_track, rejected, verification, cfg, roi_mask)


def event_rows(events: list[Event], camera_id: str, clip_id: str) -> list[dict]:
    return [ev.as_output(camera_id, clip_id, f"{clip_id}_e{k:03d}") for k, ev in enumerate(events, 1)]


# ---------------------------------------------------------------- один клип

def process_clip(video: Path, clip_id: str, camera_id: str, cfg: dict, opts: argparse.Namespace,
                 log: Log) -> dict:
    t_wall = time.time()
    info = probe(video)
    w, h = info["width"], info["height"]
    cache_dir = Path(opts.cache_dir)
    pose_path = cache_dir / "pose" / f"{clip_id}__{pose_tag(cfg)}.npz"
    ver_path = cache_dir / "verify" / f"{clip_id}__{verify_tag(cfg)}.json"

    t0 = time.time()
    dets = run_pose_tracking(video, cfg, pose_path, log)
    pose_wall = time.time() - t0
    from_cache = pose_wall < 0.5 * float(dets.meta.get("seconds", 0) or 0)
    w = w or int(dets.meta.get("width", 0))
    h = h or int(dets.meta.get("height", 0))

    tracks = make_tracks(dets, cfg)
    signals = compute_signals(tracks, cfg)
    device = dets.meta.get("device") or resolve_device(cfg["pose"].get("device", "auto"))

    verification = None
    if opts.verify:
        verification = run_verification(video, signals, cfg, w, h, device, ver_path, log)

    roi_file = find_roi_file(opts.roi, clip_id, camera_id)
    roi_mask = load_roi(roi_file, w, h) if roi_file else None
    if opts.roi and roi_file is None:
        log(f"  зона интереса для {clip_id} не найдена — берём весь кадр")

    t2 = time.time()
    res = analyze(tracks, signals, verification, cfg, roi_mask)
    rows = event_rows(res["events"], camera_id, clip_id)
    rules_s = time.time() - t2

    out_dir = Path(opts.out_dir)
    # сигналы — только числа
    sig_frames = []
    for tid, sig in signals.items():
        df = sig.copy()
        df.insert(1, "track_id", tid)
        sig_frames.append(df)
    (out_dir / "signals").mkdir(parents=True, exist_ok=True)
    if sig_frames:
        pd.concat(sig_frames).round(4).to_csv(out_dir / "signals" / f"{clip_id}.csv", index=False)

    n_cycles = sum(len(v) for v in res["cycles"].values())
    vlm_counts: dict[str, int] = {}
    for it in (verification or {}).get("items", []):
        vlm_counts[it["vlm"]] = vlm_counts.get(it["vlm"], 0) + 1
    debug = {
        "clip_id": clip_id, "camera_id": camera_id, "file": video.name, "video": info,
        "pose_cache": str(pose_path), "verify_cache": str(ver_path) if opts.verify else None,
        "roi_file": str(roi_file) if roi_file else None,
        "pose_meta": {k: v for k, v in dets.meta.items() if k != "key"},
        "tracks": [{"track_id": tid, "source_ids": tr.source_ids, "t0": round(float(tr.t[0]), 3),
                    "t1": round(float(tr.t[-1]), 3), "n": int(len(tr.t)),
                    "median_height": round(float(np.median(tr.heights())), 1),
                    "has_signal": tid in signals} for tid, tr in tracks.items()],
        "cycles": [c.as_dict() for cs in res["cycles"].values() for c in cs],
        "rejected": [r.as_dict() for r in res["rejected"]],
        "verification": {"vlm_used": (verification or {}).get("vlm_used", False),
                         "object_veto_used": (verification or {}).get("object_veto_used", False),
                         "n_crops": len((verification or {}).get("items", [])), "labels": vlm_counts,
                         "seconds": (verification or {}).get("seconds", 0.0)},
        "events": [dict(row, n_cycles=len(ev.cycles), has_object=ev.has_object,
                        median_height=round(ev.median_height, 1),
                        cycles=[c.as_dict() for c in ev.cycles]) for row, ev in zip(rows, res["events"])],
        "dropped_events": [{"track_id": ev.track_id, "start": round(ev.start, 3), "end": round(ev.end, 3),
                            "confidence": round(ev.confidence, 3), "reason": why} for ev, why in res["dropped"]],
    }
    (out_dir / "debug").mkdir(parents=True, exist_ok=True)
    (out_dir / "debug" / f"{clip_id}.json").write_text(json.dumps(debug, ensure_ascii=False, indent=1),
                                                       encoding="utf-8")

    render_s = 0.0
    if opts.render_dir and (not opts.render_only_events or rows):
        from .render import build_plan, render_video

        plan = build_plan(tracks, signals, res["cycles"], res["events"], cfg,
                          info["duration_s"] or float(dets.frame_t[-1] if len(dets.frame_t) else 0), roi_mask)
        out_video = Path(opts.render_dir) / f"{clip_id}.mp4"
        log(f"  рендер → {out_video}")
        rr = render_video(video, out_video, plan, cfg, log)
        render_s = rr["seconds"]
        log(f"  рендер готов: {rr['frames']} кадров за {rr['seconds']:.0f} с ({rr['encoder']})")

    pose_s = float(dets.meta.get("seconds", pose_wall))
    verify_s = float((verification or {}).get("seconds", 0.0))
    processing = pose_s + verify_s + rules_s
    dur = float(info["duration_s"] or 0.0)
    timing = {
        "clip_id": clip_id, "camera_id": camera_id, "file": video.name, "status": "ok",
        "duration_s": round(dur, 3), "fps": info["fps"], "width": w, "height": h,
        "frames_processed": int(dets.meta.get("frames", len(dets.frame_t))),
        "pose_s": round(pose_s, 2), "verify_s": round(verify_s, 2), "rules_s": round(rules_s, 3),
        "processing_s": round(processing, 2),
        "sec_per_min": round(processing / dur * 60, 2) if dur else "",
        "render_s": round(render_s, 2), "device": device, "pose_model": cfg["pose"]["model"],
        "vlm_used": (verification or {}).get("vlm_used", False), "n_tracks": len(tracks),
        "n_cycles": n_cycles, "n_events": len(rows), "from_cache": from_cache, "error": "",
    }
    log(f"  треков {len(tracks)}, циклов {n_cycles}, событий {len(rows)}, "
        f"обработка {processing:.1f} с на {dur:.1f} с видео (стена {time.time() - t_wall:.1f} с)")
    return {"rows": rows, "timing": timing, "debug": debug}


# ---------------------------------------------------------------- запись результатов

def write_events(out_dir: Path, rows: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "events.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(out_dir / "events.csv", "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=EVENT_FIELDS)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r[k] for k in EVENT_FIELDS})


def write_timing(out_dir: Path, timing: list[dict]) -> None:
    with open(out_dir / "timing.csv", "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=TIMING_FIELDS)
        wr.writeheader()
        for r in timing:
            wr.writerow({k: r.get(k, "") for k in TIMING_FIELDS})


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def _versions() -> dict:
    from importlib import metadata

    out = {"python": platform.python_version()}
    for pkg in ("ultralytics", "torch", "torchvision", "numpy", "opencv-python", "av", "lap", "pandas",
                "pillow", "pyyaml", "mlx", "mlx-vlm", "transformers"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    ff = _run(["ffmpeg", "-version"]).splitlines()
    out["ffmpeg"] = ff[0] if ff else ""
    return out


def _hardware(device: str) -> dict:
    hw = {"platform": platform.platform(), "machine": platform.machine(), "device": device}
    if sys.platform == "darwin":
        hw["chip"] = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        mem = _run(["sysctl", "-n", "hw.memsize"])
        hw["memory_gb"] = round(int(mem) / 2**30, 1) if mem.isdigit() else None
        hw["macos"] = platform.mac_ver()[0]
    else:
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    hw["chip"] = line.split(":", 1)[1].strip()
                    break
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal"):
                    hw["memory_gb"] = round(int(line.split()[1]) / 2**20, 1)
        except OSError:
            pass
        hw["cpu_count"] = os.cpu_count()
    return hw


def _sha256(path: str) -> str | None:
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git() -> dict:
    root = str(PROJECT_ROOT)
    commit = _run(["git", "-C", root, "rev-parse", "HEAD"])
    if not commit:
        return {}
    # dirty — изменены отслеживаемые файлы (код, конфиг); новые результаты прогона не в счёт
    return {"commit": commit, "dirty": bool(_run(["git", "-C", root, "status", "--porcelain", "-uno"])),
            "tag": _run(["git", "-C", root, "describe", "--tags", "--always"])}


def build_manifest(cfg: dict, args: argparse.Namespace, clips: list[dict], timing: list[dict],
                   device: str) -> dict:
    clean_cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    c, e = cfg["cycles"], cfg["events"]
    smoke_words = "cigarette или vape" if cfg["vlm"].get("vape_is_smoke", True) else "cigarette"
    n_votes = int(cfg["vlm"]["min_votes"])
    hand_veto = ", пустую руку у рта (касание лица)" if cfg["vlm"].get("hand_veto", True) else ""
    both_veto = (", либо обе кисти у лица больше половины паузы без прямого признака"
                 if cfg.get("kinematics", {}).get("both_hands_veto", True) else "")
    pose_w = resolve_weights(cfg["pose"]["model"], download=False)
    obj_w = resolve_weights(cfg["object_veto"]["model"], download=False)
    ok = [t for t in timing if t.get("status") == "ok"]
    total_video = sum(float(t["duration_s"] or 0) for t in ok)
    total_proc = sum(float(t["processing_s"] or 0) for t in ok)
    return {
        "project": "smoking-detector", "version": __version__,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "command": " ".join([Path(sys.executable).name, "-m", "smokedet.run", *sys.argv[1:]]),
        "git": _git(),
        "config": {"path": cfg.get("_meta", {}).get("config_path"), "preset": cfg.get("_meta", {}).get("preset"),
                   "overrides": args.set, "hash": config_hash(cfg), "values": clean_cfg},
        "label": "smoking_like",
        "confidence_threshold": float(e.get("output_min_confidence", 0.0)),
        "decision_rules": {
            "signal": "d(t) = min по двум рукам ‖кисть − рот‖ / длина торса; рот = нос + "
                      f"{cfg['geometry']['mouth_offset']}·(нос − середина глаз); кисть = запястье + "
                      f"{cfg['geometry']['hand_extension']}·(запястье − локоть); заполнение провалов до "
                      f"{cfg['signal']['max_gap_s']} с, центрированная медиана {cfg['signal']['smooth_window_s']} с",
            "cycle": f"рука у рта: вход d < {c['enter_threshold']}, выход d > {c['exit_threshold']} (гистерезис)"
                     + (f" или подъём d над минимумом паузы на {c['relative_exit']} и выше порога входа дольше "
                        f"{c.get('relative_exit_hold_s', 0.25)} с (затяжки, между которыми рука остаётся у лица)"
                        if float(c.get("relative_exit", 0) or 0) > 0 else "") + "; "
                     f"пауза у рта {c['min_pause_s']}–{c['max_pause_s']} с (правила: 0,5–5 с с допуском 0,2 с на шум); "
                     "начало цикла — начало подхода руки, конец — конец отвода",
            "event": f"не меньше {e['min_cycles']} циклов в окне {e['window_s']} с"
                     + (f" или 1 цикл + прямой признак (VLM видит {smoke_words} не меньше чем в {n_votes} "
                        f"{'кадре' if n_votes % 10 == 1 and n_votes % 100 != 11 else 'кадрах'} паузы "
                        "и таких кадров больше, чем кадров с другим предметом)"
                        if e.get("allow_single_with_object", True) else ""),
            "merge": f"циклы одного person_track_id объединяются, если пауза между ними ≤ {e['merge_gap_s']} с; "
                     "события разных людей не объединяются",
            "veto": "цикл не засчитывается, если в кадрах паузы у рта детектор COCO или VLM видит телефон, бутылку, "
                    f"банку, чашку, еду, леденец, зубочистку или ручку{hand_veto}{both_veto}",
            "visibility": f"медианная высота человека за время события ≥ {cfg['geometry']['min_person_height_px']} px",
            "roi": "человек в зоне, если середина нижней стороны рамки внутри маски не меньше чем в "
                   f"{cfg.get('roi', {}).get('min_inside_frac', 0.5):.0%} кадров события; без маски — весь кадр",
            "peak_and_box": "peak_sec — середина паузы самого надёжного цикла (с прямым признаком, иначе с минимальным d); "
                            "рамка — детекция человека, ближайшая к peak_sec, в пикселях исходного кадра",
            "confidence": "0,45 / 0,60 / 0,70 / 0,80 при 1 / 2 / 3 / 4+ циклах, +0,20 за прямой признак, максимум 0,99",
            "time": "секунды от начала ролика по меткам времени кадров (PTS)",
        },
        "models": [
            {"name": "YOLO26-pose", "file": Path(pose_w).name, "sha256": _sha256(pose_w),
             "source": "Ultralytics, github.com/ultralytics/assets/releases", "license": "AGPL-3.0",
             "purpose": "люди, 17 ключевых точек COCO, трекинг BoT-SORT"},
            {"name": "YOLO26 (COCO)", "file": Path(obj_w).name, "sha256": _sha256(obj_w),
             "source": "Ultralytics, github.com/ultralytics/assets/releases", "license": "AGPL-3.0",
             "purpose": "вето по предметам у лица: cell phone, bottle, cup, wine glass",
             "enabled": bool(cfg["object_veto"].get("enabled", True)) and args.verify},
            {"name": "Qwen3-VL-4B-Instruct (4-bit MLX)", "file": cfg["vlm"]["model"],
             "source": "huggingface.co/" + cfg["vlm"]["model"], "license": "Apache-2.0",
             "purpose": "проверка кадров паузы у рта: что у рта", "runtime": "mlx-vlm, локально",
             "enabled": bool(cfg["vlm"].get("enabled", True)) and args.verify},
        ],
        "vlm": {"prompt_name": prompt_of(cfg)[0], "prompt": prompt_of(cfg)[1]["text"], "labels": VLM_LABELS,
                "temperature": 0.0,
                "max_tokens": prompt_of(cfg)[1]["max_tokens"], "crop_size": cfg["vlm"]["crop_size"],
                "used_in_run": any(t.get("vlm_used") in (True, "True") for t in ok)},
        "training": "дообучение не проводилось: все модели предобученные, параметры правил подобраны на эталонном наборе",
        "data_handling": "видео объекта обрабатывается локально на ноутбуке команды, облачные модели к нему не "
                         "подключались; отдельные кадры для отладки и разметки эталона просматривал ИИ-ассистент "
                         "(с письменного разрешения ментора 9.10.2026)",
        "data_sources": [
            "сценарные ролики организаторов (эталон, 8.10.2026) — для подбора порогов; не публикуются",
            "публичные данные — см. data/public/SOURCES.md",
        ],
        "fonts": [{"name": "Golos Text", "license": "SIL OFL 1.1", "files": ["static/GolosText-Regular.ttf",
                                                                             "static/GolosText-SemiBold.ttf"]}],
        "libraries": _versions(),
        "hardware": _hardware(device),
        "run": {"input": str(args.input), "clips": len(timing), "ok": len(ok),
                "errors": [t["clip_id"] for t in timing if t.get("status") != "ok"],
                "events": sum(len(cl["rows"]) for cl in clips),
                "video_seconds": round(total_video, 1), "processing_seconds": round(total_proc, 1),
                "sec_per_min": round(total_proc / total_video * 60, 2) if total_video else None},
    }


# ---------------------------------------------------------------- CLI

def _parse_set(items: list[str]) -> dict:
    out: dict = {}
    for it in items or []:
        key, _, raw = it.partition("=")
        if not key or not _:
            raise SystemExit(f"--set ожидает ключ=значение, получено: {it!r}")
        node = out
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return out


def _camera_map(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        sample = f.read(2048)
        f.seek(0)
        delim = ";" if sample.count(";") > sample.count(",") else ","
        return {r["clip_id"].strip(): r["camera_id"].strip() for r in csv.DictReader(f, delimiter=delim)
                if r.get("clip_id") and r.get("camera_id")}


def collect_clips(args: argparse.Namespace, log: Log) -> list[tuple[Path, str, str]]:
    src = Path(args.input)
    cams = _camera_map(args.camera_map)
    if src.is_file():
        clip_id = args.clip_id or src.stem
        camera = cams.get(clip_id) or args.camera_id or src.parent.name
        return [(src, clip_id, camera)]
    if not src.is_dir():
        raise SystemExit(f"Нет такого файла или папки: {src}")
    videos = sorted(list_videos(src), key=lambda p: p.stat().st_size)   # короткие первыми: раньше видно результат
    if args.clips:
        wanted = {c.strip() for c in args.clips.split(",") if c.strip()}
        videos = [v for v in videos if v.stem in wanted]
    out, seen, warned = [], set(), False
    for v in videos:
        clip_id = v.stem
        if clip_id in seen:
            clip_id = f"{v.parent.name}__{v.stem}"
            log(f"  повтор имени {v.stem}: clip_id = {clip_id}")
        seen.add(clip_id)
        rel = v.relative_to(src)
        camera = cams.get(clip_id) or args.camera_id
        if not camera:
            camera = rel.parts[0] if len(rel.parts) > 1 else src.name
            if not warned:
                log("  ВНИМАНИЕ: camera_id взят из имени папки. Проверьте его перед сдачей "
                    "(--camera-id или --camera-map clip_id,camera_id)")
                warned = True
        out.append((v, clip_id, camera))
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m smokedet.run",
                                 description="Детектор поведения, похожего на курение: ролики → события")
    ap.add_argument("--input", required=True, help="папка с роликами (ищет во вложенных) или один файл")
    ap.add_argument("--out-dir", required=True, help="куда писать events.json/csv, timing.csv, manifest.json")
    ap.add_argument("--render-dir", help="куда писать видео с отметками (для видео объекта — private/renders/...)")
    ap.add_argument("--render-only-events", action="store_true", help="рендерить только клипы с событиями")
    ap.add_argument("--camera-id", help="camera_id для всех клипов")
    ap.add_argument("--camera-map", help="CSV с колонками clip_id,camera_id")
    ap.add_argument("--clip-id", help="clip_id, если --input — один файл")
    ap.add_argument("--clips", help="обработать только эти clip_id (через запятую)")
    ap.add_argument("--roi", help="маска зоны интереса: файл (PNG или JSON) или папка с <clip_id>.* / <camera_id>.*")
    ap.add_argument("--config", help="конфиг (по умолчанию configs/default.yaml)")
    ap.add_argument("--preset", choices=["fast", "accurate"], help="fast — быстрее, accurate — точнее")
    ap.add_argument("--set", action="append", default=[], metavar="КЛЮЧ=ЗНАЧЕНИЕ",
                    help="переопределить параметр, например --set cycles.enter_threshold=0.18")
    ap.add_argument("--no-vlm", action="store_true", help="без VLM (детектор предметов и правила остаются)")
    ap.add_argument("--no-verify", action="store_true", help="без этапа проверки вообще (только правила)")
    ap.add_argument("--gt", help="разметка (labels/reference.csv): сразу посчитать метрики")
    ap.add_argument("--cache-dir", default=str(PROJECT_ROOT / "results" / "cache"))
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = Log(out_dir / "run.log")
    overrides = _parse_set(args.set)
    if args.no_vlm:
        overrides = deep_merge(overrides, {"vlm": {"enabled": False}})
    cfg = load_config(args.config, args.preset, overrides)
    args.verify = not args.no_verify
    clips = collect_clips(args, log)
    if not clips:
        log(f"Роликов не найдено в {args.input}")
        return 1
    log(f"smokedet {__version__}: {len(clips)} клип(ов), конфиг {config_hash(cfg)}, пресет {args.preset or '—'}")

    all_rows: list[dict] = []
    timing: list[dict] = []
    results: list[dict] = []
    device = resolve_device(cfg["pose"].get("device", "auto"))
    t_start = time.time()
    for k, (video, clip_id, camera_id) in enumerate(clips, 1):
        log(f"[{k}/{len(clips)}] {clip_id} (камера {camera_id})")
        try:
            res = process_clip(video, clip_id, camera_id, cfg, args, log)
            results.append(res)
            all_rows += res["rows"]
            timing.append(res["timing"])
            device = res["timing"]["device"]
        except KeyboardInterrupt:
            log("Остановлено пользователем; сохраняю то, что успели")
            break
        except Exception as exc:
            log(f"  ОШИБКА: {type(exc).__name__}: {exc}")
            (out_dir / "errors.log").open("a", encoding="utf-8").write(
                f"--- {clip_id}\n{traceback.format_exc()}\n")
            timing.append({"clip_id": clip_id, "camera_id": camera_id, "file": video.name, "status": "error",
                           "error": f"{type(exc).__name__}: {exc}"[:300]})
        write_events(out_dir, all_rows)
        write_timing(out_dir, timing)

    manifest = build_manifest(cfg, args, results, timing, device)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Готово за {time.time() - t_start:.0f} с: событий {len(all_rows)}, "
        f"клипов с ошибкой {len(manifest['run']['errors'])}. Результаты: {out_dir}")

    if args.gt:
        from .evaluate import evaluate, load_durations, load_gt, to_markdown

        m = evaluate(all_rows, load_gt(args.gt), load_durations(out_dir / "timing.csv"))
        md = to_markdown(m, title=f"Метрики: {Path(args.input).name}")
        (out_dir / "metrics.md").write_text(md, encoding="utf-8")
        (out_dir / "metrics.json").write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
        print(md)
    return 0 if not manifest["run"]["errors"] else 2


if __name__ == "__main__":
    sys.exit(main())
