"""Тяжёлый этап: YOLO26-pose находит людей и 17 точек, BoT-SORT с ReID даёт постоянные ID.
Результат кэшируется, чтобы правила и пороги пересчитывать без повторного прогона моделей."""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import yaml

from .config import PROJECT_ROOT
from .tracks import Detections
from .video import iter_frames, probe

# операции, которых нет на mps, выполняются на CPU вместо ошибки
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# без телеметрии и автоустановки пакетов Ultralytics во время прогона (веса всё равно скачиваются)
os.environ.setdefault("YOLO_OFFLINE", "1")


def resolve_device(device: str = "auto") -> str:
    if device and device != "auto":
        return device
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "0"
    except Exception:
        pass
    return "cpu"


def resolve_weights(name: str, download: bool = True) -> str:
    """Веса ищем в папке models/ проекта; если их нет — скачиваем туда же из релизов Ultralytics,
    чтобы веса лежали в проекте и попадали в архив для сдачи."""
    p = Path(name)
    if p.is_file():
        return str(p)
    local = PROJECT_ROOT / "models" / p.name
    if local.is_file():
        return str(local)
    if download:
        try:
            from ultralytics.utils.downloads import attempt_download_asset

            local.parent.mkdir(parents=True, exist_ok=True)
            got = attempt_download_asset(local)
            if Path(got).is_file():
                return str(got)
        except Exception:
            pass
    return name


def write_tracker_yaml(cfg: dict, fps: float, out_dir: Path, with_reid: bool | None = None) -> Path:
    tr = cfg["tracker"]
    reid = tr["with_reid"] if with_reid is None else with_reid
    data = {
        "tracker_type": "botsort",
        "track_high_thresh": float(tr["track_high_thresh"]),
        "track_low_thresh": float(tr["track_low_thresh"]),
        "new_track_thresh": float(tr["new_track_thresh"]),
        "track_buffer": max(1, int(round(float(tr["track_buffer_s"]) * fps))),  # в обработанных кадрах
        "match_thresh": float(tr["match_thresh"]),
        "fuse_score": True,
        "gmc_method": "none",            # камера неподвижна, компенсация движения не нужна
        "proximity_thresh": float(tr["proximity_thresh"]),
        "appearance_thresh": float(tr["appearance_thresh"]),
        "with_reid": bool(reid),
        "model": "auto",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"botsort_{'reid' if reid else 'noreid'}_{data['track_buffer']}.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def cache_key(video: Path, cfg: dict) -> dict:
    st = video.stat()
    return {
        "video": str(video.resolve()), "size": st.st_size, "mtime": int(st.st_mtime),
        "target_fps": cfg["video"]["target_fps"], "model": cfg["pose"]["model"],
        "imgsz": cfg["pose"]["imgsz"], "conf": cfg["pose"]["conf"], "tracker": cfg["tracker"],
    }


def same_key(a: dict | None, b: dict) -> bool:
    """Ключи кэша совпадают; путь к видео сравниваем по имени файла, чтобы кэш читался
    и из другой копии проекта (например, из Linux-среды на том же компьютере)."""
    if not isinstance(a, dict):
        return False
    rest_a = {k: v for k, v in a.items() if k not in ("video", "mtime")}
    rest_b = {k: v for k, v in b.items() if k not in ("video", "mtime")}
    return rest_a == rest_b and Path(str(a.get("video", ""))).name == Path(str(b.get("video", ""))).name


def run_pose_tracking(video: str | Path, cfg: dict, cache_path: str | Path | None = None,
                      log=print) -> Detections:
    video = Path(video)
    key = cache_key(video, cfg)
    if cache_path and Path(cache_path).is_file():
        try:
            dets = Detections.load(cache_path)
            if same_key(dets.meta.get("key"), key):
                log(f"  кэш: {cache_path}")
                return dets
        except Exception:
            pass

    from ultralytics import YOLO

    info = probe(video)
    target_fps = float(cfg["video"]["target_fps"] or 0)
    eff_fps = min(target_fps, info["fps"]) if target_fps and info["fps"] else (info["fps"] or 25.0)
    device = resolve_device(cfg["pose"].get("device", "auto"))
    tracker_dir = PROJECT_ROOT / "results" / "cache" / "trackers"
    model = YOLO(resolve_weights(cfg["pose"]["model"]))
    reid = bool(cfg["tracker"]["with_reid"])
    tracker = write_tracker_yaml(cfg, eff_fps, tracker_dir, reid)

    frame_t, det_frame, det_track, det_box, det_conf, det_kpts = [], [], [], [], [], []
    t_begin = time.time()
    n_expected = int(info["duration_s"] * eff_fps) if info["duration_s"] else None
    for fi, (t, img) in enumerate(iter_frames(video, target_fps)):
        kwargs = dict(persist=True, tracker=str(tracker), imgsz=int(cfg["pose"]["imgsz"]),
                      conf=float(cfg["pose"]["conf"]), device=device, verbose=False)
        try:
            res = model.track(img, **kwargs)[0]
        except Exception as exc:          # ReID «auto» может не поддерживаться — работаем без него
            if not reid:
                raise
            log(f"  ReID недоступен ({type(exc).__name__}: {exc}); продолжаю без ReID")
            reid = False
            model = YOLO(resolve_weights(cfg["pose"]["model"]))
            tracker = write_tracker_yaml(cfg, eff_fps, tracker_dir, False)
            kwargs["tracker"] = str(tracker)
            res = model.track(img, **kwargs)[0]
        frame_t.append(t)
        boxes = res.boxes
        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else np.full(len(xyxy), -1)
            kp = res.keypoints.data.cpu().numpy() if res.keypoints is not None else np.zeros((len(xyxy), 17, 3))
            for j in range(len(xyxy)):
                det_frame.append(fi)
                det_track.append(int(ids[j]))
                det_box.append(xyxy[j])
                det_conf.append(float(conf[j]))
                det_kpts.append(kp[j] if kp.shape[0] > j else np.zeros((17, 3)))
        if fi and fi % 150 == 0:
            speed = fi / max(1e-6, time.time() - t_begin)
            total = f"/{n_expected}" if n_expected else ""
            log(f"  кадр {fi}{total}, {speed:.1f} кадр/с")

    meta = {
        "key": key, "video_name": video.name, "width": info["width"], "height": info["height"],
        "fps": info["fps"], "duration_s": info["duration_s"], "processed_fps": eff_fps,
        "device": device, "reid": reid, "frames": len(frame_t),
        "seconds": round(time.time() - t_begin, 2),
    }
    dets = Detections(
        frame_t=np.asarray(frame_t, dtype=float),
        det_frame=np.asarray(det_frame, dtype=int),
        det_track=np.asarray(det_track, dtype=int),
        det_box=np.asarray(det_box, dtype=float).reshape(-1, 4),
        det_conf=np.asarray(det_conf, dtype=float),
        det_kpts=np.asarray(det_kpts, dtype=float).reshape(-1, 17, 3),
        meta=meta,
    )
    if cache_path:
        dets.save(cache_path)
    return dets
