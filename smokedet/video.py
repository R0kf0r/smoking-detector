"""Чтение видео: время кадра берём из меток времени файла (PTS), а не номер кадра / fps."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Iterator

import numpy as np

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".ts", ".mts", ".webm", ".mpg", ".mpeg", ".dav", ".h264",
             ".wmv", ".flv", ".3gp", ".asf"}


def list_videos(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXT
                  and not p.name.startswith("."))


def probe(path: str | Path) -> dict:
    """Длительность, частота, размер и кодек через ffprobe (только числа)."""
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
           "stream=width,height,avg_frame_rate,r_frame_rate,codec_name,nb_frames:format=duration",
           "-of", "json", str(path)]
    out = json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout or "{}")
    st = (out.get("streams") or [{}])[0]

    def rate(s: str | None) -> float:
        if not s or s in {"0/0", "0"}:
            return 0.0
        num, _, den = s.partition("/")
        return float(num) / float(den or 1) if float(den or 1) else 0.0

    fps = rate(st.get("avg_frame_rate")) or rate(st.get("r_frame_rate"))
    dur = float((out.get("format") or {}).get("duration") or 0.0)
    return {
        "width": int(st.get("width") or 0), "height": int(st.get("height") or 0),
        "fps": round(fps, 3), "duration_s": round(dur, 3), "codec": st.get("codec_name", ""),
        "nb_frames": int(st["nb_frames"]) if str(st.get("nb_frames", "")).isdigit() else None,
    }


def _iter_pyav(path: str | Path) -> Iterator[tuple[float, np.ndarray]]:
    import av  # PyAV

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        tb = float(stream.time_base) if stream.time_base else None
        t0 = None
        for i, frame in enumerate(container.decode(stream)):
            if frame.pts is not None and tb:
                t = frame.pts * tb
            elif frame.time is not None:
                t = float(frame.time)
            else:
                rate = float(stream.average_rate or 25)
                t = i / rate
            if t0 is None:
                t0 = t
            yield float(t - t0), frame.to_ndarray(format="bgr24")


def _iter_opencv(path: str | Path) -> Iterator[tuple[float, np.ndarray]]:
    import cv2

    cap = cv2.VideoCapture(str(path))
    t0 = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if t0 is None:
            t0 = t
        yield float(t - t0), frame
    cap.release()


def iter_frames(path: str | Path, target_fps: float = 0.0) -> Iterator[tuple[float, np.ndarray]]:
    """Кадры с временем от начала ролика; при target_fps > 0 — не чаще target_fps в секунду."""
    try:
        source = _iter_pyav(path)
        first = next(source)
    except Exception:
        source = _iter_opencv(path)
        first = next(source, None)
        if first is None:
            return

    step = 1.0 / target_fps if target_fps and target_fps > 0 else 0.0
    next_keep: float | None = None

    def all_frames():
        yield first
        yield from source

    for t, img in all_frames():
        if step == 0.0:
            yield t, img
            continue
        if next_keep is None or t >= next_keep - 1e-6:
            yield t, img
            next_keep = t + step if next_keep is None else next_keep + step
            if next_keep <= t:          # был пропуск в записи — догоняем
                next_keep = t + step


def frames_at(path: str | Path, times: list[float]) -> Iterator[tuple[float, float, np.ndarray]]:
    """Для каждого запрошенного времени отдаёт ближайший кадр: (запрошенное, фактическое, кадр).
    Видео читается один раз, времена должны быть отсортированы по возрастанию."""
    if not times:
        return
    wanted = sorted(times)
    j = 0
    prev = None
    for t, img in iter_frames(path, 0.0):
        while j < len(wanted) and t >= wanted[j]:
            r = wanted[j]
            if prev is not None and abs(prev[0] - r) < abs(t - r):
                yield r, prev[0], prev[1]
            else:
                yield r, t, img
            j += 1
        if j >= len(wanted):
            return
        prev = (t, img)
    while prev is not None and j < len(wanted):
        yield wanted[j], prev[0], prev[1]
        j += 1
