"""Результат тяжёлого этапа (рамки, ID, точки по кадрам), кэш и склейка обрывков треков."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Detections:
    """Все детекции людей в клипе. Кэшируется в .npz, чтобы пороги перебирать без моделей."""

    frame_t: np.ndarray          # [F] время обработанных кадров, с (от начала ролика)
    det_frame: np.ndarray        # [M] индекс кадра в frame_t
    det_track: np.ndarray        # [M] ID трекера (-1, если трекер не дал ID)
    det_box: np.ndarray          # [M, 4] x1, y1, x2, y2 в пикселях исходного кадра
    det_conf: np.ndarray         # [M] уверенность рамки
    det_kpts: np.ndarray         # [M, 17, 3] x, y, уверенность точки
    meta: dict = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            frame_t=self.frame_t, det_frame=self.det_frame, det_track=self.det_track,
            det_box=self.det_box, det_conf=self.det_conf, det_kpts=self.det_kpts,
            meta=np.array(json.dumps(self.meta, ensure_ascii=False)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "Detections":
        with np.load(path, allow_pickle=False) as z:
            return cls(
                frame_t=z["frame_t"], det_frame=z["det_frame"], det_track=z["det_track"],
                det_box=z["det_box"], det_conf=z["det_conf"], det_kpts=z["det_kpts"],
                meta=json.loads(str(z["meta"])),
            )

    @classmethod
    def empty(cls, meta: dict | None = None) -> "Detections":
        return cls(
            frame_t=np.zeros(0), det_frame=np.zeros(0, dtype=int), det_track=np.zeros(0, dtype=int),
            det_box=np.zeros((0, 4)), det_conf=np.zeros(0), det_kpts=np.zeros((0, 17, 3)),
            meta=meta or {},
        )


@dataclass
class Track:
    """Один человек во времени (после склейки обрывков)."""

    track_id: int
    t: np.ndarray            # [n]
    box: np.ndarray          # [n, 4]
    kpts: np.ndarray         # [n, 17, 3]
    conf: np.ndarray         # [n]
    source_ids: list = field(default_factory=list)

    def box_at(self, time: float) -> np.ndarray:
        i = int(np.argmin(np.abs(self.t - time)))
        return self.box[i]

    def heights(self) -> np.ndarray:
        return self.box[:, 3] - self.box[:, 1]


def _box_center(b: np.ndarray) -> np.ndarray:
    return np.array([(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0])


def stitch_ids(dets: Detections, max_gap_s: float, max_center_shift: float) -> dict[int, int]:
    """Склеивает обрывки одного человека: трекер потерял ID и выдал новый.

    Обрывок B присоединяется к группе A, если B начался не позже чем через max_gap_s
    после конца A, группы не пересекаются по времени, центр рамки сдвинулся не больше чем на
    max_center_shift высоты рамки и высота изменилась не больше чем в 1,6 раза.
    Возвращает словарь «исходный ID → ID группы» (ID группы — первый ID в ней).
    """
    ids = sorted(int(i) for i in np.unique(dets.det_track) if i >= 0)
    if not ids:
        return {}
    info = {}
    for tid in ids:
        rows = np.where(dets.det_track == tid)[0]
        times = dets.frame_t[dets.det_frame[rows]]
        order = np.argsort(times)
        rows, times = rows[order], times[order]
        info[tid] = {
            "t0": float(times[0]), "t1": float(times[-1]),
            "box0": dets.det_box[rows[0]], "box1": dets.det_box[rows[-1]],
        }
    groups: dict[int, dict] = {}   # id группы -> {t1, box1}
    mapping: dict[int, int] = {}
    for tid in sorted(ids, key=lambda i: info[i]["t0"]):
        cur = info[tid]
        best, best_key = None, None
        for gid, g in groups.items():
            gap = cur["t0"] - g["t1"]
            if gap <= 0 or gap > max_gap_s:
                continue
            h_prev = g["box1"][3] - g["box1"][1]
            h_cur = cur["box0"][3] - cur["box0"][1]
            if h_prev <= 0 or h_cur <= 0:
                continue
            ratio = h_cur / h_prev
            if ratio < 1 / 1.6 or ratio > 1.6:
                continue
            shift = np.linalg.norm(_box_center(cur["box0"]) - _box_center(g["box1"])) / ((h_prev + h_cur) / 2)
            if shift > max_center_shift:
                continue
            key = (gap, shift)
            if best_key is None or key < best_key:
                best, best_key = gid, key
        if best is None:
            groups[tid] = {"t1": cur["t1"], "box1": cur["box1"]}
            mapping[tid] = tid
        else:
            groups[best] = {"t1": cur["t1"], "box1": cur["box1"]}
            mapping[tid] = best
    return mapping


def build_tracks(dets: Detections, mapping: dict[int, int] | None = None) -> dict[int, Track]:
    """Собирает детекции в треки (с учётом склейки). Детекции без ID отбрасываются."""
    mapping = mapping or {}
    tracks: dict[int, Track] = {}
    if dets.det_track.size == 0:
        return tracks
    new_ids = np.array([mapping.get(int(i), int(i)) for i in dets.det_track])
    for gid in sorted(set(int(i) for i in new_ids if i >= 0)):
        rows = np.where(new_ids == gid)[0]
        times = dets.frame_t[dets.det_frame[rows]]
        order = np.lexsort((-dets.det_conf[rows], times))
        rows, times = rows[order], times[order]
        # если в одном кадре два обрывка одной группы — оставляем более уверенный
        keep = np.ones(len(rows), dtype=bool)
        keep[1:] = np.diff(times) > 1e-9
        rows, times = rows[keep], times[keep]
        src = sorted(set(int(i) for i in dets.det_track[rows]))
        tracks[gid] = Track(
            track_id=gid, t=times, box=dets.det_box[rows], kpts=dets.det_kpts[rows],
            conf=dets.det_conf[rows], source_ids=src,
        )
    return tracks
