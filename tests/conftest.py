"""Общие заготовки для тестов: конфиг и синтетический человек, который подносит руку ко рту."""

from __future__ import annotations

import numpy as np
import pytest

from smokedet.config import load_config
from smokedet.tracks import Detections

# поза стоящего человека (рамка 100..200 × 100..400, торс 100 px)
BASE = {
    0: (150, 140),                    # нос
    1: (145, 132), 2: (155, 132),     # глаза
    3: (140, 135), 4: (160, 135),     # уши
    5: (130, 180), 6: (170, 180),     # плечи
    7: (125, 230), 8: (175, 230),     # локти
    9: (125, 280), 10: (175, 280),    # запястья
    11: (135, 280), 12: (165, 280),   # бёдра
    13: (135, 340), 14: (165, 340), 15: (135, 395), 16: (165, 395),
}
UP_ELBOW, UP_WRIST = np.array([170.0, 200.0]), np.array([160.0, 160.0])


@pytest.fixture
def cfg():
    return load_config()


def person_kpts(lift: float, dx: float = 0.0) -> np.ndarray:
    """17 точек; lift ∈ [0, 1] — насколько правая рука поднята ко рту."""
    k = np.zeros((17, 3))
    for i, (x, y) in BASE.items():
        k[i] = (x + dx, y, 0.9)
    el = np.array(BASE[8], float) * (1 - lift) + UP_ELBOW * lift
    wr = np.array(BASE[10], float) * (1 - lift) + UP_WRIST * lift
    k[8, :2] = el + (dx, 0)
    k[10, :2] = wr + (dx, 0)
    return k


def lift_profile(t: np.ndarray, cycles: list[tuple[float, float]], ramp: float = 1.0) -> np.ndarray:
    """cycles: [(начало паузы у рта, конец паузы)]; подход и отход длятся ramp секунд."""
    out = np.zeros_like(t)
    for a, b in cycles:
        up = np.clip((t - (a - ramp)) / ramp, 0, 1)
        down = np.clip(((b + ramp) - t) / ramp, 0, 1)
        out = np.maximum(out, np.minimum(up, down))
    return out


def synthetic_detections(people: list[dict], duration: float = 20.0, fps: float = 15.0) -> Detections:
    """people: [{"id": 1, "cycles": [(a, b), ...], "dx": 0}] — каждый в своей колонке кадра."""
    t = np.arange(0, duration, 1 / fps)
    det_frame, det_track, det_box, det_conf, det_kpts = [], [], [], [], []
    for p in people:
        lift = lift_profile(t, p.get("cycles", []))
        dx = p.get("dx", 0.0)
        for fi, li in enumerate(lift):
            det_frame.append(fi)
            det_track.append(p["id"])
            det_box.append([100 + dx, 100, 200 + dx, 400])
            det_conf.append(0.9)
            det_kpts.append(person_kpts(float(li), dx))
    return Detections(frame_t=t, det_frame=np.array(det_frame), det_track=np.array(det_track),
                      det_box=np.array(det_box, float), det_conf=np.array(det_conf),
                      det_kpts=np.array(det_kpts), meta={"width": 640, "height": 480, "fps": fps})
