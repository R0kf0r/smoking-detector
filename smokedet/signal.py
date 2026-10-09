"""След жеста: расстояние «кисть — рот», нормированное на масштаб человека, во времени.

17 точек COCO: 0 нос, 1–2 глаза, 3–4 уши, 5–6 плечи, 7–8 локти, 9–10 запястья, 11–12 бёдра.
Точек «рот» и «кисть» нет, поэтому оцениваем:
    рот   = нос + mouth_offset · (нос − середина глаз)
    кисть = запястье + hand_extension · (запястье − локоть)
Масштаб = длина торса (середина плеч → середина бёдер), иначе bbox_scale_factor · диагональ рамки.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .tracks import Track

NOSE, LEYE, REYE = 0, 1, 2
LSH, RSH, LEL, REL, LWR, RWR, LHIP, RHIP = 5, 6, 7, 8, 9, 10, 11, 12


def _vis(k: np.ndarray, idx: int, kmin: float) -> np.ndarray:
    return k[:, idx, 2] >= kmin


def _xy(k: np.ndarray, idx: int) -> np.ndarray:
    return k[:, idx, :2].astype(float)


def mouth_points(k: np.ndarray, kmin: float, offset: float, scale: np.ndarray) -> np.ndarray:
    """[n, 2] координаты рта; NaN, если нос не виден."""
    n = len(k)
    out = np.full((n, 2), np.nan)
    nose_ok = _vis(k, NOSE, kmin)
    le, re = _vis(k, LEYE, kmin), _vis(k, REYE, kmin)
    nose = _xy(k, NOSE)
    eyes_mid = np.where((le & re)[:, None], (_xy(k, LEYE) + _xy(k, REYE)) / 2.0,
                        np.where(le[:, None], _xy(k, LEYE), _xy(k, REYE)))
    has_eye = le | re
    with_eye = nose_ok & has_eye
    out[with_eye] = nose[with_eye] + offset * (nose[with_eye] - eyes_mid[with_eye])
    # глаза не видны (профиль, капюшон): рот чуть ниже носа, сдвиг по масштабу человека
    no_eye = nose_ok & ~has_eye
    if np.any(no_eye):
        out[no_eye] = nose[no_eye] + np.stack([np.zeros(no_eye.sum()), 0.08 * scale[no_eye]], axis=1)
    return out


def hand_points(k: np.ndarray, kmin: float, extension: float) -> tuple[np.ndarray, np.ndarray]:
    """Две [n, 2] точки кистей (левая, правая); NaN, если запястье не видно."""
    hands = []
    for wr, el in ((LWR, LEL), (RWR, REL)):
        w_ok, e_ok = _vis(k, wr, kmin), _vis(k, el, kmin)
        w, e = _xy(k, wr), _xy(k, el)
        h = np.full_like(w, np.nan)
        both = w_ok & e_ok
        h[both] = w[both] + extension * (w[both] - e[both])
        only_w = w_ok & ~e_ok
        h[only_w] = w[only_w]
        hands.append(h)
    return hands[0], hands[1]


def person_scale(k: np.ndarray, box: np.ndarray, kmin: float, bbox_factor: float) -> np.ndarray:
    """Длина торса; если плеч или бёдер не видно — bbox_factor · диагональ рамки."""
    sh_ok = _vis(k, LSH, kmin) | _vis(k, RSH, kmin)
    hip_ok = _vis(k, LHIP, kmin) | _vis(k, RHIP, kmin)

    def mid(a: int, b: int) -> np.ndarray:
        va, vb = _vis(k, a, kmin), _vis(k, b, kmin)
        pa, pb = _xy(k, a), _xy(k, b)
        return np.where((va & vb)[:, None], (pa + pb) / 2.0, np.where(va[:, None], pa, pb))

    torso = np.linalg.norm(mid(LSH, RSH) - mid(LHIP, RHIP), axis=1)
    diag = np.hypot(box[:, 2] - box[:, 0], box[:, 3] - box[:, 1])
    fallback = bbox_factor * diag
    ok = sh_ok & hip_ok & (torso > 0.15 * fallback)   # торс не может быть почти нулевым
    return np.where(ok, torso, fallback)


def _odd(n: int) -> int:
    n = max(1, int(round(n)))
    return n if n % 2 == 1 else n + 1


def effective_fps(t: np.ndarray) -> float:
    if len(t) < 2:
        return 1.0
    dt = np.diff(t)
    dt = dt[dt > 1e-6]
    return float(1.0 / np.median(dt)) if len(dt) else 1.0


def smooth_series(t: np.ndarray, x: np.ndarray, max_gap_s: float, window_s: float) -> np.ndarray:
    """Заполняет короткие провалы и сглаживает центрированной медианой без задержки."""
    if len(x) == 0:
        return x.copy()
    fps = effective_fps(t)
    s = pd.Series(x, dtype=float)
    limit = max(0, int(round(max_gap_s * fps)))
    filled = s.interpolate(limit=limit, limit_area="inside") if limit > 0 else s
    w = _odd(window_s * fps)
    smoothed = filled.rolling(w, center=True, min_periods=1).median()
    smoothed[filled.isna()] = np.nan     # длинные провалы не заполняем
    return smoothed.to_numpy()


def track_signal(track: Track, cfg: dict) -> pd.DataFrame:
    """Таблица по кадрам трека: t, d_raw, d (сглаженный), dl/dr по рукам, масштаб, рост, точки."""
    g, sg = cfg["geometry"], cfg["signal"]
    kmin = float(cfg["pose"]["kpt_conf_min"])
    k, box, t = track.kpts, track.box, track.t
    scale_raw = person_scale(k, box, kmin, float(g["bbox_scale_factor"]))
    scale = smooth_series(t, scale_raw, max_gap_s=1e9, window_s=float(g["scale_smooth_s"]))
    mouth = mouth_points(k, kmin, float(g["mouth_offset"]), scale)
    hl, hr = hand_points(k, kmin, float(g["hand_extension"]))
    dl = np.linalg.norm(hl - mouth, axis=1) / scale
    dr = np.linalg.norm(hr - mouth, axis=1) / scale
    d_raw = np.fmin(dl, dr)       # fmin игнорирует NaN одной руки
    use_left = np.where(np.isnan(dr), True, np.where(np.isnan(dl), False, dl <= dr))
    hand = np.where(use_left[:, None], hl, hr)
    d = smooth_series(t, d_raw, float(sg["max_gap_s"]), float(sg["smooth_window_s"]))
    dl_s = smooth_series(t, dl, float(sg["max_gap_s"]), float(sg["smooth_window_s"]))
    dr_s = smooth_series(t, dr, float(sg["max_gap_s"]), float(sg["smooth_window_s"]))
    return pd.DataFrame({
        "t": t,
        "d_raw": d_raw,
        "d": d,
        "dl": dl_s,
        "dr": dr_s,
        "scale": scale,
        "height": box[:, 3] - box[:, 1],
        "mouth_x": mouth[:, 0], "mouth_y": mouth[:, 1],
        "hand_x": hand[:, 0], "hand_y": hand[:, 1],
        "x1": box[:, 0], "y1": box[:, 1], "x2": box[:, 2], "y2": box[:, 3],
    })
