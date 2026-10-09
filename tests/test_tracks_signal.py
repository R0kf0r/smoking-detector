"""Склейка обрывков треков и геометрия сигнала."""

from __future__ import annotations

import numpy as np

from smokedet.signal import hand_points, mouth_points, person_scale, track_signal
from smokedet.tracks import Detections, build_tracks, stitch_ids

from .conftest import person_kpts


def dets_from(pieces):
    """pieces: [(track_id, t0, t1, x)] — человек стоит в точке x."""
    t = np.round(np.arange(0, 20, 0.1), 3)
    fr, tr, bx = [], [], []
    for tid, t0, t1, x in pieces:
        for i, ti in enumerate(t):
            if t0 <= ti <= t1:
                fr.append(i)
                tr.append(tid)
                bx.append([x, 100, x + 100, 400])
    n = len(fr)
    return Detections(frame_t=t, det_frame=np.array(fr), det_track=np.array(tr), det_box=np.array(bx, float),
                      det_conf=np.full(n, 0.9), det_kpts=np.zeros((n, 17, 3)))


def test_stitch_close_pieces():
    d = dets_from([(1, 0, 5, 100), (2, 6, 10, 110)])      # перерыв 1 с, сдвиг 10 px
    assert stitch_ids(d, 2.0, 0.6) == {1: 1, 2: 1}
    tracks = build_tracks(d, stitch_ids(d, 2.0, 0.6))
    assert list(tracks) == [1] and tracks[1].source_ids == [1, 2]


def test_no_stitch_after_long_gap_or_far():
    d = dets_from([(1, 0, 5, 100), (2, 9, 12, 100)])      # перерыв 4 с
    assert stitch_ids(d, 2.0, 0.6) == {1: 1, 2: 2}
    d = dets_from([(1, 0, 5, 100), (2, 6, 10, 400)])      # сдвиг 300 px = 1 высота рамки
    assert stitch_ids(d, 2.0, 0.6) == {1: 1, 2: 2}


def test_no_stitch_when_overlapping_in_time():
    d = dets_from([(1, 0, 5, 100), (2, 4, 10, 110)])      # два человека одновременно
    assert stitch_ids(d, 2.0, 0.6) == {1: 1, 2: 2}


def test_geometry_points_and_scale():
    k = person_kpts(0.0)[None]
    box = np.array([[100, 100, 200, 400]], float)
    scale = person_scale(k, box, 0.3, 0.3)
    assert scale[0] == 100.0                               # торс: плечи y=180 → бёдра y=280
    mouth = mouth_points(k, 0.3, 0.6, scale)
    assert np.allclose(mouth[0], [150, 144.8])             # нос + 0,6·(нос − середина глаз)
    hl, hr = hand_points(k, 0.3, 0.25)
    assert np.allclose(hr[0], [175, 292.5])                # запястье + 0,25·(запястье − локоть)
    k2 = k.copy()
    k2[0, 11:13, 2] = 0.0                                  # бёдер не видно → 0,3·диагональ
    assert np.isclose(person_scale(k2, box, 0.3, 0.3)[0], 0.3 * np.hypot(100, 300))


def test_signal_low_when_hand_at_mouth(cfg):
    from smokedet.tracks import Track

    t = np.arange(0, 3, 1 / 15)
    lift = (t > 1).astype(float)
    k = np.stack([person_kpts(li) for li in lift])
    tr = Track(1, t, np.tile([100, 100, 200, 400], (len(t), 1)).astype(float), k, np.full(len(t), 0.9))
    sig = track_signal(tr, cfg)
    assert sig["d"].iloc[:10].min() > 1.0                  # рука внизу
    assert sig["d"].iloc[-10:].max() < 0.15                # рука у рта
