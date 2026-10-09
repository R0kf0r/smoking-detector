"""Правила событий SPEC §3–4."""

from __future__ import annotations

from smokedet.cycles import Cycle
from smokedet.events import build_events


def cyc(a: float, b: float, tid: int = 1, verdict: str = "none", d_min: float = 0.1) -> Cycle:
    """Цикл: подход 1 с, пауза у рта [a, b], отход 1 с."""
    return Cycle(track_id=tid, t_start=a - 1, t_enter=a, t_exit=b, t_end=b + 1, pause_s=b - a,
                 d_min=d_min, t_peak=(a + b) / 2, verdict=verdict)


def test_two_cycles_within_20s_make_event(cfg):
    ev = build_events({1: [cyc(5, 7), cyc(12, 14)]}, cfg)
    assert len(ev) == 1
    assert (ev[0].start, ev[0].end) == (4, 15)
    assert len(ev[0].cycles) == 2 and not ev[0].has_object


def test_one_cycle_is_not_enough(cfg):
    assert build_events({1: [cyc(5, 7)]}, cfg) == []


def test_one_cycle_with_object_is_enough(cfg):
    ev = build_events({1: [cyc(5, 7, verdict="smoke")]}, cfg)
    assert len(ev) == 1 and ev[0].has_object
    assert ev[0].confidence > build_events({1: [cyc(5, 7), cyc(12, 14)]}, cfg)[0].confidence - 0.2


def test_single_with_object_can_be_disabled(cfg):
    cfg["events"]["allow_single_with_object"] = False
    assert build_events({1: [cyc(5, 7, verdict="smoke")]}, cfg) == []


def test_gap_over_15s_splits(cfg):
    # перерыв 25 с: две отдельные цепочки по одному циклу → событий нет
    assert build_events({1: [cyc(5, 7), cyc(33, 35)]}, cfg) == []


def test_window_20s(cfg):
    # перерыв 14 с (≤ 15), но два цикла не помещаются в 20 с: 4 → 26
    assert build_events({1: [cyc(5, 7), cyc(22, 25)]}, cfg) == []
    # третий цикл рядом со вторым — пара 2–3 помещается в 20 с, событие от первого до третьего
    ev = build_events({1: [cyc(5, 7), cyc(22, 25), cyc(30, 32)]}, cfg)
    assert len(ev) == 1 and (ev[0].start, ev[0].end) == (4, 33)


def test_merge_within_15s(cfg):
    ev = build_events({1: [cyc(5, 7), cyc(12, 14), cyc(27, 29), cyc(33, 35)]}, cfg)
    assert len(ev) == 1 and len(ev[0].cycles) == 4 and (ev[0].start, ev[0].end) == (4, 36)


def test_two_people_two_events(cfg):
    ev = build_events({1: [cyc(5, 7, 1), cyc(12, 14, 1)], 2: [cyc(6, 8, 2), cyc(13, 15, 2)]}, cfg)
    assert len(ev) == 2 and {e.track_id for e in ev} == {1, 2}


def test_people_are_never_merged(cfg):
    # по одному циклу у двух людей рядом во времени — это не событие
    assert build_events({1: [cyc(5, 7, 1)], 2: [cyc(12, 14, 2)]}, cfg) == []


def test_veto_cycles_are_excluded(cfg):
    assert build_events({1: [cyc(5, 7), cyc(12, 14, verdict="veto")]}, cfg) == []


def test_peak_prefers_smoke_then_min_d(cfg):
    ev = build_events({1: [cyc(5, 7, d_min=0.05), cyc(12, 14, d_min=0.15)]}, cfg)
    assert ev[0].peak == 6.0
    ev = build_events({1: [cyc(5, 7, d_min=0.05), cyc(12, 14, verdict="smoke", d_min=0.15)]}, cfg)
    assert ev[0].peak == 13.0


def test_output_fields_match_spec(cfg):
    ev = build_events({1: [cyc(5, 7), cyc(12, 14)]}, cfg)[0]
    ev.box = (1.04, 2.0, 3.0, 4.0)
    row = ev.as_output("cam01", "clip01", "clip01_e001")
    assert list(row) == ["camera_id", "clip_id", "event_id", "start_sec", "end_sec", "confidence", "label",
                         "person_track_id", "peak_sec", "x1", "y1", "x2", "y2"]
    assert row["label"] == "smoking_like" and 0 <= row["confidence"] <= 1
    assert isinstance(row["person_track_id"], int) and row["x1"] == 1.0


def test_confirm_time(cfg):
    ev = build_events({1: [cyc(5, 7), cyc(12, 14), cyc(20, 22)]}, cfg)[0]
    assert ev.confirm == 14                          # конец паузы второго цикла
    ev = build_events({1: [cyc(5, 7, verdict="smoke"), cyc(12, 14)]}, cfg)[0]
    assert ev.confirm == 7
