"""Оценщик по SPEC §6 на примерах, посчитанных вручную."""

from __future__ import annotations

import pytest

from smokedet.evaluate import GT, evaluate, load_gt

BOX = (100.0, 100.0, 200.0, 400.0)


def pred(start, end, clip="c1", box=BOX, track=1, eid="e"):
    return {"clip_id": clip, "camera_id": "cam", "event_id": eid, "start_sec": start, "end_sec": end,
            "confidence": 0.8, "person_track_id": track, "peak_sec": (start + end) / 2,
            "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3]}


def pos(start, end, clip="c1", box=BOX, scen="P01"):
    return GT(clip_id=clip, camera_id="cam", label="POSITIVE", start=start, end=end, scenario=scen, box=box)


def test_a_exact_match_is_tp():
    m = evaluate([pred(9, 21)], [pos(10, 20)])
    assert (m["TP"], m["FP"], m["FN"]) == (1, 0, 0)
    assert m["f1"] == 1.0
    assert m["median_latency_s"] == 0.0           # max(0, 9 − 10)


def test_b_overlap_only_inside_extension():
    # эталон 10–20 расширяется до 7–23; предсказание 22–25 пересекает его на 1 с → TP
    m = evaluate([pred(22, 25)], [pos(10, 20)])
    assert (m["TP"], m["FP"], m["FN"]) == (1, 0, 0)
    assert m["median_latency_s"] == 12.0
    # 22,5–25: пересечение 0,5 с < 1 с → FP и FN
    m = evaluate([pred(22.5, 25)], [pos(10, 20)])
    assert (m["TP"], m["FP"], m["FN"]) == (0, 1, 1)
    assert m["f1"] == 0.0


def test_c_other_person_by_iou():
    far = (400.0, 100.0, 500.0, 400.0)            # IoU = 0
    m = evaluate([pred(10, 20, box=far)], [pos(10, 20)])
    assert (m["TP"], m["FP"], m["FN"]) == (0, 1, 1)
    # IoU ровно 1/3 ≥ 0,30 → тот же человек
    half = (150.0, 100.0, 250.0, 400.0)           # пересечение 50×300, объединение 150×300
    m = evaluate([pred(10, 20, box=half)], [pos(10, 20)])
    assert m["TP"] == 1


def test_d_one_to_one_matching():
    m = evaluate([pred(10, 20, eid="a"), pred(11, 19, eid="b")], [pos(10, 20)])
    assert (m["TP"], m["FP"], m["FN"]) == (1, 1, 0)
    assert m["precision"] == 0.5 and m["recall"] == 1.0
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["pairs"][0]["event_id"] == "a"         # жадно по наибольшему пересечению (10 с против 8 с)


def test_e_ignore_interval():
    ign = GT(clip_id="c1", camera_id="cam", label="IGNORE", start=30, end=40)
    # 31–35 целиком в IGNORE и не пересекает позитив → не FP
    m = evaluate([pred(31, 35)], [pos(10, 20), ign])
    assert (m["TP"], m["FP"], m["FN"], m["ignored_predictions"]) == (0, 0, 1, 1)
    # 28–36: в IGNORE 6 с из 8 (≥ половины) → тоже не FP
    m = evaluate([pred(28, 36)], [ign])
    assert m["FP"] == 0 and m["ignored_predictions"] == 1
    # 25–33: в IGNORE 3 с из 8 (< половины) → FP
    m = evaluate([pred(25, 33)], [ign])
    assert m["FP"] == 1
    # в IGNORE, но пересекает позитив другого человека → FP (и FN позитива)
    ign2 = GT(clip_id="c1", camera_id="cam", label="IGNORE", start=8, end=22)
    far = (400.0, 100.0, 500.0, 400.0)
    m = evaluate([pred(10, 20, box=far)], [pos(10, 20), ign2])
    assert (m["TP"], m["FP"], m["FN"]) == (0, 1, 1)


def test_f_clip_and_camera_must_match():
    m = evaluate([pred(10, 20, clip="c2")], [pos(10, 20)])
    assert (m["TP"], m["FP"], m["FN"]) == (0, 1, 1)
    p = pred(10, 20)
    p["camera_id"] = "other"
    m = evaluate([p], [pos(10, 20)])
    assert m["TP"] == 0


def test_g_fp_per_hour():
    # 1 FP; клип 100 с, из них позитив 10 с → 90 с отрицательного видео
    m = evaluate([pred(10, 20), pred(60, 70, eid="x")], [pos(10, 20)], durations={"c1": 100.0})
    assert m["FP"] == 1
    assert m["fp_per_hour"] == pytest.approx(1 / (90 / 3600))


def test_h_track_id_when_no_box(tmp_path):
    p = tmp_path / "gt.csv"
    p.write_text("clip_id;camera_id;label;start_sec;end_sec;scenario;person_track_id\n"
                 "c1;cam;POSITIVE;10,0;20,5;P01;7\n"
                 "c1;cam;NEGATIVE;30;35;N02;\n"
                 "c1;cam;мусор;1;2;;\n", encoding="utf-8")
    gts = load_gt(p)
    assert len(gts) == 2
    assert gts[0].end == 20.5 and gts[0].person_track_id == 7 and gts[0].box is None
    assert evaluate([pred(10, 20, track=7)], gts)["TP"] == 1
    assert evaluate([pred(10, 20, track=3)], gts)["TP"] == 0


def test_by_scenario_counts_negative_fp():
    neg = GT(clip_id="c1", camera_id="cam", label="NEGATIVE", start=50, end=55, scenario="N02")
    m = evaluate([pred(50, 54)], [pos(10, 20), neg])
    assert m["by_scenario"]["N02"]["FP"] == 1
    assert m["by_scenario"]["P01"]["FN"] == 1
