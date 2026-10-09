"""Конвейер после модели позы на синтетических людях: сигнал → циклы → события → вывод → рендер."""

from __future__ import annotations

import json
import shutil
import subprocess

import numpy as np
import pytest

from smokedet.calibrate import apply_to_config
from smokedet.config import DEFAULT_CONFIG
from smokedet.roi import inside_fraction, load_roi
from smokedet.run import analyze, compute_signals, event_rows, make_tracks
from smokedet.video import frames_at, iter_frames, probe

from .conftest import synthetic_detections

HAS_FFMPEG = shutil.which("ffmpeg") is not None


def run_rules(dets, cfg, verification=None, roi=None):
    tracks = make_tracks(dets, cfg)
    signals = compute_signals(tracks, cfg)
    return tracks, signals, analyze(tracks, signals, verification, cfg, roi)


def test_two_puffs_give_one_event(cfg):
    dets = synthetic_detections([{"id": 1, "cycles": [(4, 6), (11, 13)]}])
    tracks, signals, res = run_rules(dets, cfg)
    assert len(res["cycles"][1]) == 2
    assert len(res["events"]) == 1
    row = event_rows(res["events"], "cam", "clip")[0]
    assert row["event_id"] == "clip_e001" and row["person_track_id"] == 1
    assert 2.5 <= row["start_sec"] <= 3.5 and 13.5 <= row["end_sec"] <= 14.6
    assert (row["x1"], row["y1"], row["x2"], row["y2"]) == (100.0, 100.0, 200.0, 400.0)
    assert row["start_sec"] <= row["peak_sec"] <= row["end_sec"]


def test_one_puff_no_event_unless_vlm_sees_cigarette(cfg):
    dets = synthetic_detections([{"id": 1, "cycles": [(4, 6)]}])
    _, _, res = run_rules(dets, cfg)
    assert len(res["cycles"][1]) == 1 and not res["events"]
    ver = {"items": [{"track_id": 1, "t": 4.5, "vlm": "cigarette", "obj": ""},
                     {"track_id": 1, "t": 5.5, "vlm": "cigarette", "obj": ""}]}
    _, _, res = run_rules(dets, cfg, ver)
    assert len(res["events"]) == 1 and res["events"][0].has_object


def test_phone_veto(cfg):
    dets = synthetic_detections([{"id": 1, "cycles": [(4, 6), (11, 13)]}])
    ver = {"items": [{"track_id": 1, "t": t, "vlm": "phone", "obj": "phone"} for t in (4.5, 5.5, 11.5, 12.5)]}
    _, _, res = run_rules(dets, cfg, ver)
    assert not res["events"]
    assert all(c.verdict == "veto" for c in res["cycles"][1])


def test_two_people_and_small_person(cfg):
    people = [{"id": 1, "cycles": [(4, 6), (11, 13)]}, {"id": 2, "cycles": [(5, 7), (12, 14)], "dx": 250}]
    _, _, res = run_rules(synthetic_detections(people), cfg)
    assert sorted(e.track_id for e in res["events"]) == [1, 2]
    cfg["geometry"]["min_person_height_px"] = 400          # рамка 300 px — ниже порога
    _, _, res = run_rules(synthetic_detections(people), cfg)
    assert not res["events"]


def test_roi_filters_events(cfg, tmp_path):
    poly = tmp_path / "roi.json"
    poly.write_text(json.dumps({"points": [[0, 0], [0.5, 0], [0.5, 1], [0, 1]]}))   # левая половина
    mask = load_roi(poly, 640, 480)
    assert mask.shape == (480, 640) and mask[:, :300].all() and not mask[:, 330:].any()
    assert inside_fraction(mask, np.array([[100, 100, 200, 400.0]])) == 1.0
    people = [{"id": 1, "cycles": [(4, 6), (11, 13)]}, {"id": 2, "cycles": [(5, 7), (12, 14)], "dx": 300}]
    _, _, res = run_rules(synthetic_detections(people), cfg, roi=mask)
    assert [e.track_id for e in res["events"]] == [1]
    assert res["dropped"] and res["dropped"][0][1] == "вне зоны интереса"


def test_apply_to_config_keeps_comments(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(DEFAULT_CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    apply_to_config(p, {"cycles.enter_threshold": 0.175, "cycles.exit_threshold": 0.3, "vlm.min_votes": 3})
    text = p.read_text(encoding="utf-8")
    import re

    assert re.search(r"enter_threshold: 0\.175\s+# d ниже", text)       # значение заменено, комментарий на месте
    assert re.search(r"exit_threshold: 0\.3\s+# d выше", text)
    assert re.search(r"min_votes: 3\s+# сколько", text)
    import yaml
    assert yaml.safe_load(text)["cycles"]["enter_threshold"] == 0.175


@pytest.fixture
def tiny_video(tmp_path):
    if not HAS_FFMPEG:
        pytest.skip("нет ffmpeg")
    path = tmp_path / "tiny.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    "testsrc=size=640x480:rate=25:duration=20", "-c:v", "mpeg4", "-q:v", "5", str(path)],
                   check=True)
    return path


def test_video_timestamps_and_subsampling(tiny_video):
    info = probe(tiny_video)
    assert (info["width"], info["height"]) == (640, 480) and abs(info["duration_s"] - 20) < 0.1
    times = [t for t, _ in iter_frames(tiny_video, 10.0)]
    assert 195 <= len(times) <= 205                       # 25 к/с → 10 к/с
    assert times[0] == 0.0 and np.all(np.diff(times) > 0.07)
    got = list(frames_at(tiny_video, [0.0, 5.02, 19.9]))
    assert [round(g[1], 2) for g in got] == [0.0, 5.0, 19.88]


def test_render_synthetic(cfg, tiny_video, tmp_path):
    from smokedet.render import build_plan, render_video

    people = [{"id": 1, "cycles": [(4, 6), (11, 13)]}, {"id": 2, "cycles": [(7, 15)], "dx": 300}]
    dets = synthetic_detections(people)
    tracks, signals, res = run_rules(dets, cfg)
    plan = build_plan(tracks, signals, res["cycles"], res["events"], cfg, 20.0, None)
    assert plan.default_track == 1
    out = tmp_path / "out.mp4"
    info = render_video(tiny_video, out, plan, cfg, log=lambda m: None)
    assert out.is_file() and info["frames"] == 500
    meta = probe(out)
    assert meta["width"] == 640 and meta["height"] > 480 and meta["codec"] in {"h264", "mpeg4"}


def test_vlm_answer_parsing():
    from smokedet.verify import VLMJudge

    assert VLMJudge.parse("Cigarette.") == "cigarette"
    assert VLMJudge.parse("a beer can") == "can"
    assert VLMJudge.parse("candy") == "unclear"          # не путаем «can» внутри слова
    assert VLMJudge.parse("Cigarettes") == "cigarette"
    assert VLMJudge.parse("") == "unclear"


def test_can_vetoes_cycle(cfg):
    dets = synthetic_detections([{"id": 1, "cycles": [(4, 6), (11, 13)]}])
    ver = {"items": [{"track_id": 1, "t": t, "vlm": "can", "obj": ""} for t in (4.5, 5.5, 11.5, 12.5)]}
    _, _, res = run_rules(dets, cfg, ver)
    assert not res["events"]


def test_smoke_evidence_beats_both_hands(cfg):
    from smokedet.cycles import Cycle
    from smokedet.verify import apply_votes

    def mk():
        return Cycle(track_id=1, t_start=3, t_enter=4, t_exit=6, t_end=7, pause_s=2, d_min=0.1, t_peak=5,
                     both_hands_frac=1.0)
    c = mk()
    apply_votes({1: [c]}, {"items": []}, cfg)
    assert c.verdict == "veto"                      # обе руки у лица без прямого признака — вето
    c = mk()
    ver = {"items": [{"track_id": 1, "t": t, "vlm": "cigarette", "obj": ""} for t in (4.5, 5.5)]}
    apply_votes({1: [c]}, ver, cfg)
    assert c.verdict == "smoke"                     # прикуривание: VLM видит сигарету


def test_apply_bool_to_config(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(DEFAULT_CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    apply_to_config(p, {"kinematics.both_hands_veto": False})
    import yaml
    assert yaml.safe_load(p.read_text(encoding="utf-8"))["kinematics"]["both_hands_veto"] is False


def test_cache_key_ignores_mount_path():
    from smokedet.pose import same_key

    a = {"video": "/Users/mac/p/private/clip.mp4", "size": 10, "mtime": 5, "target_fps": 15, "model": "m",
         "imgsz": 1280, "conf": 0.25, "tracker": {"x": 1}}
    assert same_key(a, dict(a, video="/sessions/u/mnt/p/private/clip.mp4", mtime=6))
    assert not same_key(a, dict(a, size=11))
    assert not same_key(a, dict(a, video="/Users/mac/p/private/other.mp4"))


def test_empty_hand_veto(cfg):
    dets = synthetic_detections([{"id": 1, "cycles": [(4, 6), (11, 13)]}])
    ver = {"items": [{"track_id": 1, "t": t, "vlm": "hand", "obj": ""} for t in (4.5, 5.5, 11.5, 12.5)]}
    cfg["vlm"]["hand_veto"] = True
    _, _, res = run_rules(dets, cfg, ver)
    assert not res["events"]                         # касание лица пустой рукой
    cfg["vlm"]["hand_veto"] = False
    _, _, res = run_rules(dets, cfg, ver)
    assert len(res["events"]) == 1


def test_verify_cache_name_ignores_decision_params(cfg):
    from smokedet.config import deep_merge
    from smokedet.run import verify_tag

    base = verify_tag(cfg)
    assert verify_tag(deep_merge(cfg, {"vlm": {"min_votes": 3, "hand_veto": False}})) == base
    assert verify_tag(deep_merge(cfg, {"cycles": {"enter_threshold": 0.3}})) == base
    assert verify_tag(deep_merge(cfg, {"vlm": {"scan_threshold": 0.5}})) != base


def test_vlm_request_cap(cfg):
    from smokedet.verify import plan_requests

    people = [{"id": i, "cycles": [(2 + i, 4 + i), (9 + i, 11 + i)], "dx": 120 * i} for i in range(1, 4)]
    tracks, signals, _ = run_rules(synthetic_detections(people), cfg)
    full = plan_requests(signals, cfg, 640, 480)
    cfg["vlm"]["max_requests_per_clip"] = 8
    capped = plan_requests(signals, cfg, 640, 480)
    assert len(full) > 8 and 0 < len(capped) <= 8
    assert [r.t for r in capped] == sorted(r.t for r in capped)


def test_parse_answer_kinds():
    from smokedet.verify import parse_answer

    assert parse_answer("nothing", "list") == "hand"
    assert parse_answer("Glove.", "list") == "hand"
    assert parse_answer("cigarette", "list") == "cigarette"
    assert parse_answer("A lit cigarette between fingers", "describe") == "cigarette"
    assert parse_answer("empty hand", "describe") == "hand"
    assert parse_answer("hand covering mouth", "describe") == "hand"
    assert parse_answer("a red beer can", "describe") == "can"
    assert parse_answer("phone held to ear", "describe") == "phone"
    assert parse_answer("plastic water bottle", "describe") == "bottle"
    assert parse_answer("blurry", "describe") == "unclear"
    assert parse_answer("Yes.", "yesno") == "cigarette"
    assert parse_answer("No", "yesno") == "hand"
    assert parse_answer("", "yesno") == "unclear"


def test_prompt_selection_and_apply(cfg, tmp_path):
    from smokedet.verify import PROMPTS, prompt_of

    assert prompt_of(cfg)[0] in PROMPTS
    p = tmp_path / "c.yaml"
    p.write_text(DEFAULT_CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    apply_to_config(p, {"vlm.prompt": "describe"})
    import yaml
    assert yaml.safe_load(p.read_text(encoding="utf-8"))["vlm"]["prompt"] == "describe"


def test_verify_cache_hits_after_json_roundtrip(cfg, tmp_path, monkeypatch):
    """Кэш проверки читается повторно: в ключе кортежи, а после JSON — списки."""
    from smokedet import verify

    reqs = [verify.Request(1, 4.5, (0, 0, 10, 10)), verify.Request(1, 5.0, (0, 0, 10, 10))]
    calls = []

    def fake_crops(video, rs, size):
        calls.append(len(rs))
        return [(r, None) for r in rs]

    monkeypatch.setattr(verify, "plan_requests", lambda *a, **k: reqs)
    monkeypatch.setattr(verify, "extract_crops", fake_crops)
    cfg["vlm"]["enabled"] = False
    cfg["object_veto"]["enabled"] = False
    path = tmp_path / "verify.json"
    first = verify.run_verification("none.mp4", {}, cfg, 64, 64, "cpu", path, log=lambda *_: None)
    assert path.is_file() and len(first["items"]) == 2 and calls == [2]
    second = verify.run_verification("none.mp4", {}, cfg, 64, 64, "cpu", path, log=lambda *_: None)
    assert calls == [2]                              # второй раз кадры не извлекаются: ответ из кэша
    assert second["items"] == first["items"]
