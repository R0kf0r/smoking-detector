"""Автомат циклов на синтетическом сигнале d(t)."""

from __future__ import annotations

import numpy as np

from smokedet.cycles import detect_cycles

FPS = 15.0


def signal(segments: list[tuple[float, float]], duration: float = 30.0, low: float = 0.08, high: float = 1.2,
           ramp: float = 0.8):
    """d(t): high вдали от рта, low во время пауз [(a, b)], линейные подходы длиной ramp."""
    t = np.arange(0, duration, 1 / FPS)
    d = np.full_like(t, high)
    for a, b in segments:
        x = np.where(t < a, (a - t) / ramp, np.where(t > b, (t - b) / ramp, 0.0))
        d = np.minimum(d, low + (high - low) * np.clip(x, 0, 1))
    return t, d


def test_two_normal_cycles(cfg):
    t, d = signal([(5, 7), (12, 13.5)])
    cycles, rejected = detect_cycles(t, d, cfg, 1)
    assert len(cycles) == 2 and not rejected
    c = cycles[0]
    assert c.t_start < c.t_enter < c.t_exit < c.t_end
    assert abs(c.pause_s - 2.0) < 0.4
    assert 4.0 <= c.t_start <= 4.6                  # начало подхода руки
    assert 7.4 <= c.t_end <= 8.0                    # конец отвода


def test_long_pause_is_not_a_cycle(cfg):
    t, d = signal([(5, 13)])                        # 8 с у рта: телефон
    cycles, rejected = detect_cycles(t, d, cfg, 1)
    assert not cycles
    assert [r.reason for r in rejected] == ["too_long"]


def test_short_touch_is_not_a_cycle(cfg):
    t, d = signal([(5, 5.0)], ramp=0.15)            # касание без паузы
    cycles, rejected = detect_cycles(t, d, cfg, 1)
    assert not cycles
    assert rejected and rejected[0].reason == "too_short"


def test_hysteresis_ignores_jitter(cfg):
    t, d = signal([(5, 7)])
    c = cfg["cycles"]
    lo, hi = float(c["enter_threshold"]), float(c["exit_threshold"])
    mid = (t > 5.5) & (t < 6.5)
    wave = np.sin(np.arange(mid.sum()) * 2.0)
    # дрожание между порогами входа и выхода не рвёт цикл (гистерезис); относительный выход
    # выключаем: он нарочно режет устойчивый подъём d у лица между затяжками
    c["relative_exit"] = 0.0
    d[mid] = (lo + hi) / 2 + 0.4 * (hi - lo) / 2 * wave
    cycles, _ = detect_cycles(t, d, cfg, 1)
    assert len(cycles) == 1
    # мелкое дрожание у самого рта не рвёт цикл и с относительным выходом
    c["relative_exit"] = 0.08
    d[mid] = 0.10 + 0.03 * wave
    cycles, _ = detect_cycles(t, d, cfg, 1)
    assert len(cycles) == 1


def test_lost_points_at_mouth(cfg):
    t, d = signal([(5, 9.5)])
    d[(t > 5.5) & (t < 8.5)] = np.nan               # 3 с без точек у рта (> 2 с)
    cycles, rejected = detect_cycles(t, d, cfg, 1)
    assert not cycles
    assert rejected[0].reason == "lost_at_mouth"


def test_both_hands_fraction(cfg):
    t, d = signal([(5, 7)])
    cycles, _ = detect_cycles(t, d, cfg, 1, dl=d.copy(), dr=d.copy())
    assert cycles[0].both_hands_frac > 0.9
    cycles, _ = detect_cycles(t, d, cfg, 1, dl=d.copy(), dr=np.full_like(d, 1.0))
    assert cycles[0].both_hands_frac == 0.0


def test_track_starting_at_mouth(cfg):
    t, d = signal([(0, 2)])                         # трек начинается с рукой у рта
    d[t < 0.5] = 0.08
    cycles, _ = detect_cycles(t, d, cfg, 1)         # по умолчанию такой цикл засчитываем
    assert len(cycles) == 1 and cycles[0].t_start == 0.0
    cfg["cycles"]["require_visible_approach"] = True
    cycles, _ = detect_cycles(t, d, cfg, 1)
    assert not cycles


def test_real_data_levels(cfg):
    # уровни с реального ролика: затяжка d ≈ 0,22, между затяжками рука с сигаретой у лица d ≈ 0,32
    t = np.arange(0, 12, 1 / FPS)
    d = np.full_like(t, 1.3)
    d[(t > 1) & (t < 3)] = 0.22
    d[(t >= 3) & (t < 4)] = 0.32
    d[(t >= 4) & (t < 5.5)] = 0.23
    d[(t >= 5.5) & (t < 7)] = 0.33
    cycles, _ = detect_cycles(t, d, cfg, 1)
    assert len(cycles) == 2                         # относительный выход делит затяжки
    assert abs(cycles[0].t_exit - 3.0) < 0.1 and abs(cycles[1].t_enter - 4.0) < 0.1
    cfg["cycles"]["relative_exit"] = 0.0
    cycles, rejected = detect_cycles(t, d, cfg, 1)
    assert not cycles and rejected[0].reason == "too_long"   # без него — одна пауза 6 с


def test_relative_exit_ignores_noise(cfg):
    # телефон у уха 8 с: d шумит ±0,03 около 0,22 — относительный выход не срабатывает
    t = np.arange(0, 12, 1 / FPS)
    rng = np.random.default_rng(0)
    d = np.full_like(t, 1.3)
    mid = (t > 2) & (t < 10)
    d[mid] = 0.22 + rng.uniform(-0.03, 0.03, mid.sum())
    cycles, rejected = detect_cycles(t, d, cfg, 1)
    assert not cycles and rejected[0].reason == "too_long"
