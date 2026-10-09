"""Автомат циклов «подход → пауза у рта → отход» с гистерезисом (два порога)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Cycle:
    track_id: int
    t_start: float          # начало подхода руки ко рту
    t_enter: float          # d опустился ниже порога входа
    t_exit: float           # d поднялся выше порога выхода
    t_end: float            # конец отвода руки
    pause_s: float
    d_min: float
    t_peak: float           # середина паузы у рта
    both_hands_frac: float = 0.0
    votes: dict = field(default_factory=dict)   # метка VLM или детектора → число кадров
    verdict: str = "none"   # smoke — прямой признак; veto — не курение; none — без решения
    reason: str = ""

    def as_dict(self) -> dict:
        return {
            "track_id": int(self.track_id), "t_start": round(self.t_start, 3),
            "t_enter": round(self.t_enter, 3), "t_exit": round(self.t_exit, 3),
            "t_end": round(self.t_end, 3), "pause_s": round(self.pause_s, 3),
            "d_min": None if np.isnan(self.d_min) else round(float(self.d_min), 4),
            "t_peak": round(self.t_peak, 3), "both_hands_frac": round(self.both_hands_frac, 3),
            "votes": dict(self.votes), "verdict": self.verdict, "reason": self.reason,
        }


@dataclass
class Rejected:
    track_id: int
    t_enter: float
    t_exit: float
    pause_s: float
    reason: str             # too_short | too_long | lost_at_mouth | track_ended

    def as_dict(self) -> dict:
        return {"track_id": int(self.track_id), "t_enter": round(self.t_enter, 3),
                "t_exit": round(self.t_exit, 3), "pause_s": round(self.pause_s, 3),
                "reason": self.reason}


def _approach_start(t: np.ndarray, d: np.ndarray, i_enter: int, max_s: float, frac: float = 0.9) -> int:
    """Начало подхода руки: последний момент перед входом, когда d было на уровне «рука далеко»
    (frac пути от d при входе до максимума d за max_s секунд до входа)."""
    lo = int(np.searchsorted(t, t[i_enter] - max_s - 1e-9))
    seg = d[lo:i_enter + 1]
    if not np.any(~np.isnan(seg)):
        return i_enter
    peak, base = float(np.nanmax(seg)), float(d[i_enter])
    if peak - base < 0.05:          # рука уже была у лица: подхода не видно
        return i_enter
    idx = np.where(seg >= base + frac * (peak - base))[0]
    return lo + int(idx[-1]) if len(idx) else i_enter


def _retreat_end(t: np.ndarray, d: np.ndarray, i_exit: int, max_s: float, frac: float = 0.9) -> int:
    """Конец отвода руки: первый момент после выхода, когда d поднялось на frac пути до максимума."""
    hi = int(np.searchsorted(t, t[i_exit] + max_s + 1e-9))
    seg = d[i_exit:hi]
    if not np.any(~np.isnan(seg)):
        return i_exit
    peak, base = float(np.nanmax(seg)), float(d[i_exit])
    if peak - base < 0.05:
        return i_exit
    idx = np.where(seg >= base + frac * (peak - base))[0]
    return i_exit + int(idx[0]) if len(idx) else i_exit


def detect_cycles(t: np.ndarray, d: np.ndarray, cfg: dict, track_id: int,
                  dl: np.ndarray | None = None, dr: np.ndarray | None = None
                  ) -> tuple[list[Cycle], list[Rejected]]:
    """Находит циклы в сглаженном сигнале d(t) одного человека.

    Вход в состояние «у рта» — d < enter_threshold, выход — d > exit_threshold.
    Цикл засчитывается, если пауза у рта в пределах [min_pause_s, max_pause_s].
    После долгой потери точек у рта новый вход возможен, только когда рука снова была далеко
    (d > exit_threshold). Трек, начавшийся с рукой у рта, даёт цикл, если не включено
    require_visible_approach (короткие ролики часто начинаются посреди затяжки).
    """
    c = cfg["cycles"]
    enter, exit_ = float(c["enter_threshold"]), float(c["exit_threshold"])
    if exit_ < enter:
        raise ValueError("exit_threshold должен быть не меньше enter_threshold")
    min_p, max_p = float(c["min_pause_s"]), float(c["max_pause_s"])
    max_nan = float(c["max_nan_at_mouth_s"])
    rel = float(c.get("relative_exit", 0.0))          # 0 — выключено
    rel_hold = float(c.get("relative_exit_hold_s", 0.25))
    look_back, look_ahead = float(c["approach_lookback_s"]), float(c["retreat_lookahead_s"])

    cycles: list[Cycle] = []
    rejected: list[Rejected] = []
    at_mouth = False
    # вход разрешён только после того, как рука была далеко (d > exit); в начале трека — по настройке
    armed = not bool(c.get("require_visible_approach", False))
    i_enter = -1
    nan_since = None
    run_min, i_rise = np.inf, None
    n = len(d)
    for i in range(n):
        x = d[i]
        if np.isnan(x):
            if at_mouth:
                if nan_since is None:
                    nan_since = t[i]
                elif t[i] - nan_since > max_nan:
                    rejected.append(Rejected(track_id, t[i_enter], t[i], t[i] - t[i_enter], "lost_at_mouth"))
                    at_mouth, nan_since, armed = False, None, False
            continue
        nan_since = None
        if not at_mouth:
            if x > exit_:
                armed = True
            elif x < enter and armed:
                at_mouth, i_enter, run_min, i_rise = True, i, x, None
            continue
        run_min = min(run_min, x)
        i_out = None
        if x > exit_:
            i_out = i
        elif rel > 0 and x >= enter and x - run_min >= rel:
            # относительный выход: рука с сигаретой отошла от рта, но осталась у лица
            # (между затяжками), подъём держится не меньше rel_hold секунд
            if i_rise is None:
                i_rise = i
            if t[i] - t[i_rise] >= rel_hold:
                i_out = i_rise
        else:
            i_rise = None
        if i_out is not None:
            i = i_out
            at_mouth = False
            pause = float(t[i] - t[i_enter])
            if pause < min_p:
                rejected.append(Rejected(track_id, t[i_enter], t[i], pause, "too_short"))
                continue
            if pause > max_p:
                rejected.append(Rejected(track_id, t[i_enter], t[i], pause, "too_long"))
                continue
            i_start = _approach_start(t, d, i_enter, look_back)
            i_end = _retreat_end(t, d, i, look_ahead)
            seg = d[i_enter:i]
            d_min = float(np.nanmin(seg)) if np.any(~np.isnan(seg)) else float("nan")
            both = 0.0
            if dl is not None and dr is not None and i > i_enter:
                near = (dl[i_enter:i] < exit_) & (dr[i_enter:i] < exit_)
                both = float(np.mean(near))
            cycles.append(Cycle(
                track_id=track_id, t_start=float(t[i_start]), t_enter=float(t[i_enter]),
                t_exit=float(t[i]), t_end=float(t[i_end]), pause_s=pause, d_min=d_min,
                t_peak=float((t[i_enter] + t[i]) / 2.0), both_hands_frac=both,
            ))
    if at_mouth and i_enter >= 0:
        rejected.append(Rejected(track_id, t[i_enter], t[-1], t[-1] - t[i_enter], "track_ended"))
    return cycles, rejected
