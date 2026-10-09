"""Правила событий из SPEC §3–4: два цикла за 20 с или один цикл с прямым признаком;
циклы одного человека объединяются при паузе до 15 с; разные люди — разные события."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .cycles import Cycle


@dataclass
class Event:
    track_id: int
    cycles: list = field(default_factory=list)
    start: float = 0.0
    end: float = 0.0
    peak: float = 0.0
    confidence: float = 0.0
    has_object: bool = False
    box: tuple = (0.0, 0.0, 0.0, 0.0)
    median_height: float = 0.0
    confirm: float = 0.0          # когда правило выполнилось (для видео: тревога с этого момента)

    def as_output(self, camera_id: str, clip_id: str, event_id: str) -> dict:
        """Запись ровно в формате SPEC §2."""
        x1, y1, x2, y2 = (float(v) for v in self.box)
        return {
            "camera_id": camera_id,
            "clip_id": clip_id,
            "event_id": event_id,
            "start_sec": round(float(self.start), 3),
            "end_sec": round(float(self.end), 3),
            "confidence": round(float(self.confidence), 3),
            "label": "smoking_like",
            "person_track_id": int(self.track_id),
            "peak_sec": round(float(self.peak), 3),
            "x1": round(x1, 1), "y1": round(y1, 1), "x2": round(x2, 1), "y2": round(y2, 1),
        }


def _chains(cycles: list[Cycle], merge_gap_s: float) -> list[list[Cycle]]:
    chains: list[list[Cycle]] = []
    for c in sorted(cycles, key=lambda c: c.t_start):
        if chains and c.t_start - chains[-1][-1].t_end <= merge_gap_s:
            chains[-1].append(c)
        else:
            chains.append([c])
    return chains


def _confirm_time(chain: list[Cycle], window_s: float, min_cycles: int, allow_single: bool) -> float | None:
    """Момент, когда правило выполнилось (конец паузы подтверждающего цикла), или None."""
    times = []
    if allow_single:
        times += [c.t_exit for c in chain if c.verdict == "smoke"]
    # циклы не пересекаются и отсортированы, поэтому достаточно соседних групп
    for i in range(len(chain) - min_cycles + 1):
        if chain[i + min_cycles - 1].t_end - chain[i].t_start <= window_s + 1e-9:
            times.append(chain[i + min_cycles - 1].t_exit)
            break
    return min(times) if times else None


def _qualifies(chain: list[Cycle], window_s: float, min_cycles: int, allow_single: bool) -> bool:
    return _confirm_time(chain, window_s, min_cycles, allow_single) is not None


def confidence_for(chain: list[Cycle]) -> float:
    n = len(chain)
    has_obj = any(c.verdict == "smoke" for c in chain)
    base = {1: 0.45, 2: 0.6, 3: 0.7}.get(n, 0.8)
    return float(min(0.99, base + (0.2 if has_obj else 0.0)))


def build_events(cycles_by_track: dict[int, list[Cycle]], cfg: dict) -> list[Event]:
    e = cfg["events"]
    window_s, merge_gap = float(e["window_s"]), float(e["merge_gap_s"])
    min_cycles = int(e["min_cycles"])
    allow_single = bool(e.get("allow_single_with_object", True))
    events: list[Event] = []
    for tid, cycles in cycles_by_track.items():
        valid = [c for c in cycles if c.verdict != "veto"]
        for chain in _chains(valid, merge_gap):
            confirm = _confirm_time(chain, window_s, min_cycles, allow_single)
            if confirm is None:
                continue
            # главный момент — самая уверенная пауза: с сигаретой по VLM, иначе с минимальным d
            def rank(c: Cycle):
                return (c.verdict == "smoke", -(c.d_min if not np.isnan(c.d_min) else 9.0))
            best = max(chain, key=rank)
            events.append(Event(
                track_id=tid, cycles=chain, start=chain[0].t_start, end=chain[-1].t_end,
                peak=best.t_peak, confidence=confidence_for(chain),
                has_object=any(c.verdict == "smoke" for c in chain), confirm=confirm,
            ))
    events.sort(key=lambda ev: (ev.start, ev.track_id))
    return events
