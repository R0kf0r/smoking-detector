"""Видео с отметками по docs/DESIGN.md.

Кадр: рамки людей по состоянию (сталь → янтарь «рука у лица» → кармин «похоже на курение»),
ярлыки шрифтом Golos Text, счётчик циклов, таймер паузы у рта, затемнение вне зоны интереса.
Под кадром: полоса времени клипа и «след жеста» — график расстояния кисть — рот.
Экспорт — H.264 (yuv420p) через ffmpeg.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PROJECT_ROOT
from .video import iter_frames, probe

GRAPHITE, SURFACE, LINE = "#1B242B", "#24313A", "#33424C"
TEXT, TEXT2 = "#E8EEF1", "#9FB0BA"
STEEL, AMBER, CARMINE = "#8FA3B0", "#F2B544", "#E5484D"


def rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def bgr(h: str) -> tuple[int, int, int]:
    r, g, b = rgb(h)
    return (b, g, r)


def fmt_num(x: float, digits: int = 1) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


def fmt_clock(sec: float) -> str:
    sec = max(0, int(sec))
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


# ---------------------------------------------------------------- данные для отрисовки

@dataclass
class TrackView:
    track_id: int
    t: np.ndarray                     # [n] время детекций
    box: np.ndarray                   # [n, 4] сглаженные рамки
    mouth: np.ndarray                 # [n, 2] точка рта (NaN, если не видна)
    d: np.ndarray                     # [n] сигнал кисть — рот (NaN, если нет)
    scale: np.ndarray                 # [n] масштаб человека, px
    cycles: list = field(default_factory=list)     # dict: t_start, t_enter, t_exit, t_end, verdict, reason
    events: list = field(default_factory=list)     # dict: start, end, confidence
    at_mouth: list = field(default_factory=list)   # [(t_in, t_out)]
    small: bool = False               # ниже порога видимости (IGNORE по правилам)


@dataclass
class RenderPlan:
    tracks: dict
    events: list                      # dict: track_id, start, end, confidence
    duration: float
    enter: float
    exit: float
    max_pause: float = 5.0
    roi: np.ndarray | None = None     # bool-маска H×W в исходном разрешении
    default_track: int | None = None


def mouth_intervals(t: np.ndarray, d: np.ndarray, enter: float, exit_: float,
                    max_nan_s: float = 2.0) -> list[tuple[float, float]]:
    """Интервалы «рука у рта» по тому же гистерезису, что и автомат циклов."""
    out: list[tuple[float, float]] = []
    at, t_in, nan_since = False, 0.0, None
    for ti, x in zip(t, d):
        if np.isnan(x):
            if at:
                if nan_since is None:
                    nan_since = ti
                elif ti - nan_since > max_nan_s:
                    out.append((t_in, nan_since))
                    at, nan_since = False, None
            continue
        nan_since = None
        if not at and x < enter:
            at, t_in = True, float(ti)
        elif at and x > exit_:
            out.append((t_in, float(ti)))
            at = False
    if at and len(t):
        out.append((t_in, float(t[-1])))
    return out


def _smooth_boxes(t: np.ndarray, box: np.ndarray, window_s: float = 0.3, gap_s: float = 0.5) -> np.ndarray:
    """Центрированное скользящее среднее рамок внутри непрерывных кусков трека (без задержки)."""
    if len(t) < 3:
        return box.astype(float).copy()
    dt = np.diff(t)
    seg = np.concatenate([[0], np.cumsum(dt > gap_s)])
    med = float(np.median(dt[dt > 1e-6])) if np.any(dt > 1e-6) else 1 / 15
    w = max(1, int(round(window_s / max(med, 1e-3))))
    w = w if w % 2 == 1 else w + 1
    df = pd.DataFrame(box, columns=["x1", "y1", "x2", "y2"])
    df["seg"] = seg
    sm = df.groupby("seg")[["x1", "y1", "x2", "y2"]].transform(
        lambda s: s.rolling(w, center=True, min_periods=1).mean())
    return sm.to_numpy(dtype=float)


def build_plan(tracks: dict, signals: dict, cycles_by_track: dict, events: list, cfg: dict,
               duration: float, roi: np.ndarray | None = None) -> RenderPlan:
    """Собирает всё, что нужно для отрисовки, из результатов конвейера.

    tracks: {id: Track}; signals: {id: DataFrame track_signal}; cycles_by_track: {id: [Cycle]};
    events: [Event] (после фильтров).
    """
    c = cfg["cycles"]
    enter, exit_ = float(c["enter_threshold"]), float(c["exit_threshold"])
    min_h = float(cfg["geometry"]["min_person_height_px"])
    views: dict[int, TrackView] = {}
    for tid, tr in tracks.items():
        sig = signals.get(tid)
        n = len(tr.t)
        if sig is not None and len(sig) == n:
            mouth = sig[["mouth_x", "mouth_y"]].to_numpy(float)
            d = sig["d"].to_numpy(float)
            scale = sig["scale"].to_numpy(float)
        else:
            mouth = np.full((n, 2), np.nan)
            d = np.full(n, np.nan)
            h = tr.box[:, 3] - tr.box[:, 1]
            scale = 0.3 * h
        cyc = [{"t_start": cy.t_start, "t_enter": cy.t_enter, "t_exit": cy.t_exit, "t_end": cy.t_end,
                "verdict": cy.verdict, "reason": cy.reason} for cy in cycles_by_track.get(tid, [])]
        # нумерация «цикл 1, цикл 2…» внутри цепочки (разрыв больше merge_gap — счёт заново)
        prev_end, k = None, 0
        for cd in sorted(cyc, key=lambda q: q["t_start"]):
            if cd["verdict"] == "veto":
                continue
            k = 1 if prev_end is None or cd["t_start"] - prev_end > float(cfg["events"]["merge_gap_s"]) else k + 1
            cd["num"], prev_end = k, cd["t_end"]
        evs = [{"start": e.start, "end": e.end, "confidence": e.confidence, "confirm": getattr(e, "confirm", e.start)}
               for e in events if e.track_id == tid]
        heights = tr.box[:, 3] - tr.box[:, 1]
        views[tid] = TrackView(
            track_id=int(tid), t=np.asarray(tr.t, float), box=_smooth_boxes(tr.t, tr.box), mouth=mouth,
            d=d, scale=scale, cycles=cyc, events=evs,
            at_mouth=mouth_intervals(tr.t, d, enter, exit_, float(c["max_nan_at_mouth_s"])),
            small=bool(np.nanmedian(heights) < min_h) if n else True,
        )
    ev_list = [{"track_id": int(e.track_id), "start": float(e.start), "end": float(e.end),
                "confidence": float(e.confidence)} for e in events]
    return RenderPlan(tracks=views, events=ev_list, duration=float(duration), enter=enter, exit=exit_,
                      max_pause=float(c["max_pause_s"]), roi=roi, default_track=_default_track(views))


def _default_track(views: dict) -> int | None:
    """Кого показывать в «следе жеста», когда рядом нет события: больше всего циклов."""
    best, best_key = None, None
    for tid, v in views.items():
        valid = sum(1 for c in v.cycles if c["verdict"] != "veto")
        key = (len(v.events), valid, len(v.cycles), len(v.t))
        if (valid or v.cycles) and (best_key is None or key > best_key):
            best, best_key = tid, key
    return best


# ---------------------------------------------------------------- текст и плашки

class Painter:
    """Текст через Pillow шрифтом Golos Text; готовые плашки кэшируются."""

    def __init__(self, frame_h: int, cfg: dict):
        from PIL import ImageFont

        r = cfg.get("render", {})
        size = max(11, int(round(float(r.get("text_size", 0.022)) * frame_h)))
        reg = PROJECT_ROOT / r.get("font_regular", "static/GolosText-Regular.ttf")
        bold = PROJECT_ROOT / r.get("font_bold", "static/GolosText-SemiBold.ttf")
        try:
            self.font = ImageFont.truetype(str(reg), size)
            self.font_bold = ImageFont.truetype(str(bold), size)
            self.font_small = ImageFont.truetype(str(reg), max(10, int(size * 0.8)))
        except OSError:                       # шрифта нет — стандартный (без кириллицы может быть хуже)
            self.font = self.font_bold = self.font_small = ImageFont.load_default()
        self.size = size
        self.pad_x = max(4, int(round(size * 0.42)))
        self.pad_y = max(2, int(round(size * 0.22)))
        self.radius = max(3, int(round(size * 0.27)))
        self._cache: dict = {}

    def _font(self, kind: str):
        return {"bold": self.font_bold, "small": self.font_small}.get(kind, self.font)

    def text_width(self, text: str, kind: str = "regular") -> int:
        return int(np.ceil(self._font(kind).getlength(text)))

    def pill(self, text: str, fg: str, bg: str | None, kind: str = "regular", alpha: float = 1.0,
             min_width: int = 0, align: str = "left"):
        """RGBA-плашка с текстом: (bgr float32 [h,w,3], alpha float32 [h,w,1])."""
        key = (text, fg, bg, kind, alpha, min_width, align)
        if key in self._cache:
            return self._cache[key]
        from PIL import Image, ImageDraw

        font = self._font(kind)
        tw = int(np.ceil(font.getlength(text)))
        h = int(round(self.size * (0.8 if kind == "small" else 1.0) * 1.15)) + 2 * self.pad_y
        pad = self.pad_x if bg else 1
        w = max(min_width, tw + 2 * pad)
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        if bg:
            draw.rounded_rectangle([0, 0, w - 1, h - 1], radius=self.radius,
                                   fill=rgb(bg) + (int(255 * alpha),))
        x = pad if align == "left" else (w - tw) / 2 if align == "center" else w - pad - tw
        draw.text((x, h / 2), text, font=font, fill=rgb(fg) + (255,), anchor="lm")
        arr = np.asarray(img).astype(np.float32)
        out = (arr[..., [2, 1, 0]], arr[..., 3:4] / 255.0)
        self._cache[key] = out
        return out


def blit(frame: np.ndarray, sprite, x: int, y: int) -> None:
    color, a = sprite
    h, w = a.shape[:2]
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    src_c = color[y0 - y:y1 - y, x0 - x:x1 - x]
    src_a = a[y0 - y:y1 - y, x0 - x:x1 - x]
    dst = frame[y0:y1, x0:x1].astype(np.float32)
    frame[y0:y1, x0:x1] = (src_c * src_a + dst * (1.0 - src_a)).astype(np.uint8)


def _blend_rect(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, color: str, alpha: float) -> None:
    H, W = img.shape[:2]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if x1 <= x0 or y1 <= y0:
        return
    region = img[y0:y1, x0:x1].astype(np.float32)
    img[y0:y1, x0:x1] = (region * (1 - alpha) + np.array(bgr(color), np.float32) * alpha).astype(np.uint8)


def _dashed_hline(img, x0: int, x1: int, y: int, color, dash: int, gap: int, thickness: int = 1) -> None:
    import cv2

    x = x0
    while x < x1:
        cv2.line(img, (x, y), (min(x + dash, x1), y), color, thickness, cv2.LINE_AA)
        x += dash + gap


# ---------------------------------------------------------------- выборка трека в момент τ

def _sample(v: TrackView, tau: float, max_gap: float = 0.6, edge: float = 0.12):
    """Рамка, рот, d и масштаб трека в момент τ (линейная интерполяция) или None."""
    t = v.t
    n = len(t)
    if n == 0 or tau < t[0] - edge or tau > t[-1] + edge:
        return None
    j = int(np.searchsorted(t, tau))
    if j < n and abs(t[j] - tau) < 1e-6:
        return v.box[j], v.mouth[j], v.d[j], v.scale[j]
    i0, i1 = j - 1, j
    if 0 <= i0 and i1 < n and t[i1] - t[i0] <= max_gap:
        w = (tau - t[i0]) / max(1e-9, t[i1] - t[i0])
        box = v.box[i0] * (1 - w) + v.box[i1] * w
        m0, m1 = v.mouth[i0], v.mouth[i1]
        if np.isnan(m0).any() or np.isnan(m1).any():
            mouth = m0 if not np.isnan(m0).any() and w < 0.5 else (m1 if not np.isnan(m1).any() else m0)
        else:
            mouth = m0 * (1 - w) + m1 * w
        d = v.d[i0] * (1 - w) + v.d[i1] * w if not (np.isnan(v.d[i0]) or np.isnan(v.d[i1])) else np.nan
        return box, mouth, d, v.scale[i0] * (1 - w) + v.scale[i1] * w
    k = i0 if i0 >= 0 and (i1 >= n or tau - t[i0] < t[i1] - tau) else i1
    if 0 <= k < n and abs(t[k] - tau) <= edge:
        return v.box[k], v.mouth[k], v.d[k], v.scale[k]
    return None


# ---------------------------------------------------------------- отрисовка

class Renderer:
    def __init__(self, plan: RenderPlan, cfg: dict, src_w: int, src_h: int):
        import cv2

        r = cfg.get("render", {})
        self.plan, self.cfg = plan, cfg
        max_h = int(r.get("max_height", 1080) or 0)
        self.s = min(1.0, max_h / src_h) if max_h and src_h > max_h else 1.0
        self.W = int(round(src_w * self.s)) // 2 * 2
        self.H = int(round(src_h * self.s)) // 2 * 2
        self.p = Painter(self.H, cfg)
        self.thick = max(2, int(round(2 * self.H / 1080)))
        self.show_trace = bool(r.get("trace_panel", True))
        self.strip_h = int(round(0.06 * self.H)) // 2 * 2
        self.panel_h = (int(round(0.30 * self.H)) // 2 * 2) if self.show_trace else 0
        self.out_h = self.H + self.strip_h + self.panel_h
        # затемнение вне зоны интереса и пунктир по границе
        self.mul = None
        self.border = None
        if plan.roi is not None:
            mask = cv2.resize(plan.roi.astype(np.uint8), (self.W, self.H), interpolation=cv2.INTER_NEAREST) > 0
            if not mask.all():
                mul = np.where(mask, 255, int(round(255 * 0.65))).astype(np.uint8)
                self.mul = np.repeat(mul[:, :, None], 3, axis=2)
                layer = np.zeros((self.H, self.W), np.uint8)
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                dash = max(6, int(round(self.H * 0.012)))
                for cnt in contours:
                    pts = cnt[:, 0, :]
                    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
                    pos = np.concatenate([[0], np.cumsum(seg)])
                    on = (pos // dash) % 2 == 0
                    for a in range(len(pts) - 1):
                        if on[a]:
                            cv2.line(layer, tuple(int(v) for v in pts[a]), tuple(int(v) for v in pts[a + 1]),
                                     255, max(1, self.thick // 2), cv2.LINE_AA)
                ys, xs = np.nonzero(layer)
                self.border = (ys, xs, (layer[ys, xs].astype(np.float32) / 255.0 * 0.6)[:, None])
        self._strip_base = self._make_strip_base()
        self._panel_cache: dict = {}

    # ---- кадр
    def _state(self, v: TrackView, tau: float):
        # тревога — с момента, когда правило выполнилось, до конца события (как увидел бы оператор)
        ev = next((e for e in v.events if e.get("confirm", e["start"]) <= tau <= e["end"]), None)
        mi = next(((a, b) for a, b in v.at_mouth if a <= tau <= b), None)
        recent = [c for c in v.cycles if c["verdict"] != "veto" and c["t_exit"] <= tau and tau - c["t_exit"] <= 20.0]
        dots = min(2, len(recent))
        veto = None
        if mi is not None:
            for c in v.cycles:
                if c["verdict"] == "veto" and c["t_enter"] - 0.3 <= mi[0] <= c["t_exit"] + 0.3:
                    veto = c["reason"].replace("предмет: ", "")
        return ev, mi, dots, veto

    def draw_frame(self, img: np.ndarray, tau: float) -> np.ndarray:
        import cv2

        if self.s != 1.0:
            img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA)
        elif img.shape[1] != self.W or img.shape[0] != self.H:
            img = img[:self.H, :self.W]
        img = np.ascontiguousarray(img)
        if self.mul is not None:
            img = cv2.multiply(img, self.mul, scale=1 / 255.0)
        if self.border is not None:
            ys, xs, a = self.border
            img[ys, xs] = (img[ys, xs].astype(np.float32) * (1 - a) + np.array(bgr(TEXT), np.float32) * a
                           ).astype(np.uint8)
        s = self.s
        p = self.p
        for tid, v in self.plan.tracks.items():
            smp = _sample(v, tau)
            if smp is None:
                continue
            box, mouth, _d, scale = smp
            x1, y1, x2, y2 = (int(round(c * s)) for c in box)
            ev, mi, dots, veto = self._state(v, tau)
            if ev is not None:
                color, state = CARMINE, f"похоже на курение {fmt_num(ev['confidence'], 2)}"
            elif mi is not None:
                color = AMBER
                state = f"рука у лица {fmt_num(tau - mi[0])} с" + (f", {veto}" if veto else "")
            else:
                color, state = STEEL, ""
            cv2.rectangle(img, (x1, y1), (x2, y2), bgr(color), self.thick, cv2.LINE_AA)
            if v.small and not state:
                continue                       # мелкие люди (IGNORE): только рамка
            fg = GRAPHITE
            tag = p.pill(f"ID {tid}", fg, color, kind="bold")
            th = tag[1].shape[0]
            ly = y1 - th - max(2, self.thick)
            if ly < 0:
                ly = y1 + max(2, self.thick)
            blit(img, tag, x1, ly)
            if state:
                blit(img, p.pill(state, fg, color), x1 + tag[1].shape[1] + max(3, p.pad_x // 2), ly)
            if dots or mi is not None or ev is not None:
                r = max(3, int(round(p.size * 0.2)))
                cy = ly - r - max(3, p.pad_y)
                if cy - r < 0:
                    cy = ly + th + r + max(3, p.pad_y)
                n_filled = 2 if ev is not None else dots
                for k in range(2):
                    cx = x1 + r + 1 + k * int(round(2.8 * r))
                    if k < n_filled:
                        cv2.circle(img, (cx, cy), r, bgr(color), -1, cv2.LINE_AA)
                    else:
                        cv2.circle(img, (cx, cy), r, bgr(color), max(1, self.thick // 2), cv2.LINE_AA)
            if mi is not None and not np.isnan(mouth).any():
                frac = float(np.clip((tau - mi[0]) / max(0.1, self.plan.max_pause), 0, 1))
                rad = max(6, int(round(0.22 * scale * s)))
                center = (int(round(mouth[0] * s)), int(round(mouth[1] * s)))
                cv2.ellipse(img, center, (rad, rad), -90, 0, 360 * frac, bgr(AMBER), self.thick, cv2.LINE_AA)
        parts = [img, self._draw_strip(tau)]
        if self.show_trace:
            parts.append(self._draw_panel(tau))
        return np.vstack(parts)

    # ---- полоса времени
    def _bar_geom(self):
        p = self.p
        time_w = p.text_width("00:00 / 00:00", "regular") + 2 * p.pad_x + 4
        x0 = int(round(0.012 * self.W))
        x1 = self.W - x0 - time_w - p.pad_x
        yc = self.strip_h // 2
        return x0, x1, yc, time_w

    def _tx(self, t: float, x0: int, x1: int) -> int:
        dur = max(1e-6, self.plan.duration)
        return int(round(x0 + (x1 - x0) * min(1.0, max(0.0, t / dur))))

    def _make_strip_base(self) -> np.ndarray:
        import cv2

        strip = np.zeros((self.strip_h, self.W, 3), np.uint8)
        strip[:] = bgr(GRAPHITE)
        x0, x1, yc, _ = self._bar_geom()
        track_h = max(4, int(round(0.012 * self.H)))
        cv2.rectangle(strip, (x0, yc - track_h // 2), (x1, yc + track_h // 2), bgr(LINE), -1, cv2.LINE_AA)
        ev_h = max(6, int(round(0.018 * self.H)))
        for e in self.plan.events:
            a, b = self._tx(e["start"], x0, x1), self._tx(e["end"], x0, x1)
            cv2.rectangle(strip, (a, yc - ev_h // 2), (max(a + 2, b), yc + ev_h // 2), bgr(CARMINE), -1, cv2.LINE_AA)
        tick_h = max(8, int(round(0.03 * self.H)))
        for v in self.plan.tracks.values():
            for c in v.cycles:
                if c["verdict"] == "veto":
                    continue
                x = self._tx((c["t_enter"] + c["t_exit"]) / 2, x0, x1)
                cv2.line(strip, (x, yc - tick_h // 2), (x, yc + tick_h // 2), bgr(AMBER), max(2, self.thick),
                         cv2.LINE_AA)
        return strip

    def _draw_strip(self, tau: float) -> np.ndarray:
        import cv2

        strip = self._strip_base.copy()
        x0, x1, yc, time_w = self._bar_geom()
        cur_h = max(10, int(round(0.036 * self.H)))
        x = self._tx(tau, x0, x1)
        cv2.line(strip, (x, yc - cur_h // 2), (x, yc + cur_h // 2), bgr(TEXT), max(2, self.thick), cv2.LINE_AA)
        label = f"{fmt_clock(tau)} / {fmt_clock(self.plan.duration)}"
        sp = self.p.pill(label, TEXT, SURFACE, min_width=time_w, align="center")
        blit(strip, sp, self.W - time_w - x0, yc - sp[1].shape[0] // 2)
        return strip

    # ---- «след жеста»
    def _featured(self, tau: float):
        best = None
        for e in self.plan.events:
            if e["start"] - 3 <= tau <= e["end"] + 3 and (best is None or e["confidence"] > best["confidence"]):
                best = e
        return best["track_id"] if best else self.plan.default_track

    def _window(self, tau: float) -> tuple[float, float]:
        dur = max(1e-6, self.plan.duration)
        span = min(dur, 30.0)
        a = min(max(0.0, tau - span / 2), max(0.0, dur - span))
        if dur <= 30.0:
            a = 0.0
        return round(a, 2), span

    def _draw_panel(self, tau: float) -> np.ndarray:
        import cv2

        tid = self._featured(tau)
        a, span = self._window(tau)
        key = (tid, a)
        base = self._panel_cache.get(key)
        if base is None:
            if len(self._panel_cache) > 64:
                self._panel_cache.clear()
            base = self._panel_static(tid, a, span)
            self._panel_cache[key] = base
        panel, gx0, gx1, gy0, gy1 = base
        panel = panel.copy()
        if a <= tau <= a + span and tid is not None:
            x = int(round(gx0 + (gx1 - gx0) * (tau - a) / span))
            cv2.line(panel, (x, gy0), (x, gy1), bgr(TEXT), max(1, self.thick // 2 + 1), cv2.LINE_AA)
        return panel

    def _panel_static(self, tid, a: float, span: float):
        import cv2

        p = self.p
        W, Hp = self.W, self.panel_h
        panel = np.zeros((Hp, W, 3), np.uint8)
        panel[:] = bgr(GRAPHITE)
        m = int(round(0.012 * self.W))
        cv2.rectangle(panel, (m, 0), (W - m, Hp - m), bgr(SURFACE), -1)
        title = p.pill(f"След жеста: ID {tid}" if tid is not None else "След жеста", TEXT, None, kind="bold")
        blit(panel, title, m + p.pad_x, int(p.pad_y * 1.5))
        sub = p.pill("расстояние кисть — рот в длинах торса; ниже пунктира — рука у рта", TEXT2, None, kind="small")
        ty = int(p.pad_y * 1.5) + title[1].shape[0]
        blit(panel, sub, m + p.pad_x, ty)
        gx0 = m + p.pad_x + p.text_width("0,5", "small") + p.pad_x
        gx1 = W - m - 2 * p.pad_x
        gy0 = ty + sub[1].shape[0] + int(1.6 * p.size)      # место под полосу события
        gy1 = Hp - m - int(1.4 * p.size)                      # место под подписи времени
        if gy1 - gy0 < 20 or tid is None or tid not in self.plan.tracks:
            if tid is None:
                msg = p.pill("Никто не подносил руку ко рту: циклов нет", TEXT2, None)
                blit(panel, msg, m + p.pad_x, (Hp - msg[1].shape[0]) // 2)
            return panel, gx0, gx1, max(gy0, 0), max(gy1, gy0 + 1)
        v = self.plan.tracks[tid]
        dmax = 0.8          # выше — «рука далеко», линия упирается в верх; так виден диапазон затяжек

        def X(t):
            return gx0 + (gx1 - gx0) * (np.asarray(t, float) - a) / span

        def Y(d):
            return gy1 - (gy1 - gy0) * np.clip(np.asarray(d, float), 0, dmax) / dmax

        # сетка и подписи оси
        for val in (0.0, 0.2, 0.4, 0.6, 0.8):
            y = int(round(float(Y(val))))
            cv2.line(panel, (gx0, y), (gx1, y), bgr(LINE), 1, cv2.LINE_AA)
            lab = p.pill(fmt_num(val, 1).replace(",0", "") if val else "0", TEXT2, None, kind="small")
            blit(panel, lab, gx0 - lab[1].shape[1] - p.pad_x // 2, y - lab[1].shape[0] // 2)
        step = 5 if span <= 30 else 10
        t_tick = np.ceil(a / step) * step
        while t_tick <= a + span + 1e-6:
            x = int(round(float(X(t_tick))))
            lab = p.pill(fmt_clock(t_tick), TEXT2, None, kind="small")
            blit(panel, lab, x - lab[1].shape[1] // 2, gy1 + p.pad_y)
            t_tick += step
        # паузы у рта по циклам: янтарная заливка ~22%
        for c in v.cycles:
            if c["t_exit"] < a or c["t_enter"] > a + span:
                continue
            xa, xb = int(round(float(X(max(a, c["t_enter"]))))), int(round(float(X(min(a + span, c["t_exit"])))))
            _blend_rect(panel, xa, gy0, max(xa + 2, xb), gy1, AMBER, 0.22)
            if c["verdict"] == "veto":
                lab = p.pill(c["reason"].replace("предмет: ", "") or "вето", TEXT2, None, kind="small")
            else:
                lab = p.pill(f"цикл {c.get('num', 1)}", TEXT, None, kind="bold")
            blit(panel, lab, xa + 2, gy0 + 2)
        # пороги гистерезиса — пунктир
        dash, gap = max(6, p.size // 2), max(4, p.size // 3)
        for thr in (self.plan.enter, self.plan.exit):
            _dashed_hline(panel, gx0, gx1, int(round(float(Y(thr)))), bgr(TEXT2), dash, gap)
        # сигнал
        sel = (v.t >= a - 0.5) & (v.t <= a + span + 0.5)
        t, d = v.t[sel], v.d[sel]
        ok = ~np.isnan(d)
        if ok.sum() >= 2:
            xs, ys = X(t), Y(d)
            brk = np.where(~ok)[0]
            start = 0
            for stop in list(brk) + [len(d)]:
                idx = np.arange(start, stop)
                if len(idx) >= 2:
                    pts = np.stack([xs[idx], ys[idx]], axis=1).round().astype(np.int32)
                    cv2.polylines(panel, [pts], False, bgr(TEXT2), max(1, self.thick // 2 + 1), cv2.LINE_AA)
                start = stop + 1
        # события — карминовая полоса над графиком с подписью
        bar_y = gy0 - int(0.7 * p.size)
        for e in v.events:
            if e["end"] < a or e["start"] > a + span:
                continue
            xa = int(round(float(X(max(a, e["start"])))))
            xb = int(round(float(X(min(a + span, e["end"])))))
            cv2.line(panel, (xa, bar_y), (max(xa + 2, xb), bar_y), bgr(CARMINE), max(4, int(0.012 * self.H)),
                     cv2.LINE_AA)
            lab = p.pill(f"событие {fmt_clock(e['start'])}–{fmt_clock(e['end'])}", TEXT, None, kind="bold")
            lx = xb + p.pad_x
            if lx + lab[1].shape[1] > gx1:
                lx = max(gx0, xa - p.pad_x - lab[1].shape[1])
            blit(panel, lab, lx, bar_y - lab[1].shape[0] // 2)
        return panel, gx0, gx1, gy0, gy1


# ---------------------------------------------------------------- кодирование

_ENCODER: str | None = None


def _encoder_args(name: str, w: int, h: int, fps: float, crf: int) -> list[str]:
    kbps = int(max(1500, min(20000, w * h * max(fps, 1) * 0.1 / 1000)))
    if name == "libx264":
        return ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf)]
    if name == "h264_videotoolbox":
        return ["-c:v", "h264_videotoolbox", "-b:v", f"{kbps}k", "-allow_sw", "1"]
    if name == "libopenh264":
        return ["-c:v", "libopenh264", "-b:v", f"{kbps}k"]
    return ["-c:v", "mpeg4", "-q:v", "3"]


def pick_encoder() -> str:
    """Первый работающий кодировщик: libx264 → h264_videotoolbox → libopenh264 → mpeg4."""
    global _ENCODER
    if _ENCODER:
        return _ENCODER
    listing = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    for name in ("libx264", "h264_videotoolbox", "libopenh264", "mpeg4"):
        if not re.search(rf"\s{re.escape(name)}\s", listing):
            continue
        with tempfile.TemporaryDirectory() as tmp:
            cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=gray:s=320x240:d=0.4:r=10",
                   *_encoder_args(name, 320, 240, 10, 23), "-pix_fmt", "yuv420p", str(Path(tmp) / "t.mp4")]
            if subprocess.run(cmd, capture_output=True).returncode == 0:
                _ENCODER = name
                return name
    raise RuntimeError("ffmpeg не нашёл ни одного кодировщика видео (libx264, videotoolbox, openh264, mpeg4)")


def render_video(video: str | Path, out_path: str | Path, plan: RenderPlan, cfg: dict, log=print) -> dict:
    video, out_path = Path(video), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    info = probe(video)
    fps = info["fps"] or 25.0
    rnd = Renderer(plan, cfg, info["width"], info["height"])
    encoder = pick_encoder()
    crf = int(cfg.get("render", {}).get("crf", 23))
    tmp_out = out_path.with_suffix(".part.mp4")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{rnd.W}x{rnd.out_h}", "-r", f"{fps:.6f}", "-i", "-",
           *_encoder_args(encoder, rnd.W, rnd.out_h, fps, crf), "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(tmp_out)]
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    n = 0
    try:
        for tau, img in iter_frames(video, 0.0):
            frame = rnd.draw_frame(img, tau)
            proc.stdin.write(frame.tobytes())
            n += 1
            if n % 500 == 0:
                log(f"  рендер: кадр {n}, {n / max(1e-6, time.time() - t0):.1f} кадр/с")
        proc.stdin.close()
    except BrokenPipeError:
        pass
    err = proc.stderr.read().decode("utf-8", "replace") if proc.stderr else ""
    code = proc.wait()
    if code != 0:
        raise RuntimeError(f"ffmpeg ({encoder}) завершился с кодом {code}: {err[-800:]}")
    tmp_out.replace(out_path)
    return {"frames": n, "seconds": round(time.time() - t0, 2), "encoder": encoder,
            "size": [rnd.W, rnd.out_h], "path": str(out_path)}
