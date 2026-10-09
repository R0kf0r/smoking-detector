"""Этап B: что у человека у рта. Локальная VLM (Qwen3-VL через mlx-vlm), детектор предметов COCO
и кинематика. Решение по каждому циклу — по большинству кадров паузы.

Кадры для проверки выбираются не по циклам, а по участкам, где рука близко ко рту
(d < scan_threshold). Поэтому результаты кэшируются и подходят для любых порогов циклов
при подборе (calibrate.py).
"""

from __future__ import annotations

import json
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .cycles import Cycle
from .video import frames_at

VLM_LABELS = ["cigarette", "vape", "lollipop", "toothpick", "pen", "phone", "bottle", "can", "cup", "food",
              "hand", "unclear"]
SMOKE = {"cigarette", "vape"}
# похожие на сигарету предметы (леденец на палочке, зубочистка, ручка) — сложные негативы N03
VETO = {"lollipop", "toothpick", "pen", "phone", "bottle", "can", "cup", "food"}

# Варианты промпта. Первый вариант (list_v1) на эталоне почти всегда отвечал «cigarette», даже на
# касание лица пустой рукой в перчатке, поэтому есть нейтральные варианты; какой лучше
# отделяет позитивные ролики от негативных, выбирает smokedet.vlm_probe.
PROMPTS = {
    "list_v1": {"kind": "list", "max_tokens": 6, "text": (
        "This is a crop from a street CCTV camera showing a person's face and hand. "
        "What is the person holding at or near the mouth? "
        "Answer with exactly one word from this list: "
        "cigarette, vape, lollipop, toothpick, pen, phone, bottle, can, cup, food, hand, unclear.")},
    "list_neutral": {"kind": "list", "max_tokens": 6, "text": (
        "This is a low-resolution crop from a street CCTV camera. Look at the person's hand near the face. "
        "People often just touch their face, cover their mouth, talk on the phone, drink or eat. "
        "Answer 'cigarette' or 'vape' only if you clearly see one. "
        "What is in the hand near the mouth? Answer with exactly one word from this list: "
        "nothing, glove, phone, bottle, can, cup, food, lollipop, toothpick, pen, cigarette, vape, unclear.")},
    "describe": {"kind": "describe", "max_tokens": 16, "text": (
        "This is a low-resolution crop from a street CCTV camera. In at most six words, describe what the "
        "person's hand near the face is holding. If the hand is empty, answer 'empty hand'.")},
    "yesno": {"kind": "yesno", "max_tokens": 3, "text": (
        "This is a low-resolution crop from a street CCTV camera. Is a cigarette or an electronic cigarette "
        "(vape) clearly visible in the person's hand or mouth? Answer only yes or no.")},
}
DEFAULT_PROMPT = "list_neutral"
VLM_PROMPT = PROMPTS[DEFAULT_PROMPT]["text"]

_KEYWORDS = [   # для ответа-описания: первая подходящая группа по приоритету
    ("cigarette", {"cigarette", "cigarettes", "cig", "cigar", "smoking", "smokes", "smoke", "joint"}),
    ("vape", {"vape", "vaping", "ecigarette", "e-cigarette", "vaporizer"}),
    ("phone", {"phone", "smartphone", "cellphone", "mobile", "telephone", "iphone"}),
    ("bottle", {"bottle", "flask", "thermos", "water"}),
    ("can", {"can", "beer", "soda"}),
    ("cup", {"cup", "glass", "mug", "coffee", "tea", "drink", "drinking", "juice"}),
    ("food", {"food", "eating", "sandwich", "apple", "banana", "snack", "candy", "chocolate", "bread",
              "burger", "gum", "chewing", "ice", "cream", "pie", "seeds"}),
    ("lollipop", {"lollipop"}),
    ("toothpick", {"toothpick"}),
    ("pen", {"pen", "pencil", "straw"}),
    ("hand", {"empty", "nothing", "hand", "hands", "glove", "gloves", "finger", "fingers", "fist", "mouth",
              "chin", "nose", "face", "covering", "touching", "scratching", "wiping", "mask", "scarf"}),
]


def parse_answer(text: str, kind: str = "list") -> str:
    """Ответ модели → метка из VLM_LABELS."""
    import re

    low = (text or "").lower()
    words = re.findall(r"[a-z][a-z-]*", low)
    if kind == "yesno":
        if words and words[0].startswith("yes"):
            return "cigarette"
        if words and words[0].startswith("no"):
            return "hand"
        return "unclear"
    if kind == "describe":
        wset = set(words)
        for label, keys in _KEYWORDS:
            if wset & keys:
                return label
        return "unclear"
    synonyms = {"nothing": "hand", "glove": "hand", "gloves": "hand"}
    for tok in words:
        if tok in synonyms:
            return synonyms[tok]
        for lbl in VLM_LABELS:
            if tok == lbl or tok == lbl + "s":
                return lbl
    return "unclear"


def prompt_of(cfg: dict) -> tuple[str, dict]:
    name = cfg.get("vlm", {}).get("prompt") or DEFAULT_PROMPT
    if name not in PROMPTS:
        raise ValueError(f"Нет промпта {name!r}; есть: {sorted(PROMPTS)}")
    return name, PROMPTS[name]


COCO_TO_LABEL = {"cell phone": "phone", "bottle": "bottle", "cup": "cup", "wine glass": "cup"}


@dataclass
class Request:
    track_id: int
    t: float
    box: tuple   # квадрат кропа x1, y1, x2, y2


def plan_requests(signals: dict[int, pd.DataFrame], cfg: dict, frame_w: int, frame_h: int) -> list[Request]:
    """Кадры для проверки: участки, где кисть близко ко рту (d < scan_threshold), по 2–4 кадра.
    Если участков слишком много (длинный фон с толпой), берём самые близкие ко рту."""
    v = cfg["vlm"]
    thr = float(v["scan_threshold"])
    hz = float(v["sample_hz"])
    max_n = int(v["max_frames_per_segment"])
    cap = int(v.get("max_requests_per_clip", 0) or 0)
    min_h = float(cfg["geometry"]["min_person_height_px"])
    segments = []      # (min d, [Request, ...])
    for tid, df in signals.items():
        d = df["d"].to_numpy()
        t = df["t"].to_numpy()
        near = np.nan_to_num(d, nan=9.0) < thr
        i = 0
        while i < len(near):
            if not near[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(near) and near[j + 1]:
                j += 1
            t0, t1 = t[i], t[j]
            dur = t1 - t0
            small = float(np.nanmedian(df["height"].to_numpy()[i:j + 1])) < min_h
            if dur >= 0.25 and not small:       # мелких людей (IGNORE) не проверяем
                n = int(np.clip(round(dur * hz), 2, max_n))
                seg = []
                for k in range(n):
                    tq = t0 + (k + 0.5) * dur / n
                    row = df.iloc[int(np.argmin(np.abs(t - tq)))]
                    seg.append(Request(int(tid), float(tq), crop_box(row, frame_w, frame_h)))
                segments.append((float(np.nanmin(d[i:j + 1])), seg))
            i = j + 1
    if cap and sum(len(s) for _, s in segments) > cap:
        segments.sort(key=lambda x: x[0])
        kept, total = [], 0
        for dmin, seg in segments:
            if total + len(seg) > cap:
                break
            kept.append((dmin, seg))
            total += len(seg)
        segments = kept
    reqs = [r for _, seg in segments for r in seg]
    reqs.sort(key=lambda r: (r.t, r.track_id))
    return reqs


def crop_box(row: pd.Series, w: int, h: int) -> tuple:
    mx, my, hx, hy, s = row["mouth_x"], row["mouth_y"], row["hand_x"], row["hand_y"], row["scale"]
    if np.isnan(mx):
        bh = row["y2"] - row["y1"]
        cx, cy, half = (row["x1"] + row["x2"]) / 2, row["y1"] + 0.15 * bh, 0.25 * bh
    else:
        if np.isnan(hx):
            cx, cy, dist = mx, my, 0.0
        else:
            cx, cy, dist = (mx + hx) / 2, (my + hy) / 2, float(np.hypot(mx - hx, my - hy))
        half = max(0.55 * float(s), 0.75 * dist, 24.0)
    x1, y1 = max(0.0, cx - half), max(0.0, cy - half)
    x2, y2 = min(float(w), cx + half), min(float(h), cy + half)
    return (float(x1), float(y1), float(x2), float(y2))


def extract_crops(video: str | Path, reqs: list[Request], size: int):
    from PIL import Image

    by_time: dict[float, list[Request]] = {}
    for r in reqs:
        by_time.setdefault(r.t, []).append(r)
    out = []
    for t_req, _t_real, img in frames_at(video, sorted(by_time)):
        for r in by_time[t_req]:
            x1, y1, x2, y2 = (int(round(v)) for v in r.box)
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            crop = img[y1:y2, x1:x2, ::-1]          # BGR → RGB
            pil = Image.fromarray(np.ascontiguousarray(crop)).resize((size, size), Image.LANCZOS)
            out.append((r, pil))
    return out


_MODELS: dict = {}


def _get_model(kind: str, factory, name: str):
    """Модели грузятся один раз на весь пакет клипов (VLM грузится 5–20 с)."""
    key = (kind, name)
    if key not in _MODELS:
        _MODELS[key] = factory()
    return _MODELS[key]


class ObjectVeto:
    """Детектор предметов COCO на увеличенном кропе: телефон, бутылка, чашка, бокал."""

    def __init__(self, cfg: dict, device: str):
        from ultralytics import YOLO

        from .pose import resolve_weights

        o = cfg["object_veto"]
        self.model = YOLO(resolve_weights(o["model"]))
        self.classes = set(o["classes"])
        self.conf = float(o["conf"])
        self.device = device

    def labels(self, images: list) -> list[str]:
        out = []
        for i in range(0, len(images), 16):
            batch = images[i:i + 16]
            results = self.model.predict(batch, conf=self.conf, device=self.device, verbose=False)
            for res in results:
                label = ""
                if res.boxes is not None and len(res.boxes):
                    names = [res.names[int(c)] for c in res.boxes.cls.cpu().numpy()]
                    for n in names:
                        if n in self.classes:
                            label = COCO_TO_LABEL.get(n, n)
                            break
                out.append(label)
        return out


class VLMJudge:
    """Qwen3-VL локально через mlx-vlm (только Mac на Apple Silicon)."""

    def __init__(self, cfg: dict, log=print):
        v = cfg["vlm"]
        from mlx_vlm import generate, load
        from mlx_vlm.prompt_utils import apply_chat_template

        self._generate = generate
        self._template = apply_chat_template
        self.model, self.processor = load(v["model"])
        self.name, self.spec = prompt_of(cfg)
        self.log = log

    @staticmethod
    def parse(text: str, kind: str = "list") -> str:
        return parse_answer(text, kind)

    def ask(self, images: list, spec: dict) -> tuple[list[str], list[str]]:
        """Метки и сырые ответы модели для каждой картинки."""
        prompt = self._template(self.processor, self.model.config, spec["text"], num_images=1)
        labels, raw = [], []
        with tempfile.TemporaryDirectory() as tmp:
            for i, im in enumerate(images):
                path = Path(tmp) / f"crop_{i}.png"
                im.save(path)
                res = self._generate(self.model, self.processor, prompt, image=[str(path)],
                                     max_tokens=int(spec["max_tokens"]), temperature=0.0, verbose=False)
                text = res.text if hasattr(res, "text") else str(res)
                raw.append(text.strip()[:80])
                labels.append(parse_answer(text, spec["kind"]))
        return labels, raw

    def labels(self, images: list) -> list[str]:
        return self.ask(images, self.spec)[0]


def run_verification(video: str | Path, signals: dict[int, pd.DataFrame], cfg: dict,
                     frame_w: int, frame_h: int, device: str, cache_path: str | Path | None = None,
                     log=print) -> dict:
    """Возвращает {"items": [{track_id, t, vlm, obj}], "vlm_used": bool, ...} и кэширует."""
    reqs = plan_requests(signals, cfg, frame_w, frame_h)
    vlm_on = bool(cfg["vlm"].get("enabled", True)) and cfg["vlm"].get("backend", "mlx") == "mlx"
    obj_on = bool(cfg["object_veto"].get("enabled", True))
    prompt_name, prompt_spec = prompt_of(cfg)
    key = {"n": len(reqs), "scan": cfg["vlm"]["scan_threshold"], "model": cfg["vlm"]["model"],
           "vlm_on": vlm_on, "obj_on": obj_on, "obj": cfg["object_veto"], "crop": cfg["vlm"]["crop_size"],
           "prompt": prompt_spec["text"], "sig": [(r.track_id, round(r.t, 2)) for r in reqs][:200]}
    if cache_path and Path(cache_path).is_file():
        try:
            cached = json.loads(Path(cache_path).read_text(encoding="utf-8"))
            if cached.get("key") == json.loads(json.dumps(key)):   # после JSON кортежи — списки
                log(f"  кэш проверки: {cache_path}")
                return cached
        except Exception:
            pass
    result = {"key": key, "items": [], "vlm_used": False, "object_veto_used": False,
              "prompt": prompt_spec["text"], "prompt_name": prompt_name, "seconds": 0.0}
    if not reqs:
        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            Path(cache_path).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        return result
    t_begin = time.time()
    crops = extract_crops(video, reqs, int(cfg["vlm"]["crop_size"]))
    images = [im for _, im in crops]
    obj = [""] * len(images)
    failed = False
    if obj_on and images:
        try:
            obj = _get_model("veto", lambda: ObjectVeto(cfg, device), cfg["object_veto"]["model"]).labels(images)
            result["object_veto_used"] = True
        except Exception as exc:
            failed = True
            log(f"  детектор предметов недоступен: {exc}")
    vlm = ["unclear"] * len(images)
    if vlm_on and images:
        try:
            judge = _get_model("vlm", lambda: VLMJudge(cfg, log), cfg["vlm"]["model"])
            log(f"  VLM: {len(images)} кропов")
            vlm = judge.labels(images)
            result["vlm_used"] = True
        except Exception as exc:
            failed = True
            log(f"  VLM недоступна ({type(exc).__name__}: {exc}); решение только по детектору и кинематике")
    for (r, _), lv, lo in zip(crops, vlm, obj):
        result["items"].append({"track_id": r.track_id, "t": round(r.t, 3), "vlm": lv, "obj": lo})
    result["n_crops"] = len(images)
    result["seconds"] = round(time.time() - t_begin, 2)
    if cache_path and not failed:     # сбой модели не кэшируем, чтобы следующий запуск попробовал снова
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        Path(cache_path).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return result


def apply_votes(cycles_by_track: dict[int, list[Cycle]], verification: dict | None, cfg: dict) -> None:
    """Проставляет каждому циклу голоса и вердикт: smoke (прямой признак), veto или none."""
    import bisect

    min_votes = int(cfg["vlm"]["min_votes"])
    both_veto = bool(cfg.get("kinematics", {}).get("both_hands_veto", True))
    # «hand» — VLM видит у рта пустую руку: касание лица, чесание, прикрывает рот (N04, N06)
    veto_labels = VETO | ({"hand"} if cfg["vlm"].get("hand_veto", True) else set())
    # «vape» модель путает с бутылкой у рта; можно считать прямым признаком только сигарету
    smoke_labels = SMOKE if cfg["vlm"].get("vape_is_smoke", True) else {"cigarette"}
    items = (verification or {}).get("items", [])
    by_track: dict[int, tuple[list, list]] = {}
    for it in sorted(items, key=lambda x: x["t"]):
        ts, its = by_track.setdefault(int(it["track_id"]), ([], []))
        ts.append(float(it["t"]))
        its.append(it)
    for tid, cycles in cycles_by_track.items():
        ts, its = by_track.get(tid, ([], []))
        for c in cycles:
            votes: dict[str, int] = {}
            lo = bisect.bisect_left(ts, c.t_enter - 0.25)
            hi = bisect.bisect_right(ts, c.t_exit + 0.25)
            for it in its[lo:hi]:
                votes[it["vlm"]] = votes.get(it["vlm"], 0) + 1
                if it.get("obj"):
                    key = "obj:" + it["obj"]
                    votes[key] = votes.get(key, 0) + 1
            c.votes = votes
            smoke = sum(v for k, v in votes.items() if k in smoke_labels)
            veto = sum(v for k, v in votes.items() if k in veto_labels) + sum(
                v for k, v in votes.items() if k.startswith("obj:"))
            # прямой признак важнее кинематики: обе руки у лица бывают и при прикуривании
            if smoke >= min_votes and smoke > veto:
                c.verdict, c.reason = "smoke", "VLM: " + ", ".join(k for k in votes if k in smoke_labels)
            elif veto >= min_votes and veto > smoke:
                c.verdict, c.reason = "veto", "предмет: " + ", ".join(
                    k.replace("obj:", "") for k in votes if (k in veto_labels or k.startswith("obj:")))
            elif both_veto and c.both_hands_frac >= 0.5:
                c.verdict, c.reason = "veto", "обе руки у лица"
            else:
                c.verdict, c.reason = "none", ""
