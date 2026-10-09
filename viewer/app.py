#!/usr/bin/env python3
"""Просмотрщик: выбрать или загрузить ролик → прогнать конвейер → смотреть видео с отметками,
события, циклы, ответы VLM, время обработки и вердикт по разметке.

    bash viewer/start.sh               # macOS: окружение smoke, браузер откроется сам
    python viewer/app.py [--port 8765]

Только стандартная библиотека Python. Слушает 127.0.0.1: с других компьютеров недоступен,
ролики не покидают этот компьютер. Обработка — тот же `python -m smokedet.run` с тем же конфигом,
что зафиксирован тегом freeze-v1. Просмотрщик пишет только в private/viewer/ (и в общий кэш
results/cache/); results/hidden и results/reference не трогает.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import queue
import re
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
VIEW_DIR = ROOT / "private" / "viewer"
UPLOADS = VIEW_DIR / "uploads"
RUNS = VIEW_DIR / "runs"
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".ts", ".mts", ".webm", ".mpg", ".mpeg", ".dav", ".h264",
             ".wmv", ".flv", ".3gp", ".asf"}
FONT_EXT = {".ttf", ".otf", ".woff", ".woff2"}
LOG_LINE = re.compile(r"^\[\d\d:\d\d:\d\d\]")
NOISE = re.compile(r"^\[(h264|hevc|mpeg4|mjpeg|aac|mov|swscaler|NULL)[^\]]*@|non-existing SPS|mmco|error while decoding|"
                   r"concealing|decode_slice|left block unavailable|\(repeated \d+ more times\)|deprecated pixel format")
MAX_POINTS = 1500

# ---------------------------------------------------------------- библиотека роликов

_items: dict[str, dict] = {}
_probe_cache: dict[str, tuple] = {}
_lock = threading.Lock()


def _key(path: Path) -> str:
    return hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:12]


def _probe(path: Path) -> dict:
    st = path.stat()
    sig = (st.st_size, int(st.st_mtime))
    hit = _probe_cache.get(str(path))
    if hit and hit[0] == sig:
        return hit[1]
    info = {"duration": None, "fps": None}
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                              "stream=avg_frame_rate,r_frame_rate:format=duration", "-of", "json", str(path)],
                             capture_output=True, text=True, timeout=20).stdout
        data = json.loads(out or "{}")
        dur = (data.get("format") or {}).get("duration")
        info["duration"] = float(dur) if dur not in (None, "N/A") else None
        st0 = (data.get("streams") or [{}])[0]
        for k in ("avg_frame_rate", "r_frame_rate"):
            num, _, den = str(st0.get(k, "0/0")).partition("/")
            if den and float(den or 0) > 0 and float(num or 0) > 0:
                info["fps"] = float(num) / float(den)
                break
    except Exception:
        pass
    _probe_cache[str(path)] = (sig, info)
    return info


def _labels() -> dict[str, str]:
    out: dict[str, str] = {}
    path = ROOT / "labels" / "reference.csv"
    if not path.is_file():
        return out
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            cid, lab = (r.get("clip_id") or "").strip(), (r.get("label") or "").strip().upper()
            if cid and lab and cid not in out:
                out[cid] = lab
    return out


def _official_counts(run: str) -> dict[str, int] | None:
    path = ROOT / "results" / run / "events.json"
    if not path.is_file():
        return None
    try:
        events = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    counts: dict[str, int] = {}
    for e in events:
        counts[e["clip_id"]] = counts.get(e["clip_id"], 0) + 1
    return counts


def _render_path(item: dict) -> Path:
    return RUNS / item["key"] / f"{item['clip_id']}.mp4"


def scan_library() -> list[dict]:
    labels = _labels()
    official = {"reference": _official_counts("reference"), "hidden": _official_counts("hidden")}
    groups = [("reference", ROOT / "private" / "reference"), ("hidden", ROOT / "private" / "hidden"),
              ("upload", UPLOADS)]
    items = []
    for group, folder in groups:
        if not folder.is_dir():
            continue
        files = sorted((p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXT
                        and not p.name.startswith(".")), key=lambda p: str(p).lower())
        for p in files:
            rel = p.relative_to(folder)
            sub = rel.parts[0] if len(rel.parts) > 1 else ""
            clip_id = p.stem
            title = {"reference": "Эталон", "hidden": "Скрытый набор", "upload": "Загруженные"}[group]
            if group == "reference" and sub in ("positive", "negative"):
                title += " — " + {"positive": "позитив", "negative": "негатив"}[sub]
            item = {"key": _key(p), "name": p.name, "clip_id": clip_id, "path": str(p), "group": group,
                    "group_title": title, "size_mb": round(p.stat().st_size / 2**20, 1),
                    "label": labels.get(clip_id) if group == "reference" else None}
            item.update(_probe(p))
            off = official.get(group)
            item["official_events"] = None if off is None else off.get(clip_id, 0)
            run_dir = RUNS / item["key"]
            item["processed"] = (run_dir / "events.json").is_file() and _render_path(item).is_file()
            if item["processed"]:
                try:
                    item["n_events"] = len(json.loads((run_dir / "events.json").read_text(encoding="utf-8")))
                except Exception:
                    item["n_events"] = None
            items.append(item)
    with _lock:
        _items.clear()
        _items.update({it["key"]: it for it in items})
    return items


def get_item(key: str) -> dict | None:
    with _lock:
        it = _items.get(key)
    if it is None:
        scan_library()
        with _lock:
            it = _items.get(key)
    return it


# ---------------------------------------------------------------- задания (по одному за раз)

_jobs: dict[str, dict] = {}
_queue: "queue.Queue[str]" = queue.Queue()


def official_busy() -> bool:
    """Идёт ли официальный прогон скрытого набора (hidden.sh): тогда не мешаем ему."""
    try:
        out = subprocess.run(["ps", "-axo", "command"], capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return False
    for line in out.splitlines():
        if "viewer/app.py" in line:
            continue
        if ("smokedet.run" in line and "results/hidden" in line) or "smokedet.demo" in line \
                or "scripts/hidden.sh" in line or "scripts/freeze.sh" in line:
            return True
    return False


def _parse_progress(job: dict, line: str) -> None:
    text = line.split("]", 1)[1].strip() if "]" in line else line
    m = re.search(r"кадр (\d+)/(\d+), ([\d.]+) кадр/с", text)
    if "рендер" in text:
        job["stage"] = "Видео с отметками"
        m2 = re.search(r"кадр (\d+), ([\d.]+) кадр/с", text)
        if m2 and job.get("frames"):
            job["progress"] = min(0.99, int(m2.group(1)) / job["frames"])
        elif "→" in text:
            job["progress"] = 0.0
    elif m:
        job["stage"] = "Поза и трекинг"
        job["progress"] = min(0.99, int(m.group(1)) / max(1, int(m.group(2))))
    elif "кэш проверки" in text:
        job["stage"], job["progress"] = "Проверка VLM: из кэша", None
    elif text.startswith("кэш:"):
        job["stage"], job["progress"] = "Поза и трекинг: из кэша", None
    elif text.startswith("VLM:"):
        job["stage"], job["progress"] = "Проверка VLM: " + text[4:].strip(), None
    elif "загружаю" in text.lower() or "модел" in text.lower():
        job["stage"] = text
    job["last"] = text


def _worker() -> None:
    while True:
        job_id = _queue.get()
        job = _jobs[job_id]
        item = get_item(job["key"])
        if item is None:
            job.update(status="error", stage="Ролик не найден", finished=time.time())
            continue
        out_dir = RUNS / item["key"]
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-u", "-m", "smokedet.run", "--input", item["path"], "--clip-id", item["clip_id"],
               "--out-dir", str(out_dir), "--render-dir", str(out_dir), "--camera-id", job.get("camera") or "viewer"]
        fps, dur = item.get("fps"), item.get("duration")
        job["frames"] = int(fps * dur) if fps and dur else None
        job.update(status="running", stage="Запуск конвейера", progress=None, started=time.time())
        try:
            env = dict(os.environ, PYTHONUNBUFFERED="1")
            proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, encoding="utf-8", errors="replace", bufsize=1, env=env)
            job["pid"] = proc.pid
            assert proc.stdout is not None
            for raw in proc.stdout:
                line = raw.rstrip()
                if not line or NOISE.search(line):
                    continue                      # шум декодера видео
                job["log"].append(line)
                if LOG_LINE.match(line):
                    _parse_progress(job, line)
            rc = proc.wait()
            ok = rc in (0, 2) and _render_path(item).is_file() and (out_dir / "events.json").is_file()
            job.update(status="done" if ok else "error", rc=rc, finished=time.time(),
                       stage="Готово" if ok else "Ошибка обработки", progress=1.0 if ok else None)
        except Exception as exc:                  # noqa: BLE001
            job.update(status="error", stage=f"Ошибка: {type(exc).__name__}: {exc}", finished=time.time())
        scan_library()


def enqueue(key: str, camera: str | None = None) -> dict:
    for j in _jobs.values():
        if j["key"] == key and j["status"] in ("queued", "running"):
            return j
    job = {"id": uuid.uuid4().hex[:10], "key": key, "status": "queued", "stage": "В очереди", "progress": None,
           "log": deque(maxlen=60), "created": time.time(), "camera": camera}
    _jobs[job["id"]] = job
    _queue.put(job["id"])
    return job


def job_view(job: dict) -> dict:
    now = job.get("finished") or time.time()
    return {"id": job["id"], "key": job["key"], "status": job["status"], "stage": job.get("stage"),
            "progress": job.get("progress"), "last": job.get("last"), "log": list(job["log"])[-12:],
            "elapsed": round(now - job.get("started", job["created"]), 1),
            "position": (list(_queue.queue).index(job["id"]) + 1) if job["status"] == "queued"
            and job["id"] in list(_queue.queue) else None}


# ---------------------------------------------------------------- результаты

def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) or math.isinf(v) else v


def _read_signals(path: Path, wanted: set[int]) -> dict[int, dict]:
    tracks: dict[int, dict] = {}
    if not path.is_file():
        return tracks
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            try:
                tid = int(float(r["track_id"]))
            except (KeyError, ValueError):
                continue
            if tid not in wanted:
                continue
            tr = tracks.setdefault(tid, {"t": [], "d": [], "h": []})
            tr["t"].append(_num(r.get("t")))
            tr["d"].append(_num(r.get("d")))
            tr["h"].append(_num(r.get("height")))
    for tr in tracks.values():
        n = len(tr["t"])
        if n > MAX_POINTS:
            step = math.ceil(n / MAX_POINTS)
            for k in ("t", "d", "h"):
                tr[k] = tr[k][::step]
        tr["t"] = [None if v is None else round(v, 3) for v in tr["t"]]
        tr["d"] = [None if v is None else round(v, 4) for v in tr["d"]]
        hs = [v for v in tr.pop("h") if v is not None]
        tr["median_height"] = round(sorted(hs)[len(hs) // 2], 1) if hs else None
    return tracks


def result(item: dict) -> dict | None:
    run_dir = RUNS / item["key"]
    ev_path, render = run_dir / "events.json", _render_path(item)
    if not ev_path.is_file() or not render.is_file():
        return None
    events = json.loads(ev_path.read_text(encoding="utf-8"))
    dbg_path = run_dir / "debug" / f"{item['clip_id']}.json"
    dbg = json.loads(dbg_path.read_text(encoding="utf-8")) if dbg_path.is_file() else {}
    man_path = run_dir / "manifest.json"
    man = json.loads(man_path.read_text(encoding="utf-8")) if man_path.is_file() else {}
    timing = {}
    t_path = run_dir / "timing.csv"
    if t_path.is_file():
        with open(t_path, encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
            timing = rows[0] if rows else {}
    by_id = {e.get("event_id"): e for e in dbg.get("events", [])}
    for e in events:
        extra = by_id.get(e["event_id"], {})
        for k in ("n_cycles", "has_object", "median_height", "cycles"):
            e[k] = extra.get(k)
    cycles = dbg.get("cycles", [])
    rejected = dbg.get("rejected", [])
    # треки для графика: с событиями, с циклами, с отброшенными кандидатами
    score: dict[int, float] = {}
    for e in events:
        score[int(e["person_track_id"])] = score.get(int(e["person_track_id"]), 0) + 100
    for c in cycles:
        score[int(c["track_id"])] = score.get(int(c["track_id"]), 0) + 10
    for r in rejected:
        score[int(r["track_id"])] = score.get(int(r["track_id"]), 0) + 1
    wanted = [tid for tid, _ in sorted(score.items(), key=lambda kv: -kv[1])][:12]
    sig = _read_signals(run_dir / "signals" / f"{item['clip_id']}.csv", set(wanted))
    tracks = [{"track_id": tid, **sig[tid]} for tid in wanted if tid in sig]
    cfg = (man.get("config") or {}).get("values") or {}
    cyc_cfg = cfg.get("cycles") or {}
    try:
        mtime = int(render.stat().st_mtime)
    except OSError:
        mtime = 0
    return {
        "key": item["key"], "name": item["name"], "clip_id": item["clip_id"], "label": item.get("label"),
        "group": item["group"], "group_title": item["group_title"],
        "video": dbg.get("video") or {}, "media": f"/media?key={item['key']}&v={mtime}",
        "events": events, "cycles": cycles, "rejected": rejected,
        "dropped_events": dbg.get("dropped_events", []), "verification": dbg.get("verification", {}),
        "n_tracks": len(dbg.get("tracks", [])), "tracks": tracks, "timing": timing,
        "thresholds": {"enter": cyc_cfg.get("enter_threshold"), "exit": cyc_cfg.get("exit_threshold")},
        "rules": man.get("decision_rules", {}), "config_hash": (man.get("config") or {}).get("hash"),
        "git": man.get("git", {}), "official_events": item.get("official_events"),
        "downloads": {k: f"/download?key={item['key']}&f={k}" for k in
                      ("events.csv", "events.json", "manifest.json", "timing.csv") if (run_dir / k).is_file()},
    }


def _clip_metrics(run: str) -> dict | None:
    rdir = ROOT / "results" / run
    if not (rdir / "events.json").is_file() or not (rdir / "timing.csv").is_file():
        return None
    events = json.loads((rdir / "events.json").read_text(encoding="utf-8"))
    with open(rdir / "timing.csv", encoding="utf-8", newline="") as f:
        timing = [r for r in csv.DictReader(f) if r.get("status") == "ok"]
    n_ev: dict[str, int] = {}
    for e in events:
        n_ev[e["clip_id"]] = n_ev.get(e["clip_id"], 0) + 1
    video_s = sum(_num(r.get("duration_s")) or 0 for r in timing)
    proc_s = sum(_num(r.get("processing_s")) or 0 for r in timing)
    out = {"clips": len(timing), "events": len(events), "video_min": round(video_s / 60, 1),
           "sec_per_min": round(proc_s / video_s * 60, 1) if video_s else None}
    if run == "reference":
        labels = _labels()
        pos = [r["clip_id"] for r in timing if labels.get(r["clip_id"]) == "POSITIVE"]
        neg = [r for r in timing if labels.get(r["clip_id"]) == "NEGATIVE"]
        tp = sum(1 for c in pos if n_ev.get(c))
        fp = sum(n_ev.get(r["clip_id"], 0) for r in neg)
        fn = len(pos) - tp
        neg_h = sum(_num(r.get("duration_s")) or 0 for r in neg) / 3600
        out.update(pos=len(pos), neg=len(neg), tp=tp, fp=fp, fn=fn,
                   precision=round(tp / (tp + fp), 3) if tp + fp else None,
                   recall=round(tp / len(pos), 3) if pos else None,
                   f1=round(2 * tp / (2 * tp + fp + fn), 3) if (2 * tp + fp + fn) else None,
                   fp_per_hour=round(fp / neg_h, 1) if neg_h else None)
    return out


def summary() -> dict:
    cfg_vals = {}
    try:
        man = json.loads((ROOT / "results" / "reference" / "manifest.json").read_text(encoding="utf-8"))
        cfg_vals = ((man.get("config") or {}).get("values") or {}).get("cycles", {})
    except Exception:
        pass
    tag = ""
    try:
        tag = subprocess.run(["git", "-C", str(ROOT), "describe", "--tags", "--always"], capture_output=True,
                             text=True, timeout=5).stdout.strip()
    except Exception:
        pass
    running = [job_view(j) for j in _jobs.values() if j["status"] in ("queued", "running")]
    return {"reference": _clip_metrics("reference"), "hidden": _clip_metrics("hidden"), "busy": official_busy(),
            "tag": tag, "thresholds": {"enter": cfg_vals.get("enter_threshold"),
                                       "exit": cfg_vals.get("exit_threshold")}, "jobs": running}


# ---------------------------------------------------------------- HTTP

class Handler(BaseHTTPRequestHandler):
    server_version = "smokedet-viewer/1.0"

    def log_message(self, fmt, *args):            # тихо: ошибки возвращаются в JSON
        pass

    # -- ответы
    def _json(self, obj, status: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path, ctype: str, download: str | None = None) -> None:
        size = path.stat().st_size
        start, end, status = 0, size - 1, 200
        rng = self.headers.get("Range")
        if rng:
            m = re.match(r"bytes=(\d*)-(\d*)", rng.strip())
            if m:
                s, e = m.groups()
                if s == "" and e:
                    start, end = max(0, size - int(e)), size - 1
                else:
                    start = int(s or 0)
                    end = min(int(e), size - 1) if e else size - 1
                if start >= size or start > end:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
                status = 206
        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        if download:
            self.send_header("Content-Disposition", f"attachment; filename*=UTF-8''{quote(download)}")
        self.end_headers()
        if self.command == "HEAD":
            return
        with open(path, "rb") as f:
            f.seek(start)
            left = length
            while left > 0:
                chunk = f.read(min(1 << 20, left))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return
                left -= len(chunk)

    def _err(self, status: int, msg: str) -> None:
        self._json({"error": msg}, status)

    # -- маршруты
    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if url.path in ("/", "/index.html"):
                return self._file(HERE / "index.html", "text/html; charset=utf-8")
            if url.path.startswith("/static/"):
                name = Path(url.path[len("/static/"):]).name
                path = ROOT / "static" / name
                if path.suffix.lower() in FONT_EXT and path.is_file():
                    return self._file(path, "font/ttf" if path.suffix.lower() == ".ttf" else "font/" + path.suffix[1:])
                return self._err(404, "нет файла")
            if url.path == "/api/videos":
                return self._json({"items": scan_library()})
            if url.path == "/api/summary":
                return self._json(summary())
            if url.path == "/api/result":
                item = get_item(q.get("key", ""))
                if item is None:
                    return self._err(404, "ролик не найден")
                res = result(item)
                return self._json(res if res else {"key": item["key"], "processed": False, "item": item})
            if url.path == "/api/job":
                job = _jobs.get(q.get("id", ""))
                return self._json(job_view(job)) if job else self._err(404, "задание не найдено")
            if url.path == "/media":
                item = get_item(q.get("key", ""))
                if item is None or not _render_path(item).is_file():
                    return self._err(404, "видео не найдено")
                return self._file(_render_path(item), "video/mp4")
            if url.path == "/download":
                item = get_item(q.get("key", ""))
                name = q.get("f", "")
                if item is None or name not in ("events.csv", "events.json", "manifest.json", "timing.csv"):
                    return self._err(404, "нет файла")
                path = RUNS / item["key"] / name
                if not path.is_file():
                    return self._err(404, "нет файла")
                ctype = "text/csv; charset=utf-8" if name.endswith(".csv") else "application/json; charset=utf-8"
                return self._file(path, ctype, download=f"{item['clip_id']}_{name}")
            return self._err(404, "нет такой страницы")
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as exc:                  # noqa: BLE001
            return self._err(500, f"{type(exc).__name__}: {exc}")

    def do_POST(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if url.path == "/api/upload":
                return self._upload(q)
            if url.path == "/api/process":
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}") if n else {}
                item = get_item(str(body.get("key", "")))
                if item is None:
                    return self._err(404, "ролик не найден")
                if official_busy():
                    return self._err(409, "Идёт прогон скрытого набора — обработка в просмотрщике подождёт, "
                                          "смотреть готовые результаты можно")
                return self._json(job_view(enqueue(item["key"], body.get("camera"))))
            return self._err(404, "нет такого действия")
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as exc:                  # noqa: BLE001
            return self._err(500, f"{type(exc).__name__}: {exc}")

    def _upload(self, q: dict) -> None:
        raw = Path(q.get("name", "video.mp4")).name
        ext = Path(raw).suffix.lower()
        if ext not in VIDEO_EXT:
            return self._err(400, f"Формат {ext or '?'} не поддерживается: нужен видеофайл")
        stem = re.sub(r"[^\w\-() ]", "_", Path(raw).stem).strip(" .") or "video"
        size = int(self.headers.get("Content-Length") or 0)
        if size <= 0:
            return self._err(411, "пустой файл")
        UPLOADS.mkdir(parents=True, exist_ok=True)
        dest = UPLOADS / f"{stem[:100]}{ext}"
        k = 1
        while dest.exists() and dest.stat().st_size != size:
            k += 1
            dest = UPLOADS / f"{stem[:100]} ({k}){ext}"
        reuse = dest.exists()
        left = size
        sink = open(os.devnull, "wb") if reuse else open(dest, "wb")
        with sink as f:
            while left > 0:
                chunk = self.rfile.read(min(1 << 20, left))
                if not chunk:
                    break
                f.write(chunk)
                left -= len(chunk)
        if left > 0:
            if not reuse:
                try:
                    dest.unlink()
                except OSError:
                    pass
            return self._err(400, "загрузка оборвалась")
        scan_library()
        item = get_item(_key(dest))
        if item is None:
            return self._err(500, "файл не появился в библиотеке")
        job = None
        if not item.get("processed") and not official_busy():
            job = job_view(enqueue(item["key"], q.get("camera")))
        return self._json({"key": item["key"], "item": item, "job": job,
                           "busy": job is None and not item.get("processed")})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python viewer/app.py", description=__doc__.split("\n")[0])
    ap.add_argument("--host", default="127.0.0.1", help="адрес (по умолчанию только этот компьютер)")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args(argv)
    os.chdir(ROOT)
    RUNS.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=_worker, daemon=True).start()
    threading.Thread(target=scan_library, daemon=True).start()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    print(f"Просмотрщик: http://{args.host}:{args.port}  (остановить — Ctrl+C)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
