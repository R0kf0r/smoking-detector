#!/usr/bin/env python3
"""Замер ресурсов конвейера на одном ролике: время по этапам, загрузка CPU и GPU, память.

    bash tools/bench.sh [ролик]          # по умолчанию private/reference/positive/sm_6.mp4

Прогон идёт с пустым кэшем (private/bench/cache), чтобы поза и VLM считались заново; конфиг тот же.
Каждые 0,5 с снимаем: %CPU и память дерева процессов (ps), загрузку GPU Apple Silicon
(ioreg, «Device Utilization %», без sudo). Итог — private/bench/bench.json и таблица в терминале.
Результаты в results/ не трогаются.
"""

from __future__ import annotations

import json
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "private" / "bench"


def sh(cmd: list[str], timeout: float = 10) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except Exception:
        return ""


def hardware() -> dict:
    hw = {"platform": platform.platform(), "machine": platform.machine()}
    if sys.platform == "darwin":
        hw["chip"] = sh(["sysctl", "-n", "machdep.cpu.brand_string"])
        hw["model"] = sh(["sysctl", "-n", "hw.model"])
        hw["cpu_cores"] = sh(["sysctl", "-n", "hw.ncpu"])
        hw["p_cores"] = sh(["sysctl", "-n", "hw.perflevel0.physicalcpu"])
        hw["e_cores"] = sh(["sysctl", "-n", "hw.perflevel1.physicalcpu"])
        mem = sh(["sysctl", "-n", "hw.memsize"])
        hw["memory_gb"] = round(int(mem) / 2**30, 1) if mem.isdigit() else None
        m = re.search(r"Total Number of Cores:\s*(\d+)", sh(["system_profiler", "SPDisplaysDataType"], 30))
        hw["gpu_cores"] = int(m.group(1)) if m else None
    else:
        hw["cpu_cores"] = sh(["nproc"])
    return hw


def gpu_util() -> float | None:
    if not shutil.which("ioreg"):
        return None
    m = re.search(r'"Device Utilization %"\s*=\s*(\d+)', sh(["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"], 5))
    return float(m.group(1)) if m else None


def tree_usage(root_pid: int) -> tuple[float, float]:
    """(%CPU, RSS ГБ) процесса и всех потомков."""
    rows = sh(["ps", "-A", "-o", "pid=,ppid=,pcpu=,rss="], 5).splitlines()
    info, kids = {}, {}
    for r in rows:
        p = r.split()
        if len(p) < 4:
            continue
        try:
            pid, ppid, cpu, rss = int(p[0]), int(p[1]), float(p[2].replace(",", ".")), float(p[3])
        except ValueError:
            continue
        info[pid] = (cpu, rss)
        kids.setdefault(ppid, []).append(pid)
    total_cpu = total_rss = 0.0
    stack = [root_pid]
    while stack:
        pid = stack.pop()
        if pid in info:
            total_cpu += info[pid][0]
            total_rss += info[pid][1]
        stack += kids.get(pid, [])
    return total_cpu, total_rss / 2**20


def main() -> int:
    clip = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "private" / "reference" / "positive" / "sm_6.mp4")
    if not clip.is_file():
        print(f"Нет ролика: {clip}")
        return 1
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    cmd = [sys.executable, "-u", "-m", "smokedet.run", "--input", str(clip), "--out-dir", str(OUT / "run"),
           "--cache-dir", str(OUT / "cache"), "--camera-id", "bench"]
    timer = ["/usr/bin/time", "-l"] if sys.platform == "darwin" else (["/usr/bin/time", "-v"] if Path("/usr/bin/time").exists() else [])
    print("Замер на", clip.name, "— поза и VLM считаются заново, 1–3 минуты…", flush=True)
    t0 = time.time()
    proc = subprocess.Popen(timer + cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            encoding="utf-8", errors="replace")
    samples: list[dict] = []
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            cpu, rss = tree_usage(proc.pid)
            samples.append({"t": round(time.time() - t0, 2), "cpu": cpu, "rss_gb": round(rss, 3), "gpu": gpu_util()})
            stop.wait(0.5)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    _, err = proc.communicate()
    stop.set()
    th.join(2)
    wall = time.time() - t0

    # границы этапов по журналу прогона
    log = (OUT / "run" / "run.log").read_text(encoding="utf-8") if (OUT / "run" / "run.log").is_file() else ""
    stamps = []
    for line in log.splitlines():
        m = re.match(r"\[(\d\d):(\d\d):(\d\d)\]\s+(.*)", line)
        if m:
            stamps.append((int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3)), m.group(4)))
    timing = {}
    tpath = OUT / "run" / "timing.csv"
    if tpath.is_file():
        import csv
        with open(tpath, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
            timing = rows[0] if rows else {}
    pose_s = float(timing.get("pose_s") or 0)
    verify_s = float(timing.get("verify_s") or 0)
    dur = float(timing.get("duration_s") or 0)
    # этапы по времени от старта: [начало позы … +pose_s], затем VLM
    first = min((s["t"] for s in samples), default=0)
    start_pose = None
    for sec, text in stamps:
        if "кадр" in text or "ReID" in text or text.startswith("кэш"):
            start_pose = sec
            break
    load_s = 0.0
    if stamps and start_pose is not None:
        load_s = max(0.0, start_pose - stamps[0][0])
    pose_from, pose_to = first + load_s, first + load_s + pose_s
    vlm_from, vlm_to = pose_to, pose_to + verify_s

    def stats(a: float, b: float) -> dict:
        sel = [s for s in samples if a <= s["t"] <= b]
        if not sel:
            return {}
        g = [s["gpu"] for s in sel if s["gpu"] is not None]
        return {"seconds": round(b - a, 1), "cpu_cores_avg": round(sum(s["cpu"] for s in sel) / len(sel) / 100, 2),
                "cpu_cores_max": round(max(s["cpu"] for s in sel) / 100, 2),
                "gpu_util_avg": round(sum(g) / len(g), 1) if g else None, "gpu_util_max": max(g) if g else None,
                "rss_gb_max": round(max(s["rss_gb"] for s in sel), 2)}

    mem = {}
    for key, pat in (("peak_footprint_gb", r"(\d+)\s+peak memory footprint"), ("max_rss_gb", r"(\d+)\s+maximum resident set size"),
                     ("max_rss_gb", r"Maximum resident set size \(kbytes\):\s*(\d+)")):
        m = re.search(pat, err or "")
        if m:
            v = int(m.group(1))
            mem[key] = round(v / 2**30, 2) if "kbytes" not in pat else round(v / 2**20, 2)
    result = {"clip": clip.name, "video_s": dur, "wall_s": round(wall, 1), "timing": timing, "hardware": hardware(),
              "stages": {"pose": stats(pose_from, pose_to), "vlm": stats(vlm_from, vlm_to), "all": stats(first, first + wall)},
              "memory": mem, "samples": samples}
    (OUT / "bench.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")

    hw = result["hardware"]
    print(f"\nЖелезо: {hw.get('chip')} ({hw.get('model')}), CPU {hw.get('cpu_cores')} ядер "
          f"({hw.get('p_cores')}P+{hw.get('e_cores')}E), GPU {hw.get('gpu_cores')} ядер, память {hw.get('memory_gb')} ГБ")
    print(f"Ролик {dur:.1f} с: поза {pose_s:.1f} с, VLM {verify_s:.1f} с, всего прогон {wall:.0f} с")
    for name, title in (("pose", "Поза и трекинг"), ("vlm", "Проверка VLM"), ("all", "Весь прогон")):
        s = result["stages"][name]
        if s:
            print(f"  {title}: CPU {s['cpu_cores_avg']} ядра в среднем (пик {s['cpu_cores_max']}), "
                  f"GPU {s['gpu_util_avg']}% в среднем (пик {s['gpu_util_max']}%), память до {s['rss_gb_max']} ГБ")
    if mem:
        print("  Память процесса:", ", ".join(f"{k} = {v} ГБ" for k, v in mem.items()))
    print(f"\nПодробно: {OUT / 'bench.json'}")
    return 0 if proc.returncode in (0, 2) else proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
