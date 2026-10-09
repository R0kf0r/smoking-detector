"""Маска зоны интереса (ROI).

Формат маски организаторы пока не уточнили, поэтому поддерживаем три варианта:
  * картинка (PNG, JPG, BMP): зона — белые или цветные пиксели на чёрном фоне;
  * JSON с полигонами: [[x, y], ...], {"points": [...]}, {"polygons": [[[x, y], ...], ...]},
    координаты в пикселях или в долях кадра (все числа ≤ 1);
  * TXT или CSV: по строке «x,y» на вершину.
Если указана папка, для клипа берётся файл <clip_id>.* , иначе <camera_id>.* .
Человек в зоне, если в зоне его опорная точка (середина нижней стороны рамки).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
POLY_EXT = {".json", ".txt", ".csv"}


def _polygons_from_json(data) -> list[list[list[float]]]:
    if isinstance(data, dict):
        for key in ("polygons", "zones", "rois"):
            if key in data:
                return [p for item in data[key] for p in _polygons_from_json(item)]
        for key in ("points", "polygon", "roi", "zone"):
            if key in data:
                return _polygons_from_json(data[key])
        return []
    if isinstance(data, list) and data and isinstance(data[0], (int, float)):
        flat = [float(v) for v in data]
        return [[flat[i:i + 2] for i in range(0, len(flat) - 1, 2)]]
    if isinstance(data, list) and data and isinstance(data[0], dict) and "x" in data[0]:
        return [[[float(p["x"]), float(p["y"])] for p in data]]
    if isinstance(data, list) and data and isinstance(data[0], (list, tuple)):
        if data[0] and isinstance(data[0][0], (int, float)):
            return [[[float(p[0]), float(p[1])] for p in data]]
        return [p for item in data for p in _polygons_from_json(item)]
    return []


def load_roi(path: str | Path, width: int, height: int) -> np.ndarray | None:
    """Булева маска height×width или None, если файл не прочитан."""
    import cv2

    path = Path(path)
    ext = path.suffix.lower()
    if ext in IMAGE_EXT:
        from PIL import Image

        img = Image.open(path).convert("L")
        if img.size != (width, height):
            img = img.resize((width, height), Image.NEAREST)
        mask = np.asarray(img) > 20
        return mask if mask.any() else None
    if ext in POLY_EXT:
        if ext == ".json":
            polys = _polygons_from_json(json.loads(path.read_text(encoding="utf-8")))
        else:
            pts = []
            for line in path.read_text(encoding="utf-8").splitlines():
                parts = line.replace(";", ",").replace("\t", ",").split(",")
                try:
                    pts.append([float(parts[0]), float(parts[1])])
                except (ValueError, IndexError):
                    continue
            polys = [pts]
        polys = [p for p in polys if len(p) >= 3]
        if not polys:
            return None
        mask = np.zeros((height, width), np.uint8)
        for poly in polys:
            arr = np.asarray(poly, dtype=float)
            if np.all(arr <= 1.0 + 1e-9):
                arr = arr * np.array([width, height])
            cv2.fillPoly(mask, [arr.round().astype(np.int32)], 1)
        return mask.astype(bool)
    return None


def find_roi_file(roi_arg: str | Path | None, clip_id: str, camera_id: str) -> Path | None:
    if not roi_arg:
        return None
    p = Path(roi_arg)
    if p.is_file():
        return p
    if p.is_dir():
        for stem in (clip_id, camera_id):
            for f in sorted(p.glob(f"{stem}.*")):
                if f.suffix.lower() in IMAGE_EXT | POLY_EXT:
                    return f
    return None


def inside_fraction(mask: np.ndarray | None, boxes: np.ndarray) -> float:
    """Доля рамок, у которых опорная точка (середина низа) внутри зоны."""
    if mask is None or len(boxes) == 0:
        return 1.0
    h, w = mask.shape
    xs = np.clip(((boxes[:, 0] + boxes[:, 2]) / 2).round().astype(int), 0, w - 1)
    ys = np.clip(boxes[:, 3].round().astype(int), 0, h - 1)
    return float(np.mean(mask[ys, xs]))
