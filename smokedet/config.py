"""Загрузка конфигурации: configs/default.yaml + пресет + точечные переопределения."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


def deep_merge(base: dict, override: dict) -> dict:
    """Рекурсивно накладывает override на base и возвращает новый словарь."""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path | None = None, preset: str | None = None,
                overrides: dict | None = None) -> dict:
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    presets = cfg.pop("presets", {}) or {}
    if preset:
        if preset not in presets:
            raise ValueError(f"Нет пресета {preset!r}; есть: {sorted(presets)}")
        cfg = deep_merge(cfg, presets[preset])
    if overrides:
        cfg = deep_merge(cfg, overrides)
    cfg["_meta"] = {"config_path": str(cfg_path), "preset": preset}
    return cfg


def config_hash(cfg: dict) -> str:
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    blob = json.dumps(clean, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def get(cfg: dict, dotted: str, default: Any = None) -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node
