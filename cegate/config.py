"""Tiny YAML config loader with path resolution."""
from __future__ import annotations

import os
from typing import Any, Dict

import yaml


def load(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path: str, base: str) -> str:
    """Resolve `path` relative to `base` (the config file's directory) unless absolute."""
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base, path))
