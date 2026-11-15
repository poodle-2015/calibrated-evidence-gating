"""Join a pool with Gate scores into evaluation records, split by calibration/test.

A scores file is JSONL with one object per example: {"id": ..., "score": ...}.
"""
from __future__ import annotations

import json
from typing import Dict, List, Tuple

from .pool import load_pool


def load_scores(path: str) -> Dict[str, float]:
    scores = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                scores[obj["id"]] = float(obj["score"])
    return scores


def records(pool_path: str, scores_path: str) -> Tuple[List[Dict], List[Dict]]:
    """Return (calibration_records, test_records). Each record carries root_id,
    variant, answerable, and score, ready for cegate.metrics."""
    pool = load_pool(pool_path)
    scores = load_scores(scores_path)
    cal, test = [], []
    missing = 0
    for e in pool:
        if e["id"] not in scores:
            missing += 1
            continue
        rec = {"root_id": e["root_id"], "variant": e["variant"],
               "answerable": e["answerable"], "score": scores[e["id"]]}
        (test if e["split"] == "test" else cal).append(rec)
    if missing:
        raise ValueError(f"{missing} pool examples have no Gate score")
    return cal, test
