"""Operating-point calibration.

Choose a single threshold `tau` on CALIBRATION records that maximizes insufficient
recall subject to the target answerable recall. Break recall ties by higher
answerable recall, then freeze the threshold for TEST records.
"""
from __future__ import annotations

from math import inf, nextafter
from typing import Dict, List

from .metrics import answerable_recall, insufficient_recall

Record = Dict


def choose_threshold(cal_records: List[Record], target: float) -> float:
    """Maximize insufficient recall subject to the answerable recall floor.

    Ties in insufficient recall are resolved by higher answerable recall, as
    specified in the paper.
    """
    if not any(r["answerable"] for r in cal_records):
        raise ValueError("no answerable records to calibrate on")
    if not any(not r["answerable"] for r in cal_records):
        raise ValueError("no insufficient records to calibrate on")
    scores = sorted({float(r["score"]) for r in cal_records})
    candidates = [*scores, nextafter(scores[-1], inf)]
    feasible = [tau for tau in candidates
                if answerable_recall(cal_records, tau) >= target]
    if not feasible:
        raise ValueError(f"no threshold reaches answerable recall target {target}")
    return max(feasible, key=lambda tau: (
        insufficient_recall(cal_records, tau),
        answerable_recall(cal_records, tau),
    ))


def calibrate_and_score(cal_records: List[Record], test_records: List[Record],
                        target: float):
    from .metrics import compute_all
    tau = choose_threshold(cal_records, target)
    return tau, compute_all(test_records, tau)
