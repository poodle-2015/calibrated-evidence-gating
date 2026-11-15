"""Evaluation metrics.

The Gate emits a continuous insufficiency `score` (higher => more insufficient). At a
threshold `tau`, a context is predicted "insufficient" iff `score >= tau`, and routed to
the Answerer ("sufficient") otherwise.

An evaluation record is a dict:
    {"root_id": str, "variant": str, "answerable": bool, "score": float}
"""
from __future__ import annotations

from typing import Dict, Iterable, List

Record = Dict


def _pred_insufficient(rec: Record, tau: float) -> bool:
    return rec["score"] >= tau


def answerable_recall(records: Iterable[Record], tau: float) -> float:
    ans = [r for r in records if r["answerable"]]
    if not ans:
        return float("nan")
    return sum(not _pred_insufficient(r, tau) for r in ans) / len(ans)


def insufficient_precision(records: Iterable[Record], tau: float) -> float:
    predicted = [r for r in records if _pred_insufficient(r, tau)]
    if not predicted:
        return 0.0
    return sum(not r["answerable"] for r in predicted) / len(predicted)


def insufficient_recall(records: Iterable[Record], tau: float) -> float:
    ins = [r for r in records if not r["answerable"]]
    if not ins:
        return float("nan")
    return sum(_pred_insufficient(r, tau) for r in ins) / len(ins)


def insufficient_f1(records: Iterable[Record], tau: float) -> float:
    tp = fp = fn = 0
    for r in records:
        pred = _pred_insufficient(r, tau)
        pos = not r["answerable"]         # insufficient is the positive class
        if pred and pos:
            tp += 1
        elif pred and not pos:
            fp += 1
        elif not pred and pos:
            fn += 1
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    return 2 * prec * rec / (prec + rec)


def variant_recall(records: Iterable[Record], tau: float, variant: str) -> float:
    """Insufficient-recall restricted to one variant (e.g. missing_required = MRE recall)."""
    sub = [r for r in records if r.get("variant") == variant]
    if not sub:
        return float("nan")
    return sum(_pred_insufficient(r, tau) for r in sub) / len(sub)


def matched_root_ordering(records: Iterable[Record]) -> float:
    """Fraction of same-root insufficient/answerable pairs ordered strictly correctly."""
    by_root: Dict[str, List[Record]] = {}
    for r in records:
        by_root.setdefault(r["root_id"], []).append(r)
    wins = pairs = 0
    for root_records in by_root.values():
        answerable = [r for r in root_records if r["answerable"]]
        insufficient = [r for r in root_records if not r["answerable"]]
        for neg in insufficient:
            for pos in answerable:
                pairs += 1
                if neg["score"] > pos["score"]:
                    wins += 1
    return wins / pairs if pairs else float("nan")


def compute_all(records: List[Record], tau: float) -> Dict[str, float]:
    return {
        "threshold": tau,
        "n": len(records),
        "insufficient_precision": insufficient_precision(records, tau),
        "insufficient_recall": insufficient_recall(records, tau),
        "insufficient_f1": insufficient_f1(records, tau),
        "answerable_recall": answerable_recall(records, tau),
        "missing_required_recall": variant_recall(records, tau, "missing_required"),
        "single_hop_only_recall": variant_recall(records, tau, "single_hop_only"),
        "distractor_only_recall": variant_recall(records, tau, "distractor_only"),
        "matched_root_ordering": matched_root_ordering(records),
    }
