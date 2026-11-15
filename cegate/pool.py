"""Build and load evaluation pools (five variants per root) with a by-root split.

An example record written to the pool JSONL:
    {"id", "root_id", "dataset", "variant", "answerable", "diagnosis_type",
     "hops", "question", "answer", "context", "kept_support",
     "num_sentences", "num_titles", "split"}
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from typing import Dict, List

from .construction import build_eval_variants
from .prompts import serialize_context
from .schema import Root, Variant


def _example(v: Variant) -> Dict:
    return {
        "id": f"{v.root_id}::{v.variant}",
        "root_id": v.root_id, "dataset": v.dataset, "variant": v.variant,
        "answerable": v.answerable, "diagnosis_type": v.diagnosis_type, "hops": v.hops,
        "question": v.question, "answer": v.answer,
        "context": serialize_context(v),
        "kept_support": [[t, i] for (t, i) in v.kept_support],
        "num_sentences": v.num_sentences(), "num_titles": v.num_titles(),
    }


def select_roots(roots: List[Root], seed: int, target_roots: int = None,
                 hop_strata: dict = None, min_hops: int = 2) -> List[Root]:
    """Deterministically subsample roots to the paper's pool size.

    Roots below `min_hops` gold support paragraphs are dropped first. With `hop_strata`
    ({hops: count}), take exactly that many roots per hop bucket; otherwise take the
    first `target_roots` after a seeded shuffle. Selection is seeded and stable."""
    roots = [r for r in roots if r.hops >= min_hops]
    rng = random.Random(seed)

    if hop_strata:
        by_hop = defaultdict(list)
        for r in roots:
            by_hop[r.hops].append(r)
        out: List[Root] = []
        for hop, n in sorted(hop_strata.items()):
            group = sorted(by_hop.get(int(hop), []), key=lambda r: r.root_id)
            rng.shuffle(group)
            out.extend(group[: int(n)])
        return out

    if target_roots:
        ordered = sorted(roots, key=lambda r: r.root_id)
        rng.shuffle(ordered)
        return ordered[: int(target_roots)]
    return roots


def _split_roots(roots: List[Root], calibration_fraction: float) -> Dict[str, str]:
    """Paper split: sort root ids by MD5, then take the calibration prefix."""
    ids = sorted({r.root_id for r in roots},
                 key=lambda rid: hashlib.md5(rid.encode("utf-8")).hexdigest())
    cut = int(round(len(ids) * calibration_fraction))
    if len(ids) > 1:
        cut = max(1, min(len(ids) - 1, cut))
    calibration = set(ids[:cut])
    return {rid: "calibration" if rid in calibration else "test" for rid in ids}


def build_pool(roots: List[Root], ccfg: dict, seed: int,
               split_cfg: dict) -> List[Dict]:
    split = _split_roots(roots, float(split_cfg["calibration_fraction"]))

    examples: List[Dict] = []
    for root in roots:
        for v in build_eval_variants(root, ccfg, seed):
            ex = _example(v)
            ex["split"] = split[root.root_id]
            examples.append(ex)
    return examples


def statistics(examples: List[Dict]) -> Dict:
    roots = {e["root_id"] for e in examples}
    ans = sum(e["answerable"] for e in examples)
    hop_hist = Counter(e["hops"] for e in examples if e["variant"] == "full_support")
    variant_hist = Counter(e["variant"] for e in examples)
    split_hist = Counter(e["split"] for e in examples)
    return {
        "roots": len(roots),
        "rows": len(examples),
        "answerable": ans,
        "insufficient": len(examples) - ans,
        "hops_by_root": dict(sorted(hop_hist.items())),
        "variant_counts": dict(variant_hist),
        "split_counts": dict(split_hist),
    }


def write_pool(examples: List[Dict], path: str) -> str:
    """Write JSONL and return a SHA-256 checksum of the canonical content."""
    h = hashlib.sha256()
    with open(path, "w", encoding="utf-8") as f:
        for e in examples:
            line = json.dumps(e, ensure_ascii=False, sort_keys=True)
            f.write(line + "\n")
            h.update(line.encode("utf-8"))
    return h.hexdigest()


def load_pool(path: str) -> List[Dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
