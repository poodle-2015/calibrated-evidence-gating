#!/usr/bin/env python3
"""Score a pool with the Evidence Gate using label likelihoods.

    python scripts/score_gate.py --pool outputs/pools/hotpotqa/pool.jsonl \
        --config configs/gate_llama31_8b.yaml --adapter <adapter_dir> \
        --out outputs/hotpotqa.gate_scores.jsonl

Emits one {"id", "score"} per example; higher score => more insufficient.
The score is logP(insufficient) - logP(sufficient) (see cegate/infer_gate.py).
"""
import argparse
import json

import _repo_path  # noqa: F401
from cegate.config import load
from cegate.infer_gate import GateScorer
from cegate.pool import load_pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--config", default="configs/gate_llama31_8b.yaml")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    gcfg = load(args.config)
    scorer = GateScorer(gcfg["model"]["base"], adapter=args.adapter,
                        dtype=gcfg["model"]["dtype"])

    with open(args.out, "w", encoding="utf-8") as f:
        for e in load_pool(args.pool):
            score = scorer.score(e["context"], e["question"])
            f.write(json.dumps({"id": e["id"], "score": score}) + "\n")
    print(f"[score_gate] wrote scores to {args.out}")


if __name__ == "__main__":
    main()
