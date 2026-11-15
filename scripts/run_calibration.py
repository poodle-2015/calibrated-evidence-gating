#!/usr/bin/env python3
"""Choose the operating-point threshold on the calibration roots.

    python scripts/run_calibration.py --pool outputs/pools/hotpotqa/pool.jsonl \
        --scores outputs/hotpotqa.gate_scores.jsonl \
        --config configs/calibration.yaml --out outputs/hotpotqa.threshold.json
"""
import argparse
import json

import _repo_path  # noqa: F401
from cegate.calibration import choose_threshold
from cegate.config import load
from cegate.eval_io import records
from cegate.metrics import answerable_recall, insufficient_recall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--config", default="configs/calibration.yaml")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load(args.config)
    cal, _ = records(args.pool, args.scores)
    tau = choose_threshold(cal, cfg["answerable_recall_target"])
    result = {
        "threshold": tau,
        "answerable_recall_target": cfg["answerable_recall_target"],
        "calibration_answerable_recall": answerable_recall(cal, tau),
        "calibration_insufficient_recall": insufficient_recall(cal, tau),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"[calibration] tau={tau:.6g}  "
          "cal answerable_recall="
          f"{result['calibration_answerable_recall']:.4f}")


if __name__ == "__main__":
    main()
