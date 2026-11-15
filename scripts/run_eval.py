#!/usr/bin/env python3
"""Evaluate on the held-out test roots at the frozen threshold.

    python scripts/run_eval.py --pool outputs/pools/hotpotqa/pool.jsonl \
        --scores outputs/hotpotqa.gate_scores.jsonl \
        --threshold outputs/hotpotqa.threshold.json --out outputs/hotpotqa.metrics.json

If --threshold is omitted, calibrates first with configs/calibration.yaml.
"""
import argparse
import json

import _repo_path  # noqa: F401
from cegate.calibration import choose_threshold
from cegate.config import load
from cegate.eval_io import records
from cegate.metrics import compute_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--threshold", default=None, help="threshold JSON from run_calibration")
    ap.add_argument("--calibration-config", default="configs/calibration.yaml")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cal, test = records(args.pool, args.scores)
    if args.threshold:
        tau = json.load(open(args.threshold))["threshold"]
    else:
        cfg = load(args.calibration_config)
        tau = choose_threshold(cal, cfg["answerable_recall_target"])

    metrics = compute_all(test, tau)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("[eval] " + "  ".join(
        f"{k}={v:.4f}" for k, v in metrics.items()
        if isinstance(v, float) and v == v))


if __name__ == "__main__":
    main()
