#!/usr/bin/env python3
"""Build the five-variant evaluation pools + statistics + checksums.

    python scripts/build_data.py --config configs/datasets.yaml --out outputs/pools

Writes, per dataset, `<out>/<dataset>/pool.jsonl` and a shared `<out>/manifest.json`
recording row/root statistics and a SHA-256 checksum of each pool.
"""
import argparse
import json
import os

import _repo_path  # noqa: F401
from cegate.config import load, resolve_path
from cegate.datasets import load_roots
from cegate.pool import build_pool, select_roots, statistics, write_pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/datasets.yaml")
    ap.add_argument("--out", default="outputs/pools")
    args = ap.parse_args()

    cfg = load(args.config)
    base = os.path.dirname(os.path.abspath(args.config))
    ccfg, seed, split_cfg = cfg["construction"], cfg["seed"], cfg["split"]

    manifest = {"seed": seed, "construction": ccfg, "datasets": {}}
    for name, dcfg in cfg["datasets"].items():
        path = resolve_path(dcfg["eval_path"], base)   # the held-out dev pool
        if not os.path.exists(path):
            print(f"[build_data] SKIP {name}: eval source not found at {path}")
            continue
        roots = load_roots(path, dcfg["schema"], name)
        roots = select_roots(roots, seed, target_roots=dcfg.get("eval_roots"),
                             hop_strata=dcfg.get("eval_hop_strata"),
                             min_hops=ccfg.get("min_hops", 2))
        examples = build_pool(roots, ccfg, seed, split_cfg)

        out_dir = os.path.join(args.out, name)
        os.makedirs(out_dir, exist_ok=True)
        pool_path = os.path.join(out_dir, "pool.jsonl")
        checksum = write_pool(examples, pool_path)
        stats = statistics(examples)

        manifest["datasets"][name] = {
            "source": path, "pool": pool_path, "sha256": checksum, **stats}
        print(f"[build_data] {name}: {stats['rows']} rows over {stats['roots']} roots "
              f"({stats['answerable']} ans / {stats['insufficient']} ins)  sha256={checksum[:12]}")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[build_data] wrote manifest to {os.path.join(args.out, 'manifest.json')}")


if __name__ == "__main__":
    main()
