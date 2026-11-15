# Evidence Sufficiency Is a Calibrated Decision in Generative Multi-hop Question Answering

Official core implementation for the paper accepted at the **EACL 2027** main conference.

This release contains the calibrated Evidence Gate path studied in the paper (the
sufficiency classifier plus its calibrated threshold):

```text
public multi-hop QA data
  -> controlled evidence variants
  -> Evidence Gate likelihood scores
  -> target-domain threshold calibration
  -> held-out sufficiency metrics
```

The Gate decides whether the supplied context contains a complete multi-hop evidence chain
before answer generation. A higher Gate score means stronger evidence insufficiency.

## Data preparation

Download the source datasets from their official releases and set the local JSONL paths in
`configs/datasets.yaml`.

| Dataset | Setting | Official source | Dataset license |
| --- | --- | --- | --- |
| HotpotQA | Distractor | [HotpotQA release](https://hotpotqa.github.io/) | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |
| 2WikiMultiHopQA | Distractor | [2WikiMultiHopQA repository](https://github.com/Alab-NII/2wikimultihop) | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| MuSiQue | MuSiQue-Ans | [MuSiQue repository](https://github.com/StonyBrookNLP/musique) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Gate inference requires a compatible PEFT adapter for the base model configured in
`configs/gate_llama31_8b.yaml`. If Gate scores are already available, skip the scoring step.
The score JSONL format is one record per pool example:

```json
{"id": "example-id::full_support", "score": -1.23}
```

## Running the core pipeline

The commands below use HotpotQA as an example.

```bash
# 1. Build the five-variant evaluation pools.
python scripts/build_data.py \
  --config configs/datasets.yaml \
  --out outputs/pools

# 2. Score each constructed context with the Evidence Gate.
python scripts/score_gate.py \
  --pool outputs/pools/hotpotqa/pool.jsonl \
  --config configs/gate_llama31_8b.yaml \
  --adapter /path/to/gate_adapter \
  --out outputs/hotpotqa.gate_scores.jsonl

# 3. Select the operating threshold on calibration roots only.
python scripts/run_calibration.py \
  --pool outputs/pools/hotpotqa/pool.jsonl \
  --scores outputs/hotpotqa.gate_scores.jsonl \
  --config configs/calibration.yaml \
  --out outputs/hotpotqa.threshold.json

# 4. Apply the frozen threshold to disjoint test roots.
python scripts/run_eval.py \
  --pool outputs/pools/hotpotqa/pool.jsonl \
  --scores outputs/hotpotqa.gate_scores.jsonl \
  --threshold outputs/hotpotqa.threshold.json \
  --out outputs/hotpotqa.metrics.json
```

The build step also writes `outputs/pools/manifest.json` with construction settings, counts,
and per-pool checksums. Missing Gate scores are treated as an error rather than silently
dropping evaluation examples.

## Repository layout

```text
calibrated-evidence-gating/
├── README.md
├── requirements.txt
├── configs/       # dataset, Gate inference, and calibration settings
├── prompts/       # exact Evidence Gate prompt
├── cegate/        # core implementation
├── scripts/       # four pipeline entry points
└── tests/         # construction and paper-alignment invariants
```

Run the included checks with:

```bash
python tests/test_construction.py
```

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{lim-etal-2027-evidence,
  title     = {Evidence Sufficiency Is a Calibrated Decision in Generative Multi-hop Question Answering},
  author    = {Lim, Hyoung jun},
  booktitle = {Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers)},
  year      = {2027},
  publisher = {Association for Computational Linguistics}
}
```
