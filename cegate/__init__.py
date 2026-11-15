"""Minimal Calibrated Evidence Gating reference implementation.

Modules:
  schema             common Root / Paragraph / Variant representation
  datasets           load HotpotQA / 2Wiki / MuSiQue into the common schema
  construction       root -> five evaluation context variants
  prompts            context serialization and template rendering
  metrics            Ins.P/R/F1 / answerable recall / per-variant metrics
  calibration        operating-point threshold selection
  infer_gate         likelihood-based sufficiency scoring
"""

__all__ = [
    "schema",
    "datasets",
    "construction",
    "prompts",
    "metrics",
    "calibration",
    "infer_gate",
]
