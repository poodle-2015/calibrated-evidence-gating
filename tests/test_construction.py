"""Construction invariants (the automated support-pointer audit).

Run:  python -m pytest tests/  -q      (or)   python tests/test_construction.py
"""
import os
import sys
import hashlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cegate.calibration import choose_threshold  # noqa: E402
from cegate.construction import build_eval_variants, build_variant  # noqa: E402
from cegate.metrics import compute_all  # noqa: E402
from cegate.pool import _split_roots  # noqa: E402
from cegate.prompts import serialize_context  # noqa: E402
from cegate.schema import Paragraph, Root  # noqa: E402

CCFG = {"max_paragraphs": 6, "max_sentences_per_paragraph": 5,
        "num_distractors": 2, "title_overlap_weight": 2.0}
SEED = 7


def _root():
    # A two-hop root: gold A (sent 0,1) and B (sent 0); three non-support paragraphs.
    return Root(
        root_id="r1", dataset="toy",
        question="Where was the director of Film X born?",
        answer="Warsaw",
        paragraphs=[
            Paragraph("A", ["The director of Film X is Jan.", "Jan directs films."]),
            Paragraph("B", ["Jan was born in Warsaw."]),
            Paragraph("C", ["Film X premiered in 2001.", "It was popular."]),
            Paragraph("D", ["An unrelated paragraph about birds."]),
            Paragraph("E", ["Another unrelated paragraph about Warsaw weather."]),
        ],
        support=[("A", 0), ("B", 0)],
    )


def test_audit_answerable_keep_all_insufficient_drop_one():
    root = _root()
    support = set(root.support)
    variants = {v.variant: v for v in build_eval_variants(root, CCFG, SEED)}
    assert set(variants) == {"full_support", "minimal_support", "missing_required",
                             "single_hop_only", "distractor_only"}

    for name in ("full_support", "minimal_support"):
        visible = variants[name].visible_pointers()
        assert support <= visible, f"{name} must keep all support pointers"

    for name in ("missing_required", "single_hop_only", "distractor_only"):
        visible = variants[name].visible_pointers()
        assert not (support <= visible), f"{name} must drop at least one support pointer"


def test_labels():
    variants = {v.variant: v for v in build_eval_variants(_root(), CCFG, SEED)}
    assert variants["full_support"].answerable
    assert variants["minimal_support"].answerable
    assert not variants["missing_required"].answerable
    assert not variants["single_hop_only"].answerable
    assert not variants["distractor_only"].answerable
    assert variants["missing_required"].diagnosis_type == "missing_required_support"
    assert variants["single_hop_only"].diagnosis_type == "single_hop_only"
    assert variants["distractor_only"].diagnosis_type == "irrelevant_context"


def test_distractor_count_and_cap():
    v = {x.variant: x for x in build_eval_variants(_root(), CCFG, SEED)}["full_support"]
    assert v.num_titles() <= CCFG["max_paragraphs"]
    for p in v.paragraphs:
        assert len(p.lines) <= CCFG["max_sentences_per_paragraph"]
    # distractors-only has exactly num_distractors paragraphs
    d = {x.variant: x for x in build_eval_variants(_root(), CCFG, SEED)}["distractor_only"]
    assert d.num_titles() == CCFG["num_distractors"]


def test_single_hop_needs_two_pointers():
    single = Root(
        root_id="r2", dataset="toy", question="q?", answer="a",
        paragraphs=[Paragraph("A", ["only one gold sentence."]),
                    Paragraph("B", ["distractor one."]),
                    Paragraph("C", ["distractor two."])],
        support=[("A", 0)],
    )
    assert build_variant(single, "single_hop_only", CCFG, SEED) is None


def test_determinism():
    a = [v.visible_pointers() for v in build_eval_variants(_root(), CCFG, SEED)]
    b = [v.visible_pointers() for v in build_eval_variants(_root(), CCFG, SEED)]
    assert a == b


def test_paper_context_format():
    variant = build_variant(_root(), "minimal_support", CCFG, SEED)
    context = serialize_context(variant)
    assert "[TITLE] A" in context
    assert "[A#0] The director of Film X is Jan." in context
    assert "(0)" not in context


def test_md5_root_split():
    roots = [_root()]
    for rid in ("r2", "r3", "r4"):
        roots.append(Root(root_id=rid, dataset="toy", question="q", answer="a",
                          paragraphs=_root().paragraphs, support=_root().support))
    assignment = _split_roots(roots, 0.5)
    ordered = sorted((r.root_id for r in roots),
                     key=lambda rid: hashlib.md5(rid.encode("utf-8")).hexdigest())
    assert {rid for rid, split in assignment.items() if split == "calibration"} == set(ordered[:2])


def test_calibration_and_paper_metrics():
    records = [
        {"root_id": "r", "variant": "full_support", "answerable": True, "score": 0.0},
        {"root_id": "r", "variant": "minimal_support", "answerable": True, "score": 1.0},
        {"root_id": "r", "variant": "missing_required", "answerable": False, "score": 3.0},
        {"root_id": "r", "variant": "single_hop_only", "answerable": False, "score": 4.0},
        {"root_id": "r", "variant": "distractor_only", "answerable": False, "score": 5.0},
    ]
    # Thresholds 1 and 3 have the same insufficient recall; the paper breaks the tie
    # using higher answerable recall, so threshold 3 must win.
    tau = choose_threshold(records, 0.5)
    assert tau == 3.0
    metrics = compute_all(records, tau)
    assert metrics["insufficient_precision"] == 1.0
    assert metrics["answerable_recall"] == 1.0
    assert metrics["matched_root_ordering"] == 1.0


if __name__ == "__main__":
    test_audit_answerable_keep_all_insufficient_drop_one()
    test_labels()
    test_distractor_count_and_cap()
    test_single_hop_needs_two_pointers()
    test_determinism()
    test_paper_context_format()
    test_md5_root_split()
    test_calibration_and_paper_metrics()
    print("all construction invariants passed")
