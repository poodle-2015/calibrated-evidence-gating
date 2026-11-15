"""Root -> five evaluation context variants.

Determinism: the only stochastic choice is which supporting sentence is removed
(missing_required) or kept (single_hop_only). It is fixed by a per-root RNG seeded from the
global seed and the root id, so a rebuild is bit-for-bit reproducible.

Variant table (see the paper's method section and construction appendix):
  full_support      gold paragraphs, then two hard distractors              -> answerable
  minimal_support   gold paragraphs, only the annotated support sentences   -> answerable
  missing_required  full support with one required sentence removed (>=2)  -> insufficient
  single_hop_only   exactly one support sentence, no distractors (>=2)      -> insufficient
  distractor_only   the two hard distractors, no gold                       -> insufficient

Each paragraph shows at most its first five sentences; a context holds at most six
paragraphs.
"""
from __future__ import annotations

import hashlib
import random
import re
from typing import List, Optional, Sequence

from .schema import Paragraph, Pointer, RenderedParagraph, Root, Variant

_WORD = re.compile(r"[A-Za-z0-9]+")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "had", "has", "have", "he", "her", "his", "in", "is",
    "it", "its", "of", "on", "or", "she", "that", "the", "their", "this",
    "to", "was", "were", "what", "when", "where", "which", "who", "whom",
    "whose", "why", "with",
}


def _tokens(text: str) -> set:
    return {w.lower() for w in _WORD.findall(text)
            if len(w) > 1 and w.lower() not in _STOPWORDS}


def _rng(root_id: str, seed: int) -> random.Random:
    h = int(hashlib.sha256(f"{seed}:{root_id}".encode("utf-8")).hexdigest(), 16)
    return random.Random(h % (2 ** 32))


# --------------------------------------------------------------------------- #
# Distractor selection (the "hard distractors")
# --------------------------------------------------------------------------- #
def _overlap(qtok: set, para: Paragraph, title_weight: float) -> float:
    body = _tokens(" ".join(para.sentences[:3]))
    title = _tokens(para.title)
    return title_weight * len(qtok & title) + len(qtok & body) / max(1, len(qtok))


def rank_distractors(root: Root, ccfg: dict) -> List[Paragraph]:
    """Rank non-support paragraphs by question overlap (title double-weighted);
    take the top `num_distractors`, keeping ties in their original order."""
    qtok = _tokens(root.question)
    tw = float(ccfg["title_overlap_weight"])
    nons = root.nonsupport_paragraphs()
    # sorted() is stable, so equal scores keep the source order.
    ranked = sorted(nons, key=lambda p: _overlap(qtok, p, tw), reverse=True)
    return ranked[: int(ccfg["num_distractors"])]


# --------------------------------------------------------------------------- #
# Sentence selection / rendering
# --------------------------------------------------------------------------- #
def _cap_indices(n: int, max_sents: int) -> List[int]:
    """Each paragraph shows at most its first `max_sents` sentences; original
    sentence numbers are preserved."""
    return list(range(min(n, max_sents)))


def _render(para: Paragraph, idxs: Sequence[int]) -> RenderedParagraph:
    return RenderedParagraph(title=para.title, lines=[(i, para.sentences[i]) for i in idxs])


def _support_idxs(root: Root, title: str) -> set:
    return {i for (t, i) in root.support if t == title}


def _kept_support(root: Root, paras: List[RenderedParagraph]) -> List[Pointer]:
    visible = set()
    for rp in paras:
        visible.update(rp.pointers())
    return [ptr for ptr in root.support if ptr in visible]


def _variant(root: Root, name: str, paras: List[RenderedParagraph]) -> Variant:
    # Paragraph order: gold paragraphs first, then distractors (paper Appendix).
    return Variant(
        root_id=root.root_id, dataset=root.dataset, variant=name,
        question=root.question, answer=root.answer, paragraphs=paras,
        kept_support=_kept_support(root, paras), hops=root.hops,
    )


# --------------------------------------------------------------------------- #
# Individual variants
# --------------------------------------------------------------------------- #
def _full_support_paras(root: Root, distractors: List[Paragraph], ccfg: dict) -> List[RenderedParagraph]:
    max_s = int(ccfg["max_sentences_per_paragraph"])
    max_p = int(ccfg["max_paragraphs"])
    paras: List[RenderedParagraph] = []
    for gp in root.gold_paragraphs():
        paras.append(_render(gp, _cap_indices(len(gp.sentences), max_s)))
    for d in distractors:
        paras.append(_render(d, _cap_indices(len(d.sentences), max_s)))
    # Cap paragraph count, keeping gold paragraphs (listed first here) preferentially.
    return paras[:max_p]


def build_variant(root: Root, name: str, ccfg: dict, seed: int,
                  removal_rank: int = 0) -> Optional[Variant]:
    """Build one named variant, or None if it is not well-formed for this root.
    `removal_rank` selects the k-th support pointer to drop/keep (for missing_heavy)."""
    rng = _rng(root.root_id, seed)
    distractors = rank_distractors(root, ccfg)
    max_s = int(ccfg["max_sentences_per_paragraph"])

    if name == "full_support":
        return _variant(root, name, _full_support_paras(root, distractors, ccfg))

    if name == "minimal_support":
        paras = []
        for gp in root.gold_paragraphs():
            sidx = sorted(i for i in _support_idxs(root, gp.title) if i < max_s)
            if sidx:
                paras.append(_render(gp, sidx))
        return _variant(root, name, paras) if paras else None

    if name == "missing_required":
        if len(root.support) < 2:
            return None
        order = list(root.support)
        rng.shuffle(order)
        drop = order[removal_rank % len(order)]
        full = _full_support_paras(root, distractors, ccfg)
        paras = []
        for rp in full:
            lines = [(i, t) for (i, t) in rp.lines if (rp.title, i) != drop]
            if lines:
                paras.append(RenderedParagraph(rp.title, lines))
        # must actually have removed a visible required sentence
        if drop not in _full_visible(full):
            return None
        return _variant(root, name, paras)

    if name == "single_hop_only":
        if len(root.support) < 2:
            return None
        order = list(root.support)
        rng.shuffle(order)
        t, i = order[removal_rank % len(order)]
        para = root.paragraph(t)
        if para is None or i >= len(para.sentences):
            return None
        return _variant(root, name, [RenderedParagraph(t, [(i, para.sentences[i])])])

    if name == "distractor_only":
        if not distractors:
            return None
        paras = [_render(d, _cap_indices(len(d.sentences), max_s)) for d in distractors]
        return _variant(root, name, paras)

    raise ValueError(f"unknown variant {name!r}")


def _full_visible(full: List[RenderedParagraph]) -> set:
    out = set()
    for rp in full:
        out.update(rp.pointers())
    return out


# --------------------------------------------------------------------------- #
# Public builders
# --------------------------------------------------------------------------- #
EVAL_VARIANTS = ["full_support", "minimal_support", "missing_required",
                 "single_hop_only", "distractor_only"]


def build_eval_variants(root: Root, ccfg: dict, seed: int) -> List[Variant]:
    """The up-to-five held-out variants, filtered by the paper's pointer audit."""
    gold = set(root.support)
    out = []
    for name in EVAL_VARIANTS:
        v = build_variant(root, name, ccfg, seed)
        if v is None:
            continue
        complete = gold <= v.visible_pointers()
        if (v.answerable and complete) or (not v.answerable and not complete):
            out.append(v)
    return out
