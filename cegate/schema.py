"""Common representation shared by all three datasets.

A *root* is one original dataset instance. Every dataset is normalized into `Root`
so that construction, serialization, and scoring are dataset-agnostic.

Sentence numbering is preserved from the source so that a support pointer
(title, sentence_number) stays valid after truncation: a constructed context keeps
each retained sentence's ORIGINAL index, and a pointer is "visible" iff a line with
that (title, index) is present.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

# A support pointer: (paragraph title, original sentence index).
Pointer = Tuple[str, int]


@dataclass
class Paragraph:
    """A source paragraph. `sentences[i]` has original sentence number `i`."""
    title: str
    sentences: List[str]

    def pointers(self) -> List[Pointer]:
        return [(self.title, i) for i in range(len(self.sentences))]


@dataclass
class RenderedParagraph:
    """A paragraph inside a constructed context. `lines` are (original_index, text)
    pairs, so support pointers remain valid after sentence selection/truncation."""
    title: str
    lines: List[Tuple[int, str]]

    def pointers(self) -> List[Pointer]:
        return [(self.title, i) for i, _ in self.lines]


@dataclass
class Root:
    """One original dataset instance, normalized."""
    root_id: str
    dataset: str
    question: str
    answer: str
    paragraphs: List[Paragraph]
    support: List[Pointer]           # annotated (title, sentence_number) support pointers

    @property
    def gold_titles(self) -> List[str]:
        """Distinct gold paragraph titles, in first-seen order (the hop proxy)."""
        seen, out = set(), []
        for t, _ in self.support:
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out

    @property
    def hops(self) -> int:
        return len(self.gold_titles)

    def paragraph(self, title: str) -> Optional[Paragraph]:
        for p in self.paragraphs:
            if p.title == title:
                return p
        return None

    def gold_paragraphs(self) -> List[Paragraph]:
        golds = set(self.gold_titles)
        return [p for p in self.paragraphs if p.title in golds]

    def nonsupport_paragraphs(self) -> List[Paragraph]:
        golds = set(self.gold_titles)
        return [p for p in self.paragraphs if p.title not in golds]


# The five context variants. `answerable` is the binary evaluated label;
# `diagnosis_type` is the coarse error category used only in the JSON contract.
VARIANT_SPECS = {
    "full_support":     {"answerable": True,  "diagnosis_type": "none"},
    "minimal_support":  {"answerable": True,  "diagnosis_type": "none"},
    "missing_required": {"answerable": False, "diagnosis_type": "missing_required_support"},
    "single_hop_only":  {"answerable": False, "diagnosis_type": "single_hop_only"},
    "distractor_only":  {"answerable": False, "diagnosis_type": "irrelevant_context"},
}


@dataclass
class Variant:
    root_id: str
    dataset: str
    variant: str                     # one of VARIANT_SPECS
    question: str
    answer: str                       # evaluation metadata; never rendered to the Gate
    paragraphs: List[RenderedParagraph]  # the constructed context
    kept_support: List[Pointer]          # annotated support pointers visible here
    hops: int

    @property
    def answerable(self) -> bool:
        return VARIANT_SPECS[self.variant]["answerable"]

    @property
    def sufficiency(self) -> str:
        return "sufficient" if self.answerable else "insufficient"

    @property
    def diagnosis_type(self) -> str:
        return VARIANT_SPECS[self.variant]["diagnosis_type"]

    def num_sentences(self) -> int:
        return sum(len(p.lines) for p in self.paragraphs)

    def num_titles(self) -> int:
        return len(self.paragraphs)

    def visible_pointers(self) -> set:
        out = set()
        for p in self.paragraphs:
            out.update(p.pointers())
        return out
