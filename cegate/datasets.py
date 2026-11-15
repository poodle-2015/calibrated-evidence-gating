"""Load HotpotQA / 2WikiMultiHopQA / MuSiQue into the common `Root` schema.

Support-fact alignment is deterministic: each support fact becomes a
(title, sentence_number) pointer; we check the title exists and the sentence
number is in range, and skip any root whose pointers cannot be aligned.
"""
from __future__ import annotations

import json
from typing import Iterator, List

from .schema import Paragraph, Pointer, Root


class AlignmentError(ValueError):
    pass


def _read_jsonl(path: str) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# --------------------------------------------------------------------------- #
# HotpotQA / 2WikiMultiHopQA (shared schema)
#   context:         [[title, [sent0, sent1, ...]], ...]
#   supporting_facts:[[title, sent_id], ...]
# --------------------------------------------------------------------------- #
def _load_hotpot_row(row: dict, dataset: str) -> Root:
    rid = str(row.get("id") or row.get("_id"))
    paragraphs = [Paragraph(title=t, sentences=list(sents)) for t, sents in row["context"]]
    by_title = {p.title: p for p in paragraphs}

    support: List[Pointer] = []
    for title, sid in row["supporting_facts"]:
        p = by_title.get(title)
        if p is None or not (0 <= int(sid) < len(p.sentences)):
            raise AlignmentError(f"{rid}: support ({title!r},{sid}) out of range")
        support.append((title, int(sid)))

    if not support:
        raise AlignmentError(f"{rid}: no alignable support")

    return Root(
        root_id=rid, dataset=dataset, question=row["question"],
        answer=str(row["answer"]), paragraphs=paragraphs, support=support,
    )


# --------------------------------------------------------------------------- #
# MuSiQue-Ans
#   paragraphs: [{idx, title, paragraph_text, is_supporting}, ...]
#   Support is paragraph-level, so each supporting paragraph contributes one
#   pointer (title, 0); the paragraph is represented as a single unit.
# --------------------------------------------------------------------------- #
def _load_musique_row(row: dict) -> Root:
    rid = str(row["id"])
    paragraphs, support = [], []
    for para in row["paragraphs"]:
        title = para["title"]
        text = para["paragraph_text"].strip()
        paragraphs.append(Paragraph(title=title, sentences=[text]))
        if para.get("is_supporting"):
            support.append((title, 0))

    if not support:
        raise AlignmentError(f"{rid}: no supporting paragraphs")

    return Root(
        root_id=rid, dataset="musique", question=row["question"],
        answer=str(row["answer"]), paragraphs=paragraphs, support=support,
    )


_LOADERS = {"hotpot": _load_hotpot_row, "musique": _load_musique_row}


def load_roots(path: str, schema: str, dataset: str) -> List[Root]:
    """Load all alignable roots from `path`. Rows that cannot be aligned are skipped."""
    if schema not in _LOADERS:
        raise ValueError(f"unknown schema {schema!r}; expected one of {list(_LOADERS)}")

    roots, skipped = [], 0
    for row in _read_jsonl(path):
        try:
            if schema == "hotpot":
                roots.append(_load_hotpot_row(row, dataset))
            else:
                roots.append(_load_musique_row(row))
        except AlignmentError:
            skipped += 1
    if skipped:
        print(f"[datasets] {dataset}: skipped {skipped} unalignable roots")
    return roots
