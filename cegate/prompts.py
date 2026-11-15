"""Context serialization and Evidence Gate prompt rendering."""
from __future__ import annotations

import os
from typing import Dict

from .schema import Variant

_PROMPT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts")


def _read(name: str) -> str:
    with open(os.path.join(_PROMPT_DIR, name), "r", encoding="utf-8") as f:
        return f.read()


def serialize_context(variant: Variant) -> str:
    """Render the paper's evidence-addressable context format.

        [TITLE] Title
        [Title#0] first sentence
        [Title#1] second sentence
    """
    lines = []
    for p in variant.paragraphs:
        paragraph_lines = []
        for idx, text in p.lines:
            paragraph_lines.append(f"[{p.title}#{idx}] {text}")
        if paragraph_lines:
            lines.append(f"[TITLE] {p.title}")
            lines.extend(paragraph_lines)
    return "\n".join(lines)


def render_gate(context: str, question: str) -> Dict[str, str]:
    return {
        "system": _read("gate_system.txt"),
        "user": _read("gate_user_template.txt").format(
            context=context, question=question),
    }
