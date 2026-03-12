"""Lightweight parsing helpers — no heavy dependencies."""

from __future__ import annotations

import re

HEADER_REGEX = re.compile(r"^\s*Pathophysiolog(?:y|ical)\s+Process:\s*(.+?)\s*$", re.I)
TRIPLES_HEADER_REGEX = re.compile(r"^\s*Triples:\s*$", re.I)
TRIPLE_LINE_REGEX = re.compile(r"^\s*([^|]+)\|([^|]+)\|([^|]+)\s*$")

REFUSAL_KEYWORDS = [
    "i cannot", "i can't", "i am unable",
    "content policy", "i apologize", "i'm sorry", "inappropriate",
]


def parse_mechanisms_and_triples(text: str | None) -> list[dict]:
    """Parse GPT output into a list of mechanism blocks with their triples.

    Each block is ``{"mechanism": str, "triples": [(subject, predicate, object), ...]}``.

    Parameters
    ----------
    text:
        Raw text from GPT-4o.  ``None`` or empty string returns ``[]``.
    """
    blocks: list[dict] = []
    current: dict | None = None
    collecting = False
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = HEADER_REGEX.match(line)
        if m:
            current = {"mechanism": m.group(1).strip(), "triples": []}
            blocks.append(current)
            collecting = False
            continue
        if TRIPLES_HEADER_REGEX.match(line):
            collecting = True
            continue
        if collecting and current is not None:
            m3 = TRIPLE_LINE_REGEX.match(line)
            if m3:
                subj, pred, obj = (x.strip() for x in m3.groups())
                current["triples"].append((subj, pred, obj))
    return blocks


def is_refusal(text: str) -> bool:
    """Return True if the text looks like a GPT refusal."""
    text_lower = text.lower()
    return any(k in text_lower for k in REFUSAL_KEYWORDS)
