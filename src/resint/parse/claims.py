"""The sentences that cite something, which are the sentences worth checking.

A citation on its own says nothing checkable. ``\\cite{vaswani2017}`` is a key;
what can be right or wrong is the *sentence it sits in* -- the assertion the
author attached that reference to. That sentence is the claim, and it is the
unit ``bib/citation-support`` works on.

Two coordinate systems have to be reconciled here. Citations are extracted
from **raw** LaTeX, because normalization deliberately discards citation keys.
Sentences are segmented over **normalized** text, because raw LaTeX has no
sentence structure worth speaking of. So sentences are walked in normalized
space, mapped back to their raw range through the ordinary offset machinery,
and citations are matched by where they fall in the raw source. That keeps one
authority for the mapping rather than inventing a second.
"""

from __future__ import annotations

import re

from ..ir.paper import CitedClaim
from ..ir.span import Source
from .extract import sentences
from .latex import Normalized

#: A cited sentence shorter than this is a pointer, not an assertion --
#: "see [12]", "following [3]". There is nothing in it to verify.
MIN_CLAIM_CHARS = 40


def extract_claims(
    doc: Normalized, src: Source, citations, sections=None
) -> list[CitedClaim]:
    """Every sentence that cites something, with the keys it cites.

    Order follows the paper. A sentence citing three works yields one claim
    with three keys rather than three claims, because the sentence is what all
    three references are being offered in support of.
    """
    if not citations:
        return []

    by_position = sorted(citations, key=lambda c: c.span.start)
    out: list[CitedClaim] = []

    for start, end in sentences(doc.text):
        text = doc.text[start:end].strip()
        if len(text) < MIN_CLAIM_CHARS:
            continue

        low, high = doc.raw_range(start, end)
        keys: list[str] = []
        for citation in by_position:
            if citation.span.start >= high:
                break
            if citation.span.start >= low and citation.key not in keys:
                keys.append(citation.key)

        if not keys:
            continue

        out.append(
            CitedClaim(
                text=_tidy(text),
                keys=tuple(keys),
                span=doc.anchor(src, low, high, "claim"),
                section=_section_at(sections, low),
            )
        )

    return out


def _tidy(text: str) -> str:
    """Close the gap a removed citation leaves behind.

    Normalization drops the citation command, so a sentence arrives reading
    "scales linearly with sequence length , which" -- with a space stranded
    before the punctuation. Cosmetic in isolation, but this text is quoted
    back to the author in the finding and handed to a model as the claim, and
    both deserve the sentence the author actually wrote.
    """
    joined = " ".join((text or "").split())
    return re.sub(r"\s+([,.;:)\]])", r"\1", joined)


def _section_at(sections, offset: int) -> str:
    """The heading a claim sits under, for reporting rather than for logic."""
    name = ""
    for section in sections or ():
        start = getattr(section, "start", None)
        if start is None or start > offset:
            continue
        name = getattr(section, "title", "") or name
    return name
