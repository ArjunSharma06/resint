"""Finding the passages of a cited paper that could bear on a claim.

Plain code. No model, no network, no cost.

This exists because the obvious way to check a citation against a cited paper
-- send the whole paper -- costs roughly thirty times what it needs to and does
not fit in a context window anyway. A manuscript cites forty works; forty full
papers is some four hundred thousand tokens per check. Narrowing to the two or
three paragraphs that could possibly bear on the claim brings that back to the
order of an abstract, and does it with code rather than judgement.

Two different questions live here, and conflating them would be a bug:

*Which paragraphs should the model read?* Answered by :func:`retrieve`, which
scores paragraphs against each other. Terms appearing in every paragraph of the
cited paper carry no information about **which** paragraph to pick, so they are
weighted down -- ordinary inverse document frequency, computed over the one
document at hand.

*Is this paper about the subject at all?* Answered by :func:`relatedness`, which
looks at the whole document at once. Here the same ubiquitous terms are the
most informative thing available: a paper that never once mentions any content
word of the claim is very likely not the paper the author meant to cite.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9\-]{2,}")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")

# Function words plus the connective vocabulary every paper shares. Left
# deliberately short: an aggressive list starts discarding terms that carry
# real meaning in some field, and the frequency weighting below already
# handles the common case.
_STOP = frozenset(
    """
    the and for that this with from are was were has have had not but its
    can may our their they these those such than then when where which while
    all any been both each into more most other some only also over under
    between during within without because however therefore thus been being
    use used using shown show shows given give gives well very much many
    """.split()
)

#: Below this a paragraph is a heading, a caption fragment, or a stray line.
MIN_PARAGRAPH_CHARS = 120

#: Above this, a single block is not a paragraph -- it is a paper that arrived
#: with no paragraph breaks. Chunking it beats retrieving the whole thing.
MAX_BLOCK_CHARS = 1_500

#: A shared number is worth far more than a shared word. Two papers both
#: saying "accuracy" means nothing; both saying "94.2" is a strong signal
#: that they are discussing the same result.
NUMBER_WEIGHT = 3.0


@dataclass(frozen=True, slots=True)
class Passage:
    """One paragraph of a cited paper, and why it was chosen."""

    text: str
    start: int
    score: float
    shared: tuple[str, ...] = ()

    def why(self) -> str:
        return f"shares {', '.join(self.shared[:6])}" if self.shared else "no overlap"


def terms(text: str) -> set[str]:
    """Content words, lowercased. Hyphenation is not load-bearing."""
    found = {w.lower().strip("-") for w in _WORD.findall(text or "")}
    return {w for w in found if w and w not in _STOP}


def numbers(text: str) -> set[str]:
    """Numeric tokens, normalised so 94.20 and 94.2 are the same number."""
    out = set()
    for raw in _NUMBER.findall(text or ""):
        try:
            value = float(raw)
        except ValueError:
            continue
        # Year-like and single-digit values match by coincidence constantly.
        if value < 10 or 1900 <= value <= 2100:
            continue
        out.add(f"{value:g}")
    return out


def _sentence_chunks(start: int, block: str) -> list[tuple[int, str]]:
    """Break an over-long block at sentence boundaries, keeping offsets.

    Reached when a paper arrives with no paragraph breaks at all. Returning it
    as one block would mean retrieving the whole paper every time, which is
    the cost this module exists to avoid.
    """
    pieces: list[tuple[int, str]] = []
    cursor = 0
    for boundary in re.finditer(r"(?<=[.!?])\s+", block):
        if boundary.start() - cursor >= MAX_BLOCK_CHARS:
            pieces.append((start + cursor, block[cursor : boundary.start()]))
            cursor = boundary.end()
    tail = block[cursor:].strip()
    if tail:
        pieces.append((start + cursor, tail))
    return pieces


def split_paragraphs(text: str) -> list[tuple[int, str]]:
    """Blank-line separated blocks, with their offsets into ``text``.

    Offsets are kept because a passage that later produces a finding has to
    anchor back to a real place, the same as everything else in this tool --
    so the separator is measured rather than assumed to be two characters.
    """
    text = text or ""
    bounds: list[tuple[int, int]] = []
    cursor = 0
    for separator in re.finditer(r"\n[ \t]*\n\s*", text):
        bounds.append((cursor, separator.start()))
        cursor = separator.end()
    bounds.append((cursor, len(text)))

    blocks: list[tuple[int, str]] = []
    for start, end in bounds:
        raw = text[start:end]
        stripped = raw.strip()
        if len(stripped) < MIN_PARAGRAPH_CHARS:
            continue
        offset = start + (len(raw) - len(raw.lstrip()))
        if len(stripped) > MAX_BLOCK_CHARS:
            blocks.extend(_sentence_chunks(offset, stripped))
        else:
            blocks.append((offset, stripped))
    return blocks


def _document_frequency(blocks) -> dict[str, int]:
    counts: dict[str, int] = {}
    for _, block in blocks:
        for term in terms(block):
            counts[term] = counts.get(term, 0) + 1
    return counts


def queryable(claim: str) -> bool:
    """Whether a claim carries enough vocabulary to search on at all.

    An empty result from :func:`retrieve` means one of two very different
    things: the cited paper says nothing on the subject, or the claim was too
    thin to form a query -- "This is well established [12]" has no content
    words to match. The first is a fact about the paper; the second is a
    limitation of ours, and a rule should abstain rather than move on quietly.
    """
    return bool(terms(claim) or numbers(claim))


def retrieve(claim: str, text: str, k: int = 3) -> list[Passage]:
    """The ``k`` paragraphs most likely to bear on ``claim``.

    Scored by inverse document frequency across the cited paper's own
    paragraphs. A term appearing in all of them cannot distinguish one from
    another, whatever it means; a term appearing in two is a strong pointer at
    those two.
    """
    blocks = split_paragraphs(text)
    if not blocks:
        return []

    wanted = terms(claim)
    wanted_numbers = numbers(claim)
    if not wanted and not wanted_numbers:
        return []

    frequency = _document_frequency(blocks)
    total = len(blocks)

    scored: list[Passage] = []
    for start, block in blocks:
        present = terms(block) & wanted
        shared_numbers = numbers(block) & wanted_numbers

        score = sum(
            math.log(total / (1 + frequency.get(term, 0))) + 1.0 for term in present
        )
        score += NUMBER_WEIGHT * len(shared_numbers)

        if score <= 0:
            continue
        scored.append(
            Passage(
                text=block,
                start=start,
                score=score,
                shared=tuple(sorted(shared_numbers) + sorted(present)),
            )
        )

    scored.sort(key=lambda p: (-p.score, p.start))
    return scored[:k]


def relatedness(claim: str, text: str) -> float:
    """How much of the claim's vocabulary appears anywhere in the paper.

    Zero means the cited paper does not contain a single content word of the
    claim -- the signature of a reference list that has slipped by one, or a
    citation copied out of somebody else's bibliography.

    A low score is evidence, not a verdict. Papers say the same thing in
    different words, and this function cannot know that "transformer" and
    "self-attention architecture" are the same subject. It is a screen for
    finding candidates worth a closer look, never a finding on its own.
    """
    wanted = terms(claim)
    if not wanted:
        return 1.0
    return len(wanted & terms(text)) / len(wanted)
