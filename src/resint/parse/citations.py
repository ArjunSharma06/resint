"""Citation extraction from raw LaTeX.

This reads the raw source rather than normalized text, because normalization
deliberately discards citation keys -- a bare key mid-sentence corrupts
sentence segmentation and looks like data to the numeric extractors. Here the
key is the whole point, so it gets its own pass.

Each use site produces its own record. A key cited five times is five
citations, because "cited once in related work" and "cited five times as
support for the central claim" are different situations.
"""

from __future__ import annotations

import re

from ..ir.paper import Citation
from ..ir.span import Source, Span

_CITE = re.compile(
    r"\\(?P<cmd>[Cc]ite[a-zA-Z]*|nocite)\s*"
    r"(?:\[[^\]]*\]\s*){0,2}"
    r"\{(?P<keys>[^{}]*)\}"
)

_COMMENT_LINE = re.compile(r"(?<!\\)%[^\n]*")


def _uncommented(text: str) -> str:
    """Blank out comment bodies, preserving length so offsets stay valid."""
    return _COMMENT_LINE.sub(lambda m: " " * len(m.group(0)), text)


def _anchor(doc, src: Source, raw: str, start: int, end: int, label: str) -> Span:
    """Anchor a raw-text range, resolving it to its real file when spliced.

    Without ``doc`` a citation in a multi-file paper carries an offset into the
    combined text while naming the root file -- a coordinate system matching no
    file on disk. A twelve-file submission produced an offset of 47,438 into a
    root file 15,585 characters long, and in a two-file one the offset resolved
    but the line number was fifteen too high, counted over content spliced in
    ahead of it. The anchor audit was the only thing that noticed either.
    """
    if doc is not None:
        return doc.anchor(src, start, end, label)
    return Span(
        src, start, max(end, start + 1), line=raw.count("\n", 0, start) + 1, label=label
    )


def extract_citations(raw: str, src: Source, doc=None) -> list[Citation]:
    """Every citation use site, one record per key per site."""
    scannable = _uncommented(raw)
    out: list[Citation] = []

    for m in _CITE.finditer(scannable):
        block = m.group("keys")
        block_start = m.start("keys")
        cursor = 0

        for piece in block.split(","):
            key = piece.strip()
            if key:
                offset = block_start + cursor + (len(piece) - len(piece.lstrip()))
                out.append(
                    Citation(
                        key=key,
                        command=m.group("cmd").lower(),
                        span=_anchor(
                            doc,
                            src,
                            raw,
                            offset,
                            offset + len(key),
                            f"\\{m.group('cmd')}{{{key}}}",
                        ),
                    )
                )
            cursor += len(piece) + 1

    return out
