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


def extract_citations(raw: str, src: Source) -> list[Citation]:
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
                        span=Span(
                            src,
                            offset,
                            offset + len(key),
                            line=raw.count("\n", 0, offset) + 1,
                            label=f"\\{m.group('cmd')}{{{key}}}",
                        ),
                    )
                )
            cursor += len(piece) + 1

    return out
