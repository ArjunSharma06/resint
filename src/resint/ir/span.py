"""Spans: the anchor primitive.

Every element the IR extracts carries a span back into its source. Findings
cite spans, never quoted strings -- a string match degrades silently the
moment anything upstream paraphrases or re-renders, and a finding that has
lost its anchor cannot be re-run, diffed, or labelled for the corpus.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

SourceKind = Literal["latex", "pdf", "bib", "code", "config", "readme"]


@dataclass(frozen=True, slots=True)
class Source:
    """A thing a span can point into."""

    id: str
    kind: SourceKind
    path: str | None = None

    def __str__(self) -> str:
        return self.path or self.id


@dataclass(frozen=True, slots=True)
class Span:
    """A half-open character range within a source.

    ``line`` and ``label`` are display affordances only. Equality and
    identity rest on (source, start, end) so that a span stays stable when
    the renderer changes.
    """

    source: Source
    start: int
    end: int
    line: int | None = None
    label: str | None = None

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError(f"span start must be non-negative, got {self.start}")
        if self.end < self.start:
            raise ValueError(f"span end {self.end} precedes start {self.start}")

    @property
    def length(self) -> int:
        return self.end - self.start

    def locate(self) -> str:
        """Human-readable location, e.g. ``train.py:31`` or ``abstract:L4``."""
        base = self.label or str(self.source)
        if self.line is not None:
            return f"{base}:L{self.line}"
        return f"{base}:{self.start}"

    def text_from(self, content: str) -> str:
        return content[self.start : self.end]


@dataclass(frozen=True, slots=True)
class Cell:
    """A table coordinate, which is a span with grid semantics attached."""

    span: Span
    table: str
    row: int
    col: int

    def locate(self) -> str:
        return f"{self.table}:r{self.row}c{self.col}"
