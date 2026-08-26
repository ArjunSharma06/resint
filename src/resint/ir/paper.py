"""Paper-side IR.

Only the slices v1's implemented rules consume are fleshed out here; the
remaining fields on ``Paper`` are declared so the contract is visible and
rules can be written against it before every parser lands.

Reported values keep their raw text. Precision is information -- "3.470"
claims a different granularity than "3.47", and GRIM is meaningless without
it, so nothing is normalized into a float on the way in.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Literal

from .span import Cell, Source, Span

TestKind = Literal["t", "F", "chi2", "r", "z"]
PComparator = Literal["=", "<", ">"]


def _decimals(raw: str) -> int:
    _, _, frac = raw.strip().partition(".")
    return len(frac)


@dataclass(frozen=True, slots=True)
class ReportedMean:
    """A mean reported alongside a sample size, with its stated precision."""

    raw: str
    n: int
    span: Span
    n_span: Span
    items: int = 1
    items_inferred: bool = False
    scale_min: int | None = None
    scale_max: int | None = None
    context: str = ""

    @property
    def value(self) -> Decimal:
        return Decimal(self.raw.strip())

    @property
    def decimals(self) -> int:
        return _decimals(self.raw)

    @property
    def granularity(self) -> int:
        """Number of integer units summed: participants times items each."""
        return self.n * self.items


@dataclass(frozen=True, slots=True)
class StatTest:
    """A null-hypothesis test as the paper reports it."""

    kind: TestKind
    statistic_raw: str
    df1: float | None
    df2: float | None
    p_raw: str
    p_comparator: PComparator
    span: Span
    p_span: Span
    tail: Literal[1, 2] = 2
    context: str = ""

    @property
    def statistic(self) -> float:
        return float(self.statistic_raw)

    @property
    def p_reported(self) -> float:
        return float(self.p_raw)

    @property
    def p_decimals(self) -> int:
        return _decimals(self.p_raw)

    def render(self) -> str:
        if self.kind == "F":
            head = f"F({self.df1:g}, {self.df2:g}) = {self.statistic_raw}"
        elif self.kind in ("t", "chi2", "r"):
            sym = {"t": "t", "chi2": "chi2", "r": "r"}[self.kind]
            head = f"{sym}({self.df1:g}) = {self.statistic_raw}"
        else:
            head = f"z = {self.statistic_raw}"
        return f"{head}, p {self.p_comparator} {self.p_raw}"


@dataclass(frozen=True, slots=True)
class TextSlice:
    """Normalized prose plus the means to anchor into it.

    Bundled rather than exposed as separate ``paper.text`` and
    ``paper.span_at`` attributes so a rule declares one requirement instead of
    two. The gate works on attribute names, and a rule that has to name a
    helper method alongside its data is a rule whose declaration has stopped
    describing what it needs.
    """

    content: str
    _offsets: tuple[int, ...]
    _source: Source
    _line_starts: tuple[int, ...]

    def __len__(self) -> int:
        return len(self.content)

    def __bool__(self) -> bool:
        return bool(self.content)

    def span(self, start: int, end: int, label: str = "") -> Span | None:
        """Anchor a range of this text back into the original source."""
        if not self._offsets or start >= len(self._offsets):
            return None
        lo = self._offsets[start]
        hi = self._offsets[min(end, len(self._offsets)) - 1] + 1 if end > start else lo
        line = bisect_right(self._line_starts, lo)
        return Span(self._source, lo, max(hi, lo + 1), line=line, label=label or "text")


@dataclass(frozen=True, slots=True)
class Citation:
    """One key at one point of use. A key cited five times yields five of these."""

    key: str
    span: Span
    command: str = "cite"
    section: str = ""


# Entry types that legitimately sit outside the major indices. A reference
# that fails to resolve is much weaker evidence of fabrication when it is a
# thesis or a technical report than when it claims to be a journal article.
UNINDEXED_TYPES = frozenset(
    {"phdthesis", "mastersthesis", "techreport", "unpublished", "misc", "booklet"}
)


@dataclass(frozen=True, slots=True)
class BibEntry:
    """A parsed bibliography entry, with spans for the fields rules compare."""

    key: str
    entry_type: str
    fields: dict[str, str]
    span: Span
    field_spans: dict[str, Span] = field(default_factory=dict)

    @property
    def title(self) -> str:
        return self.fields.get("title", "")

    @property
    def year(self) -> str:
        return self.fields.get("year", "")

    @property
    def doi(self) -> str:
        return self.fields.get("doi", "")

    @property
    def authors(self) -> list[str]:
        raw = self.fields.get("author", "")
        return [a.strip() for a in raw.split(" and ") if a.strip()]

    @property
    def venue(self) -> str:
        for name in ("journal", "booktitle", "publisher", "institution", "school"):
            if self.fields.get(name):
                return self.fields[name]
        return ""

    @property
    def likely_unindexed(self) -> bool:
        return self.entry_type in UNINDEXED_TYPES

    @property
    def from_bbl(self) -> bool:
        """Whether this came from a compiled bibliography rather than a .bib.

        BibTeX has already flattened the fields into rendered prose by that
        point, so the title was recovered by convention and may be a
        fragment. Rules that search an index on it should weigh a failure
        less heavily than one against a field read directly.
        """
        return self.entry_type == "bibitem"

    def span_for(self, *names: str) -> Span:
        """Span of the first named field present, falling back to the entry."""
        for name in names:
            if name in self.field_spans:
                return self.field_spans[name]
        return self.span

    def render(self) -> str:
        who = self.authors[0].split(",")[0] if self.authors else "?"
        return f"{who} {self.year}, {self.title!r}" if self.title else f"{who} {self.year}"


@dataclass(frozen=True, slots=True)
class Number:
    """A numeric value in running text or a table cell."""

    raw: str
    span: Span
    label: str = ""
    unit: str | None = None
    cell: Cell | None = None
    section: str = ""

    @property
    def value(self) -> Decimal:
        return Decimal(self.raw.strip())

    @property
    def decimals(self) -> int:
        return _decimals(self.raw)

    def render(self) -> str:
        return f"{self.raw}{self.unit or ''}"

    def locate(self) -> str:
        return self.cell.locate() if self.cell else self.span.locate()


@dataclass
class Paper:
    """The paper-side IR. Fields are populated lazily by declared need."""

    source_id: str
    text: TextSlice | None = None
    sections: list = field(default_factory=list)
    sentences: list = field(default_factory=list)
    equations: list = field(default_factory=list)
    tables: list = field(default_factory=list)
    numbers: list[Number] = field(default_factory=list)
    hyperparameters: list[Number] = field(default_factory=list)
    means: list[ReportedMean] = field(default_factory=list)
    stats: list[StatTest] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    bib: list[BibEntry] = field(default_factory=list)
    # key -> Resolution. Declaring paper.resolutions is what causes any
    # network access at all; no rule asking for it means no lookups.
    resolutions: dict = field(default_factory=dict)
    claims: list = field(default_factory=list)
    unchecked: list[str] = field(default_factory=list)
