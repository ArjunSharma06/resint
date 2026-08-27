"""Tables, references and citations out of JATS XML.

The prose comes from :mod:`resint.parse.jats`; this is the structured matter
alongside it. Each function produces exactly what its LaTeX counterpart
produces -- ``Table``, ``BibEntry``, ``Citation`` -- so every rule downstream
reads a journal article and a preprint through the same IR and cannot tell
which it was given.

Every span points into the raw XML, because that is the file the reader opens.
"""

from __future__ import annotations

import re

from ..ir.paper import BibEntry, Citation
from ..ir.span import Source, Span
from .bibtex import BibFile
from .jats import _TAG, _emit_text, _masked, _Builder
from .tables import Table, TableCell

_ATTR = re.compile(r"""(?P<key>[\w.:-]+)\s*=\s*["'](?P<value>[^"']*)["']""")

#: JATS marks a bibliographic citation with ref-type="bibr". The other kinds --
#: figures, tables, sections, footnotes -- are cross references to the paper's
#: own furniture and are not citations of anything.
_XREF = re.compile(r"<xref\b(?P<attrs>[^>]*)>", re.IGNORECASE)

_ROW = re.compile(r"<tr\b[^>]*>(?P<body>.*?)</tr>", re.DOTALL | re.IGNORECASE)
_CELL = re.compile(r"<(?P<tag>t[dh])\b(?P<attrs>[^>]*)>(?P<body>.*?)</(?P=tag)>",
                   re.DOTALL | re.IGNORECASE)


def _attrs(raw: str) -> dict:
    return {m.group("key").lower(): m.group("value") for m in _ATTR.finditer(raw or "")}


def _plain(raw: str, lo: int, hi: int) -> str:
    """Text content of a raw range, tags removed and entities decoded."""
    masked = _masked(raw)
    builder = _Builder(chars=[], offsets=[])
    cursor = lo
    for m in _TAG.finditer(masked, lo, hi):
        if m.start() > cursor:
            _emit_text(raw, cursor, m.start(), builder)
        cursor = m.end()
    if cursor < hi:
        _emit_text(raw, cursor, hi, builder)
    return "".join(builder.chars).strip()


def _elements(raw: str, name: str):
    """Raw (start-of-content, end-of-content, open-tag-start) per element."""
    masked = _masked(raw)
    depth = 0
    begin = content = 0
    for m in _TAG.finditer(masked):
        if m.group("name").lower() != name or m.group("empty"):
            continue
        if not m.group("closing"):
            if depth == 0:
                begin, content = m.start(), m.end()
            depth += 1
        else:
            depth -= 1
            if depth == 0:
                yield content, m.start(), begin
            depth = max(depth, 0)


# --- tables --------------------------------------------------------------


def extract_tables(raw: str, src: Source, doc=None) -> list[Table]:
    """Every ``<table-wrap>``, as the same Table the LaTeX path produces."""
    out: list[Table] = []

    for index, (lo, hi, begin) in enumerate(_elements(raw, "table-wrap"), 1):
        caption = ""
        for c_lo, c_hi, _ in _elements(raw[lo:hi], "caption"):
            caption = _plain(raw[lo:hi], c_lo, c_hi)
            break

        label = ""
        for l_lo, l_hi, _ in _elements(raw[lo:hi], "label"):
            label = _plain(raw[lo:hi], l_lo, l_hi)
            break

        rows: list[list[TableCell]] = []
        for row_index, row in enumerate(_ROW.finditer(raw, lo, hi)):
            cells: list[TableCell] = []
            body_at = row.start("body")
            for col, cell in enumerate(_CELL.finditer(row.group("body"))):
                start = body_at + cell.start("body")
                end = body_at + cell.end("body")
                text = _plain(raw, start, end)
                attrs = _attrs(cell.group("attrs"))
                try:
                    colspan = max(1, int(attrs.get("colspan", "1")))
                except ValueError:
                    colspan = 1
                cells.append(
                    TableCell(
                        raw=raw[start:end],
                        text=text,
                        span=_anchor(doc, src, raw, start, max(end, start + 1),
                                     f"table{index}:r{row_index}c{col}"),
                        row=row_index,
                        col=col,
                        colspan=colspan,
                    )
                )
            if cells:
                rows.append(cells)

        if not rows:
            continue

        # Effective width, counting a spanning cell for every column it
        # covers. A section header written <td colspan="4"> is one cell across
        # four columns; measuring it as one made real tables read as ragged and
        # skipped them wholesale -- widths [1, 3, 2, 4, 4, 4, 4, 1] for a table
        # that is four columns throughout.
        widths = {sum(c.colspan for c in r) for r in rows}
        irregular = (
            f"row widths disagree ({sorted(widths)})" if len(widths) > 1 else ""
        )

        out.append(
            Table(
                index=index,
                rows=rows,
                span=_anchor(doc, src, raw, begin, hi, f"table{index}"),
                caption=caption,
                label=label,
                irregular=irregular,
            )
        )

    return out


def _anchor(doc, src: Source, raw: str, start: int, end: int, label: str) -> Span:
    if doc is not None:
        return doc.anchor(src, start, end, label)
    return Span(
        src, start, max(end, start + 1), line=raw.count("\n", 0, start) + 1, label=label
    )


# --- references ----------------------------------------------------------

_YEAR = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")


def extract_bib(raw: str, src: Source, doc=None) -> BibFile:
    """Every ``<ref>``, as a BibEntry keyed by its XML id.

    The id is what ``<xref rid=...>`` points at, so it plays exactly the role a
    BibTeX key plays -- which is what lets ``bib/orphans`` work unchanged.
    """
    out = BibFile()

    for lo, hi, begin in _elements(raw, "ref"):
        attrs = _attrs(raw[begin : raw.find(">", begin) + 1])
        key = attrs.get("id", "")
        if not key:
            out.malformed.append("a <ref> with no id attribute")
            continue

        fields: dict[str, str] = {}
        field_spans: dict[str, Span] = {}

        for tag, name in (
            ("article-title", "title"),
            ("source", "journal"),
            ("year", "year"),
        ):
            for f_lo, f_hi, _ in _elements(raw[lo:hi], tag):
                value = _plain(raw[lo:hi], f_lo, f_hi)
                if not value:
                    break
                fields[name] = value
                field_spans[name] = _anchor(
                    doc, src, raw, lo + f_lo, lo + f_hi, f"[{key}].{name}"
                )
                break

        surnames = [
            _plain(raw[lo:hi], s_lo, s_hi) for s_lo, s_hi, _ in _elements(raw[lo:hi], "surname")
        ]
        if surnames:
            fields["author"] = " and ".join(s for s in surnames if s)

        for tag, name in (("pub-id", "doi"),):
            for f_lo, f_hi, f_begin in _elements(raw[lo:hi], tag):
                kind = _attrs(raw[lo:hi][f_begin : (raw[lo:hi]).find(">", f_begin) + 1])
                if kind.get("pub-id-type", "").lower() == "doi":
                    fields[name] = _plain(raw[lo:hi], f_lo, f_hi)
                    break

        if "year" not in fields:
            # Structured markup is optional; plenty of references are a single
            # <mixed-citation> string, and the year is the one field that can
            # be recovered from it without guessing.
            years = _YEAR.findall(_plain(raw, lo, hi))
            if years:
                fields["year"] = years[-1]

        out.entries.append(
            BibEntry(
                key=key,
                entry_type="ref",
                fields=fields,
                span=_anchor(doc, src, raw, begin, hi, f"[{key}]"),
                field_spans=field_spans,
            )
        )

    return out


# --- citations -----------------------------------------------------------


def extract_citations(raw: str, src: Source, doc=None) -> list[Citation]:
    """Every ``<xref ref-type="bibr">``, one record per key per use site.

    ``rid`` may name several references at once, exactly as ``\\cite{a,b}``
    does, so one element can be several citations.
    """
    out: list[Citation] = []

    for m in _XREF.finditer(_masked(raw)):
        attrs = _attrs(m.group("attrs"))
        if attrs.get("ref-type", "").lower() != "bibr":
            continue

        rid = attrs.get("rid", "")
        if not rid:
            continue

        cursor = 0
        for piece in rid.split():
            key = piece.strip()
            if key:
                out.append(
                    Citation(
                        key=key,
                        command="xref",
                        span=_anchor(
                            doc, src, raw, m.start(), m.end(), f"<xref {key}>"
                        ),
                    )
                )
            cursor += len(piece) + 1

    return out
