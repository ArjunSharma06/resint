"""Table extraction from LaTeX ``tabular`` environments.

This runs on raw source rather than normalized text, because normalization
deliberately destroys the two characters that carry all the structure: ``&``
separates cells and ``\\\\`` ends rows. By the time text is readable prose the
grid is gone.

Recovery is deliberately conservative. A table whose row lengths disagree, or
whose column specification cannot be read, is returned marked ``irregular``
and the arithmetic rules skip it -- a misparsed grid produces confident
nonsense, and "could not read this table" is a far better report than a
column total that was never in the paper.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

from ..ir.span import Source, Span
from .latex import _skip_group

_TABULAR = re.compile(r"\\begin\{(tabular\*?|tabularx|longtable|array)\}")
_CAPTION = re.compile(r"\\caption\s*(?:\[[^\]]*\])?\s*\{")
_LABEL = re.compile(r"\\label\s*\{([^}]*)\}")
_RULE_CMD = re.compile(
    r"\\(?:hline|toprule|midrule|bottomrule|cmidrule|cline|addlinespace|"
    r"noalign|rowcolor|hhline)\b(?:\s*\([^)]*\))?(?:\s*\{[^}]*\})?(?:\s*\[[^\]]*\])?"
)
_MULTICOLUMN = re.compile(r"\\multicolumn\s*\{\s*(\d+)\s*\}")
_MULTIROW = re.compile(r"\\multirow\s*\{\s*(\d+)\s*\}")

# The whole spanning form, so the content survives and the span count and
# alignment specification do not.
_SPANNING = re.compile(
    r"\\multi(?:column|row)\s*\{[^{}]*\}\s*(?:\[[^\]]*\])?\s*\{[^{}]*\}\s*\{([^{}]*)\}"
)

# Layout commands whose braced arguments are measurements, not content.
# Stripping the command but keeping the braces turns \rule{0pt}{2.0ex} into
# the cell text "0pt2.0ex", which then reads as data.
_LAYOUT_CMD = re.compile(
    r"\\(?:rule|hspace|vspace|raisebox|makebox|parbox|resizebox|scalebox|"
    r"adjustbox|rowcolor|cellcolor|columncolor|arrayrulecolor)\*?"
    r"(?:\s*\[[^\]]*\])*(?:\s*\{[^{}]*\})*"
)
_MARKUP = re.compile(r"\\[a-zA-Z]+\s*\*?")
_NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?")

# A comment runs to end of line unless the % is escaped. Blanked rather than
# removed so every offset in the source stays valid.
_COMMENT = re.compile(r"(?<!\\)%[^\n]*")


def uncomment(text: str) -> str:
    """Blank out comment bodies, preserving length so offsets stay valid.

    Tables are read from raw source rather than normalized text, because
    normalization destroys the & and \\\\ that carry the grid. That means
    comments have to be handled here too -- a real paper carries whole
    commented-out tables from earlier drafts, and parsing one produces a
    grid of \\hline rows that is reported as malformed.
    """
    return _COMMENT.sub(lambda m: " " * len(m.group(0)), text)


@dataclass(frozen=True, slots=True)
class TableCell:
    raw: str
    text: str
    span: Span
    row: int
    col: int
    colspan: int = 1

    @property
    def is_blank(self) -> bool:
        return not self.text.strip()

    @property
    def number(self) -> Decimal | None:
        """The single numeric value in this cell, if it holds exactly one.

        Cells like "94.2 +/- 0.3" or "12/48" hold two numbers and no single
        value; returning the first would silently misread the table.
        """
        found = _NUMBER.findall(self.text)
        if len(found) != 1:
            return None
        try:
            return Decimal(found[0])
        except InvalidOperation:
            return None

    @property
    def decimals(self) -> int:
        found = _NUMBER.findall(self.text)
        if len(found) != 1:
            return 0
        _, _, frac = found[0].partition(".")
        return len(frac)

    def locate(self) -> str:
        return f"r{self.row}c{self.col}"


@dataclass
class Table:
    index: int
    rows: list[list[TableCell]]
    span: Span
    caption: str = ""
    label: str = ""
    irregular: str = ""

    @property
    def name(self) -> str:
        return f"table{self.index}"

    @property
    def width(self) -> int:
        return max((len(r) for r in self.rows), default=0)

    @property
    def header(self) -> list[str]:
        return [c.text for c in self.rows[0]] if self.rows else []

    def column(self, index: int) -> list[TableCell]:
        return [r[index] for r in self.rows if index < len(r)]

    def body_rows(self) -> list[list[TableCell]]:
        """Rows below the header."""
        return self.rows[1:] if len(self.rows) > 1 else []

    def cells(self):
        for row in self.rows:
            yield from row


def _clean(raw: str) -> str:
    text = _LAYOUT_CMD.sub(" ", raw)
    # \multicolumn{2}{c}{BLEU} -> BLEU. Dropping only the command leaves the
    # alignment spec glued to the content as "cBLEU", which then fails to
    # match the metric name the prose uses.
    text = _SPANNING.sub(r"\1", text)
    text = _MULTICOLUMN.sub("", text)
    text = _MULTIROW.sub("", text)
    text = _RULE_CMD.sub("", text)
    text = _MARKUP.sub(" ", text)
    text = text.replace("$", "").replace("{", "").replace("}", "")
    text = text.replace("\\%", "%").replace("~", " ")
    return re.sub(r"\s+", " ", text).strip()


def _anchor(doc, src: Source, raw: str, start: int, end: int, label: str) -> Span:
    """Anchor a raw-text range, resolving it to its real file when spliced.

    Without ``doc`` a cell in a multi-file paper is anchored to an offset in
    the combined text, which corresponds to no file the author has open — and
    puts the cell in a different coordinate system from the prose anchor it
    gets compared against. That mismatch is invisible until something audits
    it.
    """
    if doc is not None:
        return doc.anchor(src, start, end, label)
    return Span(src, start, max(end, start + 1), line=raw.count("\n", 0, start) + 1, label=label)


def _split_rows(body: str) -> list[tuple[int, str]]:
    """Split on unescaped row terminators, keeping each row's source offset."""
    rows: list[tuple[int, str]] = []
    start, i, n = 0, 0, len(body)
    while i < n:
        if body[i] == "\\" and i + 1 < n:
            if body[i + 1] == "\\":
                rows.append((start, body[start:i]))
                i += 2
                # \\[2pt] style spacing argument
                if i < n and body[i] == "[":
                    close = body.find("]", i)
                    if close != -1:
                        i = close + 1
                start = i
                continue
            i += 2
            continue
        i += 1
    if body[start:].strip():
        rows.append((start, body[start:]))
    return rows


def _split_cells(row: str) -> list[tuple[int, str]]:
    """Split a row on unescaped ampersands, keeping each cell's offset."""
    cells: list[tuple[int, str]] = []
    depth, start, i, n = 0, 0, 0, len(row)
    while i < n:
        c = row[i]
        if c == "\\":
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        elif c == "&" and depth == 0:
            cells.append((start, row[start:i]))
            start = i + 1
        i += 1
    cells.append((start, row[start:]))
    return cells


def _read_braced(src: str, match_end: int) -> tuple[str, int]:
    """Read the braced group starting at ``match_end - 1``."""
    end = _skip_group(src, match_end - 1)
    return src[match_end : end - 1], end


def _surrounding_float(src: str, tabular_start: int) -> tuple[str, str, int]:
    """Caption, label, and float start for the table enclosing this tabular."""
    window_start = max(0, tabular_start - 4000)
    opener = src.rfind("\\begin{table", window_start, tabular_start)
    if opener == -1:
        opener = tabular_start

    closer = src.find("\\end{table", tabular_start)
    window_end = closer + 40 if closer != -1 else min(len(src), tabular_start + 4000)
    window = src[opener:window_end]

    caption = ""
    cap = _CAPTION.search(window)
    if cap:
        body, _ = _read_braced(window, cap.end())
        caption = _clean(body)

    lab = _LABEL.search(window)
    label = lab.group(1).strip() if lab else ""
    return caption, label, opener


def extract_tables(raw: str, src: Source, doc=None) -> list[Table]:
    """Every tabular in the source, as a grid of anchored cells.

    ``doc`` carries the region map for a document spliced from several
    files. Without it, cells in a multi-file paper are anchored to offsets
    in the combined text -- which corresponds to no file the author has
    open, and puts a table anchor in a different coordinate system from
    the prose anchor it is compared against.
    """
    tables: list[Table] = []

    # Offsets are preserved by blanking rather than deleting, so every span
    # produced below still points at the right place in the file the user has
    # open. Spans are taken from `raw`; only the parsing reads `scannable`.
    scannable = uncomment(raw)

    for index, m in enumerate(_TABULAR.finditer(scannable), 1):
        env = m.group(1)
        after = m.end()

        # tabular takes an optional [pos] then a mandatory column spec;
        # tabular* and tabularx take a width argument first.
        if env in ("tabular*", "tabularx"):
            while after < len(raw) and raw[after] in " \n":
                after += 1
            if after < len(scannable) and scannable[after] == "{":
                after = _skip_group(scannable, after)
        while after < len(raw) and raw[after] in " \n":
            after += 1
        if after < len(scannable) and scannable[after] == "[":
            close = scannable.find("]", after)
            after = close + 1 if close != -1 else after
        while after < len(raw) and raw[after] in " \n":
            after += 1

        spec = ""
        if after < len(scannable) and scannable[after] == "{":
            body_start = _skip_group(scannable, after)
            spec = scannable[after + 1 : body_start - 1]
            after = body_start

        end_tag = f"\\end{{{env}}}"
        stop = scannable.find(end_tag, after)
        if stop == -1:
            continue

        body = scannable[after:stop]
        caption, label, float_start = _surrounding_float(scannable, m.start())

        rows: list[list[TableCell]] = []
        for r, (row_offset, row_text) in enumerate(_split_rows(body)):
            if not _clean(row_text):
                continue
            cells: list[TableCell] = []
            col = 0
            for cell_offset, cell_raw in _split_cells(row_text):
                span_start = after + row_offset + cell_offset
                stripped = cell_raw.strip()
                lead = len(cell_raw) - len(cell_raw.lstrip())
                mc = _MULTICOLUMN.search(cell_raw)
                colspan = int(mc.group(1)) if mc else 1
                begin = span_start + lead
                finish = max(begin + len(stripped), begin + 1)
                # Not `label` — that name holds the table's own \label{} for
                # the whole of this loop, and shadowing it silently gave every
                # table the label of its last cell.
                cell_label = f"table{index}:r{len(rows)}c{col}"
                cells.append(
                    TableCell(
                        raw=cell_raw,
                        text=_clean(cell_raw),
                        span=_anchor(doc, src, raw, begin, finish, cell_label),
                        row=len(rows),
                        col=col,
                        colspan=colspan,
                    )
                )
                col += colspan
            rows.append(cells)

        irregular = ""
        if not rows:
            irregular = "no rows recovered"
        elif not spec.strip():
            irregular = "column specification unreadable"
        else:
            widths = {sum(c.colspan for c in row) for row in rows}
            if len(widths) > 1:
                irregular = f"row widths disagree ({sorted(widths)})"

        tables.append(
            Table(
                index=index,
                rows=rows,
                caption=caption,
                label=label,
                irregular=irregular,
                span=_anchor(
                    doc, src, raw, float_start, stop + len(end_tag), f"table{index}"
                ),
            )
        )

    return tables
