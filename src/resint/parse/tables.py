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
_MARKUP = re.compile(r"\\[a-zA-Z]+\s*\*?")
_NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?")


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
    text = _MULTICOLUMN.sub("", raw)
    text = _MULTIROW.sub("", text)
    text = _RULE_CMD.sub("", text)
    text = _MARKUP.sub(" ", text)
    text = text.replace("$", "").replace("{", "").replace("}", "")
    text = text.replace("\\%", "%").replace("~", " ")
    return re.sub(r"\s+", " ", text).strip()


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


def extract_tables(raw: str, src: Source) -> list[Table]:
    """Every tabular in the source, as a grid of anchored cells."""
    tables: list[Table] = []

    for index, m in enumerate(_TABULAR.finditer(raw), 1):
        env = m.group(1)
        after = m.end()

        # tabular takes an optional [pos] then a mandatory column spec;
        # tabular* and tabularx take a width argument first.
        if env in ("tabular*", "tabularx"):
            while after < len(raw) and raw[after] in " \n":
                after += 1
            if after < len(raw) and raw[after] == "{":
                after = _skip_group(raw, after)
        while after < len(raw) and raw[after] in " \n":
            after += 1
        if after < len(raw) and raw[after] == "[":
            close = raw.find("]", after)
            after = close + 1 if close != -1 else after
        while after < len(raw) and raw[after] in " \n":
            after += 1

        spec = ""
        if after < len(raw) and raw[after] == "{":
            body_start = _skip_group(raw, after)
            spec = raw[after + 1 : body_start - 1]
            after = body_start

        end_tag = f"\\end{{{env}}}"
        stop = raw.find(end_tag, after)
        if stop == -1:
            continue

        body = raw[after:stop]
        caption, label, float_start = _surrounding_float(raw, m.start())

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
                cells.append(
                    TableCell(
                        raw=cell_raw,
                        text=_clean(cell_raw),
                        span=Span(
                            src,
                            span_start + lead,
                            max(span_start + lead + len(stripped), span_start + lead + 1),
                            line=raw.count("\n", 0, span_start + lead) + 1,
                            label=f"table{index}:r{len(rows)}c{col}",
                        ),
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
                span=Span(
                    src,
                    float_start,
                    stop + len(end_tag),
                    line=raw.count("\n", 0, float_start) + 1,
                    label=f"table{index}",
                ),
            )
        )

    return tables
