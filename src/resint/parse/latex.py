"""LaTeX normalization with a truthful offset map.

Every character of normalized text remembers where it came from in the
original source. That is the whole point: a finding that says "line 98" has
to mean line 98 of the file the author is looking at, not line 98 of some
intermediate the tool invented. Carrying the map through normalization is
cheaper than reconstructing it afterwards, and it is the difference between
an anchor and a guess.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass, field

from ..ir.span import Source, Span

# Commands whose single braced argument is kept as running text.
_KEEP_ARG = {
    "textbf", "textit", "textrm", "textsf", "texttt", "textsc", "textnormal",
    "emph", "underline", "text", "mathrm", "mathbf", "mathit", "operatorname",
    "section", "subsection", "subsubsection", "paragraph", "title", "caption",
}

# Commands dropped together with their braced argument.
#
# Citations are dropped from running text rather than kept: a bare key like
# "dosovitskiy2020" sitting mid-sentence corrupts sentence segmentation and
# can be mistaken for data by the numeric extractors. The bib/ rules read
# citations from the raw source in their own pass, where the key is the
# point rather than noise.
_DROP_ARG = {
    "label", "ref", "eqref", "pageref", "includegraphics", "usepackage",
    "documentclass", "bibliographystyle", "bibliography", "input", "include",
    "hspace", "vspace", "setlength", "color", "textcolor", "footnote",
    "cite", "citep", "citet", "citeauthor", "citeyear", "citealp",
    "autoref", "cref", "Cref", "url", "href",
}

# Definition commands. The whole definition is skipped -- name, optional
# arity, and body -- otherwise the body leaks into the text as content.
_DEFINE_CMD = {"newcommand", "renewcommand", "providecommand", "def", "DeclareMathOperator"}

# Commands dropped entirely, argument-less.
_DROP_BARE = {
    "noindent", "centering", "clearpage", "newpage", "hline", "toprule",
    "midrule", "bottomrule", "small", "footnotesize", "scriptsize", "large",
    "Large", "huge", "normalsize", "bfseries", "itshape", "raggedright",
    "maketitle", "tableofcontents", "linebreak", "par",
}

# Environments whose body is discarded wholesale.
_DROP_ENV = {
    "figure", "figure*", "thebibliography", "lstlisting", "verbatim", "comment",
}

# Symbol commands mapped to plain text so statistics survive normalization.
_SYMBOLS = {
    "chi": "chi", "alpha": "alpha", "beta": "beta", "mu": "mu",
    "sigma": "sigma", "leq": "<=", "geq": ">=", "neq": "!=", "approx": "~",
    "times": "x", "pm": "+/-", "ldots": "...", "cdot": ".",
    "%": "%", "&": "&", "_": "_", "#": "#", "$": "$", "{": "{", "}": "}",
}

_SECTION_CMD = re.compile(r"\\(section|subsection|subsubsection)\*?\s*\{")
_NEWCOMMAND = re.compile(
    r"\\(?:newcommand|renewcommand|providecommand)\s*\*?\s*"
    r"\{?\\(?P<name>[A-Za-z@]+)\}?\s*(?:\[(?P<arity>\d+)\])?\s*\{"
)


@dataclass(frozen=True, slots=True)
class Section:
    name: str
    kind: str
    start: int
    end: int


@dataclass
class Normalized:
    """Plain text plus the map back into the source it came from."""

    text: str
    offsets: list[int]
    raw: str
    sections: list[Section] = field(default_factory=list)
    # Set when the document was spliced from several files. Lets a raw
    # offset be translated back to the file it actually came from, so
    # "results.tex:42" means line 42 of results.tex rather than line 42 of
    # a concatenation that exists only inside this process.
    regions: tuple = ()
    files: dict | None = None
    _line_starts: list[int] = field(default_factory=list, repr=False)
    _file_lines: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if len(self.offsets) != len(self.text):
            raise ValueError(
                f"offset map has {len(self.offsets)} entries "
                f"for {len(self.text)} characters"
            )
        if not self._line_starts:
            starts = [0]
            for i, ch in enumerate(self.raw):
                if ch == "\n":
                    starts.append(i + 1)
            self._line_starts = starts

    def raw_offset(self, index: int) -> int:
        """Source offset for a position in the normalized text."""
        if not self.offsets:
            return 0
        if index >= len(self.offsets):
            return self.offsets[-1] + 1
        return self.offsets[max(0, index)]

    def raw_range(self, start: int, end: int) -> tuple[int, int]:
        lo = self.raw_offset(start)
        hi = self.raw_offset(end - 1) + 1 if end > start else lo
        return lo, max(hi, lo)

    def line_of(self, raw_offset: int) -> int:
        origin = self.origin_of(raw_offset)
        if origin is None:
            return bisect_right(self._line_starts, raw_offset)
        _, local = origin
        return bisect_right(self._starts_for(origin[0]), local)

    def origin_of(self, raw_offset: int) -> tuple[str, int] | None:
        """(filename, offset within that file) for a combined-text offset."""
        for region in self.regions:
            if region.start <= raw_offset < region.end:
                return region.name, region.local(raw_offset)
        return None

    def _starts_for(self, name: str) -> list[int]:
        cached = self._file_lines.get(name)
        if cached is None:
            text = (self.files or {}).get(name, "")
            starts = [0]
            for i, ch in enumerate(text):
                if ch == "\n":
                    starts.append(i + 1)
            self._file_lines[name] = starts
            cached = starts
        return cached

    def anchor(self, src, raw_start: int, raw_end: int, label: str = ""):
        """A Span for a range of the *raw* text, resolved to its real file.

        Extractors that work on raw source rather than normalized text --
        tables, chiefly -- need this too. Doing the resolution in only one of
        them produced findings whose two anchors were in different coordinate
        systems: prose local to its included file, table cells offset into the
        spliced whole. The line numbers on the second kind matched no file the
        author had open.
        """
        origin = self.origin_of(raw_start)
        if origin is None:
            return Span(
                src,
                raw_start,
                max(raw_end, raw_start + 1),
                line=self.line_of(raw_start),
                label=label or str(src),
            )

        name, local = origin
        length = max(raw_end - raw_start, 1)
        return Span(
            Source(name, src.kind, path=name),
            local,
            local + length,
            line=self.line_of(raw_start),
            label=label or name,
        )

    def section_at(self, index: int) -> str:
        for sec in self.sections:
            if sec.start <= index < sec.end:
                return sec.name
        return ""


class _Writer:
    __slots__ = ("chars", "offsets")

    def __init__(self) -> None:
        self.chars: list[str] = []
        self.offsets: list[int] = []

    def put(self, text: str, offset: int) -> None:
        for ch in text:
            self.chars.append(ch)
            self.offsets.append(offset)

    def space(self, offset: int) -> None:
        if self.chars and self.chars[-1] not in " \n":
            self.put(" ", offset)

    @property
    def pos(self) -> int:
        return len(self.chars)


def _find_macros(src: str) -> dict[str, str]:
    """Collect zero-argument \\newcommand definitions for expansion."""
    macros: dict[str, str] = {}
    for m in _NEWCOMMAND.finditer(src):
        if m.group("arity"):
            continue
        depth, i = 1, m.end()
        while i < len(src) and depth:
            if src[i] == "\\":
                i += 2
                continue
            depth += (src[i] == "{") - (src[i] == "}")
            i += 1
        body = src[m.end() : i - 1]
        if "\\" not in body and len(body) < 200:
            macros[m.group("name")] = body
    return macros


def _read_command(src: str, i: int) -> tuple[str, int]:
    """Read a command name at the backslash. Returns (name, next index)."""
    j = i + 1
    if j < len(src) and not src[j].isalpha():
        return src[j], j + 1
    while j < len(src) and src[j].isalpha():
        j += 1
    return src[i + 1 : j], j


def _skip_group(src: str, i: int) -> int:
    """Given the index of '{', return the index just past its match."""
    depth, j = 1, i + 1
    while j < len(src) and depth:
        if src[j] == "\\":
            j += 2
            continue
        depth += (src[j] == "{") - (src[j] == "}")
        j += 1
    return j


def _skip_optional(src: str, i: int) -> int:
    if i < len(src) and src[i] == "[":
        j = src.find("]", i)
        return j + 1 if j != -1 else i
    return i


def normalize(raw: str, regions: tuple = (), files: dict | None = None) -> Normalized:
    """Normalize LaTeX source to running text, preserving source offsets."""
    macros = _find_macros(raw)
    out = _Writer()
    sections: list[Section] = []
    pending: list[tuple[str, str, int]] = []

    i, n = 0, len(raw)
    while i < n:
        ch = raw[i]

        if ch == "%" and (i == 0 or raw[i - 1] != "\\"):
            nl = raw.find("\n", i)
            i = n if nl == -1 else nl + 1
            continue

        if ch == "\\":
            if raw.startswith("\\begin{", i) or raw.startswith("\\end{", i):
                brace = raw.index("{", i)
                close = raw.index("}", brace)
                env = raw[brace + 1 : close]
                after = _skip_optional(raw, close + 1)
                while after < n and raw[after] == "{":
                    after = _skip_group(raw, after)
                if raw.startswith("\\begin{", i) and env in _DROP_ENV:
                    end_tag = "\\end{" + env + "}"
                    stop = raw.find(end_tag, after)
                    i = n if stop == -1 else stop + len(end_tag)
                    continue
                out.space(i)
                i = after
                continue

            sec = _SECTION_CMD.match(raw, i)
            if sec:
                body_end = _skip_group(raw, sec.end() - 1)
                name = re.sub(r"[\\{}]", "", raw[sec.end() : body_end - 1]).strip()
                if pending:
                    prev, kind, start = pending.pop()
                    sections.append(Section(prev, kind, start, out.pos))
                out.space(i)
                pending.append((name, sec.group(1), out.pos))
                i = body_end
                continue

            name, j = _read_command(raw, i)

            if name in _DEFINE_CMD:
                j = _skip_optional(raw, j)
                while j < n and raw[j] in " \n":
                    j += 1
                if j < n and raw[j] == "{":       # {\name}
                    j = _skip_group(raw, j)
                elif j < n and raw[j] == "\\":    # bare \name, as in \def
                    _, j = _read_command(raw, j)
                j = _skip_optional(raw, j)        # [arity]
                j = _skip_optional(raw, j)        # [default]
                while j < n and raw[j] in " \n":
                    j += 1
                if j < n and raw[j] == "{":       # {body}
                    j = _skip_group(raw, j)
                out.space(i)
                i = j
                continue

            if name in macros:
                out.put(macros[name], i)
                i = j
                continue
            if name in _SYMBOLS:
                out.put(_SYMBOLS[name], i)
                i = j
                continue
            if name in _DROP_BARE:
                out.space(i)
                i = j
                continue
            if name in _DROP_ARG:
                j = _skip_optional(raw, j)
                while j < n and raw[j] in " \n":
                    j += 1
                if j < n and raw[j] == "{":
                    j = _skip_group(raw, j)
                out.space(i)
                i = j
                continue
            if name in _KEEP_ARG:
                j = _skip_optional(raw, j)
                while j < n and raw[j] in " \n":
                    j += 1
                if j < n and raw[j] == "{":
                    inner_end = _skip_group(raw, j)
                    inner = normalize(raw[j + 1 : inner_end - 1])
                    for k, c in enumerate(inner.text):
                        out.put(c, j + 1 + inner.offsets[k])
                    i = inner_end
                else:
                    i = j
                continue
            if name == "\\":
                out.put("\n", i)
                i = j
                continue

            # Unknown command: drop the token, keep whatever follows it.
            out.space(i)
            i = j
            continue

        if ch in "{}$":
            i += 1
            continue
        if ch in "~&":
            out.put(" ", i)
            i += 1
            continue
        if ch == "\n":
            nxt = i + 1
            while nxt < n and raw[nxt] in " \t":
                nxt += 1
            if nxt < n and raw[nxt] == "\n":
                out.put("\n", i)
                i = nxt
                continue
            out.space(i)
            i += 1
            continue
        if ch in " \t":
            out.space(i)
            i += 1
            continue

        out.put(ch, i)
        i += 1

    for name, kind, start in pending:
        sections.append(Section(name, kind, start, out.pos))

    return Normalized(
        text="".join(out.chars),
        offsets=out.offsets,
        raw=raw,
        sections=sections,
        regions=regions,
        files=files,
    )
