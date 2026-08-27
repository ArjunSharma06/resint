"""JATS XML: the format published papers actually arrive in.

resint has read LaTeX since the beginning, which means it reads *preprints*.
The literature that reports statistics the way ``stats/grim`` and
``stats/pvalue-mismatch`` expect -- psychology, clinical medicine,
epidemiology -- publishes through journals, and PubMed Central serves several
million of those as JATS XML under open-access terms.

The gap was measured rather than assumed: across 204 real arXiv papers the
statistics extractor found something in **2**. Three rules had never once run
on real input, because inline NHST (``t(20) = 2.086, p = .03``) is a
disciplinary convention and arXiv's fields do not use it.

The whole design here is one decision: **produce the same
:class:`~resint.parse.latex.Normalized` the LaTeX path produces.** Plain text,
plus an offset map back into the source. Everything downstream -- number
extraction, GRIM, the anchor audit, the two-anchor invariant -- then works
without knowing the input was XML. No rule changes.

Offsets point into the raw XML, because that is the file the reader has open.
A finding at line 412 means line 412 of the ``.nxml``.

Written on a scanner rather than ElementTree because ElementTree discards
source positions, and a span that cannot say *where* is not evidence. The
scanner also survives the malformed markup that a strict parser rejects
outright, which matters when the input is several million files nobody
validated.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .latex import Normalized, Section

#: Content we take prose from. A whitelist, because ``<front>`` is mostly
#: metadata -- affiliations, funding, ISSNs -- and letting it through would
#: bury the abstract in publisher boilerplate.
PROSE_ELEMENTS = ("article-title", "abstract", "body")

#: Dropped wherever they appear. Tables and references are parsed separately
#: into their own IR; the rest is apparatus that is not the author's argument.
SKIP_ELEMENTS = frozenset(
    {
        "table-wrap",
        "ref-list",
        "fn-group",
        "front-stub",
        "graphic",
        "inline-graphic",
        "media",
        "supplementary-material",
        "disp-formula",  # Rendered maths; the LaTeX path drops it too.
        "inline-formula",
        "tex-math",
        "mml:math",
        "math",
    }
)

_TAG = re.compile(r"<(?P<closing>/?)(?P<name>[A-Za-z_][\w.:-]*)(?P<attrs>[^>]*?)(?P<empty>/?)>")
_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_CDATA = re.compile(r"<!\[CDATA\[(.*?)\]\]>", re.DOTALL)
_DOCTYPE = re.compile(r"<!DOCTYPE[^>[]*(\[[^\]]*\])?[^>]*>", re.DOTALL)
_PI = re.compile(r"<\?.*?\?>", re.DOTALL)

#: The handful of entities that appear in real articles. A full table is not
#: needed: numeric references cover everything else and are decoded directly.
_ENTITIES = {
    "amp": "&",
    "lt": "<",
    "gt": ">",
    "quot": '"',
    "apos": "'",
    "nbsp": " ",
    "mdash": "—",
    "ndash": "–",
    "lsquo": "‘",
    "rsquo": "’",
    "ldquo": "“",
    "rdquo": "”",
    "hellip": "…",
    "deg": "°",
    "plusmn": "±",
    "times": "×",
    "alpha": "α",
    "beta": "β",
    "chi": "χ",
    "sigma": "σ",
    "mu": "μ",
}

_ENTITY = re.compile(r"&(#x?[0-9A-Fa-f]+|[A-Za-z][A-Za-z0-9]*);")

#: Elements that end a paragraph. Blank lines are load-bearing downstream:
#: ``resolve.passages`` splits on them and ``claim/unsupported`` locates the
#: end of the front matter by section, so a document flattened into one block
#: silently degrades both.
BLOCK_ELEMENTS = frozenset(
    {"p", "sec", "title", "abstract", "list-item", "disp-quote", "article-title"}
)


def _decode_entity(match: re.Match) -> str:
    body = match.group(1)
    if body.startswith("#"):
        try:
            decoded = chr(int(body[2:], 16) if body[1] in "xX" else int(body[1:]))
        except (ValueError, OverflowError):
            return " "
    else:
        decoded = _ENTITIES.get(body, " ")

    # Any decoded space becomes an ordinary one. Otherwise &nbsp; and &#160;
    # -- the same character written two ways -- would behave differently, the
    # named form collapsing and the numeric form surviving as U+00A0 to be
    # matched against a literal space by every extractor downstream.
    return " " if decoded.isspace() else decoded


@dataclass
class _Builder:
    """Accumulates text and the raw offset each character came from."""

    chars: list
    offsets: list

    def add(self, text: str, raw_index: int) -> None:
        for ch in text:
            self.chars.append(ch)
            self.offsets.append(raw_index)

    def newline(self, raw_index: int, count: int = 2) -> None:
        """End a block, without stacking blank lines."""
        existing = 0
        for ch in reversed(self.chars):
            if ch == "\n":
                existing += 1
            elif ch in " \t":
                continue
            else:
                break
        for _ in range(max(0, count - existing)):
            self.chars.append("\n")
            self.offsets.append(raw_index)

    @property
    def length(self) -> int:
        return len(self.chars)


def _masked(raw: str) -> str:
    """Blank comments, processing instructions and the doctype, keeping length.

    Length preservation is what lets every offset below index straight into the
    original string, so a span is a position in the file the reader opens.
    """
    out = raw
    for pattern in (_COMMENT, _PI, _DOCTYPE):
        out = pattern.sub(lambda m: " " * len(m.group(0)), out)
    return out


#: Elements that occur once as the paper's own and again inside every
#: reference. ``<ref><element-citation><article-title>`` is the title of a
#: *cited* work, and taking it emitted other people's titles as though they
#: were this paper's prose. The article's own always comes first, in
#: ``<front><article-meta><title-group>``.
ONCE_ONLY = frozenset({"article-title"})


def find_regions(raw: str, names=PROSE_ELEMENTS) -> list[tuple[int, int, str]]:
    """Raw ranges of the elements prose is taken from, outermost only."""
    masked = _masked(raw)
    found: list[tuple[int, int, str]] = []

    for name in names:
        depth = 0
        start = -1
        for m in _TAG.finditer(masked):
            if m.group("name") != name:
                continue
            if m.group("empty"):
                continue
            if not m.group("closing"):
                if depth == 0:
                    start = m.end()
                depth += 1
            else:
                depth -= 1
                if depth == 0 and start >= 0:
                    found.append((start, m.start(), name))
                    start = -1
                    if name in ONCE_ONLY:
                        break
                depth = max(depth, 0)

    found.sort()
    return found


def _text_of(raw: str, masked: str, lo: int, hi: int, builder: _Builder, sections: list) -> None:
    """Emit the prose inside one raw range, skipping apparatus."""
    cursor = lo
    skip_depth = 0
    skip_name = ""
    open_sections: list[tuple[str, int, int]] = []
    pending_title: int | None = None

    for m in _TAG.finditer(masked, lo, hi):
        if skip_depth == 0 and m.start() > cursor:
            _emit_text(raw, cursor, m.start(), builder)

        name = m.group("name")
        closing = bool(m.group("closing"))
        empty = bool(m.group("empty"))

        if skip_depth:
            if name == skip_name and not empty:
                skip_depth += -1 if closing else 1
            cursor = m.end()
            continue

        if name in SKIP_ELEMENTS and not closing and not empty:
            skip_depth = 1
            skip_name = name
            cursor = m.end()
            continue

        if name in BLOCK_ELEMENTS and not empty:
            builder.newline(m.start())

        if name == "sec" and not empty:
            if not closing:
                open_sections.append(("", builder.length, m.start()))
            elif open_sections:
                title, begin, _ = open_sections.pop()
                sections.append(
                    Section(name=title or "section", kind="section",
                            start=begin, end=builder.length)
                )

        if name == "title" and not empty and open_sections:
            if not closing:
                pending_title = builder.length
            elif pending_title is not None:
                title = "".join(builder.chars[pending_title:]).strip()
                held = open_sections[-1]
                open_sections[-1] = (title or held[0], held[1], held[2])
                pending_title = None

        cursor = m.end()

    if skip_depth == 0 and cursor < hi:
        _emit_text(raw, cursor, hi, builder)

    # A section left open by truncated markup still describes real text.
    while open_sections:
        title, begin, _ = open_sections.pop()
        sections.append(
            Section(name=title or "section", kind="section",
                    start=begin, end=builder.length)
        )


def _emit_text(raw: str, lo: int, hi: int, builder: _Builder) -> None:
    """Character data, with entities decoded and whitespace collapsed."""
    cursor = lo
    for m in _ENTITY.finditer(raw, lo, hi):
        if m.start() > cursor:
            _emit_plain(raw, cursor, m.start(), builder)
        builder.add(_decode_entity(m), m.start())
        cursor = m.end()
    if cursor < hi:
        _emit_plain(raw, cursor, hi, builder)


def _emit_plain(raw: str, lo: int, hi: int, builder: _Builder) -> None:
    """Runs of whitespace become one space, so prose reads as prose.

    Each emitted character keeps the offset of the character it came from, so
    collapsing never costs the ability to point at the source.
    """
    previous_space = bool(builder.chars) and builder.chars[-1] in " \n"
    for i in range(lo, hi):
        ch = raw[i]
        if ch.isspace():
            if not previous_space:
                builder.add(" ", i)
                previous_space = True
            continue
        builder.add(ch, i)
        previous_space = False


def normalize(raw: str) -> Normalized:
    """Plain text from JATS XML, with an offset map back into the XML."""
    masked = _masked(raw)
    builder = _Builder(chars=[], offsets=[])
    sections: list[Section] = []

    regions = find_regions(raw)
    if not regions:
        # Not recognisably JATS, or an article with no body. Falling back to
        # the whole document would emit publisher metadata as though it were
        # the paper.
        return Normalized(text="", offsets=[], raw=raw, sections=[])

    for lo, hi, _name in regions:
        _text_of(raw, masked, lo, hi, builder, sections)
        builder.newline(hi)

    sections.sort(key=lambda s: s.start)
    return Normalized(
        text="".join(builder.chars),
        offsets=builder.offsets,
        raw=raw,
        sections=sections,
    )


def looks_like_jats(raw: str) -> bool:
    """Whether this is a JATS article rather than some other XML.

    Checked on content, never on the extension. PMC serves ``.nxml``, ``.xml``
    and bare downloads, and a paper is not the only XML a user may point at.
    """
    head = raw[:4000].lower()
    if "<article" not in head:
        return False
    return any(
        marker in head
        for marker in ("jats", "article-meta", "journal-meta", "front", "article-title")
    )
