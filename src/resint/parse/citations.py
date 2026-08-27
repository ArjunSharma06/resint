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


_BIBLIOGRAPHY = re.compile(
    r"\\begin\{thebibliography\}.*?(?:\\end\{thebibliography\}|\Z)", re.DOTALL
)


def _without_bibliography(text: str) -> str:
    """Blank out the reference list, preserving length.

    A citation inside the bibliography is not a use site, and natbib writes
    real ``\\cite``-shaped commands in there. Its rendered labels look like:

        \\bibitem[\\protect\\citeauthoryear{Duan, Hong, Meeker, and Gu}{Duan
          et~al.}{2017}]{DuanEtAl2017}

    ``\\citeauthoryear`` matches the citation pattern, but its argument holds
    *author names*, not keys -- and splitting on commas turned one reference
    into phantom citations of "Duan", "Hong", "Meeker" and "and Gu", each
    reported as cited-but-undefined. Across 204 papers this was the largest
    single source of false positives in the tool.
    """
    return _BIBLIOGRAPHY.sub(lambda m: " " * len(m.group(0)), text)


#: A BibTeX key cannot contain whitespace, a comma or a brace. Anything that
#: does came from parsing rendered text or a macro template.
_KEY = re.compile(r"[^\s{}\\,#]+\Z")


#: Two families, because they are shaped differently after the name.
#:
#: ``\newcommand{\foo}[2][d]{body}``      -- optional arity in brackets
#: ``\NewDocumentCommand\foo{mm}{body}``  -- argument spec in braces
#:
#: The second matters more than its rarity suggests: pandoc writes
#: ``\NewDocumentCommand\citeproc{mm}{...\cite{#1}...}``, and ``{mm}`` is the
#: spec meaning "two mandatory arguments". Read as a citation it became a key
#: named "mm", reported as cited-but-undefined on every pandoc-produced paper
#: in the corpus. Treating it like the ``\newcommand`` family would be worse
#: than ignoring it -- the spec would be blanked as though it were the body,
#: leaving the real body, and its ``\cite``, still exposed.
_MACRO_DEF = re.compile(
    r"\\(?P<spec>(?:New|Renew|Provide|Declare)DocumentCommand\*?)"
    r"|\\(?P<plain>(?:new|renew|provide)command\*?|DeclareRobustCommand\*?"
    r"|def(?=[\s\\{]))"
)


def _skip_group(text: str, i: int, opener: str, closer: str) -> int:
    """Index just past a balanced group starting at ``i``, or ``i`` unchanged."""
    if i >= len(text) or text[i] != opener:
        return i
    depth = 0
    while i < len(text):
        if text[i] == "\\":
            i += 2  # An escaped brace is not a brace.
            continue
        if text[i] == opener:
            depth += 1
        elif text[i] == closer:
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return i


def _without_macro_definitions(text: str) -> str:
    """Blank out macro bodies, preserving length.

    A definition is not a use site. Pandoc emits

        \\newcommand{\\citeproc}[2]{\\hyper@linkstart{cite}{ref-#1}{#2}...}

    and the ``\\cite``-shaped commands inside it are a template -- there is no
    key there to resolve. Reading them as citations reported keys like ``#1``
    and ``mm`` as cited-but-undefined on every pandoc-produced paper.

    Brace matching rather than a regex, because a macro body nests: the naive
    pattern stops at the first ``}`` and leaves the rest of the body exposed.
    """
    out = list(text)
    for match in _MACRO_DEF.finditer(text):
        i = match.end()
        while i < len(text) and text[i] in " \t":
            i += 1

        # The name, given either as {\foo} or bare as \foo.
        if i < len(text) and text[i] == "{":
            i = _skip_group(text, i, "{", "}")
        elif i < len(text) and text[i] == "\\":
            i += 1
            while i < len(text) and (text[i].isalpha() or text[i] == "@"):
                i += 1

        # Arity and default-argument brackets.
        while True:
            j = i
            while j < len(text) and text[j] in " \t":
                j += 1
            if j < len(text) and text[j] == "[":
                i = _skip_group(text, j, "[", "]")
                continue
            break

        # xparse states its argument specification in braces, so the group
        # immediately after the name is the spec and the body is the one after.
        if match.group("spec"):
            j = i
            while j < len(text) and text[j] in " \t\r\n":
                j += 1
            i = _skip_group(text, j, "{", "}")

        while i < len(text) and text[i] in " \t\r\n":
            i += 1

        end = _skip_group(text, i, "{", "}")

        # From the start of the definition, not from the body. The name and
        # argument specification are themselves cite-shaped: pandoc's
        # ``\NewDocumentCommand\citeproc{mm}{...}`` matches the citation
        # pattern at ``\citeproc{mm}`` before the body is even reached, so
        # blanking only the body leaves the phantom key in place.
        for k in range(match.start(), end):
            if out[k] != "\n":  # Keep line numbers truthful.
                out[k] = " "
    return "".join(out)


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
    scannable = _without_macro_definitions(_without_bibliography(_uncommented(raw)))
    out: list[Citation] = []

    for m in _CITE.finditer(scannable):
        block = m.group("keys")
        block_start = m.start("keys")
        cursor = 0

        for piece in block.split(","):
            key = piece.strip()
            if key and _KEY.fullmatch(key):
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
