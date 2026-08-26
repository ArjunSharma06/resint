"""Reading a compiled bibliography (``.bbl``).

arXiv submissions frequently ship the *output* of BibTeX rather than its
input, because that is what the compiler needs and what the submission
guidelines ask for. Refusing to read it means the three ``bib/`` rules sit out
on a large share of real papers -- which is what happened the first time
resint was pointed at an arXiv bundle.

A ``.bbl`` is prose, not data. BibTeX has already flattened the fields into a
rendered citation, so authors, title and venue are recovered by convention
rather than read off. That convention is the ``\\newblock`` separator, which
every standard style emits, but the *meaning* of each block varies between
styles and nothing guarantees the split is right.

So entries from here are marked ``from_bbl``, and the rules treat them with
more caution than a parsed ``.bib``: what is certain is the citation key,
which is enough for ``bib/orphans`` to work exactly as well as it ever does.
"""

from __future__ import annotations

import re

from ..ir.paper import BibEntry
from ..ir.span import Source, Span
from .bibtex import BibFile, clean_value

_BIBITEM = re.compile(r"\\bibitem\s*(?:\[[^\]]*\])?\s*\{([^}]*)\}")
_NEWBLOCK = re.compile(r"\\newblock\s*")
_YEAR = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")
_DOI = re.compile(r"\b10\.\d{4,9}/[^\s,;}\)]+", re.IGNORECASE)
_ARXIV = re.compile(r"arXiv[:\s]*(\d{4}\.\d{4,5})", re.IGNORECASE)
_URL = re.compile(r"\\url\s*\{([^}]*)\}")

# A "title" shorter than this is punctuation or an artefact of an unusual
# style, never a real title worth searching an index for.
_MIN_TITLE = 12


_TRAILING_YEAR = re.compile(r"[,;]?\s*\(?\b(1[89]\d{2}|20\d{2})\b\)?\s*[.,]?\s*$")


def _strip_trailing(text: str) -> str:
    """Trim punctuation, and a year that the style appended to the title.

    Some styles close the title block with the year rather than opening the
    venue block with it, leaving titles like "...long-term dependencies,
    2001". Searching an index on that finds nothing.
    """
    text = text.strip().strip(".,;: ").strip()
    text = _TRAILING_YEAR.sub("", text).strip()
    return text.strip(".,;: ").strip()


def parse(text: str, src: Source) -> BibFile:
    """Parse a compiled bibliography into entries.

    Only the key is read directly; everything else is inferred from the
    rendered citation and may be wrong. Entries carry ``from_bbl`` so rules
    can weigh that.
    """
    out = BibFile()
    matches = list(_BIBITEM.finditer(text))
    if not matches:
        return out

    for index, match in enumerate(matches):
        key = match.group(1).strip()
        if not key:
            continue

        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        stop = text.find(r"\end{thebibliography}", start, end)
        if stop != -1:
            end = stop
        body = text[start:end]

        chunks = [c for c in (b.strip() for b in _NEWBLOCK.split(body)) if c]

        fields: dict[str, str] = {}
        field_spans: dict[str, Span] = {}

        # Without \newblock there is nothing to split on, and guessing which
        # part of a run-together citation is the title would produce index
        # searches against fragments. Better to know only the key.
        if len(chunks) >= 2:
            # Position, not content. Every standard BibTeX style emits the
            # authors before the first \newblock and the title after it.
            # Guessing from content instead was actively worse: a single
            # author with no initials ("Francois Chollet") failed a
            # looks-like-a-name-list test and got promoted to title.
            authors = _strip_trailing(clean_value(chunks[0]))
            title = _strip_trailing(clean_value(chunks[1]))
            if authors:
                fields["author"] = authors
            if len(title) >= _MIN_TITLE:
                fields["title"] = title
                offset = start + body.find(chunks[1])
                field_spans["title"] = Span(
                    src,
                    max(offset, start),
                    max(offset, start) + len(title),
                    line=text.count("\n", 0, max(offset, start)) + 1,
                    label=f"[{key}].title",
                )
            if len(chunks) > 2:
                fields["journal"] = _strip_trailing(clean_value(chunks[2]))[:200]

        flat = clean_value(body)
        years = _YEAR.findall(flat)
        if years:
            # The last year in a rendered citation is the publication year;
            # earlier ones belong to volume numbers and page ranges.
            fields["year"] = years[-1]

        doi = _DOI.search(body)
        if doi:
            fields["doi"] = doi.group(0).rstrip(".")
        else:
            arxiv = _ARXIV.search(body) or _URL.search(body)
            if arxiv and "arxiv" in (arxiv.group(0) or "").lower():
                fields["eprint"] = arxiv.group(1) if arxiv.re is _ARXIV else arxiv.group(0)

        out.entries.append(
            BibEntry(
                key=key,
                entry_type="bibitem",
                fields=fields,
                span=Span(
                    src,
                    match.start(),
                    end,
                    line=text.count("\n", 0, match.start()) + 1,
                    label=f"[{key}]",
                ),
                field_spans=field_spans,
            )
        )

    without_title = [e.key for e in out.entries if not e.title]
    if without_title:
        count = (
            "1 entry has" if len(without_title) == 1
            else f"{len(without_title)} entries have"
        )
        out.notes.append(
            f"{count} no recoverable title in the compiled bibliography; "
            "they can be checked for citation but not looked up"
        )

    return out


def looks_like_bbl(text: str) -> bool:
    return r"\bibitem" in text
