"""Assembling a Paper from a source file.

Population is driven by what the running rules declared, so a run that only
needs ``paper.stats`` never pays for mean extraction -- and, more to the
point, never opens a socket. Reference resolution is the only slice that
touches the network, and it happens exactly when some rule asked for
``paper.resolutions`` and not otherwise.

Whatever the extractors had to skip lands in ``paper.unchecked`` and is
reported, because a checker that implies completeness is less trustworthy
than one that names its blind spots.
"""

from __future__ import annotations

from pathlib import Path

from ..ir.paper import Paper, TextSlice
from ..ir.span import Source
from ..resolve.base import NullResolver, Resolver, Status, resolve_all
from .bibtex import parse as parse_bibtex
from .citations import extract_citations
from .extract import (
    extract_hyperparameters,
    extract_labeled_numbers,
    extract_means,
    extract_stats,
    usable_labels,
)
from .latex import normalize
from .tables import extract_tables

ALL_SLICES = {
    "paper.text",
    "paper.stats",
    "paper.means",
    "paper.tables",
    "paper.numbers",
    "paper.hyperparameters",
    "paper.citations",
    "paper.bib",
    "paper.resolutions",
}


def find_bibliography(tex_path: Path) -> Path | None:
    """The .bib file sitting alongside the source, if there is exactly one."""
    candidates = sorted(tex_path.parent.glob("*.bib"))
    return candidates[0] if len(candidates) == 1 else None


def paper_from_latex(
    text: str,
    source_id: str = "paper.tex",
    path: str | None = None,
    needs: set[str] | None = None,
    bib_text: str | None = None,
    bib_id: str = "refs.bib",
    resolver: Resolver | None = None,
    progress=None,
) -> Paper:
    """Build a Paper, filling only the slices in ``needs``."""
    src = Source(source_id, "latex", path=path or source_id)
    doc = normalize(text)
    paper = Paper(source_id=source_id)
    paper.sections = list(doc.sections)

    wanted = needs if needs is not None else set(ALL_SLICES)

    if "paper.text" in wanted:
        paper.text = TextSlice(
            content=doc.text,
            _offsets=tuple(doc.offsets),
            _source=src,
            _line_starts=tuple(doc._line_starts),
        )

    # paper.numbers is matched against column headings, so the tables have to
    # be read first even when only the numbers were asked for.
    if {"paper.tables", "paper.numbers"} & wanted:
        paper.tables = extract_tables(text, src)
        for table in paper.tables:
            if table.irregular:
                paper.unchecked.append(
                    f"{table.name} not checked: {table.irregular}"
                )

    if "paper.numbers" in wanted:
        headings = [
            h
            for table in paper.tables
            if not table.irregular
            for h in table.header[1:]
        ]
        paper.numbers = extract_labeled_numbers(doc, src, usable_labels(headings))

    if "paper.hyperparameters" in wanted:
        paper.hyperparameters = extract_hyperparameters(doc, src)

    if "paper.stats" in wanted:
        paper.stats = extract_stats(doc, src)

    if "paper.means" in wanted:
        result = extract_means(doc, src)
        paper.means = result.means
        paper.unchecked.extend(result.unchecked)

    if "paper.citations" in wanted:
        paper.citations = extract_citations(text, src)

    needs_bib = {"paper.bib", "paper.resolutions"} & wanted
    if needs_bib:
        if bib_text is None:
            if "paper.citations" in wanted and paper.citations:
                paper.unchecked.append(
                    "bibliography not checked: no .bib file supplied"
                )
        else:
            bib_src = Source(bib_id, "bib", path=bib_id)
            parsed = parse_bibtex(bib_text, bib_src)
            paper.bib = parsed.entries
            for bad in parsed.malformed:
                paper.unchecked.append(f"bibliography entry skipped: {bad}")

    if "paper.resolutions" in wanted and paper.bib:
        active = resolver or NullResolver()
        paper.resolutions = resolve_all(active, paper.bib, progress=progress)
        unknown = sum(
            1
            for r in paper.resolutions.values()
            if r.status is Status.UNKNOWN
        )
        if unknown:
            noun = "reference" if unknown == 1 else "references"
            paper.unchecked.append(
                f"{unknown} {noun} could not be looked up "
                "(offline or no resolver configured); not reported as missing"
            )

    return paper


def paper_from_path(
    path: str | Path,
    needs: set[str] | None = None,
    bib: str | Path | None = None,
    resolver: Resolver | None = None,
    progress=None,
) -> Paper:
    p = Path(path)
    bib_path = Path(bib) if bib else find_bibliography(p)
    bib_text = (
        bib_path.read_text(encoding="utf-8")
        if bib_path and bib_path.exists()
        else None
    )
    return paper_from_latex(
        p.read_text(encoding="utf-8"),
        source_id=p.name,
        path=str(p),
        needs=needs,
        bib_text=bib_text,
        bib_id=bib_path.name if bib_path else "refs.bib",
        resolver=resolver,
        progress=progress,
    )
