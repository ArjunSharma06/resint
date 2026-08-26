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
from ..resolve.http import ResolvePolicy
from .acquire import UnreadableInput, acquire
from .bbl import looks_like_bbl, parse as parse_bbl
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
    """The bibliography sitting alongside the source, if there is one.

    A .bib is preferred; a compiled .bbl is the fallback, since a submission
    often ships only that.
    """
    for pattern in ("*.bib", "*.bbl"):
        candidates = sorted(tex_path.parent.glob(pattern))
        if len(candidates) == 1:
            return candidates[0]
    return None


def resolve_bibliography(source, path, bib=None) -> tuple[str | None, str]:
    """Which bibliography text belongs to this input, and what to call it.

    Shared by ``paper_from_path`` and the sweep runner so the two cannot
    disagree about it. The beside-lookup applies to a loose .tex only: an
    archive carries its own bibliography or has none, and globbing its parent
    means that in a sweep -- where hundreds of bundles share one cache folder
    -- a single stray .bib would be adopted by every paper in the run.
    """
    if source.bib_text is not None or bib is not None:
        return source.bib_text, source.bib_name
    if source.from_archive:
        return None, source.bib_name

    beside = find_bibliography(Path(path))
    if beside is not None and beside.is_file():
        return beside.read_text(encoding="utf-8"), beside.name
    return None, source.bib_name


def paper_from_latex(
    text: str,
    source_id: str = "paper.tex",
    path: str | None = None,
    needs: set[str] | None = None,
    bib_text: str | None = None,
    bib_id: str = "refs.bib",
    bib_kind: str = "bib",
    resolver: Resolver | None = None,
    progress=None,
    policy: ResolvePolicy | None = None,
    regions: tuple = (),
    files: dict | None = None,
) -> Paper:
    """Build a Paper, filling only the slices in ``needs``."""
    src = Source(source_id, "latex", path=path or source_id)
    doc = normalize(text, regions=regions, files=files)
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
        paper.tables = extract_tables(text, src, doc)
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
        # A submission that inlines its compiled bibliography carries no
        # separate file at all -- the thebibliography environment sits in the
        # source. Very common on arXiv, where inlining lets the paper compile
        # without running BibTeX.
        if bib_text is None and looks_like_bbl(text):
            bib_text, bib_id, bib_kind = text, source_id, "bbl"

        if bib_text is None:
            if "paper.citations" in wanted and paper.citations:
                paper.unchecked.append(
                    "bibliography not checked: no .bib file supplied"
                )
        else:
            bib_src = Source(bib_id, "bib", path=bib_id)
            compiled = bib_kind == "bbl" or looks_like_bbl(bib_text)
            parsed = (parse_bbl if compiled else parse_bibtex)(bib_text, bib_src)
            paper.bib = parsed.entries
            for bad in parsed.malformed:
                paper.unchecked.append(f"bibliography entry skipped: {bad}")
            paper.unchecked.extend(parsed.notes)

    if "paper.resolutions" in wanted and paper.bib:
        active = resolver or NullResolver()
        rules_of_engagement = policy or ResolvePolicy()
        paper.resolutions = resolve_all(
            active,
            paper.bib,
            workers=rules_of_engagement.workers,
            budget=rules_of_engagement.budget,
            progress=progress,
        )
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
    policy: ResolvePolicy | None = None,
) -> Paper:
    """Build a Paper from a file, an arXiv source bundle, or a zip.

    Bibliography discovery differs by input. A loose .tex takes the single
    .bib sitting beside it; an archive uses whatever it carries, since a
    stray .bib in the download folder has nothing to do with the paper.
    """
    source = acquire(path, bib=bib)

    bib_text, bib_id = resolve_bibliography(source, path, bib)

    paper = paper_from_latex(
        source.text,
        source_id=source.name,
        path=source.path,
        needs=needs,
        bib_text=bib_text,
        bib_id=bib_id,
        bib_kind=source.bib_kind,
        resolver=resolver,
        progress=progress,
        policy=policy,
        regions=source.regions,
        files=source.files,
    )
    paper.unchecked[:0] = list(source.notes)
    return paper
