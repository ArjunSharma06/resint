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

from dataclasses import dataclass

from pathlib import Path

from ..ir.paper import Paper, TextSlice
from ..ir.span import Source
from ..resolve.base import NullResolver, Resolver, Status, resolve_all
from ..resolve.http import ResolvePolicy
from .acquire import UnreadableInput, acquire
from .bbl import looks_like_bbl, parse as parse_bbl
from .bibtex import parse as parse_bibtex
from .citations import extract_citations
from .claims import extract_claims
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
    # Always populated, because normalization produces it either way. Declared
    # like anything else so a rule reading section structure says so.
    "paper.sections",
    "paper.stats",
    "paper.means",
    "paper.tables",
    "paper.numbers",
    "paper.hyperparameters",
    "paper.citations",
    "paper.bib",
    "paper.resolutions",
    "paper.claims",
    "paper.cited_texts",
}

#: How many cited papers one run will download. A survey cites three hundred
#: works; fetching every one of them to check a dozen claims is not a trade
#: anybody would make knowingly.
MAX_CITED_FETCHES = 60


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


def paper_from_source_if_jats(
    source,
    needs: set[str] | None = None,
    resolver: Resolver | None = None,
    progress=None,
    policy: ResolvePolicy | None = None,
    full_text=None,
) -> Paper | None:
    """A Paper if this acquired source is JATS, else None.

    Shared by ``paper_from_path`` and the sweep runner for the same reason
    ``resolve_bibliography`` is: the two must not disagree about what the input
    is. They did. The sweep called ``paper_from_latex`` directly, so six real
    PubMed Central articles were parsed as LaTeX -- prose survived, because
    stripping XML tags leaves the words behind, and every structural extractor
    silently found nothing. Coverage read ``bib 0/6, citations 0/6, tables
    0/6`` and looked like six unusual papers rather than one wrong branch.

    Content decides, never the extension: the same rule the PDF sniffing
    follows. PMC serves .nxml, .xml and bare downloads, and an .xml file is
    not necessarily a paper at all.
    """
    from .jats import looks_like_jats

    if not looks_like_jats(source.text):
        return None

    return paper_from_jats(
        source.text,
        source_id=source.name,
        path=source.path,
        needs=needs,
        resolver=resolver,
        progress=progress,
        policy=policy,
        full_text=full_text,
    )


def _fetch_cited(paper: Paper, source, progress=None) -> dict:
    """Download the cited papers that some claim actually rests on.

    Only those. A bibliography is mostly background reading, and a reference
    nobody attached an assertion to has nothing to check, so fetching it would
    be paid for and then discarded. In practice this turns a hundred-entry
    bibliography into a dozen or so downloads.
    """
    from ..resolve.fulltext import NullFullText

    active = source or NullFullText()
    by_key = {entry.key: entry for entry in paper.bib}

    wanted: list[str] = []
    for claim in paper.claims:
        for key in claim.keys:
            if key in by_key and key not in wanted:
                wanted.append(key)

    if len(wanted) > MAX_CITED_FETCHES:
        paper.unchecked.append(
            f"{len(wanted) - MAX_CITED_FETCHES} cited papers not fetched: "
            f"the per-run limit of {MAX_CITED_FETCHES} was reached"
        )
        wanted = wanted[:MAX_CITED_FETCHES]

    fetched: dict = {}
    for number, key in enumerate(wanted, 1):
        if progress is not None:
            progress(f"fetching cited paper {number}/{len(wanted)}")
        resolution = paper.resolutions.get(key)
        record = resolution.record if resolution is not None else None
        fetched[key] = active.fetch(by_key[key], record)
    return fetched


def _body_start(raw: str, offsets) -> int:
    """The normalized index where the document body begins.

    Normalization strips command names but keeps their arguments, so a LaTeX
    preamble survives into the text as a run of noise -- theorem-environment
    declarations, editorial-note macros, author blocks. A rule scanning for
    numbers is unbothered by it. A prompt is not: it lands in the position a
    model attends to most, and under truncation it displaces real content.

    Returns 0 when there is no preamble to skip, so text that was never a
    full document is unaffected.
    """
    marker = raw.find(r"\begin{document}")
    if marker < 0:
        return 0
    for index, offset in enumerate(offsets):
        if offset >= marker:
            return index
    return 0


def _latex_bibliography(text, src, doc, bib_text, bib_id, bib_kind, source_id):
    """Where a LaTeX paper's references live: a .bib, a .bbl, or the document."""
    if bib_text is None and looks_like_bbl(text):
        # A submission that inlines its compiled bibliography carries no
        # separate file at all -- the thebibliography environment sits in the
        # source. Very common on arXiv, where inlining lets the paper compile
        # without running BibTeX.
        bib_text, bib_id, bib_kind = text, source_id, "bbl"

    if bib_text is None:
        return None

    bib_src = Source(bib_id, "bib", path=bib_id)
    if bib_kind == "bbl" or looks_like_bbl(bib_text):
        # Only when the bibliography is the document itself do its offsets
        # need region resolution. A separate .bib or .bbl file is never
        # spliced, so its offsets are already local to it -- passing doc there
        # would resolve them against the wrong text.
        return parse_bbl(bib_text, bib_src, doc if bib_text is text else None)
    return parse_bibtex(bib_text, bib_src)


def _jats_bibliography(text, src, doc, bib_text, bib_id, bib_kind, source_id):
    """A journal article carries its references inside itself, always."""
    from .jats_parts import extract_bib as jats_bib

    return jats_bib(text, src, doc)


@dataclass(frozen=True)
class Dialect:
    """The three things that differ between input formats.

    Everything after these -- numbers, statistics, means, claims, resolution,
    the anchor audit -- is shared, which is the whole point: a rule reads a
    journal article and a preprint through one IR and cannot tell which it was
    given. Parameterising the differences rather than forking the function is
    what keeps that true as rules are added.
    """

    kind: str
    normalize: object
    tables: object
    citations: object
    bibliography: object


LATEX = Dialect(
    kind="latex",
    normalize=lambda text, regions, files: normalize(text, regions=regions, files=files),
    tables=extract_tables,
    citations=extract_citations,
    bibliography=_latex_bibliography,
)


def _jats_dialect() -> Dialect:
    from . import jats as jats_module
    from . import jats_parts

    return Dialect(
        kind="jats",
        normalize=lambda text, regions, files: jats_module.normalize(text),
        tables=jats_parts.extract_tables,
        citations=jats_parts.extract_citations,
        bibliography=_jats_bibliography,
    )


def paper_from_jats(
    text: str,
    source_id: str = "article.nxml",
    path: str | None = None,
    needs: set[str] | None = None,
    resolver: Resolver | None = None,
    progress=None,
    policy: ResolvePolicy | None = None,
    full_text=None,
) -> Paper:
    """Build a Paper from a JATS XML article, as PubMed Central serves them."""
    return _build(
        text,
        _jats_dialect(),
        source_id=source_id,
        path=path,
        needs=needs,
        resolver=resolver,
        progress=progress,
        policy=policy,
        full_text=full_text,
    )


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
    full_text=None,
) -> Paper:
    """Build a Paper, filling only the slices in ``needs``."""
    return _build(
        text,
        LATEX,
        source_id=source_id,
        path=path,
        needs=needs,
        bib_text=bib_text,
        bib_id=bib_id,
        bib_kind=bib_kind,
        resolver=resolver,
        progress=progress,
        policy=policy,
        regions=regions,
        files=files,
        full_text=full_text,
    )


def _build(
    text: str,
    dialect: Dialect,
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
    full_text=None,
) -> Paper:
    """Build a Paper, filling only the slices in ``needs``."""
    src = Source(source_id, dialect.kind, path=path or source_id)
    doc = dialect.normalize(text, regions, files)
    paper = Paper(source_id=source_id)
    paper.sections = list(doc.sections)

    wanted = needs if needs is not None else set(ALL_SLICES)

    if "paper.text" in wanted:
        paper.text = TextSlice(
            content=doc.text,
            _offsets=tuple(doc.offsets),
            _source=src,
            _line_starts=tuple(doc._line_starts),
            body_start=_body_start(text, doc.offsets),
        )

    # paper.numbers is matched against column headings, so the tables have to
    # be read first even when only the numbers were asked for.
    if {"paper.tables", "paper.numbers"} & wanted:
        paper.tables = dialect.tables(text, src, doc)
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

    # Claims are built out of citations, so the citation pass has to run even
    # when only the claims were asked for.
    if {"paper.citations", "paper.claims"} & wanted:
        paper.citations = dialect.citations(text, src, doc)

    if "paper.claims" in wanted:
        paper.claims = extract_claims(doc, src, paper.citations, doc.sections)

    needs_bib = {"paper.bib", "paper.resolutions", "paper.cited_texts"} & wanted
    if needs_bib:
        parsed = dialect.bibliography(
            text, src, doc, bib_text, bib_id, bib_kind, source_id
        )
        if parsed is None:
            if "paper.citations" in wanted and paper.citations:
                paper.unchecked.append(
                    "bibliography not checked: no .bib file supplied"
                )
        else:
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

    if "paper.cited_texts" in wanted and paper.bib:
        paper.cited_texts = _fetch_cited(paper, full_text, progress)

    return paper


def paper_from_path(
    path: str | Path,
    needs: set[str] | None = None,
    bib: str | Path | None = None,
    resolver: Resolver | None = None,
    progress=None,
    policy: ResolvePolicy | None = None,
    full_text=None,
) -> Paper:
    """Build a Paper from a file, an arXiv source bundle, or a zip.

    Bibliography discovery differs by input. A loose .tex takes the single
    .bib sitting beside it; an archive uses whatever it carries, since a
    stray .bib in the download folder has nothing to do with the paper.
    """
    source = acquire(path, bib=bib)

    jats = paper_from_source_if_jats(
        source,
        needs=needs,
        resolver=resolver,
        progress=progress,
        policy=policy,
        full_text=full_text,
    )
    if jats is not None:
        return jats

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
        full_text=full_text,
    )
    paper.unchecked[:0] = list(source.notes)
    return paper
