"""Checking one paper, defensively, and reporting what happened either way.

This runs inside a worker process. Three things follow from that:

Nothing may escape. A runaway regex, a MemoryError, a parser that trips on
markup nobody anticipated — all of it has to come back as a record rather than
take down the run. So the body catches ``BaseException``, which is normally
wrong and is right here.

Nothing large may be returned. ``TextSlice._offsets`` holds one integer per
character of the paper, and pickling that across the process boundary would
dominate the cost of the entire sweep. The worker returns a plain dict.

And the paper is audited before it is discarded, because the audit needs the
source text and the source text is the thing we are about to drop.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

from .. import __version__
from ..engine import plan, run
from ..parse.acquire import UnreadableInput, acquire
from ..parse.document import (
    paper_from_latex,
    paper_from_source_if_jats,
    resolve_bibliography,
)
from ..resolve.base import NullResolver
from ..rules import load_all
from .record import PaperRecord, audit_anchors, fingerprint


def source_texts(source, paper, bib_text=None, bib_id=None) -> dict[str, str]:
    """Map every source id a finding might cite to the text behind it.

    Anchors point into three different things — the paper (or, for a spliced
    multi-file document, whichever included file the region came from) and the
    bibliography. The audit needs all of them keyed the way spans name them.
    """
    texts: dict[str, str] = {}

    if source.regions:
        # Spliced document: every span is resolved to the file it came from and
        # carries offsets local to that file. The combined text exists only
        # inside the parser and must not appear here — the root file's basename
        # collides with it, and mapping that name to the combined text measures
        # local offsets against the wrong string. That produced anchor failures
        # reading "line says 895, offset is on line 935".
        texts.update(source.files or {})
    else:
        texts[source.name] = source.text

    for entry in paper.bib:
        sid = entry.span.source.id
        if sid not in texts:
            # An inlined bibliography lives in the paper itself, in which case
            # the id already resolves above.
            texts[sid] = _bib_text_for(sid, source, bib_text, bib_id)

    return {k: v for k, v in texts.items() if v is not None}


def _bib_text_for(source_id: str, source, bib_text=None, bib_id=None) -> str | None:
    if bib_text is not None and source_id == bib_id:
        return bib_text
    if source.bib_text is not None and source_id == source.bib_name:
        return source.bib_text
    if source.files and source_id in source.files:
        return source.files[source_id]
    return None


def _reason_code(message: str) -> str:
    """A groupable code for why a paper could not be read.

    Acquisition yield is a real metric -- PDF-only submissions bound what the
    tool can ever cover -- and prose messages do not aggregate.
    """
    low = message.lower()
    for needle, code in (
        ("is a pdf", "pdf-only"),
        ("no latex source", "no-latex"),
        ("could not be unpacked", "unpack-failed"),
        ("larger than", "too-large"),
        ("unpacks to over", "expands-too-far"),
        ("more than", "too-many-members"),
        ("binary file", "binary"),
        ("not text in any encoding", "encoding"),
    ):
        if needle in low:
            return code
    return "other"


def _census(paper, wanted: set[str]) -> dict:
    """How much each requested parser actually found.

    A parser bug shows up here long before it shows up as a wrong finding: a
    slice empty on nine papers in ten is a broken extractor, not a corpus that
    happens to have no tables.

    Only requested slices are recorded. A slice nobody asked for would read as
    zero, which is indistinguishable from an extractor that found nothing --
    and that is exactly the confusion this whole census exists to prevent.
    """
    counts = {
        "sections": len(paper.sections),
        "tables": len(paper.tables),
        "numbers": len(paper.numbers),
        "means": len(paper.means),
        "stats": len(paper.stats),
        "hyperparameters": len(paper.hyperparameters),
        "citations": len(paper.citations),
        "bib": len(paper.bib),
        "resolutions": len(paper.resolutions),
    }
    census = {
        name: value
        for name, value in counts.items()
        if f"paper.{name}" in wanted
    }
    if "paper.text" in wanted:
        census["text_chars"] = len(paper.text.content) if paper.text else 0
    return census


def build_provider(spec):
    """Construct a model provider inside the worker that will use it.

    A spec -- {"provider": "groq", "name": "..."} -- rather than a built
    provider, because ProcessPoolExecutor pickles its arguments and both
    CachingProvider and DiskStore hold locks. Building per worker is also the
    right shape: each gets its own HTTP client and its own adaptive pacing,
    while the sqlite cache underneath is shared and in WAL mode, so an answer
    one worker pays for is free for the rest.
    """
    if not spec:
        return None

    from ..model.base import CachingProvider
    from ..model.openai_compat import OpenAICompatProvider
    from ..model.store import DiskStore

    inner = OpenAICompatProvider(
        model=spec.get("name", ""),
        provider=spec.get("provider", ""),
        base_url=spec.get("base_url", ""),
    )
    if not inner.configured:
        return None
    return CachingProvider(inner=inner, store=DiskStore(spec.get("cache_db")))


#: One resolver per worker process, keyed by the spec that built it. Not a
#: global for convenience -- see build_resolver.
_RESOLVERS: dict = {}


def build_resolver(spec):
    """The reference resolver for this worker, built once and reused.

    Same reasoning as build_provider for why it is constructed inside the
    worker at all: CachingResolver holds a lock and cannot be pickled across a
    process boundary. Each worker gets its own, which also means each paces
    itself -- so the worker count is what bounds the request rate against
    Crossref, not a shared budget nobody enforces.

    Memoised because a worker handles many papers in its lifetime and rebuilt
    per paper the caches were thrown away between them. Bibliographies overlap
    heavily -- the same landmark papers are cited across a corpus -- so a fresh
    cache per paper means re-querying DOIs already resolved, and at doi.org's
    pacing those repeats would dominate the wall clock rather than the work.
    The same applies to HttpResolver's registration-agency cache, which is
    keyed by DOI prefix and is worth almost nothing if it lives for one paper.

    Safe now in a way it was not before: workers used to be recycled by
    ``max_tasks_per_child``, and removing that for the deadlock means a worker
    lives for the whole run. The cache is bounded by the corpus, a few hundred
    entries, not by anything unbounded.
    """
    if not spec:
        return NullResolver()

    key = repr(sorted(spec.items()))
    resolver = _RESOLVERS.get(key)
    if resolver is None:
        from ..resolve import CachingResolver
        from ..resolve.http import HttpResolver

        resolver = _RESOLVERS[key] = CachingResolver(
            HttpResolver(mailto=spec.get("mailto") or None)
        )
    return resolver


def check_one(
    path: str | Path,
    *,
    paper_id: str | None = None,
    config=None,
    resolver=None,
    commit: str = "",
    model=None,
    repo_path=None,
    resolve=None,
) -> dict:
    """Run one paper end to end. Never raises."""
    path = Path(path)
    started = time.perf_counter()
    record = PaperRecord(
        paper_id=paper_id or path.name,
        resint_version=__version__,
        resint_commit=commit,
    )

    try:
        record.source_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        pass

    try:
        registry = load_all()
        provider = build_provider(model)
        active_resolver = resolver or build_resolver(resolve)
        chosen = plan(
            registry,
            config,
            has_repo=repo_path is not None,
            has_provider=provider is not None,
        )
        record.needs = sorted(chosen.paper_slices)

        acquired_at = time.perf_counter()
        try:
            source = acquire(path)
        except UnreadableInput as exc:
            # Not a crash. The tool looked and said why it could not read it,
            # which is a result worth recording rather than an error.
            record.status = "unreadable"
            record.acquire = {
                "outcome": "unreadable",
                "reason_code": _reason_code(str(exc)),
                "reason": str(exc),
            }
            record.timings = {"total": time.perf_counter() - started}
            return record.as_dict()

        bib_text, bib_id = resolve_bibliography(source, path)

        record.acquire = {
            "outcome": "ok",
            "root": source.name,
            "from_archive": source.from_archive,
            "files": len(source.files or {}),
            "spliced": len({r.name for r in source.regions}) if source.regions else 0,
            "bib_kind": source.bib_kind if bib_text else None,
        }

        parsed_at = time.perf_counter()
        # Same branch the CLI takes. Calling paper_from_latex directly here is
        # how six real PubMed Central articles came to be parsed as LaTeX.
        paper = paper_from_source_if_jats(
            source,
            needs=chosen.paper_slices,
            resolver=active_resolver,
        )
        if paper is None:
            paper = paper_from_latex(
                source.text,
                source_id=source.name,
                path=source.path,
                needs=chosen.paper_slices,
                bib_text=bib_text,
                bib_id=bib_id,
                bib_kind=source.bib_kind,
                resolver=active_resolver,
                regions=source.regions,
                files=source.files,
            )
        record.slice_census = _census(paper, chosen.paper_slices)

        # The five repro rules and claim/unimplemented have never run on a
        # real repository. Reading one is the only thing that changes that.
        repo = None
        if repo_path is not None:
            from ..parse.repo import read_repo

            try:
                repo = read_repo(repo_path, needs=chosen.repo_slices)
            except Exception as exc:  # noqa: BLE001
                record.notes.append(f"repository not read: {type(exc).__name__}: {exc}")

        ruled_at = time.perf_counter()
        report = run(
            paper, repo=repo, registry=registry, config=config, prepared=chosen,
            model=provider,
        )

        record.findings = [f.to_dict() for f in report.findings]
        record.unchecked = list(report.unchecked)
        record.notes = list(report.notes)
        record.skipped = dict(report.skipped)
        record.ran = list(report.ran)

        # Audited here, while the source text is still in hand.
        record.anchor_audit = audit_anchors(
            report.findings, source_texts(source, paper, bib_text, bib_id)
        ).as_dict()

        finished = time.perf_counter()
        record.timings = {
            "acquire": round(parsed_at - acquired_at, 4),
            "parse": round(ruled_at - parsed_at, 4),
            "rules": round(finished - ruled_at, 4),
            "total": round(finished - started, 4),
        }

        del paper

    except BaseException as exc:  # noqa: BLE001 — a worker must not die silently
        import traceback

        record.status = "error"
        record.error = {
            "type": type(exc).__name__,
            "message": str(exc)[:500],
            "fingerprint": fingerprint(exc),
            "traceback": traceback.format_exc()[-4000:],
        }
        record.timings = {"total": round(time.perf_counter() - started, 4)}

    return record.as_dict()
