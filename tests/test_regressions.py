"""Regressions found by running resint on a real paper.

Every test here corresponds to something that went wrong on first contact
with a genuine 48 KB LaTeX source and a 35-entry bibliography. The corpus
fixtures did not catch any of them, which is the argument for running the
tool on real work early and often.
"""

import time

import pytest

from resint.ir.finding import Finding
from resint.ir.paper import BibEntry, Citation, Paper
from resint.ir.span import Source, Span
from resint.resolve import Record, Resolution, Status
from resint.resolve.base import CachingResolver, StaticResolver, resolve_all
from resint.resolve.http import _best, title_matches, title_similarity
from resint.rules import load_all
from resint.rules.registry import Context

BIB = Source("refs.bib", "bib", path="refs.bib")
TEX = Source("paper.tex", "latex", path="paper.tex")
REG = load_all()


def entry(key, **fields):
    return BibEntry(
        key=key,
        entry_type=fields.pop("entry_type", "article"),
        fields=fields,
        span=Span(BIB, 0, 40, line=1, label=f"[{key}]"),
        field_spans={},
    )


# --- title matching -----------------------------------------------------
#
# The metric used overlap over the *smaller* token set, so a short title
# fully contained in a longer one scored near 1.0. That is how Linformer
# matched a completely unrelated paper, and then drift reported its year as
# wrong against it.


@pytest.mark.parametrize(
    "entry_title, found_title",
    [
        (
            "Linformer: Self-Attention with Linear Complexity",
            "Mult-Pool Self Attention: a lightweight attention with linear complexity",
        ),
        (
            "Mamba-2: Transformers are SSMs",
            "Efficient Estimation of Generalized Structured Models",
        ),
        (
            "Gated Linear Attention",
            "FLAA: Fused Linear Attention Accelerator for Efficient Inference",
        ),
        (
            "A Survey of Transformers",
            "Survey on Multimodal Transformers for Robots",
        ),
    ],
)
def test_unrelated_papers_do_not_match(entry_title, found_title):
    assert not title_matches(entry_title, found_title)


@pytest.mark.parametrize(
    "left, right",
    [
        ("Attention Is All You Need", "Attention is all you need"),
        ("LoRA: Low-Rank Adaptation", "LoRA: Low Rank Adaptation"),
        ("The Information Bottleneck Method", "Information Bottleneck Method"),
    ],
)
def test_the_same_paper_still_matches(left, right):
    assert title_matches(left, right)


def test_similarity_is_symmetric():
    """The old metric was not, which is what let subset titles through."""
    a, b = "Linformer: Self-Attention with Linear Complexity", "Self Attention"
    assert title_similarity(a, b) == title_similarity(b, a)


def test_best_match_wins_over_first_over_the_line():
    """Search endpoints rank by their relevance, not ours."""
    candidates = [
        Record(source="x", title="Attention Is All You Need, Revisited Again"),
        Record(source="x", title="Attention Is All You Need"),
    ]
    assert _best("Attention Is All You Need", candidates).title == (
        "Attention Is All You Need"
    )


def test_best_returns_none_when_nothing_clears_the_floor():
    assert _best("Linformer", [Record(source="x", title="Something Else")]) is None


# --- drift requires an authoritative record -----------------------------


def paper_with(entries, resolutions):
    p = Paper(source_id="paper.tex")
    p.bib = list(entries)
    p.resolutions = dict(resolutions)
    return p


def test_drift_abstains_on_a_title_matched_record():
    """A guess cannot support a claim that someone's metadata is wrong."""
    e = entry("vaswani2017", title="Attention Is All You Need", year="2017")
    record = Record(
        source="openalex",
        title="Attention Is All You Need",
        year="2025",
        matched_by="title",
    )
    ctx = Context(paper=paper_with([e], {"vaswani2017": Resolution(Status.FOUND, record=record)}))
    findings = REG.get("bib/metadata-drift").run(ctx)

    assert findings == []
    assert any("found by title search" in a for a in ctx.abstentions)


def test_drift_still_fires_on_a_doi_matched_record():
    e = entry("hu2021", title="LoRA", year="2019", doi="10.1/x")
    record = Record(source="crossref", title="LoRA", year="2021", matched_by="doi")
    ctx = Context(paper=paper_with([e], {"hu2021": Resolution(Status.FOUND, record=record)}))
    findings = REG.get("bib/metadata-drift").run(ctx)

    assert len(findings) == 1
    assert "2019" in findings[0].message and "2021" in findings[0].message


def test_records_default_to_authoritative():
    assert Record(source="crossref").authoritative
    assert not Record(source="crossref", matched_by="title").authoritative


# --- batch resolution ---------------------------------------------------
#
# Sequential resolution over 35 entries against 3 indices made the tool
# appear to hang. It had to be interrupted.


class _Slow:
    def __init__(self, delay):
        self.delay = delay

    def resolve(self, entry):
        time.sleep(self.delay)
        return Resolution(Status.NOT_FOUND)


def test_budget_caps_total_time():
    entries = [entry(f"k{i}", title=f"T{i}") for i in range(20)]
    started = time.monotonic()
    results = resolve_all(_Slow(0.5), entries, workers=4, budget=0.6)
    elapsed = time.monotonic() - started

    assert elapsed < 3.0, "the budget must actually bound the wall clock"
    assert len(results) == len(entries), "every entry gets an answer"


def test_entries_beyond_the_budget_are_unknown_never_missing():
    """Running out of time must not manufacture a fabricated-citation finding."""
    entries = [entry(f"k{i}", title=f"T{i}") for i in range(20)]
    results = resolve_all(_Slow(0.5), entries, workers=2, budget=0.3)

    unknown = [r for r in results.values() if r.status is Status.UNKNOWN]
    assert unknown, "some should have been cut off"
    assert all(r.status is not Status.NOT_FOUND for r in unknown)
    assert any("budget" in r.detail for r in unknown)


def test_progress_is_reported():
    seen = []
    entries = [entry(f"k{i}", title=f"T{i}") for i in range(5)]
    resolve_all(StaticResolver(), entries, progress=lambda d, t: seen.append((d, t)))
    assert seen and seen[-1] == (5, 5)


def test_a_probe_that_raises_becomes_unknown():
    class _Boom:
        def resolve(self, entry):
            raise RuntimeError("network exploded")

    results = resolve_all(_Boom(), [entry("k", title="T")])
    assert results["k"].status is Status.UNKNOWN


def test_caching_resolver_is_safe_under_concurrency():
    calls = []

    class _Counting:
        def resolve(self, e):
            calls.append(e.key)
            time.sleep(0.01)
            return Resolution(Status.NOT_FOUND)

    resolver = CachingResolver(_Counting())
    same = [entry(f"k{i}", doi="10.1/SAME") for i in range(12)]
    results = resolve_all(resolver, same, workers=6)

    assert len(results) == 12
    assert len(calls) < 12, "the cache should collapse repeated lookups"


def test_empty_bibliography_resolves_to_nothing():
    assert resolve_all(StaticResolver(), []) == {}


# --- report readability -------------------------------------------------


def test_uncited_entries_are_one_finding_not_thirteen():
    """Thirteen separate low findings buried everything else in the report."""
    entries = [entry(f"unused{i}", title=f"T{i}") for i in range(13)]
    paper = Paper(source_id="paper.tex")
    paper.bib = entries
    paper.citations = []

    findings = REG.get("bib/orphans").run(Context(paper=paper))
    assert len(findings) == 1
    assert "13 entries are defined" in findings[0].message
    assert "and 5 more" in findings[0].message


def test_undefined_keys_stay_separate():
    """Each is a distinct broken reference in the compiled document."""
    paper = Paper(source_id="paper.tex")
    paper.bib = [entry("present", title="T")]
    paper.citations = [
        Citation(key="present", span=Span(TEX, 0, 7, line=1)),
        Citation(key="ghost_a", span=Span(TEX, 10, 17, line=2)),
        Citation(key="ghost_b", span=Span(TEX, 20, 27, line=3)),
    ]
    findings = REG.get("bib/orphans").run(Context(paper=paper))
    assert len(findings) == 2


def test_location_line_truncates_beyond_two_anchors():
    f = Finding(
        rule_id="bib/orphans",
        severity="low",
        tier="deterministic",
        message="m",
        anchors=[Span(BIB, i * 10, i * 10 + 5, line=i + 1, label=f"e{i}") for i in range(5)],
        absent_from="the paper",
    )
    where = f.locate()
    assert "+3 more" in where
    assert where.count("<->") == 3
    assert len(f.anchors) == 5, "all anchors are still retained for JSON"
