"""Resolution semantics and the bibliography rules.

The property under test throughout: a lookup that fails must never become a
finding. Reporting a reference as fabricated because the network was down
would be the worst bug this tool could ship, so the boundary between
NOT_FOUND and UNKNOWN is pinned from several directions.
"""

import pytest

from resint.ir.paper import BibEntry, Citation, Paper
from resint.ir.span import Source, Span
from resint.resolve import (
    CachingResolver,
    NullResolver,
    Record,
    Resolution,
    StaticResolver,
    Status,
)
from resint.resolve.http import HttpResolver, normalize_doi, title_matches
from resint.rules import load_all
from resint.rules.bib.drift import title_overlap
from resint.rules.registry import Context

BIB = Source("refs.bib", "bib", path="refs.bib")
TEX = Source("paper.tex", "latex", path="paper.tex")
REG = load_all()


def entry(key, **fields):
    etype = fields.pop("entry_type", "article")
    spans = {name: Span(BIB, i * 10, i * 10 + 5) for i, name in enumerate(fields, 1)}
    return BibEntry(
        key=key,
        entry_type=etype,
        fields=fields,
        span=Span(BIB, 0, 200, line=1, label=f"[{key}]"),
        field_spans=spans,
    )


def paper_with(entries, resolutions, citations=()):
    p = Paper(source_id="paper.tex")
    p.bib = list(entries)
    p.resolutions = dict(resolutions)
    p.citations = list(citations)
    return p


def fire(rule_id, paper):
    return REG.get(rule_id).run(Context(paper=paper))


# --- resolution outcomes ------------------------------------------------


def test_null_resolver_always_reports_unknown():
    r = NullResolver().resolve(entry("k", title="T"))
    assert r.status is Status.UNKNOWN
    assert not r.checkable


def test_static_resolver_distinguishes_not_found_from_unknown():
    resolver = StaticResolver(
        records={"real": Record(source="crossref", title="Real")},
        unknown={"flaky"},
    )
    assert resolver.resolve(entry("real")).status is Status.FOUND
    assert resolver.resolve(entry("fake")).status is Status.NOT_FOUND
    assert resolver.resolve(entry("flaky")).status is Status.UNKNOWN


def test_caching_resolver_queries_once_per_doi():
    calls = []

    class _Counting:
        def resolve(self, e):
            calls.append(e.key)
            return Resolution(Status.NOT_FOUND)

    resolver = CachingResolver(_Counting())
    resolver.resolve(entry("a", doi="10.1/X"))
    resolver.resolve(entry("b", doi="10.1/x"))  # same DOI, different case
    assert len(calls) == 1


def test_caching_falls_back_to_title_and_year():
    calls = []

    class _Counting:
        def resolve(self, e):
            calls.append(e.key)
            return Resolution(Status.NOT_FOUND)

    resolver = CachingResolver(_Counting())
    resolver.resolve(entry("a", title="Same Title", year="2020"))
    resolver.resolve(entry("b", title="same title", year="2020"))
    assert len(calls) == 1


# --- bib/unresolved -----------------------------------------------------


def test_unresolved_fires_on_a_not_found_article():
    e = entry("ghost", title="A Paper That Does Not Exist", doi="10.5555/nope")
    findings = fire(
        "bib/unresolved",
        paper_with([e], {"ghost": Resolution(Status.NOT_FOUND, queried=("crossref",))}),
    )
    assert len(findings) == 1
    assert findings[0].severity.value == "high"
    assert "10.5555/nope" in findings[0].message


def test_unresolved_is_silent_when_the_lookup_failed():
    """UNKNOWN is the network's problem, not the paper's."""
    e = entry("ghost", title="Something", doi="10.5555/nope")
    findings = fire(
        "bib/unresolved",
        paper_with([e], {"ghost": Resolution(Status.UNKNOWN, detail="timeout")}),
    )
    assert findings == []


def test_unresolved_is_silent_on_a_found_record():
    e = entry("real", title="Real Work")
    findings = fire(
        "bib/unresolved",
        paper_with(
            [e],
            {"real": Resolution(Status.FOUND, record=Record(source="crossref", title="Real Work"))},
        ),
    )
    assert findings == []


def test_unresolved_is_silent_when_no_resolution_was_attempted():
    findings = fire("bib/unresolved", paper_with([entry("k", title="T")], {}))
    assert findings == []


@pytest.mark.parametrize(
    "etype, expected",
    [
        ("article", "high"),
        ("inproceedings", "high"),
        ("phdthesis", "low"),
        ("techreport", "low"),
        ("misc", "low"),
    ],
)
def test_severity_reflects_how_indexable_the_type_is(etype, expected):
    e = entry("k", title="Some Work", entry_type=etype)
    findings = fire(
        "bib/unresolved",
        paper_with([e], {"k": Resolution(Status.NOT_FOUND, queried=("crossref",))}),
    )
    assert findings[0].severity.value == expected


def test_a_failed_doi_outweighs_an_unindexable_type():
    """A thesis claiming a registered DOI that does not resolve is different."""
    e = entry("k", title="Work", doi="10.5555/nope", entry_type="phdthesis")
    findings = fire(
        "bib/unresolved",
        paper_with([e], {"k": Resolution(Status.NOT_FOUND, queried=("crossref",))}),
    )
    assert findings[0].severity.value == "high"


# --- bib/metadata-drift -------------------------------------------------


def test_year_drift_is_reported():
    e = entry("k", title="Same Work", year="2019")
    record = Record(source="crossref", title="Same Work", year="2021")
    findings = fire(
        "bib/metadata-drift",
        paper_with([e], {"k": Resolution(Status.FOUND, record=record)}),
    )
    assert len(findings) == 1
    assert "2019" in findings[0].message and "2021" in findings[0].message


def test_matching_metadata_is_silent():
    e = entry("k", title="Same Work", year="2021")
    record = Record(source="crossref", title="Same Work", year="2021")
    findings = fire(
        "bib/metadata-drift",
        paper_with([e], {"k": Resolution(Status.FOUND, record=record)}),
    )
    assert findings == []


def test_wholly_different_title_is_high():
    e = entry("k", title="Attention Is All You Need", year="2017")
    record = Record(
        source="crossref", title="Deep Residual Learning for Image Recognition", year="2017"
    )
    findings = fire(
        "bib/metadata-drift",
        paper_with([e], {"k": Resolution(Status.FOUND, record=record)}),
    )
    assert len(findings) == 1
    assert findings[0].severity.value == "high"


@pytest.mark.parametrize(
    "left, right",
    [
        ("LoRA: Low-Rank Adaptation", "LoRA: Low Rank Adaptation"),
        ("The Study of Things", "Study of Things"),
        ("Étude des méthodes", "Etude des methodes"),
        ("A Title: With a Subtitle", "A Title With a Subtitle"),
    ],
)
def test_cosmetic_title_differences_do_not_drift(left, right):
    assert title_overlap(left, right) >= 0.5


def test_missing_fields_are_not_treated_as_disagreement():
    e = entry("k", title="Work")  # no year at all
    record = Record(source="crossref", title="Work", year="2021")
    findings = fire(
        "bib/metadata-drift",
        paper_with([e], {"k": Resolution(Status.FOUND, record=record)}),
    )
    assert findings == []


# --- bib/orphans --------------------------------------------------------


def cite(key, offset=0):
    return Citation(key=key, span=Span(TEX, offset, offset + len(key), line=1))


def test_cited_without_an_entry():
    findings = fire(
        "bib/orphans",
        paper_with([entry("present", title="T")], {}, [cite("absent"), cite("present", 50)]),
    )
    assert len(findings) == 1
    assert "[absent]" in findings[0].message
    assert findings[0].absent_from == "refs.bib"


def test_entry_never_cited():
    findings = fire(
        "bib/orphans",
        paper_with([entry("unused", title="T")], {}, [])
    )
    assert len(findings) == 1
    assert "never cited" in findings[0].message
    assert findings[0].absent_from == "the paper"


def test_repeated_uses_are_counted_and_anchored():
    findings = fire(
        "bib/orphans",
        paper_with([entry("x", title="T")], {}, [cite("gone", i * 20) for i in range(4)]),
    )
    undefined = next(f for f in findings if "[gone]" in f.message)
    assert "4 times" in undefined.message
    assert len(undefined.anchors) == 3, "anchors are capped at three use sites"


def test_no_bibliography_means_abstain_not_accuse():
    findings = fire("bib/orphans", paper_with([], {}, [cite("anything")]))
    assert findings == []


def test_fully_consistent_bibliography_is_silent():
    findings = fire(
        "bib/orphans",
        paper_with([entry("a", title="T"), entry("b", title="U")], {}, [cite("a"), cite("b", 40)]),
    )
    assert findings == []


# --- http resolver helpers (no network) ---------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("10.1000/XYZ", "10.1000/xyz"),
        ("https://doi.org/10.1000/xyz", "10.1000/xyz"),
        ("doi:10.1000/xyz", "10.1000/xyz"),
        ("  10.1000/xyz  ", "10.1000/xyz"),
    ],
)
def test_doi_normalisation(raw, expected):
    assert normalize_doi(raw) == expected


def test_title_matching_requires_high_overlap():
    assert title_matches("Attention Is All You Need", "Attention is all you need")
    assert not title_matches("Attention Is All You Need", "Deep Residual Learning")


def test_resolver_abstains_without_anything_to_search_on():
    result = HttpResolver().resolve(entry("bare"))
    assert result.status is Status.UNKNOWN
    assert "neither a DOI nor a title" in result.detail
