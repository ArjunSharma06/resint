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
    Registration,
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


def _dead(queried=("crossref", "doi.org")):
    """A DOI the DOI system itself denies. The only thing that may fire."""
    return Resolution(
        Status.NOT_FOUND, queried=queried, registration=Registration.DEAD
    )


def test_unresolved_fires_on_a_not_found_article():
    e = entry("ghost", title="A Paper That Does Not Exist", doi="10.5555/nope")
    findings = fire(
        "bib/unresolved",
        paper_with([e], {"ghost": _dead()}),
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


@pytest.mark.parametrize("etype", ["article", "inproceedings", "phdthesis"])
def test_a_title_only_miss_is_not_bib_unresolved_at_all(etype):
    """The rule now reports dead DOIs and nothing else.

    It used to report title-search misses too, at the same table: 176 of them
    against 18 DOI findings across 68 real papers, and the 176 buried the 18.
    That half is bib/unindexed now, off by default.
    """
    e = entry("k", title="Some Work", entry_type=etype)
    findings = fire(
        "bib/unresolved",
        paper_with([e], {"k": _dead()}),
    )
    assert findings == []


def test_a_dead_doi_is_high_severity():
    """A DOI is a claim about one registered record: it resolves or it does
    not. That is the fabrication signal, and the only thing this rule reports."""
    findings = fire(
        "bib/unresolved",
        paper_with(
            [entry("k", title="Work", doi="10.5555/nope")],
            {"k": _dead()},
        ),
    )
    assert len(findings) == 1
    assert findings[0].severity.value == "high"
    assert "10.5555/nope" in findings[0].message


def test_a_dead_doi_on_an_unindexable_type_is_softened_not_dropped():
    """A thesis with a DOI that does not resolve is still a dead DOI. But such
    work is more often deposited where these indices do not reach, so the
    claim is made more quietly rather than not at all."""
    e = entry("k", title="Work", doi="10.5555/nope", entry_type="phdthesis")
    findings = fire(
        "bib/unresolved",
        paper_with([e], {"k": _dead()}),
    )
    assert len(findings) == 1
    assert findings[0].severity.value == "med"


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


def test_metadata_drift_no_longer_judges_titles():
    """A title disagreeing under a resolving DOI is not drift -- it means the
    DOI points at a different paper, which is a separate claim with different
    evidence and a different fix. It moved to bib/doi-mismatch, where the
    author list has to corroborate before anything is reported."""
    e = entry("k", title="Attention Is All You Need", year="2017")
    record = Record(
        source="crossref", title="Deep Residual Learning for Image Recognition", year="2017"
    )
    findings = fire(
        "bib/metadata-drift",
        paper_with([e], {"k": Resolution(Status.FOUND, record=record)}),
    )
    assert findings == []


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


# --- an unreachable index is not a search -------------------------------


def test_an_unreachable_index_is_not_counted_as_searched():
    """The safety property of bib/unresolved: a reference is reported missing
    only when the indices were reached and had nothing.

    _get() used to return None for both "answered, no match" and "could not
    connect", so an offline machine looked like proof that a paper does not
    exist. Found when DBLP was added and its TLS handshake failed locally --
    every reference then claimed four indices had been searched when three had.
    """
    from resint.resolve.http import HttpResolver, Unreachable

    class _AllDown(HttpResolver):
        def _crossref(self, entry):
            raise Unreachable("crossref unreachable")

        def _openalex(self, entry):
            raise Unreachable("openalex unreachable")

        def _arxiv(self, entry):
            raise Unreachable("arxiv unreachable")

        def _dblp(self, entry):
            raise Unreachable("dblp unreachable")

    result = _AllDown().resolve(entry("k", title="Some Paper"))
    assert result.status is Status.UNKNOWN
    assert result.queried == ()
    assert not result.checkable, "UNKNOWN can never support a finding"


def test_a_partly_reachable_run_names_only_what_answered():
    """The finding's claim must be the same size as the evidence behind it."""
    from resint.resolve.http import HttpResolver, Unreachable

    class _OneDown(HttpResolver):
        def _crossref(self, entry):
            return None  # answered, no match

        def _openalex(self, entry):
            return None

        def _arxiv(self, entry):
            return None

        def _dblp(self, entry):
            raise Unreachable("dblp unreachable")

    result = _OneDown().resolve(entry("k", title="Some Paper"))
    assert result.status is Status.NOT_FOUND
    assert "dblp" not in result.queried
    assert "dblp" in result.detail


def test_an_index_being_down_no_longer_blocks_a_finding():
    """The claim used to be "no index has this", which an unsearched index
    left a hole in. It is now "the DOI system has no such handle", which one
    authority answers alone -- DBLP being down cannot make a registered DOI
    look unregistered. Eight of nine findings on batch-1c carried that caveat
    while DBLP was down for the whole sweep."""
    e = entry("k", title="Some Paper", doi="10.5555/nope")
    resolution = Resolution(
        Status.NOT_FOUND,
        queried=("crossref", "openalex", "doi.org"),
        registration=Registration.DEAD,
        detail="dblp could not be reached and was not searched",
    )
    findings = fire("bib/unresolved", paper_with([e], {"k": resolution}))
    assert len(findings) == 1
    assert "10.5555/nope" in findings[0].message


def test_a_live_doi_outside_our_indices_is_never_a_finding():
    """The bias this rule shipped with. Two of nine findings on batch-1c were
    DOIs registered through the Chinese agency, resolving via chndoi.org and
    unknown to all four indices -- reported at high severity as fabrication.
    A tool that does this fires on papers citing Chinese-language work."""
    e = entry("k", title="Real Chinese Paper", doi="10.16718/j.1009-7708.2025.06.002")
    resolution = Resolution(
        Status.NOT_FOUND,
        queried=("crossref", "openalex", "arxiv", "dblp", "doi.org"),
        registration=Registration.REGISTERED,
        agency="Chinese Academy of Sciences",
    )
    findings = fire("bib/unresolved", paper_with([e], {"k": resolution}))
    assert findings == []


def test_a_doi_org_that_could_not_be_reached_is_never_a_finding():
    """Absence of an answer is not an answer. The indices missing it and the
    authority being unreachable is exactly the UNCHECKED case."""
    e = entry("k", title="Some Paper", doi="10.5555/maybe")
    resolution = Resolution(
        Status.NOT_FOUND,
        queried=("crossref",),
        registration=Registration.UNCHECKED,
    )
    findings = fire("bib/unresolved", paper_with([e], {"k": resolution}))
    assert findings == []


# =========================================================================
# The DOI system as authority: the paths that must never fire
#
# bib/unresolved fires on one thing only -- doi.org denying a handle. Every
# other outcome of that lookup has to stay silent, and the ways it can fail
# are exactly the ways the original bug could come back.
# =========================================================================


class _Answer:
    """A urlopen result: context manager, status, body, final URL."""

    def __init__(self, body=b"", status=200, url="https://example.org/landing"):
        self._body, self.status, self.url = body, status, url

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._body


def _resolver(recorder=None):
    from resint.resolve.http import HttpResolver

    return HttpResolver(pacer=recorder or _Waits())


class _Waits:
    """A Pacer that records rather than sleeps."""

    def __init__(self):
        self.seen = []

    def wait(self, index):
        self.seen.append(index)


@pytest.mark.parametrize("code", [429, 500, 502, 503])
def test_a_throttled_doi_org_is_never_read_as_unregistered(monkeypatch, code):
    """The bug's worst possible return. A 429 read as "no such DOI" would
    fire on precisely the bibliographies with many non-Crossref references --
    the population we just stopped accusing -- and it would look like a
    finding rather than an outage."""
    import urllib.error
    import urllib.request

    def throttled(request, timeout=None):
        raise urllib.error.HTTPError(
            request.full_url, code, "throttled", {}, None
        )

    monkeypatch.setattr(urllib.request, "urlopen", throttled)
    registration, agency, record = _resolver()._doi_org("10.5555/x")

    assert registration is Registration.UNCHECKED
    assert record is None
    # And UNCHECKED cannot reach a finding, whatever else is true.
    e = entry("k", title="A Paper", doi="10.5555/x")
    resolution = Resolution(
        Status.NOT_FOUND, queried=("crossref",), registration=registration
    )
    assert fire("bib/unresolved", paper_with([e], {"k": resolution})) == []


def test_only_a_404_from_doi_org_means_unregistered(monkeypatch):
    """404 is the one status that answers the question asked."""
    import urllib.error
    import urllib.request

    def missing(request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, 404, "no", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", missing)
    assert _resolver()._doi_org("10.5555/x")[0] is Registration.DEAD


def test_both_doi_org_calls_are_paced(monkeypatch):
    """Two requests per miss -- the record and the agency -- and a
    bibliography heavy in non-Crossref agencies makes that the common path,
    not the rare one."""
    import urllib.request

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda request, timeout=None: _Answer(b'{"title": "T", "DOI": "10.5/x"}'),
    )
    waits = _Waits()
    _resolver(waits)._doi_org("10.5/x")
    assert waits.seen == ["doi.org", "doi.org"]


def test_a_live_doi_is_still_reported_when_the_agency_lookup_fails():
    """doiRA failing must not swallow the coverage gap. The abstention is the
    only place our blind spots are visible, and it matters more than the name
    of the agency behind them."""
    from resint.rules.registry import Context

    e = entry("k", title="A Paper", doi="10.16718/j.x")
    paper = paper_with(
        [e],
        {"k": Resolution(
            Status.NOT_FOUND,
            queried=("crossref", "doi.org"),
            registration=Registration.REGISTERED,
            agency="",          # doiRA did not answer
        )},
    )
    ctx = Context(paper=paper, rule=load_all().get("bib/unresolved"))
    findings = list(load_all().get("bib/unresolved").fn(ctx))

    assert findings == []
    assert any("registered and resolves" in a for a in ctx.abstentions)


def test_the_agency_falls_back_to_whoever_answered(monkeypatch):
    """A worse name, but a true one."""
    import urllib.error
    import urllib.request

    def only_the_landing_page(request, timeout=None):
        if "doiRA" in request.full_url:
            raise urllib.error.HTTPError(request.full_url, 503, "down", {}, None)
        return _Answer(b"<html>not JSON</html>",
                       url="https://www.chndoi.org/Resolution/Handler?doi=10.16718/x")

    monkeypatch.setattr(urllib.request, "urlopen", only_the_landing_page)
    registration, agency, record = _resolver()._doi_org("10.16718/x")

    # An unparseable body is still proof the handle resolved.
    assert registration is Registration.REGISTERED
    assert record is None
    assert agency == "chndoi.org"


def test_with_no_resolver_the_authority_path_never_engages():
    """The default, and --offline. Everything UNKNOWN, nothing fires, and the
    census says the lookups did not happen rather than staying silent."""
    from resint.rules.registry import Context

    e = entry("k", title="A Paper", doi="10.5555/x")
    resolution = NullResolver().resolve(e)
    assert resolution.status is Status.UNKNOWN
    assert resolution.registration is Registration.UNCHECKED

    paper = paper_with([e], {"k": resolution})
    ctx = Context(paper=paper, rule=load_all().get("bib/unresolved"))
    findings = list(load_all().get("bib/unresolved").fn(ctx))

    assert findings == []
    census = [a for a in ctx.abstentions if "reference" in a]
    assert census and "0 looked up" in census[0]
    assert "could not be looked up and were not judged" in census[0]


def test_the_agency_is_looked_up_once_per_prefix(monkeypatch):
    """The agency is a property of the DOI prefix, so a bibliography with
    forty Chinese-registered references asks once. Without this the slowest
    path in the resolver runs on exactly the papers it was added to stop
    accusing."""
    import urllib.request

    calls = []

    def answer(request, timeout=None):
        calls.append(request.full_url)
        if "doiRA" in request.full_url:
            return _Answer(b'[{"DOI": "x", "RA": "Chinese Academy of Sciences"}]')
        return _Answer(b"<html>landing</html>", url="https://chndoi.org/x")

    monkeypatch.setattr(urllib.request, "urlopen", answer)
    resolver = _resolver()

    first = resolver._doi_org("10.16718/j.1")
    second = resolver._doi_org("10.16718/j.2")
    other = resolver._doi_org("10.13604/j.3")

    assert first[1] == second[1] == "Chinese Academy of Sciences"
    # Three handle lookups, but only two agency lookups: one per prefix.
    assert sum("doiRA" in c for c in calls) == 2
    assert other[1] == "Chinese Academy of Sciences"


def test_a_failed_agency_lookup_is_not_cached(monkeypatch):
    """doiRA being down for one reference must not blank the agency for the
    rest of the bibliography."""
    import urllib.error
    import urllib.request

    state = {"fail": True}

    def flaky(request, timeout=None):
        if "doiRA" in request.full_url:
            if state["fail"]:
                state["fail"] = False
                raise urllib.error.HTTPError(request.full_url, 503, "down", {}, None)
            return _Answer(b'[{"RA": "JaLC"}]')
        return _Answer(b"<html>landing</html>", url="https://jalc.jst.go.jp/x")

    monkeypatch.setattr(urllib.request, "urlopen", flaky)
    resolver = _resolver()

    assert resolver._doi_org("10.5555/a")[1] == "jalc.jst.go.jp"  # fallback
    assert resolver._doi_org("10.5555/b")[1] == "JaLC"            # retried
