"""bib/unindexed: no DOI, and a title search found nothing.

Split out of ``bib/unresolved``, which reported this at the same table as a
DOI that fails to resolve. Across 68 real papers that meant 176 title-only
findings against 18 DOI ones, and the 176 buried the 18.

The rule is **off by default**, which is the point of most of these tests: a
failed title search is weak evidence about the world and strong evidence about
our index coverage.
"""

import pytest

from resint.config import parse as parse_config
from resint.engine import plan
from resint.ir.paper import BibEntry, Paper
from resint.ir.span import Source, Span
from resint.resolve import Resolution, Status
from resint.rules import load_all
from resint.rules.registry import Context

REG = load_all()
BIB = Source("refs.bib", "bib", path="refs.bib")


def entry(key="k", entry_type="article", **fields):
    spans = {name: Span(BIB, i * 10, i * 10 + 5) for i, name in enumerate(fields, 1)}
    return BibEntry(
        key=key,
        entry_type=entry_type,
        fields=fields,
        span=Span(BIB, 0, 200, line=1, label=f"[{key}]"),
        field_spans=spans,
    )


def fire(entries, resolutions, rule_id="bib/unindexed"):
    paper = Paper(source_id="paper.tex")
    paper.bib = list(entries)
    paper.resolutions = dict(resolutions)
    return REG.get(rule_id).run(Context(paper=paper))


NOT_FOUND = {"k": Resolution(Status.NOT_FOUND, queried=("crossref", "openalex"))}


# --- what it reports ----------------------------------------------------


def test_a_title_only_miss_is_reported_here():
    findings = fire([entry(title="Some Paper", year="2020")], NOT_FOUND)
    assert len(findings) == 1
    assert findings[0].severity.value == "low"
    assert "crossref, openalex" in findings[0].message


def test_the_fix_points_at_the_thing_that_would_help():
    """Adding a DOI turns an unanswerable search into a check."""
    findings = fire([entry(title="Some Paper")], NOT_FOUND)
    assert "DOI" in findings[0].fix


def test_an_entry_with_a_doi_belongs_to_bib_unresolved():
    findings = fire([entry(title="Some Paper", doi="10.5555/nope")], NOT_FOUND)
    assert findings == []


# --- the denominator, shrunk on purpose ---------------------------------


@pytest.mark.parametrize(
    "etype", ["phdthesis", "mastersthesis", "techreport", "unpublished", "misc"]
)
def test_unindexable_types_are_excluded_not_downgraded(etype):
    """A thesis Crossref has never heard of is not a finding at any severity.

    Reporting it quietly still spends a line of the report and still asks the
    reader to judge something the tool cannot know. Excluding it is a large
    part of why the original rule fired on three papers in four.
    """
    findings = fire([entry(title="A Thesis", entry_type=etype)], NOT_FOUND)
    assert findings == []


@pytest.mark.parametrize("status", [Status.FOUND, Status.UNKNOWN])
def test_only_a_completed_search_can_report_absence(status):
    """UNKNOWN means the network answered badly, not that the paper is wrong."""
    assert fire([entry(title="Some Paper")], {"k": Resolution(status)}) == []


# --- off by default -----------------------------------------------------


def test_the_rule_is_not_in_a_default_run():
    chosen = plan(REG, has_repo=False)
    assert "bib/unindexed" not in {r.id for r in chosen.runnable}


def test_being_off_is_reported_not_silent():
    """A rule that silently did not run is indistinguishable from one that ran
    and found nothing -- the confusion this whole tool exists to avoid."""
    chosen = plan(REG, has_repo=False)
    assert "off by default" in chosen.skipped["bib/unindexed"]


def test_the_config_can_switch_it_on():
    cfg = parse_config("rules:\n  bib/unindexed: on\n")
    chosen = plan(REG, cfg, has_repo=False)
    assert "bib/unindexed" in {r.id for r in chosen.runnable}


def test_switching_it_on_and_off_at_once_leaves_it_off():
    """'off' is the more conservative reading and wins."""
    cfg = parse_config("rules:\n  bib/unindexed: on\n  bib/unindexed: off\n")
    chosen = plan(REG, cfg, has_repo=False)
    assert "bib/unindexed" not in {r.id for r in chosen.runnable}


def test_a_normal_rule_is_unaffected_by_the_mechanism():
    chosen = plan(REG, has_repo=False)
    assert "bib/unresolved" in {r.id for r in chosen.runnable}


# --- it declares itself -------------------------------------------------


def test_the_rule_says_what_it_cannot_know():
    rule = REG.get("bib/unindexed")
    assert rule.opt_in
    assert "coverage" in rule.cannot_detect
    assert "bib/unresolved" in rule.cannot_detect
