"""End to end: source files in, anchored findings out.

The negative fixture carries the heavier guarantee. Catching planted defects
shows the rules work; staying silent on a clean paper shows they can be
trusted, and only the second one determines whether anybody keeps using this.
"""

from pathlib import Path

import pytest

from resint.engine import run, selectable
from resint.ir.finding import Severity, Tier
from resint.parse.document import find_bibliography, paper_from_path
from resint.resolve import NullResolver, StaticResolver, Status
from resint.rules import load_all

CORPUS = Path(__file__).resolve().parents[1] / "corpus"
POSITIVE = CORPUS / "planted" / "paper.tex"
NEGATIVE = CORPUS / "clean" / "paper.tex"


@pytest.fixture(scope="module")
def planted(corpus_resolver):
    return run(paper_from_path(POSITIVE, resolver=corpus_resolver))


@pytest.fixture(scope="module")
def clean(corpus_resolver):
    return run(paper_from_path(NEGATIVE, resolver=corpus_resolver))


def sources_for(tex: Path) -> dict[str, str]:
    files = {tex.name: tex.read_text(encoding="utf-8")}
    bib = find_bibliography(tex)
    if bib:
        files[bib.name] = bib.read_text(encoding="utf-8")
    return files


# --- the negative fixture: silence is the requirement -------------------


def test_clean_paper_produces_no_findings(clean):
    assert clean.findings == [], [
        f"{f.rule_id}: {f.message}" for f in clean.findings
    ]


def test_clean_paper_still_reports_what_it_could_not_check(clean):
    """Two sample sizes in one sentence is an abstention, not a pass."""
    assert any("2 sample sizes" in u for u in clean.unchecked)


def test_accented_entry_does_not_drift(clean, corpus_resolver):
    """A title full of LaTeX accents must match its index record after folding."""
    paper = paper_from_path(NEGATIVE, resolver=corpus_resolver)
    entry = next(e for e in paper.bib if e.key == "accented2019")
    assert entry.title == "Étude des méthodes adaptatives"
    assert paper.resolutions["accented2019"].status is Status.FOUND


# --- the positive fixture: each planted defect, once --------------------


def test_expected_defect_count(planted):
    # One fewer than before the bib/unresolved split: the unindexable thesis
    # is no longer counted at all, because a thesis Crossref has never heard
    # of is not a finding at any severity.
    assert planted.summary() == "9 findings (3 high, 5 med, 1 low)"


def test_grim_violation_is_caught(planted):
    grim = [f for f in planted.findings if f.rule_id == "stats/grim"]
    assert len(grim) == 1
    assert "3.45" in grim[0].message and "3.50" in grim[0].message
    assert grim[0].severity is Severity.MED, "item count was assumed"


def test_decision_error_is_caught_and_escalated(planted):
    decision = [
        f
        for f in planted.findings
        if f.rule_id == "stats/pvalue-mismatch" and f.severity is Severity.HIGH
    ]
    assert len(decision) == 1
    assert "0.244" in decision[0].message


def test_plain_inconsistency_is_caught_at_medium(planted):
    plain = [
        f
        for f in planted.findings
        if f.rule_id == "stats/pvalue-mismatch" and f.severity is Severity.MED
    ]
    assert len(plain) == 1


def test_unresolvable_reference_is_high_when_it_claims_a_doi(planted):
    found = [
        f
        for f in planted.findings
        if f.rule_id == "bib/unresolved" and f.severity is Severity.HIGH
    ]
    assert len(found) == 1
    assert "zhang2023adaptive" in found[0].message
    assert "does not recognise" in found[0].message


def test_an_unindexable_thesis_is_not_reported_at_all(planted):
    """Plenty of real work is simply not indexed, and a thesis Crossref has
    never heard of is not a finding at any severity.

    It used to be reported at low severity, which still spent a line of the
    report and still asked the reader to judge something the tool could not
    know. Excluding it from the denominator is the honest version -- and it is
    a large part of why the rule fired on three papers in four.
    """
    assert not [
        f
        for f in planted.findings
        if f.rule_id == "bib/unresolved" and "obscure2018" in f.message
    ]


def test_year_drift_is_caught(planted):
    drift = [f for f in planted.findings if f.rule_id == "bib/metadata-drift"]
    assert len(drift) == 1
    assert "gives year 2019" in drift[0].message and "2021" in drift[0].message


def test_undefined_citation_is_caught(planted):
    orphans = [
        f
        for f in planted.findings
        if f.rule_id == "bib/orphans" and "missing_entry2022" in f.message
    ]
    assert len(orphans) == 1
    assert orphans[0].absent_from == "refs.bib"


def test_uncited_entry_is_caught(planted):
    orphans = [
        f
        for f in planted.findings
        if f.rule_id == "bib/orphans" and "never_cited2020" in f.message
    ]
    assert len(orphans) == 1
    assert orphans[0].absent_from == "the paper"


def test_resolved_entries_do_not_fire(planted):
    """dosovitskiy2020 resolves cleanly and is cited. It must be untouched."""
    assert not any("dosovitskiy2020" in f.message for f in planted.findings)


def test_the_consistent_test_does_not_fire(planted):
    assert not any("4.35" in f.message for f in planted.findings)


def test_figure_caption_never_reaches_the_extractors(planted):
    assert not any("caption" in f.message.lower() for f in planted.findings)


# --- anchors have to be true -------------------------------------------


def test_every_finding_anchors_into_real_source_text(planted):
    files = sources_for(POSITIVE)
    for f in planted.findings:
        assert f.anchors, f.rule_id
        for a in f.anchors:
            text = files[a.source.id]
            assert 0 <= a.start < a.end <= len(text)
            assert text[a.start : a.end].strip(), f"{f.rule_id} anchors empty text"


def test_anchors_land_on_the_planted_lines(planted):
    tex = POSITIVE.read_text(encoding="utf-8").splitlines()
    bib = (CORPUS / "planted" / "refs.bib").read_text(encoding="utf-8").splitlines()

    grim = next(f for f in planted.findings if f.rule_id == "stats/grim")
    assert "M = 3.47" in tex[grim.anchors[0].line - 1]
    assert "N = 20" in tex[grim.anchors[1].line - 1]

    drift = next(f for f in planted.findings if f.rule_id == "bib/metadata-drift")
    assert "2019" in bib[drift.anchors[0].line - 1]

    undefined = next(
        f for f in planted.findings if "missing_entry2022" in f.message
    )
    assert "missing_entry2022" in tex[undefined.anchors[0].line - 1]


def test_absence_findings_carry_one_anchor_and_name_what_is_missing(planted):
    for f in planted.findings:
        if f.absent_from:
            assert len(f.anchors) >= 1
            assert f.absent_from.strip()
        else:
            assert len(f.anchors) >= 2


def test_findings_are_ordered_worst_first(planted):
    order = [f.severity.value for f in planted.findings]
    rank = {"high": 0, "med": 1, "low": 2}
    assert order == sorted(order, key=lambda s: rank[s])


# --- resolution must never manufacture a finding ------------------------


def test_failed_lookups_never_become_findings():
    """Offline must degrade to "could not check", never to "fabricated"."""
    offline = StaticResolver(
        records={}, unknown={"zhang2023adaptive", "obscure2018", "hu2021lora",
                             "dosovitskiy2020", "never_cited2020"}
    )
    report = run(paper_from_path(POSITIVE, resolver=offline))
    assert not any(f.rule_id == "bib/unresolved" for f in report.findings)
    assert not any(f.rule_id == "bib/metadata-drift" for f in report.findings)
    assert any("could not be looked up" in u for u in report.unchecked)


def test_null_resolver_is_the_default_and_reports_nothing_missing():
    report = run(paper_from_path(POSITIVE, resolver=NullResolver()))
    assert not any(f.rule_id == "bib/unresolved" for f in report.findings)
    assert any("not reported as missing" in u for u in report.unchecked)


def test_missing_bibliography_abstains_rather_than_flagging_every_key(tmp_path):
    """No .bib file means we did not look, which is not the same as not found."""
    tex = tmp_path / "solo.tex"
    tex.write_text(r"Text with \cite{someone2020} in it.", encoding="utf-8")
    report = run(paper_from_path(tex))
    assert not any(f.rule_id == "bib/orphans" for f in report.findings)
    assert any("no .bib file supplied" in u for u in report.unchecked)


# --- selection and laziness --------------------------------------------


def test_repo_and_model_rules_are_skipped_with_a_stated_reason():
    runnable, skipped = selectable(
        load_all().all(), has_repo=False, has_provider=False
    )
    assert all(not r.needs_repo for r in runnable)
    assert all(r.tier is Tier.DETERMINISTIC for r in runnable)
    assert all(reason for reason in skipped.values())


def test_a_deterministic_run_reports_no_model_use(planted):
    assert planted.used_model is False


def test_lazy_population_skips_unrequested_slices(corpus_resolver):
    paper = paper_from_path(POSITIVE, needs={"paper.stats"}, resolver=corpus_resolver)
    assert paper.stats
    assert paper.means == [] and paper.bib == [] and paper.resolutions == {}


def test_resolution_only_happens_when_a_rule_asks_for_it():
    """The only network-touching slice must not run unless it was declared."""

    class _Tripwire:
        called = 0

        def resolve(self, entry):
            self.__class__.called += 1
            return NullResolver().resolve(entry)

    paper_from_path(POSITIVE, needs={"paper.bib"}, resolver=_Tripwire())
    assert _Tripwire.called == 0

    paper_from_path(POSITIVE, needs={"paper.bib", "paper.resolutions"},
                    resolver=_Tripwire())
    assert _Tripwire.called > 0


def test_min_severity_filters(corpus_resolver):
    high_only = run(
        paper_from_path(POSITIVE, resolver=corpus_resolver),
        min_severity=Severity.HIGH,
    )
    assert {f.severity for f in high_only.findings} == {Severity.HIGH}
