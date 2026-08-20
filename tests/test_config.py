"""Configuration and suppression.

The property under test: a suppression must never be able to hide a
regression. Suppressed findings survive into the report marked with their
reason, and a suppression that stops matching says so instead of going quiet.
"""

from datetime import date

import pytest

from resint.config import Config, ConfigError, Suppression, discover, parse
from resint.engine import run
from resint.ir.finding import Finding
from resint.ir.span import Source, Span
from resint.parse.document import paper_from_path
from resint.rules import load_all

from conftest import CORPUS_RECORDS
from resint.resolve import StaticResolver
from pathlib import Path

SRC = Source("paper.tex", "latex", path="paper.tex")
CORPUS = Path(__file__).resolve().parents[1] / "corpus"
PLANTED = CORPUS / "planted" / "paper.tex"
RESOLVER = StaticResolver(records=dict(CORPUS_RECORDS))

SAMPLE = """
version: 1

suppress:
  - rule: bib/metadata-drift
    match: "[vaswani2017]"
    reason: "Cites the NeurIPS proceedings version deliberately."
    expires: "2099-01-01"

  - rule: numbers/table-arithmetic
    reason: "Totals are rounded per journal style."

rules:
  stats/grim: off
  stats/pvalue-mismatch: on
"""


def finding(rule_id="bib/metadata-drift", message="[vaswani2017] gives year 2017"):
    return Finding(
        rule_id=rule_id,
        severity="med",
        tier="deterministic",
        message=message,
        anchors=[Span(SRC, 0, 4), Span(SRC, 10, 14)],
    )


# --- parsing ------------------------------------------------------------


def test_reads_suppressions_and_disabled_rules():
    cfg = parse(SAMPLE)
    assert [s.rule for s in cfg.suppressions] == [
        "bib/metadata-drift",
        "numbers/table-arithmetic",
    ]
    assert cfg.disabled == {"stats/grim"}


def test_fields_are_unquoted_and_typed():
    first = parse(SAMPLE).suppressions[0]
    assert first.match == "[vaswani2017]"
    assert first.reason.startswith("Cites the NeurIPS")
    assert first.expires == date(2099, 1, 1)


def test_comments_are_ignored_but_not_inside_quotes():
    cfg = parse(
        'suppress:\n'
        '  - rule: bib/orphans  # trailing comment\n'
        '    reason: "keeps the # character"\n'
    )
    assert cfg.suppressions[0].rule == "bib/orphans"
    assert cfg.suppressions[0].reason == "keeps the # character"


def test_a_suppression_without_a_reason_is_rejected():
    """The config file is the record of every judgement about the work."""
    with pytest.raises(ConfigError, match="no reason"):
        parse("suppress:\n  - rule: bib/orphans\n")


def test_a_suppression_without_a_rule_is_rejected():
    with pytest.raises(ConfigError, match="no rule"):
        parse("suppress:\n  - reason: because\n")


def test_a_bad_expiry_date_is_rejected():
    with pytest.raises(ConfigError, match="expires|isoformat|Invalid"):
        parse(
            "suppress:\n  - rule: bib/orphans\n"
            "    reason: r\n    expires: not-a-date\n"
        )


def test_empty_config_is_valid():
    cfg = parse("")
    assert cfg.suppressions == [] and cfg.disabled == set()


# --- application --------------------------------------------------------


def test_a_matching_suppression_marks_but_does_not_remove():
    cfg = parse(SAMPLE)
    out, _ = cfg.apply([finding()])
    assert len(out) == 1, "the finding survives so it stays visible in JSON"
    assert out[0].suppressed
    assert "NeurIPS" in out[0].suppressed_reason


def test_the_match_string_narrows_to_one_finding():
    cfg = parse(SAMPLE)
    out, _ = cfg.apply([finding(message="[other2020] gives year 2017")])
    assert not out[0].suppressed


def test_an_expired_suppression_stops_applying_and_says_so():
    cfg = parse(
        'suppress:\n  - rule: bib/metadata-drift\n'
        '    reason: "temporary"\n    expires: "2020-01-01"\n'
    )
    out, notes = cfg.apply([finding()], today=date(2026, 1, 1))
    assert not out[0].suppressed
    assert any("expired" in n for n in notes)


def test_a_suppression_that_matches_nothing_is_reported():
    """Silent dead config is how a suppression file rots."""
    cfg = parse('suppress:\n  - rule: stats/grim\n    reason: "not applicable"\n')
    _, notes = cfg.apply([finding()])
    assert any("matched nothing" in n for n in notes)


def test_disabled_rules_are_reported_as_skipped():
    cfg = parse("rules:\n  stats/grim: off\n")
    report = run(paper_from_path(PLANTED, resolver=RESOLVER), config=cfg)
    assert "stats/grim" in report.skipped
    assert report.skipped["stats/grim"] == "disabled in .resint.yml"
    assert not any(f.rule_id == "stats/grim" for f in report.findings)


def test_suppressed_findings_are_excluded_from_counts_but_kept_in_output():
    cfg = parse(
        'suppress:\n  - rule: stats/grim\n    reason: "multi-item scale"\n'
    )
    report = run(paper_from_path(PLANTED, resolver=RESOLVER), config=cfg)
    grim = [f for f in report.findings if f.rule_id == "stats/grim"]
    assert len(grim) == 1 and grim[0].suppressed
    assert report.counts()["med"] == 4, "one fewer than unsuppressed"


def test_suppression_cannot_hide_a_regression_from_json():
    cfg = parse('suppress:\n  - rule: stats/grim\n    reason: "accepted"\n')
    report = run(paper_from_path(PLANTED, resolver=RESOLVER), config=cfg)
    payload = [f.to_dict() for f in report.findings]
    grim = next(f for f in payload if f["rule"] == "stats/grim")
    assert grim["suppressed"] is True
    assert grim["suppressed_reason"] == "accepted"


# --- discovery ----------------------------------------------------------


def test_discovery_walks_upward(tmp_path):
    (tmp_path / ".resint.yml").write_text(
        'suppress:\n  - rule: bib/orphans\n    reason: "root level"\n',
        encoding="utf-8",
    )
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    cfg = discover(nested)
    assert [s.rule for s in cfg.suppressions] == ["bib/orphans"]


def test_discovery_returns_an_empty_config_when_absent(tmp_path):
    assert discover(tmp_path).suppressions == []


def test_yaml_extension_is_also_found(tmp_path):
    (tmp_path / ".resint.yaml").write_text(
        'rules:\n  stats/grim: off\n', encoding="utf-8"
    )
    assert discover(tmp_path).disabled == {"stats/grim"}
