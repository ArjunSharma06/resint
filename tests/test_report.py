"""Terminal rendering and the CLI.

Output is where a linter crashes in front of a first-time user, so the
encoding fallback is pinned here rather than left to the platform.
"""

import io
import json
from pathlib import Path

import pytest

from resint.cli import main
from resint.engine import run
from resint.ir.finding import Finding, Severity, Tier
from resint.parse.document import paper_from_path
from resint.report.terminal import marks, render, render_finding, _Paint
from conftest import CORPUS_RECORDS
from resint.resolve import StaticResolver

CORPUS = Path(__file__).resolve().parents[1] / "corpus"
RESOLVER = StaticResolver(records=dict(CORPUS_RECORDS))
POSITIVE = CORPUS / "planted" / "paper.tex"
NEGATIVE = CORPUS / "clean" / "paper.tex"


class _Stream(io.StringIO):
    def __init__(self, encoding: str):
        super().__init__()
        self._encoding = encoding

    @property
    def encoding(self):
        return self._encoding

    def isatty(self):
        return False


# --- encoding fallback --------------------------------------------------


def test_unicode_tick_used_when_the_stream_supports_it():
    assert marks(_Stream("utf-8")) == ("✓", "~")


@pytest.mark.parametrize("encoding", ["cp1252", "ascii", "cp437"])
def test_ascii_mark_used_on_legacy_encodings(encoding):
    """Regression: a bare tick raised UnicodeEncodeError on a Windows console."""
    det, _ = marks(_Stream(encoding))
    assert det == "+"
    det.encode(encoding)


def test_render_never_emits_unencodable_characters_on_cp1252():
    report = run(paper_from_path(POSITIVE, resolver=RESOLVER))
    stream = _Stream("cp1252")
    out = render(report, "planted.tex", 0.4, stream=stream)
    out.encode("cp1252")


def test_render_uses_unicode_separator_when_available():
    report = run(paper_from_path(POSITIVE, resolver=RESOLVER))
    assert " · " in render(report, "p.tex", 0.4, stream=_Stream("utf-8"))
    assert " | " in render(report, "p.tex", 0.4, stream=_Stream("cp1252"))


# --- content ------------------------------------------------------------


def test_report_names_the_tier_and_counts():
    report = run(paper_from_path(POSITIVE, resolver=RESOLVER))
    out = render(report, "planted.tex", 1.8, stream=_Stream("utf-8"))
    assert "10 findings (3 high, 5 med, 2 low)" in out
    assert "no API key used" in out


def test_clean_paper_says_so_and_still_lists_unchecked():
    report = run(paper_from_path(NEGATIVE, resolver=RESOLVER))
    out = render(report, "clean.tex", 0.2, stream=_Stream("utf-8"))
    assert "No findings." in out
    assert "unchecked:" in out


def test_skipped_rules_are_reported_with_their_reason():
    """"No findings" and "did not look" must not read the same."""
    report = run(paper_from_path(POSITIVE, resolver=RESOLVER))
    report.skipped = {
        "repro/seed-claim": "no repository supplied",
        "repro/hparam-drift": "no repository supplied",
        "claim/unimplemented": "no model provider configured",
    }
    out = render(report, "planted.tex", 0.2, stream=_Stream("utf-8"))
    assert "skipped: 2 rules, no repository supplied" in out
    assert "skipped: 1 rule, no model provider configured" in out


def test_repo_rules_are_skipped_when_no_repository_is_given():
    """Skipping is reported, never silently counted as passing."""
    report = run(paper_from_path(POSITIVE, resolver=RESOLVER))
    assert report.skipped, "repro/ rules cannot run without --repo"
    assert set(report.skipped.values()) == {"no repository supplied"}
    assert all(r.startswith("repro/") for r in report.skipped)


def test_suppressed_findings_are_hidden_from_the_terminal():
    report = run(paper_from_path(POSITIVE, resolver=RESOLVER))
    report.findings = [f.suppress("known and accepted") for f in report.findings]
    out = render(report, "planted.tex", 0.2, stream=_Stream("utf-8"))
    assert "No findings." in out


def test_long_messages_wrap_rather_than_run_off():
    report = run(paper_from_path(POSITIVE, resolver=RESOLVER))
    grim = next(f for f in report.findings if f.rule_id == "stats/grim")
    body = render_finding(grim, _Paint(False), ("+", "~"))
    assert all(len(line) < 100 for line in body)


def test_identical_anchor_locations_collapse_in_display():
    report = run(paper_from_path(POSITIVE, resolver=RESOLVER))
    for f in report.findings:
        parts = f.locate().split(" <-> ")
        assert len(parts) == len(set(parts)), f"{f.rule_id} repeats a location"


# --- CLI ----------------------------------------------------------------


def test_check_exits_nonzero_when_a_high_finding_is_present(capsys):
    assert main(["check", str(POSITIVE), "--offline"]) == 1
    assert "stats/pvalue-mismatch" in capsys.readouterr().out


def test_check_exits_zero_on_a_clean_paper(capsys):
    assert main(["check", str(NEGATIVE), "--offline"]) == 0
    assert "No findings." in capsys.readouterr().out


def test_missing_file_reports_cleanly(capsys):
    assert main(["check", "does-not-exist.tex"]) == 2
    assert "no such file" in capsys.readouterr().err


def test_json_output_is_machine_readable(capsys):
    main(["check", str(POSITIVE), "--offline", "--format", "json"])
    payload = json.loads(capsys.readouterr().out)
    # Offline: the two resolution rules abstain, so only the local ones fire.
    assert payload["counts"] == {"high": 2, "med": 4, "low": 1}
    assert any("not reported as missing" in u for u in payload["unchecked"])
    for f in payload["findings"]:
        expected = 1 if f["absent_from"] else 2
        assert len(f["anchors"]) >= expected
    assert payload["unchecked"] == [] or isinstance(payload["unchecked"], list)


def test_min_severity_filters_output(capsys):
    main(["check", str(POSITIVE), "--offline", "--min-severity", "high", "--format", "json"])
    payload = json.loads(capsys.readouterr().out)
    assert {f["severity"] for f in payload["findings"]} == {"high"}


def test_rules_command_lists_blind_spots(capsys):
    assert main(["rules"]) == 0
    out = capsys.readouterr().out
    assert "stats/grim" in out
    assert "cannot detect:" in out
