"""Baselining: stable identity, and suppressions written beside the line.

Without both of these nobody runs a linter twice on the same paper. A finding
identified by line number moves when a paragraph is added, so the second run
reports the whole document as new; and a judgement about one reference that
has to be written in a config file three directories away mostly does not get
written at all.
"""

import pytest

from resint.engine import run
from resint.parse.document import paper_from_latex
from resint.parse.inline import apply_inline, find_directives
from resint.rules import load_all

REG = load_all()

GRIM = "With N = 20 the mean was 3.47 on the scale."


def build(body):
    return paper_from_latex(
        "\\documentclass{article}\\begin{document}\n" + body + "\n\\end{document}\n"
    )


def findings_for(body):
    return run(build(body), registry=REG).findings


# --- fingerprints -------------------------------------------------------


def test_a_fingerprint_survives_text_moving_down_the_page():
    """The property baselining rests on. Adding a paragraph shifts every line
    below it; a fingerprint that moved would report the whole paper as new
    findings, which is the same as reporting nothing."""
    before = findings_for(GRIM)
    after = findings_for("An added introductory paragraph.\n\n" + GRIM)

    assert before and after
    assert before[0].anchors[0].line != after[0].anchors[0].line, "the line moved"
    assert before[0].fingerprint() == after[0].fingerprint(), "the identity did not"


def test_different_findings_of_the_same_rule_differ():
    """Two GRIM violations in one paper are distinguished by their mean."""
    a = findings_for("With N = 20 the mean was 3.47 on the scale.")
    b = findings_for("With N = 20 the mean was 3.42 on the scale.")
    assert a[0].fingerprint() != b[0].fingerprint()


def test_the_fingerprint_reaches_json():
    finding = findings_for(GRIM)[0]
    assert finding.to_dict()["fingerprint"] == finding.fingerprint()


def test_a_fingerprint_is_short_enough_to_paste():
    assert len(findings_for(GRIM)[0].fingerprint()) == 12


# --- inline suppression -------------------------------------------------


def test_a_directive_above_a_line_suppresses_it():
    body = (
        "% resint: ignore stats/grim -- responses were averaged before reporting\n"
        + GRIM
    )
    findings = findings_for(body)
    assert len(findings) == 1
    assert findings[0].suppressed
    assert findings[0].suppressed_reason == (
        "responses were averaged before reporting"
    )


def test_a_suppressed_finding_still_exists():
    """It is still produced, still counted, and still in JSON with its reason,
    so a suppression can never hide a regression from the corpus."""
    body = "% resint: ignore stats/grim -- deliberate\n" + GRIM
    finding = findings_for(body)[0]
    assert finding.to_dict()["suppressed_reason"] == "deliberate"


def test_a_directive_for_another_rule_does_not_apply():
    body = "% resint: ignore bib/orphans -- unrelated\n" + GRIM
    assert not findings_for(body)[0].suppressed


def test_a_directive_elsewhere_in_the_document_does_not_reach():
    body = (
        "% resint: ignore stats/grim -- too far away\n"
        "Some unrelated prose sits between them.\n"
        "More unrelated prose.\n" + GRIM
    )
    assert not findings_for(body)[0].suppressed


@pytest.mark.parametrize("sep", ["--", ":"])
def test_either_separator_is_accepted(sep):
    directives, _ = find_directives(f"% resint: ignore stats/grim {sep} a reason\n")
    assert len(directives) == 1
    assert directives[0].reason == "a reason"


def test_a_directive_without_a_reason_is_refused_and_explained():
    """Same rule as the config file: a silenced finding with no explanation is
    unauditable six months later. Refusing silently would be worse -- someone
    wrote it meaning to suppress something."""
    directives, notes = find_directives("% resint: ignore stats/grim\n")
    assert directives == []
    assert notes and "has no reason" in notes[0]
    assert "stats/grim" in notes[0]


def test_an_unused_directive_is_reported():
    """A suppression that matches nothing usually means the finding is fixed
    and the comment is now stale."""
    _, notes = apply_inline([], find_directives(
        "% resint: ignore stats/grim -- was a problem once\n"
    )[0])
    assert notes and "matched nothing" in notes[0]


def test_directives_are_read_from_the_source_not_the_prose():
    """A comment is exactly what normalization strips, and its line number has
    to be the author's."""
    paper = build("% resint: ignore stats/grim -- deliberate\n" + GRIM)
    assert len(paper.inline_suppressions) == 1
    # Line 2: the helper puts the preamble and body-start on line 1.
    assert paper.inline_suppressions[0].line == 2


def test_a_paper_with_no_directives_is_unaffected():
    assert build(GRIM).inline_suppressions == []
    assert not findings_for(GRIM)[0].suppressed


# --- the labelling harness's own discipline -----------------------------


def test_ambiguous_is_recorded_against_the_rule():
    """Decided before labelling starts, so it cannot be argued case by case.

    A finding a careful reader cannot adjudicate in three minutes is one a
    user will not adjudicate either. Same asymmetry as UNKNOWN never becoming
    a finding, applied to the person holding the keyboard.
    """
    import pathlib

    source = pathlib.Path("tools/review.py").read_text(encoding="utf-8")
    skip_branch = source.split('if answer == "s":', 1)[1].split("continue", 1)[0]
    assert '"correct": False' in skip_branch
    assert '"ambiguous"' in skip_branch


def test_a_rate_is_refused_on_too_few_labels():
    """Wilson on a handful is noise wearing a percentage sign."""
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("review", "tools/review.py")
    review = module_from_spec(spec)
    spec.loader.exec_module(review)

    assert review.MIN_FOR_A_RATE >= 10
    lo, hi = review._interval(9, 10)
    assert hi - lo > 0.3, "ten labels cannot pin a rate down, and must not pretend to"
