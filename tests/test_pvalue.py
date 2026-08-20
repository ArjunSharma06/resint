"""stats/pvalue-mismatch -- recomputation, severity split, precision gate."""

import pytest

from resint.ir.finding import Severity
from resint.rules.registry import REGISTRY, Context
from resint.rules.stats.pvalue import evaluate, recompute

RULE = REGISTRY.get("stats/pvalue-mismatch")


class _Paper:
    def __init__(self, stats):
        self.stats = stats


# (kind, statistic, df1, df2, p, comparator, tail)
CONSISTENT = [
    ("t", "2.086", 20, None, "0.05", "=", 2),
    ("t", "5.00", 30, None, "0.001", "<", 2),
    ("t", "1.725", 20, None, "0.05", "=", 1),      # declared one-tailed
    ("F", "4.3512", 1, 20, "0.05", "=", 2),
    ("chi2", "3.8415", 1, None, "0.05", "=", 2),
    ("z", "1.96", None, None, "0.05", "=", 2),
    ("r", "0.5", 18, None, "0.02", "=", 2),        # p = .0248 -> rounds to .02
]

# disagreements that do not move the result across alpha
INCONSISTENT = [
    ("t", "2.086", 20, None, "0.01", "=", 2),      # both sides significant
    ("t", "2.10", 30, None, "0.001", "<", 2),      # .0442, still significant
]

# disagreements that flip the significance decision
DECISION = [
    ("t", "1.20", 20, None, "0.03", "=", 2),       # actually .244
    ("chi2", "2.50", 1, None, "0.03", "=", 2),     # actually .114
    ("F", "1.00", 1, 20, "0.02", "=", 2),          # actually .329
    ("z", "1.20", None, None, "0.04", "=", 2),     # actually .230
]


def _mk(stat_factory, spec):
    kind, statistic, df1, df2, p, comparator, tail = spec
    return stat_factory(kind, statistic, df1=df1, df2=df2, p=p, comparator=comparator, tail=tail)


@pytest.mark.parametrize("spec", CONSISTENT)
def test_consistent_fixtures_are_silent(stat_factory, spec):
    assert evaluate(_mk(stat_factory, spec)).verdict == "consistent"


@pytest.mark.parametrize("spec", INCONSISTENT)
def test_plain_inconsistencies_classified(stat_factory, spec):
    assert evaluate(_mk(stat_factory, spec)).verdict == "inconsistent"


@pytest.mark.parametrize("spec", DECISION)
def test_decision_errors_classified(stat_factory, spec):
    assert evaluate(_mk(stat_factory, spec)).verdict == "decision"


def test_one_tailed_is_trusted_as_declared(stat_factory):
    """The same statistic is consistent one-tailed and inconsistent two-tailed."""
    one = stat_factory("t", "1.725", df1=20, p="0.05", tail=1)
    two = stat_factory("t", "1.725", df1=20, p="0.05", tail=2)
    assert evaluate(one).verdict == "consistent"
    assert evaluate(two).flagged


def test_r_converts_through_t(stat_factory):
    assert recompute(stat_factory("r", "0.5", df1=18)) == pytest.approx(0.02477, abs=1e-4)


@pytest.mark.parametrize(
    "spec",
    [
        ("t", "2.0", None, None, "0.05", "=", 2),     # missing df
        ("r", "1.0", 18, None, "0.05", "=", 2),       # |r| >= 1
        ("F", "2.0", 1, None, "0.05", "=", 2),        # missing second df
    ],
)
def test_unsupported_inputs_abstain(stat_factory, spec):
    result = evaluate(_mk(stat_factory, spec))
    assert result.verdict == "unsupported"
    assert not result.flagged


# --- rule wiring --------------------------------------------------------


def test_decision_error_escalates_to_high(stat_factory):
    stats = [_mk(stat_factory, DECISION[0])]
    findings = RULE.run(Context(paper=_Paper(stats)))
    assert len(findings) == 1
    f = findings[0]
    assert f.severity is Severity.HIGH
    assert "disagree about significance" in f.message
    assert f.affects == ("the significance claim resting on this test",)
    assert len(f.anchors) == 2, "the statistic and the reported p"


def test_plain_inconsistency_stays_at_declared_severity(stat_factory):
    stats = [_mk(stat_factory, INCONSISTENT[0])]
    f = RULE.run(Context(paper=_Paper(stats)))[0]
    assert f.severity is Severity.MED
    assert "not across" in f.message


def test_rule_silent_on_clean_paper(stat_factory):
    stats = [_mk(stat_factory, s) for s in CONSISTENT]
    assert RULE.run(Context(paper=_Paper(stats))) == []


def test_rule_declares_its_blind_spot():
    assert "One-tailed" in RULE.cannot_detect
    assert "multiple" in RULE.cannot_detect
    assert RULE.requires == ("paper.stats",)


def test_message_renders_the_test_as_reported(stat_factory):
    f = RULE.run(Context(paper=_Paper([_mk(stat_factory, DECISION[2])])))[0]
    assert "F(1, 20) = 1.00" in f.message


# --- precision gate -----------------------------------------------------

PRECISION_FLOOR = 1.0


def test_measured_precision_meets_the_declared_floor(stat_factory):
    flagged = INCONSISTENT + DECISION
    true_positives = sum(
        bool(RULE.run(Context(paper=_Paper([_mk(stat_factory, s)])))) for s in flagged
    )
    false_positives = sum(
        bool(RULE.run(Context(paper=_Paper([_mk(stat_factory, s)])))) for s in CONSISTENT
    )
    precision = true_positives / (true_positives + false_positives)

    assert precision >= PRECISION_FLOOR, f"precision {precision:.2f}"
    assert true_positives == len(flagged), "recall"
