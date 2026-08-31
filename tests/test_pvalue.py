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


# --- the reported statistic is an interval, not a point -----------------
#
# "t = 2.086" is not a claim that t equals 2.086; it is a claim that the
# author had a value rounding to 2.086, so t lies in [2.0855, 2.0865). p is
# recomputed at both ends and a finding requires the whole resulting range to
# fall outside the range the reported p claims.
#
# Every one of these came from a real paper, and every one was reported as an
# error by the point-estimate version. All three were the author's rounding.


@pytest.mark.parametrize(
    "spec, note",
    [
        (("t", "0.18", 267, None, "0.859", "=", 2), "computed p in [0.8534, 0.8612]"),
        (("t", "0.42", 267, None, "0.676", "=", 2), "computed p in [0.6712, 0.6785]"),
        (("F", "0.02", 1, 267, "0.900", "=", 2), "computed p in [0.8745, 0.9026]"),
    ],
)
def test_rounding_in_the_statistic_is_not_an_error(stat_factory, spec, note):
    """PMC12933109 reported three of these. All three were false positives,
    and the tool had no way to know because precision was never measured."""
    assert evaluate(_mk(stat_factory, spec)).verdict == "consistent", note


def test_a_genuine_disagreement_still_fires(stat_factory):
    """The interval must not swallow real errors: t(20)=2.086 gives p ~ .0499,
    which is nowhere near the [0.025, 0.035) that 'p = .03' claims."""
    result = evaluate(_mk(stat_factory, ("t", "2.086", 20, None, "0.03", "=", 2)))
    assert result.flagged


def test_a_less_precise_statistic_widens_the_interval(stat_factory):
    """Fewer decimals is a weaker claim, so it must be harder to contradict.
    't = 2.1' admits [2.05, 2.15); 't = 2.100' admits [2.0995, 2.1005)."""
    from resint.rules.stats.pvalue import computed_interval

    coarse = stat_factory("t", "2.1", df1=20, p="0.05")
    fine = stat_factory("t", "2.100", df1=20, p="0.05")

    c_lo, c_hi = computed_interval(coarse)
    f_lo, f_hi = computed_interval(fine)
    assert (c_hi - c_lo) > (f_hi - f_lo)


def test_a_straddling_interval_is_not_called_a_decision_error(stat_factory):
    """When the recomputed range spans alpha, the recomputation has not
    established which side the result falls on. Claiming the conclusion
    changed would be asserting more than the arithmetic supports."""
    from resint.rules.stats.pvalue import ALPHA, computed_interval

    test = stat_factory("t", "2.09", df1=20, p="0.20")
    lo, hi = computed_interval(test)
    if lo < ALPHA <= hi:
        assert evaluate(test).verdict == "inconsistent"


def test_the_statistic_precision_is_read_from_what_was_printed(stat_factory):
    assert stat_factory("t", "2.086", df1=20, p="0.05").statistic_decimals == 3
    assert stat_factory("t", "2.1", df1=20, p="0.05").statistic_decimals == 1
    assert stat_factory("t", "18", df1=20, p="0.05").statistic_decimals == 0
