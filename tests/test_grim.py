"""stats/grim -- algorithm, rule wiring, and the precision gate.

Structured to the contributor bar: positive fixtures the rule must catch,
negative fixtures it must stay silent on, and a measured precision floor that
fails CI on regression.
"""

from decimal import Decimal

import pytest

from resint.ir.finding import Severity
from resint.rules.registry import REGISTRY, Context
from resint.rules.stats.grim import grim

RULE = REGISTRY.get("stats/grim")

# (mean, n, decimals) -- means unreachable from any integer sum
POSITIVE = [
    ("3.47", 20, 2),
    ("5.19", 28, 2),
    ("1.234", 50, 3),
    ("4.05", 30, 2),
]

# means that ARE reachable, or where the test has no power. Silence required.
NEGATIVE = [
    ("3.45", 20, 2),    # 69/20 exactly
    ("3.50", 20, 2),    # 70/20 exactly
    ("2.75", 20, 2),    # 55/20 exactly
    ("3.47", 200, 2),   # granularity >= 10^2: no power
    ("3.4", 20, 1),     # 68/20 exactly
    ("4", 17, 0),       # no decimals reported: no power
]


@pytest.mark.parametrize("raw, n, dp", POSITIVE)
def test_positive_fixtures_are_inconsistent(raw, n, dp):
    assert grim(Decimal(raw), n, dp).verdict == "inconsistent"


@pytest.mark.parametrize("raw, n, dp", NEGATIVE)
def test_negative_fixtures_are_not_flagged(raw, n, dp):
    assert grim(Decimal(raw), n, dp).verdict != "inconsistent"


def test_nearest_attainable_values_bracket_the_reported_mean():
    result = grim(Decimal("3.47"), 20, 2)
    assert result.nearest == (Decimal("3.45"), Decimal("3.50"))
    assert result.nearest[0] < Decimal("3.47") < result.nearest[1]


def test_no_power_states_its_reason():
    result = grim(Decimal("3.47"), 200, 2)
    assert result.verdict == "no-power"
    assert "carries no information" in result.reason


def test_items_per_participant_changes_granularity(mean_factory):
    """3.47 is unreachable from 20 single responses but reachable from 20x3."""
    assert grim(Decimal("3.47"), 20, 2).verdict == "inconsistent"
    assert grim(Decimal("3.47"), 60, 2).verdict == "consistent"

    single = mean_factory("3.47", n=20)
    multi = mean_factory("3.47", n=20, items=3)
    assert len(RULE.run(Context(paper=_paper([single])))) == 1
    assert RULE.run(Context(paper=_paper([multi]))) == []


def test_rejects_nonpositive_granularity():
    with pytest.raises(ValueError, match="granularity must be positive"):
        grim(Decimal("3.0"), 0, 2)


# --- rule wiring --------------------------------------------------------


class _Paper:
    def __init__(self, means):
        self.means = means


def _paper(means):
    return _Paper(means)


def test_rule_emits_two_anchors_and_high_severity(mean_factory):
    findings = RULE.run(Context(paper=_paper([mean_factory("3.47", n=20, context="condition A")])))
    assert len(findings) == 1
    f = findings[0]
    assert f.rule_id == "stats/grim"
    assert f.severity is Severity.HIGH
    assert len(f.anchors) == 2, "the mean and the N it was computed from"
    assert "3.45" in f.message and "3.50" in f.message
    assert "condition A" in f.message
    assert f.locate() == "results:L10 <-> method:L8"


def test_rule_is_silent_on_a_clean_paper(mean_factory):
    clean = [mean_factory(raw, n=n) for raw, n, _ in NEGATIVE if n < 100]
    assert RULE.run(Context(paper=_paper(clean))) == []


def test_rule_declares_its_blind_spot():
    assert "non-integer" in RULE.cannot_detect
    assert RULE.requires == ("paper.means",)
    assert not RULE.needs_repo


# --- precision gate -----------------------------------------------------

PRECISION_FLOOR = 1.0


def test_measured_precision_meets_the_declared_floor(mean_factory):
    """No false positives across the negative corpus. Regressions fail CI."""
    true_positives = sum(
        bool(RULE.run(Context(paper=_paper([mean_factory(raw, n=n)]))))
        for raw, n, _ in POSITIVE
    )
    false_positives = sum(
        bool(RULE.run(Context(paper=_paper([mean_factory(raw, n=n)]))))
        for raw, n, _ in NEGATIVE
    )
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / len(POSITIVE)

    assert precision >= PRECISION_FLOOR, f"precision {precision:.2f}"
    assert recall == 1.0, f"recall {recall:.2f}"
