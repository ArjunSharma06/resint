"""The two-anchor constraint, enforced rather than encouraged."""

import pytest

from resint.ir.finding import AnchorError, Finding, Severity, Tier

from conftest import span


def build(**over):
    base = dict(
        rule_id="numbers/internal-mismatch",
        severity="high",
        tier="deterministic",
        message="94.2 in the abstract, 93.8 in Table 3.",
        anchors=[span(0, 4, line=4, label="abstract"), span(90, 94, line=40, label="table3")],
    )
    base.update(over)
    return Finding(**base)


def test_two_anchors_accepted():
    assert len(build().anchors) == 2


@pytest.mark.parametrize("anchors", [[], [span(0, 1)]])
def test_fewer_than_two_anchors_rejected(anchors):
    with pytest.raises(AnchorError, match="at least two anchors"):
        build(anchors=anchors)


def test_empty_message_rejected():
    with pytest.raises(ValueError, match="must not be empty"):
        build(message="   ")


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_confidence_bounds(bad):
    with pytest.raises(ValueError, match="confidence"):
        build(confidence=bad)


def test_locate_names_both_sides():
    assert build().locate() == "abstract:L4 <-> table3:L40"


def test_severity_orders():
    assert Severity.LOW < Severity.MED < Severity.HIGH


def test_suppression_preserves_the_finding():
    """A suppressed finding survives into output, so it cannot mask a regression."""
    f = build().suppress("Cites the proceedings version deliberately.")
    assert f.suppressed
    assert f.to_dict()["suppressed_reason"].startswith("Cites")
    assert f.message == build().message
    assert f.anchors == build().anchors


def test_suppression_requires_a_reason():
    with pytest.raises(ValueError, match="must state a reason"):
        build().suppress("  ")


def test_findings_are_immutable():
    with pytest.raises(AttributeError):
        build().severity = Severity.LOW


def test_to_dict_round_trips_anchors():
    d = build().to_dict()
    assert d["tier"] == "deterministic"
    assert [a["locate"] for a in d["anchors"]] == ["abstract:L4", "table3:L40"]
