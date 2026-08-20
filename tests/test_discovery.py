"""Discovery: adding a rule file is the only step required to ship a rule."""

from resint.ir.finding import Tier
from resint.rules import load_all


def test_discovery_finds_the_implemented_rules():
    reg = load_all()
    ids = {r.id for r in reg.all()}
    assert {"stats/grim", "stats/pvalue-mismatch"} <= ids


def test_every_registered_rule_meets_the_contributor_bar():
    """The bar is structural: CI enforces it, review does not have to argue it."""
    for r in load_all().all():
        assert "/" in r.id, r.id
        assert r.requires, f"{r.id} declares no requirements"
        assert len(r.cannot_detect) > 30, f"{r.id}: cannot_detect is too thin to be honest"
        assert r.cannot_detect.rstrip().endswith("."), f"{r.id}: write it as prose"
        for req in r.requires:
            assert req.split(".")[0] in {"paper", "repo"}, req


def test_deterministic_rules_need_no_provider():
    reg = load_all()
    det = reg.by_tier(Tier.DETERMINISTIC)
    assert det, "v1 must ship usable rules that require no API key"
    assert all(r.tier is Tier.DETERMINISTIC for r in det)
