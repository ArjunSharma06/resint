"""Discovery: adding a rule file is the only step required to ship a rule."""

from resint.ir.finding import Tier
from resint.ir.paper import Paper
from resint.ir.repo import Repo
from resint.parse.document import ALL_SLICES
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


def test_every_declared_paper_slice_is_one_the_builder_populates():
    """A requirement nobody fills reads as an empty list, forever, silently.

    ``requires=`` is validated for *shape* at registration but not against the
    set of slices ``paper_from_latex`` actually knows how to build. So a rule
    can declare ``paper.claims`` -- a real field on Paper -- register cleanly,
    gate cleanly, and quietly do nothing. This is the guard that turns that
    into a failing build instead of a rule that appears to pass.
    """
    for rule in load_all().all():
        for req in rule.requires:
            if req.startswith("paper."):
                assert req in ALL_SLICES, (
                    f"{rule.id} requires {req!r}, which paper_from_latex does "
                    f"not populate. Add it to ALL_SLICES and give it a builder, "
                    f"or the rule will silently see nothing."
                )


def test_every_declared_slice_exists_as_a_field():
    """Catches a typo'd requirement before it becomes silent emptiness."""
    paper_fields = set(vars(Paper(source_id="x")))
    repo_fields = set(vars(Repo(root=".")))

    for rule in load_all().all():
        for req in rule.requires:
            root, _, attr = req.partition(".")
            known = paper_fields if root == "paper" else repo_fields
            assert attr in known, f"{rule.id} requires {req!r}, which is not a field"


def test_deterministic_rules_need_no_provider():
    reg = load_all()
    det = reg.by_tier(Tier.DETERMINISTIC)
    assert det, "v1 must ship usable rules that require no API key"
    assert all(r.tier is Tier.DETERMINISTIC for r in det)
