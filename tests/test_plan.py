"""Rule selection precedes data loading.

Before the plan seam existed, the loader had no idea which rules were active
and built every slice -- so a run with the bibliography rules switched off
still parsed the bibliography and still opened sockets. The architecture
described laziness the code did not deliver.
"""

from pathlib import Path

import pytest

from resint.config import parse as parse_config
from resint.engine import Plan, plan, required_slices, run
from resint.ir.finding import Tier
from resint.parse.document import ALL_SLICES, paper_from_path
from resint.rules import load_all

CORPUS = Path(__file__).resolve().parents[1] / "corpus"
PLANTED = CORPUS / "planted" / "paper.tex"
REG = load_all()


def test_a_plan_asks_only_for_what_its_rules_declared():
    chosen = plan(REG, has_repo=False)
    assert chosen.paper_slices == required_slices(chosen.runnable) & ALL_SLICES
    assert chosen.paper_slices < ALL_SLICES or chosen.paper_slices == ALL_SLICES


def test_disabling_the_bib_rules_removes_the_network_slice():
    """This is the whole point: no bib rules, no sockets."""
    cfg = parse_config(
        "rules:\n"
        "  bib/unresolved: off\n"
        "  bib/metadata-drift: off\n"
        "  bib/orphans: off\n"
    )
    chosen = plan(REG, cfg, has_repo=False)

    assert "paper.resolutions" not in chosen.paper_slices
    assert not chosen.opens_network
    assert {"bib/unresolved", "bib/metadata-drift"} <= set(chosen.skipped)


def test_the_default_plan_does_want_the_network():
    assert plan(REG, has_repo=False).opens_network


def test_repo_slices_come_from_runnable_rules_not_every_rule():
    """A disabled repo rule must not cause its slices to be walked."""
    with_repo = plan(REG, has_repo=True)
    assert with_repo.repo_slices

    cfg = parse_config("rules:\n  repro/seed-claim: off\n")
    without = plan(REG, cfg, has_repo=True)
    assert "repo.seeds" not in without.repo_slices
    assert "repo.seeds" in with_repo.repo_slices


def test_no_repo_means_no_repo_slices_at_all():
    chosen = plan(REG, has_repo=False)
    assert chosen.repo_slices == set()
    # Every repo rule is skipped, and for the repository reason specifically --
    # model rules are skipped too, for a different reason of their own.
    assert {
        rule_id
        for rule_id, why in chosen.skipped.items()
        if why == "no repository supplied"
    } == {r.id for r in REG.all() if r.needs_repo}


def test_model_rules_are_skipped_without_a_provider():
    chosen = plan(REG, has_repo=True, has_provider=False)
    assert all(r.tier is not Tier.MODEL_ASSISTED for r in chosen.runnable)


def test_the_plan_drives_the_run_so_selection_cannot_drift():
    """One decision, used by both the loader and the runner."""
    cfg = parse_config("rules:\n  stats/grim: off\n")
    chosen = plan(REG, cfg, has_repo=False)
    paper = paper_from_path(PLANTED, needs=chosen.paper_slices)

    report = run(paper, registry=REG, config=cfg, prepared=chosen)
    assert "stats/grim" not in report.ran
    assert report.skipped["stats/grim"] == "disabled in .resint.yml"


def test_a_narrow_plan_loads_a_narrow_paper():
    cfg = parse_config(
        "rules:\n"
        + "".join(
            f"  {r.id}: off\n" for r in REG.all() if not r.id.startswith("stats/grim")
        )
    )
    chosen = plan(REG, cfg, has_repo=False)
    assert chosen.paper_slices == {"paper.means"}

    paper = paper_from_path(PLANTED, needs=chosen.paper_slices)
    assert paper.bib == [], "bibliography parsed when nothing asked for it"
    assert paper.tables == [], "tables parsed when nothing asked for it"
    assert paper.stats == [], "statistics parsed when nothing asked for it"


def test_no_socket_is_opened_when_no_rule_needs_one():
    """The property the docs already advertise, now enforced."""

    class _Tripwire:
        touched = False

        def resolve(self, entry):
            _Tripwire.touched = True
            raise AssertionError("a socket was opened for a run that needs none")

    cfg = parse_config(
        "rules:\n  bib/unresolved: off\n  bib/metadata-drift: off\n"
    )
    chosen = plan(REG, cfg, has_repo=False)
    paper_from_path(PLANTED, needs=chosen.paper_slices, resolver=_Tripwire())
    assert not _Tripwire.touched


# --- determinism ---------------------------------------------------------
#
# The precondition for comparing two sweeps. Any nondeterminism -- a set
# iterated into output, a dict ordered by hash -- shows up as phantom churn
# on every paper and makes the diff worthless.


def _json_of(target, **kw):
    import json

    from resint.engine import run as _run

    chosen = plan(REG, has_repo=False)
    paper = paper_from_path(target, needs=chosen.paper_slices, **kw)
    report = _run(paper, registry=REG, prepared=chosen)
    return json.dumps(
        {
            "findings": [f.to_dict() for f in report.findings],
            "unchecked": report.unchecked,
            "notes": report.notes,
            "skipped": report.skipped,
            "ran": report.ran,
        },
        indent=2,
    )


@pytest.mark.parametrize("name", ["planted", "clean"])
def test_two_runs_produce_byte_identical_output(name, corpus_resolver):
    target = CORPUS / name / "paper.tex"
    first = _json_of(target, resolver=corpus_resolver)
    second = _json_of(target, resolver=corpus_resolver)
    assert first == second


def test_unused_suppression_notes_are_ordered(corpus_resolver):
    """Set iteration here made every diff show churn that was not real."""
    cfg = parse_config(
        'suppress:\n'
        '  - rule: aaa/one\n    reason: "r"\n'
        '  - rule: zzz/two\n    reason: "r"\n'
        '  - rule: mmm/three\n    reason: "r"\n'
    )
    _, notes = cfg.apply([])
    rules = [n.split()[2] for n in notes]
    assert rules == sorted(rules)
