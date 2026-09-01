"""A known-positive for every rule.

Across 352 real papers, eight rules executed and never fired once. That is not
evidence either way: a rule may be correct and rare -- GRIM violations *are*
rare, which is the point of the rule -- or it may be silently broken. More
corpus cannot separate those. It gives you a larger zero.

Only a planted case can: a document constructed so the rule **must** fire.
Together with the real corpus, which measures false positives, this measures
the other half.

Each case is a minimal reproduction, deliberately not a realistic paper. The
point is one defect, isolated, so a failure here names the rule that broke
rather than sending someone to bisect a 400-line fixture.
"""

import pytest

from resint.ir.paper import Paper
from resint.ir.repo import ConfigKey, ConfigSet, Dependency, Link, Repo, SeedCall, Symbol
from resint.ir.span import Source, Span
from resint.model.base import Completion, Outcome
from resint.parse.document import paper_from_latex
from resint.resolve import Registration, Record, Resolution, Status
from resint.rules import load_all
from resint.rules.registry import Context

REG = load_all()
RSRC = Source("train.py", "python", path="train.py")



def _span(path="train.py", line=1):
    return Span(Source(path, "python", path=path), 0, 5, line)


def paper(body, **needs):
    return paper_from_latex(
        "\\documentclass{article}\\begin{document}\n" + body + "\n\\end{document}\n",
        needs=set(needs.get("needs", ())) or None,
    )


class Answers:
    """A model that says exactly what a case requires."""

    model = "planted"

    def __init__(self, payload):
        self.payload = payload

    def complete(self, request):
        return Completion(Outcome.ANSWERED, payload=self.payload, model="planted")


def run_rule(rule_id, paper_obj, repo=None, model=None):
    ctx = Context(paper=paper_obj, repo=repo, model=model)
    return REG.get(rule_id).run(ctx)


# =========================================================================
# numbers/
# =========================================================================


def test_numbers_internal_mismatch():
    from resint.ir.paper import Number

    p = paper(
        "\\section{Results}\nWe reach accuracy of 94.2 on the benchmark.\n"
        "\\begin{tabular}{lc}\nMethod & Accuracy \\\\\n"
        "Baseline & 91.4 \\\\\nOurs & 93.8 \\\\\n\\end{tabular}\n"
    )
    assert len(run_rule("numbers/internal-mismatch", p)) == 1


def test_numbers_table_arithmetic():
    p = paper(
        "\\begin{tabular}{lr}\nSplit & Count \\\\\n"
        "Train & 800 \\\\\nVal & 100 \\\\\nTest & 100 \\\\\n"
        "Total & 1200 \\\\\n\\end{tabular}\n"
    )
    findings = run_rule("numbers/table-arithmetic", p)
    assert len(findings) == 1
    assert "1200" in findings[0].message


# =========================================================================
# stats/
# =========================================================================


def test_stats_grim():
    """3.47 is unreachable from 20 integer responses."""
    p = paper("With N = 20 participants the mean was 3.47 on the scale.")
    findings = run_rule("stats/grim", p)
    assert len(findings) == 1
    assert "3.45" in findings[0].message


def test_stats_pvalue_mismatch():
    """t(20) = 1.20 gives p ~ .244, not .03 -- and that crosses alpha."""
    p = paper("The effect held, t(20) = 1.20, p = .03.")
    findings = run_rule("stats/pvalue-mismatch", p)
    assert len(findings) == 1
    assert findings[0].severity.value == "high", "a decision error is high"


def test_stats_significance_unsupported():
    p = paper(
        "\\section{Results}\nThe improvement is statistically significant "
        "and the difference between conditions is reliable.\n"
    )
    assert len(run_rule("stats/significance-unsupported", p)) >= 1


# =========================================================================
# bib/
# =========================================================================


def _bib_paper(entries, resolutions, citations=()):
    p = Paper(source_id="paper.tex")
    p.bib = list(entries)
    p.resolutions = dict(resolutions)
    p.citations = list(citations)
    return p


def _entry(key, **fields):
    from resint.ir.paper import BibEntry

    bib = Source("refs.bib", "bib", path="refs.bib")
    return BibEntry(
        key=key,
        entry_type=fields.pop("entry_type", "article"),
        fields=fields,
        span=Span(bib, 0, 200, line=1, label=f"[{key}]"),
        field_spans={
            name: Span(bib, i * 10, i * 10 + 5) for i, name in enumerate(fields, 1)
        },
    )


def test_bib_unresolved():
    """A DOI that resolves nowhere. The fabrication signal."""
    e = _entry("ghost2020", title="A Paper", doi="10.5555/nope", year="2020")
    findings = run_rule(
        "bib/unresolved",
        _bib_paper([e], {"ghost2020": _dead()}),
    )
    assert len(findings) == 1
    assert findings[0].severity.value == "high"


def test_bib_unresolved_clears_a_live_doi_our_indices_cannot_see():
    """The planted *negative*, and the more valuable of the pair.

    A DOI registered outside Crossref -- through the Chinese agency, JaLC,
    KISTI -- resolves perfectly well and is invisible to all four of our
    indices. Until 2026-09-01 this rule reported exactly that as fabrication,
    at high severity, which made it fire on papers citing Chinese-language
    literature. A known-positive cannot catch a bias; only a known-negative
    that must stay silent can, which is why this fixture is permanent.
    """
    e = _entry(
        "chinese2025",
        title="2024 CHINET Surveillance of Bacterial Resistance in China",
        doi="10.16718/j.1009-7708.2025.06.002",
        year="2025",
    )
    resolution = Resolution(
        Status.NOT_FOUND,
        queried=("crossref", "openalex", "arxiv", "dblp", "doi.org"),
        registration=Registration.REGISTERED,
        agency="Chinese Academy of Sciences",
    )
    assert run_rule("bib/unresolved", _bib_paper([e], {"chinese2025": resolution})) == []


def test_bib_unresolved_names_the_coverage_gap_it_hit():
    """Silence about a live DOI we could not read would hide the blind spot.
    The gaps are not random -- they fall on whole literatures -- so the rule
    says which agency it could not follow."""
    from resint.rules.registry import Context

    e = _entry("chinese2025", title="A Paper", doi="10.16718/j.x", year="2025")
    paper = _bib_paper(
        [e],
        {"chinese2025": Resolution(
            Status.NOT_FOUND,
            queried=("crossref", "doi.org"),
            registration=Registration.REGISTERED,
            agency="Chinese Academy of Sciences",
        )},
    )
    ctx = Context(paper=paper, rule=REG.get("bib/unresolved"))
    list(REG.get("bib/unresolved").fn(ctx))
    assert any("Chinese Academy of Sciences" in a for a in ctx.abstentions)


def test_bib_doi_mismatch():
    """The DOI resolves -- to somebody else's paper."""
    e = _entry(
        "wrong2017",
        title="Attention Is All You Need",
        doi="10.5555/12345",
        author="Vaswani, Ashish",
        year="2017",
    )
    record = Record(
        source="crossref",
        title="Deep Residual Learning for Image Recognition",
        authors=("Kaiming He",),
        matched_by="doi",
    )
    findings = run_rule(
        "bib/doi-mismatch",
        _bib_paper([e], {"wrong2017": Resolution(Status.FOUND, record=record)}),
    )
    assert len(findings) == 1
    assert findings[0].severity.value == "high"


def test_bib_metadata_drift():
    e = _entry("stale2016", title="A Paper", doi="10.1/x", year="2016")
    record = Record(
        source="crossref", title="A Paper", year="2018", matched_by="doi"
    )
    findings = run_rule(
        "bib/metadata-drift",
        _bib_paper([e], {"stale2016": Resolution(Status.FOUND, record=record)}),
    )
    assert len(findings) == 1
    assert "2018" in findings[0].message


def test_bib_orphans():
    from resint.ir.paper import Citation

    bib = Source("refs.bib", "bib", path="refs.bib")
    e = _entry("never2020", title="Uncited Work", year="2020")
    cited = Citation(key="missing2021", span=Span(bib, 0, 5, 1))
    findings = run_rule("bib/orphans", _bib_paper([e], {}, [cited]))
    assert len(findings) == 2, "one cited-but-undefined, one defined-but-uncited"


def test_bib_unindexed():
    """Off by default, so this calls it directly."""
    e = _entry("obscure2019", title="Some Untraceable Paper", year="2019")
    findings = run_rule(
        "bib/unindexed",
        _bib_paper(
            [e], {"obscure2019": _dead()}
        ),
    )
    assert len(findings) == 1
    assert findings[0].severity.value == "low"


# =========================================================================
# repro/
# =========================================================================


def test_repro_ghost_repo():
    """A repository holding only a promise."""
    repo = Repo(
        root=".",
        files=["README.md"],
        readme="Code will be released upon acceptance.",
        readme_source=Source("README.md", "markdown", path="README.md"),
    )
    findings = run_rule("repro/ghost-repo", Paper(source_id="p.tex"), repo=repo)
    assert len(findings) == 1


def test_repro_unpinned_deps():
    repo = Repo(
        root=".",
        files=["requirements.txt"],
        deps=[
            Dependency(name=n, constraint="", pinned=False, span=_span(), manifest="requirements.txt")
            for n in ("torch", "numpy", "scipy")
        ],
    )
    findings = run_rule("repro/unpinned-deps", Paper(source_id="p.tex"), repo=repo)
    assert len(findings) == 1
    assert "torch" in findings[0].message


def test_repro_seed_claim():
    p = paper("\\section{Results}\nWe report results averaged over 5 random seeds.")
    repo = Repo(
        root=".",
        files=["train.py"],
        seeds=[SeedCall(library="torch", argument="42", span=_span())],
    )
    findings = run_rule("repro/seed-claim", p, repo=repo)
    assert len(findings) == 1
    assert "42" in findings[0].message


def test_repro_entrypoint_missing():
    """The README documents a command whose target is not in the tree."""
    from resint.parse.repo import Entrypoint

    repo = Repo(
        root=".",
        files=["README.md"],
        readme="Run `python train.py`.",
        entrypoints=[
            Entrypoint(
                command="python train.py",
                target="train.py",
                span=_span("README.md"),
                exists=False,
            )
        ],
    )
    findings = run_rule("repro/entrypoint-missing", Paper(source_id="p.tex"), repo=repo)
    assert len(findings) == 1
    assert "train.py" in findings[0].message


def test_repro_hparam_drift():
    p = paper("\\section{Method}\nWe train with a learning rate of 3e-4.")
    repo = Repo(
        root=".",
        files=["configs/base.yaml"],
        configs=ConfigSet(
            [
                ConfigKey(
                    name="learning_rate",
                    raw_name="learning_rate",
                    value="1e-4",
                    binding=1,
                    span=_span("configs/base.yaml"),
                    origin="configs/base.yaml",
                )
            ]
        ),
    )
    findings = run_rule("repro/hparam-drift", p, repo=repo)
    assert len(findings) == 1
    assert "3e-4" in findings[0].message and "1e-4" in findings[0].message


# =========================================================================
# claim/ and eval/ -- the model tier
# =========================================================================

MODEL_PAPER = (
    "\\begin{abstract}\nOur general-purpose method applies across diverse "
    "domains and achieves improved calibration under distribution shift.\n"
    "\\end{abstract}\n"
    "\\section{Methods}\nWe train our model for 200 epochs on the full split.\n"
    "The baseline was trained for 50 epochs following its original recipe.\n"
    "\\section{Results}\n"
    "Our method significantly outperforms the strongest baseline on CIFAR-10.\n"
    "We evaluate on CIFAR-10 and CIFAR-100 under the standard protocol.\n"
    "Our framework supports distributed training across multiple nodes.\n"
    + "Section prose describing the protocol at length. " * 120
    + "\n\\begin{tabular}{lc}\nMethod & Accuracy \\\\\n"
    "Baseline & 93.9 \\\\\nOurs & 94.2 \\\\\n\\end{tabular}\n"
)

SURVEY = {
    "comparisons": [
        {
            "claim": "Our method significantly outperforms the strongest baseline on CIFAR-10.",
            "ours": "94.2",
            "baseline": "93.9",
            "metric": "accuracy",
        }
    ],
    "budgets": [
        {
            "dimension": "training epochs",
            "ours": "We train our model for 200 epochs on the full split.",
            "baseline": "The baseline was trained for 50 epochs following its original recipe.",
        }
    ],
    "scope_claims": [
        "Our general-purpose method applies across diverse domains and "
        "achieves improved calibration under distribution shift."
    ],
    "evaluated_on": ["CIFAR-10", "CIFAR-100"],
    "abstract_claims": [
        {
            "claim": "Our general-purpose method applies across diverse domains and "
            "achieves improved calibration under distribution shift.",
            "about": "improved calibration",
            "terms": ["calibration", "distribution"],
        }
    ],
    "capabilities": [
        {
            "claim": "Our framework supports distributed training across multiple nodes.",
            "capability": "distributed training",
            "terms": ["distributed", "allreduce"],
        }
    ],
}


@pytest.mark.parametrize(
    "rule_id, needs_repo",
    [
        ("claim/overreach", False),
        ("claim/scope-creep", False),
        ("claim/unsupported", False),
        ("eval/baseline-fairness", False),
        ("claim/unimplemented", True),
    ],
)
def test_model_rules_fire_on_a_planted_case(rule_id, needs_repo):
    """These four produced 2 findings across 71 real papers. That says nothing
    about whether they work -- a planted case does."""
    p = paper(MODEL_PAPER)
    repo = Repo(root=".", files=["train.py"], symbols=[
        Symbol(name="train_one_epoch", kind="function", span=_span())
    ]) if needs_repo else None

    findings = run_rule(rule_id, p, repo=repo, model=Answers(SURVEY))
    assert len(findings) >= 1, f"{rule_id} did not fire on a case built for it"


# =========================================================================
# coverage: nothing may quietly lack a known-positive
# =========================================================================


def test_every_rule_has_a_planted_case():
    """The guard on this file. A rule added without a known-positive is a rule
    whose silence on the real corpus means nothing at all."""
    import pathlib

    source = pathlib.Path(__file__).read_text(encoding="utf-8")
    missing = [
        r.id
        for r in REG.all()
        # bib/citation-support needs a fetched cited paper; it is exercised in
        # tests/test_citation_support.py against a StaticFullText instead.
        if r.id != "bib/citation-support"
        and r.id.replace("/", "_").replace("-", "_") not in source
        and f'"{r.id}"' not in source
    ]
    assert not missing, f"no planted case for: {missing}"


# =========================================================================
# the loader's default
# =========================================================================


def test_read_repo_without_needs_loads_everything():
    """It used to load *nothing*. read_repo(root) returned a Repo with every
    slice empty, so all five repro rules looked at an empty world and stayed
    quiet -- indistinguishable from a clean repository.

    Silence is what hid it. The coverage census on hparam-drift is what showed
    it: "2 hyperparameters named, 0 located" on a fixture built so both must
    be found.
    """
    from resint.parse.repo import read_repo

    repo = read_repo("corpus/planted/repo")
    assert repo.configs, "configs"
    assert repo.seeds, "seeds"
    assert repo.deps, "dependencies"
    assert repo.entrypoints, "entrypoints"


def test_the_planted_repo_exercises_every_repro_rule():
    """The guard the last one lacked: a count, not a vibe."""
    from resint.engine import run
    from resint.parse.document import paper_from_path
    from resint.parse.repo import read_repo

    report = run(
        paper_from_path("corpus/planted/paper.tex"),
        repo=read_repo("corpus/planted/repo"),
        registry=REG,
    )
    fired = {f.rule_id for f in report.findings if f.rule_id.startswith("repro/")}
    assert len(fired) >= 4, f"only {sorted(fired)} fired"


def test_hparam_drift_reports_coverage_even_with_no_findings():
    """A fire rate is the wrong measure for a coverage check. This rule is
    useful at zero findings, and saying what it looked at is how."""
    from resint.engine import run
    from resint.parse.document import paper_from_path
    from resint.parse.repo import read_repo

    report = run(
        paper_from_path("corpus/planted/paper.tex"),
        repo=read_repo("corpus/planted/repo"),
        registry=REG,
    )
    census = [u for u in report.unchecked if "hyperparameters named" in u]
    assert census, "the rule must say what it examined"
    assert "located in the repository" in census[0]


def test_the_sweep_pool_does_not_retire_workers():
    """max_tasks_per_child deadlocks the pool on this platform.

    Reproduced deliberately: five tasks per child, one worker, twelve trivial
    tasks -- the run hangs before completing the fifth, the retired worker is
    never replaced, and the parent blocks in wait() forever. Pool shutdown
    never returns either.

    That stalled a sweep at exactly 25 completions and probably truncated an
    earlier batch by three papers. A guard rather than a comment, because the
    setting looks harmless and reads like good hygiene.
    """
    import pathlib

    source = pathlib.Path("tools/sweep.py").read_text(encoding="utf-8")
    body = source.split("with ProcessPoolExecutor", 1)[1].split(")", 1)[0]
    assert "max_tasks_per_child" not in body


# =========================================================================
# coverage censuses: a rule that finds nothing must still say what it looked at
# =========================================================================


def test_bib_unresolved_reports_what_it_examined():
    """Silence is indistinguishable from not having run. That is exactly how
    read_repo() hid an empty world, and how a truncated batch went unnoticed."""
    from resint.rules.registry import Context

    e = _entry("k", title="A Paper", doi="10.1/x", year="2020")
    paper = _bib_paper([e], {"k": Resolution(Status.FOUND, record=Record(source="crossref"))})
    ctx = Context(paper=paper, rule=REG.get("bib/unresolved"))
    list(REG.get("bib/unresolved").fn(ctx))

    census = [a for a in ctx.abstentions if "reference" in a]
    assert census, "the rule must say how many references it looked at"
    assert "1 with a DOI" in census[0]


def test_stats_pvalue_reports_what_it_examined():
    from resint.rules.registry import Context

    p = paper("Reliable, t(20) = 2.086, p = .05.")
    ctx = Context(paper=p, rule=REG.get("stats/pvalue-mismatch"))
    list(REG.get("stats/pvalue-mismatch").fn(ctx))

    census = [a for a in ctx.abstentions if "test statistic" in a]
    assert census
    assert "1 recomputed" in census[0]


def test_a_census_is_emitted_even_when_nothing_is_wrong():
    """The whole point: useful at zero findings."""
    from resint.rules.registry import Context

    p = paper("Reliable, t(20) = 2.086, p = .05.")
    findings = run_rule("stats/pvalue-mismatch", p)
    assert findings == [], "this statistic and p agree"

    ctx = Context(paper=p, rule=REG.get("stats/pvalue-mismatch"))
    list(REG.get("stats/pvalue-mismatch").fn(ctx))
    assert any("test statistic" in a for a in ctx.abstentions)


def _dead():
    """A DOI the DOI system itself denies -- the only thing that may fire
    bib/unresolved. Missing from every index is not enough, and a fixture
    that cannot express the difference is how the old premise survived."""
    return Resolution(
        Status.NOT_FOUND,
        queried=("crossref", "doi.org"),
        registration=Registration.DEAD,
    )
