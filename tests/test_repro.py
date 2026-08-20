"""Repository reading and the repro/ rule family.

repro/hparam-drift is the rule with the most room to do damage -- it tells an
author they misreported their own experiment. Most of the tests here are
about the cases where it must refuse to speak.
"""

from pathlib import Path

import pytest

from resint.engine import run
from resint.ir.repo import Binding, ConfigKey, ConfigSet, Repo, canonical
from resint.ir.span import Source, Span
from resint.parse.code import read_python
from resint.parse.configs import read_json, read_yaml
from resint.parse.document import paper_from_path
from resint.parse.repo import read_repo
from resint.rules import load_all
from resint.rules.registry import Context

CORPUS = Path(__file__).resolve().parents[1] / "corpus"
PLANTED = CORPUS / "planted"
CLEAN = CORPUS / "clean"
SRC = Source("train.py", "code", path="train.py")
REG = load_all()
REPO_NEEDS = {q for r in REG.all() for q in r.requires if q.startswith("repo.")}


@pytest.fixture(scope="module")
def planted_repo():
    return read_repo(PLANTED / "repo", needs=REPO_NEEDS)


@pytest.fixture(scope="module")
def planted_report(corpus_resolver, planted_repo):
    paper = paper_from_path(PLANTED / "paper.tex", resolver=corpus_resolver)
    return run(paper, repo=planted_repo)


@pytest.fixture(scope="module")
def clean_report(corpus_resolver):
    paper = paper_from_path(CLEAN / "paper.tex", resolver=corpus_resolver)
    repo = read_repo(CLEAN / "repo", needs=REPO_NEEDS)
    return run(paper, repo=repo)


def key(name, value, binding=Binding.ARGPARSE, origin="x"):
    return ConfigKey(
        name=name,
        raw_name=name,
        value=value,
        binding=binding,
        span=Span(SRC, 0, 4, line=1),
        origin=origin,
    )


# --- python reading -----------------------------------------------------


def test_argparse_defaults_are_read():
    facts = read_python(
        "import argparse\n"
        "p = argparse.ArgumentParser()\n"
        "p.add_argument('--learning-rate', type=float, default=3e-4)\n",
        SRC,
        "train.py",
    )
    assert [(c.raw_name, c.value) for c in facts.configs] == [("learning-rate", "3e-4")]
    assert facts.configs[0].binding is Binding.ARGPARSE


def test_a_computed_default_is_reported_not_guessed():
    """A number that only exists at runtime must not enter the diff table."""
    facts = read_python(
        "p.add_argument('--workers', default=len(devices))\n", SRC, "train.py"
    )
    assert facts.configs == []
    assert any("computed at runtime" in u for u in facts.unchecked)


def test_arguments_without_a_default_are_ignored():
    facts = read_python("p.add_argument('--name', type=str)\n", SRC, "train.py")
    assert facts.configs == []


@pytest.mark.parametrize(
    "source, library",
    [
        ("import random\nrandom.seed(42)", "random"),
        ("import numpy as np\nnp.random.seed(42)", "np"),
        ("import torch\ntorch.manual_seed(42)", "torch"),
        ("from lightning import seed_everything\nseed_everything(42)", "lightning"),
    ],
)
def test_seed_calls_are_recognised(source, library):
    facts = read_python(source, SRC, "train.py")
    assert len(facts.seeds) == 1
    assert facts.seeds[0].library == library
    assert facts.seeds[0].argument == "42"
    assert not facts.seeds[0].varies


def test_a_seed_from_a_variable_counts_as_varying():
    facts = read_python("random.seed(args.seed)", SRC, "train.py")
    assert facts.seeds[0].varies


def test_a_seed_inside_a_loop_counts_as_varying():
    facts = read_python(
        "for s in range(5):\n    random.seed(s)\n", SRC, "train.py"
    )
    assert facts.seeds[0].in_loop and facts.seeds[0].varies


def test_symbols_are_indexed():
    facts = read_python("def train():\n    pass\n\nclass Model:\n    pass\n", SRC, "train.py")
    assert {(s.name, s.kind) for s in facts.symbols} == {
        ("train", "function"),
        ("Model", "class"),
    }


def test_a_syntax_error_is_reported_not_raised():
    facts = read_python("def broken(:\n", SRC, "bad.py")
    assert facts.configs == []
    assert any("not parsed" in u for u in facts.unchecked)


# --- config files -------------------------------------------------------


def test_yaml_flattens_nested_keys():
    parsed = read_yaml(
        "learning_rate: 1e-4\nmodel:\n  hidden_size: 768\n",
        SRC,
        "base.yaml",
        Binding.CONFIG_FILE,
    )
    names = {k.name: k.value for k in parsed.keys}
    assert names["learning_rate"] == "1e-4"
    assert names["model.hidden_size"] == "768"


def test_yaml_comments_are_stripped_but_quoted_hashes_survive():
    parsed = read_yaml(
        'lr: 1e-4  # the good one\nname: "a # b"\n', SRC, "c.yaml", Binding.CONFIG_FILE
    )
    values = {k.raw_name: k.value for k in parsed.keys}
    assert values["lr"] == "1e-4"
    assert values["name"] == "a # b"


def test_unsupported_yaml_is_reported_not_half_parsed():
    parsed = read_yaml("<<: *base\nlr: 1e-4\n", SRC, "c.yaml", Binding.CONFIG_FILE)
    assert any("not supported" in u for u in parsed.unchecked)


def test_hydra_defaults_list_records_composition():
    parsed = read_yaml(
        "defaults:\n  - base\n  - model: large\nlr: 1e-4\n",
        SRC,
        "exp.yaml",
        Binding.CONFIG_FILE,
    )
    assert "base" in parsed.composes
    assert [k.raw_name for k in parsed.keys] == ["lr"]


def test_json_configs_are_read():
    parsed = read_json('{"lr": 0.0003, "nested": {"depth": 12}}', SRC, "c.json", Binding.CONFIG_FILE)
    values = {k.name: k.value for k in parsed.keys}
    assert values["lr"] == "0.0003" and values["nested.depth"] == "12"


def test_invalid_json_is_reported():
    parsed = read_json("{not json", SRC, "c.json", Binding.CONFIG_FILE)
    assert parsed.keys == [] and parsed.unchecked


# --- precedence ---------------------------------------------------------


@pytest.mark.parametrize(
    "written, expected",
    [
        ("lr", "learning_rate"),
        ("learning rate", "learning_rate"),
        ("Learning-Rate", "learning_rate"),
        ("batch size", "batch_size"),
        ("n_epochs", "epochs"),
        ("attention heads", "num_heads"),
        ("unmapped_thing", "unmapped_thing"),
    ],
)
def test_alias_normalisation(written, expected):
    assert canonical(written) == expected


def test_a_stronger_binding_wins():
    configs = ConfigSet([
        key("lr", "3e-4", Binding.ARGPARSE, "argparse"),
        key("lr", "1e-4", Binding.CONFIG_FILE, "base.yaml"),
    ])
    effective, candidates = configs.effective("learning rate")
    assert effective.value == "1e-4"
    assert len(candidates) == 2


def test_equal_bindings_that_disagree_produce_no_effective_value():
    """Precedence cannot be established, so the rule must not guess."""
    configs = ConfigSet([
        key("lr", "1e-4", Binding.CONFIG_FILE, "a.yaml"),
        key("lr", "5e-5", Binding.CONFIG_FILE, "b.yaml"),
    ])
    effective, candidates = configs.effective("lr")
    assert effective is None and len(candidates) == 2


def test_equal_bindings_that_agree_are_fine():
    configs = ConfigSet([
        key("lr", "1e-4", Binding.CONFIG_FILE, "a.yaml"),
        key("lr", "1e-4", Binding.CONFIG_FILE, "b.yaml"),
    ])
    effective, _ = configs.effective("lr")
    assert effective.value == "1e-4"


def test_an_unknown_name_yields_nothing():
    assert ConfigSet([key("lr", "1e-4")]).effective("temperature") == (None, [])


# --- repro/hparam-drift -------------------------------------------------


def paper_repo(hparams, configs):
    from resint.ir.paper import Number, Paper

    paper = Paper(source_id="paper.tex")
    paper.hyperparameters = [
        Number(raw=raw, label=label, span=Span(SRC, 0, len(raw), line=1))
        for label, raw in hparams
    ]
    repo = Repo(root=".", configs=ConfigSet(configs))
    return paper, repo


def drift(hparams, configs):
    """Run the rule and hand back the context, which collects abstentions."""
    paper, repo = paper_repo(hparams, configs)
    ctx = Context(paper=paper, repo=repo)
    return REG.get("repro/hparam-drift").run(ctx), ctx


def test_a_genuine_mismatch_is_reported():
    findings, _ = drift([("learning rate", "3e-4")], [key("lr", "1e-4", Binding.CONFIG_FILE)])
    assert len(findings) == 1
    assert "3e-4" in findings[0].message and "1e-4" in findings[0].message


@pytest.mark.parametrize(
    "paper_value, code_value",
    [("3e-4", "0.0003"), ("0.0003", "3e-4"), ("128", "128.0"), ("1e-4", "0.0001")],
)
def test_the_same_number_written_differently_is_not_a_mismatch(paper_value, code_value):
    findings, _ = drift(
        [("learning rate", paper_value)], [key("lr", code_value, Binding.CONFIG_FILE)]
    )
    assert findings == []


def test_ambiguous_precedence_abstains_and_records_why():
    """Silence without a reason is indistinguishable from a pass."""
    findings, ctx = drift(
        [("learning rate", "3e-4")],
        [
            key("lr", "1e-4", Binding.CONFIG_FILE, "a.yaml"),
            key("lr", "5e-5", Binding.CONFIG_FILE, "b.yaml"),
        ],
    )
    assert findings == []
    assert any("equally binding values" in a for a in ctx.abstentions)
    assert all(a.startswith("repro/hparam-drift:") for a in ctx.abstentions)


def test_abstention_reaches_the_report(corpus_resolver):
    from resint.ir.paper import Number, Paper

    paper = paper_from_path(PLANTED / "paper.tex", resolver=corpus_resolver)
    repo = Repo(
        root=".",
        configs=ConfigSet(
            [
                key("lr", "1e-4", Binding.CONFIG_FILE, "a.yaml"),
                key("lr", "5e-5", Binding.CONFIG_FILE, "b.yaml"),
            ]
        ),
    )
    report = run(paper, repo=repo)
    assert any("equally binding values" in u for u in report.unchecked)


def test_ctx_abstain_outside_a_rule_is_an_error():
    from resint.rules.registry import RuleDefinitionError

    with pytest.raises(RuleDefinitionError, match="only available inside a rule"):
        Context().abstain("nope")


def test_a_hyperparameter_absent_from_the_repo_is_silent():
    findings, _ = drift([("temperature", "0.07")], [key("lr", "1e-4")])
    assert findings == []


def test_a_hundredfold_ratio_reads_as_a_unit_convention():
    """Percent versus fraction is a convention, not a misreport."""
    findings, _ = drift([("dropout", "10")], [key("dropout", "0.1", Binding.CONFIG_FILE)])
    assert len(findings) == 1
    assert findings[0].severity.value == "low"
    assert "percentage" in findings[0].message


def test_the_resolution_trail_is_shown():
    findings, _ = drift(
        [("learning rate", "3e-4")],
        [
            key("lr", "3e-4", Binding.ARGPARSE, "argparse default in train.py"),
            key("lr", "1e-4", Binding.CONFIG_FILE, "configs/base.yaml"),
        ],
    )
    assert "Resolved through" in findings[0].message
    assert "argparse default in train.py" in findings[0].message


# --- corpus behaviour ---------------------------------------------------


def test_planted_repo_yields_every_repro_finding(planted_report):
    fired = {f.rule_id for f in planted_report.findings if f.rule_id.startswith("repro/")}
    assert fired == {
        "repro/hparam-drift",
        "repro/seed-claim",
        "repro/entrypoint-missing",
        "repro/unpinned-deps",
    }


def test_seed_claim_names_the_consequence(planted_report):
    seed = next(f for f in planted_report.findings if f.rule_id == "repro/seed-claim")
    assert "5 runs" in seed.message and "42" in seed.message
    assert "error bar" in seed.affects[0]


def test_entrypoint_finding_names_the_missing_file(planted_report):
    entry = next(
        f for f in planted_report.findings if f.rule_id == "repro/entrypoint-missing"
    )
    assert "evaluate.py" in entry.message
    assert entry.absent_from == "the repository tree"


def test_computed_default_reaches_the_report(planted_report):
    assert any("computed at runtime" in u for u in planted_report.unchecked)


def test_clean_repository_produces_nothing(clean_report):
    assert clean_report.findings == [], [
        f"{f.rule_id}: {f.message}" for f in clean_report.findings
    ]


def test_seed_from_an_argument_does_not_fire_even_with_a_five_run_claim(clean_report):
    """The clean paper claims five seeds; the code reads its seed from argv."""
    assert not any(f.rule_id == "repro/seed-claim" for f in clean_report.findings)


def test_pinned_dependencies_do_not_fire(clean_report):
    assert not any(f.rule_id == "repro/unpinned-deps" for f in clean_report.findings)


# --- walking ------------------------------------------------------------


def test_repo_reads_only_the_declared_slices():
    repo = read_repo(PLANTED / "repo", needs={"repo.deps"})
    assert repo.deps and repo.configs == [] and repo.seeds == []


def test_missing_directory_is_reported_not_raised():
    repo = read_repo(PLANTED / "does-not-exist", needs=REPO_NEEDS)
    assert repo.files == []
    assert any("not a directory" in u for u in repo.unchecked)


def test_ghost_repo_fires_on_a_readme_only_tree(tmp_path):
    (tmp_path / "README.md").write_text(
        "# Method\n\nCode will be released upon acceptance.\n", encoding="utf-8"
    )
    repo = read_repo(tmp_path, needs=REPO_NEEDS)
    from resint.ir.paper import Paper

    findings = REG.get("repro/ghost-repo").run(
        Context(paper=Paper(source_id="p.tex"), repo=repo)
    )
    assert len(findings) == 1
    assert "no code" in findings[0].message
    assert findings[0].severity.value == "high"


def test_ghost_repo_is_silent_when_code_exists(planted_repo):
    from resint.ir.paper import Paper

    findings = REG.get("repro/ghost-repo").run(
        Context(paper=Paper(source_id="p.tex"), repo=planted_repo)
    )
    assert findings == []
