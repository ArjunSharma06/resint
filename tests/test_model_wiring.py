"""Reaching the model tier from the command line, and what it costs to.

Two properties matter here and neither is about any individual rule.

**A key never comes from the config file.** ``.resint.yml`` is committed
alongside the paper; a key in it is a key on GitHub. The file names a provider,
the environment holds the secret, and there is no code path that reads one from
the other.

**A model finding never fails a build by default.** A model rule misfiring in
someone's pipeline damages trust in the deterministic rules too, and those are
the asset. The finding is still reported, still in the JSON, still counted --
it just does not turn the build red on a judgement call unless asked.
"""

from pathlib import Path

import pytest

from resint.cli import main
from resint.config import parse as parse_config
from resint.engine import plan, run
from resint.ir.finding import Finding, Severity, Tier
from resint.ir.span import Source, Span
from resint.model.base import Completion, Outcome
from resint.model.openai_compat import from_config
from resint.parse.document import paper_from_latex
from resint.rules import load_all

CORPUS = Path(__file__).resolve().parents[1] / "corpus"
PLANTED = CORPUS / "planted" / "paper.tex"
REG = load_all()


# --- the config names a provider, never a key ---------------------------


def test_a_provider_is_built_from_the_config():
    cfg = parse_config("model:\n  provider: openai\n  name: gpt-4o-mini\n")
    provider = from_config(cfg.model)
    assert provider is not None
    assert provider.model == "gpt-4o-mini"


def test_no_model_section_means_no_provider():
    assert from_config(parse_config("rules:\n  stats/grim: off\n").model) is None


def test_a_half_configured_model_is_no_provider():
    """A provider with no model name would fail at the first call with a
    message about the API rather than about the config."""
    assert from_config(parse_config("model:\n  provider: openai\n").model) is None
    assert from_config(parse_config("model:\n  name: gpt-4o\n").model) is None


def test_a_key_in_the_config_is_never_read(monkeypatch):
    """The property this file exists for. Someone will put a key here; it must
    do nothing rather than work and then leak."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg = parse_config(
        "model:\n"
        "  provider: openai\n"
        "  name: gpt-4o-mini\n"
        "  api_key: sk-do-not-use-me\n"
    )
    provider = from_config(cfg.model)
    assert provider.api_key is None
    assert not provider.configured


def test_the_key_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    cfg = parse_config("model:\n  provider: openai\n  name: gpt-4o-mini\n")
    assert from_config(cfg.model).api_key == "sk-from-env"


def test_the_model_section_does_not_disturb_the_rest_of_the_config():
    cfg = parse_config(
        "model:\n"
        "  provider: ollama\n"
        "  name: llama3\n"
        "\n"
        "rules:\n"
        "  stats/grim: off\n"
        "\n"
        "suppress:\n"
        "  - rule: bib/orphans\n"
        "    reason: intentional\n"
    )
    assert cfg.model["provider"] == "ollama"
    assert cfg.disabled == {"stats/grim"}
    assert len(cfg.suppressions) == 1


# --- selection ----------------------------------------------------------


def test_a_configured_provider_makes_the_model_rules_runnable():
    chosen = plan(REG, has_repo=True, has_provider=True)
    assert not any(
        why == "no model provider configured" for why in chosen.skipped.values()
    )


def test_without_a_provider_every_model_rule_is_skipped_and_named():
    chosen = plan(REG, has_repo=True, has_provider=False)
    skipped = {r for r, why in chosen.skipped.items() if "model provider" in why}
    assert skipped == {r.id for r in REG.all() if r.tier is Tier.MODEL_ASSISTED}


def test_passing_a_provider_to_run_is_enough_to_enable_them():
    """The engine should not need telling twice."""
    class _Silent:
        model = "test"

        def complete(self, request):
            return Completion(Outcome.UNAVAILABLE, detail="quiet")

    paper = paper_from_latex(
        "\\begin{document}\nA paper with some prose in it.\n\\end{document}\n",
        needs={"paper.text", "paper.sections"},
    )
    report = run(paper, registry=REG, model=_Silent())
    assert "claim/unsupported" not in report.skipped


# --- the exit code ------------------------------------------------------


def span():
    return Span(Source("paper.tex", "latex", path="paper.tex"), 0, 5, 1)


def finding(tier, severity="high"):
    return Finding(
        rule_id="claim/overreach" if tier is Tier.MODEL_ASSISTED else "stats/grim",
        severity=Severity(severity),
        tier=tier,
        message="m",
        anchors=[span(), span()],
    )


@pytest.mark.parametrize("flag, expected", [([], 0), (["--fail-on-model"], 1)])
def test_a_model_finding_fails_the_build_only_when_asked(monkeypatch, flag, expected):
    """The headline behaviour. Same finding, different exit code."""
    import resint.cli as cli

    monkeypatch.setattr(
        cli, "run", lambda *a, **k: _report([finding(Tier.MODEL_ASSISTED)])
    )
    assert main(["check", str(PLANTED), "--offline", *flag]) == expected


def test_a_deterministic_finding_always_fails_the_build(monkeypatch):
    """Those thirteen rules are the asset, and they are trusted with the exit
    code without a flag."""
    import resint.cli as cli

    monkeypatch.setattr(
        cli, "run", lambda *a, **k: _report([finding(Tier.DETERMINISTIC)])
    )
    assert main(["check", str(PLANTED), "--offline"]) == 1


def test_a_suppressed_finding_does_not_fail_the_build(monkeypatch):
    import resint.cli as cli

    suppressed = finding(Tier.DETERMINISTIC).suppress("known and accepted")
    monkeypatch.setattr(cli, "run", lambda *a, **k: _report([suppressed]))
    assert main(["check", str(PLANTED), "--offline"]) == 0


def _report(findings):
    from resint.engine import Report

    report = Report()
    report.findings = list(findings)
    return report


# --- the flags ----------------------------------------------------------


def test_no_model_skips_the_tier_even_when_configured(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    config = tmp_path / ".resint.yml"
    config.write_text(
        "model:\n  provider: openai\n  name: gpt-4o-mini\n", encoding="utf-8"
    )
    # Without --no-model the rules would be selected and would then try to
    # reach a network this test has no business touching.
    assert (
        main(
            [
                "check",
                str(PLANTED),
                "--offline",
                "--no-model",
                "--config",
                str(config),
            ]
        )
        in (0, 1)
    )


def test_the_report_says_why_the_model_rules_did_not_run(capsys):
    main(["check", str(PLANTED), "--offline"])
    out = capsys.readouterr().out
    assert "no model provider configured" in out


def test_a_run_with_no_model_configured_uses_no_key(capsys):
    """The line a first-time user reads before deciding to trust this."""
    main(["check", str(PLANTED), "--offline"])
    assert "no API key used" in capsys.readouterr().out
