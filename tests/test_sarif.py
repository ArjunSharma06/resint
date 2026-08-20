"""SARIF output and the CLI surface that emits it."""

import json
from pathlib import Path

import pytest

from resint.cli import main
from resint.config import parse as parse_config
from resint.engine import run
from resint.parse.document import paper_from_path
from resint.report.sarif import SCHEMA, VERSION, render
from resint.resolve import StaticResolver
from resint.rules import load_all

from conftest import CORPUS_RECORDS

CORPUS = Path(__file__).resolve().parents[1] / "corpus"
PLANTED = CORPUS / "planted" / "paper.tex"
CLEAN = CORPUS / "clean" / "paper.tex"
RESOLVER = StaticResolver(records=dict(CORPUS_RECORDS))
REG = load_all()


@pytest.fixture(scope="module")
def document():
    report = run(paper_from_path(PLANTED, resolver=RESOLVER))
    return json.loads(render(report, REG))


# --- shape --------------------------------------------------------------


def test_declares_schema_and_version(document):
    assert document["$schema"] == SCHEMA
    assert document["version"] == VERSION
    assert len(document["runs"]) == 1


def test_driver_names_the_tool(document):
    driver = document["runs"][0]["tool"]["driver"]
    assert driver["name"] == "resint"
    assert driver["informationUri"].startswith("https://")


def test_every_finding_becomes_a_result(document):
    report = run(paper_from_path(PLANTED, resolver=RESOLVER))
    assert len(document["runs"][0]["results"]) == len(report.findings)


def test_only_rules_that_fired_are_described(document):
    driver = document["runs"][0]["tool"]["driver"]
    described = {r["id"] for r in driver["rules"]}
    fired = {r["ruleId"] for r in document["runs"][0]["results"]}
    assert described == fired


# --- content ------------------------------------------------------------


@pytest.mark.parametrize(
    "severity, level", [("high", "error"), ("med", "warning"), ("low", "note")]
)
def test_severity_maps_to_sarif_level(document, severity, level):
    report = run(paper_from_path(PLANTED, resolver=RESOLVER))
    expected = {
        f.rule_id for f in report.findings if f.severity.value == severity
    }
    got = {
        r["ruleId"] for r in document["runs"][0]["results"] if r["level"] == level
    }
    assert expected <= got


def test_primary_anchor_becomes_the_location(document):
    for result in document["runs"][0]["results"]:
        region = result["locations"][0]["physicalLocation"]["region"]
        assert region["charLength"] >= 1
        assert region["startLine"] >= 1


def test_extra_anchors_become_related_locations(document):
    multi = [
        r for r in document["runs"][0]["results"] if r.get("relatedLocations")
    ]
    assert multi, "two-anchor findings must expose both sides"
    assert all(loc["id"] for r in multi for loc in r["relatedLocations"])


def test_cannot_detect_travels_into_the_rule_help(document):
    """The limitation has to survive into whatever reads the file."""
    for rule in document["runs"][0]["tool"]["driver"]["rules"]:
        assert rule["help"]["text"].startswith("Cannot detect:")
        assert rule["help"]["markdown"].startswith("**Cannot detect:**")


def test_tier_is_exposed_so_consumers_can_distinguish_proof_from_judgement(document):
    for result in document["runs"][0]["results"]:
        assert result["properties"]["tier"] == "deterministic"
        assert result["properties"]["deterministic"] is True


def test_absence_findings_name_what_was_missing(document):
    absent = [
        r
        for r in document["runs"][0]["results"]
        if r["properties"].get("absentFrom")
    ]
    assert absent, "bib/orphans emits absence findings"


def test_unchecked_notes_are_carried_as_notifications():
    report = run(paper_from_path(CLEAN, resolver=RESOLVER))
    document = json.loads(render(report, REG))
    notifications = document["runs"][0]["invocations"][0][
        "toolExecutionNotifications"
    ]
    assert any("2 sample sizes" in n["message"]["text"] for n in notifications)


def test_suppressed_findings_are_marked_not_dropped():
    """A reviewer must be able to tell "accepted" from "never ran"."""
    cfg = parse_config(
        'suppress:\n  - rule: stats/grim\n    reason: "multi-item scale"\n'
    )
    report = run(paper_from_path(PLANTED, resolver=RESOLVER), config=cfg)
    document = json.loads(render(report, REG))
    grim = next(
        r for r in document["runs"][0]["results"] if r["ruleId"] == "stats/grim"
    )
    assert grim["suppressions"][0]["justification"] == "multi-item scale"


# --- CLI ----------------------------------------------------------------


def test_sarif_format_is_valid_json(capsys):
    main(["check", str(PLANTED), "--offline", "--format", "sarif"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["version"] == VERSION


@pytest.mark.parametrize(
    "fail_on, expected", [("high", 1), ("med", 1), ("low", 1), ("none", 0)]
)
def test_fail_on_controls_the_exit_code(capsys, fail_on, expected):
    code = main(["check", str(PLANTED), "--offline", "--fail-on", fail_on])
    capsys.readouterr()
    assert code == expected


def test_clean_paper_exits_zero_at_every_threshold(capsys):
    for level in ("high", "med", "low"):
        assert main(["check", str(CLEAN), "--offline", "--fail-on", level]) == 0
        capsys.readouterr()


def test_rules_command_filters_by_family(capsys):
    main(["rules", "--family", "bib", "--format", "json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload and all(r["id"].startswith("bib/") for r in payload)


def test_rules_command_filters_by_tier(capsys):
    main(["rules", "--tier", "deterministic", "--format", "json"])
    payload = json.loads(capsys.readouterr().out)
    assert all(r["tier"] == "deterministic" for r in payload)
    assert all(r["cannot_detect"] for r in payload)


def test_init_writes_a_config_and_refuses_to_clobber(tmp_path, capsys):
    assert main(["init", str(tmp_path)]) == 0
    written = (tmp_path / ".resint.yml").read_text(encoding="utf-8")
    assert "suppress:" in written and "reason" in written

    assert main(["init", str(tmp_path)]) == 2
    assert "already exists" in capsys.readouterr().err

    assert main(["init", str(tmp_path), "--force"]) == 0


def test_written_config_is_parseable(tmp_path, capsys):
    main(["init", str(tmp_path)])
    capsys.readouterr()
    cfg = parse_config((tmp_path / ".resint.yml").read_text(encoding="utf-8"))
    assert cfg.suppressions == [] and cfg.disabled == set()


def test_no_config_ignores_a_present_file(tmp_path, capsys):
    (tmp_path / ".resint.yml").write_text(
        "rules:\n  stats/grim: off\n", encoding="utf-8"
    )
    paper = tmp_path / "paper.tex"
    paper.write_text(
        PLANTED.read_text(encoding="utf-8"), encoding="utf-8"
    )

    main(["check", str(paper), "--offline", "--format", "json", "--fail-on", "none"])
    with_config = json.loads(capsys.readouterr().out)
    assert "stats/grim" in with_config["skipped"]

    main([
        "check", str(paper), "--offline", "--no-config",
        "--format", "json", "--fail-on", "none",
    ])
    without = json.loads(capsys.readouterr().out)
    assert "stats/grim" not in without["skipped"]


def test_missing_config_file_is_a_usage_error(capsys):
    code = main(["check", str(PLANTED), "--offline", "--config", "nope.yml"])
    assert code == 2
    assert "no such config file" in capsys.readouterr().err
