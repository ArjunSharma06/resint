"""SARIF output.

SARIF is what GitHub code scanning ingests, so emitting it means findings
appear as annotations on a pull request with no UI work on our side. It costs
one serializer and buys the entire review surface.

Two details matter for this tool specifically. ``cannot_detect`` goes into
each rule's help text, so the limitation travels with the finding into
whatever reads the file. And suppressed findings are emitted with a SARIF
suppression rather than dropped, so a reviewer can see that a check fired and
was consciously accepted -- which is different from it never having run.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..engine import Report
from ..ir.finding import Finding, Severity, Tier
from ..rules import Registry

SCHEMA = "https://json.schemastore.org/sarif-2.1.0.json"
VERSION = "2.1.0"

_LEVEL = {Severity.HIGH: "error", Severity.MED: "warning", Severity.LOW: "note"}


def _uri(path: str) -> str:
    return Path(path).as_posix()


def _location(span) -> dict:
    region: dict = {
        "charOffset": span.start,
        "charLength": max(span.end - span.start, 1),
    }
    if span.line is not None:
        region["startLine"] = span.line
    return {
        "physicalLocation": {
            "artifactLocation": {"uri": _uri(span.source.path or span.source.id)},
            "region": region,
        }
    }


def _result(finding: Finding) -> dict:
    primary, *related = finding.anchors
    result: dict = {
        "ruleId": finding.rule_id,
        "level": _LEVEL[finding.severity],
        "message": {"text": finding.message},
        "locations": [_location(primary)],
        "properties": {
            "tier": finding.tier.value,
            "deterministic": finding.tier is Tier.DETERMINISTIC,
        },
    }
    if related:
        result["relatedLocations"] = [
            dict(_location(span), id=index) for index, span in enumerate(related, 1)
        ]
    if finding.absent_from:
        result["properties"]["absentFrom"] = finding.absent_from
    if finding.fix:
        result["properties"]["fix"] = finding.fix
    if finding.affects:
        result["properties"]["affects"] = list(finding.affects)
    if finding.suppressed:
        result["suppressions"] = [
            {"kind": "external", "justification": finding.suppressed_reason}
        ]
    return result


def _rule_descriptor(rule) -> dict:
    return {
        "id": rule.id,
        "name": rule.id.replace("/", "-"),
        "shortDescription": {"text": rule.id},
        "fullDescription": {"text": f"Cannot detect: {rule.cannot_detect}"},
        "help": {
            "text": f"Cannot detect: {rule.cannot_detect}",
            "markdown": f"**Cannot detect:** {rule.cannot_detect}",
        },
        "defaultConfiguration": {"level": _LEVEL[rule.severity]},
        "properties": {
            "tier": rule.tier.value,
            "requires": list(rule.requires),
            "tags": [rule.family, rule.tier.value],
        },
    }


def render(report: Report, registry: Registry, version: str = "0.1.0") -> str:
    """Serialize a report as SARIF 2.1.0."""
    fired = {f.rule_id for f in report.findings}
    rules = [r for r in registry.all() if r.id in fired]

    invocation = {
        "executionSuccessful": True,
        "toolExecutionNotifications": [
            {
                "level": "note",
                "message": {"text": note},
            }
            for note in (*report.unchecked, *report.notes)
        ],
    }

    document = {
        "$schema": SCHEMA,
        "version": VERSION,
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "resint",
                        "informationUri": "https://github.com/ArjunSharma06/resint",
                        "version": version,
                        "rules": [_rule_descriptor(r) for r in rules],
                    }
                },
                "invocations": [invocation],
                "results": [_result(f) for f in report.findings],
            }
        ],
    }
    return json.dumps(document, indent=2)
