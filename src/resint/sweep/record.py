"""What one paper's run produces, and the checks that need no labels.

Two of the checks here are the whole reason a sweep is worth running.

**Crashes** are unambiguous — every traceback is a bug — but they arrive in
bulk and mostly duplicated, so they are fingerprinted by their innermost
frames and forty crashes collapse into three problems.

**The anchor audit** is the one that verifies itself. Every finding claims a
location; the audit goes back to the source, slices at that location, and
checks the text is really there. No ground truth, no labelling, no judgement
— and it catches the exact class of bug that real papers have found again and
again, automatically.
"""

from __future__ import annotations

import hashlib
import traceback as tb
from dataclasses import asdict, dataclass, field

from ..ir.finding import Finding

SCHEMA_VERSION = 1


@dataclass
class AnchorFailure:
    rule_id: str
    source: str
    start: int
    end: int
    reason: str


@dataclass
class AnchorAudit:
    """Whether every finding points at text that exists."""

    checked: int = 0
    failed: int = 0
    missing_sources: list[str] = field(default_factory=list)
    failures: list[AnchorFailure] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return self.failed == 0

    def as_dict(self) -> dict:
        return {
            "checked": self.checked,
            "failed": self.failed,
            "missing_sources": sorted(set(self.missing_sources)),
            "failures": [asdict(f) for f in self.failures[:20]],
        }


def audit_anchors(findings, texts: dict[str, str]) -> AnchorAudit:
    """Re-slice every anchor against its real source.

    ``texts`` maps a source id to the content that id refers to. A source we
    were not given is recorded as unverifiable rather than counted as a
    failure — calling a finding broken because the auditor lacked the file
    would be the same mistake the tool exists to avoid.
    """
    audit = AnchorAudit()

    for finding in findings:
        for anchor in finding.anchors:
            content = texts.get(anchor.source.id)
            if content is None:
                audit.missing_sources.append(anchor.source.id)
                continue

            audit.checked += 1
            reason = _anchor_problem(anchor, content)
            if reason:
                audit.failed += 1
                audit.failures.append(
                    AnchorFailure(
                        rule_id=finding.rule_id,
                        source=anchor.source.id,
                        start=anchor.start,
                        end=anchor.end,
                        reason=reason,
                    )
                )

    return audit


def _anchor_problem(anchor, content: str) -> str:
    if anchor.start < 0 or anchor.end > len(content):
        return f"out of bounds: {anchor.start}-{anchor.end} in {len(content)} chars"
    if anchor.end <= anchor.start:
        return "empty span"
    if not content[anchor.start : anchor.end].strip():
        return "anchors only whitespace"
    if anchor.line is not None:
        actual = content.count("\n", 0, anchor.start) + 1
        if actual != anchor.line:
            return f"line says {anchor.line}, offset is on line {actual}"
    return ""


def fingerprint(exc: BaseException) -> str:
    """A stable id for a crash, from its type and innermost frames.

    Grouping by this turns "forty crashes" into "three bugs". Innermost frames
    rather than outermost, because the top of the stack is identical for every
    paper in the run and carries no information.
    """
    frames = tb.extract_tb(exc.__traceback__)
    tail = frames[-3:] if frames else []
    parts = [type(exc).__name__] + [f"{f.filename}:{f.name}:{f.lineno}" for f in tail]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:12]


@dataclass
class PaperRecord:
    """One paper's outcome, as stored.

    ``source_sha256`` and ``resint_commit`` are what make a later diff
    *provable*: you can show the input did not change and only the code did.
    """

    paper_id: str
    status: str = "ok"  # ok | error | timeout | unreadable
    schema: int = SCHEMA_VERSION
    source_sha256: str = ""
    resint_version: str = ""
    resint_commit: str = ""
    needs: list[str] = field(default_factory=list)

    acquire: dict = field(default_factory=dict)
    slice_census: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)

    findings: list[dict] = field(default_factory=list)
    unchecked: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    skipped: dict = field(default_factory=dict)
    ran: list[str] = field(default_factory=list)

    anchor_audit: dict = field(default_factory=dict)
    error: dict | None = None

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "PaperRecord":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})

    def loaded_findings(self, pool: dict | None = None) -> list[Finding]:
        shared = pool if pool is not None else {}
        return [Finding.from_dict(f, shared) for f in self.findings]

    @property
    def crashed(self) -> bool:
        return self.status in ("error", "timeout")
