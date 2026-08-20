"""Findings: evidence, not opinion.

The central constraint lives here. A finding must cite at least two spans,
enforced at construction rather than by convention. One anchor is an
assertion the reader has to go and check; two make it a comparison they can
verify by reading the finding itself. That difference is the product.

The single exception is an absence finding, where one side of the comparison
is a negative with no location -- a key with no bibliography entry, an entry
never cited. Those pass ``absent_from`` naming exactly what was searched,
which restores checkability without inventing a span to satisfy the rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

from .span import Span

_SEVERITY_ORDER = {"low": 0, "med": 1, "high": 2}


class Severity(str, Enum):
    LOW = "low"
    MED = "med"
    HIGH = "high"

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Severity):
            return NotImplemented
        return _SEVERITY_ORDER[self.value] < _SEVERITY_ORDER[other.value]


class Tier(str, Enum):
    """How a verdict was reached.

    DETERMINISTIC verdicts are computed and carry no model in the loop, even
    if a model helped extract the inputs. MODEL_ASSISTED verdicts involve
    judgment and are marked as such wherever they are displayed.
    """

    DETERMINISTIC = "deterministic"
    MODEL_ASSISTED = "model-assisted"


class AnchorError(ValueError):
    """Raised when a finding is constructed with insufficient evidence."""


@dataclass(frozen=True, slots=True)
class Finding:
    rule_id: str
    severity: Severity
    tier: Tier
    message: str
    anchors: tuple[Span, ...]
    absent_from: str | None = None
    fix: str | None = None
    affects: tuple[str, ...] = ()
    confidence: float | None = None
    suppressed_reason: str | None = None

    def __init__(
        self,
        *,
        rule_id: str,
        severity: Severity | str,
        tier: Tier | str,
        message: str,
        anchors: Sequence[Span],
        absent_from: str | None = None,
        fix: str | None = None,
        affects: Sequence[str] = (),
        confidence: float | None = None,
        suppressed_reason: str | None = None,
    ) -> None:
        anchors = tuple(anchors)

        # Absence findings are the one legitimate single-anchor case. Half of
        # the comparison is a negative -- a reference never cited, a key with
        # no entry -- and that half has no location by definition. Rather than
        # inventing a span to satisfy the rule, the finding must name exactly
        # what was searched and came up empty, which is what makes the claim
        # checkable. Everything else still needs two.
        if absent_from is not None:
            if not absent_from.strip():
                raise ValueError(
                    f"{rule_id}: absent_from must name what was searched"
                )
            if not anchors:
                raise AnchorError(
                    f"{rule_id}: an absence finding still needs the anchor for "
                    "the side that does exist."
                )
        elif len(anchors) < 2:
            raise AnchorError(
                f"{rule_id}: a finding needs at least two anchors, got {len(anchors)}. "
                "One anchor is an assertion; two are a comparison the reader can "
                "check. If the other side is genuinely absent, pass absent_from= "
                "naming where you looked."
            )
        if not message.strip():
            raise ValueError(f"{rule_id}: message must not be empty")
        if confidence is not None and not 0.0 <= confidence <= 1.0:
            raise ValueError(f"{rule_id}: confidence {confidence} outside [0, 1]")

        object.__setattr__(self, "rule_id", rule_id)
        object.__setattr__(self, "severity", Severity(severity))
        object.__setattr__(self, "tier", Tier(tier))
        object.__setattr__(self, "message", message)
        object.__setattr__(self, "anchors", anchors)
        object.__setattr__(self, "absent_from", absent_from)
        object.__setattr__(self, "fix", fix)
        object.__setattr__(self, "affects", tuple(affects))
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "suppressed_reason", suppressed_reason)

    @property
    def suppressed(self) -> bool:
        return self.suppressed_reason is not None

    def locate(self) -> str:
        """The two-sided location line, e.g. ``abstract:L4 <-> table3:r2c4``.

        Repeated locations collapse. Both anchors are always retained on the
        finding and in JSON output -- this only affects display, where
        "L19 <-> L19" reads as a formatting bug rather than as evidence.
        """
        seen: list[str] = []
        for a in self.anchors:
            where = a.locate()
            if where not in seen:
                seen.append(where)
        if self.absent_from:
            seen.append(f"{self.absent_from} (absent)")
        return " <-> ".join(seen)

    def suppress(self, reason: str) -> "Finding":
        """Return a suppressed copy.

        Suppression happens at the reporting layer, never during evaluation:
        the finding survives into JSON output marked with its reason, so a
        suppression can never hide a regression from the corpus.
        """
        if not reason.strip():
            raise ValueError("a suppression must state a reason")
        return Finding(
            rule_id=self.rule_id,
            severity=self.severity,
            tier=self.tier,
            message=self.message,
            anchors=self.anchors,
            absent_from=self.absent_from,
            fix=self.fix,
            affects=self.affects,
            confidence=self.confidence,
            suppressed_reason=reason,
        )

    def to_dict(self) -> dict:
        return {
            "rule": self.rule_id,
            "severity": self.severity.value,
            "tier": self.tier.value,
            "message": self.message,
            "anchors": [
                {
                    "source": a.source.id,
                    "kind": a.source.kind,
                    "start": a.start,
                    "end": a.end,
                    "line": a.line,
                    "label": a.label,
                    "locate": a.locate(),
                }
                for a in self.anchors
            ],
            "absent_from": self.absent_from,
            "fix": self.fix,
            "affects": list(self.affects),
            "confidence": self.confidence,
            "suppressed": self.suppressed,
            "suppressed_reason": self.suppressed_reason,
        }
