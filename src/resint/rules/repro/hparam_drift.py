"""repro/hparam-drift -- the paper's hyperparameters against the code's.

This is the rule with the most room to do damage. It tells an author that
they misreported their own experiment, and if it is wrong about that, it is
wrong in the most costly possible way. So the guards are heavier here than
anywhere else in the codebase:

    A value is only compared when the repository yields exactly one effective
    value for it. Two sources at the same binding strength that disagree means
    precedence cannot be established, and the rule abstains and says so.

    Values are compared numerically, so 3e-4 and 0.0003 are the same number.
    A rule that reported those as a mismatch would be worse than useless.

    Ratios that look like a unit convention rather than a mistake -- a paper
    reporting a percentage where the code stores a fraction -- are reported at
    reduced severity, because they usually are a convention.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Iterator

from ...ir.repo import canonical
from ..registry import Context, rule

# Below this relative difference the two values are the same number written
# differently, and any gap is float rendering rather than disagreement.
SAME = Decimal("0.0001")


def _decimal(raw: str) -> Decimal | None:
    try:
        return Decimal(raw.strip())
    except (InvalidOperation, ValueError):
        return None


def _relative(a: Decimal, b: Decimal) -> Decimal:
    if b == 0:
        return abs(a)
    return abs(a - b) / abs(b)


def _unit_convention(paper: Decimal, code: Decimal) -> bool:
    """Whether the pair looks like percent-versus-fraction rather than a bug."""
    if code == 0 or paper == 0:
        return False
    ratio = abs(paper / code)
    return ratio in (Decimal(100), Decimal("0.01"))


@rule(
    id="repro/hparam-drift",
    severity="high",
    tier="deterministic",
    requires=["paper.hyperparameters", "repo.configs"],
    cannot_detect=(
        "Values supplied only on a command line in a launch script that was "
        "never committed, and values computed at runtime rather than "
        "declared. Where the effective value is genuinely ambiguous the rule "
        "abstains rather than guessing, and records why."
    ),
)
def check(ctx: Context) -> Iterator:
    stated = ctx.paper.hyperparameters
    if not stated:
        return

    for number in stated:
        name = canonical(number.label)
        effective, candidates = ctx.repo.configs.effective(name)

        if not candidates:
            continue

        if effective is None:
            # Several equally strong sources disagree. Which one a run uses
            # depends on invocation, which the repository does not record.
            values = sorted({c.value for c in candidates})
            origins = sorted({c.origin for c in candidates})
            ctx.abstain(
                f"{number.label} not compared -- the repository holds "
                f"{len(values)} equally binding values ({', '.join(values)}) "
                f"across {', '.join(origins)}"
            )
            continue

        paper_value = _decimal(number.raw)
        code_value = _decimal(effective.value)
        if paper_value is None or code_value is None:
            continue
        if _relative(paper_value, code_value) <= SAME:
            continue

        if _unit_convention(paper_value, code_value):
            yield ctx.finding(
                severity="low",
                message=(
                    f"{number.label} is {number.raw} in the paper and "
                    f"{effective.value} in {effective.origin}. The ratio is "
                    "exactly 100, which usually means one is a percentage and "
                    "the other a fraction rather than a disagreement."
                ),
                anchors=[number.span, effective.span],
                fix="Confirm the units match, or state them in the paper.",
            )
            continue

        others = [c for c in candidates if c is not effective]
        trail = ""
        if others:
            trail = (
                " Resolved through "
                + " -> ".join(
                    f"{c.origin} ({c.value})" for c in sorted(others, key=lambda c: c.binding)
                )
                + f" -> {effective.origin} ({effective.value})."
            )

        yield ctx.finding(
            message=(
                f"{number.label} is {number.raw} in the paper, but a run would "
                f"use {effective.value} from {effective.origin}.{trail}"
            ),
            anchors=[number.span, effective.span],
            fix="Reconcile the reported value with the one the code uses.",
            affects=("every result produced with this setting",),
        )
