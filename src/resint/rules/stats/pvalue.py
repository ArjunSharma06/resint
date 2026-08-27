"""stats/pvalue-mismatch -- recompute p from the test statistic.

A reported test carries its own check: the statistic and degrees of freedom
determine p exactly. Where the reported p disagrees with the recomputed one,
the finding is arithmetic, not opinion.

Severity splits on consequence, following the distinction statcheck draws.
A disagreement in the fourth decimal is a typo. A disagreement that moves the
result across the threshold the paper is arguing from changes what the paper
claims, and is reported as high.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_EVEN, ROUND_HALF_UP
from typing import Iterator, Literal

from ...ir.paper import StatTest
from ...mathx.special import chi2_sf, f_sf, norm_sf, t_sf
from ..registry import Context, rule

ALPHA = 0.05
_CONVENTIONS = (ROUND_HALF_UP, ROUND_HALF_EVEN)

Verdict = Literal["consistent", "inconsistent", "decision", "unsupported"]


@dataclass(frozen=True, slots=True)
class PResult:
    verdict: Verdict
    computed: float | None = None
    reason: str = ""

    @property
    def flagged(self) -> bool:
        return self.verdict in ("inconsistent", "decision")


def recompute(test: StatTest) -> float | None:
    """The p-value implied by the reported statistic, or None if unsupported."""
    s, df1, df2 = test.statistic, test.df1, test.df2

    if test.kind == "t":
        if df1 is None or df1 <= 0:
            return None
        one = t_sf(abs(s), df1)
        return one if test.tail == 1 else 2.0 * one

    if test.kind == "z":
        one = norm_sf(abs(s))
        return one if test.tail == 1 else 2.0 * one

    if test.kind == "r":
        if df1 is None or df1 <= 0 or abs(s) >= 1.0:
            return None
        t = abs(s) * math.sqrt(df1 / (1.0 - s * s))
        one = t_sf(t, df1)
        return one if test.tail == 1 else 2.0 * one

    if test.kind == "F":
        if df1 is None or df2 is None or df1 <= 0 or df2 <= 0 or s < 0:
            return None
        return f_sf(s, df1, df2)

    if test.kind == "chi2":
        if df1 is None or df1 <= 0 or s < 0:
            return None
        return chi2_sf(s, df1)

    return None


def _rounds_to(computed: float, reported: Decimal, decimals: int) -> bool:
    """Does ``computed`` round to ``reported`` under either convention?"""
    quantum = Decimal(1).scaleb(-decimals)
    exact = Decimal(repr(computed))
    return any(
        exact.quantize(quantum, rounding=c) == reported for c in _CONVENTIONS
    )


def _significant(value: float, comparator: str, bound: float) -> bool:
    """Whether a reported p implies significance at ALPHA."""
    if comparator == "<":
        return bound <= ALPHA
    if comparator == ">":
        return False
    return value < ALPHA


def evaluate(test: StatTest) -> PResult:
    computed = recompute(test)
    if computed is None:
        return PResult("unsupported", reason=f"cannot recompute p for {test.kind}")

    reported = test.p_exact
    bound = float(reported)

    if test.p_comparator == "<":
        consistent = computed < bound
    elif test.p_comparator == ">":
        consistent = computed > bound
    else:
        consistent = _rounds_to(computed, reported, test.p_decimals)

    if consistent:
        return PResult("consistent", computed=computed)

    reported_sig = _significant(bound, test.p_comparator, bound)
    computed_sig = computed < ALPHA
    verdict: Verdict = "decision" if reported_sig != computed_sig else "inconsistent"
    return PResult(verdict, computed=computed)


def _fmt(p: float) -> str:
    return f"{p:.2e}" if p < 0.0001 else f"{p:.4f}".rstrip("0").rstrip(".")


@rule(
    id="stats/pvalue-mismatch",
    severity="med",
    tier="deterministic",
    requires=["paper.stats"],
    cannot_detect=(
        "One-tailed tests not declared as such, and corrections for multiple "
        "comparisons that were applied but not reported. Both make a correct "
        "p look inconsistent, so a declared tail is trusted as given."
    ),
)
def check(ctx: Context) -> Iterator:
    for test in ctx.paper.stats:
        result = evaluate(test)
        if not result.flagged:
            continue

        computed = _fmt(result.computed)
        where = f" ({test.context})" if test.context else ""

        if result.verdict == "decision":
            side = "below" if result.computed < ALPHA else "at or above"
            yield ctx.finding(
                severity="high",
                message=(
                    f"{test.render()}{where} recomputes to p = {computed}, which falls "
                    f"{side} the {ALPHA} threshold the paper argues from. The reported "
                    "and recomputed values disagree about significance."
                ),
                anchors=[test.span, test.p_span],
                fix="Recheck the statistic, the degrees of freedom, and the reported p.",
                affects=("the significance claim resting on this test",),
            )
        else:
            yield ctx.finding(
                message=(
                    f"{test.render()}{where} recomputes to p = {computed}. "
                    "Reported and recomputed values disagree, though not across "
                    f"the {ALPHA} threshold."
                ),
                anchors=[test.span, test.p_span],
                fix="Recheck the statistic, the degrees of freedom, and the reported p.",
            )
