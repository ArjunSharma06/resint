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


def recompute(test: StatTest, statistic: float | None = None) -> float | None:
    """The p-value implied by a statistic, or None if unsupported.

    ``statistic`` overrides the reported value so the same arithmetic can be
    run at both ends of the interval the paper's rounding actually claims.
    """
    s = test.statistic if statistic is None else statistic
    df1, df2 = test.df1, test.df2

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


def statistic_interval(test: StatTest) -> tuple[float, float]:
    """The range of statistics consistent with what the paper printed.

    ``t = 2.086`` does not mean 2.086 exactly; it means the author had a value
    that rounds to 2.086, so t lies in [2.0855, 2.0865). Reporting a
    disagreement computed from the midpoint alone turns the author's rounding
    into a finding, which is the largest false-positive source in this whole
    family of checks.
    """
    half = 0.5 * (10.0 ** -test.statistic_decimals)
    magnitude = abs(test.statistic)
    return max(magnitude - half, 0.0), magnitude + half


def computed_interval(test: StatTest) -> tuple[float, float] | None:
    """Every p the reported statistic could imply, smallest first.

    p falls as the statistic grows for every test here, so the interval's ends
    come from the interval's ends -- the larger statistic gives the smaller p.
    """
    low_stat, high_stat = statistic_interval(test)
    p_high = recompute(test, low_stat)
    p_low = recompute(test, high_stat)
    if p_low is None or p_high is None:
        return None
    return min(p_low, p_high), max(p_low, p_high)


def reported_interval(reported: Decimal, decimals: int) -> tuple[float, float]:
    """The range of p-values that round to what the paper printed."""
    half = Decimal(5).scaleb(-(decimals + 1))
    return float(reported - half), float(reported + half)


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

    span = computed_interval(test)
    if span is None:
        return PResult("unsupported", reason=f"cannot recompute p for {test.kind}")
    p_low, p_high = span

    reported = test.p_exact
    bound = float(reported)

    # Both sides are intervals: the statistic's precision bounds what p can
    # be, and the reported p's precision bounds what was claimed. A finding
    # requires those two ranges not to overlap at all -- anything less is the
    # author's rounding, not their error.
    if test.p_comparator == "<":
        consistent = p_low < bound
    elif test.p_comparator == ">":
        consistent = p_high > bound
    else:
        claim_low, claim_high = reported_interval(reported, test.p_decimals)
        consistent = p_low <= claim_high and p_high >= claim_low

    if consistent:
        return PResult("consistent", computed=computed)

    reported_sig = _significant(bound, test.p_comparator, bound)

    # A decision error means the recomputed result lands on the other side of
    # the threshold the paper argues from. When the interval straddles alpha
    # the recomputation does not establish a side, so it stays "inconsistent"
    # rather than claiming a conclusion changed.
    straddles = p_low < ALPHA <= p_high
    computed_sig = p_high < ALPHA
    verdict: Verdict = (
        "decision"
        if not straddles and reported_sig != computed_sig
        else "inconsistent"
    )
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
        "p look inconsistent, so a declared tail is trusted as given. "
        "Most importantly, any result reported without its test statistic. "
        "This rule recomputes p *from* the statistic, so an odds ratio or a "
        "hazard ratio quoted with a confidence interval and a bare p is "
        "outside it entirely -- and a large share of clinical and "
        "epidemiological work reports results that way, so the rule reaches "
        "a minority of that literature rather than most of it. A low finding "
        "count there is expected rather than a malfunction. Sampled "
        "proportions, and what they were sampled from, are in "
        "notes/sweep-log.md."
    ),
)
def check(ctx: Context) -> Iterator:
    checked = 0
    unsupported = 0

    for test in ctx.paper.stats:
        result = evaluate(test)
        if result.verdict == "unsupported":
            unsupported += 1
            continue
        checked += 1
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

    # Reported whether or not anything disagreed. A rule that finds nothing and
    # says nothing is indistinguishable from a rule that did not run, and this
    # one legitimately finds nothing on most papers.
    found = len(ctx.paper.stats)
    if found:
        noun = "test statistic" if found == 1 else "test statistics"
        census = f"{found} {noun} found, {checked} recomputed"
        if unsupported:
            census += f", {unsupported} of a kind this rule cannot recompute"
        ctx.abstain(census)
