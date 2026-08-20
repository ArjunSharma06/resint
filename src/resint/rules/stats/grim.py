"""stats/grim -- granularity-related inconsistency of means.

If a mean is computed from N integer responses, the sum is an integer, so
only multiples of 1/N are reachable. A mean of 3.47 from N=20 is not one of
them, and no amount of rounding produces it. The arithmetic is trivial; the
value is in knowing to do it.

Two decisions keep the false-positive rate near zero. The test abstains
entirely once granularity reaches 10^decimals, because past that point almost
every value is reachable and a "finding" would be noise. And consistency is
checked under both half-up and half-even rounding -- papers are not
consistent about which they use, and a mean reachable under either convention
is not evidence of anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_EVEN, ROUND_HALF_UP
from typing import Iterator, Literal

from ...ir.paper import ReportedMean
from ..registry import Context, rule

_CONVENTIONS = (ROUND_HALF_UP, ROUND_HALF_EVEN)

Verdict = Literal["consistent", "inconsistent", "no-power"]


@dataclass(frozen=True, slots=True)
class GrimResult:
    verdict: Verdict
    nearest: tuple[Decimal, ...] = ()
    reason: str = ""

    @property
    def inconsistent(self) -> bool:
        return self.verdict == "inconsistent"


def grim(mean: Decimal, granularity: int, decimals: int) -> GrimResult:
    """Is ``mean`` reachable as the average of ``granularity`` integers?"""
    if granularity <= 0:
        raise ValueError(f"granularity must be positive, got {granularity}")
    if decimals < 1:
        return GrimResult("no-power", reason="mean reported without decimal places")
    if granularity >= 10**decimals:
        return GrimResult(
            "no-power",
            reason=(
                f"granularity {granularity} exceeds 10^{decimals}; nearly every "
                "value is reachable, so the test carries no information"
            ),
        )

    quantum = Decimal(1).scaleb(-decimals)
    target = mean * granularity
    floor_sum = int(target.to_integral_value(rounding="ROUND_FLOOR"))

    reachable: list[Decimal] = []
    for candidate in (floor_sum - 1, floor_sum, floor_sum + 1, floor_sum + 2):
        exact = Decimal(candidate) / Decimal(granularity)
        for convention in _CONVENTIONS:
            rounded = exact.quantize(quantum, rounding=convention)
            if rounded == mean:
                return GrimResult("consistent")
            if rounded not in reachable:
                reachable.append(rounded)

    nearest = tuple(sorted(reachable, key=lambda v: abs(v - mean))[:2])
    return GrimResult(
        "inconsistent",
        nearest=tuple(sorted(nearest)),
        reason=f"no integer sum over {granularity} values rounds to {mean}",
    )


@rule(
    id="stats/grim",
    severity="high",
    tier="deterministic",
    requires=["paper.means"],
    cannot_detect=(
        "Anything on a non-integer response scale. Requires an inferable "
        "integer scale and an exact N; abstains otherwise, including whenever "
        "granularity reaches 10^decimals."
    ),
)
def check(ctx: Context) -> Iterator:
    for mean in ctx.paper.means:
        result = grim(mean.value, mean.granularity, mean.decimals)
        if not result.inconsistent:
            continue

        nearest = ", ".join(str(v) for v in result.nearest)
        items = "" if mean.items == 1 else f" x {mean.items} items"
        label = f" for {mean.context}" if mean.context else ""

        # A multi-item scale multiplies granularity, and a mean unreachable
        # over N responses is often reachable over N x items. Where the item
        # count had to be assumed, the inference is weaker than the
        # arithmetic, and the finding says so rather than overclaiming.
        assumed = (
            " Assumes one response per participant; a multi-item scale would "
            "raise granularity and may make this mean attainable."
            if mean.items_inferred
            else ""
        )

        yield ctx.finding(
            severity="med" if mean.items_inferred else "high",
            message=(
                f"Mean {mean.raw}{label} is not attainable from N={mean.n}"
                f"{items} integer responses. Nearest attainable: {nearest}.{assumed}"
            ),
            anchors=[mean.span, mean.n_span],
            fix="Recheck the reported mean, the sample size, or the response scale.",
        )
