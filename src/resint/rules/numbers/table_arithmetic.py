"""numbers/table-arithmetic -- totals that do not total.

Two checks, both pure arithmetic over recovered cells.

A stated total is compared against the sum of the values above it. A
percentage column is compared against 100. Both use a tolerance derived from
the precision the paper itself reported: a column written to one decimal
place cannot be held to more than one decimal place of accuracy, and holding
it to more manufactures findings out of rounding.

Any table the extractor marked irregular is skipped outright. A misparsed
grid produces confident nonsense, and reporting "could not read this table"
costs far less trust than reporting a total that was never in the paper.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Iterator

from ..registry import Context, rule

_TOTAL_WORDS = frozenset({"total", "sum", "overall", "all", "combined", "aggregate"})
_PERCENT_WORDS = frozenset({"%", "percent", "percentage", "share", "proportion"})

#: How far from 100 a column may sum and still be a partition that is *wrong*
#: rather than a column that was never a partition at all.
#:
#: A column of independent rates -- employment rate by region, prevalence by
#: subgroup -- has every value between 0 and 100 and sums to whatever it sums
#: to. One real table summed to 474.8 and was reported as "percentages of a
#: whole but sums to 474.8, not 100", which is not a defect in the paper: each
#: row is its own denominator.
#:
#: Being far from 100 is evidence the column is not parts-of-a-whole. Being
#: near but not equal is evidence of a missing category or a rounding
#: convention, which is the finding worth making.
NEAR_WHOLE = Decimal(5)


def tolerance(cells) -> Decimal:
    """Half a unit in the last place the paper actually reported."""
    places = max((c.decimals for c in cells), default=0)
    return Decimal(5).scaleb(-(places + 1)) * len(cells)


def _is_total_row(row) -> bool:
    return bool(row) and row[0].text.strip().lower().rstrip(":") in _TOTAL_WORDS


@rule(
    id="numbers/table-arithmetic",
    severity="med",
    tier="deterministic",
    requires=["paper.tables"],
    cannot_detect=(
        "Tables whose cell structure did not survive extraction; those are "
        "skipped and reported as unchecked rather than guessed at. Also "
        "totals over a subset of rows, where the paper sums some entries and "
        "not others without saying so. A percentage column missing more than "
        "a few points is not reported either: at that distance a partition "
        "with a dropped category cannot be told apart from a column of "
        "independent rates, each row having its own denominator, and "
        "guessing between them produces confident nonsense."
    ),
)
def check(ctx: Context) -> Iterator:
    for table in ctx.paper.tables:
        if table.irregular or len(table.rows) < 3:
            continue

        total_rows = [r for r in table.body_rows() if _is_total_row(r)]

        for col_index, heading in enumerate(table.header):
            column = [c for c in table.column(col_index) if c.row > 0]
            numeric = [c for c in column if c.number is not None]
            if len(numeric) < 3:
                continue

            for total_row in total_rows:
                if col_index >= len(total_row):
                    continue
                stated = total_row[col_index]
                if stated.number is None:
                    continue

                parts = [c for c in numeric if c.row != stated.row]
                if len(parts) < 2:
                    continue

                summed = sum((c.number for c in parts), Decimal(0))
                slack = tolerance(parts)
                if abs(summed - stated.number) <= slack:
                    continue

                heading_label = heading.strip() or f"column {col_index}"
                yield ctx.finding(
                    message=(
                        f"{table.name} states a total of {stated.number} for "
                        f"{heading_label}, but the {len(parts)} entries above it "
                        f"sum to {summed}."
                    ),
                    anchors=[stated.span, parts[0].span],
                    fix="Recheck the entries or the stated total.",
                )

            if _looks_like_percentages(heading, numeric, total_rows):
                parts = [
                    c for c in numeric if not any(c.row == t[0].row for t in total_rows)
                ]
                summed = sum((c.number for c in parts), Decimal(0))
                slack = tolerance(parts)
                if parts and slack < abs(summed - Decimal(100)) <= NEAR_WHOLE:
                    yield ctx.finding(
                        severity="low",
                        message=(
                            f"{table.name} column {heading.strip()!r} reads as "
                            f"percentages of a whole but sums to {summed}, not 100."
                        ),
                        anchors=[parts[0].span, parts[-1].span],
                        fix="Check for a missing category or a rounding convention.",
                    )


def _looks_like_percentages(heading: str, numeric, total_rows) -> bool:
    """Only treat a column as parts-of-a-whole when it plainly says so."""
    label = heading.strip().lower()
    if not any(word in label for word in _PERCENT_WORDS):
        return False
    if total_rows:
        return False  # an explicit total row is checked directly instead
    values = [c.number for c in numeric]
    return all(Decimal(0) <= v <= Decimal(100) for v in values) and len(values) >= 3
