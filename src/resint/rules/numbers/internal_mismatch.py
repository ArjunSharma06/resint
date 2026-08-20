"""numbers/internal-mismatch -- prose and tables disagree about one value.

The defect this catches is mundane and extremely common: a table gets rerun,
the abstract does not get updated, and the paper ships claiming a number its
own results section contradicts.

Matching prose to a cell is where this rule could easily become a
false-positive machine, so it turns on a near-miss test rather than on
equality. If the abstract says 94.2 and the accuracy column holds
{91.4, 93.8}, those are the same quantity a revision apart. If the abstract
says 94.2 and the column holds {41.0, 38.2}, they are different quantities
that happen to share a heading, and the rule stays quiet. Proximity is the
signal; a value far from every cell is not evidence of anything.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Iterator

from ..registry import Context, rule

# A prose value within this relative distance of a cell is the same quantity
# reported at a different revision. Beyond it, they are unrelated.
NEAR_MISS = Decimal("0.02")

# Columns wider than this are lookup tables, not headline results; a prose
# number "missing" from eighty cells says nothing.
MAX_COLUMN = 12


def _relative_gap(value: Decimal, other: Decimal) -> Decimal:
    if other == 0:
        return abs(value)
    return abs(value - other) / abs(other)


@rule(
    id="numbers/internal-mismatch",
    severity="high",
    tier="deterministic",
    requires=["paper.numbers", "paper.tables"],
    cannot_detect=(
        "Values that differ because they describe genuinely different "
        "quantities sharing a column heading, and prose numbers that "
        "legitimately summarise several cells such as an average across "
        "configurations. Distance from the nearest cell is the only guard."
    ),
)
def check(ctx: Context) -> Iterator:
    columns = _numeric_columns(ctx.paper.tables)
    if not columns:
        return

    for number in ctx.paper.numbers:
        label = number.label.strip().lower()
        if label not in columns:
            continue

        for table, col_index, cells in columns[label]:
            values = {c.number for c in cells if c.number is not None}
            if not values or len(values) > MAX_COLUMN:
                continue
            if number.value in values:
                continue

            nearest_cell = min(
                (c for c in cells if c.number is not None),
                key=lambda c: abs(c.number - number.value),
            )
            gap = _relative_gap(number.value, nearest_cell.number)
            if gap > NEAR_MISS or gap == 0:
                continue

            where = number.section or "the text"
            reported = ", ".join(str(v) for v in sorted(values))
            yield ctx.finding(
                message=(
                    f"{where} reports {label} of {number.raw}, but "
                    f"{table.name} reports {nearest_cell.number} for the same "
                    f"quantity (column holds {reported}). A revised table with "
                    "an unrevised claim looks exactly like this."
                ),
                anchors=[number.span, nearest_cell.span],
                fix=(
                    "Reconcile the two values, and check whether other claims "
                    "downstream of this number moved with it."
                ),
            )
            break


def _numeric_columns(tables) -> dict[str, list]:
    """Map each column heading to the numeric cells beneath it."""
    columns: dict[str, list] = {}
    for table in tables:
        if table.irregular or not table.rows:
            continue
        for col_index, heading in enumerate(table.header):
            key = heading.strip().lower().rstrip(":")
            if not key:
                continue
            cells = [
                c
                for c in table.column(col_index)
                if c.row > 0 and c.number is not None
            ]
            if cells:
                columns.setdefault(key, []).append((table, col_index, cells))
    return columns
