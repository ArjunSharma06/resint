"""bib/unindexed -- no DOI, and a title search found nothing.

Split out of ``bib/unresolved``, which used to report this at the same table
as a DOI that fails to resolve. Across 68 real papers that produced 176
title-only findings against 18 DOI ones -- and the 176 buried the 18.

**Off by default**, because a failed title search is weak evidence about the
world and strong evidence about our coverage. Titles miss for entirely
ordinary reasons: abbreviated in the bibliography, a non-English venue, a book
or a standard or a chapter, a workshop paper nobody registered, or simply a
gap in what Crossref, OpenAlex, arXiv and DBLP happen to hold. None of those
mean the reference does not exist.

It is still worth having. A bibliography where thirty entries resolve and one
does not is telling you something, and someone auditing a reference list
deliberately wants the whole picture. That is an opt-in job, not a default
one:

    rules:
      bib/unindexed: on

Entry types that legitimately sit outside these indices -- theses, technical
reports, ``@misc``, software -- are excluded outright rather than reported
quietly. Counting them would inflate the rate with entries nobody should
expect to find, which is how the original rule came to fire on three papers in
four.
"""

from __future__ import annotations

from typing import Iterator

from ...resolve.base import Status
from ..registry import Context, rule


@rule(
    id="bib/unindexed",
    severity="low",
    tier="deterministic",
    opt_in=True,
    requires=["paper.bib", "paper.resolutions"],
    cannot_detect=(
        "Whether the reference exists. This rule reports only that four "
        "indices were searched by title and none held a match, which is a "
        "statement about coverage rather than about the work. Abbreviated "
        "titles, non-English venues, books, standards, chapters and workshop "
        "papers all fail to match while being entirely real. Entry types that "
        "sit outside these indices by nature are excluded rather than "
        "reported, so absence here is not evidence of fabrication -- for that, "
        "see bib/unresolved, which requires a DOI."
    ),
)
def check(ctx: Context) -> Iterator:
    for entry in ctx.paper.bib:
        # A DOI that fails to resolve is a different, much stronger claim and
        # belongs to bib/unresolved.
        if entry.doi:
            continue

        # Excluded from the denominator, not merely downgraded. A thesis that
        # Crossref has never heard of is not a finding at any severity.
        if entry.likely_unindexed:
            continue

        resolution = ctx.paper.resolutions.get(entry.key)
        if resolution is None or resolution.status is not Status.NOT_FOUND:
            continue

        where = ", ".join(resolution.queried) or "any configured index"
        recovered = (
            " The title was recovered from a compiled bibliography and may "
            "not be exact."
            if entry.from_bbl
            else ""
        )

        yield ctx.finding(
            message=(
                f"[{entry.key}] {entry.render()} was not found by title in "
                f"{where}, and carries no DOI to check against.{recovered}"
            ),
            anchors=[entry.span_for("title"), entry.span],
            fix=(
                "Add a DOI if the work has one -- that turns an unanswerable "
                "search into a check. Otherwise confirm the entry by hand."
            ),
        )
