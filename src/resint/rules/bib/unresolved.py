"""bib/unresolved -- references that exist in no index.

The signal this is really after is fabrication. A reference that resolves in
none of Crossref, OpenAlex, arXiv or Semantic Scholar, while claiming to be a
journal article with a DOI, is very often a work that was never written.

Three guards keep this from becoming an accusation machine:

    A failed lookup is never a finding. UNKNOWN means the network answered
    badly, not that the paper is wrong, and it is reported as unchecked.

    Entry types that legitimately sit outside the indices -- theses,
    technical reports, misc -- drop to low severity. Plenty of real work is
    simply not indexed.

    A DOI that fails to resolve is treated as stronger evidence than a title
    that fails to match, because a DOI is a claim about a specific registered
    record rather than a string that might be spelled differently.
"""

from __future__ import annotations

from typing import Iterator

from ...resolve.base import Status
from ..registry import Context, rule


@rule(
    id="bib/unresolved",
    severity="high",
    tier="deterministic",
    requires=["paper.bib", "paper.resolutions"],
    cannot_detect=(
        "Genuinely obscure work absent from all four indices: theses, "
        "institutional reports, non-English venues, and very recent "
        "preprints. Severity is reduced for entry types where that is "
        "expected, but the rule cannot tell obscure from invented."
    ),
)
def check(ctx: Context) -> Iterator:
    for entry in ctx.paper.bib:
        resolution = ctx.paper.resolutions.get(entry.key)
        if resolution is None or resolution.status is not Status.NOT_FOUND:
            continue

        where = ", ".join(resolution.queried) or "any configured index"
        by = "DOI, title, or author-year" if entry.doi else "title or author-year"

        if entry.doi:
            detail = (
                f"The DOI {entry.doi} does not resolve, and no index returned "
                "a matching record."
            )
            severity = "high"
        elif entry.likely_unindexed:
            detail = (
                f"No match found, though {entry.entry_type} entries are often "
                "absent from these indices legitimately."
            )
            severity = "low"
        else:
            detail = "No index returned a matching record."
            severity = "high"

        yield ctx.finding(
            severity=severity,
            message=(
                f"[{entry.key}] {entry.render()} could not be resolved by {by} "
                f"in {where}. {detail}"
            ),
            anchors=[entry.span_for("doi", "title"), entry.span],
            fix="Verify the reference exists and correct or remove the entry.",
        )
