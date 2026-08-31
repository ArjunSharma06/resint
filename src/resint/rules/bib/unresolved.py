"""bib/unresolved -- a DOI that does not resolve.

The fabrication signal, and now only that. A DOI is a claim about one
registered record: it either resolves or it does not, and a DOI that resolves
nowhere while the entry claims to be a published article is very often a work
that was never written.

Everything weaker moved out. A title search that finds nothing used to be
reported here at the same table, and across 68 real papers that meant 176
title-only findings burying 18 DOI ones -- the rule fired on three papers in
four and read as a warning banner. Titles miss for ordinary reasons; DOIs do
not. That half is now ``bib/unindexed``, off by default.

Two guards remain, and both are about not overstating:

    A failed lookup is never a finding. UNKNOWN means the network answered
    badly, not that the paper is wrong, and it is reported as unchecked.

    Only indices that actually answered are named. An index that was down was
    not a search, and saying otherwise would make an absence claim larger than
    the evidence behind it.
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
        "Anything without a DOI -- that is bib/unindexed, which is off by "
        "default because a failed title search says more about index "
        "coverage than about the work. It also cannot tell a fabricated DOI "
        "from one mistyped by a character, nor a DOI registered so recently "
        "that the indices have not caught up. A DOI that resolves to the "
        "wrong paper is bib/doi-mismatch's job, not this one's."
    ),
)
def check(ctx: Context) -> Iterator:
    # A census, reported however the run went. "218 references, 214 resolved,
    # 4 unknown" is useful at zero findings; silence is indistinguishable from
    # not having run, which is how read_repo() hid an empty world for weeks.
    from ...resolve.base import Status as _S

    total = len(ctx.paper.bib)
    settled = sum(
        1
        for e in ctx.paper.bib
        if (r := ctx.paper.resolutions.get(e.key)) is not None and r.checkable
    )
    with_doi = sum(1 for e in ctx.paper.bib if e.doi)

    for entry in ctx.paper.bib:
        # No DOI, no claim to check. A title that fails to match is weak
        # evidence about the world and strong evidence about our coverage;
        # bib/unindexed reports it, off by default.
        if not entry.doi:
            continue

        resolution = ctx.paper.resolutions.get(entry.key)
        if resolution is None or resolution.status is not Status.NOT_FOUND:
            continue

        # Only the indices that actually answered. An index that was down was
        # not a search, and naming it here would inflate the evidence behind
        # an absence claim -- the one thing this rule must not do.
        where = ", ".join(resolution.queried) or "any configured index"
        missed = (
            resolution.detail
            if "could not be reached" in (resolution.detail or "")
            else ""
        )

        # A thesis or technical report with a DOI that fails to resolve is
        # still a dead DOI, but such work is more often deposited somewhere
        # these indices do not reach.
        severity = "med" if entry.likely_unindexed else "high"

        yield ctx.finding(
            severity=severity,
            message=(
                f"[{entry.key}] {entry.render()} gives the DOI {entry.doi}, "
                f"which does not resolve in {where}."
                + (f" Note that {missed}." if missed else "")
            ),
            anchors=[entry.span_for("doi", "title"), entry.span],
            fix="Verify the reference exists and correct or remove the entry.",
        )

    if total:
        noun = "reference" if total == 1 else "references"
        unknown = total - settled
        census = (
            f"{total} {noun}, {with_doi} with a DOI, {settled} looked up"
        )
        if unknown:
            census += f", {unknown} could not be looked up and were not judged"
        ctx.abstain(census)
