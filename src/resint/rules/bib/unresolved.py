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

The premise was wrong until 2026-09-01, not merely the code. The rule fired
when Crossref, OpenAlex, arXiv and DBLP all missed a DOI, which reads absence
from four *metadata indices* as proof that a registration does not exist.
There are ten registration agencies. Two of nine findings on batch-1c were
live DOIs registered through the Chinese agency, resolving through chndoi.org
and unknown to all four -- reported at high severity as fabrication. The rule
was therefore biased against papers citing Chinese-language literature.

It now asks doi.org, which is the authority on whether a handle exists, and
fires only on a denial from it. Three outcomes, as everywhere else:

    doi.org 404s          the DOI does not exist         -> finding
    doi.org resolves it   real, just outside our indices -> coverage note
    doi.org unreachable   nothing is known               -> never a finding

A failed lookup is still never a finding: UNKNOWN means the network answered
badly, not that the paper is wrong.
"""

from __future__ import annotations

from typing import Iterator

from ...resolve.base import Registration, Status
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

        # The whole rule, in one line. Absence from our indices is evidence
        # about our coverage; only doi.org denying the handle is evidence
        # about the world. Firing on the former reported live DOIs registered
        # outside Crossref as fabrications -- see resolve.base.Registration.
        if not resolution.fabricated:
            continue

        # An index being down no longer bears on whether we may fire, and it
        # is worth saying why the old caveat is gone. The claim used to be
        # "no index has this", so an unsearched index left a hole in it. The
        # claim is now "the DOI system has no such handle", which one
        # authority answers on its own: DBLP being down cannot make a
        # registered DOI look unregistered. Eight of the nine findings on
        # batch-1c carried that caveat while DBLP was down for the whole
        # sweep, and under the old framing they were arguably UNKNOWN.
        # Reported anyway, because the evidence they rest on is different now.
        yield ctx.finding(
            # A thesis or technical report is more often deposited where no
            # DOI was ever minted, so a dead one there is likelier a stale
            # citation than an invention.
            severity="med" if entry.likely_unindexed else "high",
            message=(
                f"[{entry.key}] {entry.render()} gives the DOI {entry.doi}, "
                "which the DOI system does not recognise: doi.org reports no "
                "such registration with any agency."
            ),
            anchors=[entry.span_for("doi", "title"), entry.span],
            fix="Verify the reference exists and correct or remove the entry.",
        )

    # Live DOIs our indices cannot see. Not a defect in the paper, so never a
    # finding -- but worth naming, because it is the only place the tool's
    # blind spots become visible, and they fall on whole literatures at a
    # time rather than at random.
    outside = [
        (e, r)
        for e in ctx.paper.bib
        if e.doi
        and (r := ctx.paper.resolutions.get(e.key)) is not None
        and r.registration is Registration.REGISTERED
        and r.record is None
    ]
    if outside:
        agencies = sorted({r.agency for _, r in outside if r.agency})
        where = f" ({', '.join(agencies)})" if agencies else ""
        noun = "reference" if len(outside) == 1 else "references"
        ctx.abstain(
            f"{len(outside)} {noun} carry a DOI that is registered and "
            f"resolves{where} but is indexed by none of the metadata sources "
            "this tool can read; their metadata was not checked"
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
