"""bib/doi-mismatch -- the DOI resolves, but to a different paper.

Distinct from ``bib/unresolved``, where nothing is found, and from
``bib/metadata-drift``, where the right record disagrees about a detail. Here
the lookup succeeds and returns *somebody else's work*.

Two things cause it, and both are worth catching:

*A key pointing at the wrong entry.* Someone copies a BibTeX block, edits the
title, and leaves the DOI. The bibliography then cites a paper the author has
probably never read, and every reader who follows the link lands somewhere
unexpected.

*A fabricated DOI that happens to resolve.* A plausible-looking DOI invented
by a language model has a real chance of colliding with a registered record.
``bib/unresolved`` cannot see that -- the DOI resolves, so nothing is missing.
This rule is what notices the record is about something else entirely, and as
writing becomes more model-assisted it is the failure mode that matters most.

**Two signals, never one.** A title that scores low on its own is weak
evidence: subtitles get dropped, translations differ, a chapter is cited under
its book's name. So the author list has to corroborate. If the first author
matches, this is reported as a title that needs checking rather than as the
wrong paper; only when neither title nor author lines up is it called a
mismatch. Single-signal disagreements do not reach ``high``.

Authoritative records only. A title-search match is a best guess, and
declaring "this is a different paper" against a guess is precisely the error
the rule exists to catch.
"""

from __future__ import annotations

import re
from typing import Iterator

from ...parse.bibtex import fold
from ...resolve.base import Status
from ..registry import Context, rule
from .drift import title_overlap

#: Below this, the two titles are not the same work. Deliberately low: a
#: shared subtitle or a dropped translation still scores well above it, so
#: reaching this floor means the strings have almost nothing in common.
MISMATCH_FLOOR = 0.35

#: Between the two floors is the ambiguous band -- different enough to report,
#: not different enough to call a different paper without corroboration.
DOUBTFUL_FLOOR = 0.55

_WORD = re.compile(r"[A-Za-z][A-Za-z'\-]+")

#: Name fragments that are not surnames. "van der Berg" and "Smith Jr." would
#: otherwise compare on the wrong token.
_PARTICLES = frozenset(
    {"van", "von", "de", "del", "della", "der", "den", "di", "da", "dos",
     "du", "la", "le", "el", "al", "bin", "ibn", "jr", "sr", "ii", "iii"}
)


def surname(name: str) -> str:
    """The family name, from either convention.

    BibTeX writes "Vaswani, Ashish"; an index returns "Ashish Vaswani". Both
    have to reduce to the same token or the corroborating signal is noise.
    """
    cleaned = fold(name or "").strip()
    if not cleaned:
        return ""

    # "Surname, Given" -- everything before the comma is the family name.
    if "," in cleaned:
        head = cleaned.split(",", 1)[0]
        parts = [w.lower() for w in _WORD.findall(head)]
    else:
        parts = [w.lower() for w in _WORD.findall(cleaned)]

    meaningful = [p for p in parts if p not in _PARTICLES]
    if not meaningful:
        return parts[-1] if parts else ""
    # Given-name-first convention puts the family name last; the comma form
    # has already been reduced to the family name alone.
    return meaningful[-1] if "," not in cleaned else meaningful[-1]


def authors_agree(entry_authors, record_authors) -> bool | None:
    """Whether the two author lists name any surname in common.

    None when either side is empty -- that is "cannot tell", which must not be
    read as disagreement. An entry with no author field is common and says
    nothing about whether the DOI is right.
    """
    ours = {surname(a) for a in entry_authors if surname(a)}
    theirs = {surname(a) for a in record_authors if surname(a)}
    if not ours or not theirs:
        return None
    return bool(ours & theirs)


@rule(
    id="bib/doi-mismatch",
    severity="high",
    tier="deterministic",
    requires=["paper.bib", "paper.resolutions"],
    cannot_detect=(
        "A DOI pointing at a genuinely similar paper -- a preprint against its "
        "published version, or one paper in a series -- where the titles "
        "overlap enough to look like the same work. It compares titles and "
        "surnames only, so a translated title, a chapter cited under its "
        "book's name, or an entry whose author field is empty will not reach "
        "the confidence needed to report. It also cannot check a DOI that no "
        "index resolves at all; that is bib/unresolved's job."
    ),
)
def check(ctx: Context) -> Iterator:
    for entry in ctx.paper.bib:
        if not entry.doi:
            continue

        resolution = ctx.paper.resolutions.get(entry.key)
        if resolution is None or resolution.status is not Status.FOUND:
            continue

        record = resolution.record
        # Only a DOI-matched record identifies one registered work. Against a
        # title-search guess, "this is a different paper" would be the very
        # mistake this rule reports.
        if record is None or not record.authoritative:
            continue
        if not entry.title or not record.title:
            continue

        overlap = title_overlap(entry.title, record.title)
        if overlap >= DOUBTFUL_FLOOR:
            continue

        agree = authors_agree(entry.authors, record.authors)

        if agree:
            # The right people, a different-looking title. Far more likely a
            # subtitle or a preprint/published pair than a wrong DOI.
            continue

        if overlap < MISMATCH_FLOOR and agree is False:
            severity, verdict = "high", (
                "Neither the title nor the authors match, so this DOI points "
                "at a different paper."
            )
        else:
            severity, verdict = "med", (
                "The titles disagree and the authors could not be compared, "
                "so this may be the wrong DOI."
            )

        theirs = ", ".join(record.authors[:2]) or "unlisted"
        yield ctx.finding(
            severity=severity,
            message=(
                f"[{entry.key}] cites {entry.title!r}, but the DOI "
                f"{entry.doi} resolves to {record.title!r} by {theirs} "
                f"({record.source}). {verdict}"
            ),
            anchors=[entry.span_for("doi", "title"), entry.span],
            fix=(
                "Check the DOI against the reference you meant to cite. A DOI "
                "copied from a neighbouring entry resolves perfectly well and "
                "sends every reader to the wrong paper."
            ),
        )
