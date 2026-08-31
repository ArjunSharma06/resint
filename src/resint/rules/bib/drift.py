"""bib/metadata-drift -- the entry resolves, but says something else.

The common cause is benign and worth catching anyway: an entry copied from a
preprint listing while the work has since appeared in proceedings, so the year
and venue are wrong in every paper that cites it. The uncommon cause is an
entry assembled from memory that happens to match a real record.

Only the year is checked here, chosen for having an unambiguous answer -- and
the finding carries the corrected line, not just the complaint. A tool that
tells you the year is wrong gets read once; a tool that hands you the field to
paste gets run again.

Titles used to be compared here too, and that was the wrong home for the
check: a title disagreeing under a resolving DOI does not mean the year is
stale, it means the DOI points at a different paper. That is a separate claim
with a different fix, and it now lives in ``bib/doi-mismatch`` where it is
corroborated against the author list instead of resting on one string
comparison.
"""

from __future__ import annotations

import re
from typing import Iterator

from ...parse.bibtex import fold
from ...resolve.base import Status
from ..registry import Context, rule

_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset({"a", "an", "the", "of", "for", "and", "on", "in", "with", "to"})

# Below this share of shared content words, the two titles are different work.
_TITLE_FLOOR = 0.5


def _tokens(title: str) -> set[str]:
    return {w for w in _WORD.findall(fold(title).lower()) if w not in _STOP}


def title_overlap(left: str, right: str) -> float:
    a, b = _tokens(left), _tokens(right)
    if not a or not b:
        return 1.0  # nothing to compare on; not evidence of anything
    return len(a & b) / min(len(a), len(b))


_YEAR_DIGITS = re.compile(r"\d{4}")


def _year_digits(value: str) -> str:
    """The four-digit year, ignoring a disambiguation suffix."""
    found = _YEAR_DIGITS.search(value or "")
    return found.group(0) if found else ""


@rule(
    id="bib/metadata-drift",
    severity="med",
    tier="deterministic",
    requires=["paper.bib", "paper.resolutions"],
    cannot_detect=(
        "Which of the two records is correct when a work legitimately exists "
        "in several versions. Author lists are not compared at all, because "
        "initials and name particles make that a false-positive generator."
    ),
)
def check(ctx: Context) -> Iterator:
    # Collected rather than reported one by one. A bibliography where most
    # entries carry no DOI produces one abstention per entry, and thirty-odd
    # identical lines drown the report exactly the way thirteen separate
    # uncited-entry findings did.
    guessed: list[str] = []

    for entry in ctx.paper.bib:
        resolution = ctx.paper.resolutions.get(entry.key)
        if resolution is None or resolution.status is not Status.FOUND:
            continue
        record = resolution.record
        if record is None:
            continue

        # Only compare against a record we are certain is the right one.
        # A DOI is a claim about a single registered work; a title search
        # returns a best guess, and reporting "your year is wrong" against a
        # guess produces exactly the failure this rule exists to avoid --
        # confident, specific, and about the wrong paper. Title-matched
        # records still count as existing, so bib/unresolved stays quiet;
        # they just cannot support a claim about metadata.
        if not record.authoritative:
            guessed.append(entry.key)
            continue

        # "2015a" and "2015b" are BibTeX's disambiguation suffixes for two
        # works by the same author in one year -- a convention, not a claim
        # about the date, and the index has no equivalent. Comparing the
        # strings reported every disambiguated entry as drifted, which on a
        # bibliography using the convention is most of it.
        stated = _year_digits(entry.year)
        canonical = _year_digits(record.year)

        if stated and canonical and stated != canonical:
            yield ctx.finding(
                message=(
                    f"[{entry.key}] gives year {entry.year}; {record.source} has "
                    f"{record.year} for {record.title!r}. Often a preprint entry "
                    "for work that later appeared in proceedings."
                ),
                anchors=[entry.span_for("year"), entry.span],
                fix=(
                    "Confirm the intended version, then replace the year with: "
                    f"year = {{{record.year}}}"
                ),
            )

        # A title that disagrees under a resolving DOI is not drift -- it means
        # the DOI points at somebody else's paper, which is a different claim
        # with different evidence and a different fix. It lives in
        # bib/doi-mismatch, where it is corroborated against the author list
        # rather than resting on one string comparison.

    if guessed:
        shown = ", ".join(f"[{k}]" for k in guessed[:4])
        more = "" if len(guessed) <= 4 else f", and {len(guessed) - 4} more"
        count = "1 entry" if len(guessed) == 1 else f"{len(guessed)} entries"
        ctx.abstain(
            f"metadata not compared for {count} -- resolved by title search "
            f"rather than by DOI, so the record may be a different work: "
            f"{shown}{more}. Adding a DOI to these entries would let them be "
            "checked."
        )
