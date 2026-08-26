"""bib/metadata-drift -- the entry resolves, but says something else.

The common cause is benign and worth catching anyway: an entry copied from a
preprint listing while the work has since appeared in proceedings, so the year
and venue are wrong in every paper that cites it. The uncommon cause is an
entry assembled from memory that happens to match a real record.

Only two fields are checked, both chosen for having an unambiguous answer.
Year is exact. Title is compared on token overlap, which tolerates
capitalization, subtitle punctuation, and brace protection while still
catching a genuinely different work. Author lists are deliberately excluded:
initials, particles, transliteration, and "and others" make string comparison
a false-positive generator.
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

        if entry.year and record.year and entry.year != record.year:
            yield ctx.finding(
                message=(
                    f"[{entry.key}] gives year {entry.year}; {record.source} has "
                    f"{record.year} for {record.title!r}. Often a preprint entry "
                    "for work that later appeared in proceedings."
                ),
                anchors=[entry.span_for("year"), entry.span],
                fix=f"Confirm the intended version and set the year accordingly.",
            )

        if entry.title and record.title:
            overlap = title_overlap(entry.title, record.title)
            if overlap < _TITLE_FLOOR:
                yield ctx.finding(
                    severity="high",
                    message=(
                        f"[{entry.key}] gives the title {entry.title!r}, but the "
                        f"record it resolves to is {record.title!r}. These are "
                        "different works."
                    ),
                    anchors=[entry.span_for("title"), entry.span],
                    fix="Check whether the key points at the intended reference.",
                )

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
