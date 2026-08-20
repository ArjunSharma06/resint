"""bib/orphans -- keys cited without an entry, entries never cited.

Both halves are absence findings, and both are cheap: no network, no model,
pure set arithmetic over what the two parsers found.

The undefined-key half is more serious than it first looks. A `\\cite{}` with
no entry renders as a bold [?] in the PDF, which means it survived to
submission without anyone reading the compiled output -- and it is a frequent
companion of references that were never real to begin with.
"""

from __future__ import annotations

from typing import Iterator

from ..registry import Context, rule


@rule(
    id="bib/orphans",
    severity="low",
    tier="deterministic",
    requires=["paper.citations", "paper.bib"],
    cannot_detect=(
        "Entries kept deliberately for a camera-ready version, and keys "
        "supplied by a bibliography style rather than the .bib file. Neither "
        "is distinguishable from an oversight by inspection alone."
    ),
)
def check(ctx: Context) -> Iterator:
    entries = {e.key: e for e in ctx.paper.bib}

    # No bibliography means nothing was looked at, which is not the same as
    # looking and finding nothing. Reporting every citation as undefined
    # because no .bib was supplied would be the absence-finding equivalent of
    # calling a reference fabricated because the network was down. The run
    # records "bibliography not checked" instead.
    if not entries:
        return

    cited: dict[str, list] = {}
    for c in ctx.paper.citations:
        cited.setdefault(c.key, []).append(c)

    bib_label = ctx.paper.bib[0].span.source.id if ctx.paper.bib else "the bibliography"

    # Cited, but no entry to render.
    for key in sorted(cited.keys() - entries.keys()):
        uses = cited[key]
        times = "once" if len(uses) == 1 else f"{len(uses)} times"
        yield ctx.finding(
            severity="med",
            message=(
                f"[{key}] is cited {times} but has no entry in {bib_label}. "
                "It will render as an unresolved marker in the compiled document."
            ),
            anchors=[u.span for u in uses[:3]],
            absent_from=bib_label,
            fix=f"Add an entry for [{key}], or remove the citation.",
        )

    # Defined, but never referenced.
    paper_label = "the paper"
    for key in sorted(entries.keys() - cited.keys()):
        entry = entries[key]
        yield ctx.finding(
            message=(
                f"[{key}] is defined in {bib_label} but never cited. "
                "It will not appear in the reference list."
            ),
            anchors=[entry.span],
            absent_from=paper_label,
            fix=f"Cite [{key}] where it is relevant, or drop the entry.",
        )
