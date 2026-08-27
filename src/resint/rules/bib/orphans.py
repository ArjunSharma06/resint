"""bib/orphans -- keys cited without an entry, entries never cited.

Both halves are absence findings, and both are cheap: no network, no model,
pure set arithmetic over what the two parsers found.

The undefined-key half is more serious than it first looks. A ``\\cite{}`` with
no entry renders as a bold [?] in the PDF, which means it survived to
submission without anyone reading the compiled output -- and it frequently
travels with references that were never real to begin with.

The uncited half is grouped into a single finding rather than one per entry.
A working bibliography routinely carries a dozen entries the draft has not
reached yet; emitting thirteen separate findings for that buries everything
else in the report and teaches the reader to skim past the whole tool. It is
one situation, so it is one finding.
"""

from __future__ import annotations

from typing import Iterator

from ..registry import Context, rule

# Beyond this many names, the message stops being scannable and the full list
# belongs in JSON output instead.
_NAMED = 8


@rule(
    id="bib/orphans",
    severity="low",
    tier="deterministic",
    requires=["paper.citations", "paper.bib"],
    cannot_detect=(
        "Entries kept deliberately for a camera-ready version or a companion "
        "paper, and keys supplied by a bibliography style rather than the .bib "
        "file. Neither is distinguishable from an oversight by inspection. It "
        "also cannot see citations produced by a macro the paper defines "
        "itself: a definition is skipped as a template, so a key only ever "
        "passed through that macro reads as uncited."
    ),
)
def check(ctx: Context) -> Iterator:
    entries = {e.key: e for e in ctx.paper.bib}

    # No bibliography means nothing was looked at, which is not the same as
    # looking and finding nothing. Reporting every citation as undefined
    # because no .bib was supplied would be the absence-finding equivalent of
    # calling a reference fabricated because the network was down.
    if not entries:
        return

    cited: dict[str, list] = {}
    for c in ctx.paper.citations:
        cited.setdefault(c.key, []).append(c)

    bib_label = ctx.paper.bib[0].span.source.id

    # Cited, but no entry to render. One finding each: every occurrence is a
    # distinct broken reference in the compiled document.
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

    # Defined, but never referenced. One finding for all of them.
    unused = sorted(entries.keys() - cited.keys())
    if not unused:
        return

    shown = ", ".join(f"[{k}]" for k in unused[:_NAMED])
    more = "" if len(unused) <= _NAMED else f", and {len(unused) - _NAMED} more"
    count = "1 entry is" if len(unused) == 1 else f"{len(unused)} entries are"

    # What happens to an uncited entry depends entirely on how the
    # bibliography is built, and the two outcomes are opposites.
    #
    # BibTeX reads a .bib and emits only what was cited, so an uncited entry
    # silently vanishes -- harmless, and routine with a shared .bib.
    #
    # A thebibliography environment is a list: LaTeX typesets every \bibitem
    # in it, cited or not. So an uncited entry *does* appear, in a reference
    # list where nothing points to it.
    #
    # This rule told every paper the first story. On 143 findings across 204
    # real papers it was telling the second kind of paper the exact opposite
    # of what would happen -- and a finding that misdescribes the evidence is
    # worse than no finding, because the reader cannot tell which to trust.
    # A JATS <ref> behaves like a \bibitem, not like a .bib entry: the
    # reference list is the article's own furniture and every entry in it is
    # typeset. Reading only "bibitem" here told six real PubMed Central
    # articles that BibTeX would drop the entry, in documents where no BibTeX
    # exists -- the same class of false statement this branch was added to fix,
    # reintroduced by a format the branch predated.
    typeset_regardless = all(e.entry_type in ("bibitem", "ref") for e in ctx.paper.bib)

    if typeset_regardless:
        listing = (
            "a reference list"
            if any(e.entry_type == "ref" for e in ctx.paper.bib)
            else "a thebibliography environment"
        )
        consequence = (
            f"and {listing} typesets every entry, so they will appear in the "
            "reference list with nothing pointing at them"
        )
        remedy = "Cite them where relevant, or remove the entries."
    else:
        consequence = (
            "so BibTeX will drop them and they will not appear in the "
            "reference list"
        )
        remedy = "Cite them where relevant, or drop them from the bibliography."

    yield ctx.finding(
        message=(
            f"{count} defined in {bib_label} but never cited, {consequence}: "
            f"{shown}{more}."
        ),
        anchors=[entries[k].span for k in unused[:3]],
        absent_from="the paper",
        fix=remedy,
    )
