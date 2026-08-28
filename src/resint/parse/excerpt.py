"""Choosing which part of a paper to show a model.

Every model rule used to send ``text.window(14_000)`` -- the first fourteen
thousand characters. Measured across 68 real papers, that is:

    99% of papers longer than the window
    median paper: the model sees 37%
    90th percentile: the model sees 14%

And not a random 37%. The *first* 37%, which is Introduction and Related Work.
Results, Discussion and Conclusion -- where the claims, the dataset lists and
the training details actually live -- were never sent at all. Five rules whose
whole job is reading results had been reading introductions.

So this is a correctness fix that happens to be cheaper. Selecting the sections
a rule needs sends **fewer** characters covering **more** of what matters.

Papers turn out to have usable structure: all 50 papers in a mixed arXiv/PMC
sample parsed into named sections. Where a paper does not, retrieval takes
over -- ``resolve.passages`` scores paragraphs against a query, which is the
right tool for a document with no headings and was already written.

**Offsets are unaffected.** Rules locate a model's quote against the full
``text.content``, never against the excerpt, so anchoring, line numbers and the
two-anchor invariant are untouched by what we choose to send. A quote the model
stitches across a gap this module created appears nowhere in the real paper and
is discarded by the machinery that already exists.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

#: What a rule asks for, and the headings that satisfy it. Substrings, checked
#: against a normalised name, because real headings carry qualifiers --
#: "Results and Discussion", "4.2 Experimental Setup", "Materials and Methods".
#: Drawn from the section names actually present in a 50-paper sample rather
#: than from what papers are supposed to be called.
ROLES: dict[str, tuple[str, ...]] = {
    "methods": (
        "method",
        "materials and methods",
        "methodology",
        "experimental setup",
        "setup",
        "implementation",
        "experiment",
        "procedure",
        "participants",
        "data collection",
        "statistical analysis",
    ),
    "results": (
        "result",
        "finding",
        "evaluation",
        "analysis",
        "performance",
    ),
    "discussion": (
        "discussion",
        "conclusion",
        "limitation",
        "concluding",
        "implication",
    ),
    "background": (
        "introduction",
        "background",
        "related work",
        "motivation",
    ),
}

#: Sections that are apparatus, never argument. Excluded even when a role
#: pattern would otherwise match them -- "Data availability" contains "data",
#: and "Author contributions" is never evidence of anything.
NEVER = (
    "author contribution",
    "data availability",
    "funding",
    "acknowledg",
    "ethics",
    "conflict of interest",
    "competing interest",
    "consent",
    "supplementary",
    "abbreviation",
)

#: Leading numbering in any of the forms papers use: "4", "4.2", "IV.", "A.".
_NUMBERING = re.compile(r"^\s*(?:\d+(?:\.\d+)*|[IVXLC]+|[A-Z])[.)]?\s+")

#: Marks where text was left out, so a model does not read two distant
#: sections as continuous prose and quote across the join.
GAP = "\n\n[...]\n\n"


@dataclass
class Excerpt:
    """What to send, and how it was chosen."""

    text: str
    roles: list[str] = field(default_factory=list)
    strategy: str = "sections"
    sections: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.text.strip())

    @property
    def chars(self) -> int:
        return len(self.text)


def normalise(name: str) -> str:
    """A section heading reduced to something matchable."""
    return _NUMBERING.sub("", (name or "").strip()).strip().lower()


def role_of(name: str) -> str:
    """Which role a heading satisfies, or "" for apparatus and the unknown."""
    cleaned = normalise(name)
    if not cleaned or any(bad in cleaned for bad in NEVER):
        return ""
    for role, patterns in ROLES.items():
        if any(pattern in cleaned for pattern in patterns):
            return role
    return ""


def _abstract(paper) -> tuple[str, int, int]:
    """The front matter: everything between the preamble and the first section.

    Not a section in its own right in either format -- LaTeX wraps it in an
    environment and JATS puts it in <front> -- so it is located by position
    rather than by name.
    """
    text = paper.text
    start = getattr(text, "body_start", 0) or 0
    ends = [s.start for s in (paper.sections or ()) if s.start > start]
    end = min(ends) if ends else min(len(text.content), start + 3_000)
    return text.content[start:end], start, end


def _sections_by_role(paper) -> dict[str, list]:
    found: dict[str, list] = {}
    for section in paper.sections or ():
        role = role_of(section.name)
        if role:
            found.setdefault(role, []).append(section)
    return found


def excerpt(paper, wanted, limit: int = 8_000, query: str = "") -> Excerpt:
    """The parts of ``paper`` a rule needs, within ``limit`` characters.

    ``wanted`` names roles in priority order: a rule asking for
    ``["results", "methods"]`` gets results first and methods only if the
    budget survives. Priority is the rule's, not the document's, because the
    rule knows which evidence it is actually looking for.
    """
    text = paper.text
    if not text or not text.content:
        return Excerpt(text="", strategy="empty")

    chunks: list[str] = []
    used: list[str] = []
    names: list[str] = []
    budget = limit

    by_role = _sections_by_role(paper)

    for role in wanted:
        if budget <= 0:
            break

        if role == "abstract":
            body, _, _ = _abstract(paper)
            body = body.strip()
            if body:
                take = body[:budget]
                chunks.append(f"=== Abstract ===\n{take}")
                used.append(role)
                names.append("abstract")
                budget -= len(take)
            continue

        for section in by_role.get(role, ()):
            if budget <= 0:
                break
            body = text.content[section.start : section.end].strip()
            if not body:
                continue
            # Truncated from the start: a section's opening says what it does,
            # which is the opposite end from where a whole-document window cuts.
            take = body[:budget]
            chunks.append(f"=== {section.name.strip() or role.title()} ===\n{take}")
            names.append(section.name.strip() or role)
            budget -= len(take)
        if role in by_role:
            used.append(role)

    # Named sections are a fast path, never a requirement. Plenty of real
    # papers are organised by subject rather than by IMRaD -- a maths paper
    # with "Retained-sample delay-neutrality", a clinical review with
    # "Spasticity and dystonia" -- and matching nothing left one 179,000
    # character paper with 507 characters of abstract. Worse than the
    # whole-document window this replaced.
    #
    # So whatever the headings are called, the budget gets filled: sections
    # first because they are the best evidence, then retrieval over the rest,
    # which finds the relevant paragraphs whatever the heading above them says.
    strategy = "sections" if chunks else ""

    if budget > 0 and query:
        from ..resolve.passages import retrieve

        # Ranges a section already contributed. Compared by offset rather than
        # by text: retrieval returns paragraphs sliced at its own boundaries,
        # so a substring check misses the overlap and pays twice to tell the
        # model the same thing.
        covered = [
            (s.start, s.end)
            for role in wanted
            for s in by_role.get(role, ())
        ]
        if "abstract" in wanted:
            _, a_start, a_end = _abstract(paper)
            covered.append((a_start, a_end))

        for passage in retrieve(query, text.content, k=12):
            if budget <= 0:
                break
            end = passage.start + len(passage.text)
            if any(passage.start < hi and end > lo for lo, hi in covered):
                continue
            take = passage.text[:budget]
            chunks.append(take)
            budget -= len(take)
        if not strategy:
            strategy = "retrieval"
        elif budget < limit:
            strategy = "sections+retrieval"

    if not chunks:
        # Nothing better available. The old behaviour, now named as the last
        # resort it always was rather than the default it used to be.
        return Excerpt(text=text.window(limit), strategy="leading")

    return Excerpt(
        text=GAP.join(chunks), roles=used, strategy=strategy, sections=names
    )
