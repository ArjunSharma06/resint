"""Which cached papers report inline null-hypothesis significance tests.

    python tools/screen_nhst.py ~/.cache/resint/pmc --keep 100

``stats/grim``, ``stats/pvalue-mismatch`` and ``stats/significance-unsupported``
have never fired on a real paper. Three explanations fit that equally well --
the rules are broken, the extraction is broken, or the corpus contained no
inline NHST -- and a corpus picked by subject area cannot tell them apart,
because "psychology journals" is not the same property as "reports t and p in
running text". Screening on the text itself removes the third explanation for
the cost of fetching more papers than are kept.

**Deliberately not our extractor.** Selecting with ``parse/extract.py`` would
choose exactly the papers that extractor already handles, and any fire rate
measured afterwards would be inflated by construction -- the corpus would be
defined as "papers we can parse" and then used as evidence that we can parse
papers. This is a regex over raw text and nothing else, and it is meant to be
cruder than the extractor rather than a second implementation of it.

That crudeness has a use beyond selection. When a paper passes this screen and
``stats/pvalue-mismatch`` still finds nothing in it, the gap between the two is
a measurement of our extraction, not of the literature.

**The threshold decides membership and nothing else does.** Papers are kept in
name order, not ranked by hit count: taking the highest-scoring N would build a
corpus of the most statistics-dense papers in medicine, and a fire rate
measured on those would then be published as a rate on papers generally. The
distribution is reported so results can be read in context, and is deliberately
not used to choose.
"""

from __future__ import annotations

import argparse
import html
import re
import sys
from pathlib import Path

#: A reported p-value. Covers "p = .03", "p<0.001", "p = 0.02", "P > .05".
#: Deliberately blind to how the number is formatted beyond the first digit.
P_VALUE = re.compile(r"\bp\s*[=<>]\s*0?\.\d", re.IGNORECASE)

#: A test statistic announcing itself. Parentheses carry the degrees of
#: freedom, which is what distinguishes "t(20) = 2.086" from a stray "t".
STATISTIC = re.compile(
    r"\b[tFrz]\s*\(\s*\d"          # t(20), F(1, 38), r(48), z(12)
    r"|\bF\s*\(\s*\d+\s*,"         # F(2, 45) with the comma
    r"|\bchi[\s-]?2|\bchi[\s-]?square|χ2|χ\s*2"   # chi2, chi-square, χ2
    r"|\bz\s*=\s*[-+]?\d"          # z = 1.96
    r"|\bt\s*=\s*[-+]?\d",         # t = 2.086, no df given
    re.IGNORECASE,
)

#: Characters between the statistic and its p-value. A sentence, roughly.
#: Wide enough for "t(20) = 2.086, 95% CI [0.1, 0.4], p = .03" and narrow
#: enough that a methods-section "p < .05" does not marry a result four
#: paragraphs away.
WINDOW = 200

_TAGS = re.compile(r"<[^>]+>")


def plain(raw: str) -> str:
    """Tags out, then entities decoded. Nothing else.

    JATS marks statistics up -- ``<italic>t</italic>(20)`` is ordinary -- so a
    match against raw XML would miss most of what is there and would be
    screening on markup style rather than content.

    The unescaping is not cosmetic. XML escapes ``<``, so every ``p < .001`` in
    the corpus is stored as ``p &lt; .001``, and a screen that skips this step
    silently loses the most common way a p-value is written. Measured: 18 of
    224 papers passed without it, 34 with.

    Order matters. Unescaping first would turn a literal ``&lt;p&gt;`` into a
    tag and then delete it.
    """
    return html.unescape(_TAGS.sub(" ", raw))


def hits(text: str) -> int:
    """How many statistic/p-value pairs sit within a sentence of each other."""
    stats = [m.start() for m in STATISTIC.finditer(text)]
    if not stats:
        return 0
    found = 0
    for match in P_VALUE.finditer(text):
        where = match.start()
        if any(abs(where - s) <= WINDOW for s in stats):
            found += 1
    return found


def screen(path: Path) -> int:
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return 0
    return hits(plain(raw))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("cache", help="directory of cached papers")
    parser.add_argument("--glob", default="*.nxml")
    parser.add_argument(
        "--ids",
        default=None,
        help=(
            "file of identifiers; screen only these. The cache accumulates "
            "across corpora, so without this a screen silently mixes pools "
            "and the resulting corpus file describes a provenance it does "
            "not have."
        ),
    )
    parser.add_argument("--keep", type=int, default=0, help="0 = all that pass")
    parser.add_argument(
        "--min-hits",
        type=int,
        default=2,
        help="pairs required; 2 rejects a lone methods-section threshold",
    )
    args = parser.parse_args(argv)

    papers = sorted(Path(args.cache).glob(args.glob))

    if args.ids:
        source = Path(args.ids)
        if not source.is_file():
            print(f"screen_nhst: no such id file: {source}", file=sys.stderr)
            return 2
        wanted = {
            line.split("#", 1)[0].strip()
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.split("#", 1)[0].strip()
        }
        papers = [p for p in papers if p.stem in wanted]

    if not papers:
        print(f"screen_nhst: nothing matching {args.glob} in {args.cache}",
              file=sys.stderr)
        return 2

    scored = [(screen(p), p) for p in papers]

    # By name, NOT by hit count. Ranking on the score and taking the top N
    # would select the most statistics-dense papers in the pool, and any fire
    # rate measured on them would be reported as a rate on papers generally.
    # The threshold decides membership; nothing else does.
    passed = sorted(
        ((n, p) for n, p in scored if n >= args.min_hits),
        key=lambda pair: pair[1].name,
    )
    kept = passed[: args.keep] if args.keep else passed

    for _, path in kept:
        print(path.stem)

    counts = sorted(n for n, _ in kept)
    spread = ""
    if counts:
        middle = counts[len(counts) // 2]
        spread = f"; hits ranged {counts[0]} to {counts[-1]}, median {middle}"

    print(
        f"\n  {len(papers)} screened, {len(passed)} carry inline NHST, "
        f"{len(papers) - len(passed)} do not, {len(kept)} kept{spread}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
