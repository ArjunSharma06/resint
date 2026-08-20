"""stats/significance-unsupported -- a claim of reliability with no test.

The paper says an effect is significant, or reliably better, and reports no
test statistic, no p-value, no confidence interval, and no variance anywhere
in the document. Not a weak test -- none at all.

The bar is deliberately set at document level rather than sentence level. A
paper that runs its tests properly and describes them in a methods section
should never trip this, even if an individual sentence in the discussion
speaks loosely. That makes the rule quiet on well-run work and loud on the
specific failure it is for: a results section written in the vocabulary of
statistics with none of the substance.
"""

from __future__ import annotations

import re
from typing import Iterator

from ..registry import Context, rule

# Wording that asserts a reliable difference rather than describing one.
_CLAIM = re.compile(
    r"\b("
    r"statistically\s+significant|significantly\s+(?:better|worse|higher|lower|"
    r"outperform\w*|improv\w*|reduc\w*|increas\w*|decreas\w*)|"
    r"significant\s+(?:difference|improvement|effect|increase|decrease|gain)|"
    r"reliably\s+(?:better|outperform\w*|improv\w*)"
    r")\b",
    re.IGNORECASE,
)

# Any of these anywhere in the document counts as statistical support.
_SUPPORT = re.compile(
    r"(?<![A-Za-z])p\s*[<>=]\s*\.?\d"
    r"|(?<![A-Za-z])t\s*\(\s*\d"
    r"|(?<![A-Za-z])F\s*\(\s*\d"
    r"|chi\s*\^?\s*2"
    r"|confidence\s+interval|\bCI\b"
    r"|standard\s+(?:error|deviation)|(?<![A-Za-z])SD\b|(?<![A-Za-z])SEM\b"
    r"|\bstd\b|±|\+/-"
    r"|bootstrap|permutation\s+test|wilcoxon|mann[- ]whitney|t-test|anova"
    r"|error\s+bars?|significance\s+test",
    re.IGNORECASE,
)


@rule(
    id="stats/significance-unsupported",
    severity="med",
    tier="deterministic",
    requires=["paper.text"],
    cannot_detect=(
        "Whether a test that is reported is the appropriate one, or whether "
        "it was applied correctly. This rule only distinguishes some "
        "statistical support from none at all, which is a much weaker "
        "question than the one a reviewer would ask."
    ),
)
def check(ctx: Context) -> Iterator:
    prose = ctx.paper.text
    if not prose:
        return

    if _SUPPORT.search(prose.content):
        return

    claims = list(_CLAIM.finditer(prose.content))
    if not claims:
        return

    spans = [prose.span(m.start(), m.end(), "claim") for m in claims[:3]]
    spans = [s for s in spans if s is not None]
    if not spans:
        return

    phrase = claims[0].group(1).lower()
    times = "once" if len(claims) == 1 else f"{len(claims)} times"

    yield ctx.finding(
        severity="high" if len(claims) > 2 else "med",
        message=(
            f"The paper claims a reliable difference {times} (first: "
            f"{phrase!r}) but reports no test statistic, p-value, confidence "
            "interval, or variance anywhere in the document."
        ),
        anchors=spans if len(spans) >= 2 else [spans[0]],
        absent_from="the whole document" if len(spans) < 2 else None,
        fix=(
            "Report the test behind the claim, or describe the difference "
            "without asserting reliability."
        ),
    )
