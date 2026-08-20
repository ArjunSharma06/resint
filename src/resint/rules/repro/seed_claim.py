"""repro/seed-claim -- averaged over five seeds, but only one seed exists.

The paper reports a mean and an error bar over several runs. The code fixes a
single seed and never varies it. Every interval in the paper downstream of
that claim describes one run's noise rather than the variability the reader is
being shown, and the finding says so explicitly -- the consequence is the
point, not the mismatch.

The rule is careful about what counts as varying. A seed read from a config or
an argument may well change between runs even though the source shows one
call, and a seed set inside a loop plainly does. Only a seed that is a literal
everywhere it appears is evidence of a single run.
"""

from __future__ import annotations

import re
from typing import Iterator

from ..registry import Context, rule

_MULTI_RUN = re.compile(
    r"(?:aver\w+|mean|std\w*|standard deviation|error bars?|variance)"
    r"[^.]{0,60}?(?:over|across|of)\s+(?P<count>\d+|three|four|five|ten)\s*"
    r"(?:random\s+|different\s+|independent\s+)?"
    r"(?:seeds?|runs?|trials?|repetitions?)"
    r"|(?P<count2>\d+|three|four|five|ten)\s*"
    r"(?:random\s+|different\s+|independent\s+)?(?:seeds?|runs?|trials?)"
    r"[^.]{0,40}?(?:aver\w+|mean|std|error bars?)",
    re.IGNORECASE,
)

_WORDS = {"three": 3, "four": 4, "five": 5, "ten": 10}


def _count(token: str) -> int:
    if token.lower() in _WORDS:
        return _WORDS[token.lower()]
    return int(token) if token.isdigit() else 0


@rule(
    id="repro/seed-claim",
    severity="high",
    tier="deterministic",
    requires=["paper.text", "repo.seeds"],
    cannot_detect=(
        "Seeds varied by an external sweep configuration, a scheduler, or a "
        "shell loop the repository does not contain. A seed read from a config "
        "or an argument is treated as possibly varying, so those are never "
        "reported."
    ),
)
def check(ctx: Context) -> Iterator:
    prose = ctx.paper.text
    seeds = ctx.repo.seeds
    if not prose or not seeds:
        return

    claim = _MULTI_RUN.search(prose.content)
    if not claim:
        return

    count = _count(claim.group("count") or claim.group("count2") or "0")
    if count < 2:
        return

    if any(seed.varies for seed in seeds):
        return

    literals = {seed.argument for seed in seeds if seed.argument is not None}
    if len(literals) > 1:
        return

    claim_span = prose.span(claim.start(), claim.end(), "claim")
    if claim_span is None:
        return

    fixed = next(iter(literals), "a literal")
    libraries = sorted({seed.library for seed in seeds})
    places = "one place" if len(seeds) == 1 else f"{len(seeds)} places"

    yield ctx.finding(
        message=(
            f"The paper reports results over {count} runs, but the repository "
            f"fixes a single seed ({fixed}) in {places} "
            f"({', '.join(libraries)}) and never varies it."
        ),
        anchors=[claim_span, seeds[0].span],
        fix=(
            "Vary the seed across runs, or describe the reported spread as "
            "something other than run-to-run variance."
        ),
        affects=(
            "every error bar and standard deviation downstream of this claim",
        ),
    )
