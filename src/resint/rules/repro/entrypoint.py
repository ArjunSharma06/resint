"""repro/entrypoint-missing -- the README's command points at nothing.

A reader's first action is to copy the command out of the README. When the
file it names is absent they conclude the repository is broken before reading
a line of it, and they are usually right: a command that was never run is a
command nobody checked.

This asks only whether the target exists. Whether it runs is a question for
the execution tier, and implying otherwise would overstate what a static read
can know.
"""

from __future__ import annotations

from typing import Iterator

from ..registry import Context, rule


@rule(
    id="repro/entrypoint-missing",
    severity="med",
    tier="deterministic",
    requires=["repo.entrypoints"],
    cannot_detect=(
        "Whether an entrypoint that exists actually runs, and commands "
        "assembled dynamically or documented outside the README. Only "
        "existence is checked, never behaviour."
    ),
)
def check(ctx: Context) -> Iterator:
    for entry in ctx.repo.entrypoints:
        if entry.exists:
            continue
        yield ctx.finding(
            message=(
                f"The README documents {entry.command!r}, but {entry.target} "
                "does not exist in the repository."
            ),
            anchors=[entry.span],
            absent_from="the repository tree",
            fix=f"Add {entry.target}, or correct the command in the README.",
        )
