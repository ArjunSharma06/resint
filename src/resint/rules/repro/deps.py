"""repro/unpinned-deps -- nothing records what actually worked.

An unpinned dependency list is a statement that the code worked against
whatever resolved on the author's machine on one particular day. Whether that
matters depends entirely on how long ago that day was, which is why this rule
sits at low severity and stays there: it is a note, not an accusation, and
treating it as more would train people to ignore the report.

The threshold is proportional rather than absolute. One unpinned package
among thirty pinned ones is an oversight; twenty among thirty means the
manifest is listing names rather than recording an environment.
"""

from __future__ import annotations

from typing import Iterator

from ..registry import Context, rule

# Below this share of unpinned dependencies the manifest is still recording
# an environment, and a finding would be noise.
UNPINNED_FLOOR = 0.5


@rule(
    id="repro/unpinned-deps",
    severity="low",
    tier="deterministic",
    requires=["repo.deps", "repo.lockfiles"],
    cannot_detect=(
        "Whether the unpinned versions still resolve to something that works, "
        "and environments captured outside the repository in a container "
        "image or a cluster module file."
    ),
)
def check(ctx: Context) -> Iterator:
    deps = ctx.repo.deps
    if not deps or ctx.repo.lockfiles:
        return

    unpinned = [d for d in deps if not d.pinned]
    if not unpinned or len(unpinned) / len(deps) < UNPINNED_FLOOR:
        return

    manifest = unpinned[0].manifest
    names = ", ".join(d.name for d in unpinned[:5])
    more = "" if len(unpinned) <= 5 else f", and {len(unpinned) - 5} more"
    anchors = [unpinned[0].span]
    if len(unpinned) > 1:
        anchors.append(unpinned[-1].span)

    yield ctx.finding(
        message=(
            f"{len(unpinned)} of {len(deps)} dependencies in {manifest} carry "
            f"no version constraint ({names}{more}), and the repository has no "
            "lockfile. Nothing records the environment the results came from."
        ),
        anchors=anchors,
        absent_from="any lockfile" if len(anchors) == 1 else None,
        fix="Pin the versions used, or commit a lockfile.",
    )
