"""repro/ghost-repo -- the linked repository holds nothing.

"Code will be released upon acceptance" is a promise the literature is full
of and the file system frequently is not. An empty repository, or one holding
only a README promising a future release, is worth reporting plainly: readers
are entitled to know before they clone it, and authors are usually better off
learning their release never landed.

The wording stays neutral throughout. The tool reports what is present, never
what it supposes about why -- there is a real difference between a release
that slipped and one that was never intended, and nothing in a file listing
distinguishes them.
"""

from __future__ import annotations

import re
from typing import Iterator

from ...ir.span import Span
from ..registry import Context, rule

_PROMISE = re.compile(
    r"(?:code|implementation|models?|weights?|data(?:set)?)\s+"
    r"(?:will\s+be|to\s+be|is|are)?\s*"
    r"(?:released|available|published|uploaded|coming)"
    r"|coming\s+soon|stay\s+tuned|under\s+construction|work\s+in\s+progress",
    re.IGNORECASE,
)

_DOC_SUFFIXES = (".md", ".txt", ".rst")


@rule(
    id="repro/ghost-repo",
    severity="high",
    tier="deterministic",
    requires=["repo.files", "repo.readme", "repo.readme_source"],
    cannot_detect=(
        "Code released under a different name or hosted elsewhere without a "
        "link from the paper, and repositories whose substantive content sits "
        "on a branch other than the one checked out."
    ),
)
def check(ctx: Context) -> Iterator:
    files = ctx.repo.files
    readme = ctx.repo.readme
    source = ctx.repo.readme_source

    if not files or source is None:
        return

    code = [
        f
        for f in files
        if not f.lower().endswith(_DOC_SUFFIXES)
        and "license" not in f.lower()
        and not f.startswith(".")
    ]
    if code:
        return

    count = f"{len(files)} file" + ("" if len(files) == 1 else "s")
    promise = _PROMISE.search(readme) if readme else None

    if promise:
        span = Span(
            source,
            promise.start(),
            promise.end(),
            line=readme.count("\n", 0, promise.start()) + 1,
            label="README",
        )
        message = (
            f"The repository contains no code -- {count}, all documentation -- "
            f"and the README states {promise.group(0)!r}."
        )
        severity = "high"
    else:
        span = Span(source, 0, max(min(len(readme), 40), 1), line=1, label="README")
        message = f"The repository contains no code: {count}, all documentation."
        severity = "med"

    yield ctx.finding(
        severity=severity,
        message=message,
        anchors=[span],
        absent_from="the repository tree",
        fix="Publish the code, or remove the link from the paper.",
    )
