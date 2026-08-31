"""Suppressions written in the paper, beside the thing they excuse.

    % resint: ignore bib/unresolved -- textbook, no DOI exists

A judgement about one reference belongs next to that reference, not in a
config file three directories away. ``.resint.yml`` stays the right home for
project-wide decisions; this is for the single entry a checker will always be
wrong about, and it survives the bibliography being reordered because it
travels with the line.

The reason is mandatory here for the same reason it is mandatory there: a
silenced finding with no explanation is unauditable six months later, and the
config file is the record of every judgement made about the work.

Scope is the line the comment sits on and the line after it, so both of these
read naturally:

    \\bibitem{knuth1984}  % resint: ignore bib/unindexed -- a book
    % resint: ignore stats/grim -- responses were averaged before reporting
    The mean was 3.47 across participants.

A suppressed finding is still produced, still counted, and still present in
JSON output marked with its reason -- so this can never hide a regression
from the corpus.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: ``% resint: ignore <rule> -- <reason>``. The separator may be ``--`` or a
#: colon; the reason is required either way.
_DIRECTIVE = re.compile(
    r"%\s*resint:\s*ignore\s+(?P<rule>[\w./-]+)\s*(?:--|:)\s*(?P<reason>\S.*?)\s*$",
    re.IGNORECASE,
)

#: A directive with no reason. Matched separately so it can be *reported*
#: rather than silently ignored -- someone who wrote one meant to suppress
#: something, and telling them why it did not work is worth a line.
_UNREASONED = re.compile(
    r"%\s*resint:\s*ignore\s+(?P<rule>[\w./-]+)\s*$", re.IGNORECASE
)


@dataclass(frozen=True, slots=True)
class InlineSuppression:
    rule: str
    reason: str
    line: int

    def covers(self, finding) -> bool:
        """Whether this directive applies to a finding.

        The comment's own line and the one after it. A trailing comment
        annotates its line; a comment on its own annotates what follows.
        """
        if finding.rule_id != self.rule:
            return False
        return any(
            a.line in (self.line, self.line + 1) for a in finding.anchors
        )


def find_directives(text: str) -> tuple[list[InlineSuppression], list[str]]:
    """Every inline suppression in a source, and complaints about malformed ones."""
    found: list[InlineSuppression] = []
    notes: list[str] = []

    for number, line in enumerate(text.splitlines(), 1):
        match = _DIRECTIVE.search(line)
        if match:
            found.append(
                InlineSuppression(
                    rule=match.group("rule"),
                    reason=match.group("reason").strip(),
                    line=number,
                )
            )
            continue

        bare = _UNREASONED.search(line)
        if bare:
            notes.append(
                f"line {number}: 'resint: ignore {bare.group('rule')}' has no "
                "reason and was not applied. Write "
                f"'% resint: ignore {bare.group('rule')} -- why' instead."
            )

    return found, notes


def apply_inline(findings, directives) -> tuple[list, list[str]]:
    """Mark findings covered by an inline directive, and report unused ones."""
    if not directives:
        return list(findings), []

    out = []
    used: set[int] = set()
    for finding in findings:
        covering = next((d for d in directives if d.covers(finding)), None)
        if covering is None:
            out.append(finding)
            continue
        used.add(covering.line)
        out.append(finding.suppress(covering.reason))

    notes = [
        f"line {d.line}: 'resint: ignore {d.rule}' matched nothing; the "
        "finding may already be fixed"
        for d in directives
        if d.line not in used
    ]
    return out, notes
