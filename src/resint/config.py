"""Project configuration and suppression.

A suppression requires a stated reason. That is the whole design: a linter
that lets findings be silenced without explanation accumulates a config file
nobody can audit, and in this domain the config file is the record of every
judgement an author made about their own work.

Suppression happens at the reporting layer, never during evaluation. The
finding is still produced, still counted, and still present in JSON output
marked with its reason. A suppression can therefore never hide a regression
from the corpus, and every suppression doubles as a labelled false positive
for the rule that produced it.

The parser is a small YAML subset rather than a dependency. resint installs
with nothing but the standard library, and a config format that needs a
third-party parser would trade that away for two levels of nesting.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

CONFIG_NAMES = (".resint.yml", ".resint.yaml")

_KEY = re.compile(r"^(?P<indent>\s*)(?P<dash>- )?(?P<key>[A-Za-z_][\w./-]*)\s*:\s*(?P<value>.*)$")
_ITEM = re.compile(r"^(?P<indent>\s*)- (?P<value>.+)$")


class ConfigError(ValueError):
    """Raised when a config file cannot be honoured as written."""


@dataclass(frozen=True, slots=True)
class Suppression:
    rule: str
    reason: str
    match: str | None = None
    expires: date | None = None

    def expired(self, today: date | None = None) -> bool:
        if self.expires is None:
            return False
        return (today or date.today()) > self.expires

    def applies_to(self, finding) -> bool:
        if finding.rule_id != self.rule:
            return False
        if self.match and self.match not in finding.message:
            return False
        return True


@dataclass
class Config:
    suppressions: list[Suppression] = field(default_factory=list)
    disabled: set[str] = field(default_factory=set)
    path: Path | None = None
    # provider/name/base_url for the model tier. Never a key: this file is
    # committed alongside the paper, and a key in it is a key on GitHub.
    model: dict[str, str] = field(default_factory=dict)

    def apply(self, findings, today: date | None = None) -> tuple[list, list[str]]:
        """Return (findings with suppressions marked, notes about the config)."""
        notes: list[str] = []
        live = [s for s in self.suppressions if not s.expired(today)]

        for stale in (s for s in self.suppressions if s.expired(today)):
            notes.append(
                f"suppression for {stale.rule} expired on {stale.expires}; "
                "the finding is reported again"
            )

        out = []
        for finding in findings:
            match = next((s for s in live if s.applies_to(finding)), None)
            out.append(finding.suppress(match.reason) if match else finding)

        # Sorted, not set order. Anything that reaches output has to be
        # deterministic or a two-run diff shows phantom churn on every paper,
        # and the diff is what makes batch-fixing safe.
        used = {s.rule for f in out if f.suppressed for s in live if s.applies_to(f)}
        for unused in sorted({s.rule for s in live} - used):
            notes.append(
                f"suppression for {unused} matched nothing; the rule may have "
                "changed or the finding may already be fixed"
            )

        return out, notes

    def enabled(self, rule_id: str) -> bool:
        return rule_id not in self.disabled


def _strip_comment(line: str) -> str:
    out, quote = [], None
    for ch in line:
        if quote:
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch == "#":
            break
        out.append(ch)
    return "".join(out)


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def parse(text: str, path: Path | None = None) -> Config:
    """Parse the .resint.yml subset: top-level keys, one list of mappings."""
    config = Config(path=path)
    section: str | None = None
    current: dict[str, str] | None = None
    raw_suppressions: list[dict[str, str]] = []

    for lineno, original in enumerate(text.splitlines(), 1):
        line = _strip_comment(original).rstrip()
        if not line.strip():
            continue

        top = _KEY.match(line)
        if top and not top.group("indent") and not top.group("dash"):
            key, value = top.group("key"), _unquote(top.group("value"))
            if key in ("suppress", "rules", "model"):
                section = key
                current = None
                continue
            section = None
            continue

        if section == "suppress":
            item = _ITEM.match(line)
            if item:
                current = {}
                raw_suppressions.append(current)
                inner = _KEY.match("  " + item.group("value"))
                if inner:
                    current[inner.group("key")] = _unquote(inner.group("value"))
                continue
            nested = _KEY.match(line)
            if nested and current is not None:
                current[nested.group("key")] = _unquote(nested.group("value"))
            continue

        if section == "model":
            nested = _KEY.match(line)
            if nested:
                config.model[nested.group("key")] = _unquote(nested.group("value"))
            continue

        if section == "rules":
            nested = _KEY.match(line)
            if nested:
                state = _unquote(nested.group("value")).lower()
                if state in ("off", "false", "no", "disabled"):
                    config.disabled.add(nested.group("key"))
            continue

    for index, entry in enumerate(raw_suppressions, 1):
        rule = entry.get("rule")
        if not rule:
            raise ConfigError(f"suppression {index} has no rule")
        reason = entry.get("reason", "").strip()
        if not reason:
            raise ConfigError(
                f"suppression {index} for {rule!r} has no reason. "
                "Every suppression must say why, so the file stays auditable."
            )
        expires = None
        if entry.get("expires"):
            try:
                expires = date.fromisoformat(entry["expires"])
            except ValueError as exc:
                raise ConfigError(
                    f"suppression {index} for {rule!r}: {exc}"
                ) from exc
        config.suppressions.append(
            Suppression(
                rule=rule, reason=reason, match=entry.get("match"), expires=expires
            )
        )

    return config


def discover(start: Path) -> Config:
    """Find the nearest config walking up from ``start``."""
    here = start if start.is_dir() else start.parent
    for directory in (here, *here.parents):
        for name in CONFIG_NAMES:
            candidate = directory / name
            if candidate.is_file():
                return parse(candidate.read_text(encoding="utf-8"), candidate)
    return Config()
