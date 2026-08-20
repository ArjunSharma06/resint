"""Terminal output.

Three things in this format are load-bearing rather than decorative. The tick
marks a finding as computed rather than judged, so a reader learns within one
run which findings need no second-guessing. The location line names both
sides of the comparison, so nothing has to be looked up to verify the claim.
And the unchecked block reports what the tool could not see, because "no
findings" and "did not look" are different statements.
"""

from __future__ import annotations

import os
import sys

from ..engine import Report
from ..ir.finding import Finding, Severity, Tier

_ANSI = {
    "high": "\033[31m",
    "med": "\033[33m",
    "low": "\033[90m",
    "ok": "\033[32m",
    "dim": "\033[90m",
    "id": "\033[36m",
    "off": "\033[0m",
}

_LABEL = {Severity.HIGH: "high", Severity.MED: " med", Severity.LOW: " low"}


def _supports_colour(stream) -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return hasattr(stream, "isatty") and stream.isatty()


def _supports_unicode(stream) -> bool:
    """Whether the stream can actually encode the tick.

    Windows consoles still default to cp1252, where a bare unicode tick
    raises rather than degrading. Crashing on output would be an absurd way
    to fail a linter, so the mark drops to ASCII instead.
    """
    encoding = getattr(stream, "encoding", None)
    if not encoding:
        return False
    try:
        "✓".encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return False
    return True


def marks(stream) -> tuple[str, str]:
    """(deterministic, model-assisted) tier marks for this stream."""
    return ("✓", "~") if _supports_unicode(stream) else ("+", "~")


class _Paint:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def __call__(self, key: str, text: str) -> str:
        if not self.enabled:
            return text
        return f"{_ANSI[key]}{text}{_ANSI['off']}"


def render_finding(
    f: Finding, paint: _Paint, tier_marks: tuple[str, str]
) -> list[str]:
    sev = f.severity.value
    det, judged = tier_marks
    mark = (
        paint("ok", det)
        if f.tier is Tier.DETERMINISTIC
        else paint("dim", judged)
    )
    head = (
        f"  {paint(sev, _LABEL[f.severity])}  {mark} "
        f"{paint('id', f.rule_id):<38} {paint('dim', f.locate())}"
    )

    lines = [head]
    for line in _wrap(f.message, 74):
        lines.append(f"        {line}")
    if f.affects:
        lines.append(paint("dim", f"        affects: {', '.join(f.affects)}"))
    if f.fix:
        lines.append(paint("dim", f"        fix: {f.fix}"))
    if f.suppressed:
        lines.append(paint("dim", f"        suppressed: {f.suppressed_reason}"))
    lines.append("")
    return lines


def _wrap(text: str, width: int) -> list[str]:
    words, out, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            out.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        out.append(cur)
    return out


def render(
    report: Report,
    title: str,
    elapsed: float,
    stream=None,
    used_provider: bool = False,
) -> str:
    stream = stream or sys.stdout
    paint = _Paint(_supports_colour(stream))
    tier_marks = marks(stream)

    lines = ["", paint("dim", title), ""]

    visible = [f for f in report.findings if not f.suppressed]
    for f in visible:
        lines.extend(render_finding(f, paint, tier_marks))

    if not visible:
        lines.append(f"  {paint('ok', 'No findings.')}")
        lines.append("")

    for note in report.unchecked:
        lines.append(paint("dim", f"  unchecked: {note}"))

    if report.skipped:
        reasons: dict[str, int] = {}
        for reason in report.skipped.values():
            reasons[reason] = reasons.get(reason, 0) + 1
        for reason, count in sorted(reasons.items()):
            noun = "rule" if count == 1 else "rules"
            lines.append(paint("dim", f"  skipped: {count} {noun}, {reason}"))

    key = "no API key used" if not used_provider else "model-assisted rules ran"
    sep = " · " if _supports_unicode(stream) else " | "
    lines.append(
        paint("dim", f"  {report.summary()}{sep}{elapsed:.1f}s{sep}{key}")
    )
    lines.append("")
    return "\n".join(lines)
