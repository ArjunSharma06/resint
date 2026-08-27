"""Compare sweep batches, and say when they cannot be compared.

    python tools/compare.py sweeps/batch-*.jsonl
    python tools/compare.py sweeps/batch-1.jsonl sweeps/batch-2.jsonl --rule bib/unresolved

Two things this exists to prevent.

**Reading a rate that moved for the wrong reason.** Fixing a rule between
batches means the next batch runs different code over different papers, and a
changed firing rate no longer says which caused it. Every record carries the
commit it ran on; when they differ, that is printed at the top as a warning
rather than left for someone to notice.

**Missing an implausible rate.** A rule firing on most papers is a rule with a
precision problem, not a corpus full of broken papers -- ``bib/orphans`` sat at
73% and turned out to be two-thirds false positives. Rates above the band are
flagged in every table, because nobody scanning twenty rows spots it reliably.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

#: Above this share of papers, a rule is suspected of misfiring. Chosen from
#: the one measurement available: bib/orphans fired on 73% of papers and was
#: 66% false positives, while the arithmetic rules sit near 8% and hold up.
IMPLAUSIBLE = 0.35

#: Below this, a rule is not being exercised -- which is not the same as being
#: correct. It may be right and rare, or it may never fire at all; only a
#: planted case can tell those apart.
STARVED = 0.01


@dataclass
class Batch:
    name: str
    papers: int = 0
    findings: int = 0
    crashes: int = 0
    unreadable: int = 0
    anchors: int = 0
    anchor_failures: int = 0
    commits: set = field(default_factory=set)
    by_rule: Counter = field(default_factory=Counter)
    papers_hit: Counter = field(default_factory=Counter)
    ran: set = field(default_factory=set)
    seconds: float = 0.0

    def rate(self, rule: str) -> float:
        return self.papers_hit[rule] / self.papers if self.papers else 0.0


def load(path: Path) -> Batch:
    batch = Batch(name=path.stem)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        batch.papers += 1
        batch.seconds += (record.get("timings") or {}).get("total", 0.0)
        if record.get("resint_commit"):
            batch.commits.add(record["resint_commit"])
        if record.get("error"):
            batch.crashes += 1
        if record.get("status") == "unreadable":
            batch.unreadable += 1

        audit = record.get("anchor_audit") or {}
        batch.anchors += audit.get("checked", 0)
        batch.anchor_failures += audit.get("failed", 0)
        batch.ran.update(record.get("ran") or ())

        seen = set()
        for finding in record.get("findings") or ():
            batch.findings += 1
            batch.by_rule[finding["rule"]] += 1
            seen.add(finding["rule"])
        for rule in seen:
            batch.papers_hit[rule] += 1
    return batch


def _flag(rate: float, fired: int) -> str:
    if rate >= IMPLAUSIBLE:
        return "!"
    if fired == 0:
        return "-"
    if rate <= STARVED:
        return "."
    return " "


def overview(batches: list[Batch]) -> None:
    print(f"  {'batch':<16}{'papers':>7}{'findings':>10}{'crash':>7}"
          f"{'anchors':>9}{'failed':>8}{'seconds':>9}  commit")
    for b in batches:
        commit = ", ".join(sorted(b.commits)) or "?"
        print(
            f"  {b.name:<16}{b.papers:>7}{b.findings:>10}{b.crashes:>7}"
            f"{b.anchors:>9}{b.anchor_failures:>8}{b.seconds:>9.0f}  {commit}"
        )

    every = set().union(*(b.commits for b in batches)) if batches else set()
    if len(every) > 1:
        print(
            "\n  ! these batches ran on different code "
            f"({', '.join(sorted(every))}).\n"
            "    A rate that moved cannot be attributed to the fix rather than\n"
            "    to different papers. Re-run the earlier batches before reading\n"
            "    the comparison below as a before-and-after."
        )


def per_rule(batches: list[Batch], only: str | None) -> None:
    rules = sorted(set().union(*(set(b.ran) for b in batches)) if batches else set())
    if only:
        rules = [r for r in rules if only in r]

    width = max((len(r) for r in rules), default=10) + 1
    header = f"  {'rule':<{width}}"
    for b in batches:
        header += f"{b.name[-8:]:>16}"
    print(header)
    print(f"  {'':<{width}}" + "".join(f"{'n / papers':>16}" for _ in batches))

    for rule in rules:
        line = f"  {rule:<{width}}"
        for b in batches:
            fired = b.by_rule[rule]
            hit = b.papers_hit[rule]
            mark = _flag(b.rate(rule), fired)
            line += f"{mark}{fired:>5} /{hit:>4} {b.rate(rule):>4.0%}"
        print(line)

    print(
        f"\n    !  fires on >= {IMPLAUSIBLE:.0%} of papers -- suspect precision, "
        "not a corpus of broken papers"
    )
    print("    .  fires on <= 1% -- barely exercised; a planted case is the only")
    print("       way to tell 'correct and rare' from 'never fires'")
    print("    -  never fired in this batch")


def movement(batches: list[Batch]) -> None:
    """What changed between the first and last batch, largest first."""
    if len(batches) < 2:
        return
    first, last = batches[0], batches[-1]
    rules = sorted(set(first.by_rule) | set(last.by_rule))

    rows = []
    for rule in rules:
        before, after = first.rate(rule), last.rate(rule)
        if abs(after - before) >= 0.05:
            rows.append((abs(after - before), rule, before, after))
    if not rows:
        return

    print(f"\n  moved between {first.name} and {last.name}:")
    for _, rule, before, after in sorted(rows, reverse=True):
        arrow = "up" if after > before else "down"
        print(f"    {rule:<34} {before:>4.0%} -> {after:>4.0%}  ({arrow})")


def coverage(batches: list[Batch], registry_rules: set[str]) -> None:
    ran = set().union(*(b.ran for b in batches)) if batches else set()
    fired = {r for b in batches for r in b.by_rule}

    print(f"\n  {len(ran)} of {len(registry_rules)} rules executed; "
          f"{len(fired)} produced a finding")
    never = sorted(registry_rules - ran)
    if never:
        print(f"    never executed: {', '.join(never)}")
    silent = sorted(ran - fired)
    if silent:
        print(f"    executed but silent: {', '.join(silent)}")
        print("      -- correct and rare, or broken. Only a planted case tells you.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("paths", nargs="+", help="sweep JSONL files, in order")
    parser.add_argument("--rule", default=None, help="only rules containing this")
    args = parser.parse_args(argv)

    expanded: list[Path] = []
    for pattern in args.paths:
        matched = sorted(glob.glob(pattern))
        expanded.extend(Path(m) for m in matched) if matched else expanded.append(Path(pattern))

    batches = [load(p) for p in expanded if p.is_file()]
    if not batches:
        print("compare: no sweep files found", file=sys.stderr)
        return 2

    print()
    overview(batches)
    print()
    per_rule(batches, args.rule)
    movement(batches)

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    try:
        from resint.rules import load_all

        coverage(batches, {r.id for r in load_all().all()})
    except ImportError:
        pass

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
