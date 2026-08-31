"""Hand-label findings, so precision becomes a number instead of a hope.

    python tools/review.py sweeps/batch-1b.jsonl --rule bib/unresolved
    python tools/review.py sweeps/*.jsonl --report

Nineteen rules with unknown precision are worth less than eight with measured
precision, and every number this project has published so far is a robustness
number: no crashes, no anchor failures, all rules executed. Those say the
plumbing works. They say nothing about whether the findings are right -- and
when precision was finally checked on one rule, ``stats/pvalue-mismatch``
turned out to be wrong every single time it fired.

So: a stratified sample, one finding at a time, four keys.

    y   correct -- the paper really has this defect
    n   wrong -- a false positive
    s   skip -- cannot tell without reading the whole paper
    q   stop, keeping everything labelled so far

Labels are stored against the finding's **fingerprint**, which survives the
paper being reformatted and the line numbers moving. Re-running the sweep does
not discard the work, and a label that no longer matches any finding is
reported rather than silently dropped -- it usually means the rule changed.
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_LABELS = Path("notes/precision-labels.json")

#: Enough per rule to separate "usually right" from "usually wrong" without
#: asking anyone to read three hundred findings. At 30 labels a rule at 90%
#: and a rule at 60% are clearly distinguishable; finer than that needs more.
SAMPLE = 30

#: Below this many labels, a percentage is theatre. Ten labels cannot separate
#: a rule at 90% from one at 60%, and printing "90%" invites a reader to
#: believe otherwise.
MIN_FOR_A_RATE = 10


def load_findings(paths) -> list[dict]:
    """Every finding across the given sweeps, tagged with its paper."""
    out = []
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            for finding in record.get("findings") or ():
                if not finding.get("fingerprint"):
                    continue  # swept before fingerprints existed
                out.append({**finding, "paper": record["paper_id"]})
    return out


def load_labels(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_labels(path: Path, labels: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(labels, indent=2, sort_keys=True), encoding="utf-8")


def stratified(findings, per_rule: int, seed: int = 0) -> list[dict]:
    """A fixed sample per rule.

    Per rule rather than overall, because a rule firing 244 times would
    otherwise consume the whole sample and the rare rules -- the ones whose
    precision is least known -- would never be looked at.

    Seeded so the same sweep yields the same sample: labelling half of one
    sample and half of another measures nothing.
    """
    by_rule = defaultdict(list)
    for finding in findings:
        by_rule[finding["rule"]].append(finding)

    chosen = []
    rng = random.Random(seed)
    for rule in sorted(by_rule):
        group = sorted(by_rule[rule], key=lambda f: f["fingerprint"])
        rng.shuffle(group)
        chosen.extend(group[:per_rule])
    return chosen


def review(findings, labels, path: Path) -> None:
    todo = [f for f in findings if f["fingerprint"] not in labels]
    if not todo:
        print("  every sampled finding is already labelled.")
        return

    print(f"  {len(todo)} to label, {len(labels)} already done.")
    print("  y = correct   n = false positive   s = skip   q = stop\n")

    for number, finding in enumerate(todo, 1):
        print("-" * 72)
        print(f"  {number}/{len(todo)}   {finding['rule']}   [{finding['severity']}]")
        print(f"  paper: {finding['paper']}")
        for anchor in finding["anchors"][:2]:
            print(f"  at   : {anchor['locate']}")
        print()
        for line in _wrap(finding["message"], 68):
            print(f"    {line}")
        if finding.get("absent_from"):
            print(f"\n    (searched: {finding['absent_from']})")
        print()

        answer = ""
        while answer not in ("y", "n", "s", "q"):
            try:
                answer = input("  correct? [y/n/s/q] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                answer = "q"

        if answer == "q":
            break

        if answer == "s":
            # Ambiguous counts against the rule, not for it -- the same
            # asymmetry as UNKNOWN never becoming a finding, applied to the
            # person labelling. A finding a careful reader cannot adjudicate
            # in three minutes is one a user will not adjudicate either.
            labels[finding["fingerprint"]] = {
                "rule": finding["rule"],
                "correct": False,
                "paper": finding["paper"],
                "why": "ambiguous",
            }
            save_labels(path, labels)
            continue

        why = ""
        if answer == "n":
            # A short tag, not prose. Collected across a rule these become the
            # fix list, which is what turns labelling from an audit into work.
            try:
                why = input("  why wrong? (short tag) ").strip()
            except (EOFError, KeyboardInterrupt):
                why = ""

        labels[finding["fingerprint"]] = {
            "rule": finding["rule"],
            "correct": answer == "y",
            "paper": finding["paper"],
            **({"why": why} if why else {}),
        }
        save_labels(path, labels)

    print(f"\n  saved {len(labels)} labels to {path}")


def _wrap(text: str, width: int) -> list[str]:
    words, lines, current = text.split(), [], ""
    for word in words:
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines


def _interval(correct: int, total: int) -> tuple[float, float]:
    """Wilson score interval at 95%.

    A range, never a single number. "9 of 10 correct" and "90 of 100 correct"
    are both 90% and mean very different things, and publishing the point
    estimate alone invites a reader to believe the first as firmly as the
    second.
    """
    if not total:
        return 0.0, 1.0
    z = 1.96
    p = correct / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    margin = (
        z * ((p * (1 - p) / total + z * z / (4 * total * total)) ** 0.5) / denom
    )
    return max(0.0, centre - margin), min(1.0, centre + margin)


def report(findings, labels) -> None:
    live = {f["fingerprint"] for f in findings}
    stale = [k for k in labels if k not in live]

    by_rule = defaultdict(Counter)
    for key, label in labels.items():
        if key in live:
            by_rule[label["rule"]]["correct" if label["correct"] else "wrong"] += 1

    fired = Counter(f["rule"] for f in findings)
    papers = {f["paper"] for f in findings}
    per_paper = {rule: fired[rule] / max(len(papers), 1) for rule in fired}

    print(
        f"\n  {'rule':<32}{'fired':>7}{'/paper':>8}"
        f"{'labelled':>10}{'precision':>22}"
    )
    for rule in sorted(fired, key=lambda r: -fired[r]):
        counts = by_rule.get(rule)
        head = f"  {rule:<32}{fired[rule]:>7}{per_paper[rule]:>8.1f}"

        if not counts:
            print(f"{head}{'--':>10}{'unmeasured':>22}")
            continue

        total = counts["correct"] + counts["wrong"]
        # Wilson on a handful of labels is noise wearing a percentage sign.
        # Say so rather than printing a number that invites belief.
        if total < MIN_FOR_A_RATE:
            print(f"{head}{total:>10}{'too few to rate':>22}")
            continue

        lo, hi = _interval(counts["correct"], total)
        print(
            f"{head}{total:>10}"
            f"{counts['correct'] / total:>12.0%}  [{lo:.0%}, {hi:.0%}]"
        )

    print(
        "\n  Findings per paper sits beside precision because it answers a\n"
        "  different question: thirty findings at 90% is a wall of text and\n"
        "  three at 90% is a useful report, and precision cannot tell them apart."
    )

    _why_tags(labels, live)
    _silent(findings, papers)

    if stale:
        print(
            f"\n  {len(stale)} label(s) match no current finding. The rule "
            "probably changed;\n  re-label rather than assuming they still hold."
        )


def _silent(findings, papers) -> None:
    """Rules that fired on nothing. Reported, never omitted.

    A rule missing from the table reads as an oversight; a rule with a
    percentage computed from two findings reads as a measurement. Neither is
    honest about a rule that legitimately finds nothing on most papers --
    which several of these do, and which is the point of them.

    The planted fixtures are what make silence interpretable -- they say the
    rule fires when it should, so a zero here is about the corpus rather than
    the code. This function does not run them, so it points at them rather
    than vouching for them.
    """
    import sys
    from pathlib import Path as _P

    sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "src"))
    try:
        from resint.rules import load_all
    except ImportError:
        return

    fired = {f["rule"] for f in findings}
    quiet = sorted(r.id for r in load_all().all() if r.id not in fired)
    if not quiet:
        return

    print("")
    print(f"  silent across {len(papers)} papers:")
    for rule in quiet:
        print(f"    {rule:<32} 0 findings; see planted fixtures and cannot_detect")
    print("")
    print("  A zero is a result, not a gap. For the stats rules it is partly")
    print("  a statement about the corpus -- GRIM's home is psychology, and a")
    print("  corpus that is mostly arXiv CS produces a small number whatever")
    print("  the rule does.")


def _why_tags(labels, live) -> None:
    """What the false positives had in common. This is the fix list."""
    tags = defaultdict(Counter)
    for key, label in labels.items():
        if key in live and not label["correct"] and label.get("why"):
            tags[label["rule"]][label["why"]] += 1
    if not tags:
        return

    print("\n  why the false positives were wrong:")
    for rule in sorted(tags):
        for tag, count in tags[rule].most_common(4):
            print(f"    {rule:<32} {count:>3}x  {tag}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("paths", nargs="+", help="sweep JSONL files")
    parser.add_argument("--labels", default=str(DEFAULT_LABELS))
    parser.add_argument("--rule", default=None, help="only this rule")
    parser.add_argument("--sample", type=int, default=SAMPLE)
    parser.add_argument("--report", action="store_true", help="summarise, do not label")
    args = parser.parse_args(argv)

    expanded = []
    for pattern in args.paths:
        matched = sorted(glob.glob(pattern))
        expanded.extend(matched or [pattern])

    findings = load_findings([p for p in expanded if Path(p).is_file()])
    if not findings:
        print("review: no findings with fingerprints found", file=sys.stderr)
        print("  (sweeps taken before fingerprints existed need re-running)",
              file=sys.stderr)
        return 2

    if args.rule:
        findings = [f for f in findings if f["rule"] == args.rule]

    path = Path(args.labels)
    labels = load_labels(path)

    if args.report:
        report(findings, labels)
        return 0

    review(stratified(findings, args.sample), labels, path)
    report(findings, labels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
