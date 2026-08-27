"""Run resint over a directory of papers and record what happened.

    python tools/sweep.py corpus/ --out sweeps/first.jsonl
    python tools/sweep.py ~/.cache/resint/eprints --out sweeps/run2.jsonl

Processes rather than threads, for three independent reasons: the parse layer
is CPU-bound pure Python so threads buy nothing under the GIL; a runaway regex
has to be killable, and you cannot kill a thread; and the per-character offset
array a Paper holds is reclaimed deterministically when a process exits.

The parent is the only writer, appending one line per completed paper. Kill the
run at paper 180 of 250 and you keep 180 valid records.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from resint.sweep import PaperRecord, check_one, write_record  # noqa: E402

PAPER_GLOBS = ("*.tex", "*.tar.gz", "*.tgz", "*.zip")


def find_papers(root: Path) -> list[Path]:
    """Every paper under ``root``, one per directory where a corpus fixture."""
    if root.is_file():
        return [root]

    found: list[Path] = []
    for pattern in PAPER_GLOBS:
        found.extend(root.rglob(pattern))

    # A corpus fixture is a directory holding paper.tex plus its repo/; the
    # repo's own .tex files, if any, are not papers.
    return sorted(p for p in found if "repo" not in p.parts)


def _identify(paper: Path, root: Path) -> str:
    """A stable, unique id. Corpus fixtures are all called paper.tex."""
    try:
        return paper.relative_to(root).as_posix()
    except ValueError:
        return paper.name


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _summarise(records: list[PaperRecord]) -> None:
    total = len(records)
    crashed = [r for r in records if r.crashed]
    unreadable = [r for r in records if r.status == "unreadable"]
    findings = sum(len(r.findings) for r in records)

    audited = sum(r.anchor_audit.get("checked", 0) for r in records)
    failed = sum(r.anchor_audit.get("failed", 0) for r in records)

    print(f"\n  {total} papers · {findings} findings")
    print(f"  anchors    {audited} checked, {failed} failed")
    print(f"  unreadable {len(unreadable)}")
    print(f"  crashed    {len(crashed)}")

    if crashed:
        # Grouped by fingerprint: forty crashes are usually three bugs.
        groups: dict[str, list[PaperRecord]] = {}
        for r in crashed:
            groups.setdefault((r.error or {}).get("fingerprint", "?"), []).append(r)
        print(f"\n  {len(groups)} distinct crash(es):")
        for fp, group in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            err = group[0].error or {}
            print(f"    [{fp}] x{len(group):<4} {err.get('type')}: {err.get('message', '')[:70]}")
            print(f"              first: {group[0].paper_id}")

    if failed:
        print("\n  anchor failures:")
        seen = 0
        for r in records:
            for f in r.anchor_audit.get("failures", []):
                if seen >= 10:
                    break
                print(f"    {r.paper_id}: {f['rule_id']} {f['reason']}")
                seen += 1

    # Slice coverage: an extractor that finds nothing on most papers is broken.
    ok = [r for r in records if r.status == "ok"]
    if ok:
        print("\n  slice coverage (papers where the extractor found anything):")
        names = sorted({k for r in ok for k in r.slice_census if k != "text_chars"})
        for key in names:
            # Denominator counts only papers that asked for the slice. A slice
            # nobody requested would otherwise read as zero, which looks
            # exactly like an extractor that is broken.
            asked = [r for r in ok if key in r.slice_census]
            hits = sum(1 for r in asked if r.slice_census[key])
            bar = "#" * round(20 * hits / len(asked)) if asked else ""
            print(f"    {key:<16} {hits:>4}/{len(asked):<4} {bar}")

        # Reported separately because it is measured in characters, not items,
        # so it does not belong on a bar chart of "papers where the extractor
        # found anything".
        sized = [r.slice_census["text_chars"] for r in ok if r.slice_census.get("text_chars")]
        if sized:
            sized.sort()
            print(f"    {'text':<16} {len(sized):>4}/{len(ok):<4} "
                  f"median {sized[len(sized) // 2]:,} chars")

        # The census names a slice "text_chars" where the requirement is called
        # "paper.text", so a straight set difference reported text as never
        # populated on every run -- while every prose rule was working from it.
        # A false alarm here costs someone an afternoon hunting a bug that is
        # not there, which is the same failure the census exists to prevent.
        measured = set(names) | {"text"}
        never = {s.split(".", 1)[1] for r in ok for s in r.needs} - measured
        if never:
            print(f"    (requested but never populated: {', '.join(sorted(never))})")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("root", help="directory of papers, or a single paper")
    parser.add_argument("--out", default="sweep.jsonl")
    parser.add_argument("--workers", type=int, default=0, help="0 = cpu_count-1")
    parser.add_argument("--timeout", type=float, default=120.0, help="seconds per paper")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)

    papers = find_papers(Path(args.root))
    if args.limit:
        papers = papers[: args.limit]
    if not papers:
        print(f"sweep: no papers under {args.root}", file=sys.stderr)
        return 2

    import os

    workers = args.workers or max(1, (os.cpu_count() or 2) - 1)
    commit = git_commit()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"  {len(papers)} papers · {workers} workers · {commit or 'no commit'}")
    started = time.perf_counter()
    records: list[PaperRecord] = []

    with out_path.open("w", encoding="utf-8") as handle:
        # max_tasks_per_child bounds memory drift without a recycling loop.
        with ProcessPoolExecutor(
            max_workers=workers, max_tasks_per_child=25
        ) as pool:
            pending = {
                pool.submit(
                    check_one, str(p), paper_id=_identify(p, Path(args.root)), commit=commit
                ): p
                for p in papers
            }
            outstanding = set(pending)
            deadline = time.perf_counter() + args.timeout * len(papers)

            while outstanding:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                done, outstanding = wait(
                    outstanding, timeout=remaining, return_when=FIRST_COMPLETED
                )
                for future in done:
                    paper = pending[future]
                    try:
                        record = PaperRecord.from_dict(future.result())
                    except Exception as exc:  # the worker itself died
                        record = PaperRecord(
                            paper_id=paper.name,
                            status="error",
                            error={"type": type(exc).__name__, "message": str(exc)[:300]},
                        )
                    records.append(record)
                    write_record(handle, record)
                    print(
                        f"\r  {len(records)}/{len(papers)}  {record.paper_id[:44]:<46}",
                        end="", file=sys.stderr, flush=True,
                    )

            for future in outstanding:
                future.cancel()
                record = PaperRecord(paper_id=pending[future].name, status="timeout")
                records.append(record)
                write_record(handle, record)

    print(f"\r{' ' * 60}\r", end="", file=sys.stderr)
    _summarise(records)
    print(f"\n  {time.perf_counter() - started:.1f}s · {out_path}")

    return 1 if any(r.crashed for r in records) else 0


if __name__ == "__main__":
    raise SystemExit(main())
