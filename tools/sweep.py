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

# .nxml and .xml are how PubMed Central serves articles. The format is
# decided by content downstream, not by this list -- these globs only say
# which files are worth opening at all.
PAPER_GLOBS = ("*.tex", "*.tar.gz", "*.tgz", "*.zip", "*.nxml", "*.xml")


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


def _identify(paper: Path, roots) -> str:
    """A stable, unique id. Corpus fixtures are all called paper.tex."""
    for root in roots if isinstance(roots, list) else [roots]:
        try:
            return paper.relative_to(root).as_posix()
        except ValueError:
            continue
    return paper.name


def _repo_for(paper: Path, repos: Path | None) -> str | None:
    """The clone belonging to this paper, if one was fetched.

    Paired by the paper's own id -- 2608.12072v1.tar.gz against a directory
    named 2608.12072v1 -- because that is the link the paper itself provided.
    """
    if repos is None:
        return None
    stem = paper.name
    for suffix in (".tar.gz", ".tgz", ".zip", ".nxml", ".xml", ".tex"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    candidate = repos / stem
    return str(candidate) if candidate.is_dir() else None


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


#: Uncommitted changes here change the findings. Everything else -- notes,
#: README, fixtures for other tests -- does not, and refusing on those would
#: train the operator to reach for the escape hatch by reflex.
_MATTERS = ("src/", "tools/sweep.py", "pyproject.toml")


def dirty_paths() -> list[str]:
    """Uncommitted code that would make this sweep's commit a lie.

    A sweep costs hours and its output is then labelled by hand, one finding at
    a time. Both are spent against a specific version of the rules, and a
    record stamped with a commit it did not actually run is worse than one
    stamped with nothing: it invites a later reader to diff two sweeps and
    attribute the difference to the commits.

    Untracked files count. ``parse/inline.py`` sat untracked for a day while
    being imported at runtime, so "tracked and modified" would have missed the
    thing most likely to be moving.
    """
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        # No git, or not a checkout. Not being able to tell is not evidence
        # of a clean tree, but refusing to sweep at all would be worse: the
        # commit is already recorded as empty, which says the same thing.
        return []
    return dirty_from_status(out.stdout)


def dirty_from_status(text: str) -> list[str]:
    """The paths in ``git status --porcelain`` output that change findings."""
    paths = []
    for line in text.splitlines():
        if not line.strip():
            continue
        # Porcelain v1: two status columns, a space, then the path. A rename
        # reads "R  old -> new", and it is the new path that is in the tree.
        # A path with a space in it arrives quoted.
        path = line[3:].strip().split(" -> ")[-1].strip('"')
        if path.startswith(_MATTERS):
            paths.append(path)
    return sorted(paths)


def corpus_ids(path: Path) -> set[str]:
    """The identifiers a corpus file names, comments and blanks dropped."""
    return {
        line.split("#", 1)[0].strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.split("#", 1)[0].strip()
    }


def paper_id_of(path: Path) -> str:
    """The identifier a cached paper's filename carries.

    Not ``Path.stem``: that leaves ``2608.1v1.tar`` behind on a ``.tar.gz``,
    and an arXiv id contains a dot of its own, so neither one suffix nor all
    of them can simply be stripped. Everything up to the first suffix that
    looks like a file extension is the id.
    """
    name = path.name
    for suffix in (".tar.gz", ".tgz", ".zip", ".nxml", ".xml", ".tex"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def keep_listed(papers, wanted: set[str]) -> list[Path]:
    """Only the papers a corpus file names."""
    return [p for p in papers if paper_id_of(p) in wanted]


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
    parser.add_argument(
        "root", nargs="+", help="directories of papers, or single papers"
    )
    parser.add_argument(
        "--ids",
        default=None,
        help=(
            "corpus file of identifiers; sweep only papers whose filename "
            "stem matches one. The cache accumulates across corpora, so "
            "without this a sweep covers whatever happens to be on disk "
            "rather than the corpus it claims to have run on."
        ),
    )
    parser.add_argument("--out", default="sweep.jsonl")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="sweep with uncommitted code; records the commit as -dirty",
    )
    parser.add_argument("--workers", type=int, default=0, help="0 = cpu_count-1")
    parser.add_argument("--timeout", type=float, default=120.0, help="seconds per paper")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing --out file that already holds records",
    )
    parser.add_argument(
        "--repos",
        default=None,
        help="directory of clones named after the paper that linked them",
    )
    parser.add_argument(
        "--resolve",
        action="store_true",
        help="look references up against Crossref and OpenAlex (opens the network)",
    )
    parser.add_argument(
        "--mailto",
        default=None,
        help="contact address for Crossref's polite pool; only sent with --resolve",
    )
    parser.add_argument(
        "--batch",
        default=None,
        help="run one slice, as N/M -- e.g. 2/5 for the second of five",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="enable the model rules, e.g. groq/openai/gpt-oss-120b",
    )
    parser.add_argument(
        "--model-workers",
        type=int,
        default=3,
        help=(
            "workers to use when --model is set. Deliberately low: every "
            "worker holds its own provider and its own pacing, so fifteen of "
            "them multiply the request rate by fifteen and every one gets "
            "rate limited."
        ),
    )
    args = parser.parse_args(argv)

    roots = [Path(r) for r in args.root]
    groups = [find_papers(r) for r in roots]

    if args.ids:
        source = Path(args.ids)
        if not source.is_file():
            print(f"sweep: no such corpus file: {source}", file=sys.stderr)
            return 2
        wanted = corpus_ids(source)
        groups = [keep_listed(g, wanted) for g in groups]
        found = sum(len(g) for g in groups)
        print(f"  corpus {source.name}: {len(wanted)} listed, {found} on disk")
        if found < len(wanted):
            print(
                f"  {len(wanted) - found} listed papers are not in these "
                "roots and will not be swept",
                file=sys.stderr,
            )

    # Interleave rather than concatenate. A batch of 75 taken off the front of
    # a concatenated list would be 75 arXiv papers, and a run that only ever
    # sees one format proves nothing about the other.
    papers: list[Path] = []
    for index in range(max((len(g) for g in groups), default=0)):
        for group in groups:
            if index < len(group):
                papers.append(group[index])

    if args.batch:
        try:
            number, _, total = args.batch.partition("/")
            number, total = int(number), int(total)
        except ValueError:
            print("sweep: --batch wants N/M, e.g. 2/5", file=sys.stderr)
            return 2
        if not 1 <= number <= total:
            print(f"sweep: batch {number} is not within 1..{total}", file=sys.stderr)
            return 2
        size = -(-len(papers) // total)  # ceiling, so the last batch is short
        papers = papers[(number - 1) * size : number * size]

    if args.limit:
        papers = papers[: args.limit]
    if not papers:
        print(f"sweep: no papers under {args.root}", file=sys.stderr)
        return 2

    import os

    spec = None
    if args.model:
        provider, _, name = args.model.partition("/")
        if not name:
            print("sweep: --model wants provider/name, e.g. groq/llama3", file=sys.stderr)
            return 2
        spec = {"provider": provider, "name": name}


    workers = args.workers or max(1, (os.cpu_count() or 2) - 1)
    if spec and not args.workers:
        # Parsing is CPU-bound and wants every core; calling a hosted model is
        # rate-limited and wants very few. When both run, the limit governs.
        workers = args.model_workers
    commit = git_commit()
    dirty = dirty_paths()
    if dirty and not args.allow_dirty:
        print(
            f"sweep: {len(dirty)} uncommitted change(s) under src/ or "
            "tools/sweep.py.",
            file=sys.stderr,
        )
        for path in dirty[:5]:
            print(f"    {path}", file=sys.stderr)
        if len(dirty) > 5:
            print(f"    ... and {len(dirty) - 5} more", file=sys.stderr)
        print(
            "  This run would be stamped with a commit it did not execute.",
            file=sys.stderr,
        )
        print(
            "  Commit, stash, or pass --allow-dirty to record it as "
            "unreproducible.",
            file=sys.stderr,
        )
        return 2
    if dirty:
        # Recorded, not hidden. Every record in the file carries the mark.
        commit = f"{commit or 'unknown'}-dirty"
    out_path = Path(args.out)

    # A sweep is expensive and interruptible, so its output is the record of
    # work already paid for. Opening it "w" truncated that on any retry:
    # batch 1 stopped at 68 of 71, and re-running it would have destroyed all
    # 68 before the first paper of the retry finished.
    if out_path.exists() and out_path.stat().st_size > 0 and not args.force:
        done = sum(1 for line in out_path.open(encoding="utf-8") if line.strip())
        print(
            f"sweep: {out_path} already holds {done} record(s).",
            file=sys.stderr,
        )
        print(
            "  Writing here would discard them. Use a different --out, or "
            "--force to overwrite.",
            file=sys.stderr,
        )
        return 2
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lookup = {"mailto": args.mailto} if args.resolve else None
    if args.mailto and not args.resolve:
        print("sweep: --mailto does nothing without --resolve", file=sys.stderr)

    repos = Path(args.repos) if args.repos else None
    paired = sum(1 for p in papers if _repo_for(p, repos)) if repos else 0

    label = f" · {args.model}" if spec else ""
    if repos:
        label += f" · {paired} with repos"
    if args.batch:
        label += f" · batch {args.batch}"
    if lookup:
        label += " · resolving"
    print(f"  {len(papers)} papers · {workers} workers · {commit or 'no commit'}{label}")
    started = time.perf_counter()
    records: list[PaperRecord] = []

    with out_path.open("w", encoding="utf-8") as handle:
        # max_tasks_per_child bounds memory drift without a recycling loop.
        # No max_tasks_per_child, and this is demonstrated rather than
        # inferred. A minimal reproduction -- max_tasks_per_child=5, one
        # worker, twelve trivial tasks -- hangs before completing the fifth on
        # CPython 3.13.1 / Windows: the retired worker exits, no replacement
        # spawns, and the parent blocks in wait() indefinitely. Even pool
        # shutdown never returns.
        #
        # That is what stalled a sweep at exactly 25 completions, and almost
        # certainly what stopped an earlier batch three papers short, which was
        # misread at the time as the session ending.
        #
        # Ruled out on the way: the Pacer, connection reuse, and the resolver's
        # thread pool -- none of which are involved, and all of which would have
        # been far worse, being live in ordinary `resint check` runs.
        #
        # The memory drift it guarded against is real but small, and a bounded
        # run exits anyway. A deadlock is worse than a large process.
        with ProcessPoolExecutor(max_workers=workers) as pool:
            pending = {
                pool.submit(
                    check_one,
                    str(p),
                    paper_id=_identify(p, roots),
                    commit=commit,
                    model=spec,
                    repo_path=_repo_for(p, repos),
                    resolve=lookup,
                ): p
                for p in papers
            }
            outstanding = set(pending)

            # Progress, not a total budget. The old deadline was
            # timeout x papers -- six hours for this run -- so anything that
            # hung was indistinguishable from work for most of a working day.
            # What actually matters is whether the run is still finishing
            # papers, and a stall shows up in seconds rather than hours.
            stalled_after = max(args.timeout, 120.0)

            while outstanding:
                done, outstanding = wait(
                    outstanding, timeout=stalled_after, return_when=FIRST_COMPLETED
                )
                if not done:
                    print("", file=sys.stderr)
                    print(
                        f"  sweep: nothing completed in {stalled_after:g}s -- "
                        f"stopping with {len(records)} of {len(papers)} done.",
                        file=sys.stderr,
                    )
                    break
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
