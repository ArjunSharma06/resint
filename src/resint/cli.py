"""Command line entry point.

Deliberately thin. Every surface after this one -- GitHub Action, MCP server,
editor extension -- wraps the same library, so any analysis that leaks into
here is analysis that has to be rewritten for surface two. Nothing in this
module decides anything about a paper; it parses arguments, calls the
library, and formats what comes back.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import __version__
from .config import Config, ConfigError, discover, parse as parse_config
from .engine import plan, run
from .ir.finding import Severity
from .parse.acquire import UnreadableInput
from .parse.document import paper_from_path
from .parse.repo import read_repo
from .report.sarif import render as render_sarif
from .report.terminal import render
from .resolve import CachingResolver, NullResolver
from .resolve.http import HttpResolver
from .rules import load_all

EXIT_OK = 0
EXIT_FINDINGS = 1
EXIT_USAGE = 2


def _load_config(args: argparse.Namespace, target: Path) -> Config:
    if args.no_config:
        return Config()
    if args.config:
        path = Path(args.config)
        if not path.is_file():
            raise ConfigError(f"no such config file: {path}")
        return parse_config(path.read_text(encoding="utf-8"), path)
    return discover(target)


def _cmd_check(args: argparse.Namespace) -> int:
    target = Path(args.target)
    if not target.exists():
        print(f"resint: no such file: {target}", file=sys.stderr)
        return EXIT_USAGE

    try:
        config = _load_config(args, target)
    except ConfigError as exc:
        print(f"resint: {exc}", file=sys.stderr)
        return EXIT_USAGE

    registry = load_all()

    # Decide what will run before loading anything. Without this the loader
    # has no idea what is wanted and builds every slice -- so a run with the
    # bib rules switched off still parses the bibliography and still opens
    # sockets. One plan drives both the loading and the running, so the two
    # cannot disagree.
    chosen = plan(
        registry,
        config,
        has_repo=bool(args.repo),
        has_provider=False,
    )

    # Only bibliographic metadata leaves the machine -- titles and DOIs of
    # works the paper already cites publicly. The manuscript itself is never
    # transmitted. --offline skips it entirely, at the cost of the reference
    # rules abstaining rather than reporting.
    resolver = (
        NullResolver()
        if args.offline or not chosen.opens_network
        else CachingResolver(HttpResolver(mailto=args.mailto))
    )

    # Resolution is the only phase that can take real time, so it is the
    # only one that reports progress. Written to stderr so --format json
    # and --format sarif stay pipeable.
    def progress(done: int, total: int) -> None:
        if args.format == "term" and sys.stderr.isatty():
            # Overwriting with spaces leaves a line of whitespace behind once
            # the report's own leading newline moves past it. ANSI erase-line
            # actually clears it, and we are already inside an isatty guard.
            tail = "" if done < total else "\r\033[2K"
            print(
                f"\r  resolving references\u2026 {done}/{total}",
                end=tail,
                file=sys.stderr,
                flush=True,
            )

    started = time.perf_counter()
    try:
        paper = paper_from_path(
            target,
            needs=chosen.paper_slices,
            bib=args.bib,
            resolver=resolver,
            progress=progress,
        )
    except UnreadableInput as exc:
        # A sentence, never a traceback. The user pointed at the wrong thing;
        # that is a usage error, not a crash.
        print(f"resint: {exc}", file=sys.stderr)
        return EXIT_USAGE

    # Only build the repository IR when a repository was actually given.
    # Walking a tree nobody asked about is the kind of cost that makes a
    # linter feel slow for no benefit.
    repo = None
    if args.repo:
        repo = read_repo(args.repo, needs=chosen.repo_slices)

    report = run(
        paper,
        repo=repo,
        registry=registry,
        min_severity=Severity(args.min_severity) if args.min_severity else None,
        config=config,
        prepared=chosen,
    )
    elapsed = time.perf_counter() - started

    if args.format == "json":
        print(
            json.dumps(
                {
                    "version": __version__,
                    "target": str(target),
                    "findings": [f.to_dict() for f in report.findings],
                    "unchecked": report.unchecked,
                    "notes": report.notes,
                    "skipped": report.skipped,
                    "counts": report.counts(),
                },
                indent=2,
            )
        )
    elif args.format == "sarif":
        print(render_sarif(report, registry, version=__version__))
    else:
        print(render(report, target.name, elapsed))

    if args.fail_on == "none":
        return EXIT_OK
    counts = report.counts()
    floor = {"high": ("high",), "med": ("high", "med"), "low": ("high", "med", "low")}
    triggered = any(counts[level] for level in floor[args.fail_on])
    return EXIT_FINDINGS if triggered else EXIT_OK


def _cmd_rules(args: argparse.Namespace) -> int:
    registry = load_all()
    rules = registry.all()
    if args.tier:
        rules = [r for r in rules if r.tier.value == args.tier]
    if args.family:
        rules = [r for r in rules if r.family == args.family]

    if args.format == "json":
        print(
            json.dumps(
                [
                    {
                        "id": r.id,
                        "severity": r.severity.value,
                        "tier": r.tier.value,
                        "requires": list(r.requires),
                        "needs_repo": r.needs_repo,
                        "cannot_detect": r.cannot_detect,
                    }
                    for r in rules
                ],
                indent=2,
            )
        )
        return EXIT_OK

    for r in rules:
        extra = "  --repo" if r.needs_repo else ""
        print(f"{r.id:<30} {r.severity.value:<5} {r.tier.value}{extra}")
        print(f"    cannot detect: {r.cannot_detect}")
        print()
    print(f"{len(rules)} rules")
    return EXIT_OK


def _cmd_init(args: argparse.Namespace) -> int:
    path = Path(args.path) / ".resint.yml"
    if path.exists() and not args.force:
        print(f"resint: {path} already exists (use --force to overwrite)", file=sys.stderr)
        return EXIT_USAGE

    path.write_text(TEMPLATE, encoding="utf-8")
    print(f"wrote {path}")
    return EXIT_OK


TEMPLATE = """\
# resint configuration. Commit this alongside the paper.
version: 1

# Every suppression states a reason. That is deliberate: this file is the
# record of each judgement made about the work, and a silenced finding with
# no explanation is unauditable six months later.
#
# Suppressed findings still appear in JSON and SARIF output marked with their
# reason, so a suppression can never hide a regression.
suppress:
  # - rule: bib/metadata-drift
  #   match: "[vaswani2017]"
  #   reason: "Cites the proceedings version deliberately."
  #   expires: "2027-01-01"

# Turn a rule off entirely when it does not apply to this work.
rules:
  # stats/grim: off        # no integer-scale response data here
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="resint", description="A linter for research papers."
    )
    parser.add_argument("--version", action="version", version=f"resint {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="check a paper")
    check.add_argument(
        "target",
        help="a .tex file, or an arXiv source bundle (.tar.gz / .zip)",
    )
    check.add_argument("--repo", default=None, help="path to the paper's repository")
    check.add_argument(
        "--bib", default=None, help="bibliography (default: the .bib beside the source)"
    )
    check.add_argument(
        "--format", choices=["term", "json", "sarif"], default="term"
    )
    check.add_argument("--min-severity", choices=["low", "med", "high"], default=None)
    check.add_argument(
        "--fail-on",
        choices=["high", "med", "low", "none"],
        default="high",
        help="lowest severity that exits non-zero (default: high)",
    )
    check.add_argument(
        "--offline",
        action="store_true",
        help="skip reference lookups; the bib rules abstain rather than guess",
    )
    check.add_argument(
        "--mailto", default=None, help="contact address for the Crossref polite pool"
    )
    check.add_argument("--config", default=None, help="path to a .resint.yml")
    check.add_argument(
        "--no-config", action="store_true", help="ignore any .resint.yml"
    )
    check.set_defaults(fn=_cmd_check)

    rules = sub.add_parser("rules", help="list rules and their blind spots")
    rules.add_argument("--tier", choices=["deterministic", "model-assisted"], default=None)
    rules.add_argument("--family", default=None, help="e.g. stats, bib, numbers")
    rules.add_argument("--format", choices=["term", "json"], default="term")
    rules.set_defaults(fn=_cmd_rules)

    init = sub.add_parser("init", help="write a .resint.yml")
    init.add_argument("path", nargs="?", default=".", help="directory (default: .)")
    init.add_argument("--force", action="store_true")
    init.set_defaults(fn=_cmd_init)

    args = parser.parse_args(argv)
    try:
        return args.fn(args)
    except KeyboardInterrupt:
        # A traceback here tells the user nothing except that the tool is
        # amateurish. 130 is the conventional shell code for SIGINT.
        print("\nresint: interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
