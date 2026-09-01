"""Cache arXiv source bundles for a sweep. Politely.

    python tools/fetch_arxiv.py --count 100 --mailto you@example.org
    python tools/fetch_arxiv.py --count 40 --categories q-bio.QM,stat.AP
    python tools/fetch_arxiv.py --ids notes/corpus-2026-09.txt

Serial, six seconds apart, cached by arXiv id. Two hundred and fifty papers
takes about twenty-five minutes, once — after that every sweep runs offline at
full speed, which is what makes a fix-and-re-run loop viable at all.

**Do not parallelise this.** Fanning out e-print downloads is how a project
gets its IP blocked, and the download is a one-off while the sweep is the
thing you repeat.

Default categories deliberately span fields. Three rules — stats/grim,
stats/pvalue-mismatch, stats/significance-unsupported — only fire on papers
reporting statistical tests, which machine-learning papers largely do not, so
an all-cs corpus would leave them untested on real input.

``--ids`` takes a file of identifiers and skips discovery entirely. Discovery
lists whatever the service has today, so a corpus built that way cannot be
rebuilt and two sweeps a week apart are not comparable; an id list makes the
corpus reproducible and lets its composition be chosen rather than accepted.

Licence: arXiv's default terms do not permit redistribution. The cache is
gitignored and must stay that way. Fixtures distilled from these papers must
be rewritten minimal reproductions of the failure, never verbatim excerpts,
unless the paper is explicitly CC-BY or CC0.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from resint.resolve.http import USER_AGENT  # noqa: E402

API = "http://export.arxiv.org/api/query"
EPRINT = "https://arxiv.org/e-print/{}"

# arXiv asks for roughly three seconds between requests. Six is deliberately
# more than asked: a 400-paper corpus is a much larger favour than a handful,
# it is fetched once and reused offline forever, and nothing downstream is
# waiting on it. Being slower than required costs a quarter of an hour once
# and removes any argument that this was rude.
POLITE_SECONDS = 6.0
DEFAULT_CACHE = Path.home() / ".cache" / "resint" / "eprints"

DEFAULT_CATEGORIES = (
    "cs.LG", "cs.CL", "cs.CV",       # tables, hyperparameters, repositories
    "q-bio.QM", "q-bio.NC",          # reported statistics
    "stat.AP", "stat.ME",            # reported statistics
    "econ.EM",                       # reported statistics, different conventions
)

_ID = re.compile(r"<id>http://arxiv\.org/abs/([^<]+)</id>")


@dataclass
class Fetched:
    arxiv_id: str
    path: Path
    cached: bool
    bytes: int = 0
    error: str = ""


def _request(url: str, mailto: str | None) -> bytes:
    agent = f"{USER_AGENT} mailto:{mailto}" if mailto else USER_AGENT
    req = urllib.request.Request(url, headers={"User-Agent": agent})
    with urllib.request.urlopen(req, timeout=30) as response:
        return response.read()


def list_ids(categories, per_category: int, mailto: str | None) -> list[str]:
    """Ask the arXiv API which papers exist. One query per category."""
    ids: list[str] = []
    for index, category in enumerate(categories):
        if index:
            time.sleep(POLITE_SECONDS)
        query = urllib.parse.urlencode(
            {
                "search_query": f"cat:{category}",
                "start": 0,
                "max_results": per_category,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }
        )
        try:
            body = _request(f"{API}?{query}", mailto).decode("utf-8", "replace")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print(f"  {category}: listing failed ({exc})", file=sys.stderr)
            continue

        found = _ID.findall(body)
        ids.extend(found)
        print(f"  {category:<10} {len(found)} ids", file=sys.stderr)

    return ids


def fetch_one(arxiv_id: str, cache: Path, mailto: str | None) -> Fetched:
    """Download one e-print, or report the cache hit."""
    safe = arxiv_id.replace("/", "_")
    target = cache / f"{safe}.tar.gz"
    sidecar = target.with_suffix(".json")

    if target.exists():
        return Fetched(arxiv_id, target, cached=True, bytes=target.stat().st_size)

    try:
        body = _request(EPRINT.format(arxiv_id), mailto)
    except urllib.error.HTTPError as exc:
        return Fetched(arxiv_id, target, cached=False, error=f"HTTP {exc.code}")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return Fetched(arxiv_id, target, cached=False, error=str(exc))

    # arXiv serves the PDF when a submission has no source. Saving that as
    # ".tar.gz" makes every downstream tool report an unpacking failure
    # instead of the one fact that matters.
    if body[:5] == b"%PDF-":
        target = cache / f"{safe}.pdf"
        sidecar = target.with_suffix(".json")

    target.write_bytes(body)
    sidecar.write_text(
        json.dumps(
            {
                "arxiv_id": arxiv_id,
                "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "bytes": len(body),
                # Recorded because a PDF-only submission is a legitimate
                # outcome, not a failure, and the sweep should say which.
                "looks_like_pdf": body[:5] == b"%PDF-",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return Fetched(arxiv_id, target, cached=False, bytes=len(body))


def read_ids(path: Path) -> list[str]:
    """Identifiers from a file, one per line.

    Discovery returns whatever the service currently lists, so a corpus built
    that way cannot be rebuilt -- the same command a week later fetches
    different papers, and a sweep can never be re-run against the material it
    actually measured. A committed id list fixes that, and turns a corpus into
    something composed on purpose rather than accepted as it arrives.

    Blank lines and #-comments are skipped so the file can say what each block
    is for, and duplicates are dropped so lists can be concatenated.
    """
    ids: list[str] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        entry = line.split("#", 1)[0].strip()
        if entry and entry not in seen:
            seen.add(entry)
            ids.append(entry)
    return ids


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument(
        "--ids",
        default=None,
        help="file of identifiers, one per line; skips discovery entirely",
    )
    parser.add_argument("--categories", default=",".join(DEFAULT_CATEGORIES))
    parser.add_argument("--cache", default=str(DEFAULT_CACHE))
    parser.add_argument(
        "--mailto", default=None, help="contact address, so arXiv can reach you"
    )
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args(argv)

    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)

    if args.ids:
        source = Path(args.ids)
        if not source.is_file():
            print(f"fetch_arxiv: no such id file: {source}", file=sys.stderr)
            return 2
        ids = read_ids(source)
        print(f"{len(ids)} ids from {source} -- discovery skipped")
    else:
        categories = [c.strip() for c in args.categories.split(",") if c.strip()]
        per_category = max(1, args.count // len(categories))

        print(f"listing {args.count} papers across {len(categories)} categories")
        ids = list_ids(categories, per_category, args.mailto)[: args.count]
        print(f"\n{len(ids)} ids")

    if args.list_only:
        for i in ids:
            print(i)
        return 0

    print(f"fetching into {cache} — {POLITE_SECONDS:g}s apart, serial\n")
    fetched = cached = failed = 0
    consecutive_failures = 0

    for number, arxiv_id in enumerate(ids, 1):
        result = fetch_one(arxiv_id, cache, args.mailto)

        if result.error:
            failed += 1
            consecutive_failures += 1
            print(f"  {number:>4}/{len(ids)}  {arxiv_id:<16} {result.error}")
            # A run of failures means something is wrong at their end or ours.
            # Continuing would just be hammering a service that said no.
            if consecutive_failures >= 5:
                print("\n  five failures in a row — stopping", file=sys.stderr)
                break
        else:
            consecutive_failures = 0
            if result.cached:
                cached += 1
            else:
                fetched += 1
                print(
                    f"  {number:>4}/{len(ids)}  {arxiv_id:<16} "
                    f"{result.bytes / 1024:>7.0f} KB"
                )

        if not result.cached and number < len(ids):
            time.sleep(POLITE_SECONDS)

    print(f"\n  {fetched} fetched · {cached} already cached · {failed} failed")
    print(f"  {cache}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
