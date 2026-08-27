"""Cache PubMed Central open-access articles. Politely.

    python tools/fetch_pmc.py --count 150
    python tools/fetch_pmc.py --count 40 --topics psychology,clinical-trial

Companion to ``fetch_arxiv.py``, and the reason it exists is measured rather
than assumed: across 204 real arXiv papers the statistics extractor found
something in **2**. ``stats/grim``, ``stats/pvalue-mismatch`` and
``stats/significance-unsupported`` had never once run on real input.

That is not a bug in those rules. Inline NHST -- ``t(20) = 2.086, p = .03`` --
is a convention of psychology, medicine and epidemiology, and arXiv is
physics, maths and computer science. The rules were aimed at a literature the
corpus did not contain. PMC is where that literature is.

**Pacing.** NCBI documents three requests a second without an API key. This
asks for two, which is inside their limit without needing a key from anyone.
That is faster than ``fetch_arxiv.py``'s six seconds because the constraints
genuinely differ: arXiv publishes a courtesy interval and serves whole source
bundles, while these are single XML documents from an endpoint built for
bulk programmatic access.

**Licence.** The open-access subset permits redistribution, unlike arXiv's
default terms -- but individual articles carry their own licences (CC-BY,
CC-BY-NC, occasionally more restrictive). The cache is gitignored and stays
that way. Fixtures distilled from these must be rewritten miniatures, the same
rule the arXiv corpus follows, because sorting per-article licences is not
work this project needs to take on.
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

ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

#: Two a second, inside NCBI's documented three.
POLITE_SECONDS = 0.5

DEFAULT_CACHE = Path.home() / ".cache" / "resint" / "pmc"

#: Searches chosen for how the literature *reports* results, not for subject
#: interest. Each of these fields writes statistics inline, which is precisely
#: what arXiv does not.
TOPICS = {
    "psychology": '"psychology"[MeSH Terms] AND "open access"[filter]',
    "clinical-trial": '"randomized controlled trial"[Publication Type] AND "open access"[filter]',
    "epidemiology": '"epidemiology"[MeSH Terms] AND "open access"[filter]',
    "public-health": '"public health"[MeSH Terms] AND "open access"[filter]',
    "neuroscience": '"neurosciences"[MeSH Terms] AND "open access"[filter]',
    "nutrition": '"nutrition therapy"[MeSH Terms] AND "open access"[filter]',
}

_ID = re.compile(r"<Id>(\d+)</Id>")


@dataclass
class Fetched:
    pmcid: str
    path: Path
    cached: bool
    bytes: int = 0
    error: str = ""


def _request(url: str, mailto: str | None) -> bytes:
    agent = f"{USER_AGENT} mailto:{mailto}" if mailto else USER_AGENT
    req = urllib.request.Request(url, headers={"User-Agent": agent})
    with urllib.request.urlopen(req, timeout=30) as response:
        return response.read()


def list_ids(topics, per_topic: int, mailto: str | None) -> list[str]:
    """Ask NCBI which articles exist. One query per topic."""
    ids: list[str] = []
    for index, topic in enumerate(topics):
        if index:
            time.sleep(POLITE_SECONDS)
        term = TOPICS.get(topic)
        if term is None:
            print(f"  {topic}: unknown topic, skipped", file=sys.stderr)
            continue

        query = urllib.parse.urlencode(
            {
                "db": "pmc",
                "term": term,
                "retmax": per_topic,
                "retmode": "xml",
                "sort": "pub_date",
                "tool": "resint",
                **({"email": mailto} if mailto else {}),
            }
        )
        try:
            body = _request(f"{ESEARCH}?{query}", mailto).decode("utf-8", "replace")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print(f"  {topic}: listing failed ({exc})", file=sys.stderr)
            continue

        found = [f"PMC{i}" for i in _ID.findall(body)]
        ids.extend(found)
        print(f"  {topic:<16} {len(found)} ids", file=sys.stderr)

    # Topics overlap -- a trial in a psychology journal answers two searches.
    return list(dict.fromkeys(ids))


def fetch_one(pmcid: str, cache: Path, mailto: str | None) -> Fetched:
    target = cache / f"{pmcid}.nxml"
    if target.exists():
        return Fetched(pmcid, target, cached=True, bytes=target.stat().st_size)

    query = urllib.parse.urlencode(
        {
            "db": "pmc",
            "id": pmcid[3:],
            "retmode": "xml",
            "tool": "resint",
            **({"email": mailto} if mailto else {}),
        }
    )
    try:
        body = _request(f"{EFETCH}?{query}", mailto)
    except urllib.error.HTTPError as exc:
        return Fetched(pmcid, target, cached=False, error=f"HTTP {exc.code}")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return Fetched(pmcid, target, cached=False, error=str(exc))

    text = body.decode("utf-8", "replace")

    # PMC answers for every id it knows, including ones outside the
    # open-access subset -- those come back as metadata with no body. Saving
    # them would pad the corpus with articles nothing can be checked against.
    if "<body" not in text:
        return Fetched(
            pmcid, target, cached=False, error="not in the open-access subset"
        )

    target.write_bytes(body)
    target.with_suffix(".json").write_text(
        json.dumps(
            {
                "pmcid": pmcid,
                "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "bytes": len(body),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return Fetched(pmcid, target, cached=False, bytes=len(body))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--count", type=int, default=150)
    parser.add_argument("--topics", default=",".join(TOPICS))
    parser.add_argument("--cache", default=str(DEFAULT_CACHE))
    parser.add_argument(
        "--mailto", default=None, help="contact address, so NCBI can reach you"
    )
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args(argv)

    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)

    topics = [t.strip() for t in args.topics.split(",") if t.strip()]
    per_topic = max(1, args.count // max(len(topics), 1))

    print(f"listing {args.count} articles across {len(topics)} topics")
    ids = list_ids(topics, per_topic, args.mailto)[: args.count]
    print(f"\n{len(ids)} ids")

    if args.list_only:
        for i in ids:
            print(i)
        return 0

    print(f"fetching into {cache} — {POLITE_SECONDS:g}s apart, serial\n")
    fetched = cached = failed = 0
    consecutive = 0

    for number, pmcid in enumerate(ids, 1):
        result = fetch_one(pmcid, cache, args.mailto)

        if result.error:
            failed += 1
            consecutive += 1
            print(f"  {number:>4}/{len(ids)}  {pmcid:<14} {result.error}")
            if consecutive >= 8:
                # Articles outside the subset are expected and are not a
                # reason to stop; a long run of them means the search itself
                # is wrong, and so is continuing.
                print("\n  eight failures in a row — stopping", file=sys.stderr)
                break
        else:
            consecutive = 0
            if result.cached:
                cached += 1
            else:
                fetched += 1
                print(
                    f"  {number:>4}/{len(ids)}  {pmcid:<14} "
                    f"{result.bytes / 1024:>7.0f} KB"
                )

        if not result.cached and number < len(ids):
            time.sleep(POLITE_SECONDS)

    print(f"\n  {fetched} fetched · {cached} already cached · {failed} unavailable")
    print(f"  {cache}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
