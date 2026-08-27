"""Clone the repositories the cached papers link to.

    python tools/fetch_repos.py
    python tools/fetch_repos.py --limit 40 --max-mb 200

Six rules need a repository and have never run on a real one: the five
``repro/`` rules plus ``claim/unimplemented``. Nothing had to be sourced for
them -- 40% of the arXiv corpus links its own code, 82 papers naming 123
distinct repositories.

Each clone is stored under the arXiv id of the paper that linked it, which is
what lets the sweep pair the two back up. A paper linking several repositories
keeps the first: the sweep checks one paper against one repository, and
guessing which of three is *the* implementation is not a guess worth making
silently.

``--depth 1 --single-branch``, because these rules read the current state of
the tree -- entrypoints, config files, dependency manifests, symbol names --
and never the history. Not ``--filter=blob:none``: a blobless clone fetches
file contents lazily, which would put a network round trip inside the analysis
and make an offline sweep quietly impossible.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path

EPRINTS = Path.home() / ".cache" / "resint" / "eprints"
DEFAULT_CACHE = Path.home() / ".cache" / "resint" / "repos"

#: Cloning is not API abuse, but 123 clones in a burst is still a burst.
POLITE_SECONDS = 1.0

#: A repository past this is a dataset or a model checkpoint, not source that
#: any of these rules can read.
DEFAULT_MAX_MB = 300

_REPO = re.compile(
    r"(?:https?://)?(?:www\.)?(?P<host>github\.com|gitlab\.com)/"
    r"(?P<owner>[A-Za-z0-9_.\-]+)/(?P<name>[A-Za-z0-9_.\-]+)"
)

#: Organisations that host tooling every paper links, never the paper's own
#: code. Cloning these would pair a paper with somebody else's framework and
#: report its config as the paper's.
NOT_THE_PAPERS_CODE = frozenset(
    {
        "pytorch", "tensorflow", "huggingface", "scikit-learn", "numpy",
        "scipy", "pandas", "matplotlib", "openai", "google", "google-research",
        "facebookresearch", "microsoft", "nvidia", "apache", "python",
        "jupyter", "conda-forge", "pypa", "actions", "docker",
    }
)

#: Files that are never a paper's own repository link.
_BADNAME = re.compile(r"\.(?:git|sty|cls|bib|tex|png|pdf|svg)$", re.IGNORECASE)


@dataclass
class Link:
    paper: str
    url: str
    slug: str


def repo_links(cache: Path, limit: int = 0) -> list[Link]:
    """The first repository each cached paper links to."""
    out: list[Link] = []
    bundles = sorted(cache.glob("*.tar.gz"))

    for bundle in bundles:
        text = ""
        try:
            with tarfile.open(bundle) as tf:
                for member in tf.getmembers():
                    if member.isfile() and member.name.endswith(".tex") and member.size < 3_000_000:
                        handle = tf.extractfile(member)
                        if handle is not None:
                            text += handle.read().decode("utf-8", "replace")
        except (tarfile.TarError, OSError):
            continue

        for match in _REPO.finditer(text):
            owner, name = match.group("owner"), match.group("name").rstrip(".,;:)}")
            if owner.lower() in NOT_THE_PAPERS_CODE or _BADNAME.search(name):
                continue
            paper_id = bundle.name.replace(".tar.gz", "")
            out.append(
                Link(
                    paper=paper_id,
                    url=f"https://{match.group('host')}/{owner}/{name}",
                    slug=f"{owner}/{name}",
                )
            )
            break  # One repository per paper: see the module docstring.

        if limit and len(out) >= limit:
            break

    return out


def _directory_mb(path: Path) -> float:
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            continue
    return total / (1024 * 1024)


def clone(link: Link, cache: Path, max_mb: float) -> tuple[str, str]:
    """Clone one repository. Returns (outcome, detail)."""
    target = cache / link.paper
    if target.exists():
        return "cached", ""

    staging = cache / f".{link.paper}.partial"
    shutil.rmtree(staging, ignore_errors=True)

    try:
        result = subprocess.run(
            [
                "git", "clone", "--depth", "1", "--single-branch",
                "--quiet", link.url, str(staging),
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        shutil.rmtree(staging, ignore_errors=True)
        return "failed", str(exc)[:80]

    if result.returncode != 0:
        shutil.rmtree(staging, ignore_errors=True)
        message = (result.stderr or "").strip().splitlines()
        detail = message[-1][:80] if message else f"exit {result.returncode}"
        return "failed", detail

    size = _directory_mb(staging)
    if size > max_mb:
        shutil.rmtree(staging, ignore_errors=True)
        return "too-big", f"{size:.0f} MB"

    shutil.rmtree(staging / ".git", ignore_errors=True)
    staging.rename(target)
    (cache / f"{link.paper}.json").write_text(
        json.dumps(
            {
                "paper": link.paper,
                "repo": link.url,
                "cloned_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "megabytes": round(size, 1),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return "cloned", f"{size:.0f} MB"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--eprints", default=str(EPRINTS))
    parser.add_argument("--cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-mb", type=float, default=DEFAULT_MAX_MB)
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args(argv)

    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)

    links = repo_links(Path(args.eprints), args.limit)
    print(f"{len(links)} papers link a repository")

    if args.list_only:
        for link in links:
            print(f"  {link.paper:<18} {link.slug}")
        return 0

    counts = {"cloned": 0, "cached": 0, "failed": 0, "too-big": 0}
    for number, link in enumerate(links, 1):
        outcome, detail = clone(link, cache, args.max_mb)
        counts[outcome] += 1
        if outcome != "cached":
            mark = {"cloned": " ", "failed": "!", "too-big": "~"}[outcome]
            print(f"  {mark} {number:>3}/{len(links)}  {link.slug:<46} {detail}")
        if outcome == "cloned":
            time.sleep(POLITE_SECONDS)

    print(
        f"\n  {counts['cloned']} cloned · {counts['cached']} already there · "
        f"{counts['failed']} unavailable · {counts['too-big']} too large"
    )
    print(f"  {cache}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
