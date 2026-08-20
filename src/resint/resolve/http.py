"""Live resolution against Crossref, OpenAlex, and arXiv.

Written on stdlib urllib so the install stays dependency-free. Every failure
path -- timeout, rate limit, malformed JSON, offline -- collapses to UNKNOWN,
never to NOT_FOUND. That distinction is the whole safety property of
``bib/unresolved``: a reference is only reported as missing when the indices
were actually reached and actually had nothing.

Lookup order is by strength of evidence. A DOI is a claim about a specific
registered record, so it is tried first and its failure means more. Title
search is a fallback and demands a high token overlap before it counts as a
match, because search endpoints return their best guess rather than nothing.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

from ..ir.paper import BibEntry
from ..parse.bibtex import fold
from .base import Record, Resolution, Status

USER_AGENT = "resint/0.1 (https://github.com/ArjunSharma06/resint; open-source paper linter)"

_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset({"a", "an", "the", "of", "for", "and", "on", "in", "with", "to"})

# A search hit below this token overlap is a different work, not a match.
_MATCH_FLOOR = 0.7


def _tokens(title: str) -> set[str]:
    return {w for w in _WORD.findall(fold(title).lower()) if w not in _STOP}


def title_matches(left: str, right: str) -> bool:
    a, b = _tokens(left), _tokens(right)
    if not a or not b:
        return False
    return len(a & b) / min(len(a), len(b)) >= _MATCH_FLOOR


def normalize_doi(raw: str) -> str:
    doi = raw.strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if doi.startswith(prefix):
            doi = doi[len(prefix) :]
    return doi


@dataclass
class HttpResolver:
    """Queries the public indices in order, stopping at the first real match."""

    mailto: str | None = None
    timeout: float = 6.0
    pause: float = 0.05
    _last_call: float = field(default=0.0, repr=False)

    @property
    def indices(self) -> tuple[str, ...]:
        return ("crossref", "openalex", "arxiv")

    # --- transport ------------------------------------------------------

    def _get(self, url: str) -> dict | None:
        """Fetch JSON. Returns None on any failure -- caller maps that to UNKNOWN."""
        gap = time.monotonic() - self._last_call
        if gap < self.pause:
            time.sleep(self.pause - gap)

        agent = USER_AGENT
        if self.mailto:
            agent = f"{agent} mailto:{self.mailto}"
        request = urllib.request.Request(url, headers={"User-Agent": agent})

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                if response.status != 200:
                    return None
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, ValueError, OSError):
            return None
        finally:
            self._last_call = time.monotonic()

    # --- per-index adapters ---------------------------------------------

    def _crossref(self, entry: BibEntry) -> Record | None:
        if entry.doi:
            doi = urllib.parse.quote(normalize_doi(entry.doi), safe="")
            payload = self._get(f"https://api.crossref.org/works/{doi}")
            if payload and payload.get("message"):
                return _from_crossref(payload["message"])
            return None

        if not entry.title:
            return None
        query = urllib.parse.urlencode(
            {"query.bibliographic": entry.title, "rows": "3"}
        )
        payload = self._get(f"https://api.crossref.org/works?{query}")
        if not payload:
            return None
        for item in payload.get("message", {}).get("items", []):
            record = _from_crossref(item)
            if record and title_matches(entry.title, record.title):
                return record
        return None

    def _openalex(self, entry: BibEntry) -> Record | None:
        if entry.doi:
            doi = urllib.parse.quote(normalize_doi(entry.doi), safe="")
            payload = self._get(f"https://api.openalex.org/works/doi:{doi}")
            if payload and payload.get("id"):
                return _from_openalex(payload)
            return None

        if not entry.title:
            return None
        query = urllib.parse.urlencode({"search": entry.title, "per-page": "3"})
        payload = self._get(f"https://api.openalex.org/works?{query}")
        if not payload:
            return None
        for item in payload.get("results", []):
            record = _from_openalex(item)
            if record and title_matches(entry.title, record.title):
                return record
        return None

    def _arxiv(self, entry: BibEntry) -> Record | None:
        if not entry.title:
            return None
        query = urllib.parse.urlencode(
            {"search_query": f'ti:"{entry.title}"', "max_results": "3"}
        )
        # The arXiv API answers in Atom XML; a title probe is enough here.
        raw = self._get_text(f"http://export.arxiv.org/api/query?{query}")
        if raw is None:
            return None
        titles = re.findall(r"<title>(.*?)</title>", raw, re.DOTALL)
        for candidate in titles[1:]:  # first <title> is the feed itself
            cleaned = " ".join(candidate.split())
            if title_matches(entry.title, cleaned):
                return Record(source="arxiv", title=cleaned)
        return None

    def _get_text(self, url: str) -> str | None:
        agent = USER_AGENT
        request = urllib.request.Request(url, headers={"User-Agent": agent})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                if response.status != 200:
                    return None
                return response.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, TimeoutError, ValueError, OSError):
            return None

    # --- the protocol ---------------------------------------------------

    def resolve(self, entry: BibEntry) -> Resolution:
        if not entry.doi and not entry.title:
            return Resolution(
                Status.UNKNOWN,
                queried=(),
                detail="entry has neither a DOI nor a title to search on",
            )

        reached_any = False
        for name, probe in (
            ("crossref", self._crossref),
            ("openalex", self._openalex),
            ("arxiv", self._arxiv),
        ):
            try:
                record = probe(entry)
            except Exception:
                continue
            if record is not None:
                return Resolution(Status.FOUND, record=record, queried=self.indices)
            reached_any = True

        if not reached_any:
            return Resolution(
                Status.UNKNOWN,
                queried=self.indices,
                detail="no index could be reached",
            )
        return Resolution(Status.NOT_FOUND, queried=self.indices)


def _from_crossref(item: dict) -> Record | None:
    titles = item.get("title") or []
    authors = tuple(
        f"{a.get('family', '')}, {a.get('given', '')}".strip(", ")
        for a in item.get("author", [])
        if a.get("family")
    )
    parts = (
        item.get("published-print")
        or item.get("published-online")
        or item.get("issued")
        or {}
    ).get("date-parts") or [[]]
    year = str(parts[0][0]) if parts and parts[0] else ""
    container = item.get("container-title") or []
    return Record(
        source="crossref",
        title=" ".join(titles[0].split()) if titles else "",
        year=year,
        authors=authors,
        venue=container[0] if container else "",
        doi=item.get("DOI", ""),
    )


def _from_openalex(item: dict) -> Record | None:
    authors = tuple(
        a["author"]["display_name"]
        for a in item.get("authorships", [])
        if a.get("author", {}).get("display_name")
    )
    venue = (item.get("primary_location") or {}).get("source") or {}
    return Record(
        source="openalex",
        title=" ".join((item.get("display_name") or "").split()),
        year=str(item.get("publication_year") or ""),
        authors=authors,
        venue=venue.get("display_name", ""),
        doi=normalize_doi(item.get("doi") or ""),
    )
