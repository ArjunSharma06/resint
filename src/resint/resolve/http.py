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
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field, replace

from ..ir.paper import BibEntry
from ..parse.bibtex import fold
from .base import Record, Registration, Resolution, Status, Unreachable

USER_AGENT = "resint/0.1 (https://github.com/ArjunSharma06/resint; open-source paper linter)"

_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset({"a", "an", "the", "of", "for", "and", "on", "in", "with", "to"})

# Jaccard, not overlap-over-the-smaller-set. The latter scores a short
# title fully contained in a longer one as a near-perfect match, which is
# how "Linformer: Self-Attention with Linear Complexity" matched "Mult-Pool
# Self Attention: a lightweight attention with linear complexity" at 0.80.
# Jaccard scores that pair 0.50 and rejects it.
_MATCH_FLOOR = 0.75


def _tokens(title: str) -> set[str]:
    return {w for w in _WORD.findall(fold(title).lower()) if w not in _STOP}


def title_similarity(left: str, right: str) -> float:
    """Symmetric token similarity. Both titles must largely agree."""
    a, b = _tokens(left), _tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def title_matches(left: str, right: str) -> bool:
    return title_similarity(left, right) >= _MATCH_FLOOR


#: A title alone may match this well and still be the wrong paper. Above the
#: floor but below this, a second signal has to agree.
_ALONE_FLOOR = 0.90


def _corroborates(entry, record) -> bool:
    """Whether something other than the title agrees.

    Title similarity is one signal and a weak one: series papers, corrected
    versions and translations all score highly against each other. Requiring
    the first author's surname or the year to line up as well turns a plausible
    string match into an identification.
    """
    if entry is None or record is None:
        return False

    from ..rules.bib.doi_mismatch import authors_agree

    if authors_agree(entry.authors, record.authors):
        return True

    stated, canonical = entry.year.strip(), (record.year or "").strip()
    if stated.isdigit() and canonical.isdigit():
        # Within a year: a preprint and its published version routinely
        # straddle a new year without being different work.
        return abs(int(stated) - int(canonical)) <= 1
    return False


def _best(entry_title: str, candidates, entry=None):
    """The closest candidate that is actually identified, or None.

    Search endpoints rank by their own relevance, which is not ours -- the
    first result over the line is regularly a worse match than the third.

    Two signals unless the title match is near-exact. A single strong-looking
    title is how a search endpoint hands back a different paper by the same
    group, and every rule downstream then reasons about the wrong record.
    """
    scored = [
        (title_similarity(entry_title, r.title), r) for r in candidates if r
    ]
    scored = [(s, r) for s, r in scored if s >= _MATCH_FLOOR]
    if not scored:
        return None

    scored.sort(key=lambda pair: -pair[0])
    for score, record in scored:
        if score >= _ALONE_FLOOR or _corroborates(entry, record):
            return record
    return None


def _as_title_match(record):
    """Mark a record as found by search rather than by identifier."""
    if record is None:
        return None
    return replace(record, matched_by="title")


def normalize_doi(raw: str) -> str:
    doi = raw.strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if doi.startswith(prefix):
            doi = doi[len(prefix) :]
    return doi


class Pacer:
    """Per-index request spacing.

    One global 20 requests/second across Crossref, OpenAlex *and* arXiv was
    wrong in both directions: too slow for the first two, and far too fast for
    arXiv, which asks for roughly three seconds between calls. Each index gets
    its own interval and its own lock, so a slow one cannot stall the others.

    The clock is injectable because a test that actually sleeps three seconds
    is a test nobody runs.
    """

    #: Seconds between requests, per index.
    DEFAULTS = {
        "crossref": 0.05, "openalex": 0.05, "arxiv": 3.0,
        "dblp": 1.0, "doi.org": 0.2,
    }

    def __init__(self, intervals: dict | None = None, clock=None, sleep=None):
        self.intervals = dict(self.DEFAULTS if intervals is None else intervals)
        self._clock = clock or time.monotonic
        self._sleep = sleep or time.sleep
        self._last: dict = {}
        self._lock = threading.Lock()

    def wait(self, index: str) -> None:
        interval = self.intervals.get(index, 0.05)
        with self._lock:
            previous = self._last.get(index)
            # Never seen this index: nothing to space out from. Defaulting the
            # timestamp to zero instead makes the very first request wait a
            # full interval, which is invisible against a real monotonic clock
            # and glaring against a fake one.
            delay = 0.0 if previous is None else interval - (self._clock() - previous)
            self._last[index] = self._clock() + max(delay, 0.0)
        if delay > 0:
            self._sleep(delay)


#: Shared by default so several resolvers in one process still pace as one.
_DEFAULT_PACER = Pacer()


@dataclass
class ResolvePolicy:
    """How hard a run is allowed to push the indices."""

    workers: int = 6
    budget: float = 40.0
    timeout: float = 5.0
    pacer: Pacer | None = None

    def paced_by(self) -> "Pacer":
        return self.pacer or _DEFAULT_PACER


@dataclass
class HttpResolver:
    """Queries the public indices in order, stopping at the first real match."""

    mailto: str | None = None
    timeout: float = 5.0
    pacer: Pacer | None = None

    #: Registration agency per DOI prefix. The agency is a property of the
    #: prefix, so this is a cache of a fact rather than of a response.
    _ra_cache: dict = field(default_factory=dict, repr=False)
    _ra_lock: object = field(default_factory=threading.Lock, repr=False)

    def _paced(self) -> Pacer:
        return self.pacer or _DEFAULT_PACER

    @property
    def indices(self) -> tuple[str, ...]:
        return ("crossref", "openalex", "arxiv", "dblp")

    #: Consulted only when all four miss a DOI, so it is not in `indices`:
    #: it answers a different question, and answers it authoritatively.
    AUTHORITY = "doi.org"

    # --- transport ------------------------------------------------------

    def _get(self, url: str, index: str = "crossref") -> dict | list | None:
        """Fetch JSON. Returns None on any failure -- caller maps that to UNKNOWN."""
        # Serialise only the pacing decision, never the request itself.
        self._paced().wait(index)

        agent = USER_AGENT
        if self.mailto:
            agent = f"{agent} mailto:{self.mailto}"
        request = urllib.request.Request(url, headers={"User-Agent": agent})

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                if response.status != 200:
                    raise Unreachable(f"{index} answered {response.status}")
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            # 404 from a lookup endpoint is a real answer: no such record.
            if exc.code == 404:
                return None
            raise Unreachable(f"{index} answered {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
            raise Unreachable(f"{index} unreachable: {exc}") from exc

    # --- per-index adapters ---------------------------------------------

    def _crossref(self, entry: BibEntry) -> Record | None:
        if entry.doi:
            doi = urllib.parse.quote(normalize_doi(entry.doi), safe="")
            payload = self._get(f"https://api.crossref.org/works/{doi}", "crossref")
            if payload and payload.get("message"):
                return _from_crossref(payload["message"])
            return None

        if not entry.title:
            return None
        query = urllib.parse.urlencode(
            {"query.bibliographic": entry.title, "rows": "3"}
        )
        payload = self._get(f"https://api.crossref.org/works?{query}", "crossref")
        if not payload:
            return None
        candidates = [
            _from_crossref(item)
            for item in payload.get("message", {}).get("items", [])
        ]
        return _as_title_match(_best(entry.title, candidates, entry))

    def _openalex(self, entry: BibEntry) -> Record | None:
        if entry.doi:
            doi = urllib.parse.quote(normalize_doi(entry.doi), safe="")
            payload = self._get(f"https://api.openalex.org/works/doi:{doi}", "openalex")
            if payload and payload.get("id"):
                return _from_openalex(payload)
            return None

        if not entry.title:
            return None
        query = urllib.parse.urlencode({"search": entry.title, "per-page": "3"})
        payload = self._get(f"https://api.openalex.org/works?{query}", "openalex")
        if not payload:
            return None
        candidates = [_from_openalex(item) for item in payload.get("results", [])]
        return _as_title_match(_best(entry.title, candidates, entry))

    def _dblp(self, entry: BibEntry) -> Record | None:
        """Search DBLP, which indexes computer-science proceedings.

        Added because Crossref, OpenAlex and arXiv between them are blind to a
        large slice of CS: workshop papers, many conference proceedings, and
        anything a publisher never registered a DOI for. Those entries were
        being reported as existing in no index, which is a claim about the
        world rather than about our coverage.
        """
        if not entry.title:
            return None

        query = urllib.parse.urlencode(
            {"q": entry.title, "format": "json", "h": "5"}
        )
        payload = self._get(
            f"https://dblp.org/search/publ/api?{query}", "dblp"
        )
        if not payload:
            return None

        hits = ((payload.get("result") or {}).get("hits") or {}).get("hit") or []
        candidates = []
        for hit in hits:
            info = hit.get("info") or {}
            title = " ".join((info.get("title") or "").split()).rstrip(".")
            if not title:
                continue
            authors = info.get("authors") or {}
            names = authors.get("author") or []
            if isinstance(names, dict):
                names = [names]
            candidates.append(
                Record(
                    source="dblp",
                    title=title,
                    year=str(info.get("year") or ""),
                    authors=tuple(
                        a.get("text", "") for a in names if isinstance(a, dict)
                    ),
                    venue=info.get("venue", "") or "",
                    doi=normalize_doi(info.get("doi") or ""),
                    matched_by="title",
                )
            )
        return _best(entry.title, candidates, entry)

    def _arxiv(self, entry: BibEntry) -> Record | None:
        if not entry.title:
            return None
        query = urllib.parse.urlencode(
            {"search_query": f'ti:"{entry.title}"', "max_results": "3"}
        )
        # The arXiv API answers in Atom XML; a title probe is enough here.
        raw = self._get_text(f"http://export.arxiv.org/api/query?{query}", "arxiv")
        if raw is None:
            return None
        titles = re.findall(r"<title>(.*?)</title>", raw, re.DOTALL)
        candidates = [
            Record(source="arxiv", title=" ".join(t.split()), matched_by="title")
            for t in titles[1:]  # the first <title> is the feed itself
        ]
        return _best(entry.title, candidates, entry)

    def _get_text(self, url: str, index: str = "arxiv") -> str | None:
        self._paced().wait(index)
        agent = USER_AGENT
        request = urllib.request.Request(url, headers={"User-Agent": agent})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                if response.status != 200:
                    raise Unreachable(f"{index} answered {response.status}")
                return response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            raise Unreachable(f"{index} answered {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
            raise Unreachable(f"{index} unreachable: {exc}") from exc

    # --- the protocol ---------------------------------------------------

    # --- the DOI system itself -------------------------------------------

    #: Content negotiation, so the request that proves a DOI exists also
    #: returns its canonical record. That record is the only metadata source
    #: bib/doi-mismatch has for a DOI none of our four indices can see.
    _CSL = "application/vnd.citationstyles.csl+json"

    def _doi_org(self, doi: str) -> tuple[Registration, str, Record | None]:
        """Ask the DOI system whether a DOI exists at all.

        Not a fifth index. Crossref, OpenAlex, arXiv and DBLP answer "do we
        have metadata for this"; doi.org answers "is this handle registered",
        and only the second question can support a claim of fabrication. See
        :class:`Registration` for what reading the first as the second cost.

        Reached only for a DOI every index already missed -- nine references
        in seventy papers -- so the extra round trip is not worth optimising.
        """
        self._paced().wait(self.AUTHORITY)

        agent = USER_AGENT
        if self.mailto:
            agent = f"{agent} mailto:{self.mailto}"
        request = urllib.request.Request(
            "https://doi.org/" + urllib.parse.quote(doi, safe="/:"),
            headers={"User-Agent": agent, "Accept": self._CSL},
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                if response.status != 200:
                    return Registration.UNCHECKED, "", None
                body = response.read()
                landed = response.url
        except urllib.error.HTTPError as exc:
            # The one status that answers the question asked. 404 from doi.org
            # is the DOI system saying no such handle is registered anywhere.
            if exc.code == 404:
                return Registration.DEAD, "", None
            return Registration.UNCHECKED, "", None
        except (urllib.error.URLError, TimeoutError, OSError):
            return Registration.UNCHECKED, "", None

        # Registered, whatever came back. Agencies outside Crossref and
        # DataCite routinely ignore the Accept header and serve their landing
        # page instead, and an unparseable body is still proof the handle
        # resolved. Reading it as "not registered" would rebuild the exact bug
        # this method exists to remove.
        return Registration.REGISTERED, self._agency(doi, landed), _from_csl(body)

    def _agency(self, doi: str, landed: str) -> str:
        """Which registration agency holds a DOI our indices could not see.

        Cached by DOI *prefix*, which is what actually determines the answer:
        every 10.16718/... handle belongs to the same agency, so a paper with
        forty Chinese-registered references asks once rather than forty times.
        Without that this is the slowest path in the resolver on exactly the
        bibliographies it was added to stop accusing.
        """
        prefix = doi.split("/", 1)[0]
        with self._ra_lock:
            hit = self._ra_cache.get(prefix)
        if hit is not None:
            return hit

        try:
            payload = self._get(
                "https://doi.org/doiRA/" + urllib.parse.quote(doi, safe="/:"),
                self.AUTHORITY,
            )
        except Unreachable:
            payload = None

        name = ""
        if isinstance(payload, list) and payload and isinstance(payload[0], dict):
            name = str(payload[0].get("RA") or "").strip()

        if name:
            with self._ra_lock:
                self._ra_cache[prefix] = name
            return name

        # doiRA did not answer. Whoever served the redirect is a worse name but
        # a true one -- and deliberately not cached: it is a degraded answer,
        # and caching it would keep the real agency from ever being learned for
        # this prefix because doiRA was down once.
        return urllib.parse.urlparse(landed or "").netloc.removeprefix("www.")

    def resolve(self, entry: BibEntry) -> Resolution:
        if not entry.doi and not entry.title:
            return Resolution(
                Status.UNKNOWN,
                queried=(),
                detail="entry has neither a DOI nor a title to search on",
            )

        reached: list[str] = []
        unreachable: list[str] = []

        for name, probe in (
            ("crossref", self._crossref),
            ("openalex", self._openalex),
            ("arxiv", self._arxiv),
            ("dblp", self._dblp),
        ):
            try:
                record = probe(entry)
            except Unreachable:
                unreachable.append(name)
                continue
            except Exception:
                unreachable.append(name)
                continue
            reached.append(name)
            if record is not None:
                return Resolution(
                    Status.FOUND, record=record, queried=tuple(reached)
                )

        if not reached:
            return Resolution(
                Status.UNKNOWN,
                queried=(),
                detail=f"no index could be reached ({', '.join(unreachable)})",
            )

        # Reporting a reference as absent rests on having actually looked.
        # Naming only the indices that answered keeps the finding's claim the
        # same size as the evidence behind it -- and an index that was down is
        # said to have been down, not silently counted as a search.
        missed = (
            f"{', '.join(unreachable)} could not be reached and was not searched"
            if unreachable
            else ""
        )

        # Every index missed it. For a DOI that settles nothing: the indices
        # are metadata, and the question a finding rests on -- does this DOI
        # exist -- belongs to the DOI system.
        if entry.doi:
            registration, agency, record = self._doi_org(normalize_doi(entry.doi))

            if registration is Registration.UNCHECKED:
                # The authority did not answer, so nothing is known about
                # existence. Exactly the case that must never fire.
                return Resolution(
                    Status.UNKNOWN,
                    queried=tuple(reached),
                    registration=registration,
                    detail="doi.org could not be reached, so whether this DOI is registered is unknown",
                )

            if registration is Registration.REGISTERED:
                if record is not None:
                    return Resolution(
                        Status.FOUND,
                        record=record,
                        queried=tuple(reached) + (self.AUTHORITY,),
                        registration=registration,
                        agency=agency,
                    )
                # Live, but its agency publishes no metadata we can read. The
                # reference is fine and there is nothing to compare it against.
                return Resolution(
                    Status.NOT_FOUND,
                    queried=tuple(reached) + (self.AUTHORITY,),
                    registration=registration,
                    agency=agency,
                    detail=(
                        f"registered with {agency or 'an agency'} but indexed "
                        "by none of the metadata sources we can read"
                    ),
                )

            return Resolution(
                Status.NOT_FOUND,
                queried=tuple(reached) + (self.AUTHORITY,),
                registration=registration,
                detail=missed,
            )

        return Resolution(
            Status.NOT_FOUND, queried=tuple(reached), detail=missed
        )


def _from_csl(body: bytes) -> Record | None:
    """A Record from the CSL JSON doi.org returns under content negotiation.

    Returns None when there is no title, rather than a Record with an empty
    one. An empty title scores zero similarity against anything, which
    ``bib/doi-mismatch`` would read as the strongest possible disagreement --
    turning a DOI we merely cannot describe into a DOI pointing at the wrong
    paper. Absent metadata must stay absent, not become contrary metadata.
    """
    try:
        item = json.loads(body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(item, dict):
        return None

    title = item.get("title")
    if isinstance(title, list):
        title = title[0] if title else ""
    title = " ".join(str(title or "").split())
    if not title:
        return None

    venue = item.get("container-title")
    if isinstance(venue, list):
        venue = venue[0] if venue else ""

    issued = ((item.get("issued") or {}).get("date-parts") or [[]])[0]
    return Record(
        source="doi.org",
        title=title,
        year=str(issued[0]) if issued else "",
        authors=tuple(
            f"{a.get('family', '')}, {a.get('given', '')}".strip(", ")
            for a in item.get("author") or []
            if isinstance(a, dict) and a.get("family")
        ),
        venue=str(venue or ""),
        doi=normalize_doi(str(item.get("DOI") or "")),
        matched_by="doi",
    )


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


def _open_access_ids(item: dict) -> tuple[str, str]:
    """The arXiv id and PMCID OpenAlex knows for a work, if any.

    These are what make full-text retrieval possible, and OpenAlex is the only
    index queried here that reports them. It records the arXiv id as a landing
    page among the work's locations rather than as a bare identifier, so the
    URL is where to look.
    """
    from .fulltext import _ARXIV_NEW, _ARXIV_OLD, _PMCID

    identifiers = item.get("ids") or {}
    pmcid = ""
    found = _PMCID.search(identifiers.get("pmcid") or "")
    if found:
        pmcid = found.group(1).upper()

    locations = list(item.get("locations") or [])
    best = item.get("best_oa_location")
    if best:
        locations.append(best)

    for location in locations:
        if not isinstance(location, dict):
            continue
        url = location.get("landing_page_url") or ""
        if "arxiv.org" not in url.lower():
            continue
        for pattern in (_ARXIV_NEW, _ARXIV_OLD):
            hit = pattern.search(url)
            if hit:
                return hit.group(1), pmcid

    return "", pmcid


def _from_openalex(item: dict) -> Record | None:
    authors = tuple(
        a["author"]["display_name"]
        for a in item.get("authorships", [])
        if a.get("author", {}).get("display_name")
    )
    venue = (item.get("primary_location") or {}).get("source") or {}
    arxiv_id, pmcid = _open_access_ids(item)
    return Record(
        source="openalex",
        title=" ".join((item.get("display_name") or "").split()),
        year=str(item.get("publication_year") or ""),
        authors=authors,
        venue=venue.get("display_name", ""),
        doi=normalize_doi(item.get("doi") or ""),
        arxiv_id=arxiv_id,
        pmcid=pmcid,
    )
