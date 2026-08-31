"""Reference resolution, behind an interface.

Resolution is the only part of the deterministic tier that touches the
network, so it is the only part that can be slow, rate-limited, or wrong for
reasons that have nothing to do with the paper. Keeping it behind a protocol
means the rules stay pure, the tests never open a socket, and an offline run
degrades to an honest "could not check" rather than a false accusation.

Three outcomes, and the third is the important one:

    FOUND      the record exists, here it is
    NOT_FOUND  every index was queried and none had it
    UNKNOWN    the query itself failed -- offline, rate-limited, timed out

UNKNOWN must never become a finding. Reporting a reference as fabricated
because the network was down would be the single worst bug this tool could
ship.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

from ..ir.paper import BibEntry


class Unreachable(Exception):
    """The index could not be contacted at all.

    Distinct from a search that ran and returned nothing, and the distinction
    is the whole safety property of ``bib/unresolved``: a reference is reported
    as missing only when the indices were actually reached and actually had
    nothing. Returning None for both makes an offline machine look like proof
    that a paper does not exist.

    Found when DBLP was added and its TLS handshake failed on the development
    machine: every reference then claimed four indices had been searched when
    three had. A second instance turned up in ``resolve/fulltext.py``, running
    the safe way -- everything collapsed to UNKNOWN, so nothing was
    over-claimed, but a paper genuinely absent from arXiv was reported as
    "could not check" forever.

    Lives beside :class:`Status` because it is the transport-level half of the
    same three-outcome contract, and because every module that raises it also
    imports Status.
    """


class Status(str, Enum):
    FOUND = "found"
    NOT_FOUND = "not-found"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class Record:
    """A canonical bibliographic record from some index."""

    source: str
    title: str = ""
    year: str = ""
    authors: tuple[str, ...] = ()
    venue: str = ""
    doi: str = ""
    # How this record was found. A DOI is a claim about one registered
    # record; a title search returns a best guess. Rules that compare
    # metadata may only trust the former -- reporting a year as wrong
    # against a record that might be a different paper entirely is worse
    # than reporting nothing.
    matched_by: str = "doi"

    #: Identifiers that lead to *readable* full text, when the index knows
    #: them. Only these two, because arXiv source and PubMed Central XML are
    #: the only open-access forms that are structured text rather than a PDF,
    #: and resint has no PDF reader. See ``resolve/fulltext.py``.
    arxiv_id: str = ""
    pmcid: str = ""

    @property
    def authoritative(self) -> bool:
        return self.matched_by == "doi"

    @property
    def has_full_text(self) -> bool:
        """Whether an open-access full text exists that we can parse."""
        return bool(self.arxiv_id or self.pmcid)

    def render(self) -> str:
        who = self.authors[0].split(",")[0] if self.authors else "?"
        return f"{who} {self.year}, {self.title!r} ({self.source})"


@dataclass(frozen=True, slots=True)
class Resolution:
    status: Status
    record: Record | None = None
    queried: tuple[str, ...] = ()
    detail: str = ""

    @property
    def found(self) -> bool:
        return self.status is Status.FOUND

    @property
    def checkable(self) -> bool:
        """Whether this resolution can support a finding at all."""
        return self.status is not Status.UNKNOWN


class Resolver(Protocol):
    def resolve(self, entry: BibEntry) -> Resolution: ...


class NullResolver:
    """Resolves nothing and says so. The default when offline."""

    name = "null"

    def resolve(self, entry: BibEntry) -> Resolution:
        return Resolution(
            Status.UNKNOWN, detail="no resolver configured", queried=()
        )


@dataclass
class StaticResolver:
    """A fixed table of answers. Used by tests and by ``--offline`` replay."""

    records: dict[str, Record] = field(default_factory=dict)
    unknown: set[str] = field(default_factory=set)
    indices: tuple[str, ...] = ("crossref", "openalex", "arxiv", "s2")

    def resolve(self, entry: BibEntry) -> Resolution:
        if entry.key in self.unknown:
            return Resolution(
                Status.UNKNOWN, queried=self.indices, detail="lookup failed"
            )
        record = self.records.get(entry.key)
        if record is None:
            return Resolution(Status.NOT_FOUND, queried=self.indices)
        return Resolution(Status.FOUND, record=record, queried=self.indices)


@dataclass
class CachingResolver:
    """Memoizes another resolver for the duration of a run.

    A bibliography cites the same landmark papers repeatedly; without this a
    single run would query the same DOI a dozen times and earn a rate limit.
    """

    inner: Resolver
    _seen: dict[str, Resolution] = field(default_factory=dict, repr=False)
    _lock: object = field(default_factory=threading.Lock, repr=False)

    def resolve(self, entry: BibEntry) -> Resolution:
        cache_key = entry.doi.lower() or f"{entry.title.lower()}|{entry.year}"
        with self._lock:
            hit = self._seen.get(cache_key)
        if hit is not None:
            return hit

        # Deliberately outside the lock: a slow lookup must not block every
        # other worker. Two threads racing the same key costs one duplicate
        # request, which is far cheaper than serialising the whole pool.
        result = self.inner.resolve(entry)
        with self._lock:
            self._seen.setdefault(cache_key, result)
            return self._seen[cache_key]


# --- batch resolution ---------------------------------------------------


def resolve_all(
    resolver: Resolver,
    entries,
    *,
    workers: int = 6,
    budget: float = 40.0,
    progress=None,
) -> dict:
    """Resolve a bibliography concurrently, under an overall time budget.

    Sequential resolution is unusable at real bibliography sizes: thirty-five
    entries against three indices is a hundred round trips, and at a second
    each the tool appears to hang. A small pool fixes the latency; the budget
    fixes the tail, where one unreachable index would otherwise hold the whole
    run hostage.

    Entries not finished inside the budget come back UNKNOWN, which the rules
    already treat as "could not check" rather than "not found". Running out of
    time can therefore never manufacture a finding -- it only shrinks what was
    checked, and the report says so.
    """
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

    entries = list(entries)
    results: dict = {}
    if not entries:
        return results

    deadline = time.monotonic() + budget
    done_count = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(resolver.resolve, e): e for e in entries}
        outstanding = set(pending)

        while outstanding:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            finished, outstanding = wait(
                outstanding, timeout=remaining, return_when=FIRST_COMPLETED
            )
            for future in finished:
                entry = pending[future]
                try:
                    results[entry.key] = future.result()
                except Exception as exc:  # a probe raised; not the paper's fault
                    results[entry.key] = Resolution(
                        Status.UNKNOWN, detail=f"lookup failed: {exc}"
                    )
                done_count += 1
                if progress is not None:
                    progress(done_count, len(entries))

        for future in outstanding:
            entry = pending[future]
            future.cancel()
            results[entry.key] = Resolution(
                Status.UNKNOWN,
                detail=f"not looked up within the {budget:g}s budget",
            )

    return results
