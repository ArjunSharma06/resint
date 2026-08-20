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

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

from ..ir.paper import BibEntry


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

    def resolve(self, entry: BibEntry) -> Resolution:
        cache_key = entry.doi.lower() or f"{entry.title.lower()}|{entry.year}"
        if cache_key not in self._seen:
            self._seen[cache_key] = self.inner.resolve(entry)
        return self._seen[cache_key]
