"""Reading the papers a manuscript cites.

Checking a citation against the cited work's **abstract** answers the wrong
question most of the time. An abstract is two hundred words summarising ten
pages, and the thing an author cites a paper *for* -- a number, a method
detail, an ablation -- usually is not in it. A rule built on abstracts alone
would report "unsupported" across half of every bibliography and be worthless.

So this fetches the whole paper. Two sources, because they are the only two
whose open-access full text is structured text rather than a PDF:

- **arXiv** -- LaTeX source from the e-print endpoint, unpacked by
  ``parse.acquire``: the same code path a user's own paper takes, so the
  archive guards and the PDF sniffing come for free.
- **PMC** -- PubMed Central's open-access subset, as JATS XML.

Everything else -- paywalled work, and open-access papers published only as a
PDF -- is out of reach, because resint has no PDF reader and acquiring one
means either a dependency or a project of its own. That gap is honest and it
belongs in ``cannot_detect``: strong coverage for machine learning and
biomedicine, weaker for economics and the social sciences. The sweep measures
the real rate per field rather than anyone guessing at one.

The three-outcome contract is the reference resolvers', for the same reason.
``NOT_FOUND`` means no readable full text exists -- permanent, and worth
caching. ``UNKNOWN`` means we could not reach it -- transient, and never
cached, because storing a timeout turns a bad afternoon into a permanent wrong
answer. Neither may ever become a finding on its own.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from ..ir.paper import BibEntry
from .base import Record, Status, Unreachable
from .http import USER_AGENT, Pacer

EPRINT = "https://arxiv.org/e-print/{}"
EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

DEFAULT_CACHE = Path.home() / ".cache" / "resint" / "fulltext"

#: A cited paper past this is an anomaly -- a thesis, or a proceedings volume
#: bundled as one file. Truncating beats holding it all in memory.
MAX_TEXT_CHARS = 400_000

# 2103.00020, 2103.00020v2, and the pre-2007 form hep-th/9901001.
_ARXIV_NEW = re.compile(r"\b(\d{4}\.\d{4,5})(v\d+)?\b")
_ARXIV_OLD = re.compile(r"\b([a-z][a-z-]{2,}(?:\.[A-Z]{2})?/\d{7})(v\d+)?\b")
_PMCID = re.compile(r"\b(PMC\d{4,})\b", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class FullText:
    """The readable body of a cited paper."""

    source: str
    identifier: str
    text: str
    title: str = ""

    @property
    def words(self) -> int:
        return len(self.text.split())


@dataclass(frozen=True, slots=True)
class Fetched:
    status: Status
    document: FullText | None = None
    queried: tuple[str, ...] = ()
    detail: str = ""

    @property
    def usable(self) -> bool:
        return self.status is Status.FOUND and self.document is not None

    @property
    def checkable(self) -> bool:
        """Whether this can support a finding at all. UNKNOWN never can."""
        return self.status is not Status.UNKNOWN


class FullTextSource(Protocol):
    def fetch(self, entry: BibEntry, record: Record | None = None) -> Fetched: ...


# --- identifying what to fetch ------------------------------------------


def arxiv_id_for(entry: BibEntry | None, record: Record | None = None) -> str:
    """The arXiv id for a reference, from whichever field carries it.

    BibTeX has no single convention. Entries carry ``eprint``, or an
    ``arxiv.org`` URL, or the id buried in a journal field reading
    "arXiv preprint arXiv:1706.03762". All three appear constantly.
    """
    if record is not None and record.arxiv_id:
        return record.arxiv_id

    if entry is None:
        return ""

    eprint = entry.fields.get("eprint", "").strip()
    archive = entry.fields.get("archiveprefix", "").lower()
    if eprint and ("arxiv" in archive or not archive):
        for pattern in (_ARXIV_NEW, _ARXIV_OLD):
            found = pattern.search(eprint)
            if found:
                return found.group(1)

    for name in ("url", "journal", "note", "howpublished", "doi"):
        value = entry.fields.get(name, "")
        if "arxiv" not in value.lower():
            continue
        for pattern in (_ARXIV_NEW, _ARXIV_OLD):
            found = pattern.search(value)
            if found:
                return found.group(1)
    return ""


def pmcid_for(entry: BibEntry | None, record: Record | None = None) -> str:
    if record is not None and record.pmcid:
        return record.pmcid
    if entry is None:
        return ""
    for name in ("pmcid", "url", "note", "eprint"):
        found = _PMCID.search(entry.fields.get(name, ""))
        if found:
            return found.group(1).upper()
    return ""


# --- turning what came back into text -----------------------------------


def latex_to_text(raw: str) -> str:
    """Strip LaTeX to prose, keeping paragraph breaks.

    Blank lines are load-bearing: ``resolve.passages`` splits on them, and a
    paper collapsed into a single block cannot be narrowed to the passage that
    matters, which is the entire point of fetching it.
    """
    from ..parse.latex import normalize

    # normalize() collapses blank lines: for the manuscript under inspection
    # offsets matter and paragraph shape does not. Here it is the other way
    # round, so split on the author's own paragraph breaks first and normalize
    # each block. Normalizing first would hand retrieval one undifferentiated
    # block and quietly defeat the point of fetching the paper at all.
    blocks = []
    for block in re.split(r"\n[ \t]*\n", raw or ""):
        cleaned = normalize(block).text.strip()
        if cleaned:
            blocks.append(re.sub(r"\s*\n\s*", " ", cleaned))
    return "\n\n".join(blocks)


def jats_to_text(raw: str) -> str:
    """Abstract and body paragraphs out of PubMed Central's XML."""
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return ""

    chunks: list[str] = []
    for tag in ("abstract", "body"):
        for section in root.iter(tag):
            for paragraph in section.iter("p"):
                # itertext() flattens the inline markup -- italics, cross
                # references, formulae -- that JATS scatters through every
                # sentence.
                joined = " ".join("".join(paragraph.itertext()).split())
                if joined:
                    chunks.append(joined)
    return "\n\n".join(chunks)


def jats_title(raw: str) -> str:
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return ""
    for node in root.iter("article-title"):
        return " ".join("".join(node.itertext()).split())
    return ""


# --- the sources ---------------------------------------------------------


class NullFullText:
    """Fetches nothing and says so. The default, and what ``--offline`` uses."""

    name = "null"

    def fetch(self, entry: BibEntry, record: Record | None = None) -> Fetched:
        return Fetched(Status.UNKNOWN, detail="full-text retrieval is disabled")


@dataclass
class StaticFullText:
    """A fixed table of papers. How every test in this tier runs."""

    documents: dict[str, FullText] = field(default_factory=dict)
    unavailable: set[str] = field(default_factory=set)

    def fetch(self, entry: BibEntry, record: Record | None = None) -> Fetched:
        key = arxiv_id_for(entry, record) or pmcid_for(entry, record)
        if not key:
            return Fetched(
                Status.NOT_FOUND, detail="no arXiv id or PMCID for this reference"
            )
        if key in self.unavailable:
            return Fetched(Status.UNKNOWN, queried=(key,), detail="lookup failed")
        document = self.documents.get(key)
        if document is None:
            return Fetched(
                Status.NOT_FOUND, queried=(key,), detail="no open-access text"
            )
        return Fetched(Status.FOUND, document=document, queried=(key,))


@dataclass
class CachingFullText:
    """Disk-backed memo of another source.

    Cited papers repeat heavily -- across one manuscript's references, and far
    more across runs, since everyone in a field cites the same landmarks.
    Fetching "Attention Is All You Need" once per machine rather than once per
    run is the difference between this rule being usable and being a
    five-minute wait.

    Only settled answers are stored. An UNKNOWN is a statement about the
    network, not about the paper, and writing it down would make a transient
    failure permanent.
    """

    inner: FullTextSource
    directory: Path = field(default_factory=lambda: DEFAULT_CACHE)
    hits: int = 0
    misses: int = 0

    def _slot(self, key: str) -> Path:
        safe = key.replace("/", "_")
        return self.directory / (safe + ".json")

    def fetch(self, entry: BibEntry, record: Record | None = None) -> Fetched:
        key = arxiv_id_for(entry, record) or pmcid_for(entry, record)
        if not key:
            return self.inner.fetch(entry, record)

        slot = self._slot(key)
        if slot.exists():
            try:
                stored = json.loads(slot.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                stored = None
            if stored is not None:
                self.hits += 1
                return _from_cache(stored, key)

        self.misses += 1
        result = self.inner.fetch(entry, record)

        if result.checkable:
            try:
                self.directory.mkdir(parents=True, exist_ok=True)
                slot.write_text(json.dumps(_to_cache(result)), encoding="utf-8")
            except OSError:
                pass  # An unwritable cache is a slow run, not a failed one.
        return result


def _to_cache(result: Fetched) -> dict:
    document = result.document
    return {
        "status": result.status.value,
        "detail": result.detail,
        "queried": list(result.queried),
        "document": (
            None
            if document is None
            else {
                "source": document.source,
                "identifier": document.identifier,
                "text": document.text,
                "title": document.title,
            }
        ),
    }


def _from_cache(stored: dict, key: str) -> Fetched:
    raw = stored.get("document")
    return Fetched(
        Status(stored.get("status", "unknown")),
        document=FullText(**raw) if raw else None,
        queried=tuple(stored.get("queried") or (key,)),
        detail=stored.get("detail", ""),
    )


@dataclass
class HttpFullText:
    """Fetches from arXiv and PubMed Central, politely.

    arXiv asks for roughly three seconds between e-print requests and NCBI
    allows three calls a second. Both are honoured through the same
    :class:`Pacer` the reference resolvers use, so a run doing both kinds of
    lookup paces as one client rather than two.
    """

    mailto: str | None = None
    timeout: float = 20.0
    pacer: Pacer | None = None
    scratch: Path = field(default_factory=lambda: DEFAULT_CACHE / "raw")

    def _paced(self) -> Pacer:
        pacer = self.pacer or Pacer()
        pacer.intervals.setdefault("arxiv-eprint", 3.0)
        pacer.intervals.setdefault("pmc", 0.4)
        return pacer

    def _agent(self) -> str:
        return USER_AGENT + (" mailto:" + self.mailto if self.mailto else "")

    def _get(self, url: str, index: str) -> bytes | None:
        """The body, or None when the server answered that there is none.

        Raises :class:`Unreachable` when we could not ask. The distinction is
        the same one ``resolve/http.py`` learned the hard way: returning None
        for both makes an offline machine indistinguishable from a paper that
        genuinely has no open-access version, and every rule downstream then
        reasons about a fact that was never established.

        Here the collapse ran the safe way -- everything became UNKNOWN, so
        nothing was over-claimed -- but a real 404 was reported as "we could
        not check" forever, which is its own kind of wrong.
        """
        self._paced().wait(index)
        request = urllib.request.Request(url, headers={"User-Agent": self._agent()})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                if response.status == 404:
                    return None
                if response.status != 200:
                    raise Unreachable(f"{index} answered {response.status}")
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            raise Unreachable(f"{index} answered {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
            raise Unreachable(f"{index} unreachable: {exc}") from exc

    def fetch(self, entry: BibEntry, record: Record | None = None) -> Fetched:
        arxiv_id = arxiv_id_for(entry, record)
        if arxiv_id:
            return self._arxiv(arxiv_id)

        pmcid = pmcid_for(entry, record)
        if pmcid:
            return self._pmc(pmcid)

        return Fetched(
            Status.NOT_FOUND, detail="no arXiv id or PMCID for this reference"
        )

    def _arxiv(self, arxiv_id: str) -> Fetched:
        try:
            body = self._get(EPRINT.format(arxiv_id), "arxiv-eprint")
        except Unreachable as exc:
            return Fetched(Status.UNKNOWN, queried=(arxiv_id,), detail=str(exc))
        if body is None:
            return Fetched(
                Status.NOT_FOUND,
                queried=(arxiv_id,),
                detail="arXiv has no e-print under this identifier",
            )

        # arXiv serves the PDF when a submission has no source. That is a
        # settled fact about the paper, not a transient failure, so it is
        # NOT_FOUND and it is worth caching.
        if body[:5] == b"%PDF-":
            return Fetched(
                Status.NOT_FOUND,
                queried=(arxiv_id,),
                detail="arXiv has only a PDF for this paper, which resint cannot read",
            )

        from ..parse.acquire import UnreadableInput, acquire

        try:
            self.scratch.mkdir(parents=True, exist_ok=True)
            staged = self.scratch / (arxiv_id.replace("/", "_") + ".tar.gz")
            staged.write_bytes(body)
            source = acquire(staged)
        except (UnreadableInput, OSError) as exc:
            return Fetched(Status.NOT_FOUND, queried=(arxiv_id,), detail=str(exc))

        text = latex_to_text(source.text)[:MAX_TEXT_CHARS]
        if not text.strip():
            return Fetched(
                Status.NOT_FOUND, queried=(arxiv_id,), detail="source held no prose"
            )
        return Fetched(
            Status.FOUND,
            document=FullText(source="arxiv", identifier=arxiv_id, text=text),
            queried=(arxiv_id,),
        )

    def _pmc(self, pmcid: str) -> Fetched:
        params = {"db": "pmc", "id": pmcid, "retmode": "xml", "tool": "resint"}
        if self.mailto:
            params["email"] = self.mailto
        try:
            body = self._get(EFETCH + "?" + urllib.parse.urlencode(params), "pmc")
        except Unreachable as exc:
            return Fetched(Status.UNKNOWN, queried=(pmcid,), detail=str(exc))
        if body is None:
            return Fetched(
                Status.NOT_FOUND,
                queried=(pmcid,),
                detail="PubMed Central has no record under this identifier",
            )

        raw = body.decode("utf-8", "replace")
        text = jats_to_text(raw)[:MAX_TEXT_CHARS]
        if not text.strip():
            # PMC answers for every id; papers outside the open-access subset
            # come back as metadata with no body.
            return Fetched(
                Status.NOT_FOUND,
                queried=(pmcid,),
                detail="not in the PubMed Central open-access subset",
            )
        return Fetched(
            Status.FOUND,
            document=FullText(
                source="pmc", identifier=pmcid, text=text, title=jats_title(raw)
            ),
            queried=(pmcid,),
        )
