"""Turning whatever the user pointed at into LaTeX source.

The natural way to obtain a paper's source is arXiv's e-print bundle -- a
``.tar.gz`` holding the ``.tex``, the ``.bib``, and a pile of figures. Asking
someone to unpack it first is asking them to do work the tool can do, and the
first thing anyone actually tried was passing the tarball directly.

Anything genuinely unreadable fails with a sentence, never a traceback. A
UnicodeDecodeError on byte 0x8b tells the user nothing except that the tool
did not expect them.
"""

from __future__ import annotations

import re
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

ARCHIVE_SUFFIXES = (".tar.gz", ".tgz", ".tar.bz2", ".tar", ".zip", ".gz")
LATEX_SUFFIXES = (".tex", ".ltx")

# Files an arXiv bundle carries that are LaTeX but never the paper itself.
_NOT_THE_PAPER = (
    "supplement", "appendix", "response", "cover", "letter", "rebuttal",
    "readme", "makefile",
)

MAX_ARCHIVE_BYTES = 80 * 1024 * 1024
# Guards on what an archive expands to, not just what it weighs on disk.
MAX_UNPACKED_BYTES = 512 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 20_000
MAX_COMPRESSION_RATIO = 200


class UnreadableInput(ValueError):
    """Raised when the target cannot be read as LaTeX, with advice attached."""


@dataclass
class Acquired:
    """LaTeX source and its bibliography, however they were obtained."""

    text: str
    name: str
    path: str
    # Whether this came out of an archive. Callers must not go looking for a
    # bibliography beside an archive: in a sweep, hundreds of bundles share
    # one cache directory, and a single stray .bib there would be adopted by
    # every paper in the run. Recorded explicitly rather than sniffed from
    # the path string, which is ambiguous on Windows ("C:\..." has a colon).
    from_archive: bool = False
    bib_text: str | None = None
    bib_name: str = "refs.bib"
    # "bib" or "bbl". A compiled bibliography needs a different parser and
    # carries less certainty, so the kind travels with the text.
    bib_kind: str = "bib"
    notes: tuple[str, ...] = ()
    # Populated when \input directives were spliced. Empty for a single file,
    # where offsets already point at the only source there is.
    regions: tuple = ()
    files: dict | None = None


def _looks_binary(raw: bytes) -> bool:
    return b"\x00" in raw[:4096]


def _decode(raw: bytes, label: str) -> str:
    for encoding in ("utf-8", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise UnreadableInput(f"{label} is not text in any encoding resint understands")


def _score_candidate(name: str, text: str) -> tuple[int, int]:
    """Rank a .tex file by how likely it is to be the paper's root."""
    lowered = Path(name).name.lower()
    score = 0
    if "\\documentclass" in text:
        score += 4
    if "\\begin{document}" in text:
        score += 4
    if "\\title" in text or "\\maketitle" in text:
        score += 2
    if any(word in lowered for word in _NOT_THE_PAPER):
        score -= 6
    if lowered in ("main.tex", "paper.tex", "ms.tex", "article.tex"):
        score += 3
    return score, len(text)


def _pick_root(candidates: dict[str, str]) -> tuple[str, str]:
    """The most plausible root document among several .tex files.

    arXiv bundles routinely contain a dozen: per-section includes, a
    supplement, a response letter. Choosing the largest is wrong often enough
    to matter -- an appendix is frequently the biggest file in the bundle.
    """
    best = max(candidates.items(), key=lambda kv: _score_candidate(*kv))
    return best


def _guard_members(label: str, members) -> None:
    """Refuse an archive before unpacking it.

    ``MAX_ARCHIVE_BYTES`` checks the *compressed* size, which an 80 MB
    tarball expanding to 40 GB passes comfortably. ``filter="data"`` does not
    close this either -- it stops traversal and device files, not volume. On
    a sweep we hand this function hundreds of archives fetched from the
    internet, so the declared sizes get checked first and the extraction only
    happens if they are plausible.
    """
    total = 0
    count = 0
    for name, size, compressed in members:
        count += 1
        total += size
        if count > MAX_ARCHIVE_MEMBERS:
            raise UnreadableInput(
                f"{label} declares more than {MAX_ARCHIVE_MEMBERS} files"
            )
        if total > MAX_UNPACKED_BYTES:
            raise UnreadableInput(
                f"{label} unpacks to over {MAX_UNPACKED_BYTES / 1e6:.0f} MB"
            )
        if compressed and size / max(compressed, 1) > MAX_COMPRESSION_RATIO:
            raise UnreadableInput(
                f"{label}: {name} expands {size / max(compressed, 1):.0f}x, "
                f"over the {MAX_COMPRESSION_RATIO}x limit"
            )


def _guard_extracted(label: str, root: Path) -> None:
    """Belt and braces: nothing may sit outside the extraction root."""
    resolved_root = root.resolve()
    for found in root.rglob("*"):
        if not found.resolve().is_relative_to(resolved_root):
            raise UnreadableInput(f"{label} wrote outside its extraction directory")


def _from_archive(path: Path) -> Acquired:
    size = path.stat().st_size
    if size > MAX_ARCHIVE_BYTES:
        raise UnreadableInput(
            f"{path.name} is {size / 1e6:.0f} MB, larger than the {MAX_ARCHIVE_BYTES / 1e6:.0f} MB limit"
        )

    with TemporaryDirectory(prefix="resint-") as tmp:
        root = Path(tmp)
        try:
            if zipfile.is_zipfile(path):
                with zipfile.ZipFile(path) as zf:
                    _guard_members(
                        path.name,
                        ((i.filename, i.file_size, i.compress_size) for i in zf.infolist()),
                    )
                    zf.extractall(root)
            else:
                with tarfile.open(path) as tf:
                    _guard_members(
                        path.name,
                        ((m.name, m.size, 0) for m in tf.getmembers() if m.isfile()),
                    )
                    # filter="data" refuses absolute paths, parent traversal,
                    # links and device files. An archive from the internet is
                    # untrusted input like any other.
                    tf.extractall(root, filter="data")
        except (tarfile.TarError, zipfile.BadZipFile, OSError) as exc:
            raise UnreadableInput(f"{path.name} could not be unpacked ({exc})") from exc

        _guard_extracted(path.name, root)

        tex: dict[str, str] = {}
        bibs: dict[str, str] = {}
        bbls: dict[str, str] = {}
        for found in sorted(root.rglob("*")):
            if not found.is_file():
                continue
            rel = str(found.relative_to(root)).replace("\\", "/")
            suffix = found.suffix.lower()
            if suffix in LATEX_SUFFIXES or (suffix == "" and found.stat().st_size < 2_000_000):
                raw = found.read_bytes()
                if _looks_binary(raw):
                    continue
                text = _decode(raw, rel)
                if suffix in LATEX_SUFFIXES or "\\documentclass" in text:
                    tex[rel] = text
            elif suffix == ".bib":
                bibs[rel] = _decode(found.read_bytes(), rel)
            elif suffix == ".bbl":
                bbls[rel] = _decode(found.read_bytes(), rel)

        if not tex:
            raise UnreadableInput(
                f"{path.name} contains no LaTeX source. arXiv bundles for "
                "papers written in Word or submitted as PDF-only have none."
            )

        name, _ = _pick_root(tex)

        # A root that is mostly \input directives is the normal shape for a
        # real submission. Reading it alone sees the abstract and nothing
        # else, so the includes are spliced in and tracked.
        text, regions = expand_inputs(name, tex)
        spliced = {r.name for r in regions} - {name}

        notes = []
        if spliced:
            notes.append(
                f"{path.name}: {name} is the root document; "
                f"{len(spliced)} included file(s) were spliced in"
            )
        elif len(tex) > 1:
            notes.append(
                f"{path.name}: {len(tex)} LaTeX files found, treating {name} as "
                "the root document"
            )

        # A .bib is preferred when both are present: it holds the fields as
        # the author wrote them, where a .bbl holds only what BibTeX rendered.
        bib_name, bib_text, bib_kind = ("refs.bib", None, "bib")
        if bibs:
            bib_name, bib_text = max(bibs.items(), key=lambda kv: len(kv[1]))
            if len(bibs) > 1:
                notes.append(
                    f"{path.name}: {len(bibs)} .bib files found, using {bib_name}"
                )
        elif bbls:
            bib_name, bib_text = max(bbls.items(), key=lambda kv: len(kv[1]))
            bib_kind = "bbl"
            notes.append(
                f"{path.name}: no .bib, reading the compiled bibliography "
                f"{bib_name}; titles there are recovered from rendered text"
            )

        return Acquired(
            text=text,
            name=Path(name).name,
            path=f"{path.name}:{name}",
            from_archive=True,
            bib_text=bib_text,
            bib_name=Path(bib_name).name,
            bib_kind=bib_kind,
            notes=tuple(notes),
            regions=regions,
            files=tex,
        )


def acquire(target: str | Path, bib: str | Path | None = None) -> Acquired:
    """Resolve a path to LaTeX source, unpacking an archive if that is what it is."""
    path = Path(target)
    if not path.exists():
        raise UnreadableInput(f"no such file: {path}")

    lowered = path.name.lower()

    # Content decides, not the extension. arXiv serves the PDF from its
    # e-print endpoint when a submission has no LaTeX source, so a file named
    # ".tar.gz" is routinely a PDF -- and reporting that as "could not be
    # unpacked" hides the one thing the user needs to know.
    head = b""
    try:
        with path.open("rb") as handle:
            head = handle.read(8)
    except OSError:
        pass

    if lowered.endswith(".pdf") or head[:5] == b"%PDF-":
        raise UnreadableInput(
            f"{path.name} is a PDF, not LaTeX source — this paper was "
            "submitted without one. resint reads source, so there is nothing "
            "here it can check."
        )

    if any(lowered.endswith(s) for s in ARCHIVE_SUFFIXES):
        acquired = _from_archive(path)
    else:
        raw = path.read_bytes()
        if _looks_binary(raw):
            raise UnreadableInput(
                f"{path.name} is a binary file, not LaTeX source."
            )
        acquired = Acquired(
            text=_decode(raw, path.name), name=path.name, path=str(path)
        )

    if bib is not None:
        bib_path = Path(bib)
        if not bib_path.is_file():
            raise UnreadableInput(f"no such bibliography: {bib_path}")
        acquired.bib_text = _decode(bib_path.read_bytes(), bib_path.name)
        acquired.bib_name = bib_path.name
        acquired.bib_kind = "bbl" if bib_path.suffix.lower() == ".bbl" else "bib"

    return acquired


# --- multi-file documents -----------------------------------------------


@dataclass(frozen=True, slots=True)
class Region:
    """A stretch of the combined text, and where it really came from.

    Splicing included files into one string is the only way to read a paper
    that is split across a dozen of them, but it destroys the property the
    whole design rests on: that an offset points at a real place in a real
    file. A region map restores it -- every offset in the combined text can
    be translated back to a filename and an offset within that file, so
    "results.tex line 42" still means line 42 of results.tex.
    """

    start: int
    end: int
    name: str
    base: int

    def local(self, offset: int) -> int:
        return offset - self.start + self.base


_INPUT = re.compile(r"\\(?:input|include)\s*\{([^}]*)\}")
_MAX_DEPTH = 6


def _resolve_name(raw: str, available: dict[str, str]) -> str | None:
    r"""Match an \input argument against the files actually present."""
    wanted = raw.strip().strip('"')
    for candidate in (wanted, f"{wanted}.tex", f"{wanted}.ltx"):
        if candidate in available:
            return candidate
    tail = Path(wanted).name
    for candidate in (tail, f"{tail}.tex"):
        for key in available:
            if Path(key).name == candidate:
                return key
    return None


def expand_inputs(
    root_name: str, files: dict[str, str]
) -> tuple[str, tuple[Region, ...]]:
    r"""Splice \input and \include directives into one text, with a region map."""
    pieces: list[str] = []
    regions: list[Region] = []
    seen: set[str] = set()
    position = 0

    def emit(name: str, text: str, depth: int) -> None:
        nonlocal position
        if depth > _MAX_DEPTH or name in seen:
            return
        seen.add(name)

        cursor = 0
        for match in _INPUT.finditer(text):
            target = _resolve_name(match.group(1), files)
            if target is None:
                continue  # a missing include is the document's problem, not ours

            chunk = text[cursor : match.start()]
            if chunk:
                pieces.append(chunk)
                regions.append(Region(position, position + len(chunk), name, cursor))
                position += len(chunk)

            emit(target, files[target], depth + 1)
            cursor = match.end()

        rest = text[cursor:]
        if rest:
            pieces.append(rest)
            regions.append(Region(position, position + len(rest), name, cursor))
            position += len(rest)

    emit(root_name, files[root_name], 0)
    return "".join(pieces), tuple(regions)


def locate_region(regions: tuple[Region, ...], offset: int) -> Region | None:
    for region in regions:
        if region.start <= offset < region.end:
            return region
    return regions[-1] if regions else None
