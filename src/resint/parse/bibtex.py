"""BibTeX parsing.

Field spans are the reason this is a parser rather than a regex sweep. A
finding about a wrong year has to point at the year, not at the entry, or the
author has to go hunting for what the tool already knew.

Deliberately tolerant: a bibliography with one malformed entry should still
yield the other two hundred. Malformed entries are recorded and skipped, not
raised.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from ..ir.paper import BibEntry
from ..ir.span import Source, Span

_ENTRY = re.compile(r"@(?P<type>[A-Za-z]+)\s*[{(]\s*(?P<key>[^\s,}()]+)\s*,")
_SKIP_TYPES = {"comment", "preamble", "string"}

# Accent commands, mapped to the combining mark they apply. Titles reach the
# indices as text, so "{\'E}tude" has to become "Etude" and not a backslash
# soup that matches nothing.
_COMBINING = {
    "'": "\u0301", "`": "\u0300", '"': "\u0308", "^": "\u0302",
    "~": "\u0303", "=": "\u0304", ".": "\u0307", "u": "\u0306",
    "v": "\u030c", "c": "\u0327", "H": "\u030b", "r": "\u030a",
    "k": "\u0328", "b": "\u0331", "d": "\u0323",
}

# Standalone letter commands with no base character to combine with.
_STANDALONE = {
    "ss": "ß", "o": "ø", "O": "Ø", "aa": "å", "AA": "Å",
    "ae": "æ", "AE": "Æ", "oe": "œ", "OE": "Œ", "l": "ł", "L": "Ł",
    "i": "i", "j": "j",
}

_ACCENT_SYMBOL = re.compile(r"\\([`'\"^~=.])\s*\{?\\?([a-zA-Z])\}?")
_ACCENT_ALPHA = re.compile(r"\\([a-zA-Z])\s*\{(\\?[a-zA-Z])\}|\\([a-zA-Z])\s+([a-zA-Z])")
_STANDALONE_RE = re.compile(r"\\([a-zA-Z]+)\s*(?:\{\})?")
_MARKUP = re.compile(r"\\[a-zA-Z]+\s*\{([^{}]*)\}")
_BRACE = re.compile(r"[{}]")
_WS = re.compile(r"\s+")


def _compose(base: str, command: str) -> str:
    mark = _COMBINING.get(command)
    if mark is None:
        return base
    return unicodedata.normalize("NFC", base.lstrip("\\") + mark)


def clean_value(raw: str) -> str:
    """Strip brace protection and decode LaTeX accents into real characters."""
    text = _ACCENT_SYMBOL.sub(lambda m: _compose(m.group(2), m.group(1)), raw)

    def _alpha(m: re.Match) -> str:
        cmd = m.group(1) or m.group(3)
        base = m.group(2) or m.group(4)
        if cmd in _COMBINING:
            return _compose(base, cmd)
        return m.group(0)

    text = _ACCENT_ALPHA.sub(_alpha, text)
    text = _STANDALONE_RE.sub(
        lambda m: _STANDALONE.get(m.group(1), m.group(0)), text
    )
    text = _MARKUP.sub(r"\1", text)
    text = _BRACE.sub("", text)
    return _WS.sub(" ", text).strip()


# Letters with no decomposition, so NFD leaves them untouched. Index search
# is ASCII, and a title stuck on "Straßen" matches nothing.
_UNFOLDABLE = str.maketrans(
    {
        "ß": "ss", "ø": "o", "Ø": "O", "æ": "ae", "Æ": "AE",
        "œ": "oe", "Œ": "OE", "ł": "l", "Ł": "L", "đ": "d", "Đ": "D",
        "ð": "d", "Ð": "D", "þ": "th", "Þ": "Th", "ı": "i",
    }
)


def fold(text: str) -> str:
    """ASCII-fold for comparison. Display keeps the accents; matching does not."""
    decomposed = unicodedata.normalize("NFD", text.translate(_UNFOLDABLE))
    return "".join(c for c in decomposed if not unicodedata.combining(c))


@dataclass
class BibFile:
    entries: list[BibEntry] = field(default_factory=list)
    malformed: list[str] = field(default_factory=list)

    def by_key(self) -> dict[str, BibEntry]:
        return {e.key: e for e in self.entries}


def _match_delimiter(src: str, i: int) -> int:
    """Index just past the group opened at ``i`` by '{' or '('."""
    opener = src[i]
    closer = "}" if opener == "{" else ")"
    depth, j = 1, i + 1
    while j < len(src) and depth:
        c = src[j]
        if c == "\\":
            j += 2
            continue
        if c == opener:
            depth += 1
        elif c == closer:
            depth -= 1
        j += 1
    return j


def _read_value(src: str, i: int) -> tuple[str, int, int, int]:
    """Read a field value. Returns (value, value_start, value_end, next_index)."""
    while i < len(src) and src[i] in " \t\n":
        i += 1
    if i >= len(src):
        return "", i, i, i

    if src[i] == "{":
        end = _match_delimiter(src, i)
        return src[i + 1 : end - 1], i + 1, end - 1, end

    if src[i] == '"':
        j, depth = i + 1, 0
        while j < len(src):
            if src[j] == "\\":
                j += 2
                continue
            if src[j] == "{":
                depth += 1
            elif src[j] == "}":
                depth -= 1
            elif src[j] == '"' and depth == 0:
                break
            j += 1
        return src[i + 1 : j], i + 1, j, j + 1

    j = i
    while j < len(src) and src[j] not in ",}\n)":
        j += 1
    return src[i:j].strip(), i, j, j


def parse(text: str, src: Source) -> BibFile:
    """Parse BibTeX source into entries carrying spans for every field."""
    out = BibFile()

    for m in _ENTRY.finditer(text):
        entry_type = m.group("type").lower()
        if entry_type in _SKIP_TYPES:
            continue

        brace = text.rfind("{", m.start(), m.end())
        paren = text.rfind("(", m.start(), m.end())
        open_at = max(brace, paren)
        if open_at == -1:
            out.malformed.append(f"{m.group('key')}: no opening delimiter")
            continue

        entry_end = _match_delimiter(text, open_at)
        if entry_end > len(text):
            out.malformed.append(f"{m.group('key')}: unterminated entry")
            continue

        fields: dict[str, str] = {}
        field_spans: dict[str, Span] = {}

        i = m.end()
        limit = entry_end - 1
        while i < limit:
            while i < limit and text[i] in " \t\n,":
                i += 1
            name_start = i
            while i < limit and (text[i].isalnum() or text[i] in "_-"):
                i += 1
            name = text[name_start:i].lower()
            if not name:
                break

            while i < limit and text[i] in " \t\n":
                i += 1
            if i >= limit or text[i] != "=":
                break
            i += 1

            value, v_start, v_end, i = _read_value(text, i)
            if name in fields:
                continue
            fields[name] = clean_value(value)
            field_spans[name] = Span(
                src,
                v_start,
                max(v_end, v_start + 1),
                line=text.count("\n", 0, v_start) + 1,
                label=f"[{m.group('key')}].{name}",
            )

        out.entries.append(
            BibEntry(
                key=m.group("key"),
                entry_type=entry_type,
                fields=fields,
                span=Span(
                    src,
                    m.start(),
                    entry_end,
                    line=text.count("\n", 0, m.start()) + 1,
                    label=f"[{m.group('key')}]",
                ),
                field_spans=field_spans,
            )
        )

    return out
