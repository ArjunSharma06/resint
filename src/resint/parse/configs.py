"""Reading YAML and JSON configuration into flat, anchored keys.

The YAML reader handles the subset research configs actually use: nested
mappings, scalars, and simple lists. It is not a YAML implementation and does
not try to be -- anchors, multi-document streams, and flow mappings are
skipped and reported rather than half-parsed, because a hyperparameter read
wrongly is worse than one not read at all.

Hydra's ``defaults:`` list is recognised specifically. A config that composes
another one overrides it, so keys from an overriding file bind more strongly;
without that distinction every Hydra repository looks like a pile of
contradictory values and the rule abstains on everything.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from ..ir.repo import Binding, ConfigKey
from ..ir.span import Source, Span

_ENTRY = re.compile(r"^(?P<indent>\s*)(?P<key>[A-Za-z_][\w.-]*)\s*:\s*(?P<value>.*?)\s*$")
_LIST_ITEM = re.compile(r"^(?P<indent>\s*)-\s+(?P<value>.*?)\s*$")
_UNSUPPORTED = re.compile(r"^\s*(?:<<:|&\w|\*\w|---|\.\.\.)")

_SCALAR_SKIP = {"", "null", "~", "|", ">", "|-", ">-", "{}", "[]"}


@dataclass
class ConfigFile:
    keys: list[ConfigKey] = field(default_factory=list)
    composes: list[str] = field(default_factory=list)
    unchecked: list[str] = field(default_factory=list)


def _clean_scalar(value: str) -> str | None:
    text = value.strip()
    if "#" in text:
        quoted = text[:1] in "\"'"
        if not quoted:
            text = text.split("#", 1)[0].strip()
    if text.lower() in _SCALAR_SKIP:
        return None
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1]
    return text


def read_yaml(text: str, src: Source, rel: str, binding: Binding) -> ConfigFile:
    """Flatten a YAML config into dotted keys with spans."""
    out = ConfigFile()
    stack: list[tuple[int, str]] = []
    offsets, total = [0], 0
    for line in text.splitlines(keepends=True):
        total += len(line)
        offsets.append(total)

    in_defaults = False
    defaults_indent = 0

    for lineno, raw in enumerate(text.splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        if _UNSUPPORTED.match(raw):
            out.unchecked.append(
                f"{rel}:{lineno}: YAML feature not supported by the built-in "
                "reader; the line was skipped rather than guessed at"
            )
            continue

        item = _LIST_ITEM.match(raw)
        if item and in_defaults and len(item.group("indent")) > defaults_indent:
            composed = _clean_scalar(item.group("value"))
            if composed and not composed.startswith("_"):
                out.composes.append(composed.split(":")[-1].strip())
            continue

        match = _ENTRY.match(raw)
        if not match:
            continue

        indent = len(match.group("indent"))
        key = match.group("key")
        value = match.group("value")

        while stack and stack[-1][0] >= indent:
            stack.pop()

        if key == "defaults" and not value:
            in_defaults, defaults_indent = True, indent
            continue
        if indent <= defaults_indent and key != "defaults":
            in_defaults = False

        scalar = _clean_scalar(value)
        if scalar is None:
            stack.append((indent, key))
            continue

        dotted = ".".join([name for _, name in stack] + [key])
        start = offsets[lineno - 1] + raw.index(value if value else key)
        out.keys.append(
            ConfigKey(
                name=dotted,
                raw_name=key,
                value=scalar,
                binding=binding,
                span=Span(
                    src,
                    start,
                    start + max(len(value.strip()), 1),
                    line=lineno,
                    label=rel,
                ),
                origin=rel,
            )
        )

    return out


def read_json(text: str, src: Source, rel: str, binding: Binding) -> ConfigFile:
    """Flatten a JSON config. Spans point at the key, which is findable."""
    out = ConfigFile()
    try:
        data = json.loads(text)
    except ValueError as exc:
        out.unchecked.append(f"{rel}: not valid JSON ({exc})")
        return out
    if not isinstance(data, dict):
        return out

    def walk(node: dict, prefix: str) -> None:
        for key, value in node.items():
            dotted = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                walk(value, dotted)
                continue
            if isinstance(value, (list, type(None))):
                continue
            rendered = str(value).lower() if isinstance(value, bool) else str(value)
            needle = f'"{key}"'
            index = text.find(needle)
            start = index if index != -1 else 0
            out.keys.append(
                ConfigKey(
                    name=dotted,
                    raw_name=key,
                    value=rendered,
                    binding=binding,
                    span=Span(
                        src,
                        start,
                        start + len(needle),
                        line=text.count("\n", 0, start) + 1,
                        label=rel,
                    ),
                    origin=rel,
                )
            )

    walk(data, "")
    return out
