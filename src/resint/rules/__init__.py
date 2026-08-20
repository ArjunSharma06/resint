"""Rule discovery.

Importing this package registers every rule. Discovery walks the family
subpackages rather than naming modules, so contributing rule 47 means adding
one file -- there is no central list to remember to edit.
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

from .registry import REGISTRY, Context, Registry, Rule, rule

_FAMILIES = ("numbers", "stats", "bib", "repro", "claim", "eval")


def load_all() -> Registry:
    root = Path(__file__).parent
    for family in _FAMILIES:
        package = root / family
        if not package.is_dir():
            continue
        for mod in pkgutil.iter_modules([str(package)]):
            if not mod.name.startswith("_"):
                importlib.import_module(f"{__name__}.{family}.{mod.name}")
    return REGISTRY


__all__ = ["REGISTRY", "Context", "Registry", "Rule", "rule", "load_all"]
