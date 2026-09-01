"""Corpus reproducibility.

A corpus assembled by discovery cannot be rebuilt: `--count 100` returns
whatever arXiv listed that morning, so a sweep can never be re-run against the
material it actually measured, and two sweeps a week apart are not comparable.
Worse, discovery decides *what gets tested* -- an all-cs corpus leaves the
statistics rules with no real input at all, which is how they reached a
release without ever having fired on a real paper.

`--ids` is what makes a corpus a deliberate object rather than a snapshot.
These tests cover the parsing, which is the part with decisions in it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"


def _tool(name: str):
    """Load a script from tools/, which is not an importable package.

    Registered in sys.modules before execution: these scripts use
    ``from __future__ import annotations`` with @dataclass, and dataclasses
    resolves a field's type by looking its module up there. Skipping that
    step fails inside the decorator, well before any test runs.
    """
    key = f"_tool_{name}"
    if key in sys.modules:
        return sys.modules[key]

    spec = importlib.util.spec_from_file_location(key, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


BOTH = pytest.mark.parametrize("tool", ["fetch_arxiv", "fetch_pmc"])


@BOTH
def test_one_identifier_per_line(tool, tmp_path):
    path = tmp_path / "ids.txt"
    path.write_text("2608.11185v1\n2608.12000v1\n", encoding="utf-8")
    assert _tool(tool).read_ids(path) == ["2608.11185v1", "2608.12000v1"]


@BOTH
def test_comments_say_what_each_block_is_for(tool, tmp_path):
    """The composition is the point of the file, so it has to be legible.
    A list that cannot record "these 50 are for the statistics rules" is a
    list nobody will maintain deliberately."""
    path = tmp_path / "ids.txt"
    path.write_text(
        "# arXiv CS -- tables, hyperparameters, repos\n"
        "2608.11185v1\n"
        "\n"
        "# stats/econ -- inline NHST\n"
        "2608.12000v1  # kept: reports t and p inline\n",
        encoding="utf-8",
    )
    assert _tool(tool).read_ids(path) == ["2608.11185v1", "2608.12000v1"]


@BOTH
def test_duplicates_are_dropped_so_lists_can_be_concatenated(tool, tmp_path):
    """Strata are composed separately and catted together; an id in two of
    them must not be fetched or swept twice."""
    path = tmp_path / "ids.txt"
    path.write_text("PMC1\nPMC2\nPMC1\n", encoding="utf-8")
    assert _tool(tool).read_ids(path) == ["PMC1", "PMC2"]


@BOTH
def test_order_is_preserved(tool, tmp_path):
    """Not sorted. The file's order is the composition, and a sweep that dies
    halfway should have covered the strata in the proportions intended."""
    path = tmp_path / "ids.txt"
    path.write_text("c\na\nb\n", encoding="utf-8")
    assert _tool(tool).read_ids(path) == ["c", "a", "b"]


@BOTH
def test_a_missing_id_file_exits_rather_than_falling_back_to_discovery(
    tool, tmp_path, capsys
):
    """Silently discovering instead would produce a corpus that looks like the
    requested one and is not -- the failure this whole flag exists to prevent,
    arriving without a message."""
    assert _tool(tool).main(["--ids", str(tmp_path / "nope.txt")]) == 2
    assert "no such id file" in capsys.readouterr().err
