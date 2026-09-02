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


# =========================================================================
# The NHST screen
#
# Corpus selection decides which rules can be measured at all, so the screen
# is a measurement instrument and not a convenience. Two properties matter
# more than its recall: it must not select using our own extractor, and it
# must not rank on the thing it is screening for.
# =========================================================================


def _screen():
    return _tool("screen_nhst")


@pytest.mark.parametrize(
    "text, why",
    [
        ("t(20) = 2.086, p = .03", "the canonical form"),
        ("<italic>t</italic>(20) = 2.086, <italic>p</italic> = .03", "JATS markup"),
        ("F(2, 45) = 3.14, p < .05", "F with degrees of freedom"),
        ("chi2 = 9.1, p = .002", "chi-square written out"),
        ("z = 1.96, p = 0.05", "no degrees of freedom"),
        ("r(48) = .31, P > .05", "capital P, correlation"),
    ],
)
def test_the_screen_finds_inline_nhst(text, why):
    assert _screen().hits(_screen().plain(text)) == 1, why


@pytest.mark.parametrize(
    "text, why",
    [
        ("We set alpha at p < .05 for all tests.", "a threshold is not a result"),
        ("The model has 12 layers, 8 heads, and 110M parameters.", "ML prose"),
        ("t(20) = 2.086. " + "x" * 400 + " p = .03", "too far apart to be one claim"),
        ("", "empty"),
    ],
)
def test_the_screen_rejects_what_is_not_a_reported_test(text, why):
    assert _screen().hits(_screen().plain(text)) == 0, why


def test_an_escaped_less_than_still_reads_as_a_p_value():
    """XML escapes `<`, so every "p < .001" in a JATS corpus is stored as
    "p &lt; .001" -- the most common way a p-value is written. Screening
    without decoding entities rejected it silently: 18 of 224 papers passed,
    against 34 once decoded."""
    assert _screen().hits(_screen().plain("t(20) = 2.086, p &lt; .001")) == 1


def test_entities_are_decoded_after_tags_not_before():
    """The other order turns a literal &lt;p&gt; into a tag and deletes it."""
    assert "<p>" in _screen().plain("a &lt;p&gt; b")


def test_markup_is_stripped_but_nothing_else_is():
    """One substitution, deliberately far short of parsing. Screening with
    parse/extract.py would select exactly the papers that extractor already
    handles, and the fire rate measured afterwards would be inflated by
    construction."""
    assert _screen().plain("<p>t</p>(20)").split() == ["t", "(20)"]
    # Text between tags survives intact; only the tags become whitespace.
    assert _screen().plain("a <b>c</b> d").split() == ["a", "c", "d"]


def test_papers_are_kept_in_name_order_not_by_score(tmp_path):
    """The threshold decides membership and nothing else does. Ranking by hit
    count and taking the top N would build a corpus of the most
    statistics-dense papers in medicine, and any rate measured on it would
    then be published as a rate on papers generally."""
    dense = "t(20) = 2.086, p = .03. F(1, 9) = 4.2, p = .01. z = 3.1, p < .001. r(8) = .7, p = .02."
    sparse = "t(20) = 2.086, p = .03. F(1, 9) = 4.2, p = .01."

    (tmp_path / "PMC900.nxml").write_text(dense, encoding="utf-8")
    (tmp_path / "PMC100.nxml").write_text(sparse, encoding="utf-8")

    import io
    from contextlib import redirect_stdout

    out = io.StringIO()
    with redirect_stdout(out):
        _screen().main([str(tmp_path), "--keep", "1"])

    # PMC100 is sparser and first alphabetically. Name order wins.
    assert out.getvalue().split() == ["PMC100"]


def test_the_screen_can_be_restricted_to_one_pool(tmp_path, capsys):
    """The cache accumulates across corpora. Screening the whole directory
    would mix August's articles into September's corpus while the corpus file
    claimed a single provenance."""
    body = "t(20) = 2.086, p = .03. F(1, 9) = 4.2, p = .01."
    for name in ("PMC1.nxml", "PMC2.nxml", "PMC3.nxml"):
        (tmp_path / name).write_text(body, encoding="utf-8")
    ids = tmp_path / "pool.txt"
    ids.write_text("PMC1\nPMC3\n", encoding="utf-8")

    _screen().main([str(tmp_path), "--ids", str(ids)])
    assert capsys.readouterr().out.split() == ["PMC1", "PMC3"]


def test_arxiv_listing_deduplicates_across_categories(monkeypatch):
    """Categories overlap -- a paper cross-listed to cs.LG and cs.CV answers
    two queries. Left in, "132 listed" and the 120 distinct papers behind it
    drift apart, and a corpus file built from the list claims a size the sweep
    does not see. Caught by round-tripping the corpus file through read_ids:
    180 lines came back as 164 identifiers."""
    tool = _tool("fetch_arxiv")
    monkeypatch.setattr(tool, "_request",
                        lambda url, mailto: b"<id>http://arxiv.org/abs/2608.1v1</id>")
    monkeypatch.setattr(tool.time, "sleep", lambda _: None)
    assert tool.list_ids(["cs.LG", "cs.CV"], 1, None) == ["2608.1v1"]
