"""BibTeX parsing, accent decoding, and citation extraction."""

import pytest

from resint.ir.span import Source
from resint.parse.bibtex import clean_value, fold, parse
from resint.parse.citations import extract_citations

BIB = Source("refs.bib", "bib", path="refs.bib")
TEX = Source("paper.tex", "latex", path="paper.tex")

SAMPLE = r"""
@article{smith2020,
  title   = {A Study of Things},
  author  = {Smith, Jane and Doe, John},
  journal = {Journal of Things},
  year    = {2020},
  doi     = {10.1000/xyz123}
}

@inproceedings{jones2019,
  title     = "Quoted Title Style",
  author    = "Jones, Alice",
  booktitle = "Proceedings of Somewhere",
  year      = 2019
}

@phdthesis{lee2018,
  title  = {{A Protected Title}},
  author = {Lee, Min},
  school = {Some University},
  year   = {2018}
}
"""


def parsed(text=SAMPLE):
    return parse(text, BIB)


# --- entries ------------------------------------------------------------


def test_reads_every_entry():
    result = parsed()
    assert [e.key for e in result.entries] == ["smith2020", "jones2019", "lee2018"]
    assert result.malformed == []


def test_entry_types_are_lowercased():
    assert [e.entry_type for e in parsed().entries] == [
        "article",
        "inproceedings",
        "phdthesis",
    ]


@pytest.mark.parametrize(
    "key, field, expected",
    [
        ("smith2020", "title", "A Study of Things"),
        ("smith2020", "doi", "10.1000/xyz123"),
        ("jones2019", "title", "Quoted Title Style"),
        ("jones2019", "year", "2019"),
        ("lee2018", "title", "A Protected Title"),
    ],
)
def test_field_values(key, field, expected):
    entry = parsed().by_key()[key]
    assert entry.fields[field] == expected


def test_authors_split_on_and():
    assert parsed().by_key()["smith2020"].authors == ["Smith, Jane", "Doe, John"]


def test_venue_falls_back_across_field_names():
    keys = parsed().by_key()
    assert keys["smith2020"].venue == "Journal of Things"
    assert keys["jones2019"].venue == "Proceedings of Somewhere"
    assert keys["lee2018"].venue == "Some University"


def test_unindexed_types_are_flagged():
    keys = parsed().by_key()
    assert keys["lee2018"].likely_unindexed
    assert not keys["smith2020"].likely_unindexed


# --- spans --------------------------------------------------------------


def test_field_spans_point_at_the_value():
    entry = parsed().by_key()["smith2020"]
    span = entry.field_spans["year"]
    assert SAMPLE[span.start : span.end] == "2020"


def test_span_for_prefers_the_first_present_field():
    entry = parsed().by_key()["smith2020"]
    assert entry.span_for("doi", "title") is entry.field_spans["doi"]
    assert entry.span_for("nonexistent", "title") is entry.field_spans["title"]
    assert entry.span_for("nonexistent") is entry.span


def test_entry_span_covers_the_whole_record():
    entry = parsed().by_key()["smith2020"]
    text = SAMPLE[entry.span.start : entry.span.end]
    assert text.startswith("@article{smith2020") and text.endswith("}")


# --- tolerance ----------------------------------------------------------


def test_a_malformed_entry_does_not_lose_the_rest():
    text = SAMPLE + "\n@article{broken2021,\n  title = {Unterminated\n"
    result = parse(text, BIB)
    assert len(result.entries) >= 3
    assert "smith2020" in result.by_key()


def test_string_and_comment_blocks_are_skipped():
    text = '@comment{ignored}\n@string{jt = "Journal of Things"}\n' + SAMPLE
    assert "ignored" not in parse(text, BIB).by_key()


def test_parenthesised_entries_are_supported():
    result = parse("@article(paren2020, title = {Works}, year = {2020})", BIB)
    assert result.by_key()["paren2020"].title == "Works"


def test_empty_input_yields_nothing():
    result = parse("", BIB)
    assert result.entries == [] and result.malformed == []


# --- accents ------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        (r"{\'E}tude", "Étude"),
        (r"m{\'e}thodes", "méthodes"),
        (r"Fran{\c c}ois", "François"),
        (r"M{\"u}ller", "Müller"),
        (r"Erd{\H o}s", "Erdős"),
        (r"Stra{\ss}e", "Straße"),
        (r"{\o}stergaard", "østergaard"),
        (r"A {Protected} Title", "A Protected Title"),
        ("Line\n  wrapped   title", "Line wrapped title"),
    ],
)
def test_accent_and_markup_decoding(raw, expected):
    assert clean_value(raw) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Étude des méthodes", "Etude des methodes"),
        ("Straße", "Strasse"),
        ("østergaard", "ostergaard"),
        ("Erdős", "Erdos"),
        ("plain ascii", "plain ascii"),
    ],
)
def test_folding_reaches_ascii_for_index_search(text, expected):
    assert fold(text) == expected


# --- citations ----------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    ["cite", "citep", "citet", "Citep", "citeauthor", "nocite"],
)
def test_all_citation_commands_are_recognised(command):
    found = extract_citations(f"Text \\{command}{{key2020}} here.", TEX)
    assert [c.key for c in found] == ["key2020"]


def test_multiple_keys_in_one_command_each_get_a_record():
    found = extract_citations(r"See \cite{a2020, b2021,c2022}.", TEX)
    assert [c.key for c in found] == ["a2020", "b2021", "c2022"]


def test_each_use_site_is_its_own_record():
    found = extract_citations(r"\cite{x2020} and later \cite{x2020}.", TEX)
    assert len(found) == 2
    assert found[0].span.start != found[1].span.start


def test_optional_arguments_are_skipped():
    found = extract_citations(r"\citep[see][p.~4]{key2020}", TEX)
    assert [c.key for c in found] == ["key2020"]


def test_commented_citations_are_ignored():
    raw = "Real \\cite{live2020}.\n% Dead \\cite{dead2020}.\n"
    assert [c.key for c in extract_citations(raw, TEX)] == ["live2020"]


def test_citation_spans_point_at_the_key_in_source():
    raw = r"Prior work~\cite{dosovitskiy2020} showed this."
    c = extract_citations(raw, TEX)[0]
    assert raw[c.span.start : c.span.end] == "dosovitskiy2020"


def test_citation_line_numbers_are_right():
    raw = "line one\nline two\nSee \\cite{k2020}.\n"
    assert extract_citations(raw, TEX)[0].span.line == 3
