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


def test_citations_inside_the_bibliography_are_not_use_sites():
    r"""natbib writes real \cite-shaped commands into its rendered labels:

        \bibitem[\protect\citeauthoryear{Duan, Hong, and Gu}{Duan
          et~al.}{2017}]{duan2017}

    \citeauthoryear matches the citation pattern, but its argument holds
    author names, not keys. Splitting on commas turned one reference into
    phantom citations of "Duan", "Hong" and "and Gu" -- across 204 real
    papers this was the largest single source of false positives in the tool.
    """
    raw = (
        r"We build on prior work \citep{duan2017}." "\n"
        r"\begin{thebibliography}{9}" "\n"
        r"\bibitem[\protect\citeauthoryear{Duan, Hong, and Gu}{Duan et~al.}{2017}]{duan2017}"
        "\n"
        r"Duan and others. A paper." "\n"
        r"\end{thebibliography}" "\n"
    )
    assert [c.key for c in extract_citations(raw, TEX)] == ["duan2017"]


def test_a_macro_template_is_not_a_citation():
    r"""\cite{#1} inside \newcommand is a definition, not a use site."""
    raw = r"\newcommand{\myref}[1]{\cite{#1}}" "\n" r"Real use \cite{smith2020}." "\n"
    assert [c.key for c in extract_citations(raw, TEX)] == ["smith2020"]


def test_an_unterminated_bibliography_still_stops_scanning():
    """A truncated source must not reopen the whole reference list."""
    raw = (
        r"Cited \citep{real2020}." "\n"
        r"\begin{thebibliography}{9}" "\n"
        r"\bibitem[\protect\citeauthoryear{Smith and Jones}{Smith}{2019}]{s2019}" "\n"
    )
    assert [c.key for c in extract_citations(raw, TEX)] == ["real2020"]


def test_a_real_citation_after_the_bibliography_is_still_found():
    """Blanking must be length-preserving and bounded, not a truncation."""
    raw = (
        r"\begin{thebibliography}{9}" "\n"
        r"\bibitem{a2020} A paper." "\n"
        r"\end{thebibliography}" "\n"
        r"Appendix cites \cite{b2021}." "\n"
    )
    found = extract_citations(raw, TEX)
    assert [c.key for c in found] == ["b2021"]
    assert raw[found[0].span.start : found[0].span.end] == "b2021"


def test_a_macro_definition_is_not_a_citation():
    r"""Pandoc emits \newcommand{\citeproc}[2]{...\cite-shaped...}, and the
    commands inside a definition are a template: there is no key there to
    resolve. Reading them as uses reported "#1" and "mm" as cited-but-undefined
    on every pandoc-produced paper in the corpus."""
    raw = (
        r"\newcommand{\citeproc}[2]{\hyper@linkstart{cite}{ref-#1}{#2}}" "\n"
        r"Real use \citep{smith2020}." "\n"
    )
    assert [c.key for c in extract_citations(raw, TEX)] == ["smith2020"]


def test_a_nested_macro_body_is_skipped_whole():
    r"""Brace matching, not a regex: the naive pattern stops at the first
    closing brace and leaves the rest of the body exposed."""
    raw = (
        r"\newcommand{\wrap}[1]{{\small \cite{#1} {\emph{\cite{inner}}}}}" "\n"
        r"Text \cite{real2021}." "\n"
    )
    assert [c.key for c in extract_citations(raw, TEX)] == ["real2021"]


@pytest.mark.parametrize(
    "definition",
    [
        r"\renewcommand{\foo}{\cite{ghost}}",
        r"\providecommand{\foo}{\cite{ghost}}",
        r"\DeclareRobustCommand{\foo}{\cite{ghost}}",
        r"\def\foo{\cite{ghost}}",
        r"\newcommand*{\foo}[2][x]{\cite{ghost}}",
    ],
)
def test_every_definition_form_is_skipped(definition):
    raw = definition + "\n" r"Body \cite{kept2020}." "\n"
    assert [c.key for c in extract_citations(raw, TEX)] == ["kept2020"]


def test_blanking_a_macro_keeps_offsets_and_lines_truthful():
    raw = (
        r"\newcommand{\foo}[1]{\cite{#1}}" "\n"
        r"\cite{real2020}" "\n"
    )
    found = extract_citations(raw, TEX)
    assert len(found) == 1
    span = found[0].span
    assert raw[span.start : span.end] == "real2020"
    assert span.line == 2


def test_an_xparse_argument_spec_is_not_a_citation_key():
    r"""pandoc writes its citation macro with xparse:

        \NewDocumentCommand\citeproc{mm}{\begingroup...\cite{#1}\endgroup}

    where {mm} is the argument specification -- "two mandatory arguments".
    Read as a citation it became a key named "mm", reported as
    cited-but-undefined on every pandoc-produced paper in the corpus.

    Note it is the *name and spec* that match the citation pattern, before
    the body is reached, so blanking only the body leaves the phantom key.
    """
    raw = (
        r"\NewDocumentCommand\citeproc{mm}{%" "\n"
        r"  \begingroup\def\citeproctext{#2}\cite{#1}\endgroup}" "\n"
        r"As shown \citeproc{ref-smith2020}{Smith 2020}." "\n"
    )
    assert [c.key for c in extract_citations(raw, TEX)] == ["ref-smith2020"]


@pytest.mark.parametrize(
    "form",
    [
        r"\NewDocumentCommand\foo{mm}{\cite{ghost}}",
        r"\RenewDocumentCommand\foo{m}{\cite{ghost}}",
        r"\ProvideDocumentCommand\foo{}{\cite{ghost}}",
        r"\DeclareDocumentCommand\foo{o m}{\cite{ghost}}",
    ],
)
def test_every_xparse_form_is_skipped(form):
    raw = form + "\n" r"Body \cite{kept2020}." "\n"
    assert [c.key for c in extract_citations(raw, TEX)] == ["kept2020"]


@pytest.mark.parametrize(
    "stated, canonical, drifted",
    [
        ("2015a", "2015", False),   # BibTeX disambiguation, not drift
        ("2015b", "2015", False),
        ("2015", "2015", False),
        ("2015", "2017", True),     # real drift: preprint vs proceedings
        ("in press", "2015", False),  # no year stated at all
    ],
)
def test_a_disambiguation_suffix_is_not_a_different_year(stated, canonical, drifted):
    """A bibliography using the 2015a/2015b convention had most of its entries
    reported as drifted, because the index has no equivalent to compare with."""
    from resint.rules.bib.drift import _year_digits

    a, b = _year_digits(stated), _year_digits(canonical)
    assert bool(a and b and a != b) is drifted
