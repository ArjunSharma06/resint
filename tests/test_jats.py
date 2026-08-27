"""Reading published papers, not just preprints.

resint read LaTeX, which meant it read arXiv. The literature that reports
statistics the way ``stats/grim`` and ``stats/pvalue-mismatch`` expect --
psychology, clinical medicine, epidemiology -- publishes through journals, and
PubMed Central serves those as JATS XML.

The gap was measured, not assumed: across 204 real arXiv papers the statistics
extractor found something in **2**. Three rules had never run on real input.

The design is one decision, and most of these tests exist to hold it: JATS
produces the *same* ``Normalized`` the LaTeX path produces, so every rule works
unchanged and cannot tell which format it was handed.
"""

from pathlib import Path

import pytest

from resint.engine import run
from resint.ir.span import Source
from resint.parse import jats_parts
from resint.parse.document import paper_from_jats, paper_from_path
from resint.parse.jats import looks_like_jats, normalize
from resint.rules import load_all

REG = load_all()
SRC = Source("article.nxml", "jats", path="article.nxml")

ARTICLE = """<?xml version="1.0"?>
<!DOCTYPE article PUBLIC "-//NLM//DTD JATS" "JATS.dtd">
<article article-type="research-article">
<front>
  <journal-meta><journal-title>Journal of Things</journal-title>
    <issn>1234-5678</issn>
  </journal-meta>
  <article-meta>
    <title-group><article-title>Effects of <italic>X</italic> on Y</article-title></title-group>
    <aff>Department of Irrelevance</aff>
    <abstract><p>Twenty people (N&#160;=&#160;20) took part, mean 3.47,
    t(20)&#x2009;=&#x2009;2.086, p&#160;=&#160;.03.</p></abstract>
  </article-meta>
</front>
<body>
  <sec id="s1"><title>Methods</title>
    <p>We recruited 20 undergraduates &amp; paid them.</p>
  </sec>
  <sec id="s2"><title>Results</title>
    <p>Accuracy reached 94.2 percent <xref ref-type="bibr" rid="B3">[3]</xref>.</p>
    <table-wrap id="t1"><label>Table 1</label>
      <caption><p>Scores by group</p></caption>
      <table>
        <tr><th>Method</th><th>Score</th></tr>
        <tr><td>Ours</td><td>94.2</td></tr>
      </table>
    </table-wrap>
  </sec>
</body>
<back>
  <ref-list><ref id="B3">
    <element-citation><article-title>A Cited Paper</article-title>
    <source>Some Journal</source><year>2019</year>
    <name><surname>Jones</surname></name></element-citation>
  </ref></ref-list>
</back>
</article>
"""


# --- what counts as the paper -------------------------------------------


def test_title_abstract_and_body_become_the_text():
    text = normalize(ARTICLE).text
    assert "Effects of X on Y" in text
    assert "mean 3.47" in text
    assert "Accuracy reached 94.2 percent" in text


@pytest.mark.parametrize(
    "apparatus",
    ["1234-5678", "Department of Irrelevance", "Journal of Things"],
)
def test_publisher_metadata_is_not_the_paper(apparatus):
    """<front> is mostly affiliations, funding and ISSNs. Letting it through
    would bury the abstract in boilerplate."""
    assert apparatus not in normalize(ARTICLE).text


def test_tables_and_references_are_not_prose():
    """Both are parsed into their own IR. Leaving them in the text would
    double-count every number in them."""
    text = normalize(ARTICLE).text
    assert "A Cited Paper" not in text
    assert "Some Journal" not in text
    # The table's own cells belong to paper.tables, not to the prose.
    assert text.count("94.2") == 1


def test_entities_are_decoded():
    text = normalize(ARTICLE).text
    assert "20 undergraduates & paid them" in text
    assert "&amp;" not in text
    assert "&#160;" not in text


def test_paragraphs_stay_separated():
    """Blank lines are load-bearing: resolve.passages splits on them and
    claim/unsupported locates the end of the front matter by section."""
    assert "\n\n" in normalize(ARTICLE).text


# --- the offset map, which is what makes a finding evidence --------------


def test_every_character_has_a_source_offset():
    doc = normalize(ARTICLE)
    assert len(doc.offsets) == len(doc.text)


@pytest.mark.parametrize("probe", ["3.47", "2.086", "94.2"])
def test_an_offset_points_at_the_real_text(probe):
    doc = normalize(ARTICLE)
    at = doc.text.find(probe)
    lo, hi = doc.raw_range(at, at + len(probe))
    assert ARTICLE[lo:hi] == probe


def test_a_span_lands_on_the_right_line_of_the_xml():
    """A finding at line 13 has to mean line 13 of the file the reader opens."""
    doc = normalize(ARTICLE)
    at = doc.text.find("2.086")
    lo, _ = doc.raw_range(at, at + 5)
    assert ARTICLE.splitlines()[doc.line_of(lo) - 1].strip().startswith("t(20)")


def test_an_anchor_inside_a_decoded_entity_still_resolves():
    """A decoded entity is several source characters for one text character,
    so the map is not one to one and cannot be assumed to be."""
    doc = normalize(ARTICLE)
    at = doc.text.find("N = 20")
    lo, hi = doc.raw_range(at, at + len("N = 20"))
    assert "N" in ARTICLE[lo:hi] and "20" in ARTICLE[lo:hi]


def test_sections_are_found_with_their_titles():
    assert [s.name for s in normalize(ARTICLE).sections] == ["Methods", "Results"]


# --- the structured matter ----------------------------------------------


def test_tables_are_extracted_with_captions_and_cells():
    tables = jats_parts.extract_tables(ARTICLE, SRC)
    assert len(tables) == 1
    assert tables[0].label == "Table 1"
    assert "Scores by group" in tables[0].caption
    assert tables[0].header == ["Method", "Score"]
    assert tables[0].rows[1][1].text == "94.2"


def test_a_table_cell_anchors_into_the_xml():
    cell = jats_parts.extract_tables(ARTICLE, SRC)[0].rows[1][1]
    assert ARTICLE[cell.span.start : cell.span.end] == "94.2"


def test_references_become_bib_entries_keyed_by_their_id():
    """The id is what <xref rid=...> points at, so it plays exactly the role a
    BibTeX key plays -- which is what lets bib/orphans work unchanged."""
    bib = jats_parts.extract_bib(ARTICLE, SRC)
    assert [e.key for e in bib.entries] == ["B3"]
    entry = bib.entries[0]
    assert entry.title == "A Cited Paper"
    assert entry.year == "2019"
    assert "Jones" in entry.fields.get("author", "")


def test_citations_come_from_bibr_cross_references():
    cites = jats_parts.extract_citations(ARTICLE, SRC)
    assert [c.key for c in cites] == ["B3"]


def test_a_cross_reference_to_a_figure_is_not_a_citation():
    """JATS marks figures, tables and sections with <xref> too. Only
    ref-type="bibr" cites anything."""
    raw = ARTICLE.replace('ref-type="bibr" rid="B3"', 'ref-type="fig" rid="f1"')
    assert jats_parts.extract_citations(raw, SRC) == []


def test_one_xref_may_name_several_references():
    raw = ARTICLE.replace('rid="B3"', 'rid="B3 B4"')
    assert [c.key for c in jats_parts.extract_citations(raw, SRC)] == ["B3", "B4"]


# --- the whole point ----------------------------------------------------


def test_the_statistics_rules_finally_see_real_input():
    """Across 204 arXiv papers the statistics extractor fired on 2. This is
    the format the literature that reports them actually uses."""
    paper = paper_from_jats(ARTICLE)
    assert len(paper.stats) == 1
    report = run(paper, registry=REG)
    assert [f.rule_id for f in report.findings] == ["stats/pvalue-mismatch"]


def test_a_cited_and_defined_reference_produces_no_orphan_finding():
    paper = paper_from_jats(ARTICLE, needs={"paper.bib", "paper.citations"})
    assert [f.rule_id for f in run(paper, registry=REG).findings] == []


def test_an_uncited_reference_is_reported():
    raw = ARTICLE.replace('<xref ref-type="bibr" rid="B3">[3]</xref>', "")
    paper = paper_from_jats(raw, needs={"paper.bib", "paper.citations"})
    findings = run(paper, registry=REG).findings
    assert [f.rule_id for f in findings] == ["bib/orphans"]
    assert "B3" in findings[0].message


# --- recognising the format ---------------------------------------------


def test_a_jats_article_is_recognised():
    assert looks_like_jats(ARTICLE)


@pytest.mark.parametrize(
    "other",
    [
        '<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg"><rect/></svg>',
        "<html><body><p>Not a paper</p></body></html>",
        "\\documentclass{article}\\begin{document}Hello\\end{document}",
        "",
    ],
)
def test_other_documents_are_not_mistaken_for_jats(other):
    """An .xml file is not necessarily a paper, so content decides -- the same
    rule the PDF sniffing follows."""
    assert not looks_like_jats(other)


def test_the_format_is_chosen_by_content_not_extension(tmp_path):
    misnamed = tmp_path / "paper.tex"
    misnamed.write_text(ARTICLE, encoding="utf-8")
    paper = paper_from_path(misnamed, needs={"paper.text", "paper.stats"})
    assert len(paper.stats) == 1, "a .tex holding JATS is still JATS"


def test_an_nxml_file_goes_through_the_whole_pipeline(tmp_path):
    path = tmp_path / "article.nxml"
    path.write_text(ARTICLE, encoding="utf-8")
    paper = paper_from_path(path)
    assert paper.text and paper.stats and paper.bib


# --- malformed input ----------------------------------------------------


@pytest.mark.parametrize(
    "broken",
    [
        "<article><body><sec><p>Unclosed paragraph</body></article>",
        "<article><body><p>Bad & entity &nosuch; here</p></body></article>",
        "<article><body><p>A < B and C > D</p></body></article>",
        "<article><front></front><body></body></article>",
    ],
)
def test_malformed_markup_does_not_raise(broken):
    """A strict parser rejects these outright. The input is several million
    files nobody validated, so refusing is not an option."""
    doc = normalize(broken)
    assert len(doc.offsets) == len(doc.text)


def test_an_article_with_no_body_yields_no_text():
    """Falling back to the whole document would emit publisher metadata as
    though it were the paper."""
    assert normalize("<article><front><issn>1234</issn></front></article>").text == ""


def test_an_uncited_reference_is_not_blamed_on_bibtex():
    """There is no BibTeX in a journal article. A JATS <ref> behaves like a
    \bibitem -- the reference list is the article's own furniture and every
    entry in it is typeset -- so an uncited one appears with nothing pointing
    at it. Six real PubMed Central articles were told the opposite.
    """
    raw = ARTICLE.replace('<xref ref-type="bibr" rid="B3">[3]</xref>', "")
    paper = paper_from_jats(raw, needs={"paper.bib", "paper.citations"})
    message = run(paper, registry=REG).findings[0].message
    assert "will appear in the reference list" in message
    assert "BibTeX" not in message
