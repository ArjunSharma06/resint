"""Choosing what to show a model.

Every model rule used to send the first 14,000 characters. Across 68 real
papers that meant the model saw 37% of the median paper and 14% of the longest
-- and always the *first* part, which is Introduction and Related Work. Five
rules whose job is reading results had been reading introductions.

The safety property these tests exist to protect: **what we send has no effect
on where a finding points.** Rules locate a quote against the full text, never
against the excerpt, so anchoring is untouched by any decision made here.
"""

import pytest

from resint.parse.document import paper_from_latex
from resint.parse.excerpt import excerpt, normalise, role_of

PAPER = r"""\documentclass{article}
\begin{document}
\begin{abstract}
We introduce a general-purpose method and evaluate it broadly.
\end{abstract}

\section{Introduction}
Prior work on this subject is extensive and mostly unrelated to our point.

\section{Methods}
We train for 200 epochs. The baseline was trained for 50 epochs.

\section{Results}
Our approach reaches 94.2 accuracy against the baseline's 93.9 on CIFAR-10.

\section{Acknowledgements}
We thank the funders, who had no role in the design of the study.
\end{document}
"""


def build(raw=PAPER):
    return paper_from_latex(raw, needs={"paper.text", "paper.sections"})


# --- picking the right sections -----------------------------------------


def test_a_rule_gets_the_sections_it_asked_for():
    got = excerpt(build(), ["results"], limit=4_000)
    assert "94.2 accuracy" in got.text
    assert "Prior work on this subject" not in got.text, "introduction was not asked for"


def test_the_abstract_is_reachable_though_it_is_not_a_section():
    """Neither format makes it one: LaTeX wraps it in an environment and JATS
    puts it in <front>. It is located by position instead."""
    got = excerpt(build(), ["abstract"], limit=4_000)
    assert "general-purpose method" in got.text


def test_priority_is_the_rules_order_not_the_documents():
    """A rule asking for results first means results matter most, whatever
    order they appear in the paper. Methods precedes Results in the document
    and still loses when the budget only stretches to one of them."""
    got = excerpt(build(), ["results", "methods"], limit=80)
    assert "94.2" in got.text
    assert "200 epochs" not in got.text, "the budget went to results, as asked"


def test_apparatus_is_never_sent():
    """'Acknowledgements' and 'Data availability' are not evidence of anything,
    and they occupy budget that results could have used."""
    got = excerpt(build(), ["abstract", "results", "discussion"], limit=8_000)
    assert "thank the funders" not in got.text


@pytest.mark.parametrize(
    "heading, expected",
    [
        ("Results", "results"),
        ("4.2 Experimental Setup", "methods"),
        ("Materials and Methods", "methods"),
        ("IV. Discussion", "discussion"),
        ("Results and Discussion", "results"),
        ("Author Contributions", ""),
        ("Data Availability", ""),
        ("Funding", ""),
    ],
)
def test_headings_are_matched_as_papers_actually_write_them(heading, expected):
    assert role_of(heading) == expected


def test_numbering_is_stripped():
    assert normalise("4.2 Experimental Setup") == "experimental setup"
    assert normalise("IV. Discussion") == "discussion"
    assert normalise("A) Methods") == "methods"


# --- the budget is always filled ----------------------------------------


def test_the_budget_is_respected():
    got = excerpt(build(), ["abstract", "results", "methods"], limit=120)
    assert got.chars <= 120 + len("=== Results ===\n") + 40


def test_a_paper_with_bespoke_headings_still_fills_the_budget():
    """Plenty of real papers are organised by subject rather than by IMRaD --
    a maths paper with 'Retained-sample delay-neutrality', a clinical review
    with 'Spasticity and dystonia'. Matching nothing left one 179,000
    character paper with 507 characters of abstract, which is worse than the
    whole-document window this replaced.
    """
    body = "\n".join(
        f"\\section{{Bespoke topic {n}}}\n"
        + "This paragraph discusses accuracy and evaluation at length. " * 12
        for n in range(8)
    )
    paper = build(
        "\\documentclass{article}\\begin{document}\n"
        "\\begin{abstract}Short.\\end{abstract}\n" + body + "\n\\end{document}\n"
    )
    got = excerpt(paper, ["results"], limit=3_000, query="accuracy evaluation")
    assert got.chars > 1_500, "the budget must be filled whatever the headings say"
    assert "retrieval" in got.strategy


def test_retrieval_does_not_repeat_what_a_section_already_gave():
    got = excerpt(build(), ["results"], limit=6_000, query="accuracy baseline")
    assert got.text.count("94.2 accuracy against") <= 1


def test_a_paper_with_no_sections_falls_back_rather_than_failing():
    paper = paper_from_latex(
        "\\begin{document}\n" + "Plain prose with no headings at all. " * 200
        + "\n\\end{document}\n",
        needs={"paper.text", "paper.sections"},
    )
    got = excerpt(paper, ["results"], limit=2_000)
    assert got.text
    assert got.strategy in ("leading", "retrieval")


def test_an_empty_paper_yields_an_empty_excerpt():
    paper = paper_from_latex("", needs={"paper.text", "paper.sections"})
    assert not excerpt(paper, ["results"], limit=1_000)


# --- the safety property ------------------------------------------------


def test_what_we_send_does_not_move_where_a_finding_points():
    """The whole reason this is safe to change. A quote taken from an excerpt
    is located against the *full* text, so anchoring, line numbers and the
    two-anchor invariant are unaffected by any choice made here."""
    from resint.model.verify import anchor_in

    paper = build()
    got = excerpt(paper, ["results"], limit=4_000)
    quote = "Our approach reaches 94.2 accuracy"
    assert quote in got.text

    span, found = anchor_in(paper.text, quote, "claim")
    assert span is not None
    assert paper.text.content[found.start : found.end] == quote


def test_text_stitched_across_a_gap_cannot_anchor():
    """Sending disjoint sections invites a model to quote across the join.

    Note the join has to be a real one. Normalization strips headings, so two
    *adjacent* sections are genuinely contiguous in the text and a quote
    spanning them legitimately anchors -- it is text the paper really contains.
    The guarantee is about sections the excerpt placed next to each other that
    the paper did not.
    """
    from resint.model.verify import anchor_in

    paper = build()
    got = excerpt(paper, ["abstract", "results"], limit=8_000)
    assert "[...]" in got.text, "the gap is marked"

    # Abstract and Results are far apart; Introduction and Methods sit between.
    stitched = "evaluate it broadly. Our approach reaches 94.2 accuracy"
    span, _ = anchor_in(paper.text, stitched, "claim")
    assert span is None
