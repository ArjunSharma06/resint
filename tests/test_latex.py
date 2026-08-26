"""LaTeX normalization, with the offset map as the property under test.

If offsets drift, every finding points at the wrong line and the tool is
worse than useless -- confidently wrong. So most of these tests assert
positions, not just text.
"""

import pytest

from resint.parse.latex import normalize


def test_comments_are_removed():
    d = normalize("before\n% a secret comment\nafter")
    assert "secret" not in d.text
    assert "before" in d.text and "after" in d.text


def test_escaped_percent_survives():
    d = normalize(r"Accuracy reached 94.2\% overall.")
    assert "94.2%" in d.text


def test_dropped_environments_lose_their_body():
    d = normalize(
        "keep this\n\\begin{figure}\nDISCARD ME\n\\end{figure}\nkeep this too"
    )
    assert "DISCARD ME" not in d.text
    assert "keep this" in d.text and "keep this too" in d.text


def test_zero_arg_macros_expand():
    d = normalize(r"\newcommand{\method}{LoRA}" "\n" r"We propose \method{} here.")
    assert "We propose LoRA here." in d.text


def test_macro_definition_body_does_not_leak_as_text():
    """The definition is scaffolding; only uses of the macro are content."""
    d = normalize(r"\newcommand{\method}{LoRA}" "\nPlain sentence.")
    assert d.text.strip() == "Plain sentence."


def test_def_form_is_also_skipped():
    d = normalize(r"\def\foo{BAR}" "\nPlain sentence.")
    assert "BAR" not in d.text


def test_citations_leave_no_key_in_running_text():
    d = normalize(r"As shown by \cite{dosovitskiy2020}, this works.")
    assert "dosovitskiy2020" not in d.text
    assert "As shown by" in d.text and "this works" in d.text


def test_formatting_commands_keep_their_argument():
    d = normalize(r"We use \textbf{ViT-B} and \emph{LoRA}.")
    assert "ViT-B" in d.text and "LoRA" in d.text
    assert "textbf" not in d.text


def test_math_delimiters_are_stripped_but_content_kept():
    d = normalize(r"The effect held, $t(20) = 2.086$.")
    assert "t(20) = 2.086" in d.text
    assert "$" not in d.text


def test_sections_are_captured_with_ranges():
    d = normalize(r"\section{Method}" "\nAlpha text.\n" r"\section{Results}" "\nBeta text.")
    assert [s.name for s in d.sections] == ["Method", "Results"]
    assert d.section_at(d.text.index("Alpha")) == "Method"
    assert d.section_at(d.text.index("Beta")) == "Results"


# --- the offset map -----------------------------------------------------


def test_offsets_map_back_to_the_exact_source_text():
    src = "Intro.\n% comment\nThe value was 2.086 exactly."
    d = normalize(src)
    i = d.text.index("2.086")
    lo, hi = d.raw_range(i, i + 5)
    assert src[lo:hi] == "2.086"


def test_line_numbers_survive_dropped_content():
    src = "\n".join(
        [
            r"\documentclass{article}",   # line 1
            "% comment",                  # line 2
            r"\begin{figure}",            # line 3
            "dropped",                    # line 4
            r"\end{figure}",              # line 5
            "The value was 2.086 here.",  # line 6
        ]
    )
    d = normalize(src)
    lo, _ = d.raw_range(d.text.index("2.086"), d.text.index("2.086") + 5)
    assert d.line_of(lo) == 6


def test_offset_map_length_matches_text():
    d = normalize(r"\section{X}" "\nSome text with \\textbf{bold} in it.")
    assert len(d.offsets) == len(d.text)


def test_every_literal_character_points_at_itself():
    """Macro expansion aside, each normalized char sits over its source char."""
    src = r"\section{Results} We found 2.086 and 94.2\% here \cite{k}."
    d = normalize(src)
    mismatches = [
        (i, c, d.raw[d.offsets[i]])
        for i, c in enumerate(d.text)
        if c not in " \n%" and d.raw[d.offsets[i]] != c
    ]
    assert mismatches == []


def test_inconsistent_offset_map_is_rejected():
    from resint.parse.latex import Normalized

    with pytest.raises(ValueError, match="offset map"):
        Normalized(text="abc", offsets=[0, 1], raw="abc")


def test_empty_input_is_handled():
    d = normalize("")
    assert d.text == "" and d.offsets == []
    assert d.raw_offset(0) == 0


# --- what a model gets shown --------------------------------------------
#
# Normalization strips command names but keeps their arguments, so a LaTeX
# preamble survives into the text as a run of noise. A rule scanning for
# numbers never noticed. A prompt cannot afford it: it lands in the position a
# model attends to most, and under truncation it displaces real content.
#
# Found by running tools/try_model.py --dry over real arXiv sources, where
# every prompt opened with six hundred characters of theorem declarations.


PREAMBLE = "\n".join(
    [
        r"\documentclass{article}",
        r"\usepackage{amsmath}",
        r"\newtheorem{theorem}{Theorem}[section]",
        r"\newtheorem{lemma}[theorem]{Lemma}",
        r"\newcommand{\dave}[1]{\textcolor{blue}{[Dave] #1}}",
        r"\begin{document}",
        "The first real sentence of the paper appears here.",
        r"\end{document}",
    ]
)


def test_the_body_starts_after_the_preamble():
    from resint.parse.document import paper_from_latex

    paper = paper_from_latex(PREAMBLE, needs={"paper.text"})
    assert paper.text.window(200).strip().startswith("The first real sentence")


def test_the_preamble_is_still_in_the_content():
    """window() is a view for prompting, not a different document. Offsets
    stay valid, so a quote taken from the window still anchors."""
    from resint.parse.document import paper_from_latex

    paper = paper_from_latex(PREAMBLE, needs={"paper.text"})
    assert paper.text.body_start > 0
    assert len(paper.text.content) > len(paper.text.window(10_000))


def test_a_quote_from_the_window_still_anchors():
    from resint.model.verify import anchor_in
    from resint.parse.document import paper_from_latex

    paper = paper_from_latex(PREAMBLE, needs={"paper.text"})
    span, found = anchor_in(paper.text, "The first real sentence of the paper", "claim")
    assert span is not None
    assert found.start >= paper.text.body_start


def test_text_with_no_preamble_is_unaffected():
    from resint.parse.document import paper_from_latex

    paper = paper_from_latex("Just prose, no document wrapper.", needs={"paper.text"})
    assert paper.text.body_start == 0
    assert paper.text.window(100) == paper.text.content[:100]


def test_a_wrapped_pdf_does_not_become_prose():
    """Some submissions typeset nothing and include a finished PDF. The
    filename and page options are not content, and they sit at the very top."""
    raw = (
        r"\documentclass{article}" "\n" r"\begin{document}" "\n"
        r"\includepdf[pages=-,fitpaper=true]{DDCM_paper_06-30-2026.pdf}" "\n"
        "Real prose follows the wrapper.\n" r"\end{document}" "\n"
    )
    assert "DDCM_paper" not in normalize(raw).text
    assert "fitpaper" not in normalize(raw).text
