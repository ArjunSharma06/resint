"""Extraction of statistics and means, and the abstention discipline.

The negative cases matter more than the positive ones here. Extraction is
upstream of every finding, so a wrong pairing at this layer becomes a
confident wrong accusation at the reporting layer.
"""

import pytest

from resint.ir.span import Source
from resint.parse.extract import extract_means, extract_stats, sentences
from resint.parse.latex import normalize

SRC = Source("paper.tex", "latex", path="paper.tex")


def parse(latex: str):
    return normalize(latex), SRC


def stats_of(latex: str):
    return extract_stats(*parse(latex))


def means_of(latex: str):
    return extract_means(*parse(latex))


# --- statistics ---------------------------------------------------------


@pytest.mark.parametrize(
    "text, kind, stat, df1, df2, p, comp",
    [
        ("The effect held, $t(20) = 2.086$, $p = .05$.", "t", "2.086", 20, None, ".05", "="),
        ("Result: $t(30)=-2.10$, $p<.001$.", "t", "-2.10", 30, None, ".001", "<"),
        ("We found $F(1, 20) = 4.35$, $p = .05$.", "F", "4.35", 1, 20, ".05", "="),
        (r"Test gave $\chi^2(1) = 3.84$, $p = .05$.", "chi2", "3.84", 1, None, ".05", "="),
        ("Correlation $r(18) = .50$, $p = .03$.", "r", ".50", 18, None, ".03", "="),
        ("Standardised $z = 1.96$, $p = .05$.", "z", "1.96", None, None, ".05", "="),
    ],
)
def test_reads_each_supported_test_form(text, kind, stat, df1, df2, p, comp):
    found = stats_of(text)
    assert len(found) == 1
    s = found[0]
    assert (s.kind, s.statistic_raw, s.df1, s.df2) == (kind, stat, df1, df2)
    assert (s.p_raw, s.p_comparator) == (p, comp)


def test_chi_square_ignores_the_sample_size_in_its_parentheses():
    s = stats_of(r"$\chi^2(1, N = 100) = 3.84$, $p = .05$.")[0]
    assert s.df1 == 1 and s.df2 is None


def test_intervening_confidence_interval_is_tolerated():
    s = stats_of(r"$t(20) = 2.086$, 95\% CI [0.12, 0.31], $p = .05$.")[0]
    assert s.statistic_raw == "2.086" and s.p_raw == ".05"


def test_unescaped_percent_truncates_the_line_as_latex_requires():
    """A bare % opens a comment. Real sources write 95\\%; we must not guess."""
    assert stats_of("$t(20) = 2.086$, 95% CI, $p = .05$.") == []


def test_declared_one_tailed_is_recorded():
    s = stats_of("Using a one-tailed test, $t(20) = 1.725$, $p = .05$.")[0]
    assert s.tail == 1


def test_two_tailed_is_the_default():
    assert stats_of("$t(20) = 1.725$, $p = .05$.")[0].tail == 2


def test_one_tailed_does_not_leak_into_a_neighbouring_sentence():
    """Regression: a character window let one 'one-tailed' halve every nearby
    p-value, which turned a correct paper into a page of false findings."""
    found = stats_of(
        "The effect was reliable, $t(20) = 2.086$, $p = .05$. "
        "Using a one-tailed test, $t(20) = 1.725$, $p = .05$."
    )
    assert [s.tail for s in found] == [2, 1]


def test_does_not_span_a_sentence_boundary():
    """A statistic in one sentence must not pair with a p in the next."""
    assert stats_of("We report $t(20) = 2.086$. Elsewhere $p = .99$ was noted.") == []


def test_ignores_letters_inside_words():
    assert stats_of("The dataset = 5 items, split = .05 by weight.") == []


def test_skips_tests_with_no_degrees_of_freedom():
    assert stats_of("$t = 2.086$, $p = .05$.") == []
    assert stats_of("$F(1) = 4.35$, $p = .05$.") == []


def test_records_section_context_and_anchors():
    doc, src = parse(r"\section{Results}" "\nThe effect held, $t(20) = 2.086$, $p = .05$.")
    s = extract_stats(doc, src)[0]
    assert s.context == "Results"
    assert doc.raw[s.span.start : s.span.end].startswith("t(20)")
    assert doc.raw[s.p_span.start : s.p_span.end] == ".05"
    assert s.span.line == 2


def test_multiple_statistics_in_one_paragraph():
    found = stats_of(
        "First $t(20) = 2.086$, $p = .05$. Then $F(1, 30) = 9.1$, $p = .004$."
    )
    assert [s.kind for s in found] == ["t", "F"]


# --- means --------------------------------------------------------------


def test_pairs_a_mean_with_the_single_sample_size_in_its_sentence():
    r = means_of("Participants rated it favourably (M = 3.47, SD = 0.82, N = 20).")
    assert len(r.means) == 1
    m = r.means[0]
    assert m.raw == "3.47" and m.n == 20 and m.items == 1
    assert m.items_inferred is True


def test_explicit_item_count_is_used_and_marked_certain():
    r = means_of("On the 3-item scale the mean was 3.47 with N = 20 participants.")
    assert r.means[0].items == 3
    assert r.means[0].items_inferred is False


def test_abstains_when_no_sample_size_is_in_the_sentence():
    r = means_of("The mean was 3.47 overall. Separately, N = 20 took part.")
    assert r.means == []
    assert any("no sample size" in u for u in r.unchecked)


def test_abstains_when_the_sentence_is_ambiguous():
    """Two candidate sample sizes means we cannot know which one applies."""
    r = means_of("Across groups (N = 20 and N = 25) the mean was 3.47.")
    assert r.means == []
    assert any("2 sample sizes" in u for u in r.unchecked)


def test_unchecked_reasons_name_a_line():
    r = means_of("Intro line.\n\nThe mean was 3.47 with no count given.")
    assert r.unchecked and "line" in r.unchecked[0]


def test_several_means_share_one_sample_size():
    r = means_of("With N = 20, we saw M = 3.45 before and M = 3.50 after.")
    assert [m.raw for m in r.means] == ["3.45", "3.50"]
    assert {m.n for m in r.means} == {20}


def test_mean_anchors_point_at_the_source():
    doc, src = parse("Participants agreed (M = 3.47, SD = 0.82, N = 20).")
    m = extract_means(doc, src).means[0]
    assert doc.raw[m.span.start : m.span.end] == "3.47"
    assert doc.raw[m.n_span.start : m.n_span.end] == "20"


# --- sentence segmentation ----------------------------------------------


def test_abbreviations_do_not_split_sentences():
    text = "As shown by Smith et al. the mean was 3.47 with N = 20 here."
    assert len(sentences(text)) == 1


def test_genuine_boundaries_do_split():
    assert len(sentences("First sentence here. Second sentence here.")) == 2


def test_a_decimal_comma_is_read_as_a_decimal():
    """Most of continental Europe and Latin America writes p < 0,001 and
    F(1, 791) = 18,65. Reading only the point turned those into 0 and 18, and
    the recomputation then disagreed with the (also misread) reported value --
    a confident finding built on two wrong numbers. The first Brazilian paper
    in the corpus produced exactly that.
    """
    doc = normalize("Houve diferenca, com F(1, 791) = 18,65; p < 0,001.")
    stats = extract_stats(doc, SRC)
    assert len(stats) == 1
    assert stats[0].statistic == 18.65
    assert stats[0].p_reported == 0.001
    assert stats[0].df1 == 1 and stats[0].df2 == 791


def test_the_comma_between_degrees_of_freedom_is_still_a_separator():
    """F(1, 791) is two numbers, not one. The fix must not reach into it."""
    doc = normalize("Result F(2, 20) = 4.11, p = .03.")
    stat = extract_stats(doc, SRC)[0]
    assert (stat.df1, stat.df2) == (2.0, 20.0)
    assert stat.statistic == 4.11


def test_a_decimal_point_still_works():
    doc = normalize("Reliable, t(20) = 2.086, p = .03.")
    stat = extract_stats(doc, SRC)[0]
    assert stat.statistic == 2.086
    assert stat.p_reported == 0.03


def test_precision_survives_a_decimal_comma():
    """GRIM needs the stated granularity, so "3,470" must stay three decimals."""
    doc = normalize("The mean was 3,470 for N = 20 participants.")
    mean = extract_means(doc, SRC).means[0]
    assert mean.decimals == 3
    assert float(mean.value) == 3.470


def test_a_rule_never_converts_a_reported_value_itself():
    """stats/pvalue.py kept its own Decimal(p_raw.strip()) and so still read a
    comma as a syntax error after the IR had learned to handle it -- a crash on
    the first Brazilian paper swept. One conversion point, and this asserts
    nothing has grown a second one."""
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[1] / "src" / "resint"
    offenders = []
    for path in root.rglob("*.py"):
        if path.name == "paper.py":
            continue  # where the conversion legitimately lives
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"(?:Decimal|float)\(\s*\w*\.?(?:p_raw|statistic_raw)", line):
                offenders.append(f"{path.name}:{n}: {line.strip()}")
    assert not offenders, "convert via decimal_text/p_exact instead:\n" + "\n".join(offenders)
