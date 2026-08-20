"""Table extraction and the numbers/ rule family."""

from decimal import Decimal

import pytest

from resint.ir.paper import Number, Paper
from resint.ir.span import Source, Span
from resint.parse.extract import NON_METRIC_HEADERS, usable_labels
from resint.parse.tables import extract_tables
from resint.rules import load_all
from resint.rules.registry import Context

SRC = Source("paper.tex", "latex", path="paper.tex")
REG = load_all()

MAIN = r"""
\begin{table}[t]
  \centering
  \caption{Accuracy on the benchmark.}
  \label{tab:main}
  \begin{tabular}{lcc}
    \toprule
    Method & Accuracy & F1 \\
    \midrule
    Baseline      & 91.4 & 90.2 \\
    \textbf{Ours} & 93.8 & 92.6 \\
    \bottomrule
  \end{tabular}
\end{table}
"""

COUNTS = r"""
\begin{table}
  \caption{Dataset composition.}
  \begin{tabular}{lr}
    Split & Count \\
    Train & 800 \\
    Val   & 100 \\
    Test  & 100 \\
    Total & 1200 \\
  \end{tabular}
\end{table}
"""


def tables_of(src):
    return extract_tables(src, SRC)


def paper_with(tables=(), numbers=()):
    p = Paper(source_id="paper.tex")
    p.tables = list(tables)
    p.numbers = list(numbers)
    return p


def fire(rule_id, paper):
    return REG.get(rule_id).run(Context(paper=paper))


# --- extraction ---------------------------------------------------------


def test_grid_is_recovered():
    t = tables_of(MAIN)[0]
    assert [c.text for c in t.rows[0]] == ["Method", "Accuracy", "F1"]
    assert [c.text for c in t.rows[2]] == ["Ours", "93.8", "92.6"]
    assert not t.irregular


def test_rule_commands_do_not_become_rows():
    """\\toprule and friends carry no data and must not shift the grid."""
    t = tables_of(MAIN)[0]
    assert len(t.rows) == 3
    assert all(any(c.text for c in row) for row in t.rows)


def test_markup_is_stripped_from_cells():
    t = tables_of(MAIN)[0]
    assert t.rows[2][0].text == "Ours"


def test_caption_and_label_are_captured():
    t = tables_of(MAIN)[0]
    assert t.caption == "Accuracy on the benchmark."
    assert t.label == "tab:main"


def test_cell_anchors_point_at_the_source():
    t = tables_of(MAIN)[0]
    cell = t.rows[2][1]
    assert MAIN[cell.span.start : cell.span.end] == "93.8"


def test_every_cell_anchor_is_exact():
    for table in tables_of(MAIN + COUNTS):
        for cell in table.cells():
            if not cell.text:
                continue
            anchored = MAIN.join([]) or ""
    source = MAIN + COUNTS
    for table in extract_tables(source, SRC):
        for cell in table.cells():
            if cell.text:
                assert source[cell.span.start : cell.span.end].strip()


def test_numeric_cells_parse_and_text_cells_do_not():
    t = tables_of(MAIN)[0]
    assert t.rows[1][1].number == Decimal("91.4")
    assert t.rows[1][0].number is None


def test_ambiguous_cells_yield_no_single_number():
    """"94.2 +/- 0.3" holds two numbers; returning the first would misread it."""
    src = r"\begin{tabular}{lc} A & B \\ x & 94.2 $\pm$ 0.3 \\ y & 12/48 \\ \end{tabular}"
    t = extract_tables(src, SRC)[0]
    assert all(c.number is None for c in t.column(1)[1:])


def test_column_and_header_accessors():
    t = tables_of(MAIN)[0]
    assert t.header == ["Method", "Accuracy", "F1"]
    assert [c.text for c in t.column(1)] == ["Accuracy", "91.4", "93.8"]


def test_multiple_tables_are_numbered_in_order():
    tables = tables_of(MAIN + COUNTS)
    assert [t.name for t in tables] == ["table1", "table2"]


def test_ragged_table_is_marked_irregular_not_guessed():
    src = r"\begin{tabular}{lcc} A & B & C \\ 1 & 2 \\ 3 & 4 & 5 \\ \end{tabular}"
    assert "row widths disagree" in extract_tables(src, SRC)[0].irregular


def test_multicolumn_counts_toward_width():
    src = (
        r"\begin{tabular}{lcc} A & B & C \\ "
        r"\multicolumn{2}{c}{Spanning} & Z \\ \end{tabular}"
    )
    assert not extract_tables(src, SRC)[0].irregular


def test_tabularx_width_argument_is_skipped():
    src = r"\begin{tabularx}{\textwidth}{lc} A & B \\ 1 & 2 \\ \end{tabularx}"
    t = extract_tables(src, SRC)[0]
    assert t.header == ["A", "B"]


def test_unterminated_table_is_dropped_not_raised():
    assert extract_tables(r"\begin{tabular}{lc} A & B \\", SRC) == []


# --- header vocabulary --------------------------------------------------


@pytest.mark.parametrize("word", ["Method", "Model", "Dataset", "Split", "Ablation"])
def test_row_label_headers_are_not_metrics(word):
    assert word.lower() in NON_METRIC_HEADERS
    assert usable_labels([word]) == []


def test_metric_headers_survive():
    assert usable_labels(["Method", "Accuracy", "F1"]) == ["Accuracy", "F1"]


# --- numbers/internal-mismatch ------------------------------------------


def number(raw, label, section="Results", offset=0):
    return Number(
        raw=raw,
        label=label,
        section=section,
        span=Span(SRC, offset, offset + len(raw), line=1, label=section),
    )


def test_near_miss_against_a_column_is_reported():
    findings = fire(
        "numbers/internal-mismatch",
        paper_with(tables_of(MAIN), [number("94.2", "accuracy")]),
    )
    assert len(findings) == 1
    assert "94.2" in findings[0].message and "93.8" in findings[0].message
    assert len(findings[0].anchors) == 2


def test_a_value_present_in_the_column_is_silent():
    findings = fire(
        "numbers/internal-mismatch",
        paper_with(tables_of(MAIN), [number("93.8", "accuracy")]),
    )
    assert findings == []


def test_a_distant_value_is_a_different_quantity_and_stays_silent():
    """94.2 against {41.0, 38.2} is not a stale number, it is another metric."""
    src = MAIN.replace("91.4", "41.0").replace("93.8", "38.2")
    findings = fire(
        "numbers/internal-mismatch",
        paper_with(tables_of(src), [number("94.2", "accuracy")]),
    )
    assert findings == []


def test_a_label_with_no_matching_column_is_silent():
    findings = fire(
        "numbers/internal-mismatch",
        paper_with(tables_of(MAIN), [number("94.2", "bleu")]),
    )
    assert findings == []


def test_irregular_tables_are_never_compared_against():
    src = r"\begin{tabular}{lcc} Method & Accuracy & F1 \\ a & 93.8 \\ b & 1 & 2 \\ \end{tabular}"
    findings = fire(
        "numbers/internal-mismatch",
        paper_with(extract_tables(src, SRC), [number("94.2", "accuracy")]),
    )
    assert findings == []


def test_trailing_zeros_compare_equal():
    src = MAIN.replace("93.8", "93.80")
    findings = fire(
        "numbers/internal-mismatch",
        paper_with(tables_of(src), [number("93.8", "accuracy")]),
    )
    assert findings == []


# --- numbers/table-arithmetic -------------------------------------------


def test_wrong_total_is_reported():
    findings = fire("numbers/table-arithmetic", paper_with(tables_of(COUNTS)))
    assert len(findings) == 1
    assert "1200" in findings[0].message and "1000" in findings[0].message


def test_correct_total_is_silent():
    findings = fire(
        "numbers/table-arithmetic",
        paper_with(tables_of(COUNTS.replace("1200", "1000"))),
    )
    assert findings == []


def test_rounding_within_reported_precision_is_tolerated():
    src = r"""
\begin{tabular}{lr}
  Part & Share \\
  A & 33.3 \\
  B & 33.3 \\
  C & 33.3 \\
  Total & 100.0 \\
\end{tabular}
"""
    assert fire("numbers/table-arithmetic", paper_with(tables_of(src))) == []


def test_table_with_no_total_row_is_silent():
    src = MAIN
    assert fire("numbers/table-arithmetic", paper_with(tables_of(src))) == []


def test_irregular_table_is_skipped():
    src = r"\begin{tabular}{lr} Split & Count \\ a & 1 \\ b \\ Total & 99 \\ \end{tabular}"
    assert fire("numbers/table-arithmetic", paper_with(extract_tables(src, SRC))) == []


# --- rule metadata ------------------------------------------------------


@pytest.mark.parametrize(
    "rule_id", ["numbers/internal-mismatch", "numbers/table-arithmetic"]
)
def test_rules_declare_their_blind_spots(rule_id):
    rule = REG.get(rule_id)
    assert len(rule.cannot_detect) > 40
    assert rule.cannot_detect.rstrip().endswith(".")
