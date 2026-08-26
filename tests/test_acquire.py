"""Input handling, and the real-world LaTeX it has to survive.

Everything here comes from pointing resint at an actual arXiv source bundle.
The first attempt crashed with a UnicodeDecodeError; the second read only the
wrapper file and reported nothing; the third parsed a table the authors had
commented out years earlier.
"""

import io
import tarfile
import zipfile
from pathlib import Path

import pytest

from resint.ir.span import Source
from resint.parse.acquire import (
    UnreadableInput,
    acquire,
    expand_inputs,
    locate_region,
)
from resint.parse.document import paper_from_path
from resint.parse.tables import extract_tables, uncomment

SRC = Source("paper.tex", "latex", path="paper.tex")

ROOT_TEX = r"""\documentclass{article}
\begin{document}
\title{A Paper}
\input{introduction}
\input{results}
\end{document}
"""
INTRO_TEX = "Intro line one.\nIntro line two with accuracy of 91.4.\n"
RESULTS_TEX = "Results line one.\nWe reach accuracy of 94.2 here.\n"


def make_tar(tmp_path, files, name="bundle.tar.gz"):
    path = tmp_path / name
    with tarfile.open(path, "w:gz") as tf:
        for member, text in files.items():
            data = text.encode("utf-8")
            info = tarfile.TarInfo(member)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return path


# --- unreadable input fails with a sentence -----------------------------


def test_a_pdf_is_refused_with_advice(tmp_path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.7\n\x00\x01binary")
    with pytest.raises(UnreadableInput, match="is a PDF, not LaTeX source"):
        acquire(pdf)


def test_a_pdf_wearing_an_archive_name_is_still_a_pdf(tmp_path):
    """arXiv serves the PDF from its e-print endpoint when a submission has no
    source, so a file named .tar.gz is routinely a PDF. Three of the first
    twenty-two papers swept were exactly this, and every one was reported as
    an unpacking failure -- which hides the single fact that matters."""
    disguised = tmp_path / "2608.24241v1.tar.gz"
    disguised.write_bytes(b"%PDF-1.7\n\x00\x01binary")
    with pytest.raises(UnreadableInput, match="is a PDF, not LaTeX source"):
        acquire(disguised)


def test_a_binary_file_is_refused(tmp_path):
    blob = tmp_path / "thing.tex"
    blob.write_bytes(b"\x1f\x8b\x00\x00binary garbage")
    with pytest.raises(UnreadableInput, match="binary"):
        acquire(blob)


def test_a_missing_file_is_refused(tmp_path):
    with pytest.raises(UnreadableInput, match="no such file"):
        acquire(tmp_path / "nope.tex")


def test_an_archive_without_latex_is_refused(tmp_path):
    path = make_tar(tmp_path, {"notes.txt": "no latex here"})
    with pytest.raises(UnreadableInput, match="no LaTeX source"):
        acquire(path)


def test_a_corrupt_archive_is_refused(tmp_path):
    path = tmp_path / "broken.tar.gz"
    path.write_bytes(b"\x1f\x8b\x08\x00 not really a gzip")
    with pytest.raises(UnreadableInput, match="could not be unpacked"):
        acquire(path)


# --- archives -----------------------------------------------------------


def test_a_tarball_is_unpacked(tmp_path):
    path = make_tar(tmp_path, {"ms.tex": ROOT_TEX, "introduction.tex": INTRO_TEX,
                               "results.tex": RESULTS_TEX})
    got = acquire(path)
    assert "Intro line one" in got.text
    assert "We reach accuracy of 94.2" in got.text


def test_a_zip_is_unpacked(tmp_path):
    path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("main.tex", ROOT_TEX)
        zf.writestr("introduction.tex", INTRO_TEX)
        zf.writestr("results.tex", RESULTS_TEX)
    assert "Intro line one" in acquire(path).text


def test_the_root_document_is_chosen_over_its_includes(tmp_path):
    """The largest file is regularly an appendix, not the paper."""
    big_appendix = "Appendix. " * 4000
    path = make_tar(tmp_path, {
        "ms.tex": ROOT_TEX,
        "introduction.tex": INTRO_TEX,
        "results.tex": RESULTS_TEX,
        "appendix.tex": big_appendix,
    })
    assert acquire(path).name == "ms.tex"


def test_a_bib_inside_the_archive_is_used(tmp_path):
    path = make_tar(tmp_path, {
        "ms.tex": ROOT_TEX,
        "introduction.tex": INTRO_TEX,
        "results.tex": RESULTS_TEX,
        "refs.bib": "@article{a2020, title={T}, year={2020}}",
    })
    got = acquire(path)
    assert got.bib_text and "a2020" in got.bib_text


def test_splicing_is_reported_as_unchecked(tmp_path):
    path = make_tar(tmp_path, {"ms.tex": ROOT_TEX, "introduction.tex": INTRO_TEX,
                               "results.tex": RESULTS_TEX})
    paper = paper_from_path(path)
    assert any("spliced in" in u for u in paper.unchecked)


# --- \input expansion keeps offsets truthful ----------------------------


def test_included_files_are_spliced_in_order():
    combined, _ = expand_inputs("ms.tex", {
        "ms.tex": ROOT_TEX, "introduction.tex": INTRO_TEX, "results.tex": RESULTS_TEX
    })
    assert combined.index("Intro line one") < combined.index("Results line one")


def test_regions_map_an_offset_back_to_its_real_file():
    files = {"ms.tex": ROOT_TEX, "introduction.tex": INTRO_TEX, "results.tex": RESULTS_TEX}
    combined, regions = expand_inputs("ms.tex", files)

    offset = combined.index("94.2")
    region = locate_region(regions, offset)
    assert region.name == "results.tex"
    local = region.local(offset)
    assert files["results.tex"][local : local + 4] == "94.2"


def test_line_numbers_are_local_to_the_included_file(tmp_path):
    """Line 2 of results.tex, not line 40 of a concatenation."""
    path = make_tar(tmp_path, {"ms.tex": ROOT_TEX, "introduction.tex": INTRO_TEX,
                               "results.tex": RESULTS_TEX})
    paper = paper_from_path(path, needs={"paper.numbers", "paper.tables"})
    reported = [n for n in paper.numbers if n.raw == "94.2"]
    assert reported, "the spliced content must be visible to the extractors"
    span = reported[0].span
    assert span.source.id == "results.tex"
    assert span.line == 2


def test_a_missing_include_is_skipped_not_fatal():
    combined, _ = expand_inputs("ms.tex", {"ms.tex": ROOT_TEX})
    assert "documentclass" in combined


def test_include_cycles_terminate():
    files = {"a.tex": r"A \input{b}", "b.tex": r"B \input{a}"}
    combined, _ = expand_inputs("a.tex", files)
    assert "A" in combined and "B" in combined


# --- comments in tables -------------------------------------------------


def test_uncomment_preserves_length_so_offsets_stay_valid():
    text = "keep % drop this\nkeep2"
    assert len(uncomment(text)) == len(text)
    assert "drop this" not in uncomment(text)


def test_an_escaped_percent_is_not_a_comment():
    assert "94.2\\%" in uncomment("94.2\\% accuracy")


def test_a_commented_out_table_is_not_parsed():
    """A real paper carries dead tables from earlier drafts."""
    src = (
        "%\\begin{tabular}{l|c}\n"
        "%\\hline\n"
        "%Method & Score \\\\\n"
        "%\\hline\n"
        "%Ours & 94.2 \\\\\n"
        "%\\end{tabular}\n"
    )
    assert extract_tables(src, SRC) == []


def test_a_live_table_beside_a_dead_one_still_parses():
    src = (
        "%\\begin{tabular}{l|c}\n%Dead & 1 \\\\\n%\\end{tabular}\n"
        "\\begin{tabular}{lc}\nMethod & Score \\\\\nOurs & 94.2 \\\\\n\\end{tabular}\n"
    )
    tables = extract_tables(src, SRC)
    assert len(tables) == 1
    assert tables[0].header == ["Method", "Score"]


def test_multicolumn_keeps_content_and_drops_alignment():
    """\\multicolumn{2}{c}{BLEU} became the header "cBLEU"."""
    src = (
        "\\begin{tabular}{lcc}\n"
        "Model & \\multicolumn{2}{c}{BLEU} \\\\\n"
        "Ours & 24.6 & 39.9 \\\\\n"
        "\\end{tabular}\n"
    )
    assert extract_tables(src, SRC)[0].rows[0][1].text == "BLEU"


def test_layout_commands_do_not_become_cell_text():
    """\\rule{0pt}{2.0ex} became the cell text "0pt2.0ex"."""
    src = (
        "\\begin{tabular}{lc}\n"
        "\\rule{0pt}{2.0ex}Model & Score \\\\\n"
        "Ours & 94.2 \\\\\n"
        "\\end{tabular}\n"
    )
    assert extract_tables(src, SRC)[0].rows[0][0].text == "Model"
