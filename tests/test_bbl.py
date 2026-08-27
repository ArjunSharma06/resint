"""Compiled bibliographies.

arXiv submissions usually ship BibTeX's output rather than its input, and
often inline it into the source rather than shipping a file at all. Every
case here comes from the Transformer bundle, where forty references were
invisible because none of this existed.
"""

import io
import tarfile

import pytest

from resint.ir.span import Source
from resint.parse.bbl import looks_like_bbl, parse
from resint.parse.document import paper_from_path
from resint.resolve import Record, Resolution, Status, StaticResolver
from resint.rules import load_all
from resint.rules.registry import Context

SRC = Source("ms.tex", "bib", path="ms.tex")
REG = load_all()

SAMPLE = r"""
\begin{thebibliography}{10}

\bibitem{chollet2016}
Francois Chollet.
\newblock Xception: Deep learning with depthwise separable convolutions.
\newblock {\em arXiv preprint arXiv:1610.02357}, 2016.

\bibitem{hochreiter2001}
Sepp Hochreiter, Yoshua Bengio, Paolo Frasconi, and J{\"u}rgen Schmidhuber.
\newblock Gradient flow in recurrent nets: the difficulty of learning long-term
  dependencies, 2001.

\bibitem{withdoi2020}
A.~Author and B.~Other.
\newblock A paper with an identifier.
\newblock {\em Journal of Things}, 2020.
\newblock doi:10.1000/xyz123.

\bibitem{noblocks2019}
Everything on one line with no newblock separator at all, 2019.

\end{thebibliography}
"""


@pytest.fixture(scope="module")
def parsed():
    return parse(SAMPLE, SRC)


def test_detection():
    assert looks_like_bbl(SAMPLE)
    assert not looks_like_bbl("@article{a2020, title={T}}")


def test_every_bibitem_becomes_an_entry(parsed):
    assert [e.key for e in parsed.entries] == [
        "chollet2016",
        "hochreiter2001",
        "withdoi2020",
        "noblocks2019",
    ]


def test_a_single_author_without_initials_is_not_mistaken_for_a_title(parsed):
    """"Francois Chollet" failed a looks-like-a-name-list test and became
    the title, so the reference was searched for under the author's name."""
    entry = parsed.by_key()["chollet2016"]
    assert entry.fields["author"] == "Francois Chollet"
    assert entry.title == (
        "Xception: Deep learning with depthwise separable convolutions"
    )


def test_a_trailing_year_is_stripped_from_the_title(parsed):
    """Some styles close the title block with the year rather than the venue."""
    entry = parsed.by_key()["hochreiter2001"]
    assert entry.title.endswith("long-term dependencies")
    assert "2001" not in entry.title
    assert entry.year == "2001"


def test_accents_are_decoded(parsed):
    assert "Jürgen" in parsed.by_key()["hochreiter2001"].fields["author"]


def test_a_doi_is_picked_up_when_the_style_prints_one(parsed):
    assert parsed.by_key()["withdoi2020"].doi == "10.1000/xyz123"


def test_an_entry_without_newblocks_yields_only_its_key(parsed):
    """Guessing which fragment is the title would search indices on noise."""
    entry = parsed.by_key()["noblocks2019"]
    assert entry.title == ""
    assert entry.year == "2019"


def test_untitled_entries_are_reported_as_a_note(parsed):
    assert any("no recoverable title" in n for n in parsed.notes)
    assert "1 entry has" in parsed.notes[0]


def test_entries_are_marked_as_coming_from_a_compiled_bibliography(parsed):
    assert all(e.from_bbl for e in parsed.entries)


def test_spans_point_into_the_source(parsed):
    entry = parsed.by_key()["chollet2016"]
    assert SAMPLE[entry.span.start : entry.span.end].startswith(r"\bibitem")
    title_span = entry.field_spans["title"]
    assert "Xception" in SAMPLE[title_span.start : title_span.end]


def test_empty_input_is_not_a_bibliography():
    assert parse("no bibitems here", SRC).entries == []


# --- severity is softened for reconstructed titles ----------------------


def paper_with(entries, resolutions):
    from resint.ir.paper import Paper

    p = Paper(source_id="ms.tex")
    p.bib = list(entries)
    p.resolutions = dict(resolutions)
    return p


def test_unresolved_from_a_bbl_is_medium_not_high(parsed):
    """A title recovered from rendered text may simply be wrong."""
    entry = parsed.by_key()["chollet2016"]
    findings = REG.get("bib/unresolved").run(
        Context(
            paper=paper_with(
                [entry],
                {"chollet2016": Resolution(Status.NOT_FOUND, queried=("crossref",))},
            )
        )
    )
    assert len(findings) == 1
    assert findings[0].severity.value == "med"
    assert "may not be exact" in findings[0].message


def test_a_failed_doi_from_a_bbl_is_still_high(parsed):
    """A printed DOI is a specific claim regardless of where it was found."""
    entry = parsed.by_key()["withdoi2020"]
    findings = REG.get("bib/unresolved").run(
        Context(
            paper=paper_with(
                [entry],
                {"withdoi2020": Resolution(Status.NOT_FOUND, queried=("crossref",))},
            )
        )
    )
    assert findings[0].severity.value == "high"


# --- end to end ---------------------------------------------------------


INLINE_TEX = (
    "\\documentclass{article}\n"
    "\\begin{document}\n"
    "Prior work~\\cite{chollet2016} and more~\\cite{withdoi2020}.\n"
    + SAMPLE
    + "\\end{document}\n"
)


def make_tar(tmp_path, files, name="bundle.tar.gz"):
    path = tmp_path / name
    with tarfile.open(path, "w:gz") as tf:
        for member, text in files.items():
            data = text.encode("utf-8")
            info = tarfile.TarInfo(member)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return path


def test_a_bibliography_inlined_in_the_source_is_found(tmp_path):
    """The Transformer bundle ships no .bib and no .bbl -- it inlines both."""
    path = make_tar(tmp_path, {"ms.tex": INLINE_TEX})
    paper = paper_from_path(path, needs={"paper.bib", "paper.citations"})
    assert len(paper.bib) == 4
    assert {c.key for c in paper.citations} == {"chollet2016", "withdoi2020"}


def test_a_separate_bbl_file_is_used_when_no_bib_exists(tmp_path):
    path = make_tar(tmp_path, {
        "ms.tex": "\\documentclass{article}\n\\begin{document}\n"
                  "Text~\\cite{chollet2016}.\n\\end{document}\n",
        "ms.bbl": SAMPLE,
    })
    paper = paper_from_path(path, needs={"paper.bib"})
    assert {e.key for e in paper.bib} >= {"chollet2016", "hochreiter2001"}


def test_a_real_bib_wins_over_a_compiled_one(tmp_path):
    """A .bib holds the fields as written; a .bbl holds what was rendered."""
    path = make_tar(tmp_path, {
        "ms.tex": "\\documentclass{article}\n\\begin{document}\nx\n\\end{document}\n",
        "ms.bbl": SAMPLE,
        "refs.bib": "@article{fromthebib, title={From The Bib}, year={2020}}",
    })
    paper = paper_from_path(path, needs={"paper.bib"})
    assert [e.key for e in paper.bib] == ["fromthebib"]
    assert not paper.bib[0].from_bbl


def test_orphans_works_on_a_compiled_bibliography(tmp_path):
    """The key is the one thing a .bbl states outright, so this rule is exact."""
    path = make_tar(tmp_path, {"ms.tex": INLINE_TEX})
    paper = paper_from_path(path, needs={"paper.bib", "paper.citations"})
    findings = REG.get("bib/orphans").run(Context(paper=paper))
    uncited = [f for f in findings if "never cited" in f.message]
    assert len(uncited) == 1
    assert "hochreiter2001" in uncited[0].message


def test_an_uncited_bibitem_is_told_it_will_appear(tmp_path):
    """A thebibliography environment is a list: LaTeX typesets every \bibitem
    in it, cited or not. So an uncited entry *does* appear, in a reference list
    nothing points at.

    The rule used to tell every paper the opposite -- that the entry would be
    dropped -- which is only true of BibTeX. On 143 findings across 204 real
    papers it was stating the reverse of what happens, and a finding that
    misdescribes its own evidence is worse than no finding at all.
    """
    path = make_tar(tmp_path, {"ms.tex": INLINE_TEX})
    paper = paper_from_path(path, needs={"paper.bib", "paper.citations"})
    uncited = [
        f
        for f in REG.get("bib/orphans").run(Context(paper=paper))
        if "never cited" in f.message
    ]
    assert len(uncited) == 1
    assert "will appear in the reference list" in uncited[0].message
    assert "will not appear" not in uncited[0].message


def test_an_uncited_bib_entry_is_told_it_will_be_dropped(tmp_path):
    """The opposite case: BibTeX emits only what was cited."""
    path = make_tar(tmp_path, {
        "ms.tex": (
            r"\documentclass{article}" "\n"
            r"\begin{document}" "\n"
            r"Cited \cite{used2020}." "\n"
            r"\end{document}" "\n"
        ),
        "refs.bib": (
            "@article{used2020, title={Used}, year={2020}}\n"
            "@article{spare2019, title={Spare}, year={2019}}\n"
        ),
    })
    paper = paper_from_path(path, needs={"paper.bib", "paper.citations"})
    uncited = [
        f
        for f in REG.get("bib/orphans").run(Context(paper=paper))
        if "never cited" in f.message
    ]
    assert len(uncited) == 1
    assert "will not appear in the reference list" in uncited[0].message
    assert "spare2019" in uncited[0].message
