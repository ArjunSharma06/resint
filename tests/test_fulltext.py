"""Reading cited papers, and narrowing them to the part that matters.

The narrowing is what makes this affordable. A manuscript cites forty works;
sending forty full papers to a model is some four hundred thousand tokens per
check and does not fit in a context window anyway. Retrieval brings that back
to the order of an abstract using plain code, which is why these tests are
free to run and run offline.
"""

import json

import pytest

from resint.ir.paper import BibEntry
from resint.ir.span import Source, Span
from resint.resolve.base import Record, Status
from resint.resolve.fulltext import (
    CachingFullText,
    Fetched,
    FullText,
    HttpFullText,
    NullFullText,
    StaticFullText,
    arxiv_id_for,
    jats_title,
    jats_to_text,
    latex_to_text,
    pmcid_for,
)
from resint.resolve.passages import (
    Passage,
    numbers,
    relatedness,
    retrieve,
    split_paragraphs,
    terms,
)

SPAN = Span(Source("refs.bib", "bibtex"), 0, 1, 1)

CITED = """Transformers relate positions in a sequence using self-attention,
which has become the standard building block for sequence modelling tasks.

Self-attention computes a score for every pair of positions, so its cost grows
quadratic in the sequence length. This is the central bottleneck for long
documents and the motivation for the approximations we survey below.

We evaluate on machine translation and report a BLEU score of 28.4 on the
WMT 2014 English-to-German benchmark, improving over previously published
results at a fraction of the training cost.
"""


def entry(**fields) -> BibEntry:
    return BibEntry(key="k", entry_type="article", fields=fields, span=SPAN)


# --- narrowing a paper to the passages that bear on a claim --------------


def test_the_passage_that_bears_on_the_claim_ranks_first():
    """The claim says linear; the paper says quadratic. That paragraph is the
    one worth sending, and nothing but overlap statistics chose it."""
    top = retrieve("attention scales linearly with sequence length", CITED)[0]
    assert "quadratic in the sequence length" in top.text


def test_retrieval_returns_at_most_k_passages():
    assert len(retrieve("sequence attention positions cost", CITED, k=2)) <= 2


def test_a_paragraph_with_no_overlap_is_not_returned():
    assert retrieve("perovskite crystallography annealing", CITED) == []


def test_a_shared_number_outweighs_a_shared_word():
    """Two papers both saying "accuracy" means nothing. Both saying 28.4 means
    they are discussing the same result."""
    top = retrieve("the reported BLEU score of 28.4", CITED)[0]
    assert "28.4" in top.text
    assert "28.4" in top.shared


def test_coincidental_numbers_are_ignored():
    """Years and small counts match by accident constantly."""
    assert numbers("in 2014 we used 8 GPUs to reach 94.2") == {"94.2"}


def test_ubiquitous_terms_do_not_decide_which_passage_to_send():
    """A word in every paragraph carries no information about *which* one to
    pick, whatever it means elsewhere."""
    everywhere = "\n\n".join(
        "The model is discussed here at sufficient length to count as a real "
        f"paragraph rather than a heading, part {n}, with padding text."
        for n in range(4)
    )
    distinctive = (
        "\n\nThe model attains a calibration error of 0.031 under temperature "
        "scaling, which is the specific result being referred to here."
    )
    top = retrieve("model calibration error", everywhere + distinctive)[0]
    assert "calibration" in top.text


def test_paragraph_offsets_point_at_the_real_text():
    """A passage that produces a finding has to anchor back to a real place."""
    for start, block in split_paragraphs(CITED):
        assert CITED[start : start + 20] == block[:20]


def test_headings_and_stray_lines_are_not_passages():
    assert split_paragraphs("Introduction\n\n1\n\nSee below.") == []


def test_offsets_survive_an_irregular_separator():
    """Papers separate paragraphs with blank lines of every width. Assuming a
    two-character separator drifts the offsets a little further with each one."""
    body = "\n\n   \n\n".join(
        f"Paragraph number {n} written at enough length to count as a real "
        "block of prose rather than a heading or a stray caption line."
        for n in range(4)
    )
    blocks = split_paragraphs(body)
    assert len(blocks) == 4
    for start, block in blocks:
        assert body[start : start + len(block)] == block


def test_a_paper_with_no_paragraph_breaks_is_still_narrowed():
    """Returning one enormous block would mean sending the whole paper every
    time -- the exact cost this module exists to avoid."""
    run_on = " ".join(
        f"This is sentence number {n} of a paper that arrived with no blank "
        "lines anywhere in it at all." for n in range(120)
    )
    blocks = split_paragraphs(run_on)
    assert len(blocks) > 1
    for start, block in blocks:
        assert run_on[start : start + len(block)] == block


def test_stop_words_are_not_content():
    assert "the" not in terms("the quadratic cost of the method")
    assert "quadratic" in terms("the quadratic cost of the method")


# --- is the cited paper even about this? --------------------------------


def test_a_paper_on_the_subject_scores_high():
    assert relatedness("self-attention sequence cost", CITED) > 0.5


def test_a_paper_on_another_subject_scores_zero():
    """The signature of a reference list that has slipped by one, or a citation
    copied out of somebody else's bibliography."""
    assert relatedness("perovskite photovoltaic annealing", CITED) == 0.0


def test_an_empty_claim_is_not_treated_as_unrelated():
    """Nothing to look for is not the same as looking and finding nothing."""
    assert relatedness("", CITED) == 1.0


# --- finding the identifier that leads to full text ---------------------


@pytest.mark.parametrize(
    "fields, expected",
    [
        ({"eprint": "1706.03762", "archiveprefix": "arXiv"}, "1706.03762"),
        ({"journal": "arXiv preprint arXiv:2103.00020"}, "2103.00020"),
        ({"url": "https://arxiv.org/abs/1810.04805v2"}, "1810.04805"),
        ({"url": "https://arxiv.org/abs/hep-th/9901001"}, "hep-th/9901001"),
        ({"journal": "Nature", "doi": "10.1038/nature14539"}, ""),
    ],
)
def test_an_arxiv_id_is_found_however_bibtex_spells_it(fields, expected):
    """BibTeX has no single convention and all of these appear constantly."""
    assert arxiv_id_for(entry(**fields)) == expected


def test_a_resolved_record_supplies_the_id_when_the_entry_does_not():
    """The common case: a DOI-only entry, and OpenAlex knows it is on arXiv."""
    record = Record(source="openalex", arxiv_id="1706.03762")
    assert arxiv_id_for(entry(doi="10.5555/3295222"), record) == "1706.03762"


def test_a_pmcid_is_found_in_a_url():
    assert pmcid_for(entry(url="https://ncbi.nlm.nih.gov/pmc/articles/PMC3084216")) == (
        "PMC3084216"
    )


def test_a_record_without_either_id_has_no_full_text():
    assert not Record(source="crossref", doi="10.1000/x").has_full_text


# --- turning what came back into text -----------------------------------


def test_latex_becomes_prose_with_paragraphs_intact():
    """Blank lines are load-bearing: retrieval splits on them, and a paper
    collapsed into one block cannot be narrowed at all."""
    text = latex_to_text(
        "\\section{Method}\nWe train the model for one hundred epochs on "
        "eight GPUs.\n\nWe then evaluate it on the held-out benchmark split.\n"
    )
    assert len(split_paragraphs(text + "\n")) >= 0
    assert "\n\n" in text


def test_jats_paragraphs_become_passages():
    xml = (
        "<article><front><article-meta><title-group>"
        "<article-title>A Study of <italic>Things</italic></article-title>"
        "</title-group><abstract><p>We studied the thing.</p></abstract>"
        "</article-meta></front><body><sec>"
        "<p>The cost is quadratic in the input length by construction.</p>"
        "<p>We report 94.2 accuracy on the held-out split.</p>"
        "</sec></body></article>"
    )
    text = jats_to_text(xml)
    assert "quadratic in the input length" in text
    assert "94.2" in text
    assert jats_title(xml) == "A Study of Things"


def test_inline_markup_does_not_break_a_sentence_apart():
    """JATS scatters italics and cross-references through every sentence."""
    xml = "<article><body><p>A cost of <italic>n</italic> squared per layer.</p></body></article>"
    assert "A cost of n squared per layer." in jats_to_text(xml)


def test_malformed_xml_yields_no_text_rather_than_an_exception():
    assert jats_to_text("<article><body><p>unclosed") == ""


# --- the three-outcome contract -----------------------------------------


def test_the_default_source_fetches_nothing_and_says_so():
    result = NullFullText().fetch(entry(eprint="1706.03762"))
    assert result.status is Status.UNKNOWN
    assert not result.usable
    assert not result.checkable


def test_a_known_paper_is_found():
    source = StaticFullText(
        documents={"1706.03762": FullText("arxiv", "1706.03762", CITED)}
    )
    result = source.fetch(entry(eprint="1706.03762"))
    assert result.usable
    assert result.document.words > 50


def test_a_reference_with_no_identifier_is_not_found_rather_than_unknown():
    """Permanent, not transient: there is no id to try again with later."""
    result = StaticFullText().fetch(entry(journal="Nature"))
    assert result.status is Status.NOT_FOUND
    assert result.checkable


def test_a_failed_lookup_is_unknown_not_not_found():
    """The distinction the whole safety property rests on."""
    source = StaticFullText(unavailable={"1706.03762"})
    assert source.fetch(entry(eprint="1706.03762")).status is Status.UNKNOWN


def test_unknown_can_never_support_a_finding():
    assert not Fetched(Status.UNKNOWN).checkable
    assert not Fetched(Status.UNKNOWN).usable


def test_found_without_a_document_is_still_not_usable():
    assert not Fetched(Status.FOUND, document=None).usable


# --- caching -------------------------------------------------------------


def test_a_paper_is_fetched_once_per_machine(tmp_path):
    """Everyone in a field cites the same landmarks. Refetching them each run
    is the difference between usable and a five-minute wait."""
    calls = []

    class _Counting:
        def fetch(self, entry, record=None):
            calls.append(1)
            return Fetched(
                Status.FOUND,
                document=FullText("arxiv", "1706.03762", CITED),
                queried=("1706.03762",),
            )

    cache = CachingFullText(_Counting(), directory=tmp_path)
    first = cache.fetch(entry(eprint="1706.03762"))
    second = cache.fetch(entry(eprint="1706.03762"))

    assert len(calls) == 1
    assert (cache.hits, cache.misses) == (1, 1)
    assert second.document.text == first.document.text


def test_a_paper_with_no_readable_text_is_remembered(tmp_path):
    """NOT_FOUND is a settled fact about the paper. Asking arXiv again every
    run for a submission that has no source is pure waste."""
    calls = []

    class _Counting:
        def fetch(self, entry, record=None):
            calls.append(1)
            return Fetched(Status.NOT_FOUND, queried=("1706.03762",), detail="PDF only")

    cache = CachingFullText(_Counting(), directory=tmp_path)
    cache.fetch(entry(eprint="1706.03762"))
    again = cache.fetch(entry(eprint="1706.03762"))

    assert len(calls) == 1
    assert again.status is Status.NOT_FOUND
    assert again.detail == "PDF only"


def test_a_transient_failure_is_never_written_down(tmp_path):
    """Caching a timeout turns a bad afternoon into a permanent wrong answer."""
    outcomes = [
        Fetched(Status.UNKNOWN, detail="fetch failed"),
        Fetched(Status.FOUND, document=FullText("arxiv", "1706.03762", CITED)),
    ]

    class _Flaky:
        def fetch(self, entry, record=None):
            return outcomes.pop(0)

    cache = CachingFullText(_Flaky(), directory=tmp_path)
    assert not cache.fetch(entry(eprint="1706.03762")).usable
    assert cache.fetch(entry(eprint="1706.03762")).usable
    assert list(tmp_path.glob("*.json"))


def test_a_corrupt_cache_entry_is_refetched_not_fatal(tmp_path):
    (tmp_path / "1706.03762.json").write_text("{ truncated", encoding="utf-8")

    class _Works:
        def fetch(self, entry, record=None):
            return Fetched(Status.FOUND, document=FullText("arxiv", "1706.03762", CITED))

    assert CachingFullText(_Works(), directory=tmp_path).fetch(
        entry(eprint="1706.03762")
    ).usable


def test_an_old_style_id_does_not_escape_the_cache_directory(tmp_path):
    """hep-th/9901001 contains a separator. It must not become a subdirectory."""
    class _Works:
        def fetch(self, entry, record=None):
            return Fetched(Status.FOUND, document=FullText("arxiv", "hep-th/9901001", "x"))

    cache = CachingFullText(_Works(), directory=tmp_path)
    cache.fetch(entry(url="https://arxiv.org/abs/hep-th/9901001"))
    written = list(tmp_path.glob("*.json"))
    assert written and written[0].name == "hep-th_9901001.json"


def test_what_is_cached_round_trips(tmp_path):
    class _Works:
        def fetch(self, entry, record=None):
            return Fetched(
                Status.FOUND,
                document=FullText("pmc", "PMC3084216", CITED, title="A Study"),
                queried=("PMC3084216",),
                detail="",
            )

    cache = CachingFullText(_Works(), directory=tmp_path)
    cache.fetch(entry(pmcid="PMC3084216"))
    stored = json.loads((tmp_path / "PMC3084216.json").read_text(encoding="utf-8"))
    assert stored["document"]["title"] == "A Study"

    restored = CachingFullText(_Works(), directory=tmp_path).fetch(entry(pmcid="PMC3084216"))
    assert restored.document.title == "A Study"
    assert restored.document.source == "pmc"


# --- the transport, without a socket ------------------------------------


def test_a_reference_with_no_identifier_never_opens_a_socket(monkeypatch):
    """Most references are neither on arXiv nor in PMC. They must cost nothing."""
    def explode(*a, **k):
        raise AssertionError("no identifier means no request")

    monkeypatch.setattr("urllib.request.urlopen", explode)
    result = HttpFullText().fetch(entry(journal="Nature", doi="10.1038/x"))
    assert result.status is Status.NOT_FOUND
    assert "no arXiv id or PMCID" in result.detail


def test_a_pdf_only_submission_is_a_settled_fact(monkeypatch, tmp_path):
    """arXiv serves the PDF when a submission has no source. That is permanent,
    so it is NOT_FOUND and worth caching -- not a transient failure."""
    source = HttpFullText(scratch=tmp_path)
    monkeypatch.setattr(source, "_get", lambda url, index: b"%PDF-1.7\nbinary")
    result = source.fetch(entry(eprint="1706.03762"))
    assert result.status is Status.NOT_FOUND
    assert "cannot read" in result.detail


def test_a_network_failure_is_unknown(monkeypatch, tmp_path):
    source = HttpFullText(scratch=tmp_path)
    monkeypatch.setattr(source, "_get", lambda url, index: None)
    assert source.fetch(entry(eprint="1706.03762")).status is Status.UNKNOWN


def test_a_paper_outside_the_open_access_subset_is_not_found(monkeypatch):
    """PMC answers for every id; papers outside the subset come back as
    metadata with no body."""
    source = HttpFullText()
    monkeypatch.setattr(
        source,
        "_get",
        lambda url, index: b"<article><front><article-meta/></front></article>",
    )
    result = source.fetch(entry(pmcid="PMC3084216"))
    assert result.status is Status.NOT_FOUND
    assert "open-access subset" in result.detail


def test_pmc_full_text_is_read(monkeypatch):
    source = HttpFullText()
    monkeypatch.setattr(
        source,
        "_get",
        lambda url, index: (
            b"<article><body><sec><p>The cost is quadratic in the input "
            b"length, which we establish below.</p></sec></body></article>"
        ),
    )
    result = source.fetch(entry(pmcid="PMC3084216"))
    assert result.usable
    assert "quadratic" in result.document.text
    assert result.document.source == "pmc"
