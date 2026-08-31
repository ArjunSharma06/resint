"""bib/doi-mismatch: the DOI resolves, but to a different paper.

The rule ``bib/unresolved`` cannot see this -- the DOI resolves, so nothing is
missing -- and it is the failure mode that grows as writing becomes more
model-assisted: an invented DOI has a real chance of colliding with a
registered record.

Almost every test here exists to hold one design decision: **two signals,
never one.** A title scoring low on its own is weak evidence, because
subtitles get dropped, translations differ, and chapters are cited under their
book's name. The author list has to corroborate before anything is reported as
the wrong paper.
"""

import pytest

from resint.ir.paper import BibEntry, Paper
from resint.ir.span import Source, Span
from resint.resolve import Record, Resolution, Status
from resint.rules import load_all
from resint.rules.bib.doi_mismatch import authors_agree, surname
from resint.rules.registry import Context

REG = load_all()
BIB = Source("refs.bib", "bib", path="refs.bib")

ATTENTION = "Attention Is All You Need"
RESNET = "Deep Residual Learning for Image Recognition"


def entry(key="k", **fields):
    fields.setdefault("doi", "10.5555/12345")
    spans = {name: Span(BIB, i * 10, i * 10 + 5) for i, name in enumerate(fields, 1)}
    return BibEntry(
        key=key,
        entry_type="article",
        fields=fields,
        span=Span(BIB, 0, 200, line=1, label=f"[{key}]"),
        field_spans=spans,
    )


def fire(entries, resolutions):
    paper = Paper(source_id="paper.tex")
    paper.bib = list(entries)
    paper.resolutions = dict(resolutions)
    return REG.get("bib/doi-mismatch").run(Context(paper=paper))


def found(title, authors=(), matched_by="doi"):
    return {
        "k": Resolution(
            Status.FOUND,
            record=Record(
                source="crossref", title=title, authors=tuple(authors),
                matched_by=matched_by,
            ),
        )
    }


# --- the finding it exists for ------------------------------------------


def test_a_doi_pointing_at_another_paper_is_reported():
    findings = fire(
        [entry(title=ATTENTION, author="Vaswani, Ashish and Shazeer, Noam")],
        found(RESNET, ["Kaiming He", "Xiangyu Zhang"]),
    )
    assert len(findings) == 1
    assert findings[0].severity.value == "high"
    assert "10.5555/12345" in findings[0].message
    assert RESNET in findings[0].message


def test_the_finding_carries_both_titles_so_a_reader_can_judge():
    message = fire(
        [entry(title=ATTENTION, author="Vaswani, Ashish")],
        found(RESNET, ["Kaiming He"]),
    )[0].message
    assert ATTENTION in message and RESNET in message


# --- two signals, never one ---------------------------------------------


def test_a_matching_author_means_a_title_variant_not_a_wrong_doi():
    """The likeliest cause of a low-scoring title with the right people is a
    dropped subtitle or a preprint/published pair -- not a wrong DOI."""
    findings = fire(
        [entry(title="Attention Is All You Need", author="Vaswani, Ashish")],
        found("Attention Is All You Need: A Transformer Architecture for "
              "Sequence Transduction Tasks", ["Ashish Vaswani"]),
    )
    assert findings == []


def test_an_uncomparable_author_list_never_reaches_high():
    """An entry with no author field is common and says nothing about whether
    the DOI is right, so it cannot carry a confident accusation."""
    findings = fire([entry(title=ATTENTION)], found(RESNET, ["Kaiming He"]))
    assert len(findings) == 1
    assert findings[0].severity.value == "med"


def test_a_similar_title_is_not_reported_at_all():
    findings = fire(
        [entry(title="Deep Residual Learning for Image Recognition")],
        found("Deep residual learning for image recognition.", ["Kaiming He"]),
    )
    assert findings == []


# --- what it refuses to judge -------------------------------------------


def test_a_title_matched_record_is_never_used():
    """A title search returns a best guess. Declaring 'this is a different
    paper' against a guess is the exact error the rule exists to catch."""
    findings = fire(
        [entry(title=ATTENTION, author="Vaswani, Ashish")],
        found(RESNET, ["Kaiming He"], matched_by="title"),
    )
    assert findings == []


def test_an_entry_with_no_doi_is_not_this_rules_business():
    e = BibEntry(
        key="k", entry_type="article",
        fields={"title": ATTENTION},
        span=Span(BIB, 0, 200, line=1),
    )
    assert fire([e], found(RESNET, ["Kaiming He"])) == []


@pytest.mark.parametrize("status", [Status.NOT_FOUND, Status.UNKNOWN])
def test_an_unresolved_doi_belongs_to_another_rule(status):
    findings = fire([entry(title=ATTENTION)], {"k": Resolution(status)})
    assert findings == []


def test_a_record_with_no_title_cannot_be_compared():
    findings = fire([entry(title=ATTENTION)], found("", ["Kaiming He"]))
    assert findings == []


# --- surname extraction, which the corroboration rests on ---------------


@pytest.mark.parametrize(
    "name, expected",
    [
        ("Vaswani, Ashish", "vaswani"),      # BibTeX convention
        ("Ashish Vaswani", "vaswani"),       # index convention
        ("van der Maaten, Laurens", "maaten"),
        ("Laurens van der Maaten", "maaten"),
        ("Smith Jr., John", "smith"),
        ("Erdős, Paul", "erdos"),            # folded for comparison
        ("", ""),
    ],
)
def test_surnames_reduce_to_the_same_token_from_either_convention(name, expected):
    assert surname(name) == expected


def test_authors_agree_on_a_shared_surname():
    assert authors_agree(["Vaswani, Ashish"], ["Ashish Vaswani", "Noam Shazeer"])


def test_authors_disagree_when_nothing_is_shared():
    assert authors_agree(["Vaswani, Ashish"], ["Kaiming He"]) is False


def test_an_empty_side_is_cannot_tell_not_disagreement():
    """None and False must stay distinct: an absent author list is not
    evidence against the DOI."""
    assert authors_agree([], ["Kaiming He"]) is None
    assert authors_agree(["Vaswani, Ashish"], []) is None


# --- the rule declares itself -------------------------------------------


def test_the_rule_names_its_blind_spots():
    rule = REG.get("bib/doi-mismatch")
    assert "preprint" in rule.cannot_detect
    assert rule.tier.value == "deterministic"


def test_every_finding_carries_two_anchors():
    findings = fire(
        [entry(title=ATTENTION, author="Vaswani, Ashish")],
        found(RESNET, ["Kaiming He"]),
    )
    assert len(findings[0].anchors) >= 2
