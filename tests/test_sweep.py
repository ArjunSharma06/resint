"""The sweep: crash isolation, the anchor audit, and the result store.

The anchor audit is the reason a sweep is worth running at all — it is the one
check that verifies itself, with no labels and no judgement. So most of these
tests are about proving it actually catches a broken anchor, rather than
passing because nothing was wrong.
"""

import json
from pathlib import Path

import pytest

from resint.ir.finding import Finding
from resint.ir.span import Source, Span
from resint.sweep import (
    PaperRecord,
    audit_anchors,
    check_one,
    fingerprint,
    read_records,
    write_record,
)

CORPUS = Path(__file__).resolve().parents[1] / "corpus"
SRC = Source("paper.tex", "latex", path="paper.tex")

TEXT = "line one here\nline two here\nline three here\n"


def finding_at(start, end, line=None, rule="numbers/internal-mismatch"):
    return Finding(
        rule_id=rule,
        severity="high",
        tier="deterministic",
        message="m",
        anchors=[
            Span(SRC, start, end, line=line, label="a"),
            Span(SRC, 0, 4, line=1, label="b"),
        ],
    )


# --- the anchor audit ----------------------------------------------------


def test_a_correct_anchor_passes():
    audit = audit_anchors([finding_at(5, 8, line=1)], {"paper.tex": TEXT})
    assert audit.clean
    assert audit.checked == 2


def test_an_out_of_bounds_anchor_is_caught():
    audit = audit_anchors([finding_at(9000, 9010)], {"paper.tex": TEXT})
    assert not audit.clean
    assert "out of bounds" in audit.failures[0].reason


def test_an_anchor_on_only_whitespace_is_caught():
    text = "abc     def"
    audit = audit_anchors([finding_at(3, 7)], {"paper.tex": text})
    assert not audit.clean
    assert "whitespace" in audit.failures[0].reason


def test_a_wrong_line_number_is_caught():
    """The exact regression the offset map keeps producing."""
    audit = audit_anchors([finding_at(30, 35, line=1)], {"paper.tex": TEXT})
    assert not audit.clean
    assert "line says 1" in audit.failures[0].reason


def test_a_correct_line_number_passes():
    start = TEXT.index("line three")
    audit = audit_anchors([finding_at(start, start + 4, line=3)], {"paper.tex": TEXT})
    assert audit.clean


def test_a_source_we_were_not_given_is_unverifiable_not_failed():
    """Calling a finding broken because the auditor lacked the file would be
    the same mistake the tool exists to avoid."""
    audit = audit_anchors([finding_at(0, 4)], {})
    assert audit.clean
    assert audit.failed == 0
    assert "paper.tex" in audit.missing_sources


def test_the_failure_list_is_capped_but_the_count_is_not():
    findings = [finding_at(9000, 9010) for _ in range(40)]
    audit = audit_anchors(findings, {"paper.tex": TEXT})
    assert audit.failed == 40
    assert len(audit.as_dict()["failures"]) == 20


# --- crash fingerprinting ------------------------------------------------


def _raise(kind, depth=0):
    if depth < 2:
        return _raise(kind, depth + 1)
    raise kind("boom")


def test_the_same_crash_from_the_same_place_groups():
    seen = []
    for _ in range(3):
        try:
            _raise(ValueError)
        except ValueError as exc:
            seen.append(fingerprint(exc))
    assert len(set(seen)) == 1, "forty crashes should collapse into one bug"


def test_different_exception_types_do_not_group():
    prints = []
    for kind in (ValueError, TypeError):
        try:
            _raise(kind)
        except Exception as exc:
            prints.append(fingerprint(exc))
    assert len(set(prints)) == 2


def test_a_fingerprint_is_short_and_stable():
    try:
        raise RuntimeError("x")
    except RuntimeError as exc:
        fp = fingerprint(exc)
    assert len(fp) == 12 and fp.isalnum()


# --- running one paper ---------------------------------------------------


def test_a_good_paper_produces_a_clean_record():
    record = check_one(CORPUS / "planted" / "paper.tex")
    assert record["status"] == "ok"
    assert record["findings"]
    assert record["anchor_audit"]["failed"] == 0
    assert record["anchor_audit"]["checked"] > 0
    assert record["timings"]["total"] > 0


def test_every_corpus_finding_survives_its_own_audit():
    """If this ever fails, the offset map has regressed."""
    for name in ("planted", "clean", "unsupported"):
        record = check_one(CORPUS / name / "paper.tex")
        audit = record["anchor_audit"]
        assert audit["failed"] == 0, (name, audit["failures"])
        assert audit["missing_sources"] == [], (name, audit["missing_sources"])


def test_an_unreadable_input_is_a_result_not_a_crash(tmp_path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.7\n\x00binary")
    record = check_one(pdf)
    assert record["status"] == "unreadable"
    assert record["error"] is None
    assert "PDF" in record["acquire"]["reason"]


def test_a_crash_is_captured_rather_than_raised(monkeypatch):
    """A worker must come back with a record, whatever happened inside it."""
    import resint.sweep.runner as runner

    def explode(*args, **kwargs):
        raise RuntimeError("parser exploded")

    monkeypatch.setattr(runner, "paper_from_latex", explode)
    record = check_one(CORPUS / "planted" / "paper.tex")

    assert record["status"] == "error"
    assert record["error"]["type"] == "RuntimeError"
    assert record["error"]["fingerprint"]
    assert "parser exploded" in record["error"]["message"]


def test_the_census_only_counts_slices_that_were_requested():
    """A slice nobody asked for reading as zero looks like a broken parser."""
    record = check_one(CORPUS / "planted" / "paper.tex")
    census, needs = record["slice_census"], record["needs"]
    for key in census:
        if key != "text_chars":
            assert f"paper.{key}" in needs


def test_a_record_carries_provenance():
    record = check_one(CORPUS / "planted" / "paper.tex")
    assert record["source_sha256"], "a diff must be able to prove the input was the same"
    assert record["resint_version"]


# --- the store -----------------------------------------------------------


def test_records_round_trip_through_jsonl(tmp_path):
    path = tmp_path / "sweep.jsonl"
    written = [
        PaperRecord.from_dict(check_one(CORPUS / n / "paper.tex"))
        for n in ("planted", "clean")
    ]
    with path.open("w", encoding="utf-8") as handle:
        for record in written:
            write_record(handle, record)

    loaded = read_records(path)
    assert [r.paper_id for r in loaded] == [r.paper_id for r in written]
    assert [r.findings for r in loaded] == [r.findings for r in written]


def test_findings_reload_losslessly(tmp_path):
    path = tmp_path / "sweep.jsonl"
    record = PaperRecord.from_dict(check_one(CORPUS / "planted" / "paper.tex"))
    with path.open("w", encoding="utf-8") as handle:
        write_record(handle, record)

    pool = {}
    reloaded = read_records(path)[0].loaded_findings(pool)
    assert [f.to_dict() for f in reloaded] == record.findings
    assert pool, "sources should be interned, not reallocated per anchor"


def test_a_truncated_final_line_does_not_lose_the_rest(tmp_path):
    """What an interrupted run leaves behind. 179 good records still load."""
    path = tmp_path / "sweep.jsonl"
    record = PaperRecord(paper_id="a")
    with path.open("w", encoding="utf-8") as handle:
        write_record(handle, record)
        write_record(handle, PaperRecord(paper_id="b"))
        handle.write('{"paper_id": "c", "stat')  # killed mid-write

    loaded = read_records(path)
    assert [r.paper_id for r in loaded] == ["a", "b"]


def test_corruption_anywhere_else_is_fatal(tmp_path):
    path = tmp_path / "sweep.jsonl"
    path.write_text('{"paper_id": "a"}\nnot json at all\n{"paper_id": "c"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        read_records(path)


def test_unknown_fields_are_ignored_so_the_schema_can_grow(tmp_path):
    path = tmp_path / "sweep.jsonl"
    path.write_text(
        json.dumps({"paper_id": "a", "invented_later": 42}) + "\n", encoding="utf-8"
    )
    assert read_records(path)[0].paper_id == "a"


# --- the dirty-tree guard -------------------------------------------------
#
# A sweep costs hours and is then labelled by hand. Both are spent against a
# specific version of the rules, so a record stamped with a commit it did not
# execute is worse than one stamped with nothing.


def _tool():
    """tools/sweep.py, which is a script rather than an importable module."""
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "tools" / "sweep.py"
    spec = importlib.util.spec_from_file_location("_sweep_tool", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_changed_source_makes_a_sweep_unreproducible():
    status = " M src/resint/rules/stats/pvalue.py\n M tools/sweep.py\n"
    assert _tool().dirty_from_status(status) == [
        "src/resint/rules/stats/pvalue.py",
        "tools/sweep.py",
    ]


def test_untracked_source_counts():
    # parse/inline.py sat untracked for a day while being imported at
    # runtime, so "tracked and modified" would miss what is most likely to
    # be moving.
    assert _tool().dirty_from_status("?? src/resint/parse/inline.py\n") == [
        "src/resint/parse/inline.py"
    ]


def test_notes_and_readme_do_not_block_a_sweep():
    # Refusing on files that cannot change a finding would train the operator
    # to reach for --allow-dirty by reflex, which costs the guard everything.
    status = " M README.md\n M notes/sweep-log.md\n?? sweeps/batch-2.jsonl\n"
    assert _tool().dirty_from_status(status) == []


def test_a_renamed_rule_is_reported_at_its_new_path():
    status = "R  src/resint/rules/bib/old.py -> src/resint/rules/bib/new.py\n"
    assert _tool().dirty_from_status(status) == ["src/resint/rules/bib/new.py"]


def test_a_clean_tree_is_empty():
    assert _tool().dirty_from_status("") == []


def test_a_sweep_can_be_restricted_to_a_corpus_file(tmp_path):
    """The cache accumulates: 619 PubMed Central articles on disk against 30
    in the September corpus. Without this the sweep covers whatever happens to
    be there while its record names the corpus it was supposed to run on."""
    tool = _tool()
    ids = tmp_path / "corpus.txt"
    ids.write_text(
        "\n".join(["# --- arXiv CS ---", "2608.1v1", "", "PMC9  # kept", ""]),
        encoding="utf-8",
    )
    papers = [Path(n) for n in
              ("2608.1v1.tar.gz", "2608.2v1.tar.gz", "PMC9.nxml", "PMC8.nxml")]

    kept = tool.keep_listed(papers, tool.corpus_ids(ids))
    assert [p.name for p in kept] == ["2608.1v1.tar.gz", "PMC9.nxml"]


@pytest.mark.parametrize(
    "name, want",
    [
        ("2608.1v1.tar.gz", "2608.1v1"),
        ("2608.1v1.tgz", "2608.1v1"),
        ("PMC13427623.nxml", "PMC13427623"),
        ("paper.tex", "paper"),
    ],
)
def test_a_papers_id_survives_its_double_suffix(name, want):
    """Path.stem leaves "2608.1v1.tar" behind on a .tar.gz, and an arXiv id
    carries a dot of its own, so neither one suffix nor all of them can just
    be stripped."""
    assert _tool().paper_id_of(Path(name)) == want


def test_a_missing_corpus_file_stops_the_sweep(tmp_path, capsys):
    """Sweeping everything on disk instead would produce a record that names
    a corpus it did not run on -- the failure the flag exists to prevent."""
    (tmp_path / "a.tex").write_text("x", encoding="utf-8")
    assert _tool().main([str(tmp_path), "--ids", str(tmp_path / "nope.txt"),
                         "--out", str(tmp_path / "o.jsonl")]) == 2
    assert "no such corpus file" in capsys.readouterr().err
