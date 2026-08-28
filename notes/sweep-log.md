# Sweep log

What running resint over real papers actually produced, batch by batch.

This file exists because a rule's firing rate is the only cheap evidence we
have about its precision, and a rate means nothing without the one before it.
Every batch records the commit it ran on: **two batches on different code are
not a before-and-after**, and saying so here is less error-prone than
remembering it.

Generated numbers come from `python tools/compare.py sweeps/*.jsonl`.

---

## The corpus

| | |
|---|---|
| arXiv e-prints | 226 fetched, 204 with readable LaTeX source, 22 PDF-only |
| PubMed Central | 148 open-access articles, JATS XML |
| Repositories | 69 cloned, paired to the papers that link them |
| **Checkable papers** | **352** — 204 `.tar.gz` + 148 `.nxml` |
| Fields | 8 arXiv categories, 6 PMC topics |

Batches are **interleaved**, not concatenated: a batch taken off the front of
a concatenated list would be all arXiv, and a run that only sees one format
proves nothing about the other.

```bash
python tools/sweep.py ~/.cache/resint/eprints ~/.cache/resint/pmc \
    --repos ~/.cache/resint/repos --resolve --mailto <you> \
    --model groq/openai/gpt-oss-120b --batch N/5 --out sweeps/batch-N.jsonl
```

---

## Batch overview

| Batch | Papers | Findings | Crashes | Anchors | Failed | Commit | Notes |
|---|---|---|---|---|---|---|---|
| 1 | 68 / 71 | 340 | 0 | 687 | 0 | `905ce21` | first run with all 19 rules; stopped early when the session ended |

**Not yet run:** batches 2–5. Held deliberately — batch 1 surfaced two bugs
that would make four more batches four more measurements of the same faults.

---

## Batch 1 — 2026-08-27

First run in which **all 19 rules executed on real papers.** `claim/unimplemented`
fired for the first time in the project's history: it needs a repository *and*
a model in the same run, and until this batch no such run had existed.

### Health

```
papers        68 of 71   (session ended before the last three)
crashes        0
anchors      687 checked, 0 failed
unreadable     0
wall clock    ~2 h at 3 workers
```

Zero crashes and zero anchor failures across 687 checked anchors is the
result that matters most: every finding points at text that is really there.

### Firing rates

| Rule | Findings | Papers | Rate | |
|---|---|---|---|---|
| `bib/unresolved` | 194 | 49 | **72%** | ⚠ |
| `bib/metadata-drift` | 62 | 31 | **46%** | ⚠ |
| `bib/orphans` | 35 | 35 | **51%** | ⚠ |
| `numbers/internal-mismatch` | 26 | 4 | 6% | |
| `numbers/table-arithmetic` | 7 | 5 | 7% | |
| `repro/hparam-drift` | 5 | 3 | 4% | |
| `repro/entrypoint-missing` | 3 | 1 | 1% | |
| `stats/pvalue-mismatch` | 3 | 1 | 1% | |
| `claim/unimplemented` | 2 | 1 | 1% | first ever |
| `repro/unpinned-deps` | 2 | 2 | 3% | |
| `stats/significance-unsupported` | 1 | 1 | 1% | |
| `bib/citation-support` | 0 | 0 | — | abstained: no full-text source |
| `claim/overreach` | 0 | 0 | — | model did not answer |
| `claim/scope-creep` | 0 | 0 | — | model did not answer |
| `claim/unsupported` | 0 | 0 | — | model did not answer |
| `eval/baseline-fairness` | 0 | 0 | — | model did not answer |
| `repro/ghost-repo` | 0 | 0 | — | silent |
| `repro/seed-claim` | 0 | 0 | — | silent |
| `stats/grim` | 0 | 0 | — | silent |

⚠ = fires on ≥35% of papers. A rule hitting most papers is a rule with a
precision problem, not a corpus of uniformly broken papers.

### Where the time went

```
rules      22,034 s     94%
parse       1,306 s      6%
acquire        12 s
```

Rule time is almost entirely model calls and reference lookups waiting on
rate limits. Parsing 68 papers takes nineteen seconds a paper including
archive extraction, which is not the bottleneck and does not need to be.

---

## What batch 1 found

### 1. `bib/unresolved` rates a failed title search as strong evidence

**194 findings, 151 of them `high`.** The rule's own docstring says:

> A DOI that fails to resolve is treated as stronger evidence than a title
> that fails to match, because a DOI is a claim about a specific registered
> record rather than a string that might be spelled differently.

The code did not implement it. The final branch rated a title-only miss
`high` — identical to a dead DOI.

```
by evidence :  176 title-only,  18 DOI
by severity :  151 high,  26 med,  17 low
```

Title search fails for ordinary reasons: abbreviated titles in
bibliographies, non-English venues, books, standards, chapters, and plain
coverage gaps in Crossref and OpenAlex. Reporting 151 of those at `high` —
where the message is effectively *this reference may not exist* — is the
accusation machine the docstring claims three guards prevent. The third guard
was documented and never written.

**Fixed:** title-only misses are now `med` at most. `high` is reserved for a
DOI that does not resolve, which is a claim about a specific registered
record.

### 2. The model tier was rate-limited into silence

```
56x  claim/scope-creep       "the model did not answer"
54x  eval/baseline-fairness  "the model did not answer"
51x  claim/unsupported       "the model did not answer"
37x  claim/overreach         "the model did not answer"
```

Across 68 papers, roughly **80% of model calls failed.** The four rules were
not being conservative — they never got an answer to be conservative about.

Worse, the abstention said only *"the model did not answer"* and discarded the
provider's reason. Rate limit, refusal, malformed reply and truncation all
collapsed into one sentence, so the sweep could not distinguish *we were
throttled* from *the prompt is broken* — the exact confusion the three-outcome
contract exists to prevent, reintroduced at the reporting layer.

**Fixed:** abstentions now carry the provider's detail.

### 3. The sweep destroyed its own partial results on retry

`--out` was opened in `"w"` mode, so re-running an interrupted batch to the
same file truncated the records already collected. Batch 1 stopped at 68 of
71; a naive retry would have wiped all 68.

**Fixed:** the sweep refuses to overwrite a non-empty output file unless
`--force` is given.

---

## Rules that executed but stayed silent

`repro/ghost-repo`, `repro/seed-claim`, `stats/grim`, and the four model rules.

**This is not evidence either way.** A rule that never fires across 68 papers
may be correct and rare — GRIM violations *are* rare, which is the point of
the rule — or it may be broken. More papers cannot separate those: they give
you a larger zero.

Only a **planted case** distinguishes them: a fixture constructed so the rule
*must* fire. `corpus/planted/` does this for some rules and is why
`stats/grim` is known to work at all. Extending it to cover every rule is
tracked separately and is the honest completion of this measurement.

---

## Method

**Fix between batches:** crashes, anchor failures, and clear bugs. The results
are uninterpretable otherwise.

**Wait until all batches are done:** precision thresholds, severities, message
wording. Those are judged *by* the rates, so changing them mid-run means a
moved rate cannot be attributed to the fix rather than to different papers.

**After any fix, re-run the earlier batches** so every number sits on one
commit. The model cache makes this nearly free — unchanged prompts cost
nothing.
