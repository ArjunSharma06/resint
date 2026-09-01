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

| Batch | Papers | Findings | Crashes | Anchors | Failed | Seconds | Commit |
|---|---|---|---|---|---|---|---|
| 1 | 68 / 71 | 340 | 0 | 687 | 0 | 23,354 | `905ce21` |
| 1b | 71 / 71 | 394 | 0 | 795 | 0 | **3,318** | `544c2e5` |

Batch 1b is batch 1 re-run after the fixes below: same papers, same model
(`groq/openai/gpt-oss-120b`), different code. **7× faster**, and the first run
to finish.

**Not yet run:** batches 2–5.

---

## Batch 1 — 2026-08-27

First run in which **all 19 rules executed on real papers.** `claim/unimplemented`
fired for the first time in the project's history: it needs a repository *and*
a model in the same run, and until this batch no such run had existed.

### Health

> **Every number below comes from a truncated run.** The batch stopped at
> 68 of 71 and the cause was recorded as "the session ended". It was not:
> the pool deadlocked at a `max_tasks_per_child` boundary, since reproduced
> deliberately. The three missing papers are **whatever was in flight**, not
> a random sample, so every rate here is wrong by an unknown amount.
>
> The conclusions drawn from them held up against batch 1b, which completed
> 71 of 71 -- but they held by luck rather than by construction. Treat the
> direction as evidence and the digits as indicative.

```
papers        68 of 71   (a deadlock, misread at the time as the session ending)
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


---

## Batch 1b — 2026-08-29

Batch 1 re-run on corrected code. The point was to test three predictions, so
they are recorded here whether or not they held.

### What was predicted, and what happened

| Prediction | Result | |
|---|---|---|
| `bib/unresolved` high severity 151 → 18 | **151 → 10** | held |
| Unanswered model calls ~198 → near zero | **198 → 68** | partly |
| Model rules produce findings | **0 → 2** | equivocal |

```
bib/unresolved severity   high 151 -> 10     med 26 -> 214     low 17 -> 20
wall clock                23,354s -> 3,318s
papers completed          68 -> 71
anchor failures           0 -> 0
```

### The severity fix worked as designed

151 high-severity findings became 10. What remains are the DOI failures --
a registered record that does not resolve -- which is the evidence the rule
was always meant to rest on. The other 214 are now `med`, where a failed
title search belongs.

### The abstention fix paid for itself immediately

Batch 1 could only say *"the model did not answer"* 198 times. Batch 1b says:

```
18x  claim/scope-creep       "rate limited, and still limited after N retries"
18x  claim/unsupported       "rate limited, and still limited after N retries"
18x  eval/baseline-fairness  "rate limited, and still limited after N retries"
12x  claim/overreach         "rate limited, and still limited after N retries"
```

So roughly **18 of 71 papers** still exhaust the token budget. Not a broken
prompt, not a refusal -- a quota, named as one. That distinction cost nothing
to record and would have cost hours to guess at.

### The survey returns content; the quotes are the weak link

The question the combined prompt raised was whether asking six things at once
degrades each. The abstentions answer it: the model **is** extracting, and the
extractions **are** reaching the rules.

```
8x  claim/unsupported       "1 abstract claim could not be checked: the quoted
                             sentence was not found"
4x  claim/unsupported       "N abstract claims could not be checked ..."
2x  eval/baseline-fairness  "1 budget comparison could not be checked ..."
2x  claim/overreach         "1 comparison could not be checked ..."
```

That is the verification mechanism doing its job -- quotes that do not appear
in the paper are discarded rather than becoming findings. But it is also the
honest cost of the shared prompt: a model asked six things quotes less
precisely than one asked a single question.

`claim/scope-creep` abstained 24 times with *"no evaluation datasets were
identified"*, which is the rule's own guard rather than a model failure. For a
corpus of mathematics and clinical papers, that is the correct answer.

**Verdict: inconclusive, and worth keeping.** Two findings across 71 papers is
too few to judge, but these are maths and biomedical papers and these rules
target machine-learning benchmark claims. A batch weighted towards `cs.LG`
would test them properly. The 7× saving is real and paid for now.

### Still open

- **18 of 71 papers hit the token ceiling.** Options: Gemini (measured ~3x the
  free throughput), splitting batches across providers, or a paid key at
  roughly a dollar for the whole corpus.
- **Six rules executed and stayed silent**, unchanged from batch 1. Only
  planted fixtures separate *correct and rare* from *broken*.
- `bib/unresolved` still fires on 76% of papers. Severity is now honest, but
  the rate says the underlying title-matching is weak.

### A harness bug this batch found

The first attempt at 1b **deadlocked at exactly 25 completions** -- the
`max_tasks_per_child=25` recycle boundary. The retired worker exited, no
replacement spawned, and the parent blocked in `wait()` for fifteen minutes
with no error.

Batch 1 stopping three papers short was almost certainly the same thing,
misread at the time as the session ending.

Fixed twice over: the recycle limit is gone, and the total deadline
(`timeout x papers`, six hours) is replaced by a **stall detector** that gives
up when nothing has *completed* for `timeout` seconds. The second fix is the
one that matters -- it catches the next hang whatever causes it.

**The mechanism is now proven rather than inferred.** The original fix rested
on a correlation: the stall was at 25, the setting was 25, removing it made the
symptom go away. The "verification" run then used fifteen workers over forty
papers -- about three tasks each, never within twenty of the boundary -- so it
tested nothing at all.

A minimal reproduction settles it. `max_tasks_per_child=5`, one worker, twelve
trivial tasks: the run hangs before completing the fifth, the retired worker is
never replaced, the parent blocks in `wait()` indefinitely, and even pool
shutdown never returns. CPython 3.13.1 on Windows.

Ruling out the alternatives mattered more than confirming this one. The Pacer,
connection reuse and the resolver's thread pool are **not** involved -- and any
of those would have been far worse, being live in ordinary `resint check` runs,
where a hang in CI is the worst failure mode available.

---

## What batch 1b changed — 2026-08-29

A review of the rules against the question *"could a reader with the paper in
front of them do this?"* produced a list of precision defects. All were fixed
before batch 2, because a firing rate measured on known-broken rules measures
nothing.

### Two rules were wrong every time they fired

| Rule | Real findings | After |
|---|---|---|
| `stats/pvalue-mismatch` | 3 | **0** |
| `numbers/table-arithmetic` | 2 | **0** |

`stats/pvalue-mismatch` treated the reported *p* as an interval and the
**statistic** as a point. `t = 2.086` is not a claim that t equals 2.086; it
claims t rounds to it, so t lies in [2.0855, 2.0865). Recomputing from the
midpoint alone turned the author's rounding into their error -- on all three
findings the tool produced. p is now computed at both ends and a finding
requires the whole range to fall outside what the reported p claims.

`numbers/table-arithmetic` fired on any percentage column not summing to 100.
A column of independent rates -- employment rate by region -- has every value
in [0,100] and sums to whatever it sums to; one real table summed to 474.8.
Firing now requires the sum to be *near* 100, because being far from it is
evidence the column was never a partition.

### `bib/unresolved` was one rule doing two jobs

176 title-only findings against 18 DOI ones, and the 176 buried the 18. Split:

- **`bib/unresolved`** -- a DOI that resolves nowhere. High. The fabrication signal.
- **`bib/doi-mismatch`** -- new. A DOI that resolves *to a different paper*, which the old rule structurally could not see. Two signals: title **and** authors must both disagree before it says so.
- **`bib/unindexed`** -- no DOI, title search failed. Low, **off by default**.

Unindexable entry types -- theses, `@misc`, technical reports -- are now
excluded from the denominator rather than downgraded. A thesis Crossref has
never heard of is not a finding at any severity.

### Three bugs found while making those changes

**An unreachable index was counted as a search.** `_get()` returned `None` both
for "answered, no match" and "could not connect", so an offline machine looked
like proof a paper does not exist -- the `UNKNOWN → NOT_FOUND` leak, in the one
place it matters most. Found when DBLP was added and its TLS handshake failed
locally: every reference then claimed four indices had been searched when three
had.

**`read_repo()` with no `needs` loaded nothing.** It read "unspecified" as
"want none", the opposite of `paper_from_path`. All five `repro/` rules looked
at an empty world and stayed quiet -- indistinguishable from a clean
repository. Silence is what hid it; the new coverage census is what showed it,
reporting *"2 hyperparameters named, 0 located"* on a fixture built so both
must be found. Four `repro/` rules now fire on the planted corpus where none
did.

**The sweep deadlocked at exactly 25 completions** -- the
`max_tasks_per_child` recycle boundary -- and sat silent for fifteen minutes.
Batch 1 stopping three papers short was almost certainly the same thing,
misread at the time as the session ending. The recycle limit is gone and the
six-hour total deadline is now a stall detector that gives up when nothing has
*completed*.

### What was added

- **A planted case for every rule.** Eight rules executed on 352 real papers and never fired once, which is not evidence either way. All 21 now have a document built so they must fire.
- **Document-level sample sizes.** GRIM required N in the same sentence as the mean, which found means in 1 paper of 148. N is now resolved from the section or the document when the sentence lacks it, and the mean records *where its N came from* so a rule can weigh it. 1/148 → 3/60 papers.
- **Stable fingerprints and inline suppression.** Without both, nobody runs a linter twice on the same paper.
- **`tools/review.py`** -- stratified sampling and hand-labelling, so precision becomes a number with an interval rather than a hope.
- **Two-signal title matching, and DBLP** as a fourth index for CS proceedings.

### Still unmeasured

Precision. Every number this log has published is a robustness number -- no
crashes, no anchor failures, all rules executed. Those say the plumbing works.
The one rule whose precision was actually checked turned out to be wrong every
time it fired, which is the strongest possible argument for checking the rest.

Batches 2--5 should run on the corrected code, and their findings labelled.

---

## The fourth UNKNOWN-collapse was not in the code -- 2026-09-01

`_get()` returned None for both "answered, no match" and "could not connect".
`read_repo()` returned an empty world for both "no repository" and "loaded
nothing". `fulltext.py` had the same shape a third time. All three were
fixed by looking at a function and noticing that one return value carried two
meanings.

`bib/unresolved` was the same bug in the rule's *premise*, and no amount of
reading the function would have found it. The code did exactly what it said:
query four indices, and if all four come back empty, report the DOI as
unresolved. Every branch was correct. The mistake was that "absent from
Crossref, OpenAlex, arXiv and DBLP" had been quietly equated with "does not
exist", and those are different propositions -- there are ten DOI registration
agencies and our four indices cover the output of roughly one.

It surfaced by reading nine findings by hand, which took ten minutes and was
prompted by the number looking wrong rather than by any test failing. Two of
the nine were live DOIs registered through the Chinese agency. The rule was
reporting them at high severity as fabrication signals, which means it fired
on papers for citing Chinese-language literature. A full test suite, a clean
anchor audit and zero crashes all held throughout.

The transferable part: the first three instances were found by inspection, so
the natural next move was to grep for more functions of that shape. That would
never have reached this one. **A rule can collapse two outcomes in its
premise while every line of its implementation is correct**, and the only
thing that catches it is reading findings against the world.

What found it was reading nine findings and checking them against reality --
ten minutes, prompted by a count that looked wrong. That is precisely the
labelling procedure, run on nine findings instead of thirty. So labelling is
not only a gate to pass before publishing precision numbers; it is the only
instrument this project has for finding premise-level bugs, and the only one
it can have, because no test written against a wrong premise will fail. The
afternoon of labelling is not an audit of finished work. It is the primary
bug-finding method, and it should be scheduled as one.

Guarded now by a planted *negative* in `tests/test_planted.py`: a DOI that
resolves through a non-Crossref agency must stay silent. A known-positive
cannot catch a bias, because the bias is in what the rule says about cases it
should never have spoken about.

## Method, revised after the deadlock — 2026-08-29

The order below is what the deadlock changed. It had been "sweep, then fix";
it is now "prove the mechanism, then audit the class, then sweep once".

### 1. Name the mechanism before fixing it

The stall at 25 was diagnosed by correlation -- the number matched a setting,
removing the setting removed the symptom -- and then "verified" by a run that
used fifteen workers over forty papers, about three tasks each, never within
twenty of the boundary. It tested nothing.

A minimal reproduction settles it in seconds: `max_tasks_per_child=5`, one
worker, twelve trivial tasks hangs before the fifth completes. That also rules
out the Pacer, connection reuse and the resolver's thread pool -- and ruling
those out mattered more than confirming this one, because all three are live
in ordinary `resint check` runs where a hang in CI is the worst outcome
available.

### 2. Audit the class, not the instance

`_get()` and `read_repo()` were one bug twice: a function returning nothing for
both "looked, found nothing" and "could not look". A deliberate pass found a
third in `resolve/fulltext.py`, running the *safe* way -- everything collapsed
to UNKNOWN, so nothing was over-claimed, but a paper genuinely absent from
arXiv was reported as "could not check" forever.

Cleared in the same pass: `parse/repo.py` reports unreadable files into
`unchecked`, `parse/tables.py` carries `irregular`, and the model provider
collapses to UNAVAILABLE *with* the reason attached. `resolve/passages.py`
gained `queryable()` so "the claim had no content words to search on" -- our
limitation -- is no longer silently identical to "the cited paper says nothing
on the subject".

Doing this before re-running is what avoids running twice.

### 3. Census before sweep, not during

Writing the census while a sweep runs would leave the recorded commit hash
describing code that no longer exists -- a silently wrong provenance record,
which is worse than none. Every external-facing rule now reports what it
examined:

```
bib/unresolved:     5 references, 3 with a DOI, 0 looked up, 5 could not be
                    looked up and were not judged
stats/pvalue:       3 test statistics found, 3 recomputed
repro/hparam-drift: 2 hyperparameters named in the paper, 2 located
```

That census is what caught `read_repo()`. It is now everywhere it can be.

### 4-5. Labelling discipline, fixed in advance

Decided before any labelling begins, so it cannot be argued case by case:

- **Ambiguous counts as a false positive.** A finding a careful reader cannot adjudicate in three minutes is one a user will not adjudicate either.
- **Findings per paper reported beside precision.** Thirty findings at 90% is a wall of text; three at 90% is a useful report. Precision alone cannot tell them apart.
- **Every false positive carries a short tag.** Aggregated per rule, those tags are the fix list.
- **No rate below ten labels.** Wilson on a handful is noise wearing a percentage sign, and `too few to rate` is printed instead.
- **Silent rules are listed, never omitted.** `0 findings; see planted fixtures and cannot_detect` is a result and reads as one. For the statistics rules a zero is partly a statement about the corpus, not the rule -- GRIM's home is psychology, and a corpus that is mostly arXiv CS produces a small number whatever the code does.

### Still to come

Labelling itself, which needs a person; then narrowing whatever it surfaces,
re-labelling only the rules whose logic changed, and publishing per-rule
numbers in the README. That last step is the one that makes the exercise worth
having done.

Two things about the coverage census are known-wrong and deliberately left
alone, written down here so they are not rediscovered.

**The census is riding in the wrong channel.** It goes out through
`ctx.abstain()`, which is the unchecked block -- built to say *I did not look
at this*. A census says the opposite: *here is what I did look at, and how much
of it*. Both now land under the same heading in the report, so a rule that
examined 214 of 218 references reads, at a glance, like a rule that examined
none. It needs its own field on the report rather than a share of that one.
The mechanism is sound and it has already earned its keep -- `read_repo()`
loading nothing was found by a census reporting "2 hyperparameters named, 0
located" -- so the fix is where the number is printed, not how it is computed.

**Three rules have a census. Eighteen do not.** Present on:

- `bib/unresolved` -- "218 references, 214 with a DOI, 214 looked up"
- `stats/pvalue-mismatch` -- "9 test statistics found, 6 recomputed"
- `repro/hparam-drift` -- "12 hyperparameters named in the paper, 9 located in
  the repository"

Absent on the other eighteen, in the order they are worth adding:

- `numbers/table-arithmetic`, `numbers/internal-mismatch`, `stats/grim`,
  `stats/significance-unsupported` -- these need it most. Each walks a
  population it can count (tables parsed, number pairs compared, means with a
  reportable N), and each is a rule whose silence is currently unreadable: no
  finding could mean a clean paper or an empty `paper.tables`, and only a
  denominator separates them.
- `repro/entrypoint-missing`, `repro/ghost-repo`, `repro/seed-claim`,
  `repro/unpinned-deps` -- the four that were silently seeing an empty
  repository until the `read_repo()` bug was found. A census on any one of
  them would have caught it a month earlier.
- `bib/orphans`, `bib/metadata-drift`, `bib/doi-mismatch`, `bib/unindexed` --
  all four already know their denominator; they simply do not print it.
- `bib/citation-support`, `claim/overreach`, `claim/scope-creep`,
  `claim/unimplemented`, `claim/unsupported`, `eval/baseline-fairness` -- the
  model tier. These already report what they *could not* check, item by item,
  which is half a census; what is missing is the total they started from.

Eighteen, not the thirteen this was first written down as. The miscount came
from counting the model rules' existing abstentions as a census, which they
are not -- they report the unchecked remainder without ever stating the
denominator it was subtracted from.
