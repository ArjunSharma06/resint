# Changelog

Notable changes, newest first. This file is the source for release notes and
for anything written up on quevo.dev, so entries record *why* a change was
made where the reason is not obvious.

Format loosely follows [Keep a Changelog](https://keepachangelog.com).
Versions follow [Semantic Versioning](https://semver.org), with the caveat
that the API is not stable before `1.0`.

## [Unreleased]

Fixes from the first run against a real 48 KB paper with a 35-entry
bibliography. None of these were caught by the corpus fixtures, which is the
argument for using the tool on real work early.

### Fixed

- **Reference resolution appeared to hang.** Lookups ran sequentially: 35
  entries against 3 indices is over a hundred round trips, and the run had to
  be interrupted. Resolution is now concurrent with an overall time budget,
  and reports progress. Entries not reached inside the budget come back
  UNKNOWN, so running out of time can never manufacture a finding.
- **`bib/metadata-drift` reported eleven false positives.** Title search used
  overlap over the smaller token set, so a short title contained in a longer
  one scored near 1.0 — "Linformer: Self-Attention with Linear Complexity"
  matched "Mult-Pool Self Attention: a lightweight attention with linear
  complexity" at 0.80. Now uses Jaccard similarity, and picks the *best*
  candidate rather than the first above the threshold.
- **Drift now requires an authoritative record.** A DOI identifies one
  registered work; a title search returns a guess. Reporting someone's
  metadata as wrong against a guess is exactly the failure the rule exists to
  avoid, so title-matched records abstain with a stated reason. They still
  count as existing, so `bib/unresolved` stays quiet.
- **`bib/orphans` flooded the report.** Thirteen uncited entries produced
  thirteen findings and buried everything else. Uncited entries are now one
  grouped finding; undefined keys stay separate, since each is a distinct
  broken reference in the compiled document.
- **Ctrl+C printed a traceback.** Now exits cleanly with code 130.
- **Location lines with several anchors were unreadable.** Display truncates
  past two with a count; every anchor is still present in JSON and SARIF.

## [0.1.0] — first public release

Thirteen rules, 381 tests, no dependencies beyond the standard library.

### Rules

**`numbers/` — internal consistency**
- `internal-mismatch` — the abstract and the results table disagree about one
  value. Turns on a *near-miss* test rather than equality: a prose value close
  to a cell is a stale number, a distant one is a different quantity.
- `table-arithmetic` — stated totals that do not equal the entries above them,
  with tolerance derived from the precision the paper itself reported.

**`stats/` — statistical forensics**
- `pvalue-mismatch` — recomputes *p* from the test statistic and degrees of
  freedom for *t*, *F*, χ², *r* and *z*. Escalates to high only when the
  disagreement flips a significance decision.
- `grim` — means arithmetically impossible for the stated *N*. Abstains once
  granularity reaches 10^decimals, where the test carries no information.
- `significance-unsupported` — claims of reliability with no test statistic,
  interval or variance anywhere in the document. Document-level on purpose, so
  a paper that runs its tests properly never lands here.

**`bib/` — citation integrity**
- `unresolved` — references that resolve in none of Crossref, OpenAlex, arXiv
  or Semantic Scholar. The fabricated-citation signal.
- `metadata-drift` — entries whose year or title disagrees with the canonical
  record. Author lists are deliberately not compared.
- `orphans` — keys cited with no entry, entries never cited.

**`repro/` — paper against code** (requires `--repo`)
- `hparam-drift` — the paper's hyperparameters against the value a run would
  actually use, resolving argparse defaults, YAML layering and Hydra
  composition.
- `seed-claim` — "averaged over five seeds" where the code fixes one.
- `entrypoint-missing` — README commands pointing at files that do not exist.
- `ghost-repo` — a linked repository holding no code.
- `unpinned-deps` — no lockfile and no version constraints.

### Design

- **Two anchors per finding**, enforced at construction. One anchor is an
  assertion the reader has to check; two make it a comparison they can verify
  by reading the finding itself.
- **Absence findings** — the one legitimate single-anchor case, where half the
  comparison is a negative with no location. They pass `absent_from` naming
  exactly what was searched.
- **`cannot_detect` is mandatory.** The engine refuses to register a rule
  without one, and the limitation travels into JSON and SARIF output.
- **Three resolution outcomes.** `UNKNOWN` can never become a finding —
  reporting a reference as fabricated because the network was down would be
  the worst bug this tool could ship.
- **Declared requirements.** Rules see only what they list, which makes
  laziness enforceable: a run whose rules never declare `paper.resolutions`
  never opens a socket.
- **Rules read, never write.** Abstention is `ctx.abstain(reason)`, collected
  by the engine. Silence without a stated reason is indistinguishable from a
  pass.

### Infrastructure

- No dependencies. The incomplete beta and gamma functions are implemented
  locally rather than pulling scipy in for four CDFs; Python's own `ast`
  replaces tree-sitter for reading code.
- LaTeX normalization with a truthful offset map, so `line 98` means line 98
  of the author's file.
- `.resint.yml` suppression, where every entry states a reason and suppressed
  findings survive into output rather than being dropped.
- Terminal, JSON and SARIF output from the same `Report`.
- Fixture corpus with positive *and* negative papers. The first real false
  positive in the project — a stray "one-tailed" halving the *p*-value of a
  statistic in a neighbouring sentence — was invisible to every unit test and
  was caught by the clean fixture on its first run.
- `docs/rules.md` is generated from the registry, and CI fails if the
  committed copy drifts.

### Known gaps

- Five of eighteen planned v1 rules are unimplemented: `repro/dead-asset`,
  `numbers/sample-size-drift`, and the model-assisted tier (`claim/`, `eval/`).
- PDF input is not supported yet — LaTeX source only.
- Only Python is read on the repository side.
