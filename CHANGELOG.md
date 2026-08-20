# Changelog

Notable changes, newest first. This file is the source for release notes and
for anything written up on quevo.dev, so entries record *why* a change was
made where the reason is not obvious.

Format loosely follows [Keep a Changelog](https://keepachangelog.com).
Versions follow [Semantic Versioning](https://semver.org), with the caveat
that the API is not stable before `1.0`.

## [Unreleased]

Nothing yet.

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
