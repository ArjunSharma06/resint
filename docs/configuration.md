# Configuration

`resint init` writes a starter file. resint looks for `.resint.yml` (or
`.resint.yaml`) beside the paper, then walks upward.

```yaml
version: 1

suppress:
  - rule: bib/metadata-drift
    match: "[vaswani2017]"
    reason: "Cites the NeurIPS proceedings version deliberately."
    expires: "2027-01-01"

  - rule: repro/unpinned-deps
    reason: "Library target; pinning is the consumer's decision."

rules:
  stats/grim: off        # no integer-scale response data in this work
```

## Suppression

| Field | | |
|---|---|---|
| `rule` | required | The rule id to silence |
| `reason` | **required** | Why. A suppression without one is a parse error. |
| `match` | optional | Substring of the message — narrows to one finding |
| `expires` | optional | ISO date. After it, the finding returns. |

### Three properties worth knowing

**A suppression without a reason is a parse error.** This file is the record
of every judgement made about the work. A silenced finding with no explanation
is unauditable six months later, by which point nobody remembers whether it
was considered or just annoying.

**Suppressed findings are not deleted.** They survive into `--format json` and
`--format sarif` marked with their reason, and SARIF emits them as
suppressions rather than dropping them — a reviewer needs to tell *"fired and
was consciously accepted"* from *"never ran"*. A suppression can therefore
never hide a regression.

**A suppression that stops matching says so.**

```
  suppression for bib/metadata-drift matched nothing; the rule may have
  changed or the finding may already be fixed
```

Silent dead config is how a suppression file rots into something nobody trusts
to edit.

### Expiry

```yaml
expires: "2027-01-01"
```

After the date the suppression stops applying and the run reports why:

```
  suppression for bib/metadata-drift expired on 2027-01-01; the finding is
  reported again
```

Useful for *"not before the camera-ready"* rather than *"never."*

## Disabling rules

```yaml
rules:
  stats/grim: off
```

A disabled rule is **reported as skipped**, not silently omitted:

```
  skipped: 1 rule, disabled in .resint.yml
```

Disable a rule when it does not apply to your work — `stats/grim` on a paper
with no integer-scale responses, for instance. Reach for suppression instead
when the rule applies but this particular finding is accounted for.

## Command line

```console
resint check paper.tex [--repo PATH]

  --bib PATH              bibliography (default: the .bib beside the source)
  --format term|json|sarif
  --min-severity low|med|high
  --fail-on high|med|low|none      lowest severity that exits non-zero
  --offline                        skip reference lookups entirely
  --mailto you@example.org         Crossref polite pool
  --config PATH                    use a specific .resint.yml
  --no-config                      ignore any .resint.yml
```

```console
resint rules [--tier deterministic|model-assisted] [--family stats] [--format json]
resint init [PATH] [--force]
```

### Exit codes

| | |
|---|---|
| `0` | Nothing at or above `--fail-on` |
| `1` | Findings at or above `--fail-on` |
| `2` | Usage error — missing file, bad config |

`--fail-on` defaults to `high`, so CI fails on the findings that change what a
paper claims and stays quiet about the rest.

## In CI

```yaml
- run: pipx install resint
- run: resint check paper.tex --repo . --format sarif > resint.sarif
- uses: github/codeql-action/upload-sarif@v3
  with:
    sarif_file: resint.sarif
```

Findings become annotations on the pull request, and each rule's
`cannot_detect` rides along in the SARIF help text.
