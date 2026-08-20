# Writing a rule

How rules work, and the bar they have to clear. This is the design record for
the rule system — read it before adding a check, and read it if you want to
understand why the engine refuses things it refuses.

A rule is one file, self-contained, with fixtures. If you can read this page
and land rule number fourteen in an afternoon, the design works.

```console
git clone https://github.com/ArjunSharma06/resint && cd resint
pip install -e ".[dev]"
pytest
```

No dependencies beyond the standard library, and none in a rule either.
If a rule needs a package, open an issue first — the install staying light is
a feature people notice.

## Writing a rule

Rules live in `src/resint/rules/<family>/<name>.py`. Discovery walks the
family directories, so adding a file is the only step — there is no central
list to remember to edit.

```python
from ..registry import Context, rule


@rule(
    id="numbers/internal-mismatch",
    severity="high",
    tier="deterministic",
    requires=["paper.numbers", "paper.tables"],
    cannot_detect=(
        "Values that differ because they describe genuinely different "
        "quantities sharing a column heading."
    ),
)
def check(ctx: Context):
    for number in ctx.paper.numbers:
        ...
        yield ctx.finding(
            message=f"{where} reports {label} of {number.raw}, but ...",
            anchors=[number.span, cell.span],
            fix="Reconcile the two values.",
        )
```

### Four things the engine enforces

**Two anchors, minimum.** A `Finding` with fewer raises at construction. One
anchor is an assertion the reader has to go and check; two make it a
comparison they can verify by reading the finding itself. That difference is
the product.

The single exception is an *absence* finding, where half the comparison is a
negative with no location — a key with no entry, an entry never cited. Pass
`absent_from=` naming exactly what you searched:

```python
yield ctx.finding(
    message=f"[{key}] is cited but has no entry in {bib}.",
    anchors=[use_site.span],
    absent_from=bib,
)
```

**Declared requirements only.** `ctx` exposes exactly what `requires` lists.
Reaching past it is an `AttributeError`. This keeps the engine lazy — a run
whose rules never declare `paper.resolutions` never opens a socket — and it
keeps your tests small, because you only stand up the slice you asked for.

**`cannot_detect` is mandatory.** It appears in `resint rules`, in JSON, and
in SARIF help text. Write it honestly and specifically: *"theses and
technical reports are often absent from these indices legitimately"* is
useful; *"some edge cases"* is not. A rule that knows its own limits is
trustworthy; one that implies completeness is not.

**Severity is a default, not a constant.** Escalate or reduce per finding via
`ctx.finding(severity=...)`. A *p*-value off in the fourth decimal is a typo;
one that flips a significance decision changes what the paper claims.

### The governing rule

**A model may extract structure. It may never render a verdict that code
could compute.**

```python
# No.  An opinion. Sometimes wrong. Unfalsifiable.
"Are the abstract's numbers consistent with Table 3?"

# Yes. The model reads fuzzy prose; the comparison is arithmetic.
"Which numbers in this abstract are reported results?"  →  then compare
```

Rules that follow this stay deterministic *at the point of judgement*, which
is where it matters.

## The quality bar

CI enforces items 1–3 mechanically, so nobody argues about quality in review.

1. **Positive fixtures.** At least one paper in `corpus/` where the rule
   fires, with the defect commented at the point it occurs.

2. **At least three negative fixtures** — papers where the rule must stay
   silent. Include the near misses: the value that *is* reachable, the
   citation that is genuinely obscure rather than invented, the table that
   cannot be parsed. Those are where a rule is tempted to guess.

   Most projects skip negatives. Do not. The first real false positive here —
   a stray "one-tailed" halving the *p*-value of a statistic in a
   *neighbouring* sentence — was invisible to every unit test and was caught
   by `corpus/clean/` on the first run.

3. **A precision floor**, asserted in your test module:

   ```python
   PRECISION_FLOOR = 1.0

   def test_measured_precision_meets_the_declared_floor():
       ...
       assert precision >= PRECISION_FLOOR
   ```

4. **`cannot_detect` written honestly.** Reviewed by a human — the one part
   of the bar that is a judgement call.

5. **An actionable message.** Not *"inconsistency found"* but what is wrong,
   why it matters, and what to do. Findings are read by people under deadline
   pressure.

### When in doubt, abstain

A rule that misses one case in five is fine. A rule that is wrong one time in
twenty is a liability. This tool tells people something is wrong with their
published work; a single viral false accusation costs more trust than a
hundred correct findings earn.

Concretely, prefer:

- Skipping an ambiguous case and appending to `paper.unchecked`
- Reporting a *narrower* claim you can prove over a broader one you cannot
- Reducing severity when an input had to be inferred rather than read

## Testing

```console
pytest                     # everything, under a second
pytest tests/test_grim.py  # one rule
```

**No test may touch the network.** Resolution goes through a protocol; use
`StaticResolver` with a fixed table of answers. A suite that depends on
Crossref being up is flaky in exactly the way that erodes trust in the tool.

Line numbers in `corpus/` are load-bearing — `tests/test_pipeline.py` asserts
that findings anchor to the exact source line, so an offset-map regression
fails the suite rather than silently mislocating every finding.

## Code style

Match the surrounding code. Two things that are not negotiable:

- **Comments explain why, not what.** The interesting comments in this
  codebase record a decision and its consequence — why UNKNOWN can never
  become a finding, why the tail must be read from one sentence. Those are
  the comments that stop someone reintroducing a bug.
- **Docstrings on modules and rules** state what the thing is *for* and what
  it deliberately does not do.

## Reporting a false positive

The most valuable issue you can open. Include the source that triggered it
and what the correct behaviour would be. False positives become negative
fixtures, which is how the precision floor rises.
