# The fixture corpus

Every rule ships with fixtures here, and CI measures precision against them.
One directory per fixture, self-contained — a fixture must not pick up a
bibliography from a sibling, because a rule's behaviour should depend only on
what its own fixture says.

```
corpus/
  <name>/
    paper.tex      the source under test
    refs.bib       optional; auto-discovered when present
```

## The two kinds

**Positive** fixtures contain planted defects. Every defect is commented at
the point it occurs, and asserted by name in `tests/test_pipeline.py`.

**Negative** fixtures are internally consistent and must produce **zero**
findings. These carry the heavier guarantee. Catching a planted defect shows
a rule works; staying silent on correct work shows it can be trusted, and
only the second one determines whether anybody keeps using this.

Most projects skip negative fixtures. Do not. The first real false positive
in this project — a stray "one-tailed" halving the p-value of a statistic in
a neighbouring sentence — was invisible to every unit test and was caught by
`clean/paper.tex` on the first run.

## Adding one

A rule pull request needs at least one positive fixture and **three**
negatives. Negatives should include the near misses: the value that is
reachable, the citation that is genuinely obscure rather than invented, the
table the extractor cannot read. Those are the cases where a rule is tempted
to guess.

Line numbers in these files are load-bearing. `tests/test_pipeline.py`
asserts that findings anchor to the exact source line, so an offset-map
regression fails the suite rather than silently mislocating every finding.
