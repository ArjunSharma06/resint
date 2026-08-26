# Field reports

Real inconsistencies people report catching in their own papers, and whether
resint could catch them. Source material for the remaining rules.

Each entry records what is *actually detectable from the text*, which is
usually narrower than the story. The gap is the `unchecked:` line.

---

## From r/PhD thread (2026-08-25)

### Statistical test interpreted backward

> "I once realized my entire third chapter was interpreting the results of a
> statistical test backward, like the conclusion was the exact opposite of
> what the numbers actually said and I'd been polishing the prose for weeks."

**Category:** interpretation, not arithmetic. First report of this kind.

**Catchable:** a reported p-value whose polarity disagrees with the
significance claim in the same sentence. p = 0.03 next to "no significant
difference", or p = 0.4 next to "significantly outperforms".

**Not catchable:** a misreading that the whole chapter is then written to be
consistent with. Prose agrees with itself; the error is between the data and
the paper. Nothing to compare against unless the raw numbers are also stated.

**Candidate rule:** `stats/claim-polarity`

- Reuses statistic parsing from `stats/pvalue-mismatch`
- Reuses proximity association from `numbers/internal-mismatch`
- Abstains unless: exactly one p-value in the sentence, alpha known or
  defaulted (and the default stated in the finding), and the claim polarity
  is unambiguous. Hedged claims ("may suggest", "appears to") are not claims.

**Follow up:** did they catch it themselves, or did someone else?

---

## Open questions this file should answer

- Are reports mostly numbers-vs-numbers (covered) or interpretation (not)?
- Which of the five unimplemented rules does anyone actually mention?
- What is the modal catcher: author, advisor, reviewer, or post-publication?
