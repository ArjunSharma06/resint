---
name: False positive
about: A rule fired on something that is actually correct
labels: false-positive
---

**Rule:** `family/name`

**The source that triggered it**

```latex
paste the smallest snippet that reproduces it
```

**Why it is correct as written**

<!-- What did the rule miss? This becomes the rule's negative fixture. -->

**resint version:** `resint --version`

---

False positives are the most valuable issue in this project. Each one becomes
a negative fixture, which is how the precision floor rises.
