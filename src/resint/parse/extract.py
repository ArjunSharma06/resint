"""Pulling statistics and means out of normalized text.

Extraction is where false positives are born. A rule can only be as precise
as what it is handed, so the discipline here is to abstain loudly rather than
pair things up hopefully: a mean is only reported when exactly one sample
size sits in the same sentence, and anything skipped is recorded in
``unchecked`` so the report can say what it could not see.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..ir.paper import Number, ReportedMean, StatTest
from ..ir.span import Source, Span
from .latex import Normalized

# Abbreviations that must not end a sentence.
_ABBREV = (
    "et al.", "e.g.", "i.e.", "cf.", "vs.", "Fig.", "Eq.", "Sec.", "Tab.",
    "approx.", "resp.", "Dr.", "Prof.", "St.", "No.", "pp.",
)

_SENT_END = re.compile(r"(?<=[.!?])[\s\n]+(?=[A-Z(\[])")

# A gap that may contain decimal points but not a sentence boundary.
_GAP = r"(?:[^.]|\.(?=\d)){0,70}?"

_STAT = re.compile(
    r"(?<![A-Za-z])"
    r"(?P<kind>chi\s*\^?\s*2|chi2|X\s*\^?\s*2|t|F|r|z)"
    r"\s*"
    r"(?:\(\s*(?P<df1>\d+(?:\.\d+)?)\s*"
    r"(?:,\s*(?:[Nn]\s*=\s*\d+|(?P<df2>\d+(?:\.\d+)?))\s*)?\)\s*)?"
    r"(?P<scomp>[=<>~])\s*"
    r"(?P<stat>-?\d*[.,]\d+|-?\d+)"
    + _GAP +
    r"(?<![A-Za-z])p\s*"
    r"(?P<pcomp><=|>=|[=<>])\s*"
    r"(?P<p>\d*[.,]\d+|\d+(?:[.,]\d+)?)",
    re.IGNORECASE | re.DOTALL,
)

_MEAN = re.compile(
    r"(?<![A-Za-z])(?:M|Mean|mean)\s*(?:=|of|was|:)\s*(?P<mean>\d+[.,]\d+|\d+)",
)
_N = re.compile(r"(?<![A-Za-z])[Nn]\s*=\s*(?P<n>\d+)")
_ITEMS = re.compile(r"(?P<items>\d+)[\s-]*(?:items|item)(?![A-Za-z])", re.IGNORECASE)
_ONE_TAILED = re.compile(r"one[\s-]?(?:tail|sid)|1[\s-]?tail", re.IGNORECASE)

_KIND_MAP = {"t": "t", "f": "F", "r": "r", "z": "z"}


def _kind_of(token: str) -> str:
    flat = token.replace(" ", "").replace("^", "").lower()
    if flat in ("chi2", "x2"):
        return "chi2"
    return _KIND_MAP[flat]


def sentences(text: str) -> list[tuple[int, int]]:
    """Sentence spans over normalized text, protecting common abbreviations."""
    guard = text
    for abbr in _ABBREV:
        guard = guard.replace(abbr, abbr[:-1] + "\0")

    bounds, last = [], 0
    for m in _SENT_END.finditer(guard):
        bounds.append((last, m.start()))
        last = m.end()
    if last < len(text):
        bounds.append((last, len(text)))
    return bounds


def _span(doc: Normalized, src: Source, start: int, end: int, label: str) -> Span:
    lo, hi = doc.raw_range(start, end)
    return doc.anchor(src, lo, hi, label)


def _enclosing_sentence(bounds: list[tuple[int, int]], index: int) -> tuple[int, int]:
    for start, end in bounds:
        if start <= index < end:
            return start, end
    return index, index


def extract_stats(doc: Normalized, src: Source) -> list[StatTest]:
    """Every reported test where statistic and p-value appear together."""
    out: list[StatTest] = []
    bounds = sentences(doc.text)

    for m in _STAT.finditer(doc.text):
        kind = _kind_of(m.group("kind"))
        df1 = float(m.group("df1")) if m.group("df1") else None
        df2 = float(m.group("df2")) if m.group("df2") else None

        if kind in ("t", "chi2", "r") and df1 is None:
            continue  # unusable without degrees of freedom
        if kind == "F" and (df1 is None or df2 is None):
            continue

        # Tail must be read from the sentence that reports the test, never a
        # character window: a fixed window reaches into neighbouring sentences,
        # and one stray "one-tailed" elsewhere in the paragraph silently halves
        # an unrelated p-value into a false finding.
        s_start, s_end = _enclosing_sentence(bounds, m.start())
        tail = 1 if _ONE_TAILED.search(doc.text[s_start:s_end]) else 2
        section = doc.section_at(m.start())

        out.append(
            StatTest(
                kind=kind,
                statistic_raw=m.group("stat"),
                df1=df1,
                df2=df2,
                p_raw=m.group("p"),
                p_comparator={"<=": "<", ">=": ">"}.get(
                    m.group("pcomp"), m.group("pcomp")
                ),
                tail=tail,
                context=section,
                span=_span(doc, src, m.start(), m.end("stat"), section),
                p_span=_span(doc, src, m.start("p"), m.end("p"), section),
            )
        )
    return out


@dataclass(frozen=True, slots=True)
class MeanExtraction:
    means: list[ReportedMean]
    unchecked: list[str]


def _sample_sizes(text: str) -> list[tuple[int, int]]:
    """Every stated sample size in a stretch of text, with its offset."""
    return [(m.start("n"), int(m.group("n"))) for m in _N.finditer(text)]


def _resolve_n(doc, s_start: int, s_end: int, sentence: str):
    """The sample size a mean in this sentence belongs to, and how sure we are.

    Papers put N in Participants, or a table header, or a group label -- almost
    never beside the mean. Requiring both in one sentence found means in 1 of
    148 real papers, so GRIM never ran.

    Widening the search is only safe while the answer stays unambiguous, so
    each step out requires exactly one candidate. Two sample sizes in a
    section is a study with groups, and picking one would be a guess -- which
    is how a confident, wrong GRIM finding gets made.
    """
    local = _sample_sizes(sentence)
    if len(local) == 1:
        offset, value = local[0]
        return value, s_start + offset, "sentence"
    if len(local) > 1:
        return None, None, f"{len(local)} sample sizes"

    section = doc.section_bounds_at(s_start)
    if section:
        lo, hi = section
        found = _sample_sizes(doc.text[lo:hi])
        if len(found) == 1:
            offset, value = found[0]
            return value, lo + offset, "section"
        if len(found) > 1:
            return None, None, f"{len(found)} sample sizes in this section"

    whole = _sample_sizes(doc.text)
    if len(whole) == 1:
        offset, value = whole[0]
        return value, offset, "document"
    if len(whole) > 1:
        return None, None, f"{len(whole)} sample sizes in the paper"
    return None, None, "no sample size"


def extract_means(doc: Normalized, src: Source) -> MeanExtraction:
    """Means paired with an unambiguous sample size in the same sentence.

    Pairing a mean with the wrong N produces a confident, wrong GRIM finding
    -- the worst failure this tool has. So a mean is only reported when its
    sentence contains exactly one sample size. Zero or several and it is
    skipped, with the reason recorded.
    """
    means: list[ReportedMean] = []
    unchecked: list[str] = []

    for s_start, s_end in sentences(doc.text):
        sentence = doc.text[s_start:s_end]
        found_means = list(_MEAN.finditer(sentence))
        if not found_means:
            continue

        n_value, n_at, how = _resolve_n(doc, s_start, s_end, sentence)
        if n_value is None:
            line = doc.line_of(doc.raw_offset(s_start + found_means[0].start()))
            unchecked.append(f"mean at line {line} not checked: {how}")
            continue

        item_match = _ITEMS.search(sentence)
        items = int(item_match.group("items")) if item_match else 1
        section = doc.section_at(s_start)

        n_span = _span(
            doc, src, n_at, n_at + len(str(n_value)), section
        )

        for mm in found_means:
            means.append(
                ReportedMean(
                    raw=mm.group("mean"),
                    n=n_value,
                    items=items,
                    items_inferred=item_match is None,
                    context=section,
                    n_source=how,
                    span=_span(
                        doc, src,
                        s_start + mm.start("mean"),
                        s_start + mm.end("mean"),
                        section,
                    ),
                    n_span=n_span,
                )
            )

    return MeanExtraction(means=means, unchecked=unchecked)


# --- labelled numbers ---------------------------------------------------
#
# Matching a prose value to a table cell needs to know what quantity the
# prose is talking about. Rather than shipping a fixed vocabulary of metric
# names -- which would work for one field and fail everywhere else -- the
# paper supplies its own: table headers name the quantities that table
# reports, and those terms are what we look for in the surrounding text.

_BASE_METRICS = (
    "accuracy", "precision", "recall", "f1", "f-1", "auc", "auroc", "map",
    "bleu", "rouge", "meteor", "perplexity", "wer", "cer", "iou", "dice",
    "mse", "rmse", "mae", "error rate", "top-1", "top-5", "exact match",
)

_NUM = r"[-+]?\d+(?:\.\d+)?"

# Header words that name rows rather than quantities. A results table's first
# column is almost always one of these, and treating "Method" as a metric lets
# "our method reaches 94.2" outcompete "94.2% accuracy" for the same digits.
NON_METRIC_HEADERS = frozenset(
    {
        "method", "methods", "model", "models", "approach", "system", "variant",
        "dataset", "datasets", "data", "split", "name", "setting", "config",
        "configuration", "task", "benchmark", "baseline", "ours", "id", "run",
        "condition", "group", "type", "size", "version", "ablation", "row",
    }
)


def usable_labels(headers) -> list[str]:
    """Header terms that plausibly name a measured quantity."""
    return [
        h
        for h in headers
        if h.strip() and h.strip().lower() not in NON_METRIC_HEADERS
    ]


def _label_pattern(labels: list[str]) -> re.Pattern | None:
    """One alternation over every quantity name this paper uses."""
    cleaned = sorted(
        {re.escape(l.strip().lower()) for l in labels if 2 <= len(l.strip()) <= 40},
        key=len,
        reverse=True,
    )
    if not cleaned:
        return None
    names = "|".join(cleaned)
    return re.compile(
        rf"(?:(?P<label_a>{names})\s*"
        rf"(?:of|was|were|is|are|reaches|reached|achieves|achieved|at|=|:)?\s*"
        rf"(?P<value_a>{_NUM})\s*%?)"
        rf"|(?:(?P<value_b>{_NUM})\s*%?\s*(?P<label_b>{names}))",
        re.IGNORECASE,
    )


def extract_labeled_numbers(
    doc: Normalized, src: Source, labels: list[str]
) -> list[Number]:
    """Numbers in prose that name the quantity they report."""
    pattern = _label_pattern(list(labels) + list(_BASE_METRICS))
    if pattern is None:
        return []

    out: list[Number] = []
    for m in pattern.finditer(doc.text):
        label = (m.group("label_a") or m.group("label_b") or "").strip().lower()
        raw = m.group("value_a") or m.group("value_b")
        group = "value_a" if m.group("value_a") else "value_b"
        section = doc.section_at(m.start())
        out.append(
            Number(
                raw=raw,
                label=label,
                section=section,
                span=_span(doc, src, m.start(group), m.end(group), section or "text"),
            )
        )
    return out


# --- hyperparameters ----------------------------------------------------
#
# The paper side of repro/hparam-drift. Values are kept as written, including
# scientific notation, because "3e-4" and "0.0003" are the same number but
# only one of them is what the author will search for when checking.

# No bare trailing point: "rank is 8." must yield 8, not "8.".
_HP_NUM = r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?"

_HP_NAMES = (
    "learning rate", "learning-rate", "lr",
    "batch size", "batch-size", "batchsize",
    "weight decay", "weight-decay",
    "dropout", "temperature", "momentum",
    "epochs", "num epochs", "number of epochs",
    "warmup steps", "warmup",
    "hidden size", "hidden dimension", "embedding size",
    "num layers", "number of layers", "depth",
    "num heads", "number of heads", "attention heads",
    "rank", "lora rank", "alpha",
    "max length", "sequence length", "context length",
    "gradient accumulation", "accumulation steps",
    "beam size", "top-k", "top-p",
)

_HP = re.compile(
    r"(?:(?P<name_a>" + "|".join(re.escape(n) for n in _HP_NAMES) + r")"
    # A bounded run of connectives, so "was set to" and "is equal to"
    # both reach the number without letting the match wander a sentence.
    r"(?:\s*(?:of|was|were|is|are|to|at|as|set|equal|fixed|kept|chosen|:|=|,)?){0,3}\s*"
    r"(?P<value_a>" + _HP_NUM + r"))"
    r"|(?:(?P<value_b>" + _HP_NUM + r")\s*(?P<name_b>"
    + "|".join(re.escape(n) for n in _HP_NAMES) + r"))",
    re.IGNORECASE,
)

# "trained for 100 epochs" reads naturally but puts the number first, and a
# bare "100 epochs" is the most common phrasing of all.
_HP_EPOCHS = re.compile(
    r"(?:for|over)?\s*(?P<value>\d+)\s*(?:training\s+)?epochs?\b", re.IGNORECASE
)


def extract_hyperparameters(doc: Normalized, src: Source) -> list[Number]:
    """Hyperparameters stated in prose, with the value as written."""
    out: list[Number] = []
    seen: set[tuple[str, int]] = set()

    for m in _HP.finditer(doc.text):
        name = (m.group("name_a") or m.group("name_b") or "").strip().lower()
        raw = m.group("value_a") or m.group("value_b")
        group = "value_a" if m.group("value_a") else "value_b"
        key = (name, m.start(group))
        if key in seen:
            continue
        seen.add(key)
        section = doc.section_at(m.start())
        out.append(
            Number(
                raw=raw,
                label=name,
                section=section,
                span=_span(doc, src, m.start(group), m.end(group), section or "text"),
            )
        )

    for m in _HP_EPOCHS.finditer(doc.text):
        key = ("epochs", m.start("value"))
        if key in seen:
            continue
        seen.add(key)
        section = doc.section_at(m.start())
        out.append(
            Number(
                raw=m.group("value"),
                label="epochs",
                section=section,
                span=_span(
                    doc, src, m.start("value"), m.end("value"), section or "text"
                ),
            )
        )

    return out
