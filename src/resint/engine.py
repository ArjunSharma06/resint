"""Rule selection and execution.

Selection is the interesting part. A rule is skipped -- silently, and without
being counted as passing -- whenever the run cannot satisfy what it declared:
no repository for a repo rule, no provider for a model-assisted one. The
report says which rules did not run and why, because "no findings" and "did
not look" are different statements and conflating them is how a checker
becomes misleading.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import Config
from .ir.finding import Finding, Severity, Tier
from .rules import REGISTRY, Context, Registry, Rule, load_all

_SEVERITY_RANK = {Severity.HIGH: 0, Severity.MED: 1, Severity.LOW: 2}


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)
    unchecked: list[str] = field(default_factory=list)
    ran: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def used_model(self) -> bool:
        return any(f.tier is Tier.MODEL_ASSISTED for f in self.findings)

    def counts(self) -> dict[str, int]:
        out = {"high": 0, "med": 0, "low": 0}
        for f in self.findings:
            if not f.suppressed:
                out[f.severity.value] += 1
        return out

    def summary(self) -> str:
        c = self.counts()
        total = sum(c.values())
        noun = "finding" if total == 1 else "findings"
        return (
            f"{total} {noun} "
            f"({c['high']} high, {c['med']} med, {c['low']} low)"
        )


def selectable(
    rules: list[Rule], *, has_repo: bool, has_provider: bool
) -> tuple[list[Rule], dict[str, str]]:
    """Split rules into those this run can honestly execute, and why not."""
    runnable, skipped = [], {}
    for r in rules:
        if r.needs_repo and not has_repo:
            skipped[r.id] = "no repository supplied"
        elif r.tier is Tier.MODEL_ASSISTED and not has_provider:
            skipped[r.id] = "no model provider configured"
        else:
            runnable.append(r)
    return runnable, skipped


def required_slices(rules: list[Rule]) -> set[str]:
    return {req for r in rules for req in r.requires}


@dataclass
class Plan:
    """Which rules will run, why the rest will not, and what data they need.

    Selection has to happen *before* the paper is built, not after. Otherwise
    the loader has no idea what to load and defaults to everything -- which
    means a run with no bibliography rules still parses the bibliography and
    still opens sockets. Computing the plan first makes the laziness the
    architecture already describes actually true, and it stops selection
    drifting between the call site that loads data and the one that runs
    rules.
    """

    runnable: list[Rule] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)

    @property
    def paper_slices(self) -> set[str]:
        return {r for r in required_slices(self.runnable) if r.startswith("paper.")}

    @property
    def repo_slices(self) -> set[str]:
        return {r for r in required_slices(self.runnable) if r.startswith("repo.")}

    @property
    def opens_network(self) -> bool:
        """Whether anything in this plan can reach the network."""
        return "paper.resolutions" in self.paper_slices


def plan(
    registry: Registry | None = None,
    config: Config | None = None,
    *,
    has_repo: bool = False,
    has_provider: bool = False,
) -> Plan:
    """Decide what will run, before any data is loaded."""
    reg = registry or load_all()
    settings = config or Config()

    candidates = [r for r in reg.all() if settings.enabled(r.id)]

    # Opt-in rules stay out unless the config names them. They are skipped
    # visibly, like everything else: a rule that silently did not run is
    # indistinguishable from one that ran and found nothing, which is the
    # confusion this whole tool exists to avoid.
    opted_out = [
        r for r in candidates
        if r.opt_in and r.id not in settings.enabled_explicitly
    ]
    candidates = [r for r in candidates if r not in opted_out]
    runnable, skipped = selectable(
        candidates, has_repo=has_repo, has_provider=has_provider
    )
    for rule in sorted(opted_out, key=lambda r: r.id):
        skipped[rule.id] = "off by default; switch on in .resint.yml to use it"

    for disabled in sorted(settings.disabled):
        if disabled in reg:
            skipped[disabled] = "disabled in .resint.yml"

    return Plan(runnable=runnable, skipped=skipped)


def run(
    paper,
    repo=None,
    *,
    registry: Registry | None = None,
    has_provider: bool = False,
    min_severity: Severity | None = None,
    config: Config | None = None,
    prepared: Plan | None = None,
    model=None,
) -> Report:
    settings = config or Config()
    chosen = prepared or plan(
        registry,
        settings,
        has_repo=repo is not None,
        has_provider=has_provider or model is not None,
    )
    runnable, skipped = chosen.runnable, dict(chosen.skipped)

    report = Report(skipped=skipped, ran=[r.id for r in runnable])
    report.unchecked = list(getattr(paper, "unchecked", []))

    if repo is not None:
        report.unchecked.extend(getattr(repo, "unchecked", []))

    ctx = Context(paper=paper, repo=repo, model=model)
    for rule in runnable:
        report.findings.extend(rule.run(ctx))
    report.unchecked.extend(ctx.abstentions)

    # Suppression is a reporting concern, applied after every rule has run:
    # a suppressed finding still exists, is still counted in JSON, and can
    # therefore never hide a regression from the corpus.
    report.findings, notes = settings.apply(report.findings)
    report.notes.extend(notes)

    # Inline directives, after the config file. A judgement written beside the
    # line it excuses is the most specific statement available about that
    # finding, so it is applied last and wins.
    inline = getattr(paper, "inline_suppressions", None) or ()
    if inline:
        from .parse.inline import apply_inline

        report.findings, inline_notes = apply_inline(report.findings, inline)
        report.notes.extend(inline_notes)

    if min_severity is not None:
        floor = _SEVERITY_RANK[min_severity]
        report.findings = [
            f for f in report.findings if _SEVERITY_RANK[f.severity] <= floor
        ]

    report.findings.sort(
        key=lambda f: (_SEVERITY_RANK[f.severity], f.rule_id, f.anchors[0].start)
    )
    return report
