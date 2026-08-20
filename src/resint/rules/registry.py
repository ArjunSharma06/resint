"""The rule registry, and the context gate that keeps laziness honest.

A rule declares what it needs. The context hands it exactly that and nothing
more -- reaching past the declaration is an AttributeError rather than a
silent dependency. Two things fall out of that. The engine can build only the
IR slices some rule actually asked for, so a deterministic-only run never
constructs a provider client; and a rule's tests only have to stand up the
slice it declared, which is what keeps rule 47 cheap to contribute.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterator, Sequence

from ..ir.finding import Finding, Severity, Tier

RuleFn = Callable[["Context"], Iterator[Finding]]


class RuleDefinitionError(ValueError):
    """Raised when a rule is declared without meeting the contributor bar."""


class UndeclaredAccess(AttributeError):
    """Raised when a rule touches IR it did not declare in ``requires``."""


@dataclass(frozen=True, slots=True)
class Rule:
    id: str
    severity: Severity
    tier: Tier
    requires: tuple[str, ...]
    cannot_detect: str
    fn: RuleFn

    @property
    def family(self) -> str:
        return self.id.split("/", 1)[0]

    @property
    def needs_repo(self) -> bool:
        return any(r.startswith("repo.") for r in self.requires)

    def run(self, ctx: "Context") -> list[Finding]:
        gated = ctx.gated_for(self)
        out: list[Finding] = []
        for finding in self.fn(gated) or ():
            if finding.rule_id != self.id:
                raise RuleDefinitionError(
                    f"{self.id} emitted a finding tagged {finding.rule_id!r}"
                )
            out.append(finding)
        return out


class _Slice:
    """Attribute-gated view over one IR object."""

    __slots__ = ("_target", "_allowed", "_name", "_rule_id")

    def __init__(self, target: object, allowed: frozenset[str], name: str, rule_id: str):
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_allowed", allowed)
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_rule_id", rule_id)

    def __getattr__(self, item: str):
        if item.startswith("_"):
            raise AttributeError(item)
        if item not in self._allowed:
            raise UndeclaredAccess(
                f"{self._rule_id} accessed {self._name}.{item} without declaring it. "
                f"Add \"{self._name}.{item}\" to requires=."
            )
        return getattr(self._target, item)

    def __setattr__(self, *_: object) -> None:
        raise AttributeError("the IR is read-only inside a rule")


@dataclass
class Context:
    """What a rule is handed. Ungated until :meth:`gated_for` narrows it."""

    paper: object = None
    repo: object = None
    rule: "Rule | None" = None
    abstentions: list[str] = field(default_factory=list)

    def abstain(self, reason: str) -> None:
        """Record that this rule declined to check something, and why.

        A first-class operation rather than a write into the IR. Rules read;
        the engine collects. Abstentions surface in the report next to
        findings because "did not look" and "found nothing" are different
        statements, and a rule that goes quiet without saying why is
        indistinguishable from one that passed.
        """
        if self.rule is None:
            raise RuleDefinitionError("ctx.abstain() is only available inside a rule")
        self.abstentions.append(f"{self.rule.id}: {reason}")

    def finding(
        self,
        *,
        message: str,
        anchors: Sequence,
        severity: Severity | str | None = None,
        absent_from: str | None = None,
        fix: str | None = None,
        affects: Sequence[str] = (),
        confidence: float | None = None,
    ) -> Finding:
        """Build a finding attributed to the running rule.

        ``rule_id`` and ``tier`` come from the declaration, so a rule cannot
        mislabel its own output or claim to be deterministic when it is not.
        ``severity`` defaults to the declared level but may be overridden per
        finding -- a p-value mismatch that flips a significance decision is a
        different animal from one in the fourth decimal place.
        """
        if self.rule is None:
            raise RuleDefinitionError("ctx.finding() is only available inside a rule")
        return Finding(
            rule_id=self.rule.id,
            severity=self.rule.severity if severity is None else severity,
            tier=self.rule.tier,
            message=message,
            anchors=anchors,
            absent_from=absent_from,
            fix=fix,
            affects=affects,
            confidence=confidence,
        )

    def gated_for(self, rule: Rule) -> "Context":
        allowed: dict[str, set[str]] = {}
        for req in rule.requires:
            root, _, attr = req.partition(".")
            allowed.setdefault(root, set()).add(attr)

        for root in allowed:
            if getattr(self, root, None) is None:
                raise RuleDefinitionError(
                    f"{rule.id} requires {root!r}, which this run does not have"
                )

        return Context(
            paper=(
                _Slice(self.paper, frozenset(allowed["paper"]), "paper", rule.id)
                if "paper" in allowed
                else None
            ),
            repo=(
                _Slice(self.repo, frozenset(allowed["repo"]), "repo", rule.id)
                if "repo" in allowed
                else None
            ),
            rule=rule,
            abstentions=self.abstentions,
        )


class Registry:
    def __init__(self) -> None:
        self._rules: dict[str, Rule] = {}

    def add(self, rule: Rule) -> None:
        if rule.id in self._rules:
            raise RuleDefinitionError(f"duplicate rule id {rule.id!r}")
        self._rules[rule.id] = rule

    def get(self, rule_id: str) -> Rule:
        return self._rules[rule_id]

    def all(self) -> list[Rule]:
        return sorted(self._rules.values(), key=lambda r: r.id)

    def by_tier(self, tier: Tier) -> list[Rule]:
        return [r for r in self.all() if r.tier == tier]

    def required_slices(self, rules: Sequence[Rule]) -> set[str]:
        """Union of everything the given rules declared -- what the engine builds."""
        return {req for rule in rules for req in rule.requires}

    def __len__(self) -> int:
        return len(self._rules)

    def __contains__(self, rule_id: object) -> bool:
        return rule_id in self._rules


REGISTRY = Registry()


def rule(
    *,
    id: str,
    severity: Severity | str,
    tier: Tier | str,
    requires: Sequence[str],
    cannot_detect: str,
    registry: Registry | None = None,
) -> Callable[[RuleFn], RuleFn]:
    """Declare a rule.

    ``cannot_detect`` is mandatory and load-bearing: it is surfaced by
    ``resint rules`` and by ``--explain``. A rule that states its blind spots
    is trustworthy; one that implies completeness is not.
    """
    if "/" not in id:
        raise RuleDefinitionError(f"rule id {id!r} must be namespaced, e.g. 'stats/grim'")
    if not requires:
        raise RuleDefinitionError(f"{id}: requires= must not be empty")
    if not cannot_detect or not cannot_detect.strip():
        raise RuleDefinitionError(
            f"{id}: cannot_detect= is mandatory. State what this rule will miss."
        )
    for req in requires:
        root, sep, attr = req.partition(".")
        if not sep or root not in {"paper", "repo"} or not attr:
            raise RuleDefinitionError(
                f"{id}: requires entry {req!r} must be 'paper.<attr>' or 'repo.<attr>'"
            )

    target = registry if registry is not None else REGISTRY

    def decorate(fn: RuleFn) -> RuleFn:
        target.add(
            Rule(
                id=id,
                severity=Severity(severity),
                tier=Tier(tier),
                requires=tuple(requires),
                cannot_detect=cannot_detect.strip(),
                fn=fn,
            )
        )
        fn.rule_id = id  # type: ignore[attr-defined]
        return fn

    return decorate
