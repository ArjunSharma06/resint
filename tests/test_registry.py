"""The declaration gate: rules get exactly what they asked for."""

from types import SimpleNamespace

import pytest

from resint.ir.finding import Severity, Tier
from resint.rules.registry import (
    Context,
    Registry,
    RuleDefinitionError,
    UndeclaredAccess,
    rule,
)

from conftest import span


@pytest.fixture
def reg():
    return Registry()


def test_declared_access_allowed(reg):
    @rule(id="t/ok", severity="low", tier="deterministic",
          requires=["paper.numbers"], cannot_detect="nothing", registry=reg)
    def _r(ctx):
        assert ctx.paper.numbers == [1]
        return iter(())

    reg.get("t/ok").run(Context(paper=SimpleNamespace(numbers=[1], tables=[2])))


def test_undeclared_access_raises(reg):
    @rule(id="t/leak", severity="low", tier="deterministic",
          requires=["paper.numbers"], cannot_detect="nothing", registry=reg)
    def _r(ctx):
        return iter([ctx.paper.tables])

    with pytest.raises(UndeclaredAccess, match=r'Add "paper.tables" to requires'):
        list(reg.get("t/leak").run(Context(paper=SimpleNamespace(numbers=[], tables=[]))))


def test_missing_repo_is_a_definition_error(reg):
    @rule(id="t/repo", severity="low", tier="deterministic",
          requires=["repo.configs"], cannot_detect="nothing", registry=reg)
    def _r(ctx):
        return iter(())

    with pytest.raises(RuleDefinitionError, match="requires 'repo'"):
        reg.get("t/repo").run(Context(paper=SimpleNamespace()))


def test_cannot_detect_is_mandatory(reg):
    with pytest.raises(RuleDefinitionError, match="cannot_detect"):
        @rule(id="t/blank", severity="low", tier="deterministic",
              requires=["paper.numbers"], cannot_detect="   ", registry=reg)
        def _r(ctx):
            return iter(())


def test_requires_must_be_namespaced(reg):
    with pytest.raises(RuleDefinitionError, match="must be 'paper."):
        @rule(id="t/bare", severity="low", tier="deterministic",
              requires=["numbers"], cannot_detect="x", registry=reg)
        def _r(ctx):
            return iter(())


def test_rule_id_must_have_a_family(reg):
    with pytest.raises(RuleDefinitionError, match="namespaced"):
        @rule(id="bare", severity="low", tier="deterministic",
              requires=["paper.numbers"], cannot_detect="x", registry=reg)
        def _r(ctx):
            return iter(())


def test_duplicate_ids_rejected(reg):
    for _ in range(1):
        @rule(id="t/dupe", severity="low", tier="deterministic",
              requires=["paper.numbers"], cannot_detect="x", registry=reg)
        def _r(ctx):
            return iter(())

    with pytest.raises(RuleDefinitionError, match="duplicate"):
        @rule(id="t/dupe", severity="low", tier="deterministic",
              requires=["paper.numbers"], cannot_detect="x", registry=reg)
        def _r2(ctx):
            return iter(())


def test_finding_factory_attributes_to_the_running_rule(reg):
    @rule(id="t/attr", severity="med", tier="deterministic",
          requires=["paper.numbers"], cannot_detect="x", registry=reg)
    def _r(ctx):
        yield ctx.finding(message="default severity", anchors=[span(0, 1), span(2, 3)])
        yield ctx.finding(message="escalated", anchors=[span(0, 1), span(2, 3)], severity="high")

    out = reg.get("t/attr").run(Context(paper=SimpleNamespace(numbers=[])))
    assert [f.rule_id for f in out] == ["t/attr", "t/attr"]
    assert [f.severity for f in out] == [Severity.MED, Severity.HIGH]
    assert all(f.tier is Tier.DETERMINISTIC for f in out)


def test_required_slices_unions_declarations(reg):
    @rule(id="t/a", severity="low", tier="deterministic",
          requires=["paper.numbers"], cannot_detect="x", registry=reg)
    def _a(ctx):
        return iter(())

    @rule(id="t/b", severity="low", tier="deterministic",
          requires=["paper.numbers", "paper.stats"], cannot_detect="x", registry=reg)
    def _b(ctx):
        return iter(())

    assert reg.required_slices(reg.all()) == {"paper.numbers", "paper.stats"}


def test_needs_repo_detection(reg):
    @rule(id="t/local", severity="low", tier="deterministic",
          requires=["paper.numbers"], cannot_detect="x", registry=reg)
    def _a(ctx):
        return iter(())

    @rule(id="t/coded", severity="low", tier="deterministic",
          requires=["paper.numbers", "repo.configs"], cannot_detect="x", registry=reg)
    def _b(ctx):
        return iter(())

    assert not reg.get("t/local").needs_repo
    assert reg.get("t/coded").needs_repo


def test_ir_is_read_only_inside_a_rule(reg):
    @rule(id="t/ro", severity="low", tier="deterministic",
          requires=["paper.numbers"], cannot_detect="x", registry=reg)
    def _r(ctx):
        ctx.paper.numbers = []
        return iter(())

    with pytest.raises(AttributeError, match="read-only"):
        reg.get("t/ro").run(Context(paper=SimpleNamespace(numbers=[])))
