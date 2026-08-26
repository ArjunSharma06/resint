"""Per-index request spacing.

One global 20 req/s across Crossref, OpenAlex and arXiv was wrong in both
directions -- too slow for the first two, far too fast for arXiv, which asks
for roughly three seconds. A sweep pushing hundreds of papers through the old
pacer would have been rate-limited off the indices.
"""

import pytest

from resint.resolve import Pacer, ResolvePolicy


class _Clock:
    """A clock that only moves when a test says so."""

    def __init__(self):
        self.now = 0.0
        self.slept = []

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds


def test_each_index_has_its_own_interval():
    assert Pacer.DEFAULTS["arxiv"] > Pacer.DEFAULTS["crossref"]
    assert Pacer.DEFAULTS["arxiv"] >= 3.0, "arXiv asks for ~3s between requests"


def test_a_slow_index_does_not_stall_a_fast_one():
    clock = _Clock()
    pacer = Pacer(clock=clock.time, sleep=clock.sleep)

    pacer.wait("arxiv")
    pacer.wait("crossref")
    pacer.wait("crossref")

    # the arxiv call must not have imposed its 3s gap on crossref
    assert all(s < 1.0 for s in clock.slept), clock.slept


def test_repeated_calls_to_one_index_are_spaced():
    clock = _Clock()
    pacer = Pacer({"arxiv": 3.0}, clock=clock.time, sleep=clock.sleep)

    pacer.wait("arxiv")
    pacer.wait("arxiv")

    assert clock.slept and clock.slept[-1] == pytest.approx(3.0)


def test_the_first_call_never_waits():
    clock = _Clock()
    Pacer(clock=clock.time, sleep=clock.sleep).wait("crossref")
    assert clock.slept == []


def test_an_unknown_index_still_gets_paced():
    clock = _Clock()
    pacer = Pacer(clock=clock.time, sleep=clock.sleep)
    pacer.wait("somewhere-new")
    pacer.wait("somewhere-new")
    assert clock.slept


def test_resolvers_share_one_pacer_by_default():
    """Otherwise a batch runner creating one resolver per paper paces nothing."""
    from resint.resolve.http import HttpResolver

    assert HttpResolver()._paced() is HttpResolver()._paced()


def test_a_policy_can_override_the_defaults():
    policy = ResolvePolicy(workers=2, budget=5.0)
    assert (policy.workers, policy.budget) == (2, 5.0)
    assert policy.paced_by() is not None
