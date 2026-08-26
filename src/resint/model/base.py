"""Talking to a language model, behind an interface.

This mirrors ``resolve/base.py`` closely, and deliberately: that module
already solves the same problem for bibliographic lookups, and a reader who
knows one should recognise the other immediately.

Three outcomes, and the third is the safety property:

    ANSWERED     the model replied and the reply matched its schema
    DECLINED     the model looked and returned a well-formed negative
    UNAVAILABLE  no key, timeout, rate limit, refusal, malformed JSON,
                 schema violation, budget spent, dependency missing

**UNAVAILABLE must never become a finding.** It is the exact analogue of
``Status.UNKNOWN`` for references: reporting a problem with someone's paper
because the model was rate-limited would be the same failure as reporting a
citation fabricated because the network blipped.

The user brings their own model. resint has no key of its own and, without one
configured, model rules are skipped and reported as skipped rather than
silently passing.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol


class Outcome(str, Enum):
    ANSWERED = "answered"
    DECLINED = "declined"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class Request:
    """One question, and the shape the answer has to take.

    ``schema`` is not decoration. A reply that does not match it is
    UNAVAILABLE, which is what keeps a confused model from becoming a finding.
    """

    system: str
    user: str
    schema: dict
    tier: str = "cheap"
    prompt_version: str = "1"
    max_tokens: int = 2048

    def cache_key(self, model: str) -> str:
        """Identity for the verdict cache.

        The rendered body is hashed, not just ``prompt_version``. A template
        edit that nobody remembers to version-bump would otherwise serve stale
        answers forever, and the body hash is the backstop against that.
        """
        body = json.dumps(
            {
                "system": self.system,
                "user": self.user,
                "schema": self.schema,
                "prompt_version": self.prompt_version,
                "model": model,
            },
            sort_keys=True,
        )
        return hashlib.sha256(body.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class Completion:
    outcome: Outcome
    payload: dict | None = None
    model: str = ""
    detail: str = ""
    usage: dict = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        """Whether this reply may contribute to a finding."""
        return self.outcome is Outcome.ANSWERED and self.payload is not None


class Provider(Protocol):
    def complete(self, request: Request) -> Completion: ...


class NullProvider:
    """Answers nothing, and says so. The default.

    Every model rule is skipped when this is in place, so the deterministic
    tier behaves exactly as it did before the model tier existed.
    """

    name = "null"

    def complete(self, request: Request) -> Completion:
        return Completion(Outcome.UNAVAILABLE, detail="no model provider configured")


@dataclass
class StaticProvider:
    """A fixed table of answers, keyed by request hash.

    How the model tier is tested. CI runs with sockets disabled, so every
    model test replays recorded answers rather than calling anything.
    """

    responses: dict = field(default_factory=dict)
    model: str = "static"
    unknown_is: Outcome = Outcome.UNAVAILABLE

    def complete(self, request: Request) -> Completion:
        payload = self.responses.get(request.cache_key(self.model))
        if payload is None:
            payload = self.responses.get(request.user)  # convenience for tests
        if payload is None:
            return Completion(
                self.unknown_is, model=self.model, detail="no recorded response"
            )
        return Completion(Outcome.ANSWERED, payload=payload, model=self.model)


@dataclass
class CachingProvider:
    """Memoizes another provider on disk.

    This is not only a cost optimisation -- it is the **determinism
    mechanism**. ``temperature=0`` does not exist on current models (the
    parameter is removed and returns a 400), so run-to-run reproducibility
    cannot come from sampling settings. It comes from here.

    Which means a cache miss during a comparison run is a real event worth
    noticing, not something to paper over with a live call.
    """

    inner: Provider
    store: object | None = None
    _memory: dict = field(default_factory=dict, repr=False)
    _lock: object = field(default_factory=threading.Lock, repr=False)
    hits: int = 0
    misses: int = 0

    def complete(self, request: Request) -> Completion:
        model = getattr(self.inner, "model", "") or "unknown"
        key = request.cache_key(model)

        with self._lock:
            hit = self._memory.get(key)
        if hit is None and self.store is not None:
            hit = self.store.get(key)

        if hit is not None:
            self.hits += 1
            return Completion(
                Outcome.ANSWERED, payload=hit, model=model, detail="cached"
            )

        self.misses += 1
        result = self.inner.complete(request)

        # Only successful answers are cached. Caching a timeout would turn a
        # transient failure into a permanent one.
        if result.usable:
            with self._lock:
                self._memory[key] = result.payload
            if self.store is not None:
                self.store.put(key, result.payload)

        return result


@dataclass
class RecordingProvider:
    """Wraps a live provider and writes every exchange to disk.

    This is how the static fixtures get made: run once with a real key, commit
    the recordings, and CI replays them offline forever. Without it, every
    model test would be either hand-written fiction or a live call.
    """

    inner: Provider
    path: object = None
    recorded: dict = field(default_factory=dict)

    def complete(self, request: Request) -> Completion:
        result = self.inner.complete(request)
        if result.usable:
            model = getattr(self.inner, "model", "") or "unknown"
            self.recorded[request.cache_key(model)] = result.payload
            if self.path is not None:
                from pathlib import Path

                Path(self.path).write_text(
                    json.dumps(self.recorded, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
        return result


@dataclass
class Budget:
    """A ceiling on what one run may spend.

    Model calls cost the user money and are not deterministic, so a rule that
    silently makes four hundred of them is a bug even when every answer is
    right. Exhaustion fails closed to UNAVAILABLE, which by the rule above
    can never become a finding.
    """

    max_calls: int = 40
    used: int = 0

    @property
    def spent(self) -> bool:
        return self.used >= self.max_calls

    def take(self) -> bool:
        if self.spent:
            return False
        self.used += 1
        return True


@dataclass
class BudgetedProvider:
    inner: Provider
    budget: Budget = field(default_factory=Budget)

    def complete(self, request: Request) -> Completion:
        if not self.budget.take():
            return Completion(
                Outcome.UNAVAILABLE,
                detail=f"run budget of {self.budget.max_calls} model calls is spent",
            )
        return self.inner.complete(request)
