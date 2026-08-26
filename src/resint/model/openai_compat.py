"""One transport, several providers.

OpenAI, Google Gemini, Groq and a locally-run Ollama all expose the same
``/chat/completions`` wire format, so one stdlib client reaches all of them and
``dependencies = []`` stays literally true. The user picks a provider and a
base URL; resint holds no key and no account.

Every failure path collapses to UNAVAILABLE — no key, refused connection,
timeout, rate limit, a reply that is not JSON, a reply that is JSON but the
wrong shape. That is the same discipline ``resolve/http.py`` applies to
reference lookups, and for the same reason: **a problem at our end must never
be reported as a problem with someone's paper.**

The reply is required to be JSON matching a schema. That is not tidiness — it
is the boundary check. A model that has drifted off-task produces something
that fails validation and therefore produces nothing, rather than producing a
confident finding about a paper it stopped reading.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from .base import Completion, Outcome, Request

#: Where each provider speaks, and which environment variable holds its key.
PROVIDERS = {
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "gemini": (
        "https://generativelanguage.googleapis.com/v1beta/openai",
        "GEMINI_API_KEY",
    ),
    "together": ("https://api.together.xyz/v1", "TOGETHER_API_KEY"),
    "deepseek": ("https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    # Local, free, and needs no key at all. The reason a contributor with no
    # budget can still work on the model tier.
    "ollama": ("http://localhost:11434/v1", ""),
}

USER_AGENT = "resint (https://github.com/ArjunSharma06/resint)"

#: Longest a single rate-limit wait may be. A provider asking for five minutes
#: is telling you to come back later, not to hold the terminal open.
MAX_BACKOFF_SECONDS = 30.0


#: Google states the wait in the error body rather than in a header:
#: "Please retry in 5.056492331s." Reading it beats guessing.
_RETRY_IN_BODY = re.compile(r"retry in ([0-9.]+)\s*s", re.IGNORECASE)


def _retry_after(exc, attempt: int, body: str = "") -> float:
    """How long to wait after a 429.

    The provider knows when its window resets and we do not, so anything it
    tells us wins. It may say so in a Retry-After header, or -- as Gemini does
    -- in prose inside the error body. Only when neither is present do we
    guess, doubling from two seconds.
    """
    try:
        header = (exc.headers or {}).get("Retry-After", "") or ""
    except AttributeError:
        header = ""

    if header:
        try:
            return min(float(header), MAX_BACKOFF_SECONDS)
        except ValueError:
            pass  # Retry-After may be an HTTP date; fall through.

    stated = _RETRY_IN_BODY.search(body or "")
    if stated:
        try:
            # A shade over what was asked for. Coming back at the exact
            # instant the window opens is how you get refused again.
            return min(float(stated.group(1)) + 0.5, MAX_BACKOFF_SECONDS)
        except ValueError:
            pass

    return min(2.0 ** (attempt + 1), MAX_BACKOFF_SECONDS)


@dataclass
class OpenAICompatProvider:
    """Talks to anything speaking the OpenAI chat-completions format."""

    model: str
    provider: str = "ollama"
    base_url: str = ""
    api_key: str | None = None
    timeout: float = 60.0
    #: How many times to wait out a rate limit before giving up. Six model
    #: rules fire five or six calls per paper in quick succession, which is
    #: over the free-tier limit of every hosted provider -- the first real run
    #: against Gemini had 8 of 8 calls refused. Rather than pace every user to
    #: the slowest tier, back off only when actually told to: a paid key never
    #: waits, and a free one still finishes.
    max_retries: int = 3
    #: Injectable so a test can exercise backoff without sleeping through it.
    sleep: object = None
    clock: object = None
    #: Seconds to leave between calls. Starts at zero -- a paid key or a local
    #: model should never be slowed -- and rises the first time a provider says
    #: it is being asked too fast. Retries count against the same quota they
    #: are waiting for, so reacting alone turns one refusal into four; the
    #: calls after it have to slow down too.
    min_interval: float = 0.0
    _endpoint: str = field(default="", repr=False)
    _last_call: float = field(default=0.0, repr=False)

    def __post_init__(self) -> None:
        default_url, env_var = PROVIDERS.get(self.provider, ("", ""))
        self._endpoint = (self.base_url or default_url).rstrip("/")
        if self.api_key is None and env_var:
            # Environment only. A key must never come from .resint.yml, which
            # is committed alongside the paper.
            self.api_key = os.environ.get(env_var)

    @property
    def configured(self) -> bool:
        """Whether this could plausibly answer. Cheap: no I/O.

        Rule selection needs to know whether a provider exists *before* any
        paper is loaded, so this must not touch the network.
        """
        if not self._endpoint:
            return False
        _, env_var = PROVIDERS.get(self.provider, ("", ""))
        return bool(self.api_key) or not env_var

    def models(self) -> list[str]:
        """What this provider actually serves, from its own /models endpoint.

        Model names change and retire without notice, and a wrong one fails as
        a 404 that looks like a broken endpoint. Every OpenAI-compatible API
        exposes this list; reading it costs nothing and settles the question.
        """
        if not self.configured:
            return []
        headers = {"User-Agent": USER_AGENT}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(f"{self._endpoint}/models", headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, ValueError, OSError):
            return []
        found = [m.get("id", "") for m in payload.get("data") or []]
        return sorted(name for name in found if name)

    def complete(self, request: Request) -> Completion:
        if not self.configured:
            _, env_var = PROVIDERS.get(self.provider, ("", ""))
            return Completion(
                Outcome.UNAVAILABLE,
                model=self.model,
                detail=f"no API key; set {env_var}" if env_var else "no endpoint",
            )

        body = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": request.system},
                    {"role": "user", "content": request.user},
                ],
                "max_tokens": request.max_tokens,
                # Ask for JSON. Providers that ignore the hint still usually
                # comply because the prompt says so, and anything that does
                # not will fail validation below rather than leak through.
                "response_format": {"type": "json_object"},
            }
        ).encode()

        headers = {
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(
            f"{self._endpoint}/chat/completions", data=body, headers=headers
        )
        pause = self.sleep or time.sleep
        now = self.clock or time.monotonic

        for attempt in range(self.max_retries + 1):
            # Space this call from the last one, if a refusal has taught us to.
            if self.min_interval > 0 and self._last_call:
                overdue = self.min_interval - (now() - self._last_call)
                # Not "> 0": subtracting two floats that should cancel leaves
                # a residue around 1e-16, and pausing for that is a wasted
                # syscall that also shows up in tests as a phantom sleep.
                if overdue > 0.01:
                    pause(overdue)
            self._last_call = now()

            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                if exc.code == 429:
                    try:
                        body = exc.read().decode("utf-8", "replace")
                    except Exception:  # noqa: BLE001 -- diagnostics only
                        body = ""
                    wait = _retry_after(exc, attempt, body)
                    # Learn the pace rather than only reacting to it.
                    self.min_interval = max(self.min_interval, min(wait, 10.0))
                    if attempt < self.max_retries:
                        pause(wait)
                        continue

                detail = f"HTTP {exc.code}"
                if exc.code == 429:
                    detail = (
                        f"rate limited, and still limited after "
                        f"{self.max_retries} retries"
                    )
                elif exc.code in (401, 403):
                    detail = "key rejected"
                return Completion(Outcome.UNAVAILABLE, model=self.model, detail=detail)
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                hint = " (is ollama running?)" if self.provider == "ollama" else ""
                return Completion(
                    Outcome.UNAVAILABLE, model=self.model, detail=f"{exc}{hint}"
                )
            except ValueError as exc:
                return Completion(
                    Outcome.UNAVAILABLE,
                    model=self.model,
                    detail=f"reply was not JSON: {exc}",
                )

            return self._read(payload, request)

        return Completion(
            Outcome.UNAVAILABLE, model=self.model, detail="rate limited"
        )

    def _read(self, payload: dict, request: Request) -> Completion:
        choices = payload.get("choices") or []
        if not choices:
            return Completion(
                Outcome.UNAVAILABLE, model=self.model, detail="reply had no content"
            )

        choice = choices[0]
        reason = choice.get("finish_reason")
        if reason in ("content_filter", "refusal"):
            # A linter over arXiv will eventually hand a model a paper on
            # pathogen engineering or offensive security. A refusal is a
            # perfectly reasonable answer and must not become a finding.
            return Completion(
                Outcome.UNAVAILABLE, model=self.model, detail=f"model declined ({reason})"
            )

        content = (choice.get("message") or {}).get("content")

        if not content and reason == "length":
            # A reasoning model spends tokens thinking before it writes
            # anything, and those tokens are not in completion_tokens. Asked
            # for JSON under a tight cap it returns a message object with no
            # content field at all -- which read as "empty reply" and sent
            # everyone looking at the wrong thing. Say what actually happened.
            return Completion(
                Outcome.UNAVAILABLE,
                model=self.model,
                detail=(
                    f"reply was cut off by the {self.model} token limit "
                    "before any content was produced; raise max_tokens"
                ),
            )

        if not content:
            return Completion(
                Outcome.UNAVAILABLE, model=self.model, detail="reply was empty"
            )

        try:
            parsed = json.loads(content)
        except ValueError:
            return Completion(
                Outcome.UNAVAILABLE,
                model=self.model,
                detail="reply was not valid JSON",
            )

        if not isinstance(parsed, dict):
            return Completion(
                Outcome.UNAVAILABLE,
                model=self.model,
                detail=f"reply was {type(parsed).__name__}, expected an object",
            )

        missing = [k for k in request.schema.get("required", []) if k not in parsed]
        if missing:
            return Completion(
                Outcome.UNAVAILABLE,
                model=self.model,
                detail=f"reply missing required field(s): {', '.join(missing)}",
            )

        return Completion(
            Outcome.ANSWERED,
            payload=parsed,
            model=self.model,
            usage=payload.get("usage") or {},
        )


def from_config(settings) -> "OpenAICompatProvider | None":
    """Build a provider from ``.resint.yml``, or None if none is configured.

    None means the model rules are skipped and reported as skipped. A provider
    that exists but has no key behaves the same way at the point of use, since
    every call comes back UNAVAILABLE -- but returning it lets the report say
    which environment variable is missing instead of saying nothing at all.
    """
    if not settings:
        return None
    provider = (settings.get("provider") or "").strip()
    # "name", because "model.model" reads like a mistake in a config file.
    model = (settings.get("name") or settings.get("model") or "").strip()
    if not provider or not model:
        return None
    return OpenAICompatProvider(
        model=model,
        provider=provider,
        base_url=settings.get("base_url", ""),
    )
