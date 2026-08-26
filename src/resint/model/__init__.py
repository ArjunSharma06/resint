"""The model-assisted tier.

The user brings their own model. resint holds no key, and with none configured
every model rule is skipped and reported as skipped -- never silently passed.
One transport reaches OpenAI, Gemini, Groq and a local Ollama, because all of
them speak the same wire format, and it is written on the standard library so
the install stays dependency-free.

Read ``verify.py`` before anything else here. It holds the single idea the
whole tier rests on: the model returns a quote, code finds the offset.
"""

from .base import (
    Budget,
    BudgetedProvider,
    CachingProvider,
    Completion,
    NullProvider,
    Outcome,
    Provider,
    RecordingProvider,
    Request,
    StaticProvider,
)
from .verify import Anchored, Located, Verdict, anchor_in, anchor_quotes, locate

__all__ = [
    "Anchored",
    "Budget",
    "BudgetedProvider",
    "CachingProvider",
    "Completion",
    "Located",
    "NullProvider",
    "Outcome",
    "Provider",
    "RecordingProvider",
    "Request",
    "StaticProvider",
    "Verdict",
    "anchor_in",
    "anchor_quotes",
    "locate",
]
