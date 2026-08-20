"""Repository-side IR.

The load-bearing type here is ``ConfigKey``. A hyperparameter has a value in
the paper and a value in the code, and the second one is genuinely hard to
determine: an argparse default can be overridden by a YAML file, which can be
overridden by a Hydra composition, which can be overridden on the command line
by a launch script nobody committed.

So a ConfigKey records not just a value but *where it came from* and *how
strongly it binds*. When several sources disagree and precedence cannot be
established, the right answer is to abstain and say so -- a confident wrong
hyperparameter finding is the worst thing this tool can produce, because it
accuses an author of misreporting their own experiment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

from .span import Span


class Binding(IntEnum):
    """How strongly a source binds a value, lowest to highest.

    Ordered so a stronger source wins outright. Two sources at the *same*
    level that disagree is the ambiguous case, and ambiguity means silence.
    """

    CONSTANT = 10         # a module-level literal
    ARGPARSE = 20         # a declared default
    CONFIG_FILE = 30      # yaml/json committed alongside the code
    CONFIG_OVERRIDE = 40  # a config that explicitly overrides a base


# Hyperparameter aliases. A paper says "learning rate"; the code says "lr".
# Without this the rule silently compares nothing at all, which reads to the
# author as a clean bill of health rather than as a check that never ran.
ALIASES = {
    "lr": "learning_rate",
    "learningrate": "learning_rate",
    "learning rate": "learning_rate",
    "base_lr": "learning_rate",
    "init_lr": "learning_rate",
    "bs": "batch_size",
    "batchsize": "batch_size",
    "batch": "batch_size",
    "train_batch_size": "batch_size",
    "per_device_train_batch_size": "batch_size",
    "n_epochs": "epochs",
    "num_epochs": "epochs",
    "max_epochs": "epochs",
    "nepochs": "epochs",
    "wd": "weight_decay",
    "weightdecay": "weight_decay",
    "dropout_rate": "dropout",
    "drop_rate": "dropout",
    "temp": "temperature",
    "tau": "temperature",
    "hidden": "hidden_size",
    "hidden_dim": "hidden_size",
    "d_model": "hidden_size",
    "n_layers": "num_layers",
    "nlayers": "num_layers",
    "depth": "num_layers",
    "n_heads": "num_heads",
    "nheads": "num_heads",
    "attention heads": "num_heads",
    "warmup": "warmup_steps",
    "warmup_ratio": "warmup_steps",
    "grad_accum": "gradient_accumulation_steps",
    "accum_steps": "gradient_accumulation_steps",
    "max_len": "max_length",
    "seq_len": "max_length",
    "max_seq_length": "max_length",
    "lora rank": "lora_rank",
    "rank": "lora_rank",
}


def canonical(name: str) -> str:
    """Normalize a hyperparameter name to its canonical form."""
    lowered = name.strip().lower()
    if lowered in ALIASES:
        return ALIASES[lowered]
    flat = lowered.replace("-", "_").replace(" ", "_").lstrip("_")
    if flat in ALIASES:
        return ALIASES[flat]
    collapsed = flat.replace("_", "")
    return ALIASES.get(collapsed, flat)


@dataclass(frozen=True, slots=True)
class ConfigKey:
    name: str
    raw_name: str
    value: str
    binding: Binding
    span: Span
    origin: str = ""

    @property
    def canonical_name(self) -> str:
        return canonical(self.raw_name)

    def render(self) -> str:
        return f"{self.raw_name} = {self.value}"


class ConfigSet(list):
    """Every declared config value, plus the precedence logic over them.

    A list subclass rather than a bare list so ``repo.configs`` carries its
    own resolution. The declaration gate works on attribute names, and a rule
    forced to name ``repo.effective`` alongside ``repo.configs`` would be
    declaring an implementation detail rather than what it needs. Same
    reasoning as ``paper.text``.
    """

    def effective(self, name: str):
        """The value a run would actually use, plus every candidate.

        Returns ``(None, candidates)`` when the strongest binding level holds
        two or more disagreeing values. Precedence between them cannot be
        established from the repository alone, and guessing would produce a
        confident accusation about an experiment nobody observed.
        """
        target = canonical(name)
        candidates = [c for c in self if c.canonical_name == target]
        if not candidates:
            return None, []

        strongest = max(c.binding for c in candidates)
        top = [c for c in candidates if c.binding == strongest]
        if len({c.value for c in top}) > 1:
            return None, candidates
        return top[0], candidates


@dataclass(frozen=True, slots=True)
class SeedCall:
    """A call that fixes a random seed."""

    library: str
    argument: str | None
    span: Span
    in_loop: bool = False
    from_config: bool = False

    @property
    def varies(self) -> bool:
        """Whether this call could plausibly produce more than one seed."""
        return self.in_loop or self.from_config or self.argument is None


@dataclass(frozen=True, slots=True)
class Symbol:
    name: str
    kind: str
    span: Span

    def locate(self) -> str:
        return self.span.locate()


@dataclass(frozen=True, slots=True)
class Dependency:
    name: str
    constraint: str
    pinned: bool
    span: Span
    manifest: str


@dataclass(frozen=True, slots=True)
class Link:
    url: str
    span: Span
    context: str = ""

    @property
    def kind(self) -> str:
        low = self.url.lower()
        if any(w in low for w in (".pt", ".pth", ".ckpt", ".safetensors", ".bin", ".h5")):
            return "checkpoint"
        if any(w in low for w in ("dataset", "data/", ".zip", ".tar", ".csv")):
            return "dataset"
        return "link"


@dataclass
class Repo:
    """The repository-side IR. Populated lazily by declared need."""

    root: str
    files: list[str] = field(default_factory=list)
    symbols: list[Symbol] = field(default_factory=list)
    configs: ConfigSet = field(default_factory=ConfigSet)
    seeds: list[SeedCall] = field(default_factory=list)
    deps: list[Dependency] = field(default_factory=list)
    links: list[Link] = field(default_factory=list)
    readme: str = ""
    readme_source: object = None
    entrypoints: list = field(default_factory=list)
    lockfiles: list[str] = field(default_factory=list)
    unchecked: list[str] = field(default_factory=list)
