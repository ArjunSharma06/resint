"""Walking a repository into the Repo IR.

The walk is bounded on purpose. A research repository can carry a checkpoint
directory, a vendored dependency tree, and a decade of notebooks, and reading
all of it to find a learning rate would make the tool slow enough that people
stop running it. Caps are generous but real, and anything skipped because of
one is reported rather than silently dropped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ..ir.repo import Binding, ConfigSet, Dependency, Link, Repo
from ..ir.span import Source, Span
from .code import read_python
from .configs import read_json, read_yaml

SKIP_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv",
    "env", ".env", "build", "dist", ".tox", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", "site-packages", ".ipynb_checkpoints", "wandb", "runs",
    "checkpoints", "outputs", ".idea", ".vscode", "third_party", "vendor",
}

CODE_SUFFIXES = {".py"}
CONFIG_SUFFIXES = {".yaml", ".yml", ".json"}
DOC_SUFFIXES = {".md", ".rst", ".txt"}

MAX_FILES = 4000
MAX_FILE_BYTES = 512 * 1024

LOCKFILES = {
    "poetry.lock", "Pipfile.lock", "uv.lock", "pdm.lock", "conda-lock.yml",
    "requirements.lock", "requirements.txt.lock",
}

_URL = re.compile(r"https?://[^\s<>\"'\)\]\},]+")
_REQ_LINE = re.compile(
    r"^\s*(?P<name>[A-Za-z0-9][\w.\-]*)\s*(?P<extras>\[[^\]]*\])?\s*"
    r"(?P<constraint>(?:[=<>!~]=?|@)\s*[^\s;#]+)?"
)
_README_CMD = re.compile(
    r"(?:^|\s)(?:python3?|torchrun|accelerate\s+launch|bash|sh|make)\s+"
    r"(?:-m\s+(?P<module>[\w.]+)|(?P<script>[\w./-]+\.(?:py|sh)))"
)

# A config file whose name says it extends another binds more strongly than
# the base it overrides.
_OVERRIDE_HINTS = ("override", "experiment", "exp_", "local", "custom")


@dataclass(frozen=True, slots=True)
class Entrypoint:
    command: str
    target: str
    span: Span
    exists: bool


def _binding_for(rel: str) -> Binding:
    lowered = rel.lower()
    if any(hint in lowered for hint in _OVERRIDE_HINTS):
        return Binding.CONFIG_OVERRIDE
    return Binding.CONFIG_FILE


def _iter_files(root: Path) -> tuple[list[Path], list[str]]:
    found: list[Path] = []
    notes: list[str] = []
    for path in sorted(root.rglob("*")):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if not path.is_file():
            continue
        found.append(path)
        if len(found) >= MAX_FILES:
            notes.append(
                f"repository walk stopped at {MAX_FILES} files; later files "
                "were not read"
            )
            break
    return found, notes


#: Every slice this loader can populate. Named so "no needs given" can mean
#: "all of it" without a caller having to enumerate them.
ALL_REPO_SLICES = frozenset(
    {
        "repo.files",
        "repo.readme",
        "repo.readme_source",
        "repo.configs",
        "repo.seeds",
        "repo.symbols",
        "repo.deps",
        "repo.links",
        "repo.entrypoints",
        "repo.lockfiles",
    }
)


def read_repo(root: str | Path, needs: set[str] | None = None) -> Repo:
    """Assemble the Repo IR from a checkout on disk."""
    base = Path(root)
    repo = Repo(root=str(base))
    if not base.is_dir():
        repo.unchecked.append(f"{base} is not a directory; no repository was read")
        return repo

    # Unspecified means everything, matching paper_from_path. It used to mean
    # *nothing*: read_repo(root) with no needs= returned a Repo with every
    # slice empty, so every repro rule looked, found an empty world and stayed
    # silent. Indistinguishable from a clean repository, and the coverage
    # census on hparam-drift is what finally showed it -- "2 named, 0 located"
    # on a fixture built so it must find both.
    wanted = set(ALL_REPO_SLICES) if needs is None else set(needs)
    files, notes = _iter_files(base)
    repo.unchecked.extend(notes)
    repo.files = [str(p.relative_to(base)).replace("\\", "/") for p in files]
    repo.lockfiles = [f for f in repo.files if Path(f).name in LOCKFILES]

    want_configs = {"repo.configs", "repo.seeds", "repo.symbols"} & wanted
    want_links = {"repo.links", "repo.readme", "repo.entrypoints"} & wanted
    want_deps = "repo.deps" in wanted

    for path in files:
        rel = str(path.relative_to(base)).replace("\\", "/")
        suffix = path.suffix.lower()
        name = path.name

        interesting = (
            (want_configs and suffix in CODE_SUFFIXES | CONFIG_SUFFIXES)
            or (want_links and suffix in DOC_SUFFIXES)
            or (want_deps and name in ("requirements.txt", "pyproject.toml", "setup.py"))
        )
        if not interesting:
            continue

        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                repo.unchecked.append(f"{rel}: larger than 512 KB, not read")
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            repo.unchecked.append(f"{rel}: could not be read ({exc.strerror})")
            continue

        src = Source(rel, "code" if suffix in CODE_SUFFIXES else "config", path=rel)

        if want_configs and suffix in CODE_SUFFIXES:
            facts = read_python(text, src, rel)
            repo.configs.extend(facts.configs)
            repo.seeds.extend(facts.seeds)
            repo.symbols.extend(facts.symbols)
            repo.unchecked.extend(facts.unchecked)

        if want_configs and suffix in CONFIG_SUFFIXES:
            reader = read_yaml if suffix in (".yaml", ".yml") else read_json
            parsed = reader(text, src, rel, _binding_for(rel))
            repo.configs.extend(parsed.keys)
            repo.unchecked.extend(parsed.unchecked)

        if want_deps and name in ("requirements.txt", "pyproject.toml", "setup.py"):
            repo.deps.extend(_read_dependencies(text, src, rel))

        if want_links and suffix in DOC_SUFFIXES:
            for match in _URL.finditer(text):
                repo.links.append(
                    Link(
                        url=match.group(0).rstrip(".,;:"),
                        span=Span(
                            src,
                            match.start(),
                            match.end(),
                            line=text.count("\n", 0, match.start()) + 1,
                            label=rel,
                        ),
                        context=rel,
                    )
                )
            if name.lower().startswith("readme"):
                repo.readme = text
                repo.readme_source = src
                repo.entrypoints.extend(
                    _read_entrypoints(text, src, rel, base, set(repo.files))
                )

    return repo


def _read_dependencies(text: str, src: Source, rel: str) -> list[Dependency]:
    if not rel.endswith("requirements.txt"):
        return []
    out: list[Dependency] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped and not stripped.startswith(("#", "-", "git+")):
            match = _REQ_LINE.match(stripped)
            if match and match.group("name"):
                constraint = (match.group("constraint") or "").strip()
                out.append(
                    Dependency(
                        name=match.group("name"),
                        constraint=constraint,
                        pinned=constraint.startswith("=="),
                        span=Span(
                            src,
                            offset,
                            offset + len(stripped),
                            line=text.count("\n", 0, offset) + 1,
                            label=rel,
                        ),
                        manifest=rel,
                    )
                )
        offset += len(line)
    return out


def _read_entrypoints(
    text: str, src: Source, rel: str, base: Path, known: set[str]
) -> list[Entrypoint]:
    out: list[Entrypoint] = []
    seen: set[str] = set()
    for match in _README_CMD.finditer(text):
        module, script = match.group("module"), match.group("script")
        if module:
            target = module.replace(".", "/") + ".py"
        else:
            target = script.lstrip("./")
        if target in seen:
            continue
        seen.add(target)

        exists = target in known or any(f.endswith("/" + target) for f in known)
        if module and not exists:
            package = module.replace(".", "/") + "/__init__.py"
            exists = package in known or any(f.endswith("/" + package) for f in known)

        out.append(
            Entrypoint(
                command=match.group(0).strip(),
                target=target,
                span=Span(
                    src,
                    match.start(),
                    match.end(),
                    line=text.count("\n", 0, match.start()) + 1,
                    label=rel,
                ),
                exists=exists,
            )
        )
    return out
