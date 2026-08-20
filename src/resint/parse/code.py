"""Reading Python source with the standard library's own parser.

``ast`` is already installed everywhere and understands the language exactly,
which makes tree-sitter an unnecessary dependency for the language most
research code is written in. Other languages will need a different strategy;
that is a later problem and should not cost the install today.

What comes out of here is deliberately narrow: declared argument defaults,
seed calls, and a symbol index. Not a call graph, not dataflow. A rule that
needs to know what a program *does* is a rule for the execution tier; these
are the facts you can read off the source without running it.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

from ..ir.repo import Binding, ConfigKey, SeedCall, Symbol
from ..ir.span import Source, Span

_SEED_FUNCTIONS = {
    "seed": "random",
    "manual_seed": "torch",
    "manual_seed_all": "torch.cuda",
    "set_seed": "transformers",
    "seed_everything": "lightning",
    "set_random_seed": "framework",
    "PRNGKey": "jax",
}

_SEED_ATTRIBUTES = {"random", "np", "numpy", "torch", "cuda", "tf", "jax"}

_LOOP_NODES = (ast.For, ast.While, ast.AsyncFor, ast.comprehension)


def _literal(node: ast.AST) -> str | None:
    """Render a literal argument, or None if it is not one."""
    try:
        value = ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError):
        return None
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float, str)):
        return str(value)
    return None


def _as_written(node: ast.AST, text: str) -> str | None:
    """A numeric literal in the form the author actually typed.

    ``literal_eval`` turns ``3e-4`` into ``0.0003``, and a drift message
    reporting 0.0003 where the file plainly says 3e-4 reads as the tool
    having misparsed the source. Comparison is numeric either way, so this
    only affects what the author is shown -- which is the part that decides
    whether they trust the finding. Strings keep the normalized form, since
    their quotes are not information.
    """
    rendered = _literal(node)
    if rendered is None:
        return None
    segment = ast.get_source_segment(text, node)
    if segment is None:
        return rendered
    try:
        float(segment)
    except ValueError:
        return rendered
    return segment


@dataclass
class CodeFacts:
    configs: list[ConfigKey] = field(default_factory=list)
    seeds: list[SeedCall] = field(default_factory=list)
    symbols: list[Symbol] = field(default_factory=list)
    unchecked: list[str] = field(default_factory=list)


class _Visitor(ast.NodeVisitor):
    def __init__(self, src: Source, text: str, rel: str) -> None:
        self.src = src
        self.text = text
        self.lines = text.splitlines()
        self.rel = rel
        self.facts = CodeFacts()
        self._loop_depth = 0
        self._offsets = self._line_offsets(text)

    @staticmethod
    def _line_offsets(text: str) -> list[int]:
        offsets, total = [0], 0
        for line in text.splitlines(keepends=True):
            total += len(line)
            offsets.append(total)
        return offsets

    def _span(self, node: ast.AST, label: str) -> Span:
        line = getattr(node, "lineno", 1)
        col = getattr(node, "col_offset", 0)
        end_line = getattr(node, "end_lineno", line) or line
        end_col = getattr(node, "end_col_offset", col + 1) or col + 1
        start = self._offsets[min(line - 1, len(self._offsets) - 1)] + col
        end = self._offsets[min(end_line - 1, len(self._offsets) - 1)] + end_col
        return Span(self.src, start, max(end, start + 1), line=line, label=label)

    # --- traversal ------------------------------------------------------

    def generic_visit(self, node: ast.AST) -> None:
        if isinstance(node, _LOOP_NODES):
            self._loop_depth += 1
            super().generic_visit(node)
            self._loop_depth -= 1
        else:
            super().generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.facts.symbols.append(
            Symbol(node.name, "function", self._span(node, f"{self.rel}:{node.name}"))
        )
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.facts.symbols.append(
            Symbol(node.name, "class", self._span(node, f"{self.rel}:{node.name}"))
        )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        self._maybe_add_argument(node)
        self._maybe_seed(node)
        self.generic_visit(node)

    # --- argparse -------------------------------------------------------

    def _maybe_add_argument(self, node: ast.Call) -> None:
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "add_argument"):
            return

        flag = None
        for arg in node.args:
            rendered = _literal(arg)
            if rendered and rendered.startswith("-"):
                flag = rendered.lstrip("-")
                break
        if not flag:
            return

        default = next((k for k in node.keywords if k.arg == "default"), None)
        if default is None:
            return

        value = _as_written(default.value, self.text)
        if value is None:
            # A computed default cannot be read off the source. Saying so is
            # the whole point; guessing would put a number in a diff table
            # that the program may never use.
            self.facts.unchecked.append(
                f"{self.rel}:{default.value.lineno}: default for --{flag} is "
                "computed at runtime and was not read"
            )
            return

        self.facts.configs.append(
            ConfigKey(
                name=flag,
                raw_name=flag,
                value=value,
                binding=Binding.ARGPARSE,
                span=self._span(default.value, f"{self.rel}"),
                origin=f"argparse default in {self.rel}",
            )
        )

    # --- seeds ----------------------------------------------------------

    def _maybe_seed(self, node: ast.Call) -> None:
        func = node.func
        name = None
        library = None

        if isinstance(func, ast.Attribute) and func.attr in _SEED_FUNCTIONS:
            root = func.value
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name) and root.id in _SEED_ATTRIBUTES:
                name, library = func.attr, root.id
            elif func.attr in ("manual_seed", "seed_everything", "set_seed"):
                name, library = func.attr, _SEED_FUNCTIONS[func.attr]
        elif isinstance(func, ast.Name) and func.id in (
            "seed_everything",
            "set_seed",
            "set_random_seed",
        ):
            name, library = func.id, _SEED_FUNCTIONS[func.id]

        if name is None:
            return

        argument = None
        from_config = False
        if node.args:
            argument = _literal(node.args[0])
            if argument is None:
                # seed(args.seed) or seed(cfg.seed): the value comes from
                # outside, so this call may well vary across runs.
                from_config = True

        self.facts.seeds.append(
            SeedCall(
                library=library or "unknown",
                argument=argument,
                span=self._span(node, self.rel),
                in_loop=self._loop_depth > 0,
                from_config=from_config,
            )
        )


def read_python(text: str, src: Source, rel: str) -> CodeFacts:
    """Extract declared defaults, seed calls, and symbols from one module."""
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        facts = CodeFacts()
        facts.unchecked.append(f"{rel}: not parsed ({exc.msg} at line {exc.lineno})")
        return facts

    visitor = _Visitor(src, text, rel)
    visitor.visit(tree)
    return visitor.facts
