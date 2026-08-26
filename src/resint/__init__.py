"""resint -- a linter for research papers.

The public surface is deliberately small. Everything a wrapper needs is here:
build a Paper, run rules over it, format the result. The CLI is one consumer
of this API and holds no analysis of its own, which is what keeps the next
surface -- Action, MCP server, editor extension -- a thin file rather than a
rewrite.
"""

from .config import Config, discover, parse as parse_config
from .engine import Report, run
from .ir.finding import Finding, Severity, Tier
from .ir.span import Source, Span
from .parse.document import paper_from_latex, paper_from_path
from .rules import REGISTRY, Context, Registry, Rule, load_all, rule

__version__ = "0.1.1.dev0"

__all__ = [
    "Config",
    "Context",
    "Finding",
    "REGISTRY",
    "Registry",
    "Report",
    "Rule",
    "Severity",
    "Source",
    "Span",
    "Tier",
    "__version__",
    "discover",
    "load_all",
    "paper_from_latex",
    "paper_from_path",
    "parse_config",
    "rule",
    "run",
]
