"""Find out whether the model tier actually works on real papers.

    python tools/try_model.py --dry
    python tools/try_model.py --provider ollama --name llama3
    python tools/try_model.py --provider openai --name gpt-4o-mini --record fixtures/

Every test of the model tier feeds the rules a hand-written answer. That proves
a rule handles a well-formed answer correctly. It proves nothing about whether
a real model *produces* one -- and if the prompts come back with quotes that
never verify, every rule silently finds nothing and the whole suite still
passes. This is the tool that closes that gap.

**--dry needs no model, no key and no network.** It assembles the exact
requests the rules would send and reports what is in them. That catches the
failure worth catching first: not a bad model, but a bad prompt. LaTeX left
unstripped, a table rendered as noise, a truncation landing mid-table -- each
one makes the tier useless in a way no unit test can see, because the tests
supply the answer the prompt was supposed to elicit.

With a provider it does the real thing and reports the number that matters:

    **quote verification rate** -- of the quotes the model returned, how many
    were actually found in the source.

That is a hallucination rate measured with no labels, no annotation and no
judgement, and it is the honest headline number for this tier. A model
scoring poorly here is not usable for this task no matter how good its
findings look.

``--record`` saves the responses so CI can replay them forever with no key.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from resint.engine import plan  # noqa: E402
from resint.ir.finding import Tier  # noqa: E402
from resint.model.base import Completion, Outcome  # noqa: E402
from resint.model.verify import locate  # noqa: E402
from resint.parse.acquire import UnreadableInput  # noqa: E402
from resint.parse.document import paper_from_path  # noqa: E402
from resint.parse.repo import read_repo  # noqa: E402
from resint.rules import load_all  # noqa: E402
from resint.rules.registry import Context  # noqa: E402

DEFAULT_CACHE = Path.home() / ".cache" / "resint" / "eprints"

#: Fields in a model reply that are supposed to be verbatim quotes. Checked
#: against the paper directly, so this measurement does not depend on any
#: rule's internal handling being right.
QUOTE_FIELDS = ("claim", "ours", "baseline", "quote", "manuscript_quote", "cited_quote")


class Capturing:
    """Records every request and answers nothing. What --dry runs on."""

    model = "dry-run"

    def __init__(self):
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        return Completion(Outcome.UNAVAILABLE, detail="dry run: no model called")


class Watching:
    """Wraps a real provider and keeps both sides of every exchange."""

    def __init__(self, inner):
        self.inner = inner
        self.model = getattr(inner, "model", "?")
        self.exchanges = []

    def complete(self, request):
        started = time.perf_counter()
        answer = self.inner.complete(request)
        self.exchanges.append(
            {
                "rule": getattr(request, "prompt_version", "?"),
                "request": request,
                "answer": answer,
                "seconds": time.perf_counter() - started,
            }
        )
        return answer


def quotes_in(payload) -> list[str]:
    """Every string a model offered as a verbatim quote, at any depth."""
    found: list[str] = []

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in QUOTE_FIELDS and isinstance(value, str):
                    found.append(value)
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(payload)
    return found


def papers(cache: Path, limit: int):
    for path in sorted(cache.glob("*.tar.gz"))[:limit]:
        yield path


def describe(request) -> dict:
    """What is actually in a prompt, without reading the whole thing."""
    user = request.user or ""
    return {
        "rule": request.prompt_version,
        "system_chars": len(request.system or ""),
        "user_chars": len(user),
        # A rough token count. Four characters per token is close enough to
        # tell 2k from 40k, which is the only distinction that matters here.
        "approx_tokens": (len(request.system or "") + len(user)) // 4,
        "has_tables": "TABLES:" in user,
        "preamble": _preamble_noise(user),
    }


#: Traces of a LaTeX preamble that survived normalization. Counting
#: backslashes -- the obvious check -- cannot work: normalization strips the
#: command names and leaves their arguments, so the noise contains no
#: backslashes at all. That check passed on prompts opening with six hundred
#: characters of "theoremTheorem[section] lemma[theorem]Lemma", which is
#: exactly what it was written to catch.
_PREAMBLE_MARKS = (
    "[theorem]",
    "[section]",
    "#1",
    "documentclass",
    "usepackage",
    "newcommand",
    "\\begin{document}",
)


#: Junk shorter than this is not worth reporting. A stray "#1 #1 1.5" from a
#: \renewcommand declared after \begin{document} is untidy and harmless; six
#: hundred characters of theorem declarations is a broken prompt. Measuring
#: the distance to real prose distinguishes them, where matching markers alone
#: reports both identically.
NOISE_BUDGET = 120

_PROSE = re.compile(r"[A-Za-z][A-Za-z ,'-]{45,}")


def _preamble_noise(user: str) -> str:
    """How much non-prose leads a prompt, and what it looks like.

    Returns "" when the body starts promptly. Only the opening is examined:
    these markers occur legitimately deeper in a paper -- a methods section may
    discuss its own macros -- but at the front they mean the body never started.
    """
    head = user[len("PAPER:\n") : 1500]
    prose = _PROSE.search(head)
    distance = prose.start() if prose else len(head)
    if distance <= NOISE_BUDGET:
        return ""
    marks = [m for m in _PREAMBLE_MARKS if m.lower() in head[:distance].lower()]
    return f"{distance} chars before prose" + (f" ({', '.join(marks)})" if marks else "")


def run_paper(path: Path, registry, provider, repo_path=None):
    chosen = plan(registry, has_repo=repo_path is not None, has_provider=True)
    model_rules = [r for r in chosen.runnable if r.tier is Tier.MODEL_ASSISTED]

    paper = paper_from_path(path, needs=chosen.paper_slices)
    repo = read_repo(repo_path, needs=chosen.repo_slices) if repo_path else None

    ctx = Context(paper=paper, repo=repo, model=provider)
    findings = []
    for rule in model_rules:
        try:
            findings.extend(rule.run(ctx))
        except Exception as exc:  # noqa: BLE001 -- a crash here is the finding
            print(f"    !! {rule.id} crashed: {type(exc).__name__}: {exc}")
    return paper, findings, ctx


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--dry", action="store_true", help="no model, no network")
    parser.add_argument("--provider", default=None, help="openai, gemini, groq, ollama")
    parser.add_argument("--name", default=None, help="the model name")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--record", default=None, help="directory for fixtures")
    parser.add_argument(
        "--cache-db", default=None, help="where to keep answers (default: ~/.cache)"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="always call the model, even for prompts already answered",
    )
    parser.add_argument("--repo", default=None)
    parser.add_argument("--show", type=int, default=0, help="print N chars of each prompt")
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="ask the provider what it serves, and exit",
    )
    args = parser.parse_args(argv)

    if args.list_models:
        return list_models(args)

    cache = Path(args.cache)
    found = list(papers(cache, args.count))
    if not found:
        print(f"no cached papers in {cache}", file=sys.stderr)
        print("run: python tools/fetch_arxiv.py --count 20", file=sys.stderr)
        return 2

    registry = load_all()

    if args.dry:
        return dry_run(found, registry, args)

    if not args.provider or not args.name:
        print("give --provider and --name, or use --dry", file=sys.stderr)
        return 2

    from resint.model.base import CachingProvider
    from resint.model.openai_compat import OpenAICompatProvider
    from resint.model.store import DiskStore

    inner = OpenAICompatProvider(
        model=args.name, provider=args.provider, base_url=args.base_url
    )
    if not inner.configured:
        print(f"{args.provider} is not configured -- no API key found", file=sys.stderr)
        return 2

    # Cache before watch, so a repeat run costs nothing and the watcher still
    # sees every exchange. Iterating on a rule means running it over the same
    # papers twenty times; without this that is twenty times the calls, which
    # is exactly how the first session exhausted a free quota re-deriving
    # answers it already had.
    provider = inner
    if not args.no_cache:
        store = DiskStore(args.cache_db)
        provider = CachingProvider(inner=inner, store=store)
        print(f"cache: {store.count()} answers on disk at {store.path}")
        print()

    watcher = Watching(provider)
    watcher.model = inner.model  # the cache wrapper has no model of its own
    code = live_run(found, registry, watcher, args)

    if not args.no_cache:
        print(f"  cache now holds {DiskStore(args.cache_db).count()} answers")
    return code


def list_models(args) -> int:
    """What a provider actually serves. Names retire without notice, and a
    stale one fails as a 404 that reads like a broken endpoint."""
    from resint.model.openai_compat import PROVIDERS, OpenAICompatProvider

    if not args.provider:
        print("known providers: " + ", ".join(sorted(PROVIDERS)))
        print("give --provider to list one's models")
        return 0

    provider = OpenAICompatProvider(
        model="?", provider=args.provider, base_url=args.base_url
    )
    if not provider.configured:
        _, env_var = PROVIDERS.get(args.provider, ("", ""))
        print(f"{args.provider} is not configured -- set {env_var}", file=sys.stderr)
        return 2

    names = provider.models()
    if not names:
        print(f"{args.provider} returned no model list", file=sys.stderr)
        return 1
    print(f"{args.provider} serves {len(names)} models:")
    print()
    for name in names:
        print(f"  {name}")
    return 0


def dry_run(found, registry, args) -> int:
    """What would be sent, and whether it looks like a usable prompt."""
    print(f"dry run over {len(found)} papers -- no model, no network\n")

    totals = Counter()
    problems = []

    for path in found:
        capture = Capturing()
        try:
            paper, _, ctx = run_paper(path, registry, capture, args.repo)
        except UnreadableInput as exc:
            print(f"  {path.name:<24} unreadable: {exc}")
            continue

        print(f"  {path.name}")
        if not capture.requests:
            print("    (no model rule assembled a request)")
            continue

        for request in capture.requests:
            facts = describe(request)
            totals["requests"] += 1
            totals["tokens"] += facts["approx_tokens"]

            flags = []
            if facts["preamble"]:
                flags.append("PREAMBLE")
                problems.append(
                    (path.name, facts["rule"], f"preamble in prompt: {facts['preamble']}")
                )
            if facts["approx_tokens"] > 12_000:
                flags.append("LARGE")
                problems.append((path.name, facts["rule"], "prompt is very large"))
            if facts["approx_tokens"] < 100:
                flags.append("TINY")
                problems.append((path.name, facts["rule"], "prompt is nearly empty"))

            print(
                f"    {facts['rule']:<22} ~{facts['approx_tokens']:>6,} tok"
                f"  tables={'y' if facts['has_tables'] else 'n'}"
                f"  {' '.join(flags)}"
            )
            if args.show:
                body = request.user[: args.show].replace("\n", "\n      ")
                print(f"      {body}\n")

    print(f"\n  {totals['requests']} requests, ~{totals['tokens']:,} tokens total")
    if totals["requests"]:
        print(f"  ~{totals['tokens'] // totals['requests']:,} tokens per request")

    if problems:
        print(f"\n  {len(problems)} prompts look wrong:")
        for name, rule, why in problems[:12]:
            print(f"    {name:<22} {rule:<22} {why}")
        return 1

    print("\n  prompts look sane. Run again with --provider to test a real model.")
    return 0


def live_run(found, registry, watcher, args) -> int:
    """The real thing, and the number that matters."""
    print(f"{watcher.model} over {len(found)} papers\n")

    verified = Counter()
    rejected: list[tuple[str, str, str]] = []
    failures: Counter = Counter()
    findings_total = 0

    for path in found:
        # Only the exchanges this paper produced. watcher.exchanges is
        # cumulative, and checking paper one's quotes against paper two's text
        # reports every one of them as a hallucination -- which is exactly
        # what it did, turning a clean run into a fake 62% failure rate.
        seen_before = len(watcher.exchanges)
        try:
            paper, findings, ctx = run_paper(path, registry, watcher, args.repo)
        except UnreadableInput as exc:
            print(f"  {path.name:<24} unreadable: {exc}")
            continue

        findings_total += len(findings)
        print(f"  {path.name}  {len(findings)} findings")
        for finding in findings:
            print(f"    {finding.severity.value:<5} {finding.rule_id}")
            print(f"      {finding.message[:150]}")
        for note in ctx.abstentions:
            print(f"    abstained: {note[:120]}")

        # The measurement. Quotes are checked against the paper directly
        # rather than trusting any rule to have done it.
        text = paper.text.content if paper.text else ""
        for exchange in watcher.exchanges[seen_before:]:
            answer = exchange["answer"]
            if not answer.usable:
                verified["unusable_replies"] += 1
                # Why, not just that. "the model did not answer" is what a
                # rule tells a user; a diagnostic tool has to say whether that
                # was a rate limit, a refusal or a broken reply.
                failures[answer.detail[:90] or answer.outcome.value] += 1
                continue
            verified["usable_replies"] += 1
            for quote in quotes_in(answer.payload):
                verified["quotes"] += 1
                found = locate(quote, text)
                verified[found.verdict.value] += 1
                if not found.usable:
                    # The whole point. A rate without the failing cases tells
                    # you there is a problem and nothing about what it is.
                    rejected.append((exchange["rule"], found.verdict.value, quote))

    print("\n  --- results " + "-" * 40)
    print(f"  replies usable      {verified['usable_replies']}")
    print(f"  replies unusable    {verified['unusable_replies']}")
    for detail, count in failures.most_common():
        print(f"      {count:>3} x {detail}")
    print(f"  findings            {findings_total}")

    quotes = verified["quotes"]
    if quotes:
        located = verified["located"]
        rate = located / quotes
        print(f"\n  quotes returned     {quotes}")
        print(f"  quotes verified     {located}  ({rate:.0%})")
        print(f"    absent            {verified['absent']}")
        print(f"    ambiguous         {verified['ambiguous']}")
        print(f"    too short         {verified['too-short']}")
        print(
            "\n  The verified rate is this model's hallucination rate on this "
            "task,\n  measured with no labels. Below about 90% the model is "
            "not usable here."
        )
    else:
        print("\n  no quotes returned at all -- check the prompts with --dry")

    if rejected:
        # A rate without the failing cases tells you there is a problem and
        # nothing whatever about what it is.
        print(f"\n  --- the {len(rejected)} quotes that did not verify " + "-" * 12)
        for rule, verdict, quote in rejected[:15]:
            print(f"\n  [{rule}] {verdict}")
            print(f"    {quote[:300]!r}")

    if args.record:
        target = Path(args.record)
        target.mkdir(parents=True, exist_ok=True)
        saved = {
            e["request"].cache_key(watcher.model): e["answer"].payload
            for e in watcher.exchanges
            if e["answer"].usable
        }
        out = target / f"{watcher.model.replace('/', '_')}.json"
        out.write_text(json.dumps(saved, indent=2), encoding="utf-8")
        print(f"\n  recorded {len(saved)} responses to {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
