"""Command-line entry point for debugging retrieval.

Usage:

    attune-rag query "how do I run a security audit?"
    attune-rag query "..." --corpus-path ./my-docs
    attune-rag query "..." --retriever hybrid
    attune-rag query "..." --min-score 5
    attune-rag query "..." --provider claude
    attune-rag corpus-info
    attune-rag corpus-info --corpus-path ./my-docs

Uses argparse so the CLI works with only the core package
installed (no typer / click dependency).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

from .pipeline import RagPipeline
from .prompts import PROMPT_VARIANTS
from .providers import list_available

# C0/C1 control characters EXCEPT tab (\t), newline (\n), and carriage
# return (\r). Used to scrub error messages before they reach the
# terminal so a path or filename containing raw ANSI escapes can't
# rewrite the surrounding output.
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def _safe_stderr(msg: str) -> str:
    """Strip terminal control characters from ``msg`` before printing."""
    return _CTRL_RE.sub("?", msg)


def _build_corpus(args: argparse.Namespace):
    """Resolve ``--corpus-path`` to a :class:`DirectoryCorpus`.

    Returns ``None`` when the flag is absent so the pipeline
    falls back to the bundled default corpus. Raises
    ``ValueError`` (from the ``DirectoryCorpus`` constructor)
    when the path is not an existing directory.
    """
    corpus_path = getattr(args, "corpus_path", None)
    if corpus_path is None:
        return None
    from .corpus import DirectoryCorpus

    return DirectoryCorpus(Path(corpus_path).expanduser())


def _build_retriever(args: argparse.Namespace):
    """Resolve ``--retriever`` / ``--min-score`` to a retriever instance.

    Returns ``None`` for the plain keyword default so the
    pipeline constructs its own. ``--min-score`` is an absolute
    keyword score, so it only composes with the keyword
    retriever — the hybrid and transformer tiers have no
    abstention threshold yet (see the safe-abstention-defaults
    spec); combining the flags raises ``ValueError``.
    """
    name = getattr(args, "retriever", "keyword")
    min_score = getattr(args, "min_score", None)
    if name == "keyword":
        if min_score is None:
            return None
        from .retrieval import KeywordRetriever

        return KeywordRetriever(min_score=min_score)
    if min_score is not None:
        raise ValueError(
            f"--min-score only applies to --retriever keyword (an absolute "
            f"keyword score has no meaning for the {name!r} retriever, and "
            f"the hybrid/transformer tiers do not support abstention yet)."
        )
    if name == "hybrid":
        from .hybrid import HybridRetriever

        return HybridRetriever()
    from .transformer import TransformerRetriever

    return TransformerRetriever()


def _cmd_query(args: argparse.Namespace) -> int:
    """Handle ``attune-rag query``.

    Two paths:

    - With ``--provider``: runs the full RAG pipeline (retrieve
      + assemble prompt + call LLM) and prints the response,
      optionally with native claim-level citations.
    - Without ``--provider``: prints retrieval results only —
      either a human-readable hit list, or JSON when ``--json``
      is set.

    Returns 0 on success (including the "no grounding context
    found" fallback) and 2 on a predictable setup error —
    missing extra, bad ``--corpus-path``, conflicting flags —
    which surfaces as a one-line message instead of a
    traceback. Unexpected exceptions (e.g. provider API
    failures) still raise; surfacing those tracebacks is
    intentional.
    """
    try:
        pipeline = RagPipeline(
            corpus=_build_corpus(args),
            retriever=_build_retriever(args),
        )
        if args.provider:
            response, result = asyncio.run(
                pipeline.run_and_generate(
                    args.query,
                    provider=args.provider,
                    k=args.k,
                    model=args.model,
                    prompt_variant=args.prompt_variant,
                    use_native_citations=args.native_citations,
                )
            )
            if result.used_native_citations:
                from .provenance import format_claim_citations_markdown

                print(format_claim_citations_markdown(response, result.claim_citations))
            else:
                print(response)
                print()
                _print_citations(result)
            return 0

        result = pipeline.run(args.query, k=args.k, prompt_variant=args.prompt_variant)
    except (RuntimeError, ValueError) as exc:
        print(f"error: {_safe_stderr(str(exc))}", file=sys.stderr)
        return 2
    if args.json:
        payload = {
            "fallback_used": result.fallback_used,
            "confidence": result.confidence,
            "elapsed_ms": result.elapsed_ms,
            "retriever": result.citation.retriever_name,
            "hits": [
                {
                    "path": h.template_path,
                    "category": h.category,
                    "score": h.score,
                }
                for h in result.citation.hits
            ],
        }
        print(json.dumps(payload, indent=2))
        return 0

    if result.fallback_used:
        print(f"No grounding context found for: {args.query!r}")
        return 0

    print(f"Top {len(result.citation.hits)} hits for: {args.query!r}\n")
    for hit in result.citation.hits:
        print(f"  {hit.score:>6.2f}  {hit.category:<14}  {hit.template_path}")
    print()
    _print_citations(result)
    return 0


def _cmd_corpus_info(args: argparse.Namespace) -> int:
    """Handle ``attune-rag corpus-info``.

    Loads the corpus — ``--corpus-path`` as a
    :class:`DirectoryCorpus` when given, the bundled default
    otherwise — and prints its name, version, total entry
    count, and a per-category breakdown sorted by descending
    count then alphabetical category. Returns 0 on success, 2
    on a predictable setup error (missing extra, bad
    ``--corpus-path``).
    """
    try:
        pipeline = RagPipeline(corpus=_build_corpus(args))
        corpus = pipeline.corpus
        entries = list(corpus.entries())
    except (RuntimeError, ValueError) as exc:
        print(f"error: {_safe_stderr(str(exc))}", file=sys.stderr)
        return 2
    categories: dict[str, int] = {}
    for entry in entries:
        categories[entry.category] = categories.get(entry.category, 0) + 1
    print(f"Corpus:  {corpus.name}")
    print(f"Version: {corpus.version}")
    print(f"Entries: {len(entries)}")
    print("Categories:")
    for cat, count in sorted(categories.items(), key=lambda kv: (-kv[1], kv[0])):
        label = cat or "(root)"
        print(f"  {label:<20} {count}")
    return 0


def _cmd_providers(args: argparse.Namespace) -> int:
    """Handle ``attune-rag providers``.

    Prints which provider extras are currently installed (e.g.
    ``claude``, ``gemini``). When nothing is installed, prints
    a one-line install hint. Returns 0 in both cases; non-zero
    exits are reserved for actual failures, not for "empty
    list."
    """
    _ = args
    available = list_available()
    if not available:
        print("No provider extras installed.")
        print("Install one: pip install 'attune-rag[claude]' (or gemini).")
        return 0
    print("Available providers:")
    for name in available:
        print(f"  - {name}")
    return 0


def _cmd_dashboard_render(args: argparse.Namespace) -> int:
    """Handle ``attune-rag dashboard render``.

    Builds a fresh snapshot for ``--corpus-package``, renders
    the packaged HTML template to ``--out``, and optionally
    opens the result in the system browser when ``--open`` is
    set. Returns 0 on success, 2 on a ``ValueError`` from
    snapshot building or rendering (typically a bad output
    path).
    """
    from .dashboard.refresh import build_snapshot
    from .dashboard.render import render

    out = Path(args.out).expanduser().resolve()
    try:
        snapshot = build_snapshot(corpus_package=args.corpus_package)
        render(out, snapshot, title=args.title)
    except ValueError as exc:
        print(f"error: {_safe_stderr(str(exc))}", file=sys.stderr)
        return 2
    print(f"Dashboard written to {out}")
    if args.open:
        import webbrowser

        webbrowser.open(out.as_uri())
    return 0


def _cmd_dashboard_show(args: argparse.Namespace) -> int:
    """Handle ``attune-rag dashboard show``.

    Thin shim over :func:`attune_rag.dashboard.show.main` —
    builds a snapshot for ``--corpus-package`` and pretty-prints
    it to the terminal via Rich. Returns the inner ``main``'s
    exit code unchanged.
    """
    from .dashboard.show import main as _show

    return _show(args.corpus_package)


def _cmd_dashboard_refresh(args: argparse.Namespace) -> int:
    """Handle ``attune-rag dashboard refresh``.

    Thin shim over :func:`attune_rag.dashboard.refresh.main` —
    rebuilds the snapshot for ``--corpus-package`` and writes it
    to the corpus's standard snapshot path. Returns the inner
    ``main``'s exit code unchanged.
    """
    from .dashboard.refresh import main as _refresh

    return _refresh(args.corpus_package)


def _print_citations(result) -> None:
    from .provenance import format_citations_markdown

    print(format_citations_markdown(result.citation))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="attune-rag",
        description="Retrieve grounded context from a corpus and optionally call an LLM.",
    )
    subs = parser.add_subparsers(dest="cmd", required=True)

    query = subs.add_parser("query", help="Run retrieval for a query.")
    query.add_argument("query", help="Question or request to ground.")
    query.add_argument("-k", type=int, default=3, help="Max hits to return (default 3).")
    query.add_argument(
        "--corpus-path",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Query a directory of markdown files (DirectoryCorpus) instead "
            "of the bundled default corpus."
        ),
    )
    query.add_argument(
        "--retriever",
        choices=["keyword", "hybrid", "transformer"],
        default="keyword",
        help=(
            "Retrieval strategy (default: keyword). 'hybrid' fuses keyword "
            "+ static embeddings via RRF (requires the [embeddings] extra; "
            "falls back to keyword-only without it). 'transformer' ranks "
            "with a sentence-transformers model (requires the heavyweight "
            "[transformers] extra)."
        ),
    )
    query.add_argument(
        "--min-score",
        type=float,
        default=None,
        metavar="SCORE",
        help=(
            "Abstention threshold (keyword retriever only): drop hits "
            "scoring below SCORE; when nothing clears it, report 'no "
            "grounding context' instead of a weak match. The score is "
            "corpus-specific — calibrate with "
            "'attune-rag-benchmark --calibrate-abstention'."
        ),
    )
    query.add_argument(
        "--prompt-variant",
        choices=sorted(PROMPT_VARIANTS),
        default="citation",
        help=(
            "Prompt template for the augmented prompt (default: citation, "
            "selected via A/B sweep for the lowest hallucination rate)."
        ),
    )
    query.add_argument(
        "--provider",
        choices=["claude", "gemini"],
        help="If set, call the named LLM and print its response.",
    )
    query.add_argument(
        "--model",
        default=None,
        help="Override the provider's default model name.",
    )
    query.add_argument(
        "--json",
        action="store_true",
        help="Emit hits as JSON instead of the human-readable format.",
    )
    query.add_argument(
        "--native-citations",
        action="store_true",
        help=(
            "Use the Anthropic Citations API for claim-level "
            "attribution (Claude provider only). Falls back to "
            "the prompt-assembly path on providers that don't "
            "support it."
        ),
    )
    query.set_defaults(func=_cmd_query)

    info = subs.add_parser(
        "corpus-info",
        help="Print stats about a corpus (bundled default or --corpus-path).",
    )
    info.add_argument(
        "--corpus-path",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Inspect a directory of markdown files (DirectoryCorpus) "
            "instead of the bundled default corpus."
        ),
    )
    info.set_defaults(func=_cmd_corpus_info)

    provs = subs.add_parser("providers", help="List LLM providers whose extras are installed.")
    provs.set_defaults(func=_cmd_providers)

    dash = subs.add_parser("dashboard", help="Render or refresh the attune-rag Cowork dashboard.")
    dash_subs = dash.add_subparsers(dest="dashboard_cmd", required=True)

    show_p = dash_subs.add_parser(
        "show", help="Run benchmark and display dashboard in the terminal."
    )
    show_p.add_argument(
        "--corpus-package",
        default="attune_help",
        metavar="NAME",
        help="Corpus package name (default: attune_help).",
    )
    show_p.set_defaults(func=_cmd_dashboard_show)

    render_p = dash_subs.add_parser(
        "render", help="Run benchmark, embed snapshot, write dashboard HTML."
    )
    render_p.add_argument("--out", required=True, metavar="PATH", help="Destination file path.")
    render_p.add_argument(
        "--corpus-package",
        default="attune_help",
        metavar="NAME",
        help="Corpus package name (default: attune_help).",
    )
    render_p.add_argument(
        "--title",
        default="attune-rag dashboard",
        metavar="TEXT",
        help="Dashboard title (default: 'attune-rag dashboard').",
    )
    render_p.add_argument(
        "--open",
        action="store_true",
        help="Open the rendered file in the default browser.",
    )
    render_p.set_defaults(func=_cmd_dashboard_render)

    refresh_p = dash_subs.add_parser("refresh", help="Emit a JSON snapshot to stdout.")
    refresh_p.add_argument(
        "--corpus-package",
        default="attune_help",
        metavar="NAME",
        help="Corpus package name (default: attune_help).",
    )
    refresh_p.set_defaults(func=_cmd_dashboard_refresh)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
