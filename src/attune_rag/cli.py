"""Command-line entry point for debugging retrieval.

Usage:

    attune-rag query "how do I run a security audit?"
    attune-rag query "..." --provider claude
    attune-rag corpus-info

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
from .providers import list_available

# C0/C1 control characters EXCEPT tab (\t), newline (\n), and carriage
# return (\r). Used to scrub error messages before they reach the
# terminal so a path or filename containing raw ANSI escapes can't
# rewrite the surrounding output.
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def _safe_stderr(msg: str) -> str:
    """Strip terminal control characters from ``msg`` before printing."""
    return _CTRL_RE.sub("?", msg)


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
    found" fallback). Does not catch exceptions from the
    pipeline or providers; surfacing the traceback is
    intentional.
    """
    pipeline = RagPipeline()
    if args.provider:
        response, result = asyncio.run(
            pipeline.run_and_generate(
                args.query,
                provider=args.provider,
                k=args.k,
                model=args.model,
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

    result = pipeline.run(args.query, k=args.k)
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

    Loads the default corpus (currently :class:`AttuneHelpCorpus`)
    and prints its name, version, total entry count, and a
    per-category breakdown sorted by descending count then
    alphabetical category. ``args`` is unused — the command
    takes no flags. Returns 0.
    """
    _ = args
    pipeline = RagPipeline()
    corpus = pipeline.corpus
    entries = list(corpus.entries())
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

    info = subs.add_parser("corpus-info", help="Print stats about the default corpus.")
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
