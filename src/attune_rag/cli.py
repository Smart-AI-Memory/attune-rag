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
import sys

from .pipeline import RagPipeline
from .providers import list_available


def _cmd_query(args: argparse.Namespace) -> int:
    pipeline = RagPipeline()
    if args.provider:
        response, result = asyncio.run(
            pipeline.run_and_generate(
                args.query,
                provider=args.provider,
                k=args.k,
                model=args.model,
            )
        )
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
    _ = args
    available = list_available()
    if not available:
        print("No provider extras installed.")
        print("Install one: pip install 'attune-rag[claude]' (or openai, gemini).")
        return 0
    print("Available providers:")
    for name in available:
        print(f"  - {name}")
    return 0


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
        choices=["claude", "openai", "gemini"],
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
    query.set_defaults(func=_cmd_query)

    info = subs.add_parser("corpus-info", help="Print stats about the default corpus.")
    info.set_defaults(func=_cmd_corpus_info)

    provs = subs.add_parser("providers", help="List LLM providers whose extras are installed.")
    provs.set_defaults(func=_cmd_providers)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
