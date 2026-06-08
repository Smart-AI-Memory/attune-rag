"""V2 verification: cache_control on document blocks (Citations API).

Submits the same batch of citation documents twice; second call
should hit the prompt cache if document-block caching works the
same as text-block caching. Reports cache_creation_input_tokens
+ cache_read_input_tokens from each call's usage.

Run:

    ANTHROPIC_API_KEY=sk-ant-... python scripts/probe_v2_cache_control.py

Cost: ~$0.01 (two ~1500-token-input calls on Sonnet).
"""

from __future__ import annotations

import os
import sys
import time

# Build a system prompt + document corpus that's at least 1024 tokens
# so cache_control actually triggers on Sonnet (the threshold below
# which Anthropic doesn't cache).
LONG_SYSTEM = (
    "You are answering questions strictly from the provided documents.\n"
    "Cite the source document for every factual claim.\n\n"
) * 4  # ~200 tokens

# Each document is ~600 tokens of repeated technical prose so
# the doc payload alone clears the caching floor.
LARGE_DOC_BODY = (
    "The Anthropic Citations API allows the model to attach "
    "structured citations to specific spans of its response. "
    "Each citation references a document and a location range "
    "within that document. For custom_content sources, the "
    "location is reported as a content_block_location with "
    "start_block_index and end_block_index pointers. "
) * 50  # ~2000 tokens, well above caching floor

QUERY = "Summarize the citations behavior in one sentence."


def _make_documents() -> list[dict]:
    """Two documents, first one carrying ``cache_control``."""
    docs: list[dict] = []
    for i, title in enumerate(
        ["concepts/citations-overview.md", "concepts/citations-locations.md"]
    ):
        block = {
            "type": "document",
            "source": {
                "type": "content",
                "content": [{"type": "text", "text": LARGE_DOC_BODY}],
            },
            "title": title,
            "citations": {"enabled": True},
        }
        if i == 0:
            block["cache_control"] = {"type": "ephemeral"}
        docs.append(block)
    return docs


def _call(client, docs: list[dict], label: str) -> dict:
    t0 = time.perf_counter()
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=128,
        messages=[
            {
                "role": "user",
                "content": docs + [{"type": "text", "text": QUERY}],
            }
        ],
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    usage = resp.usage
    print(f"--- {label} ---")
    print(f"  input_tokens:                {getattr(usage, 'input_tokens', '?')}")
    print(f"  output_tokens:               {getattr(usage, 'output_tokens', '?')}")
    print(f"  cache_creation_input_tokens: {getattr(usage, 'cache_creation_input_tokens', 0) or 0}")
    print(f"  cache_read_input_tokens:     {getattr(usage, 'cache_read_input_tokens', 0) or 0}")
    print(f"  elapsed:                     {elapsed_ms:.0f} ms")
    return {
        "creation": getattr(usage, "cache_creation_input_tokens", 0) or 0,
        "read": getattr(usage, "cache_read_input_tokens", 0) or 0,
    }


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("error: ANTHROPIC_API_KEY not set", file=sys.stderr)
        return 2
    from anthropic import Anthropic

    client = Anthropic()
    docs = _make_documents()

    first = _call(client, docs, "first call (priming the cache)")
    print()
    second = _call(client, docs, "second call (should read cache)")

    print()
    print("=== verdict ===")
    if second["read"] > 0:
        print(
            f"PASS: cache_control on document block produced a hit "
            f"({second['read']} cached tokens read on second call)."
        )
        print(
            "ACTION: wire cache_control onto first document in "
            "_build_documents_payload (default behavior)."
        )
        return 0
    if first["creation"] > 0 and second["read"] == 0:
        print("MIXED: first call wrote a cache entry but second didn't read it.")
        print("ACTION: investigate — possible TTL or invalidation issue.")
        return 1
    print(
        "FAIL: no cache activity. Document-block caching may not work the "
        "same as text-block caching for the citations API."
    )
    print("ACTION: leave cache_control OFF on the citations path (current default).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
