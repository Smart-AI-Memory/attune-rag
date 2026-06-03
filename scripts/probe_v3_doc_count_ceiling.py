"""V3 verification: per-request document-count ceiling.

The current code hardcodes ``MAX_CITATION_DOCUMENTS = 20`` in
``ClaudeProvider`` based on a conservative recall. This probe
walks the count up until Anthropic refuses (or until it gets to
a configurable max), so we can pin the real ceiling.

Run:

    ANTHROPIC_API_KEY=sk-ant-... python scripts/probe_v3_doc_count_ceiling.py

Strategy: bisect upward in chunks (5, 10, 20, 50, 100). On the
first 4xx that mentions a document limit, log the threshold and
stop. We use ``max_tokens=8`` to keep cost minimal — each call
generates almost nothing.

Cost: ~$0.01–$0.10 depending on how high we walk.
"""

from __future__ import annotations

import os
import sys


def _doc(i: int) -> dict:
    return {
        "type": "document",
        "source": {
            "type": "content",
            "content": [{"type": "text", "text": f"Document number {i}: short body."}],
        },
        "title": f"doc-{i}.md",
        "citations": {"enabled": True},
    }


def _try(client, n: int) -> tuple[bool, str]:
    """Return (accepted, error_message)."""
    try:
        client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=8,
            messages=[
                {
                    "role": "user",
                    "content": [_doc(i) for i in range(n)] + [{"type": "text", "text": "ok"}],
                }
            ],
        )
        return True, ""
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("error: ANTHROPIC_API_KEY not set", file=sys.stderr)
        return 2
    from anthropic import Anthropic

    client = Anthropic()

    # Probe ladder: small enough to be cheap, dense enough to find
    # the real cap to within 5–10 documents. Stops on the first
    # rejection.
    candidates = [5, 10, 20, 30, 50, 75, 100, 150, 200]
    last_ok = 0
    failed_at: int | None = None
    fail_msg = ""

    for n in candidates:
        print(f"trying n={n}...", end=" ", flush=True)
        ok, msg = _try(client, n)
        if ok:
            print("ACCEPTED")
            last_ok = n
            continue
        print("REJECTED")
        print(f"  reason: {msg[:200]}")
        failed_at = n
        fail_msg = msg
        break

    print()
    print("=== verdict ===")
    print(f"highest accepted: n = {last_ok}")
    if failed_at is None:
        print(f"never rejected up to n = {candidates[-1]}.")
        print(
            f"ACTION: raise MAX_CITATION_DOCUMENTS to {candidates[-1]} "
            "(conservative; the real cap is higher)."
        )
    else:
        print(f"first rejected:   n = {failed_at}")
        if "document" in fail_msg.lower() or "limit" in fail_msg.lower():
            print(
                f"ACTION: set MAX_CITATION_DOCUMENTS to {last_ok} "
                "(or somewhere in the gap; bisect within if you want a precise number)."
            )
        else:
            print("WARNING: rejection wasn't an obvious document-count error.")
            print("Check the reason above before adjusting the cap.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
