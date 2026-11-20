"""Offline grounding journey using a temporary synthetic corpus and a stub provider.

Run after installing the candidate wheel: python examples/grounding_evidence.py
No API credentials, provider SDK, network calls, or model downloads are needed.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

from attune_rag import RagPipeline
from attune_rag.corpus import DirectoryCorpus
from attune_rag.eval.reporting import summarize_answers
from attune_rag.providers.base import CitedResponse


class DemoProvider:
    """Return fixed text to illustrate transport, not live answer quality."""

    name = "offline-demo"
    supports_native_citations = True

    def __init__(self):
        self.plain_calls = 0
        self.native_calls = 0

    async def generate(self, prompt, **kwargs):
        self.plain_calls += 1
        return "This is an ungrounded demonstration fallback."

    async def generate_with_citations(self, documents, query, system=None, **kwargs):
        self.native_calls += 1
        return CitedResponse("The scanner inspects files.", ())


async def main():
    with TemporaryDirectory(prefix="attune-rag-grounding-") as temporary:
        corpus_root = Path(temporary)
        source = "\nThe scanner inspects files.\n  "
        (corpus_root / "scanner.md").write_text(source, encoding="utf-8")
        pipeline = RagPipeline(corpus=DirectoryCorpus(corpus_root))

        # This named route needs neither SDK nor credentials: no hits in strict mode.
        refusal, strict = await pipeline.run_and_generate(
            "galactic pineapple orbit",
            "claude",
            prompt_variant="strict",
            use_native_citations=True,
        )
        stub = DemoProvider()
        fallback, non_strict = await pipeline.run_and_generate(
            "galactic pineapple orbit",
            stub,
            prompt_variant="baseline",
        )
        answer, grounded = await pipeline.run_and_generate(
            "scanner",
            stub,
            prompt_variant="strict",
            use_native_citations=True,
        )

        # Fixed unsupported-claim fixture. It does not run a judge.
        fixture = {
            "id": "synthetic-unsupported",
            "status": "completed",
            "answer": "The scanner inspects and deletes files.",
            "context": grounded.context,
            "supported_claims": ["The scanner inspects files."],
            "unsupported_claims": ["The scanner deletes files."],
            "reasoning": "The supplied source says nothing about deletion.",
            "score": 0.5,
            "abstained": False,
        }
        print(
            json.dumps(
                {
                    "strict": {
                        "answer": refusal,
                        "abstained": strict.abstained,
                        "fallback_used": strict.fallback_used,
                        "context": strict.context,
                    },
                    "fallback": {
                        "answer": fallback,
                        "abstained": non_strict.abstained,
                        "fallback_used": non_strict.fallback_used,
                    },
                    "grounded": {
                        "answer": answer,
                        "context": grounded.context,
                        "generation_documents": [asdict(d) for d in grounded.generation_documents],
                        "used_native_citations": grounded.used_native_citations,
                    },
                    "stub_calls": {"plain": stub.plain_calls, "native": stub.native_calls},
                    "fixed_report_fixture": {
                        "per_query": [fixture],
                        "summary": summarize_answers([fixture], 1),
                    },
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    asyncio.run(main())
