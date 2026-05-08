# Native Anthropic Citations API in attune-rag

> Design + decision note for the opt-in native citations path
> introduced in 0.1.13. Tracks the spec at
> `specs/rag-native-citations/` (workspace umbrella).

## TL;DR

attune-rag now has **two parallel grounding paths**:

1. **Legacy `[P{n}]` path** (default): retrieved hits are inlined
   into the augmented prompt as numbered passages; the model is
   *instructed* to cite as `[P1]`, `[P2]`. Soft, training-anchored
   convention. Selected via 2026-04-19 A/B sweep that dropped
   hallucination 46.7% → 6.7%.
2. **Native citations path** (opt-in): retrieved hits become
   `custom_content` document blocks on the Anthropic Messages
   API. The model emits structured citations attached to its
   response text — claim-level, character/block-precise,
   programmatically verifiable.

The legacy path is **kept intact** because its `[P{n}] source:`
header format is anchored to the model's citation training (an
A/B with `id="P{n}"` XML tripled per-claim hallucination). The
new path is **additive and opt-in**, default off.

## When to use which

| Use case | Path | Reason |
|---|---|---|
| You don't ship Claude (Gemini-only) | Legacy | Native API is Claude-only |
| You want a single rendered prompt for caching, eval, audit logs | Legacy | The citations path doesn't produce a single rendered prompt |
| You want claim-level attribution with character precision | Native | Each sentence carries a structured pointer |
| You want a future faithfulness eval to programmatically verify "did the model actually use what it said it used?" | Native | Native citations are model-asserted; verifier can check `cited_text` against the source |
| You need the lowest token cost | Legacy | Native sends documents as separate blocks (slightly more overhead) |

## API surface

```python
from attune_rag import RagPipeline, format_claim_citations_markdown

pipeline = RagPipeline()
response, result = await pipeline.run_and_generate(
    "How does the security audit pipeline work?",
    provider="claude",
    use_native_citations=True,    # NEW in 0.1.13
)

if result.used_native_citations:
    print(format_claim_citations_markdown(response, result.claim_citations))
else:
    print(response)  # legacy path returned plain text
```

`RagResult` gains two fields (both default-empty so existing
callers are unaffected):

- `claim_citations: tuple[ClaimCitation, ...]` — model-emitted
  per-claim attribution.
- `used_native_citations: bool` — whether the citations API was
  actually called for this run (False on the legacy path, on
  empty-hit fallbacks, and on Gemini-fallback).

## Fallback behavior

| `use_native_citations` | provider supports? | hits empty? | what runs |
|---|---|---|---|
| False | — | — | legacy prompt-assembly path |
| True | True | False | native citations path |
| True | True | True | fallback prompt + plain `generate` (no docs to cite) |
| True | False | — | warning logged, legacy path runs |

`used_native_citations` is True only on row 2.

## Caching

Caching is **off** on the native path in v1. The legacy path
continues to flag the stable prompt prefix with
`cache_control: ephemeral`. Document-block caching needs the V2
verification gate (an empirical 2-call test that confirms
document-block caching behaves the same as text-block caching);
once confirmed, attach `cache_control` to the first document.

## Document-count ceiling

`MAX_CITATION_DOCUMENTS = 20` is enforced by `ClaudeProvider`.
Exceeding it raises `ValueError` with a clean message. The
ceiling will be re-verified by the V3 gate before the default
flips. Today this is well above the project's `k=3` retrieval
default.

## Benchmark

```sh
attune-rag-benchmark \
  --with-faithfulness \
  --native-citations \
  --min-faithfulness 0.85
```

This runs the full faithfulness sweep twice (once each path) and
prints a side-by-side table showing mean faithfulness, refusal
rate, hallucination rate, citation emit rate, and p95 latency.
The decision to ever flip the default belongs to a follow-up
spec citing the resulting CSV.

The benchmark gates on the **legacy** path's faithfulness floor
because that's the established baseline; native is exploratory.

## Open verification gates (V2, V3)

These need real API calls and were not run in the implementing
PR. They affect optional polish, not correctness:

- **V2 — `cache_control` on document blocks.** Empirically
  confirm a 2-call test yields cache hits when documents are
  identical. If yes, wire `cache_control: ephemeral` onto the
  first document in `_build_documents_payload`.
- **V3 — document-count ceiling.** Confirm 20 is still the
  per-request cap. If higher, raise `MAX_CITATION_DOCUMENTS`.

Findings should land in this doc as a follow-up commit.

## Why not replace the legacy path?

Three reasons:

1. **Provider parity.** Gemini doesn't have a compatible API.
   Replacing the legacy path entirely would cut Gemini callers
   off from any grounding behavior.
2. **Training anchor.** The `[P{n}] source:` format is anchored
   in the model's citation training. We have data showing
   variants regress hallucination. Native citations are a
   *different* mode (the model isn't writing markers, the API
   is extracting cited spans). It's not strictly better
   everywhere — we need the benchmark.
3. **Opt-in lets us A/B.** A default flip is a separate decision
   that should be evidence-driven. The opt-in kwarg keeps the
   blast radius small.

## Rollback

Soft rollback is "callers stop passing `use_native_citations=True`"
— no release needed. See `specs/rag-native-citations/tasks.md`
for the full rollback ladder.
