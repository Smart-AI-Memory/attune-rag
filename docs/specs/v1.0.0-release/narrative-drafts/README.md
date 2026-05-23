# Release-narrative drafts — v1.0.0

> **Status: v1 full drafts, not commitments.** All three posts (plus
> the LinkedIn adaptation of B) are now first-draft complete. They
> sit here so the *shape and substance* are captured while the
> v1.0.0-release spec is still scaffolding. None of these ship until
> the spec is scoped, the narrative task is owned, and a polish pass
> resolves each draft's "Open questions for fleshing-out" tail.

## Why three posts

Post A (marquee) makes the counter-narrative claim. Post B
(methodology) earns the right to make that claim. Post C (BM25
deep-cut) closes the loop by explaining why most people think rerank
works in the first place. The trilogy *is* the v1.0.0 thesis:
*"We ship measurement, not opinion."*

## Suggested sequencing

| Relative to v1.0.0 cut | Post | Channel |
|---|---|---|
| Cut day | [A — Rerank measurement](A-rerank-measurement.md) | README hero + Show HN + r/LocalLLaMA |
| Cut + 7 days | [B — RAG methodology](B-rag-methodology.md) | Blog (canonical) + [LinkedIn adaptation](B-linkedin-adaptation.md) 24–48h later + cross-link from A |
| Cut + 21 days | [C — BM25 broken](C-bm25-broken.md) | Blog + r/LocalLLaMA + HN |

Spacing rationale: B too soon dilutes A's hook; C too soon looks
defensive. Three staggered drops = three visibility moments, not one.

## Draft maturity

| Draft | State | Words | Notes |
|---|---|---|---|
| [A — Rerank measurement](A-rerank-measurement.md) | v1 full draft (2026-05-22) | ~1570 | Grounded in D5 diagnostic-1 receipts. Hero table from the verdict-lock. |
| [B — RAG methodology](B-rag-methodology.md) | v1 full draft (2026-05-22) | ~2320 | Grounded in locked v2 baseline at commit `6fbe6d7`. Four-levers framing + σ=3.0→2.0 walkback. |
| [B — LinkedIn adaptation](B-linkedin-adaptation.md) | v1 full draft (2026-05-22) | ~920 body | Founder-voice, no code blocks, provocation CTA. Companion to canonical B. |
| [C — BM25 broken](C-bm25-broken.md) | v1 full draft (2026-05-22) | ~1980 | Grounded in alias-expansion-sweep memory + project arc. Five diagnostic patterns + the M12 near-regression. |

All v1 drafts. Polish-pass + scoping-pinned owner/channel still
required before publication. Every draft carries its own
"Open questions for fleshing-out" tail naming the calls a human
needs to make.

## Originating context

- Discussion: perf-baseline-multi-run M3 wrap, 2026-05-22.
- Evidence base: reranker-evaluation verdict lock (#133–#136),
  perf-baseline-multi-run v2 schema (#131/#137/#139/#142),
  alias-expansion-sweep arc lessons.
- Pre-scoping flag: [../tasks.md](../tasks.md) §"Scoping-time
  considerations (flagged 2026-05-22)".
