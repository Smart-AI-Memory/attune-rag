# Release-narrative drafts — v1.0.0

> **Status: drafts, not commitments.** These are skeleton outlines
> for the three-post release narrative flagged in
> [../tasks.md](../tasks.md) "Scoping-time considerations" on
> 2026-05-22. They sit here so the *shape* is captured while the
> v1.0.0-release spec is still scaffolding. None of these ship until
> the spec is scoped and the narrative task is owned.

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

## Skeleton convention

Each draft is headers + 1–2 sentences per section. Enough to know
what each section argues; not enough to ship. Fleshing out happens
during M2 of v1.0.0-release after scoping pins owner and channel.

## Originating context

- Discussion: perf-baseline-multi-run M3 wrap, 2026-05-22.
- Evidence base: reranker-evaluation verdict lock (#133–#136),
  perf-baseline-multi-run v2 schema (#131/#137/#139/#142),
  alias-expansion-sweep arc lessons.
- Pre-scoping flag: [../tasks.md](../tasks.md) §"Scoping-time
  considerations (flagged 2026-05-22)".
