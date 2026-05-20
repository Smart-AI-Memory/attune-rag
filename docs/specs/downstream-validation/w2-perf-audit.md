# W2.2 — Mid-phase performance audit

> Per [`docs/specs/downstream-validation/tasks.md`](./tasks.md) W2.2.
> `attune-ai:performance_audit` over retrieval + reranker hot paths,
> cross-checked against [`perf-baseline.md`](./perf-baseline.md).
> Date: 2026-05-20. Baseline commit: `28a3293` (perf-baseline lock,
> N=50, σ=3.0, `include_llm=true`).
>
> Backfills the missing audit doc referenced by
> [`docs/specs/phase-5-backlog/items.md`](../phase-5-backlog/items.md)
> ("items surfaced during the audit but not captured in a committed
> doc; sourced from the Phase-5 capture prompt (2026-05-20)"). The
> items themselves are unchanged — this document is their source of
> truth.

## Scope

| Module | LOC | Function under audit |
|---|---:|---|
| `src/attune_rag/retrieval.py` | 353 | `KeywordRetriever.retrieve`, `._score_entry`, `._entry_field_tokens`, `._category_weight` |
| `src/attune_rag/reranker.py` | 144 | `LLMReranker.rerank` |
| **Total** | **497** | |

Cross-checked against the locked perf baseline at
[`perf-baseline.md`](./perf-baseline.md) (N=50, σ=3.0,
`include_llm=true`, commit `28a3293`).

## Health snapshot

| Pass | Result | Blocking? |
|---|---|---|
| Retrieval hot path | 4 micro-opt candidates, all behaviour-equivalent | No |
| Reranker hot path | 0 candidates — code shape already optimal; latency budget dominated by Anthropic round-trip | No |
| Baseline cross-check | All metrics inside thresholds; no candidate would shift a metric outside σ=3.0 noise on the current golden set | No |

W2.2 gate: **green.** No findings need to land during the freeze. All
four perf candidates captured to
[`docs/specs/phase-5-backlog/items.md` § Perf](../phase-5-backlog/items.md#perf-w22-perf-audit)
as items P1–P4 for the post-freeze 0.2.x → 0.3.0 perf-only PR.

## Baseline cross-check

Per-metric thresholds from [`perf-baseline.md`](./perf-baseline.md):

| Metric | Mean (s) | σ × 3 (s) | Threshold (s) | Audit observation |
|---|---:|---:|---:|---|
| `keyword_retriever_retrieve.cpu` | 0.002079 | 0.036405 | 0.038484 | Stdev dominated by JIT/import warm-up of the first run; steady-state is well below mean. Headroom is plentiful. |
| `keyword_retriever_retrieve.wall` | 0.002079 | 0.036405 | 0.038484 | Same — CPU-bound, no I/O on the steady path. |
| `rag_pipeline_run.cpu` | 0.000630 | 0.000169 | 0.000799 | Pipeline composition cost is negligible relative to the retriever it calls. |
| `rag_pipeline_run.wall` | 0.000630 | 0.000169 | 0.000799 | Same. |
| `llm_reranker_rerank.cpu` | 0.012409 | 0.214206 | 0.226615 | CPU cost is JSON parse + Python orchestration; tiny. Wide σ is harmless because the gate is one-sided (regression-only). |
| `llm_reranker_rerank.wall` | 0.799094 | 1.562608 | 2.361702 | Wall-clock is dominated by the Anthropic round-trip. σ=3.0 absorbs network jitter; this metric stays advisory through W3 by spec design. |
| `directory_corpus_load.cpu` / `.wall` | 0.000045 | 0.000021 | 0.000066 | Trivial. |

**No metric is close to threshold; no candidate found in this audit
would shift any metric outside the σ=3.0 noise band on the current
golden set.** This is *why* every candidate below is freeze-deferred:
the gate can't tell the difference, so there's no signal to validate
the change against during the burn-in.

## Retrieval hot path

Findings are micro-optimisations on a path that's already CPU-bound,
already memoised at the right grain (`_entry_field_tokens` /
`_related_summary_tokens` cache on the entry itself), and already
gated by `MIN_SCORE` for short-circuit semantics. All four are
behaviour-equivalent.

### Findings (effort-sorted)

1. **[P1 — Lazy match-reason build]**
   [`retrieval.py::_score_entry`](../../../src/attune_rag/retrieval.py)
   — the `reasons: list[str]` and `"+".join(reasons)` at lines 300–313
   build the human-readable match-reason string for *every* entry,
   then `retrieve` discards the score (and the reason) for any entry
   below `MIN_SCORE` (line 349). Most corpus entries fall below
   `MIN_SCORE` for any given query; the reason string is built and
   thrown away. **Fix:** defer reason construction — either return a
   lambda / `functools.partial` that builds the string on demand, or
   inline the `MIN_SCORE` check inside `_score_entry` and skip reason
   construction when `score < MIN_SCORE`. The second is simpler and
   matches the existing one-shot dropout contract.

2. **[P2 — Class-level cache-key constant]**
   [`retrieval.py::_entry_field_tokens`](../../../src/attune_rag/retrieval.py)
   line 235 — `cache_key = ("field_tokens", self.CONTENT_PREVIEW_CHARS)`
   allocates a fresh tuple on every call. **Fix:** promote to
   `_FIELD_TOKENS_CACHE_KEY: ClassVar[tuple[str, int]] = ("field_tokens", CONTENT_PREVIEW_CHARS)`.
   Sub-microsecond per call, but compounds across thousands of
   entries per retrieval.

3. **[P3 — `heapq.nlargest` for top-k]**
   [`retrieval.py::retrieve`](../../../src/attune_rag/retrieval.py)
   lines 352–353 — `scored.sort(key=...); return scored[:k]` is O(n
   log n); the natural top-k pattern is `heapq.nlargest(k, scored,
   key=...)` at O(n log k). **Fix:** straightforward swap; preserve
   the tie-break key `(-h.score, h.entry.path)`. **Caveat:** the
   asymptotic win does not show on the current golden-set corpus size
   (k=3, n≈100). Wait for the Phase-5 inter-run baseline (item M1) +
   a corpus-scaling perf scenario; otherwise the change will sit
   inside σ.

4. **[P4 — Inline `_category_weight`]**
   [`retrieval.py::_score_entry`](../../../src/attune_rag/retrieval.py)
   line 297 — `_category_weight` is a one-line wrapper around
   `CATEGORY_WEIGHTS.get(entry.category, DEFAULT_CATEGORY_WEIGHT)`.
   Method-call overhead is small but non-zero and the call sits on
   the hottest line of the score loop. **Fix:** inline the `.get(...)`
   directly in `_score_entry`. Subclasses that override
   `_category_weight` today (none in tree, none in attune-gui/help)
   would need to override `_score_entry` instead — acceptable trade
   for an internal method.

### Notable (not findings)

- **`_entry_field_tokens` memoisation is correct and well-keyed.**
  The cache key includes `CONTENT_PREVIEW_CHARS` (line 235) so a
  subclass that changes the preview window gets independent cache
  entries instead of stale ones. The matching `_related_summary_tokens`
  cache (line 256) keys on `corpus.name` for the same reason. Both
  are good shape — no findings here.
- **`MIN_ALIAS_OVERLAP` is well-documented** (line 218–221 comment
  + the override-rationale link to `selection-criteria-robustness`).
  The 0.1.22 escape-hatch contract is clear.
- **`_tokenize` / `_stem` are already cached at the corpus-build
  layer** (the `_tokens_cache` sidecar on `RetrievalEntry`). No
  reason to add a second cache layer at the retriever.
- **Sort stability under tie-break.** The current sort is stable on
  the secondary key `entry.path`. `heapq.nlargest` preserves equality
  ordering by the key tuple, which is the same — but verify in the
  P3 PR that the test suite's exact tie-breaking expectations
  survive.

## Reranker hot path

**0 findings.** The reranker is in good shape:

- CPU cost is dominated by JSON parse + list comprehensions; both are
  already as tight as Python allows without C extension.
- Wall-clock cost is dominated by the Anthropic round-trip
  (`llm_reranker_rerank.wall` mean = 0.799 s vs `.cpu` mean = 0.012 s).
  Network jitter is irreducible at this layer; σ=3.0 absorbs it.
- The reranker prompt (`_SYSTEM`) is single-block, cache-friendly,
  and already structured for prompt caching (per the comment cluster
  near `MAX_CITATION_DOCUMENTS=200` in `providers/claude.py` — the
  W2.1 deep-review called this "exemplary").
- No premature optimisation candidates: no JSON path that could be
  swapped for `orjson` would survive the σ envelope. The cost is
  network, not compute.

The reranker stays advisory through W3 by spec design (see W3.1 in
`tasks.md`) precisely because its wall-clock noise is dominated by
external variance.

## Disposition

| Class | Items |
|---|---|
| **Blocking (must fix this freeze week)** | none |
| **Fix during freeze (`### Changed` / `### Fixed`)** | none — every candidate is a behaviour-equivalent micro-opt with no signal at the current corpus size. |
| **W3.1 input (perf gate promotion)** | The audit confirms `keyword_retriever_retrieve.cpu` and `rag_pipeline_run.cpu` are gating-ready (mean stable, headroom plentiful). Reranker stays advisory per spec. |
| **Phase-5 ticket** | P1 (lazy reason), P2 (class-level cache key), P3 (`heapq.nlargest`), P4 (inline `_category_weight`) — captured in [`phase-5-backlog/items.md` § Perf](../phase-5-backlog/items.md#perf-w22-perf-audit). Bundle as one perf-only PR after the inter-run baseline (item M1) lands so the methodology is in place to validate the deltas. |

## Hand-off to W3

W3.1 promotes the perf gate from advisory → blocking for
`keyword_retriever_retrieve.cpu` and `rag_pipeline_run.cpu`
(CPU-time only; wall-clock and reranker stay advisory). This audit
confirms both metrics are stable enough on the current baseline to
gate without flapping.

W3.3 (coverage push via `/test-audit` + `/smart-test`) does not
overlap with the perf findings — none of P1–P4 touch the public
surface, and none would require a new test that the W3.3 coverage
work isn't already adding.
