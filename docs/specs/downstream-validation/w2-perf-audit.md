# W2.2 — Mid-phase performance audit

> Per [`docs/specs/downstream-validation/tasks.md`](./tasks.md) W2.2.
> Static audit of retrieval + reranker hot paths plus cross-check
> against [`perf-baseline.md`](./perf-baseline.md) /
> [`perf-thresholds.json`](./perf-thresholds.json). Date: 2026-05-20.
> Commit: `11afde1` (post W2.1).

## Headline

- **CI perf-gate flagged a single-axis advisory regression on PR #72** (the 0.1.22 release): `rag_pipeline_run.cpu` and `.wall` both ~+34 % over threshold; `keyword_retriever_retrieve` axes well under threshold; `directory_corpus_load` axes well under threshold.
- **Static review of the 0.1.22 diff finds no plausible mechanism** for a +182 µs CPU-time regression on `rag_pipeline_run`. The actual code change is ~2 extra suffix-endswith checks per query token (memoized once after first call) plus one new `if X >= Y` per scored entry.
- **Working hypothesis:** noise on a sub-millisecond benchmark. The baseline `rag_pipeline_run.cpu.stdev` is 0.000044 s (8 % RSD) measured on a single CI run; one re-measurement is not a confirmed trend.
- **Recommendation:** do **not** file as Phase-5 perf work yet. Re-measure on the next main-branch perf workflow run; if the regression persists across two consecutive readings, dig deeper. If we reach W3.1 (gate promotion to blocking on CPU-time) still over threshold, options are (a) real fix, (b) re-baseline, (c) widen N to dampen single-point noise.
- **Reranker hot path is LLM-bound** (network dominates). Python overhead is negligible; no actionable findings.
- **Static audit surfaces four Phase-5 micro-opt candidates** in `retrieval.py` (none would obviously close the ~+34 % advisory regression on its own).
- **Tooling note:** `attune-ai:performance_audit` MCP tool failed (`AttributeError: 'str' object has no attribute 'get'`) on both absolute and repo-relative paths. Filed as an attune-ai bug — not blocking; static audit substitutes.

## Cross-check vs perf-baseline.json

Baseline locked at commit `3149a0c` (pre-0.1.22). Current HEAD is
post-0.1.22 (commit `11afde1`). PR #72 ran the advisory perf workflow
on the 0.1.22 retrieval diff and posted:

| Metric | Baseline mean (s) | Current mean (s) | Δ | Threshold (s) | Status |
|---|---:|---:|---:|---:|---|
| `rag_pipeline_run.cpu` | 0.000537 | 0.000719 | **+33.9 %** | 0.000625 | ⚠️ **over** |
| `rag_pipeline_run.wall` | 0.000536 | 0.000718 | **+34.0 %** | 0.000624 | ⚠️ **over** |
| `keyword_retriever_retrieve.cpu` | 0.003212 | 0.010196 | +217 % | 0.034493 | ok (under threshold) |
| `keyword_retriever_retrieve.wall` | 0.003212 | 0.010196 | +217 % | 0.034494 | ok (under threshold) |
| `directory_corpus_load.cpu` | 0.000047 | 0.000053 | +12.8 % | 0.000066 | ok |
| `directory_corpus_load.wall` | 0.000046 | 0.000053 | +15.2 % | 0.000066 | ok |

Source: PR #72 advisory comment under marker `<!-- attune-rag-perf-gate -->`.

### Interpretation

**Why `keyword_retriever_retrieve` is "ok" despite +217 %:** baseline
`stdev = 0.01564 s`, dwarfing the 0.003212 s mean. Threshold = `mean +
2σ = 0.034493 s`, so the 0.010 s current reading is still inside 1σ of
the baseline. The benchmark is dominated by first-call cache-population
cost, hence the loose threshold. This is the intended W0.5 behaviour:
the loose retrieve threshold is documented in
[`perf-baseline.md`](./perf-baseline.md).

**Why `rag_pipeline_run` is concerning:** baseline `stdev = 0.000044 s`
on a 0.000537 s mean — 8 % RSD, tight. Current 0.000719 s is **4.1 σ**
above baseline mean. That's signal, *if* the new measurement was taken
under the same conditions as the baseline. But:

- The benchmark shares `corpus`, `retriever`, and `pipeline` instances
  with the previous benchmarks in the same process, so by the time
  `pipeline.run` is timed the entry-token cache is fully warm. The
  measurement is therefore mostly pipeline-orchestration overhead
  (citation-record build, prompt formatting, structlog emit) plus a
  cache-hit retrieve.
- Baseline `N = 30` is fine for a millisecond-scale benchmark; for a
  ~500 µs benchmark on a noisy cloud runner it's at the edge. Single
  GC pauses, page faults, or co-tenancy noise on Azure dominate.
- The 0.1.22 retrieval diff (3 ops, all upstream of the cache) is
  inadequate to explain a +182 µs change on a warm-cache `pipeline.run`.

## Static perf audit — `retrieval.py`

### Strengths

- Two-layer memoization on entries: `_entry_field_tokens` (path/summary/content/aliases tokens) and `_related_summary_tokens` (related-entry summary tokens), both keyed on the relevant scope (`CONTENT_PREVIEW_CHARS`, `corpus.name`). Per the `_entry_field_tokens` docstring this was the review's primary perf concern and is correctly handled.
- `_score_entry` uses Python `set & set` intersections (C-implemented).
- Single `scored.sort(...)` at the end of `retrieve`; no in-loop sorts.
- `_tokenize` does a single regex sub + split + comprehension; no quadratic patterns.
- `_stem` is a linear suffix scan over a small tuple (18 entries) — but its result is memoized downstream via `_entry_field_tokens`, so query tokens are the only ones repeatedly stemmed (~5–10 tokens × 18 tries = ~100 endswith calls per query). Negligible.

### Phase-5 micro-opt candidates

None of these would obviously close the ~+34 % advisory regression on
their own; they are real but small wins that can land in a single
post-freeze `### Changed` housekeeping commit.

1. **`_score_entry` always builds `reasons: list[str]` + `"+".join(reasons)` even for entries below `MIN_SCORE`.** The match-reason string is only useful for entries that survive the threshold filter in `retrieve()`. Cost: ~5 list-appends + a join per scored entry — small per entry, multiplied by corpus size.
   **Fix:** lazy-build `match_reason` only when the caller filters by `score >= MIN_SCORE`, or inline the threshold check into `_score_entry` so we skip reason construction below threshold.
2. **Cache-key tuple `("field_tokens", self.CONTENT_PREVIEW_CHARS)` is allocated on every `_entry_field_tokens` invocation** even on cache hits.
   **Fix:** promote to a class-level constant (`_FIELD_TOKENS_CACHE_KEY = ("field_tokens", CONTENT_PREVIEW_CHARS)`) or just use an int / str. Tuple-allocation is cheap but happens N times per query.
3. **`scored.sort(key=lambda h: (-h.score, h.entry.path))` is O(n log n).** As corpora grow (currently ~80 entries in `.help/templates/`), `heapq.nlargest(k, scored, key=...)` is O(n log k) and asymptotically faster.
   **Fix:** swap `sort + [:k]` for `heapq.nlargest`. Same output, less work as corpus grows.
4. **`_category_weight` is a one-line dict lookup wrapped in a method.** Per-entry function call overhead is real on hot paths.
   **Fix:** inline `self.CATEGORY_WEIGHTS.get(entry.category, self.DEFAULT_CATEGORY_WEIGHT)` directly in `_score_entry`.

## Static perf audit — `reranker.py`

- The hot path is `LLMReranker.rerank` → an Anthropic API call. Latency is network-bound (Haiku rerank calls return in <2 s typical; the timeout is 60 s). Python work pre/post call is microsecond-scale.
- System prompt has `cache_control: ephemeral` — prompt-cache hits expected after first call, halving cost on repeated reranks.
- API key is never logged; the broad-`except Exception` block uses `logger.debug(..., type(exc).__name__, exc)` and **deliberately omits `exc_info=True`** (comment explains the call-frame-locals concern). This was already cleared in W09.A.007 and re-confirmed in W2.1.
- Trivial nit: the `seen: set[int]` + two-pass dedup in `rerank` could be a single `dict.fromkeys(...)` order-preserving dedup. Already O(n) and readable; not worth churning.

No actionable findings.

## Static perf audit — `pipeline.py` (`run` / `_retrieve` paths)

- `_retrieve` does retrieval, optional re-ranking, then a slice. Fine.
- `run` does retrieve + `build_citation_record` + `join_context*` + `build_augmented_prompt` + `logger.info`. All linear in `k`, which is bounded (default 3).
- `logger.info(..., query=query, ...)` emits the user query verbatim to the structlog logger. Already cleared in W2.1 security pass; flagged here only because structlog field serialization is a measurable fraction of a 500 µs benchmark. If the advisory regression turns out to be real, **a structlog renderer change between baseline and current is a non-zero hypothesis** worth eliminating before going deeper into retrieval.

## Recommended actions

| # | Action | Class | Where |
|---|---|---|---|
| 1 | **Re-run perf workflow** on the current main-branch tip (commit `fed00f5` or later). If `rag_pipeline_run` is still over threshold, we have a confirmed regression; if it slips back under, the PR #72 reading was noise. | Verify | Trigger `.github/workflows/perf.yml` on a dummy PR or next merge |
| 2 | If still over threshold after (1): **eliminate the structlog hypothesis first** — patch out the `logger.info` call in the benchmark and re-measure. If `rag_pipeline_run.cpu` drops back to baseline, the regression was a structlog renderer change, not retrieval. | Diagnose | scoped throwaway PR |
| 3 | If still over threshold after (1) + (2): apply Phase-5 micro-opt #1 (lazy reason build) — the only one of the four that touches the per-entry hot loop in a visible way. Re-measure. | Fix-by-elimination | post-freeze `### Changed` |
| 4 | Independent of the regression: **widen N from 30 → 50 for the rag_pipeline_run benchmark specifically.** Re-locked baseline would close the noise question at source. Document the change in `perf-baseline.md`. | Tighten gate | post-W3.1 or sooner if W3.1 promotion blocks on noise |
| 5 | **File Phase-5 backlog ticket** capturing the four retrieval micro-opts above for the post-freeze housekeeping commit. Net wall-clock gain probably <50 µs aggregate, but the lazy-reason change is also a readability win. | Backlog | Phase-5 spec when opened |
| 6 | **File attune-ai bug:** `mcp__plugin_attune-ai_attune-ai__performance_audit` raises `AttributeError: 'str' object has no attribute 'get'` on both absolute and repo-relative `path:` arguments. W2.2 succeeded via static audit; the official tool was unusable. | External | attune-ai issue tracker |

## Freeze-clock impact

None of the recommendations introduce new public surface. Cadence
clock is **not** reset.

- Recommendations 1–3 are diagnosis / fix work that lands as `### Changed` if needed.
- Recommendation 4 (widen N) is benchmark-runner internal — no API.
- Recommendation 5 is post-freeze housekeeping.
- Recommendation 6 is in another repo.

## Hand-off

- **W2 housekeeping commit** (optional, can bundle W2.1 + W2.2 fix-during-freeze items): docstring fixes from W2.1 + structlog elimination test from W2.2 rec #2 if pursued.
- **W3.1 perf-gate promotion** depends on whether rec #1 + rec #4 land before then. If `rag_pipeline_run.cpu` is still over threshold at W3.1 start, the promotion should either be deferred for that one metric or block on a re-baseline.
- **W3.3 `/test-audit`** can absorb the structlog-hypothesis test if it surfaces a real renderer change.
