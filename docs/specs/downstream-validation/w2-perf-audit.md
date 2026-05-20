# W2.2 — Mid-phase performance audit

> Per [`docs/specs/downstream-validation/tasks.md`](./tasks.md) W2.2.
> Static audit of retrieval + reranker hot paths plus cross-check
> against [`perf-baseline.md`](./perf-baseline.md) /
> [`perf-thresholds.json`](./perf-thresholds.json). Date: 2026-05-20.
> Commit: `11afde1` (post W2.1).

## Headline

- **PR #72's advisory perf-gate flagged a single-axis regression on `rag_pipeline_run.cpu/.wall` (~+34 % over threshold). PR #74's second reading on the same code (no perf-relevant changes between them) showed `rag_pipeline_run.cpu` at +10.8 % — back under threshold.** Two consecutive readings on identical code, ±23 % swing on a sub-millisecond benchmark. **Verdict: noise, not regression.**
- The ostensibly-changed code (0.1.22's two added stem suffixes + one new conditional) was inadequate to explain the +182 µs PR #72 magnitude; the second reading confirms the static-diff intuition.
- **Reranker hot path is LLM-bound** (network dominates). Python overhead is negligible; no actionable findings.
- **Static audit still surfaces four Phase-5 micro-opt candidates** in `retrieval.py` — independent of the false-alarm regression, they are real (small) wins.
- **Persistent recommendation: widen `rag_pipeline_run` N from 30 → 50 before the W3.1 promotion to blocking.** The 23 % single-run swing makes the current gate unreliable: the same code crossed the threshold once and stayed under it the second time. If the perf workflow becomes blocking with the current noise floor, real PRs will eat false-positive blocks.
- **Tooling note:** `attune-ai:performance_audit` MCP tool failed (`AttributeError: 'str' object has no attribute 'get'`) on both absolute and repo-relative paths. Filed as an attune-ai bug — not blocking; static audit substitutes.

## Cross-check vs perf-baseline.json

Baseline locked at commit `3149a0c` (pre-0.1.22). Current HEAD is
post-0.1.22 (commit `99996fd`). Two advisory perf-workflow readings
on the same code (no perf-relevant changes between PR #72 merge and
PR #74's docs commits):

| Metric | Baseline (s) | PR #72 (first) | PR #74 (second) | Threshold | PR #72 Δ | PR #74 Δ |
|---|---:|---:|---:|---:|---:|---:|
| `rag_pipeline_run.cpu` | 0.000537 | 0.000719 ⚠️ | **0.000595 ✅** | 0.000625 | +33.9 % | +10.8 % |
| `rag_pipeline_run.wall` | 0.000536 | 0.000718 ⚠️ | **0.000595 ✅** | 0.000624 | +34.0 % | +11.0 % |
| `keyword_retriever_retrieve.cpu` | 0.003212 | 0.010196 | 0.010355 | 0.034493 | +217 % | +222 % |
| `keyword_retriever_retrieve.wall` | 0.003212 | 0.010196 | 0.010357 | 0.034494 | +217 % | +222 % |
| `directory_corpus_load.cpu` | 0.000047 | 0.000053 | 0.000054 | 0.000066 | +12.8 % | +14.9 % |
| `directory_corpus_load.wall` | 0.000046 | 0.000053 | 0.000054 | 0.000066 | +15.2 % | +17.4 % |

Source: PR #72 + PR #74 advisory comments under marker
`<!-- attune-rag-perf-gate -->` on issue-comments stream.

### Interpretation

**Why `keyword_retriever_retrieve` is "ok" despite +217 %:** baseline
`stdev = 0.01564 s`, dwarfing the 0.003212 s mean. Threshold = `mean +
2σ = 0.034493 s`, so the 0.010 s current reading is still inside 1σ of
the baseline. The benchmark is dominated by first-call cache-population
cost, hence the loose threshold. This is the intended W0.5 behaviour:
the loose retrieve threshold is documented in
[`perf-baseline.md`](./perf-baseline.md).

**Why the PR #72 `rag_pipeline_run` reading looked concerning, and why
the PR #74 reading exonerates it:** baseline `stdev = 0.000044 s` on a
0.000537 s mean — 8 % RSD, tight. The PR #72 reading of 0.000719 s was
**4.1 σ** above baseline mean. That would have been signal *if it were
repeatable*. It wasn't:

- PR #72 reading: 0.000719 s (+33.9 %, over threshold)
- PR #74 reading: 0.000595 s (+10.8 %, under threshold) — same code path
- Spread between the two readings: 0.000124 s = ±23 % of mean

The two readings are on identical perf-relevant code (PR #74 added only
docs/markdown commits), yet they sit on opposite sides of the gate. The
benchmark shares `corpus`, `retriever`, and `pipeline` instances with
the previous benchmarks in the same process, so by the time
`pipeline.run` is timed the entry-token cache is fully warm — the
measurement is mostly orchestration overhead (citation-record build,
prompt formatting, structlog emit) plus a cache-hit retrieve, all on
the sub-millisecond scale where one GC pause / page fault / co-tenancy
hiccup on Azure dominates.

**Conclusion:** PR #72's flag was single-point noise. The 0.1.22
retrieval diff (3 ops, all upstream of the cache) was already
inadequate to explain a +182 µs change on a warm-cache `pipeline.run`;
the second reading confirms the static-diff intuition.

**Open issue:** the gate is currently unreliable for this metric. A
±23 % run-to-run swing on a benchmark whose threshold sits at +16 %
above mean means **the same code can flip the gate purely on runner
noise.** This is fine while the gate is advisory, but if W3.1 promotes
it to blocking unchanged, real PRs will eat false-positive blocks.

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
| 1 | ✅ **DONE.** Second perf reading on PR #74 confirms PR #72's flag was noise; `rag_pipeline_run` back under threshold (+10.8 %). | Verified | — |
| 2 | **Widen N from 30 → 50 for the `rag_pipeline_run` benchmark before W3.1 gate promotion.** Re-locked baseline with N=50 closes the ±23 % single-run swing at source. Update `perf-baseline.md` + `perf-thresholds.json`. *This is the only blocking remediation before W3.1.* | Tighten gate | scoped PR, pre-W3.1 (target ≤ 2026-06-01) |
| 3 | **File Phase-5 backlog ticket** capturing the four retrieval micro-opts (lazy reason build, cache-key tuple constant, `heapq.nlargest`, inline category-weight). Aggregate net wall-clock gain probably <50 µs but the lazy-reason change is also a readability win. | Backlog | Phase-5 spec when opened |
| 4 | **File attune-ai bug:** `mcp__plugin_attune-ai_attune-ai__performance_audit` raises `AttributeError: 'str' object has no attribute 'get'` on both absolute and repo-relative `path:` arguments. W2.2 succeeded via static audit; the official tool was unusable. | External | attune-ai issue tracker |

## Freeze-clock impact

None of the recommendations introduce new public surface. Cadence
clock is **not** reset.

- Recommendations 1–3 are diagnosis / fix work that lands as `### Changed` if needed.
- Recommendation 4 (widen N) is benchmark-runner internal — no API.
- Recommendation 5 is post-freeze housekeeping.
- Recommendation 6 is in another repo.

## Hand-off

- **Pre-W3.1 perf-gate prep:** rec #2 (widen N=30→50 for `rag_pipeline_run`) is the only blocker. Without it, the W3.1 promotion to blocking is unsafe — same code crossed the threshold once and slipped back under it on the very next run.
- **W2 housekeeping commit** (optional, can bundle W2.1 + W2.2 items): the ~7 docstring/style nits from W2.1. W2.2 contributes no fix-during-freeze items — the advisory was noise.
- **W3.1 perf-gate promotion** stays on schedule for `rag_pipeline_run.cpu` *provided rec #2 lands first*. Wall-clock axis stays advisory per the spec.
