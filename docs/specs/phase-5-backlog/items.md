# Phase-5 backlog — items

Eleven items deferred during the Phase 4 freeze. Provenance:

- **W2.1 deep-review** ([docs/specs/downstream-validation/w2-deep-review.md](../downstream-validation/w2-deep-review.md)) — Disposition table "Phase-5 ticket" + "W3.3 `/test-audit` follow-up" rows.
- **W2.2 perf audit** — items surfaced during the audit but not captured in a committed doc; sourced from the Phase-5 capture prompt (2026-05-20).
- **Methodology gap** — inter-run noise observation from PRs [#72](https://github.com/Smart-AI-Memory/attune-rag/pull/72) vs [#74](https://github.com/Smart-AI-Memory/attune-rag/pull/74), only partially addressed by [#75](https://github.com/Smart-AI-Memory/attune-rag/pull/75) + [#77](https://github.com/Smart-AI-Memory/attune-rag/pull/77).

Effort: **S** ≤ 1 hr, **M** ≤ 4 hr, **L** > 4 hr.

## Quality (W2.1 deep-review)

| # | Item | Target file(s) | Effort | Why deferred during freeze |
|---|---|---|---:|---|
| Q1 | Extract `_RollbackState` helper / context manager for the three rollback layers in `apply_rename` (~90 LOC, moves → staging → sequential rename) | `src/attune_rag/editor/rename.py:426-515` | M | Internal refactor with high regression surface on the freeze's highest-risk module; bundle with the W3.3 rollback test gaps so refactor + tests land together. |
| Q2 | Lift the three near-identical `_iter_entries(corpus)` helpers (~7 LOC each) into a shared internal module | `src/attune_rag/editor/references.py:164-171`, `rename.py:586-593`, `autocomplete.py:63-76` | S | Blocked: the `editor/_*.py` 0.3.0 compatibility shims complicate a clean lift. Revisit post-shim removal. |
| Q3 | Rename `validator = err.validator` → `keyword = err.keyword` (jsonschema's own term for the discriminator) | `src/attune_rag/editor/schema.py:90` | S | Cosmetic; touches a freeze-locked module for no behaviour change. |
| Q4 | Alphabetise `__all__` | `src/attune_rag/providers/__init__.py:59` | S | Cosmetic; bundle with the 0.2.0 public-surface freeze rather than churning `__all__` mid-freeze. |

## Perf (W2.2 perf audit)

| # | Item | Target file(s) | Effort | Why deferred during freeze |
|---|---|---|---:|---|
| P1 | Lazy match-reason build in `_score_entry` — skip the `reasons: list[str]` + `"+".join` for entries below `MIN_SCORE` | `src/attune_rag/retrieval.py::_score_entry` | S | Behaviour-equivalent micro-opt on a hot path; the freeze gate's σ already absorbs this scale of variance, so no signal would distinguish the change from noise. Land post-freeze with the inter-run baseline (item M1) in place. |
| P2 | Promote the cache-key tuple to a class-level constant | `src/attune_rag/retrieval.py::_entry_field_tokens` | S | Same reasoning as P1 — micro-opt with no observable freeze-gate effect. |
| P3 | Replace `sorted(...)[:k]` with `heapq.nlargest(k, ...)` (O(n log k) vs O(n log n)) | `src/attune_rag/retrieval.py::retrieve` | S | The asymptotic win matters at larger corpora than the current golden set; verify with a corpus-scaling perf scenario as part of Phase 5's perf re-baseline, not during freeze. |
| P4 | Inline `_category_weight` call into `_score_entry` | `src/attune_rag/retrieval.py` | S | Tiny hot-path call-site optimisation; defer with P1/P2 to land as one perf-only PR post-freeze. |

## Test-audit (W2.1 shallow-but-covered)

| # | Item | Target file(s) | Effort | Why deferred during freeze |
|---|---|---|---:|---|
| T1 | VCR-style recorded fixture for `ClaudeProvider.generate_with_citations` to catch Anthropic SDK shape drift | `tests/unit/providers/fixtures/` (new), `tests/unit/providers/test_claude.py` | M | Requires one live SDK call to record the cassette; the freeze policy keeps live-network steps out of CI. Schedule with Phase 5's final `/security-audit` + `/deep-review` pass. |
| T2 | Realistic disk-fault simulation for `apply_rename` rollback paths (replace timed `target.exists()` patches with an injected filesystem-fault layer) | `tests/unit/test_editor_rename.py` | M | Pairs naturally with Q1's `_RollbackState` extraction — easier to inject faults after the refactor, so do them together. |
| T3 | Real-SDK `cached_prefix`-flagging contract test (current tests assert the kwarg was *passed*, not that the SDK actually flagged the block for caching) | `tests/unit/providers/test_claude.py`, `tests/unit/test_pipeline_native_citations.py` | M | Same network-recording requirement as T1; bundle. |

## Methodology (inter-run noise)

| # | Item | Target file(s) | Effort | Why deferred during freeze |
|---|---|---|---:|---|
| M1 | Multi-run perf-baseline methodology: lock-baseline runs K **separate** workflow invocations (different runner instances), measures once each, aggregates means. Current baseline only models intra-run stdev; PR [#72](https://github.com/Smart-AI-Memory/attune-rag/pull/72) vs [#74](https://github.com/Smart-AI-Memory/attune-rag/pull/74) showed a ±23 % swing on `rag_pipeline_run.cpu` across two consecutive workflow runs on identical perf-relevant code. PRs [#75](https://github.com/Smart-AI-Memory/attune-rag/pull/75) + [#77](https://github.com/Smart-AI-Memory/attune-rag/pull/77) widened σ 2.0→3.0 and bumped N (per-PR 10→30, baseline 30→50), but neither captures inter-runner variance. | `scripts/measure_perf_baseline.py` (new aggregation mode), `.github/workflows/perf.yml` (orchestrate K invocations) | L | Touches the perf gate itself mid-freeze, which would invalidate the in-flight baseline. The current σ=3.0 gate is already wide enough to ship the freeze; the proper inter-run baseline is a Phase-5 deliverable so the v1.0.0 perf claim has a defensible noise model. |
