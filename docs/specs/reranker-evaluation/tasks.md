# Spec: reranker-evaluation — tasks

> **Status:** scoped 2026-05-22 — executable. Entry gates inherited
> from [`requirements.md`](requirements.md) §"Entry gates"; this
> file describes what to *do* once those gates open.

## Scoping decisions locked (2026-05-22)

The seven open questions inherited from
[design.md §9](design.md#9-open-questions-for-scoping) +
[requirements.md §"Open questions for scoping"](requirements.md#open-questions-for-scoping)
are answered as follows:

| # | Question | Decision | Rationale |
|---|---|---|---|
| 1 | Scope α (retrieval-only) vs β (+faithfulness) | **α** | Bundled-corpus baseline P@1 already 1.00; faithfulness scoring doubles judge-LLM spend for marginal signal. β reserved as follow-up if α's verdict is "marginal lift, need faithfulness data to break tie." |
| 2 | Measurement implementation | **B — new `scripts/measure_reranker.py`** | Decoupled from `test_golden.py`; matches existing `scripts/measure_*.py` pattern; future test-runner refactors don't break the diagnostic. |
| 3 | Run cardinality | **N=5 for `rerank=on`, N=1 for `rerank=off`** | rerank-off is deterministic — one pass is correct. rerank-on is Anthropic-bound (non-deterministic); N=5 catches prompt-noise on the 7 residual queries for ~$0.30 of API spend. |
| 4 | Verdict thresholds | tightened from design.md §4 — see table in M3 below | Illustrative numbers in design.md replaced with concrete cutoffs at scoping. |
| 5 | Default-flip PR timing | **defer to v1.0.0 cut PR** | If verdict is `rerank-default-off`, the flip rides on the cut PR's CHANGELOG narrative. Cleaner exit-summary; avoids a mid-Phase-5 selection-criteria flip that complicates Phase 4's hand-off story. |
| 6 | Report filename | **`diagnostic-1.md`** | Matches [`embedding-retriever/diagnostic-1.md`](../embedding-retriever/diagnostic-1.md) — both are "diagnostic that produces a defer-or-ship verdict" artifacts. |
| 7 | Token-cost source | **`response.usage` from the Anthropic SDK** | Per requirements.md R2. Payload-compute is fallback only if `response.usage` is unavailable (which itself would be a diagnostic bug worth surfacing). |

These decisions are load-bearing: subsequent milestones reference
them. Changes require a new `/spec` pass.

## Milestones

### M0 — Entry-gate verification (pre-flight)

| # | Task | Layer | Notes |
|---|---|---|---|
| M0.1 | Verify Phase 4 W4.2 green: read `docs/specs/downstream-validation/cadence-week-{1,2,3,4}.md` and confirm all four report `**Status:** ON TRACK`. | attune-rag | If any reads RESET, D5 doesn't start — the freeze hasn't held. |
| M0.2 | Verify 0.2.0 shipped to PyPI. Run `pip index versions attune-rag` (or equivalent) and confirm `0.2.0` is the latest stable. | attune-rag | The diagnostic must measure the pipeline shape that 0.2.0 ships, not an intermediate state. |
| M0.3 | Verify 7-day no-hotfix watch closed clean per `api-v0.2.0-cut` M3.3. | attune-rag | A late hotfix during the watch would change the pipeline shape and force a re-measurement. |

### M1 — Build the diagnostic (`scripts/measure_reranker.py`)

| # | Task | Layer | Notes |
|---|---|---|---|
| M1.1 | Create `scripts/measure_reranker.py` skeleton: `argparse` CLI mirroring `scripts/measure_baseline_variance.py`. Flags: `--baseline-queries`, `--paraphrased-queries`, `--rerank-runs N` (default 5), `--out PATH` (report path), `--candidate-multiplier` (default 3). | attune-rag | Pure stdlib + existing `attune_rag` imports. No new deps. |
| M1.2 | Wire `RagPipeline` construction inside the script: build `pipeline_off` with `reranker=None`, `pipeline_on` with the default Haiku reranker. Both share the bundled `AttuneHelpCorpus` and the same `candidate_multiplier`. | attune-rag | Use the public `RagPipeline(...)` constructor — no internal API surfaces. |
| M1.3 | Implement Run A (`rerank=off`): iterate baseline + paraphrased query YAMLs, capture per-query top-3 paths, compute P@1 and R@3 against ground truth. **Strict-dominance check (R1):** assert numbers reproduce 1.00 / 1.00 / 0.9125 / 1.00. Fail loudly if not. | attune-rag | This is the regression net — if Run A doesn't reproduce, the diagnostic is wrong. |
| M1.4 | Implement Run B (`rerank=on`, N=5): for each of 5 runs, same iteration as M1.3, capture per-query top-3 + `response.usage.input_tokens` / `.output_tokens`. Aggregate: mean / p50 / p95 per metric across the 5 runs. | attune-rag | The N=5 dimension applies to token + P@1 mean; R@3 is reported per-run (will likely be stable). |
| M1.5 | Implement report rendering: write the markdown shape from design.md §3 to `--out` path. Include all reproducibility metadata (Q3): query YAML SHA-256, commit SHA, reranker model string, Anthropic SDK version, ISO timestamp. | attune-rag | Deterministic — same input data + same N runs produce byte-identical metadata block (modulo the timestamp). |
| M1.6 | Tests under `tests/unit/test_measure_reranker.py`: cover query loading (YAML parse), Run-A scoring (mocked corpus), report rendering (golden snapshot), reproducibility metadata. **Do not** test the actual Anthropic call — that's M2's job. | attune-rag | Mock the reranker for unit tests; live API hit is M2. |

### M2 — Run the diagnostic, draft the report

| # | Task | Layer | Notes |
|---|---|---|---|
| M2.1 | Execute `python scripts/measure_reranker.py --rerank-runs 5 --out docs/specs/reranker-evaluation/diagnostic-1.md` with `ANTHROPIC_API_KEY` set. | attune-rag | Wall-clock budget: ~5 min (5 × 80 queries × ~0.8 s). API spend: ~$0.30 at Haiku list. |
| M2.2 | Verify R1 holds: Run A reproduces 1.00 / 1.00 / 0.9125 / 1.00 byte-identically. If not, M2.2 fails and M1 is re-opened. | attune-rag | This is the diagnostic's self-test. |
| M2.3 | Inspect the per-query residuals table (the 7 paraphrased P@1 misses). For each: does any of the 5 rerank-on runs lift the expected doc to rank 1? Annotate the per-query "rerank fixes?" column with a stability indicator (e.g. `5/5`, `3/5`, `0/5`). | attune-rag | Stability matters: a 1/5 lift is noise, not a fix. |
| M2.4 | Commit the report at `docs/specs/reranker-evaluation/diagnostic-1.md` (CHANGELOG `### Changed` — internal docs, no `### Added`). | attune-rag | Report is the artifact; commit and PR follow the standard pattern. |

### M3 — Lock the verdict + downstream cross-link

| # | Task | Layer | Notes |
|---|---|---|---|
| M3.1 | Apply the locked verdict logic to the report numbers: <br><br>**`rerank-default-on`** iff: <br>• ≥3 of the 7 paraphrased P@1 misses fixed at ≥4/5 stability AND <br>• token cost mean ≤ $0.002 / query AND <br>• Run B baseline P@1 + R@3 = 1.00 / 1.00 AND Run B paraphrased R@3 = 1.00. <br><br>**`rerank-default-off`** iff: <br>• ≤1 of the 7 misses fixed at ≥4/5 stability, OR <br>• Run B baseline P@1 or R@3 drops below 1.00 (rerank moves a winning doc off rank 1) OR Run B paraphrased R@3 drops below 1.00, OR <br>• token cost mean > $0.005 / query. <br><br>**`corpus-shape-dependent-default`** iff: <br>• exactly 2 of the 7 misses fixed at ≥4/5 stability AND no regression — i.e. the lift is real but small enough that corpus shape (size, density of paraphrase) plausibly tips the cost/benefit either way. | attune-rag | Thresholds locked at scoping; verdict is the report's tightest binding output. |
| M3.2 | Fill in the report's "Verdict" cell with one of the three locked strings + write the one-paragraph rationale. | attune-rag | No mealy-mouthed text in the verdict cell (R5). |
| M3.3 | Fill in the report's "Implications for `user-corpus-onboarding`" paragraph: what the harness default should be, what the guide should say about when to enable rerank on a user corpus. | attune-rag | This is what `user-corpus-onboarding/risks.md` §7 references. |
| M3.4 | Update [`user-corpus-onboarding/risks.md` §7](../user-corpus-onboarding/risks.md#7-the-reranker-evaluation-diagnostic-d5-and-the-harness-happen-in-different-timeframes): replace "D5 not yet run" language with a cross-link to D5's report and the locked harness-default decision. | attune-rag | Closes the dependency `user-corpus-onboarding` declared on D5. |
| M3.5 | If verdict is `rerank-default-off`: file a follow-up issue (not a PR) titled "v1.0.0-cut: flip RagPipeline reranker default to None per D5." The cut PR's own `/spec` pass picks up the flip. **Do not flip in this milestone.** | attune-rag | Decision 5 of scoping: flip rides on v1.0.0 cut PR. |
| M3.6 | If verdict is `rerank-default-on`: file a follow-up note in `docs/specs/v1.0.0-release/design.md` noting D5 ratified the existing default; no flip needed. | attune-rag | Symmetric outcome — both verdicts produce a cut-PR artifact. |
| M3.7 | If verdict is `corpus-shape-dependent-default`: file a follow-up note in `docs/specs/v1.0.0-release/design.md` recommending the cut PR adds a POLICY paragraph clarifying that the toggle is exposed without a strong default. | attune-rag | Per `[[feedback_policy_llm_metric_commitments]]`: corpus-shape-dependent verdict is a *capability* commitment, not a metric commitment — fine for POLICY. |

## Dependencies

```
M0.1 → M0.2 → M0.3   (entry gates; cannot start M1 until all green)
M1.1 → M1.2 → M1.3 → M1.4 → M1.5 → M1.6  (script build)
M1.6 → M2.1 → M2.2 → M2.3 → M2.4         (diagnostic run)
M2.4 → M3.1 → M3.2 → M3.3 → M3.4         (verdict + cross-link)
M3.4 → exactly one of {M3.5, M3.6, M3.7}  (cut-PR handoff, verdict-conditional)
```

## Definition of done (Phase 5 D5 closes)

- [ ] All M0 gates green.
- [ ] `scripts/measure_reranker.py` lands with tests; ruff clean; freeze gate green.
- [ ] `docs/specs/reranker-evaluation/diagnostic-1.md` committed; R1 reproduction verified inside the report.
- [ ] Verdict cell contains exactly one of the three locked strings (R5).
- [ ] `user-corpus-onboarding/risks.md` §7 updated with cross-link.
- [ ] Exactly one of M3.5 / M3.6 / M3.7 follow-up filed.
- [ ] D5 status banner across README/design/requirements/risks/tasks promoted to `complete`.

## Out-of-scope (not in D5; tracked elsewhere)

Per [`README.md`](README.md#whats-not-in-scope) and
[`requirements.md`](requirements.md#out-of-scope-requirements-deferred):

- Prompt re-tuning, cross-vendor rerankers, `candidate_multiplier` sweeps,
  multi-corpus measurement, user-facing toggle UX, faithfulness measurement
  under rerank (Scope β; revisit if α's verdict is "marginal lift").

## Calendar (proposed; Phase 5 opens after 0.2.0 + 7-day watch)

If 0.2.0 cuts ~2026-06-20 (per Phase 4 calendar) and 7-day watch
closes ~2026-06-27:

- **D5 starts** ~2026-06-27 (Phase 5 opens for `/spec`).
- M1 (script build): ~3-4 days of attention.
- M2 (run + report): ~half-day (wall-clock dominated by 5 × 80 Haiku calls + report polish).
- M3 (verdict + cross-link): ~1 day.

Total D5 attention budget: **~5-6 days**. Must close **before**
`user-corpus-onboarding` M1 scoping closes (the harness default
inherits the verdict).
