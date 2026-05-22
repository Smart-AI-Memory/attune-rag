# Spec: reranker-evaluation — requirements

> **Status:** scaffolding — not executable. Invariants below are the
> non-negotiables the `/spec` scoping pass + implementation must
> satisfy.

## Entry gates (inherited from upstream phases)

The **scaffold** itself (these four files plus README) is docs-only
and freeze-compliant — it lands today regardless of gate state.

The **diagnostic run** activates when **all of** the following hold:

1. **Phase 4 W4.2 green** — four consecutive `cadence-week-{1,2,3,4}.md`
   reports show zero `### Added` under `[Unreleased]`. The diagnostic
   itself doesn't add public surface, but its verdict may trigger a
   default flip that must respect freeze semantics — which means the
   freeze must be demonstrably held before D5 produces a verdict.
2. **0.2.0 cut shipped to PyPI.** Per
   [`api-v0.2.0-cut`](../api-v0.2.0-cut/) M2. D5 must measure
   against the same pipeline shape that ships at 0.2.0, not against
   an intermediate state.
3. **0.2.0 7-day no-hotfix watch closed clean.** Per
   [`api-v0.2.0-cut`](../api-v0.2.0-cut/) M3.3.
4. **Phase 5 opens for `/spec` scoping.** Per
   [ROADMAP-v1.md Phase 5](../ROADMAP-v1.md#phase-5--100-release).

D5 is also sequenced internally to land **before**
[`user-corpus-onboarding`](../user-corpus-onboarding/) M1's scoping
pass closes — so the harness default inherits D5's verdict.

## Functional requirements

### R1: The diagnostic reproduces the bundled-baseline numbers under `rerank=off`

Run A (`rerank=off`) of the diagnostic against the bundled
`attune-help` corpus + `tests/golden/queries.yaml` +
`tests/golden/queries_paraphrased.yaml` must produce numbers
identical to the locked post-sweep baseline:

- Baseline P@1: 1.00 (40/40)
- Baseline R@3: 1.00 (40/40)
- Paraphrased P@1: 0.9125 (73/80)
- Paraphrased R@3: 1.00 (80/80)

If Run A produces different numbers, the diagnostic's measurement
machinery is wrong (not the reranker), and the verdict cannot lock
until the discrepancy is investigated.

This is the regression-net property: keyword-only is deterministic;
the diagnostic must reproduce it byte-identically before its
reranker-on numbers are trustworthy.

### R2: The diagnostic captures token usage from `response.usage`

Each Anthropic Haiku call's `response.usage.input_tokens` and
`response.usage.output_tokens` are captured and reported. The
diagnostic does not fall back to estimating tokens from the request
payload unless `response.usage` is unavailable (which would be a
diagnostic-itself bug worth surfacing).

Cost calculation uses the **Haiku list price at measurement time**,
documented in the report header so the number is auditable later.

### R3: Latency numbers are cited, not re-measured

The report's latency table cites
[`docs/specs/downstream-validation/perf-baseline.md`](../downstream-validation/perf-baseline.md)
directly. D5 doesn't re-run pytest-benchmark. The N=50 numbers
locked in 0.1.22 are good enough; re-measuring them mid-Phase-5
would muddy the diagnostic's purpose.

(Exception: if `perf-baseline-multi-run` re-locks the baseline
mid-Phase-5 with new numbers, D5 cites the new numbers — but still
doesn't re-measure itself.)

### R4: The report uses the same shape as `baseline-1.md`

Tables and section ordering match
[`docs/specs/release-quality-baseline/baseline-1.md`](../release-quality-baseline/baseline-1.md)
where the metric types overlap. New tables (token cost, residuals)
follow the same column-style — right-aligned numerics, ISO-format
metadata, query SHA-256 in the header.

This is what makes the report comparable to prior diagnostic
reports and citable from other specs (especially
`user-corpus-onboarding/design.md`).

### R5: The verdict is one of three fixed values

The verdict cell in the report must contain exactly one of:

- `rerank-default-on`
- `rerank-default-off`
- `corpus-shape-dependent-default`

No mealy-mouthed "the reranker generally helps but..." text in the
verdict cell. The rationale paragraph is where nuance lives; the
verdict is binary-ish so downstream specs can consume it
programmatically.

### R6: The diagnostic is single-corpus (bundled-corpus only)

D5 measures the **bundled `attune-help` corpus** only. User-corpus
measurement is what
[`user-corpus-onboarding`](../user-corpus-onboarding/) ships as a
harness; D5 is upstream of that harness and informs its default.

A second-corpus measurement (e.g. attune-help's *help-templates*
directory vs concept-docs, or a synthetic harder corpus) is **not**
in D5's scope. If the verdict turns on corpus shape, that's the
`corpus-shape-dependent-default` verdict — and the user-corpus
harness lets users measure for themselves whether they fall on the
"rerank helps" or "rerank doesn't help" side.

## Quality requirements

### Q1: Strict-dominance held across any flip PR D5 triggers

If D5's verdict triggers a default-flip PR (rerank-default-off
flipping the `RagPipeline` default), that PR runs the full baseline
diagnostic and must show:

- Baseline P@1: 1.00 (unchanged)
- Baseline R@3: 1.00 (unchanged)
- Paraphrased R@3: 1.00 (unchanged)
- Paraphrased P@1: documented expected delta (whatever D5 measured)

The flip PR's body cites D5's report by commit SHA + report path.

If the verdict is `rerank-default-on` (no flip), this requirement
doesn't fire.

### Q2: The diagnostic respects freeze semantics

- The diagnostic script (whether reusing `test_golden.py`,
  introducing a new `scripts/measure_reranker.py`, or extending
  `measure_baseline_variance.py`) lands as `### Changed` if it
  touches existing scripts, or as part of an internal script
  (`scripts/`) that doesn't appear in `__all__`.
- The report markdown is docs-only; no public-surface impact.
- A default-flip (if triggered) is `### Changed` (selection-criteria;
  the reranker stays available, just opt-in instead of opt-out) and
  legal during Phase 4 freeze, but scoping may defer it to the
  v1.0.0 cut PR for cleanliness.

### Q3: The diagnostic is reproducible

- Query YAMLs SHA-256-recorded in the report header.
- Commit SHA recorded.
- Reranker model string recorded.
- Anthropic SDK version recorded.
- API key not recorded (obviously); the report must run on any
  machine with `ANTHROPIC_API_KEY` set and produce the same
  retrieval numbers (token-cost numbers and rerank picks may vary
  per-run by model nondeterminism — captured by the N=5 plan in
  [design.md §6](design.md#6-single-run-vs-multi-run) if scoping
  picks that).

### Q4: The report is self-contained

A reader who lands on the report cold (no other context) can answer:

- What was measured?
- Against what corpus and query set?
- What did the numbers say?
- What's the verdict and why?
- What downstream decisions does the verdict shape?

…without needing to chase 6 other links. Links are for *deepening*
context, not establishing it.

This mirrors the
[`diagnostic-1.md`](../embedding-retriever/diagnostic-1.md) shape
from the embedding-retriever defer report.

## Out-of-scope requirements (deferred)

These are not requirements for D5; listed so they don't slip in
unbudgeted:

- **No prompt re-tuning.** The reranker prompt is measured as-is.
- **No cross-vendor reranker comparison.** Haiku only.
- **No `candidate_multiplier` sweep.** Measured at default `=3`.
- **No multi-corpus measurement.** Bundled corpus only.
- **No faithfulness measurement.** (Scope α; see
  [design.md §1.3](design.md#13-faithfulness-vs-retrieval-only)).
  Scoping may pick Scope β; if so this line moves.
- **No user-facing toggle UX work.** Whatever the verdict, the
  toggle already exists (`RagPipeline(reranker=...)`). UX for
  surfacing the toggle is a separate spec if it's a spec at all.

## Open questions for scoping

(Each gets a decision at the `/spec` pass; recorded here as the
unresolved set the scoping pass inherits.)

1. **Scope α vs β** — measure faithfulness as well? See
   [design.md §1.3](design.md#13-faithfulness-vs-retrieval-only).
2. **Approach A / B / C** for the measurement implementation. See
   [design.md §2](design.md#2-how-to-measure-candidate-approaches).
3. **Run cardinality.** Single-pass vs N=5 rerank-on. See
   [design.md §6](design.md#6-single-run-vs-multi-run).
4. **Verdict thresholds.**
   [design.md §4](design.md#4-verdict-logic-proposed-scoping-confirms)
   has illustrative numbers; pin them.
5. **Default-flip PR timing** if verdict is `rerank-default-off` —
   Phase 4 freeze or v1.0.0 cut PR?
6. **Report file name.** `report.md` vs `diagnostic-1.md`.
7. **Token-cost source.** `response.usage` (R2) vs payload
   compute? Confirmed R2 favors SDK; double-check during scoping.

## Definition of done (sketch)

To be finalized at scoping. Initial bullets:

- [ ] Measurement script implemented (Approach picked per
      [design.md §2](design.md#2-how-to-measure-candidate-approaches)).
- [ ] Run A (`rerank=off`) reproduces locked baseline numbers (R1).
- [ ] Run B (`rerank=on`) produces P@1 / R@3 + token-cost numbers.
- [ ] Report markdown rendered at the path the scoping pass picks.
- [ ] Verdict locked, rationale paragraph written.
- [ ] Cross-link added in
      [`user-corpus-onboarding/risks.md` §7](../user-corpus-onboarding/risks.md#7-the-reranker-evaluation-diagnostic-d5-and-the-harness-happen-in-different-timeframes)
      pointing at D5's report.
- [ ] If verdict triggers a default-flip PR: that PR runs the full
      baseline diagnostic and holds strict-dominance per Q1.
