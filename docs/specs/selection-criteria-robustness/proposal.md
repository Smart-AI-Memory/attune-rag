# Proposal: More robust selection criteria for KeywordRetriever

> **Status:** Workstream A shipped in 0.1.22 (`-ity`/`-ities` stemming + `MIN_ALIAS_OVERLAP=2` default, under `freeze-override`). Workstream B (embedding co-signal) deferred to 0.2.0+ semantic-floor lift — see [api-v0.2-public-surface follow-ups](../api-v0.2-public-surface/tasks.md#follow-ups-post-0118).
> **Workstream:** A (internal retriever tuning) — shipped. B (embeddings) — deferred.
> **Freeze posture:** `### Changed`, no `__all__` delta, no new public surface, no new dependency. Compatible with the Phase 4 symbol-level freeze.

> **Update 2026-05-21:** Workstream B's premise — that a semantic co-signal would be needed to lift the paraphrased-query floor — was **superseded by the [alias-expansion sweep](../alias-expansion-sweep/)** (shipped in 0.1.23). The sweep added 180+ multi-token aliases via the new `aliases_override.json` mechanism and closed paraphrased R@3 from 28.75% → 100% on the attune-help corpus without any new dependency. So:
>
> - **For the attune-help corpus**, Workstream B is **permanently deferred** — the embedding co-signal is no longer needed; the keyword-only path now lands paraphrased R@3 = 100%.
> - **For arbitrary user corpora** (post-v1.0.0 framework framing — see [v1.0.0-release Phase 5 scope](../v1.0.0-release/design.md#phase-5-scope-decided-2026-05-21)), embedding retrieval remains viable as a future feature if the [`user-corpus-onboarding`](../user-corpus-onboarding/) quality harness surfaces gaps the frontmatter-alias + override path can't close. The [`embedding-retriever`](../embedding-retriever/#scope-of-the-defer) spec carries the scope-specific defer.
>
> The "Why not workstream B or C now" section below predates the sweep and should be read as "as of pre-sweep evaluation (2026-05-19)." The rest of this proposal is preserved unchanged as the historical record of what shipped in 0.1.22.

## Problem

`docs/specs/release-quality-baseline/thresholds.json` locks Precision@1 at **0.95** — the floor and the current measured value. There is zero headroom; the next regression that flips a single query unlocks an indefinite re-baseline window. The two queries currently sitting at the cliff are diagnostic, not random:

| ID | Query | Top-1 (actual) | Top-1 (expected) | Why it fails |
|---|---|---|---|---|
| gq-011 | "vulnerability scan" | `quickstarts/skill-security-audit.md` | `concepts/tool-security-audit.md` | Stemming asymmetry: query token `vulnerability` (no suffix matches) and corpus token `vulnerabilities` → `vulnerabilit` (via `"ies"`) never collide. One quickstart that happens to also write the singular wins on raw summary hit count. |
| gq-020 | "write unit tests" | `concepts/tool-fix-test.md` | `concepts/tool-smart-test.md` or `quickstarts/generate-tests.md` | Single-token alias hit: `tool-fix-test`'s alias list contains `"broken pipeline tests"`. The query's only content token `test` matches one token in the alias union and credits the full `ALIASES_WEIGHT=1.5`, breaking what should be a tie in fix-test's favor. |

Neither failure mode is a "more semantics" problem; both are token-pipeline bugs. Fix them where they live before reaching for embeddings.

## Proposed change

Two edits in `src/attune_rag/retrieval.py`:

1. **Symmetric stem for `-ity` / `-ities`** (`### Changed`, freeze-compatible). Insert `"ities"` and `"ity"` into `_STEM_SUFFIXES`, with `"ities"` placed **before** `"ies"` so plurals collapse to the same stem as the singular. `_MIN_STEM_LEN = 3` prevents the obvious false friends (`city`, `unity`, `pity` all leave too little stem and are not modified).
2. **`MIN_ALIAS_OVERLAP` class attribute (default `2`)** (`### Added`, ships under `freeze-override`). Requires at least 2 distinct query tokens to overlap an entry's alias-token union before crediting `aliases_hits`. Multi-token aliases (the design intent — `"CI pipeline failing"`, `"publish to PyPI"`) still fire; lone-token coincidences (e.g. `test` overlapping `"broken pipeline tests"`) no longer dominate ties. Flipping to `1` restores pre-0.1.20 behavior.

### Why ship the alias knob mid-freeze

The freeze gate is symbol-level. Initially this work was scoped to land only #1 and defer #2 to 0.2.0. Measurement traced the tradeoff: under #2, gq-020 ("write unit tests") flips from miss → top-1 (`quickstarts/generate-tests.md`), while gq-026 ("version bump and changelog") flips from top-1 → miss (loses single-token `vers` alias hit on `"ship a version"`). **Net P@1 unchanged at 0.975** — but the remaining miss kind matters:

- *Without* #2: gq-020 is a documented semantic-tie miss that needs workstream B (embeddings) to resolve.
- *With* #2: gq-026 is a corpus-side alias-content miss that attune-help can resolve by tightening the alias phrasing.

The second failure mode is solvable inside the freeze; the first isn't. Combined with the exposed-knob safety valve (downstream consumers can set `MIN_ALIAS_OVERLAP = 1` to opt out), shipping mid-freeze is the cleaner end state. The freeze-override is filed in CHANGELOG `[Override-rationale]`; cadence clock not reset (`Security`-scoped exception pattern applied to an internal quality knob with comparable reversibility).

### Discarded during measurement

A weight-lowering variant (`ALIASES_WEIGHT` 1.5 → 1.0 or 1.25) was traced as an alternative to the coverage threshold; it regresses gq-013 / gq-035 because references-category docs (category weight 1.0) lean on single-token alias signal to overcome the concept-category boost. The coverage-threshold rule wins on aliases that pass the bar regardless of weight.

## Measured metric impact

Against the locked 40-query golden set (`tests/golden/queries.yaml` at sha256 `f47486d…`):

| Metric | Before | After | Threshold |
|---|---:|---:|---:|
| Precision@1 | 0.950 (38/40) | **0.975 (39/40)** | 0.95 (to be re-baselined to 0.975) |
| Recall@3    | 1.000 (40/40) | 1.000 (40/40)     | 1.00 |

Per-difficulty (combined stem fix + alias knob):

| Tier | Before P@1 | After P@1 |
|---|---:|---:|
| easy   | 9/10  | **10/10** (gq-011 + gq-020 fixed) |
| medium | 26/27 | 26/27 (gq-026 newly missed — alias `vers`) |
| hard   | 3/3   | 3/3 |

Per-query: gq-011 retrieves `concepts/tool-security-audit.md` at top-1 (stem fix). gq-020 retrieves `quickstarts/generate-tests.md` at top-1 (alias rule). gq-026 retrieves `quickstarts/skill-release-prep.md` at top-1 (alias rule regression; expected `concepts/tool-release-prep.md` ranks #2). gq-013, gq-021, gq-035 still pass top-1. All 713 unit tests pass; golden suite passes 37 + 3 xpassed.

Mean faithfulness is structurally unaffected — the reranker and answer generation never see the stem table or the alias overlap rule. A fresh 20-run baseline measurement re-locks `thresholds.json` (P@1 floor lifts from 0.95 → 0.975).

## Why not workstream B or C now

- **Embedding retrieval (B)** is a more interesting medium-term move but adds dep weight (`fastembed` ≈ 50 MB, `model2vec` ≈ 20 MB) for two queries already in top-3. With R@3 already at 1.0, the marginal value of a semantic co-signal is small until the corpus or query distribution shifts. Re-evaluate post-freeze with the `### Added` public knob.
- **Reranker prompt tuning (C)** moves a metric we don't currently miss on (faithfulness sits at 0.979 vs 0.9686 threshold) and risks the only metric with non-trivial stdev. Defer.

## Risks

- Adding `"ity"`/`"ities"` to the stem table widens collision surface. The 40-query golden set is the regression net; the full unit suite (673 tests) plus golden (37 + 3 xpassed) is green. If a future corpus update flips an unrelated query, back out the two suffixes.

## Rollout (executed)

1. Pre-change snapshot: `artifacts/before-selection-criteria.json` (P@1 38/40).
2. Stemming edit in `retrieval.py`; targeted unit test in `tests/unit/test_retrieval.py::test_stemming_collapses_ity_and_ities`.
3. Post-change snapshot: `artifacts/after-selection-criteria.json` (P@1 39/40).
4. `scripts/check_freeze.py` returns 0. Golden + unit suites green.
5. CHANGELOG entry under `[Unreleased] ### Changed` quotes the measured delta.
