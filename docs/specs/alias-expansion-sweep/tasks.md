# Spec: alias-expansion-sweep — Tasks

> **Status: scaffolding — not executable; promotes via `/spec` scoping pass.**

This file is a **scaffold**, not a runnable task list. A `tasks.md` in a scoping spec is **not executable** — see user memory `feedback_spec_scoping_vs_approved`. The `/spec` pass promotes this into an approved spec with concrete scripts and acceptance criteria.

## Phase 1: Tasks

**Status:** scoping. No work in this spec is approved to start.

### Implementation order (sketch)

M1 builds the override mechanism. M2 commits the bug-predict aliases already validated by D3. M3–M12 sweep the remaining clusters. M13 measures the aggregate and decides whether embedding-retriever revives.

| # | Task | Layer | Status | Notes (to be filled by scoping) |
|---|------|-------|--------|---------------------------------|
| **M1 — Override mechanism** | | | | |
| M1.1 | Add `src/attune_rag/corpus/aliases_override.json` (empty stub) with the same path-keyed shape as `summaries_override.json`. Value is a list of strings appended to the entry's existing aliases. | attune-rag | scaffold | unscoped |
| M1.2 | Extend `AttuneHelpCorpus.__init__` to load `aliases_override.json` and merge each path's extra aliases into the entry's `aliases` tuple. Use `dataclasses.replace` on the frozen `RetrievalEntry`. | attune-rag | scaffold | unscoped |
| M1.3 | Unit test: corpus entry for a path in the override file exposes the merged aliases; entries not in the override file are unchanged. | attune-rag | scaffold | unscoped |
| **M2 — bug-predict aliases (commit D3's result)** | | | | |
| M2.1 | Move the 18 aliases from [`run_diagnostic_3.py`](../embedding-retriever/run_diagnostic_3.py) `_BUG_PREDICT_EXTRA_ALIASES` into `aliases_override.json` under the three bug-predict paths. Fix the 2 residual misses (`gqp-015a`, `gqp-036a`) by adding `"fails silently"` and `"diff bites"`. | attune-rag | scaffold | unscoped |
| M2.2 | Re-run [`run_diagnostic_1.py`](../embedding-retriever/run_diagnostic_1.py) against `queries_paraphrased.yaml`. Verify bug-predict R@3 = 100% and overall R@3 ≥ baseline + 8pp. | attune-rag | scaffold | unscoped |
| **M3–M12 — Per-cluster sweeps** | | | | |
| M3 | **security-audit** (8 queries). Hand-author aliases against D1 misses (gqp-001a/b, 011a/b, 023a/b, 032a/b). Add to `aliases_override.json`. Measure with parameterized `run_diagnostic_3.py`. Acceptance: cluster R@3 ≥ +30pp, no baseline regression. | attune-rag | scaffold | unscoped |
| M4 | **release-prep** (8 queries: 005a is deep-review-tagged but ranks against release; clarify cluster boundaries during scoping). | attune-rag | scaffold | unscoped |
| M5 | **smart-test** (6 queries). | attune-rag | scaffold | unscoped |
| M6 | **fix-test** (6 queries). | attune-rag | scaffold | unscoped |
| M7 | **code-quality** (6 queries). | attune-rag | scaffold | unscoped |
| M8 | **refactor-plan** (6 queries). | attune-rag | scaffold | unscoped |
| M9 | **planning** (6 queries). | attune-rag | scaffold | unscoped |
| M10 | **doc-gen** (6 queries). | attune-rag | scaffold | unscoped |
| M11 | **doc-orchestrator** (6 queries). | attune-rag | scaffold | unscoped |
| M12 | **deep-review** (4 queries) + **doc-audit** (4 queries). Bundled because they're small. | attune-rag | scaffold | unscoped |
| **M13 — Aggregate measurement and defer/revive decision** | | | | |
| M13.1 | Re-run [`run_diagnostic_1.py`](../embedding-retriever/run_diagnostic_1.py) with all aliases committed. Report final paraphrased P@1 / R@3 vs the D1 baseline. | attune-rag | scaffold | unscoped |
| M13.2 | If overall paraphrased R@3 ≥ 70% → embedding-retriever spec stays `deferred`, closed permanently. If < 70% → identify the residual miss queries, update [embedding-retriever/README.md](../embedding-retriever/README.md) defer rationale to point at the specific residual, and promote that spec back to `scoping` with a narrowed mandate. | attune-rag + docs | scaffold | unscoped |
| M13.3 | Promote `tests/golden/queries_paraphrased.yaml` to the regression suite per [embedding-retriever/diagnostic-2.md §Recommended next steps](../embedding-retriever/diagnostic-2.md#recommended-next-steps) — info-only assertions in `tests/golden/test_golden.py` for 0.3.x, gating threshold decision deferred. | attune-rag | scaffold | unscoped |
| M13.4 | Open an upstream-promotion task in attune-help for the aliases that proved out, to be executed in a single attune-help release. **Not** part of this spec's surface — separate task tracked in [attune-help backlog]. | attune-help | scaffold | unscoped |

### Dependencies (sketch)

```
M1 (override mechanism) ─→ M2 (bug-predict commit) ─→ M3..M12 (per-cluster sweeps, parallelizable) ─→ M13 (aggregate)
```

M3–M12 are mutually independent once M1 lands. They can ship as separate PRs or one batched PR; per-cluster PRs make the alias-authoring choices easier to review.

### Definition of done (sketch)

To be finalized during scoping. Initial bullets:

- [ ] `aliases_override.json` lives in `src/attune_rag/corpus/` with a working merge mechanism.
- [ ] All 12 clusters have aliases committed. Each cluster's R@3 lift documented in its PR description.
- [ ] Aggregate paraphrased R@3 ≥ 70% (target — finalize during scoping).
- [ ] No regression on `queries.yaml` baseline (P@1 ≥ 0.95, R@3 ≥ 1.00, faithfulness ≥ 0.9686).
- [ ] `queries_paraphrased.yaml` is loaded by `test_golden.py` as info-only.
- [ ] [embedding-retriever/README.md](../embedding-retriever/README.md) reflects the final defer/revive decision.
- [ ] Upstream-promotion task opened against attune-help (separate work).

### Risks & mitigations (sketch)

| Risk | Mitigation sketch |
|---|---|
| A cluster's miss queries don't respond to alias expansion the way bug-predict did (e.g., the noise attractor is a category-weight problem, not an alias-coverage problem). | Acceptance criterion (≥ +30pp cluster R@3) is per-cluster — if a cluster fails, document why and move on; that cluster becomes part of the residual that informs the embedding-retriever defer/revive decision at M13.2. |
| Alias bloat hurts retrieval on the baseline (lots of low-quality aliases pulling adjacent features). | `MIN_ALIAS_OVERLAP = 2` is the structural guard; each PR re-runs the baseline `queries.yaml` and gates on no regression. |
| Authoring time exceeds the 1–2 hour estimate from D3. | Methodology is established by D3; the sweep is mechanical. If a cluster's authoring is hard, that's signal it doesn't respond well — capture it as M13 residual rather than spending more time. |
| `aliases_override.json` proves the wrong shape (e.g., we need per-alias provenance for upstream-promotion later). | Adopt the simplest shape now (path → list of strings); if we need richer metadata for M13.4, refactor at that point. |

### Out of scope (deferred)

- Upstream promotion to attune-help frontmatter. Tracked in M13.4 as a separate task, not work this spec performs.
- Embedding-retriever revival. M13.2 decides; the revival work itself lives in [embedding-retriever/](../embedding-retriever/) if it happens.
- QueryExpander default-on positioning. Separate decision.
- The `attune-hub` noise-attractor finding. Orthogonal; may close indirectly per cluster.
