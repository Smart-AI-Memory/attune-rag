# Spec: alias-expansion-sweep — Tasks

> **Status: complete — M2–M12 and M13.1–M13.3 landed via PRs [#94](https://github.com/Smart-AI-Memory/attune-rag/pull/94)–[#105](https://github.com/Smart-AI-Memory/attune-rag/pull/105) on 2026-05-21. M13.4 (upstream promotion to attune-help) is in flight at [Smart-AI-Memory/attune-help#9](https://github.com/Smart-AI-Memory/attune-help/issues/9) and intentionally outside this spec's surface. The 121 proven aliases live in [`src/attune_rag/corpus/aliases_override.json`](../../../src/attune_rag/corpus/aliases_override.json) as a temporary home until the attune-help release ships.**

## Phase 1: Tasks

**Status:** complete.

### Implementation order (as shipped)

M1 built the override mechanism. M2 committed the bug-predict aliases already validated by D3. M3–M12 swept the remaining clusters. M13 captured the aggregate and the defer/revive decision.

| # | Task | Layer | Status | Notes |
|---|------|-------|--------|-------|
| **M1 — Override mechanism** | | | | |
| M1.1 | Add `src/attune_rag/corpus/aliases_override.json` with the same path-keyed shape as `summaries_override.json`. | attune-rag | **done** | PR [#94](https://github.com/Smart-AI-Memory/attune-rag/pull/94) |
| M1.2 | Extend `DirectoryCorpus.__init__` with `extra_aliases` kwarg + `AttuneHelpCorpus.__init__` to load the JSON. | attune-rag | **done** | PR [#94](https://github.com/Smart-AI-Memory/attune-rag/pull/94) |
| M1.3 | Unit tests for the merge mechanism (6 in `test_corpus_directory.py`) + integration tests on the bundled corpus. | attune-rag | **done** | PR [#94](https://github.com/Smart-AI-Memory/attune-rag/pull/94) |
| **M2 — bug-predict aliases (commit D3's result)** | | | | |
| M2.1 | Move the 18 aliases from D3's `_BUG_PREDICT_EXTRA_ALIASES` into the override; add `"fails silently"` for gqp-015a; original `"diff bites"` corrected to `"diff bite"` (stemmer collapses `bites → bit`). | attune-rag | **done** | PR [#94](https://github.com/Smart-AI-Memory/attune-rag/pull/94) |
| M2.2 | Verify bug-predict R@3 = 100% and overall R@3 ≥ baseline + 8pp. **Observed: bug-predict R@3 14/14 = 100%, overall R@3 +11.25pp.** | attune-rag | **done** | PR [#94](https://github.com/Smart-AI-Memory/attune-rag/pull/94) |
| **M3–M12 — Per-cluster sweeps** | | | | |
| M3 | **security-audit** (8 queries; closed 7/8). Known residual gqp-023a documented. | attune-rag | **done** | PR [#95](https://github.com/Smart-AI-Memory/attune-rag/pull/95) — +8.75pp R@3 |
| M4 | **release-prep** (8 queries; closed 5/5 failing, 3 already passed). | attune-rag | **done** | PR [#96](https://github.com/Smart-AI-Memory/attune-rag/pull/96) — +6.25pp R@3, +2.5pp baseline P@1 |
| M5 | **smart-test** (6 queries; closed 4/4 failing). | attune-rag | **done** | PR [#97](https://github.com/Smart-AI-Memory/attune-rag/pull/97) — +5.00pp R@3 |
| M6 | **fix-test** (6 queries; closed 3/3 failing). | attune-rag | **done** | PR [#98](https://github.com/Smart-AI-Memory/attune-rag/pull/98) — +3.75pp R@3 |
| M7 | **code-quality** (6 queries; closed 5/5 failing). **M13 acceptance criterion met at this PR.** | attune-rag | **done** | PR [#99](https://github.com/Smart-AI-Memory/attune-rag/pull/99) — +6.25pp R@3 |
| M8 | **refactor-plan** (6 queries; closed 6/6 failing). | attune-rag | **done** | PR [#101](https://github.com/Smart-AI-Memory/attune-rag/pull/101) — +7.50pp R@3 |
| M9 | **planning** (6 queries; closed 2/2 failing). | attune-rag | **done** | PR [#102](https://github.com/Smart-AI-Memory/attune-rag/pull/102) — +2.50pp R@3 |
| M10 | **doc-gen** (6 queries; closed 1/1 failing). | attune-rag | **done** | PR [#103](https://github.com/Smart-AI-Memory/attune-rag/pull/103) — +1.25pp R@3 |
| M11 | **doc-orchestrator** (6 queries; closed 6/6 failing — pushed paraphrased hard R@3 to 100%). | attune-rag | **done** | PR [#104](https://github.com/Smart-AI-Memory/attune-rag/pull/104) — +7.50pp R@3 |
| M12 | **deep-review** (4 queries; closed 3/3 failing) + **doc-audit** (4 queries; closed 3/3 failing). Captured the only near-regression of the sweep (`"readme lies about code"` tipping gq-017) and fixed pre-commit. | attune-rag | **done** | PR [#105](https://github.com/Smart-AI-Memory/attune-rag/pull/105) — +7.50pp R@3 |
| **M13 — Aggregate measurement and defer/revive decision** | | | | |
| M13.1 | Final diagnostic-1 against committed aliases. **Result: paraphrased P@1 82.50%, R@3 96.25%; baseline 100%/100%.** | attune-rag | **done** | M12 PR description |
| M13.2 | embedding-retriever stays `deferred` permanently — alias-expansion closed the gap that justified it. | docs | **done** | spec banner stays `deferred` |
| M13.3 | Promote `queries_paraphrased.yaml` to `tests/golden/test_golden.py` as info-only regression input + aggregate watermark guard. | attune-rag | **done** | PR [#100](https://github.com/Smart-AI-Memory/attune-rag/pull/100) |
| M13.4 | Open upstream-promotion task in attune-help. **Not in this spec's surface** — separate work. | attune-help | **in flight** | [attune-help#9](https://github.com/Smart-AI-Memory/attune-help/issues/9) |

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
