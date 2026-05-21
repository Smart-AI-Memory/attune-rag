# Spec: alias-expansion-sweep

> **Status: complete — M2–M12 and M13.1–M13.3 landed via PRs [#94](https://github.com/Smart-AI-Memory/attune-rag/pull/94)–[#100](https://github.com/Smart-AI-Memory/attune-rag/pull/100), [#101](https://github.com/Smart-AI-Memory/attune-rag/pull/101)–[#105](https://github.com/Smart-AI-Memory/attune-rag/pull/105) on 2026-05-21. M13.4 (upstream promotion) tracked at [Smart-AI-Memory/attune-help#9](https://github.com/Smart-AI-Memory/attune-help/issues/9); not in this spec's surface. Embedding-retriever stays permanently deferred.**

- **Owner:** Patrick
- **Created:** 2026-05-21
- **Completed:** 2026-05-21 (same day — methodology was mechanical once M2 + the mechanism landed)
- **Target version (shipped):** 0.3.0 — additive `aliases_override.json`, no public surface change.
- **Predecessor work:** [embedding-retriever/](../embedding-retriever/) diagnostics D1–D3 (deferred parent spec, kept as archival evidence + revival path).
- **Entry condition:** D3 closed +50pp R@3 on the bug-predict cluster with zero baseline regression and zero new dependency. Sweep the same lever across the remaining feature clusters.

## Result

| | D1 keyword-only | After M2–M12 | Δ |
|---|---:|---:|---:|
| Paraphrased P@1 | 11.25% | **82.50%** | **+71.25pp** |
| Paraphrased R@3 | 28.75% | **96.25%** | **+67.50pp** |
| Baseline P@1 | 97.50% | **100.00%** | **+2.50pp** ✨ |
| Baseline R@3 | 100% | 100% | 0 |

By paraphrased difficulty:
- easy R@3: 30% → **100%**
- medium R@3: 30% → **94.44%**
- hard R@3: 17% → **100%**

**Strict-dominance held across the sweep.** Baseline P@1 actually improved 2.5pp from M4's release-prep aliases. The one near-regression (M12's `"readme lies about code"` tipping `gq-017` off the doc-gen top-1) was caught at the pre-commit diagnostic step and fixed before shipping. Lesson captured in M12's PR description.

3 paraphrased queries remain in R@3-miss: all `medium` difficulty, all flagged as cluster-boundary ambiguities (the longest-standing is `gqp-023a "look for ways my code could be exploited"` documented since M3).

## Purpose (archived for reference)

Generalize the alias-expansion result from [diagnostic-3.md](../embedding-retriever/diagnostic-3.md) across all under-served feature clusters surfaced by [diagnostic-1.md](../embedding-retriever/diagnostic-1.md). For each cluster, hand-author multi-token aliases against the cluster's paraphrased misses, measure the before/after lift with [run_diagnostic_3.py](../embedding-retriever/run_diagnostic_3.py) (parameterized per feature), and persist the aliases as a new override file in `attune-rag`.

**Outcome at completion (target was R@3 ≥ 70%):** shipped at **R@3 = 96.25%**, well above target. M13 acceptance criterion met at the M7 cluster (code-quality, PR [#99](https://github.com/Smart-AI-Memory/attune-rag/pull/99)); subsequent M8–M12 PRs were optional polish that pushed R@3 from 70% to 96.25%.

## Why this spec exists separately from embedding-retriever

embedding-retriever is **deferred**, not closed — the artifact bundle there is the supporting evidence and the revival path. This spec is the **execution** of the cheaper alternative that justified the defer. Keeping them separate:

- Lets this spec ship without re-litigating the embedding case.
- Lets the embedding-retriever spec revive cleanly (without un-deferring the wrong thing) if the sweep leaves a residual.
- Avoids burying actionable per-cluster work inside an archival spec.

## Cluster sizes (from `queries_paraphrased.yaml`)

Authoring effort prioritization order:

| Cluster | Paraphrased queries | Notes |
|---|---|---|
| bug-predict | 14 | **Done in D3** — commit the aliases via the new override mechanism. |
| security-audit | 8 | Largest unaddressed cluster. |
| release-prep | 8 | Tied. |
| smart-test | 6 | |
| fix-test | 6 | |
| code-quality | 6 | |
| refactor-plan | 6 | |
| planning | 6 | |
| doc-gen | 6 | |
| doc-orchestrator | 6 | |
| deep-review | 4 | |
| doc-audit | 4 | |

Total: 80 paraphrased queries across 12 clusters.

## Alias home — override in attune-rag first

Aliases currently live in attune-help frontmatter. Two paths:

| Path | When to use |
|---|---|
| **`aliases_override.json` in attune-rag (this spec)** | Fast iteration during the sweep. Same shape as the existing `summaries_override.json` (path → list of strings). Loaded by `AttuneHelpCorpus` at construction time and merged into `entry.aliases`. **No attune-help rev needed.** |
| Upstream attune-help frontmatter | Long-term home once aliases stabilize. Done as a separate task after the sweep, in a single attune-help release that promotes all proven aliases at once. |

The override-first path keeps each cluster iteration to a single attune-rag PR. Promotion upstream is a follow-up; doing it inline would couple every cluster to an attune-help release.

## Spec files

- [`tasks.md`](tasks.md) — milestones: (M1) override mechanism, (M2) bug-predict commit, (M3–M12) per-cluster sweeps, (M13) aggregate measurement and embedding-retriever defer/revive decision.

`requirements.md` and `design.md` are intentionally omitted — the methodology is fully specified by [diagnostic-3.md](../embedding-retriever/diagnostic-3.md) and the override mechanism is a small implementation task. If this spec grows or the override mechanism needs design, add them then.

## Out of scope

- Embedding retriever revival. Separate decision after M13.
- Upstream alias promotion to attune-help. Follow-up task once the sweep completes.
- QueryExpander positioning (default vs opt-in). Separate small task; the precision cost in [diagnostic-2.md](../embedding-retriever/diagnostic-2.md) is the deciding evidence.
- The `attune-hub` noise-attractor finding from D1. Orthogonal; aliases may close it indirectly per cluster, but it deserves its own measurement if the indirect close doesn't fire.
- Re-authoring the paraphrase set with a second author. Tracked in diagnostic-1's caveats; not an entry condition for this spec.

## Activation path

This spec promotes from `scoping` to `approved` when the `/spec` scoping pass:

1. Confirms the override file format (proposed: `src/attune_rag/corpus/aliases_override.json`, same path-keyed shape as `summaries_override.json`, value is a list of strings appended to the entry's existing aliases).
2. Confirms acceptance criteria per cluster: `≥ +30pp R@3` on the cluster's queries, zero regression on the baseline `queries.yaml`.
3. Confirms the aggregate success criterion at M13: paraphrased R@3 ≥ 70%.
4. Fills in concrete acceptance criteria on each tasks.md milestone.
