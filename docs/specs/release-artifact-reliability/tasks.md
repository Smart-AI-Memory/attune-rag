# Phase 1 task overview

> **Status: Phase 1 complete locally; all five tasks accepted under authorized auto-run.**

The [XML plan](../../../.claude/plans/release-artifact-reliability.md) is the task-definition authority. Patrick approved execution and auto-run of the remaining tasks (D5–D9). The embedded state and canonical workspace record task acceptance; the linked receipts record measured results and their limits.

| Task | Deliverable | Requirements | Result |
|---|---|---|---|
| 1 | Package resources and source/wheel parity | R1–R2 | Accepted; [local review](receipts/task-1-local-review.json) |
| 2 | Validate exactly what publication consumes | R3 | Accepted; [11 tests and artifact-transfer review](receipts/task-2-review.json) |
| 3 | Quality outcome propagation and metric validation | R4–R5 | Accepted; [254 targeted tests and review](receipts/task-3-review.json) |
| 4 | Valid evidence for the two existing CPU gates | R6 | Accepted; [158 targeted tests and review](receipts/task-4-review.json) |
| 5 | Public claims and final integrated verification | R7 | Accepted; [final review](receipts/task-5-review.json) |

**Task 1.** Both corpus JSON overrides are package data. The checker validates all required resources and package source bytes in both archives, verifies isolated import/metadata origin, enforces the locked retrieval thresholds, and compares ordered query paths and scores. Missing/corrupt resources, Python-module omissions, invalid metrics, import fallback, and report/artifact collisions fail validation.

**Task 2.** The test workflow runs the shared checker. Publication validates the report checksum and exact artifact filenames/hashes before the unchanged OIDC publisher consumes the same bytes. No rebuild occurs after validation. Remote CI, OIDC authentication, and environment reviewer enforcement were not tested locally.

**Task 3.** The benchmark persists valid retrieval and primary faithfulness separately. Regressions, invalid measurements, and local failures fail. Only typed transient primary-provider errors qualify for at most one workflow retry, corroborated by passing retrieval and an explicitly unavailable primary result. Required retrieval and any completed faithfulness remain checked. Existing benchmark stdout consumers pass.

**Task 4.** Missing or invalid selected CPU evidence and failed local measurements fail validation. The same two CPU axes remain blocking; valid wall-clock, directory-load, and reranker regressions remain advisory. Baselines and variance methodology are unchanged.

**Task 5.** README, tests README, and Unreleased notes describe the implemented checks and their limits. Final serial suite: **1,505 passed, 2 XFAIL, 78 XPASS; 93.03% coverage**. The final clean sdist-to-wheel check validates 45 package/resource members and identical ordered results on all 40 locked queries (P@1/R@3 both 1.0 with attune-help 0.13.0). Final-wheel missing-resource controls and the actual publication checksum scripts reject altered bytes. See [evidence](evidence.md), [distribution receipt](receipts/task-5-distribution.json), [negative controls](receipts/task-5-negative-controls.json), and [suite receipt](receipts/task-5-tests.xml).

**Review distinction.** Local Codex review and tests satisfy the authorized task-review mechanism. Patrick's separately requested Claude subscription cross-review is advisory. Patrick supplied an explicit subscription review command after the exact-payload question. The six-file Task 3 review completed. Full-code triage found three real issues, now corrected (F3/F6/F9), seven unsupported claims, and four intentional behaviors. The [original claims and dispositions](cross-review.md), [status receipt](receipts/task-3-cross-review.json), and [independent follow-up](receipts/task-3-cross-review-independent.md) remain separate from the local task gate. The refreshed full-suite and distribution receipts include the corrections; the original five Task 5 receipts are preserved under `task-5-pre-cross-review-*`. The follow-up review assigns no new numerical score.

**Release and later phases.** The original Phase 1 checkout remains at source version 1.2.0; its earlier release check stopped because that version already exists on PyPI. Patrick subsequently invoked release-execute (D13). Release 1.2.1 preparation is in progress in the separate `/private/tmp/attune-rag-release-1.2.1` checkout, based on verified current main `f00815d`. Phase 1 remains complete. These receipts do not claim a published release or passing remote CI. Cache, editor, generation, provider lifecycle, and broader retrieval work remain in the [review](../../reviews/library-review-2026-09-09.md) and [goal map](../RELIABILITY_GOALS.md) for separate scoping. The [workflow portability TODOs](../WORKFLOW_PORTABILITY_TODO.md) include multiple backends and explicit subscription/API routing.
