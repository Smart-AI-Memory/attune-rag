> **Status: Phase 2 approved 2026-10-19; Milestone 1 execution authorized.**

# Decisions

## D1 — Program goal and authority (inherited)

The accepted goal is that the installed library preserves measured retrieval
quality, returns current and correctly grounded results, and fulfills documented
contracts. Patrick directed that the code be the ultimate authority for current
state. Reviews supply evidence and candidate goals; desired behavior is a separate
claim. See [Phase 1 decisions D1–D4](../release-artifact-reliability/decisions.md).

## D2 — Phase 2 starts with dependencies and grounding (confirmed)

On 2026-10-19 Patrick requested a Phase 2 spec using the review and current code,
starting with critical dependency alerts and grounding issues. This authorizes
spec creation. The two-milestone plan and task details remain proposed for review.
Phase 1 is released; its old drafting status is not the current implementation state.

## D3 — Strict mode abstains without evidence (confirmed)

On 2026-10-19 Patrick selected: “Strict mode abstains; other variants keep
fallback.” Apply strict rules consistently to native and legacy generation.
The proposed deterministic exact refusal uses the existing strict prompt's
sentence; no provider is needed when retrieval returned zero hits. Nonempty
strict context still uses generation. The selected policy does not change the
current default `citation` variant or retrieval threshold.

## D4 — Two independently reviewable milestones (proposed)

Finish critical dependency remediation first; keep grounding in a separate
milestone/PR. Address all four critical advisory families, not only demonstrated
exploit entry points. Inventory and disposition the remaining high/medium alerts
without silently expanding this into a general dependency modernization project.

## D5 — Preserve native evidence selection (proposed)

Retain full native source documents, record them exactly, and stop separately
capping their evaluation rendering. Keep the legacy cap. Add defaulted
`generation_documents` and `abstained` result fields, with no new public export.
Review constructor/serialization compatibility and privacy before accepting the
implementation. A common context budget is deferred pending measurement.

## D6 — Define all as runtime features (proposed)

Include the complete Claude runtime extra in `all`; exclude separately named
maintainer author/dev tools and correct the misleading “every extra” wording.
Validate metadata against the chosen runtime union.

## D7 — Keep current gates; expose answer-level evidence (proposed)

Upload public/synthetic evaluation reports with explicit per-answer and per-claim
metrics and 14-day retention. Preserve the current query set, mean-faithfulness
threshold, refusal handling, and provider-unavailability classification. New
answer-level blocking targets require representative evidence and a later decision;
this draft does not promise a numerical unsupported-answer reduction.

## D8 — Approval and spending boundary

This is a reviewable spec draft. No Phase 2 implementation, provider call,
workflow dispatch, merge, release, or publication is approved here. Earlier release
permissions and its $50 planning allowance are not treated as a standing Phase 2
budget. Implementation approval must name the accepted scope; paid validation
requires an explicit plan and allowance.

Before implementation, ratify D4–D7, recheck current code/advisories, resolve any
incompatible secure dependency floor, and select an API-compatible release plan.
Do not infer approval from the existence of these documents.

## D9 — Phase 2 approved; execute Milestone 1 (confirmed 2026-10-19)

Patrick stated: “Phase 2 is approved. Go ahead with milestone 1—the critical
dependency repairs.” This approves the Phase 2 plan and ratifies D4–D7; execution
is authorized for T1–T2 only. D8's former unapproved status is historical and
superseded to that extent. Milestone 2 remains unstarted. No new paid model run,
merge, release, or publication authority is inferred.

The refreshed graph places AnyIO on Claude, Gemini, embeddings, transformers,
all, and dev paths. Critical bounds will be explicit on those affected extras.
Compatible adjacent MCP, fsspec, urllib3, and virtualenv bounds are included to
avoid retaining the same snapshot's high/medium affected versions on those paths;
this does not authorize general dependency modernization.

## D10 — Merge M1 and execute M2 (confirmed 2026-10-20)

Patrick requested: “Looks good. Merge PR #224, then start Milestone 2.”
PR #224 merged as `e95a3b50fff001f9e58238c6df2398d2bd694d19`; its tree
`955d0f1876ad94ee1fe6365837a49aaf54d21c53` equals the green reviewed head.
This authorizes T3–T5 implementation in a separate PR, not its merge or release.

## D11 — Bounded CI allowance (confirmed 2026-10-20)

Patrick approved a new $50 Anthropic allowance for the automatic post-M1-merge
40-query benchmark and one Milestone 2 PR 40-query benchmark, each retaining
the existing single classified transient retry. Use the existing Sonnet 5 answer
and Fable 5.1 judge defaults, locked queries and thresholds. Planning estimate:
$10–30 total before unusual retries; actual billing depends on tokens. No extra
paid experiments, native comparison passes, or manual reruns are approved.
