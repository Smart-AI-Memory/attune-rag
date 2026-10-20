> **Status: completion review complete for this planning draft; implementation remains unapproved.**

# Completion review for the Phase 2 draft

**Goal:** The installed library preserves measured retrieval quality, returns
current and correctly grounded results, and fulfills its documented contracts.
This review updates existing dependency/grounding opportunities; it does not
start a new workstream or authorize implementation.

## R2 update — Secure supported optional dependencies

**Lens: review.** The refreshed snapshot still has 27 alerts, 4 critical. Current
source permits caller-selected transformer model loading on an affected locked
version. Other critical exploit preconditions are absent from inspected direct
calls; that limits exposure claims, not the need for supported dependency repair.
[E2](evidence.md#e2--critical-dependencies-refreshed-2026-10-19) and the metadata
receipt pin the basis.

**Follow-up:** proposed M1 in this spec, including published requirements,
resolved installation checks, and remaining-alert dispositions.
**Done when:** known critical affected ranges are excluded across supported paths
and candidate-specific receipts prove that property. **Effort:** bounded dependency
repair, with Python/platform resolution as the main uncertainty.
**Authority:** scoped proposal only; no dependency update or release approved.

## R3 update — Preserve inspectable grounding evidence

**Lens: expert.** The pinned replay confirms all-variant native policy bypass and
25,025 sent versus 20,000 recorded characters. Historical quality runs each flag
11/40 answers while passing their mean gate; missing per-query artifacts prevent
inspection of those historical claims. Strict empty-hit behavior was previously
intentional; Patrick has now selected local strict abstention with other fallbacks
retained. [E3–E4](evidence.md#e3--grounding-counterexamples-replayed-against-current-code)
separate deterministic observations, product choice, and live measurement limits.

**Follow-up:** proposed M2 records actual supplied evidence, enforces the selected
policy, and preserves per-answer reports before any new numerical target.
**Done when:** reviewers can trace unsupported claims to actual supplied sources,
strict empty hits make zero generation calls, and all provider paths carry the
selected policy without weakening inherited gates. **Effort:** medium core-path
and workflow repair, with dataclass compatibility and report privacy reviewed.
**Authority:** spec draft only; no implementation, new paid run, or publication.

## Next decision — numerical grounding targets

**Lens: ladder.** The current aggregate is insufficient to choose an answer-level
error budget or claim a numerical improvement. R6/D7 preserve this as a decision
after representative inspectable evidence, including appropriate refusal and
usefulness, is available. No new threshold or benchmark is authorized here.

**First user encounter:** an optional-feature installer meets the dependency
protection first; a strict caller then meets predictable no-evidence abstention.
A maintainer needs the per-answer artifact to assess grounding quality beyond the
mean. The draft decisions, requirements, opportunity entries, and validation results
have been read back. R2 and R3 are promoted to scoped proposals, not resolved
defects. The answer-level target remains an explicit future decision.

**Completion review: complete for spec drafting.** The
[validation receipt](receipts/spec-validation.json) records passing consistency,
evidence, and scope checks. No independent implementation review or live quality
measurement was performed; implementation remains unapproved.

## Milestone 1 / T1 checkpoint — 2026-10-19

**Lens: requirements.** Published-wheel metadata now excludes all four critical
families on the identified extras. Twelve affected-version requests fail against
the repair and resolve against the hash-verified published control. The lock
refresh excludes the 27 advertised affected ranges. This closes the T1 metadata
and lock implementation step; T2 foreign-platform execution and milestone review
remain pending. Evidence: [M1 validation](m1-validation.md) and its linked receipts.

**R2 disposition:** implemented in the candidate, awaiting CI and review. The
published 1.2.1 wheel remains unchanged, so downstream users do not receive this
repair until a separately authorized release. The next step is candidate review,
not automatic grounding work or publication. Completion review: pending for M1.

## Milestone 1 implementation checkpoint — 2026-10-19

**Goal:** secure supported installations while preserving measured retrieval and
existing documented contracts. **Completion review: complete for implementation;**
remote candidate acceptance remains visible in PR #224, and no merge/release is
performed by this review.

**R2 update (requirements):** T1–T2 now implement secure optional bounds, the
runtime-extra union, a targeted lock refresh, and inspectable environment/alert
receipts. Twelve repaired-wheel conflicts have twelve successful published-wheel
controls; 240 resolution cases and both local installations pass. Both offline
suites pass with unchanged 40-query artifact retrieval. Each of the 27 advertised
affected ranges is excluded in the candidate; OSV returns no known findings in
245 lookups. Scope/limits are retained in [validation](m1-validation.md).
The remaining acceptance work is exact-head CI plus human review. Users receive
the fix only after a separately authorized merge and release.

**R2 follow-up (ladder):** advisory data ages after a release. The existing
`security-scan.yml` reviews selected source patterns; this milestone's installed
and range audit is a point-in-time receipt. Candidate follow-up: scope a repeatable,
LLM-free dependency audit across lock and supported extras, with database timestamp,
explicit unavailable state, and minimum/current environments. Done when a maintainer
can refresh and inspect those results without weakening the gate or manually
reconstructing package paths. Effort: small CI/tooling task. This note grants no
new automation, implementation, spending, or publication authority.

**Expert lens:** completing `[all]` changes SDK availability for existing judge
auto-detection inside Claude Code. Installation docs now name that consequence
and the existing direct-API override. Neither a package import nor the stubbed
SDK test certifies account authentication or billing.

**R3 disposition:** Milestone 2 remains unstarted; the native policy/evidence and
strict fallback issues are still present. No grounding outcome is claimed here.
**First user encounter:** the optional-feature install gains secure version
bounds; maintainers encounter the dated audit-refresh follow-up next.

## M1 merge checkpoint — 2026-10-20

**Goal lens:** secure supported installations and inspectable grounded answers.
PR #224 is merged as `e95a3b5`; the full tree equals the reviewed candidate
`c2fbd85`. All 12 platform/Python checks passed before merge. The automatic
post-merge quality gate is a separate, budgeted run, not a new release.

**R2 update:** implementation is accepted on main; published 1.2.1 remains
unchanged. The existing audit-refresh opportunity remains dated and open; no
additional dependency work is inferred. **R3 next rung:** retain partial answer
evidence before changing generation. This is already authorized T3, not a new
opportunity. No distinct new opportunity was found at this merge checkpoint.
A strict caller still encounters the known generation defects first; M2 addresses
them. Completion review: complete for the merge checkpoint; pending for M2.

## Milestone 2 local implementation checkpoint — 2026-10-20

**Goal lens:** preserve measured retrieval and make grounded answers inspectable.
The strict caller now receives predictable local abstention; the maintainer can
trace an answer and unsupported claims to the evidence actually supplied. T3–T5
local evidence and limitations are in [validation](m2-validation.md).

**R3 update (requirements):** policy forwarding, no-provider strict abstention,
exact native evidence, defaulted result fields, per-attempt partial reports and
explicit denominators are implemented and pass local controls. Three mutations
prove that removing the repaired boundaries fails the acceptance tests. The final
minimum/current suites each pass 1,637 tests; source/distribution retrieval remains
1.0/1.0 on the locked set. Exact-head remote acceptance remains pending.

**Expert/review lens:** the independent review exposed a provenance class error:
filesystem location alone cannot establish code ownership. Both installed-wheel
and untracked-source counterexamples were reproduced before repair; unavailable
Git provenance stays null. No additional analogous source helper was found in the
bounded search. This closes the in-scope finding, not a new workstream.

**Existing next decision update (ladder):** inspectable answer-level evidence now
makes it possible to assess unsupported answers and zero-claim responses beyond
the mean. A representative usefulness/error target still needs later scoping;
local stub results provide no numerical live improvement claim. Retain the
existing numerical-grounding opportunity rather than inventing a new target.

**First user encounter:** strict callers see local refusal without an SDK/key;
maintainers then see explicit per-answer report denominators. Completion review:
complete for the local implementation checkpoint; pending for the PR CI handoff.
This record grants no additional implementation, spend, M2 merge, or publication.
