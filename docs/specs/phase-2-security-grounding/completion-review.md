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
