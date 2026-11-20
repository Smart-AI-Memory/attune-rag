> **Status: M1 merged in PR #224; M2 implementation complete locally, with candidate CI and review tracked separately.**

# Phase 2 — Dependency security and grounding integrity

**Owner:** Patrick Roebuck. **Created:** 2026-10-19. **Layer:** attune-rag.
**Code baseline:** `0ebb72dd0727bf8261a762f472cfdcf38873ddac` (released 1.2.1).
**Target version:** choose at implementation review; this draft makes no release change.

The installed library should preserve measured retrieval quality, return current
and correctly grounded results, and fulfill its documented contracts. This phase
starts with the critical dependency alerts and the grounding issues identified
in the library review and rechecked against released code.

The two milestones are independently reviewable and can ship separately:

| Order | Milestone | Observable result |
|---|---|---|
| 1 | M1 — Secure supported dependency paths | Published requirements and the lock exclude known critical affected versions across supported installs; resolved environments and remaining alerts have inspectable receipts. |
| 2 | M2 — Preserve grounding policy and evidence | Native and legacy generation honor the selected policy; strict empty retrieval abstains locally; evaluation records the evidence actually supplied and exposes unsupported answers. |

Patrick selected this no-evidence policy on 2026-10-19: **strict mode abstains;
other prompt variants retain a documented generative fallback.** This decision
applies to both generation paths. It does not change the default prompt variant,
retriever, scoring, or corpus-specific abstention threshold.

The pre-M1 dependency snapshot contained 27 open lockfile alerts: 4 critical,
11 high, and 12 medium. The reviewed critical packages are sentence-transformers,
claude-agent-sdk, AnyIO, and PyJWT. M1 excludes those affected ranges on main; published 1.2.1 remains unchanged. Their exposure
and the limits of lockfile evidence are recorded in [evidence](evidence.md).

The reviewed 1.2.1 native path omitted the selected grounding instructions and can send
more source text than it records for evaluation. Both strict empty-hit paths called generation. Deterministic replay against the baseline confirms these
behaviors; it does not measure live answer quality.

## Reading order

- [Requirements and acceptance criteria](requirements.md)
- [Design and compatibility choices](design.md)
- [Implementation tasks and milestone gates](tasks.md)
- [Confirmed decisions and proposals](decisions.md)
- [Current-code and review evidence](evidence.md)
- [Risks and deferred work](risks.md)
- [Completion review](completion-review.md)

## Session contract

**Original drafting mode:** specification and planning. **Outcome:** a reviewable Phase 2 spec that
turns the dependency and grounding review into bounded, code-backed milestones.
**Done when:** requirements trace to current code and saved evidence; each task
has positive and negative acceptance cases; compatibility, uncertainty, product
decisions, and authorization boundaries are explicit; the draft passes link and
scope checks; completion review is recorded. **Scope ceiling:** one spec pass,
two milestones; no implementation, provider calls, release, or publication.

**Implementation done when:** M1 and M2 meet their acceptance criteria with
candidate-specific evidence. A spec draft, an installed-version inventory,
and a passing mean faithfulness score alone do not establish that outcome.

This phase addresses [G3 and the new dependency goal G7](../RELIABILITY_GOALS.md).
[Phase 1](../release-artifact-reliability/README.md) shipped in
[1.2.1, PR #223](https://github.com/Smart-AI-Memory/attune-rag/pull/223).
Cache freshness, editor safety, general onboarding, async responsiveness, and
workflow portability remain separate work. The older safe-abstention spec concerns
retrieval thresholds; this phase concerns generation after retrieval.

**Completion review:** complete for this draft; [record and limits](completion-review.md).

## Milestone 1 execution contract — 2026-10-19

Patrick approved Phase 2 and explicitly authorized Milestone 1, the critical
dependency repairs. Scope: T1–T2, R1–R2, and their inherited R7 checks. Done when
published metadata and the lock exclude known critical affected versions,
minimum/current installation and platform-resolution evidence is inspectable,
and remaining advisories have explicit dispositions. No grounding implementation,
new paid model calls, merge, or publication is included. Completion review: complete for the implementation checkpoint; CI acceptance
and Patrick's review remain tracked by PR #224. The earlier planning completion
remains historical.

[Milestone 1 implementation and validation](m1-validation.md): local checks pass;
[PR #224](https://github.com/Smart-AI-Memory/attune-rag/pull/224) tracks exact-head CI and review.

## Milestone 2 execution contract — 2026-10-20

Patrick authorized M1 merge and T3–T5 implementation, followed by the bounded
CI allowance in D11. M1 is merged at `e95a3b5`, whose tree matches the reviewed
candidate. M2 has a separate branch from that merge. Runtime policy/evidence,
report capture, and offline examples are implemented; see
[validation](m2-validation.md), [core design](m2-implementation-design.md), and
[independent review](m2-review.md). Exact-head PR checks govern remote acceptance.
No M2 merge, release, or publication is authorized. Completion review: pending
for the candidate CI checkpoint; local implementation review is recorded below.
