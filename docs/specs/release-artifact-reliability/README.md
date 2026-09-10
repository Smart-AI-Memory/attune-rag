# Phase 1 — Release artifact and CI reliability

> **Status: Phase 1 complete locally; all five tasks accepted under authorized auto-run.**

**Owner:** Patrick Roebuck. **Created:** 2026-09-09.

**Program goal, accepted by Patrick:** The installed library preserves measured retrieval quality, returns current and correctly grounded results, and fulfills its documented contracts.

**Phase 1 outcome:** The built distribution preserves source retrieval behavior, and CI cannot report successful validation when a required local check failed or a measured regression occurred.

Patrick selected the full reliability program, then clarified that this session should create a **Phase 1 spec**, using the grounded review to scope separate Phase 2+ specs later. Phase 1 covers packaging and CI integrity. Patrick subsequently approved implementation and auto-run of this phase (D5–D9). Publication and deferred retrieval/editor/generation changes remain outside that Phase 1 authorization. D13 separately records Patrick’s later release-execute request and the 1.2.1 release preparation now in progress.

The checked-out code, including build and workflow configuration, is the ultimate authority for current state. The [executive review](../../reviews/library-review-2026-09-09.md) supplies goals and code-grounded findings; the [evidence record](evidence.md) pins the reviewed checkout and Phase 1 reproduction results. Resolve conflicting prose against the code and recheck those premises before implementation if the checkout changes. Requirements describe desired behavior, not a claim that it already exists.

The [reliability goal map](../RELIABILITY_GOALS.md) turns the review into goals for this and later specs. This phase addresses G1 only; the remaining goals are candidates for collaborative scoping, not preapproved work.

- [Requirements and acceptance criteria](requirements.md)
- [Design and failure-outcome policy](design.md)
- [Task overview](tasks.md)
- [Canonical XML task ladder](../../../.claude/plans/release-artifact-reliability.md)
- [Confirmed scoping decisions](decisions.md)

**Current boundary:** All five tasks are accepted under Patrick’s authorized auto-run; the canonical workspace reached terminal revision 19. [Decisions](decisions.md), [evidence](evidence.md), and [lifecycle receipts](gate-receipts.jsonl) distinguish approval, local test results, and external checks. The separate Claude subscription cross-review is complete: all 14 claims were checked against the full code, three were corrected, seven were disproven, and four describe intentional behavior. [Triage and follow-up evidence](cross-review.md) remain advisory, separate from task gates. Release 1.2.1 preparation is now separately authorized under D13; publication and remote CI success are not claimed here.

**Session done when:** The approved five-task implementation and its local validation are complete, the canonical state records the outcomes, and the evidence identifies any remaining limits. The initial planning-only done condition was extended by D6 and D9.

**Implementation done when:** Required runtime data survives sdist and wheel construction; source/wheel ordered retrieval results agree and meet the existing locked thresholds; the release workflow validates the artifact bytes it subsequently publishes; regression, invalid-data, local-failure, and narrow provider-unavailability cases have tested outcomes; README gate claims match those outcomes.

**Implemented behavior to retain:** Base dependency footprint and opt-in retrieval tiers; current query set and quality/performance baselines; conditional faithfulness selection and disclosed missing-credential mode; configured OIDC publication and `pypi` environment. Current remote required-reviewer settings were not inspected.

**Implemented policy repair:** Task 3 restricts retry handling to typed primary provider transport, rate-limit, and server errors. A retry also requires valid passing retrieval; completed primary results are preserved. This Phase 1 repair did not itself authorize a release; the later D13 request supplies the separate release-execution authorization.

**Later scoping:** Cache isolation/freshness, effective corpus fingerprints, grounding and abstention, editor safety, async responsiveness, CLI and onboarding behavior, provider lifecycle, richer diagnostics, and representative evaluation remain in the executive review. No later phase is approved or automatically activated by this spec.
