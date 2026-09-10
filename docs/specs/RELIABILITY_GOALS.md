# attune-rag reliability goals from the executive review

**Status: goal inventory for collaborative scoping. Phase 1's five-task plan and execution are approved; Task 1 is undergoing review. Later phases remain candidates.**

Patrick accepted the [2026-09-09 executive review](../reviews/library-review-2026-09-09.md) as a source of goals for the specs. Preserve the evidence and the reasoning behind each goal; fresh scoping must verify that its findings still apply.

**Current-state authority:** The checked-out code, including build configuration and workflow logic, is the ultimate authority for what the library currently implements. The review supplies candidate goals and explanations. Tests and reproductions support claims about the paths and environments they exercise; passing tests do not override contradictory implementation. Documentation and earlier specs describe claims or intended behavior. If they conflict with the code, record the discrepancy and use the code to describe current state. Each new spec must trace its premise to the selected code revision and relevant configuration; a proposed improvement remains a desired outcome until implemented and verified.

**Program goal:** The installed library preserves measured retrieval quality, returns current and correctly grounded results, and fulfills its documented contracts.

| Goal | Desired outcome | Review basis | Spec disposition |
|---|---|---|---|
| **G1 — Trust the release and its checks** | The distributable preserves source retrieval behavior; measured regressions and failed required local checks cannot pass; public gate claims match reality | Priority findings 1 and 3; source/wheel comparison; active-threshold drift | [Phase 1: release-artifact-reliability](release-artifact-reliability/README.md), execution in progress; Task 1 review pending |
| **G2 — Retrieve current content from the intended corpus** | Cache identity cannot cross corpora; updates invalidate stale matrices; effective corpus fingerprints cover retrieval-affecting inputs | Priority finding 2; corpus-version and hybrid-fallback probes | Candidate for later spec scoping |
| **G3 — Make grounding, abstention, and provenance trustworthy** | Generation paths honor one explicit grounding policy; recorded evidence matches supplied evidence; missing grounding/degradation is visible; every cited excerpt remains traceable | Priority finding 4; findings 6, 7, and 11; diagnostic-record opportunity | Candidate for later spec scoping; fallback policy requires a product decision |
| **G4 — Preserve corpus integrity during editing** | Supported alias/path edits preserve YAML values, references, permissions, and containment; invalid/incomplete documents yield diagnostics rather than crashes | Priority finding 5; findings 12 and parser portion of 13 | Candidate for later spec scoping |
| **G5 — Make documented integrations work as written** | JSON output is machine-readable; examples run; corpus fields behave as documented; supported provider defaults work; async calls remain responsive; cache cost claims reflect actual behavior | Findings 8–10, 13–14; onboarding opportunity | Candidate for later spec scoping; may split into smaller independent specs |
| **G6 — Demonstrate useful retrieval on representative user data** | Measurements disclose denominators, distinguish hit rate from answer usefulness, include refusal/negative behavior, and guide improvements for long or changing documents | Measurement/positioning pushback; utility opportunities 2–3 | Candidate research/scoping goal; no new model, corpus-size claim, or feature design approved |

**Keep as design constraints:** Small corpus/retriever/provider interfaces; deterministic and explainable keyword baseline; optional heavyweight dependencies; explicit provenance; corpus-specific calibration; reusable headless editor primitives; existing cross-platform test coverage. Reliability work should strengthen these properties.

**How this inventory feeds a spec:** Select the goal and smallest useful outcome, inspect the intended code base and recheck the review finding, record code references and bounded reproduction evidence, define observable acceptance criteria, identify compatibility/product decisions, then create and review that phase's spec. Correct or retire findings that no longer match the code. Do not treat this table or a review recommendation as authorization to execute all phases.

**Sequencing:** Phase 1 is packaging and CI. The review identifies cache isolation and safe editing as substantial risks, but Patrick has not assigned a fixed order or phase numbers to the remaining goals. Retain that choice for the next scoping conversation rather than silently deciding it here.

**Cross-project follow-up requested during execution:** [Workflow portability TODOs](WORKFLOW_PORTABILITY_TODO.md) track checking Attune workflows with Claude and at least one non-Claude backend. Their likely implementation owner is attune-ai's workflow/runner layer; they do not expand this library's approved Phase 1 task ladder.
