> **Status: approved 2026-10-19; T1–T2 in progress under Milestone 1 authorization. T3–T5 have not started.**

# Tasks and milestone acceptance

Order: T1 → T2 → M1 acceptance; then T3 → T4 → T5 → M2 acceptance.
M1 can ship independently. T3's report work precedes live evaluation of T4.
T1 is locally complete. T2 local validation passes; remote CI is pending. T3–T5 remain **not started**.

| Task | Deliverable | Requirements | Evidence required |
|---|---|---|---|
| T1 | Critical dependency bounds and regenerated lock | R1–R2 | Fresh advisory/graph inventory; wheel metadata; affected-version conflicts; compatible secure resolutions |
| T2 | Install compatibility and remaining-alert dispositions | R1–R2, R7 | Minimum/current environments; platform/Python coverage; lazy imports; benign SDK smoke tests; audit/disposition report |
| T3 | Inspectable grounding evaluation artifacts | R6–R7 | Schema and math fixtures; actual workflow-script pass/fail controls; per-attempt upload and privacy/retention checks |
| T4 | Shared grounding policy, strict abstention, exact evidence | R3–R5 | All-variant/provider-path matrix; long-tail evidence replay; no-SDK strict refusal; compatibility tests |
| T5 | Integrated verification and documented usage | R2–R7 | Relevant suite, golden and artifact parity receipts; runnable examples; candidate-specific limits |

## T1 — Exclude affected critical versions

Recheck the baseline and advisory boundaries. Trace each affected package through
the declared extras, choose compatible secure floors, update all affected metadata,
and regenerate `uv.lock`. Align the runtime meaning of `all` with the proposed
design. Do not change library version, provider model, retrieval scoring, or gate
thresholds to accommodate resolution.

Done when wheel metadata rejects affected critical versions and secure versions
resolve for each relevant path, with an explicit graph and unchanged lightweight
base. If a floor is not compatible, report the conflict and proposed alternative
before altering Python support or disabling a feature.

## T2 — Validate M1 as a supported installation

Check minimum secure and current dependency sets. Resolve the tested platform/
Python matrix, run isolated affected-extra installs on Python 3.10 and 3.13, and
retain the normal 12-job suite. Exercise package metadata, `pip check`, lazy
imports, injected encoders, and structured SDK option/result handling offline.
Review high/medium alerts individually; fix adjacent compatible dependencies where
required by the chosen secure graph and record the rest.

**M1 acceptance:** zero known critical affected versions in the supported lock and
tested installs; published metadata excludes affected critical ranges; remaining
alerts and test coverage are inspectable. A passing import or closed Dependabot
count by itself is insufficient. A release is a separate decision.

## T3 — Preserve evaluation evidence before changing behavior

Extend the existing report with the fields in R6 and upload reports for each
attempt, including partial/failing runs. Preserve required failure propagation and
retry classification. Document public/synthetic-only CI inputs and 14-day retention.
Use stored/stubbed verdicts to verify denominators and missing evidence handling.
Do not attempt to reconstruct the old release's missing per-query answers.

Done when a reviewer can trace a fixed unsupported claim from answer to the exact
supplied source, model/judge configuration, and run; failed/incomplete evidence
cannot look like a passing completed evaluation.

## T4 — Repair policy and evidence together

Write a short implementation design note for affected core paths before changing
them. Implement shared variant semantics, early input validation, strict local
empty-hit abstention, the defaulted result metadata, and native evidence capture
from the same documents sent to the provider. Update the tests that intentionally
codify the former strict fallback behavior.

Done when every R3–R5 path passes, including non-strict compatibility controls,
invalid input, unsupported-native fallback, long-tail evidence, citation-index
mapping, and existing result construction. Negative controls remove the specific
policy forwarding, abstention short-circuit, or exact-evidence behavior and show
the relevant tests fail. Payload tests are not live model quality measurements.

## T5 — Verify the complete supported journey

Run appropriate offline suites and inherited API/golden/artifact checks after the
final implementation changes. Exercise the documented install → corpus → strict
query → result-evidence sequence using a synthetic corpus outside the source tree.
Show a non-strict fallback example and an unsupported-claim report example.

If a live quality run is required by CI or proposed for acceptance, first present
its dataset/model/judge plan, estimated cost, retry limit, and requested budget.
The completed release's earlier $50 authorization is not a new Phase 2 allowance.
Use the unchanged thresholds and preserve all outcomes. Report any paid run that
was not performed; do not invent a before/after quality improvement.

**M2 acceptance:** deterministic policy/evidence/abstention contracts pass; reports
are inspectable; source/wheel and retrieval contracts remain intact; selected
required CI checks have candidate-specific receipts. Mean faithfulness and
answer-level observations are disclosed separately. No zero-hallucination claim.

## Checkpoints and boundaries

At each milestone, re-anchor on the program goal, inspect the evidence and limits,
record useful opportunities, and review the next task's premises before advancing.
Scope changes and incompatible support changes require a named decision. Spec
approval and implementation approval are recorded separately; neither constitutes
paid-run, merge, or publication authority.

**Planning acceptance:** draft files, source/advisory traceability, link/scope
validation, confirmed D3, explicit remaining proposals, and completed
[completion review](completion-review.md). No implementation task is marked done
by completion of this planning session.

## Execution record

- 2026-10-19: Phase 2 approved; Patrick authorized Milestone 1 only. T1 started
  after refreshing main/advisories and inspecting dependency paths. T2 validation
  is pending. Completion review: pending for the milestone.

- 2026-10-19: T1 requirements/lock repair and T2 local installation, resolver,
  artifact, SDK-type, and offline suite checks pass. See [Milestone 1 validation](m1-validation.md).
  Remote CI and final milestone completion review remain pending.
