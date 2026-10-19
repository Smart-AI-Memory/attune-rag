# Phase 1 — Release artifact and CI reliability

> **Status: Phase 1 complete locally; all five tasks accepted under authorized auto-run.**

**Program goal:** The installed library preserves measured retrieval quality, returns current and correctly grounded results, and fulfills its documented contracts.

**This phase:** Packaging and CI integrity only, grounded in the [review](../../docs/reviews/library-review-2026-09-09.md) and [Phase 1 requirements](../../docs/specs/release-artifact-reliability/requirements.md). Creation and goal agreement do not approve task execution.

**Current-state authority:** Inspect the selected code revision and active build/workflow configuration before implementation. Code governs current-state claims; the review supplies candidate goals, and prior specs describe intent. Correct any conflicting premise before acting on it, without silently expanding this phase's scope.

**Acceptance:** Source and isolated installed-wheel retrieval agree; required resources are present; the release uses validated artifact bytes; measured regressions and invalid/local failures cannot pass; catch-all inconclusive handling is restricted to classified transient provider failures; public gate claims match the active checks. No baseline reset or broader retrieval/editor/generation changes.

<task id="1" name="Preserve package resources and prove source-wheel parity">
  <objective>Repair omitted bundled resources and add an installed-artifact check using the existing locked query set and matching dependencies.</objective>
  <files-to-create>
    <file path="scripts/check_distribution.py">Portable clean build/resource/import/parity verification helper; no provider calls.</file>
    <file path="tests/unit/test_distribution_contract.py">Behavioral checks for required resources, isolated import origin, deterministic parity, and removed-resource negative controls.</file>
  </files-to-create>
  <files-to-modify>
    <file path="pyproject.toml">Declare the complete required runtime package data.</file>
  </files-to-modify>
  <validation>
    <check>Clean tracked snapshot builds sdist, then wheel from sdist; required resource bytes survive both.</check>
    <check>Outside-checkout wheel import is verified; source and wheel return identical ordered paths and scores on the locked 40-query set using identical attune-help dependency.</check>
    <check>Both runs satisfy the active threshold artifact's retrieval subset (P@1/R@3), explicitly skipping faithfulness in this provider-free artifact check and leaving the baseline unchanged.</check>
    <check>Remove each corpus override file separately and prove validation fails; replay the check against reviewed commit 25d2cf0.</check>
  </validation>
  <risks><risk severity="medium">Editable imports or stale build outputs can falsely pass; isolate builds and assert module origin.</risk></risks>
</task>

<task id="2" name="Validate the artifact consumed by the publisher">
  <objective>Run package validation in the build job before publication and ensure the publisher consumes exactly the validated files.</objective>
  <files-to-create>
    <file path="tests/unit/test_distribution_workflow.py">Workflow behavior checks for failed validation, hash mismatches, and absence of a rebuild after validation.</file>
  </files-to-create>
  <files-to-modify>
    <file path=".github/workflows/tests.yml">Add a bounded PR artifact-validation job using the shared distribution checker.</file>
    <file path=".github/workflows/publish.yml">Validate built artifacts, carry checksum evidence, and preserve the configured OIDC publisher and pypi environment.</file>
  </files-to-modify>
  <validation>
    <check>A deliberately incomplete wheel fails before upload/publication.</check>
    <check>Hash checks bind build validation to the files downloaded for publication; altered bytes fail.</check>
    <check>OIDC permissions and the pypi environment reference remain configured; remote reviewer enforcement is not inferred from this file, release-approval requirements are unchanged, and tests do not publish.</check>
  </validation>
  <dependencies><dep>1</dep></dependencies>
  <risks><risk severity="medium">A second unchecked build invalidates the evidence; publish only the already-validated artifact set.</risk></risks>
</task>

<task id="3" name="Preserve regression failures through the quality workflow">
  <objective>Distinguish pass, measured regression, invalid measurement, local failure, and known transient provider unavailability without losing independent retrieval results.</objective>
  <files-to-create>
    <file path="tests/unit/test_quality_workflow_outcomes.py">Run the real command/workflow control flow with stubbed provider and process outcomes.</file>
  </files-to-create>
  <files-to-modify>
    <file path="src/attune_rag/benchmark.py">Expose sufficient structured outcome information to distinguish causes and retain required retrieval evidence.</file>
    <file path="scripts/check_thresholds.py">Reject invalid/nonfinite/out-of-domain required metrics while preserving existing valid report behavior.</file>
    <file path=".github/workflows/benchmark.yml">Always honor completed regressions; replace catch-all retries/inconclusive handling with the proposed classified provider exception.</file>
    <file path="tests/unit/test_benchmark.py">Verify structured outcomes and existing CLI compatibility.</file>
    <file path="tests/unit/test_check_thresholds.py">Cover missing, malformed, nonfinite, out-of-domain, and SHA-mismatch measurements.</file>
  </files-to-modify>
  <validation>
    <check>Measured regressions fail on both sides of internal CLI cutoffs and active locked thresholds; a completed low score is never retried to pass.</check>
    <check>NaN, infinity, missing/corrupt reports, and unclassified exceptions cannot become successful validation.</check>
    <check>Known transient provider failure gets at most one retry; continued unavailability affects only explicitly reported faithfulness status.</check>
    <check>Required retrieval pass/fail is preserved when faithfulness is unavailable or intentionally skipped; replay the 0.50 review receipt.</check>
    <check>Preserve benchmark stdout fields consumed by scripts/measure_baseline_variance.py and run tests/unit/test_measure_baseline_variance.py alongside the benchmark compatibility checks.</check>
  </validation>
  <dependencies><dep>2</dep></dependencies>
  <risks><risk severity="medium">Overbroad outage classification recreates the defect; making all provider outages blocking would depart from the narrow availability exception proposed in this plan.</risk></risks>
</task>

<task id="4" name="Require valid measurements for existing blocking performance checks">
  <objective>Make failed or invalid local CPU measurement visible as failed validation while retaining advisory status for other axes.</objective>
  <files-to-create>
    <file path="tests/unit/test_perf_workflow_outcomes.py">Actual workflow outcome checks for process failure and invalid selected metrics.</file>
  </files-to-create>
  <files-to-modify>
    <file path=".github/workflows/perf.yml">Remove successful-validation handling for failed required local measurement.</file>
    <file path="scripts/format_perf_delta.py">Validate selected blocking metric inputs and preserve advisory-only deltas.</file>
    <file path="tests/unit/test_format_perf_delta.py">Check valid/pass/regression/invalid metric outcomes without resetting baselines.</file>
  </files-to-modify>
  <validation>
    <check>CPU regression, missing/nonfinite selected CPU metric, and required local measurement crash fail.</check>
    <check>Advisory-only wall-clock/reranker/directory changes remain advisory.</check>
    <check>Existing baseline values, two blocking CPU axes, and variance method are unchanged.</check>
  </validation>
  <dependencies><dep>3</dep></dependencies>
  <risks><risk severity="medium">Measurement flakiness may surface as red CI; diagnose the infrastructure rather than silently reporting a pass.</risk></risks>
</task>

<task id="5" name="Reconcile Phase 1 claims and verify the completed boundary">
  <objective>Align public packaging/gate claims with verified behavior and preserve the remaining review findings for separate phase scoping.</objective>
  <files-to-modify>
    <file path="README.md">Correct only package quality, active thresholds, conditional faithfulness, and performance-gate claims.</file>
    <file path="tests/README.md">Document source versus distribution checks and workflow outcome verification.</file>
    <file path="CHANGELOG.md">Record completed Phase 1 changes once implemented, without selecting a release version here.</file>
    <file path="docs/specs/release-artifact-reliability/tasks.md">Record actual approved task outcomes and remaining scope.</file>
    <file path="docs/specs/release-artifact-reliability/evidence.md">Append final artifact hashes and verification receipts, preserving review evidence.</file>
  </files-to-modify>
  <validation>
    <check>Run the applicable serial suite and the isolated distribution check in a fresh correctly installed environment.</check>
    <check>Replay the negative controls and workflow outcomes after integration; record exactly which checks were local, mocked, or live.</check>
    <check>README values and blocking/advisory claims agree with active configuration; broader review findings remain deferred.</check>
  </validation>
  <dependencies><dep>4</dep></dependencies>
  <risks><risk severity="low">Do not expand documentation reconciliation into unrelated API/provider fixes or claim remote/live validation that was not run.</risk></risks>
</task>

<!-- spec-state: {"schema_version": 1, "completed": ["1", "2", "3", "4", "5"], "current": null, "auto_run": true, "last_updated": "2026-09-10T02:19:23.005698+00:00"} -->
