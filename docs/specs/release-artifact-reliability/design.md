# Phase 1 design

> **Status: Phase 1 complete locally; all five tasks accepted under authorized auto-run.**

Use the existing build backend, golden-query loader, scoring implementation, threshold files, and workflows. Add the missing checks at their boundaries rather than creating a second benchmark framework.

**Artifact flow.** Clean tracked source → sdist → wheel from sdist → isolated installed-wheel probe → recorded checksums → upload → publish the verified files. Source comparison runs separately with the same dependency versions. An artifact probe must assert its import origin before measuring; a successful import from `src/` would invalidate the test. Runtime resource membership and byte content are checked independently of score parity so currently empty override files remain protected.

Packaging tests should use temporary directories and portable paths. The review's existing local virtual environment contains stale attune-rag distribution metadata; do not rewrite production version constants to match it or use it as the release environment. Fresh artifact validation resolves this ambiguity by construction.

**Implemented outcome policy.** The reviewed baseline converted every remaining benchmark failure into inconclusive success. Task 3 replaces that broad handling with these classified outcomes:

| Observed outcome | Required retrieval/local check | Conditional faithfulness |
|---|---|---|
| Valid measurements meet active thresholds | Pass | Pass |
| Valid measurements violate active thresholds | Fail; no retry for a better score | Fail; no retry for a better score |
| Missing/malformed/nonfinite/out-of-domain result | Validation failure | Validation failure when selected |
| Unexpected local exception/process failure | Fail | Fail; do not label a provider outage |
| Known transient provider unavailability | Continue independent local checks | Retry once; still unavailable → explicit inconclusive |
| Credentials absent before optional stage selection | Continue independent local checks | Explicitly skipped under current policy |

Prefer separate retrieval and faithfulness outcomes with an aggregate verdict. Do not infer the cause of failure solely from exit code 1 or absence of a report. If extending the benchmark output/exit contract is necessary, keep normal reports backward compatible, name new outcomes, and test current CLI consumers. Typed external errors should carry sanitized categories rather than full credential-bearing exception payloads.

The same distinction applies to performance: existing advisory-only changes remain advisory, while no valid result for a required CPU measurement is a failed check. Do not promote wall-clock or other axes by accident. The per-PR measurement is local and explicitly LLM-free.

**Verification architecture.** Three kinds of checks cover different promises:

1. Resource and installed-package tests verify the distribution people consume, including negative controls with a removed resource.
2. Behavioral parity compares ordered query results using identical corpus inputs. Existing active thresholds provide the minimum quality bar.
3. Workflow tests exercise actual shell/command outcome propagation with stubbed provider responses, process failures, and corrupt reports. A helper-only test cannot prove that the workflow does not skip the helper.

The release workflow must consume the exact validated artifact set; checksum evidence guards against a second unchecked build. Run package checks in the build job before the configured `pypi` environment publication job. Preserve OIDC and that environment reference. Remote required-reviewer enforcement was not inspected; this design does not claim it exists or change release-approval requirements.

**Implemented machinery to retain.** `scripts/check_thresholds.py` verifies the configured query hash and thresholds; `.github/workflows/benchmark.yml` selects conditional faithfulness and invokes `scripts/smoke_check_gate.sh`; `.github/workflows/perf.yml` selects `keyword_retriever_retrieve.cpu` and `rag_pipeline_run.cpu` as blocking metrics. These files establish current behavior. The archived `release-quality-baseline` spec supplies historical intent for a narrower transient-error policy; implementing that distinction is a Phase 1 change. `perf-baseline-multi-run` follow-up statistics remain separate; this is not a re-baseline or a new soak period.

**Tradeoffs.** Required local measurement failures now make CI red; this is stricter than the reviewed baseline performance crash path. The advantage is truthful validation. The strongest counterargument is infrastructure noise; address it by fixing reproducible measurement failures and reporting their cause, while retaining the narrowly defined provider exception. Do not weaken thresholds to make a failing artifact pass.

**Rollback.** Each task is independently reviewable. Restore the last known-good artifact/check implementation if a regression is found. Preserve failure evidence and never use a silent success conversion as the rollback strategy. Release or rollback actions require their existing authorization and validation.
