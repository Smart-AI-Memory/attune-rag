> **Status: approved scope and risk record; Milestone 1 in progress.**

# Risks and limits

| Risk | Response and acceptance limit |
|---|---|
| Secure dependency floors conflict with supported Python/platforms or SDK contracts | Resolve minimum/current sets and test benign adapters. Surface conflicts before changing support; no vulnerable fallback. |
| Lock remediation leaves downstream installs affected | Enforce floors in affected published extras and inspect wheel metadata; deliberate conflicting requests must fail. |
| No listed fix for an advisory is mistaken for repaired | Refresh the advisory and inspect affected ranges. Record an upstream block/explicit disposition rather than suppressing it. |
| Native grounding rules request legacy citation syntax | Preserve policy semantics while using provider-native attribution; assert payload and citation mapping. |
| Larger recorded native context increases memory/judge input cost | Preserve actual evidence, disclose size, and plan any live run. Never hide truncation behind a completed score. Common budgets are deferred. |
| Strict local refusal changes existing caller behavior | Limit it to explicitly selected strict + zero hits, document it, preserve other variants, add observable metadata, test no-SDK route. |
| Empty or incomplete answers improve apparent faithfulness | Show unsupported-answer rates, claim counts, appropriate refusal, pipeline abstention, and answerability/usefulness separately. No new numerical target without evidence. |
| CI artifacts disclose user corpus data | Automatic upload only for approved public/synthetic inputs; explicit retention and no automatic private-corpus upload. |
| Additive result fields alter dataclass serialization/repr | Preserve old construction defaults, inspect serializers/snapshots, document new evidence text exposure and release compatibility. |
| Green stubs are described as live model validation | Label payload, dependency resolution, installation, and live measurements separately. Record unperformed checks. |

## Deliberate deferrals

- Cache ownership, fingerprints, freshness, dense matrix invalidation, and hybrid
  recovery belong to G2. This phase records the evidence used; it does not certify
  that a corpus remained current before retrieval.
- Expansion-outage caching, stage degradation diagnostics, and event-loop blocking
  remain separate G2/G5 candidates. Strict abstention is triggered by zero hits,
  not an inference that an optional stage was healthy.
- Editor YAML/path/permission/containment defects remain G4. Do not bundle mutation
  safety work with dependency metadata or generation changes.
- General CLI JSON cleanup, corpus documentation mismatches, provider lifecycle
  updates, dashboard corpus binding, and broader usage fixes remain G5/G6.
- New retrievers, universal confidence thresholds, chunking, default model changes,
  new corpus benchmarks, and general telemetry remain outside this phase.
- [Workflow portability](../WORKFLOW_PORTABILITY_TODO.md) remains separately owned;
  this spec does not execute Claude-backed workflows or grant cross-project edits.
- Quantitative answer-level grounding targets and representative held-out evaluation
  are follow-up decisions after evidence is inspectable. Preserve existing gates.

M1's zero-known-critical criterion is time- and advisory-database-bound. It is not
a guarantee that dependencies are vulnerability-free. M2 makes grounding policy
and evidence trustworthy; it does not prove all generated claims are true or that
prompt injection has been eliminated.
