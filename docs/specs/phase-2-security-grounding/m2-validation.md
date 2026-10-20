# Milestone 2 validation

**Status:** local implementation and independent review pass; exact-head PR CI
and human acceptance remain separate. Base: merged M1 `e95a3b5`. This record
makes no M2 merge, release, or publication claim.

[Local receipt](receipts/m2-local-validation.json) binds the four runtime files,
final wheel/sdist, query lock, suite counts, and negative controls. Full local
XML, artifact, frozen review, and reproduction evidence is retained in the
project's private review evidence. CI publishes its own candidate-specific
artifact and grounding reports for inspection.

| Check | Result |
|---|---|
| Full minimum Python 3.10.11 suite | 1,637 passed; 2 expected failures; 78 advisory passes. |
| Full current Python 3.13.14 suite | Same result; no failures or errors. |
| Policy/path matrix | All four variants on legacy/native/unsupported-native paths; overrides, validation, source-data separation. |
| Strict empty hits | Exact local response, no named-provider initialization, no calls, empty evidence/citations, explicit abstention metadata. |
| Native evidence | Exact immutable titles/text; duplicate titles, Unicode, whitespace, tail-only facts; fixtures around/above the legacy 20,000-character cap. |
| Report/workflow controls | Partial judge failures, denominators, malformed verdicts, pass/regression/auth/outage/missing reports, both attempts and retention. |
| Negative controls | Removing native policy: 4 failures; removing strict short circuit: 6 failures; restoring native truncation: 16 failures. Unmodified control: 57 passes. |
| Independent review | One P2 provenance finding reproduced and closed; final delta accepted, actual reviewer runtime identity unavailable. |
| Source/sdist/wheel | Byte/resource checks and identical ordered 40-query retrieval pass; P@1/R@3 remain 1.0. |
| Base-only installed wheel | 15 installed distributions; no Anthropic/Agent SDK/torch/model2vec/sentence-transformers; named strict refusal and documented example execute without keys. Dependency check passes. |
| Static checks | Ruff passes. The existing unrelated invalid-noqa warning remains. |

The final suites ran after the reviewed provenance changes. A later equivalent
type-check syntax adjustment required by pinned lint passed all 13 report controls
on Python 3.10, received delta review, and has a regenerated artifact receipt. The final artifact
resources match those same reviewed runtime bytes. Model defaults, dependency
metadata/lock, golden queries, active quality thresholds, and performance
thresholds are unchanged. No API key was present in offline suite environments.
No model was downloaded or loaded for these controls.

T3 preserves evidence before T4's changed generation. Completed gate measurements
remain in their original keys; partial records have no fabricated scores.
[The guide and executable example](../../grounding.md) cover the installed
synthetic-corpus journey and a fixed unsupported-claim fixture.

The approved live plan is one automatic post-M1 main 40-query gate and one M2 PR
40-query gate, with only the existing classified transient retry. The allowance
is $50; no manual reruns, native comparisons, or extra paid experiments are
approved. Actual provider billing is not available from these local receipts.
The live gate uses unchanged mean faithfulness 0.9698 and the locked queries.
Mean faithfulness, unsupported-answer rate, zero-claim/non-answer counts and
refusals must be disclosed separately. Payload checks do not establish live model
quality, useful refusals, or prompt-injection resistance.
