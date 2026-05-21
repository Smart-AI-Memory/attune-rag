# Diagnostic-1 — KeywordRetriever paraphrase robustness

corpus: attune-help  version: 0.11.0

## Baseline (original golden queries)

| difficulty | n | P@1 | R@3 |
|---|---|---|---|
| easy | 10 | 100.00% | 100.00% |
| medium | 27 | 96.30% | 100.00% |
| hard | 3 | 100.00% | 100.00% |
| **overall** | **40** | **97.50%** | **100.00%** |

## Paraphrased (no-token-overlap variants)

| difficulty | n | P@1 | R@3 |
|---|---|---|---|
| easy | 20 | 0.00% | 30.00% |
| medium | 54 | 14.81% | 29.63% |
| hard | 6 | 16.67% | 16.67% |
| **overall** | **80** | **11.25%** | **28.75%** |

## Deltas

- Δ P@1 (overall): -86.25%
- Δ R@3 (overall): -71.25%

## Verdict

**STRONG** — keyword is brittle to paraphrasing. **EmbeddingRetriever case is real.** Proceed to design a spec: local-model first (no API dep), hybrid scoring, and a benchmark that includes this paraphrase set.

## Paraphrased R@3 misses (57)

- **gqp-001a** — `what catches unsafe input handling and leaked credentials`
  - expected: ['concepts/tool-security-audit.md', 'quickstarts/run-security-audit.md']
  - got: ['quickstarts/task-error-handling-design.md', 'concepts/task-error-handling-design.md', 'tasks/task-error-handling-design.md']
- **gqp-001b** — `I want to find places attackers could break in`
  - expected: ['concepts/tool-security-audit.md', 'quickstarts/run-security-audit.md']
  - got: ['concepts/tool-bug-predict.md', 'references/tool-security-audit.md', 'concepts/tool-workflow-orchestration.md']
- **gqp-002a** — `I have functions with no safety nets`
  - expected: ['concepts/tool-smart-test.md', 'quickstarts/generate-tests.md']
  - got: ['concepts/tool-doc-gen.md', 'concepts/tool-release-prep.md', 'quickstarts/skill-release-prep.md']
- **gqp-003a** — `suite went red after last merge`
  - expected: ['concepts/tool-fix-test.md']
  - got: ['tips/after-bug-predict.md', 'concepts/task-git-workflow.md', 'concepts/workflow-chain-prediction.md']
- **gqp-004a** — `give me a once-over on this module`
  - expected: ['concepts/tool-code-quality.md']
  - got: ['concepts/tool-doc-gen.md', 'references/skill-doc-gen.md', 'references/tool-test-generation.md']
- **gqp-004b** — `evaluate craftsmanship of my changes`
  - expected: ['concepts/tool-code-quality.md']
  - got: ['concepts/task-database-migrations.md', 'concepts/tool-bug-predict.md', 'concepts/task-dependency-management.md']
- **gqp-005a** — `go over everything in this branch before I ship`
  - expected: ['references/tool-deep-review.md']
  - got: ['concepts/tool-release-prep.md', 'tasks/use-release-prep.md', 'concepts/task-git-workflow.md']
- **gqp-005b** — `audit all touched files before landing`
  - expected: ['references/tool-deep-review.md']
  - got: ['concepts/tool-bug-predict.md', 'quickstarts/run-security-audit.md', 'quickstarts/skill-security-audit.md']
- **gqp-006a** — `where are the landmines in this commit`
  - expected: ['concepts/tool-bug-predict.md', 'references/tool-bug-predict.md']
  - got: ['troubleshooting/pre-commit-infinite-loop.md', 'tasks/task-git-workflow.md', 'references/task-git-workflow.md']
- **gqp-007a** — `untangle this module`
  - expected: ['concepts/tool-refactor-plan.md', 'references/tool-refactor-plan.md']
  - got: ['concepts/tool-doc-gen.md', 'references/skill-doc-gen.md', 'references/tool-test-generation.md']
- **gqp-007b** — `rework the tangled bits in my project`
  - expected: ['concepts/tool-refactor-plan.md', 'references/tool-refactor-plan.md']
  - got: ['concepts/tool-memory-and-context.md', 'quickstarts/task-configuration-setup.md', 'tasks/task-ci-cd-pipeline.md']
- **gqp-008b** — `what's the gate before I push v0.5 out`
  - expected: ['concepts/tool-release-prep.md', 'references/skill-release-prep.md']
  - got: ['concepts/tool-spec.md', 'tasks/use-attune-hub.md', 'concepts/socratic-discovery.md']
- **gqp-009a** — `find places where my readme lies about the code`
  - expected: ['references/tool-doc-audit.md']
  - got: ['quickstarts/run-code-review.md', 'quickstarts/skill-code-quality.md', 'concepts/tool-bug-predict.md']
- **gqp-009b** — `what readme bits are wrong about how the app works now`
  - expected: ['references/tool-doc-audit.md']
  - got: ['concepts/tool-attune-hub.md', 'concepts/tool-doc-gen.md', 'concepts/task-precursor-warning-design.md']
- **gqp-011a** — `look for ways my service could be compromised`
  - expected: ['concepts/tool-security-audit.md', 'quickstarts/run-security-audit.md']
  - got: ['concepts/tool-coach.md']
- **gqp-011b** — `check my project for exploit surface`
  - expected: ['concepts/tool-security-audit.md', 'quickstarts/run-security-audit.md']
  - got: ['references/tool-health-check.md', 'quickstarts/check-health.md', 'references/tool-dependency-check.md']
- **gqp-013a** — `wire up all my readme-related jobs as one process`
  - expected: ['references/tool-doc-orchestrator.md']
  - got: ['quickstarts/skill-fix-test.md', 'quickstarts/skill-workflow-orchestration.md', 'tasks/task-configuration-setup.md']
- **gqp-013b** — `run readme tasks back-to-back across the repo`
  - expected: ['references/tool-doc-orchestrator.md']
  - got: ['quickstarts/task-code-migration.md', 'tasks/run-doctor.md', 'tasks/run-workflow.md']
- **gqp-014a** — `what's potentially harmful in my source`
  - expected: ['concepts/tool-bug-predict.md', 'references/tool-bug-predict.md']
  - got: ['tasks/use-attune-hub.md', 'tasks/use-security-audit.md', 'concepts/audience-adaptation.md']
- **gqp-015a** — `where might my service fail silently`
  - expected: ['concepts/tool-bug-predict.md', 'references/tool-bug-predict.md']
  - got: ['troubleshooting/gpg-signing-fails.md', 'concepts/tool-fix-test.md', 'quickstarts/skill-fix-test.md']
- **gqp-016b** — `what are the weak points in my source`
  - expected: ['concepts/tool-bug-predict.md', 'quickstarts/skill-bug-predict.md']
  - got: ['concepts/tool-attune-hub.md', 'tasks/use-attune-hub.md', 'concepts/audience-adaptation.md']
- **gqp-017a** — `spin up an explainer for my module`
  - expected: ['concepts/tool-doc-gen.md', 'quickstarts/skill-doc-gen.md']
  - got: ['concepts/tool-coach.md', 'quickstarts/task-configuration-setup.md', 'tasks/task-configuration-setup.md']
- **gqp-018a** — `give my module a once-over`
  - expected: ['concepts/tool-code-quality.md', 'quickstarts/skill-code-quality.md']
  - got: ['concepts/tool-doc-gen.md', 'references/skill-doc-gen.md', 'references/tool-test-generation.md']
- **gqp-019a** — `what should I do before pushing v0.4 to users`
  - expected: ['concepts/tool-release-prep.md', 'references/skill-release-prep.md']
  - got: ['concepts/socratic-discovery.md', 'tasks/use-planning.md', 'concepts/feedback-loop.md']
- **gqp-019b** — `push my code out to the world`
  - expected: ['concepts/tool-release-prep.md', 'references/skill-release-prep.md']
  - got: ['concepts/task-code-migration.md', 'concepts/tool-code-quality.md', 'quickstarts/run-code-review.md']
- **gqp-020a** — `build safety nets for these functions`
  - expected: ['concepts/tool-smart-test.md', 'quickstarts/generate-tests.md']
  - got: ['concepts/tool-refactor-plan.md', 'concepts/task-ci-cd-pipeline.md', 'concepts/task-package-publishing.md']
- **gqp-021a** — `the codebase has accumulated junk over time`
  - expected: ['concepts/tool-refactor-plan.md', 'references/tool-refactor-plan.md']
  - got: ['concepts/feedback-loop.md', 'concepts/task-code-migration.md', 'concepts/tool-security-audit.md']
- **gqp-021b** — `what should I prune from this codebase`
  - expected: ['concepts/tool-refactor-plan.md', 'references/tool-refactor-plan.md']
  - got: ['concepts/tool-security-audit.md', 'concepts/task-code-migration.md', 'concepts/tool-doc-gen.md']
- **gqp-022a** — `shore up untouched parts of my module`
  - expected: ['concepts/tool-smart-test.md', 'quickstarts/generate-tests.md']
  - got: ['quickstarts/task-configuration-setup.md', 'tasks/task-configuration-setup.md', 'concepts/tool-coach.md']
- **gqp-022b** — `what functions need assertions`
  - expected: ['concepts/tool-smart-test.md', 'quickstarts/generate-tests.md']
  - got: ['concepts/tool-attune-hub.md', 'tasks/use-attune-hub.md', 'concepts/tool-doc-gen.md']
- **gqp-023a** — `look for ways my code could be exploited`
  - expected: ['concepts/tool-security-audit.md', 'quickstarts/run-security-audit.md']
  - got: ['concepts/tool-code-quality.md', 'concepts/task-code-migration.md', 'quickstarts/run-code-review.md']
- **gqp-023b** — `look for leaked tokens and risky function calls`
  - expected: ['concepts/tool-security-audit.md', 'quickstarts/run-security-audit.md']
  - got: ['concepts/tool-bug-predict.md', 'quickstarts/task-error-handling-design.md', 'concepts/task-authentication-patterns.md']
- **gqp-024a** — `my suite is red — figure out why`
  - expected: ['concepts/tool-fix-test.md', 'references/skill-fix-test.md']
  - got: ['concepts/sync-paradigm.md', 'concepts/workflow-chain-prediction.md', 'concepts/cross-linking.md']
- **gqp-024b** — `auto-repair pytest errors`
  - expected: ['concepts/tool-fix-test.md', 'references/skill-fix-test.md']
  - got: ['quickstarts/skill-fix-test.md', 'concepts/tool-smart-test.md', 'quickstarts/generate-tests.md']
- **gqp-025a** — `what guidance is no longer accurate`
  - expected: ['references/tool-doc-audit.md']
  - got: ['concepts/tool-attune-hub.md', 'concepts/tool-release-prep.md', 'concepts/task-authentication-patterns.md']
- **gqp-026a** — `what do I need before pushing this out`
  - expected: ['concepts/tool-release-prep.md', 'references/tool-release-prep.md']
  - got: ['concepts/tool-attune-hub.md', 'tasks/use-attune-hub.md', 'concepts/socratic-discovery.md']
- **gqp-027a** — `what part of this PR is dangerous`
  - expected: ['concepts/tool-bug-predict.md', 'references/tool-bug-predict.md']
  - got: ['concepts/tool-attune-hub.md', 'tasks/use-attune-hub.md', 'tasks/use-bug-predict.md']
- **gqp-027b** — `highlight worrisome spots in the diff`
  - expected: ['concepts/tool-bug-predict.md', 'references/tool-bug-predict.md']
  - got: []
- **gqp-028a** — `I need a design pass before I write code`
  - expected: ['concepts/tool-planning.md', 'quickstarts/skill-planning.md']
  - got: ['concepts/tool-bug-predict.md', 'quickstarts/skill-code-quality.md', 'concepts/tool-code-quality.md']
- **gqp-029a** — `raise the bar on this codebase`
  - expected: ['concepts/tool-code-quality.md', 'references/skill-code-quality.md']
  - got: ['concepts/task-code-migration.md', 'concepts/tool-security-audit.md', 'quickstarts/run-security-audit.md']
- **gqp-029b** — `what's keeping this module from being clean`
  - expected: ['concepts/tool-code-quality.md', 'references/skill-code-quality.md']
  - got: ['concepts/tool-doc-gen.md', 'concepts/tool-spec.md', 'quickstarts/skill-doc-gen.md']
- **gqp-030a** — `simplify this gnarly module`
  - expected: ['concepts/tool-refactor-plan.md', 'references/skill-refactor-plan.md']
  - got: ['concepts/tool-doc-gen.md', 'references/tool-simplify-code.md', 'tips/after-simplify-code.md']
- **gqp-030b** — `this module has too many branches and nesting`
  - expected: ['concepts/tool-refactor-plan.md', 'references/skill-refactor-plan.md']
  - got: ['concepts/task-git-workflow.md', 'concepts/tool-doc-gen.md', 'quickstarts/task-git-workflow.md']
- **gqp-031b** — `what's blocking my merges`
  - expected: ['concepts/tool-fix-test.md', 'quickstarts/skill-fix-test.md']
  - got: ['tasks/use-attune-hub.md', 'concepts/tool-attune-hub.md', 'quickstarts/skill-planning.md']
- **gqp-032a** — `look for exploit surface in my repo`
  - expected: ['concepts/tool-security-audit.md', 'quickstarts/run-security-audit.md']
  - got: ['concepts/tool-coach.md', 'notes/decision-d1-template-schemas-live-in-the-repo.md', 'quickstarts/skill-bug-predict.md']
- **gqp-032b** — `what's potentially attackable here`
  - expected: ['concepts/tool-security-audit.md', 'quickstarts/run-security-audit.md']
  - got: ['tasks/use-attune-hub.md', 'concepts/tool-attune-hub.md', 'concepts/progressive-depth.md']
- **gqp-034a** — `wire up readme tasks as one flow`
  - expected: ['references/tool-doc-orchestrator.md']
  - got: ['concepts/task-debugging-sessions.md', 'quickstarts/task-configuration-setup.md', 'quickstarts/task-debugging-sessions.md']
- **gqp-034b** — `run all my reference-material jobs in sequence`
  - expected: ['references/tool-doc-orchestrator.md']
  - got: ['quickstarts/run-code-review.md', 'concepts/tool-workflow-orchestration.md', 'quickstarts/skill-fix-test.md']
- **gqp-035a** — `push this package out to users`
  - expected: ['concepts/tool-release-prep.md', 'references/tool-release-prep.md']
  - got: ['quickstarts/task-package-publishing.md', 'concepts/task-package-publishing.md', 'tasks/task-package-publishing.md']
- **gqp-036a** — `where's the diff going to bite me`
  - expected: ['concepts/tool-bug-predict.md', 'references/tool-bug-predict.md']
  - got: ['tasks/use-bug-predict.md', 'references/tool-analyze-image.md']
- **gqp-036b** — `what part of this commit is shaky`
  - expected: ['concepts/tool-bug-predict.md', 'references/tool-bug-predict.md']
  - got: ['troubleshooting/pre-commit-infinite-loop.md', 'concepts/task-git-workflow.md', 'concepts/tool-attune-hub.md']
- **gqp-037b** — `look at every changed file before landing`
  - expected: ['references/tool-deep-review.md']
  - got: ['concepts/tool-bug-predict.md', 'concepts/socratic-discovery.md', 'concepts/task-database-migrations.md']
- **gqp-038a** — `wire up readme jobs across all modules`
  - expected: ['references/tool-doc-orchestrator.md']
  - got: ['concepts/tool-memory-and-context.md', 'quickstarts/task-ci-cd-pipeline.md', 'tasks/task-configuration-setup.md']
- **gqp-038b** — `run my reference-material tasks in sequence repo-wide`
  - expected: ['references/tool-doc-orchestrator.md']
  - got: ['tasks/use-coach.md', 'tasks/run-doctor.md', 'tasks/run-workflow.md']
- **gqp-039a** — `what should I tackle in the next two weeks`
  - expected: ['concepts/tool-planning.md', 'references/skill-planning.md']
  - got: ['concepts/tool-attune-hub.md', 'concepts/task-precursor-warning-design.md', 'concepts/tool-memory-and-context.md']
- **gqp-039b** — `break my upcoming work into pieces`
  - expected: ['concepts/tool-planning.md', 'references/skill-planning.md']
  - got: ['concepts/tool-workflow-orchestration.md', 'quickstarts/skill-planning.md', 'concepts/feedback-loop.md']
- **gqp-040b** — `what could go wrong once this is live`
  - expected: ['concepts/tool-bug-predict.md', 'references/tool-bug-predict.md']
  - got: ['concepts/tool-release-prep.md', 'tasks/use-release-prep.md', 'concepts/tool-attune-hub.md']

