# Diagnostic-3 — Alias expansion on bug-predict

corpus: attune-help  version: 0.11.0
queries: tests/golden/queries_paraphrased.yaml (80 total)
added aliases: 18 multi-token aliases on each of 3 bug-predict entries

## All-corpus view (sanity check — should not regress on non-bug-predict)

### Before (aliases as-is)

| difficulty | n | P@1 | R@3 |
|---|---|---|---|
| easy | 20 | 0.00% | 30.00% |
| medium | 54 | 14.81% | 29.63% |
| hard | 6 | 16.67% | 16.67% |
| **overall** | **80** | **11.25%** | **28.75%** |

### After (augmented aliases)

| difficulty | n | P@1 | R@3 |
|---|---|---|---|
| easy | 20 | 0.00% | 35.00% |
| medium | 54 | 25.93% | 40.74% |
| hard | 6 | 16.67% | 16.67% |
| **overall** | **80** | **18.75%** | **37.50%** |

Δ P@1 (overall): +7.50%
Δ R@3 (overall): +8.75%

## bug-predict focus

### Before (aliases as-is) — bug-predict only (14 queries)

- P@1: **14.29%** (2/14)
- R@3: **35.71%** (5/14)

R@3 misses (9):
  - **gqp-006a** — `where are the landmines in this commit` → ['troubleshooting/pre-commit-infinite-loop.md', 'tasks/task-git-workflow.md', 'references/task-git-workflow.md']
  - **gqp-014a** — `what's potentially harmful in my source` → ['tasks/use-attune-hub.md', 'tasks/use-security-audit.md', 'concepts/audience-adaptation.md']
  - **gqp-015a** — `where might my service fail silently` → ['troubleshooting/gpg-signing-fails.md', 'concepts/tool-fix-test.md', 'quickstarts/skill-fix-test.md']
  - **gqp-016b** — `what are the weak points in my source` → ['concepts/tool-attune-hub.md', 'tasks/use-attune-hub.md', 'concepts/audience-adaptation.md']
  - **gqp-027a** — `what part of this PR is dangerous` → ['concepts/tool-attune-hub.md', 'tasks/use-attune-hub.md', 'tasks/use-bug-predict.md']
  - **gqp-027b** — `highlight worrisome spots in the diff` → []
  - **gqp-036a** — `where's the diff going to bite me` → ['tasks/use-bug-predict.md', 'references/tool-analyze-image.md']
  - **gqp-036b** — `what part of this commit is shaky` → ['troubleshooting/pre-commit-infinite-loop.md', 'concepts/task-git-workflow.md', 'concepts/tool-attune-hub.md']
  - **gqp-040b** — `what could go wrong once this is live` → ['concepts/tool-release-prep.md', 'tasks/use-release-prep.md', 'concepts/tool-attune-hub.md']

### After (augmented aliases) — bug-predict only (14 queries)

- P@1: **57.14%** (8/14)
- R@3: **85.71%** (12/14)

R@3 misses (2):
  - **gqp-015a** — `where might my service fail silently` → ['troubleshooting/gpg-signing-fails.md', 'concepts/tool-fix-test.md', 'quickstarts/skill-fix-test.md']
  - **gqp-036a** — `where's the diff going to bite me` → ['tasks/use-bug-predict.md', 'references/tool-analyze-image.md']

Δ P@1 (bug-predict only): +42.86% (+6 queries)
Δ R@3 (bug-predict only): +50.00% (+7 queries)

## Verdict

**STRONG** — alias expansion closes most of the bug-predict paraphrase gap. If this generalizes across other feature clusters, the embedding-retriever spec defers in favor of an alias-authoring task.
