---
type: note
name: reranker-note
feature: reranker
depth: note
generated_at: 2026-05-20T03:36:00.794729+00:00
source_hash: e8daea3d6507f630a80c7cd27c8ac195aab73bcc409c0b5cadc15e215ce9e11d
status: generated
---

# Note: reranker

## Context

The `reranker` module (`src/attune_rag/reranker.py`) adds a second-pass relevance ranking step on top of keyword retrieval. After an initial keyword search returns candidates, `LLMReranker` sends those candidates to Claude Haiku, which acts as a relevance judge and returns them sorted from most to least relevant.

The reranker is opt-in. If the Claude API call fails for any reason, the module falls back silently to the original keyword-only order, so retrieval continues uninterrupted.

## How it works

`LLMReranker.__init__` accepts four parameters:

| Parameter | Default | Purpose |
|---|---|---|
| `model` | `claude-haiku-4-5-20251001` | The Claude model used for ranking |
| `api_key` | `None` | API key; falls back to environment if `None` |
| `candidate_multiplier` | `3` | Multiplier applied to the requested top-K to generate the candidate pool passed to Claude |
| `timeout` | `60.0` | Seconds before the API call times out |

`LLMReranker.rerank(query, hits)` takes the query string and the list of `RetrievalHit` objects from keyword retrieval, calls Claude Haiku with a structured system prompt, and returns the hits reordered by relevance.

## Ranking heuristics baked into the system prompt

The system prompt (`_SYSTEM`) instructs Claude to apply several document-type heuristics specific to the attune-ai workflow documentation structure:

- **`tool-*` paths** (e.g., `concepts/tool-release-prep.md`) are canonical workflow references and rank above `skill-*`, `task-*`, and `use-*` paths for workflow-goal queries.
- **Release and publishing queries** (`version bump`, `changelog`, `ship`, `publish to PyPI`) prefer `tool-release-prep.md` over task guides such as `task-package-publishing.md`.
- **CI/test failure queries** prefer `tool-fix-test.md` or `skill-fix-test.md` over `task-ci-cd-pipeline.md`, which is a setup guide rather than a fix tool.
- **Orchestration queries** (`orchestrate`, `coordinate`, `manage`) prefer `tool-doc-orchestrator.md`.

Claude returns a JSON array of zero-based indices covering every candidate exactly once. The module uses that ordering to reorder the `RetrievalHit` list before returning it.

## Source files

- `src/attune_rag/reranker.py`

**Tags:** `reranker`, `hybrid-retrieval`, `precision`, `claude`, `haiku`
