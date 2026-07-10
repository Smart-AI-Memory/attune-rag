---
type: concept
name: reranker-concept
feature: reranker
depth: concept
generated_at: 2026-07-10T13:06:04.019518+00:00
source_hash: c828b6c3ccd4f66d997d42c41fc386540dc0978c13f780db0e2920cdbb911f6d
status: generated
---

# Reranker

The reranker is a second-pass relevance filter that uses Claude Haiku to sort keyword-retrieval candidates by how well they actually answer a query — improving precision without replacing the underlying search.

## The problem it solves

Keyword retrieval is fast but imprecise: it returns documents that contain the right words, not necessarily documents that answer the right question. A query like "how do I fix failing tests?" may surface a CI/CD setup guide alongside the actual fix tool, because both documents share vocabulary.

`LLMReranker` addresses this by treating Claude Haiku as a relevance judge. After keyword retrieval produces a candidate list, the reranker asks the model to order those candidates from most to least relevant to the original query. The final ranked list is what the caller receives.

## How the pieces fit together

Retrieval in attune-rag works in two stages:

1. **Keyword retrieval** — runs first, quickly, and returns a broad candidate set. To give the reranker enough material to work with, it fetches `candidate_multiplier × N` results (default multiplier: 3) rather than exactly the N results the caller wants.
2. **LLM reranking** — `LLMReranker.rerank(query, hits)` sends the candidate paths and summaries to Claude Haiku with a structured system prompt. The model returns a JSON array of 0-based indices ordered by relevance. The reranker uses that index order to reorder the `RetrievalHit` list before returning it.

The system prompt encodes specific routing rules so the model can apply domain knowledge rather than generic relevance signals. For example:

- "version bump", "release", or "publish" → prefer `tool-release-prep.md` over `task-package-publishing.md`
- "CI pipeline failing" or "fix tests" → prefer `tool-fix-test.md` over `task-ci-cd-pipeline.md` (a setup guide, not a fix tool)
- `tool-*` paths rank above `skill-*`, `task-*`, and `use-*` paths for workflow-goal queries

## Failure behavior

The reranker is opt-in and fail-safe. If the Claude API call times out (default: 60 seconds) or returns an error, `LLMReranker` returns the original keyword-retrieval order unchanged. Callers that do not configure a reranker receive keyword results directly.

## When the reranker matters

The reranker has the most impact when:

- The user's query expresses a goal ("release my package", "fix failing tests") rather than a keyword ("PyPI", "pytest")
- Multiple documents share surface-level vocabulary but serve different purposes — for example, a setup guide and a fix tool that both mention "CI pipeline"
- Retrieval precision is more important than latency, since the LLM call adds a network round-trip
