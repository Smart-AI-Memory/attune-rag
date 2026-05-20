---
type: concept
name: reranker-concept
feature: reranker
depth: concept
generated_at: 2026-05-20T03:36:00.762647+00:00
source_hash: e8daea3d6507f630a80c7cd27c8ac195aab73bcc409c0b5cadc15e215ce9e11d
status: generated
---

# Reranker

The reranker is an optional second-pass stage that uses Claude Haiku as a relevance judge to reorder keyword-retrieval candidates before they reach the caller, improving precision without replacing the underlying search index.

## How reranking works

Keyword retrieval is fast but ranks by term frequency, not intent. The reranker addresses this by fetching more candidates than you ultimately need — controlled by `candidate_multiplier` (default `3`) — then asking Claude Haiku to sort them by relevance to the original query.

The flow looks like this:

1. Keyword retrieval returns `N × candidate_multiplier` hits (for example, if you want 5 results, retrieval fetches 15).
2. `LLMReranker.rerank(query, hits)` sends Claude Haiku a numbered list of candidate document paths and summaries.
3. Claude Haiku responds with a JSON array of 0-based indices ordered from most to least relevant. The system prompt encodes domain-specific ranking rules — for example, `tool-release-prep.md` is preferred for queries about versioning or publishing, while `tool-fix-test.md` is preferred over `task-ci-cd-pipeline.md` for failing-test queries.
4. `rerank` maps those indices back to the original `RetrievalHit` objects and returns the reordered list.

If the Claude API call fails for any reason, `rerank` falls back to the original keyword-ranked order, so retrieval continues to work without interruption.

## Domain-aware ranking rules

The system prompt baked into `LLMReranker` teaches Claude Haiku about the attune-ai documentation taxonomy:

| Path prefix | Document type |
|---|---|
| `tool-` | Canonical attune workflow references (highest priority for goal queries) |
| `skill-`, `task-`, `use-` | Quickstarts and task guides |

Several query patterns trigger explicit preferences — for instance, queries mentioning "version bump", "changelog", or "publish to PyPI" are steered toward `tool-release-prep.md` rather than the more generic `task-package-publishing.md`.

## Key parameters

| Parameter | Default | Effect |
|---|---|---|
| `model` | `claude-haiku-4-5-20251001` | The Claude model used as the relevance judge |
| `candidate_multiplier` | `3` | Multiplier applied to the desired result count to widen the candidate pool before reranking |
| `timeout` | `60.0` s | Maximum wait for the Claude API response before falling back to keyword order |
| `api_key` | `None` (uses environment) | Anthropic API key; omit to rely on the default credential chain |

## Relationship to retrieval

`LLMReranker` operates on `RetrievalHit` objects produced by keyword retrieval — it does not perform its own search. You pass it the raw hits and get back the same objects in a new order. This means you can adopt it incrementally: plug it in after your existing retrieval step, and remove it at any time without changing upstream or downstream code.
