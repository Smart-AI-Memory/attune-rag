---
type: concept
name: reranker-concept
feature: reranker
depth: concept
generated_at: 2026-05-20T02:45:30.116900+00:00
source_hash: e8daea3d6507f630a80c7cd27c8ac195aab73bcc409c0b5cadc15e215ce9e11d
status: generated
---

# Reranker

The reranker is a second-pass relevance filter that uses Claude Haiku to reorder keyword-retrieval results before they reach the user, trading a small amount of latency for meaningfully higher retrieval precision.

## The problem it solves

Keyword retrieval is fast but shallow: it ranks documents by term frequency, not by whether a document actually answers the query. The reranker sits between the keyword retrieval step and the final result list. It takes the top-K candidates that keyword search returns, asks Claude Haiku to judge their relevance against the original query, and returns them in the order a reader would find most useful.

## How LLMReranker works

`LLMReranker` (in `src/attune_rag/reranker.py`) is the single class that implements this pipeline:

1. **Candidate expansion.** Before reranking, the retriever fetches `candidate_multiplier × N` documents (default multiplier: `3`). This gives Claude a wider pool to reorder rather than a tight list that keyword ranking has already over-constrained.

2. **Relevance judgment.** `rerank(query, hits)` passes the query and a numbered list of candidate paths and summaries to Claude Haiku. The model acts as a relevance judge: it returns a JSON array of 0-based indices ranked from most to least relevant — no explanation, just the ordering.

3. **Ranking guidance baked into the prompt.** The system prompt encodes attune-specific heuristics. For example:
   - `tool-*` paths (e.g. `concepts/tool-release-prep.md`) are canonical workflow references and rank above `skill-*`, `task-*`, and `use-*` paths for workflow-goal queries.
   - Queries mentioning "version bump", "changelog", or "publish" prefer `tool-release-prep.md`.
   - Queries about failing CI or tests prefer `tool-fix-test.md` or `skill-fix-test.md` over `task-ci-cd-pipeline.md`, which is a setup guide rather than a fix tool.

4. **Fail-safe fallback.** If the Claude API call times out or returns an error, `LLMReranker` returns the original keyword-ranked order unchanged. Reranking is opt-in: the rest of the retrieval pipeline keeps working with no reranking if `LLMReranker` is not wired in.

## Configuration

`LLMReranker.__init__` accepts four parameters:

| Parameter | Default | Effect |
|---|---|---|
| `model` | `claude-haiku-4-5-20251001` | The Claude model used as the relevance judge |
| `api_key` | `None` (uses environment) | Anthropic API key; omit to read from the environment |
| `candidate_multiplier` | `3` | How many extra candidates to retrieve before reranking |
| `timeout` | `60.0` | Seconds before the API call is abandoned and the fallback order is used |

## When reranking matters

Reranking has the most impact when keyword overlap between the query and the document is misleading — for example, when multiple documents mention the same terms but only one actually addresses the user's goal. It matters less for lookup queries where the top keyword result is almost always correct.
