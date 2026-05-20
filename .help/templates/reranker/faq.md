---
type: faq
name: reranker-faq
feature: reranker
depth: faq
generated_at: 2026-05-20T03:36:00.787833+00:00
source_hash: e8daea3d6507f630a80c7cd27c8ac195aab73bcc409c0b5cadc15e215ce9e11d
status: generated
---

# Reranker FAQ

## What does the reranker do?

`LLMReranker` uses Claude Haiku as a relevance judge to re-order the candidates returned by keyword retrieval, so the most relevant results appear first. If the Claude API call fails for any reason, the reranker returns the original keyword-retrieval order unchanged.

## When should I use it?

Use `LLMReranker` when keyword retrieval alone isn't surfacing the right documents at the top of your results — for example, when queries are ambiguous or when you need higher precision for workflow-goal queries. If retrieval quality is already acceptable without it, you don't need it; it's opt-in.

## What class do I instantiate?

`LLMReranker` in `src/attune_rag/reranker.py`. Construct it with your API key and, optionally, a `candidate_multiplier` to control how many keyword candidates are passed to Claude for re-ranking:

```python
reranker = LLMReranker(api_key="YOUR_KEY", candidate_multiplier=3)
reranked_hits = reranker.rerank(query, hits)
```

`rerank()` accepts a query string and a list of `RetrievalHit` objects, and returns a re-ordered list of the same hits.

## What does `candidate_multiplier` control?

It multiplies the number of keyword-retrieval candidates passed to Claude before re-ranking. The default is `3`. A higher value gives Claude more to work with but increases latency and token usage.

## What model does it use?

Claude Haiku (`claude-haiku-4-5-20251001`) by default. You can pass a different model name to the `model` parameter of `LLMReranker.__init__()`, but the prompting logic is tuned for Haiku.

## What ranking rules does Claude apply?

The system prompt instructs Claude to apply these priorities, among others:

- For workflow-goal queries, `tool-*` docs rank above `skill-*`, `task-*`, and `use-*` docs.
- Queries about version bumps, changelogs, releases, or publishing prefer `tool-release-prep.md`.
- Queries about failing CI or tests prefer `tool-fix-test.md` or `skill-fix-test.md` over `task-ci-cd-pipeline.md`.
- Queries about orchestrating or coordinating documentation prefer `tool-doc-orchestrator.md`.

Claude returns a JSON array of 0-based indices (every index exactly once), and `LLMReranker` uses that order to sort the hits.

## What happens if the Claude API call times out?

The default timeout is 60 seconds. If the request exceeds that or any other API error occurs, `rerank()` returns the original hits in their keyword-retrieval order — no exception is raised to your caller.

## How do I debug it?

Run the related tests first: `pytest -k "reranker" -v`. If they pass but your code still fails, add a `logger.debug` statement before and after the `reranker.rerank()` call to inspect the input hits and returned order. Check that your API key is valid and that Claude Haiku is accessible from your environment.

## Where is the source?

`src/attune_rag/reranker.py`

**Tags:** `reranker`, `hybrid-retrieval`, `precision`, `claude`, `haiku`
