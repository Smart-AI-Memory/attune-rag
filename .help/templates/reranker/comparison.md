---
type: comparison
name: reranker-comparison
feature: reranker
depth: comparison
generated_at: 2026-05-20T03:36:00.797070+00:00
source_hash: e8daea3d6507f630a80c7cd27c8ac195aab73bcc409c0b5cadc15e215ce9e11d
status: generated
---

# Reranker vs. keyword-only retrieval

## Context

`LLMReranker` adds a second retrieval stage: it takes the top-K candidates from keyword search and asks Claude Haiku to re-rank them by semantic relevance to the query. The final order reflects meaning, not just term frequency. If the Claude API call fails for any reason, the system returns the original keyword-ranked order unchanged, so enabling the reranker carries no availability risk.

## Feature comparison

| Capability | Keyword-only retrieval | `LLMReranker` |
|---|---|---|
| Ranking signal | Term overlap (BM25-style) | Semantic relevance judged by Claude Haiku |
| Latency | Near-zero (local index scan) | Adds one Claude API round-trip (default timeout: 60 s) |
| API dependency | None | Requires Anthropic API key |
| Handles vocabulary mismatch | No — misses synonyms and paraphrases | Yes — Claude understands intent, not just tokens |
| Domain-specific ranking rules | No | Yes — system prompt encodes attune-specific path heuristics (e.g. `tool-*` docs ranked above `skill-*` for workflow-goal queries) |
| Candidate pool | All index hits | `candidate_multiplier × k` keyword hits (default: 3×), then re-ranked to top k |
| Failure behavior | N/A | Falls back to keyword order on any API error |
| Cost | Free | One Claude Haiku inference call per query |

## When to use `LLMReranker`

Enable `LLMReranker` when any of the following apply:

- **Queries use natural language or goals, not keywords.** A query like "how do I ship a release?" may not share tokens with a document titled `tool-release-prep.md`, but Claude can match them correctly.
- **Precision matters more than latency.** If showing one wrong result at position 1 degrades the user experience more than a 1–2 second API round-trip hurts it, the reranker improves the outcome.
- **You work with attune workflow docs.** The reranker's system prompt encodes specific ranking rules for this corpus — for example, routing "publish to PyPI" queries to `tool-release-prep.md` rather than `task-package-publishing.md`, and preferring `tool-fix-test.md` over `task-ci-cd-pipeline.md` for "CI failing" queries. Keyword search has no equivalent logic.
- **You already have an Anthropic API key.** There is no additional infrastructure to provision.

## When to skip the reranker

- **Latency is the primary constraint.** Keyword-only retrieval is synchronous and local. The reranker adds a network round-trip that can reach the 60-second timeout under load.
- **Queries are already keyword-shaped.** If users consistently search with exact document identifiers or precise technical terms, keyword ranking is sufficient and the API cost is unnecessary.
- **You have no Anthropic API key.** The reranker requires one. Without it, the class instantiates but every `rerank()` call falls back to keyword order — you get keyword behavior with added overhead.
- **You are writing a throwaway script or spike.** Configuring an API key and a `candidate_multiplier` for a single exploratory run is more setup than the benefit warrants.

## Recommendation

**Use `LLMReranker` for any user-facing query path in the attune-ai workflow documentation system.** The system prompt is purpose-built for this corpus, and the keyword fallback means there is no downside risk. Keyword-only retrieval is the right default only for offline tooling, latency-sensitive batch jobs, or environments where outbound API calls are restricted.

## Source files

- `src/attune_rag/reranker.py`

**Tags:** `reranker`, `hybrid-retrieval`, `precision`, `claude`, `haiku`
