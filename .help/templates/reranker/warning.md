---
type: warning
name: reranker-warning
feature: reranker
depth: warning
generated_at: 2026-05-20T03:36:00.783293+00:00
source_hash: e8daea3d6507f630a80c7cd27c8ac195aab73bcc409c0b5cadc15e215ce9e11d
status: generated
---

# Reranker cautions

## What to watch for

`LLMReranker` calls the Claude Haiku API during every `rerank()` invocation to score and reorder keyword-retrieved candidates. If the API call fails for any reason, the reranker silently returns the original keyword-ranked order rather than raising an exception. This fail-safe behavior means degraded retrieval precision can go undetected if you are not actively monitoring for it.

## Risk areas

**Silent fallback masks API and quota failures.** When `LLMReranker.rerank()` encounters an API error or timeout, it returns the unmodified `hits` list without signaling a problem. If your Claude API key is invalid, rate-limited, or the request exceeds the 60-second `timeout`, your application continues running on keyword-only results with no visible indication. Log or instrument the return path if retrieval quality matters to your use case.

**`candidate_multiplier` inflates token usage and latency.** `LLMReranker` passes `candidate_multiplier × k` candidates to Claude for scoring. The default multiplier of `3` triples the number of document summaries sent in each prompt. For long document summaries or high-traffic workloads, this can push requests toward context-length limits and significantly increase per-query cost and latency. Set `candidate_multiplier` deliberately rather than leaving it at the default.

**Path-pattern ranking rules are baked into the system prompt.** The relevance-judge prompt in `_SYSTEM` encodes fixed heuristics that rank `tool-*` paths above `skill-*`, `task-*`, and `use-*` paths for workflow-goal queries, and maps specific keywords (for example, "publish to PyPI") to particular documents. If your documentation hierarchy diverges from the `attune-ai` path conventions, these hardcoded rules will produce incorrect rankings. Audit `_SYSTEM` before deploying against a custom document corpus.

**The `_SYSTEM` constant is private and subject to change.** Because `_SYSTEM` begins with an underscore, it is not part of the public API. Referencing or patching it in your own code creates a fragile dependency that may break without notice across versions.

## How to avoid problems

1. **Monitor for silent fallbacks.** Wrap `rerank()` calls with logging that records whether the returned order differs from the input order. A result that exactly matches the input `hits` order is a signal that the API call may have failed.

2. **Tune `candidate_multiplier` for your workload.** Start with a lower value in cost-sensitive or latency-sensitive environments, and increase it only after benchmarking retrieval precision against the added expense.

3. **Validate path conventions before switching corpora.** If you use `LLMReranker` against documentation that does not follow the `tool-`, `skill-`, `task-`, `use-` path scheme, the embedded ranking heuristics in `_SYSTEM` will misfire. Consider whether the out-of-the-box prompt is appropriate for your content structure.

4. **Set `timeout` based on your SLA.** The default `timeout` of `60.0` seconds may be too permissive for interactive use cases. Pass a lower value to `LLMReranker.__init__()` to bound worst-case latency, accepting that more requests will fall back to keyword order under load.

## Source files

- `src/attune_rag/reranker.py`

**Tags:** `reranker`, `hybrid-retrieval`, `precision`, `claude`, `haiku`
