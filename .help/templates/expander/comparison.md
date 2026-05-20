---
type: comparison
name: expander-comparison
feature: expander
depth: comparison
generated_at: 2026-05-20T03:34:53.283760+00:00
source_hash: 8d3d8b3b23e7c8dfce64df28069feb915c0f099b70568ec577cb0a58dddf9b78
status: generated
---

# QueryExpander vs. raw query retrieval

## Context

`QueryExpander` uses Claude Haiku to rewrite a single query into 3–5 alternative phrasings before retrieval. It targets the core recall problem in documentation search: a user's wording rarely matches the exact terms in the indexed documents. By exposing synonyms, feature names, tool categories, and developer jargon, the expander increases the chance that at least one phrasing overlaps with the target content.

Expansion is opt-in and fail-safe. If the Claude API call fails, retrieval continues with the original query unchanged.

## Feature comparison

| Capability | Raw query retrieval | QueryExpander |
|---|---|---|
| Query variants sent to retrieval | 1 | 3–5 |
| Handles vocabulary mismatch | No | Yes — generates synonyms and developer jargon |
| LLM dependency | None | Claude Haiku (`claude-haiku-4-5-20251001`) |
| Latency added | None | One Haiku API call per query |
| Response caching | N/A | Yes (`cache=True` by default) |
| Async support | Depends on retriever | Yes — `expand_async()` |
| API failure behavior | N/A | Falls back to original query automatically |
| Output format | — | JSON array of strings |

## Tradeoffs

**Where `QueryExpander` wins**

- **Recall on low-overlap queries.** When a user asks "how do I hook into the build pipeline" and your docs say "CI integration", raw keyword retrieval misses entirely. The expander surfaces phrasings like "CI integration", "build hook", and "pipeline plugin" in a single call.
- **No schema changes required.** Expansion happens before retrieval; your index and retrieval layer stay unchanged.
- **Cached by default.** Repeated identical queries hit the cache rather than the API, so the latency cost is paid only once per unique query.

**Where raw retrieval wins**

- **Latency-sensitive paths.** Every non-cached query requires a round trip to the Claude API. If your retrieval SLA is tight and your query vocabulary is consistent with your document vocabulary, the overhead is not justified.
- **Deterministic output.** LLM-generated phrasings vary. If you need reproducible retrieval results (for testing or auditing), raw retrieval is more predictable.
- **Offline or air-gapped environments.** `QueryExpander` requires an active Anthropic API connection. Raw retrieval has no external dependency.

## Use `QueryExpander` when…

- Your users phrase queries differently from how your documentation is written — common in developer tools where users say "token limit" but docs say "context window".
- You are seeing low retrieval recall on integration, workflow, or conceptual queries.
- You can tolerate one additional API call per unique query (or you expect cache hits to absorb most of the cost).
- You want async retrieval pipelines — `expand_async()` makes this straightforward.

## Stick with raw query retrieval when…

- Your query and document vocabularies are already well-aligned and recall is acceptable.
- You are running in an environment without Anthropic API access.
- You need fully deterministic retrieval results.
- You are writing a quick exploratory script — wiring up `QueryExpander` adds an API key dependency that is not worth it for a one-off.

## Source files

- `src/attune_rag/expander.py`

**Tags:** `expander`, `hybrid-retrieval`, `recall`, `claude`, `haiku`
