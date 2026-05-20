---
type: warning
name: expander-warning
feature: expander
depth: warning
generated_at: 2026-05-20T03:34:53.270721+00:00
source_hash: 8d3d8b3b23e7c8dfce64df28069feb915c0f099b70568ec577cb0a58dddf9b78
status: generated
---

# Expander cautions

## What to watch for

`QueryExpander` uses Claude Haiku to rewrite a query into 3–5 alternative phrasings before retrieval. This improves recall when a user's wording diverges from document vocabulary, but it introduces an external LLM call into your retrieval path. If the API call fails, `QueryExpander` silently falls back to the original query — retrieval still works, but without expansion. Understanding this behavior is important for debugging recall regressions in production.

## Risk areas

### Cached expansion results may mask model or API changes

When `cache=True` (the default), `QueryExpander` reuses previously generated expansions for identical query strings. If you rotate API keys, switch models, or update the system prompt, cached results from earlier calls will persist until the process restarts. This means a model change or prompt fix may appear to have no effect during a live session.

**Mitigation:** Instantiate `QueryExpander` with `cache=False` when testing prompt or model changes. In production, be aware that redeploying without a process restart preserves the in-memory cache.

---

### Silent fallback hides API failures during retrieval

When the Claude Haiku API call raises an exception, `QueryExpander.expand()` and `expand_async()` fall back to the original query without raising or logging the error to the caller. Retrieval continues, but at reduced recall — and nothing in the return value signals that expansion was skipped.

**Mitigation:** Monitor your retrieval pipeline's recall metrics over time. If you need explicit visibility into expansion failures, wrap `expand()` calls in a try/except or add instrumentation around the `QueryExpander` instance rather than relying on its return value to indicate success.

---

### The `_SYSTEM` prompt is private and may change without notice

The system prompt that shapes expansion behavior — instructing the model to return a JSON array of 3–5 alternative phrasings with no explanation — is defined in the private constant `_SYSTEM`. Code that reads or patches `_SYSTEM` directly is depending on an internal implementation detail that can change without a deprecation notice.

**Mitigation:** Use `QueryExpander`'s public interface only: `__init__`, `expand`, and `expand_async`. If you need custom expansion behavior, subclass `QueryExpander` and override `expand` rather than patching `_SYSTEM`.

---

### Malformed JSON from the model produces silent empty expansion

`QueryExpander` expects the model to return a raw JSON array of strings. The system prompt explicitly instructs the model to return only JSON with no markdown fences or explanation, but LLM output is non-deterministic. If the model returns malformed JSON, the expansion list may be empty or partial, and retrieval falls back to the original query.

**Mitigation:** If you observe unexpectedly low recall on specific queries in production, log the raw model response before parsing to confirm whether the model is returning well-formed JSON.

## Source files

- `src/attune_rag/expander.py`

**Tags:** `expander`, `hybrid-retrieval`, `recall`, `claude`, `haiku`
