---
type: note
name: expander-note
feature: expander
depth: note
generated_at: 2026-05-20T03:34:53.281427+00:00
source_hash: 8d3d8b3b23e7c8dfce64df28069feb915c0f099b70568ec577cb0a58dddf9b78
status: generated
---

# Note: expander

## Context

`QueryExpander` improves retrieval recall by rewriting a user query into 3–5 alternative phrasings before the retrieval step runs. This matters most when the query and the target documents share little surface-level vocabulary — for example, when a user asks about "setting up auth" but the documentation uses "configuring OAuth providers."

Expansion is opt-in and fail-safe: if the Claude API call fails, the pipeline falls back to the original query unchanged.

## How it works

`QueryExpander` (defined in `src/attune_rag/expander.py`) sends the query to Claude Haiku with a fixed system prompt that instructs the model to surface the user's actual intent — feature names, tool categories, workflow synonyms, and developer jargon — and return the results as a bare JSON array of strings.

The system prompt explicitly forbids prose or markdown fences in the response, so the returned value can be parsed directly without post-processing.

Both a synchronous path (`expand`) and an asynchronous path (`expand_async`) are available. Either method returns a `list[str]` of alternative phrasings.

Response caching is enabled by default (controlled by the `cache` parameter at construction time), which avoids redundant API calls for repeated queries.

## Default model

The default model is `claude-haiku-4-5-20251001`. You can override this by passing a different `model` string to `QueryExpander.__init__`.

## Source files

- `src/attune_rag/expander.py`

**Tags:** `expander`, `hybrid-retrieval`, `recall`, `claude`, `haiku`
