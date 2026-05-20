---
type: faq
name: expander-faq
feature: expander
depth: faq
generated_at: 2026-05-20T03:34:53.275043+00:00
source_hash: 8d3d8b3b23e7c8dfce64df28069feb915c0f099b70568ec577cb0a58dddf9b78
status: generated
---

# Expander FAQ

## What does the expander feature do?

`QueryExpander` sends your query to Claude Haiku and returns 3–5 alternative phrasings as a list of strings. You pass those expanded queries to your retrieval pipeline to improve recall when the user's wording doesn't closely match the language in your documents.

## When should I use it?

Use `QueryExpander` when keyword or embedding search returns poor results because users phrase queries differently from how your documentation is written. It's particularly useful for developer-facing documentation where the same concept has multiple common names — for example, "how to mock a dependency" vs. "stub an interface" vs. "test double setup".

If your recall problems come from incomplete document coverage rather than query phrasing, query expansion won't help.

## How do I create a QueryExpander?

Instantiate `QueryExpander` directly:

```python
from attune_rag.expander import QueryExpander

expander = QueryExpander()  # uses claude-haiku-4-5-20251001 by default
```

You can pass a custom `model`, an `api_key`, or set `cache=False` to disable response caching.

## How do I expand a query?

Call `expand()` for synchronous use or `expand_async()` for async contexts. Both accept a query string and return a `list[str]`:

```python
alternatives = expander.expand("how do I set up authentication")
# e.g. ["configure auth", "authentication setup", "enable login flow", ...]
```

## What happens if the Claude API call fails?

The call fails with whatever exception the underlying API client raises — there is no silent fallback built into `QueryExpander` itself. Wrap `expand()` or `expand_async()` in a try/except and fall back to the original query string in your own retrieval logic if needed.

## What model does it use by default?

`claude-haiku-4-5-20251001`. You can override this by passing a different model name to the `model` parameter when constructing `QueryExpander`.

## How do I debug unexpected expansions?

Inspect the raw list returned by `expand()` — the model is prompted to return only a JSON array with no explanation or markdown. If the output looks malformed, check whether a custom `model` you've passed returns a different response format than the default prompt expects.

To run the relevant tests: `pytest -k "expander" -v`.

## Where is the source code?

`src/attune_rag/expander.py`

**Tags:** `expander`, `hybrid-retrieval`, `recall`, `claude`, `haiku`
