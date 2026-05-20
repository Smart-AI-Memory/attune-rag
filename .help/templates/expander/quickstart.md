---
type: quickstart
name: expander-quickstart
feature: expander
depth: quickstart
generated_at: 2026-05-20T03:34:53.277287+00:00
source_hash: 8d3d8b3b23e7c8dfce64df28069feb915c0f099b70568ec577cb0a58dddf9b78
status: generated
---

# Quickstart: expander

`QueryExpander` uses Claude Haiku to rewrite a query into 3–5 alternative phrasings, improving retrieval recall when the user's wording doesn't closely match your documentation.

```python
from attune_rag.expander import QueryExpander

expander = QueryExpander(api_key="YOUR_API_KEY")
results = expander.expand("how do I set up authentication")
print(results)
```

Expected output:

```json
[
  "how do I set up authentication",
  "configure auth flow",
  "user login setup guide",
  "implement authentication tokens",
  "enable user sign-in"
]
```

## Prerequisites

- The package is installed locally from the cloned repository
- You have an Anthropic API key with access to Claude Haiku

## Steps

1. **Import and instantiate `QueryExpander`.** Pass your Anthropic API key. Response caching is on by default, so repeated identical queries won't consume extra API calls.

   ```python
   from attune_rag.expander import QueryExpander

   expander = QueryExpander(api_key="YOUR_API_KEY")
   ```

2. **Expand a query.** Call `expand()` with your query string. The expander returns a JSON array of alternative phrasings that expose feature names, tool categories, workflow synonyms, and developer jargon related to the original query.

   ```python
   expansions = expander.expand("how do I configure rate limiting")
   ```

3. **Use the expansions in your retrieval pipeline.** Pass each string in the returned list as a separate retrieval query to broaden recall. If the API call fails, your original query is preserved — the expander never drops it.

   ```python
   for query in expansions:
       results = your_retriever.search(query)
   ```

## Source files

- `src/attune_rag/expander.py`

Next: To run expansion without blocking your event loop, replace `expand()` with `expand_async()` and await the result inside an async function.

**Tags:** `expander`, `hybrid-retrieval`, `recall`, `claude`, `haiku`
