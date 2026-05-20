---
type: task
name: expander-task
feature: expander
depth: task
generated_at: 2026-05-20T02:45:15.476035+00:00
source_hash: 8d3d8b3b23e7c8dfce64df28069feb915c0f099b70568ec577cb0a58dddf9b78
status: generated
---

# Work with the QueryExpander

Use `QueryExpander` when you want to improve retrieval recall by expanding a user's query into 3–5 alternative phrasings before keyword search — particularly for queries with low surface-level overlap against your target documents.

## Prerequisites

- Access to `src/attune_rag/expander.py`
- A Claude Haiku API key (set as `ANTHROPIC_API_KEY` or passed directly to the constructor)

## Steps

1. **Instantiate `QueryExpander`.**
   Import and create a `QueryExpander` instance. Pass your API key explicitly or let the class read it from the environment. Enable caching (the default) to avoid redundant API calls for repeated queries:

   ```python
   from attune_rag.expander import QueryExpander

   expander = QueryExpander(
       model="claude-haiku-4-5-20251001",
       api_key="YOUR_API_KEY",  # or omit to use the environment variable
       cache=True,
   )
   ```

2. **Expand a query.**
   Call `expand()` with the user's raw query string. The method returns a list of alternative phrasings as strings. If the API call fails, it falls back to the original query so retrieval is never blocked:

   ```python
   variants = expander.expand("how do I configure logging?")
   # Example result:
   # ["how do I configure logging?",
   #  "set up log output",
   #  "logging configuration options",
   #  "enable debug logs",
   #  "adjust log level settings"]
   ```

   Use `expand_async()` instead if your retrieval pipeline is async.

3. **Feed the expanded variants into your retrieval step.**
   Pass the returned list to your keyword or vector search function. Each variant is a standalone query string ready for retrieval.

4. **Extend `QueryExpander` for custom behavior.**
   If you need a different model, prompt, or output format, subclass `QueryExpander` rather than editing the base class directly. Override `expand()` or `expand_async()` and match the existing return type (`list[str]`).

5. **Run the tests.**
   Confirm your changes work without regressions:

   ```bash
   pytest -k "expander"
   ```

## Verify success

Your setup is working correctly when:

- `expand()` returns a list of 3–5 strings for a well-formed query.
- Passing an invalid API key causes `expand()` to return a list containing only the original query string, with no exception raised.
- `pytest -k "expander"` passes with no failures.

## Key files

- `src/attune_rag/expander.py` — contains `QueryExpander`, the `expand()` and `expand_async()` methods, and the system prompt used to instruct Claude Haiku.
