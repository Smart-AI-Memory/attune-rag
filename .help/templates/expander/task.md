---
type: task
name: expander-task
feature: expander
depth: task
generated_at: 2026-05-20T03:34:53.257151+00:00
source_hash: 8d3d8b3b23e7c8dfce64df28069feb915c0f099b70568ec577cb0a58dddf9b78
status: generated
---

# Expand queries with QueryExpander

Use `QueryExpander` when you want to improve retrieval recall by automatically broadening a user's query into 3–5 alternative phrasings before searching your documentation.

## Prerequisites

- A valid Anthropic API key (or set the `ANTHROPIC_API_KEY` environment variable)
- Access to `src/attune_rag/expander.py`

## Expand a query

1. **Instantiate `QueryExpander`.**
   Create an instance with your desired configuration. All parameters are optional:

   ```python
   from attune_rag.expander import QueryExpander

   expander = QueryExpander(
       model="claude-haiku-4-5-20251001",  # default
       api_key="YOUR_API_KEY",             # or set via environment variable
       cache=True,                         # cache results for repeated queries
   )
   ```

2. **Call `expand()` with your query string.**
   Pass the user's raw query. The method returns a list of alternative phrasings as strings:

   ```python
   results = expander.expand("how do I configure logging")
   ```

   If your application is async, use `expand_async()` instead:

   ```python
   results = await expander.expand_async("how do I configure logging")
   ```

3. **Pass the expanded queries to your retrieval pipeline.**
   Use the returned list to query your documentation index. Each string is a standalone alternative phrasing of the original query, surfacing feature names, tool categories, workflow synonyms, and developer jargon:

   ```python
   # results might look like:
   # ["set up logging", "configure log output", "logging configuration options", ...]

   all_hits = []
   for phrase in results:
       all_hits.extend(index.search(phrase))
   ```

4. **Verify the output.**
   Confirm the task succeeded: `expand()` returns a Python `list` of 3–5 strings. If the API call fails for any reason, `QueryExpander` falls back to returning the original query, so your retrieval pipeline continues without interruption.

5. **Run the related tests.**
   Confirm nothing is broken after any changes:

   ```bash
   pytest -k "expander"
   ```

## Key files

- `src/attune_rag/expander.py` — contains `QueryExpander`, the only public class in this module
