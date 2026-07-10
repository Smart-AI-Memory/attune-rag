---
type: task
name: expander-task
feature: expander
depth: task
generated_at: 2026-07-10T13:05:48.726010+00:00
source_hash: 8645de9f31cc8aa82ed6ad99d147639060d87eaed712a4c14f1355d03
status: generated
---

# Work with QueryExpander

Use `QueryExpander` when you want to improve retrieval recall by expanding a narrow or jargon-heavy query into 3–5 alternative phrasings before it hits your keyword index.

## Prerequisites

- Read access to `src/attune_rag/expander.py`
- An Anthropic API key (or a pre-configured `api_key` argument)
- `pytest` installed if you intend to verify your changes with the test suite

## Steps

1. **Instantiate `QueryExpander`.**
   Create an instance with your preferred options. All parameters are optional:

   ```python
   from attune_rag.expander import QueryExpander

   expander = QueryExpander(
       model=None,    # defaults to Claude Haiku
       api_key=None,  # falls back to environment variable
       cache=True,    # cache repeated queries
   )
   ```

2. **Call `expand` or `expand_async` with your query.**
   Pass the raw user query as a string. The method returns a list of alternative phrasings as strings:

   ```python
   # Synchronous
   variants = expander.expand("how do I set up git hooks?")

   # Asynchronous
   variants = await expander.expand_async("how do I set up git hooks?")
   ```

   If the Claude API call fails for any reason, `QueryExpander` automatically falls back to returning the original query, so retrieval continues uninterrupted.

3. **Feed the expanded list into your retrieval pipeline.**
   Use the returned phrasings as additional search terms alongside — or instead of — the original query. Each string in the list is a standalone rephrasing suitable for keyword or vector search.

4. **Extend `QueryExpander` for custom behavior.**
   If you need to change the expansion logic (for example, to target a different model or alter the system prompt), subclass `QueryExpander` rather than editing the base class directly. Override only the methods you need to change, and keep your naming, error handling, and logging consistent with the existing class.

5. **Run the related tests.**
   Confirm nothing is broken:

   ```bash
   pytest -k "expander"
   ```

## Key files

- `src/attune_rag/expander.py` — defines `QueryExpander`, the system prompt (`_SYSTEM`), and the `expand` / `expand_async` methods

## Verify success

After calling `expand`, check that:

- The return value is a list of strings with between 3 and 5 entries
- Each entry is a distinct rephrasing of your original query (not a repetition)
- If you deliberately pass an invalid API key, the method still returns a list containing the original query string, confirming the fail-safe fallback is active
