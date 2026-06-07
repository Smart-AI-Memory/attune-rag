---
type: task
name: expander-task
feature: expander
depth: task
generated_at: 2026-06-07T07:13:53.297236+00:00
source_hash: fee9ea3e96d976b16a96673b646ba25f945f41ad6136204efbd13aaa334ccf76
status: generated
---

# Expand queries with QueryExpander

Use `QueryExpander` when retrieval results are missing relevant documents because the user's query shares little surface-level wording with your indexed content — expanding the query into 3–5 alternative phrasings improves recall without changing your retrieval pipeline.

## Prerequisites

- An Anthropic API key, or the `ANTHROPIC_API_KEY` environment variable set in your shell
- `attune_rag` installed in your Python environment

## Expand a query

1. **Import `QueryExpander`** from `attune_rag.expander`:

   ```python
   from attune_rag.expander import QueryExpander
   ```

2. **Instantiate the expander.** The constructor accepts three optional parameters:

   - `model` — the Claude model to use (default: `'claude-haiku-4-5'`)
   - `api_key` — your Anthropic API key; omit this if the key is already set as an environment variable
   - `cache` — set to `False` to disable response caching (default: `True`)

   ```python
   expander = QueryExpander()
   ```

   To supply an API key explicitly or swap the model:

   ```python
   expander = QueryExpander(model='claude-haiku-4-5', api_key='sk-...', cache=True)
   ```

3. **Call `expand`** with the user's original query. The method returns a list of alternative phrasings:

   ```python
   variants = expander.expand("how do I set up pre-commit hooks")
   ```

   If you are working in an async context, call `expand_async` instead:

   ```python
   variants = await expander.expand_async("how do I set up pre-commit hooks")
   ```

4. **Pass the returned list to your retrieval pipeline.** Each string in `variants` is a standalone query you can use to broaden your keyword or vector search. If the API call fails, `QueryExpander` falls back to the original query automatically — no error handling is required on your end.

## Verify the expansion worked

Print `variants` and confirm it contains between 3 and 5 strings, each rephrasing the original intent:

```python
print(variants)
# ['how do I set up pre-commit hooks',
#  'configure git pre-commit automation',
#  'install pre-commit framework hooks',
#  'pre-commit hook setup workflow']
```

If the list contains exactly one entry that matches your original query, the API call fell back gracefully — check that your API key is valid and that the `claude-haiku-4-5` model is accessible on your account.
