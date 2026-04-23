---
type: task
feature: prompts
depth: task
generated_at: 2026-04-23T03:34:39.272580+00:00
source_hash: 3c35e8e0b30791a15e2c36afa2edf13ea407da219b38803d3325fc50b9980c18
status: generated
---

# Work with prompts

Use the prompts module when you need to build LLM prompts with retrieved context or customize how search results are formatted for prompt injection.

## Prerequisites

- Access to the project source code
- Understanding of retrieval-augmented generation (RAG) patterns
- Familiarity with `src/attune_rag/prompts.py`

## Build an augmented prompt

1. **Import the prompt builder:**
   ```python
   from attune_rag.prompts import build_augmented_prompt
   ```

2. **Prepare your query and context:**
   Your query is the user's question. Your context comes from search results, typically formatted using `join_context()` or `join_context_numbered()`.

3. **Choose a prompt variant:**
   Use `'baseline'` (default), or check the function documentation for other available variants.

4. **Build the prompt:**
   ```python
   prompt = build_augmented_prompt(
       query="How do I configure authentication?",
       context="<passage>Authentication requires...</passage>",
       variant='baseline'
   )
   ```

## Format search results as context

1. **For simple concatenation, use `join_context()`:**
   ```python
   from attune_rag.prompts import join_context

   context = join_context(
       hits=search_results,
       corpus=your_corpus,
       max_chars=4000
   )
   ```

2. **For numbered passages, use `join_context_numbered()`:**
   ```python
   from attune_rag.prompts import join_context_numbered

   context = join_context_numbered(
       hits=search_results,
       corpus=your_corpus,
       max_chars=4000
   )
   ```
   This creates `[P1]`, `[P2]` prefixed passages that you can reference in responses.

## Extend prompt variants

1. **Locate the variant logic in `build_augmented_prompt()`:**
   Open `src/attune_rag/prompts.py` and find where variants are handled.

2. **Add your variant:**
   Follow the existing pattern for variant selection and template formatting.

3. **Test your variant:**
   Run `pytest -k "prompts"` to verify your changes don't break existing functionality.

## Verify your changes

Your prompt integration works correctly when:
- `build_augmented_prompt()` returns a well-formed prompt string
- Search context appears wrapped in `<passage>` tags
- The LLM receives injection-resistant prompts (check for the injection defense clause)
- Tests pass with `pytest -k "prompts"`
