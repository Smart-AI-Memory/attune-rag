---
type: tip
name: pipeline-tip
feature: pipeline
depth: tip
generated_at: 2026-05-20T03:20:40.540605+00:00
source_hash: f5cc845ee3957a76674328c9a162ce4a86e404c42291f721ed77a3b4c3b27569
status: verified
---

# Tip: Check `fallback_used` before trusting a `RagResult`

## Recommendation

After calling `RagPipeline.run()` or `run_and_generate()`, check `RagResult.fallback_used` before presenting the result. If `fallback_used` is true, retrieval returned zero hits. Check `abstained` to identify a strict local refusal. Other variants can generate an answer without sources.

Zero-hit results carry a confidence score of zero.

## Tradeoff

Reading `fallback_used` adds a conditional branch to your response handling. If you skip it, you lose the ability to distinguish a well-grounded answer from an honest "I don't know" — and those two cases often need different UI treatment or downstream logging.

## Example

```python
answer, result = await pipeline.run_and_generate(query, provider=provider, prompt_variant=variant)

if result.abstained:
    display_refusal(answer)
elif result.fallback_used:
    # No corpus match — surface a warning or lower-confidence indicator
    warn_user("Answer is not grounded in the corpus.")
else:
    # Safe to display citations from result.citation
    display(answer, result.citation)
```

## Source files

- `src/attune_rag/pipeline.py`
- `src/attune_rag/__init__.py`

**Tags:** `pipeline`, `orchestration`, `rag`, `result`
