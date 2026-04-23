---
type: task
feature: eval
depth: task
generated_at: 2026-04-23T03:36:09.603864+00:00
source_hash: c51ee93d206242f5b5e8a7ad4ed5772aaf45bc2940d459444097d7a899d4513c
status: generated
---

# Work with eval

Use eval when you need to measure answer quality in your RAG system or compare prompt variants against a test dataset.

## Prerequisites

- Access to the project source code
- API key for Claude (for faithfulness scoring)
- Test dataset with queries and expected answers

## Steps

1. **Choose your evaluation approach.**

   For faithfulness scoring (checking if answers are grounded in retrieved passages):
   ```python
   from attune_rag.eval import FaithfulnessJudge

   judge = FaithfulnessJudge()
   result = judge.score(query, answer, passages)
   ```

   For prompt benchmarking against a golden dataset:
   ```bash
   python -m attune_rag.eval.bench_prompts
   ```

2. **Set up faithfulness evaluation.**

   Create a FaithfulnessJudge instance with your preferred model:
   ```python
   judge = FaithfulnessJudge(
       model="claude-sonnet-4-6",  # default
       timeout=30.0
   )
   ```

3. **Score individual answers.**

   Pass your query, generated answer, and retrieved context:
   ```python
   result = judge.score(
       query="How do I configure authentication?",
       answer="You need to set the API_KEY environment variable",
       passages=["The system reads API_KEY from environment variables..."]
   )

   print(f"Score: {result.score}")
   print(f"Supported: {result.supported_claims}")
   print(f"Unsupported: {result.unsupported_claims}")
   ```

4. **Run prompt benchmarks.**

   Execute the benchmarking script to compare prompt variants:
   ```bash
   pytest -k "eval"  # Run evaluation tests first
   python -m attune_rag.eval.bench_prompts  # Then run benchmarks
   ```

## Verify success

Your evaluation is working when:
- FaithfulnessResult objects return scores between 0.0 and 1.0
- Supported claims list contains facts directly stated in passages
- Unsupported claims list contains inferences or invented details
- Benchmark script completes without errors and outputs comparison metrics

## Key files

- `src/attune_rag/eval/faithfulness.py` - FaithfulnessJudge class and scoring logic
- `src/attune_rag/eval/bench_prompts.py` - Prompt comparison benchmarking
- `src/attune_rag/eval/__init__.py` - Module exports and main entry point
