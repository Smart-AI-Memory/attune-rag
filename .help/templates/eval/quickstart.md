---
type: quickstart
name: eval-quickstart
feature: eval
depth: quickstart
generated_at: 2026-05-20T03:28:38.744529+00:00
source_hash: 9d4d7626c287ef8da26b5caa6cd8470542d6e7b12acbf7d3e678d0c442cc9f43
status: generated
---

# Score a RAG answer for faithfulness

`FaithfulnessJudge` uses Claude as a strict judge to check whether every factual claim in a RAG answer is directly supported by the retrieved passages — flagging hallucinations claim by claim.

```python
import asyncio
from attune_rag.eval import FaithfulnessJudge

judge = FaithfulnessJudge()  # uses claude-sonnet-4-6 by default

result = asyncio.run(judge.score(
    query="What is the return policy?",
    answer="Returns are accepted within 30 days. Items must be unworn.",
    passages=["Our return policy allows returns within 30 days of purchase."],
))

print(result.score, result.supported_claims, result.unsupported_claims)
```

Expected output:

```
0.5 ['Returns are accepted within 30 days.'] ['Items must be unworn.']
```

A `score` of `0.5` means one of two claims was grounded in the retrieved passage. The unsupported claim appeared in the answer but has no backing in the passages.


## Step 1: Set your API key

`FaithfulnessJudge` accepts an `api_key` argument, or reads `ANTHROPIC_API_KEY` from the environment:

```python
judge = FaithfulnessJudge(api_key="sk-ant-...")
```


## Step 2: Inspect the full result

`FaithfulnessResult` exposes everything the judge produced:

```python
print(result.score)               # float: supported / total_claims
print(result.total_claims)        # int: len(supported) + len(unsupported)
print(result.supported_claims)    # list[str]: grounded claims
print(result.unsupported_claims)  # list[str]: hallucinated or inferred claims
print(result.reasoning)           # str: judge's chain-of-thought
print(result.model)               # str: which Claude model was used
```


## Step 3: Enable extended thinking for harder judgments

Pass `use_thinking=True` when the answer is long or the passages are ambiguous:

```python
result = asyncio.run(judge.score(
    query="...",
    answer="...",
    passages=["..."],
    use_thinking=True,
))
print(result.thinking_used)  # True
```


## Step 4: Serialize for logging or CI

Call `to_dict()` to convert the result to a plain dictionary suitable for JSON logging or golden-set comparison:

```python
import json
print(json.dumps(result.to_dict(), indent=2))
```


---

Next: run `FaithfulnessJudge.score` across your full golden set and assert `result.score == 1.0` for every expected answer to wire faithfulness checking into your CI pipeline.


**Tags:** `eval`, `faithfulness`, `judge`, `hallucination`, `scoring`
