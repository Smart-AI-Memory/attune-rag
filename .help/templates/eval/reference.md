---
type: reference
feature: eval
depth: reference
generated_at: 2026-04-23T03:36:21.817462+00:00
source_hash: c51ee93d206242f5b5e8a7ad4ed5772aaf45bc2940d459444097d7a899d4513c
status: generated
---

# Eval reference

Judge RAG answer faithfulness and run prompt A/B benchmarks using LLM-based evaluation.

## Classes

| Class | Description |
|-------|-------------|
| `FaithfulnessResult` | Per-answer faithfulness verdict from LLM judge |
| `FaithfulnessJudge` | Scores RAG answers for grounding in retrieved context |

## FaithfulnessResult fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `score` | `float` | — | Faithfulness score between 0.0 and 1.0 |
| `supported_claims` | `list[str]` | — | Claims backed by retrieved passages |
| `unsupported_claims` | `list[str]` | — | Claims not found in passages or inferred |
| `reasoning` | `str` | — | Judge's explanation of scoring decision |
| `model` | `str` | — | Model used for judgment |

## FaithfulnessResult methods

| Method | Returns | Description |
|--------|---------|-------------|
| `to_dict()` | `dict[str, Any]` | Converts result to dictionary format |

## FaithfulnessResult properties

| Property | Type | Description |
|----------|------|-------------|
| `total_claims` | `int` | Total number of claims evaluated |

## FaithfulnessJudge methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `__init__` | `client: AsyncAnthropic \| None = None, api_key: str \| None = None, model: str = DEFAULT_JUDGE_MODEL, timeout: float = DEFAULT_JUDGE_TIMEOUT_SECONDS` | `None` | Initialize faithfulness judge with Anthropic client |
| `score` | `query: str, answer: str, passages: str \| list[str], max_tokens: int = 2048` | `FaithfulnessResult` | Evaluate answer faithfulness against retrieved passages |

## FaithfulnessJudge properties

| Property | Type | Description |
|----------|------|-------------|
| `model` | `str` | Model name used for judgment |

## Functions

| Function | Parameters | Returns | Description |
|----------|------------|---------|-------------|
| `main` | `argv: list[str] \| None = None` | `int` | Run prompt benchmark evaluation |

## Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `DEFAULT_JUDGE_MODEL` | `'claude-sonnet-4-6'` | Default model for faithfulness scoring |

## Source files

- `src/attune_rag/eval/__init__.py`
- `src/attune_rag/eval/faithfulness.py`
- `src/attune_rag/eval/bench_prompts.py`

## Tags

`eval`, `faithfulness`, `judge`, `hallucination`, `scoring`
