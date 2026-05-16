---
type: reference
name: eval-reference
feature: eval
depth: reference
generated_at: 2026-05-15T20:03:10.208788+00:00
source_hash: 9d4d7626c287ef8da26b5caa6bc9f43
status: generated
---

# Eval reference

Use this API to score RAG answer faithfulness and run prompt A/B benchmarks. `FaithfulnessJudge` calls Claude with a tool-use protocol to classify each claim in an answer as supported or unsupported by the retrieved passages.

## Classes

| Class | Description |
|-------|-------------|
| `FaithfulnessResult` | Per-answer faithfulness verdict. |
| `FaithfulnessJudge` | Scores RAG answers for grounding in retrieved context. |

### `FaithfulnessResult`

Per-answer faithfulness verdict. Dataclass returned by `FaithfulnessJudge.score`.

#### Fields

| Field | Type | Default |
|-------|------|---------|
| `score` | `float` | — |
| `supported_claims` | `list[str]` | — |
| `unsupported_claims` | `list[str]` | — |
| `reasoning` | `str` | — |
| `model` | `str` | — |
| `thinking_used` | `bool` | `False` |

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `total_claims` | `int` | Total number of claims (supported + unsupported). |

#### Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `to_dict` | `self` | `dict[str, Any]` | Serializes the verdict to a plain dictionary. |

---

### `FaithfulnessJudge`

Scores RAG answers for grounding in retrieved context using Claude tool-use.

#### Constructor

| Parameter | Type | Default |
|-----------|------|---------|
| `client` | `AsyncAnthropic \| None` | `None` |
| `api_key` | `str \| None` | `None` |
| `model` | `str` | `DEFAULT_JUDGE_MODEL` |
| `timeout` | `float` | `DEFAULT_JUDGE_TIMEOUT_SECONDS` |

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `model` | `str` | The Claude model identifier used for judging. |

#### Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `score` | `self, query: str, answer: str, passages: str \| list[str], max_tokens: int = 2048, *, use_thinking: bool = False, thinking_budget_tokens: int = DEFAULT_THINKING_BUDGET_TOKENS` | `FaithfulnessResult` | Judges each factual claim in `answer` against `passages` and returns a verdict. |

---

## Functions

| Function | Parameters | Returns | Description |
|----------|------------|---------|-------------|
| `main` | `argv: list[str] \| None = None` | `int` | Runs the prompt A/B benchmark harness; returns `0` on success. |

## Constants

| Constant | Type | Value |
|----------|------|-------|
| `DEFAULT_JUDGE_MODEL` | `str` | `'claude-sonnet-4-6'` |

## Source files

- `src/attune_rag/eval/__init__.py`
- `src/attune_rag/eval/faithfulness.py`
- `src/attune_rag/eval/bench_prompts.py`

## Tags

`eval`, `faithfulness`, `judge`, `hallucination`, `scoring`
