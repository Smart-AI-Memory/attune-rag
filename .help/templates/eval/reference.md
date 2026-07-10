---
type: reference
name: eval-reference
feature: eval
depth: reference
generated_at: 2026-07-10T13:05:28.724296+00:00
source_hash: 15853b482b28c9a8b23b9156fb0655f28895187b4de4d1e26cdb96202d404731
status: generated
---

# Eval reference

Use this API to score RAG answers for faithfulness and run prompt A/B benchmarks. `FaithfulnessJudge` calls Claude with tool-use to determine which claims in an answer are directly supported by the retrieved passages; `FaithfulnessResult` carries the per-answer verdict.

## Classes

| Class | Description |
|-------|-------------|
| `FaithfulnessResult` | Per-answer faithfulness verdict. |
| `FaithfulnessJudge` | Scores RAG answers for grounding in retrieved context. |

### `FaithfulnessResult`

`[dataclass]`

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
| `total_claims` | `int` | Total number of claims (supported plus unsupported). |

#### Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `to_dict` | `self` | `dict[str, Any]` | Serializes the verdict to a plain dictionary. |

---

### `FaithfulnessJudge`

Scores RAG answers for grounding in retrieved context.

#### Constructor

| Parameters | Type | Default | Description |
|------------|------|---------|-------------|
| `client` | `AsyncAnthropic \| None` | `None` | Async Anthropic client to use. |
| `api_key` | `str \| None` | `None` | Anthropic API key; falls back to environment variable if `None`. |
| `model` | `str \| None` | `None` | Model to use for judging; resolved via `default_judge_model()` if `None`. |
| `timeout` | `float` | `DEFAULT_JUDGE_TIMEOUT_SECONDS` | Request timeout in seconds. |
| `auth_mode` | `str \| None` | `None` | Authentication mode override. |

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `model` | `str` | The resolved model identifier used for judging. |

#### Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `score` | `self, query: str, answer: str, passages: str \| list[str], max_tokens: int = 3072, *, use_thinking: bool = False, thinking_budget_tokens: int = DEFAULT_THINKING_BUDGET_TOKENS` | `FaithfulnessResult` | Judges whether each claim in `answer` is supported by `passages` and returns a per-answer verdict. |

---

## Functions

| Function | Parameters | Returns | Description |
|----------|------------|---------|-------------|
| `main` | `argv: list[str] \| None = None` | `int` | Entry point for the prompt benchmark CLI; returns `0` on success. |
| `default_judge_model` | — | `str` | Resolves the judge model to the premium tier (`ATTUNE_MODEL_PREMIUM`). |

---

## Module constants

| Constant | Type | Description |
|----------|------|-------------|
| `__all__` | `list` | Public exports: `{'FaithfulnessJudge', 'FaithfulnessResult'}`. |
| `_JUDGE_SYSTEM_PROMPT` | `str` | System prompt that instructs the judge to decompose answers into atomic claims and mark each as `SUPPORTED` or `UNSUPPORTED` based solely on the retrieved passages. |
| `_JUDGE_USER_TEMPLATE` | `str` | User message template with `{query}`, `{passages}`, and `{answer}` slots; instructs the model to call the `report_faithfulness` tool with results. |

---

## Source files

- `src/attune_rag/eval/__init__.py`
- `src/attune_rag/eval/faithfulness.py`
- `src/attune_rag/eval/bench_prompts.py`

## Tags

`eval`, `faithfulness`, `judge`, `hallucination`, `scoring`
