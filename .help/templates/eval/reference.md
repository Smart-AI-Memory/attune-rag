---
type: reference
name: eval-reference
feature: eval
depth: reference
generated_at: 2026-05-20T03:28:38.726662+00:00
source_hash: 9d4d7626c287ef8da26b5caa6cd8470542d6e7b12acbf7d3e678d0c442cc9f43
status: generated
---

# Eval reference

Use this module to score RAG answers for faithfulness and run prompt A/B benchmarks against a golden set. `FaithfulnessJudge` calls Claude with tool use to decompose each answer into atomic claims and label each one as supported or unsupported by the retrieved passages.

## Classes

| Class | Description |
|-------|-------------|
| `FaithfulnessResult` | Per-answer faithfulness verdict. |
| `FaithfulnessJudge` | Scores RAG answers for grounding in retrieved context. |

---

### `FaithfulnessResult`

Dataclass that holds the verdict for a single scored answer.

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
| `to_dict` | `self` | `dict[str, Any]` | Serializes the result to a plain dictionary. |

---

### `FaithfulnessJudge`

Scores RAG answers by checking each factual claim against the retrieved passages. Uses Claude's tool-use API to decompose answers into atomic claims and classify each as supported or unsupported.

#### Constructor

| Parameters | Type | Default | Description |
|------------|------|---------|-------------|
| `client` | `AsyncAnthropic \| None` | `None` | An existing `AsyncAnthropic` client to reuse. |
| `api_key` | `str \| None` | `None` | Anthropic API key; used when `client` is `None`. |
| `model` | `str` | `DEFAULT_JUDGE_MODEL` | Claude model to use as judge. |
| `timeout` | `float` | `DEFAULT_JUDGE_TIMEOUT_SECONDS` | Request timeout in seconds. |

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `model` | `str` | The Claude model identifier used for judging. |

#### Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `score` | `self, query: str, answer: str, passages: str \| list[str], max_tokens: int = 2048, *, use_thinking: bool = False, thinking_budget_tokens: int = DEFAULT_THINKING_BUDGET_TOKENS` | `FaithfulnessResult` | Scores one answer against its retrieved passages and returns a per-claim verdict. |

---

## Functions

| Function | Parameters | Returns | Description |
|----------|------------|---------|-------------|
| `main` | `argv: list[str] \| None = None` | `int` | Runs the prompt A/B benchmark CLI; returns `0` on success. |

---

## Constants

| Constant | Type | Value |
|----------|------|-------|
| `DEFAULT_JUDGE_MODEL` | `str` | `'claude-sonnet-4-6'` |

---

## Source files

- `src/attune_rag/eval/__init__.py`
- `src/attune_rag/eval/faithfulness.py`
- `src/attune_rag/eval/bench_prompts.py`

## Tags

`eval`, `faithfulness`, `judge`, `hallucination`, `scoring`
