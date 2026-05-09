# attune-rag tests

## Running locally

```bash
# Install dev deps (includes pytest-cov)
pip install -e ".[dev]"

# Full suite
pytest

# With coverage (matches CI's ubuntu x py3.11 cell)
pytest --cov --cov-report=term-missing

# Just unit tests (skip the golden retrieval suite)
pytest tests/unit/

# Just golden retrieval (requires [attune-help] extra; long-running)
pytest tests/golden/
```

## LLM mocking standard

attune-rag itself is LLM-agnostic — the heavy lifting happens in
attune-author, which is the **reference layer** for LLM mocking
patterns. Cross-layer integration tests follow:

- Strip `ANTHROPIC_API_KEY` via an autouse fixture so a misconfigured
  test never reaches the network.
- Patch `anthropic.Anthropic` at import time, not at call site.
- Reset module-level singletons (e.g. `_RagPipeline`) between tests.

See `attune-author/tests/conftest.py` (`_lenient_polish_by_default`,
`_reset_rag_pipeline`). Pass 2 of the test-strategy spec will formalize
this into a shared `docs/testing-conventions.md`.

## Public API contract tests

`tests/unit/test_contracts.py` pins the public surface of `attune_rag`:

- Every name in `__all__` must be importable, the right kind (class /
  callable / dict), and have a docstring.
- Function signatures for the most-consumed callables (`build_augmented_prompt`,
  `RagPipeline.run`) preserve documented kwargs.
- `CitedSource` keeps the consumer-facing `template_path`/`score`/`excerpt`/`category`
  fields (attune-gui maps `result.citation.hits` directly into its own
  `RagHit` shape).

Adding a new public export requires updating `EXPECTED_ALL` in this
file — that's deliberate friction. attune-rag is the API contract
source for attune-gui, attune-help, and attune-author.

## Golden retrieval suite

`tests/golden/` runs each `queries.yaml` entry through a real
`RagPipeline` and asserts overlap with `expected_in_top_3`. Hard
queries are dynamically `@pytest.mark.xfail(strict=False)` so retriever
upgrades surface as `XPASS` rather than silent regressions.

Adding a query is a one-line YAML edit in `tests/golden/queries.yaml`.

## What's tested vs. not

Tracked in
`/Users/patrickroebuck/attune/specs/test-strategy/current-state.md`. After
pass 1, the highest-value remaining gaps in this layer are:

- `dashboard/show.py` (172 statements, 0% covered) — Rich CLI module
  with no tests; biggest remaining gap by line count.
- `dashboard/refresh.py` (~66%) — refresh paths.
- `eval/bench_prompts.py` (~70%) — prompt builders.

Pass 2 will revisit; stretch ceiling for this layer is **90%**
(rag is the contract source of truth, gets the highest gate).
