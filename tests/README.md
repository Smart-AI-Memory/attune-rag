# attune-rag tests

## Running locally

```bash
# Install dev + embeddings deps, as in source-test CI
python -m pip install -e ".[dev,embeddings]"

# Default suite (live-marked tests are excluded by pyproject.toml)
python -m pytest tests/

# With coverage (matches CI's ubuntu x py3.11 cell)
python -m pytest tests/ --cov --cov-report=term-missing --cov-report=xml

# Just unit tests (skip the golden retrieval suite)
python -m pytest tests/unit/

# Just golden retrieval (attune-help is included in the dev extra)
python -m pytest tests/golden/
```

## LLM mocking standard, `live` marker, CI guard, cost policy

See **`testing-conventions.md`** in the attune workspace umbrella —
the canonical reference (mocking pattern, `live` marker semantics, CI
guard expectation, cost & quota policy). Applies to all four layers.

attune-rag itself is LLM-agnostic. The `live` marker is registered in
`pyproject.toml` so any future opt-in tests have a consistent home.

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

[test_golden.py](golden/test_golden.py) runs all 40 entries in
`queries.yaml` through a real `RagPipeline` and asserts overlap with
`expected_in_top_3`. Every difficulty, including hard, uses a hard assertion.
The suite skips if `attune-help` is unavailable.

The separate 80-query `queries_paraphrased.yaml` set retains per-query
`xfail(strict=False)` marks: `XPASS` and `XFAIL` are informational.
Its aggregate R@3 watermark of 0.85 is a hard assertion. These per-query
results are distinct from the locked 40-query quality gate.

The active [quality threshold JSON](../docs/specs/release-quality-baseline/thresholds.json)
requires P@1 ≥ 0.975 and R@3 = 1.00, with mean faithfulness ≥ 0.9698
when selected and measured. It also locks the SHA-256 of `queries.yaml`;
changing that file without updating the baseline evidence causes a
validation failure.

## Source tests and installed-artifact validation

The unit and golden suites exercise the checkout's code. They do not by
themselves establish that a built sdist or wheel contains the required
runtime files. [check_distribution.py](../scripts/check_distribution.py)
provides that separate check:

```bash
# From the repository root; dist must be empty or use a new output directory
python -m pip install build ".[attune-help]"
python scripts/check_distribution.py --output-dir dist --report artifact-validation.json
```

The checker builds from a clean snapshot of current Git-tracked files,
then builds the wheel from the sdist. Untracked files are excluded;
tracked edits and deletions are reflected. It checks both archives for
byte-identical package `.py`/`.pyi` files and five required resources:
`py.typed`, both corpus override JSON files, the editor schema, and
dashboard HTML.

It then installs the wheel in a temporary environment and checks import
origins in isolated source and wheel subprocesses. Both use the same
prepared dependency versions. Ordered top-three paths and scores must
match for every locked query, and both probes must pass the active
retrieval thresholds. This check makes no generation or faithfulness
calls. `--sdist PATH --wheel PATH --report PATH` validates existing
artifacts without rebuilding them.

The JSON receipt records hashes, dependency versions, origins, and query
results. The [test workflow](../.github/workflows/tests.yml) runs this
checker separately; the [publish workflow](../.github/workflows/publish.yml)
verifies the receipt and exact artifact checksums before publishing the
validated files. These changes are listed under
[Unreleased](../CHANGELOG.md#unreleased).

## Gate behavior tests

| Tests | Evidence checked |
|---|---|
| `test_distribution_contract.py` | Missing or altered resources/source, import isolation, result parity, and receipt failures |
| `test_distribution_workflow.py` | Actual publish-shell verification rejects altered receipts/artifacts and propagates checker failure |
| `test_check_thresholds.py` | Quality schema, finite values in `[0, 1]`, query SHA, and exit codes 0/1/2 |
| `test_benchmark.py` | Benchmark stage outcomes and typed provider failures with mocked calls |
| `test_quality_workflow_outcomes.py` | Actual workflow shell with a fake benchmark and real threshold checker |
| `test_format_perf_delta.py` / `test_perf_workflow_outcomes.py` | Required CPU evidence, measurement failures, and blocking/advisory performance outcomes |

Retrieval remains required during a provider outage. Faithfulness is
selected only for configured-key full runs; other runs disclose the skip.
A full run gets at most one workflow retry for a classified transient
primary-provider failure, after retrieval passes. Missing or malformed
data, local failures, and measured regressions block instead of becoming
an inconclusive success. A persistent classified outage leaves
faithfulness unavailable and disclosed, with retrieval still required.

The perf workflow blocks only on `keyword_retriever_retrieve.cpu` and
`rag_pipeline_run.cpu` regressions. Wall-clock, directory loading, and
reranker regressions remain advisory. A crashed, cancelled, or skipped
measurement, or missing/invalid required CPU evidence, fails validation.
The workflow tests exercise these branches without provider calls.
