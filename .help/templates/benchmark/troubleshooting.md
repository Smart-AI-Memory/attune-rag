---
type: troubleshooting
name: benchmark-troubleshooting
feature: benchmark
depth: troubleshooting
generated_at: 2026-06-10T06:07:59.719871+00:00
source_hash: 82975cf88c844b87657deb87845f45f4f5fbc32319ccba10e0eb8a798867630f
status: generated
---

# Troubleshoot benchmark

## Before you start

`attune-rag-benchmark` is a precision/recall/faithfulness benchmark runner that gates CI on configurable thresholds. It supports three retrieval tiers (`keyword`, `hybrid`, `transformer`), custom query files, abstention-threshold calibration, and optional faithfulness scoring. All functionality is exposed through `main()` in `attune_rag.benchmark`.

Key exit codes to know:
- `0` — all thresholds passed
- `2` — a retrieval tier's required extra is not installed (the error message includes an install hint)

## Symptom table

| If you observe | Check |
|---|---|
| Exit code `2` with an install hint | Run `pip show attune-rag` and confirm the extra for your `--retriever` tier is installed (e.g., `pip install "attune-rag[transformers]"`) |
| `Queries file not found` on a pip install | The default golden query sets live in the repo checkout (`tests/golden/`), not the installed wheel — run from a clone or pass `--queries` (and optionally `--negatives`) pointing at your own set |
| Scores unexpectedly low or missing | Confirm your query file format matches what the runner expects and that `--with-faithfulness` is set if you need faithfulness scores |
| Abstention threshold mismatch in CI | Check whether `--calibrate-abstention` was run on the same dataset used in CI; a threshold calibrated on a different corpus will produce unreliable results |
| Runner exits `0` but CI still fails | Verify the threshold flags passed to `main()` match the values your CI configuration expects |
| Intermittent failures across runs | Check for environment drift — model weights, index state, or cached embeddings that differ between runs |
| Slow benchmark execution | Identify whether the bottleneck is the retriever tier (`keyword` is fastest; `transformer` is slowest) and confirm no unnecessary re-indexing is happening on each run |

## Diagnosis steps

1. **Reproduce with the minimal invocation.**
   Run `attune-rag-benchmark` with only the required arguments and your exact `--retriever` value. Confirm the failure occurs before adding optional flags like `--with-faithfulness` or `--calibrate-abstention`.

2. **Check the exit code and stderr output.**
   Exit code `2` means a missing extra — read the install hint printed to stderr. Any other non-zero exit points to a threshold failure or an unhandled exception; the traceback names the file and line.

3. **Enable verbose output.**
   Re-run with `--verbose` to surface per-query results at the point of failure. The per-query table often identifies whether the issue is in retrieval, scoring, or threshold comparison.

4. **Run the benchmark tests.**
   Execute `pytest -k "benchmark" -v` to confirm which paths are covered. If a test exercises the failing case, use its fixtures to narrow down the input that triggers the bug.

5. **Isolate the retriever tier.**
   If the failure is tier-specific, run each `--retriever` value (`keyword`, `hybrid`, `transformer`) in sequence to determine whether the problem is tier-dependent or present across all three.

## Common fixes

- **Missing retriever extra.** Exit code `2` means the selected tier's package extra is not installed. Install the correct extra for your tier:
  ```
  pip install "attune-rag[embeddings]"    # for --retriever hybrid
  pip install "attune-rag[transformers]"  # for --retriever transformer
  ```
  `keyword` requires no extra.

- **`Queries file not found` after `pip install attune-rag`.** The default golden query sets are part of the repo checkout (`tests/golden/`), not the published wheel. Either run from a clone (`git clone https://github.com/Smart-AI-Memory/attune-rag`) or pass `--queries` with your own set. To score your own corpus, `attune-rag-measure` is the purpose-built tool.

- **Faithfulness scoring not appearing.** Faithfulness scoring is opt-in. Pass `--with-faithfulness` explicitly; omitting it produces no faithfulness output and is not a bug.

- **Calibrated abstention threshold is stale.** If abstention behavior changed after a data or model update, re-run calibration against the current dataset:
  ```
  attune-rag-benchmark --calibrate-abstention --retriever <tier>
  ```

- **CI threshold mismatch.** If `main()` returns `0` locally but CI reports a failure, compare the threshold flags used in both environments. A threshold set in a CI config file that differs from your local invocation will produce different pass/fail results.

- **Dependency version drift.** A retriever or model dependency upgrade can shift scores between runs. Run `pip show <package>` to confirm installed versions match across environments, then pin the relevant packages in your CI requirements file.

## Source files

- `src/attune_rag/benchmark.py`

**Tags:** `benchmark`, `ci`, `precision`, `recall`, `quality`, `retriever-tiers`
