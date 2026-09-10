---
type: troubleshooting
name: benchmark-troubleshooting
feature: benchmark
depth: troubleshooting
generated_at: 2026-06-10T06:07:59.719871+00:00
source_hash: 82975cf88c844b87657deb87845f45f4f5fbc32319ccba10e0eb8a798867630f
status: verified
---

# Troubleshoot benchmark

## Before you start

`attune-rag-benchmark` is a precision/recall/faithfulness benchmark runner that gates CI on configurable thresholds. It supports three retrieval tiers (`keyword`, `hybrid`, `transformer`), custom query files, abstention-threshold calibration, and optional faithfulness scoring. All functionality is exposed through `main()` in `attune_rag.benchmark`.

Key exit codes to know:

- `0` — the requested mode completed successfully (normal CLI cutoffs passed, or calibration completed)
- `1` — a completed measurement missed a CLI cutoff
- `2` — a local, credential, dependency, schema, or invalid-data failure
- `3` — a classified transient failure during the primary provider pass

## Symptom table

| If you observe | Check |
|---|---|
| Exit code `2` with an install hint | Run `pip show attune-rag` and confirm the extra for your `--retriever` tier is installed (e.g., `pip install "attune-rag[transformers]"`) |
| `Queries file not found` on a pip install | The default golden query sets live in the repo checkout (`tests/golden/`), not the installed wheel — run from a clone or pass `--queries` (and optionally `--negatives`) pointing at your own set |
| Scores unexpectedly low or missing | Confirm your query file format matches what the runner expects and that `--with-faithfulness` is set if you need faithfulness scores |
| Abstention threshold mismatch in CI | Check whether `--calibrate-abstention` was run on the same dataset used in CI; a threshold calibrated on a different corpus will produce unreliable results |
| Runner exits `0` but CI still fails | CLI defaults (`0.70` precision, `0.85` faithfulness) differ from repository CI thresholds (`0.975` precision, `1.0` recall@3, `0.9698` faithfulness when enabled); CI also verifies the query-file hash |
| Exit `2` after malformed queries or metrics | Read `outcomes.detail` for controlled local diagnostics; check YAML/UTF-8, unique IDs, required fields, finite rates in `[0, 1]`, and nonnegative finite latency |
| Exit `3` or a later comparison failure | Inspect current `outcomes` and saved measurements; primary outages retain retrieval, while comparison failures retain the completed primary result |
| Intermittent failures across runs | Check for environment drift — model weights, index state, or cached embeddings that differ between runs |
| Slow benchmark execution | Identify whether the bottleneck is the retriever tier (`keyword` is fastest; `transformer` is slowest) and confirm no unnecessary re-indexing is happening on each run |

## Diagnosis steps

1. **Reproduce with the minimal invocation.**
   Run `attune-rag-benchmark` with only the required arguments and your exact `--retriever` value. Confirm the failure occurs before adding optional flags like `--with-faithfulness` or `--calibrate-abstention`.

2. **Check the exit code and stderr output.**
   Exit `1` is a regression. Exit `2` covers local failures, not just missing extras. Exit `3` is a classified transient primary-provider failure. Use `--json report.json` and read `outcomes.reason` plus any controlled `outcomes.detail`; arbitrary provider exception text is redacted rather than printed in a traceback.

3. **Enable verbose output.**
   Re-run with `--verbose` to surface per-query results at the point of failure. The per-query table often identifies whether the issue is in retrieval, scoring, or threshold comparison.

4. **Run the benchmark tests.**
   Execute `pytest -k "benchmark" -v` to confirm which paths are covered. If a test exercises the failing case, use its fixtures to narrow down the input that triggers the bug.

5. **Isolate the retriever tier.**
   If the failure is tier-specific, run each `--retriever` value (`keyword`, `hybrid`, `transformer`) in sequence to determine whether the problem is tier-dependent or present across all three.

## Common fixes

- **Missing retriever extra.** When stderr identifies a missing dependency, install the correct extra:
  ```
  pip install "attune-rag[embeddings]"    # for --retriever hybrid
  pip install "attune-rag[transformers]"  # for --retriever transformer
  ```
  `keyword` requires no retriever extra. Hybrid can fall back to keyword-only when its embedding leg is unavailable, so a successful hybrid run alone does not prove embeddings ran. The default bundled corpus separately needs `[attune-help]`.

- **`Queries file not found` after `pip install attune-rag`.** The default golden query sets are part of the repo checkout (`tests/golden/`), not the published wheel. Either run from a clone (`git clone https://github.com/Smart-AI-Memory/attune-rag`) or pass `--queries` with your own set. To score your own corpus, `attune-rag-measure` is the purpose-built tool.

- **Faithfulness scoring not appearing.** Faithfulness scoring is opt-in. Pass `--with-faithfulness` explicitly; omitting it produces no faithfulness output and is not a bug.

- **Calibration is not a completed benchmark.** It recommends an absolute keyword-score threshold without applying it. Both JSON stages are `skipped` with reason `calibration_only`, even with `--with-faithfulness`. Re-run against representative legitimate and negative queries when the data changes:
  ```
  attune-rag-benchmark --calibrate-abstention --queries queries.yaml --negatives negatives.yaml --json calibration.json
  ```

- **CI threshold mismatch.** Check `docs/specs/release-quality-baseline/thresholds.json` and the current dump with `scripts/check_thresholds.py`. Do not lower locked thresholds to match the CLI defaults. A changed query-file hash requires an intentional baseline re-measurement. Missing or invalid evidence is a validation failure, not an unavailable-provider pass.

- **Malformed query input.** Supply UTF-8 YAML with a nonempty `queries` list, unique nonempty string IDs, and nonempty string queries. `expected_in_top_3`, when present, must be a list of nonempty path strings. Local diagnostics name the problem without echoing invalid query contents.

- **Dependency version drift.** A retriever or model dependency upgrade can shift scores between runs. Run `pip show <package>` to confirm installed versions match across environments, then pin the relevant packages in your CI requirements file.

## Source files

- `src/attune_rag/benchmark.py`

**Tags:** `benchmark`, `ci`, `precision`, `recall`, `quality`, `retriever-tiers`
