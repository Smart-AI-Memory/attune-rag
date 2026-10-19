---
type: reference
name: benchmark-reference
feature: benchmark
depth: reference
generated_at: 2026-06-10T06:07:59.704964+00:00
source_hash: 82975cf88c844b87657deb87845f45f4f5fbc32319ccba10e0eb8a798867630f
status: verified
---

# Benchmark reference

Precision/recall/faithfulness benchmark runner, installed as the `attune-rag-benchmark` console script. Gates CI on configurable thresholds; `--retriever {keyword,hybrid,transformer}` benchmarks each retrieval tier (keyword needs no retriever extra; hybrid uses `[embeddings]` and can fall back; transformer requires `[transformers]`); supports custom query files, abstention-threshold calibration via `--calibrate-abstention`, and optional faithfulness scoring via `--with-faithfulness`.

## Functions

| Function | Parameters | Returns | Description |
|----------|------------|---------|-------------|
| `main` | `argv: list[str] \| None = None` | `int` | Runs the benchmark suite and returns an exit code. Returns `0` on success. |

## Exit codes

| Code | Meaning |
|---|---|
| `0` | Requested mode completed successfully: normal CLI cutoffs passed, or calibration completed |
| `1` | A completed precision or primary faithfulness measurement fell below a CLI cutoff |
| `2` | Local, credential, schema, dependency, or invalid-data failure, including a failed optional comparison |
| `3` | Classified transient failure in the primary provider pass: typed connection/timeout, rate-limit, or server error |

## Cutoffs and repository CI

`--min-precision` defaults to `0.70`; `--min-faithfulness` defaults to `0.85` and gates the primary pass when `--with-faithfulness` is enabled. Both must be finite numbers between `0` and `1`. `-k` is a positive integer, default `3`. The CLI reports recall but does not gate it.

Repository CI uses `scripts/check_thresholds.py` and `docs/specs/release-quality-baseline/thresholds.json`: precision@1 ≥ `0.975`, recall@3 ≥ `1.0`, and mean faithfulness ≥ `0.9698` when enabled. It verifies the query-file SHA-256 and rejects missing or invalid measurements. The checker exits `0` for pass, `1` for regression, and `2` for invalid evidence.

The workflow retries once only for exit `3` corroborated by a current, valid retrieval pass and an unavailable primary faithfulness stage. A persistent classified provider outage can leave faithfulness unavailable while retrieval still gates. Local failures and measured regressions do not become outage passes.

## JSON results

`--json PATH` records `queries_path`, completed measurements, and an additive `outcomes` object with `retrieval` and `faithfulness` stage statuses. Completed retrieval is saved before later work; completed primary faithfulness is saved before optional comparisons. The usual primary key is `faithfulness_legacy`; `--compare-thinking` instead uses `faithfulness_thinking_off` and `faithfulness_thinking_on` and is outside the current CI checker's default-pass contract.

Stages can be `pending`, `completed`, `skipped`, `failed`, or (for primary faithfulness) `unavailable`. Failures add `reason`; controlled local validation errors can add `detail` naming the field or file without echoing invalid query or provider exception contents. Invalid measurements are rejected before that stage is recorded as completed. If the output path cannot be written, the command exits `2` and no usable receipt is guaranteed.

Successful `--calibrate-abstention` writes `calibration`, marks both benchmark stages `skipped`, sets reason `calibration_only`, and exits `0`. It recommends an absolute keyword-score threshold without applying it or running the faithfulness pass.

## Query format

`--queries PATH` accepts UTF-8 YAML with a nonempty `queries` list. Rows require unique nonempty string `id` values and nonempty string `query` values. If supplied, `expected_in_top_3` must be a list of nonempty path strings; these paths define successful retrieval. Invalid YAML, duplicate IDs, and malformed rows are local failures. The default golden files are in the repository's `tests/golden/`, not the wheel.

## Source files

- `src/attune_rag/benchmark.py`
- `scripts/check_thresholds.py`
- `.github/workflows/benchmark.yml`
- `docs/specs/release-quality-baseline/thresholds.json`

## Tags

`benchmark`, `ci`, `precision`, `recall`, `quality`, `retriever-tiers`
