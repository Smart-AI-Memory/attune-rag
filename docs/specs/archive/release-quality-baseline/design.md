# Spec: Release Quality Baseline (attune-rag)

## Phase 2: Design

**Status**: draft

### Architecture

#### Components

```
scripts/
  measure_baseline_variance.py   # new: runs benchmark N times, emits stats

docs/specs/release-quality-baseline/
  requirements.md
  design.md
  tasks.md
  baseline-1.md                  # locked: methodology + measured numbers
  thresholds.json                # machine-readable per-metric gate values

.github/workflows/
  benchmark.yml                  # new: per-PR + post-merge benchmark gate

src/attune_rag/
  # unchanged — gate is purely external
```

Rationale: the gate is **additive infrastructure**. It
must not require touching any module inside
`src/attune_rag/`, so Phase 3's API freeze is not coupled
to Phase 1's deliverable.

#### Data flow

```
PR opened
  → benchmark.yml runs `attune-rag-benchmark` against locked queries.yaml
  → script reads docs/specs/release-quality-baseline/thresholds.json
  → for each metric: fail if value < threshold[metric]
  → PR check: pass / fail / inconclusive
```

### API changes

None to the public Python API. CLI changes are limited to
the variance-measurement script, which is internal
tooling.

#### `scripts/measure_baseline_variance.py`

```
python scripts/measure_baseline_variance.py \
    --runs 20 \
    --queries attune_rag/eval/golden_queries.yaml \
    --out docs/specs/release-quality-baseline/baseline-1.md \
    --thresholds-out docs/specs/release-quality-baseline/thresholds.json \
    [--sigma 2.0]
```

Required flags:
- `--runs N` — number of benchmark runs (N ≥ 10).
- `--out PATH` — markdown report destination.
- `--thresholds-out PATH` — JSON thresholds destination.

Optional flags:
- `--queries PATH` — defaults to the bundled golden set.
- `--sigma FLOAT` — defaults to `2.0`; threshold =
  `mean − sigma × stdev`.

Exit codes: `0` on success, `1` on benchmark failure, `2`
on validation error (e.g. N < 10).

### Data model

#### `thresholds.json`

```json
{
  "measured_at": "2026-05-16T00:00:00Z",
  "commit": "<sha>",
  "queries_sha256": "<hash>",
  "runs": 20,
  "sigma": 2.0,
  "metrics": {
    "precision_at_1": {"mean": 0.823, "stdev": 0.014, "threshold": 0.795},
    "recall_at_3":    {"mean": 0.941, "stdev": 0.011, "threshold": 0.919},
    "faithfulness":   {"mean": 0.781, "stdev": 0.023, "threshold": 0.735}
  }
}
```

`queries_sha256` lets CI fail loud if the query set
changes without a re-measurement.

#### `baseline-N.md`

Locked human-readable record: date, commit, N, raw
per-run numbers, mean / stdev / threshold per metric, link
back to the thresholds.json that was generated.

### CI workflow shape (`.github/workflows/benchmark.yml`)

Triggers:
- `pull_request` against `main`
- `push` to `main` (post-merge sanity check)

Steps:
1. Checkout, install with `[all]` extra.
2. Run `attune-rag-benchmark` against the locked
   `queries.yaml`, dump JSON output.
3. Run a small `scripts/check_thresholds.py` that loads
   both the dumped JSON and `thresholds.json`, fails if
   any metric is below threshold.
4. On failure: post a PR comment with the metric, the
   measured value, and the threshold.

Conditional faithfulness: skip the faithfulness judge
unless the PR touches the modules listed in
`requirements.md` or includes `[full-bench]` in the title
— keeps CI cost predictable while still catching the
regressions that matter.

### Re-measurement procedure

Triggered manually whenever a judge-affecting change
lands (Phase 2's `--thinking` default flip is the
expected first user). Procedure:

1. Land the judge-affecting change on a topic branch.
2. Run `measure_baseline_variance.py --runs 20` on that
   branch.
3. Land a follow-up PR that updates **both**
   `baseline-N+1.md` (new locked numbers, link to N)
   **and** `thresholds.json`.
4. Same PR adds a `Changed` entry to CHANGELOG noting the
   re-measurement and the reason.

### Tradeoffs & alternatives

#### Threshold strategy

| Option | Pros | Cons | Chosen? |
|---|---|---|---|
| `mean − 2σ` per metric, measured floor | Empirically grounded; pre-empts noise objections; symmetric across metrics | Requires the upfront measurement run; baseline must be re-measured on judge changes | **Yes (Decision 1, 2026-05-15)** |
| Fixed `−2 pp` faithfulness, `−3 pp` P@1/R@3 | Simple; ships immediately | Guess; will produce false positives if σ is large or false negatives if σ is small | No |
| Per-query strict equality | Maximum sensitivity | Judge non-determinism (40 pp single-query swings) makes this unusable | No |

#### Cost containment for the faithfulness judge

| Option | Pros | Cons | Chosen? |
|---|---|---|---|
| Always run faithfulness on every PR | Maximum coverage | Material LLM cost per PR; CI gets slow | No (default) |
| Run faithfulness only on PRs touching retrieval / eval / prompts | Bounded cost; targets the surface that affects faithfulness | Misses faithfulness drift from non-obvious paths | **Yes (initial default)** |
| Run faithfulness on a `[full-bench]` opt-in label | Author opts in for risky changes | Easy to skip; requires discipline | **Yes (also wired)** |

#### Workflow location

| Option | Pros | Cons | Chosen? |
|---|---|---|---|
| New `benchmark.yml` workflow | Isolated; easy to disable independently | Two workflow files to maintain | **Yes** |
| Extend `tests.yml` with a new job | One file | Pytest and benchmark have different cost / latency profiles; mixing complicates `concurrency:` groups | No |

### Security

- The script runs in CI with whatever `ANTHROPIC_API_KEY`
  the existing workflows already use; no new secret.
- `--out` and `--thresholds-out` are written under the
  spec dir; path-traversal validation reuses the helper
  in `src/attune_rag/dashboard/render.py`.

### References

- Decision 1 in
  [docs/specs/ROADMAP-v1.md](../ROADMAP-v1.md) — locked
  2026-05-15.
- Calibration evidence for judge non-determinism:
  [docs/rag/faithfulness-thinking-calibration.md](../../rag/faithfulness-thinking-calibration.md)
  and the v1 + v2 ground-truth artifacts under
  `artifacts/calibration/`.
- Benchmark entry point: `src/attune_rag/benchmark.py`
  (`main()` — argparse-driven, supports `--json` for
  machine-readable output).
- Existing CI: `.github/workflows/tests.yml`.
