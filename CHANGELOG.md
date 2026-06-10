# Changelog

All notable changes to `attune-rag` are documented here.
Format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **The CLI now reaches the whole retrieval surface** (usability audit
  step 2 — previously these were library-only):
  - `attune-rag query --corpus-path DIR` and
    `attune-rag corpus-info --corpus-path DIR` — query/inspect your own
    markdown corpus (`DirectoryCorpus`) instead of the bundled default.
  - `attune-rag query --retriever {keyword,hybrid,transformer}` — the
    opt-in retrieval ladder from the terminal.
  - `attune-rag query --min-score SCORE` — abstention (keyword retriever
    only; combining with other retrievers is rejected with an
    explanation, pending the safe-abstention-defaults work).
  - `attune-rag query --prompt-variant {anti_prior,baseline,citation,strict}`.
- **Measure the retrieval tiers on your own corpus** (usability audit
  step 3 — users previously had no shipped way to reproduce the
  transformer-tier numbers before paying the ~GB torch install):
  - `attune-rag-measure --retriever {keyword,hybrid,transformer}` (and
    `measure(retriever=...)` in the Python API; `MeasureResult` and the
    JSON payload record which retriever scored the run). Keyword-path
    report bytes are unchanged — the golden snapshot still pins them.
  - `attune-rag-benchmark --retriever` gains the `transformer` choice
    (previously `keyword`/`hybrid` only).
  - Both CLIs exit 2 with the install hint — not a traceback — when the
    requested tier's extra is missing.
- **Predictable setup errors now exit cleanly.** Missing extras
  (`[attune-help]`, `[transformers]`), a bad `--corpus-path`, and
  conflicting flags print a one-line actionable `error: ...` and exit 2
  instead of dumping a traceback — the previous first-run experience of
  a bare `pip install attune-rag` + `attune-rag query`.

### Fixed
- **Benchmark + dashboard now explain themselves on a pip install.**
  Their default query-set paths resolve to `tests/golden/` in the repo
  checkout, which doesn't ship in the wheel — previously
  `attune-rag-benchmark` failed with a bare "Queries file not found"
  and `attune-rag dashboard show` rendered an unexplained partial
  snapshot. Both now state where the golden sets live and what to pass
  instead (`--queries` / `queries_path`), and the benchmark points
  corpus authors at `attune-rag-measure`. Explicit-path misses keep the
  short error.
- **`attune-rag-benchmark` console script now actually installs.** The
  README, CHANGELOG, and docs have documented it since 0.1.x, but it was
  never declared in `[project.scripts]` — every documented invocation
  failed with "command not found" and only `python -m attune_rag.benchmark`
  worked. Packaging defect fix; no new functionality. Note: the default
  golden query sets still ship in the repo checkout, not the wheel — on a
  pip install, pass `--queries` explicitly.
- **Type hints are no longer invisible downstream.** attune-rag has been
  fully annotated all along, but without a `py.typed` marker (PEP 561)
  mypy/pyright treated the package as untyped. The marker now ships in
  the wheel.
- **README editor example no longer calls a nonexistent
  `DirectoryCorpus.load()`** — the corpus API is lazy; constructing the
  corpus is enough.
- **README staleness swept**: Status section updated 0.2.0 → 0.5.1 (the
  classifier has been `4 - Beta` since 0.5.1, not `3 - Alpha`); "an
  `EmbeddingRetriever` is on the post-freeze roadmap" corrected — it
  shipped in 0.5.0; the Public API list now includes
  `EmbeddingRetriever`, `HybridRetriever`, and `TransformerRetriever`,
  which were exported and snapshot-tested but undocumented in that
  section.

## [0.5.1] — 2026-06-07

Packaging + docs corrections on top of 0.5.0. **No library code changes** —
the retrieval behavior is identical to 0.5.0.

### Changed
- **Packaging metadata** that was intended for the 0.5.0 cut: Development
  Status classifier `3 - Alpha` → `4 - Beta`; PyPI Homepage +
  Documentation URLs → `https://attune-rag.dev`.
- **README** now leads with the measured retrieval numbers (bundled
  100%/100%; unseen-corpus hard-paraphrase precision@1 25%→90%, recall@3
  25%→100% with the transformer tier; abstention 92%→8%) and a
  lightweight-vs-transformer "two ways to run it" guide. **Install
  instructions corrected** to list the `[embeddings]` and `[transformers]`
  retrieval tiers.
- **`.help/` corpus regenerated** to match the 0.5.0 retrieval code (6
  features refreshed: retrieval, corpus, providers, cli, expander,
  reranker).

## [0.5.0] — 2026-06-07

The retrieval-capabilities release. A full opt-in retrieval ladder lands on
top of the keyword default — static hybrid, configurable abstention, and a
heavyweight transformer tier — each measured against an unseen-corpus
benchmark and shipped without changing any default. The base install and
the keyword retriever behave exactly as in 0.2.0.

### Added

- **Transformer retrieval (`[transformers]` extra) — heavyweight opt-in.**
  `TransformerRetriever` ranks by a real sentence-transformers model
  (default `BAAI/bge-small-en-v1.5`), reusing `EmbeddingRetriever`'s
  matrix cache + cosine ranking and a new asymmetric `query_prefix` hook
  (BGE query instruction, applied to the query only). It is
  **embedding-primary** and for **arbitrary corpora where paraphrase
  recall matters** — it tanks a keyword-tuned corpus's top-1, so it is
  opt-in and never a default. Validated on two unseen corpora
  (`docs/specs/transformer-retriever/`): hard-tier paraphrase
  precision@1 **≈0.50 (torch-free ceiling) → 0.85–0.90**, recall@3 →
  1.00 — the one goal no torch-free retriever reaches. Cost: torch
  (~GB) + one-time model download (then offline), ~10–300 ms/query.
  `sentence_transformers` is imported lazily, so the base install is
  unaffected; `pip install attune-rag[transformers]` to enable.
- **Hybrid retrieval (`[embeddings]` extra) — opt-in.** `EmbeddingRetriever`
  (static `model2vec` embeddings: no torch, offline, ms-encode) and
  `HybridRetriever` (keyword + embedding fused via weighted RRF, with
  graceful keyword-only fallback when the extra is absent). Benchmark gains
  `--retriever {keyword,hybrid}`. KeywordRetriever stays the pipeline
  default. Measured (`docs/specs/rag-strengthening/` Phase 3): **+9pts
  recall@3 on an unseen/unstructured corpus**; tuned-corpus default path
  unchanged. Equal-weight RRF trades tuned-corpus top-1 precision, so the
  default weighting favors keyword (`keyword_weight=2.0`, tunable).
- **Generalization benchmark track (Phase 2).** `--corpus` /
  `--corpus-queries` measure retrieval against an unseen second corpus
  (`tests/golden/corpus_b/`); reported under `generalization` (advisory).
- **Abstention threshold + calibration (Phase 5).** `KeywordRetriever`
  now accepts `min_score=` — when every candidate is below it the
  retriever returns nothing (abstains) instead of surfacing a weak,
  likely-wrong match. New `attune-rag-benchmark --calibrate-abstention`
  sweeps the legit + out-of-corpus query sets and recommends a `min_score`
  for the corpus (the threshold is an absolute keyword score, so it is
  corpus-specific). Measured on the bundled corpus: `min_score=5` cuts the
  false-answer rate **92% → 8%** for a 2pt legit-recall cost. Default
  (`2.0`) unchanged — opt-in.

### Changed

- **Corpus-quality guide §6.4 (“Wiring into CI”) hardened into a
  standing-guard recipe.** The CI recipe now gates on **both**
  `--watermark-p1` and `--watermark-r3` (R@3 sits near 1.0 with
  headroom; real regressions surface in P@1 first), adds a
  pick-your-floors-from-a-measured-baseline step, and offers a
  second in-suite `pytest` shape (`measure()` +
  `watermark_failures()`) for repos that prefer a test over a
  standalone workflow. Also fixes the guide’s §7→§9 heading gap
  (the override-mechanism section is now §8). Docs-only; no API or
  CLI surface change — every primitive the recipe uses already
  shipped in 0.2.0.
- **CI now enforces `uv.lock` matches `pyproject.toml`** via the new
  `.github/workflows/lockfile.yml` workflow (single job:
  `uv lock --check` per PR + push to main). Closes a silent-drift gap
  — `uv.lock` existed in the repo but nothing verified it, and it
  silently drifted from `0.1.22` to `0.1.23` (then would have drifted
  to `0.2.0`+) until a release-time reproducibility issue would have
  surfaced. Same class of bug caught on attune-ai's v7.1.0 ship,
  which surfaced there only because attune-ai's pre-commit happens
  to call `uv run` (which auto-regenerates uv.lock as a side effect).
  This workflow is the explicit version of that protection.

## [0.2.0] — 2026-05-25

> **Cut via the freeze-override mechanism (W4.2 hard gate).**
> Phase 4's review/gate deliverables landed ~3 weeks ahead of the
> nominal calendar; rather than wait the remaining cadence-soak
> weeks for calendar-elapse alone, the user-facing additions below
> shipped via the per-PR override mechanism (`freeze-override`
> label + `[Override-rationale]` PR-body block). Full per-PR
> rationales:
> [#130](https://github.com/Smart-AI-Memory/attune-rag/pull/130)
> (`load_aliases_from_file`) and
> [#136](https://github.com/Smart-AI-Memory/attune-rag/pull/136)
> (`attune_rag.measure_corpus`). Override-release notation in
> [`docs/specs/downstream-validation/exit-summary.md`](docs/specs/downstream-validation/exit-summary.md).

### Added

- **`attune_rag.measure_corpus` public module — quality harness
  promoted from the v0 `scripts/measure_corpus.py` (user-corpus-onboarding
  M1).** Public Python API: `measure(corpus_path | bundled, queries_path,
  paraphrased_path=None, rerank=False, candidate_multiplier=3,
  extra_aliases_file=None) -> MeasureResult` + `MeasureResult` frozen
  dataclass with `p1` / `r3` / `n` / `paraphrased_*` / `per_query_table`
  / `paraphrased_per_query` / `per_difficulty_breakdown` / metadata
  (corpus_label, queries_path, queries_sha, etc.). `MeasureResult.report_markdown()`
  renders the same byte-identical shape the v0 script produced (golden
  snapshot at `tests/golden/measure_corpus_bundled.golden.md`
  unchanged); `MeasureResult.to_json()` for the alternate format;
  `MeasureResult.watermark_failures()` for CI-suitable threshold checks.
  CLI via `python -m attune_rag.measure_corpus ...` + the new
  `attune-rag-measure` console_script (`[project.scripts]`). Default
  `rerank=False` per D5's verdict (`reranker-evaluation/diagnostic-1.md`).
  `scripts/measure_corpus.py` retained as a backward-compat shim
  (~25 lines) so existing invocations keep working and the
  bundled-corpus golden snapshot test pins the regression net unchanged.
  Implements M1 of [`user-corpus-onboarding`](docs/specs/user-corpus-onboarding/).
  Adds 2 symbols to the v1.0.0 surface budget (running total: 4 of 5
  — `measure`, `MeasureResult`, `load_aliases_from_file`,
  `extra_aliases_file=` kwarg).

- **`attune_rag.corpus.load_aliases_from_file(path)` public helper**
  + **`DirectoryCorpus(extra_aliases_file=...)` kwarg** for loading
  path-keyed extra aliases from a JSON file. Schema mirrors
  `attune-rag`'s own `aliases_override.json`:
  `{"rel/path.md": ["alias one", "alias two"], "_comment": "..."}`.
  Underscore-prefixed keys are dropped; missing files raise
  `FileNotFoundError` with path in message; malformed JSON or
  non-list/non-string values raise `ValueError`. When both
  `extra_aliases` (inline dict) and `extra_aliases_file` are
  provided, inline wins per-path collision. `AttuneHelpCorpus`'s
  internal bundled-aliases loader refactored to use the new
  function (wrapped in a tolerant try/except for backward-compat
  on the bundled file). Strict-dominance verified: bundled
  `tests/golden/measure_corpus_bundled.golden.md` byte-identical
  pre/post refactor; `pytest tests/golden` unchanged. Implements
  M2 of [`user-corpus-onboarding`](docs/specs/user-corpus-onboarding/).

### Changed

- **`scripts/measure_perf_baseline.py` `DEFAULT_SIGMA` rolled back
  3.0 → 2.0 (Phase 5 perf-baseline-multi-run M3 follow-up).** Closes
  the M3 cycle started by [#137](https://github.com/Smart-AI-Memory/attune-rag/pull/137)
  (workflow matrix) + [#139](https://github.com/Smart-AI-Memory/attune-rag/pull/139)
  (first v2 locked baseline). The σ=3.0 inflation in v1 absorbed
  inter-run noise the single-invocation methodology couldn't see;
  the v2 methodology (`scripts/aggregate_perf_baseline.py`, K=5
  invocations) now captures that noise explicitly as
  `inter_run_stdev`, so the inflation is no longer load-bearing.
  Aligns `measure_perf_baseline.py`'s default with
  `aggregate_perf_baseline.py`'s `DEFAULT_SIGMA = 2.0` so v1-path
  single-invocation runs (per-PR `delta-check`'s `current.json`,
  future manual locks without `--per-invocation-out`) use the same
  σ as the v2-path aggregator. **The gate that matters — the
  locked baseline's threshold on main — is unchanged**: it was
  re-computed in #139 as `mean + 2.0 × inter_run_stdev` and is
  unaffected by this script-default change. Threshold-comparison
  check (per spec M3) in the PR body; only `rag_pipeline_run.{cpu,wall}`
  show a hypothetical σ=2.0 v1-path tightening vs v2 (~7%), all
  other metrics loosen substantially under v1 — expected and not
  operationally relevant since v1-path is non-canonical now.
  Internal tooling; no public API; one-line script change + comment
  update; freeze-legal.

- **`docs/POLICY.md` §4.1: "Surface budget (0.2.0 → v1.0.0)" — durable
  home for the 5-symbol public-surface cap committed in
  [`user-corpus-onboarding/tasks.md` scoping decision #5](docs/specs/user-corpus-onboarding/tasks.md).**
  Adds a running ledger of the four symbols already used (4/5 slots)
  cross-linked to their source PRs + specs. Names the test-snapshot
  mechanism that mechanically enforces the cap and the
  /spec-pass requirement to re-open scoping decision #5 if a 6th
  symbol becomes load-bearing. Tripwire intent: before the next
  PR that wants new public surface, this paragraph is the first
  place to check "is this worth burning the last slot?" Docs-only;
  no public API surface; no test impact.

- **`USER_CORPUS_GUIDE.md` §6 reframed around the packaged
  `attune-rag-measure` entry point (user-corpus-onboarding M3
  partial).** §6.2 now leads with three equivalent paths — the
  `attune-rag-measure` console script, `python -m
  attune_rag.measure_corpus`, and the `measure()` Python API — and
  ends with a worked Python-API example showing
  `per_difficulty_breakdown` and `watermark_failures()` use. The
  legacy 20-line scripted loop stays as the "build your own" path.
  §6.4 (CI wiring) updated: the gate is now `attune-rag-measure
  --watermark-r3 0.85` with an artifact upload, not
  `python scripts/measure_corpus.py`. **Depends on**: this guide
  references the public `attune_rag.measure_corpus` module promoted
  in #136; merge order should be #136 → this PR. The script
  (`scripts/measure_corpus.py`) remains as a backward-compat shim;
  existing invocations keep working unchanged. Docs-only; no public
  API surface; no test impact.

- **`.github/workflows/perf.yml` `lock-baseline` job restructured to
  multi-run methodology v2 (Phase 5 perf-baseline-multi-run M2).** The
  single-invocation lock job is replaced by a K=5 parallel matrix
  (invocation_index 0..4), each running
  `scripts/measure_perf_baseline.py --per-invocation-out <id>.json`
  with the existing `--include-llm` toggle preserved per matrix entry.
  A new `aggregate-baseline` job (`needs: lock-baseline`, runs after
  all matrix entries) downloads the K artifacts, runs
  `scripts/aggregate_perf_baseline.py`, and opens the bot-PR with the
  v2 locked baseline (mean / intra_run_stdev / inter_run_stdev /
  threshold = mean + σ × inter_run_stdev). `fail-fast: false` on the
  matrix so one bad invocation doesn't kill the others; the aggregator
  catches missing/duplicate invocation_index. Matrix entries run with
  `contents: read` only; the aggregator widens to `contents: write` +
  `pull-requests: write` for the bot-PR. Runner fingerprint logged
  per invocation so M5's clustering check has the raw data. The
  per-PR `delta-check` job is unchanged — Phase 5 M5 reads the v2
  threshold but the gate continues to consume `mean / stdev / threshold`
  via the backward-compat schema alias. **M3** (live v2 lock-baseline
  dispatch on main + σ 3.0 → 2.0 rollback PR) ships in a follow-up
  once the workflow change lands. Per
  [`perf-baseline-multi-run/tasks.md`](docs/specs/perf-baseline-multi-run/tasks.md)
  M2; freeze-legal (no public API; workflow + docs only).

- **D5 (reranker-evaluation) closed — verdict `rerank-default-off`.**
  [`docs/specs/reranker-evaluation/diagnostic-1.md`](docs/specs/reranker-evaluation/diagnostic-1.md)
  committed with N=5 live measurement against the bundled
  `AttuneHelpCorpus` + golden query sets. Multiple regression triggers
  fired against the rubric: Run B baseline P@1 dropped from 1.00 to
  0.985 (rerank demotes winning docs ~1.5% of runs), baseline R@3
  dropped to 0.995, paraphrased R@3 dropped to 0.9825. Only 1 of 10
  paraphrased P@1 residuals lifted at ≥4/5 stability — well short of
  the ≥3-of-7 the rubric needs for `rerank-default-on`. **D5 ratifies
  the existing `RagPipeline.reranker=None` default** — no flip at the
  v1.0.0 cut. Cross-link landed at
  [`user-corpus-onboarding/risks.md` §7](docs/specs/user-corpus-onboarding/risks.md);
  follow-up note in
  [`v1.0.0-release/design.md`](docs/specs/v1.0.0-release/design.md).
  Spec status banners (README/design/requirements/risks/tasks) promoted
  to `complete`. Spec drift corrected: `user-corpus-onboarding` scoping
  decision #7 claimed "Mirror RagPipeline default (currently `on`)";
  the actual default was always `None`. Script extended (M2.3):
  `_residual_stability()` + per-query residuals table in the report;
  +3 unit tests. ~$0.50 of Haiku spend (~600 calls), ~10 min wall-clock.

- **Reframe rerank-mode messaging from "measure the lift" to "measure
  whether rerank earns its keep" — informed by the N=1 bundled-corpus
  result (#133, rerank-neutral on a well-aliased corpus).** Updates
  `USER_CORPUS_GUIDE.md` §6.2 to surface both outcomes (lift vs.
  neutral) as informative + links to `baseline-with-rerank.md` as the
  worked example. Updates `scripts/measure_corpus.py`'s keyword-only
  footer to match. Bundled-corpus golden snapshot
  (`tests/golden/measure_corpus_bundled.golden.md`) regenerated to
  reflect the new footer wording — same numbers, same per-query
  results; only the marketing line changed.

- **`docs/specs/release-quality-baseline/baseline-with-rerank.md` published —
  N=1 indicative rerank-lift artifact (marketing/onboarding companion to
  `scripts/measure_corpus.py --with-rerank`).** Real-data example users
  can point at when reading [`USER_CORPUS_GUIDE.md`](docs/USER_CORPUS_GUIDE.md)
  §6.2. Result against the bundled `AttuneHelpCorpus` + golden query
  sets: **net rerank lift on aggregate = zero** (paraphrased P@1 70/80
  and R@3 79/80 unchanged with rerank on). Two individual queries flipped
  in opposite directions and canceled (`gqp-003b` ✗→✓, `gqp-014b` ✓→✗).
  Wall-clock ~2.5 min; API spend < $0.10 at Haiku list pricing.
  **Important framing in the artifact**: this is NOT the D5 verdict —
  N=1 doesn't apply the rubric; D5 ([`reranker-evaluation/`](docs/specs/reranker-evaluation/))
  uses N=5 + statistical CI for the default-flip decision, which gets
  written to `diagnostic-1.md` and supersedes this file.
  Docs-only PR; no public API surface; no test impact.

- **`scripts/measure_perf_baseline.py` extended for multi-run methodology
  (Phase 5 perf-baseline-multi-run M1).** New flags: `--per-invocation-out`,
  `--invocations K`, `--invocation-index I`. When `--per-invocation-out`
  is set, the script emits one invocation's raw N-trial timings as JSON
  (skipping v1 lock emission); a workflow matrix runs the script K times
  to collect K JSONs. **New companion** `scripts/aggregate_perf_baseline.py`
  combines K per-invocation JSONs into the v2 locked baseline with the
  dual-noise schema: `intra_run_stdev` (within-invocation jitter) +
  `inter_run_stdev` (between-invocation drift). Threshold rebases on
  inter-run noise (`mean + sigma × inter_run_stdev`) so per-PR
  delta-check gates the noise floor a single sequential run can't see.
  Backward-compatible v1 schema: `mean`, `stdev`, `threshold` keep their
  meaning (`stdev` aliases `inter_run_stdev`); new keys
  (`intra_run_stdev`, `inter_run_stdev`, `runs_per_invocation`,
  `invocations`, `methodology_version: 2`) added beside (R2 of
  [`perf-baseline-multi-run`](docs/specs/perf-baseline-multi-run/)).
  Internal tooling; no public API surface; pure stdlib. Workflow
  matrix wiring (M2) + first v2 lock-baseline run (M3) ship in
  follow-up PRs once Phase 5 opens.

- **`scripts/measure_reranker.py` ships as the D5 diagnostic harness
  (Phase 5 reranker-evaluation M1).** Compares `RagPipeline(reranker=None)`
  (Run A — deterministic, N=1) against `RagPipeline(reranker=LLMReranker())`
  (Run B — non-deterministic, N=5 by default), emitting a markdown
  diagnostic report with per-metric mean/p50/p95 and full reproducibility
  metadata (query SHA-256, commit SHA, reranker model, Anthropic SDK
  version, ISO timestamp). Run A includes an R1 strict-dominance check:
  fails loudly if the bundled-baseline numbers (1.00/1.00/0.8750/0.9875)
  don't reproduce. Uses the shared `attune_rag._scoring.score_queries`
  helper (no duplicated scoring logic; cross-link satisfied per
  `tasks.md` M1.1 annotation). `--skip-run-b` enables R1-only emission
  for the M1 PR without API spend; M2 (live run + verdict) and M3
  (cross-link to user-corpus-onboarding/risks.md) ship in follow-up
  PRs. Internal tooling; no public API surface. 9 unit tests cover
  query loading, R1 check pass/fail, aggregation math (mean/p50/p95),
  report rendering, deterministic metadata, and CLI smoke.

- **`scripts/measure_corpus.py` ships as the v0 user-corpus measurement
  harness (Phase 5 accelerator).** Standalone script that scores any
  attune-rag-compatible markdown corpus against a queries YAML, emitting
  a deterministic markdown report with aggregate P@1 / R@3 + per-query
  breakdown. Defaults match `RagPipeline()` (keyword-only, no rerank,
  deterministic, no API spend); opt-in `--with-rerank` adds a
  side-by-side comparison pass for data-backed corpus polish (requires
  `ANTHROPIC_API_KEY`, ~$0.05 per 80-query set at Haiku pricing).
  CI-suitable via `--watermark-r3` (default 0.85; non-zero exit on fail).
  Extracts a private scoring helper to `attune_rag._scoring` shared with
  `tests/golden/test_golden.py`; refactor is strict-dominance verified
  (bundled-baseline numbers byte-identical pre/post). Bundled-corpus
  output pinned at `tests/golden/measure_corpus_bundled.golden.md` via
  byte-identical golden-snapshot diff in
  `tests/golden/test_measure_corpus_bundled.py` — the strict-dominance
  regression net for the keyword path. Internal tooling; no public API
  surface growth. The v1.0.0 spec
  ([`user-corpus-onboarding/`](docs/specs/user-corpus-onboarding/))
  promotes this script into `attune_rag.measure_corpus` post-D5; the
  script stays as a backward-compat entry point. `docs/USER_CORPUS_GUIDE.md`
  §6.2 updated to document the new path. Cross-linked from
  `reranker-evaluation/tasks.md` M1.1/M1.3 so D5's harness reuses
  `_scoring.score_queries` rather than duplicating it.

- **`docs/USER_CORPUS_GUIDE.md` published (v0 forerunner of the
  Phase 5 framework-framing).** ~750-line user-facing guide
  documenting the working-today path for pointing attune-rag at a
  user corpus: directory layout + frontmatter schema, multi-token
  alias intent (the `MIN_ALIAS_OVERLAP = 2` consequence), authoring
  patterns + the `_tokenize()` validation discipline (the `bites →
  bit` lesson), the override file pattern + override-then-promote
  workflow, the `MIN_ALIAS_OVERLAP` knob (when to flip), stemmer
  gotchas + the `_MIN_STEM_LEN` floor, quality measurement (a
  20-line script users can build today), strict-dominance
  discipline + CI wiring, the `QueryExpander` lever, and where the
  override mechanism fits in package-ships-a-corpus shipping
  patterns. The guide is the v0 forerunner; the v1.0.0 spec
  ([`user-corpus-onboarding/`](docs/specs/user-corpus-onboarding/))
  upgrades it with packaged-harness ergonomics. Pure docs; no
  public-surface impact. Ships under freeze. Cross-linked from
  README "Quick start — custom corpus" section.

- **`docs/specs/perf-baseline-multi-run/` promoted from scaffolding →
  scoped (Phase 5 inter-run noise methodology).** Spec was authored
  decision-complete; the scoping pass added a confirmed-decisions
  table at the top of `tasks.md` (Option B matrix orchestration with
  K=5 parallel jobs and escape-to-Option-C if M5 shows clustering;
  N=20 per invocation; backward-compatible schema additions;
  σ rollback 3.0 → 2.0 lands in the same PR as the first v2 lock;
  three-trigger re-lock cadence; `methodology_version: 2`; cost
  ceiling K=5 N=20 inside 20-min workflow timeout) plus an M0
  entry-gate verification milestone for symmetry with D5 +
  user-corpus-onboarding. Activates when Phase 5 opens after the
  0.2.0 cut + 7-day no-hotfix watch. Docs-only; no public-surface
  impact.

- **`docs/specs/api-v0.2.0-cut/` promoted from scaffolding → scoped
  (W4.4 → mechanical).** Spec was authored decision-complete in
  `design.md`; the scoping pass added an M0 entry-gate verification
  milestone (×5 mechanical checks: cadence-clean / perf-baseline-holds
  / downstream-gate-green / security-clean / exit-summary-recommends-cut)
  and an explicit decisions table at the top of `tasks.md`. The eight
  scoping decisions confirmed: 0.2.0 = SemVer freeze only (not
  Production/Stable); classifier stays `3 - Alpha` through 0.2.x; no
  symbol additions in the cut PR (`### Changed` only); the cut
  ratifies the surface locked by `test_api_surface.py` as-is;
  POLICY.md gets a single past-tense copy edit; PyPI publish via the
  existing release workflow + `/attune-release-check` skill;
  deprecation shims stay through 0.2.x; 7-day hotfix watch
  acknowledged as Phase-5-reset risk. Spec is now executable when
  W4 closes and M0 gates green — making W4.4 ("open the 0.2.0
  successor spec for `/spec` scoping") mechanical. Docs-only; no
  public-surface impact.

- **W4.3 exit-summary skeleton drafted ahead of W4 close.**
  `docs/specs/downstream-validation/exit-summary.md` lands as a
  structured template with `<TODO>` placeholders for the numbers
  that fill in mechanically as W4 closes: phase calendar (with
  observed −2.5-week W2/W3 slack), four cadence-report verdicts,
  per-metric perf trend (gated vs advisory columns), security
  findings disposition (HIGH/MEDIUM/LOW counts), downstream-gate
  green record, W2 hand-off pointers, recommendation (CUT-0.2.0 /
  EXTEND-FREEZE), Phase 5 spec readiness table, and token-spend
  projection. The structure is reviewable today; only the
  measured values await. Per `tasks.md` W4.3.

- **W3.2 task row updated with `✅ done 2026-05-20` annotation**
  citing PR #81 (drift fix: the row was never updated when W3.2
  shipped). Same shape as the W1.1 / W2.1 / W2.2 / W3.1 / W3.3
  done annotations in the same file.

- **W3.3 coverage push — public `__all__` surface raised to ≥ 90 %
  per module (aggregate 90.04 %).** Twenty new tests across
  `tests/unit/test_editor_references.py` (+8) and
  `tests/unit/test_editor_rename.py` (+12) cover branches that
  end-to-end fixtures didn't naturally exercise: duck-typed
  corpus shapes via `entries()` callable, entries with empty
  content/path, docs with no frontmatter, block-style alias
  scans with blank-line/dedent terminators, unsupported rename
  kinds, normalizer rejections (empty / absolute / escape),
  `FileMove.to_dict` round-trip, `apply_rename` failure modes
  (source-missing, target-missing, no `_root`). Net delta:
  `editor/references.py` 88 % → 99 %; `editor/rename.py` 87 %
  → 92 %; total 88.5 % → 90.04 %. Also added `pragma: no cover`
  on the two Protocol-stub ellipses in
  `src/attune_rag/corpus/help_adapter.py` (uncoverable by
  design). 849/849 tests green. Closes W3.3 of
  [`docs/specs/downstream-validation/tasks.md`](docs/specs/downstream-validation/tasks.md).

- **`docs/specs/reranker-evaluation/` promoted from scaffolding →
  scoped (Phase 5 D5).** Seven open scoping questions resolved and
  recorded in the new [`tasks.md`](docs/specs/reranker-evaluation/tasks.md):
  Scope α (retrieval-only, no faithfulness); Approach B (new
  `scripts/measure_reranker.py`); N=5 for `rerank=on` + N=1 for
  `rerank=off`; tightened verdict thresholds (≥3 of 7 paraphrased
  misses fixed at ≥4/5 stability AND token cost ≤ $0.002/query →
  `rerank-default-on`; ≤1 fix OR baseline regression OR
  > $0.005/query → `rerank-default-off`; exactly 2 fixes →
  `corpus-shape-dependent-default`); default-flip PR deferred to
  v1.0.0 cut PR (cleaner exit-summary); report filename
  `diagnostic-1.md` (matches `embedding-retriever/diagnostic-1.md`
  convention); token cost via `response.usage` (SDK). Activates
  when Phase 5 opens after the 0.2.0 cut + 7-day no-hotfix watch.
  Docs-only; no public-surface impact.

- **`docs/specs/user-corpus-onboarding/` promoted from scaffolding →
  scoped (Phase 5 v1.0.0 framework-framing).** Seven open scoping
  questions resolved and recorded in the new
  [`tasks.md`](docs/specs/user-corpus-onboarding/tasks.md): module
  name `attune_rag.measure_corpus`; CLI entry shipped both as
  `python -m attune_rag.measure_corpus` and as `attune-rag-measure`
  console_scripts; default watermark R@3 ≥ 0.85 (matches the bundled
  floor); leaner 600-line v1 guide with worked-example §8 deferred
  to v1.1.0; `load_aliases_from_file` ships as a public helper (1
  symbol against the 5-symbol budget); `extra_summaries_file=` kwarg
  deferred to v1.1.0; harness mirrors the `RagPipeline` rerank
  default (with `--no-rerank` flag for ablation) and inherits D5's
  verdict at execution time. Four milestones (M0 entry gates, M1
  harness, M2 kwarg refactor, M3 guide, M4 polish + cut prep);
  M2 runs parallel to M1. Sequenced strictly after
  [`reranker-evaluation/`](docs/specs/reranker-evaluation/) D5
  closes so the harness inherits D5's verdict. Symbol budget: 4 new
  public symbols (within the spec's ≤ 5 cap). Docs-only scoping;
  no public-surface impact at this PR. Activates when Phase 5 opens
  after the 0.2.0 cut + 7-day no-hotfix watch + D5 closure.

- **`scripts/changelog_cadence.py` detects `> **Freeze override`
  blockquotes in CHANGELOG section headers.** Override-flagged
  `### Added` bullets no longer trigger `Status: RESET` in the
  weekly cadence report; they're surfaced as
  `Added | N (M under override)` while status reads `ON TRACK`.
  Closes a false-positive gap that would have reset the freeze
  clock on Monday 2026-05-25 against 0.1.22's authorized
  `MIN_ALIAS_OVERLAP` addition. Schema documented at
  [`docs/specs/downstream-validation/cadence-week-schema.md`](docs/specs/downstream-validation/cadence-week-schema.md).
  Internal-tooling change; no public surface impact.
- **Perf gate promoted advisory → blocking on CPU-time axis of
  `KeywordRetriever.retrieve` + `RagPipeline.run` (Phase 4 W3.1).**
  `scripts/format_perf_delta.py` gains `--gate-metric METRIC`
  (repeatable). When set, only regressions in the named metrics
  affect the exit code; other metrics still appear in the comment
  table with the ⚠️ icon but don't fail the job. Gated regressions
  render as ⛔ blocking. `.github/workflows/perf.yml` switches from
  `--advisory` to two `--gate-metric` flags and propagates the
  script's exit code instead of force-exiting 0. Measurement-failure
  branch stays advisory (RC=0) — a script crash shouldn't block
  merge. Wall-clock, reranker, and `directory_corpus_load` axes
  remain advisory through W3; wall-clock promotion re-evaluated in
  W4 per [`tasks.md`](docs/specs/downstream-validation/tasks.md)
  W3.1. Verified by smoke test: clean → exit 0, gated regression
  → exit 1, advisory-only regression → exit 0.

## [0.1.23] - 2026-05-21

> **Patch release under the Phase 4 freeze.** This release is
> entirely internal retrieval improvements + housekeeping. Zero
> `### Added` entries, zero new public symbols in any `__all__`.
> The freeze clock is not reset.
>
> Headline: the **alias-expansion sweep** (PRs #94–#108) closes
> paraphrased R@3 from 28.75% → 100% on the new 80-query
> regression set, with baseline R@3 holding at 100% across all
> 13 PRs. Full arc + lessons in
> [`docs/specs/alias-expansion-sweep/`](docs/specs/alias-expansion-sweep/);
> the diagnostic chain that motivated the alias approach (and
> ruled out an embedding retriever) is in
> [`docs/specs/embedding-retriever/`](docs/specs/embedding-retriever/),
> now permanent-deferred.
>
> The perf-gate σ=3.0 widening (#75 / #77) ships here as the
> short-term noise mitigation; the principled fix is scoped in
> [`docs/specs/perf-baseline-multi-run/`](docs/specs/perf-baseline-multi-run/)
> (Phase 5 work) and will restore σ=2.0 once inter-run stdev is
> measured directly.

### Changed

- **Alias-expansion sweep — paraphrased R@3 lifted 28.75% → 100%,
  baseline held at 100%.** Thirteen-PR sequence (#94–#108) that
  adds **180+ multi-token aliases across 13 attune-help feature
  clusters** via a new `aliases_override.json` mechanism in
  `DirectoryCorpus.extra_aliases` + `AttuneHelpCorpus`. The
  override mechanism mirrors the existing `summaries_override.json`
  path-for-path and is internal — `extra_aliases` is a new kwarg
  on existing classes, no new public symbols. Measured on the
  new 80-query [`tests/golden/queries_paraphrased.yaml`](tests/golden/queries_paraphrased.yaml)
  set (authored as no-token-overlap variants of the 40-query
  baseline goldens): paraphrased P@1 lifted **11.25% → 91.25%**;
  paraphrased R@3 lifted **28.75% → 100%**. Baseline 40-query set
  unchanged at P@1=100%, R@3=100%. **Zero new dependencies. Zero
  baseline regression across all 13 PRs** — strict-dominance
  held by running the full baseline diagnostic before each PR
  commit (one near-miss caught + corrected pre-merge on M12). One
  golden update (gq-013) at #108 to close a documented
  ranking-list incompleteness. Per-cluster PRs: #94 bug-predict +
  mechanism, #95 security-audit, #96 release-prep, #97 smart-test,
  #98 fix-test, #99 code-quality, #101 refactor-plan, #102
  planning, #103 doc-gen, #104 doc-orchestrator, #105 deep-review
  + doc-audit, #108 residuals (M12 + D4). Diagnostics + arc:
  [`docs/specs/alias-expansion-sweep/`](docs/specs/alias-expansion-sweep/),
  [`docs/specs/embedding-retriever/`](docs/specs/embedding-retriever/)
  (D1–D4 chain).

- **Paraphrase regression suite wired into `tests/golden/test_golden.py`
  with an aggregate watermark.** PR #100 added per-query
  `xfail(strict=False)` markers for the 80-query paraphrased set;
  PR #107 tightened the aggregate R@3 watermark floor from 0.50
  → **0.85** once the sweep stabilized; the current run shows
  80/80 xpass with R@3 = 1.00. Failure mode: if a future change
  regresses paraphrased R@3 below 0.85, the watermark assertion
  fails; per-query xpass→fail transitions surface as individual
  test failures.

- **W3.2: downstream-attune-gui gate promoted from advisory to blocking
  (internal CI only — no public API impact).** Removed
  `continue-on-error: true` from the test step in
  `.github/workflows/downstream-attune-gui.yml` so an attune-gui
  editor + RAG test failure now fails the workflow and blocks merge.
  The PR-comment step still runs under `if: always()` so reviewers
  see the verdict regardless. Per [`docs/specs/downstream-validation/design.md`](docs/specs/downstream-validation/design.md)
  §4's promotion ramp + Decision 2 of the v1.0 roadmap (attune-gui is
  the gating downstream for Phase 4). Rollback: re-add
  `continue-on-error: true` to the test step.
- **Perf-gate noise tolerance widened pre-W3.1 (internal tooling
  only — no public API impact).** Three coordinated changes to
  `scripts/measure_perf_baseline.py` and `.github/workflows/perf.yml`:
  (a) `DEFAULT_SIGMA` 2.0 → 3.0 (threshold formula tolerance);
  (b) per-PR `delta-check` `--runs` 10 → 30 (tighter per-PR mean);
  (c) `lock-baseline` workflow_dispatch default `runs` 30 → 50
  (tighter baseline mean). Rationale: PR #72 vs PR #74 produced
  a ±23 % swing on `rag_pipeline_run.cpu` despite zero perf-relevant
  code changes between them, exposing inter-run noise the original
  N=10 / 2σ formula didn't tolerate. With the gate set to promote
  to blocking in W3.1, the original setting would have produced
  false-positive blocks. Re-locked baseline at N=50 lands in a
  follow-up auto-PR from the `lock-baseline` workflow.
- **Lazy schema-key cache in `editor/lint.py`** — replaced module-import-time
  `_KNOWN_FRONTMATTER_KEYS = set(load_schema()["properties"].keys())` with
  an `@lru_cache(maxsize=1)` helper `_known_frontmatter_keys()`. Parallels
  `_validator()` in `editor/schema.py`; defers the schema-load cost (and
  any malformed-schema crash) to first use rather than import time.
- **Type-hint completion in `providers/__init__.py`** — `get_provider`
  gained `**kwargs: Any` (previously untyped). Pure annotation; runtime
  behaviour unchanged.
- **Style normalisation in `providers/gemini.py`** — `cached_prefix:
  (str | None) = None` → bare `str | None = None` to match the rest of
  the providers package.

### Fixed

- **`editor/rename.py` symlink-handling docstring** — the comment in
  `_normalize_corpus_relpath` claimed `Path.resolve(strict=False)` runs
  "WITHOUT following symlinks". It does follow symlinks; the containment
  check itself is correct (compares resolved candidate against resolved
  root, so symlink-escapes are caught). Docstring rewritten so future
  maintainers don't drop the resolution and reintroduce a containment
  hole. (W2.1 INFO finding; no behaviour change.)
- **Docstring fills** — added one-line docstrings to `_stem`
  (suffix-order contract), `Hunk.to_dict` / `FileEdit.to_dict` /
  `FileMove.to_dict` / `Diagnostic.to_dict`, and a "non-cryptographic
  content hash" comment to `_make_hunk`'s sha256 use. All from the
  W2.1 deep-review nit list.

## [0.1.22] - 2026-05-20

> **Freeze override (Phase 4 of v1.0 roadmap).** The
> `KeywordRetriever.MIN_ALIAS_OVERLAP` knob below is a new
> public class attribute that lands inside the Phase 4
> symbol-level freeze under the override mechanism
> documented in
> [`docs/specs/downstream-validation/tasks.md`](docs/specs/downstream-validation/tasks.md)
> §"Freeze semantics". `[Override-rationale]`: the knob
> changes the default ranking behavior for attune-help in a
> structurally meaningful way (replaces a documented
> semantic-tie miss with a corpus-side alias-tuning miss),
> and gating its default behind a separate 0.2.0 cut would
> have shipped the behavior change without the
> user-controllable safety valve. Cadence clock not reset
> per the spec's `Security`-scoped exception pattern,
> applied here for an internal quality knob with comparable
> reversibility — flipping `MIN_ALIAS_OVERLAP = 1` restores
> pre-0.1.22 behavior exactly.

### Added

- **`KeywordRetriever.MIN_ALIAS_OVERLAP` class attribute
  (default `2`).** Requires at least this many distinct
  query tokens to overlap an entry's alias-token union
  before crediting `aliases_hits`. Multi-token alias matches
  (the design intent — `"CI pipeline failing"`,
  `"publish to PyPI"`) still fire; single common tokens
  riding in via one alias no longer dominate ties. Set to
  `1` (via subclass or class-attribute override) to restore
  pre-0.1.22 behavior. Trade-off measured on the locked
  golden set: gq-020 ("write unit tests") now retrieves
  `quickstarts/generate-tests.md` at top-1 (was
  `concepts/tool-fix-test.md` — phantom `test` alias hit);
  gq-026 ("version bump and changelog") drops from top-1
  (`concepts/tool-release-prep.md` lost its single-token
  `vers` alias hit). Net Precision@1 unchanged at 0.975;
  remaining miss is now a corpus-side alias-content
  question rather than a documented embedding-gap.

### Changed

- **`KeywordRetriever` stemming: collapse `-ity` / `-ities`
  to a shared stem.** `_STEM_SUFFIXES` now includes `"ities"`
  and `"ity"` (ordered before `"ies"` so plurals strip to the
  same stem as the singular). Previously `vulnerability`
  stayed unstemmed while `vulnerabilities` stemmed to
  `vulnerabilit` via the `"ies"` suffix, so a singular query
  token never overlapped the plural in summaries. Measured
  delta on the locked 40-query golden set: Precision@1
  lifted from **0.950 (38/40) to 0.975 (39/40)** — gq-011
  ("vulnerability scan") now retrieves
  `concepts/tool-security-audit.md` at top-1 instead of
  `quickstarts/skill-security-audit.md`. Recall@3 holds at
  1.00 (40/40). `_MIN_STEM_LEN = 3` continues to protect
  short tokens (`city`, `pity`, `unity` left unchanged).
  Rationale and per-query trace:
  [`docs/specs/selection-criteria-robustness/proposal.md`](docs/specs/selection-criteria-robustness/proposal.md).

- **Locked baseline lifted (20-run, with faithfulness).**
  `docs/specs/release-quality-baseline/thresholds.json`
  re-measured after the changes above. P@1 floor lifts
  0.950 → 0.975 (stdev 0); mean_faithfulness floor lifts
  0.9686 → 0.9698 (mean 0.9801, stdev 0.0052); R@3 holds at
  1.0. `baseline-1.md` refreshed with the new per-run table.

## [0.1.21] - 2026-05-20

> **Phase 4 W0 setup ships.** Twelve weeks of W0 machinery + four
> security passes (HIGH + LOW) + the polished `.help/` corpus + the
> locked perf baseline. No public API change — every CHANGELOG bullet
> below is either `Security` (hardening), `Fixed` (CDN-supply-chain +
> XSS), or `Changed` (docs, internal tooling, freeze workflows).
> Phase 4 W1–W4 burn-in starts from this commit; the formal `0.2.0`
> SemVer cut follows once the four-week soak completes cleanly.
>
> Supersedes the never-published 0.1.20 (tagged at 1f9a7d7) so the
> security follow-up from PR #68 (W09.A.005..008) ships in the same
> release as the rest of W0.

### Security

- **Close macOS direct-path bypass in `_SYSTEM_DIRS` denylist**
  (`attune_rag.eval.bench_prompts`). Found during W0.11 triage of
  W0.9 finding W09.S.011: `/etc`/`/sys`/etc. were caught via
  symlink-resolution (those paths resolve to `/private/...` on
  macOS), but a user typing `--output /private/etc/passwd`
  directly bypassed the guard because the raw-path arm of the
  check didn't have `/private/etc` in its denylist. Mirrored
  each original entry under `/private/` so the direct form is
  blocked too. Did NOT add bare `/private`, `/var`, or
  `/usr/...` — those would over-block legitimate
  user-writable temp roots (pytest tmp_path lives under
  `/private/var/folders/...`). 3 new test cases added to
  `tests/unit/test_eval_bench_prompts.py`. Threat model is
  developer-typo, not a hardened jail.

- **W0.9 Source 2 LOW hardening (Phase 4).** Three LOW findings surfaced
  by a read-only security-review agent on `src/attune_rag/`, addressed
  in this PR rather than deferred:
  - **W09.A.005 — Rich-markup injection in `dashboard/show.py`.**
    Snapshot fields (error message, retriever / corpus name, feature
    labels, kind names, per-query feature) were interpolated raw into
    Rich markup strings. A corpus value containing `[blink]X[/blink]`
    would alter the developer's terminal styling. All untrusted fields
    now flow through `rich.markup.escape`. Tests in
    `tests/unit/test_dashboard_show.py`.
  - **W09.A.006 — ANSI escape in CLI stderr.** `attune-rag dashboard
    render --out` interpolates the user-supplied path into the
    `ValueError` rendered to stderr. A path with raw ANSI bytes would
    repaint the terminal. New `_safe_stderr(msg)` helper strips C0/C1
    control characters (preserves `\t \n \r`) before printing.
    Tests in `tests/unit/test_cli.py`.
  - **W09.A.007 — `exc_info=True` on Anthropic-SDK exceptions.**
    `LLMReranker.rerank` and `QueryExpander.expand` logged failures
    with full traceback (debug-level). Traceback frames can capture
    SDK locals that may include secret-adjacent material under future
    SDK changes. Now logs exception type + message only.
    Tests in `tests/unit/test_expander_reranker.py`.
- **Close render-path macOS denylist bypass.** Same class as W09.S.011
  but in `dashboard/render.py`. `Path("/etc/foo").resolve()` returns
  `/private/etc/foo` on macOS, slipping past the `/etc` denylist
  entry. Added `/private/etc`, `/private/sys`, `/private/dev`
  mirrors. Two new tests in `tests/unit/test_dashboard_render.py`.

### Fixed

- **Dashboard XSS hardening** (Phase 4 W0.9 / W0.11). Three HIGH-severity
  findings closed against `dashboard/render.py` and
  `dashboard/templates/dashboard.html`:
  - The embedded `snapshot` JSON now goes through `_json_for_script_block()`,
    which `\u`-escapes the less-than byte and the U+2028 / U+2029 line
    separators so a corpus value containing a literal `</script>` cannot
    terminate the inline `<script>` block.
  - The `title` argument is HTML-escaped via `html.escape(…, quote=True)`
    before substitution into `<title>…</title>`, so values containing
    `</title><script>…` cannot break out of the title element.
  - The Chart.js CDN `<script>` tag now carries
    `integrity="sha384-…"` (Subresource Integrity) plus
    `crossorigin="anonymous"` and `referrerpolicy="no-referrer"`,
    closing the CDN-compromise vector.
  Tests under `tests/unit/test_dashboard_render.py` cover all three.
  No public API change; freeze-compatible.

### Changed

- **README task #6 closeout.** Reranker row now ships real numbers
  (`llm_reranker_rerank.wall`: mean 728 ms, threshold 1.07 s, σ ≈ 170 ms),
  sourced from the full 4-benchmark lock that landed in #64. The other
  three rows refreshed in lockstep: the full lock measured slightly
  different timings than the LLM-free lock because the reranker
  benchmark exercises corpus paths that warm and re-evaluate adjacent
  hot paths, so the CPU means and σs shifted. New numbers:
  `keyword_retriever_retrieve.cpu` 3,212 µs / 34,493 µs (σ ≈ 15.6 ms,
  cold-cache-noise dominated, as before); `directory_corpus_load.cpu`
  47 µs / 66 µs (essentially unchanged); `rag_pipeline_run.cpu`
  537 µs / 625 µs (up from the LLM-free numbers, reflecting the
  full-pipeline measurement). Footnote updated to explain both
  noise profiles (cold-cache for keyword, Anthropic-network for
  reranker). Removed the "re-lock pending" placeholder; closes
  task #6.
- **README task #6 partial pass.** Filled real locked numbers into 3
  of the 4 perf-table rows (`keyword_retriever_retrieve`,
  `directory_corpus_load`, `rag_pipeline_run` — all `.cpu` axis,
  measured at N=30 on `ubuntu-latest` / CPython 3.11.15). Added a
  one-sentence note that the `keyword_retriever_retrieve` threshold
  is wide because cold-cache σ ≈ 3.5 ms dominates the `mean+2σ`
  formula. Added a "Bundled `.help/` corpus" section calling out the
  143 polished templates (13 features × 11 kinds) that landed in
  #58 + #61. Reranker row still placeholder — drops once the
  full lock runs with `include_llm=true`.
- **`.help/` corpus repolished for release readiness.** Cache cleared,
  every feature re-run through `attune-author generate --no-rag
  --all-kinds --fact-check strict`. Each of the 13 features now ships
  all 11 `.help/` template kinds (concept, task, reference, quickstart,
  faq, error, warning, tip, note, comparison, troubleshooting) — up
  from 3 kinds per feature in #58. 143 polished templates total (39
  pre-existing refreshed; 104 new). The `--no-rag` flag was needed
  because the default RAG-grounded polish was cross-contaminating
  attune-rag templates with attune-help vocabulary (function names,
  file paths, command names that don't exist in this repo); switching
  to no-RAG keeps polish doing its prose-rewriting work without
  pulling in foreign references. The `--fact-check strict` enforces
  that every reference (function, class, file path, link) resolves
  in the repo; the pass shipped here cleared strict on all 143 files.
  Project-doc kinds (docs/how-to, docs/tutorials) were attempted but
  systematically failed strict — wrong package paths and dead
  cross-doc links — and reverted; the dedicated `attune-author docs`
  three-stage pipeline is tracked as a follow-up.
- **`docs/specs/downstream-validation/security-findings.md`** —
  W0.11 partial triage: 10 of 11 stdlib findings confirmed
  `non-issue` after deeper code reads; the 11th (W09.S.011)
  surfaced the macOS denylist gap above and is being closed in
  this same PR. Source 2 (attune-ai deep sweep) still pending.

- **`.help/` corpus refreshed and extended.** Regenerated stale
  `pipeline` and `retrieval` templates against post-0.1.18 source.
  Added four new feature areas to close public-surface coverage
  gaps: `editor` (template-editor primitives), `dashboard`
  (living-docs three-stage pipeline), `expander` (LLM-driven
  query expansion), `reranker` (LLM-driven re-ranking). Each
  feature ships `concept`, `reference`, and `task` templates
  produced by `attune-author regenerate` with the polish pass
  applied. `features.yaml` extended in lockstep. No code change.
- **README** rewritten with the eval-as-marketing thesis foregrounded.
  New "Why attune-rag" section quotes the locked retrieval +
  faithfulness thresholds plus the per-hot-path latency baseline
  ([docs/specs/release-quality-baseline/baseline-1.md](docs/specs/release-quality-baseline/baseline-1.md),
  [docs/specs/downstream-validation/perf-baseline.md](docs/specs/downstream-validation/perf-baseline.md))
  as the primary differentiator. Added a comparison table vs
  LangChain / LlamaIndex and a "What attune-rag is not" section
  for honest self-disqualification. Status section bumped from
  stale `v0.1.10` to `v0.1.19`; roadmap section repositioned as
  post-freeze 0.2.0+ instead of "next minor release." No public
  API change.

- **Deep-review MEDIUM/LOW capture marked closed.** `security-findings.md`
  Source 2 now lists W09.A.005..007 as `fix-now (closed in this PR)` and
  documents the read-only-agent path as the workaround for the broken
  `security_audit` MCP. Hard gate (`zero severity: high open`) still
  holds; LOWs are now zero open as well.

- **`benchmark.yml` threshold-gate heredoc fix.** The mode-decision step used the `key=value` form to write `reason` to `$GITHUB_OUTPUT`, which GitHub Actions rejected with `Invalid format` when the PR's diff touched ≥ 2 faithfulness-affecting paths (multi-line value). Switched to the documented heredoc form (`reason<<EOF_REASON \n … \n EOF_REASON`). The gate now correctly emits `mode=full` with multi-line rationale. Surfaced on PR #68 which touched both `reranker.py` and `expander.py`.
## [0.1.19] - 2026-05-16

> **Phase 2 of the v1.0 roadmap** — the `--thinking` default
> decision is locked. No behavioral change ships in this
> version (the default was already OFF and stays OFF). The
> v3 ground-truth round (n = 30) gives a bootstrap 95 % CI
> on `(wins_off − wins_on)` of `[−1, +13]` — point estimate
> `+6`, CI includes zero. The decision rests on
> "off-favored but not statistically distinguishable at this
> sample size" plus judge variance well below the
> escalation threshold, NOT on a positive CI. Phase 3's
> API-surface groundwork (snapshot tests + deprecation
> policy + `__all__` audit) landed in 0.1.18 in parallel;
> the formal API freeze still targets 0.2.0. See
> [ROADMAP-v1.md](docs/specs/ROADMAP-v1.md) for sequencing.

### Changed

- **`--thinking` default decision locked: stays OFF (Phase 2
  of v1.0 roadmap).** v3 ground-truth round at n=30 (15 shift
  + 15 random + 2 controls). Bootstrap 95 % CI on
  `(wins_off − wins_on)` = `[−1, +13]` — point estimate +6,
  CI includes 0 (off favored but not statistically
  distinguishable). v3 off-to-on ratio 2.5× reverses the
  v1 → v2 narrowing (1.5× → 1.2× → 2.5×). Judge variance is
  small: `margin_stdev = 0.0189`, far below the 0.10
  escalation threshold (K=8 random-bucket queries × M=5 runs;
  5 of 8 hit σ=0 in both conditions). No baseline
  re-measurement needed because the default doesn't flip.
  Locked record:
  [`docs/specs/faithfulness-thinking-decision/decision.md`](docs/specs/faithfulness-thinking-decision/decision.md).
  Calibration writeup:
  [`docs/rag/faithfulness-thinking-calibration.md`](docs/rag/faithfulness-thinking-calibration.md).
- **`docs/rag/faithfulness-thinking-calibration.md` rewritten.**
  Top blockquote now states the locked Phase 2 decision
  (was "pending"). The duplicated v2 ground-truth section
  (two near-identical copies in the file) has been
  deduplicated. A new "v3 (2026-05-16, n = 30 rubric + 2
  controls)" section adds the aggregate-alignment table,
  bootstrap CI, judge-variance discussion with the σ=0
  finding, phantom-claim examples, the v1 → v2 → v3
  comparison table, and a trace of all six rubric rules.
- **`docs/specs/ROADMAP-v1.md`** marks Phase 2 complete
  and Phase 3 unblocked; current-version row reflects
  0.1.19.

### Added

- **`scripts/measure_judge_variance.py`** — judge-only
  variance measurement. Re-runs the FaithfulnessJudge M times
  per query in each condition (off, on) against captured
  answer + context from a calibration artifact. Outputs
  per-query mean/stdev + aggregate pooled stdevs +
  `margin_stdev`. Used by Phase 2 to anchor the rubric's
  noise-floor escalation rule.
- **Bootstrap CI + phantom-claim rate in
  `scripts/score_against_ground_truth.py`.** Extended with
  `--rubric-rule {legacy,design}`, `--control-ids`,
  `--bootstrap-iters`, `--seed`, and `--variance`. The
  design-rule classifier flags a tie iff `|off−on|`,
  `|off−label|`, `|on−label|` are all `< 0.025`. Bootstrap
  resamples per-query verdicts B times and reports the 2.5 %
  / 97.5 % quantiles. Phantom-claim detection uses a
  content-word overlap heuristic (overlap `< 0.40` ⇒
  flagged); honest about the limits — a literal-substring
  matcher was tried first and yielded a meaningless 100 %
  rate. The 6-rule acceptance rubric from `design.md` runs at
  the bottom of every score run.
- **`build_calibration_labeling_kit.py --n-random N` +
  `--seed`.** New bucket: N queries drawn uniformly from the
  remaining pool after shift+controls. Anchors the
  noise-floor measurement on typical queries instead of
  high-shift outliers. `_select_queries` now returns
  `(shifted, controls, random)`; the kit script's previous
  flat-list return is gone (M2.1 / M2.2 of Phase 2).
- **`docs/specs/faithfulness-thinking-decision/`** — full
  Phase 2 spec (requirements, design, tasks).
- **`docs/specs/faithfulness-thinking-decision/decision.md`** —
  locked, machine-readable YAML record of the Phase 2 verdict:
  per-round win counts, bootstrap CI bounds, phantom rate,
  variance numbers, prior-round comparisons, and the
  methodology footnote about the mid-round labeler shift.
  Future re-evaluations should start a successor spec
  directory rather than amending this record.
- **v3 calibration artifacts** at
  `artifacts/calibration/thinking-2026-05-16.json` (paired
  off+on benchmark at n=40),
  `ground-truth-2026-05-16.md` (n=32 labels; 30 rubric +
  2 controls), and `variance-2026-05-16.json` (K=8 × M=5
  judge-variance measurement).
- **New unit-test coverage for the calibration toolchain.**
  `tests/unit/test_measure_judge_variance.py` (9 tests:
  fake-judge integration, aggregate stdev math, CLI
  validation) and an expanded `tests/unit/test_calibration_scripts.py`
  (27 new tests across the design tie rule, content-word
  tokenizer, phantom-rate detector, bootstrap CI, and all
  six rubric branches; total 52 tests pass in 0.13 s).

## [0.1.18] - 2026-05-16

> Two parallel tracks ship together: **Phase 1 of the v1.0 roadmap**
> (release-quality baseline + CI gate) and **API-surface groundwork
> toward the 0.2.0 freeze**. The public surface is now documented and
> snapshot-tested, but formal SemVer commitments still begin at 0.2.0
> — see [docs/POLICY.md](docs/POLICY.md). Roadmap:
> [docs/specs/ROADMAP-v1.md](docs/specs/ROADMAP-v1.md). API spec:
> [docs/specs/api-v0.2-public-surface/](docs/specs/api-v0.2-public-surface/).

### Added

- **Release quality gate (Phase 1 of v1.0 roadmap).** Every PR
  is now gated against a locked retrieval + faithfulness
  baseline. Thresholds set at `mean − 2σ` per metric, measured
  from N = 20 back-to-back benchmark runs on commit `d98fabe`:
  `precision_at_1 ≥ 0.95`, `recall_at_3 ≥ 1.00`,
  `mean_faithfulness ≥ 0.9686`. Aggregate faithfulness σ
  measured at 0.0052 — much tighter than per-query judge
  non-determinism (40+ pp single-query swings) suggested.
- **`scripts/measure_baseline_variance.py`** — runs the
  benchmark N times on an unchanged HEAD and emits a locked
  `baseline-N.md` + machine-readable `thresholds.json`. Pure
  stdlib, subprocess-driven, stdout-parsed. Supports
  `--skip-faithfulness` for cheap script validation.
- **`scripts/check_thresholds.py`** — pure-stdlib threshold
  checker invoked by CI. Exits 0 (pass) / 1 (regression) /
  2 (malformed input or `queries_sha256` mismatch). Emits a
  deterministic markdown PR-comment body via `--comment-out`,
  wrapped in a `<!-- attune-rag-quality-gate -->` marker so the
  CI workflow edits the comment in place instead of stacking
  one per push. Supports `--skip-metric` (repeatable) for
  retrieval-only runs.
- **`scripts/smoke_check_gate.sh`** — six in-CI assertions on
  the gate's own plumbing (good / bad / broken dumps → exit
  0 / 1 / 2; comment written on regression, not on validation
  errors). Keeps the gate's logic exercised on every PR even
  when the real benchmark passes or is skipped.
- **`.github/workflows/benchmark.yml`** — runs the benchmark
  on every PR + push to main. Decides retrieval-only vs full
  faithfulness mode from PR title (`[full-bench]` opt-in) or
  diff (faithfulness-affecting paths). Degrades gracefully to
  retrieval-only when `ANTHROPIC_API_KEY` isn't in repo
  Secrets — auto-engages faithfulness gating once the secret
  lands, no workflow edit needed. Full pass retries once on
  transient API errors with 30 s backoff; inconclusive runs
  exit 0 with a `::warning::` annotation.
- **`docs/specs/release-quality-baseline/`** — full Phase 1
  spec (requirements, design, tasks) plus the locked
  `baseline-1.md`, `thresholds.json`, and re-measurement
  procedure at `re-measure.md`.
- **`docs/specs/ROADMAP-v1.md`** — 5-phase plan from 0.1.x to
  v1.0.0 with a decisions log.
- **Public API surface documented and snapshot-tested.**
  `tests/unit/test_api_surface.py` locks `__all__` for every PUBLIC
  module and the importability of every PUBLIC submodule. Symbol
  additions or removals must update the test in the same PR. See
  [docs/POLICY.md](docs/POLICY.md) for the deprecation policy (which
  takes effect formally in 0.2.0) and the README's new "Public API"
  section for the enumerated surface.
- **`AttuneHelpCorpus` re-exported at the package root.** Already
  reachable via `attune_rag.corpus.attune_help.AttuneHelpCorpus`;
  the root re-export makes the constraint visible in the surface
  test so accidental removal fails CI.
- **New `author` optional extra** (`pip install 'attune-rag[author]'`)
  pinning `attune-author>=0.13.0`. Centralizes the docs-authoring
  toolchain for `.help/templates/` regeneration so the venv doesn't
  drift onto pre-polish versions. New `Makefile` wraps the workflow
  (`make help-regen` / `make help-regen-batch` etc.) with a feature
  probe that sidesteps the stale-`__version__` packaging bug in the
  0.13.0 wheel.

### Changed

- **`attune-rag-benchmark --json` no longer requires
  `--with-faithfulness`.** Retrieval-only `--json` now emits a
  dump containing `retrieval` + `queries_path`. The dump shape
  is additive — existing consumers that read
  `faithfulness_legacy` see the same key when
  `--with-faithfulness` is passed. Enables the CI quality gate
  to dump retrieval metrics without spending API tokens.
- **Five editor submodules renamed** to drop the misleading
  underscore prefix (the underscore convention falsely signaled
  "private" for modules that downstream tools already imported by
  name). New canonical paths:
  - `attune_rag.editor.rename` (was `_rename`)
  - `attune_rag.editor.schema` (was `_schema`)
  - `attune_rag.editor.lint` (was `_lint`)
  - `attune_rag.editor.autocomplete` (was `_autocomplete`)
  - `attune_rag.editor.references` (was `_references`)

  The public symbols re-exported from `attune_rag.editor` are
  unchanged.

### Deprecated

- **`attune_rag.editor._rename` and four sibling underscore modules.**
  These now exist as deprecation shims that re-export the renamed
  modules and emit `DeprecationWarning`. They are removed in
  **attune-rag 0.3.0**.

## [0.1.17] - 2026-05-15

### Fixed

- **`attune_rag.__version__` was stale.** Both v0.1.15 and
  v0.1.16 shipped with `__version__ = "0.1.14"` because the
  release-prep flow only bumped `pyproject.toml` (the version
  PyPI uses) and never touched the in-source constant. Synced
  `__version__` to `0.1.16` and added
  `tests/unit/test_package_metadata.py` asserting it matches
  `importlib.metadata.version("attune-rag")` so the next
  release-prep PR will fail CI if it forgets the bump.

### Added

- **Calibration ground-truth labeling kit.** Two scripts under
  `scripts/`:
  - `build_calibration_labeling_kit.py` picks N queries from a
    `--compare-thinking --json` artifact (largest shifts + a
    few unchanged controls) and emits a markdown labeling
    template.
  - `score_against_ground_truth.py` reads the labeled markdown
    plus the artifact and reports which judge pass (off / on)
    aligned more closely with the human labels — the empirical
    signal that gates a future `--thinking` default-flip
    decision.

  Workflow documented in
  `docs/rag/faithfulness-thinking-calibration.md`. First kit
  for the 2026-05-15 run is committed at
  `artifacts/calibration/ground-truth-2026-05-15.template.md`
  (8 queries: 5 highest-shift + 3 controls). Known gap: the
  benchmark JSON doesn't yet capture the generator's answer
  text or retrieved passages, so the kit surfaces the judge's
  claim lists as a proxy; a follow-up will enrich the JSON.

- **Larger calibration kit (17 queries, v2).** A re-run of
  `--compare-thinking` against the 40-query golden set on
  enriched-JSON output (post-#26, with `answer` + `context`
  embedded per query) produced a fresh artifact at
  `artifacts/calibration/thinking-2026-05-15-v2.json`. From
  that, a 17-query labeling kit at
  `artifacts/calibration/ground-truth-2026-05-15-v2.template.md`
  (13 highest-shift + 4 controls). Since `answer` and
  `context` are embedded per query, the kit is now
  self-contained — no live API calls needed at label time.
  Surfaced a call-to-call-variance observation worth noting:
  v2's high-shift set differs significantly from v1's
  (e.g., gq-017 went from Δ=+0.182 to Δ=−0.250 across the
  two runs); judge non-determinism means each calibration
  captures a snapshot, not ground truth itself. See
  `docs/rag/faithfulness-thinking-calibration.md`.

- **Ground-truth validation of the v2 calibration (17 queries).**
  Follow-up to the v1 round below. Patrick labeled the larger
  17-query v2 kit under the same strict-lens philosophy.
  Outcome: **off-closer 6, on-closer 5, tied 6** — option B
  confirmed at 2× sample size. V1's off-vs-on margin narrowed
  from 1.5× to 1.2× — on more competitive than v1 suggested,
  but not enough to flip the call. Labels at
  `artifacts/calibration/ground-truth-2026-05-15-v2.md`;
  results appended to
  `docs/rag/faithfulness-thinking-calibration.md`.

- **Ground-truth validation of the v0.1.15 calibration.**
  Patrick labeled the 8-query kit interactively under a strict
  lens; results are committed at
  `artifacts/calibration/ground-truth-2026-05-15.md` and
  written into the calibration doc. Outcome: among the 5
  high-shift queries, **off-closer 3, on-closer 2, tied 0**
  (3 controls were tied). Also surfaced a phantom-claim
  pattern in judge-on (paraphrases the answer into more
  specific claims, then flags its own paraphrases). Decision
  Option B (keep `--thinking` opt-in) is now empirically
  backed rather than absence-of-evidence-based.

## [0.1.16] - 2026-05-15

### Added

- **`attune-rag-benchmark --compare-thinking`** runs the judge
  twice (thinking off, thinking on at `--thinking-budget`) and
  prints a side-by-side aggregate table plus a per-query
  verdict-shift list. Mutually exclusive with `--thinking` and
  `--native-citations`. Calibration data lives in
  `docs/rag/faithfulness-thinking-calibration.md` (resolves
  [#17](https://github.com/Smart-AI-Memory/attune-rag/issues/17)).
  Outcome of the 2026-05-15 calibration run: 80 % verdict-shift
  rate but mean faithfulness barely moves (−0.005) and
  hallucination rate worsens slightly (+2.5 pp); `--thinking`
  stays opt-in pending hand-labeled ground-truth queries.
- **`attune-rag-benchmark --json PATH`** dumps the full
  structured faithfulness report — including per-query
  reasoning text and the supported / unsupported claim lists —
  to a JSON file for offline analysis. Per-query benchmark
  records also gain `supported_claims`, `unsupported_claims`,
  `reasoning`, and `latency_ms` fields.
- **`plan_rename(..., kind="template_path")`** is now
  implemented (was `NotImplementedError` in v0.1.15). Moves a
  template file within its corpus root and updates path-keyed
  sidecars (`summaries.json` / `summaries_by_path.json`) when
  present. `RenamePlan` grows a new `moves: list[FileMove]`
  field and `FileMove(old_path, new_path)` is exported from
  `attune_rag.editor`. `apply_rename` applies moves first
  (creating missing parent directories, tracked for rollback)
  then text edits, reversing prior work on any mid-flight
  failure. Out of scope: cross-corpus moves, `cross_links.json`
  updates, attune-help static index, git history. The gui-side
  "Rename file…" trigger and WS path-change handling live in
  attune-gui. See `docs/specs/template-path-rename/`.

## [0.1.15] - 2026-05-15

### Added

- **Extended thinking on the faithfulness judge.**
  `FaithfulnessJudge.score` now accepts keyword-only
  `use_thinking: bool = False` and
  `thinking_budget_tokens: int = 32768`. When enabled, the
  judge sends Anthropic's `thinking={"type": "enabled", ...}`
  block and swaps `tool_choice` to `"auto"` (required by
  Anthropic when thinking + tools are used together on Claude 4
  models). The response parser now prefers `tool_use` blocks
  (schema-guaranteed happy path) and falls back to JSON-parsing
  a `text` block only when the model declines the tool. The
  `_strip_code_fences` helper handles ` ```json ` wrappers in
  thinking-mode text responses.
- **`FaithfulnessResult.thinking_used`** — new boolean field
  (defaults to `False`) surfaces whether the verdict was
  produced with thinking enabled. Included in `to_dict()`.
- **`attune-rag-benchmark --thinking`** flag plus
  `ATTUNE_RAG_FAITHFULNESS_THINKING` env-var default and
  `ATTUNE_RAG_FAITHFULNESS_THINKING_BUDGET` for budget override.
  Per-query benchmark output gains a `thinking_used` column.

### Changed

- **`anthropic` SDK floor pinned to `>=0.95,<1.0`** (was
  `>=0.40.0,<1.0`) across the `[claude]`, `[all]`, and `[dev]`
  extras. Required for stable extended-thinking + tool-use
  support on Claude 4 models.

## [0.1.14] - 2026-05-08

### Changed

- **Native citations: caching enabled by default.** The first
  document in a citations request now carries
  `cache_control: {"type": "ephemeral"}` — one marker covers
  the entire document prefix per Anthropic's caching semantics.
  Empirically verified by the V2 probe: a 3799-token payload
  yielded full cache hits on the second call
  (`cache_read_input_tokens=3799`,
  `cache_creation_input_tokens=0`) with ~29% latency reduction
  (3102ms → 2190ms). No code change for callers; identical
  inputs to `RagPipeline.run_and_generate(use_native_citations=True)`
  now get cheaper on repeat calls.
- **`MAX_CITATION_DOCUMENTS`: 20 → 200.** V3 probe accepted every
  count in `{5, 10, 20, 30, 50, 75, 100, 150, 200}` without
  rejection; Anthropic's actual cap is higher still. The new
  ceiling gives generous headroom while still surfacing a clean
  `ValueError` if a caller accidentally tries hundreds.
- **Docs (`docs/rag/native-citations.md`):** "Open verification
  gates" section updated to "Verification gates — resolved
  2026-05-08" with the V2 / V3 findings inline. The "Caching"
  and "Document-count ceiling" sections now reflect the
  defaults.

### Added

- **Verification probes** at
  `scripts/probe_v2_cache_control.py` and
  `scripts/probe_v3_doc_count_ceiling.py`. Manual one-shot
  scripts that re-run the V2 / V3 verifications against the
  live Anthropic API. Cost ~$0.01 each. Useful when the SDK or
  service contract may have changed.

## [0.1.13] - 2026-05-08

### Added

- **Native Anthropic Citations API** (opt-in). New
  `use_native_citations` kwarg on `RagPipeline.run_and_generate` —
  when True (and the provider supports it), retrieved hits are
  sent as `custom_content` document blocks and the model emits
  claim-level citations attached to its response text. Falls
  back to the legacy `[P{n}]`-marker path on providers without
  support (Gemini) and on empty-hit retrievals.
  - New types: `attune_rag.ClaimCitation` (response-span +
    document-index + cited-text); `attune_rag.providers.base.
    CitationDocument`, `CitedResponse`.
  - New helper: `attune_rag.format_claim_citations_markdown`
    renders response text with footnote-style claim
    attribution.
  - New protocol surface: `LLMProvider.supports_native_citations`
    flag and `LLMProvider.generate_with_citations`.
    `ClaudeProvider` implements both. `GeminiProvider` declares
    the flag = False; pipeline detects this and falls back.
  - `RagResult` gains `claim_citations: tuple[ClaimCitation, ...]`
    and `used_native_citations: bool` (both default empty/False
    so all existing callers are unaffected).
- **`attune-rag query --native-citations`** CLI flag — opt into
  the native path from the command line. Renders the response
  with footnote citations when active.
- **`attune-rag-benchmark --native-citations`** flag (with
  `--with-faithfulness`) — runs a side-by-side faithfulness
  pass on both paths and prints a comparison table including
  hallucination rate, citation emit rate, and p95 latency.
- **Docs:** `docs/rag/native-citations.md` design note covering
  when to use which path, fallback behavior, open verification
  gates (V2 cache_control, V3 doc-count ceiling), and rollback.

### Removed

- **OpenAI provider** (`attune_rag.providers.openai.OpenAIProvider`) and
  the `[openai]` install extra. Use `[claude]` or `[gemini]` instead.
  No external consumer of attune-rag was importing `OpenAIProvider`;
  the provider is fully removable without callsite migrations.

## [0.1.12] - 2026-05-05

### Added

- **`attune_rag.editor` submodule** — template-editor backend for
  attune-gui. Provides:
  - **Linting** (`lint_template`) — frontmatter validation against the
    bundled JSON schema, plus body-level checks (broken aliases, unknown
    tags, missing references).
  - **Autocomplete** (`autocomplete_aliases`, `autocomplete_tags`) —
    case-insensitive prefix lookups over a corpus's known aliases and
    tags, with rich result records for the editor's command palette.
  - **Rename refactoring** (`plan_rename`, `apply_rename`,
    `RenameCollisionError`, `RenameError`) — preview + apply rename for
    aliases, tags, and template paths. Atomic multi-file write with
    in-memory snapshot rollback on partial failure.
  - **Frontmatter schema** (`_schema.load_schema`) — single source of
    truth for both server-side validation and the browser form. Covers
    template type, name, tags, aliases, summary, source, confidence,
    subtype, category, plus regen-pipeline read-only fields (hash,
    source_hash, feature, depth, generated_at, status).
- **`attune_rag.editor._rename._hunks`** — line-based hunk computation
  used by the editor's per-hunk save flow.

## [0.1.8] - 2026-04-24

### Changed

- **`LLMReranker(timeout=60.0)`** — new parameter, passed through
  to `Anthropic(...)` client init. Matches the `FaithfulnessJudge`
  pattern and protects callers from hanging network calls during
  re-ranking. Default 60 seconds; override if needed.

- **Fail-safe fallback logging now includes tracebacks.**
  `QueryExpander.expand()` and `LLMReranker.rerank()` already
  degrade to keyword-only on any API or parse error. The debug
  log line now carries `exc_info=True` so the full traceback is
  available when diagnosing why a fallback triggered.

- **`KeywordRetriever.retrieve()` now explicitly rejects a `None`
  corpus** with a clear `ValueError`. `RagPipeline` already
  guards this; the retriever is public, so the check hardens the
  direct-use path.

### Fixed

- **`attune_rag.__version__`** now tracks `pyproject.toml` (was
  stuck at `0.1.6` through the `0.1.7` release).

### Added

- **`from __future__ import annotations`** added to `__init__.py`
  so forward references in re-exports stay safe if type hints
  are refactored later.

- **Dashboard source formalised in the git repository.** The
  `attune_rag.dashboard` subpackage, its shipped template, and
  the accompanying unit tests (`tests/unit/test_dashboard_*`)
  have been tracked in git since this release. The code itself
  has been on PyPI since 0.1.6 — this commit catches the
  repository up to what the wheel has been shipping. Also
  includes `docs/specs/dashboard-v0.2.0.md` with an
  Implementation Note reconciling the locked spec against the
  shipped design.

### Docs

- **Roadmap — embeddings** section added to README pointing at
  the planned `attune-rag[embeddings]` extra using
  [`fastembed`](https://github.com/qdrant/fastembed). The
  decision record was already in this changelog; this makes it
  visible to new readers at the top level.

## [0.1.7] - 2026-04-24

### Changed

- **Raise `attune-help` cap from `<0.8` to `<0.10`** so consumers
  can co-install `attune-help` 0.9.0, which adds
  `Feature.doc_paths` (list form) and a top-level `_docs:` bucket
  to the manifest schema. No behavioural change in `attune-rag`
  itself — this is a compatibility-only bump.

  The 0.9.0 schema additions are backward compatible: legacy
  manifests using `doc_path:` (scalar) continue to load via
  dataclass coalescence. `attune-rag` does not read manifests
  directly, so neither the sidecar loader nor any retrieval
  path is affected.

## [0.1.6] - 2026-04-23

### Added

- **`QueryExpander`** (`attune_rag.expander`) — uses Claude Haiku to generate
  alternative phrasings for a query before keyword retrieval, improving recall
  for synonym and paraphrase queries. Requires `[claude]` extra. Caches
  expansions in-process; falls back gracefully to keyword-only on any API error.

- **`LLMReranker`** (`attune_rag.reranker`) — uses Claude Haiku to re-rank
  keyword retrieval candidates by semantic relevance. Retrieves
  `k × candidate_multiplier` candidates then returns the top-k in ranked order.
  Requires `[claude]` extra. Falls back to keyword order on any API error.
  System prompt includes attune-domain ranking guidance (prefer `tool-*` concept
  docs over `skill-*/task-*` quickstarts for workflow-goal queries).

- **`RagPipeline(expander=..., reranker=...)`** — both components are opt-in and
  composable. Expander enriches recall; reranker corrects precision.

- **`summaries_override.json`** — bundled override sidecar with 55 keyword-enriched
  summaries for attune-help corpus entries. `AttuneHelpCorpus` merges this on top
  of the installed sidecar; overrides win. Add entries here to fix retrieval gaps
  without touching the `attune-help` package.

- **`DirectoryCorpus(extra_summaries=...)`** — new parameter for programmatic
  summary overrides.

- **Dashboard CLI** (`attune-rag dashboard show|render`) — terminal dashboard with
  retrieval metrics, corpus health, feature coverage, and per-difficulty breakdown.

### Changed

- **`tasks/` category weight 1.5 → 1.2** — prevents task templates from
  outranking concept docs on feature-name queries due to higher keyword density.

- **`rich>=13.0`** added as a core dependency (required by the dashboard).

### Retrieval metrics (attune-help corpus, 40 golden queries)

| Mode | P@1 | R@3 |
|---|---|---|
| v0.1.5 baseline | 47.5% | 70.0% |
| v0.1.6 keyword-only | **87.5%** | **95.0%** |
| v0.1.6 + LLMReranker | ≈88–90% | ≈95–97% |

## [0.1.5] - 2026-04-19

### Added (security)

- **`<passage>...</passage>` sentinel wrapping around every
  retrieved passage** produced by `join_context` and
  `join_context_numbered`. Each prompt variant gained a
  clear **injection-defense clause** telling the model
  that content inside `<passage>` tags is documentation
  data, never instructions — even when a passage body
  contains adversarial text like "Ignore prior
  instructions" or a literal `</passage>` attempt to
  break out of the wrapping.
  - Addresses a class of risk orthogonal to the v0.1.4
    citation-forced faithfulness gains: **claim
    hallucination** (the model making up facts) and
    **prompt injection from retrieved content** (the
    model executing instructions embedded in the corpus)
    are separate threat models and need separate
    mitigations. v0.1.4 addressed the first; 0.1.5
    addresses the second.
  - The inner passage format is intentionally unchanged
    from v0.1.4 — each passage still begins with its
    pre-sentinel `[source: <path>]` (baseline) or
    `[P{n}] source: <path>` (citation-numbered) header
    line. Experimenting with an XML-attribute form
    (`<passage id="P1" source="path">` with `id` as the
    citation anchor) regressed citation faithfulness
    from 1.00 to 0.97 in an A/B sweep; the header-inside-
    sentinel hybrid recovered it to 0.99.

### Changed (A/B verification)

A/B sweep confirmed the new sentinel format keeps all four
prompt variants well above the pre-committed 0.85
faithfulness gate:

| variant | v0.1.4 | v0.1.5 | Δ faith | Δ hallu rate |
|---|---|---|---|---|
| baseline | 0.94 / 47% | 0.94 / 40% | flat | −7pp |
| strict | 0.97 / 27% | 0.96 / 33% | −0.01 | +6pp |
| **citation** | **1.00 / 6.7%** | **0.99 / 13.3%** | **−0.01** | **+6.7pp** |
| anti_prior | 0.95 / 33% | 0.98 / 20% | +0.03 | −13pp |

Citation remains the top variant and the pinned default.
The +6.7pp bucket-level hallucination rate on citation
corresponds to 2 extra queries with a single unsupported
claim each (per-claim hallucination went from ~0.5% to
~3.2%) — a real but small regression traded for real
injection defense.

## [0.1.4] - 2026-04-19

### Changed (reliability + robustness)

- **`FaithfulnessJudge` now sets a 60s request timeout** on
  the default `AsyncAnthropic` client. A stalled network
  no longer hangs the benchmark loop indefinitely. Override
  via the new ``timeout=`` constructor kwarg.
- **Per-query error handling in `bench_prompts`.** A single
  transient failure (rate-limit 429, network blip,
  malformed fixture entry) now records an `error` on the
  `_QueryRun` and lets the sweep continue to the next
  query. Errored runs are excluded from the means and
  surfaced in a new `err` column in the printed table.
  Before: one bad query killed the full ~10 min run with
  no partial results.
- **Tool-use payload is schema-validated** before the
  judge computes a score. Non-list claim arrays or
  non-string reasoning now raise a clear `RuntimeError`
  naming the offending field rather than a cryptic
  `TypeError` from `len()`. Non-string items inside the
  claim lists are coerced to strings (occasional model
  quirk) rather than failing.
- **`FaithfulnessJudge.__init__` rejects ambiguous
  config** — passing both `client=` and `api_key=` now
  raises `ValueError` instead of silently ignoring one.

### Changed (CLI robustness in `python -m attune_rag.benchmark`
and `python -m attune_rag.eval.bench_prompts`)

- **Path validation on `--queries` and `--output`**: null
  bytes rejected, resolution errors reported clearly,
  system directories (`/etc`, `/sys`, `/proc`, `/dev`,
  `/bin`, `/sbin`, `/boot`) refused for writes on both
  Linux and macOS (handles `/etc` → `/private/etc`
  symlink).
- **Clear error when the default golden fixture is
  missing** — the `tests/golden/` directory is not
  shipped in the wheel, so `pip install attune-rag` users
  who run the benchmark with no `--queries` now get a
  message explaining they must pass their own fixture,
  not a cryptic `FileNotFoundError`.

### Added

- `test_eval_bench_prompts.py` — 21 new tests covering
  `_aggregate` metric math (including the new
  errored-run exclusion), `main()` argument + env
  validation (missing key, bad path, unknown variant,
  null byte, system-directory output), and the new
  `_validate_read_path` / `_validate_write_path` guards.
- 5 new tests in `test_eval_faithfulness.py` for
  constructor validation, the missing `[claude]` extra
  path, and tool-use payload schema violations.

### Notes

Every change in this release was surfaced by a
three-pass deep review (security + quality + test gaps)
on the v0.1.3 eval module. No behavioral regressions;
all public APIs unchanged.

## [0.1.3] - 2026-04-19

### Added

- **`attune_rag.eval.FaithfulnessJudge`** — LLM-as-judge
  scorer for answer grounding. Uses Anthropic forced
  tool-use for guaranteed-schema JSON output. Decomposes
  each answer into atomic factual claims and scores
  supported / unsupported against the retrieved passages.
  Returns a 0-1 score plus per-claim verdicts. Requires
  the `[claude]` extra.
- **Three prompt variants** in `attune_rag.prompts`
  alongside `baseline`: `strict`, `citation`,
  `anti_prior`. Registry exposed as
  `attune_rag.PROMPT_VARIANTS`. Select via
  `RagPipeline.run(..., prompt_variant="...")`.
- **A/B benchmark runner** at
  `attune_rag.eval.bench_prompts`. Sweeps every variant
  × the golden query set through retrieval + generation
  + judging, prints a comparison table with P@1, R@k,
  faithfulness, refusal rate, hallucination rate.
- **`--with-faithfulness` / `--min-faithfulness` flags**
  on `python -m attune_rag.benchmark`. Opt-in
  faithfulness pass that runs the default variant
  through the judge and gates CI on mean faithfulness.
  Default off; retrieval-only mode stays offline and
  free.
- **`RagResult.context`** — the exact passage block
  passed to the generator, preserved on the result so
  downstream evaluators can score against identical
  text (not the 200-char excerpts in
  `CitedSource.excerpt`).

### Changed

- **Default prompt variant is now `citation`.** Selected
  by A/B sweep on 2026-04-19 (decision doc:
  [faithfulness-decision-2026-04-19.md](https://github.com/Smart-AI-Memory/attune-ai/blob/main/docs/rag/faithfulness-decision-2026-04-19.md)).
  Hallucination rate on the golden set drops **46.7% →
  6.7%** vs baseline while retrieval quality stays
  identical. Mean faithfulness reaches **1.00**. Opt
  back in to the old template with
  `run(..., prompt_variant="baseline")` if your use
  case needs prose-style answers without inline
  citations.

### Faithfulness quality

| variant | P@1 | R@3 | faith | hallu. |
|---|---|---|---|---|
| baseline | 73.3% | 86.7% | 0.94 | 46.7% |
| strict | 73.3% | 86.7% | 0.97 | 26.7% |
| **citation** | 73.3% | 86.7% | **1.00** | **6.7%** |
| anti_prior | 73.3% | 86.7% | 0.95 | 33.3% |

## [0.1.2] - 2026-04-18

### Changed

- **`AttuneHelpCorpus` now loads the path-keyed summary
  sidecar shipped in attune-help 0.7.0+.** Previously the
  adapter passed no sidecar to `DirectoryCorpus` because
  attune-help's legacy `summaries.json` was feature-keyed
  and silently ignored. The 0.7.0 release adds
  `summaries_by_path.json` which attune-rag now reads by
  default. Effect: summary signal reaches the retriever
  for the first time, materially improving retrieval
  quality for attune-help corpora.
- **`attune-help` dep floor bumped** from `>=0.5.1,<0.6`
  to `>=0.7.0,<0.8`. Users with older attune-help will
  need to upgrade.

### Retrieval quality

See the post-upgrade benchmark in
[attune-ai/docs/rag/embeddings-decision-2026-04-17.md](https://github.com/Smart-AI-Memory/attune-ai/blob/main/docs/rag/embeddings-decision-2026-04-17.md)
for updated Precision@1 / Recall@3 numbers on the 15-query
golden set.

## [0.1.1] - 2026-04-18

### Changed

- `KeywordRetriever` gains three tuning mechanisms:
  - **Category weights.** `entry.category` multiplies the
    base score. Primary material (`concepts/`, `quickstarts/`,
    `tasks/`) boosted 1.5x; lesson-style material (`errors/`,
    `warnings/`, `faqs/`) penalized 0.4x. Overridable via
    `KeywordRetriever.CATEGORY_WEIGHTS`.
  - **Path-hit cap.** Long keyword-dense filenames common in
    `errors/` no longer dominate scoring. Hits against the
    path are capped at `PATH_HIT_CAP` (default 3).
  - **Suffix stemmer.** Token equivalence for common English
    suffixes (`-ing`, `-ion`, `-ate`, `-ator`, `-ed`, `-s`,
    etc.). Covers query/target pairs like "bugs"/"bug" and
    "orchestrator"/"orchestrate".
- `match_reason` now includes a `cat:<category>×<weight>`
  marker when the category weight is non-unity.

### Retrieval quality (attune-help 0.5.1 corpus, 15 golden
queries)

| Metric | 0.1.0 | 0.1.1 | Delta |
|---|---|---|---|
| Precision@1 | 53.33% | **66.67%** | **+13.34 pts** |
| Recall@3 | 60.00% | **73.33%** | **+13.33 pts** |
| Easy P@1 | 4/5 | 5/5 | +1 |
| Hard P@1 | 0/6 | 1/6 | +1 |
| Mean latency | 11.83 ms | 42.45 ms | +30.6 ms |

The 70% P@1 gate committed in
[docs/rag/embeddings-decision-2026-04-17.md](https://github.com/Smart-AI-Memory/attune-ai/blob/main/docs/rag/embeddings-decision-2026-04-17.md)
is not met. Remaining misses are queries with zero token
overlap against their targets (e.g. "vulnerability scan" →
`concepts/tool-security-audit.md`) — the semantic gaps the
decision doc anticipated. Routes next work to v0.2.0
`attune-rag[embeddings]` using `fastembed`.

## [0.1.0] - 2026-04-17

### Added

- Initial package scaffold (task 1.1 of the attune-ai RAG
  grounding spec, v4.0): pyproject.toml, public API
  surface, README with multi-LLM quickstarts, LICENSE.
- Optional extras: `[attune-help]`, `[claude]`, `[openai]`,
  `[gemini]`, `[all]`, `[dev]`. Core install has zero LLM
  SDK deps.
- Core modules: `pipeline`, `retrieval`, `corpus`,
  `provenance`, `prompts`, `providers` (Claude, OpenAI,
  Gemini adapters).
- Benchmark harness (`python -m attune_rag.benchmark`) with
  15-query golden set in `tests/golden/queries.yaml`.
