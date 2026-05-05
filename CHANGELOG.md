# Changelog

All notable changes to `attune-rag` are documented here.
Format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
