# Changelog

All notable changes to `attune-rag` are documented here.
Format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
