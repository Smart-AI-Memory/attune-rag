# attune-rag

Lightweight, LLM-agnostic RAG pipeline with pluggable
corpora. Works with Claude, Gemini, or any LLM.

- **No LLM SDK at install time.** All provider deps are
  optional extras. Two required runtime deps: `structlog`,
  `jinja2`.
- **Pluggable corpus.** Use attune-help (the default), any
  markdown directory, or your own `CorpusProtocol`.
- **Returns a prompt string + citation records** by default
  — `pipeline.run()` never opens a network connection. You
  call your own LLM however you like. Optional provider
  adapters ship convenience wrappers.
- **Optional hybrid retrieval.** `QueryExpander` and
  `LLMReranker` layer Claude Haiku on top of keyword
  retrieval to improve recall and precision — both opt-in,
  both fail-safe.

## Why attune-rag

Most RAG libraries ship features. attune-rag ships **measured
quality numbers** and gates merges against them. The CI badge
isn't "tests pass" — it's `P@1 ≥ 0.95, R@3 = 1.00, mean
faithfulness ≥ 0.9686` (locked at
[`docs/specs/release-quality-baseline/baseline-1.md`](docs/specs/release-quality-baseline/baseline-1.md))
plus per-axis CPU + wall-clock perf thresholds (locked at
[`docs/specs/downstream-validation/perf-baseline.md`](docs/specs/downstream-validation/perf-baseline.md)).

A PR that drops `mean_faithfulness` below `0.9686` fails CI
automatically. Same for any latency hot-path regressing past
`mean + 2σ`. That's the differentiator.

### vs LangChain / LlamaIndex

| | attune-rag | LangChain | LlamaIndex |
|---|---|---|---|
| Required runtime deps | 2 | many (transitively, ~30+) | many (~25+) |
| LLM SDK at install | none | bundled | bundled |
| Published quality regression thresholds | yes (P@1, R@3, faithfulness) | no | no |
| Published perf thresholds (wall + CPU) | yes | no | no |
| Citation primitives built-in | yes | add-on | add-on |
| "Get a string back, call your own LLM" | default | possible w/ effort | possible w/ effort |

LangChain and LlamaIndex are fantastic frameworks if you want
batteries-included orchestration. attune-rag is the alternative
when you want a RAG component you can drop into an existing app
without buying into a framework — and want the quality bar
quantified, not implied.

### What attune-rag is **not**

Honest exclusions, so you can self-disqualify if you need any
of these:

- **Not an agent framework.** No multi-step chains, no tool-use
  orchestration, no agent loops.
- **Not a document-parsing toolkit.** Bring your markdown
  already-parsed; use `unstructured.io` or similar upstream.
- **Not a vector DB integration.** Keyword retrieval is the
  default; you wire your own vector store if you need one (an
  `EmbeddingRetriever` is on the post-freeze roadmap — see
  below).
- **Not a one-line-install batteries-included framework.** That's
  LangChain / LlamaIndex. attune-rag is for the case where that's
  too much.

## Install

```bash
pip install attune-rag                     # core only
pip install 'attune-rag[attune-help]'      # + bundled help corpus
pip install 'attune-rag[claude]'           # + Claude adapter
pip install 'attune-rag[gemini]'           # + Gemini adapter
pip install 'attune-rag[all]'              # everything
```

## Quick start — Claude

```bash
pip install 'attune-rag[attune-help,claude]'
```

```python
import asyncio
from attune_rag import RagPipeline

async def main():
    pipeline = RagPipeline()  # defaults to AttuneHelpCorpus
    response, result = await pipeline.run_and_generate(
        "How do I run a security audit with attune?",
        provider="claude",
    )
    print(response)
    print("\nSources:", [h.entry.path for h in result.citation.hits])

asyncio.run(main())
```

## Quick start — Gemini

```bash
pip install 'attune-rag[attune-help,gemini]'
```

```python
response, result = await pipeline.run_and_generate(
    "...", provider="gemini", model="gemini-1.5-pro",
)
```

## Quick start — custom corpus, any LLM

```python
from pathlib import Path
from attune_rag import RagPipeline, DirectoryCorpus

pipeline = RagPipeline(corpus=DirectoryCorpus(Path("./my-docs")))
result = pipeline.run("How do I...?")

# Send result.augmented_prompt to whatever LLM you use.
# The pipeline itself does NOT call an LLM unless you use
# run_and_generate or call a provider adapter yourself.
```

## Hybrid retrieval (optional)

`QueryExpander` and `LLMReranker` require the `[claude]` extra and an
`ANTHROPIC_API_KEY`. Both are opt-in and fail-safe — any API error
falls back to keyword-only order automatically.

```python
from attune_rag import RagPipeline, LLMReranker, QueryExpander

# Reranker only (recommended for precision):
pipeline = RagPipeline(reranker=LLMReranker())

# Expander + reranker (max coverage):
pipeline = RagPipeline(
    expander=QueryExpander(),
    reranker=LLMReranker(),
)
```

## Template editor primitives (`attune_rag.editor`)

Headless toolkit for tools that need to validate, lint, and refactor a
template corpus — used by the [`attune-gui`](https://pypi.org/project/attune-gui/)
template editor and the [`attune-author`](https://pypi.org/project/attune-author/)
`edit` CLI, but works standalone with any
[`CorpusProtocol`](https://github.com/Smart-AI-Memory/attune-rag).

| API | What it does |
|-----|---------------|
| `load_schema()` | Loads `template_schema.json` (the v1 frontmatter contract: required `type` enum + `name`; optional `tags`, `aliases`, `summary`, `source`, `hash`; `additionalProperties: true`). |
| `parse_frontmatter(text)` / `validate_frontmatter(data)` | Split a template into frontmatter + body and report typed `FrontmatterIssue`s — used by linters and editors. |
| `lint_template(text, rel_path, corpus)` | Returns `Diagnostic[]` for schema violations, broken `[[alias]]` references, and depth-marker sequence errors. 1-indexed line/col ranges. |
| `autocomplete_tags(corpus, prefix, limit)` / `autocomplete_aliases(corpus, prefix, limit)` | Prefix-match completions ranked by frequency (tags) or lexical proximity (aliases). Sub-ms on 1k templates. |
| `find_references(corpus, name, kind)` | Locate every alias/tag/path occurrence across body, frontmatter, and `cross_links.json`. |
| `plan_rename(corpus, old, new, kind)` | Build a `RenamePlan` (one `FileEdit` per affected file with unified-diff hunks) for `kind="alias"` or `"tag"`. Raises `RenameCollisionError` on existing alias targets. |
| `apply_rename(corpus, plan)` | Atomically apply the plan (tempfile-per-file + sequential rename + drift-detection rollback). Returns the list of affected paths. |

Schema, lint, and rename are pure functions over `CorpusProtocol` — no I/O,
no global state. All three pieces are tested as a unit and used live by the
attune-gui editor's `/api/corpus/<id>/lint`, `/autocomplete`, and
`/refactor/rename/{preview,apply}` routes.

```python
from attune_rag import DirectoryCorpus
from attune_rag.editor import lint_template, plan_rename, apply_rename

corpus = DirectoryCorpus(Path("./templates")).load()

# Validate a template before saving
diagnostics = lint_template(
    text=Path("./templates/concepts/foo.md").read_text(),
    rel_path="concepts/foo.md",
    corpus=corpus,
)

# Rename an alias across the whole corpus
plan = plan_rename(corpus, old="oldname", new="newname", kind="alias")
print(f"Affects {len(plan.edits)} files")
affected = apply_rename(corpus, plan)
```

## Dashboard

```bash
attune-rag dashboard show    # live terminal dashboard
attune-rag dashboard render --out report.html  # HTML snapshot
```

## Quality baselines

attune-rag locks two baselines, both gated by CI. Thresholds
are empirically derived (`mean ± 2σ`) from back-to-back
benchmark runs on an unchanged HEAD — grounded, not guessed.

### Retrieval + faithfulness

| Metric | Threshold (current) | Source |
|---|---:|---|
| `precision_at_1` | **≥ 0.95** | retrieval, deterministic |
| `recall_at_3` | **= 1.00** | retrieval, deterministic |
| `mean_faithfulness` | **≥ 0.9686** | Claude judge, σ ≈ 0.005 |

Gated by [`.github/workflows/benchmark.yml`](.github/workflows/benchmark.yml).
Faithfulness gating engages when the PR touches retrieval,
reranker, expander, pipeline, prompts, or eval paths, or when
the PR title contains `[full-bench]`. Methodology + raw numbers
in [`docs/specs/release-quality-baseline/`](docs/specs/release-quality-baseline/baseline-1.md).

### Per-hot-path latency

Locked dual-axis (wall-clock + CPU-time) thresholds on the four
benchmarks. CPU-time is the gating axis (deterministic);
wall-clock is advisory through Phase 4 burn-in, then revisited.

| Benchmark | Axis | Mean (µs) | Threshold (µs) |
|---|---|---:|---:|
| `keyword_retriever_retrieve` | cpu | 730 | 7,783 |
| `directory_corpus_load` | cpu | 47 | 62 |
| `rag_pipeline_run` (retrieval-only) | cpu | 270 | 307 |
| `llm_reranker_rerank` | wall | _re-lock pending_ | _re-lock pending_ |

Numbers measured from N=30 back-to-back runs on the
locked-baseline runner (Linux `ubuntu-latest`, CPython 3.11.15).
The `keyword_retriever_retrieve` threshold is wide because its
measured σ (~3.5 ms) reflects cold-cache effects on the first
few iterations — the threshold formula is `mean + 2σ`, which
is empirically derived rather than tuned for tightness.

Gated by [`.github/workflows/perf.yml`](.github/workflows/perf.yml)
per-PR (advisory comment in Phase 4 W1–W2, blocking in W3.1).
Reranker numbers ship once `ANTHROPIC_API_KEY` is configured
in repo Secrets and a full lock is run with `include_llm=true`.
Raw numbers + hardware fingerprint:
[`docs/specs/downstream-validation/perf-baseline.md`](docs/specs/downstream-validation/perf-baseline.md).

### Why this is the differentiator

Most RAG libraries A/B-test internally and ship the result.
attune-rag publishes the thresholds, gates merges against them,
and re-measures whenever the corpus, judge prompt, or hardware
changes. The receipts are checked in.

## Bundled `.help/` corpus

The repo ships a polished `.help/` corpus that documents
attune-rag's own surface — 143 templates across 13 features ×
11 kinds (`concept`, `task`, `reference`, `quickstart`, `faq`,
`error`, `warning`, `tip`, `note`, `comparison`,
`troubleshooting`). Generated by
[`attune-author`](https://pypi.org/project/attune-author/) with
strict fact-check; queryable via `AttuneHelpCorpus` or as the
bundled default for `RagPipeline()`. See
[`.help/features.yaml`](.help/features.yaml) for the feature
map and [`.help/templates/`](.help/templates/) for the content.

The 13 features: `pipeline`, `retrieval`, `corpus`, `prompts`,
`provenance`, `providers`, `eval`, `benchmark`, `cli`, `editor`,
`dashboard`, `expander`, `reranker`.

### What faithfulness measures

Faithfulness scores how well an answer is **grounded in the retrieved
passages** — `1.0` means every claim in the answer is supported by a
cited source; lower scores mean some claims have no support in the
context. It catches hallucination in a way that `precision_at_k` and
`recall_at_k` can't: those only measure whether the *right documents*
were retrieved, not whether the *generated answer* actually used them.

attune-rag uses **Claude as the judge** via Anthropic's tool-use API
to produce a structured score in `[0.0, 1.0]` for each
`(query, answer, retrieved_context)` triple. The reported metric is
the mean over the golden query set. Aggregate σ ≈ `0.005` over 40
queries even though per-query judge non-determinism can swing 40+
percentage points on individual queries — averaging absorbs the noise.

### Run faithfulness manually

```bash
pip install 'attune-rag[claude]'
export ANTHROPIC_API_KEY=sk-ant-...

# Retrieval metrics only (free, deterministic):
attune-rag-benchmark --queries queries.yaml --json out.json

# Add faithfulness (~1 Claude API call per query, costs tokens):
attune-rag-benchmark --queries queries.yaml --with-faithfulness --json out.json

# Compare extended-thinking on vs off (2× judge cost):
attune-rag-benchmark --queries queries.yaml --with-faithfulness --compare-thinking --json out.json
```

The judge implementation lives at
`attune_rag.eval.faithfulness.FaithfulnessJudge`. Note: `attune_rag.eval.*`
is currently INTERNAL and may move — the `attune-rag-benchmark
--with-faithfulness` CLI is the stable contract.

For the methodology behind the `0.9686` threshold, the v1/v2 ground-truth
calibration runs, and the extended-thinking-vs-default decision record, see
[`docs/rag/faithfulness-thinking-calibration.md`](https://github.com/Smart-AI-Memory/attune-rag/blob/main/docs/rag/faithfulness-thinking-calibration.md).

## Roadmap — embeddings (post-freeze 0.2.0+)

Keyword retrieval + optional Claude reranker currently meet
the locked `P@1 ≥ 0.95, R@3 = 1.00` thresholds against the
attune-help golden set. The remaining hard queries
(3 of 28, currently `xpass`-gated under `[no-embeddings]`)
have zero token overlap against their target doc (e.g.
"vulnerability scan" → `tool-security-audit.md`). Closing
that gap needs vector search.

The plan is to ship `attune-rag[embeddings]` using
[`fastembed`](https://github.com/qdrant/fastembed) for local,
CPU-only embeddings — no new network dependency, no API key
required at retrieval time. Keyword retrieval stays the default;
embeddings layer in opt-in, same shape as `QueryExpander` and
`LLMReranker`. Shipping is paced by the
[Phase 4 feature-freeze](docs/specs/downstream-validation/)
currently in progress — public surface additions wait for the
0.2.0 cut.

See
[CHANGELOG.md](https://github.com/Smart-AI-Memory/attune-rag/blob/main/CHANGELOG.md)
for the decision record and remaining-gap analysis.

## Prompt caching (Claude only)

When using the Claude provider, `run_and_generate` automatically enables
[Anthropic prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)
on the stable RAG context prefix (≥ 1 024 chars). This eliminates
repeated token costs on the corpus portion of the prompt when the same
context block is reused across calls.

No configuration needed — the provider handles the `cache_control`
header automatically.

## Public API

attune-rag's public surface is documented below and snapshot-tested
in [tests/unit/test_api_surface.py](tests/unit/test_api_surface.py).
Formal SemVer commitments begin with the 0.2.0 release — see
[docs/POLICY.md](docs/POLICY.md) for the deprecation policy. Until
then the surface is honor-system: the lock test catches drift, but
treat 0.1.x as still-evolving.

**Top-level (`from attune_rag import ...`):**

- Pipeline — `RagPipeline`, `RagResult`
- Corpus — `CorpusProtocol`, `RetrievalEntry`, `DirectoryCorpus`,
  `AttuneHelpCorpus`
- Retrieval — `KeywordRetriever`, `RetrievalHit`, `RetrieverProtocol`
- Provenance — `CitationRecord`, `CitedSource`, `ClaimCitation`,
  `format_citations_markdown`, `format_claim_citations_markdown`
- Prompting — `build_augmented_prompt`, `PROMPT_VARIANTS`
- Hybrid retrieval — `QueryExpander`, `LLMReranker`

**PUBLIC submodules** (importable by qualified path):

- `attune_rag.corpus` — exposes `AliasInfo`, `DuplicateAliasError`
  in addition to the top-level corpus names
- `attune_rag.corpus.attune_help` — `AttuneHelpCorpus`
- `attune_rag.corpus.help_adapter` — `HelpCorpusAdapter` Protocol
- `attune_rag.providers` — `LLMProvider`, `get_provider`,
  `list_available`
- `attune_rag.editor` — template-editor primitives (lint, schema,
  rename, autocomplete, references); see "Template editor primitives"
  above for the symbol list
- `attune_rag.editor.{rename,schema,lint,autocomplete,references}` —
  the individual editor submodules

Anything not listed above is INTERNAL and may change in any release.
The underscore-prefixed editor modules (`attune_rag.editor._rename`
etc.) shipped in 0.1.x are deprecation shims as of 0.2.0; they
re-export the new non-underscore names and emit `DeprecationWarning`.
They are removed in 0.3.0.

## Status

v0.1.19 (alpha). In
[Phase 4 of the v1.0 roadmap](docs/specs/ROADMAP-v1.md) — a
four-week feature freeze + downstream validation soak; formal
`0.2.0` SemVer cut follows. Until then the public surface is
honor-system + lock-tested; deprecation policy at
[`docs/POLICY.md`](docs/POLICY.md).

Part of the attune ecosystem
([attune-ai](https://github.com/Smart-AI-Memory/attune-ai),
[attune-help](https://github.com/Smart-AI-Memory/attune-help),
[attune-author](https://github.com/Smart-AI-Memory/attune-author),
[attune-gui](https://github.com/Smart-AI-Memory/attune-gui)).

## License

Apache 2.0. See
[LICENSE](https://github.com/Smart-AI-Memory/attune-rag/blob/main/LICENSE).
