# attune-rag

Lightweight, LLM-agnostic RAG pipeline with pluggable
corpora. Works with Claude, Gemini, or any LLM.

🌐 **Docs & guides: [attune-rag.dev](https://attune-rag.dev)**

## Proven retrieval — the numbers

Every figure is reproducible with `attune-rag-benchmark`. The bundled-corpus
row is a **hard CI gate** — a PR that regresses it fails automatically.

| | Bundled corpus¹ | Unseen corpus, overall² | Unseen corpus, **hard paraphrases**³ |
|---|---|---|---|
| **precision@1** | **100%** | 73% (lightweight, zero torch) | 25% lightweight → **90%** transformer tier |
| **recall@3** | **100%** | 82% (lightweight, zero torch) | 25% lightweight → **100%** transformer tier |
| **faithfulness** | **0.97** mean | — | — |

Plus: configurable **abstention** drops out-of-corpus false answers from
**92% → 8%** — so the retriever stays quiet instead of confidently wrong.

The headline: on a corpus attune-rag has **never seen or been tuned on**,
asking questions worded *nothing* like the docs, the transformer tier goes
from a keyword baseline of **1-in-4** to **9-in-10** top-1 correct, and
**finds the right doc in the top 3 every single time** (recall@3 100%).

<sub>¹ bundled attune-help corpus, gated in CI at `P@1 ≥ 0.95 / R@3 = 1.00 /
faithfulness ≥ 0.9686`; actuals shown.  ² lightweight keyword on an unseen
corpus (`corpus_b`), no embeddings.  ³ pure-paraphrase stress test
(`corpus_c`, queries with almost no vocabulary overlap with the docs) —
measured in `docs/specs/transformer-retriever/`.</sub>

## Two ways to run it

- 🪶 **Lightweight (default).** Keyword retrieval, optionally fused with
  **torch-free** static embeddings (`pip install attune-rag[embeddings]`).
  5 pure-Python deps, no LLM SDK, no torch, fully offline, ~1 ms/query —
  the dependency-light path that holds **100% / 100%** on a tuned corpus
  and **73% precision@1** on an unseen one.
- 🤖 **Transformer tier (opt-in).** Real sentence-transformers embeddings
  (`pip install attune-rag[transformers]`) for paraphrase-heavy or arbitrary
  corpora. On pure-paraphrase queries it lifts precision@1 **25% → 90%** and
  recall@3 **25% → 100%** — the generalization no torch-free retriever
  reaches. Heavyweight (pulls torch, ~GB); embedding-primary; never a default.

- **No LLM SDK at install time; footprint scales with your setup.**
  The base install pulls **5 small pure-Python deps** (`structlog`,
  `jinja2`, `pyyaml`, `rich`, `jsonschema`) — no LLM SDK, no torch.
  Retrieval tiers add only what they need: `[embeddings]` adds
  torch-free `model2vec`; `[transformers]` adds the torch stack. You
  pay for exactly the setup you choose.
- **Pluggable corpus.** Use attune-help (the default), any
  markdown directory, or your own `CorpusProtocol`.
- **Returns a prompt string + citation records** by default
  — `pipeline.run()` never opens a network connection. You
  call your own LLM however you like. Optional provider
  adapters ship convenience wrappers.
- **Opt-in retrieval ladder.** Keyword retrieval by default;
  add a torch-free static-embedding `HybridRetriever`
  (`[embeddings]`) or a `TransformerRetriever` (`[transformers]`)
  for paraphrase-heavy corpora, plus configurable **abstention**
  (`min_score=`) to suppress confident out-of-corpus answers.
  Every rung is opt-in and fail-safe; the keyword default is
  unchanged.

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
| Required runtime deps | 5 (pure-Python) | many (transitively, ~30+) | many (~25+) |
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

Beyond drop-in retrieval, attune-rag is the grounding foundation
for the `attune-*` family's content-quality discipline. The
`attune-author` polish/fact-check pipeline uses attune-rag's
retrieval + faithfulness primitives to verify generated help
content is grounded in source material before it's marked
authoritative — the same `mean_faithfulness ≥ 0.9686` discipline
that gates this library's own benchmarks, extended to the
authoring loop.

### What attune-rag is **not**

Honest exclusions, so you can self-disqualify if you need any
of these:

- **Not an agent framework.** No multi-step chains, no tool-use
  orchestration, no agent loops.
- **Not a document-parsing toolkit.** Bring your markdown
  already-parsed; use `unstructured.io` or similar upstream.
- **Not a vector DB integration.** Keyword retrieval is the
  default; the optional `[embeddings]` / `[transformers]` tiers
  embed in-process (model2vec / sentence-transformers). There is
  no external vector-store integration — you wire your own if
  you need one.
- **Not a one-line-install batteries-included framework.** That's
  LangChain / LlamaIndex. attune-rag is for the case where that's
  too much.

## Install

```bash
pip install attune-rag                      # core — keyword retrieval, 5 pure-Python deps, no LLM SDK
# Retrieval tiers (opt-in):
pip install 'attune-rag[embeddings]'        # + torch-free static hybrid retrieval
pip install 'attune-rag[transformers]'      # + transformer retrieval tier (pulls torch, ~GB)
# Corpus & LLM adapters (opt-in):
pip install 'attune-rag[attune-help]'       # + bundled help corpus
pip install 'attune-rag[claude]'            # + Claude adapter
pip install 'attune-rag[gemini]'            # + Gemini adapter
# Convenience:
pip install 'attune-rag[all]'               # every extra, incl. the transformer tier (pulls torch, ~GB)
```

Extras compose, e.g. `pip install 'attune-rag[embeddings,claude]'`. The base
install stays dependency-light on purpose; only `[transformers]` (and
therefore `[all]`) pulls torch.

## Authentication

The faithfulness judge is subscription-first: inside a Claude Code
session (`CLAUDECODE=1`) with the `[claude]` extra installed, judge
calls route through your Claude subscription via the Agent SDK — no
`ANTHROPIC_API_KEY` needed. From a plain terminal (or in CI) it uses
`ANTHROPIC_API_KEY` as before. Override with
`FaithfulnessJudge(auth_mode="api"|"sub")`, `ATTUNE_RAG_AUTH_MODE`,
or `attune-rag-benchmark --auth-mode`. Note: RAG answer *generation*
(`ClaudeProvider`) is API-key-only — the subscription route covers
the judge.

## Prompt-cache TTL

`ClaudeProvider` marks the stable prompt prefix (and the first citation
document block) with Anthropic prompt caching. By default the cache window
is the 5-minute `ephemeral` tier. Set `ATTUNE_RAG_CACHE_TTL=1h` to extend it
to one hour — at the **same per-token price** — for workloads that issue
clusters of related queries within the hour (dashboards, benchmark sweeps):

```bash
export ATTUNE_RAG_CACHE_TTL=1h   # default: 5m
```

Leave it unset for one-off queries: the cache rarely survives long enough to
pay off, and the default wire shape is byte-identical to prior behavior.

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

> 📖 **Building a quality corpus.** See [`docs/USER_CORPUS_GUIDE.md`](docs/USER_CORPUS_GUIDE.md)
> for the corpus-authoring discipline that produced the bundled
> attune-help corpus's 100% / 100% baseline + 100% paraphrased R@3:
> frontmatter aliases, multi-token intent, the `MIN_ALIAS_OVERLAP`
> knob, stemmer traps, the override file pattern, and the
> strict-dominance measurement loop. The guide is the v0 forerunner
> of the v1.0.0 framework framing (`user-corpus-onboarding` spec).

## CLI

Everything above is also reachable from the terminal — your own
corpus, the retrieval tiers, and abstention included:

```bash
attune-rag query "how do I run a security audit?"   # bundled corpus
attune-rag query "..." --corpus-path ./my-docs      # your markdown corpus
attune-rag query "..." --retriever hybrid           # [embeddings] tier
attune-rag query "..." --retriever transformer      # [transformers] tier
attune-rag query "..." --min-score 5                # abstain below threshold
attune-rag query "..." --prompt-variant strict      # prompt template
attune-rag query "..." --provider claude            # full RAG + LLM call
attune-rag query "..." --json                       # hits as JSON
attune-rag corpus-info --corpus-path ./my-docs      # corpus stats
attune-rag providers                                # installed LLM extras
```

The bundled default corpus requires the `[attune-help]` extra; on a
bare `pip install attune-rag`, pass `--corpus-path`.

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

### Embedding / hybrid retrieval (`[embeddings]` extra)

`HybridRetriever` fuses the keyword retriever with static **model2vec**
embeddings via Reciprocal Rank Fusion. No torch, no GPU, no API key —
offline, millisecond encode. Install: `pip install attune-rag[embeddings]`.

```python
from attune_rag import RagPipeline, HybridRetriever

# Opt-in. KeywordRetriever stays the default.
pipeline = RagPipeline(retriever=HybridRetriever())
```

**When to use it:** on an **unstructured / arbitrary corpus** (raw markdown
with no curated summaries or aliases), embeddings recover the paraphrase
recall that token overlap misses — measured **+9pts recall@3** on an unseen
corpus. On a corpus that's already **keyword-tuned** (curated
summaries/aliases, like the bundled `.help/`), an equal blend trades away
top-1 precision, so the default weighting favors keyword (`keyword_weight=2.0`);
raise it to fully protect a tuned corpus, lower toward `1.0` to maximize the
embedding contribution. Falls back to keyword-only if the extra isn't
installed.

### Transformer retrieval (`[transformers]` extra) — heavyweight

`TransformerRetriever` ranks by a real **sentence-transformers** model
(default `BAAI/bge-small-en-v1.5`). This is the **heaviest** rung of the
opt-in ladder (`keyword` → `[embeddings]` static → `[transformers]`):
it pulls **torch (~GB)** and downloads a model once (then offline,
~10–300 ms/query). Install: `pip install attune-rag[transformers]`.

```python
from attune_rag import RagPipeline, TransformerRetriever

# Heavyweight opt-in — embedding-primary, for arbitrary corpora.
pipeline = RagPipeline(retriever=TransformerRetriever())
```

**When to use it:** only on an **arbitrary corpus where paraphrasing is
heavy** and the torch-free static embeddings fall short. It is
embedding-primary and **tanks a keyword-tuned corpus's top-1 precision**,
so it is never a default. Measured on two unseen corpora
(`docs/specs/transformer-retriever/`): hard-tier paraphrase **precision@1
≈0.50 (the torch-free ceiling) → 0.85–0.90, recall@3 → 1.00** — the one
goal no torch-free retriever reaches. For a keyword-tuned corpus use
`KeywordRetriever`; for a lexically-aligned arbitrary corpus the lighter
`[embeddings]` `HybridRetriever` is usually enough.

Pass `query_prefix=""` for symmetric models (e.g. `all-MiniLM-L6-v2`); the
default prefix is tuned for BGE's asymmetric query instruction.

Measure the lift on **your** corpus before paying the torch install:

```bash
attune-rag-measure --corpus-path ./my-docs --queries ./queries.yaml \
    --retriever transformer
```

### Abstention — don't answer out-of-corpus queries

By default the retriever returns its best match even for a question the
corpus can't answer. Raise `min_score` so it **returns nothing** when no
candidate clears the bar — cutting the false-answer rate on out-of-corpus
queries (measured 92% → 8% on the bundled corpus at `min_score=5`, for a
2pt recall cost).

```python
from attune_rag import RagPipeline, KeywordRetriever

pipeline = RagPipeline(retriever=KeywordRetriever(min_score=5))
```

The threshold is an **absolute keyword score**, so calibrate it per corpus
— the benchmark recommends one from your legit + out-of-corpus query sets:

```bash
python -m attune_rag.benchmark --calibrate-abstention
# -> Recommended: min_score=5 (legit kept 98%, false-answer rate 8%)
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

corpus = DirectoryCorpus(Path("./templates"))

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
wall-clock is advisory.

Numbers measured under the V2 multi-run methodology (5
invocations × 20 runs = 100 measurements per metric) on the
locked-baseline runner (Linux `ubuntu-latest`, CPython 3.11.15).
Inter-run and intra-run variance are tracked separately;
thresholds are `mean + 2σ × inter_run_stdev`. **Full 8-row
dual-axis table + hardware fingerprint + per-metric noise
profile:**
[`docs/specs/downstream-validation/perf-baseline.md`](docs/specs/downstream-validation/perf-baseline.md).

Why two threshold styles in the locked table:

- **`keyword_retriever_retrieve`** has a wider CPU band because
  measured intra-run variance reflects cold-cache effects on the
  first few iterations — empirically derived, not tuned for
  tightness.
- **`llm_reranker_rerank`** is wall-clock-only because Anthropic
  network variance dominates the CPU axis; the gate is set
  generously.

Gated by [`.github/workflows/perf.yml`](.github/workflows/perf.yml)
per-PR (blocking on the CPU axis as of W3.1).

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

The same discipline powers `attune-author`'s polish/fact-check
pipeline — generated help content is scored against retrieved
source passages before being marked authoritative. attune-rag's
faithfulness primitives aren't just instrumentation; they're the
contract the family's content-quality story is built on.

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

## Embeddings — shipped (`[embeddings]` extra)

Local, CPU-only, offline embeddings shipped via
[`model2vec`](https://github.com/MinishLab/model2vec) static models (no
torch, no API key). Keyword retrieval remains the default; embeddings
layer in opt-in through `HybridRetriever` — see
[Embedding / hybrid retrieval](#embedding--hybrid-retrieval-embeddings-extra)
above.

**Measured impact** (`docs/specs/rag-strengthening/`): on an unseen,
unstructured corpus, hybrid lifts **recall@3 +9pts**; on the keyword-tuned
`.help/` corpus the default keyword path is unchanged. The benchmark can
compare either retriever:

```bash
attune-rag-benchmark --retriever keyword      # default
attune-rag-benchmark --retriever hybrid       # keyword + embeddings (RRF)
attune-rag-benchmark --retriever transformer  # [transformers] tier
```

See
[CHANGELOG.md](https://github.com/Smart-AI-Memory/attune-rag/blob/main/CHANGELOG.md)
for the decision record.

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
Formal SemVer commitments have been in effect since 0.2.0 — see
[docs/POLICY.md](docs/POLICY.md) for the deprecation policy. Symbols
PUBLIC in a minor line stay PUBLIC through every patch of that line;
the snapshot test catches drift.

**Top-level (`from attune_rag import ...`):**

- Pipeline — `RagPipeline`, `RagResult`
- Corpus — `CorpusProtocol`, `RetrievalEntry`, `DirectoryCorpus`,
  `AttuneHelpCorpus`
- Retrieval — `KeywordRetriever`, `EmbeddingRetriever`,
  `HybridRetriever`, `TransformerRetriever`, `RetrievalHit`,
  `RetrieverProtocol`
- Provenance — `CitationRecord`, `CitedSource`, `ClaimCitation`,
  `format_citations_markdown`, `format_claim_citations_markdown`
- Prompting — `build_augmented_prompt`, `PROMPT_VARIANTS`
- Hybrid retrieval — `QueryExpander`, `LLMReranker`

**PUBLIC submodules** (importable by qualified path):

- `attune_rag.corpus` — exposes `AliasInfo`, `DuplicateAliasError`,
  `load_aliases_from_file` in addition to the top-level corpus names
- `attune_rag.corpus.attune_help` — `AttuneHelpCorpus`
- `attune_rag.corpus.help_adapter` — `HelpCorpusAdapter` Protocol
- `attune_rag.providers` — `LLMProvider`, `get_provider`,
  `list_available`
- `attune_rag.measure_corpus` — `measure(...)` function +
  `MeasureResult` dataclass for scoring a corpus against a query
  set. CLI via `python -m attune_rag.measure_corpus ...` or the
  `attune-rag-measure` console script. See
  [`docs/USER_CORPUS_GUIDE.md`](docs/USER_CORPUS_GUIDE.md) §6 for
  the worked example.
- `attune_rag.editor` — template-editor primitives (lint, schema,
  rename, autocomplete, references); see "Template editor primitives"
  above for the symbol list
- `attune_rag.editor.{rename,schema,lint,autocomplete,references}` —
  the individual editor submodules

**Console scripts:**

- `attune-rag` — CLI entry point (`attune_rag.cli:main`)
- `attune-rag-measure` — quality measurement
  (`attune_rag.measure_corpus:main`); CI-suitable via `--watermark-r3`
  (non-zero exit on fail)
- `attune-rag-benchmark` — retrieval + optional faithfulness
  benchmark (`attune_rag.benchmark:main`). The default golden query
  sets ship in the repo checkout, not the wheel — on a pip install,
  point `--queries` (and optionally `--negatives`) at your own sets.

Anything not listed above is INTERNAL and may change in any release.
The underscore-prefixed editor modules (`attune_rag.editor._rename`
etc.) shipped in 0.1.x are deprecation shims as of 0.2.0; they
re-export the new non-underscore names and emit `DeprecationWarning`.
They are removed in 0.3.0.

## Status

**0.5.1 — the retrieval-capabilities line.** 0.5.0 landed the full
opt-in retrieval ladder — torch-free static hybrid (`[embeddings]`),
transformer dense tier (`[transformers]`), and configurable
abstention (`min_score=`); 0.5.1 is a packaging/docs correction on
top. Quality baselines (P@1 ≥ 0.95, R@3 = 1.00, mean faithfulness ≥
0.9686) hold and gate CI throughout.

SemVer commitments have been binding since 0.2.0 —
[`docs/POLICY.md`](docs/POLICY.md) §2; symbols PUBLIC in a minor
line stay PUBLIC through every patch of that line, and the snapshot
test catches drift.

Classifier is `4 - Beta` — the Production/Stable flip is a v1.0.0
deliverable.

Part of the attune ecosystem
([attune-ai](https://github.com/Smart-AI-Memory/attune-ai),
[attune-help](https://github.com/Smart-AI-Memory/attune-help),
[attune-author](https://github.com/Smart-AI-Memory/attune-author),
[attune-gui](https://github.com/Smart-AI-Memory/attune-gui)).

## License

Apache 2.0. See
[LICENSE](https://github.com/Smart-AI-Memory/attune-rag/blob/main/LICENSE).
