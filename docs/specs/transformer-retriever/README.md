# Spec: transformer-retriever (attune-rag)

> **Status:** **scoping** (2026-06-07). Scaffold is docs-only and
> freeze-compliant — it lands now. The retriever is **not** built here;
> it's a **heavyweight opt-in `[transformers]` tier**, likely v1.1.0+,
> with its own `/spec` pass.
>
> **Narrowly reopens** the [`embedding-retriever`](../embedding-retriever/)
> permanent defer — which was specifically a *torch* defer. This spec
> brings torch back **only** as an opt-in tier for arbitrary corpora,
> **never** as a default.

## Purpose

Give users pointing attune-rag at an **arbitrary BYO corpus** where
**paraphrase recall matters** the best achievable retrieval quality —
which the measurements show requires a real transformer embedding model
and is unreachable by any torch-free option.

This is the one goal the
[`confidence-gated-retrieval` M1b comparison](../confidence-gated-retrieval/tasks.md)
identified as **uniquely** requiring sentence-transformers. For every
other goal (bundled/tuned corpus, zero-regression safe-everywhere) torch
is unnecessary; this spec is scoped tightly to the case where it isn't.

## The anchoring measurement (2026-06-07)

n=26 hard paraphrase set (`queries_corpus_b_hard.yaml`), attune-help as
the regression guard. "Torch-free ceiling" = the best gated/hybrid/static
result.

| config | hard P@1 | hard R@3 | help P@1/R@3 |
|---|---:|---:|---:|
| torch-free ceiling (gated, static) | 0.50 | 0.65–0.73 | 1.00 / 1.00 |
| `all-MiniLM-L6-v2` embedding-only | 0.58 | **0.92** | 0.42 |
| **`bge-small-en-v1.5` embedding-only** | **0.69** | 0.81 | 0.53 |

- **Torch genuinely exceeds the torch-free ceiling** on paraphrase: hard
  P@1 0.50→**0.69** (bge-small), R@3 →**0.92** (MiniLM). No static /
  keyword / gated config reaches this.
- The gain is **embedding-primary**: a transformer leg leading, not
  gated behind keyword. That tanks a *tuned* corpus (help 0.42–0.53),
  which is why it is for **arbitrary corpora only**, where there is no
  keyword-tuned precision to protect.

## Why this is opt-in, never a default

The
[`confidence-gated-retrieval` M1b](../confidence-gated-retrieval/tasks.md)
showed there is **no single configuration** that gets the transformer's
0.69 hard *and* attune-help 1.00 — the right operating point is
corpus-type-dependent:

| corpus type | retriever | torch? |
|---|---|---|
| bundled / keyword-tuned | keyword | no |
| arbitrary, paraphrase-heavy | **transformer, embedding-primary** | **yes** |

So this tier is a **new rung on the existing opt-in ladder**
(keyword default → `[embeddings]` static hybrid → `[transformers]`), not
a replacement for any of them. The base install and the keyword default
are untouched.

## Cost (the reason it's a separate, heavy tier)

- **Dependency:** torch (~GB) + sentence-transformers — far heavier than
  the torch-free `[embeddings]` (`model2vec`, ~30 MB).
- **Latency:** ~3 s first-load (model init), then ~10–300 ms/query vs
  <1 ms keyword / ~1 ms static.
- **Offline:** like `[embeddings]`, the model downloads once from
  HuggingFace then runs offline/deterministic. Torch is the heavy part,
  not the network behavior.

These costs are acceptable for an explicit opt-in; they are disqualifying
for a default.

## Implementation is light (the encoder is already injectable)

[`EmbeddingRetriever`](../../../src/attune_rag/embedding.py) already
accepts an injectable `encoder` exposing `encode(list[str]) -> 2D array`
— which is exactly the `SentenceTransformer.encode` interface. The M1b
measurement ran by injecting a `SentenceTransformer` with **zero changes**
to the retriever. So this tier is mostly: a `[transformers]` extra, a
small encoder adapter, a default model, and docs — not a new retriever.

## What's *not* in scope

- **Changing any default.** Keyword stays the base default; static hybrid
  stays the `[embeddings]` default. No default flip.
- **The bundled attune-help corpus.** It's keyword-optimal already; this
  tier is for arbitrary corpora.
- **Rerank / LLM expansion.** Both data-gated out; orthogonal.
- **Editing `queries.yaml`** (SHA-locked).

## Layout

- [`requirements.md`](requirements.md) — opt-in-only, base-install
  untouched, footprint disclosure, determinism, second-corpus validation.
- [`design.md`](design.md) — encoder adapter, model choice, symmetric vs
  asymmetric (query-prefix) encoding, embedding-primary config, open Qs.
- [`risks.md`](risks.md) — footprint/latency, offline/download,
  determinism across torch versions, single-corpus n=26 caveat,
  maintenance burden.
- [`tasks.md`](tasks.md) — M0 reopen → M1 second-corpus + asymmetric
  validation → M2 implement extra → M3 docs → v1.1.0+ sequencing.

## Provenance

Opened 2026-06-07 from the
[`confidence-gated-retrieval` M1b torch comparison](../confidence-gated-retrieval/tasks.md),
which answered "is sentence-transformers the only way?" — yes, for this
goal only. Probes: `/tmp/torch_ceiling.py`, `/tmp/torch_tsweep.py`
(authoring-time; promote to `scripts/` at M1).
