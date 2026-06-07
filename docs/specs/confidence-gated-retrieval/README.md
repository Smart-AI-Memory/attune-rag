# Spec: confidence-gated-retrieval (attune-rag)

> **Status:** **scoping** (2026-06-07). Scaffold is docs-only and
> freeze-compliant — it lands now. The retriever it proposes is **not**
> built here; it activates after a ≥30-query hard-set validation
> ([`requirements.md` R1](requirements.md)) with its own `/spec` pass.
>
> **Reopens** the [`embedding-retriever`](../embedding-retriever/)
> permanent defer — with a *torch-free* design that the earlier defer
> did not have on the table.

## Purpose

Close the unseen-corpus **paraphrase / hard-tier gap** (corpus_b hard
P@1 stuck at 50%) **without** the dependencies every prior lever
required — no torch, no API, deterministic — and **without regressing
the bundled corpus**.

This spec exists because a same-day amendment to the
[`rag-strengthening` hard-tier diagnostic](../rag-strengthening/tasks.md)
overturned its own "measured-out" verdict: after the LLM-expansion
ceiling was established (0.75), two cheap torch-free levers were run and
one of them reached that ceiling with **zero** tuned-corpus regression.

## The anchoring measurement (2026-06-07)

P@1 / R@3 on corpus_b hard tier, full corpus_b, and attune-help
(regression guard). Target: the LLM-expansion ceiling = **0.75** hard P@1.

| config | hard P@1 | corpus_b P@1 / R@3 | attune-help P@1 / R@3 |
|---|---:|---:|---:|
| keyword (today's default) | 0.50 | 0.73 / 0.82 | **1.00 / 1.00** |
| RRF re-weighting (any weight, 8M) | 0.50 | — | degrades help |
| LLM query expansion (ceiling) | 0.75 | — | — |
| embedding-only `potion-retrieval-32M` | 0.75 | 0.91 / 0.91 | 0.28 💥 |
| **confidence-gated fusion** (keyword-primary, 32M rescue, T=3) | **0.75** | **0.82 / 0.91** | **1.00 / 1.00** |

Three findings drive the design:

1. **RRF re-weighting is a dead end.** Hard-tier P@1 stays 0.50 at every
   keyword:embedding weight; leaning embedding-heavy only trades away
   attune-help precision. The fusion *rule*, not the weight, is the
   problem.
2. **A retrieval-tuned static model reaches the ceiling, torch-free.**
   `minishlab/potion-retrieval-32M` (model2vec, no torch, ms-encode)
   used embedding-only hits hard 0.75 / corpus_b 0.91 — but **tanks the
   tuned corpus (0.28)**, so it cannot be a global default.
3. **Confidence-gated fusion captures both.** Route to the embedding leg
   only when keyword top-1 is below a threshold T. At **T=3**: hard-tier
   **0.75** *and* attune-help **1.00 / 1.00 — zero regression.** No
   torch, no API, deterministic.

## The unifying insight (why this and abstention are one signal)

The gate key is **keyword top-1 confidence** — the *same* signal that
drives [`safe-abstention-defaults`](../safe-abstention-defaults/). When
keyword top-1 is low, two things can be true:

- the embedding leg finds a strong match → **rescue** (in-corpus
  paraphrase), or
- nothing scores well anywhere → **abstain** (out-of-corpus).

Paraphrase-rescue and abstention are the *same mechanism* keyed on the
same threshold. This spec and `safe-abstention-defaults` MUST be
designed together so v1.0.0 ships one coherent "do I trust this
retrieval?" decision, not two thresholds that disagree.

## What ships (eventually — not in this scaffold)

An **opt-in** retriever (under the `[embeddings]` extra) that:

- uses a retrieval-tuned static model (`potion-retrieval-32M` or
  better), and
- fuses via a **confidence gate** (keyword-primary, embedding-rescue
  when keyword top-1 < T), with T calibrated per corpus (same per-corpus
  lesson as `min_score`).

Keyword stays the zero-dependency default. This is a new rung on the
existing opt-in ladder, not a default flip.

## What's *not* in scope

- **Torch / sentence-transformers.** The whole point is that the
  retrieval-tuned *static* model reaches the ceiling without it. A torch
  model is out of scope unless a ≥30-query validation shows static
  cannot close the gap.
- **Changing the keyword default.** Keyword-only stays the base-install
  default; this is opt-in.
- **LLM query expansion / rerank.** Both already data-gated out
  ([#165](https://github.com/Smart-AI-Memory/attune-rag/pull/165)); this
  spec is the dependency-light alternative to them.
- **Editing `queries.yaml`** (SHA-locked). The ≥30-query hard set lands
  as a new advisory side-file.

## Layout

- [`requirements.md`](requirements.md) — invariants (zero attune-help
  regression, ≥30-query validation gate, shared-threshold consistency
  with abstention, footprint bound, determinism).
- [`design.md`](design.md) — the measurement, the gated-fusion mechanism,
  model choice, threshold calibration, and open questions.
- [`risks.md`](risks.md) — n=4 thinness, model footprint, T
  corpus-relativity, reopening a permanent defer, abstention interaction.
- [`tasks.md`](tasks.md) — M0 entry → M1 build+validate the ≥30-query
  hard set → M2 implement → M3 calibrate → M4 tests/docs.

## Provenance

Opened 2026-06-07 from the
[`rag-strengthening` hard-tier amendment](../rag-strengthening/tasks.md).
Probes: `gated_fusion.py` / `static_levers.py` (authoring-time; promote
to `scripts/` at M1).
