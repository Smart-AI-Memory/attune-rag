# Spec: transformer-retriever — risks

> **Status:** scoping (2026-06-07).

## 1. Single-corpus signal (n=26, corpus_b only)

**Risk:** the 0.69 / 0.92 advantage is measured on one synthetic corpus.
It may not generalize, or corpus_b's paraphrases may flatter transformers.

**Mitigation:** R1 makes a **second-corpus validation a hard entry gate**.
No build until the margin reproduces on a different arbitrary corpus.

## 2. Dependency footprint

**Risk:** torch is ~GB — orders of magnitude heavier than the entire rest
of attune-rag (and the `model2vec` extra at ~30 MB). It dwarfs the package
and can dominate install time / image size for downstream users.

**Mitigation:** strictly opt-in (`[transformers]`), never pulled by the
base install or `[embeddings]` (R2). Disclose the delta (R4). Consider a
CPU-only torch wheel in the extra's pin (design §8 Q4). Document that this
tier is for users who *need* paraphrase recall and accept the weight.

## 3. Latency

**Risk:** ~3 s first-load + 10–300 ms/query vs <1 ms keyword. For
latency-sensitive callers this is a large regression.

**Mitigation:** opt-in; document the profile (R4). Model loads once per
process (cached), so steady-state is per-query encode only.

## 4. Offline / download expectation

**Risk:** attune-rag's value prop includes offline/deterministic
retrieval. A transformer model requires a one-time HuggingFace download;
a user expecting fully-offline-from-install could be surprised.

**Mitigation:** same model as `[embeddings]` already does (model2vec also
downloads once). Document the one-time download + an offline/pre-cache
path (R6). Torch is the new weight, not new network behavior.

## 5. Determinism across torch versions

**Risk:** float results can drift across torch/hardware (CPU vs MPS vs
CUDA), so the embedding matrix — and thus ranking ties — could differ
between environments.

**Mitigation:** pin a torch floor, force eval mode + float32, keep the
path-based tie-break. Document that exact scores are environment-sensitive
while top-k ordering is stable for realistic gaps (R5).

## 6. Reopening a *permanent* defer (again)

**Risk:** `embedding-retriever` was deferred **permanently** on the torch
cost; bringing torch back risks looking like churn / relitigating.

**Mitigation:** the reopen is narrow and explicitly conditional — torch as
an **opt-in tier for arbitrary corpora only**, never a default, justified
by a measured capability (0.69 vs 0.50) the defer never had data for.
Record the reopen in the `embedding-retriever` spec so history is legible.
The original defer's "no torch in the default/core" intent is preserved.

## 7. Maintenance burden

**Risk:** torch + sentence-transformers is a fast-moving, heavy
dependency surface (version churn, security advisories, wheel/platform
issues) for a feature only some users enable.

**Mitigation:** isolate behind the extra + lazy import; keep tests on a
fake encoder so the core suite never depends on torch; treat the real-model
job as optional/non-gating. Revisit the tier if maintenance cost outweighs
adoption.
