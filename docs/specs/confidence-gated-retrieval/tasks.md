# Spec: confidence-gated-retrieval — tasks

> **Status:** scoping (2026-06-07). M1+ are **not** executable until the
> entry gate ([`requirements.md`](requirements.md#entry-gates)) opens.
> Scoping decisions are filled at the `/spec` pass.

## Scoping decisions (locked at `/spec` — TBD)

From [`design.md` §8](design.md#8-open-questions-for-scoping) /
[`requirements.md` "Open questions"](requirements.md#open-questions-for-scoping):

1. Hard switch vs below-T RRF blend — _TBD_ (decided by M2)
2. Gate signal: top-1 score vs top1−top2 gap — _TBD_
3. `GatedRetriever` class vs `gate=`/`mode=` on `HybridRetriever` — _TBD_
4. Shared calibration tool with `safe-abstention-defaults` vs linked — _TBD_
5. Default model: `potion-retrieval-32M` vs larger static — _TBD_

## Milestones

### M0 — Entry + reopen
- [ ] Record the reopen in [`embedding-retriever`](../embedding-retriever/)
      (status note → here).
- [ ] Confirm joint design with `safe-abstention-defaults` (shared signal).

### M1 — Build + validate the ≥30-query hard set (the gate)
- [x] Author a **≥30-query paraphrase/hard set**
      (`tests/golden/queries_corpus_b_hard.yaml`, 32 queries: 26 hard + 6
      medium, authored blind to retrieval behavior). Advisory side-file;
      SHA-locked `queries.yaml` untouched.
- [x] Promote the probe to `scripts/validate_gated_fusion.py`.
- [x] Re-measure keyword vs hybrid vs embedding-only vs gated (T sweep) on
      the expanded set + attune-help guard.

### M1 results — validation PASSES the formal gates, but corrects the n=4 hype (2026-06-07)

| config | hard P@1 | hard R@3 | full-cbh P@1/R@3 | attune-help P@1/R@3 |
|---|---:|---:|---:|---:|
| keyword (default) | 0.31 | 0.38 | 0.34 / 0.44 | 1.00 / 1.00 |
| hybrid 2:1 (8M, **shipped**) | 0.50 | **0.73** | 0.53 / 0.78 | 0.85 / 1.00 |
| embedding-only (ret-32M) | 0.46 | 0.69 | 0.53 / 0.75 | 0.28 / 0.68 |
| **gated T=2 (ret-32M)** | **0.50** | 0.65 | 0.53 / 0.69 | **1.00 / 1.00** |
| gated T=3–6 (ret-32M) | 0.46 | 0.65 | 0.50 / 0.69 | 1.00→0.95 |

- **The 0.75 "ceiling" was an n=4 artifact.** At n=26 the torch-free
  hard-tier ceiling is **~0.50 across every approach** (hybrid, embedding,
  gated). The original 4-query sample over-stated the lift; this n≥30 gate
  is exactly what caught it.
- **Gated still passes R1/R2/R3:** hard P@1 0.31→0.50 (**+19pts**), R@3
  0.38→0.65 (**+27pts**), attune-help held at **1.00/1.00**. T=2 is the
  knee (max lift + zero regression).
- **But the build case is weaker than it looked.** The already-shipped
  `HybridRetriever` matches gated on hard P@1 (0.50) and beats it on hard
  R@3 (0.73 vs 0.65). **Gated's *only* edge is zero tuned-corpus
  regression (1.00 vs hybrid's 0.85)** — which matters for a
  *default / bundled-safe* retriever, NOT for the opt-in BYO case (where
  the user isn't querying attune-help and hybrid already suffices).

**Decision gate (R1): PASS — but reframed.** Gated works as designed, yet
its marginal value over shipped hybrid is the zero-regression property
alone. So building it as *another opt-in BYO retriever* is **not**
justified (hybrid already covers that). It is only worth building if the
goal becomes **a single safe-everywhere retriever** — i.e. toward a
better-than-keyword default or a bundled-corpus-safe option, designed
jointly with [`safe-abstention-defaults`](../safe-abstention-defaults/).
That scope decision is escalated before M2. (Also reframes the original
question: ~0.50 is the torch-free hard ceiling, so "does torch exceed it"
is now the sharper open question.)

### M1b — torch / sentence-transformers comparison (2026-06-07)

The "does torch beat the ~0.50 torch-free ceiling?" question, measured
directly. Real transformer encoders injected into `EmbeddingRetriever`
(symmetric encoding), same n=26 hard set + attune-help guard.

| config | hard P@1 | hard R@3 | help P@1/R@3 |
|---|---:|---:|---:|
| torch-free gated T=2 (ret-32M) | 0.50 | 0.65 | 1.00 / 1.00 |
| embedding-only `all-MiniLM-L6-v2` | 0.58 | **0.92** | 0.42 |
| embedding-only `bge-small-en-v1.5` | **0.69** | 0.81 | 0.53 |
| gated T=3 + bge-small (zero-regression) | 0.54 | 0.73 | **1.00 / 1.00** |
| gated T=4–6 + bge-small | 0.62 | 0.73 | 0.97 |
| gated T=10 + bge-small | **0.69** | 0.81 | 0.82 |

**Findings:**
- **Torch genuinely exceeds the torch-free ceiling** on paraphrase — but
  only **embedding-primary**: bge-small hits hard P@1 **0.69** (vs 0.50),
  MiniLM hits R@3 **0.92**. No torch-free option reaches this.
- **Gating throttles the gain.** The threshold that protects attune-help
  also keeps deferring to weak keyword, so at zero help-regression (T=3)
  torch buys only hard 0.50→0.54 over torch-free. The transformer's full
  strength (0.69) needs T=10+, which costs attune-help (→0.82).
- There is **no single gate setting** that gets both the transformer's
  0.69 hard *and* help 1.00 — the operating point is corpus-type-dependent.

**Conclusion — the design space, fully mapped:**

| goal | best retriever | torch? |
|---|---|---|
| bundled/tuned corpus | keyword | no (already 1.00) |
| safe-everywhere, zero tuned-corpus regression | torch-free gated (~0.50 hard) | no — torch adds only ~+4pts |
| **max paraphrase quality on arbitrary BYO corpus** | **bge-small, embedding-primary** (0.69 / 0.92) | **yes — uniquely** |

So sentence-transformers is the only way **for one specific goal**: best
paraphrase retrieval on an arbitrary corpus, where the user accepts an
embedding-primary config (no keyword-tuned precision to protect). It is a
heavyweight (~GB torch dep, ~3 s first-load, 10–300 ms/query) → fits a
separate **opt-in `[transformers]`** rung, never a default. For the
bundled exemplar and for a zero-regression safe-everywhere retriever,
torch is **not** justified. This narrowly reopens the
[`embedding-retriever`](../embedding-retriever/) defer (which was a torch
defer) — as an opt-in heavyweight tier, not a default flip.

### M2 — Mechanism bake-off
- [ ] Measure hard-switch vs below-T RRF blend (Q1), and gate-on-score vs
      gate-on-gap (Q2), on the ≥30-set incl. medium tier.
- [ ] Lock the rule + gate signal.

### M3 — Implement (the build PR)
- [ ] `GatedRetriever` (or `HybridRetriever(gate=…)`, per Q3), opt-in
      under `[embeddings]`, default model per Q5.
- [ ] Prove R2 (help 100/100), R3 (hard lift), R5 (base install
      unchanged) in CI. Disclose footprint delta (risk §2).
- [ ] `### Added`/`### Changed` per freeze decision (Q3 / R-NFR).

### M4 — Shared calibration + threshold
- [ ] Per-corpus T via the **shared** abstention/gate calibration (R4,
      Q4). One threshold, one tool, reproducible (R6).

### M5 — Docs
- [ ] README/onboarding: when to enable gated retrieval; footprint; the
      one "do I trust this retrieval?" decision tree (rescue vs abstain).

## Done when

- The unseen-corpus hard-tier lift is real at **n≥30** (not a 4-query
  artifact), torch-free and deterministic.
- attune-help stays 100% / 100% (zero regression), base install
  unchanged.
- One shared keyword-confidence threshold governs both rescue and
  abstention.
- The `embedding-retriever` reopen is recorded and the calibration is
  reproducible.

## Provenance

Opened 2026-06-07 from the
[`rag-strengthening` hard-tier amendment](../rag-strengthening/tasks.md)
after confidence-gated fusion (keyword-primary, `potion-retrieval-32M`
rescue, T=3) hit the LLM-expansion ceiling (hard P@1 0.75) with zero
attune-help regression — torch-free, no API. Reopens
[`embedding-retriever`](../embedding-retriever/); designed jointly with
[`safe-abstention-defaults`](../safe-abstention-defaults/).
