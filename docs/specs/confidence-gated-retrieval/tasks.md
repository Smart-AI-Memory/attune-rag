# Spec: confidence-gated-retrieval — tasks

> **Status:** **active — M0+M2 complete (2026-08-10).** Entry gates all
> opened: R1's >=30-query validation passed at M1, v1.0.0 shipped
> 2026-08-10 (freeze lifted), and
> [`safe-abstention-defaults`](../safe-abstention-defaults/) completed
> 2026-08-10 with the shared calibration machinery (its Q6 explicitly
> defers cross-tier confidence to this spec). M2 locked Q1/Q2 by
> measurement (`scripts/measure_gated_mechanism.py`). **M3 build is
> gated on the chair ratifying the escalated scope decision + Q3.**

## Scoping decisions (locked at `/spec` — TBD)

From [`design.md` §8](design.md#8-open-questions-for-scoping) /
[`requirements.md` "Open questions"](requirements.md#open-questions-for-scoping):

1. Hard switch vs below-T RRF blend — **below-gate 1:1 RRF blend
   (LOCKED at M2)**: ties switch on corpus_b, +15pts hard P@1 over
   switch on corpus_c (0.70 vs 0.55) — weak-but-nonzero keyword signal
   still contributes below the gate.
2. Gate signal: top-1 score vs top1−top2 gap — **top-1 score (LOCKED at
   M2)**: every gap-keyed config breaks the attune-help guard
   (0.75–0.95, never 1.00). Structural, not tunable: a tuned corpus
   routinely holds two strong close-scored relevant docs, so the gap
   misreads redundant strength as doubt.
3. `GatedRetriever` class vs `gate=`/`mode=` on `HybridRetriever` —
   **recommended: a `gate_threshold=` option on `HybridRetriever`**
   (the blend below the gate IS hybrid's existing 1:1 RRF; above the
   gate is a short-circuit return of the keyword leg — simpler-is-
   better, no new public class). **Ratification owed (API surface).**
4. Shared calibration tool with `safe-abstention-defaults` vs linked —
   **shared (per R4)**: extend the `RagPipeline.calibrated()` /
   `_calibrate_abstention` sweep to emit the gate threshold from the
   same keyword-confidence distribution. Mechanics finalized at M4.
5. Default model: `potion-retrieval-32M` vs larger static — **`potion-
   retrieval-32M` (LOCKED at M1/M2)**: `potion-base-8M` never reached
   the torch-free ceiling; larger static models unmeasured and not
   needed for the observed plateau.

## Milestones

### M0 — Entry + reopen
- [x] Record the reopen in [`embedding-retriever`](../embedding-retriever/)
      (status note → here). *(2026-08-10: narrow reopen note added atop
      its README — opt-in rescue-leg scope only; the attune-help-corpus
      defer stands.)*
- [x] Confirm joint design with `safe-abstention-defaults` (shared signal).
      *(2026-08-10: that spec is COMPLETE — bundled corpus abstains at
      the calibrated `min_score=5.0`; its Q6 defers cross-tier
      confidence here; `RagPipeline.calibrated()` is the shared
      calibration entry point R4 requires. And the M2 plateau below
      CONTAINS T=5 — the single shared threshold is empirically
      comfortable, not a compromise.)*

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
- [x] Measure hard-switch vs below-T RRF blend (Q1), and gate-on-score vs
      gate-on-gap (Q2), on the ≥30-set incl. medium tier — **plus
      corpus_c** (20 hard + 4 medium; second arbitrary corpus, absent at
      M1). Script: `scripts/measure_gated_fusion.py` promoted to
      `scripts/measure_gated_mechanism.py`.
- [x] Lock the rule + gate signal. **Q1 = below-gate 1:1 RRF blend;
      Q2 = keyword top-1 score** (scoping decisions above).

### M2 results (2026-08-10) — blend/score wins; the plateau contains the abstention threshold

`PYTHONPATH=src python3 scripts/measure_gated_mechanism.py`
(torch-free, deterministic; P@1/R@3 per cell):

| config | cb-hard (26) | cb-med (6) | cc-hard (20) | cc-med (4) | help (40) |
|---|---:|---:|---:|---:|---:|
| keyword (default) | 0.31/0.38 | 0.50/0.67 | 0.25/0.25 | 0.50/0.50 | 1.00/1.00 |
| hybrid 2:1 (8M, shipped) | 0.50/0.73 | 0.67/1.00 | 0.50/0.90 | 0.50/1.00 | 0.95/1.00 |
| embedding-only (ret-32M) | 0.46/0.69 | 0.83/1.00 | 0.55/0.85 | 0.25/0.75 | 0.47/0.70 |
| switch/score T=2..6 | 0.46–0.50/0.65 | 0.67/0.83 | 0.45–0.55/0.75–0.85 | 0.25–0.50/0.75 | **1.00/1.00** |
| **blend/score T=4–6** | **0.50/0.65** | 0.67/0.83 | **0.70/0.85** | 0.50/0.50 | **1.00/1.00** |
| switch/gap G=1..4 | 0.46/0.65 | 0.67–0.83 | 0.55–0.60/0.85 | 0.25/0.50–0.75 | 0.75–0.85 💥 |
| blend/gap G=1..4 | 0.46–0.50/0.65 | 0.67/0.83–1.00 | 0.70/0.80–0.85 | 0.50/0.50 | 0.90–0.95 💥 |

- **Q2 is structural:** every gap-keyed config breaks the attune-help
  guard (R2), at every threshold, in both mechanisms. Tuned corpora
  hold multiple strong close-scored relevant docs; top1−top2 misreads
  that redundant strength as doubt and rescues queries that were right.
  Top-1 score holds 1.00/1.00 at **every** T measured (2–6).
- **Q1 is a corpus_c story:** switch and blend tie on corpus_b (0.50
  hard-P@1 ceiling, unchanged from M1), but below-gate 1:1 fusion lifts
  corpus_c hard P@1 to **0.70** — vs 0.55 switch, 0.55 embedding-only,
  0.50 shipped hybrid, 0.25 keyword. Fusing the weak keyword leg below
  the gate beats discarding it.
- **The safe plateau T=4–6 contains T=5** — exactly the bundled-corpus
  abstention threshold `safe-abstention-defaults` calibrated. One
  shared per-corpus threshold (R4) costs nothing on this data.
- **Cross-spec confirmation:** M1 measured switch T=3–6 costing help
  (→0.95); today the same sweep holds 1.00/1.00 through T=6. The delta
  is abstention-M3's alias remediation (gq-032 top-1 3.75 → 12.75) —
  fixing the corpus widened the safe gate range, exactly as the
  shared-signal design predicts.
- **Watch item (not a blocker):** cc-med R@3 is blend's soft spot
  (0.50 vs hybrid's 1.00; n=4, two queries). Blend matches the keyword
  baseline there; re-check at M3 with the shipped config.

**The escalated scope decision now has its data.** M1's verdict was
"gated's only edge over shipped hybrid is zero tuned-corpus
regression." M2 with a second corpus strengthens that materially:
blend-gated at T=5 is **+20pts hard P@1 over shipped hybrid on
corpus_c (0.70 vs 0.50) AND zero-regression (1.00/1.00 vs hybrid's
0.95)** — stronger on the corpus it was built for and safer on the
corpus it must not harm. That is the safe-everywhere default-candidate
profile the M1 reframe asked for. Recommendation to the chair: build
M3 as the opt-in `[embeddings]` rung with explicit default-candidate
framing; any default flip stays a separate decision (R5 intact).

### M3 — Implement (the build PR)
- [ ] `GatedRetriever` (or `HybridRetriever(gate=…)`, per Q3), opt-in
      under `[embeddings]`, default model per Q5.
- [ ] Prove R2 (help 100/100), R3 (hard lift), R5 (base install
      unchanged) in CI. Disclose footprint delta (risk §2).
- [ ] `### Added`/`### Changed` per freeze decision (Q3 / R-NFR).
- [ ] State the `RagResult.confidence` contract (requirements.md 2026-06-10
      audit input): either the gated retriever normalizes it or the field
      is documented retriever-relative — decided in the M3 design.

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
