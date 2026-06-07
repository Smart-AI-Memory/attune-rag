# Spec: confidence-gated-retrieval — risks

> **Status:** scoping (2026-06-07).

## 1. The signal is n=4 (the headline risk)

**Risk:** the whole spec rests on corpus_b's 4-query hard tier. "0.50 →
0.75" is one recovered query. The full-corpus_b signal (n=11: P@1
0.73→0.82, R@3 0.82→0.91) is more credible but still small. Building a
retriever + heavier model on this would violate the program's own
measure-before-build discipline.

**Mitigation:** R1 makes a **≥30-query hard-set validation a hard entry
gate**. If the lift doesn't survive n≥30, the spec closes. This is
exactly the reopen trigger written into the hard-tier diagnostic.

## 2. Model footprint regression

**Risk:** `potion-retrieval-32M` is materially larger than the current
`potion-base-8M` (bigger download, higher RSS, slower first encode). A
silent footprint jump in the `[embeddings]` extra would surprise users.

**Mitigation:** keep the model configurable (default disclosed), state
the delta in the PR, and keep `potion-base-8M` available. Still
torch-free, so the *dependency class* is unchanged — it's a size, not a
new dependency.

## 3. Threshold T is corpus-relative

**Risk:** T=3 fit both corpora here, but T is an absolute keyword score;
a hardcoded global T will mis-gate some corpus (over-rescue → tuned-corpus
regression, or under-rescue → no paraphrase lift). Same trap as a global
`min_score`.

**Mitigation:** R4 + R6 — T is calibrated per corpus by a reproducible
tool, shared with abstention. No global constant.

## 4. Reopening a *permanent* defer

**Risk:** `embedding-retriever` was deferred **permanently** (2026-05-21)
on the basis that aliases+QueryExpander closed the gap on attune-help and
a new dependency wasn't justified. Reopening it risks thrash / relitigating
a settled call.

**Mitigation:** the defer's own reopen clause covers exactly this — "a
failure mode aliases+QueryExpander demonstrably can't close (e.g. a new
corpus where alias-authoring cost dominates)." corpus_b is that corpus.
And the reopen is *narrow*: torch-free static + gating, opt-in, not the
torch retriever the original defer rejected. Record the reopen in the
`embedding-retriever` spec so the history is legible.

## 5. Interaction with abstention and hybrid

**Risk:** three overlapping mechanisms now key on keyword confidence —
abstention (`min_score`), the gate (T), and `HybridRetriever`'s RRF.
Shipped independently they could contradict (e.g. gate rescues a query
that abstention would have suppressed).

**Mitigation:** R4 — one shared threshold/calibration, and joint design
with `safe-abstention-defaults`. Define the precedence explicitly:
below-confidence → embedding-rescue **if** the embedding leg is strong,
else abstain. One decision tree, not three.

## 6. Hard-switch brittleness

**Risk:** the prototype hard-switches keyword→embedding at T. A query
just under T goes fully to embedding even if keyword's #2/#3 were good,
potentially hurting medium-difficulty queries not represented in the
tiny sample.

**Mitigation:** M2 measures hard-switch vs a below-T RRF blend on the
≥30-query set, including medium-tier queries, before locking the rule.
