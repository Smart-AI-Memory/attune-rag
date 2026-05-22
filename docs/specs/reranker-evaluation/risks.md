# Spec: reranker-evaluation — risks

> **Status:** **scoped 2026-05-22.** Risk register; mitigations are
> implementation-time work tracked in [`tasks.md`](tasks.md).

## 1. The verdict moves on Haiku nondeterminism

**Risk:** the reranker is the only Anthropic-bound component on the
measurement path. Two runs of `rerank=on` against the same 80
paraphrased queries may produce different P@1 numbers — the model
genuinely re-orders candidates non-deterministically when several
are plausible. A single-pass measurement could land the verdict
anywhere across that noise band.

If the noise band straddles the
`rerank-default-on` ↔ `rerank-default-off` boundary, the verdict is
*noise-determined*, not measurement-determined. That's
indistinguishable from coin-flipping the v1.0.0 default.

**Posture:** **characterize the noise before locking the verdict.**

The N=5 rerank-on plan in
[design.md §6](design.md#6-single-run-vs-multi-run) is the proposed
mitigation. If the across-run stdev on paraphrased P@1 is < ~1
point (one query out of 80), single-pass is fine. If stdev is > ~2
points, N=5 is the floor and the verdict cites mean ± stdev rather
than a point estimate. Scoping pass picks the N.

If even at N=5 the noise straddles the verdict boundary, that's
itself a finding — `corpus-shape-dependent-default` is the honest
verdict (the reranker's contribution is in the same order as its
nondeterminism, which means "default doesn't matter much").

## 2. The 7 paraphrased P@1 misses are wrong about the world

**Risk:** the 7 paraphrased queries that miss P@1 under keyword-only
were synthesized as part of the paraphrase set's authoring — they
were *deliberately phrased* to be hard for keyword retrieval. They
may not represent the distribution of real user paraphrases.

If the reranker fixes all 7, that's measured against a synthetic
hard set, not real user traffic. Confidence in the "rerank lifts
P@1" finding is bounded by how representative those 7 are.

**Posture:** acknowledge the limit in the report's verdict
rationale.

The verdict paragraph should explicitly state: "the lift figure is
measured against the bundled paraphrased set, which was synthesized
to be hard for keyword-only retrieval. Real-world paraphrase
distributions may differ; telemetry (planned for v1.1.0+ per
[`telemetry/`](../telemetry/)) will eventually let us measure
real-distribution lift."

This is honest framing, not a blocker. The diagnostic's purpose is
to make a v1.0.0 default decision *now* with the data available
*now*; telemetry will let v1.1.0+ revisit the decision with more
ground truth.

## 3. The default-flip lands during freeze and breaks attune-gui

**Risk:** if D5's verdict is `rerank-default-off` and the flip lands
within Phase 4 freeze as `### Changed`, attune-gui's existing
integration assumes the rerank default is on. A behavior flip — even
a SemVer-legal one for selection-criteria — could surprise the
downstream consumer.

The downstream-validation Phase 4 gate is supposed to catch this,
but the gate measures *correctness* (does the pipeline still run),
not *quality* (are the returned documents the same).

**Posture:** **the flip PR is conservative; defaults to v1.0.0 cut.**

If verdict is `rerank-default-off`, the scoping pass for D5's
default-flip should consider whether the flip is genuinely
`### Changed`-shaped (no public-surface impact, no SemVer break) or
whether it's *behaviorally* a break that v1.0.0 cut PR is the right
home for.

The default position for the flip: **defer to the v1.0.0 cut PR.**
That gives downstream consumers (attune-gui) the natural SemVer
boundary to react to a behavior change. Flipping during freeze is
the optimization; the v1.0.0 cut PR is the safe path.

## 4. D5 measures a stable prompt that's about to change

**Risk:** the reranker prompt at
[`reranker.py:14-30`](../../../src/attune_rag/reranker.py) has had
multiple iterations. If a prompt-tuning round happens between D5's
measurement and v1.0.0 ship, the verdict is measuring a prompt that
isn't the v1.0.0 prompt — making the verdict obsolete.

**Posture:** **freeze the prompt before D5 measures, until the
verdict locks.**

Practically: D5's `/spec` scoping pass declares the reranker prompt
as "frozen during the diagnostic" — no prompt-tuning PRs land
between D5's measurement and the verdict locking. After the verdict
locks, the prompt is unfrozen.

If a prompt-tuning round genuinely needs to happen mid-Phase-5 (e.g.
a bug discovered in the prompt), D5 re-runs. The cost is small —
~$0.30 of Haiku spend and an afternoon of measurement.

## 5. The `corpus-shape-dependent` verdict is a cop-out

**Risk:** a `corpus-shape-dependent-default` verdict sounds nuanced
but is functionally indistinguishable from "we didn't decide." It
pushes the decision to every individual user, which is the wrong
default for the bundled-corpus exemplar — attune-help is a known
corpus shape, and v1.0.0's job is to make a decision about it.

**Posture:** **`corpus-shape-dependent` requires the bundled-corpus
default to be picked anyway.**

The verdict logic in
[design.md §4](design.md#4-verdict-logic-proposed-scoping-confirms)
allows `corpus-shape-dependent-default` only when the bundled-corpus
numbers genuinely don't argue for one side or the other (lift is
real but within noise, or lift is real but cost is high). Even in
that case, **the bundled-corpus default still gets a pick** — the
`corpus-shape-dependent` label means "the user-corpus harness should
default to mirroring the pipeline default but document the toggle
prominently," not "the package ships without picking a default."

The `RagPipeline` constructor cannot ship without *some* default for
the `reranker` argument. The verdict picks that default; the label
captures the confidence around it.

## 6. Token-cost measurement misses the "list price isn't real price"

**Risk:** the report cites Haiku *list price* per the SDK's
`response.usage`. Real-world prices vary — prompt caching, batch
discounts, enterprise contracts, model deprecation pricing. A
verdict that turns on token-cost (e.g. "the lift is real but too
expensive at $X/query") may be wrong about $X for many users.

**Posture:** **report token *count* prominently; cost in dollars is
secondary.**

The token-count tables are the durable measurement. The dollar
conversion is convenience for the rationale paragraph. The verdict
should turn on "token count per query is X" — which is invariant —
not "dollars per query is Y" — which depends on the user's billing
arrangement.

This is also why
[requirements.md R2](requirements.md#r2-the-diagnostic-captures-token-usage-from-responseusage)
captures tokens specifically (not dollars).

## 7. The diagnostic itself drifts from the test_golden runner

**Risk:** if D5 picks Approach B (separate script;
[design.md §2.2](design.md#22-approach-b-separate-diagnostic-script-under-scripts)),
the diagnostic and `test_golden.py` measure subtly different things
over time as one or the other refactors. A future "the diagnostic
says X but the regression suite says Y" investigation costs days.

**Posture:** **R1 is the regression net.**

[requirements.md R1](requirements.md#r1-the-diagnostic-reproduces-the-bundled-baseline-numbers-under-rerankoff)
demands that Run A reproduce the locked baseline numbers
byte-identically. If the diagnostic and the test suite drift,
Run A's numbers stop matching the locked baseline and the
discrepancy surfaces immediately — not in some future debug session.

The N=1 cost of Run A acting as the regression check is negligible
(it's deterministic; 40 + 80 queries; <10 seconds wall-clock).

## 8. The verdict doesn't survive the embedding-retriever revival

**Risk:** `embedding-retriever` is scope-deferred — but the scope
specifically allows revival if user corpora consistently fail the
keyword+frontmatter discipline (see
[`embedding-retriever/#scope-of-the-defer`](../embedding-retriever/#scope-of-the-defer)).
If that revival happens in v1.1.0+, the pipeline shape changes —
the reranker may be operating on candidates from a different
retriever — and D5's verdict (measured against keyword-only) may
not generalize.

**Posture:** D5's verdict is **scoped to v1.0.0's keyword-retrieval
pipeline.** The report says so explicitly.

If the embedding-retriever spec revives in v1.1.0+ and lands, D5's
verdict gets a follow-up measurement against the new pipeline shape
— D5-rerun-for-embedding-retriever. That's a future spec's problem,
not D5's. D5 just notes the scope in its rationale.

---

None of the above blocks scaffolding the spec or the `/spec` pass
that promotes it. The risk register names each so the implementation
milestones can design against them.
