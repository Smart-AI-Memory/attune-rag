# Spec: safe-abstention-defaults — risks

> **Status:** scoping (2026-06-07).

## 1. Over-abstention on lean / short-query corpora

**Risk:** a default tuned to suppress attune-help's spurious matches
(neg median 3.38) cuts real answers on corpora whose legit scores are
naturally low. The measurement shows corpus_b legit top-1 median is
**6.0** — a global `min_score=5` would already start abstaining on
genuine corpus_b answers.

**Mitigation:** R3 forbids a universal absolute constant; the bundled
corpus gets a calibrated value, BYO corpora auto-calibrate or use a
conservative floor. Any C3 relative heuristic must clear a both-corpora
*legit-recall* bar (M2) before shipping — measure on legit sets, not
just negatives.

## 2. Default flip during freeze

**Risk:** changing the bundled abstention default is a behavior change.
Shipping it while the feature freeze is active could violate freeze
semantics or surprise downstream pins.

**Mitigation:** the scaffold is docs-only and lands now. The change
itself sequences to the v1.0.0 cut (R-entry-gate). Precedent
([#120](https://github.com/Smart-AI-Memory/attune-rag/pull/120),
the perf-gate promotion) shows a behavior change with no new public
surface ships as `### Changed` under freeze; if the chosen mechanism
adds a construction path, it takes a `freeze-override` with rationale.
Decision deferred to scoping (`design.md` §6 Q1/Q2).

## 3. POLICY §7 interaction

**Risk:** POLICY §7 codifies a retrieval-quality floor as a
downstream-facing contract. A new abstention default could be read as a
*new* behavioral commitment — and committing to an LLM-free but
corpus-sensitive metric as a SemVer floor has the same over-commitment
hazard flagged in
[`feedback_policy_llm_metric_commitments`](../../../).

**Mitigation:** abstention is deterministic and offline (no Anthropic
dependency), so it is safer to commit to than faithfulness — but the
*value* is corpus-relative. Recommended posture: state the bundled
default's behavior factually in POLICY without committing to a numeric
false-answer-rate floor across arbitrary corpora. Resolved at scoping
(`design.md` §6 Q5).

## 4. Negatives-set representativeness

**Risk:** the bundled default is calibrated against
`queries_negative.yaml` (12 queries: far + near flavors). If real
out-of-corpus traffic looks different, the calibrated threshold may
under- or over-abstain in production.

**Mitigation:** the negatives set is advisory, not a SemVer gate.
Calibration is reproducible (R5), so the threshold can be re-derived as
the negatives set grows. Document the threshold's provenance and the
set it was calibrated against so it is auditable, not magic.

## 5. Two surfaces disagree (harness vs bundled default)

**Risk:** identical to the reranker-default hazard
([`user-corpus-onboarding/risks.md` §7](../user-corpus-onboarding/risks.md)):
if `user-corpus-onboarding` ships a harness default while this spec
ships a bundled default and they are computed independently, v1.0.0 has
two surfaces saying different things.

**Mitigation:** R4 — single source of truth + a cross-link landing
before either spec closes scoping.

## 6. Short-query degeneracy under a per-token heuristic

**Risk:** C3's per-token criterion (`top1 / n_query_tokens`) can
misbehave on single-token queries — one strong token can clear any
floor (false answer) or a legit one-word lookup can be penalized.

**Mitigation:** if C3 is adopted, pair the relative criterion with a
modest absolute floor (keep-if-either), and include single-token
queries in the M2 legit-recall measurement explicitly.
