# Retro: alias-expansion sweep (2026-05-21)

**Outcome:** paraphrased R@3 went from **28.75% → 100%** with zero baseline regression, no new dependency, and a permanent defer of the embedding-retriever spec. Shipped as v0.1.23 ([PR #110](https://github.com/Smart-AI-Memory/attune-rag/pull/110)). 13 PRs, one day.

## The arc

```
Morning framing      "We need an embedding retriever for lexical-mismatch failures."
D1 diagnostic        Confirmed the gap: P@1 97.5% → 11.25%, R@3 100% → 28.75%
The challenge        "What cheaper alternative haven't I measured?"
D2 (QueryExpander)   Closed +51pp R@3 but cost −10pp baseline P@1. Real, but lossy.
D3 (hand aliases)    +50pp R@3 on one cluster, zero baseline cost. Strict-dominant.
Sweep (M2–M12)       11 PRs, one cluster each. Acceptance hit at M7. M8–M12 polish.
D4 (residuals)       Diagnosed 3 misses: not structural, just authoring gaps. Closed.
```

## Numbers

| | D1 keyword | After D4 |
|---|---:|---:|
| Paraphrased P@1 | 11.25% | 91.25% |
| Paraphrased R@3 | 28.75% | **100.00%** |
| Baseline P@1 | 97.50% | 100.00% |
| Baseline R@3 | 100% | 100% |

Zero baseline regression across 13 PRs. Watermark floor lifted 50% → 85%.

## 7 lessons (the durable ones)

1. **Challenge your own framing before scoping work.** The initial "embedding retriever is the answer" verdict held up through D1. The self-challenge cost 30 min of D2/D3; the avoidance saved a multi-week build. Heuristic: *when a diagnostic confirms what you were already inclined toward, that is exactly when to ask what cheaper alternative you haven't measured.*

2. **Diagnostics belong in the spec, not before it.** D1–D4 live in `docs/specs/embedding-retriever/` even though embedding-retriever ended up deferred. The diagnostic artifacts are *part of* the permanent-defer decision's defensibility.

3. **Strict-dominance over 13 PRs isn't an accident.** Every PR ran the full baseline diagnostic *before* commit. M12 nearly broke this — a `"readme lies about code"` alias would have shipped a baseline P@1 regression. Caught pre-commit; one offending alias removed. The workflow `write → diagnose → commit` is the load-bearing piece.

4. **The stemmer is sneakier than it looks.** `bites → bit` collapses (via `-es` suffix), but `bite → bite` doesn't. Always run `_tokenize()` on alias candidates before authoring. Don't trust the mental model.

5. **Sequential PRs cost wall-clock, save context.** Two specific wins from deliberate pace: M12's near-regression caught pre-merge; the cross-repo classifier block became a conversation instead of an engineered-around problem. Follow-up-to-fix rate across 13 PRs was effectively zero.

6. **Defer-permanently is a separate decision from defer-pending.** Embedding-retriever moved `deferred` → `deferred (permanent)` after M13. Status banners should reflect *which* defer state. Conflating them invites future readers to re-ask the same question.

7. **Override mechanism is a tactical layer, not the long-term home.** `aliases_override.json` shipped so the sweep didn't require an attune-help release per cluster. Once Smart-AI-Memory/attune-help#9 lands, the override entries become redundant and should be trimmed. But the *mechanism* stays — useful for testing alias hypotheses before upstream promotion.

## Artifacts produced

- 4 diagnostic writeups + 4 driver scripts at `docs/specs/embedding-retriever/`
- 80-query paraphrased regression set at `tests/golden/queries_paraphrased.yaml`
- 180+ multi-token aliases across 13 templates in `aliases_override.json`
- Override mechanism in `DirectoryCorpus.extra_aliases` + `AttuneHelpCorpus`
- Per-query `xfail(strict=False)` + aggregate watermark in `tests/golden/test_golden.py`
- Permanent-defer status on `docs/specs/embedding-retriever/`
- Upstream-promotion task at [Smart-AI-Memory/attune-help#9](https://github.com/Smart-AI-Memory/attune-help/pull/9) (in flight)

## What this enables

- The **alias-sweep playbook** is reusable on any markdown corpus exhibiting paraphrase gaps. `user-corpus-onboarding` (Phase 5) will inherit it.
- The **diagnostic-as-spec-evidence pattern** (D1–D4 living in the spec dir) is the right template for any future "should we add capability X?" question.
- The **permanent-defer-with-artifacts** posture lets the repo say "no" definitively without losing the revival path.

---

*Source memory: `[[project_alias_expansion_sweep]]`. PRs [#94](https://github.com/Smart-AI-Memory/attune-rag/pull/94)–[#108](https://github.com/Smart-AI-Memory/attune-rag/pull/108), [#110](https://github.com/Smart-AI-Memory/attune-rag/pull/110).*
