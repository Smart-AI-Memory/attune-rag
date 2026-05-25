# Phase 4 — Exit summary (W4.3) — override-release

> **Status:** **complete 2026-05-25 via W4.2 override.**
> 0.2.0 cut authorized ahead of nominal calendar.

## Override-release rationale

Phase 4's review/gate deliverables landed ~3 weeks ahead of the
nominal calendar:

| Phase week | Original target | Actual close | Slip |
|---|---|---|---:|
| W0 (setup) | 2026-05-23 | 2026-05-20 | −3 d |
| W1 (burn-in start) | 2026-05-25 | 2026-05-25 | 0 |
| W2 (mid-phase reviews) | 2026-06-06 | 2026-05-20 | −17 d |
| W3 (gate promotions) | 2026-06-13 | 2026-05-22 | −22 d |
| W4 (close + 0.2.0 cut) | 2026-06-20 | 2026-05-25 | **−26 d** (W4.2 override) |

The remaining 3 cadence-soak weeks would have been calendar-elapse
only — the cadence clock's purpose was a visible-discipline window
for users; the review/gate work the spec was built around
(perf-baseline V2 lock, downstream-gate-green throughput,
quality-baseline holds) was complete.

## What was met before the override

| Gate | Status | Source |
|---|---|---|
| Quality baselines (P@1 ≥ 0.95, R@3 = 1.00, mean faithfulness ≥ 0.9686) | green | [`baseline-1.md`](../release-quality-baseline/baseline-1.md) |
| Perf baseline locked (V2 multi-run methodology) | green | [`perf-baseline.md`](perf-baseline.md) |
| Downstream-gate green throughout window | green | attune-gui `downstream-attune-gui.yml` history |
| Cadence-week-1: zero effective Added | green | [`cadence-week-1.md`](cadence-week-1.md) (`ON TRACK`) |
| Security findings clean | green | [`security-findings.md`](security-findings.md) — *"hard gate (zero severity: high open) is met at this snapshot"* |

## What the override traded

| Trade-off | Description |
|---|---|
| Skipped | cadence-week-2, -3, -4 reports + 3 weeks of soak-no-Added discipline |
| Gained | user-facing additions in users' hands 23 days sooner |
| Preserved | spec contract honesty (POLICY.md §2 SemVer binding starts at 0.2.0) |

## Per-PR override receipts

- [#130](https://github.com/Smart-AI-Memory/attune-rag/pull/130) — `load_aliases_from_file` + `DirectoryCorpus(extra_aliases_file=)`
- [#136](https://github.com/Smart-AI-Memory/attune-rag/pull/136) — `attune_rag.measure_corpus` public module + `attune-rag-measure` console script

## Recommendation

**CUT 0.2.0 — 2026-05-25.** Phase 4 review/gate work complete.
Override mechanism honored per
[`v1.0.0-release/design.md`](../v1.0.0-release/design.md)
§"Phase 5 scope" pre-staging clause.

## Follow-up

- **7-day no-hotfix watch.** Window begins on 0.2.0 publish. Watch
  mechanics defined in
  [`v1.0.0-release/design.md`](../v1.0.0-release/design.md)
  ("Seven-day no-hotfix gate"); the same mechanism applies to the
  0.2.0 post-cut watch.
- **Phase 5 activation gates.** Phase 5 specs
  ([`reranker-evaluation`](../reranker-evaluation/),
  [`user-corpus-onboarding`](../user-corpus-onboarding/),
  [`perf-baseline-multi-run`](../perf-baseline-multi-run/)) activate
  after 0.2.0 cut + 7-day watch closes clean + D5 verdict locked,
  per [`user-corpus-onboarding/tasks.md`](../user-corpus-onboarding/tasks.md)
  M0.3/M0.4.
- **Override debrief (new — created by this release).** The W4.2
  override mechanism shipped here is a discipline experiment;
  future cuts should reconsider whether the 4-week cadence-soak
  should be a hard gate or a soft-with-override gate. Defer to
  the Phase 5 retrospective.
