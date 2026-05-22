# Phase 4 — Exit summary (W4.3)

> **Status:** **skeleton 2026-05-22** — placeholders for the numbers
> that fill in as W4 closes. Per
> [`tasks.md`](tasks.md) W4.3: "Write
> `docs/specs/downstream-validation/exit-summary.md` — perf baseline
> trend, security findings disposition, downstream-green record,
> recommendation for the 0.2.0 cut. Hands off to the 0.2.0 successor
> spec."
>
> **Drafted as a skeleton ahead of W4 close so the hand-off shape is
> reviewable before the numbers land.** When W4 closes, the
> `<TODO>` placeholders get filled in with the measured values; the
> structure stays as-is. Status banner flips to `complete YYYY-MM-DD`
> when the recommendation is locked.

## TL;DR (one paragraph for the 0.2.0 cut PR body)

`<TODO at W4 close: 2-3 sentences. Template:>`

> Phase 4 closed clean: <N> consecutive cadence weeks at `ON TRACK`,
> perf baseline held (no gated regressions), attune-gui downstream
> blocking gate stayed green, zero open `severity: high` security
> findings. The exit recommendation is to **cut 0.2.0** on
> `<date>` per [`api-v0.2.0-cut/tasks.md`](../api-v0.2.0-cut/tasks.md)
> M1.

## Phase 4 calendar (filled at close)

| Phase week | Original target | Actual close | Slip |
|---|---|---|---:|
| W0 (setup) | 2026-05-23 | 2026-05-20 | **−3 d** |
| W1 (burn-in start) | 2026-05-25 | 2026-05-25 | 0 |
| W2 (mid-phase reviews) | 2026-06-06 | 2026-05-20 | **−17 d** (W2.1+W2.2 done early) |
| W3 (gate promotions) | 2026-06-13 | 2026-05-22 | **−22 d** (W3.1+W3.2+W3.3 done early) |
| W4 (close + 0.2.0 cut) | 2026-06-20 | `<TODO>` | `<TODO>` |

> **Calendar observation:** Phase 4 ran ~2.5–3 weeks ahead of the
> nominal schedule because W2/W3 work landed eagerly while the
> cadence clock continued its mechanical 4-week elapse. The
> calendar floor remains the cadence-week count, not the
> review/gate work.

## Cadence reports (W4.2 hard gate)

| Week | Window | Status | Cited file |
|---|---|---|---|
| 1 | `<TODO 2026-05-19 → 2026-05-25>` | `<TODO ON TRACK / RESET>` | [`cadence-week-1.md`](cadence-week-1.md) |
| 2 | `<TODO 2026-05-26 → 2026-06-01>` | `<TODO>` | [`cadence-week-2.md`](cadence-week-2.md) |
| 3 | `<TODO 2026-06-02 → 2026-06-08>` | `<TODO>` | [`cadence-week-3.md`](cadence-week-3.md) |
| 4 | `<TODO 2026-06-09 → 2026-06-15>` | `<TODO>` | [`cadence-week-4.md`](cadence-week-4.md) |

**W4.2 gate verdict:** `<TODO PASS / FAIL>`. If any week `RESET`,
this row records which week and which `### Added` entry triggered
it; the freeze extends another 4 weeks from that release.

**Override-flagged Added entries observed during W1-W4:**
`<TODO list any, with rationale. Expected: zero, since 0.1.22's
MIN_ALIAS_OVERLAP override is pre-freeze and won't appear in any
cadence-week window starting 2026-05-19 — but the cadence script's
override-detection logic (per PR #119 / cadence-week-schema.md)
handles them correctly if they do.>`

## Perf baseline trend

Locked baseline at W4 start (post PR #77): `mean / σ / N` of
`rag_pipeline_run.cpu`, `keyword_retriever_retrieve.cpu`,
`llm_reranker_rerank.{cpu,wall}`, `directory_corpus_load.{cpu,wall}`
per [`perf-thresholds.json`](perf-thresholds.json).

| Metric | Baseline (W4 start) | Trend during W1-W4 | Notes |
|---|---|---|---|
| `keyword_retriever_retrieve.cpu` (gated W3.1) | `<TODO>` | `<TODO>` | |
| `rag_pipeline_run.cpu` (gated W3.1) | `<TODO>` | `<TODO>` | |
| `llm_reranker_rerank.wall` (advisory) | `<TODO>` | `<TODO>` | Inter-Anthropic variance dominant; per `perf-baseline-multi-run` spec, multi-run methodology re-locks post-cut. |
| `keyword_retriever_retrieve.wall` (advisory) | `<TODO>` | `<TODO>` | Wall-clock promotion re-evaluated in W4 per tasks.md W3.1. |
| `rag_pipeline_run.wall` (advisory) | `<TODO>` | `<TODO>` | Same. |
| `directory_corpus_load.{cpu,wall}` (advisory) | `<TODO>` | `<TODO>` | |

**Gated regressions during W1-W4:** `<TODO count + per-PR list. Expected: zero, since W3.1 only promoted to blocking on 2026-05-22 and no PRs landed afterward that touched the gated hot paths.>`

**Wall-clock promotion recommendation for W4:** `<TODO PROMOTE / STAY ADVISORY based on observed σ during W3>`. If the wall-clock σ tightened materially during W3, recommend promotion; otherwise stay advisory and let `perf-baseline-multi-run` (Phase 5) re-evaluate after the multi-run methodology lock.

## Security findings disposition

From [`security-findings.md`](security-findings.md):

| Severity | Open at W4 close | Disposition |
|---|---:|---|
| HIGH | `<TODO 0 / N>` | `<TODO fix-now / non-issue-with-rationale / Phase-5-ticket>` |
| MEDIUM | `<TODO>` | `<TODO>` |
| LOW | `<TODO>` | `<TODO>` |
| INFO | `<TODO>` | `<TODO>` |

**Gate:** zero open HIGH (per W0.11). `<TODO PASS / FAIL>`.

**New findings during W1-W4 (per-PR security-scan.yml runs):**
`<TODO count + summary. Expected: low single digits, all triaged in-PR.>`

## Downstream gate (attune-gui)

Per [`tasks.md` W3.2](tasks.md):

| Metric | Value |
|---|---|
| Window when blocking | `<TODO W3.2 promotion date → W4 close>` |
| Total PRs gated | `<TODO N>` |
| Red runs in window | `<TODO 0 / N>` |
| Recovery time on red (if any) | `<TODO median minutes>` |
| Last green run on `main` | `<TODO run-id + timestamp>` |

**Gate:** zero unresolved red runs on `main` at W4 close.
`<TODO PASS / FAIL>`.

## W2 review hand-off

| Pass | Output | Headline |
|---|---|---|
| W2.1 (deep-review) | [`w2-deep-review.md`](w2-deep-review.md) | 2,774 LOC reviewed; 0 blocking findings; 89.78% coverage (W3.3 lifted to 90.04%). |
| W2.2 (perf audit) | [`w2-perf-audit.md`](w2-perf-audit.md) | 4 micro-opt candidates on retrieval (P1-P4); 0 on reranker; all filed as [`phase-5-backlog/items.md`](../phase-5-backlog/items.md) entries. |

**Follow-up items still pending at W4 close:**
`<TODO list any W2 items that weren't closed during W3 or in this exit summary.>`

## Recommendation

**`<TODO at W4 close: one of CUT-0.2.0 / EXTEND-FREEZE.>`**

Rationale: `<TODO 1-2 sentences citing the gates above.>`

Hand-off:
- If CUT-0.2.0: open the cut PR per
  [`api-v0.2.0-cut/tasks.md`](../api-v0.2.0-cut/tasks.md) M1 (the
  spec is already scoped + executable; M0 entry-gate checks
  mechanically verify this exit-summary's conclusions).
- If EXTEND-FREEZE: identify the reset trigger, document the new
  4-week cadence start, return to W1.

## Phase 5 spec status (informational hand-off)

All three Phase 5 specs are pre-staged + scoped as of 2026-05-22:

| Spec | Status | Executes |
|---|---|---|
| [`reranker-evaluation/`](../reranker-evaluation/) (D5) | scoped | Phase 5 D5 |
| [`user-corpus-onboarding/`](../user-corpus-onboarding/) | scoped | Phase 5 M1-M4; M1 inherits D5's verdict |
| [`perf-baseline-multi-run/`](../perf-baseline-multi-run/) | scoped | Phase 5 (multi-run noise methodology + σ rollback) |

The 0.2.0 cut spec ([`api-v0.2.0-cut/`](../api-v0.2.0-cut/)) is
also scoped + executable; W4.4 of this phase is a no-op.

## Token spend during Phase 4

Per [risks.md](requirements.md) and tasks.md's "Attune-ai workflows
burn API tokens" mitigation:

| Workflow | Runs in W0-W4 | Estimated token spend |
|---|---:|---|
| `/security-audit` (W0.9 + W2.1 deep-review chain) | `<TODO>` | `<TODO>` |
| `/deep-review` (W2.1 + W4.1) | `<TODO>` | `<TODO>` |
| `attune-ai:performance_audit` (W2.2) | 1 | `<TODO>` |
| Per-PR `security-scan.yml` (W0.10 → present) | `<TODO>` | `<TODO>` (stdlib-only; negligible) |

Phase 5 budget projection: D5's reranker measurement is the main
new Anthropic spend (~$0.30 per lock, per the D5 spec); other Phase 5
work is stdlib.

---

*Drafted 2026-05-22 as a skeleton; `<TODO>` placeholders fill in as
W4 closes. Update the status banner to `complete YYYY-MM-DD` once
the recommendation is locked.*
