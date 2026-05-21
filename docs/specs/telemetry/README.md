# Spec: Query-distribution telemetry (attune-rag)

> **Status:** scoping memo — not executable. No `tasks.md`, no
> implementation plan, no code. This is a "should we and how" memo;
> a tasks file is added only if/when the proposal is approved.
> **Workstream:** post-v0.2.0, not calendar-gated. Sequenced after
> M13.4 upstream promotion lands.

## Problem

The [alias-expansion-sweep](../alias-expansion-sweep/) closed
paraphrased R@3 from 28.75% → 100% using a **synthetic** 80-query
paraphrased set ([`tests/golden/queries_paraphrased.yaml`](../../../tests/golden/queries_paraphrased.yaml)).
That set is a useful regression guard and was the right tool for the
sweep, but it is *my* construction — every paraphrase was authored
by me as a no-token-overlap variant of a baseline golden.

Two consequences:

1. **Future override authoring is judgment-driven, not
   evidence-driven.** When I add an entry to
   [`aliases_override.json`](../../../src/attune_rag/corpus/aliases_override.json),
   I am guessing at the shape of real developer queries from the
   shape of synthetic paraphrases I authored. There is no feedback
   loop from actual CLI usage.
2. **The paraphrase-vs-baseline ratio is unmeasured.** The sweep
   assumed paraphrased queries matter. They almost certainly do, but
   "almost certainly" is not "we have N weeks of data showing X% of
   live queries don't lexically match any template alias before
   expansion." The ratio is the lever for prioritizing future
   work (more aliases vs. embedding retriever vs. neither).

Real telemetry would convert both of these from intuition into data.

## Why this spec exists now, not later

Three timing reasons:

1. **The synthetic set is at 100% R@3.** Further authoring against
   it has diminishing returns. The next force multiplier on retrieval
   quality is real-distribution data, not more synthetic paraphrases.
2. **Before v0.2.0 cut is the right window.** Telemetry semantics
   are a public surface — once shipped, the wire format and the
   opt-in toggle live forever. Cheaper to scope the surface before
   the API freeze than to retrofit after.
3. **The override mechanism is now a stable tactical layer.** Any
   telemetry-driven additions to `aliases_override.json` flow
   through the same mechanism the sweep already validated. The
   telemetry layer doesn't need to invent a retrieval-modification
   path; it just needs to inform the existing one.

## Proposed defaults (full rationale in [design.md](design.md))

| Decision | Default | Alternatives considered |
|---|---|---|
| **Storage** | Local file (JSONL under `~/.attune-rag/telemetry/`) | Local + opt-in Redis; opt-in cloud upload — both deferred |
| **Privacy** | Hash query text by default (sha256); log full text only with second explicit toggle | Full text by default (rejected — preserves local-only intent but blocks future cloud upload story) |
| **Sampling** | Log every query with distinct-query dedup (counter per hashed query) | Append-only stream (rejected — unbounded growth for marginal info); configurable sample rate (rejected — premature for the CLI scale) |
| **Rollout** | Opt-in via `attune.config.json`, off by default, one-time first-run prompt on first `/rag` invocation | Env-var-only (rejected — discoverability); silent opt-in via config (rejected — no consent surface) |

Each of these has cheaper and more aggressive variants; see
[design.md](design.md) for the trade-off matrix and
[open-questions.md](open-questions.md) for the decisions deliberately
left unresolved at scoping time.

## Strict-dominance constraint

The [alias-expansion-sweep](../alias-expansion-sweep/) shipped 13
PRs without any baseline regression. That discipline carries
forward to telemetry. **Whatever telemetry ships:**

1. Must not regress paraphrased R@3 (current floor: 100%, watermark
   85%) — by definition, since the telemetry layer is observation,
   not retrieval modification.
2. Must not regress baseline P@1 / R@3 (current: 100% / 100%) —
   same reason.
3. **Must not regress retrieval latency.** This is the non-obvious
   one. A naive "log every query synchronously" implementation can
   move the perf-baseline gate.
   [`docs/specs/perf-baseline-multi-run/`](../perf-baseline-multi-run/)
   exists to make that gate trustworthy; telemetry must fit inside
   it, not break it. See [risks.md](risks.md) §1.

## What this spec is not

- **Not a build plan.** No `tasks.md`. Approval gate before any
  code path is written.
- **Not a cloud-upload story.** Local file is the only storage
  destination proposed for v1. Cloud upload is enumerated as a
  future extension in [open-questions.md](open-questions.md) §4
  but is explicitly out of scope here.
- **Not a replacement for the synthetic paraphrase set.** The 80-
  query regression suite stays as a guard. Telemetry data informs
  *new* alias authoring; it does not replace the existing test
  signal.
- **Not a v0.2.0 blocker.** Sequenced *before* v0.2.0 cut so the
  config surface lands inside the API window, but the actual
  emission path can ship in a v0.2.1 or v0.3.0 — the memo is about
  reserving the surface, not promising the implementation.

## Layout

- [design.md](design.md) — proposed mechanism for each decision
  axis, with the alternatives that were considered and rejected.
- [open-questions.md](open-questions.md) — decisions deliberately
  left unresolved at scoping time. Each one has a "we need to know
  X before we can resolve this" hook.
- [risks.md](risks.md) — strict-dominance risks (latency,
  retrieval-quality side effects), privacy-regret risks (the kind
  of telemetry decision you can't undo), and scope-creep risks.

When/if approved, this directory also gains:

- `requirements.md` — invariants the implementation must satisfy.
- `tasks.md` — the implementation milestones.

Until those exist, this directory is a memo, not a spec.

## Provenance

Sequenced after the alias-expansion-sweep arc closed on 2026-05-21.
The motivation comes directly from
[`project_alias_expansion_sweep.md`](../alias-expansion-sweep/) §"The
override mechanism is a tactical layer, not the long-term home" and
from the recognition that the synthetic paraphrase set, while
sufficient for the sweep, is not the long-term signal for which
aliases matter.
