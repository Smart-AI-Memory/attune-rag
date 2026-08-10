# Spec: attune-rag 1.0.0 release

> **Status: approved 2026-08-09** — scoped and ratified by Patrick in
> the same session (D1–D10 locked, including D7 = Option A). Execution
> is unblocked; first task is M0.5. The scoping audit found three stale
> premises in the scaffolding (0.2.0-as-parent, an Alpha classifier
> that never existed, and consumer pins that all exclude 1.0.0) —
> decisions.md §0 records them; the files here are corrected inline.

- **Owner:** Patrick
- **Created:** 2026-05-20
- **Scoped:** 2026-08-09
- **Target version:** 1.0.0
- **Roadmap phase:** [Phase 5](../ROADMAP-v1.md#phase-5--100-release)
- **Shape parent:** [docs/specs/api-v0.2-public-surface/](../archive/api-v0.2-public-surface/)
  — this spec mirrors its `requirements.md` / `design.md` / `tasks.md`
  layout.

## Purpose

Produce the formal **attune-rag 1.0.0** release. 1.0.0 is a
**stability claim**, not new public surface:

- The public surface was frozen at 0.2.0 (see
  [api-v0.2-public-surface/](../archive/api-v0.2-public-surface/)).
- Phase 4 ([downstream-validation/](../downstream-validation/)) is
  the burn-in that earns the stability claim.
- Phase 5 takes the claim and ratifies it on the package itself:
  flip the classifier `4 - Beta` → `5 - Production/Stable` in
  [pyproject.toml](../../../pyproject.toml) (corrected — the package
  was never `3 - Alpha`; see [decisions.md](decisions.md) D2), widen
  consumer pins so 1.0.0 is installable (M0, D3), publish a support
  window + 1.x deprecation policy in [docs/POLICY.md](../../POLICY.md),
  cut and tag 1.0.0, watch for hotfixes for seven days.

## What this spec is not

- **Not new public API.** Any surface addition lands as 1.0.x or
  1.1.0 *after* the cut, under the policy this spec ratifies.
- **Not a perf or eval re-baseline.** Phase 1 + Phase 4 own those
  numbers; Phase 5 inherits them.
- **Not a Phase-4 retrospective.** That belongs in Phase 4's
  `exit-summary.md` (see W4.3 of
  [downstream-validation/tasks.md](../downstream-validation/tasks.md)).

## Inherited entry-gates — audited closed 2026-08-09

Each gate is owned upstream; this spec only checks them. The full
audit with evidence is in [requirements.md](requirements.md)
"Entry gates".

- [x] **Phase 4 closed.** `exit-summary.md` (W4.3) exists and
      recommends the cut.
- [x] **Predecessor cut closed.** 0.2.0 shipped 2026-05-25; the line
      has since advanced to **0.9.0** — the cut's parent per
      [decisions.md](decisions.md) D1.
- [x] **Soak.** N pinned at 14 days (D4); 0.9.0 has 22 with zero
      hotfixes.
- [x] **Downstream clean** on current pins.
- [x] **Perf baseline** — methodology holds; numbers re-measured at
      M1.0 (D8).
- [x] **No open severity-high security findings.**
- [ ] **Consumer pins admit 1.0.0** — ❌ **the one open gate**, found
      during scoping (D3). All consumers cap below 1.0; milestone M0
      fixes this before the cut, starting with M0.5 (the ratified
      `DirectoryCorpus` override, D7 = Option A) so consumers validate
      against the final surface.

## Files

| File | Purpose |
|---|---|
| [README.md](README.md) | This one-pager. |
| [decisions.md](decisions.md) | **Scoping decisions D1–D10 (2026-08-09)** — read first; binding on the cut PR. |
| [requirements.md](requirements.md) | Entry-gate audit + scope/non-goals. |
| [design.md](design.md) | What 1.0.0 means; classifier flip; support window; 1.x deprecation policy; M0 pin-widening design. |
| [tasks.md](tasks.md) | Work-tracker (M0–M4). Scoped 2026-08-09. |

## See also

- [docs/specs/ROADMAP-v1.md](../ROADMAP-v1.md) — Phase 5 section is
  the source of truth for outcome, gate, and attune-ai workflows.
- [docs/specs/api-v0.2-public-surface/](../archive/api-v0.2-public-surface/)
  — what 1.0.0 ratifies as the stable surface (symbol-level lock).
- [docs/specs/api-v0.2.0-cut/](../api-v0.2.0-cut/) — the SemVer-level
  0.2.0 cut (W4.4 successor spec); ships *before* 1.0.0 and is one
  of this spec's entry gates.
- [docs/specs/downstream-validation/](../downstream-validation/)
  — the burn-in that feeds W4.3's `exit-summary.md` into Phase 5.
- [docs/specs/perf-baseline-multi-run/](../perf-baseline-multi-run/)
  — multi-run perf-baseline methodology fix (promoted from
  phase-5-backlog M1 in
  [#86](https://github.com/Smart-AI-Memory/attune-rag/pull/86));
  Phase 5 deliverable in its own right, ships *outside* this cut spec.
- [docs/specs/phase-5-backlog/](../phase-5-backlog/) — deferred-
  during-freeze items triaged into / out of Phase 5 at scoping time
  (see [items.md](../phase-5-backlog/items.md)). Item M1 has
  already promoted to its own spec (above).
- [docs/POLICY.md](../../POLICY.md) — the public-API + deprecation
  policy that Phase 5 extends with the 1.x support window.
