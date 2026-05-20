# Spec: attune-rag 1.0.0 release

> **Status: scaffolding — not yet scoped; activates after the 0.2.0
> cut closes cleanly + the 7-day post-1.0.0 no-hotfix gate is
> achievable (per [ROADMAP-v1.md](../ROADMAP-v1.md) Phase 5 gate).**

- **Owner:** Patrick
- **Created:** 2026-05-20
- **Target version:** 1.0.0
- **Roadmap phase:** [Phase 5](../ROADMAP-v1.md#phase-5--100-release)
- **Shape parent:** [docs/specs/api-v0.2-public-surface/](../api-v0.2-public-surface/)
  — this spec mirrors its `requirements.md` / `design.md` / `tasks.md`
  layout.

## Purpose

Produce the formal **attune-rag 1.0.0** release. 1.0.0 is a
**stability claim**, not new public surface:

- The public surface was frozen at 0.2.0 (see
  [api-v0.2-public-surface/](../api-v0.2-public-surface/)).
- Phase 4 ([downstream-validation/](../downstream-validation/)) is
  the burn-in that earns the stability claim.
- Phase 5 takes the claim and ratifies it on the package itself:
  flip the classifier `3 - Alpha` → `5 - Production/Stable` in
  [pyproject.toml](../../../pyproject.toml), publish a support
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

## Inherited entry-gates

Phase 5 cannot start scoping until all of the following are true.
Each gate is owned upstream; this spec only checks them.

- [ ] **Phase 4 closed.** [`docs/specs/downstream-validation/`](../downstream-validation/)
      `exit-summary.md` (W4.3) exists and recommends the 0.2.0 cut.
- [ ] **0.2.0 cut closed.** [`docs/specs/api-v0.2.0-cut/`](../api-v0.2.0-cut/)
      (the W4.4 successor spec, scaffolded in
      [#83](https://github.com/Smart-AI-Memory/attune-rag/pull/83))
      has been scoped, executed, and shipped. PyPI shows
      `attune-rag==0.2.0`. The classifier flip is *not* part of
      0.2.0 (that's this spec's job — see
      [api-v0.2.0-cut/README.md](../api-v0.2.0-cut/README.md)
      "What 0.2.0 means here").
- [ ] **0.2.0 has soaked.** N days on PyPI with zero hotfix releases
      (N is a placeholder — pinned during the formal scoping pass).
- [ ] **attune-gui is pinned to 0.2.x** and reports clean across
      one full weekly downstream-validation cycle on the new pin.
- [ ] **Perf baseline holds.** `perf-thresholds.json` from Phase 4 is
      not breached by 0.2.0.
- [ ] **No open severity-high security findings.** Phase 4's
      `security-findings.md` is at zero open.

If any gate is missing, this spec stays in scaffolding state.

## Files

| File | Purpose |
|---|---|
| [README.md](README.md) | This one-pager. |
| [requirements.md](requirements.md) | Entry-gate requirements + scope/non-goals. |
| [design.md](design.md) | What 1.0.0 means; classifier flip; support window; 1.x deprecation policy; backlog disposition. |
| [tasks.md](tasks.md) | Unscoped milestone skeleton (M1–M4). Promotes via `/spec` after 0.2.0 closes. |

## See also

- [docs/specs/ROADMAP-v1.md](../ROADMAP-v1.md) — Phase 5 section is
  the source of truth for outcome, gate, and attune-ai workflows.
- [docs/specs/api-v0.2-public-surface/](../api-v0.2-public-surface/)
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
