# Spec: attune-rag 0.2.0 cut (post-W4.2 successor)

> **Status: scaffolding — not yet scoped; activates after Phase 4 W4.2.**

- **Owner:** Patrick
- **Created:** 2026-05-20
- **Target version:** 0.2.0
- **Predecessor spec:** [docs/specs/api-v0.2-public-surface/](../api-v0.2-public-surface/) (shape parent)
- **Activation gate:** [`docs/specs/downstream-validation/tasks.md`](../downstream-validation/tasks.md) W4.2 — "Verify all four weekly cadence reports show zero `Added`. If any reset happened, the freeze extends."

## Purpose

Produce the formal **0.2.0** cut of `attune-rag` once the Phase 4
cadence-clean gate (W4.2) holds. The Phase 3 spec
[`api-v0.2-public-surface`](../api-v0.2-public-surface/) **named** and
**snapshot-tested** the public surface at the symbol level — it landed
as 0.1.18 + 0.1.19 because every change was backward-compatible. This
spec is the SemVer-level event that ratifies that surface into a frozen
0.2.x line.

## What 0.2.0 means here

| Layer | Phase 3 (0.1.18 / 0.1.19) | 0.2.0 (this spec) |
|---|---|---|
| Public surface enumerated | ✓ | unchanged |
| Snapshot test gating drift | ✓ | unchanged |
| `docs/POLICY.md` published | ✓ | unchanged |
| SemVer commitment binding | honor-system (0.1.x is still-evolving per [POLICY.md §2](../../POLICY.md#2-semver-commitment)) | **binding from 0.2.0 onward** |
| `pyproject.toml` Development Status classifier | `3 - Alpha` | **stays at `3 - Alpha`** |
| Production/Stable claim | — | deferred to Phase 5 / v1.0.0 |

**0.2.0 is the SemVer freeze, not the stability claim.** The
classifier flip (Alpha → Production/Stable) is Phase 5's job — see
[ROADMAP-v1.md Phase 5](../ROADMAP-v1.md). 0.2.0 ratifies that "what
is PUBLIC today stays through every 0.2.z" per the policy already
documented in [POLICY.md §2](../../POLICY.md#2-semver-commitment).

## Activation criteria (W4.2 → this spec)

This spec promotes from "scaffolding" to "scoping" only after the
Phase 4 W4 deliverables land:

- W4.2 confirms four consecutive `cadence-week-{1,2,3,4}.md` reports
  show zero `### Added` under `[Unreleased]`.
- W4.3 writes `docs/specs/downstream-validation/exit-summary.md` with
  a recommendation to cut.
- W4.4 opens this spec for `/spec` scoping.

Calendar target for W4 close: ~2026-06-17 (W4.2 gate) / ~2026-06-20
(W4.3 exit-summary), per the Phase 4 calendar in
[ROADMAP-v1.md Phase 4](../ROADMAP-v1.md#phase-4--quality-burn-in--downstream-validation).

## Spec files

- [`requirements.md`](requirements.md) — entry gates inherited from
  Phase 4 (cadence-clean, perf baseline locked, downstream gate green,
  security findings clean, exit-summary written).
- [`design.md`](design.md) — SemVer interpretation, public surface
  map, classifier-flip placement (Phase 5, not here), deprecation
  policy inheritance.
- [`tasks.md`](tasks.md) — three milestones (M1 pre-cut audit, M2 cut
  & release, M3 post-cut verification) at the **scaffold** level; tasks
  are unscoped until the `/spec` pass after W4.2.

## Not in scope

- Adding any new public symbol. The surface was frozen at the symbol
  level in Phase 3; 0.2.0 ratifies it unchanged.
- Production/Stable classifier flip (Phase 5).
- `py.typed` marker (tracked as 0.3.0 candidate per Phase 3 follow-ups).
- Signature locking (Phase 5 candidate per Phase 4 out-of-scope list).
