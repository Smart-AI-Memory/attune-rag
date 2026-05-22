# Spec: attune-rag 0.2.0 cut — Tasks

> **Status:** **scoped 2026-05-22** — executable when Phase 4 W4
> closes and the entry gates in
> [`requirements.md`](requirements.md#entry-gates-inherited-from-phase-4)
> are green. The spec was authored decision-complete in
> [`design.md`](design.md); this pass adds the M0 entry-gate
> verification milestone and confirms the decisions.

## Scoping decisions confirmed (2026-05-22)

| # | Decision | Source |
|---|---|---|
| 1 | 0.2.0 = **SemVer freeze only**, not Production/Stable claim | [design.md "SemVer interpretation"](design.md#semver-interpretation-for-this-cut) |
| 2 | Classifier **stays at `Development Status :: 3 - Alpha`** through 0.2.x; flip is Phase 5's job | [design.md "Classifier flip (where it does NOT happen)"](design.md#classifier-flip-where-it-does-not-happen) |
| 3 | **No symbol additions** in the cut PR — `### Changed` only | [requirements.md Process](requirements.md#process) + [README out-of-scope](README.md#not-in-scope) |
| 4 | The 0.2.0 cut ratifies the surface locked by [`tests/unit/test_api_surface.py`](../../../tests/unit/test_api_surface.py) **as-is**; no re-audit | [design.md "Public surface map"](design.md#public-surface-map-reference) |
| 5 | POLICY.md gets a single copy edit: "**formal SemVer commitments take effect with 0.2.0** (gated on Phase 4…)" → "took effect at 0.2.0 on `<date>`" | [design.md "Deprecation policy"](design.md#deprecation-policy-inheritance-not-new) |
| 6 | Tag + PyPI publish use the **existing release workflow** + `/attune-release-check` skill + manual `pypi` environment approval | [requirements.md Process](requirements.md#process) |
| 7 | Deprecation shims (`editor/_{rename,schema,lint,autocomplete,references}.py`) **stay through 0.2.x**; removable at 0.3.0 | [design.md "Deprecation shims"](design.md#public-surface-map-reference) |
| 8 | 7-day no-hotfix watch starts at cut day; any hotfix resets Phase 5 entry — acceptable, tracked but not contractually preventable | [design.md risks](design.md#risks--mitigations-sketch) row 4 |

## Phase 3: Tasks

### Implementation order

Four milestones (M0 added at scoping for entry-gate verification;
M1-M3 are the original work shape). Mirrors
[`api-v0.2-public-surface/tasks.md`](../api-v0.2-public-surface/tasks.md)
at lower granularity since the substantive design work happened in
Phase 3 already.

| # | Task | Layer | Status | Notes |
|---|------|-------|--------|-------|
| **M0 — Entry-gate verification** | | | | |
| M0.1 | Verify cadence-clean: read all four `docs/specs/downstream-validation/cadence-week-{1,2,3,4}.md` reports and confirm each ends with `**Status:** ON TRACK`. (W4.2 hard gate.) | attune-rag | pending | If any reads RESET, M0 fails and the freeze extends — do not proceed to M1. |
| M0.2 | Verify perf baseline holds: `docs/specs/downstream-validation/perf-thresholds.json` is locked at N=50, σ=3.0, `include_llm: true`. Confirm no perf-gate red on `main` during the four-week soak (gh run list --workflow perf.yml --status failure --created '>YYYY-MM-DD'). | attune-rag | pending | Inherited from PR #77. Independent of `perf-baseline-multi-run` — that re-locks post-0.2.0. |
| M0.3 | Verify downstream gate stayed green: `gh run list --workflow downstream-attune-gui.yml --branch main --status failure --created '>YYYY-MM-DD'` returns no rows for the W3.2 → W4.4 window. | attune-rag | pending | The blocking-mode gate from W3.2. |
| M0.4 | Verify security findings clean: `docs/specs/downstream-validation/security-findings.md` has zero open `severity: high` items at W4 close. Any open items are MEDIUM/LOW with rationale or Phase-5 tickets. | attune-rag | pending | Inherited from W0.11 + any new findings during W1-W4. |
| M0.5 | Verify exit-summary written: `docs/specs/downstream-validation/exit-summary.md` exists (Phase 4 W4.3 deliverable), summarizes perf trend + security disposition + downstream-green record, and **explicitly recommends cutting 0.2.0**. | attune-rag | pending | The summary's recommendation IS the trigger for M1. |
| **M1 — Pre-cut audit** | | | | |
| M1.1 | Re-run `tests/unit/test_api_surface.py` against current `main`; verify zero drift from the frozen surface tables in [`api-v0.2-public-surface/design.md`](../api-v0.2-public-surface/design.md). | attune-rag | pending | Exit code 0; no skips. |
| M1.2 | Re-grep attune-gui (gating downstream per ROADMAP Decision 2) for `attune_rag.*` imports; verify every consumed symbol is in the frozen `__all__`. | attune-rag | scaffold | unscoped |
| M1.3 | Refresh README "Public API" section if any wording is stale; refresh [`docs/POLICY.md`](../../POLICY.md) to replace "**formal SemVer commitments take effect with 0.2.0** (gated on Phase 2…)" with "took effect at 0.2.0 on `<date>`". | attune-rag | scaffold | unscoped |
| **M2 — Cut & release** | | | | |
| M2.1 | Bump `pyproject.toml` version `0.1.z → 0.2.0`. Classifier stays at `Development Status :: 3 - Alpha`. | attune-rag | scaffold | unscoped |
| M2.2 | Sync `src/attune_rag/__init__.py` `__version__` to `0.2.0`. | attune-rag | scaffold | unscoped |
| M2.3 | Roll CHANGELOG `[Unreleased]` into `[0.2.0] — <date>`. Section ordering per existing CHANGELOG convention. No `### Added` entries (would fail the W0.1 freeze enforcer on the cut PR itself). | attune-rag | scaffold | unscoped |
| M2.4 | Tag + PyPI publish via the existing release workflow (`/attune-release-check` skill + manual `pypi` environment approval). | attune-rag | scaffold | unscoped |
| **M3 — Post-cut verification** | | | | |
| M3.1 | Verify `pip install attune-rag==0.2.0` from PyPI in a clean venv; smoke-test the public surface import-by-import. | attune-rag | scaffold | unscoped |
| M3.2 | In attune-gui: bump the floor pin from `attune-rag>=0.1.18,<0.2` to `attune-rag>=0.2.0,<0.3`; run the contract test suite (`test_contract_attune_rag.py`) against the published 0.2.0 wheel. | attune-gui | scaffold | unscoped |
| M3.3 | Start the 7-day no-hotfix watch. Any hotfix during the window blocks Phase 5 (v1.0.0) entry per [ROADMAP-v1.md Phase 5 Gate](../ROADMAP-v1.md#phase-5--100-release). | attune-rag | scaffold | unscoped |

### Dependencies

```
M0.1 → M0.2 → M0.3 → M0.4 → M0.5   (entry gates; M0.5 includes the explicit
                                    exit-summary "recommend cutting 0.2.0")

M0.5 → M1.1 → M1.2 → M1.3          (pre-cut audit)

M1.3 → M2.1 → M2.2 → M2.3 → M2.4   (cut + release; M2.4 is the PyPI publish)

M2.4 → M3.1 → M3.2 → M3.3          (post-cut verification + 7-day watch start)
```

M0 is the hard gate from Phase 4. If any M0 check fails the freeze
extends and the spec stays in "scoped, waiting" until the gate
re-greens.

### Definition of done

- [ ] 0.2.0 on PyPI.
- [ ] `tests/unit/test_api_surface.py` green against the published wheel.
- [ ] attune-gui pinned at `attune-rag>=0.2.0,<0.3`; its contract test
      passes against the published wheel.
- [ ] CHANGELOG `[0.2.0]` section dated; `[Unreleased]` reset to empty.
- [ ] `pyproject.toml` Development Status classifier still
      `3 - Alpha` (Phase 5 owns the flip).
- [ ] [`docs/POLICY.md`](../../POLICY.md) updated to past-tense on the
      "takes effect at 0.2.0" clauses.
- [ ] 7-day no-hotfix watch started; tracked in `exit-summary.md`
      (Phase 4 W4.3) or a new file at this spec's root.

### Risks & mitigations

Locked in [`design.md`](design.md#risks--mitigations-sketch):
surface-drift catch via M1.1, downstream-symbol catch via M1.2,
pin-syntax breakage caught via M3.2, hotfix-in-watch acknowledged
as acceptable Phase-5-reset risk.

### Out of scope (deferred)

- **Production/Stable classifier flip.** Phase 5 (v1.0.0).
- **Adding any new public symbol.** Goes through deprecation policy at
  0.3.0 or later.
- **`py.typed` marker.** Tracked as a 0.3.0 candidate per Phase 3
  follow-ups in [`api-v0.2-public-surface/tasks.md`](../api-v0.2-public-surface/tasks.md).
- **Signature locking.** Phase 5 candidate per Phase 4 out-of-scope.
- **Re-auditing the surface.** Audit was Phase 3 work; this spec
  ratifies the verdict, it does not re-litigate it.

### Follow-ups (post-0.2.0)

- Hand off to Phase 5 once the 7-day no-hotfix watch clears (per
  [ROADMAP-v1.md Phase 5 gate](../ROADMAP-v1.md#phase-5--100-release)).
  Phase 5 specs are pre-staged: [`reranker-evaluation/`](../reranker-evaluation/),
  [`user-corpus-onboarding/`](../user-corpus-onboarding/),
  [`perf-baseline-multi-run/`](../perf-baseline-multi-run/).
- Any post-freeze items waiting on the cut: land them as new
  `[Unreleased]` entries on top of the rolled `[0.2.0]` section
  (NOT inside `[0.2.0]` — the cut is a snapshot, not an aggregator).
- D5 (reranker-evaluation) executes first in Phase 5 to inform the
  rerank default for user-corpus-onboarding M1.
