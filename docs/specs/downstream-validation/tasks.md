# Spec: Downstream Validation (Phase 4)

## Phase 3: Tasks

**Status**: scoping (not started — pending spec approval)

### Implementation order

The phase splits into a **setup week** (lands the machinery) plus a
**four-week freeze** (the actual soak). The freeze clock starts when the
enforcer + downstream gates are both live + green on main.

| # | Task | Layer | Notes |
|---|------|-------|-------|
| **W0 — Setup (this week)** | | | |
| W0.1 | Write `scripts/check_freeze.py` — parses public `__all__` + CHANGELOG `### Added` diff; exits 0/1/2 mirroring `check_thresholds.py`. | attune-rag | Pure stdlib + pyyaml. Tests under `tests/unit/test_check_freeze.py`. |
| W0.2 | Write `.github/workflows/freeze.yml`. Runs `check_freeze.py` on every PR. Recognizes `freeze-override` label; requires `[Override-rationale]` block in PR body when override is set. | attune-rag | Wire after the script is green locally. |
| W0.3 | Write `scripts/measure_perf_baseline.py` — pytest-benchmark subprocess runner mirroring `measure_baseline_variance.py`. Targets `KeywordRetriever.retrieve`, `LLMReranker.rerank`, `RagPipeline.run`, `DirectoryCorpus.load`. | attune-rag | Tests under `tests/unit/test_measure_perf_baseline.py`. |
| W0.4 | Lock initial perf baseline: 30 runs on CI runner SKU `ubuntu-latest`. Commit `perf-baseline.md` + `perf-thresholds.json`. | attune-rag | Records hardware fingerprint + Python version. |
| W0.5 | Write `.github/workflows/perf.yml` — advisory mode (comment, don't fail) for W1–W2. | attune-rag | Promoted to gating in W3.1. |
| W0.6 | Write `scripts/changelog_cadence.py` — emits weekly Added/Fixed/Changed/Security breakdown. | attune-rag | Tests under `tests/unit/test_changelog_cadence.py`. |
| W0.7 | Write `.github/workflows/cadence-report.yml` — runs weekly, commits result to `docs/specs/downstream-validation/cadence-week-{N}.md`. | attune-rag | Cron schedule: `0 12 * * 1` (Mondays noon UTC). |
| W0.8 | Write `.github/workflows/downstream-attune-gui.yml` — calls attune-gui's test suite against the PR's HEAD SHA. Comment-only severity for W1–W2. | attune-rag | Promoted to blocking in W3.2. |
| W0.9 | Run attune-ai `/security-audit` repo-wide. Capture findings in `security-findings.md`. | attune-rag | One-shot; per-PR scans are W0.10. |
| W0.10 | Write `.github/workflows/security-scan.yml` — stdlib eval/exec/path-traversal/secrets/deserialization checks on PRs touching `editor/`, `providers/`, `dashboard/`, `cli.py`. | attune-rag | Per-PR; lighter than the full audit. |
| W0.11 | Triage W0.9 findings: fix-now / non-issue-with-rationale / Phase-5-ticket. | attune-rag | Gate: zero `severity: high` open at end of W0. |
| **W1–W4 — Burn-in** | | | |
| W1.1 | Tag start of freeze in CHANGELOG (e.g. `> Feature freeze begins YYYY-MM-DD.`). First weekly cadence report runs Monday W1. | attune-rag | Reset clock if any `Added` slips in. |
| W1.2 | Attune-gui M5.3 follow-up: run the editor + rag contract tests against installed 0.1.19 weekly. Track in cadence reports. | attune-gui | Surfaces regressions during the freeze. |
| W2.1 | Mid-phase `/deep-review` pass on `editor/`, `providers/`, `pipeline.py`, `retrieval.py`. | attune-rag | Findings either fixed or downgraded. |
| W2.2 | Run `attune-ai:performance_audit` on retrieval + reranker hot paths. Cross-check vs `perf-baseline.json`. | attune-rag | If audit finds optimizations, file as Phase 5 work; no Adding during freeze. |
| W3.1 | Promote perf gate from advisory → blocking for `KeywordRetriever.retrieve` and `RagPipeline.run`. Reranker stays advisory if its observed σ over W1–W2 is loose. | attune-rag | Verify by deliberately regressing retrieval; gate must fire. |
| W3.2 | Promote downstream gate (attune-gui) from comment → blocking. | attune-rag | |
| W3.3 | `/test-audit` to surface mocked-or-shallow tests; `/smart-test` to raise coverage on the public surface without adding new public symbols. | attune-rag | Coverage targets ~90% on `__all__` symbols, no Added. |
| W4.1 | End-of-phase `/deep-review`. | attune-rag | Final findings → fixed / non-issue / Phase-5. |
| W4.2 | Verify all four weekly cadence reports show zero `Added`. If any reset happened, the freeze extends. | attune-rag | Hard gate. |
| W4.3 | Write `docs/specs/downstream-validation/exit-summary.md` — perf baseline trend, security findings disposition, downstream-green record, recommendation for the 0.2.0 cut. | attune-rag | Hands off to the 0.2.0 successor spec. |
| W4.4 | If all gates green: open the 0.2.0 successor spec. | attune-rag | New spec at `docs/specs/api-v0.2.0-cut/` or similar. Mirrors api-v0.2-public-surface shape. |

### Dependencies

```
W0.1 → W0.2 (enforcer)
W0.3 → W0.4 → W0.5 (perf machinery)
W0.6 → W0.7 (cadence)
W0.8 (downstream — independent)
W0.9 → W0.10 → W0.11 (security)

All W0 must be green before W1 starts.
W3.1 needs ≥ 2 weeks of perf data (W1–W2 readings).
W4.4 only fires if W4.2 confirms the cadence gate held.
```

### Definition of done (Phase 4)

- [ ] Feature-freeze enforcer live, gating PRs since W1.
- [ ] Perf baseline locked and gating in W3.
- [ ] `security-findings.md` complete; zero `severity: high` open.
- [ ] Four consecutive weekly `cadence-week-{1,2,3,4}.md` reports with zero
      `Added` entries.
- [ ] Downstream gate (attune-gui) was blocking in W3–W4 and stayed green.
- [ ] `exit-summary.md` written, recommending the 0.2.0 cut.

### Risks & mitigations

| Risk | Mitigation |
|---|---|
| A security fix needs a new public symbol mid-phase. | `freeze-override` label + `[Security]` CHANGELOG section; document in `freeze-overrides.md`. Doesn't reset the cadence clock if the override is `Security`-scoped. |
| Perf baseline noise is too high to gate reliably. | Advisory mode (W1–W2) gives time to learn the noise profile before promoting. Reranker latency stays advisory if its observed σ stays loose — wall-clock noise to Anthropic is hard to remove. |
| attune-gui's main has unrelated failing tests during the phase. | Use `feature/attune-rag-0.2-editor-rename` as the downstream target until cowork-test failures + `test/pass-2-llm-discipline` work reconcile. Documented in requirements.md edge cases. |
| Calendar slips — the four-week soak runs into other commitments. | The clock is mechanical (cadence reports), not aspirational. If a regression resets the clock, that's data, not failure. Phase 5 doesn't depend on a calendar date. |
| Attune-ai workflows (`/security-audit`, `/deep-review`, etc.) burn API tokens. | Run them at scheduled cadence (W0.9, W2.1, W2.2, W4.1), not per-commit. Document token spend in `exit-summary.md` for the next phase's budgeting. |

### Out of scope (Phase 5 candidates)

- **Signature locking.** 0.2.0 is symbol-level freeze; signature contracts
  come at 1.0.0.
- **`py.typed` marker.** Track separately.
- **Coverage above 90%.** `/smart-test` targets ~90%; the remaining ~10%
  (CLI argv parsing, dashboard JS bridge, niche provider error paths) needs
  case-by-case attention that exceeds the freeze window.
- **`Production/Stable` classifier flip.** Phase 5 work.

### Calendar (proposed)

- **W0:** 2026-05-17 → 2026-05-23 (setup, this week)
- **W1:** 2026-05-24 → 2026-05-30 (freeze starts)
- **W2:** 2026-05-31 → 2026-06-06 (mid-phase reviews)
- **W3:** 2026-06-07 → 2026-06-13 (gates promote to blocking)
- **W4:** 2026-06-14 → 2026-06-20 (close-out, hand off to 0.2.0)
- **Slack week:** 2026-06-21 → 2026-06-27 (buffer for resets / 0.2.0 spec
  drafting)

Total: ~6 calendar weeks of attention budget, mapped onto the ~4 weeks of
soak time the roadmap estimates.
