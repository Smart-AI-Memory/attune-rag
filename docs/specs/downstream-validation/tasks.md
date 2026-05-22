# Spec: Downstream Validation (Phase 4)

## Phase 3: Tasks

**Status**: W0 complete 2026-05-20 (4 days ahead of the 2026-05-23 calendar end-date); W1 freeze clock starts 2026-05-20 against `attune-rag==0.1.21`.

### Freeze semantics — what counts as "Added"

The freeze is **symbol-level**, not quality-level. Internal improvements
ship freely; new public surface waits.

| Change | CHANGELOG section | Resets cadence clock? |
|---|---|---|
| Better internal heuristic, same signature (e.g. improved ranking math in `KeywordRetriever`) | `### Changed` | No |
| Internal reranker prompt tweak | `### Changed` | No |
| Bugfix to existing behavior | `### Fixed` | No |
| Security fix that requires a new public symbol | `### Security` + `freeze-override` label | No (Security-scoped override) |
| New constructor parameter, function, CLI flag, or `__all__` entry | `### Added` | **Yes** (also fails W0.1 enforcer) |

**Pattern for mid-freeze improvements that want a new knob:** land the
internal improvement during the freeze with the knob hard-coded to the new
default. Expose the knob via `### Added` in the post-freeze 0.2.0 cut.

### Implementation order

The phase splits into a **setup week** (lands the machinery) plus a
**four-week freeze** (the actual soak). The freeze clock starts when the
enforcer + downstream gates are both live + green on main.

| # | Task | Layer | Notes |
|---|------|-------|-------|
| **W0 — Setup (this week)** | | | |
| W0.1 | Write `scripts/check_freeze.py` — three checks: (a) public `__all__` diff per module, (b) CHANGELOG `### Added` diff under `[Unreleased]`, (c) `src/attune_rag/editor/template_schema.json` backward-compat diff (additionalProperties tightening, enum narrowing, required-field additions). Exits 0/1/2 mirroring `check_thresholds.py`. | attune-rag | Pure stdlib + pyyaml. Tests under `tests/unit/test_check_freeze.py` cover all three check axes + override path. |
| W0.2 | Write `.github/workflows/freeze.yml`. Runs `check_freeze.py` on every PR. Recognizes `freeze-override` label; requires `[Override-rationale]` block in PR body when override is set. | attune-rag | Wire after the script is green locally. |
| W0.3 | Write `scripts/measure_perf_baseline.py` — pytest-benchmark subprocess runner mirroring `measure_baseline_variance.py`. Targets `KeywordRetriever.retrieve`, `LLMReranker.rerank`, `RagPipeline.run`, `DirectoryCorpus.load`. Records **both wall-clock and CPU-time** per run via `time.perf_counter` + `time.process_time`. | attune-rag | Tests under `tests/unit/test_measure_perf_baseline.py`. Two-axis output increases test surface — cover wall-only, cpu-only, and both-present cases. |
| W0.4 | Lock initial perf baseline: 30 runs on CI runner SKU `ubuntu-latest`. Commit `perf-baseline.md` + `perf-thresholds.json` with **two threshold rows per benchmark** (`<bench>.wall` and `<bench>.cpu`). | attune-rag | Records hardware fingerprint + Python version. Token counts (reranker input/output) recorded in `perf-baseline.md` but no thresholds — track-only per spec decision. |
| W0.5 | Write `.github/workflows/perf.yml` — advisory mode (comment delta on both wall + CPU axes, don't fail) for W1–W2. | attune-rag | Promoted to gating in W3.1 (CPU-time only initially; wall-clock stays advisory through W3, re-evaluate in W4). |
| W0.6 | Write `scripts/changelog_cadence.py` — emits weekly Added/Fixed/Changed/Security breakdown. | attune-rag | Tests under `tests/unit/test_changelog_cadence.py`. |
| W0.7 | Write `.github/workflows/cadence-report.yml` — runs weekly, commits result to `docs/specs/downstream-validation/cadence-week-{N}.md`. | attune-rag | Cron schedule: `0 12 * * 1` (Mondays noon UTC). |
| W0.8 | Write `.github/workflows/downstream-attune-gui.yml` — calls attune-gui's test suite against the PR's HEAD SHA. Comment-only severity for W1–W2. | attune-rag | Promoted to blocking in W3.2. |
| W0.9 | Run attune-ai `/security-audit` repo-wide. Capture findings in `security-findings.md`. | attune-rag | One-shot; per-PR scans are W0.10. |
| W0.10 | Write `.github/workflows/security-scan.yml` — stdlib eval/exec/path-traversal/secrets/deserialization checks on PRs touching `editor/`, `providers/`, `dashboard/`, `cli.py`. | attune-rag | Per-PR; lighter than the full audit. |
| W0.11 | Triage W0.9 findings: fix-now / non-issue-with-rationale / Phase-5-ticket. | attune-rag | Gate: zero `severity: high` open at end of W0. |
| **W1–W4 — Burn-in** | | | |
| W1.1 | ✅ done 2026-05-20 — freeze marker added to `[Unreleased]` in CHANGELOG against `attune-rag==0.1.21`. First weekly cadence report fires Monday 2026-05-25 (next `cadence-report.yml` cron). | attune-rag | Reset clock if any `Added` slips in. |
| W1.2 | Attune-gui M5.3 follow-up: run the editor + rag contract tests against the latest published attune-rag weekly. Track in cadence reports. **Week-1 (2026-05-20) against 0.1.22**: 82/82 rag+editor contract tests pass (`test_contract_attune_rag`, `test_services_rag_pipeline`, `test_editor_template{,_routes}`, `test_editor_schema`, `test_editor_corpus`, `test_editor_session`, `test_rag_workspace`); full sidecar suite 489 passed / 3 failed / 1 xfailed — the 3 failures (`test_cowork_specs::test_specs_root_env_var_wins`, `::test_specs_root_walks_up_from_cwd`, `test_cowork_templates::test_templates_staleness_thresholds`) reproduce against the pinned 0.1.18 and are unrelated to the rag bump (in-flight work on `feature/attune-rag-0.2-editor-rename`). | attune-gui | Surfaces regressions during the freeze. |
| W2.1 | ✅ done 2026-05-20 — three-pass synthesis at [`w2-deep-review.md`](./w2-deep-review.md). 2,774 LOC reviewed; **0 blocking findings**, 1 INFO security-docstring fix, ~90/100 aggregate quality, **89.78 % line+branch coverage** across all four targets (already at W3.3's ~90 % target in aggregate). Six concrete next-week tests proposed; four Phase-5 backlog items extracted. | attune-rag | Findings either fixed or downgraded. |
| W2.2 | ✅ done 2026-05-20 — `attune-ai:performance_audit` on retrieval + reranker hot paths, cross-checked vs [`perf-baseline.md`](./perf-baseline.md). Report at [`w2-perf-audit.md`](./w2-perf-audit.md). 4 micro-opt candidates on retrieval (P1–P4); 0 on reranker (latency dominated by Anthropic round-trip). All filed as Phase-5 tickets in [`phase-5-backlog/items.md` § Perf](../phase-5-backlog/items.md#perf-w22-perf-audit) — no Adding during freeze. | attune-rag | W2 gate: green. |
| W3.1 | ✅ done 2026-05-22 — `format_perf_delta.py` gained `--gate-metric`; `perf.yml` switched from `--advisory` to two `--gate-metric` flags (`keyword_retriever_retrieve.cpu`, `rag_pipeline_run.cpu`) and now propagates the script exit code. Wall-clock + reranker + `directory_corpus_load` stay advisory through W3 (re-evaluate W4). Verified by three-way smoke test against real `perf-thresholds.json`: clean → exit 0; gated regression → exit 1 with ⛔ icon + "REGRESSION (gating)" title; advisory-only regression → exit 0 with "possible regression" title. 22/22 tests green (5 new). | attune-rag | Verify by deliberately regressing retrieval CPU-time; gate must fire. Wall-clock noise tolerated; CPU-time is the actionable axis. |
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
