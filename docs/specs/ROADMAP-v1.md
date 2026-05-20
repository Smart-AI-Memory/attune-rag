# attune-rag → 1.0 roadmap

Rolling tracker for the path from v0.1.17 (Alpha) to v1.0.0
(Production/Stable). One spec per phase. Update the
**Status** and **Next action** rows at the end of every
working session so state is recoverable cold from disk.

| Field | Value |
|---|---|
| Current version | 0.1.22 (0.1.19 shipped Phase 2; 0.1.20–0.1.22 shipped intra-freeze `### Changed` / `### Fixed` work — see PRs [#75](https://github.com/Smart-AI-Memory/attune-rag/pull/75), [#77](https://github.com/Smart-AI-Memory/attune-rag/pull/77), and the selection-criteria-robustness spec) |
| Target version | 1.0.0 |
| Current phase | Phase 4 — **W0 complete 2026-05-20** (3 days ahead of 2026-05-23 target); **W1 freeze clock live** against `attune-rag==0.1.21`; **W2 closed 2026-05-20** (W2.1 deep-review at [`w2-deep-review.md`](downstream-validation/w2-deep-review.md), W2.2 perf audit at [`w2-perf-audit.md`](downstream-validation/w2-perf-audit.md)); **W3.2 downstream gate already promoted to blocking** (PR [#81](https://github.com/Smart-AI-Memory/attune-rag/pull/81)); **Phase 3 M5.3 closed 2026-05-20** ([attune-gui#38](https://github.com/Smart-AI-Memory/attune-gui/pull/38) — pin bumped to 0.1.22 with 82/82 rag+editor contract tests green). Next mechanical tick is the first weekly cadence-report cron Monday 2026-05-25. Phase 3 M5 fully closed. Successor spec `docs/specs/api-v0.2.0-cut/` scaffolded ([PR #83](https://github.com/Smart-AI-Memory/attune-rag/pull/83)); activates after W4.2. |
| Last updated | 2026-05-20 |

---

## Phase ordering rationale

1. **Eval gate before API freeze.** The faithfulness /
   P@1 / R@3 numbers are the release gate. If the gate is
   noisy, every later phase is unverifiable.
2. **API freeze before quality burn-in.** A "no new
   features for 4 weeks" rule is meaningless until we
   know which surface counts as "the feature set."
3. **Downstream validation before 1.0.** Internal-only
   stability claims aren't credible — at least one
   external consumer has to survive a minor bump.

---

## Phase 1 — Baseline lock-in

**Spec:** [docs/specs/release-quality-baseline/](release-quality-baseline/)
**Status:** complete (2026-05-16). Baseline locked at [baseline-1.md](release-quality-baseline/baseline-1.md). Thresholds: faithfulness ≥ 0.9686 (mean 0.979, σ 0.0052), P@1 ≥ 0.95, R@3 ≥ 1.00. CI gate wired at [`.github/workflows/benchmark.yml`](../../.github/workflows/benchmark.yml). M3.5 verified 2026-05-16 via throwaway [#34](https://github.com/Smart-AI-Memory/attune-rag/pull/34): gate fires, comment posts under marker, edit-in-place on subsequent failures.
**Estimate:** ~1 week of attention

**Outcome:** every release reports P@1, R@3, and
faithfulness. CI fails on regression past a threshold set
at 2σ above the measured noise floor (per Decision 1).

**Gate:** a deliberately-bad PR fails the metric gate in
CI; README quotes a numeric baseline; the noise-floor
measurement and resulting per-metric thresholds are
documented in `docs/specs/release-quality-baseline/`.

**First task — noise-floor measurement.** Run the full
benchmark N times back-to-back on an unchanged HEAD
(target N ≥ 10, ideally 20) to quantify aggregate
variance per metric. Set the gate at `mean − 2σ` for each
of faithfulness, P@1, and R@3. Re-measure when
`--thinking` defaults change in Phase 2.

**attune-ai workflows:**
- `/smart-test` on `src/attune_rag/dashboard/refresh.py`,
  `src/attune_rag/eval/faithfulness.py`.
- `/release-prep` dry runs.
- `attune-ai:bug-predict` on `src/attune_rag/benchmark.py`.

**Next action:** spin up the spec with `/spec`; first
deliverable is the noise-floor measurement script and
results doc.

---

## Phase 2 — Eval story landed

**Spec:** [docs/specs/faithfulness-thinking-decision/](faithfulness-thinking-decision/)
**Status:** **complete 2026-05-16**. Final v3 results (design.md tie rule, controls excluded): wins_off = 10, wins_on = 4, ties = 16. Bootstrap 95 % CI on `(wins_off − wins_on)` = `[−1, +13]` includes 0; point estimate +6. Judge variance `margin_stdev = 0.0189` (well below 0.10 escalation threshold). Phantom-claim rate 7.4 % (heuristic). **Verdict: `off-forever` — keep `--thinking` opt-in.** Locked at [`faithfulness-thinking-decision/decision.md`](faithfulness-thinking-decision/decision.md). Ships at 0.1.19. No baseline re-measurement required.
**Estimate:** ~3 weeks of attention
**Depends on:** Phase 1 complete ✓ (PR #33 merged 2026-05-16)

**Outcome:** `--thinking` default is decided. Calibration
doc has a conclusion, not a "pending" tag. Judge
non-determinism has a quantified confidence interval.

**Gate:** n ≥ 30 hand-labeled queries; calibration doc's
top paragraph states the default; CHANGELOG records it
under `Changed`.

**attune-ai workflows:**
- `/spec` brainstorm → plan → execute.
- `/code-quality` and `/deep-review` on
  `src/attune_rag/eval/faithfulness.py` and
  `src/attune_rag/eval/bench_prompts.py`.
- `/doc-gen` regenerates the calibration doc at the end.

---

## Phase 3 — Public API freeze, ship 0.2.0

**Spec:** [docs/specs/api-v0.2-public-surface/](api-v0.2-public-surface/)
**Status:** **all in-scope work complete 2026-05-20.** Groundwork shipped in attune-rag 0.1.18 (PR #36) — M1–M3 + M4 done; surface lock test, deprecation shims, POLICY.md, README "Public API" section, `AttuneHelpCorpus` re-export all in. **M5 (attune-gui downstream cleanup) closed 2026-05-20:** M5.1 + M5.2 landed on branch `feature/attune-rag-0.2-editor-rename` (commits `5bf35ec`, `b5f4d3b`); M5.3 — pin bump to 0.1.22 + branch merge — landed as [attune-gui#38](https://github.com/Smart-AI-Memory/attune-gui/pull/38) with W1.2 evidence (82/82 rag+editor contract tests green against the 0.1.22 wheel). **The formal 0.2.0 SemVer freeze + classifier flip remain queued** — the freeze ships via the [`api-v0.2.0-cut`](api-v0.2.0-cut/) successor spec after Phase 4 W4.2; the classifier flip stays Phase 5's (0.2.0 stays at `Development Status :: 3 - Alpha`).
**Estimate:** ~3 weeks of attention (realized; M5 closed)
**Depends on:** Phase 2 ships before 0.2.0 ships. ✓ Phase 2 shipped as 0.1.19 (2026-05-16).
**Sequencing (per Decision 3):** soft-parallel. Phase 3
scoping — surface mapping, `__all__` audit, spec drafting
— may begin during Phase 2. 0.2.0 release ships only
after Phase 2 lands.
**Surface coverage requirement (per Decision 2):** the
frozen public API must include everything attune-gui
consumes. Audit attune-gui's imports of `attune_rag.*`
before locking `__all__`.

**Outcome:** documented, frozen public surface for
`RagPipeline`, `CorpusProtocol`, `DirectoryCorpus`,
provider adapters, `RenamePlan` / `FileMove`, CLI.
Internals marked private. Breaking changes after 0.2.0
require a deprecation cycle.

**Gate:** 0.2.0 on PyPI; README lists the exact public
symbols; `tests/unit/test_api_surface.py` snapshots
`__all__` and fails on accidental surface changes.

**attune-ai workflows:**
- `/refactor` on `src/attune_rag/`.
- `/doc-audit` then `/doc-gen` for undocumented public
  symbols.
- `/deep-review` — one module per PR, not one mega-PR.

---

## Phase 4 — Quality burn-in + downstream validation

**Spec:** [docs/specs/downstream-validation/](downstream-validation/)
**Status:** **approved 2026-05-19**, W0 setup begins 2026-05-23. 23 tasks (W0.1–W4.4) walked task-by-task. Five design-level open questions resolved: override label `freeze-override`; perf gate measures both wall-clock + CPU-time, gates CPU-time first; downstream gate ramps comment (W1–W2) → block (W3–W4); reranker token-cost track-only; W0.1 freeze enforcer covers `__all__` + CHANGELOG + `template_schema.json`. Phase 4 has explicit "freeze semantics" callout in tasks.md so the `### Added` vs `### Changed` distinction is documented (internal selection-criteria improvements ship freely as `### Changed`; new public knobs wait for the 0.2.0 cut).
**Estimate:** ~4 weeks soak + 1 week setup (W0). Calendar target W0 2026-05-23 → W4 close 2026-06-20, with 2026-06-21 → 06-27 slack week.
**Depends on:** Phase 3 complete

**Outcome:** four consecutive weeks of CHANGELOG entries
containing only `Fixed` / `Changed` / `Security` — no
`Added`. **attune-gui** (per Decision 2) pins attune-rag
and survives a 0.2.x → 0.3.x bump without breakage; its
CI is treated as a second release gate.

**Gate:** four-week feature freeze delivered cleanly;
downstream consumer green; perf baseline documented at
`docs/specs/downstream-validation/perf-baseline.md`;
security audit clean.

**attune-ai workflows:**
- `/security-audit` repo-wide, then per-PR.
- `attune-ai:performance_audit` on retrieval + reranker
  hot paths.
- `/test-audit` to find mocked-or-shallow tests.
- `/smart-test` to raise public-surface coverage to ~90 %.
- One `/deep-review` mid-phase, one at end.

---

## Phase 5 — 1.0.0 release

**Spec:** `docs/specs/v1.0.0-release/` (to create)
**Status:** not started
**Estimate:** ~2 weeks of attention
**Depends on:** Phase 4 complete

**Outcome:** `Development Status :: 3 - Alpha` →
`5 - Production/Stable` in `pyproject.toml`.
`docs/POLICY.md` documents support window and deprecation
policy.

**Gate:** 1.0.0 on PyPI; support policy published; seven
days post-release with no hotfix.

**attune-ai workflows:**
- `/release-prep` (`attune-release-check` skill enforces
  no-stale-`__version__`, clean tree, green CI, changelog
  entry).
- Final `/security-audit`.
- Final `/deep-review`.
- `/doc-orchestrator` for one-shot doc refresh across
  README + API reference + calibration doc.

---

## Decisions log

| # | Date | Decision |
|---|---|---|
| 1 | 2026-05-15 | **Regression threshold = `mean − 2σ` per metric, with the noise floor measured from N ≥ 10 back-to-back benchmark runs on an unchanged HEAD.** Re-measure when `--thinking` defaults change. Rationale: judge non-determinism (e.g., gq-017's 43 pp single-query swing) makes any fixed round-number threshold a guess; measuring the floor first pre-empts the "false positive" objection. Lands as Phase 1's first deliverable. |
| 2 | 2026-05-15 | **attune-gui is the gating downstream for Phase 4.** Rationale: tightest coupling — consumes pipeline, dashboard refresh, and rename plans, so it stresses the largest fraction of attune-rag's public surface. Phase 3's `__all__` audit must cover everything attune-gui imports from `attune_rag.*`. |
| 3 | 2026-05-15 | **Phase 2 / Phase 3 sequencing = soft-parallel.** Phase 3 scoping (surface mapping, `__all__` audit, spec drafting) may begin during Phase 2; 0.2.0 release ships only after Phase 2 lands. Rationale: captures ~2 weeks of calendar without freezing the API around an unstable gate. |

## Open decisions

None — Phase 1 is unblocked.
