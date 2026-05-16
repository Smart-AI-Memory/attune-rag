# attune-rag → 1.0 roadmap

Rolling tracker for the path from v0.1.17 (Alpha) to v1.0.0
(Production/Stable). One spec per phase. Update the
**Status** and **Next action** rows at the end of every
working session so state is recoverable cold from disk.

| Field | Value |
|---|---|
| Current version | 0.1.18 (Phase 1 + Phase 3 groundwork shipped on PyPI 2026-05-16) |
| Target version | 1.0.0 |
| Current phase | Phase 2 — **complete 2026-05-16**. Decision: keep `--thinking` opt-in (`off-forever`). Ships at 0.1.19. Phase 3's API-surface groundwork shipped in parallel as part of 0.1.18 (see PR #36). |
| Last updated | 2026-05-16 |

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
**Status:** **complete 2026-05-16**. Final v3 results (design.md tie rule, controls excluded): wins_off = 10, wins_on = 4, ties = 16. Bootstrap 95 % CI on `(wins_off − wins_on)` = `[−1, +13]` includes 0; point estimate +6. Judge variance `margin_stdev = 0.0189` (well below 0.10 escalation threshold). Phantom-claim rate 7.4 % (heuristic). **Verdict: `off-forever` — keep `--thinking` opt-in.** Locked at [`faithfulness-thinking-decision/decision.md`](faithfulness-thinking-decision/decision.md). Ships at 0.1.18. No baseline re-measurement required.
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

**Spec:** `docs/specs/api-v0.2-public-surface/` (to create)
**Status:** not started
**Estimate:** ~3 weeks of attention
**Depends on:** Phase 2 ships before 0.2.0 ships.
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

**Spec:** `docs/specs/downstream-validation/` (to create)
**Status:** not started
**Estimate:** ~4 weeks of attention
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
