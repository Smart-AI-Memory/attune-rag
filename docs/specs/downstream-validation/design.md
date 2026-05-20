# Spec: Downstream Validation (Phase 4)

## Phase 2: Design

**Status**: approved (2026-05-19)

### Architecture overview

Phase 4 has four mechanical pillars + one calendar gate. Each pillar lands as
a thin, testable piece of CI machinery; the calendar gate is the soak time
itself.

```
Feature-freeze enforcer    →  CI fails PRs that add to __all__ or "### Added"
Perf baseline + gate       →  CI advisory (week 1-2), gating (week 3+)
Security audit cadence     →  Repo-wide week 1, per-PR for sensitive paths
Downstream-CI integration  →  attune-rag PRs trigger attune-gui's test suite
Cadence dashboard          →  Weekly report on Added/Fixed/Changed/Security
```

### 1. Feature-freeze enforcer

**Mechanism:** new CI job in `.github/workflows/freeze.yml` running on every
PR. Three checks:

1. **Public surface diff.** Parse `__all__` from each public module on the
   PR base and PR head; fail if any symbol was added.
2. **CHANGELOG `### Added` diff.** Parse the `[Unreleased]` block; fail if a
   new `### Added` bullet appeared on the PR head that wasn't on the base.
3. **Editor schema diff.** Diff `src/attune_rag/editor/template_schema.json`
   for backward-incompatible changes (loosening additionalProperties from
   `true` to anything stricter, narrowing an enum, etc.). Fail those — they
   need `freeze-override` per Phase 4 policy.

**Override:** PR label `freeze-override` makes the job warn instead of fail.
The override must also include a `[Security]` or `[Override-rationale]`
section in the PR body. Job parses that and surfaces it as a check
annotation.

**Implementation:** stdlib + `pyyaml` only. Reuses
`scripts/check_thresholds.py`-style exit-code semantics (0 pass, 1 freeze
violation, 2 malformed input).

**Test plan:** unit tests under `tests/unit/test_freeze_enforcer.py`
covering: clean PR passes; surface addition fails; CHANGELOG addition fails;
override label flips fail to warn; missing rationale on override fails;
schema tightening fails.

### 2. Perf baseline + gate

**Hot paths to measure:**
- `KeywordRetriever.retrieve()` — pure CPU, deterministic, the most-called
  function in the pipeline.
- `LLMReranker.rerank()` — Claude Haiku call; measure latency p50/p95 +
  per-call token cost (input/output).
- `RagPipeline.run()` end-to-end — composite of the above plus prompt
  assembly.
- `DirectoryCorpus.load()` cold start — disk IO + parse, gated separately
  since it runs once per process.

**Measurement methodology:** N=30 runs per benchmark, `pytest-benchmark`
(already a dev dep). Each benchmark fixes its random seed if any sampling is
involved. Each benchmark records **both wall-clock and CPU-time** per run
(`time.perf_counter` + `time.process_time`); pytest-benchmark stores
wall-clock natively, CPU-time is captured via a fixture wrapper. Results
dumped as `perf-baseline.json` alongside `baseline-1.md` (narrative) and
`perf-thresholds.json` (machine-readable, two threshold rows per benchmark
— `<bench>.wall` and `<bench>.cpu`).

Threshold strategy: **same as Phase 1 quality gate** — `mean + 2σ` per
metric for latencies (higher = worse, so we gate the upper bound), applied
independently to wall-clock and CPU-time. CPU-time gates regressions in
our code; wall-clock tracks user-perceived latency including network. For
token counts, `mean + 2σ` per input/output dimension (track-only, no
gate — see Resolved decisions). Re-measure trigger:
- Python minor version change (3.11 → 3.12 etc.).
- Hardware change in the CI runner SKU.
- Any change to `KeywordRetriever`, `LLMReranker`, or `pipeline.py`.

**Promotion ramp:**
- Week 1–2: advisory only. Workflow comments the regression delta on PRs
  (both wall-clock and CPU-time axes) but doesn't fail.
- Week 3: gates **CPU-time** for retrieval perf (deterministic, low noise).
  Wall-clock stays advisory for retrieval through W3 — re-evaluate W4 once
  noise floor is characterized.
- Week 4: gates reranker CPU-time if its observed σ stays tight. Reranker
  wall-clock stays advisory (Anthropic network variance is real and
  unactionable on our side).

**Output files:**
- `docs/specs/downstream-validation/perf-baseline.md` — narrative + raw
  numbers, mirrors `release-quality-baseline/baseline-1.md` shape.
- `perf-thresholds.json` — alongside it, machine-readable for CI.
- `scripts/measure_perf_baseline.py` — pure stdlib + pytest-benchmark
  subprocess invocation, mirrors `measure_baseline_variance.py`.

### 3. Security audit cadence

**Week 1: repo-wide pass.** Triggered manually via attune-ai's
`/security-audit` skill. Findings go into
`docs/specs/downstream-validation/security-findings.md` with severity +
disposition (fix in-phase / deferred / non-issue).

**Per-PR scans:** new workflow `.github/workflows/security-scan.yml`
triggers `/security-audit`-equivalent stdlib checks on PRs touching:
- `src/attune_rag/editor/**` — rename and lint code paths walk user paths
- `src/attune_rag/providers/**` — handles API tokens, may log requests
- `src/attune_rag/dashboard/**` — emits snapshots; ensure no secret leakage
- `src/attune_rag/cli.py` — user-facing argv parsing

The stdlib check is conservative: `grep` + `ast` walk for the four classic
patterns (eval/exec, path traversal, hardcoded secrets, untrusted-input
deserialization). False positives are OK; false negatives are not.
attune-ai's deeper analysis is the secondary sweep on findings.

**Exit criteria:** all findings either fixed, downgraded to non-issue with
written rationale, or filed as tracked-debt with a target Phase 5 ticket.

### 4. Downstream CI integration

**attune-rag side:** `.github/workflows/downstream-attune-gui.yml`. Triggers
on PRs. Inputs:
- `attune_gui_ref` — defaults to `feature/attune-rag-0.2-editor-rename` (the
  M5.2-clean branch); user can override via PR comment `/downstream test
  <ref>`.
- `attune_rag_ref` — the PR's HEAD SHA, used by attune-gui to `pip install
  git+https://github.com/Smart-AI-Memory/attune-rag.git@<sha>`.

Job runs attune-gui's `pytest sidecar/tests -k 'editor or rag'` against the
installed attune-rag. Failures comment on the PR with the failing test
names + a link to the attune-gui run.

**attune-gui side:** the existing reusable workflow is good enough; no
attune-gui-side changes needed beyond confirming `pip install
attune-rag==<sha>` works in the runner.

### 5. Cadence dashboard

**Mechanism:** `scripts/changelog_cadence.py` walks the last 7 days (or any
window passed as `--window-days N`) of CHANGELOG entries, parses
`[Unreleased]` and tagged releases, and emits a Markdown report:

```
Week of 2026-05-19 → 2026-05-26
  Added:    0
  Changed:  3 (releases: 0.1.19)
  Fixed:    1
  Security: 0
  Status:   ON TRACK (freeze active, week 1/4)
```

Run weekly via a `schedule:` cron in
`.github/workflows/cadence-report.yml`. Output committed to
`docs/specs/downstream-validation/cadence-week-{N}.md` and posted as a PR
comment if there's an `Added` entry (reinforcing the freeze enforcer).

### Resolved decisions (2026-05-19, spec approval pass)

- **Override label name:** `freeze-override`. Neutral, phase-agnostic so the
  same enforcer survives Phase 5+.
- **Perf threshold metric:** **both wall-clock and CPU-time**, tracked
  separately per hot path. CPU-time gates regressions in our code;
  wall-clock tracks user-perceived latency. Doubles the rows in
  `perf-thresholds.json` but the signal split is worth it for a freeze gate.
  Reranker stays advisory in W3 if its observed σ stays loose on either
  axis (per W3.1).
- **Downstream gate severity:** comment-only in W1–W2, blocking in W3–W4.
  Already encoded in tasks.md (W0.8 advisory → W3.2 blocking). Two weeks of
  characterization before failures become PR blockers.
- **Token-cost tracking:** track input/output tokens per reranker call in
  the perf report; do **not** gate. Cost spikes are signal worth capturing,
  but Anthropic-side pricing/model shifts aren't our bug to block on.

(Previous open-questions section consolidated here once the spec moved from
scoping to approved. See git history if you need the original framing.)

### Notes

- Phase 4 deliberately does *not* try to add tests for the freeze itself
  via attune-ai workflows (`/test-audit`, `/smart-test`, `/deep-review`).
  Those are listed in the roadmap as attune-ai workflows to *use* during
  Phase 4, not as deliverables of Phase 4. They're applied per-PR, not
  scoped here.
- The freeze enforcer is *Phase 4's* defining mechanic. Without it, "four
  consecutive weeks of no Added" is aspirational. With it, the
  enforcement is mechanical and the gate is real.
