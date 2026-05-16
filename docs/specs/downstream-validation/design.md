# Spec: Downstream Validation (Phase 4)

## Phase 2: Design

**Status**: scoping

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
involved. Results dumped as `perf-baseline.json` alongside `baseline-1.md`
(narrative) and `perf-thresholds.json` (machine-readable).

Threshold strategy: **same as Phase 1 quality gate** — `mean + 2σ` per
metric for latencies (higher = worse, so we gate the upper bound). For
token counts, `mean + 2σ` per input/output dimension. Re-measure trigger:
- Python minor version change (3.11 → 3.12 etc.).
- Hardware change in the CI runner SKU.
- Any change to `KeywordRetriever`, `LLMReranker`, or `pipeline.py`.

**Promotion ramp:**
- Week 1–2: advisory only. Workflow comments the regression delta on PRs
  but doesn't fail.
- Week 3: gates retrieval perf (deterministic, low noise).
- Week 4: gates reranker perf too if its observed σ stays tight.

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

### Open questions to resolve before implementation

- **Override label name** — `freeze-override` reads OK; alternatives:
  `freeze-exempt`, `phase-4-override`. Pick now to avoid bikeshedding mid-flight.
- **Perf threshold metric for reranker latency** — wall-clock or
  CPU-time? Wall-clock includes network variance to Anthropic. CPU-time
  excludes it. Suggest wall-clock with a documented "expect noise" disclaimer.
- **Downstream gate severity** — does an attune-gui failure block the PR or
  just comment? Initial recommendation: comment in week 1–2, block in
  weeks 3–4.
- **Token-cost tracking** — useful or noise? attune-rag's pipeline calls
  out to Claude Haiku for reranker; tracking input/output tokens per call
  gives an early warning if a prompt change increased cost. Initial
  recommendation: track but don't gate.

### Notes

- Phase 4 deliberately does *not* try to add tests for the freeze itself
  via attune-ai workflows (`/test-audit`, `/smart-test`, `/deep-review`).
  Those are listed in the roadmap as attune-ai workflows to *use* during
  Phase 4, not as deliverables of Phase 4. They're applied per-PR, not
  scoped here.
- The freeze enforcer is *Phase 4's* defining mechanic. Without it, "four
  consecutive weeks of no Added" is aspirational. With it, the
  enforcement is mechanical and the gate is real.
