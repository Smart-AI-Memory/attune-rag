# Re-measuring the quality baseline

The thresholds in
[`thresholds.json`](thresholds.json) are pinned to a specific
measurement of judge non-determinism. When something that
affects the judge changes, those thresholds become stale —
either falsely failing real PRs or falsely passing real
regressions. Re-measure in those cases. Don't re-measure
casually; every run costs ~3.6 hours of wall-clock and
hundreds of API calls.

## When to re-measure

Re-measure if **any** of the following lands:

- **`--thinking` default flips** (currently opt-in; Phase 2
  may flip it).
- **Judge model changes** — the faithfulness judge is pinned
  to a specific Claude model; swapping it is a re-measure
  event.
- **Judge prompt changes** in
  [`src/attune_rag/eval/bench_prompts.py`](../../../src/attune_rag/eval/bench_prompts.py).
- **Corpus changes** that shift `tests/golden/queries.yaml`
  (CI catches this automatically — `queries_sha256` mismatch
  → exit 2 with a re-measure hint).
- **Generator prompt changes** in
  [`src/attune_rag/prompts.py`](../../../src/attune_rag/prompts.py)
  if they materially alter the answers the judge sees.

Do **not** re-measure for:

- Code changes that don't affect the judge or the answers it
  sees (most refactors, doc edits, test-only changes).
- A single PR that happens to fail the gate — investigate the
  regression first.

## How to re-measure

1. **Land the judge-affecting change on a topic branch.**
   Don't bundle the re-measurement into the same commit as the
   change — keep them separate so the cause-and-effect is
   reviewable.

2. **Run the variance script** from that branch:

   ```bash
   export ANTHROPIC_API_KEY=...
   python scripts/measure_baseline_variance.py \
       --runs 20 \
       --out docs/specs/release-quality-baseline/baseline-N.md \
       --thresholds-out docs/specs/release-quality-baseline/thresholds.json
   ```

   Use the next sequence number for `N` (next after the
   latest existing `baseline-*.md`). The script writes both
   the locked record and the machine-readable thresholds.

3. **Eyeball the numbers** before committing. Sanity checks:

   - `stdev` on `mean_faithfulness` should be in the range we
     saw at the prior baseline (≈ 0.005 at the v1.0 baseline).
     A 10× jump means the judge got dramatically less stable
     and the change should be reconsidered.
   - `mean` should be plausible (≥ 0.90 for a healthy run on
     the current golden set).
   - Retrieval metrics should be deterministic (`stdev == 0`).
     If they're not, something changed in retrieval — that's
     out of scope for "judge change" and you should stop and
     understand it.

4. **Open a follow-up PR** that contains **only**:

   - The new `baseline-N.md`.
   - The updated `thresholds.json`.
   - A `CHANGELOG.md` entry under `Changed` noting the
     re-measurement and the reason.

   Include `[baseline-update]` in the PR title. The CI gate's
   own logic ignores this opt-in label for now (we still gate
   the threshold-update PR against the *old* thresholds — the
   delta should be small enough to pass). If it doesn't pass,
   that's information: the judge-affecting change is moving
   the floor by more than 2σ, which is worth thinking about.

5. **Merge in order.** The judge-affecting change first, then
   the threshold update. Don't merge the threshold update
   before the change lands or future PRs run against a
   threshold whose underlying conditions don't yet exist.

## Sanity-checking a re-measurement

If a re-measurement looks weird, three quick diagnostics
before committing it:

- **Diff `mean` against the prior baseline.** A move of more
  than ~5 pp on `mean_faithfulness` is a flag — either the
  judge got materially better/worse or something about the
  setup is different.
- **Spot-check 2–3 random queries by hand.** If the judge is
  emitting different verdicts for queries you eyeballed
  before, the change is real; if verdicts look the same but
  the aggregate moved, suspect a scoring or aggregation bug.
- **Re-run on the prior commit.** If the new numbers
  *also* show on the unchanged HEAD, the variance script or
  its environment changed, not the judge.

## Related docs

- Decision 1 in
  [docs/specs/ROADMAP-v1.md](../ROADMAP-v1.md) — why
  `mean − 2σ` and not a round-number threshold.
- The variance script:
  [`scripts/measure_baseline_variance.py`](../../../scripts/measure_baseline_variance.py).
- The check script (used by CI):
  [`scripts/check_thresholds.py`](../../../scripts/check_thresholds.py).
- The locked v1 measurement:
  [`baseline-1.md`](baseline-1.md).
