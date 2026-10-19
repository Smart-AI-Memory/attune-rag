# Benchmark advisory cross-review triage

Scope: findings F1, F4, F9, F10, F11, F12, F13 (one-based order) from
`/private/tmp/attune-rag-task3-subscription-review-authorized/review-result.json`.
Reviewed the full current code, not only the diff. No source files were edited
and no provider/model calls were made. Line references below describe the
reviewed snapshot before any follow-up fixes by the owning agent.

## Disposition

| Finding | Classification | Evidence and action |
|---|---|---|
| F1: thinking/native comparison varies two axes | **Noise: unreachable through the CLI** | `src/attune_rag/benchmark.py:921-928` rejects `--compare-thinking` with `--native-citations` before either provider pass. Existing `tests/unit/test_benchmark.py:320-340` covers this combination. Thus the secondary kwargs at `benchmark.py:1105-1108` cannot have both switches true on an accepted CLI invocation. A local `main()` probe returned 2 and the explicit incompatibility message, with zero provider-pass calls. No change needed. |
| F4: early precision failure omits advisory diagnostics | **Intentional behavior change; optional diagnostics improvement deferred** | Current `benchmark.py:990-1002` validates, persists, and prints the primary retrieval report before returning 1 on a measured regression. Negatives, extended, and generalization work begins at `:1004`, `:1012`, and `:1022`. HEAD previously did this advisory work at `:878-910` before gating at `:912-917`; the observation is accurate. Early return preserves the primary failure and avoids later advisory failure replacing it. Primary per-query evidence remains in JSON (`:991-993`), and verbose diagnostics are printed before the gate (`:994`, `:317-325`). Restoring optional advisory diagnostics is a separate usability choice, not a reason to weaken or delay the required verdict. |
| F9: calibration success leaves faithfulness pending | **Real, low-severity new outcome defect; reproduced** | Payload initialization sets faithfulness to pending whenever `--with-faithfulness` is present (`benchmark.py:876-882`). Calibration returns 0 after marking only retrieval skipped and setting `reason=calibration_only` (`:965-980`), so final serialization preserves pending faithfulness (`:900-905`). Actual local calibration produced exit 0, 29 calibration rows, retrieval skipped, faithfulness pending, and zero provider-pass calls. The backward-compatible fix is to mark faithfulness skipped in this calibration-only branch; calibration has always documented “then exit” (`:723-730`). Add one CLI regression for the combined flags. |
| F10: calibration lacks an earlier strict-JSON precheck | **Noise as a correctness claim; uniform failure receipts may be deferred** | `_dump_json` strictly serializes the entire payload before writing (`benchmark.py:635-637`), and `main` returns 2 on a serialization failure (`:900-904`). The old output is removed before execution (`:884-885`), so a bad calibration payload cannot pass or preserve an old successful receipt. Calibration is the sole operation in this mode; there is no completed retrieval/faithfulness measurement to preserve. The actual implementation emits integer-derived thresholds and finite count ratios (`:379-400`), with a fixed finite target in the CLI path. No ordinary input path producing nonfinite calibration output was demonstrated. An injected malformed calibration result could lack a failure JSON, which is optional diagnostic consistency, not a false-pass bug. |
| F11: repeated checkpoint serialization and stderr messages | **Intentional checkpoints; cosmetic log deduplication deferred** | The retrieval checkpoint is at `benchmark.py:993`, primary faithfulness checkpoint at `:1089`, and final write at `:901`. Final writes capture subsequent optional diagnostics/comparison results and final status/reason. All writes use the same strict writer, and its location message goes to stderr (`:638`), while the baseline consumer parses stdout (`scripts/measure_baseline_variance.py:45-58,177-182`). Three writes and repeated path announcements can occur, but no correctness/consumer break is established. Retain checkpoints; quieting repeated notices is optional. |
| F12: generator variable shadows the path parameter | **Noise/style only** | The `path` in the generator expression at `benchmark.py:128` has generator-local scope; it does not rebind the function parameter. The function’s earlier file read/error use the outer path (`:113-116`), and the function returns parsed queries (`:131`). A variable rename could improve style but fixes no behavior. |
| F13: bad-options test may assume a missing guard | **Noise: guard exists and was exercised** | `benchmark.py:929-934` explicitly rejects bare `--compare-thinking` with the “requires --with-faithfulness” message. That matches `tests/unit/test_benchmark.py:813-826`. A local `main(['--compare-thinking'])` probe returned 2 with that message and zero provider-pass calls. No change needed. |

## Behavioral receipts

Observed F9 output before follow-up fixes:

```json
{"calibration_rows":29,"exit":0,"outcomes":{"faithfulness":"pending","reason":"calibration_only","retrieval":"skipped"},"provider_pass_calls":0,"report":"/private/tmp/attune-rag-cross-review-f9-calibration.json"}
```

Reproduce from `/Users/patrickroebuck/attune-rag`; this uses real local
calibration and replaces only the provider pass with a fail-if-called guard:

```bash
.venv/bin/python - <<'PY'
import contextlib
import io
import json
import os
from pathlib import Path
from unittest.mock import patch
from attune_rag import benchmark

os.environ.pop('ANTHROPIC_API_KEY', None)
output = Path('/private/tmp/attune-rag-cross-review-f9-calibration.json')
with patch.object(benchmark, '_score_faithfulness', side_effect=AssertionError('Provider pass must not run')) as provider, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    rc = benchmark.main(['--calibrate-abstention', '--with-faithfulness', '--json', str(output)])
report = json.loads(output.read_text())
print(json.dumps({'exit': rc, 'outcomes': report['outcomes'], 'provider_pass_calls': provider.call_count, 'calibration_rows': len(report['calibration']['rows'])}, sort_keys=True))
PY
```

F1/F13 full-code guard probes returned:

```text
F1 exit=2: error: --compare-thinking and --native-citations cannot be combined (would require 4 faithfulness passes). Run them in separate invocations.
F13 exit=2: error: --compare-thinking requires --with-faithfulness.
Both: provider_pass_calls=0.
```

Only F9 warrants a bounded correctness fix among these seven findings.
This is advisory triage, not a replacement for the already accepted Phase 1
gates or authorization for unrelated changes.

## Subsequent closeout

The preceding notes are preserved as pre-correction evidence. F9 was subsequently corrected and regression-tested; the final [disposition](../cross-review.md) and [independent follow-up](task-3-cross-review-independent.md) record the current state. The integrated refreshed suite and artifact checks are bound in [task-3-cross-review-followup.json](task-3-cross-review-followup.json).
