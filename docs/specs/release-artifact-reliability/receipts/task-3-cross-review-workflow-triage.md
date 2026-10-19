# Task 3 Claude advisory findings: bounded workflow/checker triage

Source review result: `/private/tmp/attune-rag-task3-subscription-review-authorized/review-result.json`.
Reviewed findings F2, F5, F6, F7, F8, F14 against the current full files and active threshold/spec configuration. Read-only except this requested scratch receipt; no providers, model calls, product edits, or remote actions. The advisory result is not a gate.

| Finding | Classification | Disposition |
|---|---|---|
| F2 | Noise / hypothetical scope extension | No current change needed. |
| F5 | Intentional; alternate-mode gate support explicitly deferred | Retain default-pass gate. |
| F6 | Real, low-severity diagnostic wording issue | Optional in-scope wording improvement; preserve missing-evidence failure. |
| F7 | Intentional precedence | Retain measured regression and separately disclose original benchmark exit. |
| F8 | Noise / incorrect reachability claim | No change needed. |
| F14 | Noise as stated; timeout concern unmeasured | No SHA-related change or timeout increase justified by this finding. |

## F2 — universal [0,1] threshold validator

The active checker supports only quality ratios: precision_at_1, recall_at_k, and mean_faithfulness (`/Users/patrickroebuck/attune-rag/scripts/check_thresholds.py:82`–`:125`). The actual locked config contains exactly mean_faithfulness=0.9698, precision_at_1=0.975, recall_at_3=1.0 (`/Users/patrickroebuck/attune-rag/docs/specs/release-quality-baseline/thresholds.json:4`–`:20`). Latency/CPU use a separate formatter and config.

The claim's implication that unknown unmeasured metrics should be ignored is contrary to the existing checker contract: every configured nonskipped threshold must have a measured counterpart (`scripts/check_thresholds.py:190`–`:217`). `git show HEAD:scripts/check_thresholds.py` confirms that unknown unmeasured metrics already produced validation errors before this change. Validation of skipped configuration is intentional (new regression test `test_skipping_metric_does_not_accept_malformed_threshold_configuration`).

A future heterogeneous metric API would require types, extraction, and direction semantics, not simply relaxing this guard. No such API is active or approved in Phase 1 (`docs/specs/release-artifact-reliability/requirements.md:37`). Reject as a current bug.

## F5 — compare-thinking dumps rejected by default-pass gate

Correct observation about keys, but explicitly intended scope. `/Users/patrickroebuck/attune-rag/scripts/check_thresholds.py:41`–`:45` says Phase 1 gates the default single-pass run and revisits thinking comparison if defaults change. The workflow constructs exactly `--with-faithfulness --json ...` or `--json ...`, never `--compare-thinking` (`/Users/patrickroebuck/attune-rag/.github/workflows/benchmark.yml:122`–`:126`). Consequently the completed pass must have `faithfulness_legacy` (`:189`–`:190`).

Manual comparison is an available benchmark feature, but integrating it with this gate is explicitly deferred, not a regression introduced by Task 3. Do not expand the approved scope merely because the JSON schema supports additional exploratory modes.

## F6 — missing check result labelled validation error

The final `always() && rc != '0' && rc != '1'` guard (`/Users/patrickroebuck/attune-rag/.github/workflows/benchmark.yml:301`–`:305`) intentionally fails when the required check produces no usable exit-code evidence. This correctly prevents a missing/failed prerequisite from becoming a pass under R4 (`docs/specs/release-artifact-reliability/requirements.md:21`–`:23`).

The printed text specifically says the quality gate "hit a validation error" and lists bad metric dumps/config, although an install/setup failure may leave the check unrun and its output empty. That wording is a real low-severity diagnostic issue. Small optional correction: state that required quality validation **failed or did not complete**, and include `check`/benchmark execution status when available. Do not weaken the always/missing-result guard. Cancellation-specific scheduling behavior was not run against GitHub; the static predicate matches an empty result, while the job may independently remain cancelled.

## F7 — measured regression wins over later benchmark error

The actual state contains TWO failures: the primary measured faithfulness value violates the locked threshold (FAITH_RC=1), and a subsequent harness/secondary operation returns BENCH_RC=2. The workflow reports a regression because one is genuinely present (`/Users/patrickroebuck/attune-rag/.github/workflows/benchmark.yml:216`–`:224`), does not retry (retry requires 3 at `:132`), and still prints the original benchmark exit separately in the summary (`:267`–`:281`). No success or provider-outage result is fabricated.

The deliberately named `completed-faithfulness-before-error` case records this contract (`/Users/patrickroebuck/attune-rag/tests/unit/test_quality_workflow_outcomes.py:176`, `:192`, `:203`–`:206`). R5 says a later exception cannot erase a completed regression (`docs/specs/release-artifact-reliability/requirements.md:25`). With a single aggregate code, preferring a proven measured regression while retaining the original process status is defensible and matches the plan. Richer combined-failure diagnostics are optional; changing this to a non-regression-only classification would weaken the existing measured-result receipt.

## F8 — math.isfinite supposedly unreachable

False: short-circuit evaluation reaches `math.isfinite(value)` whenever `0 <= value <= 1` is true (`/Users/patrickroebuck/attune-rag/scripts/check_thresholds.py:75`–`:78`). It is redundant for accepted built-in numeric values, but not unreachable. The comment correctly explains why range rejection precedes conversion of an arbitrarily large integer.

Local no-file probe using `runpy.run_path` and a temporary in-process isfinite spy:

```
F8_VALID_INPUT {'result': 0.5, 'isfinite_calls': [0.5]}
F8_HUGE_INT {'rejected': True, 'isfinite_calls': []}
```

No product modifications or providers. An explicit finite check can stay for clarity; this is not a reliability defect.

## F14 — query-SHA coupling / 20-second timeout

The test fixture points to the CURRENT repo query file, not a frozen old SHA (`/Users/patrickroebuck/attune-rag/tests/unit/test_quality_workflow_outcomes.py:79`). THRESHOLDS likewise points at the CURRENT active config (`:18`). The checker hashes the file at runtime and compares to the current config (`scripts/check_thresholds.py:161`–`:188`). Updating query bytes and their baseline SHA together therefore continues to work. The claim that any legitimate SHA update breaks every replay case is false.

A temporary-files-only proof used the real checker with current query bytes/config, then a changed query file plus matching recomputed SHA. Both returned zero failures and zero errors:

```
F14_SHA_PAIR {'label': 'original', 'failure_count': 0, 'errors': []}
F14_SHA_PAIR {'label': 'updated-pair', 'failure_count': 0, 'errors': []}
```

Separate maintenance caveat: the synthetic pass fixture uses faith=0.98 (`test_quality_workflow_outcomes.py:73`), so an eventual legitimate faithfulness threshold above 0.98 would require revisiting positive-control fixture values. That is different from the alleged SHA break and is not occurring with the active 0.9698 threshold. The 20-second timeout (`:105`) bounds each shell invocation; the retry delay is explicitly replaced by a logging stub (`:31`). No timing failure or evidence establishes that this bound is currently too tight. Do not report an observed timeout defect or increase it speculatively.

## Integrated recommendation

Accept only F6 as a low diagnostic improvement from this subset. Treat F2/F8/F14 as rejected claims and F5/F7 as intentional behavior with the evidence above. None establishes that required checks can falsely pass. Keep advisory review findings separate from execution gates; any voluntary correction should carry its own local verification receipt.

## Subsequent closeout

The preceding notes are preserved as pre-correction evidence. F6 was subsequently corrected and regression-tested; the final [disposition](../cross-review.md) and [independent follow-up](task-3-cross-review-independent.md) record the current state. The integrated refreshed suite and artifact checks are bound in [task-3-cross-review-followup.json](task-3-cross-review-followup.json).
