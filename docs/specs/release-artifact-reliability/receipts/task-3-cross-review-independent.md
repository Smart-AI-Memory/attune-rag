# Independent follow-up review: F3, F9, F6

Conclusion: no actionable defects found within the requested correction scope.

Scope: read-only inspection of the frozen F3/F9 controlled benchmark diagnostics and calibration outcomes, plus the frozen F6 final workflow diagnostic and corresponding tests. This is source/diff evidence, not a test-run receipt. The parent owns integrated behavioral validation. No model/provider/network/release calls, product edits, or git mutations were made by this follow-up review.

## F3: actionable local diagnostics without arbitrary exception payloads

- `src/attune_rag/benchmark.py:50` defines the dedicated private validation-error type. `:913` admits detail only when `type(exc) is _BenchmarkValidationError`; an ordinary ValueError with similar wording, or a subclass, does not gain this privilege. Generic errors retain stage/class only at `:912` and `:917`.
- Inspected every constructor and every `_bounded_number` callsite using `rg -n '_BenchmarkValidationError|_bounded_number' src/attune_rag/benchmark.py`. Schema checks at `:129-144` use fixed messages, without row IDs, query contents, expected path contents, or parser exception text. YAML/Unicode replacements at `:119-127` include the requested file path via repr and controlled syntax/encoding guidance. File-path disclosure is intentional troubleshooting context, not a promise of removing all user data from output.
- Numeric diagnostics at `:656-670` use fixed labels from `:678-698` and `:935-938`, state the valid domain, and do not interpolate the invalid value. Overflow from a huge integer is converted to a controlled failure. Specific validation precedes strict serialization at `:701`; unrelated serialization exceptions remain class-only at the CLI boundary.
- Primary provider exception handling at `:1102-1111` still emits only the controlled stage and exception class. Secondary provider exceptions reach the generic class-only handler at `:909-917`; primary completion remains intact. No new raw-provider exception printing is introduced.
- Test evidence inspected: `tests/unit/test_benchmark.py:864` invalid CLI limits, `:908` schema redaction, `:935` malformed YAML/UTF-8 redaction with path context, `:958` invalid measurement labels, `:976` arbitrary ValueError redaction in retrieval/secondary phases, and `:692` existing typed-provider outage/redaction matrix. These exercise the public main boundary and actual JSON output with an AsyncMock provider; they do not invoke a provider. Test execution was not repeated here.
- Deliberate scope limit: the pre-existing retrieval RuntimeError setup-message branch at `benchmark.py:1011-1016` is unchanged. This assessment concerns the new F3 diagnostic path and provider boundaries; it is not an assertion that every pre-existing diagnostic throughout the CLI is redacted.

## F9: calibration outcome consistency

- `src/attune_rag/benchmark.py:994-1007` retains the actual calibration calculation and stores its result, then sets retrieval and faithfulness to skipped with reason calibration_only. This occurs before `_run_benchmark` (`:1010`) and the later faithfulness dispatch (`:1081`); success leaves no pending benchmark stage even when --with-faithfulness was supplied.
- `tests/unit/test_benchmark.py:1007-1039` executes real calibration/top-score collection with deterministic local pipeline results. It asserts return 0, the exact two skipped statuses/reason, recommendation 1.0, absence of retrieval/primary measurement payloads, and no awaited provider. The test does not replace the calibration implementation with a success stub.

## F6: incomplete validation diagnostic

- `.github/workflows/benchmark.yml:302` preserves `always() && steps.check.outputs.rc != '0' && steps.check.outputs.rc != '1'`; `:311` still exits 1. The guard therefore does not become fail-open when setup or an earlier step never produces a checker result.
- `:304` transfers the output through an environment variable. `:306` tests the safely quoted value without executing/interpolating it as shell source. Empty/unset output gets the controlled did-not-complete message (`:307`); nonempty invalid output retains existing validation guidance (`:309`). No command or release behavior changed.
- `tests/unit/test_quality_workflow_outcomes.py:322-340` loads the actual workflow YAML and executes its final shell body for empty, 2, and 42. It asserts exit 1 and the corresponding mutually exclusive message. `:305-319` retains independent guard/non-continue-on-error checks. Its helper runs Bash directly, so this provides shell-behavior coverage; the GitHub Actions expression itself is inspected/asserted, not evaluated by a hosted runner here.

## Files reviewed

All short paths/line anchors above resolve under `/Users/patrickroebuck/attune-rag/`.

- `/Users/patrickroebuck/attune-rag/src/attune_rag/benchmark.py` SHA-256 `1542a1114e4d0f09f85f303eae2c2e59e1066a3f3d65e2ab7def5a6082015ac1`
- `/Users/patrickroebuck/attune-rag/tests/unit/test_benchmark.py` SHA-256 `31de2041ad59e515022ae201df106893d362c011807a4bafe71c53ad19f4e1f4`
- `/Users/patrickroebuck/attune-rag/.github/workflows/benchmark.yml` SHA-256 `c161874b1ab2de87b50384fd9919e60a9b33c132c0448825e3790761867770f7`
- `/Users/patrickroebuck/attune-rag/tests/unit/test_quality_workflow_outcomes.py` SHA-256 `49e15c9f2e6907766bd7acb04ccf5a4c8472c6380e14314df38fed51df7f9098`

Review confidence/limits: source review complete for these three small corrections; no remaining actionable finding. No claim is made about remote CI execution or unrelated pre-existing behavior. No subjective numerical score is used as a substitute for the parent's behavioral checks.
