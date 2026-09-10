# Release 1.2.1 secret-scanner audit

Status: completed; no unexpected credential finding. Only `.secrets.baseline` was edited in the release checkout. Product files, tests, hooks and all 41 raw receipt files retained their bytes.

## Scope and results

- Checkout: `/private/tmp/attune-rag-release-1.2.1`.
- Scanner: cached pre-commit detect-secrets 1.5.0 installation at `/Users/patrickroebuck/.cache/pre-commit/repo93elicgr/py_env-python3.11/bin/detect-secrets`.
- Scanned 711 tracked/staged paths, including the five newly staged release-specific receipts. The baseline itself is excluded as scanner bookkeeping, consistent with its existing baseline filter.
- Command: `detect-secrets scan --no-verify --baseline <temporary baseline copy> <tracked paths>`; complete exact argv is saved in `/private/tmp/attune-rag-release-secret-scan/command.json`.
- Added 275 individually audited false positives across 18 files: 271 digest findings, 2 deliberate test sentinels, and 2 pytest temporary fixture paths.
- Detector categories: {'Hex High Entropy String': 271, 'Base64 High Entropy String': 2, 'Secret Keyword': 2}.
- Existing 19 baseline entries and every detector/filter/configuration/metadata value were preserved. No exclusion was added or broadened. New entries use supported `is_secret: false` audit verdicts; `is_verified: false` remains because no network verification was performed.

## Independent proof

Each candidate was matched to its scanner `hashed_secret` using the scanner's SHA-1 fingerprint, then inspected in its full source context. All 271 hex candidates are 40-character Git commit IDs or 64-character SHA-256 values in explicit source/resource/artifact/query/snapshot/file/receipt digest fields. 254 digest entries were independently corroborated by hashing available local source/artifact/receipt bytes or resolving the named Git commit. Remaining historical digests are classified from the structured receipt field and hashing writer contract; they are not claimed as freshly recomputed. All current 1.2.1 artifact hashes were directly recomputed from the actual files named in the release distribution receipt and matched.

The two keyword findings at `tests/unit/test_benchmark.py:655,984` are uppercase symbolic exception-message sentinels in `_provider_exception` / `test_only_typed_primary_provider_outages_are_retryable` and `test_arbitrary_value_errors_remain_redacted`. Their test data flow injects synthetic exceptions and checks that messages are absent from output; these constants are not used for authentication.

The two Base64 findings at `docs/specs/release-artifact-reliability/receipts/task-5-initial-environment-tests.xml:1,30` are complete absolute pytest `tmp_path` directory values in preserved failure diagnostics. Their scanner fingerprints match those filesystem paths. The parent explicitly confirmed including these independently established path false positives. No receipt text was changed.

## Validation and limits

The merged baseline covers every candidate in the captured final scan. All original baseline entries/configuration compare equal, and all raw receipt hashes before/after this audit match. Detailed candidate-to-field evidence, file counts, and baseline hashes are saved to `/private/tmp/attune-rag-release-secret-scan/audit-proof.json`; raw scan results are `/private/tmp/attune-rag-release-secret-scan/scan-raw.json`. Scanner stdout/stderr were captured, not displayed. The default global scanner was initially incompatible with the baseline plugin set; the matching cached pre-commit version completed successfully without changing configuration.

No model calls, network verification, broad allowlists, product/test changes, or publication occurred in this subtask. A scanner plus contextual audit is bounded evidence, not proof that the repository contains no possible secret. New receipt content after this scan requires an incremental scan.
