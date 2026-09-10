# attune-rag 1.2.1 release preparation

**Status: local validation passed; release in progress.** Patrick requested the release-execute workflow after reviewing Phase 1. The target is a patch release because these changes repair packaging and validation contracts.

The release branch starts at current origin/main, `f00815d22cfa28c0abca6ab7b4bcd5d6a1059dcd`. It includes the approved Phase 1 implementation, a version bump, repository-pinned formatting, and source-reviewed benchmark help. Existing thresholds, query lock, and dependency versions are unchanged. The lockfile changes only the attune-rag project version; both local uv and CI-pinned uv 0.9.22 validate it.

[Local release receipt](receipts/release-1.2.1-local.json): **1,505 passed, 2 XFAIL, 78 XPASS; 93.03% coverage**. The built 1.2.1 sdist and wheel pass resource/source checks and identical ordered retrieval on 40 locked queries. Actual publication scripts accept the verified artifacts and reject modified copies. See [distribution](receipts/release-1.2.1-distribution.json), [negative controls](receipts/release-1.2.1-negative-controls.json), and [suite](receipts/release-1.2.1-tests.xml).

All measurements here are local macOS/Python 3.11 using prepared dependencies without model calls. The original Phase 1 records remain historical evidence for their 1.2.0 working snapshot. Remote CI, merge, tagging, and PyPI verification must be recorded separately before this release is complete.

The help generator's signature-based staleness check missed benchmark behavior changes. Eleven help pages were corrected directly against the source and passed existing schema, lint, and help-parser checks. Generation timestamps and source hashes were retained; status is source-reviewed. No paid regeneration or API preflight was run.

Pre-commit whitespace mutators preserve the raw JSON/XML receipts so recorded checksums remain valid. JSON validation and secret scanning still apply; cryptographic digests and deliberate redaction sentinels are reviewed individually in the existing secrets baseline.

All repository-pinned pre-commit checks pass, including secret scanning. The [scanner audit](receipts/release-1.2.1-secret-audit.md) records individually reviewed hashes, test sentinels and fixture paths; scanner settings remain intact.

## First remote run and Windows follow-up

[PR #223](https://github.com/Smart-AI-Memory/attune-rag/pull/223) passed lint, lockfile, quality, performance, artifact parity, and all eight Linux/macOS test jobs. The cloud artifact checker used normal build isolation and validated 45 members with identical ordered results on all 40 queries (P@1/R@3 both 1.0).

The [first test run](https://github.com/Smart-AI-Memory/attune-rag/actions/runs/34430634861) found the same three test portability failures on each of Python 3.10–3.13 on Windows: two assertions expected unescaped path separators in a diagnostic that deliberately uses `repr`, and one test read UTF-8 formatter output using the Windows default encoding. The follow-up makes the expected path representation explicit and reads/writes the new performance test fixtures as UTF-8. Production behavior, metric thresholds, and diagnostic redaction remain unchanged.

The two affected test files pass together locally: **189 passed** using `pytest tests/unit/test_benchmark.py tests/unit/test_format_perf_delta.py -q -rN`. This targeted follow-up does not replace the pending rerun of the full remote matrix; the original local suite and artifact receipts above remain measurements of the first release candidate.
