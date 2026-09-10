# attune-rag 1.2.1 release preparation

**Status: local validation passed; release in progress.** Patrick requested the release-execute workflow after reviewing Phase 1. The target is a patch release because these changes repair packaging and validation contracts.

The release branch starts at current origin/main, `f00815d22cfa28c0abca6ab7b4bcd5d6a1059dcd`. It includes the approved Phase 1 implementation, a version bump, repository-pinned formatting, and source-reviewed benchmark help. Existing thresholds, query lock, and dependency versions are unchanged. The lockfile changes only the attune-rag project version; both local uv and CI-pinned uv 0.9.22 validate it.

[Local release receipt](receipts/release-1.2.1-local.json): **1,505 passed, 2 XFAIL, 78 XPASS; 93.03% coverage**. The built 1.2.1 sdist and wheel pass resource/source checks and identical ordered retrieval on 40 locked queries. Actual publication scripts accept the verified artifacts and reject modified copies. See [distribution](receipts/release-1.2.1-distribution.json), [negative controls](receipts/release-1.2.1-negative-controls.json), and [suite](receipts/release-1.2.1-tests.xml).

All measurements here are local macOS/Python 3.11 using prepared dependencies without model calls. The original Phase 1 records remain historical evidence for their 1.2.0 working snapshot. Remote CI, merge, tagging, and PyPI verification must be recorded separately before this release is complete.

The help generator's signature-based staleness check missed benchmark behavior changes. Eleven help pages were corrected directly against the source and passed existing schema, lint, and help-parser checks. Generation timestamps and source hashes were retained; status is source-reviewed. No paid regeneration or API preflight was run.

Pre-commit whitespace mutators preserve the raw JSON/XML receipts so recorded checksums remain valid. JSON validation and secret scanning still apply; cryptographic digests and deliberate redaction sentinels are reviewed individually in the existing secrets baseline.

All repository-pinned pre-commit checks pass, including secret scanning. The [scanner audit](receipts/release-1.2.1-secret-audit.md) records individually reviewed hashes, test sentinels and fixture paths; scanner settings remain intact.
