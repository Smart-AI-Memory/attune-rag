# Milestone 1 validation

**Status: local implementation checks passed; remote CI pending.**
**Baseline:** released `0ebb72dd0727bf8261a762f472cfdcf38873ddac`.

Published optional requirements now reject the four critical affected ranges.
The same refresh requires compatible patched MCP, fsspec, urllib3, and development
virtualenv versions. Base dependencies, runtime source, library version, models,
retrieval query SHA, and quality/performance baselines are unchanged. `[all]`
now includes the full runtime-feature union, including the Agent SDK.

| Check | Observed result |
|---|---|
| Snapshot advisories | All 27 advertised affected ranges excluded by the candidate lock; explicit per-advisory dispositions retained |
| OSV lookup | 245 package/version lookups covering lock and both installed sets; no known findings returned |
| Resolution | 240/240 cases pass: 3 platforms × 4 Python versions × 10 install paths × minimum-secure/current sets |
| Installed extras | Python 3.10 minimum-secure and Python 3.13 current `[all,dev]` installs pass `pip check` |
| Full offline suites | 1,558 passed, 2 XFAIL, 78 XPASS in each environment |
| Final contracts | 25 passed in each environment, including actual minimum/current SDK option/result types with only the query boundary replaced |
| Published-wheel bounds | 12 affected-version requests rejected; the same 12 requests resolve against the hash-verified published 1.2.1 wheel; secure minimum/current sets resolve |
| Source/sdist/wheel | Package resources match; ordered results agree for all 40 queries; P@1/R@3 both 1.0 |
| Base-only installed wheel | 15 distributions including attune-rag; synthetic keyword query passes with no optional SDK, numpy, model2vec, sentence-transformers, or torch import |
| Formatting and lock | Ruff and changed-test Black checks pass; CI's uv 0.9.22 accepts the lock |

The full suites ran before the final real-SDK-types test was added; the subsequent
25-test contract runs include that addition. The native/legacy grounding code
remains unchanged. Real-SDK type tests replace query execution and do not launch
the Claude CLI or contact a model.

The lock changes eight alerted packages plus three required transitive packages.
The new virtualenv needs packaging 26.3 and python-discovery >=1.6.1; it and the
updated AnyIO require typing-extensions 4.16 on relevant Python versions. No
unrelated package-version upgrades are included.

[Local validation](receipts/m1-local-validation.json),
[resolution matrix](receipts/m1-resolution-matrix.json),
[wheel controls](receipts/m1-wheel-contracts.json),
[advisory dispositions](receipts/m1-advisory-dispositions.json), and
[OSV response](receipts/m1-osv-audit.json) retain the evidence and limits.

The PyJWT options-mutation advisory still has no listed first-fixed version.
Selected PyJWT versions are outside its published affected range; source inspection
of installed 2.15.1 confirms options are copied before mutation. This does not
assign a new upstream first-fixed release or constitute an exploit test.

The 240 cases validate metadata and wheel availability for Linux x86_64 with
glibc 2.28+, macOS arm64, and Windows x86_64. They do not prove foreign-platform
execution or other architectures. The inherited 12-job test matrix remains the
execution evidence. GitHub's main-branch alerts remain open until the repair is
merged and reanalyzed; no alert was dismissed or suppressed. No paid model run,
merge, or publication occurred.
