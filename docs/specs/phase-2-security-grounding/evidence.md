> **Status: current-state evidence for a draft spec — no repair is claimed.**

# Evidence and traceability

## E1 — Baseline and review provenance

Current-code authority is released commit
`0ebb72dd0727bf8261a762f472cfdcf38873ddac`, version 1.2.1, tree
`b17f2f3870be4ab7f1e9936dfb23675e0bea6cb9`. The dedicated managed spec worktree
starts at that exact commit. The original dirty checkout is preserved.

The [September executive review](../../reviews/library-review-2026-09-09.md)
supplies the program goals. The October 19 full-library review refreshed those
findings before the release; its original observations used the earlier dirty
checkout, not this released commit. This spec rechecks the selected dependency
and grounding findings against the exact release. It retains bounded sanitized
receipts below, rather than treating review prose as current implementation proof.

Phase 1's artifact repair and incomplete-judge rejection shipped in
[PR #223](https://github.com/Smart-AI-Memory/attune-rag/pull/223).
The former malformed-judge finding is **repaired**, not a Phase 2 open defect.
The [1.2.1 release](https://pypi.org/project/attune-rag/1.2.1/) preserves the
quality threshold and query set. Its release receipts are historical validation,
not fresh Phase 2 test runs.

## E2 — Critical dependencies, refreshed 2026-10-19

The [alert metadata receipt](receipts/dependency-alerts.json) contains 27 open
`uv.lock` alerts: 4 critical, 11 high, 12 medium, across eight normalized package
names. GitHub's runtime scope field alone does not classify optional versus
maintainer use. Code/lock paths must supply that distinction.

| Package | Locked | Critical first fixed | Supported path and bounded exposure assessment |
|---|---|---|---|
| sentence-transformers | 5.5.1 | 5.6.0 | Direct `transformers`/`all` extra. `TransformerRetriever` passes caller-selected `model_name` to `SentenceTransformer`; a malicious local model directory satisfies the advisory's source-control precondition. |
| claude-agent-sdk | 0.2.97 | 0.2.121 | Direct `claude` extra. The only `ClaudeAgentOptions` call has fixed options and no `resume`; the advisory's untrusted-resume entry point is not used by that call. |
| AnyIO | 4.13.0 | 4.14.2 | Optional SDK/network graph, including the Agent SDK. No direct `connect_tcp`/`TLSStream` use in attune-rag. The advisory requires internationalized hostnames and a redirected connection/certificate condition. |
| PyJWT | 2.13.0 | 2.14.0 | Agent SDK → mcp → PyJWT. No direct JWT/JWK verification flow in attune-rag. The critical advisory requires mixed HMAC/asymmetric verification algorithms and specially formatted raw PEM keys. |

Primary advisory records:
[sentence-transformers](https://github.com/advisories/GHSA-jhr6-gm9c-rqjv),
[Agent SDK](https://github.com/anthropics/claude-agent-sdk-python/security/advisories/GHSA-h4mw-j7qp-8mwm),
[AnyIO](https://github.com/agronholm/anyio/security/advisories/GHSA-82r6-8w77-94w6),
[PyJWT](https://github.com/jpadilla/pyjwt/security/advisories/GHSA-ffc3-869f-jxw9).
These assessments are bounded source inspection, not comprehensive audits of
SDK internals, all downstream configurations, or exploit demonstrations.

The base lock closure contains attune-rag plus attrs, Jinja2, jsonschema,
jsonschema-specifications, markdown-it-py, MarkupSafe, mdurl, Pygments, PyYAML,
referencing, rich, rpds-py, structlog, and typing-extensions. None matches an alerted
package. This does not make the supported optional paths safe. The benchmark
workflow installs `[all,dev]` with pip instead of consuming the frozen uv lock;
lock alerts do not identify the exact versions that any particular CI run installed.

Other alerts concern mcp, fsspec, urllib3, virtualenv, AnyIO, and 13 additional
PyJWT advisories. PyJWT 2.15.0 is a candidate to address later listed fixes;
[GHSA-gvp8-978c-rx2q](https://github.com/advisories/GHSA-gvp8-978c-rx2q)
still lists no patched version in this snapshot. Its disposition needs explicit
verification. The spec does not declare all high/medium alerts repaired by a lock bump.

## E3 — Grounding counterexamples replayed against current code

[Grounding replay](receipts/grounding-current-state.json) records a deterministic
provider/client-stub probe with socket connection entry points blocked and the
module import origin asserted. The original review probe was replayed with only
its source-root binding changed to the exact release worktree. Its source hash
and the current pipeline/prompt hashes are retained. Existing Python dependencies
were reused; this is not a fresh installed-wheel or live-model test.

| Case | Current behavior | Requirement |
|---|---|---|
| Four variants, native citations | Every call receives `system=None`; the selected template is not applied | R3 |
| Native long document | 25,025 document characters sent; reported context capped at 20,000 and omits the tail fact | R5 |
| Strict, zero hits, legacy/native requested | Each makes one plain generation call and returns the stub's answer | R4 |

These are pre-repair counterexamples. They confirm policy/evidence transport
gaps; they do not demonstrate a live model hallucination or establish its frequency.
The selected strict policy intentionally changes the existing tested fallback.

## E4 — Existing live quality evidence and its limits

| Run | Code | Displayed mean faithfulness | Answers with unsupported claims | Required threshold |
|---|---|---:|---:|---:|
| [PR quality, attempt 2](https://github.com/Smart-AI-Memory/attune-rag/actions/runs/39294595323/attempts/2) | ac81bbca89dfdec5d66e1788a9b109253741b8b0 | 0.983 | 11/40 (27.5%) | 0.9698 |
| [Merged-main quality](https://github.com/Smart-AI-Memory/attune-rag/actions/runs/39300254562) | 0ebb72dd0727bf8261a762f472cfdcf38873ddac | 0.975 | 11/40 (27.5%) | 0.9698 |

[PR receipt](receipts/release-pr-quality.json) and
[main receipt](receipts/release-main-quality.json) retain the original full-log
hashes and measurement limits. Both runs passed; the means are stdout-rounded.
The two commits have identical trees. The aggregates do not show whether the same
11 answers were flagged. Per-query JSON was runner-local and not uploaded, so
individual historical claims cannot be reconstructed from these receipts.

`benchmark.py` counts one unsupported answer if its verdict has at least one
unsupported claim. It averages per-answer supported-claim ratios for mean
faithfulness; these are different denominators. Its zero-claim count is labeled
refusal, but does not itself prove useful or appropriate refusal. The primary
pass explicitly sets native citations false, so its zero native-citation emission
is expected and says nothing about legacy inline citation presence.

R6 preserves reviewable reports before setting any new answer-level target.
No paid evaluation was performed during this spec pass.

## E5 — Current source anchors

All links below are pinned to the code baseline, not mutable main.

| Requirement | Current authoritative code/configuration |
|---|---|
| R1–R2 | [Dependency metadata](https://github.com/Smart-AI-Memory/attune-rag/blob/0ebb72dd0727bf8261a762f472cfdcf38873ddac/pyproject.toml#L42), [lock](https://github.com/Smart-AI-Memory/attune-rag/blob/0ebb72dd0727bf8261a762f472cfdcf38873ddac/uv.lock), [transformer loader](https://github.com/Smart-AI-Memory/attune-rag/blob/0ebb72dd0727bf8261a762f472cfdcf38873ddac/src/attune_rag/transformer.py#L69), [subscription options](https://github.com/Smart-AI-Memory/attune-rag/blob/0ebb72dd0727bf8261a762f472cfdcf38873ddac/src/attune_rag/auth.py#L182) |
| R3–R4 | [Legacy fallback](https://github.com/Smart-AI-Memory/attune-rag/blob/0ebb72dd0727bf8261a762f472cfdcf38873ddac/src/attune_rag/pipeline.py#L336), [native path](https://github.com/Smart-AI-Memory/attune-rag/blob/0ebb72dd0727bf8261a762f472cfdcf38873ddac/src/attune_rag/pipeline.py#L458), [variant definitions](https://github.com/Smart-AI-Memory/attune-rag/blob/0ebb72dd0727bf8261a762f472cfdcf38873ddac/src/attune_rag/prompts.py#L58), [provider system support](https://github.com/Smart-AI-Memory/attune-rag/blob/0ebb72dd0727bf8261a762f472cfdcf38873ddac/src/attune_rag/providers/claude.py#L164) |
| R5 | [RagResult evidence contract](https://github.com/Smart-AI-Memory/attune-rag/blob/0ebb72dd0727bf8261a762f472cfdcf38873ddac/src/attune_rag/pipeline.py#L67), [full native documents and capped rendering](https://github.com/Smart-AI-Memory/attune-rag/blob/0ebb72dd0727bf8261a762f472cfdcf38873ddac/src/attune_rag/pipeline.py#L529), [context cap](https://github.com/Smart-AI-Memory/attune-rag/blob/0ebb72dd0727bf8261a762f472cfdcf38873ddac/src/attune_rag/prompts.py#L179), [document/citation structures](https://github.com/Smart-AI-Memory/attune-rag/blob/0ebb72dd0727bf8261a762f472cfdcf38873ddac/src/attune_rag/providers/base.py#L12) |
| R6 | [Per-query verdicts and denominators](https://github.com/Smart-AI-Memory/attune-rag/blob/0ebb72dd0727bf8261a762f472cfdcf38873ddac/src/attune_rag/benchmark.py#L475), [primary legacy selection](https://github.com/Smart-AI-Memory/attune-rag/blob/0ebb72dd0727bf8261a762f472cfdcf38873ddac/src/attune_rag/benchmark.py#L1145), [workflow report/retry handling](https://github.com/Smart-AI-Memory/attune-rag/blob/0ebb72dd0727bf8261a762f472cfdcf38873ddac/.github/workflows/benchmark.yml#L35) |
| R7 | [Thresholds](../release-quality-baseline/thresholds.json), [POLICY](../../POLICY.md), [source/distribution parity check](../../../scripts/check_distribution.py) |

**Premise correction, 2026-10-19:** older Phase 1/goal-map drafting statuses and
the safe-abstention README lag released code. `pipeline.py` already sets the
bundled keyword threshold to 5 while explicit BYO corpus construction retains its
retriever-specific default. This phase does not reopen that retrieval calibration.
The goal map is corrected with a dated note; historical spec receipts are preserved.

## E6 — What this pass validated

Read-only code/lock inspection, a fresh GitHub alert query, primary advisory
checks, and the offline current-behavior replay ground this draft. Link, JSON,
source-hash, requirement/task-coverage, and changed-path checks are saved in
[spec validation](receipts/spec-validation.json) after drafting. No dependency
update, new install, exploit, model download, provider call, CI dispatch, full
implementation test run, or publication occurred.
