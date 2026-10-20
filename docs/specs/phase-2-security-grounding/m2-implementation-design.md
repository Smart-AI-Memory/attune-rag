# Milestone 2 implementation design

Prepared before runtime edits. Scope: T3–T5 of the approved Phase 2 spec.
PR #224 merged as e95a3b5; its tree equals the tested candidate. Milestone 2
starts on codex/phase2-grounding from that merge.

## Evaluation evidence first (T3)

Keep completed gate measurements in their existing keys. Add a separate
`evaluation` object for in-progress/partial per-pass records so that a failed
judge cannot masquerade as `faithfulness_legacy` completion or alter the existing
outage classifier. Persist each generated answer before judging it and each
validated verdict afterward. Record safe error classes, never exception bodies,
request headers, or credentials. A missing verdict has no score.

Add schema/run/code/query/dependency provenance and available corpus identity.
Use explicit unavailable values for provider-returned model IDs or corpus hashes
that the current interfaces do not expose. Report scored, generated, and requested
denominators separately; distinguish exact refusal text, pipeline abstention,
empty answers, and other zero-claim answers without inferring semantic usefulness.
Preserve current mean-faithfulness and zero-claim scoring contracts.

The existing CI loop saves every attempt before retrying. Always upload only its
approved bundled public benchmark inputs, with 14-day retention. Missing or invalid
reports produce an explicit diagnostic; they do not alter benchmark exit or retry
rules. Replay the actual workflow scripts against pass, regression, malformed,
outage, authentication, and missing-report stubs.

## Grounding policy and exact evidence (T4)

Validate the query and variant before retrieval/provider initialization. Retrieve
once, then enforce strict zero-hit abstention before resolving a named provider.
Other variants retain the existing plain fallback. Keep run() as prompt assembly.
Reuse a shared internal assembly helper rather than duplicating retrieval or
strict decisions across native and legacy paths.

Keep legacy prompt text stable where possible. Factor variant instructions so
native system messages preserve the same only-source/refusal semantics, adapting
citation syntax to native attribution. Structured source documents remain data;
instruction-like text and literal delimiters stay in document content.

Append defaulted RagResult fields: abstained=False and immutable ordered
generation_documents=(). Native generation captures exact CitationDocument values
once and supplies those same values to the provider. Render evaluation context
from them without stripping whitespace or applying the legacy 20,000-character
cap. Keep legacy context and its cap unchanged. Preserve citation indices, explicit
model/token overrides, lazy optional imports, and old constructors. Suppress newly
available full-document text from repr; document explicit serialization exposure.

## Experiments and limits (T5)

Use all-variant/path recording tests, provider initialization traps, and evidence
fixtures below/at/above the legacy cap, with Unicode, duplicate titles, literal
delimiters, and tail-only facts. Remove policy forwarding, abstention enforcement,
and exact evidence independently as negative controls and verify targeted failures.
Run API/golden tests, source/sdist/wheel parity, and the documented synthetic corpus
journey outside the checkout. Preserve locked queries and thresholds. Obtain a
different-model read-only review of an immutable candidate before PR readiness.

Rejected: a common truncation budget (changes native evidence selection), retrying
malformed judgments (changes failure policy), and counting partial means as gate
completion (hides missing evidence). Payload checks are not live quality or prompt
injection proofs. Paid CI validation requires its separately approved allowance.

Completion review: pending.
