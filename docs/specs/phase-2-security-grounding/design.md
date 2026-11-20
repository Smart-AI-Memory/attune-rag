> **Status: approved; M1 merged, M2 implemented locally pending candidate CI acceptance.**

# Design

## Dependency remediation

Start from fresh advisory metadata, then derive actual dependency paths from the
lock and package metadata. The current critical first-fixed boundaries are
sentence-transformers 5.6.0, claude-agent-sdk 0.2.121, AnyIO 4.14.2, and PyJWT
2.14.0. PyJWT 2.15.0 is the proposed candidate floor because later noncritical
advisories also affect 2.14.0. These are advisory boundaries, not verified
cross-platform compatibility results.

Raise direct floors in every affected extra, including repeated declarations in
`all`. Put needed transitive security bounds in the extras whose graphs include
those packages; do not pull optional dependencies into the base graph to enforce
a global constraint. Derive that placement from the graph rather than assuming
that only `claude` reaches AnyIO. Regenerate the lock using the repository's
normal tooling and inspect unrelated resolver changes.

Published wheel metadata and deliberately conflicting resolver requests prove the
protection reaches downstream users. Lockfile audit proves a different property:
the selected maintainer environment avoids affected versions. Both are needed.
Refresh high/medium dispositions at acceptance, including the PyJWT advisory whose
snapshot has no listed first-fixed release. Do not infer its resolution solely
from a larger version number.

The proposed `all` meaning is all runtime feature extras: bundled help, Claude
(including subscription SDK), Gemini, static embeddings, and transformers.
`author` and `dev` remain maintainer tools. Test the union against metadata so
future duplicated declarations cannot silently diverge.

An incompatible secure floor is an implementation stop for the affected path:
propose a supported alternative or an explicit scope/support change. Do not ship
a critical vulnerable version because exploitation was not seen in our tests.
M1 can be reviewed and released without waiting for M2; release authorization is
separate from spec approval.

## Shared policy, separate transports

Extract or reuse one internal definition of variant semantics. Legacy rendering
continues to produce a prompt; native rendering supplies equivalent policy through
the existing `system` argument alongside structured citation documents. The native
`citation` variant asks for native attribution rather than literal `[Pn]` markers.
No new provider method or public export is needed.

Validate query and variant before dispatch. Retrieve once for the chosen path.
When strict has no hits, short-circuit generation before named-provider construction.
Use the existing strict refusal sentence as a shared internal constant, avoiding
wording drift between prompt assembly, local generation, and tests.

| Evidence | Variant | Generation behavior | Result flags |
|---|---|---|---|
| Empty | strict | Local exact refusal; no provider initialization/call | fallback_used=True, abstained=True, used_native_citations=False |
| Empty | other valid variants | Existing plain generative fallback | fallback_used=True, abstained=False, used_native_citations=False |
| Nonempty | any valid variant | Selected policy via supported transport | fallback_used=False, abstained=False; native flag reflects actual call |
| Any | invalid query/variant | Input error before generation | No successful result |

`abstained` describes a pipeline decision only. It is not a semantic detector for
a model's refusal or proof that a response after nonempty retrieval is grounded.
The fallback template for other variants stays explicit about missing context;
the current default `citation` variant remains unchanged.

## Evidence representation

Keep legacy's 20,000-character rendered context behavior. Native currently sends
full document bodies; retain that behavior in this repair to avoid changing its
evidence selection and citation offsets without measurements.

Proposed additive field: `RagResult.generation_documents`, defaulting to an empty
tuple, reuses immutable `CitationDocument` values for the ordered native payload.
No new public type is exported. Capture these exact documents once, pass them to
the provider, and derive the native evaluation context from the same values,
preserving full text and source order without a second cap. Existing consumers
continue to receive `context`; new consumers can inspect exact native documents.
Legacy evidence remains available as the exact rendered `context`.

Do not use retrieval excerpts as a substitute for supplied evidence. Do not
remap native document indices to different retrieval hits. Preserve original
whitespace in structured evidence even when the readable rendering adds headings.
A long native context is an accuracy repair with a memory/judge-cost tradeoff;
address future context budgeting in a separately measured change. A judge input
limit must fail or be explicitly unavailable, never silently trim evidence and
still claim a completed comparable score.

## Evaluation artifacts and targets

Extend the existing JSON report and workflow upload, rather than introducing an
observability service. Retain each attempt before a retry overwrites its working
path. Recommend 14-day CI artifact retention, with code/run/query identifiers in
the manifest and explicit partial/completed state. Test the actual extracted
workflow scripts with stubs; do not pay for provider calls to test upload logic.

The report must expose both the mean supported-claim ratio per answer and the
count/rate of answers with any unsupported claim. Keep per-query claim evidence
so a reviewer can inspect the difference. Record appropriate refusal and pipeline
abstention separately from answerable questions that received no useful answer.

The previous 11/40 unsupported-answer observations are historical baselines with
missing per-query artifacts, not a new acceptable error budget. Keep the current
CI threshold; do not add an arbitrary answer-level blocking target in this phase.
After inspectable representative evidence exists, propose a numerical target with
its dataset, refusal/usefulness tradeoff, variance, cost, and approval separately.
The deterministic contract repairs can be accepted without claiming a measured
reduction in live unsupported claims.

## Compatibility and rollback

Defaulted dataclass fields preserve existing keyword and positional construction;
check this explicitly and update relevant serialization/snapshot expectations.
Describe strict's zero-hit behavior change in public usage docs and changelog at
implementation. Keep the default variant and all non-strict fallbacks. Select the
release version after reviewing dependency support and API compatibility.

M1 and M2 should use separate PRs. A rollback of M2 retains M1's security bounds.
Do not recommend downgrading to known vulnerable dependencies as a routine rollback;
fix forward or disable the affected optional feature if necessary. Neither this
design nor approval of the spec authorizes merge or publication.
