# Grounding policy and evaluation evidence

This guide describes the Phase 2 candidate. These changes are not in the published
1.2.1 package. Install the candidate wheel for verification; publication is separate.

## Run the offline example

From a clone containing this change, build and install its wheel:

```bash
python -m pip install build
python -m build --wheel
python -m pip install --force-reinstall dist/attune_rag-*.whl
python examples/grounding_evidence.py
```

The example creates a synthetic Markdown corpus in a temporary directory outside
the checkout. It uses the base install, a named-provider strict query with no
hits, and a recording stub for fallback and native transport. It makes no model
calls. The final output includes the local refusal, fallback disclosure, ordered
native source documents, and a fixed unsupported-claim report fixture. These are
transport and reporting demonstrations, not live quality measurements.

## Choose a policy

`run()` assembles a prompt. `run_and_generate()` optionally calls a provider.
Both validate empty/whitespace queries and unknown variants before retrieval or
provider initialization. The default remains `citation`.

| Variant | When sources are retrieved | When retrieval returns zero hits |
|---|---|---|
| `baseline` | Use context, disclose missing information, avoid invented APIs. | Call the plain generator with the existing ungrounded fallback instruction. |
| `strict` | Use only source facts; refuse when evidence is insufficient. | Return exactly `The provided context does not cover this question.` locally. |
| `citation` | Use only source facts and attribute factual claims. | Call the plain generator with the existing ungrounded fallback instruction. |
| `anti_prior` | Ignore prior ecosystem knowledge and use source facts. | Call the plain generator with the existing ungrounded fallback instruction. |

Strict local refusal initializes and calls no answer provider, including when the
provider is named `claude` and its SDK or key is absent. Explicitly configured
query expansion or reranking can still run during retrieval and can have their
own cost or failure behavior. `run()` emits strict refusal instructions and
metadata; it does not generate an answer.

Native citations receive the same variant semantics through the provider's
`system` parameter. Native attribution replaces literal `[Pn]` markers. Source
titles and bodies are data, including instruction-like text or literal passage
tags. Payload separation tests do not prove resistance to every prompt injection.
A provider without native support uses the legacy prompt path.

## Inspect the result

| Field | Meaning |
|---|---|
| `fallback_used` | Retrieval returned zero hits. Preserved for compatibility, including strict local refusal. |
| `abstained` | The pipeline enforced strict empty-hit abstention. It does not classify a model's refusal after nonempty retrieval. Defaults to `False` in existing constructors. |
| `context` | Exact legacy evidence block, or an uncapped rendering of the supplied native documents. Empty on fallback or local refusal. |
| `generation_documents` | Immutable ordered native `CitationDocument` values with exact titles and text. Empty on legacy/fallback/abstention paths; defaulted for existing constructors. |
| `citation` | Retrieval provenance, including preview excerpts. It is not the complete generation evidence. |
| `claim_citations` | Model-asserted attribution. A document index refers to the same position in `generation_documents`; attribution is not an entailment verdict. |

Legacy evidence retains its 20,000-character rendered cap. Native documents retain
their existing full text, including whitespace, and have no second evaluation
cap. The judge receives `context` from that same evidence selection. If the
provider or judge rejects the input size, the evaluation fails explicitly; it does
not silently trim evidence and report a comparable score.

`generation_documents` is excluded from the result's repr. `dataclasses.asdict()`
includes its full text. Existing `context` and `augmented_prompt` also contain
source text. Apply your own logging, serialization, and access policy to all three.

## Inspect benchmark reports

The quality workflow uploads `grounding-evaluation-<run>-<run-attempt>` with
**14-day retention**. Each `attempt-N` contains `benchmark.json` and an attempt
diagnostic. The existing retry preserves attempt 1 before writing attempt 2.
Missing/malformed reports leave explicit diagnostics; required validation still
fails. Upload errors also fail CI.

Automatic uploads are limited to this workflow's fixed public/synthetic bundled
benchmark inputs. Reports contain queries, generated answers, and full source
passages. The library does not automatically upload arbitrary user corpora or
queries. Do not add private inputs to this upload workflow without a separate
privacy decision. Reports do not include API keys or request headers.

The additive `manifest` records schema version, code/query hashes, resolved
dependencies, and GitHub run/attempt identifiers. Code SHA is captured only from
a verified source checkout whose reported revision contains the executing module; an installed wheel does not inherit the consuming
application’s Git SHA. Tracked changes are disclosed when a checkout is available. The additive `evaluation` holds
per-pass evidence, including answers saved before judging, actual supplied text,
ordered native documents, retrieval sources, claim attributions, supported and
unsupported claim lists, reasoning, configured model IDs, variant, and fallback/
abstention flags. Native document indices are zero-based; legacy `[Pn]` headers
identify the supplied passages and their source paths. Retrieval-source lists
describe retrieval order and can include passages truncated out of legacy evidence.

Provider interfaces do not return actual served model IDs; those values are
explicitly unavailable, distinct from configured IDs. Corpus name/version and an
existing body fingerprint are recorded when available. The body fingerprint
excludes aliases and summaries and does not establish complete cache identity.
Unavailable provenance is labeled; it is not reconstructed from guesses.

Only completed, validated measurements occupy the existing `faithfulness_*` gate
keys. Partial records stay under `evaluation` and have no fabricated verdicts.
Each summary separates requested, generated, scored, and unscored answers:

- Mean per-answer faithfulness is the average of supported/total claim ratios.
  Existing zero-claim answers retain their score of 1.0.
- Unsupported-answer rate counts scored answers with any unsupported claim;
  its denominator is scored answers, not total claims.
- Supported/unsupported claim counts, zero-claim answers, exact context refusals,
  empty answers, other zero-claim answers, and pipeline abstentions are explicit.
  Refusal appropriateness and answer usefulness remain **not assessed**.

The example's fixed answer claims both inspection and deletion, while its source
only states inspection. Its mean faithfulness is 0.5, and its unsupported-answer
rate is 1/1. Those values measure different things. A high average or a refusal
does not by itself establish useful, fully supported answers.

The locked 40-query set, retrieval thresholds, mean-faithfulness threshold 0.9698,
provider-unavailability classification, and single transient retry remain unchanged.
New answer-level blocking targets require a separately approved decision.
