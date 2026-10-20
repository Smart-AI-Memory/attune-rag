> **Status: approved; M1 merged, M2 implemented locally pending candidate CI acceptance.**

# Phase 2 requirements

Current state is pinned in [evidence](evidence.md). Desired behavior below is not
a claim that the release already implements it. The confirmed strict-mode choice
is [D3](decisions.md#d3--strict-mode-abstains-without-evidence-confirmed).

## M1 — Dependency security

### R1 — Exclude known critical affected versions in supported installations

Repair all four critical advisory families in the snapshot, including optional
paths whose exploit preconditions are not exercised by attune-rag itself.
Update published dependency requirements as well as `uv.lock`; a maintainer's
constraint file or a modern resolver choosing a newer version is insufficient
protection for downstream installs.

Acceptance:

- Refresh the advisory snapshot at implementation and before milestone acceptance.
  Record affected ranges, first fixed releases, resolved versions, dependency
  paths, Python/platform markers, and the advisory database timestamp/source.
- The supported lock and tested resolved environments contain zero known critical
  affected versions. Published metadata must prevent selection of each critical
  affected range whenever the corresponding dependency path is installed.
- Resolver negative controls deliberately request affected versions and fail with
  a dependency conflict. Fixed-boundary and current compatible resolution succeed.
  Do not execute exploits or load an untrusted model to prove version exclusion.
- Cover base, `attune-help`, `claude`, `gemini`, `embeddings`, `transformers`,
  `author`, `all`, and `dev`, including the existing `[all,dev]` CI combination.
  Distinguish runtime extras from maintainer tools using the actual graph.
- Record every remaining high/medium advisory as repaired, blocked upstream, or
  deferred with exposure, reason, owner, and a proposed follow-up. An unknown
  fixed release remains unknown until verified. No silent suppression or blanket
  claim that all 27 alerts affect the default install.

### R2 — Preserve supported behavior and the lightweight default

Keep Python 3.10–3.13 and the existing Linux/macOS/Windows test coverage. Keep the
keyword path independent of provider SDKs and heavyweight model packages.

Acceptance:

- Resolve dependency metadata for the supported platform/Python matrix. Exercise
  isolated installs on the oldest and newest tested Python versions for changed
  extras, with `pip check`, distribution metadata, and benign adapter smoke tests.
  Record any untested platform installation separately from successful resolution.
- Verify both minimum secure dependency bounds and a current compatible set;
  retain the existing cross-platform suite. No model download is required for
  lazy-import and injected-encoder tests. Do not claim a real model-load test from
  an injected encoder.
- Run the structured-completion adapter offline against the repaired Agent SDK
  option/result interfaces. Retain `setting_sources=[]`, no tool use, and no
  forwarding of untrusted `resume` or arbitrary CLI options.
- Build the wheel and inspect its `Requires-Dist` and extras. Verify security
  bounds in wheel metadata, not only `pyproject.toml` or the source environment.
- Resolve the `all` contract explicitly: the proposed runtime-feature union includes
  `claude-agent-sdk`; author/dev tooling remains separately named and documented.
  Requirements and installation examples must agree.
- Base imports and keyword retrieval must not import or initialize optional SDKs,
  torch, or a transformer model. Security bounds belong to the affected extras,
  not unconditional heavyweight base dependencies.

## M2 — Grounding integrity

### R3 — Apply the selected grounding policy on every generation path

Native citations must preserve the semantics of `baseline`, `strict`, `citation`,
and `anti_prior`. Source documents remain data. Native citation transport can
replace legacy `[Pn]` syntax; it cannot discard only-source or refusal rules.

Acceptance:

- Recording-provider tests assert the effective instructions and evidence for all
  four variants on legacy, native, and unsupported-native capability fallback.
- Validate empty/whitespace queries and unknown variants consistently before any
  generation call, including the zero-hit and native paths. Report invalid input
  as input failure, not an abstention or provider outage.
- Include benign adversarial fixtures containing instruction-like documentation
  and literal passage delimiters. Check instruction/data separation in payloads;
  do not present a payload assertion as proof that live prompt injection is solved.
- Preserve explicit model and token overrides, native capability detection, and
  existing model-asserted claim citation parsing. Citations are attribution, not
  an independent entailment verdict.

### R4 — Enforce strict empty-retrieval abstention locally

For a valid strict query with zero hits, return exactly:
`The provided context does not cover this question.`
Answer generation makes zero provider calls, regardless of native citation
preference or provider capability. This does not disable explicitly configured
optional retrieval stages; their recovery/cost policy remains separately scoped. Other variants keep their existing generative fallback.

Acceptance:

- Test strict zero-hit generation with native on/off, a provider without native
  support, and the named-provider route with no SDK/key available. Local abstention
  must not require provider initialization or authentication.
- Return empty supplied evidence and claim citations, zero confidence, and
  `used_native_citations=False`. Preserve `fallback_used=True` for the empty-hit
  compatibility signal; add a defaulted `abstained` flag to distinguish locally
  enforced abstention from generated fallback. The flag does not classify a
  model-generated refusal after nonempty retrieval.
- `run(..., prompt_variant="strict")` is prompt assembly: with zero hits it emits
  the strict refusal instruction and the same metadata. It does not fabricate a
  model response. `run_and_generate` enforces the local response above.
- With nonempty evidence, strict still reaches the provider with strict rules.
  With zero hits, each other variant still calls the plain generator once and
  discloses fallback; requesting native citations does not invent citations.

### R5 — Record the evidence actually supplied to generation

`RagResult.context` must cover the supplied source content, and native source
identity/order must be reconstructable without guessing from a truncated preview.
The generator and judge consume the same evidence selection.

Acceptance:

- Record immutable ordered native documents using their exact titles and text.
  Derive evaluation context from those documents without a second hidden cap.
  Legacy context remains the exact rendered context included in its prompt.
- Use fixtures below, at, and above the 20,000-character legacy boundary, with
  multiple documents, whitespace, Unicode, duplicate titles, and a fact present
  only in the final document/tail. Compare provider arguments and returned evidence.
- Native document indices remain tied to the supplied document order. Keep
  retrieval provenance distinct from generation evidence and model assertions.
- Disclose different legacy/native evidence budgets. This repair retains full
  native documents; a shared truncation budget would be a separate behavior change.
- Preserve compatible existing `RagResult` construction using defaulted additive
  fields; review serialization and repr/log exposure of newly available text.

### R6 — Make answer-level grounding inspectable

Preserve the existing per-query benchmark evidence as a downloadable CI artifact.
Keep existing gate outcomes and thresholds. A high average does not imply that
every answer is supported, and a citation marker is not proof of support.

Acceptance:

- For each completed query retain ID, answer, actual supplied evidence and source
  mapping, supported/unsupported claim lists, judge reasoning, prompt variant,
  provider/model identifiers, citation mode, and pipeline-abstention/fallback state.
  Bind the receipt to code SHA, query SHA, corpus identity/version and available
  content fingerprint, resolved dependencies, run/attempt, and report schema version.
  Label unavailable identifiers explicitly; do not fabricate provenance.
- Report mean per-answer faithfulness, answers with any unsupported claim, claim
  counts, refusals, and pipeline abstentions with explicit denominators. Keep
  factual refusals separate from non-answers that improve a score by saying nothing.
- Preserve both attempts when the existing transient-failure retry runs. Upload
  completed/partial evidence even when validation fails; an incomplete report
  remains visibly incomplete and cannot satisfy a required gate.
- Stub the real workflow shell/control path to prove upload/report handling on
  pass, measured regression, malformed judge output, transient unavailability,
  authentication failure, and missing report. Report handling must not change
  those outcomes or add retries seeking a passing measurement.
- Automatic CI uploads use only approved public/synthetic benchmark corpora.
  Arbitrary user corpus/query text is not uploaded or logged automatically.
  Set an explicit artifact retention period and disclose that reports contain
  answers and source passages. No credentials or raw request headers belong there.
- Check report math offline against fixed complete, unsupported, zero-claim,
  partial, and malformed fixtures. Retain 1.2.1's malformed-judge rejection.
  Numeric answer-level targets remain advisory until representative evidence and
  a separately approved decision establish them.

## R7 — Preserve existing quality, packaging, and failure contracts

Keep the 40-query SHA lock and current threshold artifact unchanged. Preserve
all 40 correct top-1/top-3 results on the bundled set and Phase 1's source/wheel
parity, required local failure handling, and narrow provider-unavailability rule.
A dependency update is not permission to reset the baseline or weaken a test.

Acceptance: run the affected offline tests, API compatibility checks, bundled
golden/paraphrase guards, source/sdist/wheel check, and relevant workflow negative
controls. The legacy mean-faithfulness CI threshold stays 0.9698; POLICY's bundled
retrieval floors remain P@1/R@3 1.0. Live evaluation, if needed for implementation,
requires a fresh run plan and spending approval. This draft performs none.
