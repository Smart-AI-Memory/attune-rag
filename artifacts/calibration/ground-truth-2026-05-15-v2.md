# Faithfulness ground-truth labels (v2 — 17 queries)

Source artifact: `artifacts/calibration/thinking-2026-05-15-v2.json`
Labeled by: Patrick Roebuck on 2026-05-15 / 2026-05-16
Strict lens (consistent with v1 round).


## gq-017 — `create documentation for my code`

```yaml
id: gq-017
verdict: partial
faithfulness_score: 0.85
notes: |
  Strict lens. Two distinct strict-lens issues in the answer:
  (1) Placeholder substitution explanation ("replace
      <placeholder> with what you want") — mild editorial
      inference from shell convention, not stated in P1.
  (2) Tip section claims code-quality review is useful BEFORE
      documenting code — editorial bridge; P3 never ties
      code-quality to documentation as a sequenced workflow.
  Judge-on (0.750) caught both; judge-off (1.000) missed both
  but judge-on inflated its count with a duplicate paraphrase
  of issue #2. On-closer at this label.

  NOTE: Same query went OPPOSITE direction vs v1 run
  (v1: off=0.82, on=1.00; v2: off=1.00, on=0.75). Excellent
  illustration of judge non-determinism — the calibration
  captures a snapshot, not ground truth.
claims: []
```

## gq-028 — `architect a new feature`

```yaml
id: gq-028
verdict: partial
faithfulness_score: 0.85
notes: |
  Strict lens. Multiple compounding editorial issues:
  (1) Editorial descriptions added for "Component analysis"
      and "Coupling assessment" (P1 lists those as bullet
      names with NO description).
  (2) Conflates Risk assessment and Scope boundaries as
      architecture-review-specific outputs — those are
      general-planning items per P1's "What it produces"
      table.
  (3) `/planning architecture review for [feature]` command
      syntax not substantiated by P3 (which only shows
      `/planning authentication feature with OAuth support`
      and natural-language variants).
  Judge-on (0.812) caught 3 issues; judge-off (0.947) caught
  only 1. On-closer.
claims: []
```

## gq-005 — `deep review my PR`

```yaml
id: gq-005
verdict: faithful
faithfulness_score: 1.0
notes: |
  Strict lens. Clean grounded reproduction of P1.
  Judge-on (0.900) flagged a `deep_review(path="src/")`
  example that doesn't exist in the answer — the answer
  uses the literal placeholder `path="..."`. Phantom flag
  (same pattern as gq-015 / gq-008 in v1 round).
  Judge-off (1.000) correct.
claims: []
```

## gq-002 — `generate tests for my code`

```yaml
id: gq-002
verdict: faithful
faithfulness_score: 1.0
notes: |
  Strict lens. Clean reproduction of P1/P2/P3 — both
  invocations (/smart-test + attune workflow), gap-types
  table, output description, guided-questions paragraph,
  and the post-gen pytest pointer all trace verbatim.
  Judge-on (0.929) flagged "There are exactly two ways" as
  an exhaustiveness claim — but the answer's "two ways,
  depending on your setup" is descriptive enumeration of
  what's in the passages, not an exhaustive cap. Phantom
  flag. Judge-off (1.000) correct.
claims: []
```

## gq-037 — `end-to-end review before merging a PR`

```yaml
id: gq-037
verdict: partial
faithfulness_score: 0.90
notes: |
  Strict lens. Two minor editorial issues:
  (1) "Security Audit — focused vulnerability scanning" —
      added "focused" framing; P1 just says "Detects
      vulnerabilities" (truncated). Judge-off caught this.
  (2) Closing claim that Bug Predict is "a strong complement
      to Deep Review for a thorough pre-merge workflow" —
      editorial synthesis; passages don't pair these tools as
      a sequenced workflow. Both judges missed this one.
  The "Bug Predict useful before merging a large PR" claim
  IS in P2 verbatim — that part is supported. Off-closer at
  0.90.
claims: []
```

## gq-025 — `find stale documentation`

```yaml
id: gq-025
verdict: partial
faithfulness_score: 0.95
notes: |
  Strict lens. One minor editorial issue: the Tip frames
  doc_orchestrator() as "more comprehensive" than doc_audit
  and suggests using it "instead" — a comparative claim
  P1/P3 don't make. Similar to gq-037 pattern.
  Judge-on (0.933) flagged "By default, it runs against the
  current directory" which is just the shell-convention
  meaning of `Default: .` — phantom flag. Judge-on closer
  numerically (Δon=0.017 vs Δoff=0.050) but for wrong
  reasons; phantom flag accidentally lowered score to near
  the right place.
claims: []
```

## gq-001 — `how do I run a security audit`

```yaml
id: gq-001
verdict: partial
faithfulness_score: 0.92
notes: |
  Strict lens. One editorial bridge: "Regardless of which
  method you use, a security audit scans for [6 categories]"
  attributes all of P3's 6 categories to both the CLI
  workflow (P1) and the Claude Code skill (P2). P1 doesn't
  list categories; P2 lists only 4 of the 6. The
  generalization is reasonable but not directly supported.
  Judge-off (0.938) caught this; judge-on (1.000) missed it.
  Off-closer.
claims: []
```

## gq-020 — `write unit tests`

```yaml
id: gq-020
verdict: partial
faithfulness_score: 0.95
notes: |
  Strict lens. One narrow editorial bridge: "You can also
  run [Smart-test] directly via the CLI: attune workflow
  run test-gen ..." links P2's test-gen CLI command to P3's
  Smart Test concept. Reasonable inference (identical
  purposes described) but no passage explicitly says
  test-gen IS Smart Test. Judge-on (0.938) caught this;
  judge-off (1.000) missed it. On-closer.
claims: []
```

## gq-032 — `SAST scan my repository`

```yaml
id: gq-032
verdict: partial
faithfulness_score: 0.95
notes: |
  Strict lens. One narrow editorial bridge: heading equates
  "security audit" with "SAST" (Static Application Security
  Testing). P1 describes what IS functionally SAST but never
  uses the term — answer bridges user's vocab (SAST) to P1's
  vocab (security audit). Factually correct but editorial
  under strict lens. Judge-on (0.938) caught it; judge-off
  (1.000) missed it. On-closer.
claims: []
```

## gq-010 — `plan a new feature`

```yaml
id: gq-010
verdict: faithful
faithfulness_score: 1.0
notes: |
  Strict lens. Answer recommends Feature spec mode for the
  user's "plan a new feature" query — direct mapping to P1's
  use-case column ("Starting a new feature or epic"), not an
  editorial leap. Judge-on (0.952) flagged the recommendation
  as "likely wants" editorial framing, but P1 explicitly ties
  Feature spec mode to new-feature queries. Phantom flag.
  Judge-off (1.000) correct.
claims: []
```

## gq-015 — `sniff out hard-to-catch bugs`

```yaml
id: gq-015
verdict: partial
faithfulness_score: 0.92
notes: |
  Strict lens. Two editorial issues:
  (1) "Type checking catches type-related bugs missed in
      review" — P2's pipeline table lists mypy/pyright but
      doesn't describe what type-checking catches. Judge-off
      caught this.
  (2) Closing "Putting It Together" recommends combining
      bug-predict + CI integration — editorial synthesis;
      neither passage makes that sequencing recommendation.
  Judge-off (0.958) caught issue 1; judge-on (1.000) missed
  both. Off-closer.

  NOTE: Same query labeled faithful 1.0 in v1 round. v2's
  answer is more thorough but introduces editorial issues
  the v1 answer didn't have. Answer-text variance, not just
  judge variance.
claims: []
```

## gq-034 — `manage the documentation pipeline`

```yaml
id: gq-034
verdict: partial
faithfulness_score: 0.85
notes: |
  Strict lens. Three editorial issues:
  (1) `doc_orchestrator(path="/your/project/root")` syntax —
      P1 only shows `doc_orchestrator()` without args.
  (2) CI/CD integration recommendation pairs Doc Orchestrator
      with P2/P3; no passage links them.
  (3) "Pipeline tools you can use alongside it" framing for
      Related Topics — same pattern as gq-038/gq-009.
  Both judges caught issues 1 and 2 (different phrasings,
  same issues). Both missed issue 3. Convergence case —
  judge-off 0.882, judge-on 0.867; both close to my 0.85.
  Judge-on slightly closer.
claims: []
```

## gq-014 — `look for dangerous eval calls`

```yaml
id: gq-014
verdict: partial
faithfulness_score: 0.90
notes: |
  Strict lens. Two editorial issues:
  (1) `bug_predict(path="...")` syntax — P1 (bug-predict
      concept) doesn't show a Python function call; the
      answer borrows P3's `security_audit(path="...")`
      pattern by analogy. Both judges caught this.
  (2) "Related security audit tool can also be used for
      broader vulnerability scanning" — "broader" framing
      isn't in P3 (no comparison). Both judges missed this.
  Judge-on (0.909) slightly closer than judge-off (0.923).
claims: []
```

## gq-011 — `vulnerability scan`

```yaml
id: gq-011
verdict: faithful
faithfulness_score: 1.0
notes: |
  Strict lens. Clean reproduction of P1/P2/P3 — both
  invocations, 6-category table (cited to P2 which IS the
  source), depth options, when-to-run. No editorial bridges.
  Notably more careful than gq-001 (similar query domain)
  — does NOT make the "regardless of method" generalization
  that flagged gq-001. Both judges 1.000 — genuine
  agreement. Tied.
claims: []
```

## gq-016 — `find bugs in my code`

```yaml
id: gq-016
verdict: faithful
faithfulness_score: 1.0
notes: |
  Strict lens. Clean reproduction of P1/P2/P3 — bug-predict
  description + 3-category table, contextual signals,
  code-quality categories, /code-quality command, default
  depth note. "Good fit for everyday bug-finding" is mild
  framing but reasonable inference from P3's depth table.
  Both judges 1.000 — tied.
claims: []
```

## gq-018 — `check code quality`

```yaml
id: gq-018
verdict: faithful
faithfulness_score: 1.0
notes: |
  Strict lens. Textbook clean — both invocations (P2 + P3),
  full 5-row category table from P1, depth levels with
  default note, guided-questions paragraph, post-review
  /smart-test follow-up all traceable verbatim. Both judges
  1.000 — tied.
claims: []
```

## gq-019 — `create a release`

```yaml
id: gq-019
verdict: faithful
faithfulness_score: 1.0
notes: |
  Strict lens. Clean reproduction of P1/P2/P3 — both
  invocations, full 5-row checklist table from P1,
  /release-prep version syntax, two-questions paragraph,
  GO/NO-GO descriptions, blockers framing, post-fix
  actions. Both judges 1.000 — tied.
claims: []
```
