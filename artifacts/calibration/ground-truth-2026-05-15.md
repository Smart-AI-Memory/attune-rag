# Faithfulness ground-truth labels

Source artifact: `artifacts/calibration/thinking-2026-05-15.json`
Labeled by: Patrick Roebuck on 2026-05-15
8 queries (5 highest-shift + 3 controls).


## gq-017 — `create documentation for my code`

```yaml
id: gq-017
verdict: faithful
faithfulness_score: 0.95
notes: |
  Answer paraphrases P1 (/doc-gen invocation) and P3 (code-quality
  tips) closely. Judge-off flagged the "Tip: especially useful after
  a refactor / when inheriting unfamiliar code" framing as editorial,
  but P3 literally says "After a large refactor — verify nothing
  degraded. When inheriting unfamiliar code — get a quick read on
  its health" — direct paraphrase, not hallucination. Thinking-on
  was right to count those as supported.
claims: []  # not enumerating; verdict drives scoring
```

## gq-015 — `sniff out hard-to-catch bugs`

```yaml
id: gq-015
verdict: faithful
faithfulness_score: 1.0
notes: |
  Judge-off (1.000) was correct. Judge-on (0.900) flagged two
  claims as unsupported, but neither phrasing literally appears
  in the answer text — judge-on appears to have paraphrased
  claims out of the answer (a "slips through code review"
  framing and a closing synthesis claim) and then flagged its
  own paraphrases. Evidence AGAINST thinking-on being more
  reliable: it can over-decompose answers and create phantom
  claims.
claims: []
```

## gq-038 — `coordinate documentation updates across the project`

```yaml
id: gq-038
verdict: partial
faithfulness_score: 0.85
notes: |
  Strict-lens scoring. Three editorial framings count as
  unsupported under strict:
  (1) "All of this is coordinated in a single pipeline" —
      extrapolation from "end-to-end documentation maintenance"
      and "Group: workflow" in P1; close but not literal.
  (2) "Doc Orchestrator is part of a broader suite of workflow
      tools, alongside Security Audit / Bug Predict / Code
      Review" — strongest leap. P1 lists those only as Related
      Topics, not as a named suite.
  (3) "documentation standards" as a memory candidate — P2
      lists "project conventions and naming rules" but not
      documentation standards specifically; small bridging
      fabrication.
  Judge-on (0.824) is closer to truth than judge-off (0.909)
  here. Judge-on did add one phantom paraphrase ("any project
  root") but its other two flags were legitimate strict-lens
  concerns.
claims: []
```

## gq-030 — `reduce code complexity`

```yaml
id: gq-030
verdict: partial
faithfulness_score: 0.85
notes: |
  Strict-lens scoring. Two editorial leaps:
  (1) "Bug prediction can help you prioritize which complex
      code to address first based on bug likelihood" —
      inference from P2 that goes beyond what P2 says.
  (2) Closing recommendation "Use simplify_code() for quick
      hotspot detection, then follow up with a Refactor Plan
      for a structured cleanup roadmap" — synthesizes a
      sequencing workflow neither P3 nor P1 suggests.
  Judge-off (0.917) is closer to truth than judge-on (1.000)
  here. Judge-off correctly flagged the bug-predict editorial
  inference; judge-on missed both editorial leaps.
claims: []
```

## gq-008 — `prepare a release`

```yaml
id: gq-008
verdict: faithful
faithfulness_score: 1.0
notes: |
  Strict-lens scoring. Answer is a clean grounded reproduction
  of P1/P2/P3 — tables verbatim, every claim traces. Judge-on
  (0.920) flagged two claims, but both are phantoms — judge-on
  paraphrased the answer's verbatim P3 table entries into
  specificities the answer doesn't claim ("specifically adds a
  missing changelog entry", "once you have a GO verdict").
  Judge-off (1.000) was correct. Same phantom-claim pattern as
  gq-015.
claims: []
```

## gq-003 — `fix failing tests`

```yaml
id: gq-003
verdict: faithful
faithfulness_score: 1.0
notes: |
  Strict-lens scoring. Control query (both judges 1.000).
  Answer is a clean re-presentation of P1/P2/P3 — examples,
  4-step process, root causes table, retry behavior all
  traceable verbatim. Useful control: confirms judges aren't
  systematically off on easy cases.
claims: []
```

## gq-007 — `refactor my code`

```yaml
id: gq-007
verdict: faithful
faithfulness_score: 1.0
notes: |
  Strict-lens scoring. Control query (both judges 1.000).
  Clean re-presentation of P1/P2/P3 — invocation, categories
  table, focus examples, post-roadmap actions all verbatim.
  Mild truncation (drops "shotgun surgery", "vestigial
  modules") but no fabrications. Second control confirms
  judges aren't off on easy cases.
claims: []
```

## gq-009 — `audit documentation for staleness`

```yaml
id: gq-009
verdict: faithful
faithfulness_score: 1.0
notes: |
  Strict-lens scoring. Control query (both judges 1.000).
  Clean grounded reproduction of P1 (with brief refs to P2/P3
  vibes). Two borderline calls considered:
  (1) "Once you've identified stale documentation, you may
      also want to explore related workflow tools" — softer
      than gq-038's "broader suite" framing; rhetorical glue,
      not a structural claim. Passes.
  (2) Truncated tool descriptions in P1 ("Detects
      vulnerabili...") completed by the answer to standard
      tool descriptions. Obvious-truncation completion, not
      fabrication. Passes.
  Third control point — confirms judges aren't off on
  straightforward queries.
claims: []
```
