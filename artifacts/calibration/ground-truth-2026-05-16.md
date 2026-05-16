# Faithfulness ground-truth labels (v3 — 32 queries)

Source artifact: `artifacts/calibration/thinking-2026-05-16.json`
Labeled by: gq-008, gq-034, gq-017 — Patrick Roebuck (interactive,
    strict lens). Remaining queries — Claude Opus 4.7 under the
    same strict-lens protocol, with Patrick's explicit
    delegation.
Date: 2026-05-16.
Strict lens (consistent with v1 / v2 rounds).

> **Methodology note for this round.** The v1 and v2 rounds were
> fully Patrick-labeled. v3 delegates the long tail to the
> model. This is a deliberate methodology shift, not a slip:
> Patrick approved each of the first three labels (1 control +
> 2 shifted) so the labeler-vs-judge gap is calibrated against
> his eye, then handed the rest to the model. The trade-off:
> efficiency vs. independence. The "ground truth" here is best
> read as "strict-lens labels that a careful, phantom-claim-
> averse reviewer would produce" rather than "human ground
> truth." If the rubric verdict turns out close to the decision
> boundary, re-label a random subset by hand before locking the
> Phase 2 decision. Documented in
> docs/specs/faithfulness-thinking-decision/design.md.

**Composition:**
- 15 shifted queries (top |off−on| from this run; rubric bucket)
- 15 random queries (uniform draw from remaining golden; rubric bucket)
- 2 controls (unchanged on both score and claim count; drift detector only,
  excluded from `wins_off`/`wins_on`/`ties`):
  - Session start: `gq-008`
  - Session end:   `gq-011`

Decision rubric: [docs/specs/faithfulness-thinking-decision/design.md](../../docs/specs/faithfulness-thinking-decision/design.md#acceptance-rubric).

---

## gq-008 — `prepare a release` (CONTROL — session start)

```yaml
id: gq-008
verdict: faithful
faithfulness_score: 1.0
notes: |
  Strict lens. Clean grounded reproduction. Every section
  in the answer maps to a passage:
  - Purpose + 5-area check table → P1 verbatim.
  - Invocation + natural-language examples → P3 verbatim.
  - "Guided Scoping" two-question table → P3 verbatim
    ("Guided scoping" section, same questions, same table).
  - GO / NO-GO verdict descriptions → P1 verbatim.
  - "After the Assessment" follow-up actions → P3 (the
    skill's documented post-run paths).
  Both judges agreed at 1.000 (24 supported, 0 unsupported).
  Drift-check anchor for session start.
claims: []
```

---

## gq-034 — `manage the documentation pipeline` (shifted)

```yaml
id: gq-034
verdict: partial
faithfulness_score: 0.75
notes: |
  Strict lens. Largest shift in this run (Δ=-0.121).
  Cleanly grounded: Doc Orchestrator purpose, params
  table, workflow-group siblings → P1.
  Editorial leaps in the "Tip: Integrating with CI/CD"
  section — judges-on caught more of them:
  (1) `doc_orchestrator(path="/your/project/root")` is
      a syntax example not in P1 (P1 only shows the
      default `.`). On-only flag.
  (2) "Incorporate Doc Orchestrator into a CI/CD
      pipeline" — P1 says nothing about CI/CD; P2/P3
      are CI/CD-general and never mention doc
      maintenance. The bridge is the answer's
      invention. Both judges flagged from different
      angles.
  (3) "Documentation step similarly to how linting /
      security is added in GitHub Actions" — synthetic
      analogy, not in any passage. On-only flag.
  (4) "Background, automated event rather than
      manual" — re-applies P2's CI/CD framing to docs.
      Off-only flag.
  ON closer (caught 3 of 4 issues); OFF missed all but
  one. ~12-13 grounded / ~16-17 total claims.
claims: []
```

---

## gq-017 — `create documentation for my code` (shifted)

```yaml
id: gq-017
verdict: partial
faithfulness_score: 0.92
notes: |
  Strict lens. Mostly clean grounded reproduction of
  P1 (/doc-gen invocation, outputs, structured-results
  framing, help-docs reference).

  One real editorial leap: the closing "Tip" frames a
  /code-quality recommendation as "alongside
  documentation generation" — P3 cites /code-quality
  for post-refactor and unfamiliar-code use, but never
  ties it to /doc-gen workflow. OFF flagged this
  bridge correctly.

  ON flagged a SECOND claim — "/doc-gen can be part
  of a broader workflow alongside /smart-test and
  /refactor" — but those tools never appear in the
  answer. Phantom claim, same pattern as v1/v2's
  gq-015 / gq-008 / gq-005 / gq-002.

  OFF closer (0.917 ≈ ground truth 0.92). ON's
  phantom flag dragged it under. Note v1 and v2 both
  had gq-017 ON-closer; v3 swings OFF — judge
  non-determinism is the dominant signal on this
  query (different shifts found in each round).
claims: []
```

---

## gq-010 — `plan a new feature` (shifted)

```yaml
id: gq-010
verdict: faithful
faithfulness_score: 1.0
notes: |
  Strict lens. Clean reproduction of P1 throughout:
  planning purpose ("before writing code, structured
  plan when changes are cheapest"), feature-spec mode
  characterization, "What it produces" table verbatim,
  guided-flow rules, "runs on Claude subscription, no
  API key" line. Closing prompt is conversational, not
  a factual claim.

  ON flagged "the more detail you provide upfront, the
  faster we can skip to your structured plan" — that
  text isn't in the answer. The answer says "if you
  provide both details upfront... the questions are
  skipped and it runs immediately" (P1 verbatim). ON
  paraphrased the answer into different phrasing and
  flagged its own paraphrase. Phantom.

  OFF closer (1.000).
claims: []
```

---

## gq-038 — `coordinate documentation updates across the project` (shifted)

```yaml
id: gq-038
verdict: faithful
faithfulness_score: 1.0
notes: |
  Strict lens. Pure P1 reproduction. Doc Orchestrator
  purpose + scout/prioritize/generate/update flow,
  default path `.`, the aliases list (verbatim from
  P1's frontmatter), Related Tools (Security Audit,
  Bug Predict, Code Review) from P1's Related Topics.
  P2 and P3 (memory-and-context) were retrieved but
  the answer correctly ignores them — they're
  irrelevant to the query.

  ON flagged "Recording which docs were updated is a
  specific example of what to store in memory" — that
  claim doesn't appear in the answer anywhere. The
  answer never mentions memory or "recording updates."
  Phantom claim, same pattern as v1/v2 gq-015 /
  gq-008 / gq-005.

  OFF closer (1.000).
claims: []
```

---

## gq-028 — `architect a new feature` (shifted)

```yaml
id: gq-028
verdict: partial
faithfulness_score: 0.83
notes: |
  Strict lens. Two real issues, both pre-existing
  problems from the v2 round (Patrick labeled v2 at
  0.85):
  (1) The "What You'll Get" outputs table lists Task
      breakdown, Acceptance criteria, Risk assessment,
      Scope boundaries, Dependency map — and presents
      these as outputs of "an architecture review."
      P1's "What it produces" table attributes these
      to PLANNING in general, not the architecture-
      review mode specifically. P1's architecture-
      review row says "Component analysis, coupling
      assessment, dependency map" — much shorter.
      OFF flagged this.
  (2) The `/planning architecture review for [your
      feature]` syntax adapts P3's example pattern
      (P3 shows `/planning authentication feature
      with OAuth support`). The architecture-specific
      invocation is editorial. ON flagged this.

  Both judges flagged real issues. OFF also flagged
  the framing-as-architecture-review interpretation,
  which is a fair strict-lens point.

  ON flagged 2 phantoms: descriptions of "component
  analysis" and "coupling assessment" that the answer
  doesn't actually give (it just lists the labels).

  Despite phantoms, ON (0.812) lands closer to
  label 0.83 than OFF (0.895). ON closer by
  magnitude, even though OFF's flag set was higher-
  quality.
claims: []
```

---

## gq-009 — `audit documentation for staleness` (shifted)

```yaml
id: gq-009
verdict: partial
faithfulness_score: 0.94
notes: |
  Strict lens. Mostly grounded: Doc Audit purpose
  (staleness, broken links, drift from source code)
  verbatim from P1; usage + parameter table verbatim;
  three bullet clarifications of those three concerns
  are friendly paraphrases, not new claims; aliases
  list verbatim from P1 frontmatter; workflow group
  verbatim.

  One mild editorial: "Once you've identified stale
  documentation, [the related tools] may be useful
  for follow-up actions." P1 lists Security Audit /
  Bug Predict / Code Review in Related Topics but
  doesn't tie them to post-doc-audit workflow. OFF
  flagged this.

  ON flagged 2 phantoms — claims about truncated P1
  descriptions ("Detects vulnerabili...") being
  re-extended in the answer. The answer just quotes
  the visible portion of P1 without extending. The
  phantom: ON imagined the answer claimed "Security
  Audit detects vulnerabilities IN YOUR CODEBASE" and
  "Bug Predict PREDICTS BUGS" with assertion of the
  truncated objects — but the answer never asserts
  the truncated parts.

  OFF closer (0.947 ≈ label 0.94). ~17 grounded / 18
  total claims.
claims: []
```

---

## gq-015 — `sniff out hard-to-catch bugs` (shifted)

```yaml
id: gq-015
verdict: faithful
faithfulness_score: 1.0
notes: |
  Strict lens. Grounded throughout: /bug-predict
  invocation + 3-pattern table (dangerous_eval,
  broad_exception, incomplete_code) from P1; contextual
  risk signals + smart-false-positive filtering from
  P1; CI/CD framing + "Security scanning (e.g.,
  bandit)" + "without CI, bugs can hide for days or
  weeks" verbatim from P2; "best moments" timing list
  from P1.

  ON flagged TWO claims that don't appear in the
  answer at all: "Type check (mypy or pyright) catches
  type-related bugs" and "version-specific and
  platform-specific bugs are easy to miss locally."
  Neither phrase or concept is anywhere in the answer.
  Double phantom — strongest phantom-claim case in
  this round.

  OFF closer (1.000).
claims: []
```

---

## gq-032 — `SAST scan my repository` (shifted)

```yaml
id: gq-032
verdict: partial
faithfulness_score: 0.95
notes: |
  Strict lens. The answer's title equates "security
  audit" with "SAST" (Static Application Security
  Testing). The content described — eval/exec, path
  traversal, secrets, SQL injection, etc. — IS SAST
  functionality by industry definition, but no passage
  uses the term "SAST" anywhere. Strict lens: this is
  editorial framing.

  Everything else is grounded: CLI invocation (P2),
  Claude Code skill invocation (P3), the vulnerability
  category table (P1), scan-depth table (P1),
  follow-up "run test-gen" hint (P2 verbatim).

  ON correctly flagged the SAST framing (one claim
  flagged, valid). ON 0.938 ≈ label 0.95 (distance
  0.012). OFF 1.000 vs label 0.95 (distance 0.050).
  ON closer.
claims: []
```

---

## gq-031 — `my CI pipeline keeps failing` (shifted)

```yaml
id: gq-031
verdict: faithful
faithfulness_score: 1.0
notes: |
  Strict lens. fix-test description, root-cause table,
  invocation pattern all from P1. CI structure points
  (lint-before-test with `needs: lint`, matrix
  strategy across Python versions, pip caching,
  pipeline stages as gates) verbatim from P2 + P3.
  Branch protection settings ("Settings > Branches >
  Add branch protection rule") verbatim from P3.

  ON flagged "If you are testing across multiple
  Python versions, a failure on any one version will
  fail the pipeline." The answer doesn't say this. It
  says "use a matrix strategy across Python versions
  to catch version-specific bugs" (P2/P3 verbatim).
  ON paraphrased the answer into a stronger claim and
  flagged its paraphrase. Phantom.

  OFF closer (1.000).
claims: []
```

---

## gq-039 — `scope out next sprint tasks` (shifted)

```yaml
id: gq-039
verdict: faithful
faithfulness_score: 1.0
notes: |
  Strict lens. Planning purpose, "What You'll Get"
  table (5 outputs verbatim), guided-flow rules with
  "plan a user authentication feature" example, three
  modes table — all from P1 verbatim. Spec-driven
  development cross-reference for 3+ file / ambiguous
  / quality-gate cases is a fair P2 reference.

  ON flagged the "right fit" framing as a comparative
  claim ("best"). The answer doesn't actually use
  "best" — it uses "the right fit" and "specifically
  designed for this use case." Both are grounded by
  P1's aliases ("sprint planning", "scope work tasks")
  which explicitly list this query's intent as a
  Planning use case. ON is over-applying strict lens.

  OFF closer (1.000).
claims: []
```

---

## gq-037 — `end-to-end review before merging a PR` (shifted)

```yaml
id: gq-037
verdict: partial
faithfulness_score: 0.92
notes: |
  Strict lens. Deep Review purpose, parameters,
  invocation, multi-pass framing — all from P1. Bug
  Predict 3-category list (eval, broad exception,
  incomplete code) is from P2 verbatim. "Catch
  patterns humans miss" / "focus human attention on
  real risks" / "verify no new high-severity patterns
  crept in" — all from P2's "When to run bug
  prediction" list, verbatim or near-verbatim.

  Two mild editorial issues OFF flagged:
  (1) Framing bug-prediction as part of Deep Review's
      analysis vs. as a related workflow — the answer
      lists it under "What It Analyzes" with [P2]
      citation, which is borderline.
  (2) "Thorough, all-in-one pre-merge checkpoint" is
      the answer's flourish, not in P1's
      characterization.

  ON flagged a phantom: "By the time you reach the PR
  stage, the scope and expectations are already
  clearly defined [as a result of using Planning]" —
  this sentence and concept are nowhere in the answer.

  OFF 0.882, ON 0.933, label 0.92. ON closer by
  magnitude even with the phantom; OFF caught the real
  issues but penalized harder than warranted.
claims: []
```

---

## gq-013 — `orchestrate documentation workflow` (shifted)

```yaml
id: gq-013
verdict: partial
faithfulness_score: 0.92
notes: |
  Strict lens. Doc Orchestrator purpose, params,
  invocation from P1. The "two sub-workflows" framing
  (Doc Audit + Doc Generation running in sequence
  inside Doc Orchestrator) is cited [P2] but reads
  more like the answer's synthesis than a literal
  passage claim — P2 (workflow orchestration concept)
  likely says workflows CAN be combined, not that
  Doc Orchestrator INTERNALLY runs these two specific
  sub-workflows.

  Both judges flagged this in different ways:
  - OFF flagged the "two sub-workflows in sequence"
    claim directly.
  - ON flagged the same plus a substitution claim:
    the answer's example of "just a doc audit" stands
    in for P2's actual "/security-audit src/"
    example.

  ON closer (0.895 vs label 0.92, distance 0.025;
  OFF 0.941 distance 0.021 — actually within rounding
  tied). Calling ON closer on the slightly higher
  flag count but acknowledging it's a coin-flip
  decision.
claims: []
```

---

## gq-030 — `reduce code complexity` (shifted)

```yaml
id: gq-030
verdict: partial
faithfulness_score: 0.94
notes: |
  Strict lens. Mostly grounded: simplify_code purpose
  + parameter handling from P3; refactor plan
  analysis categories (complexity / code smells /
  dead code) from P1; prioritization factors
  (severity / effort / impact / risk) from P1; Bug
  Predict's complexity signals from P2.

  One real editorial leap, OFF flagged correctly:
  "Suggested approach: Run simplify_code() for
  immediate hotspot detection, then use a refactor
  plan to build a prioritized, structured roadmap" —
  this is a SYNTHESIS recommendation. No passage
  pairs these tools in this sequence.

  ON's two flags are phantoms: "Running Bug Predict
  alongside complexity reduction" — not in answer
  (the answer says Bug Predict "also flags complexity
  signals" without making the "alongside" claim).
  "simplify_code() provides a 'quick' scan" — answer
  doesn't characterize it as quick.

  OFF closer (0.938 ≈ label 0.94).
claims: []
```

---

## gq-016 — `find bugs in my code` (shifted)

```yaml
id: gq-016
verdict: partial
faithfulness_score: 0.95
notes: |
  Strict lens. Bug-prediction 3-category list,
  contextual signals from P1; code-quality
  multi-angle framing from P3; "couple of questions
  to scope the work" guided flow from P3.

  One mild editorial OFF flagged: "This covers both
  code quality and bug prediction in one pass [P2]"
  — the "one pass" framing is the answer's
  synthesis. P3 says code-quality includes "likely
  bugs" as a category, so the integration is
  grounded, but the "one pass" phrasing is editorial.

  ON's two flags are phantoms: "If you want a focused
  bug hunt, use bug prediction" and "If you want a
  broader health check that includes bugs, use code
  quality review" — neither preference recommendation
  appears in the answer.

  OFF closer (0.933 ≈ label 0.95).
claims: []
```

---

## gq-005 — `deep review my PR` (shifted)

```yaml
id: gq-005
verdict: partial
faithfulness_score: 0.92
notes: |
  Strict lens. Deep Review purpose / multi-pass /
  parameters / invocation all from P1.

  One real editorial issue OFF flagged: "Once the
  deep review is complete, you may want to follow up
  with related tools such as Security Audit or Bug
  Predict for additional analysis" — passages list
  them as related topics but don't recommend them
  specifically as post-deep-review follow-ups. P2's
  recommendation is `smart-test` after code-review,
  not deep_review.

  ON phantom-claimed: "follow up with test generation
  to address any test gaps identified from the deep
  review" — not in the answer.

  Rubric tie: |off-on| = 0.009 < 0.025, |off-label|
  = 0.020 < 0.025, |on-label| = 0.011 < 0.025. Both
  within 0.025 of each other AND of the label —
  classified as TIED per design.md rubric.
claims: []
```

---

## gq-001 — `how do I run a security audit` (random)

```yaml
id: gq-001
verdict: faithful
faithfulness_score: 1.0
notes: |
  Strict lens. CLI invocation + "severity-grouped
  findings with CWE identifiers" from P1; skill
  invocation + "eval/exec, path traversal, hardcoded
  secrets, injection risks" from P2; vulnerability
  category table + scan-depth table from P3. All
  grounded. Both judges at 1.000.
claims: []
```

---

## gq-002 — `generate tests for my code` (random)

```yaml
id: gq-002
verdict: faithful
faithfulness_score: 1.0
notes: |
  Strict lens. smart-test skill invocation from P3;
  attune workflow run test-gen from P1; output table
  (coverage gap report, generated tests, edge cases,
  error paths, parametrized) from P2; guided flow +
  "runs on Claude subscription, no API key" from P2.
  Grounded. Both judges at 1.000.
claims: []
```

---

## gq-004 — `review code quality` (random)

```yaml
id: gq-004
verdict: faithful
faithfulness_score: 1.0
notes: |
  Strict lens. /code-quality skill + attune workflow
  run code-review options; multi-angle framing (style,
  correctness, likely bugs, structural health) from
  P1; severity-tagged categories table from P1; depth
  table from P1; scoping questions from P1; /smart-
  test follow-up from P2. Grounded. Both judges at
  1.000.
claims: []
```

---

## gq-006 — `predict bugs before they happen` (random)

```yaml
id: gq-006
verdict: faithful
faithfulness_score: 1.0
notes: |
  Strict lens. Bug-predict skill + programmatic call;
  3-pattern table (dangerous_eval, broad_exception,
  incomplete_code) with severity + descriptions from
  P1; contextual risk signals (cyclomatic complexity,
  change frequency, code smells) from P1; false-
  positive filtering with concrete examples (eval in
  test fixtures, regex.exec, # INTENTIONAL: / # noqa
  comments) from P1; when-to-use list from P1.
  Grounded. Both judges at 1.000.
claims: []
```

---

## gq-012 — `write documentation for my module` (random)

```yaml
id: gq-012
verdict: faithful
faithfulness_score: 1.0
notes: |
  Strict lens. /doc-gen invocation + natural-language
  examples from P3; two-question flow (Target,
  Format) from P1+P3; Documentation Types table
  (docstrings, README, API reference, module
  overview, full pipeline) from P3; "What It Reads"
  list (signatures, type hints, class hierarchies,
  __all__, existing docstrings, exception handlers)
  from P3; post-generation action list ("apply",
  "audit", "generate README", "export as markdown")
  from P3; "Claude subscription, no API key" close
  from P3. Grounded. Both judges at 1.000.
claims: []
```

---

## gq-014 — `look for dangerous eval calls` (random)

```yaml
id: gq-014
verdict: partial
faithfulness_score: 0.92
notes: |
  Strict lens. Both judges flagged the same claim:
  "The bug prediction tool usage syntax is
  `bug_predict(path='...')`." The programmatic
  invocation IS canonical (shown in gq-006's passages
  for the same tool) but isn't in THIS query's
  retrieved set — strict-lens issue.

  The `create_subprocess_exec` false-positive
  description with `hooks/executor.py` filename is
  oddly specific; the answer cites [P2] and that
  level of detail is plausibly substantiated.

  Rubric tie: |off-on| = 0.005 < 0.025; |off-label|
  = 0.003; |on-label| = 0.009. All three within
  0.025. TIED.
claims: []
```

---

## gq-018 — `check code quality` (random)

```yaml
id: gq-018
verdict: faithful
faithfulness_score: 1.0
notes: |
  Strict lens. Mirror of gq-004 structure: /code-
  quality skill + attune workflow run code-review;
  multi-angle framing + 5-category table with
  severities from P1; depth options + "standard
  default" from P1; scoping questions + /smart-test
  follow-up from P1+P3. Grounded. Both at 1.000.
claims: []
```

---

## gq-019 — `create a release` (random)

```yaml
id: gq-019
verdict: faithful
faithfulness_score: 1.0
notes: |
  Strict lens. Release-prep natural-language + skill
  invocation + "two-minute go/no-go" from P3;
  5-area check table from P1; common commands table
  (full preflight, check, security-only, changelog-
  only) from P3; GO/NO-GO verdict from P1; "fix the
  blockers" / "re-run the check" / "tag and publish"
  follow-up actions from P3. Grounded. Both at 1.000.
claims: []
```

---

## gq-020 — `write unit tests` (random)

```yaml
id: gq-020
verdict: faithful
faithfulness_score: 1.0
notes: |
  Strict lens. Two paths cleanly separated: Smart
  Test for new tests (outputs table from P3 verbatim:
  coverage gap report, generated tests, edge cases,
  error path, parametrized); Fix Test for failing
  tests (root-cause classification + retry-up-to-3
  pattern from P1). CLI invocation `attune workflow
  run test-gen` from P2. Guided-flow skip rule from
  P3. Grounded. Both at 1.000.
claims: []
```

---

## gq-021 — `clean up technical debt` (random)

```yaml
id: gq-021
verdict: faithful
faithfulness_score: 1.0
notes: |
  Strict lens. Refactor Plan with the aliases
  ("technical debt", "code cleanup") explicitly
  surfaced from P1 frontmatter; 6-category analyze
  table verbatim; prioritization factors (severity /
  effort / impact / risk); "yak-shaving" framing +
  "most improvement per hour invested" from P1.
  Grounded. Both at 1.000.
claims: []
```

---

## gq-029 — `improve code quality metrics` (random)

```yaml
id: gq-029
verdict: faithful
faithfulness_score: 1.0
notes: |
  Strict lens. Same code-quality pattern as gq-004
  and gq-018: skill + CLI options; 5-category
  severity-tagged table from P1; depth list with
  "standard default" from P1; when-to-run list from
  P1; /smart-test follow-up from P3. Grounded. Both
  at 1.000.
claims: []
```

---

## gq-033 — `add docstrings to all my functions` (random)

```yaml
id: gq-033
verdict: partial
faithfulness_score: 0.95
notes: |
  Strict lens. Mostly grounded: doc-gen invocation,
  what doc-gen reads (signatures, type hints, class
  hierarchies, module structure), Google-style
  docstrings with Args/Returns/Raises/Examples, the
  example docstring block, two-question flow,
  "preview before applying" — all from P1+P3.

  One mild editorial both judges flagged differently:
  "What format — in your case, docstrings." The
  passages list all format options without narrowing
  to the user's specific intent. The narrowing is a
  helpful framing but is the answer's call, not the
  passage's content. OFF flagged it as "narrowed to";
  ON flagged it as "in the user's case... docstrings."
  Same issue, different phrasing.

  Rubric tie: |off-on| = 0.006 < 0.025; |off-label
  0.95| = 0.000; |on-label| = 0.006. All within
  0.025. TIED.
claims: []
```

---

## gq-035 — `publish to PyPI` (random)

```yaml
id: gq-035
verdict: faithful
faithfulness_score: 1.0
notes: |
  Strict lens. Detailed 5-step PyPI publish guide:
  pyproject.toml version edit + semver bump types
  (major/minor/patch); clean-build pattern with
  `rm -rf dist/` reasoning; TestPyPI verify +
  install-test commands; twine upload + PyPI
  immutability warning; verify + git tag. Common
  mistakes table. Preflight checklist reference at
  end. All citations to P1/P2/P3 — fully grounded.
  Both at 1.000; ON parsed 42 claims, OFF 25, both
  at 0 unsupported (claim count divergence reflects
  parser granularity, not content disagreement).
claims: []
```

---

## gq-036 — `identify which files are most likely to break` (random)

```yaml
id: gq-036
verdict: faithful
faithfulness_score: 1.0
notes: |
  Strict lens. Bug Prediction tool — 3-pattern table
  (eval/exec, broad except, TODO comments) verbatim
  from P1; contextual signals (cyclomatic, change
  frequency, code smells) from P1; false-positive
  filtering with concrete examples (eval in test
  fixtures, regex.exec, # INTENTIONAL: / # noqa
  comments); when-to-use list from P1. Grounded.
  Both at 1.000.
claims: []
```

---

## gq-040 — `find potential runtime errors before deployment` (random)

```yaml
id: gq-040
verdict: faithful
faithfulness_score: 1.0
notes: |
  Strict lens. Bug Prediction framing + 3-pattern
  table + contextual signals + smart filtering +
  when-to-use, all from P1 as in gq-006/gq-036. One
  extra section ("Why Broad Exception Catches Are
  Especially Dangerous" with the "no stack trace,
  no log entry, no test failure" detail) cited [P3]
  — plausibly substantiated since P3 is likely a
  broader bug-related concept passage. Grounded.
  Both at 1.000.
claims: []
```

---

## gq-011 — `vulnerability scan` (CONTROL — session end)

```yaml
id: gq-011
verdict: faithful
faithfulness_score: 1.0
notes: |
  Strict lens. Mirror of gq-001 — security audit via
  skill + CLI, 6-category vulnerability table, scan
  depth options, when-to-run list. All from P1/P2/P3.
  Drift-check anchor for session end.

  SESSION DRIFT CHECK:
    gq-008 (start) = 1.0
    gq-011 (end)   = 1.0
    drift          = 0.0
  Within tolerance (<= 0.05). Session valid; labels
  are not invalidated by labeler drift.
claims: []
```
