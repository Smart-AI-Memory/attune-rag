# Faithfulness ground-truth labels

Source artifact: `artifacts/calibration/thinking-2026-05-15.json`
Selected 8 queries (5 shifted + 3 controls; shift threshold = 0.05).

## How to label

For each query below, fill in the YAML block under "Your labels". The scoring script compares your `faithfulness_score` against both judge passes (off / on) to decide whether thinking-on aligns better with ground truth than thinking-off. See `docs/rag/faithfulness-thinking-calibration.md` for the decision matrix this feeds.

---

## gq-017 — `create documentation for my code`

> ⚠️ The benchmark artifact does not currently capture the generator
> ANSWER text or the retrieved passages — only the judge's verdict
> on each. To label this query against ground truth you need to
> re-run the RAG pipeline locally and read the answer + context. A
> follow-up commit should add answer/context capture to the
> benchmark JSON output. For now, the judge's claim lists are
> provided below as a proxy.

### Judge verdicts (off → on)

| | Score | Supported | Unsupported |
| - | -: | -: | -: |
| Thinking off | 0.818 | 9 | 2 |
| Thinking on  | 1.000 | 10 | 0 |

**Reasoning (thinking off):**

> The vast majority of claims in the answer are directly and explicitly supported by passage P1, which covers the /doc-gen skill, its outputs (docstrings, READMEs, API references), the command syntax, the Claude Code conversation context, structured results, and the help-docs reference command. The tip at the end partially draws on P3 (code quality review after a refactor) and P2 (asking /code-quality to scan migrated code), but the specific framing — recommending /code-quality specifically after doc-gen use and attributing deprecated patterns to refactors/migrations — is an editorial inference that goes slightly beyond what the passages explicitly state.

**Reasoning (thinking on):**

> Every atomic factual claim in the answer is directly supported by the retrieved passages. The core /doc-gen claims (skill name, outputs: docstrings/READMEs/API references, command syntax, Claude Code conversation context, structured results, and the help-docs command) are all explicitly stated in P1. The closing tip about running a code quality review after a refactor or migration to verify nothing degraded is supported by P3 ("After a large refactor — verify nothing degraded"), and the reference to /code-quality catching style issues and deprecated patterns is explicitly stated in P2. No hallucinated details, invented flags, or claims sourced from outside knowledge were found.

**Claims identified by judge (thinking off):**

Supported:
  - The /doc-gen skill generates documentation from source code.
  - The /doc-gen skill can generate docstrings.
  - The /doc-gen skill can generate READMEs.
  - The /doc-gen skill can generate API references.
  - The command to use the skill is: /doc-gen <path or module to document>
  - The command is run in a Claude Code conversation.
  - After running the command, you get structured results directly in your Claude Code conversation.
  - The full reference on this skill can be accessed by running: attune help-docs ref-skill-doc-gen
  - You can run a code quality review with /code-quality to catch style issues or deprecated patterns.

Unsupported:
  - Running /code-quality is specifically recommended after a large refactor or migration (the passages mention this as one use case for code-quality, but the specific framing of 'if you've recently done a large refactor or migration' as a tip tied to doc-gen is an editorial inference not directly stated in any passage).
  - Deprecated patterns may be 'introduced during' a refactor or migration (this causal framing is not stated in the passages).

**Claims identified by judge (thinking on):**

Supported:
  - The /doc-gen skill can be used to create documentation from source code.
  - The /doc-gen skill generates docstrings.
  - The /doc-gen skill generates READMEs.
  - The /doc-gen skill generates API references.
  - The command to use is `/doc-gen <path or module to document>`.
  - The command is run in a Claude Code conversation.
  - After running the command, you get structured results directly in your Claude Code conversation.
  - The full reference guide can be accessed by running `attune help-docs ref-skill-doc-gen`.
  - After a large refactor, a code quality review can be run to verify nothing degraded.
  - The /code-quality skill can scan migrated code for style issues and deprecated patterns.

Unsupported:


### Your labels

```yaml
id: gq-017
# Overall verdict: faithful | partial | unfaithful | refusal_ok
verdict: TBD
# Your scalar score = supported / (supported + unsupported), 0.0 to 1.0.
# For "refusal_ok" answers (no factual claims), score 1.0.
faithfulness_score: TBD
# Optional: list claims as you see them. Counts only the bottom number.
claims:
  - text: "..."
    supported: true
  - text: "..."
    supported: false
notes: |
  Optional commentary on what the judge missed or got wrong.
```

---

## gq-015 — `sniff out hard-to-catch bugs`

> ⚠️ The benchmark artifact does not currently capture the generator
> ANSWER text or the retrieved passages — only the judge's verdict
> on each. To label this query against ground truth you need to
> re-run the RAG pipeline locally and read the answer + context. A
> follow-up commit should add answer/context capture to the
> benchmark JSON output. For now, the judge's claim lists are
> provided below as a proxy.

### Judge verdicts (off → on)

| | Score | Supported | Unsupported |
| - | -: | -: | -: |
| Thinking off | 1.000 | 23 | 0 |
| Thinking on  | 0.900 | 18 | 2 |

**Reasoning (thinking off):**

> Every factual claim in the answer maps directly to content in the retrieved passages. The bug prediction scanner details (patterns, severities, contextual risk factors, false-positive filtering) all come from P1. The CLI syntax comes from P3. The CI/CD pipeline details (security scan stage tools, blocking behavior, early bug detection rationale) come from P2. No claims appear to rely on outside knowledge or invented details.

**Reasoning (thinking on):**

> The answer is overwhelmingly faithful to the retrieved passages. Nearly all factual claims about /bug-predict — its purpose, syntax, the three risk pattern categories (dangerous_eval, broad_exception, incomplete_code) with their severities, contextual risk factors (cyclomatic complexity, change frequency, code smells), and the recommended use-case timing — are drawn verbatim or near-verbatim from P1 and P3. The CI/CD section accurately reflects P2's pipeline stage table and automation rationale. Two minor unsupported claims exist: the framing that CI/CD specifically addresses "bugs that slip through code review" (P2 frames CI as catching bugs early, not as a backstop for code review gaps), and the concluding synthesis sentence asserting that the combination of both tools gives "the best chance" of catching bugs, which is an inferential editorial claim not found in any passage.

**Claims identified by judge (thinking off):**

Supported:
  - The /bug-predict command scans your codebase for patterns that historically cause production incidents
  - It analyzes code structure, complexity, and known anti-patterns to identify where failures are most likely to happen
  - The command syntax is: /bug-predict <path or directory to scan>
  - The scanner looks for three categories of risk
  - The dangerous_eval pattern has HIGH severity
  - dangerous_eval catches eval(), exec(), compile() on any input — code injection vectors
  - The broad_exception pattern has MEDIUM severity
  - broad_exception catches bare except:, unlogged except Exception: — errors silently swallowed
  - The incomplete_code pattern has LOW severity
  - incomplete_code catches TODO, FIXME, HACK, XXX comments — unfinished code paths that break under edge cases
  - The scanner also weighs cyclomatic complexity — deeply nested conditionals and long function bodies
  - Change frequency — frequently modified ('hot') files are more likely to contain regressions
  - Code smells — functions over 50 lines, classes with too many methods, duplicated logic across modules
  - The scanner automatically suppresses known-safe patterns such as eval() inside test fixture strings
  - The scanner suppresses JavaScript regex.exec() method calls
  - A typical Python CI pipeline includes a Security scan stage using tools like bandit, pip-audit, detect-secrets
  - The security scan stage blocks deployment if issues are found
  - Running checks on every commit means bugs are caught early rather than hiding until someone manually runs the test suite — which might be days or weeks after broken code was merged
  - Before merging a large PR — catch patterns humans miss
  - During code review — focus human attention on real risks
  - After onboarding unfamiliar code — map risk hotspots fast
  - As a periodic health check on high-churn modules
  - Before a release — verify no new high-severity patterns crept in

Unsupported:


**Claims identified by judge (thinking on):**

Supported:
  - The /bug-predict tool scans your codebase for patterns that historically cause production incidents
  - The tool analyzes code structure, complexity, and known anti-patterns to identify where failures are most likely to happen
  - The tool is run with: /bug-predict <path or directory to scan>
  - The scanner looks for three categories of risk
  - The dangerous_eval pattern has HIGH severity and involves eval(), exec(), compile() on input as code injection vectors
  - The broad_exception pattern has MEDIUM severity and involves bare except: or unlogged exceptions where errors are silently swallowed
  - The incomplete_code pattern has LOW severity and involves TODO, FIXME, HACK, XXX comments representing unfinished code paths that break on edge cases
  - Cyclomatic complexity (deeply nested conditionals and long function bodies) correlates with higher defect rates
  - Frequently modified ('hot') files are more likely to contain regressions
  - Code smells include functions over 50 lines, classes with too many methods, and duplicated logic across modules
  - A typical CI/CD pipeline includes a security scan stage using tools like bandit that blocks deployment if issues are found
  - Running tests catches regressions before they reach main
  - Security scanning finds vulnerabilities before they ship
  - Dependency auditing catches known CVEs in your supply chain
  - Before merging a large PR is a use case for bug prediction to catch patterns humans miss
  - During code review is a use case for bug prediction to focus human attention on real risks
  - After onboarding unfamiliar code is a use case for bug prediction to map risk hotspots fast
  - Before a release is a use case for bug prediction to verify no new high-severity patterns crept in

Unsupported:
  - For bugs that slip through code review, automate detection on every push (the specific framing that CI/CD addresses bugs slipping through code review is not stated in the passages — P2 states CI catches bugs early on every commit but does not specifically frame it as addressing gaps in code review)
  - The combination of proactive bug prediction scanning and automated CI/CD checks gives you the best chance of catching hard-to-find bugs before they reach production (this is an editorial synthesis conclusion not stated in any passage)

### Your labels

```yaml
id: gq-015
# Overall verdict: faithful | partial | unfaithful | refusal_ok
verdict: TBD
# Your scalar score = supported / (supported + unsupported), 0.0 to 1.0.
# For "refusal_ok" answers (no factual claims), score 1.0.
faithfulness_score: TBD
# Optional: list claims as you see them. Counts only the bottom number.
claims:
  - text: "..."
    supported: true
  - text: "..."
    supported: false
notes: |
  Optional commentary on what the judge missed or got wrong.
```

---

## gq-038 — `coordinate documentation updates across the project`

> ⚠️ The benchmark artifact does not currently capture the generator
> ANSWER text or the retrieved passages — only the judge's verdict
> on each. To label this query against ground truth you need to
> re-run the RAG pipeline locally and read the answer + context. A
> follow-up commit should add answer/context capture to the
> benchmark JSON output. For now, the judge's claim lists are
> provided below as a proxy.

### Judge verdicts (off → on)

| | Score | Supported | Unsupported |
| - | -: | -: | -: |
| Thinking off | 0.909 | 10 | 1 |
| Thinking on  | 0.824 | 14 | 3 |

**Reasoning (thinking off):**

> Almost all claims in the answer are directly and explicitly supported by passage P1, which describes the Doc Orchestrator tool, its parameters, default values, usage syntax, group membership, and related tools. The tip about persisting documentation decisions is grounded in P2, which does list architecture decisions and project conventions as good candidates for memory. The only mildly unsupported claim is the specific framing that storing documentation decisions ensures "future sessions stay consistent" — P2 says memory is for things you learn along the way and gives examples, but doesn't explicitly tie documentation decisions to session consistency in exactly that way; however, this is very close to what P2 states ("Save a project convention so every session follows the same rules"), making it borderline. Overall the answer is highly faithful to the retrieved passages.

**Reasoning (thinking on):**

> The answer is largely faithful to P1 and P2. Nearly all claims about the Doc Orchestrator's description, parameters, defaults, and related tools map directly onto P1's tabular reference. The tip about persisting decisions in memory is well-grounded in P2. Three minor unsupported claims arise from inferential overreach: (1) "point it at any project root" extrapolates beyond what the parameter description confirms; (2) "in one step" is editorial framing not present in P1, which only says "end-to-end"; and (3) calling the related tools a "broader workflow toolset" imposes a structural grouping that P1 does not assert — it merely lists them as related topics.

**Claims identified by judge (thinking off):**

Supported:
  - The Doc Orchestrator tool is purpose-built for coordinating documentation updates across a project.
  - The Doc Orchestrator handles end-to-end documentation maintenance.
  - The Doc Orchestrator scouts gaps, prioritizes, generates, and updates docs.
  - The tool is called with doc_orchestrator().
  - The Doc Orchestrator accepts an optional `path` parameter.
  - The `path` parameter is a string for the project root path.
  - The `path` parameter defaults to `.` if not specified.
  - The Doc Orchestrator is part of a workflow group of tools.
  - Related tools include Security Audit, Bug Predict, and Code Review.
  - Architecture decisions and project conventions are good candidates for persistent memory.

Unsupported:
  - Storing documentation decisions in memory helps future sessions stay consistent (the passages mention storing decisions, but the specific framing of 'stay consistent' in the context of documentation decisions is an inference not directly stated).

**Claims identified by judge (thinking on):**

Supported:
  - The Doc Orchestrator tool handles end-to-end documentation maintenance
  - The Doc Orchestrator includes scouting gaps, prioritizing, generating, and updating docs
  - The tool is called with doc_orchestrator()
  - The `path` parameter is of type string
  - The `path` parameter description is 'Project root path'
  - The default value for the `path` parameter is '.'
  - By default the tool runs against the current directory
  - The Doc Orchestrator belongs to the 'workflow' group
  - Security Audit detects vulnerabilities
  - Bug Predict analyzes code patterns
  - Code Review provides code quality checks
  - Security Audit, Bug Predict, and Code Review are listed as related topics to the Doc Orchestrator
  - Storing documentation conventions or architecture decisions in memory allows future sessions to follow the same rules
  - Memory persists across sessions so future sessions pick up without repeating yourself

Unsupported:
  - You can point the tool at any project root (beyond what the parameter description implies — not explicitly stated in passages)
  - The Doc Orchestrator coordinates the entire documentation pipeline in one step ('in one step' framing is not in P1)
  - Security Audit, Bug Predict, and Code Review collectively form a 'broader workflow toolset' alongside the Doc Orchestrator (P1 lists them only as related topics, not as members of a named shared toolset)

### Your labels

```yaml
id: gq-038
# Overall verdict: faithful | partial | unfaithful | refusal_ok
verdict: TBD
# Your scalar score = supported / (supported + unsupported), 0.0 to 1.0.
# For "refusal_ok" answers (no factual claims), score 1.0.
faithfulness_score: TBD
# Optional: list claims as you see them. Counts only the bottom number.
claims:
  - text: "..."
    supported: true
  - text: "..."
    supported: false
notes: |
  Optional commentary on what the judge missed or got wrong.
```

---

## gq-030 — `reduce code complexity`

> ⚠️ The benchmark artifact does not currently capture the generator
> ANSWER text or the retrieved passages — only the judge's verdict
> on each. To label this query against ground truth you need to
> re-run the RAG pipeline locally and read the answer + context. A
> follow-up commit should add answer/context capture to the
> benchmark JSON output. For now, the judge's claim lists are
> provided below as a proxy.

### Judge verdicts (off → on)

| | Score | Supported | Unsupported |
| - | -: | -: | -: |
| Thinking off | 0.917 | 11 | 1 |
| Thinking on  | 1.000 | 13 | 0 |

**Reasoning (thinking off):**

> Almost all claims in the answer are directly and explicitly supported by the retrieved passages. P3 supports the simplify_code tool description, its default path parameter, and its no-required-parameters usage. P1 supports all refactor plan claims about structural analysis, prioritization factors, and the yak-shaving warning. P2 explicitly states that cyclomatic complexity and deep nesting correlate with higher defect rates. The one unsupported claim is the interpretive synthesis that bug prediction helps identify 'where complexity is most likely causing real risk' — the passage notes complexity as a risk factor but does not frame it specifically as a tool for identifying risk caused by complexity. The recommended starting point combining both tools is also a synthesized recommendation not explicitly stated in the passages, but it is framed as advice rather than a factual claim about the tools, so it is borderline; however, since it asserts a workflow not described in any passage, it could be considered unsupported. That said, it is a soft recommendation rather than a hard factual claim, so it is not included in unsupported claims. Overall, the answer is highly faithful to the passages.

**Reasoning (thinking on):**

> Every atomic factual claim in the answer maps directly to explicit statements in the retrieved passages. The description of `simplify_code()` and its default parameter are taken verbatim from P3. All Refactor Plan details (what it analyzes, prioritization factors, use-cases) are stated explicitly in P1. The claim about complexity being a bug risk factor and correlating with defect rates is stated word-for-word in P2. No outside knowledge, invented details, or unsupported inferences were introduced.

**Claims identified by judge (thinking off):**

Supported:
  - The `simplify_code()` tool finds complex code hotspots and suggests simplifications to reduce cognitive load.
  - The `simplify_code()` tool scans a directory defaulting to the current directory.
  - The `simplify_code()` tool can be run with no required parameters.
  - A refactor plan scans your code for structural problems and builds a prioritized roadmap to fix them.
  - The refactor plan analyzes complexity — high cyclomatic complexity, deep nesting, and long chains.
  - The refactor plan analyzes code smells — long methods, god classes, feature envy, and data clumps.
  - The refactor plan analyzes duplication — copy-pasted blocks, near-duplicates, and DRY violations.
  - The refactor plan prioritizes fixes by weighing severity, effort, impact, and risk.
  - High-severity, low-effort, high-impact items float to the top.
  - Refactoring without a plan can lead to yak-shaving — starting to fix one thing and ending up touching 20 files.
  - The bug prediction scanner factors in complexity signals, noting that deeply nested conditionals and long function bodies correlate with higher defect rates.

Unsupported:
  - Bug prediction can help you identify where complexity is most likely causing real risk.

**Claims identified by judge (thinking on):**

Supported:
  - The `simplify_code()` tool finds complex code hotspots and suggests simplifications to reduce cognitive load.
  - The `simplify_code()` tool scans a directory, defaulting to the current directory.
  - The `simplify_code()` tool can be run with no required parameters.
  - A Refactor Plan builds a prioritized roadmap to fix structural problems, including complexity issues.
  - The Refactor Plan analyzes high cyclomatic complexity, deep nesting, and long chains.
  - The Refactor Plan analyzes long methods and god classes.
  - The Refactor Plan analyzes dead code such as unreachable branches and unused parameters.
  - Items in a Refactor Plan are prioritized by severity, effort, impact, and risk.
  - High-severity, low-effort, high-impact fixes float to the top of the prioritized list.
  - A Refactor Plan is especially useful before adding features to a tangled area.
  - A Refactor Plan is useful when you need data to justify refactoring time to stakeholders.
  - High complexity is a bug risk factor — deeply nested conditionals and long function bodies correlate with higher defect rates.
  - Running a bug prediction scan can help identify which complex areas are most likely to cause production incidents.

Unsupported:


### Your labels

```yaml
id: gq-030
# Overall verdict: faithful | partial | unfaithful | refusal_ok
verdict: TBD
# Your scalar score = supported / (supported + unsupported), 0.0 to 1.0.
# For "refusal_ok" answers (no factual claims), score 1.0.
faithfulness_score: TBD
# Optional: list claims as you see them. Counts only the bottom number.
claims:
  - text: "..."
    supported: true
  - text: "..."
    supported: false
notes: |
  Optional commentary on what the judge missed or got wrong.
```

---

## gq-008 — `prepare a release`

> ⚠️ The benchmark artifact does not currently capture the generator
> ANSWER text or the retrieved passages — only the judge's verdict
> on each. To label this query against ground truth you need to
> re-run the RAG pipeline locally and read the answer + context. A
> follow-up commit should add answer/context capture to the
> benchmark JSON output. For now, the judge's claim lists are
> provided below as a proxy.

### Judge verdicts (off → on)

| | Score | Supported | Unsupported |
| - | -: | -: | -: |
| Thinking off | 1.000 | 25 | 0 |
| Thinking on  | 0.920 | 23 | 2 |

**Reasoning (thinking off):**

> Every atomic factual claim in the answer is directly and explicitly supported by at least one of the retrieved passages. The command syntax, natural language invocations, guided scoping questions and their defaults, the five check areas and their descriptions, the GO/NO-GO verdict definitions, the single-blocker rule, and all the post-assessment follow-up phrases are all verbatim or near-verbatim reproductions from P1, P2, and P3. There are no hallucinated details, invented CLI flags, or outside-knowledge drift present in the answer.

**Reasoning (thinking on):**

> The vast majority of the answer is well-supported by the retrieved passages. The command syntax, natural language aliases, guided scoping questions, five check areas with details, GO/NO-GO verdict definitions, and post-assessment follow-up phrases are all directly stated in P1, P2, and P3. Two minor unsupported claims exist: (1) the answer characterizes "update the changelog for this release" as specifically being for "adding a missing changelog entry," while P3 only labels the goal as "Update changelog" — adding specificity not stated in the passage; (2) the answer qualifies "tag and publish" with "once you have a GO verdict," but P3 simply lists it as a post-assessment action without that conditional framing.

**Claims identified by judge (thinking off):**

Supported:
  - To prepare a release, use the release prep skill, which runs a preflight checklist before you publish.
  - The command to invoke is: /release-prep <version or 'check'>
  - The command is run in a Claude Code conversation.
  - You can use natural language 'is this ready to release?' to invoke it.
  - You can use natural language 'prepare release 5.6.0' to invoke it.
  - You can use natural language 'run the release preflight' to invoke it.
  - You can use natural language 'check if I can ship this' to invoke it.
  - Before running, the skill asks two questions.
  - The first question is about what version, with options being a semver string or 'check' to auto-detect, defaulting to the current version in pyproject.toml.
  - The second question is about what stage, with options being full prep, prep check, changelog only, or security only, defaulting to full prep.
  - The skill checks five areas: Health, Security, Changelog, Dependencies, and Version.
  - Health checks that tests pass, lint is clean, and coverage is above threshold.
  - Security checks for no CVEs, no eval/exec, and secrets scan clean.
  - Changelog checks that an entry exists for this version and the date is current.
  - Dependencies checks for pinned versions, no vulnerabilities, and compatible ranges.
  - Version checks that the semver bump matches changes, pyproject.toml is updated, and dist builds cleanly.
  - After all checks complete, you receive a single GO or NO-GO verdict.
  - GO means every check passed or has only advisory warnings and it is safe to tag, build, and publish.
  - NO-GO means one or more blocking issues were found and the report lists exactly what to fix before retrying.
  - A single blocking issue is enough for a NO-GO verdict.
  - The report lists all blockers so you can fix them in one pass.
  - After the assessment, you can say 'fix the blockers' to fix all blockers.
  - After the assessment, you can say 'update the changelog for this release' to update the changelog.
  - After the assessment, you can say 'run release prep again' to re-run after fixes.
  - After the assessment, you can say 'tag and publish' to tag and publish.

Unsupported:


**Claims identified by judge (thinking on):**

Supported:
  - To prepare a release, use the /release-prep skill in Claude Code.
  - The command syntax is: /release-prep <version or 'check'>
  - The natural language phrase 'prepare release 5.6.0' can be used to invoke release prep.
  - The natural language phrase 'run the release preflight' can be used to invoke release prep.
  - The natural language phrase 'check if I can ship this' can be used to invoke release prep.
  - The natural language phrase 'is this ready to release?' can be used to invoke release prep.
  - The skill asks two scoping questions before running.
  - The first question is 'What version?' with options being a semver string or 'check' to auto-detect, defaulting to the current version in pyproject.toml.
  - The second question is 'What stage?' with options Full prep, prep check, changelog only, security only, defaulting to Full prep.
  - Release prep runs a preflight checklist across five check areas.
  - The Health check verifies tests pass, lint clean, coverage above threshold, and is blocking if tests fail.
  - The Security check verifies no new CVEs, no eval/exec, secrets scan clean, and is blocking if critical findings.
  - The Changelog check verifies an entry exists for this version and the date is current, and is blocking if missing.
  - The Dependencies check verifies pinned versions, no known vulnerabilities, compatible ranges, and is blocking if a vulnerable dep is found.
  - The Version check verifies the semver bump matches changes, pyproject.toml is updated, and dist builds cleanly, and is blocking if there is a version mismatch.
  - After all checks complete, you receive either a GO or NO-GO verdict.
  - GO means every check passed or has only advisory warnings and it is safe to tag, build, and publish.
  - NO-GO means one or more blocking issues were found and the report lists exactly what to fix before retrying.
  - A single blocking issue is enough for a NO-GO verdict.
  - All blockers are listed in one report so you can fix them in one pass.
  - After the assessment, saying 'fix the blockers' addresses blocking issues.
  - After the assessment, saying 'run release prep again' re-runs after fixes.
  - After the assessment, saying 'tag and publish' triggers tagging and publishing.

Unsupported:
  - Saying 'update the changelog for this release' specifically adds a missing changelog entry (the passage only says the goal is 'Update changelog', not specifically adding a missing entry).
  - Saying 'tag and publish' is specifically for use 'once you have a GO verdict' (the passage lists it as a post-assessment action but does not condition it on having a GO verdict).

### Your labels

```yaml
id: gq-008
# Overall verdict: faithful | partial | unfaithful | refusal_ok
verdict: TBD
# Your scalar score = supported / (supported + unsupported), 0.0 to 1.0.
# For "refusal_ok" answers (no factual claims), score 1.0.
faithfulness_score: TBD
# Optional: list claims as you see them. Counts only the bottom number.
claims:
  - text: "..."
    supported: true
  - text: "..."
    supported: false
notes: |
  Optional commentary on what the judge missed or got wrong.
```

---

## gq-003 — `fix failing tests`

> ⚠️ The benchmark artifact does not currently capture the generator
> ANSWER text or the retrieved passages — only the judge's verdict
> on each. To label this query against ground truth you need to
> re-run the RAG pipeline locally and read the answer + context. A
> follow-up commit should add answer/context capture to the
> benchmark JSON output. For now, the judge's claim lists are
> provided below as a proxy.

### Judge verdicts (off → on)

| | Score | Supported | Unsupported |
| - | -: | -: | -: |
| Thinking off | 1.000 | 16 | 0 |
| Thinking on  | 1.000 | 16 | 0 |

**Reasoning (thinking off):**

> All factual claims in the answer are directly and explicitly supported by the retrieved passages. The quick start commands, the four-step process (Scoping, Diagnosis, Repair+retry, Report), the root cause table with auto-fixability ratings, and the behavior when all 3 attempts are exhausted are all drawn verbatim or near-verbatim from P1, P2, and P3. No outside knowledge or invented details are present.

**Reasoning (thinking on):**

> Every atomic factual claim in the answer maps directly to content in the retrieved passages. The inline-naming behavior and skipping of questions is stated in P3. Both example commands appear verbatim in P3. The three-step flow (diagnosis, repair+retry up to 3 attempts, report) is described in P1. The root-cause table with all six entries and their fixability ratings is taken directly from P1. The exhausted-attempts summary behaviour is documented in P3. All four post-results follow-up suggestions are listed word-for-word in P3. No hallucinations or unsupported inferences were detected.

**Claims identified by judge (thinking off):**

Supported:
  - Naming the test directly skips setup questions and jumps straight to diagnosis
  - You can run 'fix tests/unit/test_auth.py' to fix a failing test
  - You can use '/fix-test tests/unit/test_auth.py::test_login_expired' to fix a specific test
  - The fix process includes a Scoping step where you're asked which test is failing and whether it broke recently
  - The Scoping step is skipped if you name the test inline
  - The fix process includes a Diagnosis step where the failing test is run and the traceback is classified into a root cause category
  - The fix process includes a Repair + retry step where the fix is applied and the test is re-run
  - There are up to 3 total attempts in the retry cycle
  - The fix process includes a Report step where you see which tests were fixed, which still fail, how many attempts were used, and what to try next
  - Import errors (renamed/moved modules) are auto-fixable
  - Stale mock targets are auto-fixable
  - Assertion drift is auto-fixable
  - Type/signature errors are auto-fixable
  - Missing fixtures are 'Usually' auto-fixable
  - Environment issues are 'Sometimes' auto-fixable
  - If all 3 attempts are exhausted, you get a clear summary of what was tried and actionable suggestions for what still needs manual attention

Unsupported:


**Claims identified by judge (thinking on):**

Supported:
  - Naming the test inline skips all setup questions and jumps straight to diagnosis
  - The command `fix tests/unit/test_auth.py` can be used to fix a failing test
  - The command `/fix-test tests/unit/test_auth.py::test_login_expired` can be used to target a specific test
  - The tool runs the failing test and classifies the traceback into a root cause category
  - A fix is applied and the test is re-run, up to 3 attempts total
  - The report shows which tests were fixed, which still fail, how many attempts were used, and what to try next
  - Import errors are auto-fixable (Yes)
  - Mock mismatches are auto-fixable (Yes)
  - Assertion drift is auto-fixable (Yes)
  - Type errors are auto-fixable (Yes)
  - Missing fixtures are auto-fixable (Usually)
  - Environment issues are auto-fixable (Sometimes)
  - If all 3 attempts are exhausted, you get a clear summary of what was tried and what remains manually
  - After results, you can fix more tests with 'fix the rest of the failing tests in tests/unit/'
  - After results, you can check for similar breakage with 'are there other tests with the same stale import?'
  - After results, you can generate new tests with 'generate tests for the module I just refactored'

Unsupported:


### Your labels

```yaml
id: gq-003
# Overall verdict: faithful | partial | unfaithful | refusal_ok
verdict: TBD
# Your scalar score = supported / (supported + unsupported), 0.0 to 1.0.
# For "refusal_ok" answers (no factual claims), score 1.0.
faithfulness_score: TBD
# Optional: list claims as you see them. Counts only the bottom number.
claims:
  - text: "..."
    supported: true
  - text: "..."
    supported: false
notes: |
  Optional commentary on what the judge missed or got wrong.
```

---

## gq-007 — `refactor my code`

> ⚠️ The benchmark artifact does not currently capture the generator
> ANSWER text or the retrieved passages — only the judge's verdict
> on each. To label this query against ground truth you need to
> re-run the RAG pipeline locally and read the answer + context. A
> follow-up commit should add answer/context capture to the
> benchmark JSON output. For now, the judge's claim lists are
> provided below as a proxy.

### Judge verdicts (off → on)

| | Score | Supported | Unsupported |
| - | -: | -: | -: |
| Thinking off | 1.000 | 14 | 0 |
| Thinking on  | 1.000 | 14 | 0 |

**Reasoning (thinking off):**

> Every factual claim in the answer maps directly to content in the retrieved passages. The quick-start commands and path-not-specified prompt come from P3. The roadmap description with effort estimates and suggested fixes is from P3. The category table (code smells, duplication, complexity, coupling, dead code) with their specific sub-items is drawn from P1. The prioritization logic (high-severity, low-effort, high-impact float to top; risky changes flagged) is from P1. The specific focus examples ('simplify src/engine.py', 'find duplication in src/', 'check coupling between auth and models') are verbatim from P3. No claims were invented or drawn from outside knowledge.

**Reasoning (thinking on):**

> Every atomic factual claim in the answer maps directly to at least one retrieved passage. The two natural-language example commands and the command table are lifted verbatim from P3. The five analysis categories (code smells, duplication, complexity, coupling, dead code) and their sub-items are drawn word-for-word from P1's table. The prioritization factors (severity, effort, impact, risk) and the statement that high-severity/low-effort/high-impact items float to the top are quoted directly from P1. The prompt quoted when no path is supplied ("Which file or directory needs refactoring analysis?") appears verbatim in P3. No outside knowledge or invented details were introduced.

**Claims identified by judge (thinking off):**

Supported:
  - You can analyze code by saying what you want to analyze, e.g. 'analyze src/ for refactoring opportunities'
  - You can use the skill directly with /refactor-plan <path to analyze>
  - If you don't specify a path, you'll be asked: 'Which file or directory needs refactoring analysis?'
  - You'll receive a prioritized roadmap with effort estimates and suggested fixes
  - The analysis covers code smells such as long methods, god classes, and feature envy
  - The analysis covers duplication such as copy-pasted blocks and DRY violations
  - The analysis covers complexity such as deep nesting and high cyclomatic complexity
  - The analysis covers coupling such as circular imports and tight dependencies
  - The analysis covers dead code such as unused params and unreachable branches
  - Items that are high-severity, low-effort, and high-impact float to the top
  - Risky changes get flagged so you can plan extra testing
  - You can narrow the analysis with 'simplify src/engine.py' for complexity only
  - You can narrow the analysis with 'find duplication in src/' for copy-paste detection
  - You can narrow the analysis with 'check coupling between auth and models' for dependency analysis

Unsupported:


**Claims identified by judge (thinking on):**

Supported:
  - You can analyze code by saying 'analyze src/auth/ for refactoring opportunities'
  - You can use the skill directly with '/refactor-plan src/auth/'
  - '/refactor-plan src/models.py' scans one file
  - '/refactor-plan src/workflows/' scans a directory tree
  - '/refactor-plan .' scans the whole project
  - The analysis produces a prioritized roadmap
  - Code smells detected include long methods, god classes, and feature envy
  - Duplication detected includes copy-pasted blocks and DRY violations
  - Complexity detected includes high cyclomatic complexity and deep nesting
  - Coupling detected includes circular imports and tight dependencies
  - Dead code detected includes unreachable branches and unused parameters
  - Items are ranked by severity, effort, impact, and risk
  - High-severity, low-effort, high-impact fixes float to the top
  - If no path is specified, the user is asked 'Which file or directory needs refactoring analysis?'

Unsupported:


### Your labels

```yaml
id: gq-007
# Overall verdict: faithful | partial | unfaithful | refusal_ok
verdict: TBD
# Your scalar score = supported / (supported + unsupported), 0.0 to 1.0.
# For "refusal_ok" answers (no factual claims), score 1.0.
faithfulness_score: TBD
# Optional: list claims as you see them. Counts only the bottom number.
claims:
  - text: "..."
    supported: true
  - text: "..."
    supported: false
notes: |
  Optional commentary on what the judge missed or got wrong.
```

---

## gq-009 — `audit documentation for staleness`

> ⚠️ The benchmark artifact does not currently capture the generator
> ANSWER text or the retrieved passages — only the judge's verdict
> on each. To label this query against ground truth you need to
> re-run the RAG pipeline locally and read the answer + context. A
> follow-up commit should add answer/context capture to the
> benchmark JSON output. For now, the judge's claim lists are
> provided below as a proxy.

### Judge verdicts (off → on)

| | Score | Supported | Unsupported |
| - | -: | -: | -: |
| Thinking off | 1.000 | 10 | 0 |
| Thinking on  | 1.000 | 10 | 0 |

**Reasoning (thinking off):**

> Every factual claim in the answer is directly supported by passage P1. The tool description, usage syntax, parameter name/type/description/default, group membership, and all three related tools (Security Audit, Bug Predict, Code Review) with their brief descriptions are all explicitly stated in P1. No claims rely on outside knowledge or invented details. Passages P2 and P3 were not drawn upon by the answer, and correctly so, as they are irrelevant to doc_audit.

**Reasoning (thinking on):**

> Every atomic factual claim in the answer is directly and explicitly supported by passage P1. The tool description, parameter name/type/description/default, usage syntax, group, source file, and all three related tool descriptions (Security Audit, Bug Predict, Code Review) are all verbatim or near-verbatim renderings of content found in P1. No outside knowledge, inference beyond the passage, or invented details were introduced.

**Claims identified by judge (thinking off):**

Supported:
  - The doc_audit tool audits existing documentation for staleness, broken links, and drift from source code.
  - The tool is called using doc_audit() with no arguments.
  - The doc_audit tool has a parameter named 'path'.
  - The 'path' parameter is of type string.
  - The 'path' parameter describes the project root path.
  - The 'path' parameter defaults to '.' (current directory).
  - The doc_audit tool belongs to the 'workflow' group.
  - Security Audit is a related tool that detects vulnerabilities in your codebase.
  - Bug Predict is a related tool that analyzes code patterns and predicts bugs.
  - Code Review is a related tool that provides comprehensive code quality review.

Unsupported:


**Claims identified by judge (thinking on):**

Supported:
  - The doc_audit() tool audits existing documentation for staleness, broken links, and drift from source code.
  - The path parameter is of type string.
  - The path parameter describes the project root path.
  - The path parameter has a default value of '.'.
  - The usage syntax for the tool is doc_audit().
  - The doc_audit tool belongs to the workflow group.
  - The doc_audit tool is sourced from src/attune/mcp/tool_schemas.py.
  - Security Audit detects vulnerabilities in the codebase.
  - Bug Predict analyzes code patterns and predicts bugs.
  - Code Review provides comprehensive code quality review.

Unsupported:


### Your labels

```yaml
id: gq-009
# Overall verdict: faithful | partial | unfaithful | refusal_ok
verdict: TBD
# Your scalar score = supported / (supported + unsupported), 0.0 to 1.0.
# For "refusal_ok" answers (no factual claims), score 1.0.
faithfulness_score: TBD
# Optional: list claims as you see them. Counts only the bottom number.
claims:
  - text: "..."
    supported: true
  - text: "..."
    supported: false
notes: |
  Optional commentary on what the judge missed or got wrong.
```

---

