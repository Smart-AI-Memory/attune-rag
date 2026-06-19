# Faithfulness ground-truth labels

Source artifact: `artifacts/calibration/thinking-2026-05-15-v2.json`
Selected 17 queries (16 shifted + 4 controls; shift threshold = 0.01).

## How to label

For each query below, fill in the YAML block under "Your labels". The scoring script compares your `faithfulness_score` against both judge passes (off / on) to decide whether thinking-on aligns better with ground truth than thinking-off. See `docs/rag/faithfulness-thinking-calibration.md` for the decision matrix this feeds.

---

## gq-017 — `create documentation for my code`

### Retrieved context

> <passage>
> [P1] source: quickstarts/skill-doc-gen.md
> ---
> type: quickstart
> name: skill-doc-gen
> tags: [skill, claude-code]
> source: plugin/skills/doc-gen/SKILL.md
> ---
>
> # Quickstart: Use /doc-gen
>
> Generate documentation from source code — docstrings, READMEs, API references.
>
> ```
> /doc-gen <path or module to document>
> ```
>
> **Result:** Structured results in your Claude Code conversation.
>
> **Next:** See full reference: `attune help-docs ref-skill-doc-gen`
>
> ## Related Topics
>
> _No related topics yet._
> </passage>
>
> <passage>
> [P2] source: concepts/task-code-migration.md
> ---
> type: concept
> name: task-code-migration
> tags: [migration, python, upgrade, compatibility]
> source: developer-guidance
> ---
>
> # Code Migration
>
> Code migration is the process of moving your codebase from
> one version of a language, framework, or library to another.
> Unlike feature work, migrations touch every file that depends
> on the thing being upgraded. The risk is not in any single
> change -- it is in the sheer number of changes and the
> interactions between them.
>
> ## Migration types
>
> | Migration type | What changes | Scope | Typical risk | Rollback possible? | Example |
> |---|---|---|---|---|---|
> | **Python version** | Syntax, stdlib, builtins | Entire codebase | Medium-High | Yes (keep old runtime) | 3.9 to 3.12 |
> | **Framework major** | APIs, config, middleware, ORM | Framework-dependent code | High | Difficult | Django 4 to 5, Flask 2 to 3 |
> | **Library upgrade** | Function signatures, defaults, behavior | Import sites | Low-Medium | Yes (pin old version) | pydantic v1 to v2 |
> | **API version** | Endpoints, request/response shapes, auth | Client code | Medium | Yes (version header) | REST v2 to v3 |
> | **Architecture** | Module boundaries, data flow, deployment | Full system | Very High | Partial at best | Monolith to services |
> | **Build system** | Config files, scripts, CI pipelines | Build layer | Low | Yes (keep old config) | setup.py to pyproject.toml |
>
> The key insight: risk scales with scope. A library upgrade
> that touches 3 files is a different problem than a Python
> version upgrade that touches 300.
>
> ## Incremental vs big-bang migration
>
> Two fundamental strategies exist, and choosing the wrong
> one is the most common migration mistake.
>
> | Strategy | How it works | Best for | Worst for |
> |---|---|---|---|
> | **Incremental** | Migrate one module or feature at a time. Old and new coexist behind compatibility layers or feature flags. Ship each piece independently. | Large codebases, production systems, teams with limited migration experience | Tightly coupled systems where partial migration is impossible |
> | **Big-bang** | Migrate everything at once. Switch over in a single deploy or merge. | Small codebases, non-production code, migrations where partial states are invalid | Anything with more than a few hundred files or external consumers |
>
> Incremental migration is almost always safer. The cost is
> maintaining compatibility layers during the transition. The
> benefit is that you can ship, test, and roll back each piece
> independently.
>
> ## Compatibility layers
>
> A compatibility layer is code that lets old and new
> implementations coexist. It absorbs the differences so that
> callers don't need to change until you're ready.
>
> Common patterns:
>
> - **Shim functions** that wrap the new API to match the
>   old signature
> - **`__future__` imports** that bring new Python behavior
>   into old versions
> - **`try`/`except ImportError`** blocks that import from
>   the new location and fall back to the old one
> - **Type aliases** that map old names to new ones
>
> Compatibility layers are temporary. They exist to make the
> migration incremental. Once migration is complete, remove
> them -- they are tech debt by design.
>
> ## Feature flags for migration
>
> Feature flags let you deploy migrated code to production
> without activating it for all users. This separates the
> deploy from the rollout:
>
> 1. Deploy migrated code behind a flag (off by default)
> 2. Enable for internal users or a percentage of traffic
> 3. Monitor for errors, performance regressions, or
>    behavior changes
> 4. Gradually increase the rollout
> 5. Remove the flag and the old code path
>
> This is the safest approach for high-risk migrations where
> rollback needs to be instant.
>
> ## Risk assessment
>
> Before starting a migration, assess:
>
> | Factor | Low risk | High risk |
> |---|---|---|
> | **Test coverage** | 80%+ coverage on migrated code | Under 50% or no tests |
> | **Dependency count** | Few downstream consumers | Many packages depend on your code |
> | **Runtime behavior** | Pure type/syntax changes | Changed defaults, removed features |
> | **Rollback plan** | Pin old version and redeploy | No clear rollback path |
> | **Timeline pressure** | Weeks of buffer | EOL deadline next week |
>
> If more than two factors are "high risk," consider breaking
> the migration into smaller phases or investing in test
> coverage before starting.
>
> ## What makes migrations fail
>
> Most migration failures share the same root causes:
> starting without sufficient test coverage, skipping the
> compatibility assessment, trying to migrate everything at
> once, or underestimating the time required for edge cases.
> The first 80% of a migration goes fast. The last 20% --
> deprecation warnings, subtle behavior changes, CI
> platform differences -- takes longer than the first 80%.
>
> ## Want to learn more?
>
> - Say **"how do I migrate my Python version?"** for a
>   step-by-step guide to planning and executing a version
>   upgrade
> - Say **"show me migration patterns"** for the full
>   reference with per-version checklists, codemods, and
>   common pitfalls
> - Say **"I need to upgrade Python"** for a 5-minute
>   quickstart
> - Ask **"/code-quality"** to scan migrated code for style
>   issues and deprecated patterns
> - Ask **"/smart-test"** to find untested code before
>   starting a migration
> - Ask **"/refactor"** to plan large-scale structural
>   changes as part of a migration
> </passage>
>
> <passage>
> [P3] source: concepts/tool-code-quality.md
> ---
> type: concept
> name: tool-code-quality
> tags: [review, quality, linting, skill]
> source: plugin/skills/code-quality/SKILL.md
> ---
>
> # Code Quality Review
>
> A code quality review looks at your code from multiple
> angles at once — style, correctness, likely bugs, and
> structural health. Instead of running a linter, then a
> bug scanner, then a reviewer, and reading three separate
> reports, you get one pass with a unified score.
>
> ## What it catches
>
> | Category | Examples | Severity | Auto-fixable? |
> |----------|---------|----------|---------------|
> | **Style** | Unused imports, line length, naming | Low | Often yes |
> | **Correctness** | Unreachable code, wrong return types | High | Sometimes |
> | **Likely bugs** | Broad exceptions, mutable defaults, race conditions | High | No |
> | **Structural** | High coupling, god classes, circular deps | Medium | No |
> | **Security overlap** | eval/exec, unvalidated paths (deep only) | Critical | No |
>
> ## When you'd use it
>
> Before opening a pull request — catch issues before
> reviewers do. After a large refactor — verify nothing
> degraded. When inheriting unfamiliar code — get a
> quick read on its health. Or any time you want a
> single number that summarizes code quality for a file
> or directory.
>
> ## How thorough it is
>
> | Depth | What it covers | Best for |
> |-------|---------------|----------|
> | **Quick** | Style and formatting only | Pre-commit check, fast feedback |
> | **Standard** | + Logic errors, likely bugs | Pull requests, regular development |
> | **Deep** | + Security, architecture, test gaps | Pre-release, critical modules |
>
> The default is standard.
>
> ## What to expect
>
> When you ask for a code quality review, you'll be asked
> a couple of questions first to scope the work — which
> files to review and how deep to go. This keeps the
> results focused on what you actually care about instead
> of dumping everything at once.
>
> ## Want to learn more?
>
> - Say **"tell me more"** for step-by-step instructions
> - Say **"what is security audit?"** for dedicated security analysis
> - Say **"tell me about bug prediction"** to predict where failures happen
> </passage>

### Answer

## Generating Documentation for Your Code

You can use the **`/doc-gen`** skill to create documentation from your source code [P1].

### What it does

The `/doc-gen` skill generates documentation from source code, including:

- **Docstrings**
- **READMEs**
- **API references**

[P1]

### How to use it

Run the following command in your Claude Code conversation:

```
/doc-gen <path or module to document>
```

Simply replace `<path or module to document>` with the path or module you want documented [P1].

### What to expect

You'll get structured results directly in your Claude Code conversation [P1].

### Want to learn more?

For the full reference guide, run: [P1]

```
attune help-docs ref-skill-doc-gen
```

---

> **Tip:** If your code has just gone through a large refactor or migration, you may also want to run a **code quality review** beforehand to verify nothing has degraded — this gives you a unified score covering style, correctness, and structural health before you document it [P3].


### Judge verdicts (off → on)

| | Score | Supported | Unsupported |
| - | -: | -: | -: |
| Thinking off | 1.000 | 10 | 0 |
| Thinking on  | 0.750 | 9 | 3 |

**Reasoning (thinking off):**

> Every factual claim in the answer maps directly to one of the retrieved passages. The `/doc-gen` usage, command syntax, output description, and help-docs reference all come verbatim from P1. The tip about running a code quality review after a refactor, including the description of a "unified score" covering style, correctness, and structural health, is explicitly stated in P3. There are no invented details, CLI flags, or workflow descriptions beyond what the passages provide.

**Reasoning (thinking on):**

> The core claims about `/doc-gen` — what it generates, its command syntax, and the full-reference command — are all directly and explicitly supported by P1. The claim about code quality review covering style, correctness, and structural health is supported by P3's table. However, two unsupported claims appear in the closing tip: (1) the suggestion to run a code quality review specifically before documenting code as a way to get "cleaner code to document" is an inference that bridges P1 and P3 in a way neither passage supports, and (2) P3 never lists documentation prep as a use case for code quality review. Additionally, the instructional gloss that the user should "replace" the placeholder in the command is a trivial but technically unstated inference from the command syntax in P1.

**Claims identified by judge (thinking off):**

Supported:
  - The `/doc-gen` skill generates documentation from source code
  - The `/doc-gen` skill can generate docstrings
  - The `/doc-gen` skill can generate READMEs
  - The `/doc-gen` skill can generate API references
  - The command to use the skill is `/doc-gen <path or module to document>`
  - The command is run in a Claude Code conversation
  - The results are structured and appear in your Claude Code conversation
  - The full reference guide can be accessed by running `attune help-docs ref-skill-doc-gen`
  - A code quality review can be used after a large refactor to verify nothing has degraded
  - A code quality review gives a unified score covering style, correctness, and structural health

Unsupported:


**Claims identified by judge (thinking on):**

Supported:
  - The `/doc-gen` skill can be used to create documentation from source code.
  - The `/doc-gen` skill can generate docstrings.
  - The `/doc-gen` skill can generate READMEs.
  - The `/doc-gen` skill can generate API references.
  - The command to use is `/doc-gen <path or module to document>`.
  - The command is run in a Claude Code conversation.
  - The result of `/doc-gen` is structured results in your Claude Code conversation.
  - The full reference guide can be accessed by running `attune help-docs ref-skill-doc-gen`.
  - A code quality review checks for style, correctness, and structural health.

Unsupported:
  - You should replace `<path or module to document>` with the path or module you want to document (this is an inference from command syntax, not explicitly stated in the passages).
  - Running a code quality review before documentation gives you cleaner code to document (P3 never mentions documentation as a use case for code quality review).
  - If your code has quality issues before you document it, you can run a code quality review first (the connection between pre-documentation cleanup and code quality review is not stated in any passage).

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

## gq-028 — `architect a new feature`

### Retrieved context

> <passage>
> [P1] source: concepts/tool-planning.md
> ---
> type: concept
> name: tool-planning
> tags: [planning, architecture, design]
> aliases: [plan a feature, architect a feature, design new feature, sprint planning, scope work tasks, feature roadmap]
> source: plugin/skills/planning/SKILL.md
> ---
>
> # Planning
>
> Planning helps you think through features, architecture,
> and TDD strategy before writing code. Instead of jumping
> straight to implementation, you get a structured plan
> with tasks, acceptance criteria, dependencies, and risk
> flags — when changes are cheapest to make.
>
> ## Planning modes
>
> | Mode | What it produces | Time | Use case |
> |------|-----------------|------|----------|
> | **Feature spec** | Goals, scope, non-goals, task breakdown with effort estimates | ~2-3 min | Starting a new feature or epic |
> | **TDD scaffold** | Test-first structure with red/green/refactor steps and test names | ~1-2 min | Complex logic that needs test coverage from the start |
> | **Architecture review** | Component analysis, coupling assessment, dependency map | ~2-3 min | Evaluating design decisions before committing to them |
>
> ## When you'd use it
>
> Before starting a new feature — define what "done" looks
> like and break the work into deliverables. When designing
> a TDD approach — get a test-first skeleton with the right
> granularity. When evaluating architecture — surface
> coupling, circular dependencies, and scaling concerns
> before they become expensive to fix.
>
> ## What it produces
>
> | Output | Description |
> |--------|-------------|
> | Task breakdown | Ordered steps with effort estimates and dependencies |
> | Acceptance criteria | Clear, testable definition of done per task |
> | Risk assessment | Blockers, unknowns, and mitigation strategies |
> | Scope boundaries | Explicit non-goals to prevent scope creep |
> | Dependency map | What depends on what, and what can run in parallel |
>
> ## What to expect
>
> When you ask for planning, you'll be guided through a
> couple of quick questions first — what you're planning
> and what kind of plan you need. This keeps the output
> focused on your actual goal instead of producing a
> generic plan. If you provide both details upfront (e.g.
> "plan a user authentication feature") the questions are
> skipped and it runs immediately.
>
> Runs on your Claude subscription — no API key or
> additional cost.
>
> ## Want to learn more?
>
> - Say **"tell me more"** for step-by-step instructions
> - Say **"what is spec?"** for spec-driven development
>   with approval loops
> - Say **"what is refactor plan?"** for code-level
>   refactoring analysis
> </passage>
>
> <passage>
> [P2] source: concepts/task-git-workflow.md
> ---
> type: concept
> name: task-git-workflow
> tags: [git, workflow, branching, merge]
> source: developer-guidance
> ---
>
> # Concept: Git workflow strategies
>
> ## What
>
> A git workflow is the branching, merging, and release
> strategy your team follows. Choosing the right one
> determines how easily you ship code, how often you hit
> conflicts, and how painful rollbacks are. The three major
> strategies are trunk-based development, feature branching,
> and gitflow.
>
> ## Why
>
> The wrong workflow creates friction at every step. A solo
> developer using gitflow drowns in ceremony. A 20-person
> team committing directly to `main` steps on each other
> constantly. The workflow should match team size, release
> cadence, and risk tolerance.
>
> ## The three strategies
>
> | Strategy | Best for | Branch lifetime | Release cadence | Conflict risk |
> |---|---|---|---|---|
> | Trunk-based | Small teams, CI/CD | Hours | Continuous | Low (small diffs) |
> | Feature branch | Most teams | Days | On merge | Medium |
> | Gitflow | Regulated releases | Days to weeks | Scheduled | High (long-lived branches) |
>
> ### Trunk-based development
>
> Everyone commits to `main` (or merges very short-lived
> branches). Code ships continuously. Feature flags hide
> incomplete work.
>
> - **Pro:** Minimal merge conflicts, fast feedback
> - **Con:** Requires strong CI and feature flags
> - **Use when:** You deploy multiple times per day
>
> ### Feature branching
>
> Each feature gets a branch off `main`. When done, the
> branch is merged back via pull request.
>
> - **Pro:** Isolates work, enables code review
> - **Con:** Long-lived branches diverge and conflict
> - **Use when:** You review code before merging
>
> ### Gitflow
>
> Separate branches for `develop`, `release/*`, `hotfix/*`,
> and `main`. Releases are cut from `develop`, stabilized,
> then merged to `main`.
>
> - **Pro:** Clear release process, parallel release prep
> - **Con:** High ceremony, frequent merge conflicts
> - **Use when:** You have scheduled releases with QA gates
>
> ## Merge vs rebase
>
> | Approach | What it does | When to use | Trade-off |
> |---|---|---|---|
> | Merge commit | Creates a merge node preserving both histories | Shared branches, PRs | Cluttered history but safe |
> | Rebase | Replays your commits on top of the target | Local feature branches before PR | Clean history but rewrites commits |
> | Squash merge | Combines all branch commits into one | Feature branches with messy history | Clean main, loses per-commit detail |
>
> **The golden rule:** Never rebase commits that other people
> have pulled. Rebase is for your local, unpushed work.
>
> ## Conflict resolution philosophy
>
> Conflicts are not failures -- they are information. Two
> people changed the same thing, and git is asking you to
> decide which version wins. The best conflict resolution
> happens *before* the conflict:
>
> - **Pull often.** The longer you wait, the more your
>   branch diverges.
> - **Keep branches short-lived.** Merge within days, not
>   weeks.
> - **Communicate.** If two people are editing the same
>   file, talk before branching.
>
> When you do hit a conflict, resolve it by understanding
> *why* both changes were made, not by blindly picking one
> side.
>
> ## Want to learn more?
>
> - Say **"how do I resolve a merge conflict?"** for the
>   step-by-step guide
> - Say **"show me all git commands"** for the full
>   reference with dangerous-command warnings
> - Say **"I have a merge conflict"** for the 5-step
>   quickstart
> - Try **/release** when preparing a release branch
> - Try **/code-quality** to check for pre-commit hook
>   issues before merging
>
> ## Related Topics
>
> - **Task**: Git workflow -- step-by-step guides for
>   conflicts, rebasing, cherry-pick, and stash
> - **Reference**: Git workflow -- full command catalog
>   with risk ratings and safer alternatives
> - **Quickstart**: Git workflow -- 5-step merge conflict
>   resolution
> </passage>
>
> <passage>
> [P3] source: tasks/use-planning.md
> ---
> type: task
> name: use-planning
> tags: [skill, task]
> source: plugin/skills/planning/SKILL.md
> ---
>
> # How to Plan Before You Code
>
> ## Quick start
>
> Say what you want to plan:
>
> ```
> plan a user authentication feature
> ```
>
> Or use the skill directly:
>
> ```
> /planning authentication feature with OAuth support
> ```
>
> You'll get a structured plan with tasks, acceptance
> criteria, and risk flags before writing any code.
>
> ## The guided flow
>
> When you ask for planning, you'll be guided through
> two quick questions before anything runs:
>
> | Step | What you're asked | Why |
> |------|-------------------|-----|
> | 1. Subject | "What are you planning?" | Names the feature, system, or behavior to plan around |
> | 2. Mode | "Feature spec, TDD approach, or architecture review?" | Each mode produces different output and depth |
>
> If you provide both upfront ("plan TDD for the payment
> module") the questions are skipped and it runs
> immediately.
>
> ## Choosing a planning mode
>
> | Mode | What you get | Time | When to use |
> |------|-------------|------|-------------|
> | **Feature spec** | Goals, scope, tasks, acceptance criteria, risks | ~2-3 min | Starting a new feature or epic |
> | **TDD scaffold** | Test-first skeleton with red/green/refactor steps | ~1-2 min | Complex logic that needs tests from day one |
> | **Architecture review** | Component analysis, coupling, dependency map | ~2-3 min | Evaluating design before committing |
>
> Natural language works:
>
> - "plan the new billing feature" — feature spec
> - "plan TDD for src/auth/" — TDD scaffold
> - "review the architecture of the plugin system" —
>   architecture review
>
> ## Reading the plan
>
> ### Feature spec output
>
> ```
> Feature Plan: User Authentication
>
> Goal
>   Allow users to log in with email/password or OAuth.
>
> Scope
>   - Email/password login and registration
>   - Google and GitHub OAuth providers
>   - Session management with JWT tokens
>
> Non-goals
>   - Multi-factor authentication (Phase 2)
>   - Social login beyond Google/GitHub
>
> Tasks
>   1. [2h] Set up auth module structure
>      Acceptance: auth/ directory with __init__.py,
>      models.py, routes.py
>   2. [4h] Implement email/password registration
>      Acceptance: POST /register returns 201 with JWT
>      Depends on: Task 1
>   3. [3h] Implement login endpoint
>      Acceptance: POST /login returns JWT for valid creds
>      Depends on: Task 2
>   4. [4h] Add Google OAuth flow
>      Acceptance: GET /auth/google redirects, callback
>      creates session
>      Depends on: Task 1
>   5. [3h] Add GitHub OAuth flow
>      Acceptance: Same pattern as Google
>      Depends on: Task 4
>
> Risks
>   - OAuth callback URL configuration varies per provider
>     Mitigation: Document setup per provider in README
>   - JWT secret rotation needs a strategy
>     Mitigation: Add rotation endpoint in Phase 2
>
> Estimated total: 16 hours
> ```
>
> ### TDD scaffold output
>
> ```
> TDD Plan: Payment Processing
>
> Red/Green/Refactor Steps
>
>   Step 1 — Red: test_charge_valid_card_returns_success
>     Write test asserting successful charge returns
>     status="success" and a transaction ID.
>     Run: pytest -k test_charge_valid_card (should FAIL)
>
>   Step 2 — Green: implement charge()
>     Minimal implementation that passes the test.
>     Run: pytest -k test_charge_valid_card (should PASS)
>
>   Step 3 — Red: test_charge_expired_card_raises
>     Write test asserting ExpiredCardError for expired
>     cards.
>     Run: pytest -k test_charge_expired (should FAIL)
>
>   Step 4 — Green: add expiry validation
>     Add date check before charging.
>     Run: pytest -k test_charge_expired (should PASS)
>
>   Step 5 — Refactor
>     Extract validation into _validate_card() helper.
>     Run: pytest (all should PASS)
> ```
>
> ## What to do next
>
> After the plan, you'll be offered follow-up options:
>
> | Goal | What to say |
> |------|-------------|
> | Start implementing | "let's build task 1" |
> | Refine the plan | "add more detail to task 3" |
> | Switch mode | "now do TDD for the auth module" |
> | Turn into a spec | "create a spec from this plan" |
> | Review architecture | "review the architecture of this plan" |
> | Go deeper | say "tell me more" for the full reference |
>
> ## Want to learn more?
>
> - Say **"tell me more"** for the complete reference
>   with all planning modes and output format
> - Say **"what is planning?"** to go back to the
>   overview
> - Say **"create a spec"** to turn a plan into a
>   spec-driven workflow with approval gates
> </passage>

### Answer

# Architecting a New Feature

Great — you're looking to do an **architecture review** before committing to a design. Here's how to get the most out of it.

---

## What an Architecture Review Produces

An architecture review gives you a **component analysis, coupling assessment, and dependency map** — surfacing design concerns before they become expensive to fix [P1]. Specifically, you'll get:

| Output | Description |
|--------|-------------|
| **Component analysis** | How your components relate and interact |
| **Coupling assessment** | Where tight coupling creates risk |
| **Dependency map** | What depends on what, and what can run in parallel |
| **Risk assessment** | Blockers, unknowns, and mitigation strategies |
| **Scope boundaries** | Explicit non-goals to prevent scope creep |

[P1, P3]

---

## How to Start

You can trigger it with natural language — just say something like:

```
review the architecture of the plugin system
```

Or be more specific about your feature:

```
/planning architecture review for [your feature name]
```

If you provide the subject and mode upfront, the questions are skipped and it runs immediately [P1, P3].

---

## The Guided Flow

If you don't provide full details, you'll be asked two quick questions [P3]:

1. **What are you planning?** — Names the feature or system
2. **Which mode?** — You'd choose **Architecture Review**

---

## When to Use It

Use an architecture review **before committing to design decisions** — to surface coupling, circular dependencies, and scaling concerns while changes are still cheap to make [P1].

---

## What's Next After the Review?

Once you have the architecture plan, you can [P3]:

- **"let's build task 1"** — move into implementation
- **"now do TDD for the [module]"** — switch to a test-first scaffold
- **"create a spec from this plan"** — turn it into a spec-driven workflow with approval gates
- **"add more detail to task 3"** — refine any specific part

What feature are you looking to architect?


### Judge verdicts (off → on)

| | Score | Supported | Unsupported |
| - | -: | -: | -: |
| Thinking off | 0.947 | 18 | 1 |
| Thinking on  | 0.812 | 13 | 3 |

**Reasoning (thinking off):**

> The answer is largely faithful to the retrieved passages. Nearly all claims about what an architecture review produces, when to use it, how to trigger it with natural language, the guided flow questions, and the follow-up options are directly supported by P1 and P3. The one unsupported claim is the specific CLI invocation format '/planning architecture review for [your feature name]' — while P3 does show '/planning authentication feature with OAuth support' as an example, the passages don't explicitly show this exact architecture-review variant of the command, and P1/P3 only show the natural language example 'review the architecture of the plugin system' for architecture mode. The table row for 'Scope boundaries' as an architecture review output is also slightly conflated — P1 lists scope boundaries as a general planning output, not specifically tied to architecture review — but since P1 attributes it to planning broadly and the answer cites P1, this is borderline supported. Overall, the answer is highly faithful with only one minor unsupported CLI flag format.

**Reasoning (thinking on):**

> The answer is largely faithful to the retrieved passages, correctly citing planning modes, timing, guided-flow steps, and follow-up options. However, two table-cell descriptions are invented rather than drawn from the passages: "Breakdown of how components interact" (for Component analysis) and "Where tight coupling could cause problems" (for Coupling assessment) — the passages simply name these outputs without those specific descriptions. Additionally, the specific CLI invocation `/planning architecture review for [your feature name]` is not shown in any passage; the passages only demonstrate `/planning authentication feature with OAuth support`, making that particular command form an unsupported invention.

**Claims identified by judge (thinking off):**

Supported:
  - An architecture review produces a component analysis, coupling assessment, and dependency map.
  - An architecture review surfaces design concerns before they become expensive to fix.
  - Architecture review outputs include component analysis showing how components relate and interact.
  - Architecture review outputs include a coupling assessment showing where tight coupling creates risk.
  - Architecture review outputs include a dependency map showing what depends on what and what can run in parallel.
  - Architecture review outputs include a risk assessment covering blockers, unknowns, and mitigation strategies.
  - Architecture review outputs include scope boundaries as explicit non-goals to prevent scope creep.
  - You can trigger an architecture review with natural language such as 'review the architecture of the plugin system'.
  - If you provide the subject and mode upfront, the questions are skipped and it runs immediately.
  - If you don't provide full details, you'll be asked two quick questions.
  - The first question is 'What are you planning?' which names the feature or system.
  - The second question is which mode you want.
  - Architecture review is used before committing to design decisions.
  - Architecture review surfaces coupling, circular dependencies, and scaling concerns while changes are still cheap to make.
  - After the review, you can say 'let's build task 1' to move into implementation.
  - After the review, you can say 'now do TDD for the [module]' to switch to a test-first scaffold.
  - After the review, you can say 'create a spec from this plan' to turn it into a spec-driven workflow with approval gates.
  - After the review, you can say 'add more detail to task 3' to refine any specific part.

Unsupported:
  - You can trigger an architecture review with '/planning architecture review for [your feature name]'.

**Claims identified by judge (thinking on):**

Supported:
  - An architecture review gives you component analysis, coupling assessment, and a dependency map.
  - The architecture review mode is designed for evaluating design decisions before committing to them.
  - The dependency map output shows what depends on what, and what can run in parallel.
  - The risk assessment output covers blockers, unknowns, and mitigation strategies.
  - The natural-language example 'review the architecture of the plugin system' triggers an architecture review.
  - If you provide enough detail upfront, the guided questions are skipped and it runs immediately.
  - If you don't specify details upfront, you'll be walked through two quick questions before the plan runs.
  - Step 1 of the guided flow asks 'What are you planning?' to name the feature or system.
  - Step 2 of the guided flow asks you to select a mode (which includes architecture review).
  - An architecture review takes approximately 2–3 minutes.
  - After the review, 'let's build task 1' is a follow-up option to start implementing.
  - After the review, 'create a spec from this plan' is a follow-up option to turn it into a spec-driven workflow.
  - After the review, 'add more detail to task 3' is a follow-up option to refine specific parts.

Unsupported:
  - Component analysis is described as 'Breakdown of how components interact' (this specific description does not appear in the passages).
  - Coupling assessment is described as 'Where tight coupling could cause problems' (this specific description does not appear in the passages).
  - The command '/planning architecture review for [your feature name]' is presented as a usable syntax (the passages only show '/planning authentication feature with OAuth support' as an example; the specific architecture-review variant of that command is not shown).

### Your labels

```yaml
id: gq-028
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

## gq-005 — `deep review my PR`

### Retrieved context

> <passage>
> [P1] source: references/tool-deep-review.md
> ---
> type: reference
> subtype: tabular
> name: tool-deep-review
> category: tool
> tags: [mcp, tool, workflow]
> aliases: [end-to-end review, PR review before merging, comprehensive code review, thorough review, deep dive review]
> source: src/attune/mcp/tool_schemas.py
> ---
>
> # Reference: Tool: Deep Review
>
> Multi-pass deep code review: security, quality, and test gap analysis with prioritized findings.
>
> **Group:** workflow
>
> ## Parameters
>
> | Parameter | Type | Description | Constraints | Default |
> | --------- | ---- | ----------- | ----------- | ------- |
> | `path` | string | Path to directory or file to review |  | required |
>
> ## Usage
>
> `deep_review(path="...")`
>
> ## Related Topics
> - **Reference**: Tool: Security Audit — Run security audit workflow on codebase. Detects vulnerabili...
> - **Reference**: Tool: Bug Predict — Run bug prediction workflow. Analyzes code patterns and pred...
> - **Reference**: Tool: Code Review — Run code review workflow. Provides comprehensive code qualit...
> </passage>
>
> <passage>
> [P2] source: quickstarts/run-code-review.md
> ---
> type: quickstart
> name: run-code-review
> tags: [workflow, code-quality]
> source: src/attune/cli_minimal.py
> ---
>
> # Quickstart: Run a code review
>
> Get AI-powered code quality analysis.
>
> ```
> attune workflow run code-review --path "src/"
> ```
>
> **Result:** Quality findings grouped by severity.
>
> **Next:** Follow up with `/smart-test` to generate tests for flagged areas.
>
> ## Related Topics
>
> _No related topics yet._
> </passage>
>
> <passage>
> [P3] source: references/tool-code-review.md
> ---
> type: reference
> subtype: tabular
> name: tool-code-review
> category: tool
> tags: [mcp, tool, workflow]
> source: src/attune/mcp/tool_schemas.py
> ---
>
> # Reference: Tool: Code Review
>
> Run code review workflow. Provides comprehensive code quality analysis with suggestions for improvement.
>
> **Group:** workflow
>
> ## Parameters
>
> | Parameter | Type | Description | Constraints | Default |
> | --------- | ---- | ----------- | ----------- | ------- |
> | `path` | string | Path to directory or file to review |  | required |
>
> ## Usage
>
> `code_review(path="...")`
>
> ## Related Topics
> - **Reference**: Tool: Security Audit — Run security audit workflow on codebase. Detects vulnerabili...
> - **Reference**: Tool: Bug Predict — Run bug prediction workflow. Analyzes code patterns and pred...
> - **Reference**: Tool: Test Generation — Generate tests for code. Can batch generate tests for multip...
> </passage>

### Answer

# Deep Review

To run a deep review on your PR, use the `deep_review` tool, which performs a **multi-pass deep code review covering security, quality, and test gap analysis with prioritized findings** [P1].

## How to Use It

Point it at the path you want reviewed:

```
deep_review(path="...")
```

Replace `"..."` with the path to your PR's directory or file [P1].

## Notes

- The `path` parameter is **required** [P1].
- The tool is part of the **workflow** group [P1].
- It is also known by aliases such as **end-to-end review**, **PR review before merging**, **comprehensive code review**, **thorough review**, and **deep dive review** [P1].

## Related Tools

You may also want to follow up with related tools such as **Security Audit**, **Bug Predict**, and **Code Review** for additional analysis [P1].


### Judge verdicts (off → on)

| | Score | Supported | Unsupported |
| - | -: | -: | -: |
| Thinking off | 1.000 | 14 | 0 |
| Thinking on  | 0.900 | 9 | 1 |

**Reasoning (thinking off):**

> Every factual claim in the answer maps directly to content in passage P1. The tool name, description (multi-pass, security/quality/test gap analysis, prioritized findings), usage syntax, parameter name and constraints, workflow group membership, all five aliases, and the three related tools are all explicitly listed in P1. There are no invented details or claims drawn from outside knowledge.

**Reasoning (thinking on):**

> The answer is largely faithful to the retrieved passages. Nearly every factual claim — the tool name, its multi-pass nature, the three coverage areas (security, quality, test gaps), prioritized findings, the call signature, and the parameter constraints — is directly stated in P1. The only unsupported claim is the concrete path example `src/` used to illustrate `deep_review(path="src/")`: that value appears in P2 but only in the context of the separate `code-review` CLI command, not the `deep_review` tool. Everything else maps cleanly to P1.

**Claims identified by judge (thinking off):**

Supported:
  - The tool used to run a deep review is called `deep_review`.
  - The deep review performs a multi-pass deep code review.
  - The deep review covers security, quality, and test gap analysis.
  - The deep review produces prioritized findings.
  - The usage syntax is `deep_review(path="...")`.
  - The `path` parameter accepts a path to a directory or file to review.
  - The `path` parameter is required.
  - The `deep_review` tool is part of the workflow group.
  - The tool has the alias 'end-to-end review'.
  - The tool has the alias 'PR review before merging'.
  - The tool has the alias 'comprehensive code review'.
  - The tool has the alias 'thorough review'.
  - The tool has the alias 'deep dive review'.
  - Related tools include Security Audit, Bug Predict, and Code Review.

Unsupported:


**Claims identified by judge (thinking on):**

Supported:
  - The tool to use is called `deep_review`.
  - The deep_review tool performs a multi-pass deep code review.
  - The review covers security, quality, and test gap analysis.
  - The review produces prioritized findings.
  - The tool is called with the syntax `deep_review(path="...")`.
  - The `path` parameter accepts either a directory or a file path.
  - The `path` parameter is of type string.
  - The `path` parameter is required (no default).
  - Security Audit and Bug Predict are listed as related tools to follow up with after a deep review.

Unsupported:
  - The example `deep_review(path="src/")` is given as a concrete illustration of the tool call — the `src/` path example does not appear in any passage for the `deep_review` tool (it appears only in P2 for the unrelated `code-review` CLI command).

### Your labels

```yaml
id: gq-005
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

## gq-002 — `generate tests for my code`

### Retrieved context

> <passage>
> [P1] source: quickstarts/generate-tests.md
> ---
> type: quickstart
> name: generate-tests
> tags: [workflow, testing]
> source: src/attune/cli_minimal.py
> ---
>
> # Quickstart: Generate tests for a module
>
> Auto-generate pytest tests for uncovered code.
>
> ```
> attune workflow run test-gen --path src/attune/help/engine.py
> ```
>
> **Result:** Generated test file with edge cases and assertions.
>
> **Next:** Run `pytest` to verify the generated tests pass.
>
> ## Related Topics
>
> _No related topics yet._
> </passage>
>
> <passage>
> [P2] source: concepts/tool-smart-test.md
> ---
> type: concept
> name: tool-smart-test
> tags: [testing, coverage, generation]
> source: plugin/skills/smart-test/SKILL.md
> ---
>
> # Smart Test
>
> Smart-test finds untested code and generates pytest tests
> to cover it. Instead of guessing what needs tests, you get
> a gap analysis that shows exactly which public functions,
> branches, and error paths have zero coverage — then it
> writes the tests for you.
>
> ## What it finds
>
> | Gap type | What it catches | Risk if untested |
> |----------|----------------|------------------|
> | **Untested functions** | Public APIs with no test file | High — regressions slip through silently |
> | **Missing branches** | if/else paths never executed | Medium — edge cases cause production bugs |
> | **Error paths** | Exception handlers never triggered | High — failures cascade unpredictably |
> | **Boundary values** | Empty inputs, None, zero, max-length | Medium — users hit edges you didn't |
> | **Parametrized combos** | Input combinations never paired | Low — interaction bugs are subtle |
>
> ## When you'd use it
>
> After writing new modules or public functions — catch
> gaps before they become tech debt. When coverage drops
> below the 80% threshold — find exactly what's missing.
> Before a release — verify error paths actually work.
> Or to bootstrap tests for legacy code with no coverage
> at all.
>
> ## What it produces
>
> | Output | Description |
> |--------|-------------|
> | Coverage gap report | Ranked list of untested functions with risk scores |
> | Generated tests | Working pytest functions with assertions |
> | Edge cases | Boundary values, empty inputs, None handling |
> | Error path tests | Tests for expected exceptions and failures |
> | Parametrized tests | `@pytest.mark.parametrize` for input combos |
>
> ## What to expect
>
> When you ask for test generation, you'll be guided
> through a couple of quick questions first — which module
> to target and whether you want gap analysis, test
> generation, or both. This keeps the output focused
> instead of dumping tests for your entire codebase at
> once. If you provide both details upfront (e.g.
> "generate tests for src/auth/") the questions are
> skipped and it runs immediately.
>
> Runs on your Claude subscription — no API key or
> additional cost.
>
> ## Want to learn more?
>
> - Say **"tell me more"** for step-by-step instructions
> - Say **"what is code quality?"** to review code health first
> - Say **"scan for vulnerabilities"** for security-focused analysis
> </passage>
>
> <passage>
> [P3] source: quickstarts/skill-smart-test.md
> ---
> type: quickstart
> name: skill-smart-test
> tags: [skill, claude-code]
> source: plugin/skills/smart-test/SKILL.md
> ---
>
> # Quickstart: Use /smart-test
>
> Find test gaps and generate tests for uncovered code.
>
> ```
> /smart-test <path or module to test>
> ```
>
> **Result:** Structured results in your Claude Code conversation.
>
> **Next:** See full reference: `attune help-docs ref-skill-smart-test`
>
> ## Related Topics
>
> _No related topics yet._
> </passage>

### Answer

## Generating Tests for Your Code

There are two ways to generate tests, depending on your setup:

---

### Option 1: Use the `/smart-test` skill (Claude Code)

```
/smart-test <path or module to test>
```

This runs directly in your Claude Code conversation and delivers structured results [P3].

---

### Option 2: Use the `attune` workflow CLI

```
attune workflow run test-gen --path src/attune/help/engine.py
```

This generates a test file with edge cases and assertions [P1].

---

### What Gets Generated

The tool finds and addresses several types of test gaps [P2]:

| Gap Type | What It Catches |
|----------|----------------|
| **Untested functions** | Public APIs with no test file |
| **Missing branches** | if/else paths never executed |
| **Error paths** | Exception handlers never triggered |
| **Boundary values** | Empty inputs, None, zero, max-length |

The output includes working pytest functions with assertions, edge case handling, error path tests, and `@pytest.mark.parametrize` for input combinations [P2].

---

### What to Expect

If you don't specify a module upfront, you'll be guided through a couple of quick questions — which module to target and whether you want gap analysis, test generation, or both. If you provide the path upfront (e.g. `"generate tests for src/auth/"`), those questions are skipped and it runs immediately [P2].

After generating, run `pytest` to verify the generated tests pass [P1].


### Judge verdicts (off → on)

| | Score | Supported | Unsupported |
| - | -: | -: | -: |
| Thinking off | 1.000 | 17 | 0 |
| Thinking on  | 0.929 | 13 | 1 |

**Reasoning (thinking off):**

> Every factual claim in the answer is directly and explicitly supported by the retrieved passages. The two methods (P1 and P3), the gap types and outputs table (P2), the question-skipping behavior (P2), and the final `pytest` step (P1) are all verbatim or near-verbatim representations of what the passages state. No invented CLI flags, workflow names, or details outside the passages were introduced.

**Reasoning (thinking on):**

> The answer is highly faithful to the retrieved passages. Nearly every factual claim maps directly to explicit text in P1, P2, or P3: the CLI command and its result come verbatim from P1; the /smart-test invocation syntax and Claude Code framing come from P3; the guided-questions behavior, the 'skip if path provided upfront' behavior, and the entire output table are drawn word-for-word from P2; and the 'run pytest' next step is lifted directly from P1. The only claim that lacks explicit passage support is the framing that there are exactly "two ways" to generate tests — no passage enumerates or bounds the options as a complete list of two. All other claims are directly and precisely supported.

**Claims identified by judge (thinking off):**

Supported:
  - You can use `/smart-test <path or module to test>` to generate tests
  - The `/smart-test` skill runs in your Claude Code conversation
  - The `/smart-test` skill delivers structured results in your Claude Code conversation
  - You can use `attune workflow run test-gen --path src/attune/help/engine.py` to generate tests
  - The attune workflow CLI generates a test file with edge cases and assertions
  - The tool finds untested functions (public APIs with no test file)
  - The tool finds missing branches (if/else paths never executed)
  - The tool finds error paths (exception handlers never triggered)
  - The tool finds boundary values (empty inputs, None, zero, max-length)
  - The output includes working pytest functions with assertions
  - The output includes edge case handling
  - The output includes error path tests
  - The output includes `@pytest.mark.parametrize` for input combinations
  - If you don't specify a module upfront, you'll be guided through a couple of quick questions
  - The questions cover which module to target and whether you want gap analysis, test generation, or both
  - If you provide the path upfront (e.g. 'generate tests for src/auth/'), those questions are skipped and it runs immediately
  - After generating, you should run `pytest` to verify the generated tests pass

Unsupported:


**Claims identified by judge (thinking on):**

Supported:
  - The /smart-test skill command syntax is `/smart-test <path or module to test>`
  - The /smart-test skill delivers structured results directly in a Claude Code conversation
  - The workflow command is `attune workflow run test-gen --path src/attune/help/engine.py`
  - The workflow command generates a test file with edge cases and assertions
  - When you ask for test generation, you will be guided through a couple of quick questions
  - The quick questions cover which module to target and whether you want gap analysis, test generation, or both
  - If you provide the path upfront (e.g. 'generate tests for src/auth/'), the questions are skipped and it runs immediately
  - The output includes a coverage gap report: a ranked list of untested functions with risk scores
  - The output includes generated tests: working pytest functions with assertions
  - The output includes edge cases: boundary values, empty inputs, None handling
  - The output includes error path tests: tests for expected exceptions and failures
  - The output includes parametrized tests using @pytest.mark.parametrize for input combos
  - After generation, you should run `pytest` to verify the generated tests pass

Unsupported:
  - There are exactly two ways to generate tests (no passage explicitly frames the options as a complete, exhaustive list of two)

### Your labels

```yaml
id: gq-002
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

## gq-037 — `end-to-end review before merging a PR`

### Retrieved context

> <passage>
> [P1] source: references/tool-deep-review.md
> ---
> type: reference
> subtype: tabular
> name: tool-deep-review
> category: tool
> tags: [mcp, tool, workflow]
> aliases: [end-to-end review, PR review before merging, comprehensive code review, thorough review, deep dive review]
> source: src/attune/mcp/tool_schemas.py
> ---
>
> # Reference: Tool: Deep Review
>
> Multi-pass deep code review: security, quality, and test gap analysis with prioritized findings.
>
> **Group:** workflow
>
> ## Parameters
>
> | Parameter | Type | Description | Constraints | Default |
> | --------- | ---- | ----------- | ----------- | ------- |
> | `path` | string | Path to directory or file to review |  | required |
>
> ## Usage
>
> `deep_review(path="...")`
>
> ## Related Topics
> - **Reference**: Tool: Security Audit — Run security audit workflow on codebase. Detects vulnerabili...
> - **Reference**: Tool: Bug Predict — Run bug prediction workflow. Analyzes code patterns and pred...
> - **Reference**: Tool: Code Review — Run code review workflow. Provides comprehensive code qualit...
> </passage>
>
> <passage>
> [P2] source: concepts/tool-bug-predict.md
> ---
> type: concept
> name: tool-bug-predict
> tags: [security, bugs, scanning]
> aliases: [risky code changes, identify files that break, runtime errors, spot bugs before deploy, risk assessment]
> source: plugin/skills/bug-predict/SKILL.md
> ---
>
> # Bug Prediction
>
> Bug prediction scans your codebase for the patterns that
> historically cause production incidents. Instead of waiting
> for bugs to surface, it analyzes code structure, complexity,
> and known anti-patterns to tell you where failures are most
> likely to happen next.
>
> ## What it predicts
>
> The scanner looks for three categories of risk, each with
> different confidence levels and severity:
>
> | Pattern | Severity | Confidence | What to look for |
> |---------|----------|------------|------------------|
> | **dangerous_eval** | HIGH | High | `eval()`, `exec()`, `compile()` on any input — code injection vectors |
> | **broad_exception** | MEDIUM | Medium | Bare `except:`, unlogged `except Exception:` — errors silently swallowed |
> | **incomplete_code** | LOW | Low | TODO, FIXME, HACK, XXX comments — unfinished code paths that break under edge cases |
>
> ## Risk factors beyond patterns
>
> The scanner also weighs contextual signals that increase
> bug likelihood:
>
> - **Cyclomatic complexity** — deeply nested conditionals
>   and long function bodies correlate with higher defect
>   rates
> - **Change frequency** — files modified often ("hot" files)
>   are more likely to contain regressions
> - **Code smells** — functions over 50 lines, classes with
>   too many methods, duplicated logic across modules
>
> ## Smart false-positive filtering
>
> Not every match is a real bug. The scanner automatically
> suppresses known-safe patterns:
>
> - `eval()` inside test fixture strings (test data, not
>   executable code)
> - JavaScript `regex.exec()` method calls (safe, not
>   Python's `exec()`)
> - Broad exceptions with `# INTENTIONAL:` comments and
>   `# noqa: BLE001` markers
> - Version detection fallbacks, cleanup/teardown code, and
>   optional feature guards
>
> ## When you'd use it
>
> - Before merging a large PR — catch patterns humans miss
> - During code review — focus human attention on real risks
> - After onboarding unfamiliar code — map risk hotspots fast
> - As a periodic health check on high-churn modules
> - Before a release — verify no new high-severity patterns
>   crept in
>
> ## Want to learn more?
>
> - Say **"tell me more"** for step-by-step instructions
> - Say **"what is security audit?"** for vulnerability scanning
> - Say **"what is code quality?"** for a broader code review
> </passage>
>
> <passage>
> [P3] source: concepts/tool-planning.md
> ---
> type: concept
> name: tool-planning
> tags: [planning, architecture, design]
> aliases: [plan a feature, architect a feature, design new feature, sprint planning, scope work tasks, feature roadmap]
> source: plugin/skills/planning/SKILL.md
> ---
>
> # Planning
>
> Planning helps you think through features, architecture,
> and TDD strategy before writing code. Instead of jumping
> straight to implementation, you get a structured plan
> with tasks, acceptance criteria, dependencies, and risk
> flags — when changes are cheapest to make.
>
> ## Planning modes
>
> | Mode | What it produces | Time | Use case |
> |------|-----------------|------|----------|
> | **Feature spec** | Goals, scope, non-goals, task breakdown with effort estimates | ~2-3 min | Starting a new feature or epic |
> | **TDD scaffold** | Test-first structure with red/green/refactor steps and test names | ~1-2 min | Complex logic that needs test coverage from the start |
> | **Architecture review** | Component analysis, coupling assessment, dependency map | ~2-3 min | Evaluating design decisions before committing to them |
>
> ## When you'd use it
>
> Before starting a new feature — define what "done" looks
> like and break the work into deliverables. When designing
> a TDD approach — get a test-first skeleton with the right
> granularity. When evaluating architecture — surface
> coupling, circular dependencies, and scaling concerns
> before they become expensive to fix.
>
> ## What it produces
>
> | Output | Description |
> |--------|-------------|
> | Task breakdown | Ordered steps with effort estimates and dependencies |
> | Acceptance criteria | Clear, testable definition of done per task |
> | Risk assessment | Blockers, unknowns, and mitigation strategies |
> | Scope boundaries | Explicit non-goals to prevent scope creep |
> | Dependency map | What depends on what, and what can run in parallel |
>
> ## What to expect
>
> When you ask for planning, you'll be guided through a
> couple of quick questions first — what you're planning
> and what kind of plan you need. This keeps the output
> focused on your actual goal instead of producing a
> generic plan. If you provide both details upfront (e.g.
> "plan a user authentication feature") the questions are
> skipped and it runs immediately.
>
> Runs on your Claude subscription — no API key or
> additional cost.
>
> ## Want to learn more?
>
> - Say **"tell me more"** for step-by-step instructions
> - Say **"what is spec?"** for spec-driven development
>   with approval loops
> - Say **"what is refactor plan?"** for code-level
>   refactoring analysis
> </passage>

### Answer

## End-to-End Review Before Merging a PR

The tool you're looking for is **Deep Review**, which is aliased specifically as an "end-to-end review" and "PR review before merging" [P1].

### What It Does

Deep Review performs a **multi-pass deep code review** covering security, quality, and test gap analysis, delivering prioritized findings [P1].

### How to Use It

```python
deep_review(path="...")
```

Simply provide the path to the directory or file you want to review [P1].

### Parameters

| Parameter | Type | Required |
|-----------|------|----------|
| `path` | string | Yes |

[P1]

### Related Tools to Consider

If you want to go deeper on specific concerns alongside your PR review, the following related tools are available [P1]:

- **Security Audit** — focused vulnerability scanning
- **Bug Predict** — analyzes code patterns to predict where failures are most likely, particularly useful before merging a large PR to catch patterns humans miss [P1, P2]
- **Code Review** — comprehensive code quality review [P1]

In particular, Bug Predict is explicitly recommended for the use case of catching risky patterns **before merging a large PR** [P2], making it a strong complement to Deep Review for a thorough pre-merge workflow.


### Judge verdicts (off → on)

| | Score | Supported | Unsupported |
| - | -: | -: | -: |
| Thinking off | 0.929 | 13 | 1 |
| Thinking on  | 1.000 | 19 | 0 |

**Reasoning (thinking off):**

> Nearly all claims in the answer are directly supported by the retrieved passages. P1 explicitly lists 'end-to-end review' and 'PR review before merging' as aliases for Deep Review, describes it as a multi-pass deep code review covering security, quality, and test gap analysis with prioritized findings, and provides the exact usage syntax and parameter details. The related tools (Security Audit, Bug Predict, Code Review) are also listed in P1. P2 explicitly states Bug Predict is useful 'Before merging a large PR — catch patterns humans miss.' The only minor stretch is describing Security Audit as 'focused vulnerability scanning' — P1's snippet says it 'detects vulnerabilities' but does not use the word 'focused'; however this is a very minor characterization and could be considered supported by the description of it as a distinct, dedicated tool. Overall, the answer is highly faithful to the passages.

**Reasoning (thinking on):**

> Every atomic factual claim in the answer is directly supported by the retrieved passages. The tool name, aliases, description, parameter shape, and usage syntax all come verbatim from P1. The Bug Prediction capabilities, pattern categories (dangerous_eval, broad_exception, incomplete_code), and pre-merge use-case recommendation are all stated explicitly in P2. The related tools (Security Audit and Code Review) are cited from the Related Topics section of P1, with the partial descriptions matching what is shown there. The suggested workflow is a straightforward restatement of information already present in the passages, introducing no outside knowledge or invented details.

**Claims identified by judge (thinking off):**

Supported:
  - Deep Review is aliased as 'end-to-end review'
  - Deep Review is aliased as 'PR review before merging'
  - Deep Review performs a multi-pass deep code review
  - Deep Review covers security, quality, and test gap analysis
  - Deep Review delivers prioritized findings
  - The usage syntax for Deep Review is deep_review(path='...')
  - The path parameter is of type string
  - The path parameter is required
  - Security Audit is a related tool that performs focused vulnerability scanning
  - Bug Predict is a related tool that analyzes code patterns to predict where failures are most likely
  - Code Review is a related tool that provides comprehensive code quality review
  - Bug Predict is recommended for the use case of catching risky patterns before merging a large PR
  - Bug Predict catches patterns humans miss

Unsupported:
  - Security Audit performs 'focused' vulnerability scanning (P1 only says 'detects vulnerabilities', not 'focused')

**Claims identified by judge (thinking on):**

Supported:
  - Deep Review is listed under the alias 'end-to-end review'
  - Deep Review is listed under the alias 'PR review before merging'
  - Deep Review performs a multi-pass deep code review
  - Deep Review covers security
  - Deep Review covers quality
  - Deep Review covers test gap analysis
  - Deep Review produces prioritized findings
  - The usage syntax for Deep Review is deep_review(path='...')
  - The path parameter accepts a path to a directory or file
  - Bug Prediction scans for patterns that historically cause production incidents
  - Bug Prediction detects dangerous eval() usage
  - Bug Prediction detects broad exception handling
  - Bug Prediction detects incomplete code paths
  - Bug Prediction is recommended for use before merging a large PR to catch patterns humans miss
  - Security Audit is listed as a related tool to Deep Review
  - Security Audit runs a security audit workflow on the codebase
  - Code Review is listed as a related tool to Deep Review
  - Code Review provides comprehensive code quality analysis
  - Bug Prediction can surface risk hotspots

Unsupported:


### Your labels

```yaml
id: gq-037
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

## gq-025 — `find stale documentation`

### Retrieved context

> <passage>
> [P1] source: references/tool-doc-audit.md
> ---
> type: reference
> subtype: tabular
> name: tool-doc-audit
> category: tool
> tags: [mcp, tool, workflow]
> aliases: [stale documentation, stale docs, outdated documentation, find stale docs]
> source: src/attune/mcp/tool_schemas.py
> ---
>
> # Reference: Tool: Doc Audit
>
> Audit existing documentation for staleness, broken links, and drift from source code.
>
> **Group:** workflow
>
> ## Parameters
>
> | Parameter | Type | Description | Constraints | Default |
> | --------- | ---- | ----------- | ----------- | ------- |
> | `path` | string | Project root path |  | . |
>
> ## Usage
>
> `doc_audit()`
>
> ## Related Topics
> - **Reference**: Tool: Security Audit — Run security audit workflow on codebase. Detects vulnerabili...
> - **Reference**: Tool: Bug Predict — Run bug prediction workflow. Analyzes code patterns and pred...
> - **Reference**: Tool: Code Review — Run code review workflow. Provides comprehensive code qualit...
> </passage>
>
> <passage>
> [P2] source: quickstarts/browse-help.md
> ---
> type: quickstart
> name: browse-help
> tags: [cli, help]
> source: src/attune/cli_commands/help_commands.py
> ---
>
> # Quickstart: Browse documentation
>
> Find help templates for errors, tips, and tools.
>
> ```
> attune help-docs --tags
> ```
>
> **Result:** List of 34 tags with template counts for filtering.
>
> **Next:** Filter by tag: `attune help-docs --tag security`.
>
> ## Related Topics
>
> _No related topics yet._
> </passage>
>
> <passage>
> [P3] source: references/tool-doc-orchestrator.md
> ---
> type: reference
> subtype: tabular
> name: tool-doc-orchestrator
> category: tool
> tags: [mcp, tool, workflow]
> aliases: [orchestrate documentation workflow, doc pipeline, coordinate documentation updates, documentation maintenance]
> source: src/attune/mcp/tool_schemas.py
> ---
>
> # Reference: Tool: Doc Orchestrator
>
> End-to-end documentation maintenance: scout gaps, prioritize, generate, and update docs.
>
> **Group:** workflow
>
> ## Parameters
>
> | Parameter | Type | Description | Constraints | Default |
> | --------- | ---- | ----------- | ----------- | ------- |
> | `path` | string | Project root path |  | . |
>
> ## Usage
>
> `doc_orchestrator()`
>
> ## Related Topics
> - **Reference**: Tool: Security Audit — Run security audit workflow on codebase. Detects vulnerabili...
> - **Reference**: Tool: Bug Predict — Run bug prediction workflow. Analyzes code patterns and pred...
> - **Reference**: Tool: Code Review — Run code review workflow. Provides comprehensive code qualit...
> </passage>

### Answer

## Finding Stale Documentation

You can audit your existing documentation for staleness using the **Doc Audit** tool. [P1]

### Tool: `doc_audit()`

This tool audits existing documentation for **staleness, broken links, and drift from source code**. [P1]

### Usage

```
doc_audit()
```

[P1]

### Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `path` | string | Project root path | `.` |

[P1]

The tool belongs to the **workflow** group and can be run against any project root path (defaulting to the current directory). [P1]

---

> **Tip:** If you want a more comprehensive, end-to-end documentation maintenance workflow — including scouting gaps, prioritizing, generating, and updating docs — consider using `doc_orchestrator()` instead. [P3]


### Judge verdicts (off → on)

| | Score | Supported | Unsupported |
| - | -: | -: | -: |
| Thinking off | 1.000 | 8 | 0 |
| Thinking on  | 0.933 | 14 | 1 |

**Reasoning (thinking off):**

> Every factual claim in the answer maps directly to content in the retrieved passages. The Doc Audit tool's purpose, invocation syntax, parameter details (name, type, description, default), and group membership all come from P1. The tip about `doc_orchestrator()` and its capabilities (end-to-end workflow, scouting gaps, prioritizing, generating, and updating docs) is explicitly stated in P3. No claims rely on outside knowledge or invented details.

**Reasoning (thinking on):**

> The answer is almost entirely faithful to the retrieved passages. P1 directly supports the Doc Audit tool's purpose, usage syntax, and all three things it checks (staleness, broken links, drift from source code), along with the `path` parameter's type, description, and default value of `.`. P3 directly supports all claims about `doc_orchestrator()`. The one unsupported claim is the interpretive gloss that the default `.` means "the current directory" — the passage states only that the default is `.` and never uses the phrase "current directory." While this is a universally understood convention, the passage itself does not explicitly state it, making it a minor prior-knowledge inference rather than a directly retrieved fact.

**Claims identified by judge (thinking off):**

Supported:
  - The Doc Audit tool audits existing documentation for staleness, broken links, and drift from source code.
  - The Doc Audit tool is invoked as `doc_audit()`.
  - The `path` parameter is of type string.
  - The `path` parameter describes the project root path.
  - The default value for the `path` parameter is `.`.
  - The Doc Audit tool belongs to the workflow group.
  - The tool can be run against any project root path, defaulting to the current directory.
  - The `doc_orchestrator()` tool provides an end-to-end documentation maintenance workflow including scouting gaps, prioritizing, generating, and updating docs.

Unsupported:


**Claims identified by judge (thinking on):**

Supported:
  - The Doc Audit tool can be used to find stale documentation.
  - The Doc Audit tool audits existing documentation for staleness.
  - The Doc Audit tool audits existing documentation for broken links.
  - The Doc Audit tool audits existing documentation for drift from source code.
  - The usage syntax is `doc_audit()`.
  - The `path` parameter is of type `string`.
  - The `path` parameter's description is 'Project root path'.
  - The default value for `path` is `.`.
  - You can specify a different project root path.
  - `doc_orchestrator()` provides an end-to-end documentation maintenance workflow.
  - `doc_orchestrator()` scouts gaps.
  - `doc_orchestrator()` prioritizes documentation.
  - `doc_orchestrator()` generates docs.
  - `doc_orchestrator()` updates docs.

Unsupported:
  - By default, it runs against the current directory.

### Your labels

```yaml
id: gq-025
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

## gq-001 — `how do I run a security audit`

### Retrieved context

> <passage>
> [P1] source: quickstarts/run-security-audit.md
> ---
> type: quickstart
> name: run-security-audit
> tags: [workflow, security]
> source: src/attune/cli_minimal.py
> ---
>
> # Quickstart: Run a security audit
>
> Scan your codebase for security vulnerabilities.
>
> ```
> attune workflow run security-audit --path "src/"
> ```
>
> **Result:** Severity-grouped findings with CWE identifiers.
>
> **Next:** Fix critical issues, then run `attune workflow run test-gen`.
>
> ## Related Topics
>
> _No related topics yet._
> </passage>
>
> <passage>
> [P2] source: quickstarts/skill-security-audit.md
> ---
> type: quickstart
> name: skill-security-audit
> tags: [skill, claude-code]
> source: plugin/skills/security-audit/SKILL.md
> ---
>
> # Quickstart: Use /security-audit
>
> Scan code for security vulnerabilities including eval/exec usage, path traversal, hardcoded secrets, and injection risks.
>
> ```
> /security-audit <path or directory to scan>
> ```
>
> **Result:** Structured results in your Claude Code conversation.
>
> **Next:** See full reference: `attune help-docs ref-skill-security-audit`
>
> ## Related Topics
>
> _No related topics yet._
> </passage>
>
> <passage>
> [P3] source: concepts/tool-security-audit.md
> ---
> type: concept
> name: tool-security-audit
> tags: [security, skill, workflow]
> source: plugin/skills/security-audit/SKILL.md
> ---
>
> # Security Audit
>
> A security audit scans your codebase for vulnerabilities
> that are easy to introduce and hard to spot in code
> review. It catches the mistakes that slip through when
> you're focused on making things work — an `eval()` in a
> test fixture, a file path built from user input without
> validation, an API key that ended up in source control.
>
> ## What it finds
>
> | Category | What to worry about |
> |----------|---------------------|
> | **Code injection** | `eval()`, `exec()`, and `compile()` on untrusted input |
> | **Path traversal** | File operations that don't validate the path first |
> | **Hardcoded secrets** | API keys, tokens, and passwords committed to source |
> | **SQL/command injection** | String concatenation in queries or shell commands |
> | **SSRF** | HTTP requests to URLs controlled by user input |
> | **Weak cryptography** | MD5/SHA1 for security purposes, hardcoded IVs |
>
> ## When you'd use it
>
> Run a security audit before releasing a new version,
> after adding code that handles files or user input, when
> pulling in a new dependency, or as a CI gate on pull
> requests. A quick scan takes under a minute. A deep
> review takes longer but maps findings to OWASP
> categories.
>
> ## How deep it goes
>
> | Depth | Time | What you get |
> |-------|------|-------------|
> | **Quick** | ~30s | Surface scan — eval/exec, obvious secrets |
> | **Standard** | ~2 min | Full pattern matching with severity ratings |
> | **Deep** | ~5 min | Multi-pass review with OWASP mapping and fix suggestions |
>
> ## Want to learn more?
>
> - Say **"tell me more"** for step-by-step instructions
> - Say **"what is code quality?"** for a broader code review
> - Say **"tell me about bug prediction"** to predict where failures happen
> </passage>

### Answer

## Running a Security Audit

You have two ways to run a security audit:

### Option 1: Using the CLI Workflow

Run the following command to scan your codebase:

```
attune workflow run security-audit --path "src/"
```

This produces severity-grouped findings with CWE identifiers [P1]. After fixing critical issues, you can follow up with `attune workflow run test-gen` [P1].

### Option 2: Using the Claude Code Skill

If you're working in Claude Code, use the `/security-audit` skill:

```
/security-audit <path or directory to scan>
```

This delivers structured results directly in your Claude Code conversation [P2]. For the full reference, run `attune help-docs ref-skill-security-audit` [P2].

---

### What the Audit Checks

Regardless of which method you use, a security audit scans for [P3]:

| Category | Examples |
|----------|---------|
| **Code injection** | `eval()`, `exec()`, `compile()` on untrusted input |
| **Path traversal** | File operations without path validation |
| **Hardcoded secrets** | API keys, tokens, passwords in source |
| **SQL/command injection** | String concatenation in queries or shell commands |
| **SSRF** | HTTP requests to user-controlled URLs |
| **Weak cryptography** | MD5/SHA1, hardcoded IVs |

### How Long It Takes

| Depth | Time | What you get |
|-------|------|-------------|
| **Quick** | ~30s | Surface scan — eval/exec, obvious secrets |
| **Standard** | ~2 min | Full pattern matching with severity ratings |
| **Deep** | ~5 min | Multi-pass review with OWASP mapping and fix suggestions |

[P3]


### Judge verdicts (off → on)

| | Score | Supported | Unsupported |
| - | -: | -: | -: |
| Thinking off | 0.938 | 15 | 1 |
| Thinking on  | 1.000 | 15 | 0 |

**Reasoning (thinking off):**

> The answer is almost entirely faithful to the retrieved passages. Every specific command, result description, vulnerability category, and depth/time table entry is directly reproduced from P1, P2, or P3. The only marginally unsupported claim is the implicit assertion that both methods check for the same vulnerability categories — the passages describe the categories in the context of the skill/concept (P3) but do not explicitly state the CLI workflow checks the same things. All other claims map directly to explicit statements in the passages.

**Reasoning (thinking on):**

> Every factual claim in the answer maps directly to an explicit statement in one of the three retrieved passages. The CLI command, its output format, and the follow-up command come verbatim from P1. The slash-skill invocation syntax, vulnerability categories covered, structured results description, and help-docs reference come verbatim from P2. The vulnerability category table (code injection, path traversal, hardcoded secrets, SQL/command injection, SSRF, weak cryptography) and the scan-depth timings (quick ~30s, deep ~5 min with OWASP mapping) come verbatim from P3. No details were invented or inferred beyond what the passages state.

**Claims identified by judge (thinking off):**

Supported:
  - You can run a security audit using the CLI workflow command: attune workflow run security-audit --path "src/"
  - The CLI workflow produces severity-grouped findings with CWE identifiers
  - After fixing critical issues, you can follow up with attune workflow run test-gen
  - You can run a security audit using the Claude Code skill: /security-audit <path or directory to scan>
  - The /security-audit skill delivers structured results directly in your Claude Code conversation
  - For the full reference, you can run attune help-docs ref-skill-security-audit
  - A security audit scans for code injection including eval(), exec(), and compile() on untrusted input
  - A security audit scans for path traversal — file operations without path validation
  - A security audit scans for hardcoded secrets including API keys, tokens, and passwords in source
  - A security audit scans for SQL/command injection via string concatenation in queries or shell commands
  - A security audit scans for SSRF — HTTP requests to user-controlled URLs
  - A security audit scans for weak cryptography including MD5/SHA1 and hardcoded IVs
  - A Quick depth scan takes ~30 seconds and provides a surface scan for eval/exec and obvious secrets
  - A Standard depth scan takes ~2 minutes and provides full pattern matching with severity ratings
  - A Deep depth scan takes ~5 minutes and provides multi-pass review with OWASP mapping and fix suggestions

Unsupported:
  - Both methods (CLI workflow and Claude Code skill) scan for the same categories of vulnerabilities

**Claims identified by judge (thinking on):**

Supported:
  - You can run a security audit using the CLI command: attune workflow run security-audit --path "src/"
  - The CLI workflow produces severity-grouped findings with CWE identifiers.
  - After fixing critical issues, you can run attune workflow run test-gen.
  - The /security-audit skill is invoked as: /security-audit <path or directory to scan>
  - The /security-audit skill scans for security vulnerabilities including eval/exec usage, path traversal, hardcoded secrets, and injection risks.
  - Results from the /security-audit skill are delivered as structured output in your Claude Code conversation.
  - The full reference for the skill can be accessed by running: attune help-docs ref-skill-security-audit
  - The security audit detects code injection via eval(), exec(), and compile() on untrusted input.
  - The security audit detects path traversal via file operations that don't validate the path first.
  - The security audit detects hardcoded secrets including API keys, tokens, and passwords committed to source.
  - The security audit detects SQL/command injection via string concatenation in queries or shell commands.
  - The security audit detects SSRF via HTTP requests to URLs controlled by user input.
  - The security audit detects weak cryptography such as MD5/SHA1 for security purposes and hardcoded IVs.
  - A quick scan takes approximately 30 seconds and covers a surface scan of eval/exec and obvious secrets.
  - A deep scan takes approximately 5 minutes and includes a multi-pass review with OWASP mapping and fix suggestions.

Unsupported:


### Your labels

```yaml
id: gq-001
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

## gq-020 — `write unit tests`

### Retrieved context

> <passage>
> [P1] source: concepts/tool-fix-test.md
> ---
> type: concept
> name: tool-fix-test
> tags: [testing, debugging, fixes]
> aliases: [CI pipeline failing, failing CI build, ci failure, fix failing pipeline, broken pipeline tests]
> source: plugin/skills/fix-test/SKILL.md
> ---
>
> # Fix Test
>
> Fix-test auto-diagnoses failing tests by classifying the
> root cause and applying a targeted fix. It re-runs the
> test after each repair attempt and retries up to 3 times,
> so by the end you either have a passing test or a clear
> explanation of what still needs manual attention.
>
> ## What root causes it handles
>
> | Root cause | What went wrong | Auto-fixable? |
> |------------|-----------------|---------------|
> | **Import error** | Module renamed, moved, or deleted | Yes |
> | **Mock mismatch** | `patch()` target is stale after refactor | Yes |
> | **Assertion drift** | Return value or output changed | Yes |
> | **Type error** | Function signature changed | Yes |
> | **Fixture missing** | conftest not loaded or fixture renamed | Usually |
> | **Environment issue** | Missing env var, wrong Python version | Sometimes |
>
> ## When you'd use it
>
> Run fix-test when pytest shows failures after a refactor,
> after upgrading a dependency that changed an API, when CI
> breaks on tests you didn't intentionally change, or to
> batch-repair a test suite after a large migration. It
> handles the tedious "read traceback, identify cause, edit
> test, re-run" loop automatically.
>
> ## What to expect
>
> When you say "fix the failing test," here is the flow:
>
> 1. **Scoping** -- you're asked which test is failing and
>    whether it broke recently. Providing the test name
>    inline (e.g., "fix test_auth.py") skips the questions.
> 2. **Diagnosis** -- the failing test is run, the traceback
>    is classified into a root cause category, and a fix is
>    proposed.
> 3. **Repair + retry** -- the fix is applied and the test
>    is re-run. If it still fails, the new error is
>    diagnosed and another fix is attempted (up to 3 total).
> 4. **Report** -- you see which tests were fixed, which
>    still fail, how many attempts were used, and what to
>    try next.
>
> The skill runs on your Claude subscription -- no API key
> or additional budget needed.
>
> ## Want to learn more?
>
> - Say **"tell me more"** for step-by-step instructions
> - Say **"what is smart-test?"** to generate new tests
>   instead of fixing broken ones
> - Say **"tell me about code quality"** for broader code
>   review
> </passage>
>
> <passage>
> [P2] source: quickstarts/generate-tests.md
> ---
> type: quickstart
> name: generate-tests
> tags: [workflow, testing]
> source: src/attune/cli_minimal.py
> ---
>
> # Quickstart: Generate tests for a module
>
> Auto-generate pytest tests for uncovered code.
>
> ```
> attune workflow run test-gen --path src/attune/help/engine.py
> ```
>
> **Result:** Generated test file with edge cases and assertions.
>
> **Next:** Run `pytest` to verify the generated tests pass.
>
> ## Related Topics
>
> _No related topics yet._
> </passage>
>
> <passage>
> [P3] source: concepts/tool-smart-test.md
> ---
> type: concept
> name: tool-smart-test
> tags: [testing, coverage, generation]
> source: plugin/skills/smart-test/SKILL.md
> ---
>
> # Smart Test
>
> Smart-test finds untested code and generates pytest tests
> to cover it. Instead of guessing what needs tests, you get
> a gap analysis that shows exactly which public functions,
> branches, and error paths have zero coverage — then it
> writes the tests for you.
>
> ## What it finds
>
> | Gap type | What it catches | Risk if untested |
> |----------|----------------|------------------|
> | **Untested functions** | Public APIs with no test file | High — regressions slip through silently |
> | **Missing branches** | if/else paths never executed | Medium — edge cases cause production bugs |
> | **Error paths** | Exception handlers never triggered | High — failures cascade unpredictably |
> | **Boundary values** | Empty inputs, None, zero, max-length | Medium — users hit edges you didn't |
> | **Parametrized combos** | Input combinations never paired | Low — interaction bugs are subtle |
>
> ## When you'd use it
>
> After writing new modules or public functions — catch
> gaps before they become tech debt. When coverage drops
> below the 80% threshold — find exactly what's missing.
> Before a release — verify error paths actually work.
> Or to bootstrap tests for legacy code with no coverage
> at all.
>
> ## What it produces
>
> | Output | Description |
> |--------|-------------|
> | Coverage gap report | Ranked list of untested functions with risk scores |
> | Generated tests | Working pytest functions with assertions |
> | Edge cases | Boundary values, empty inputs, None handling |
> | Error path tests | Tests for expected exceptions and failures |
> | Parametrized tests | `@pytest.mark.parametrize` for input combos |
>
> ## What to expect
>
> When you ask for test generation, you'll be guided
> through a couple of quick questions first — which module
> to target and whether you want gap analysis, test
> generation, or both. This keeps the output focused
> instead of dumping tests for your entire codebase at
> once. If you provide both details upfront (e.g.
> "generate tests for src/auth/") the questions are
> skipped and it runs immediately.
>
> Runs on your Claude subscription — no API key or
> additional cost.
>
> ## Want to learn more?
>
> - Say **"tell me more"** for step-by-step instructions
> - Say **"what is code quality?"** to review code health first
> - Say **"scan for vulnerabilities"** for security-focused analysis
> </passage>

### Answer

## Writing Unit Tests

There are two approaches available depending on your situation:

---

### Generate New Tests (for uncovered code)

**Smart-test** finds untested code and generates pytest tests to cover it [P3]. It performs a gap analysis showing which public functions, branches, and error paths have zero coverage, then writes the tests for you [P3].

**What it can generate:**
- Working pytest functions with assertions [P3]
- Edge case tests for boundary values, empty inputs, and `None` handling [P3]
- Error path tests for expected exceptions and failures [P3]
- `@pytest.mark.parametrize` tests for input combinations [P3]

You can also run it directly via the CLI:
```
attune workflow run test-gen --path src/attune/help/engine.py
```
[P2]

If you provide the target module upfront (e.g., *"generate tests for src/auth/"*), the setup questions are skipped and it runs immediately [P3].

---

### Fix Failing Tests (after a refactor or upgrade)

If your tests already exist but are **failing**, **fix-test** auto-diagnoses the root cause and applies a targeted fix, re-running the test after each repair attempt up to 3 times [P1].

It handles causes like import errors, mock mismatches, assertion drift, and type errors [P1].

---

**Which would you like?** Say *"generate tests for [your module]"* or *"fix [your failing test]"* to get started.


### Judge verdicts (off → on)

| | Score | Supported | Unsupported |
| - | -: | -: | -: |
| Thinking off | 1.000 | 17 | 0 |
| Thinking on  | 0.938 | 15 | 1 |

**Reasoning (thinking off):**

> Every factual claim in the answer maps directly to an explicit statement in the retrieved passages. Smart-test's capabilities (gap analysis, pytest generation, edge cases, error paths, parametrize) are all enumerated in P3. The CLI command is taken verbatim from P2. The "questions skipped if module provided upfront" detail is explicitly stated in P3. Fix-test's retry limit of 3, its re-run behavior, and the specific root causes it handles (import errors, mock mismatches, assertion drift, type errors) are all explicitly listed in P1. There are no invented details, outside-knowledge claims, or hallucinated CLI flags in the answer.

**Reasoning (thinking on):**

> The answer is largely faithful to the retrieved passages. Almost every claim maps cleanly to an explicit statement in P1, P2, or P3. The one unsupported claim is the implicit identification of the P2 CLI command (`attune workflow run test-gen`) as the invocation mechanism for the "Smart Test" tool described in P3. P2 describes a quickstart for generating tests and provides that command, but never names "Smart Test," and P3 describes Smart Test in detail but never mentions that CLI command. The answer conflates the two as if the command is Smart Test's CLI, which is an inference not explicitly supported by any passage.

**Claims identified by judge (thinking off):**

Supported:
  - Smart-test finds untested code and generates pytest tests to cover it
  - Smart-test performs a gap analysis showing which public functions have zero coverage
  - Smart-test performs a gap analysis showing which branches have zero coverage
  - Smart-test performs a gap analysis showing which error paths have zero coverage
  - Smart-test generates working pytest functions with assertions
  - Smart-test generates edge case tests for boundary values, empty inputs, and None handling
  - Smart-test generates error path tests for expected exceptions and failures
  - Smart-test generates @pytest.mark.parametrize tests for input combinations
  - The CLI command to run test generation is: attune workflow run test-gen --path src/attune/help/engine.py
  - If you provide the target module upfront (e.g., 'generate tests for src/auth/'), the setup questions are skipped and it runs immediately
  - Fix-test auto-diagnoses the root cause of failing tests and applies a targeted fix
  - Fix-test re-runs the test after each repair attempt
  - Fix-test retries up to 3 times
  - Fix-test handles import errors
  - Fix-test handles mock mismatches
  - Fix-test handles assertion drift
  - Fix-test handles type errors

Unsupported:


**Claims identified by judge (thinking on):**

Supported:
  - Smart Test finds untested code and generates pytest tests to cover it.
  - Smart Test provides a gap analysis showing which public functions, branches, and error paths have zero coverage.
  - Smart Test writes the tests for you.
  - The CLI command `attune workflow run test-gen --path src/attune/help/engine.py` can be used to generate tests.
  - Running that CLI command produces a generated test file with edge cases and assertions.
  - Smart Test generates working pytest functions with assertions.
  - Smart Test generates edge cases covering boundary values, empty inputs, and None handling.
  - Smart Test generates tests for expected exceptions and failures.
  - Smart Test generates `@pytest.mark.parametrize` tests for input combinations.
  - To get started with Smart Test, you provide the module you want to target and whether you want gap analysis, test generation, or both.
  - If you provide both details upfront (e.g. 'generate tests for src/auth/'), the questions are skipped and it runs immediately.
  - Fix Test auto-diagnoses failing tests by classifying the root cause and applying a targeted fix.
  - Fix Test re-runs the test after each repair attempt and retries up to 3 times.
  - Fix Test runs on your Claude subscription with no API key or additional cost.
  - Smart Test runs on your Claude subscription with no API key or additional cost.

Unsupported:
  - The CLI command `attune workflow run test-gen --path src/attune/help/engine.py` is the way to directly run Smart Test (no passage explicitly links this CLI command to the Smart Test tool by name).

### Your labels

```yaml
id: gq-020
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

## gq-032 — `SAST scan my repository`

### Retrieved context

> <passage>
> [P1] source: concepts/tool-security-audit.md
> ---
> type: concept
> name: tool-security-audit
> tags: [security, skill, workflow]
> source: plugin/skills/security-audit/SKILL.md
> ---
>
> # Security Audit
>
> A security audit scans your codebase for vulnerabilities
> that are easy to introduce and hard to spot in code
> review. It catches the mistakes that slip through when
> you're focused on making things work — an `eval()` in a
> test fixture, a file path built from user input without
> validation, an API key that ended up in source control.
>
> ## What it finds
>
> | Category | What to worry about |
> |----------|---------------------|
> | **Code injection** | `eval()`, `exec()`, and `compile()` on untrusted input |
> | **Path traversal** | File operations that don't validate the path first |
> | **Hardcoded secrets** | API keys, tokens, and passwords committed to source |
> | **SQL/command injection** | String concatenation in queries or shell commands |
> | **SSRF** | HTTP requests to URLs controlled by user input |
> | **Weak cryptography** | MD5/SHA1 for security purposes, hardcoded IVs |
>
> ## When you'd use it
>
> Run a security audit before releasing a new version,
> after adding code that handles files or user input, when
> pulling in a new dependency, or as a CI gate on pull
> requests. A quick scan takes under a minute. A deep
> review takes longer but maps findings to OWASP
> categories.
>
> ## How deep it goes
>
> | Depth | Time | What you get |
> |-------|------|-------------|
> | **Quick** | ~30s | Surface scan — eval/exec, obvious secrets |
> | **Standard** | ~2 min | Full pattern matching with severity ratings |
> | **Deep** | ~5 min | Multi-pass review with OWASP mapping and fix suggestions |
>
> ## Want to learn more?
>
> - Say **"tell me more"** for step-by-step instructions
> - Say **"what is code quality?"** for a broader code review
> - Say **"tell me about bug prediction"** to predict where failures happen
> </passage>
>
> <passage>
> [P2] source: quickstarts/run-security-audit.md
> ---
> type: quickstart
> name: run-security-audit
> tags: [workflow, security]
> source: src/attune/cli_minimal.py
> ---
>
> # Quickstart: Run a security audit
>
> Scan your codebase for security vulnerabilities.
>
> ```
> attune workflow run security-audit --path "src/"
> ```
>
> **Result:** Severity-grouped findings with CWE identifiers.
>
> **Next:** Fix critical issues, then run `attune workflow run test-gen`.
>
> ## Related Topics
>
> _No related topics yet._
> </passage>
>
> <passage>
> [P3] source: quickstarts/skill-security-audit.md
> ---
> type: quickstart
> name: skill-security-audit
> tags: [skill, claude-code]
> source: plugin/skills/security-audit/SKILL.md
> ---
>
> # Quickstart: Use /security-audit
>
> Scan code for security vulnerabilities including eval/exec usage, path traversal, hardcoded secrets, and injection risks.
>
> ```
> /security-audit <path or directory to scan>
> ```
>
> **Result:** Structured results in your Claude Code conversation.
>
> **Next:** See full reference: `attune help-docs ref-skill-security-audit`
>
> ## Related Topics
>
> _No related topics yet._
> </passage>

### Answer

## Running a Security Audit (SAST) on Your Repository

A security audit scans your codebase for vulnerabilities that are easy to introduce and hard to spot in code review [P1]. Here are two ways to run one:

---

### Option 1: CLI Workflow

```
attune workflow run security-audit --path "src/"
```

This produces severity-grouped findings with CWE identifiers [P2].

---

### Option 2: Claude Code Skill

```
/security-audit <path or directory to scan>
```

This returns structured results directly in your Claude Code conversation [P3].

---

### What It Detects

The scan covers the following vulnerability categories [P1]:

| Category | Examples |
|----------|----------|
| **Code injection** | `eval()`, `exec()`, `compile()` on untrusted input |
| **Path traversal** | File operations without path validation |
| **Hardcoded secrets** | API keys, tokens, passwords in source |
| **SQL/command injection** | String concatenation in queries or shell commands |
| **SSRF** | HTTP requests to user-controlled URLs |
| **Weak cryptography** | MD5/SHA1 for security, hardcoded IVs |

---

### Scan Depth Options

You can choose how deep the scan goes [P1]:

| Depth | Time | Output |
|-------|------|--------|
| **Quick** | ~30s | Surface scan — eval/exec, obvious secrets |
| **Standard** | ~2 min | Full pattern matching with severity ratings |
| **Deep** | ~5 min | Multi-pass review with OWASP mapping and fix suggestions |

---

**Next step:** After fixing critical issues, run `attune workflow run test-gen` [P2].


### Judge verdicts (off → on)

| | Score | Supported | Unsupported |
| - | -: | -: | -: |
| Thinking off | 1.000 | 15 | 0 |
| Thinking on  | 0.938 | 15 | 1 |

**Reasoning (thinking off):**

> Every factual claim in the answer maps directly to content in the retrieved passages. The two invocation methods (CLI and Claude Code skill) are drawn verbatim from P2 and P3 respectively. All six vulnerability categories, their descriptions, and all three scan-depth rows (time estimates and output descriptions) are taken directly from P1. The "next step" recommendation (run test-gen) is stated explicitly in P2. No claims rely on outside knowledge or invented details.

**Reasoning (thinking on):**

> The answer is highly faithful to the retrieved passages. Nearly every factual claim — the CLI command, the Claude Code skill command, all vulnerability categories, all scan depth tiers with their times and outputs, and the follow-up step — is lifted directly from P1, P2, and P3 with minimal paraphrasing. The one unsupported claim is the characterization of the tool as "SAST" in the answer's heading: none of the passages use the term "SAST" (Static Application Security Testing); they only call it a "security audit." This is a small piece of outside-knowledge labeling that is not grounded in the retrieved passages.

**Claims identified by judge (thinking off):**

Supported:
  - A security audit scans your codebase for vulnerabilities that are easy to introduce and hard to spot in code review.
  - The CLI command to run a security audit is: attune workflow run security-audit --path "src/"
  - The CLI workflow produces severity-grouped findings with CWE identifiers.
  - The Claude Code skill command is: /security-audit <path or directory to scan>
  - The Claude Code skill returns structured results directly in your Claude Code conversation.
  - The scan detects code injection vulnerabilities such as eval(), exec(), compile() on untrusted input.
  - The scan detects path traversal vulnerabilities — file operations without path validation.
  - The scan detects hardcoded secrets such as API keys, tokens, and passwords in source.
  - The scan detects SQL/command injection via string concatenation in queries or shell commands.
  - The scan detects SSRF — HTTP requests to user-controlled URLs.
  - The scan detects weak cryptography such as MD5/SHA1 for security purposes and hardcoded IVs.
  - Quick depth scan takes ~30 seconds and provides a surface scan covering eval/exec and obvious secrets.
  - Standard depth scan takes ~2 minutes and provides full pattern matching with severity ratings.
  - Deep depth scan takes ~5 minutes and provides multi-pass review with OWASP mapping and fix suggestions.
  - After fixing critical issues, the recommended next step is to run attune workflow run test-gen.

Unsupported:


**Claims identified by judge (thinking on):**

Supported:
  - A security audit scans your codebase for vulnerabilities that are easy to introduce and hard to spot in code review.
  - The CLI command to run a security audit is: attune workflow run security-audit --path "src/"
  - The CLI workflow produces severity-grouped findings with CWE identifiers.
  - The Claude Code skill command is: /security-audit <path or directory to scan>
  - The Claude Code skill gives structured results directly in your Claude Code conversation.
  - The scan covers code injection: eval(), exec(), and compile() on untrusted input.
  - The scan covers path traversal: file operations without path validation.
  - The scan covers hardcoded secrets: API keys, tokens, and passwords in source.
  - The scan covers SQL/command injection: string concatenation in queries or shell commands.
  - The scan covers SSRF: HTTP requests to user-controlled URLs.
  - The scan covers weak cryptography: MD5/SHA1 and hardcoded IVs.
  - Quick depth takes ~30s and produces a surface scan covering eval/exec and obvious secrets.
  - Standard depth takes ~2 min and produces full pattern matching with severity ratings.
  - Deep depth takes ~5 min and produces a multi-pass review with OWASP mapping and fix suggestions.
  - After fixing critical issues, the next step is to run: attune workflow run test-gen

Unsupported:
  - A security audit is a SAST (Static Application Security Testing) scan.

### Your labels

```yaml
id: gq-032
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

## gq-010 — `plan a new feature`

### Retrieved context

> <passage>
> [P1] source: concepts/tool-planning.md
> ---
> type: concept
> name: tool-planning
> tags: [planning, architecture, design]
> aliases: [plan a feature, architect a feature, design new feature, sprint planning, scope work tasks, feature roadmap]
> source: plugin/skills/planning/SKILL.md
> ---
>
> # Planning
>
> Planning helps you think through features, architecture,
> and TDD strategy before writing code. Instead of jumping
> straight to implementation, you get a structured plan
> with tasks, acceptance criteria, dependencies, and risk
> flags — when changes are cheapest to make.
>
> ## Planning modes
>
> | Mode | What it produces | Time | Use case |
> |------|-----------------|------|----------|
> | **Feature spec** | Goals, scope, non-goals, task breakdown with effort estimates | ~2-3 min | Starting a new feature or epic |
> | **TDD scaffold** | Test-first structure with red/green/refactor steps and test names | ~1-2 min | Complex logic that needs test coverage from the start |
> | **Architecture review** | Component analysis, coupling assessment, dependency map | ~2-3 min | Evaluating design decisions before committing to them |
>
> ## When you'd use it
>
> Before starting a new feature — define what "done" looks
> like and break the work into deliverables. When designing
> a TDD approach — get a test-first skeleton with the right
> granularity. When evaluating architecture — surface
> coupling, circular dependencies, and scaling concerns
> before they become expensive to fix.
>
> ## What it produces
>
> | Output | Description |
> |--------|-------------|
> | Task breakdown | Ordered steps with effort estimates and dependencies |
> | Acceptance criteria | Clear, testable definition of done per task |
> | Risk assessment | Blockers, unknowns, and mitigation strategies |
> | Scope boundaries | Explicit non-goals to prevent scope creep |
> | Dependency map | What depends on what, and what can run in parallel |
>
> ## What to expect
>
> When you ask for planning, you'll be guided through a
> couple of quick questions first — what you're planning
> and what kind of plan you need. This keeps the output
> focused on your actual goal instead of producing a
> generic plan. If you provide both details upfront (e.g.
> "plan a user authentication feature") the questions are
> skipped and it runs immediately.
>
> Runs on your Claude subscription — no API key or
> additional cost.
>
> ## Want to learn more?
>
> - Say **"tell me more"** for step-by-step instructions
> - Say **"what is spec?"** for spec-driven development
>   with approval loops
> - Say **"what is refactor plan?"** for code-level
>   refactoring analysis
> </passage>
>
> <passage>
> [P2] source: concepts/tool-refactor-plan.md
> ---
> type: concept
> name: tool-refactor-plan
> tags: [refactoring, complexity, code-smells]
> aliases: [technical debt, clean up code, code cleanup, messy codebase, code smells]
> source: plugin/skills/refactor-plan/SKILL.md
> ---
>
> # Refactor Plan
>
> A refactor plan scans your code for structural problems
> and builds a prioritized roadmap to fix them. It catches
> the issues that accumulate quietly — a class that grew
> into a god object, copy-pasted blocks that drifted apart,
> a function with 12 levels of nesting that nobody wants to
> touch.
>
> ## What it analyzes
>
> | Category | What it finds |
> |----------|---------------|
> | **Code smells** | Long methods, god classes, feature envy, data clumps |
> | **Duplication** | Copy-pasted blocks, near-duplicates, DRY violations |
> | **Complexity** | High cyclomatic complexity, deep nesting, long chains |
> | **Coupling** | Circular imports, tight dependencies, shotgun surgery |
> | **Naming** | Abbreviations, generic names, inconsistent conventions |
> | **Dead code** | Unreachable branches, unused params, vestigial modules |
>
> ## When you'd use it
>
> Run a refactor plan when a module feels hard to change or
> test, before adding features to a tangled area, after a
> deep-review flags complexity hotspots, or when you need
> data to justify refactoring time to stakeholders.
> Refactoring without a plan leads to yak-shaving — you
> start fixing one thing and end up touching 20 files. The
> roadmap tells you which changes deliver the most
> improvement per hour invested.
>
> ## How it prioritizes
>
> | Factor | What it means |
> |--------|---------------|
> | **Severity** | How much the issue hurts readability, testability, or safety |
> | **Effort** | Lines of code affected, number of files touched |
> | **Impact** | How much better the code gets after the fix |
> | **Risk** | Chance of introducing regressions during the change |
>
> Items that are high-severity, low-effort, and high-impact
> float to the top. Risky changes get flagged so you can
> plan extra testing.
>
> ## Want to learn more?
>
> - Say **"tell me more"** for step-by-step instructions
> - Say **"what is code quality?"** for a broader code review
> - Say **"tell me about security audit"** to scan for
>   vulnerabilities instead
> </passage>
>
> <passage>
> [P3] source: quickstarts/skill-refactor-plan.md
> ---
> type: quickstart
> name: skill-refactor-plan
> tags: [skill, claude-code]
> source: plugin/skills/refactor-plan/SKILL.md
> ---
>
> # Quickstart: Use /refactor-plan
>
> Code-level refactoring analysis and roadmap.
>
> ```
> /refactor-plan <path to analyze>
> ```
>
> **Result:** Structured results in your Claude Code conversation.
>
> **Next:** See full reference: `attune help-docs ref-skill-refactor-plan`
>
> ## Related Topics
>
> _No related topics yet._
> </passage>

### Answer

# Planning a New Feature

Great! The **Planning** tool is designed exactly for this use case.

## What It Does

Planning helps you think through features, architecture, and TDD strategy **before** writing code, giving you a structured plan when changes are cheapest to make [P1].

For a new feature, it will produce a **Feature spec**, which includes goals, scope, non-goals, and a task breakdown with effort estimates — typically taking around 2–3 minutes [P1].

## What You'll Get

| Output | Description |
|--------|-------------|
| Task breakdown | Ordered steps with effort estimates and dependencies |
| Acceptance criteria | Clear, testable definition of done per task |
| Risk assessment | Blockers, unknowns, and mitigation strategies |
| Scope boundaries | Explicit non-goals to prevent scope creep |
| Dependency map | What depends on what, and what can run in parallel |

[P1]

## How It Works

You'll be guided through a couple of quick questions first — what you're planning and what kind of plan you need. However, if you provide both details upfront (e.g. *"plan a user authentication feature"*), the questions are skipped and it runs immediately [P1].

It runs on your Claude subscription — no API key or additional cost [P1].

---

**To get started**, just tell me: **what feature would you like to plan?**


### Judge verdicts (off → on)

| | Score | Supported | Unsupported |
| - | -: | -: | -: |
| Thinking off | 1.000 | 13 | 0 |
| Thinking on  | 0.952 | 20 | 1 |

**Reasoning (thinking off):**

> Every factual claim in the answer maps directly to explicit statements in passage P1. The description of the Planning tool, its modes (Feature spec), timing (~2–3 min), outputs (task breakdown, acceptance criteria, risk assessment, scope boundaries, dependency map), the guided-questions flow, the upfront-details shortcut, and the no-API-key detail are all word-for-word or near-verbatim from P1. There are no invented details, unsupported inferences, or hallucinations present in the answer.

**Reasoning (thinking on):**

> Nearly all factual claims in the answer are reproduced verbatim or near-verbatim from P1, including the planning-mode table, the output descriptions, the behavior around guiding questions, and the next-step suggestions. The only unsupported claim is the editorial recommendation that the user "likely wants" a Feature spec — P1 lists Feature spec's use case as "Starting a new feature or epic," but never explicitly prescribes it as the preferred mode for the user's situation; that framing is an inference added by the answer. Everything else is faithfully grounded in P1.

**Claims identified by judge (thinking off):**

Supported:
  - Planning helps you think through features, architecture, and TDD strategy before writing code.
  - Planning gives you a structured plan when changes are cheapest to make.
  - For a new feature, Planning produces a Feature spec.
  - A Feature spec includes goals, scope, non-goals, and a task breakdown with effort estimates.
  - A Feature spec takes around 2–3 minutes to produce.
  - Planning produces a task breakdown of ordered steps with effort estimates and dependencies.
  - Planning produces acceptance criteria as a clear, testable definition of done per task.
  - Planning produces a risk assessment covering blockers, unknowns, and mitigation strategies.
  - Planning produces scope boundaries as explicit non-goals to prevent scope creep.
  - Planning produces a dependency map showing what depends on what and what can run in parallel.
  - You'll be guided through a couple of quick questions first — what you're planning and what kind of plan you need.
  - If you provide both details upfront (e.g. 'plan a user authentication feature'), the questions are skipped and it runs immediately.
  - Planning runs on your Claude subscription — no API key or additional cost.

Unsupported:


**Claims identified by judge (thinking on):**

Supported:
  - When you ask for planning, you'll be guided through a couple of quick questions first — what you're planning and what kind of plan you need.
  - The quick questions keep the output focused on your actual goal instead of producing a generic plan.
  - If you provide both details upfront (e.g. 'plan a user authentication feature'), the questions are skipped and it runs immediately.
  - The Feature spec planning mode produces goals, scope, non-goals, and a task breakdown with effort estimates.
  - The Feature spec planning mode's use case is starting a new feature or epic.
  - The Feature spec planning mode takes approximately 2–3 minutes.
  - The TDD scaffold planning mode produces a test-first structure with red/green/refactor steps and test names.
  - The TDD scaffold planning mode takes approximately 1–2 minutes.
  - The TDD scaffold planning mode's use case is complex logic that needs test coverage from the start.
  - The Architecture review planning mode produces component analysis, coupling assessment, and a dependency map.
  - The Architecture review planning mode takes approximately 2–3 minutes.
  - The Architecture review planning mode's use case is evaluating design decisions before committing to them.
  - The plan produces a task breakdown of ordered steps with effort estimates and dependencies.
  - The plan produces acceptance criteria described as a clear, testable definition of done per task.
  - The plan produces a risk assessment covering blockers, unknowns, and mitigation strategies.
  - The plan produces scope boundaries as explicit non-goals to prevent scope creep.
  - The plan produces a dependency map showing what depends on what and what can run in parallel.
  - Saying 'tell me more' provides step-by-step instructions.
  - Saying 'what is spec?' covers spec-driven development with approval loops.
  - Saying 'what is refactor plan?' covers code-level refactoring analysis.

Unsupported:
  - For a new feature, the user 'likely wants' a Feature spec (editorial recommendation not explicitly stated in the passages).

### Your labels

```yaml
id: gq-010
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

### Retrieved context

> <passage>
> [P1] source: concepts/tool-bug-predict.md
> ---
> type: concept
> name: tool-bug-predict
> tags: [security, bugs, scanning]
> aliases: [risky code changes, identify files that break, runtime errors, spot bugs before deploy, risk assessment]
> source: plugin/skills/bug-predict/SKILL.md
> ---
>
> # Bug Prediction
>
> Bug prediction scans your codebase for the patterns that
> historically cause production incidents. Instead of waiting
> for bugs to surface, it analyzes code structure, complexity,
> and known anti-patterns to tell you where failures are most
> likely to happen next.
>
> ## What it predicts
>
> The scanner looks for three categories of risk, each with
> different confidence levels and severity:
>
> | Pattern | Severity | Confidence | What to look for |
> |---------|----------|------------|------------------|
> | **dangerous_eval** | HIGH | High | `eval()`, `exec()`, `compile()` on any input — code injection vectors |
> | **broad_exception** | MEDIUM | Medium | Bare `except:`, unlogged `except Exception:` — errors silently swallowed |
> | **incomplete_code** | LOW | Low | TODO, FIXME, HACK, XXX comments — unfinished code paths that break under edge cases |
>
> ## Risk factors beyond patterns
>
> The scanner also weighs contextual signals that increase
> bug likelihood:
>
> - **Cyclomatic complexity** — deeply nested conditionals
>   and long function bodies correlate with higher defect
>   rates
> - **Change frequency** — files modified often ("hot" files)
>   are more likely to contain regressions
> - **Code smells** — functions over 50 lines, classes with
>   too many methods, duplicated logic across modules
>
> ## Smart false-positive filtering
>
> Not every match is a real bug. The scanner automatically
> suppresses known-safe patterns:
>
> - `eval()` inside test fixture strings (test data, not
>   executable code)
> - JavaScript `regex.exec()` method calls (safe, not
>   Python's `exec()`)
> - Broad exceptions with `# INTENTIONAL:` comments and
>   `# noqa: BLE001` markers
> - Version detection fallbacks, cleanup/teardown code, and
>   optional feature guards
>
> ## When you'd use it
>
> - Before merging a large PR — catch patterns humans miss
> - During code review — focus human attention on real risks
> - After onboarding unfamiliar code — map risk hotspots fast
> - As a periodic health check on high-churn modules
> - Before a release — verify no new high-severity patterns
>   crept in
>
> ## Want to learn more?
>
> - Say **"tell me more"** for step-by-step instructions
> - Say **"what is security audit?"** for vulnerability scanning
> - Say **"what is code quality?"** for a broader code review
> </passage>
>
> <passage>
> [P2] source: concepts/task-ci-cd-pipeline.md
> ---
> type: concept
> name: task-ci-cd-pipeline
> tags: [ci, cd, github-actions, automation]
> source: developer-guidance
> ---
>
> # Concept: CI/CD pipelines
>
> ## What
>
> CI/CD is the practice of automatically building, testing,
> and deploying your code every time you push a change.
> **Continuous Integration** (CI) catches bugs early by
> running checks on every commit. **Continuous Deployment**
> (CD) ships verified code to production without manual
> steps. Together they replace "it works on my machine"
> with "it works everywhere, every time."
>
> ## Why
>
> Without CI, bugs hide until someone manually runs the
> test suite -- which might be days or weeks after the
> broken code was merged. Without CD, deployments are
> stressful rituals involving checklists and crossed
> fingers. A pipeline turns both into boring, automatic
> events that happen in the background while you keep
> coding.
>
> ## CI vs CD
>
> | Aspect | CI (Continuous Integration) | CD (Continuous Deployment) |
> |---|---|---|
> | **Goal** | Catch problems fast | Ship verified code automatically |
> | **When it runs** | Every push or pull request | After CI passes on the main branch |
> | **What it does** | Lint, test, type-check, security scan | Build artifacts, publish packages, deploy |
> | **Fails if** | Any check reports an error | Build or deploy step fails |
> | **Who fixes it** | The developer who pushed | The on-call or release owner |
>
> Some teams use **Continuous Delivery** instead of
> Continuous Deployment -- the pipeline builds and
> verifies everything but waits for a human to press
> the deploy button.
>
> ## Pipeline stages
>
> A typical Python CI pipeline runs these stages in
> order. Each stage acts as a gate -- if it fails, the
> later stages never run.
>
> | Stage | What runs | Blocks deploy? | Typical time |
> |---|---|---|---|
> | **Lint** | ruff, black, trailing-whitespace checks | Yes | 10-30 seconds |
> | **Type check** | mypy or pyright | Optional | 30-60 seconds |
> | **Unit tests** | pytest on your test suite | Yes | 1-10 minutes |
> | **Security scan** | bandit, pip-audit, detect-secrets | Yes | 30-60 seconds |
> | **Build** | Build wheel and sdist | Yes | 10-30 seconds |
> | **Integration tests** | Tests that hit real services or databases | Sometimes | 2-15 minutes |
> | **Deploy** | Publish to PyPI, push container, etc. | N/A (this IS the deploy) | 1-5 minutes |
>
> ## What to automate and why
>
> | Automate this | Because |
> |---|---|
> | Linting and formatting | Eliminates style debates in code review |
> | Running tests | Catches regressions before they reach main |
> | Security scanning | Finds vulnerabilities before they ship |
> | Dependency auditing | Catches known CVEs in your supply chain |
> | Building artifacts | Ensures the package actually builds cleanly |
> | Version checks | Prevents releasing with a stale version |
>
> ## GitHub Actions basics
>
> GitHub Actions is the most common CI/CD platform for
> GitHub-hosted projects. Key concepts:
>
> - **Workflow**: A YAML file in `.github/workflows/` that
>   defines when and what to run
> - **Job**: A set of steps that run on a single runner
>   (virtual machine)
> - **Step**: One command or action within a job
> - **Runner**: The machine that executes the job (Ubuntu,
>   macOS, or Windows)
> - **Action**: A reusable step published by the community
>   (e.g., `actions/checkout`)
> - **Matrix**: Run the same job across multiple
>   configurations (Python versions, operating systems)
>
> ## Matrix builds
>
> A matrix strategy runs your test suite across multiple
> Python versions and operating systems in parallel:
>
> | Dimension | Common values | Why |
> |---|---|---|
> | Python version | 3.10, 3.11, 3.12, 3.13 | Catch version-specific bugs |
> | Operating system | ubuntu-latest, macos-latest, windows-latest | Catch platform-specific issues (encoding, paths) |
> | Dependency set | minimal, full | Verify optional extras don't break core |
>
> A 4-version x 3-OS matrix creates 12 parallel jobs.
> This sounds expensive but each job runs independently,
> so the total wall-clock time is roughly the time of the
> slowest single job.
>
> ## Want to learn more?
>
> - Say **"how do I set up CI for my Python project?"**
>   for the step-by-step guide
> - Say **"show me the GitHub Actions reference"** for
>   workflow syntax, trigger events, and caching strategies
> - Say **"I need CI for my Python project"** for the
>   5-step quickstart
> - Try **/security** to add security scanning to your
>   pipeline
> - Try **/smart-test** to identify which tests to run
>   in CI
> - Try **/release** to automate deployment checks
>
> ## Related Topics
>
> - **Task**: CI/CD pipeline -- step-by-step guide to
>   setting up GitHub Actions for a Python project
> - **Reference**: CI/CD pipeline -- GitHub Actions syntax,
>   triggers, caching, and common problems
> - **Quickstart**: CI/CD pipeline -- 5-step guide to a
>   working pipeline
> </passage>
>
> <passage>
> [P3] source: quickstarts/skill-bug-predict.md
> ---
> type: quickstart
> name: skill-bug-predict
> tags: [skill, claude-code]
> source: plugin/skills/bug-predict/SKILL.md
> ---
>
> # Quickstart: Use /bug-predict
>
> Predict likely bug locations from code patterns and complexity.
>
> ```
> /bug-predict <path or directory to scan>
> ```
>
> **Result:** Structured results in your Claude Code conversation.
>
> **Next:** See full reference: `attune help-docs ref-skill-bug-predict`
>
> ## Related Topics
>
> _No related topics yet._
> </passage>

### Answer

# Sniffing Out Hard-to-Catch Bugs

Here are the best approaches for finding bugs that are easy to miss before they hit production.

---

## 1. Use Bug Prediction Scanning (`/bug-predict`)

Bug prediction scans your codebase for patterns that historically cause production incidents, analyzing code structure, complexity, and known anti-patterns to identify where failures are most likely to happen [P1].

The scanner targets three specific risk categories that are notoriously easy to overlook [P1]:

| Pattern | Severity | Why It's Sneaky |
|---|---|---|
| **`eval()`/`exec()`/`compile()`** | HIGH | Code injection vectors hidden in input handling |
| **Bare `except:` / unlogged exceptions** | MEDIUM | Errors silently swallowed — failures disappear |
| **TODO/FIXME/HACK comments** | LOW | Unfinished code paths that break under edge cases |

Beyond these patterns, the scanner also weighs contextual signals like **cyclomatic complexity** (deeply nested conditionals), **change frequency** (frequently modified "hot" files), and **code smells** like functions over 50 lines or duplicated logic [P1].

### When to run it:
- Before merging a large PR — catch patterns humans miss [P1]
- After onboarding unfamiliar code — map risk hotspots fast [P1]
- Before a release — verify no new high-severity patterns crept in [P1]

### To use it:
```
/bug-predict <path or directory to scan>
```
[P3]

---

## 2. Add Automated Checks to Your CI Pipeline

Hard-to-catch bugs often slip through because they're only checked manually. A CI pipeline runs checks automatically on every commit, catching bugs early before they reach production [P2].

Key pipeline stages that catch hidden bugs include [P2]:

- **Security scanning** (e.g., `bandit`) — finds vulnerabilities before they ship
- **Type checking** (e.g., `mypy`) — catches type-related bugs missed in review
- **Unit tests** — catches regressions before they reach main
- **Dependency auditing** (e.g., `pip-audit`) — catches known CVEs in your supply chain

Running a **matrix build** across multiple Python versions and operating systems can also surface version-specific or platform-specific bugs that are invisible in a single environment [P2].

---

## Putting It Together

For the most thorough coverage, run `/bug-predict` on your codebase to identify risky patterns and hotspots [P1], and integrate automated scanning into a CI pipeline so every future change is checked automatically [P2].


### Judge verdicts (off → on)

| | Score | Supported | Unsupported |
| - | -: | -: | -: |
| Thinking off | 0.958 | 23 | 1 |
| Thinking on  | 1.000 | 21 | 0 |

**Reasoning (thinking off):**

> The answer is overwhelmingly faithful to the retrieved passages. Nearly every factual claim maps directly to explicit statements in P1, P2, or P3. The one claim marked unsupported—that type checking "catches type-related bugs missed in review"—is a reasonable inference but is not explicitly stated in P2, which only says type checking is an optional pipeline stage without specifying what kinds of bugs it catches. All other claims about bug prediction patterns, risk factors, when to use the tool, the CLI command, and CI/CD pipeline details are directly and explicitly supported by the passages.

**Reasoning (thinking on):**

> Every atomic factual claim in the answer maps directly to an explicit statement in one of the retrieved passages. The bug prediction descriptions (patterns, severities, risk factors, false-positive suppression, and use-case timing) are all drawn verbatim or near-verbatim from P1. The command syntax is taken directly from P3. The CI/CD pipeline details (security scan stage, tools named, blocking behavior, and regressions claim) are all directly stated in P2. There are no invented details, extrapolated inferences, or outside-knowledge additions present in the answer.

**Claims identified by judge (thinking off):**

Supported:
  - Bug prediction scans your codebase for patterns that historically cause production incidents
  - Bug prediction analyzes code structure, complexity, and known anti-patterns to identify where failures are most likely to happen
  - The scanner targets dangerous_eval patterns (eval(), exec(), compile()) with HIGH severity
  - eval(), exec(), compile() on any input are code injection vectors
  - The scanner targets broad_exception patterns (bare except:, unlogged except Exception:) with MEDIUM severity
  - Broad exceptions cause errors to be silently swallowed
  - The scanner targets incomplete_code patterns (TODO, FIXME, HACK, XXX comments) with LOW severity
  - TODO/FIXME/HACK comments represent unfinished code paths that break under edge cases
  - The scanner weighs cyclomatic complexity (deeply nested conditionals) as a contextual signal
  - The scanner weighs change frequency (frequently modified 'hot' files) as a contextual signal
  - The scanner weighs code smells like functions over 50 lines as a contextual signal
  - The scanner weighs duplicated logic as a contextual signal
  - Bug prediction is useful before merging a large PR to catch patterns humans miss
  - Bug prediction is useful after onboarding unfamiliar code to map risk hotspots fast
  - Bug prediction is useful before a release to verify no new high-severity patterns crept in
  - The command to use bug prediction is: /bug-predict <path or directory to scan>
  - A CI pipeline runs checks automatically on every commit
  - CI catches bugs early before they reach production
  - Security scanning (e.g., bandit) finds vulnerabilities before they ship
  - Type checking (e.g., mypy) is a pipeline stage
  - Unit tests catch regressions before they reach main
  - Dependency auditing (e.g., pip-audit) catches known CVEs in your supply chain
  - Running a matrix build across multiple Python versions and operating systems can surface version-specific or platform-specific bugs

Unsupported:
  - Type checking catches type-related bugs missed in review

**Claims identified by judge (thinking on):**

Supported:
  - The /bug-predict tool scans your codebase for patterns that historically cause production incidents
  - The tool analyzes code structure, complexity, and known anti-patterns to identify where failures are most likely to happen
  - The command syntax is: /bug-predict <path or directory to scan>
  - The scanner looks for three categories of risk
  - The 'dangerous_eval' pattern has HIGH severity and catches eval(), exec(), compile() on any input as code injection vectors
  - The 'broad_exception' pattern has MEDIUM severity and catches bare except: and unlogged except Exception: where errors are silently swallowed
  - The 'incomplete_code' pattern has LOW severity and catches TODO, FIXME, HACK, XXX comments as unfinished code paths that break under edge cases
  - The scanner weighs cyclomatic complexity — deeply nested conditionals and long function bodies
  - The scanner weighs change frequency — frequently modified 'hot' files are more likely to contain regressions
  - The scanner weighs code smells — functions over 50 lines, classes with too many methods, duplicated logic across modules
  - The scanner automatically suppresses eval() inside test fixture strings
  - The scanner suppresses JavaScript regex.exec() calls
  - The scanner suppresses broad exceptions marked with # INTENTIONAL: comments
  - A typical Python CI pipeline includes a Security scan stage running tools like bandit, pip-audit, detect-secrets
  - The Security scan stage blocks deployment if it fails
  - A good time to run bug prediction is before merging a large PR to catch patterns humans miss
  - A good time to run bug prediction is during code review to focus human attention on real risks
  - A good time to run bug prediction is after onboarding unfamiliar code to map risk hotspots fast
  - A good time to run bug prediction is before a release to verify no new high-severity patterns crept in
  - Security scanning finds vulnerabilities before they ship
  - Running scanning automatically means regressions are caught before they reach main

Unsupported:


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

## gq-034 — `manage the documentation pipeline`

### Retrieved context

> <passage>
> [P1] source: references/tool-doc-orchestrator.md
> ---
> type: reference
> subtype: tabular
> name: tool-doc-orchestrator
> category: tool
> tags: [mcp, tool, workflow]
> aliases: [orchestrate documentation workflow, doc pipeline, coordinate documentation updates, documentation maintenance]
> source: src/attune/mcp/tool_schemas.py
> ---
>
> # Reference: Tool: Doc Orchestrator
>
> End-to-end documentation maintenance: scout gaps, prioritize, generate, and update docs.
>
> **Group:** workflow
>
> ## Parameters
>
> | Parameter | Type | Description | Constraints | Default |
> | --------- | ---- | ----------- | ----------- | ------- |
> | `path` | string | Project root path |  | . |
>
> ## Usage
>
> `doc_orchestrator()`
>
> ## Related Topics
> - **Reference**: Tool: Security Audit — Run security audit workflow on codebase. Detects vulnerabili...
> - **Reference**: Tool: Bug Predict — Run bug prediction workflow. Analyzes code patterns and pred...
> - **Reference**: Tool: Code Review — Run code review workflow. Provides comprehensive code qualit...
> </passage>
>
> <passage>
> [P2] source: concepts/task-ci-cd-pipeline.md
> ---
> type: concept
> name: task-ci-cd-pipeline
> tags: [ci, cd, github-actions, automation]
> source: developer-guidance
> ---
>
> # Concept: CI/CD pipelines
>
> ## What
>
> CI/CD is the practice of automatically building, testing,
> and deploying your code every time you push a change.
> **Continuous Integration** (CI) catches bugs early by
> running checks on every commit. **Continuous Deployment**
> (CD) ships verified code to production without manual
> steps. Together they replace "it works on my machine"
> with "it works everywhere, every time."
>
> ## Why
>
> Without CI, bugs hide until someone manually runs the
> test suite -- which might be days or weeks after the
> broken code was merged. Without CD, deployments are
> stressful rituals involving checklists and crossed
> fingers. A pipeline turns both into boring, automatic
> events that happen in the background while you keep
> coding.
>
> ## CI vs CD
>
> | Aspect | CI (Continuous Integration) | CD (Continuous Deployment) |
> |---|---|---|
> | **Goal** | Catch problems fast | Ship verified code automatically |
> | **When it runs** | Every push or pull request | After CI passes on the main branch |
> | **What it does** | Lint, test, type-check, security scan | Build artifacts, publish packages, deploy |
> | **Fails if** | Any check reports an error | Build or deploy step fails |
> | **Who fixes it** | The developer who pushed | The on-call or release owner |
>
> Some teams use **Continuous Delivery** instead of
> Continuous Deployment -- the pipeline builds and
> verifies everything but waits for a human to press
> the deploy button.
>
> ## Pipeline stages
>
> A typical Python CI pipeline runs these stages in
> order. Each stage acts as a gate -- if it fails, the
> later stages never run.
>
> | Stage | What runs | Blocks deploy? | Typical time |
> |---|---|---|---|
> | **Lint** | ruff, black, trailing-whitespace checks | Yes | 10-30 seconds |
> | **Type check** | mypy or pyright | Optional | 30-60 seconds |
> | **Unit tests** | pytest on your test suite | Yes | 1-10 minutes |
> | **Security scan** | bandit, pip-audit, detect-secrets | Yes | 30-60 seconds |
> | **Build** | Build wheel and sdist | Yes | 10-30 seconds |
> | **Integration tests** | Tests that hit real services or databases | Sometimes | 2-15 minutes |
> | **Deploy** | Publish to PyPI, push container, etc. | N/A (this IS the deploy) | 1-5 minutes |
>
> ## What to automate and why
>
> | Automate this | Because |
> |---|---|
> | Linting and formatting | Eliminates style debates in code review |
> | Running tests | Catches regressions before they reach main |
> | Security scanning | Finds vulnerabilities before they ship |
> | Dependency auditing | Catches known CVEs in your supply chain |
> | Building artifacts | Ensures the package actually builds cleanly |
> | Version checks | Prevents releasing with a stale version |
>
> ## GitHub Actions basics
>
> GitHub Actions is the most common CI/CD platform for
> GitHub-hosted projects. Key concepts:
>
> - **Workflow**: A YAML file in `.github/workflows/` that
>   defines when and what to run
> - **Job**: A set of steps that run on a single runner
>   (virtual machine)
> - **Step**: One command or action within a job
> - **Runner**: The machine that executes the job (Ubuntu,
>   macOS, or Windows)
> - **Action**: A reusable step published by the community
>   (e.g., `actions/checkout`)
> - **Matrix**: Run the same job across multiple
>   configurations (Python versions, operating systems)
>
> ## Matrix builds
>
> A matrix strategy runs your test suite across multiple
> Python versions and operating systems in parallel:
>
> | Dimension | Common values | Why |
> |---|---|---|
> | Python version | 3.10, 3.11, 3.12, 3.13 | Catch version-specific bugs |
> | Operating system | ubuntu-latest, macos-latest, windows-latest | Catch platform-specific issues (encoding, paths) |
> | Dependency set | minimal, full | Verify optional extras don't break core |
>
> A 4-version x 3-OS matrix creates 12 parallel jobs.
> This sounds expensive but each job runs independently,
> so the total wall-clock time is roughly the time of the
> slowest single job.
>
> ## Want to learn more?
>
> - Say **"how do I set up CI for my Python project?"**
>   for the step-by-step guide
> - Say **"show me the GitHub Actions reference"** for
>   workflow syntax, trigger events, and caching strategies
> - Say **"I need CI for my Python project"** for the
>   5-step quickstart
> - Try **/security** to add security scanning to your
>   pipeline
> - Try **/smart-test** to identify which tests to run
>   in CI
> - Try **/release** to automate deployment checks
>
> ## Related Topics
>
> - **Task**: CI/CD pipeline -- step-by-step guide to
>   setting up GitHub Actions for a Python project
> - **Reference**: CI/CD pipeline -- GitHub Actions syntax,
>   triggers, caching, and common problems
> - **Quickstart**: CI/CD pipeline -- 5-step guide to a
>   working pipeline
> </passage>
>
> <passage>
> [P3] source: quickstarts/task-ci-cd-pipeline.md
> ---
> type: quickstart
> name: task-ci-cd-pipeline
> tags: [ci, cd, github-actions, automation]
> source: developer-guidance
> ---
>
> # I Need CI for My Python Project
>
> Five steps to a working GitHub Actions pipeline that
> lints, tests, and scans your code on every push.
>
> ## Step 1: Create the workflow directory
>
> ```
> mkdir -p .github/workflows
> ```
>
> GitHub Actions only looks for workflow files in this
> exact directory.
>
> ## Step 2: Create the workflow file
>
> Create `.github/workflows/ci.yml` with this content:
>
> ```yaml
> name: CI
>
> on:
>   push:
>     branches: [main]
>   pull_request:
>     branches: [main]
>
> jobs:
>   lint:
>     runs-on: ubuntu-latest
>     steps:
>       - uses: actions/checkout@v4
>       - uses: actions/setup-python@v5
>         with:
>           python-version: "3.12"
>           cache: pip
>       - run: pip install ruff
>       - run: ruff check src/
>
>   test:
>     needs: lint
>     runs-on: ubuntu-latest
>     strategy:
>       matrix:
>         python-version: ["3.10", "3.12"]
>     steps:
>       - uses: actions/checkout@v4
>       - uses: actions/setup-python@v5
>         with:
>           python-version: ${{ matrix.python-version }}
>           cache: pip
>       - run: pip install -e '.[dev]'
>       - run: pytest --tb=short
> ```
>
> Adjust `src/` and `'.[dev]'` to match your project
> layout and test extras.
>
> ## Step 3: Push and verify
>
> ```
> git add .github/workflows/ci.yml
> git commit -m "ci: add GitHub Actions pipeline"
> git push
> ```
>
> Go to the **Actions** tab on your GitHub repository.
> You should see the workflow running within a few seconds.
>
> ## Step 4: Open a test pull request
>
> Create a branch, make a small change, and open a PR.
> Verify that CI runs on the pull request and the status
> checks appear on the PR page.
>
> ## Step 5: Protect your main branch
>
> Go to **Settings > Branches > Add branch protection
> rule** for `main`:
>
> - Enable **Require status checks to pass before merging**
> - Select the `lint` and `test` jobs
> - Save
>
> Now no one can merge a PR that breaks CI.
>
> **Done.** Every push and PR now runs linting and tests
> automatically.
>
> ## Quick additions
>
> | Want to add | What to do |
> |---|---|
> | Security scanning | Add a `security` job with `bandit -r src/` and `pip-audit` |
> | More Python versions | Add versions to the matrix list |
> | macOS/Windows testing | Add `os: [ubuntu-latest, macos-latest, windows-latest]` to the matrix |
> | Dependency caching with uv | Replace `setup-python` with `astral-sh/setup-uv@v4` and `enable-cache: true` |
> | Auto-publish to PyPI | Add a deploy job triggered by tag push |
>
> ## Want to learn more?
>
> - Say **"what is CI/CD?"** for pipeline concepts -- CI
>   vs CD, stages, and matrix builds explained
> - Say **"walk me through the full CI setup"** for the
>   detailed guide with security scanning and caching
> - Say **"show me the GitHub Actions reference"** for
>   trigger events, caching options, secrets, and
>   troubleshooting
> - Try **/security** to add security gates to your
>   pipeline
> - Try **/smart-test** to generate tests for uncovered
>   code before enabling CI
> - Try **/release** to automate deployment after CI
>   passes
>
> ## Related Topics
>
> - **Concept**: CI/CD pipeline -- CI vs CD, pipeline
>   stages, matrix builds, and what to automate
> - **Task**: CI/CD pipeline -- step-by-step guide to
>   setting up GitHub Actions for a Python project
> - **Reference**: CI/CD pipeline -- GitHub Actions syntax,
>   triggers, caching, secrets, and common problems
> </passage>

### Answer

## Managing the Documentation Pipeline

You can manage the documentation pipeline using the **Doc Orchestrator** tool, which handles end-to-end documentation maintenance.

### What It Does

The Doc Orchestrator performs the full documentation maintenance workflow: it scouts gaps, prioritizes, generates, and updates docs [P1].

### How to Use It

Call the tool with:

```python
doc_orchestrator()
```

It accepts a single optional parameter [P1]:

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `path` | string | Project root path | `.` |

So to run it on a specific project directory:

```python
doc_orchestrator(path="/your/project/root")
```

### Related Workflows

The Doc Orchestrator is part of the **workflow** group and is related to several other pipeline tools you can use alongside it [P1]:

- **Security Audit** — detects vulnerabilities in your codebase
- **Bug Predict** — analyzes code patterns and predicts bugs
- **Code Review** — provides comprehensive code quality analysis

### Tip: Integrate with CI/CD

If you want documentation maintenance to run automatically, you can incorporate it into a CI/CD pipeline that triggers on every push or pull request [P2]. GitHub Actions workflows defined in `.github/workflows/` are a common way to automate such tasks [P2, P3].


### Judge verdicts (off → on)

| | Score | Supported | Unsupported |
| - | -: | -: | -: |
| Thinking off | 0.882 | 15 | 2 |
| Thinking on  | 0.867 | 13 | 2 |

**Reasoning (thinking off):**

> Most claims in the answer are well-supported by P1, which explicitly documents the doc_orchestrator tool, its parameters, defaults, group, and related tools. The CI/CD integration tip is where the answer goes beyond the passages: while P2 and P3 describe CI/CD pipelines and GitHub Actions in general, neither passage mentions the Doc Orchestrator in a CI/CD context — this connection is an inference not explicitly stated in any passage. The specific usage example doc_orchestrator(path="/your/project/root") is also an inference (though a reasonable one) not explicitly shown in any passage.

**Reasoning (thinking on):**

> Most claims about the Doc Orchestrator tool are well-supported by P1, which provides its description, usage signature, parameter table, and related tools. The CI/CD background facts about pipeline stages and the .github/workflows/ directory are supported by P2 and P3. Two claims are unsupported: (1) the specific parameterized call syntax `doc_orchestrator(path="...")` is an inference — P1 only shows the no-argument form `doc_orchestrator()`; and (2) the suggestion to integrate doc_orchestrator into a CI/CD pipeline is an inference the answer draws by combining P1 and P2/P3 — no passage explicitly connects the doc orchestrator tool to CI/CD integration. These two represent minor prior-knowledge drift and cross-passage inference beyond what the sources directly state.

**Claims identified by judge (thinking off):**

Supported:
  - The Doc Orchestrator handles end-to-end documentation maintenance.
  - The Doc Orchestrator scouts gaps, prioritizes, generates, and updates docs.
  - The tool is called with doc_orchestrator().
  - The tool accepts a parameter named 'path'.
  - The 'path' parameter is of type string.
  - The 'path' parameter describes the project root path.
  - The default value for 'path' is '.'.
  - The 'path' parameter is optional.
  - The Doc Orchestrator is part of the 'workflow' group.
  - The Doc Orchestrator is related to the Security Audit tool, which detects vulnerabilities in the codebase.
  - The Doc Orchestrator is related to the Bug Predict tool, which analyzes code patterns and predicts bugs.
  - The Doc Orchestrator is related to the Code Review tool, which provides comprehensive code quality analysis.
  - CI/CD pipelines can trigger on every push or pull request.
  - GitHub Actions workflows are defined in '.github/workflows/'.
  - GitHub Actions workflows are a common way to automate tasks.

Unsupported:
  - You can incorporate the Doc Orchestrator into a CI/CD pipeline to run documentation maintenance automatically.
  - doc_orchestrator(path="/your/project/root") is a valid usage example for running on a specific directory.

**Claims identified by judge (thinking on):**

Supported:
  - The Doc Orchestrator handles end-to-end documentation maintenance automatically
  - The Doc Orchestrator scouts gaps, prioritizes, generates, and updates docs
  - Basic usage is doc_orchestrator()
  - The tool accepts a parameter named `path`
  - The `path` parameter is of type string
  - The `path` parameter describes the project root path
  - The `path` parameter defaults to '.'
  - The Doc Orchestrator belongs to the 'workflow' group of tools
  - Security Audit is a related tool that runs a security audit workflow on the codebase
  - Bug Predict is a related tool that analyzes code patterns and predicts bugs
  - Code Review is a related tool that provides comprehensive code quality review
  - A typical CI pipeline runs stages in order where each stage acts as a gate
  - GitHub Actions workflow files are located in .github/workflows/

Unsupported:
  - doc_orchestrator(path="/your/project/root") is a valid usage example (specific syntax not shown in passages; P1 only shows doc_orchestrator())
  - The Doc Orchestrator can be integrated into a CI/CD pipeline as a documentation step (no passage connects doc_orchestrator to CI/CD integration)

### Your labels

```yaml
id: gq-034
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

## gq-014 — `look for dangerous eval calls`

### Retrieved context

> <passage>
> [P1] source: concepts/tool-bug-predict.md
> ---
> type: concept
> name: tool-bug-predict
> tags: [security, bugs, scanning]
> aliases: [risky code changes, identify files that break, runtime errors, spot bugs before deploy, risk assessment]
> source: plugin/skills/bug-predict/SKILL.md
> ---
>
> # Bug Prediction
>
> Bug prediction scans your codebase for the patterns that
> historically cause production incidents. Instead of waiting
> for bugs to surface, it analyzes code structure, complexity,
> and known anti-patterns to tell you where failures are most
> likely to happen next.
>
> ## What it predicts
>
> The scanner looks for three categories of risk, each with
> different confidence levels and severity:
>
> | Pattern | Severity | Confidence | What to look for |
> |---------|----------|------------|------------------|
> | **dangerous_eval** | HIGH | High | `eval()`, `exec()`, `compile()` on any input — code injection vectors |
> | **broad_exception** | MEDIUM | Medium | Bare `except:`, unlogged `except Exception:` — errors silently swallowed |
> | **incomplete_code** | LOW | Low | TODO, FIXME, HACK, XXX comments — unfinished code paths that break under edge cases |
>
> ## Risk factors beyond patterns
>
> The scanner also weighs contextual signals that increase
> bug likelihood:
>
> - **Cyclomatic complexity** — deeply nested conditionals
>   and long function bodies correlate with higher defect
>   rates
> - **Change frequency** — files modified often ("hot" files)
>   are more likely to contain regressions
> - **Code smells** — functions over 50 lines, classes with
>   too many methods, duplicated logic across modules
>
> ## Smart false-positive filtering
>
> Not every match is a real bug. The scanner automatically
> suppresses known-safe patterns:
>
> - `eval()` inside test fixture strings (test data, not
>   executable code)
> - JavaScript `regex.exec()` method calls (safe, not
>   Python's `exec()`)
> - Broad exceptions with `# INTENTIONAL:` comments and
>   `# noqa: BLE001` markers
> - Version detection fallbacks, cleanup/teardown code, and
>   optional feature guards
>
> ## When you'd use it
>
> - Before merging a large PR — catch patterns humans miss
> - During code review — focus human attention on real risks
> - After onboarding unfamiliar code — map risk hotspots fast
> - As a periodic health check on high-churn modules
> - Before a release — verify no new high-severity patterns
>   crept in
>
> ## Want to learn more?
>
> - Say **"tell me more"** for step-by-step instructions
> - Say **"what is security audit?"** for vulnerability scanning
> - Say **"what is code quality?"** for a broader code review
> </passage>
>
> <passage>
> [P2] source: errors/bug-predict-dangerous-eval-flags-subprocess-exec.md
> ---
> type: error
> name: bug-predict-dangerous-eval-flags-subprocess-exec
> confidence: Verified
> tags: [security, claude-code]
> source: .claude/CLAUDE.md
> ---
>
> # Error: Bug-predict `dangerous_eval` flags `subprocess_exec`
>
> ## Signature
>
> Bug-predict `dangerous_eval` flags `subprocess_exec`
>
> ## Root Cause
>
> The scanner's regex matches `create_subprocess_exec` as containing `exec`, producing a false positive for `dangerous_eval` in `hooks/executor.py`. There is no actual `eval()` or `exec()` usage. Always verify HIGH severity scanner findings against the source before treating them as real vulnerabilities.
>
> ## Resolution
>
> 1. Always verify HIGH severity scanner findings against the source before treating them as real vulnerabilities
>
> ## Confidence
>
> **Verified** — Confirmed by prior incident (Lessons Learned)
>
> ## Related Topics
> - Tip: Best practice: Bug-predict `dangerous_eval` flags `subprocess_exec`
> </passage>
>
> <passage>
> [P3] source: references/tool-security-audit.md
> ---
> type: reference
> subtype: tabular
> name: tool-security-audit
> category: tool
> tags: [mcp, tool, workflow]
> source: src/attune/mcp/tool_schemas.py
> ---
>
> # Reference: Tool: Security Audit
>
> Run security audit workflow on codebase. Detects vulnerabilities, dangerous patterns, and security issues. Returns findings with severity levels.
>
> **Group:** workflow
>
> ## Parameters
>
> | Parameter | Type | Description | Constraints | Default |
> | --------- | ---- | ----------- | ----------- | ------- |
> | `path` | string | Path to directory or file to audit |  | required |
>
> ## Usage
>
> `security_audit(path="...")`
>
> ## Related Topics
> - **Reference**: Tool: Bug Predict — Run bug prediction workflow. Analyzes code patterns and pred...
> - **Reference**: Tool: Code Review — Run code review workflow. Provides comprehensive code qualit...
> - **Reference**: Tool: Test Generation — Generate tests for code. Can batch generate tests for multip...
> </passage>

### Answer

## Scanning for Dangerous Eval Calls

I can run bug prediction to look for dangerous eval patterns in your codebase. Here's what you need to know:

### What the Scanner Detects

The **`dangerous_eval`** pattern flags uses of `eval()`, `exec()`, and `compile()` on any input, as these are code injection vectors. This category is rated **HIGH severity** with **High confidence** [P1].

### How to Run It

Use the bug predict tool on your target path:

```
bug_predict(path="...")
```

The related security audit tool can also be used for broader vulnerability scanning [P3].

---

### ⚠️ Important: Verify HIGH Severity Findings

Before treating any HIGH severity result as a real vulnerability, **always verify it against the source code** [P2].

A known false positive exists: the scanner's regex matches `create_subprocess_exec` as containing `exec`, flagging it as `dangerous_eval` in files like `hooks/executor.py` — even when there is no actual `eval()` or `exec()` usage [P2].

### Other Known False Positives to Watch For

The scanner automatically suppresses some safe patterns, including:
- `eval()` inside **test fixture strings** (test data, not executable code)
- JavaScript **`regex.exec()`** method calls (safe, not Python's `exec()`) [P1]

---

**Recommendation:** Run the scan, then cross-check any HIGH severity `dangerous_eval` findings directly in the flagged source files before acting on them [P2].


### Judge verdicts (off → on)

| | Score | Supported | Unsupported |
| - | -: | -: | -: |
| Thinking off | 0.923 | 12 | 1 |
| Thinking on  | 0.909 | 10 | 1 |

**Reasoning (thinking off):**

> Almost all claims in the answer are directly supported by the retrieved passages. P1 covers the dangerous_eval pattern details (severity, confidence, what it flags), as well as the false-positive suppression rules for test fixture strings and regex.exec(). P2 covers the known false positive involving `create_subprocess_exec` in `hooks/executor.py` and the recommendation to verify HIGH severity findings. P3 covers the security audit tool and its `security_audit(path="...")` syntax. The one unsupported claim is the specific invocation syntax `bug_predict(path="...")` — P3 only shows the `security_audit` tool schema; while a bug predict tool is referenced in P3's related topics, its exact call signature is not provided in any passage.

**Reasoning (thinking on):**

> Most of the answer's factual claims are well-supported by P1 (dangerous_eval pattern details, severity, confidence, auto-suppressed false positives) and P2 (the known false positive involving create_subprocess_exec in hooks/executor.py and the advice to verify HIGH severity findings). The one unsupported claim is the invocation syntax bug_predict(path="..."). P3 provides the syntax only for the security_audit tool (security_audit(path="...")), and while the Bug Predict tool is referenced as a related topic in P3, no explicit call signature for it appears in any retrieved passage. This is a case of the answer inferring the syntax by analogy with security_audit, which goes beyond what the passages directly state.

**Claims identified by judge (thinking off):**

Supported:
  - The `dangerous_eval` pattern flags uses of `eval()`, `exec()`, and `compile()` on any input
  - These are code injection vectors
  - The `dangerous_eval` category is rated HIGH severity
  - The `dangerous_eval` category is rated High confidence
  - The scanner's regex matches `create_subprocess_exec` as containing `exec`, producing a false positive for `dangerous_eval`
  - This false positive occurs in files like `hooks/executor.py`
  - There is no actual `eval()` or `exec()` usage in that file
  - Always verify HIGH severity scanner findings against the source before treating them as real vulnerabilities
  - The scanner suppresses `eval()` inside test fixture strings (test data, not executable code)
  - The scanner suppresses JavaScript `regex.exec()` method calls (safe, not Python's `exec()`)
  - A security audit tool can be used for broader vulnerability scanning
  - The security audit tool uses the syntax `security_audit(path="...")`

Unsupported:
  - The bug predict tool is invoked with the syntax `bug_predict(path="...")`

**Claims identified by judge (thinking on):**

Supported:
  - The Bug Predict tool scans for dangerous eval patterns.
  - The scanner looks for eval(), exec(), and compile() on any input.
  - eval(), exec(), and compile() on any input are considered code injection vectors.
  - The dangerous_eval pattern has a severity of HIGH.
  - The dangerous_eval pattern has a confidence level of High.
  - The scanner's regex matches create_subprocess_exec as containing exec, producing a false positive for dangerous_eval in hooks/executor.py.
  - In that false positive case, there is no actual eval() or exec() usage.
  - You should always verify HIGH severity scanner findings against the source before treating them as real vulnerabilities.
  - The scanner automatically suppresses eval() inside test fixture strings (test data, not executable code).
  - The scanner automatically suppresses JavaScript regex.exec() method calls, which are safe and not Python's exec().

Unsupported:
  - The tool call syntax for Bug Predict is bug_predict(path="...").

### Your labels

```yaml
id: gq-014
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

## gq-011 — `vulnerability scan`

### Retrieved context

> <passage>
> [P1] source: quickstarts/skill-security-audit.md
> ---
> type: quickstart
> name: skill-security-audit
> tags: [skill, claude-code]
> source: plugin/skills/security-audit/SKILL.md
> ---
>
> # Quickstart: Use /security-audit
>
> Scan code for security vulnerabilities including eval/exec usage, path traversal, hardcoded secrets, and injection risks.
>
> ```
> /security-audit <path or directory to scan>
> ```
>
> **Result:** Structured results in your Claude Code conversation.
>
> **Next:** See full reference: `attune help-docs ref-skill-security-audit`
>
> ## Related Topics
>
> _No related topics yet._
> </passage>
>
> <passage>
> [P2] source: concepts/tool-security-audit.md
> ---
> type: concept
> name: tool-security-audit
> tags: [security, skill, workflow]
> source: plugin/skills/security-audit/SKILL.md
> ---
>
> # Security Audit
>
> A security audit scans your codebase for vulnerabilities
> that are easy to introduce and hard to spot in code
> review. It catches the mistakes that slip through when
> you're focused on making things work — an `eval()` in a
> test fixture, a file path built from user input without
> validation, an API key that ended up in source control.
>
> ## What it finds
>
> | Category | What to worry about |
> |----------|---------------------|
> | **Code injection** | `eval()`, `exec()`, and `compile()` on untrusted input |
> | **Path traversal** | File operations that don't validate the path first |
> | **Hardcoded secrets** | API keys, tokens, and passwords committed to source |
> | **SQL/command injection** | String concatenation in queries or shell commands |
> | **SSRF** | HTTP requests to URLs controlled by user input |
> | **Weak cryptography** | MD5/SHA1 for security purposes, hardcoded IVs |
>
> ## When you'd use it
>
> Run a security audit before releasing a new version,
> after adding code that handles files or user input, when
> pulling in a new dependency, or as a CI gate on pull
> requests. A quick scan takes under a minute. A deep
> review takes longer but maps findings to OWASP
> categories.
>
> ## How deep it goes
>
> | Depth | Time | What you get |
> |-------|------|-------------|
> | **Quick** | ~30s | Surface scan — eval/exec, obvious secrets |
> | **Standard** | ~2 min | Full pattern matching with severity ratings |
> | **Deep** | ~5 min | Multi-pass review with OWASP mapping and fix suggestions |
>
> ## Want to learn more?
>
> - Say **"tell me more"** for step-by-step instructions
> - Say **"what is code quality?"** for a broader code review
> - Say **"tell me about bug prediction"** to predict where failures happen
> </passage>
>
> <passage>
> [P3] source: quickstarts/run-security-audit.md
> ---
> type: quickstart
> name: run-security-audit
> tags: [workflow, security]
> source: src/attune/cli_minimal.py
> ---
>
> # Quickstart: Run a security audit
>
> Scan your codebase for security vulnerabilities.
>
> ```
> attune workflow run security-audit --path "src/"
> ```
>
> **Result:** Severity-grouped findings with CWE identifiers.
>
> **Next:** Fix critical issues, then run `attune workflow run test-gen`.
>
> ## Related Topics
>
> _No related topics yet._
> </passage>

### Answer

# Running a Vulnerability Scan

There are two ways to run a vulnerability scan on your codebase:

## Option 1: Claude Code Skill

Use the `/security-audit` command directly in your Claude Code conversation [P1]:

```
/security-audit <path or directory to scan>
```

## Option 2: CLI Workflow

Run the audit via the `attune` CLI [P3]:

```
attune workflow run security-audit --path "src/"
```

This produces severity-grouped findings with CWE identifiers [P3].

---

## What Gets Scanned

A security audit checks for the following vulnerability categories [P2]:

| Category | Examples |
|----------|---------|
| **Code injection** | `eval()`, `exec()`, `compile()` on untrusted input |
| **Path traversal** | File operations without path validation |
| **Hardcoded secrets** | API keys, tokens, passwords in source |
| **SQL/command injection** | String concatenation in queries or shell commands |
| **SSRF** | HTTP requests to user-controlled URLs |
| **Weak cryptography** | MD5/SHA1, hardcoded IVs |

## Scan Depth Options

You can choose how deep the scan goes [P2]:

| Depth | Time | What you get |
|-------|------|-------------|
| **Quick** | ~30s | Surface scan — eval/exec, obvious secrets |
| **Standard** | ~2 min | Full pattern matching with severity ratings |
| **Deep** | ~5 min | Multi-pass review with OWASP mapping and fix suggestions |

## When to Run It

Consider running a scan before releasing a new version, after adding code that handles files or user input, when pulling in a new dependency, or as a CI gate on pull requests [P2].


### Judge verdicts (off → on)

| | Score | Supported | Unsupported |
| - | -: | -: | -: |
| Thinking off | 1.000 | 17 | 0 |
| Thinking on  | 1.000 | 17 | 0 |

**Reasoning (thinking off):**

> Every atomic factual claim in the answer is directly and explicitly supported by at least one of the retrieved passages. The two usage methods (skill and CLI) map precisely to P1 and P3 respectively. All vulnerability categories, scan depth options, timings, and "when to run" scenarios are verbatim or near-verbatim from P2. No invented details, CLI flags, or outside knowledge were introduced.

**Reasoning (thinking on):**

> Every atomic factual claim in the answer is directly traceable to one of the three retrieved passages. The two usage methods (Claude Code skill via `/security-audit` and the `attune` CLI workflow) are each explicitly described in P1 and P3 respectively. The vulnerability categories, scan depth tiers (with timings and descriptions), and recommended timing for running a scan are all taken verbatim from P2. No invented CLI flags, workflow names, OWASP categories beyond what the passages state, or outside-knowledge details were introduced. The answer is fully faithful to the retrieved passages.

**Claims identified by judge (thinking off):**

Supported:
  - You can use the /security-audit command directly in your Claude Code conversation
  - The /security-audit command syntax is: /security-audit <path or directory to scan>
  - You can run the audit via the attune CLI using: attune workflow run security-audit --path "src/"
  - The CLI workflow produces severity-grouped findings with CWE identifiers
  - Code injection (eval(), exec(), compile() on untrusted input) is a vulnerability category scanned
  - Path traversal (file operations without path validation) is a vulnerability category scanned
  - Hardcoded secrets (API keys, tokens, passwords in source) is a vulnerability category scanned
  - SQL/command injection (string concatenation in queries or shell commands) is a vulnerability category scanned
  - SSRF (HTTP requests to user-controlled URLs) is a vulnerability category scanned
  - Weak cryptography (MD5/SHA1, hardcoded IVs) is a vulnerability category scanned
  - Quick depth scan takes ~30s and provides a surface scan covering eval/exec and obvious secrets
  - Standard depth scan takes ~2 min and provides full pattern matching with severity ratings
  - Deep depth scan takes ~5 min and provides multi-pass review with OWASP mapping and fix suggestions
  - Consider running a scan before releasing a new version
  - Consider running a scan after adding code that handles files or user input
  - Consider running a scan when pulling in a new dependency
  - Consider running a scan as a CI gate on pull requests

Unsupported:


**Claims identified by judge (thinking on):**

Supported:
  - The `/security-audit` command is used directly in a Claude Code conversation
  - The syntax for the Claude Code skill is `/security-audit <path or directory to scan>`
  - The CLI command to run a vulnerability scan is `attune workflow run security-audit --path "src/"`
  - The CLI workflow produces severity-grouped findings with CWE identifiers
  - The scan detects code injection vulnerabilities: `eval()`, `exec()`, and `compile()` on untrusted input
  - The scan detects path traversal: file operations that don't validate the path first
  - The scan detects hardcoded secrets: API keys, tokens, and passwords committed to source
  - The scan detects SQL/command injection: string concatenation in queries or shell commands
  - The scan detects SSRF: HTTP requests to URLs controlled by user input
  - The scan detects weak cryptography: MD5/SHA1, hardcoded IVs
  - Quick scan depth takes ~30s and covers surface scan — eval/exec, obvious secrets
  - Standard scan depth takes ~2 min and provides full pattern matching with severity ratings
  - Deep scan depth takes ~5 min and provides multi-pass review with OWASP mapping and fix suggestions
  - A good time to run a security audit is before releasing a new version
  - A good time to run a security audit is after adding code that handles files or user input
  - A good time to run a security audit is when pulling in a new dependency
  - A good time to run a security audit is as a CI gate on pull requests

Unsupported:


### Your labels

```yaml
id: gq-011
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

## gq-016 — `find bugs in my code`

### Retrieved context

> <passage>
> [P1] source: concepts/tool-bug-predict.md
> ---
> type: concept
> name: tool-bug-predict
> tags: [security, bugs, scanning]
> aliases: [risky code changes, identify files that break, runtime errors, spot bugs before deploy, risk assessment]
> source: plugin/skills/bug-predict/SKILL.md
> ---
>
> # Bug Prediction
>
> Bug prediction scans your codebase for the patterns that
> historically cause production incidents. Instead of waiting
> for bugs to surface, it analyzes code structure, complexity,
> and known anti-patterns to tell you where failures are most
> likely to happen next.
>
> ## What it predicts
>
> The scanner looks for three categories of risk, each with
> different confidence levels and severity:
>
> | Pattern | Severity | Confidence | What to look for |
> |---------|----------|------------|------------------|
> | **dangerous_eval** | HIGH | High | `eval()`, `exec()`, `compile()` on any input — code injection vectors |
> | **broad_exception** | MEDIUM | Medium | Bare `except:`, unlogged `except Exception:` — errors silently swallowed |
> | **incomplete_code** | LOW | Low | TODO, FIXME, HACK, XXX comments — unfinished code paths that break under edge cases |
>
> ## Risk factors beyond patterns
>
> The scanner also weighs contextual signals that increase
> bug likelihood:
>
> - **Cyclomatic complexity** — deeply nested conditionals
>   and long function bodies correlate with higher defect
>   rates
> - **Change frequency** — files modified often ("hot" files)
>   are more likely to contain regressions
> - **Code smells** — functions over 50 lines, classes with
>   too many methods, duplicated logic across modules
>
> ## Smart false-positive filtering
>
> Not every match is a real bug. The scanner automatically
> suppresses known-safe patterns:
>
> - `eval()` inside test fixture strings (test data, not
>   executable code)
> - JavaScript `regex.exec()` method calls (safe, not
>   Python's `exec()`)
> - Broad exceptions with `# INTENTIONAL:` comments and
>   `# noqa: BLE001` markers
> - Version detection fallbacks, cleanup/teardown code, and
>   optional feature guards
>
> ## When you'd use it
>
> - Before merging a large PR — catch patterns humans miss
> - During code review — focus human attention on real risks
> - After onboarding unfamiliar code — map risk hotspots fast
> - As a periodic health check on high-churn modules
> - Before a release — verify no new high-severity patterns
>   crept in
>
> ## Want to learn more?
>
> - Say **"tell me more"** for step-by-step instructions
> - Say **"what is security audit?"** for vulnerability scanning
> - Say **"what is code quality?"** for a broader code review
> </passage>
>
> <passage>
> [P2] source: quickstarts/skill-code-quality.md
> ---
> type: quickstart
> name: skill-code-quality
> tags: [skill, claude-code]
> source: plugin/skills/code-quality/SKILL.md
> ---
>
> # Quickstart: Use /code-quality
>
> Code review and bug prediction to find quality issues, style violations, and likely bugs.
>
> ```
> /code-quality <path or directory to review>
> ```
>
> **Result:** Structured results in your Claude Code conversation.
>
> **Next:** See full reference: `attune help-docs ref-skill-code-quality`
>
> ## Related Topics
>
> _No related topics yet._
> </passage>
>
> <passage>
> [P3] source: concepts/tool-code-quality.md
> ---
> type: concept
> name: tool-code-quality
> tags: [review, quality, linting, skill]
> source: plugin/skills/code-quality/SKILL.md
> ---
>
> # Code Quality Review
>
> A code quality review looks at your code from multiple
> angles at once — style, correctness, likely bugs, and
> structural health. Instead of running a linter, then a
> bug scanner, then a reviewer, and reading three separate
> reports, you get one pass with a unified score.
>
> ## What it catches
>
> | Category | Examples | Severity | Auto-fixable? |
> |----------|---------|----------|---------------|
> | **Style** | Unused imports, line length, naming | Low | Often yes |
> | **Correctness** | Unreachable code, wrong return types | High | Sometimes |
> | **Likely bugs** | Broad exceptions, mutable defaults, race conditions | High | No |
> | **Structural** | High coupling, god classes, circular deps | Medium | No |
> | **Security overlap** | eval/exec, unvalidated paths (deep only) | Critical | No |
>
> ## When you'd use it
>
> Before opening a pull request — catch issues before
> reviewers do. After a large refactor — verify nothing
> degraded. When inheriting unfamiliar code — get a
> quick read on its health. Or any time you want a
> single number that summarizes code quality for a file
> or directory.
>
> ## How thorough it is
>
> | Depth | What it covers | Best for |
> |-------|---------------|----------|
> | **Quick** | Style and formatting only | Pre-commit check, fast feedback |
> | **Standard** | + Logic errors, likely bugs | Pull requests, regular development |
> | **Deep** | + Security, architecture, test gaps | Pre-release, critical modules |
>
> The default is standard.
>
> ## What to expect
>
> When you ask for a code quality review, you'll be asked
> a couple of questions first to scope the work — which
> files to review and how deep to go. This keeps the
> results focused on what you actually care about instead
> of dumping everything at once.
>
> ## Want to learn more?
>
> - Say **"tell me more"** for step-by-step instructions
> - Say **"what is security audit?"** for dedicated security analysis
> - Say **"tell me about bug prediction"** to predict where failures happen
> </passage>

### Answer

## Finding Bugs in Your Code

There are a couple of tools well-suited for this:

---

### Option 1: Bug Prediction

Bug prediction scans your codebase for patterns that historically cause production incidents, analyzing code structure, complexity, and known anti-patterns to identify where failures are most likely to happen [P1].

It looks for three categories of risk [P1]:

| Pattern | Severity | What it finds |
|---------|----------|---------------|
| **dangerous_eval** | HIGH | `eval()`, `exec()`, `compile()` on any input |
| **broad_exception** | MEDIUM | Bare `except:` or unlogged exceptions that silently swallow errors |
| **incomplete_code** | LOW | TODO, FIXME, HACK, XXX comments — unfinished code paths |

It also weighs contextual signals like cyclomatic complexity, change frequency, and code smells (e.g., functions over 50 lines) [P1].

---

### Option 2: Code Quality Review

A code quality review covers bugs as part of a broader pass — style, correctness, likely bugs, and structural health — all in one unified report [P3].

For bug-finding specifically, it catches things like broad exceptions, mutable defaults, and race conditions (all rated **High** severity), as well as unreachable code and wrong return types [P3].

---

### How to Run It

Use the `/code-quality` command with your file or directory path [P2]:

```
/code-quality <path or directory to review>
```

The default depth is **standard**, which covers logic errors and likely bugs — a good fit for everyday bug-finding [P3].


### Judge verdicts (off → on)

| | Score | Supported | Unsupported |
| - | -: | -: | -: |
| Thinking off | 1.000 | 15 | 0 |
| Thinking on  | 1.000 | 15 | 0 |

**Reasoning (thinking off):**

> Every factual claim in the answer maps directly to an explicit statement in the retrieved passages. Bug prediction details (patterns, severities, contextual signals) come from P1. Code quality review details (categories, severities, examples) come from P3. The /code-quality command syntax and the default depth of "standard" come from P2 and P3 respectively. No outside knowledge, invented details, or unsupported inferences were identified.

**Reasoning (thinking on):**

> Every atomic factual claim in the answer maps directly to explicit statements in the retrieved passages. The bug prediction risk categories, severities, and contextual signals all come verbatim from P1. The code quality categories, severities, and depth levels come directly from P3's tables and prose. The /code-quality command syntax comes from P2. The claim about the default depth being "standard" is explicitly stated in P3 ("The default is standard"). No outside knowledge, invented details, or unsupported inferences were detected.

**Claims identified by judge (thinking off):**

Supported:
  - Bug prediction scans your codebase for patterns that historically cause production incidents.
  - Bug prediction analyzes code structure, complexity, and known anti-patterns to identify where failures are most likely to happen.
  - Bug prediction looks for three categories of risk: dangerous_eval, broad_exception, and incomplete_code.
  - dangerous_eval has HIGH severity and finds eval(), exec(), compile() on any input.
  - broad_exception has MEDIUM severity and finds bare except: or unlogged exceptions that silently swallow errors.
  - incomplete_code has LOW severity and finds TODO, FIXME, HACK, XXX comments — unfinished code paths.
  - Bug prediction weighs contextual signals like cyclomatic complexity, change frequency, and code smells.
  - Code smells include functions over 50 lines.
  - A code quality review covers style, correctness, likely bugs, and structural health in one unified report.
  - Code quality review catches broad exceptions, mutable defaults, and race conditions, all rated High severity.
  - Code quality review catches unreachable code and wrong return types.
  - The /code-quality command is used with a file or directory path.
  - The /code-quality command syntax is: /code-quality <path or directory to review>.
  - The default depth is standard.
  - The standard depth covers logic errors and likely bugs.

Unsupported:


**Claims identified by judge (thinking on):**

Supported:
  - Bug prediction scans your codebase for patterns that historically cause production incidents.
  - The scanner looks for three categories of risk.
  - Dangerous eval is HIGH severity and covers eval(), exec(), compile() on any input — code injection vectors.
  - Broad exceptions is MEDIUM severity and covers bare except: or unlogged except Exception: that silently swallow errors.
  - Incomplete code is LOW severity and covers TODO, FIXME, HACK, XXX comments indicating unfinished code paths.
  - The scanner weighs contextual signals like cyclomatic complexity, change frequency, and code smells.
  - Functions over 50 lines is an example of a code smell.
  - A code quality review looks at your code from multiple angles at once — style, correctness, likely bugs, and structural health.
  - Code quality review gives one unified pass instead of multiple separate reports.
  - Likely bugs category includes broad exceptions, mutable defaults, and race conditions at High severity.
  - Correctness issues category includes unreachable code and wrong return types at High severity.
  - Security overlap category includes eval/exec and unvalidated paths at Critical severity, for deep scan only.
  - The /code-quality command takes a path or directory to review.
  - The default scan depth is standard, which covers logic errors and likely bugs.
  - A deep scan additionally covers security and architecture issues.

Unsupported:


### Your labels

```yaml
id: gq-016
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

## gq-018 — `check code quality`

### Retrieved context

> <passage>
> [P1] source: concepts/tool-code-quality.md
> ---
> type: concept
> name: tool-code-quality
> tags: [review, quality, linting, skill]
> source: plugin/skills/code-quality/SKILL.md
> ---
>
> # Code Quality Review
>
> A code quality review looks at your code from multiple
> angles at once — style, correctness, likely bugs, and
> structural health. Instead of running a linter, then a
> bug scanner, then a reviewer, and reading three separate
> reports, you get one pass with a unified score.
>
> ## What it catches
>
> | Category | Examples | Severity | Auto-fixable? |
> |----------|---------|----------|---------------|
> | **Style** | Unused imports, line length, naming | Low | Often yes |
> | **Correctness** | Unreachable code, wrong return types | High | Sometimes |
> | **Likely bugs** | Broad exceptions, mutable defaults, race conditions | High | No |
> | **Structural** | High coupling, god classes, circular deps | Medium | No |
> | **Security overlap** | eval/exec, unvalidated paths (deep only) | Critical | No |
>
> ## When you'd use it
>
> Before opening a pull request — catch issues before
> reviewers do. After a large refactor — verify nothing
> degraded. When inheriting unfamiliar code — get a
> quick read on its health. Or any time you want a
> single number that summarizes code quality for a file
> or directory.
>
> ## How thorough it is
>
> | Depth | What it covers | Best for |
> |-------|---------------|----------|
> | **Quick** | Style and formatting only | Pre-commit check, fast feedback |
> | **Standard** | + Logic errors, likely bugs | Pull requests, regular development |
> | **Deep** | + Security, architecture, test gaps | Pre-release, critical modules |
>
> The default is standard.
>
> ## What to expect
>
> When you ask for a code quality review, you'll be asked
> a couple of questions first to scope the work — which
> files to review and how deep to go. This keeps the
> results focused on what you actually care about instead
> of dumping everything at once.
>
> ## Want to learn more?
>
> - Say **"tell me more"** for step-by-step instructions
> - Say **"what is security audit?"** for dedicated security analysis
> - Say **"tell me about bug prediction"** to predict where failures happen
> </passage>
>
> <passage>
> [P2] source: quickstarts/skill-code-quality.md
> ---
> type: quickstart
> name: skill-code-quality
> tags: [skill, claude-code]
> source: plugin/skills/code-quality/SKILL.md
> ---
>
> # Quickstart: Use /code-quality
>
> Code review and bug prediction to find quality issues, style violations, and likely bugs.
>
> ```
> /code-quality <path or directory to review>
> ```
>
> **Result:** Structured results in your Claude Code conversation.
>
> **Next:** See full reference: `attune help-docs ref-skill-code-quality`
>
> ## Related Topics
>
> _No related topics yet._
> </passage>
>
> <passage>
> [P3] source: quickstarts/run-code-review.md
> ---
> type: quickstart
> name: run-code-review
> tags: [workflow, code-quality]
> source: src/attune/cli_minimal.py
> ---
>
> # Quickstart: Run a code review
>
> Get AI-powered code quality analysis.
>
> ```
> attune workflow run code-review --path "src/"
> ```
>
> **Result:** Quality findings grouped by severity.
>
> **Next:** Follow up with `/smart-test` to generate tests for flagged areas.
>
> ## Related Topics
>
> _No related topics yet._
> </passage>

### Answer

## Checking Code Quality

Here are the ways you can run a code quality review:

### Option 1: Claude Code Skill
```
/code-quality <path or directory to review>
```
This gives you structured results directly in your Claude Code conversation [P2].

### Option 2: Attune CLI Workflow
```
attune workflow run code-review --path "src/"
```
This returns quality findings grouped by severity [P3].

---

### What Gets Checked

A code quality review covers multiple angles at once — style, correctness, likely bugs, and structural health [P1]. Here's a breakdown:

| Category | Examples | Severity |
|----------|---------|----------|
| **Style** | Unused imports, line length, naming | Low |
| **Correctness** | Unreachable code, wrong return types | High |
| **Likely bugs** | Broad exceptions, mutable defaults, race conditions | High |
| **Structural** | High coupling, god classes, circular deps | Medium |
| **Security overlap** | eval/exec, unvalidated paths (deep only) | Critical |

[P1]

### Depth Levels

You can choose how thorough the review is [P1]:

- **Quick** — Style and formatting only (good for pre-commit checks)
- **Standard** — Adds logic errors and likely bugs *(this is the default)*
- **Deep** — Adds security, architecture, and test gaps (best for pre-release)

[P1]

### What to Expect

When you request a code quality review, you'll be asked a couple of questions first to scope the work — which files to review and how deep to go [P1]. After a code review, you can follow up with `/smart-test` to generate tests for flagged areas [P3].


### Judge verdicts (off → on)

| | Score | Supported | Unsupported |
| - | -: | -: | -: |
| Thinking off | 1.000 | 18 | 0 |
| Thinking on  | 1.000 | 18 | 0 |

**Reasoning (thinking off):**

> Every factual claim in the answer maps directly to content in the retrieved passages. The two options (Claude Code skill from P2, Attune CLI from P3), the full category table (P1), the depth levels table (P1, including the 'default is standard' note), the scoping questions expectation (P1), and the /smart-test follow-up suggestion (P3) are all explicitly stated in the passages. No invented details, flags, or workflow names appear in the answer beyond what the passages contain.

**Reasoning (thinking on):**

> Every atomic factual claim in the answer is directly and explicitly supported by at least one of the retrieved passages. The /code-quality command and Claude Code conversation output come from P2. The attune CLI command, severity grouping, and /smart-test follow-up come from P3. All category names, example issues, severity levels, depth tiers, defaults, and scoping-question details come verbatim or near-verbatim from P1's tables and prose. No details appear to have been invented or inferred beyond what the passages state.

**Claims identified by judge (thinking off):**

Supported:
  - You can run a code quality review using /code-quality <path or directory to review>
  - The /code-quality skill gives structured results directly in your Claude Code conversation
  - You can run a code quality review using: attune workflow run code-review --path "src/"
  - The attune CLI workflow returns quality findings grouped by severity
  - A code quality review covers style, correctness, likely bugs, and structural health
  - Style issues include unused imports, line length, and naming, and are Low severity
  - Correctness issues include unreachable code and wrong return types, and are High severity
  - Likely bugs include broad exceptions, mutable defaults, and race conditions, and are High severity
  - Structural issues include high coupling, god classes, and circular deps, and are Medium severity
  - Security overlap issues include eval/exec and unvalidated paths (deep only), and are Critical severity
  - Quick depth covers style and formatting only
  - Quick depth is good for pre-commit checks
  - Standard depth adds logic errors and likely bugs
  - Standard depth is the default
  - Deep depth adds security, architecture, and test gaps
  - Deep depth is best for pre-release
  - When you request a code quality review, you'll be asked a couple of questions first to scope the work — which files to review and how deep to go
  - After a code review, you can follow up with /smart-test to generate tests for flagged areas

Unsupported:


**Claims identified by judge (thinking on):**

Supported:
  - The Claude Code skill command is /code-quality <path or directory to review>
  - The Claude Code skill delivers structured results directly in your Claude Code conversation
  - The Attune CLI command is: attune workflow run code-review --path "src/"
  - The CLI returns quality findings grouped by severity
  - You can follow up with /smart-test to generate tests for flagged areas
  - A code quality review covers style, correctness, likely bugs, and structural health
  - A code quality review gives a unified score instead of separate reports
  - Style issues (unused imports, naming, line length) have Low severity
  - Correctness issues (unreachable code, wrong return types) have High severity
  - Likely bugs (broad exceptions, race conditions) have High severity
  - Structural issues (high coupling, circular deps) have Medium severity
  - Security overlap (eval/exec, unvalidated paths) has Critical severity
  - Quick depth is best for pre-commit checks and fast feedback
  - Standard depth is best for pull requests and regular development
  - Standard is the default depth
  - Deep depth is best for pre-release and critical modules
  - When invoking the skill, you will be asked scoping questions about which files to review and how deep to go
  - The scoping questions keep results focused on what you actually care about

Unsupported:


### Your labels

```yaml
id: gq-018
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

## gq-019 — `create a release`

### Retrieved context

> <passage>
> [P1] source: concepts/tool-release-prep.md
> ---
> type: concept
> name: tool-release-prep
> tags: [release, publishing, ci]
> aliases: [publish to PyPI, PyPI release, cut a release, ship a version, package publishing]
> source: plugin/skills/release-prep/SKILL.md
> ---
>
> # Release Prep
>
> Release prep runs a preflight checklist across your
> project before you publish. It checks health, security,
> changelog, dependencies, and version consistency, then
> produces a go/no-go assessment telling you whether the
> release is safe to ship.
>
> ## What it checks
>
> | Check area | What it verifies | Blocking? |
> |------------|------------------|-----------|
> | **Health** | Tests pass, lint clean, coverage above threshold | Yes if tests fail |
> | **Security** | No new CVEs, no eval/exec, secrets scan clean | Yes if critical findings |
> | **Changelog** | Entry exists for this version, date is current | Yes if missing |
> | **Dependencies** | Pinned versions, no known vulnerabilities, compatible ranges | Yes if vulnerable dep |
> | **Version** | Semver bump matches changes, pyproject.toml updated, dist builds cleanly | Yes if version mismatch |
>
> ## The go/no-go decision
>
> After all checks complete, you get a single verdict:
>
> - **GO** -- every check passed or has only advisory
>   warnings. Safe to tag, build, and publish.
> - **NO-GO** -- one or more blocking issues found. The
>   report lists exactly what to fix before retrying.
>
> The assessment is conservative. A stale changelog entry
> or a failing test is enough to block. Better to catch
> it here than after the package is on PyPI.
>
> ## When to use it
>
> - Before bumping the version in pyproject.toml
> - Before running `twine upload` or publishing to PyPI
> - After merging a large feature branch to main
> - As the final gate before tagging a release
> - When you're unsure whether the codebase is release-ready
>
> ## Want to learn more?
>
> - Say **"how do I run release prep?"** for step-by-step
>   instructions
> - Say **"tell me more"** for the full reference with
>   all check areas, scoring, and configuration
> - Say **"run a security audit"** to scan for
>   vulnerabilities separately
> - Say **"check my test coverage"** to look at tests
>   before releasing
> </passage>
>
> <passage>
> [P2] source: quickstarts/skill-release-prep.md
> ---
> type: quickstart
> name: skill-release-prep
> tags: [skill, claude-code]
> source: plugin/skills/release-prep/SKILL.md
> ---
>
> # Quickstart: Use /release-prep
>
> Pre-release preparation with health checks, security audit, changelog validation, version bumps, and dependency audits.
>
> ```
> /release-prep <version or 'check'>
> ```
>
> **Result:** Structured results in your Claude Code conversation.
>
> **Next:** See full reference: `attune help-docs ref-skill-release-prep`
>
> ## Related Topics
>
> _No related topics yet._
> </passage>
>
> <passage>
> [P3] source: tasks/use-release-prep.md
> ---
> type: task
> name: use-release-prep
> tags: [release, skill, task]
> source: plugin/skills/release-prep/SKILL.md
> ---
>
> # How to Run Release Prep
>
> ## Quick start
>
> The fastest way: just say what you want.
>
> ```
> is this project ready to release?
> ```
>
> Or use the skill directly:
>
> ```
> /release-prep check
> ```
>
> That kicks off the full preflight checklist. You'll get a
> go/no-go assessment in about two minutes.
>
> ## The guided flow
>
> When you invoke release prep, it asks two questions
> before running:
>
> 1. **What version?** -- "What version are you releasing?
>    Or should I check the current version and suggest the
>    next bump?"
>
> 2. **What stage?** -- "Full release prep, or a specific
>    check?"
>    - **Full** -- health, security, changelog, deps, version
>    - **Prep check** -- just verify readiness without
>      version bump
>    - **Changelog only** -- validate the changelog entry
>    - **Security only** -- run the security gate
>
> Answer in natural language. If you say "just check
> everything," it runs the full assessment.
>
> ## What you'll type
>
> | Goal | What to say |
> |------|-------------|
> | Full preflight | `/release-prep 5.6.0` |
> | Check without version bump | `/release-prep check` |
> | Just the security gate | `run the security check for release` |
> | Just changelog validation | `check if the changelog is ready for release` |
>
> ## Reading the assessment
>
> Results come back as a structured report:
>
> ```
> Release Readiness Assessment
> Verdict: NO-GO
> Version: 5.5.0 -> 5.6.0
>
> Health          PASS   Tests: 15,482 passing, coverage 87%
> Security        PASS   No critical findings, 2 advisory
> Changelog       FAIL   No entry for v5.6.0
> Dependencies    PASS   All pinned, no known CVEs
> Version         PASS   Semver valid, dist builds clean
>
> Blockers (1)
>   Changelog — No entry for v5.6.0 in CHANGELOG.md
>
> Recommendations
>   1. Add a v5.6.0 section to CHANGELOG.md
>   2. Re-run release prep after fixing
> ```
>
> Each check area shows PASS or FAIL with a one-line
> summary. Blockers are listed separately so you know
> exactly what to fix.
>
> ## Handling blockers
>
> After reviewing the report:
>
> - **"Fix the blockers"** -- the skill will attempt to
>   resolve each issue (update changelog, fix lint, etc.)
> - **"Update the changelog"** -- generates a changelog
>   entry from recent commits
> - **"Re-run the check"** -- runs the assessment again
>   after you've made fixes
> - **"Tag and publish"** -- if the verdict is GO, proceeds
>   to tag and upload
>
> ## Want to learn more?
>
> - Say **"tell me more"** for the full reference with
>   every check area, scoring, and configuration
> - Say **"what is release prep?"** to go back to the
>   overview
> - Say **"run a security audit"** to scan for
>   vulnerabilities separately
> </passage>

### Answer

## Creating a Release

Here's how to run release prep before publishing your package:

### Quick Start

The fastest way is to say what you want in plain language:

```
is this project ready to release?
```

Or invoke the skill directly:

```
/release-prep check
```

This kicks off the full preflight checklist and you'll get a go/no-go assessment in about two minutes [P3].

---

### What Gets Checked

Release prep runs a preflight checklist that covers [P1]:

| Check Area | What It Verifies | Blocking? |
|---|---|---|
| **Health** | Tests pass, lint clean, coverage above threshold | Yes if tests fail |
| **Security** | No new CVEs, no eval/exec, secrets scan clean | Yes if critical findings |
| **Changelog** | Entry exists for this version, date is current | Yes if missing |
| **Dependencies** | Pinned versions, no known vulnerabilities | Yes if vulnerable dep |
| **Version** | Semver bump matches changes, pyproject.toml updated | Yes if version mismatch |

---

### Specifying a Version

If you know your target version, run:

```
/release-prep 5.6.0
```

The skill will ask two questions before running: **what version** you're releasing, and **what stage** (full release prep, prep check only, changelog only, or security only) [P3].

---

### Reading the Result

After checks complete, you get a single verdict [P1]:

- ✅ **GO** — every check passed or has only advisory warnings. Safe to tag, build, and publish.
- ❌ **NO-GO** — one or more blocking issues found. The report lists exactly what to fix before retrying.

Blockers are listed separately so you know exactly what to fix [P3].

---

### After Fixing Blockers

Once issues are resolved, you can say [P3]:
- **"Re-run the check"** — runs the assessment again
- **"Tag and publish"** — if the verdict is GO, proceeds to tag and upload


### Judge verdicts (off → on)

| | Score | Supported | Unsupported |
| - | -: | -: | -: |
| Thinking off | 1.000 | 18 | 0 |
| Thinking on  | 1.000 | 18 | 0 |

**Reasoning (thinking off):**

> Every atomic factual claim in the answer is directly and explicitly supported by the retrieved passages. The quick start commands, check areas and their blocking conditions, the two-question guided flow, the GO/NO-GO verdict descriptions, and the post-blocker actions are all verbatim or near-verbatim from P1, P2, and P3. No claims appear to rely on outside knowledge or invented details. The answer is a faithful synthesis of the passages with no hallucinations detected.

**Reasoning (thinking on):**

> Every atomic factual claim in the answer maps directly to explicit text in the retrieved passages. The five check areas and their descriptions come verbatim from P1's table. The GO/NO-GO definitions are quoted directly from P1. The natural-language invocation phrases and their effects come from P3's "Handling blockers" section. The /release-prep command syntax and the example version (5.6.0) appear in P2 and P3. The two-minute estimate is stated literally in P3. No outside knowledge, invented flags, or inferred details were introduced.

**Claims identified by judge (thinking off):**

Supported:
  - The fastest way to start a release is to say 'is this project ready to release?' in plain language
  - You can invoke the skill directly with '/release-prep check'
  - Running '/release-prep check' kicks off the full preflight checklist
  - You get a go/no-go assessment in about two minutes
  - Release prep checks Health: tests pass, lint clean, coverage above threshold, blocking if tests fail
  - Release prep checks Security: no new CVEs, no eval/exec, secrets scan clean, blocking if critical findings
  - Release prep checks Changelog: entry exists for this version, date is current, blocking if missing
  - Release prep checks Dependencies: pinned versions, no known vulnerabilities, blocking if vulnerable dep
  - Release prep checks Version: semver bump matches changes, pyproject.toml updated, blocking if version mismatch
  - You can specify a version by running '/release-prep 5.6.0'
  - The skill asks what version you're releasing before running
  - The skill asks what stage (full release prep, prep check only, changelog only, or security only) before running
  - After all checks complete, you get a single verdict
  - GO means every check passed or has only advisory warnings and it is safe to tag, build, and publish
  - NO-GO means one or more blocking issues found and the report lists exactly what to fix before retrying
  - Blockers are listed separately so you know exactly what to fix
  - You can say 'Re-run the check' to run the assessment again after making fixes
  - You can say 'Tag and publish' and if the verdict is GO, it proceeds to tag and upload

Unsupported:


**Claims identified by judge (thinking on):**

Supported:
  - The fastest way to begin is to say: 'is this project ready to release?'
  - You can invoke the skill directly with /release-prep check
  - This kicks off the full preflight checklist
  - Returns a go/no-go assessment in about two minutes
  - Release prep runs checks across five areas
  - Health check verifies: tests pass, lint clean, coverage above threshold
  - Security check verifies: no new CVEs, no eval/exec, secrets scan clean
  - Changelog check verifies: entry exists for this version, date is current
  - Dependencies check verifies: pinned versions, no known vulnerabilities
  - Version check verifies: semver bump matches changes, pyproject.toml updated, dist builds cleanly
  - You'll get either a GO or NO-GO verdict
  - GO means every check passed or has only advisory warnings; safe to tag, build, and publish
  - NO-GO means one or more blocking issues found; the report lists exactly what to fix before retrying
  - 'Fix the blockers' causes the skill to attempt to resolve each issue
  - 'Update the changelog' generates a changelog entry from recent commits
  - 'Re-run the check' runs the assessment again after fixes
  - 'Tag and publish' — if the verdict is GO, proceeds to tag and upload
  - Use /release-prep <version> (e.g. /release-prep 5.6.0) for a full preflight tied to a specific version

Unsupported:


### Your labels

```yaml
id: gq-019
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
