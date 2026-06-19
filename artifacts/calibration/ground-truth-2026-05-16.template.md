# Faithfulness ground-truth labels

Source artifact: `artifacts/calibration/thinking-2026-05-16.json`
Selected 32 queries (15 shifted + 2 controls + 15 random; shift threshold = 0.00; seed = 42).

## How to label

For each query below, fill in the YAML block under "Your labels". The scoring script compares your `faithfulness_score` against both judge passes (off / on) to decide whether thinking-on aligns better with ground truth than thinking-off. See `docs/rag/faithfulness-thinking-calibration.md` for the decision matrix this feeds.

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

You can manage your documentation pipeline using the **Doc Orchestrator** tool, which handles end-to-end documentation maintenance automatically.

### What It Does

The Doc Orchestrator performs the full documentation maintenance workflow: it scouts gaps, prioritizes, generates, and updates docs [P1].

### How to Use It

Call the tool with:

```python
doc_orchestrator()
```

It accepts one optional parameter:

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `path` | string | Project root path | `.` |

So to run it on a specific project directory, you would pass the path accordingly [P1].

### Workflow Group

The Doc Orchestrator belongs to the **workflow** group of tools, alongside related tools such as:

- **Security Audit** — runs a security audit workflow on the codebase
- **Bug Predict** — analyzes code patterns and predicts bugs
- **Code Review** — provides comprehensive code quality review

[P1]

### Tip: Integrating with CI/CD

If you want documentation maintenance to run automatically on every push, you could incorporate it into a CI/CD pipeline. A typical pipeline can be triggered on every push or pull request to your main branch, with jobs running sequentially as gates [P2, P3]. This would allow documentation updates to happen as a background, automated event rather than a manual process [P2].


### Judge verdicts (off → on)

| | Score | Supported | Unsupported |
| - | -: | -: | -: |
| Thinking off | 0.933 | 14 | 1 |
| Thinking on  | 0.812 | 13 | 3 |

**Reasoning (thinking off):**

> The vast majority of claims in the answer are directly supported by P1, which describes the Doc Orchestrator tool, its parameters, defaults, workflow group, and related tools. The CI/CD concepts (pipeline triggers, sequential gates) are supported by P2 and P3. The one unsupported claim is the specific suggestion that the Doc Orchestrator should be integrated into CI/CD so that documentation updates become automated background events — while the CI/CD passages do describe automation as making events happen "in the background," they do so in the context of code/deployment, not the Doc Orchestrator specifically. No passage explicitly connects the Doc Orchestrator to CI/CD integration, so that specific recommendation goes beyond what the passages state.

**Reasoning (thinking on):**

> The answer is largely faithful to the retrieved passages. Most claims about the Doc Orchestrator tool — its description, usage syntax, parameter table, and related tools — are directly supported by P1. The claim about CI pipeline stages acting as gates is directly supported by P2. However, three claims are unsupported: (1) the specific extended call syntax `doc_orchestrator(path="/your/project/root")` is never shown in the passages — P1 only shows `doc_orchestrator()` with no argument; (2) the suggestion to integrate doc_orchestrator into a CI/CD pipeline is an inference that no passage makes — P1 never mentions CI/CD and P2/P3 never mention doc_orchestrator; and (3) the analogy of adding a documentation step "similarly to how linting or security scanning is added in GitHub Actions" is the answer's own editorial inference, not stated in any passage.

**Claims identified by judge (thinking off):**

Supported:
  - The Doc Orchestrator tool handles end-to-end documentation maintenance automatically.
  - The Doc Orchestrator scouts gaps, prioritizes, generates, and updates docs.
  - The tool is called with doc_orchestrator().
  - The tool accepts a parameter named 'path'.
  - The 'path' parameter is of type string.
  - The 'path' parameter describes the project root path.
  - The default value for 'path' is '.'.
  - The Doc Orchestrator belongs to the 'workflow' group of tools.
  - Security Audit is a related tool that runs a security audit workflow on the codebase.
  - Bug Predict is a related tool that analyzes code patterns and predicts bugs.
  - Code Review is a related tool that provides comprehensive code quality review.
  - A typical CI/CD pipeline can be triggered on every push or pull request to the main branch.
  - CI/CD pipeline jobs run sequentially as gates.
  - CI/CD turns documentation/deployment updates into boring, automatic events that happen in the background while you keep coding.

Unsupported:
  - Incorporating the Doc Orchestrator into a CI/CD pipeline would allow documentation updates to happen as a background, automated event rather than a manual process (the passages do not state this about the Doc Orchestrator specifically; the CI/CD background-event claim is made in the context of general code deployment, not documentation maintenance).

**Claims identified by judge (thinking on):**

Supported:
  - The Doc Orchestrator tool handles end-to-end documentation maintenance.
  - The Doc Orchestrator scouts gaps, prioritizes, generates, and updates docs.
  - Basic usage of the tool is `doc_orchestrator()`.
  - The tool accepts one parameter.
  - The `path` parameter is of type string.
  - The `path` parameter's description is 'Project root path'.
  - The default value of the `path` parameter is `.`
  - The Doc Orchestrator belongs to a 'workflow' group of tools.
  - Security Audit runs a security audit workflow on a codebase.
  - Bug Predict analyzes code patterns and predicts bugs.
  - Code Review provides comprehensive code quality review.
  - Security Audit, Bug Predict, and Code Review are listed as related tools to Doc Orchestrator.
  - A typical CI pipeline runs stages in order, where each stage acts as a gate — if it fails, later stages don't run.

Unsupported:
  - You can call the tool on a specific directory using `doc_orchestrator(path="/your/project/root")`.
  - You can integrate doc_orchestrator into a CI/CD workflow to run documentation maintenance automatically as part of a pipeline.
  - You could add a documentation step similarly to how linting or security scanning is added as a job in GitHub Actions.

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

### What It Does

The `/doc-gen` skill generates documentation from source code, including:
- **Docstrings**
- **READMEs**
- **API references**

[P1]

### How to Use It

Run the following command in your Claude Code conversation:

```
/doc-gen <path or module to document>
```

Replace `<path or module to document>` with the path or module you want documented [P1].

### What to Expect

After running the command, you'll get **structured results directly in your Claude Code conversation** [P1].

### Learn More

For the full reference guide, you can run:

```
attune help-docs ref-skill-doc-gen
```

[P1]

---

> **Tip:** If you've recently done a large refactor or are working with unfamiliar code, you may also want to run a **code quality review** (`/code-quality`) alongside documentation generation to verify the overall health of your code [P3].


### Judge verdicts (off → on)

| | Score | Supported | Unsupported |
| - | -: | -: | -: |
| Thinking off | 0.917 | 11 | 1 |
| Thinking on  | 0.818 | 9 | 2 |

**Reasoning (thinking off):**

> Nearly all claims in the answer are directly and explicitly supported by passage P1, which describes the /doc-gen skill, its command syntax, its outputs (docstrings, READMEs, API references), and the attune help-docs reference command. The tip about /code-quality being useful after a large refactor and with unfamiliar code is supported by P3. However, the suggestion to run /code-quality *alongside* documentation generation to verify health is a slight inferential leap beyond what either passage says — P3 mentions /code-quality in those scenarios but never connects it to running alongside doc-gen. This is a minor unsupported inference rather than a hallucination, but strictly speaking it goes beyond what the passages explicitly state.

**Reasoning (thinking on):**

> The core claims about /doc-gen — its purpose, command syntax, output types (docstrings, READMEs, API references), and the attune help-docs reference command — are all directly and explicitly supported by P1. The /code-quality capability for scanning style issues and deprecated patterns is supported by P2, but the answer reframes it as something to do "before or after documenting," a context not present in the passages (P2 only discusses it in the migration context). Most notably, the answer invents a workflow link between /doc-gen and /smart-test//refactor that has no basis in any passage; P2 mentions those tools only in the context of migration planning, never in connection with documentation generation.

**Claims identified by judge (thinking off):**

Supported:
  - The /doc-gen skill can be used to create documentation from source code
  - The /doc-gen skill generates docstrings
  - The /doc-gen skill generates READMEs
  - The /doc-gen skill generates API references
  - The command syntax is: /doc-gen <path or module to document>
  - The command is run in a Claude Code conversation
  - After running the command, you get structured results in your Claude Code conversation
  - The full reference guide can be accessed by running: attune help-docs ref-skill-doc-gen
  - A code quality review can be run with /code-quality
  - A code quality review is useful after a large refactor
  - A code quality review is useful when working with unfamiliar code

Unsupported:
  - Running /code-quality alongside documentation generation helps verify the overall health of your code

**Claims identified by judge (thinking on):**

Supported:
  - You can use the /doc-gen skill to create documentation from your source code
  - The /doc-gen skill generates documentation from source code
  - The /doc-gen skill generates docstrings
  - The /doc-gen skill generates READMEs
  - The /doc-gen skill generates API references
  - The command syntax is: /doc-gen <path or module to document>
  - You get structured results directly in your Claude Code conversation
  - The /code-quality skill can scan code for style issues and deprecated patterns
  - The full reference for the doc-gen skill can be accessed by running: attune help-docs ref-skill-doc-gen

Unsupported:
  - Running a code quality review with /code-quality is something to consider before or after documenting code (the passages only mention /code-quality in the context of post-migration scanning, not documentation)
  - The /doc-gen skill can be part of a broader workflow alongside /smart-test and /refactor (no passage links /doc-gen to these tools; P2 only mentions /smart-test and /refactor in the context of migration, not documentation)

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

Great! The **Planning** tool is designed exactly for this. Here's what you can expect:

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

You'll be guided through a couple of quick questions — what you're planning and what kind of plan you need. However, if you provide both details upfront (e.g., *"plan a user authentication feature"*), the questions are skipped and it runs immediately [P1].

It runs on your Claude subscription with no API key or additional cost [P1].

---

**To get started**, just tell me: **what feature are you planning?**


### Judge verdicts (off → on)

| | Score | Supported | Unsupported |
| - | -: | -: | -: |
| Thinking off | 1.000 | 13 | 0 |
| Thinking on  | 0.909 | 10 | 1 |

**Reasoning (thinking off):**

> Every factual claim in the answer maps directly to explicit statements in passage P1. The answer accurately reproduces information about what Planning does, the Feature spec mode and its outputs, the interactive questioning flow, the shortcut when details are provided upfront, and the subscription/cost model. The output table is a verbatim reproduction of the table in P1. No claims rely on outside knowledge or invented details.

**Reasoning (thinking on):**

> Nearly all factual claims in the answer are directly and verbatim supported by P1. The one unsupported claim is the closing call-to-action statement that "the more detail you provide upfront, the faster we can skip to your structured plan." P1 only states a binary condition: if you provide *both* details upfront, the questions are skipped entirely. The answer reframes this as a gradual/scalar relationship ("more detail → faster"), which is an inference not explicitly stated in any passage.

**Claims identified by judge (thinking off):**

Supported:
  - Planning helps you think through features, architecture, and TDD strategy before writing code.
  - Planning gives you a structured plan when changes are cheapest to make.
  - For a new feature, Planning produces a Feature spec.
  - A Feature spec includes goals, scope, non-goals, and a task breakdown with effort estimates.
  - A Feature spec takes around 2–3 minutes to produce.
  - The task breakdown output consists of ordered steps with effort estimates and dependencies.
  - Acceptance criteria output provides a clear, testable definition of done per task.
  - Risk assessment output covers blockers, unknowns, and mitigation strategies.
  - Scope boundaries output lists explicit non-goals to prevent scope creep.
  - The dependency map output shows what depends on what and what can run in parallel.
  - You'll be guided through a couple of quick questions — what you're planning and what kind of plan you need.
  - If you provide both details upfront (e.g., 'plan a user authentication feature'), the questions are skipped and it runs immediately.
  - It runs on your Claude subscription with no API key or additional cost.

Unsupported:


**Claims identified by judge (thinking on):**

Supported:
  - When you plan a new feature, you'll get a Feature spec.
  - A Feature spec includes goals, scope, non-goals, and a task breakdown with effort estimates.
  - The Task breakdown output consists of ordered steps with effort estimates and dependencies.
  - The Acceptance criteria output consists of a clear, testable definition of done per task.
  - The Risk assessment output consists of blockers, unknowns, and mitigation strategies.
  - The Scope boundaries output consists of explicit non-goals to prevent scope creep.
  - The Dependency map output describes what depends on what, and what can run in parallel.
  - You'll be guided through a couple of quick questions — what you're planning and what kind of plan you need.
  - The questions keep the output focused on your actual goal instead of producing a generic plan.
  - If you provide both details upfront (e.g. 'plan a user authentication feature'), the questions are skipped and it runs immediately.

Unsupported:
  - The more detail you provide upfront, the faster we can skip to your structured plan.

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

## gq-038 — `coordinate documentation updates across the project`

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
> [P2] source: concepts/tool-memory-and-context.md
> ---
> type: concept
> name: tool-memory-and-context
> tags: [memory, context, persistence]
> source: plugin/skills/memory-and-context/SKILL.md
> ---
>
> # Memory and Context
>
> Every new Claude Code session starts with a blank slate.
> Memory and context bridges that gap — it stores notes,
> preferences, project decisions, and working state that
> persist across sessions. The next session picks up where
> the last one left off without you repeating yourself.
>
> ## What it does
>
> | Operation | What it does | Persists? |
> |-----------|-------------|-----------|
> | **Store** | Save a note, decision, or pattern with a key | Yes, across sessions |
> | **Recall** | Retrieve a specific memory by key | -- |
> | **Search** | Find memories matching a query | -- |
> | **Forget** | Remove a memory you no longer need | Permanent deletion |
> | **Context** | Store temporary state for this session only | No, session only |
>
> ## When you'd use it
>
> Store a debugging pattern you figured out so you don't
> have to rediscover it next week. Save a project convention
> ("we use snake_case for file names") so every session
> follows the same rules. Record an architecture decision
> so future sessions know why you chose SQLite over
> Postgres.
>
> Good candidates for memory:
>
> - Debugging patterns that took effort to figure out
> - Project conventions and naming rules
> - Architecture decisions and their rationale
> - Working state ("I'm in the middle of refactoring auth")
> - Team preferences ("Patrick prefers concise output")
>
> ## How it works
>
> You describe what you want to remember or find, and the
> skill asks clarifying questions before storing or
> searching. Everything runs on your Claude subscription
> with no API key or extra cost.
>
> ```
> remember that we use pytest-asyncio for all async tests
> ```
>
> The skill stores that note with a key you can recall
> later. When you start a new session and need that
> context:
>
> ```
> what testing framework do we use for async?
> ```
>
> The skill searches your stored memories and returns
> the match.
>
> ## Memory vs CLAUDE.md
>
> | Use case | Where it belongs |
> |----------|-----------------|
> | Permanent project rules | CLAUDE.md |
> | Session-to-session notes | Memory |
> | Architecture decisions | Memory (or CLAUDE.md for critical ones) |
> | Temporary working state | Context (session only) |
> | Debugging patterns | Memory |
>
> CLAUDE.md is for rules that apply to every session.
> Memory is for things you learn along the way.
>
> ## Want to learn more?
>
> - Say **"tell me more"** for step-by-step instructions
> - Say **"what is code quality?"** for code review tools
> - Say **"tell me about security audit"** for vulnerability scanning
> </passage>
>
> <passage>
> [P3] source: quickstarts/skill-memory-and-context.md
> ---
> type: quickstart
> name: skill-memory-and-context
> tags: [skill, claude-code]
> source: plugin/skills/memory-and-context/SKILL.md
> ---
>
> # Quickstart: Use /memory-and-context
>
> Store, retrieve, search, and manage persistent memory across sessions.
>
> ```
> /memory-and-context <operation: store|retrieve|search|forget|empathy>
> ```
>
> **Result:** Structured results in your Claude Code conversation.
>
> **Next:** See full reference: `attune help-docs ref-skill-memory-and-context`
>
> ## Related Topics
>
> _No related topics yet._
> </passage>

### Answer

## Coordinating Documentation Updates

You can use the **Doc Orchestrator** tool to coordinate documentation updates across your project. It handles end-to-end documentation maintenance, including scouting gaps, prioritizing, generating, and updating docs [P1].

### How to Use It

Simply call:

```
doc_orchestrator()
```

By default, it runs against the current directory (`.`), but you can specify a different project root by passing a `path` parameter [P1].

### What It Does

The Doc Orchestrator covers the full documentation pipeline [P1]:

- **Scouts** for documentation gaps
- **Prioritizes** what needs attention
- **Generates** new documentation
- **Updates** existing docs

### Aliases

You can also refer to this workflow by any of these names [P1]:
- `orchestrate documentation workflow`
- `doc pipeline`
- `coordinate documentation updates`
- `documentation maintenance`

### Related Tools

The Doc Orchestrator is part of a broader **workflow** group of tools that also includes [P1]:
- **Security Audit** — for vulnerability detection
- **Bug Predict** — for analyzing code patterns
- **Code Review** — for comprehensive code quality checks


### Judge verdicts (off → on)

| | Score | Supported | Unsupported |
| - | -: | -: | -: |
| Thinking off | 1.000 | 16 | 0 |
| Thinking on  | 0.917 | 11 | 1 |

**Reasoning (thinking off):**

> Every factual claim in the answer is directly and explicitly supported by Passage P1. The tool name, usage syntax, default path value, path parameter, aliases, workflow description (scout gaps, prioritize, generate, update), group classification, and related tools (Security Audit, Bug Predict, Code Review) with their brief descriptions are all present verbatim or near-verbatim in P1. There are no hallucinated details, invented flags, or claims drawn from outside knowledge.

**Reasoning (thinking on):**

> The answer is almost entirely faithful to the retrieved passages. Nearly every claim maps directly to P1 (Doc Orchestrator tool name, purpose, pipeline steps, usage syntax, path parameter type and default) or P2/P3 (Memory and Context skill, what persists across sessions, good memory candidates with verbatim examples, the slash-command syntax). The one unsupported claim is the specific example of "recording which docs were updated" as a memory use-case — P2 mentions "working state" generically with a different concrete example ("I'm in the middle of refactoring auth"), but the docs-update-tracking example is an inference not stated in any passage. Overall the answer demonstrates high faithfulness with only a minor invented illustrative example.

**Claims identified by judge (thinking off):**

Supported:
  - The Doc Orchestrator tool handles end-to-end documentation maintenance
  - The Doc Orchestrator scouts for documentation gaps
  - The Doc Orchestrator prioritizes what needs attention
  - The Doc Orchestrator generates new documentation
  - The Doc Orchestrator updates existing docs
  - The tool is called with doc_orchestrator()
  - By default, it runs against the current directory ('.')
  - You can specify a different project root by passing a 'path' parameter
  - The alias 'orchestrate documentation workflow' is associated with this tool
  - The alias 'doc pipeline' is associated with this tool
  - The alias 'coordinate documentation updates' is associated with this tool
  - The alias 'documentation maintenance' is associated with this tool
  - The Doc Orchestrator belongs to the 'workflow' group of tools
  - Security Audit is a related tool for vulnerability detection
  - Bug Predict is a related tool for analyzing code patterns
  - Code Review is a related tool for comprehensive code quality checks

Unsupported:


**Claims identified by judge (thinking on):**

Supported:
  - The Doc Orchestrator tool is designed for end-to-end documentation maintenance
  - The Doc Orchestrator scouts gaps, prioritizes, generates, and updates docs
  - The tool is invoked by calling doc_orchestrator()
  - The Doc Orchestrator accepts a 'path' parameter that is a string for the project root path
  - The 'path' parameter defaults to '.' if not specified
  - The Memory and Context skill can be used to persist decisions or notes between sessions
  - Project conventions and naming rules are good candidates for memory
  - Architecture decisions and their rationale are good candidates for memory
  - Working state such as 'I'm in the middle of refactoring auth' is a good candidate for memory
  - Notes can be stored and retrieved using /memory-and-context <operation: store|retrieve|search|forget>
  - Each new session can pick up where the last one left off without you repeating yourself

Unsupported:
  - Recording which docs were updated is a specific example of what to store in memory for documentation work

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

Great — you're looking to do an **architecture review** before committing to a design. Here's how to get started:

## What You'll Get

An architecture review produces a **component analysis, coupling assessment, and dependency map** — surfacing design concerns before they become expensive to fix [P1].

Specifically, the output includes:

| Output | Description |
|--------|-------------|
| Task breakdown | Ordered steps with effort estimates and dependencies [P1] |
| Acceptance criteria | Clear, testable definition of done per task [P1] |
| Risk assessment | Blockers, unknowns, and mitigation strategies [P1] |
| Scope boundaries | Explicit non-goals to prevent scope creep [P1] |
| Dependency map | What depends on what, and what can run in parallel [P1] |

## How to Trigger It

You can ask naturally:

```
review the architecture of the plugin system
```

Or provide the subject and mode upfront to skip the guided questions entirely [P3]:

```
/planning architecture review for [your feature]
```

## The Guided Flow

If you don't provide full details upfront, you'll be walked through two quick questions [P3]:

| Step | What you're asked |
|------|-------------------|
| 1. Subject | "What are you planning?" |
| 2. Mode | "Feature spec, TDD approach, or architecture review?" |

Providing both details upfront skips these questions and runs immediately [P1, P3].

## What to Do Next

After receiving your architecture review, you can:

| Goal | What to say |
|------|-------------|
| Start implementing | "let's build task 1" |
| Refine the plan | "add more detail to task 3" |
| Turn into a spec | "create a spec from this plan" |
| Go deeper | Say "tell me more" for the full reference [P3] |

---

**Ready to start?** Tell me: **what feature or system are you architecting?**


### Judge verdicts (off → on)

| | Score | Supported | Unsupported |
| - | -: | -: | -: |
| Thinking off | 0.895 | 17 | 2 |
| Thinking on  | 0.812 | 13 | 3 |

**Reasoning (thinking off):**

> Most of the factual claims in the answer are directly supported by P1 and P3. The planning modes, guided flow steps, output types, and follow-up options are all explicitly documented in the passages. However, two notable issues arise: (1) The answer assumes the user wants an 'architecture review' specifically, when the passages don't confirm this mapping from 'architect a new feature' — though the aliases in P1 do include 'architect a feature', making this a reasonable but still inferential leap. (2) More critically, the table of outputs (task breakdown, acceptance criteria, risk assessment, scope boundaries, dependency map) is presented as what an architecture review produces, but in P1 this table appears under the general 'What it produces' section for planning overall — not specifically scoped to architecture review mode, which is described as producing 'component analysis, coupling assessment, dependency map.' This is a subtle but real misattribution.

**Reasoning (thinking on):**

> The answer is largely faithful to the retrieved passages, especially P1 and P3. Most claims about what architecture review produces, how to trigger it, the guided flow, and the follow-up options are directly supported. However, the answer invents two specific table-cell descriptions: "breakdown of the components involved" for Component analysis and "how tightly components depend on each other" for Coupling assessment — neither phrase appears in any passage. Additionally, the specific CLI invocation '/planning architecture review for \u003cyour feature\u003e' is an adaptation of the example in P3 (which only shows '/planning authentication feature with OAuth support'), making it an inference rather than a directly supported claim.

**Claims identified by judge (thinking off):**

Supported:
  - An architecture review produces a component analysis, coupling assessment, and dependency map
  - Architecture review surfaces design concerns before they become expensive to fix
  - The output of planning includes a task breakdown with ordered steps, effort estimates, and dependencies
  - The output of planning includes acceptance criteria — clear, testable definition of done per task
  - The output of planning includes a risk assessment covering blockers, unknowns, and mitigation strategies
  - The output of planning includes scope boundaries with explicit non-goals to prevent scope creep
  - The output of planning includes a dependency map showing what depends on what and what can run in parallel
  - You can trigger an architecture review by saying 'review the architecture of the plugin system'
  - You can use '/planning architecture review for [your feature]' to trigger planning
  - If you don't provide full details upfront, you'll be walked through two quick questions
  - Step 1 asks 'What are you planning?'
  - Step 2 asks 'Feature spec, TDD approach, or architecture review?'
  - Providing both details upfront skips the guided questions and runs immediately
  - After the plan you can say 'let's build task 1' to start implementing
  - After the plan you can say 'add more detail to task 3' to refine the plan
  - After the plan you can say 'create a spec from this plan' to turn it into a spec
  - After the plan you can say 'tell me more' for the full reference

Unsupported:
  - The answer frames the user's request specifically as wanting an 'architecture review' — the passages do not confirm this interpretation; 'architect a new feature' could map to a feature spec or architecture review
  - The table of outputs (task breakdown, acceptance criteria, risk assessment, scope boundaries, dependency map) is presented as specific output of an architecture review, but the passages attribute this output table to planning in general, not specifically to architecture review mode

**Claims identified by judge (thinking on):**

Supported:
  - An architecture review produces a component analysis.
  - An architecture review produces a coupling assessment.
  - An architecture review produces a dependency map.
  - Architecture review is designed for evaluating design decisions before committing to them.
  - The dependency map describes what depends on what, and what can run in parallel.
  - The natural language trigger example 'review the architecture of the plugin system' can be used to start an architecture review.
  - If you provide both the subject and the mode upfront, the guided questions are skipped.
  - If you provide both the subject and the mode upfront, the plan runs immediately.
  - In the guided flow, Step 1 asks 'What are you planning?'
  - In the guided flow, Step 2 asks about the mode: 'Feature spec, TDD approach, or architecture review?'
  - After the plan, saying 'let's build task 1' starts implementing.
  - Saying 'create a spec from this plan' turns the plan into a spec-driven workflow.
  - Saying 'add more detail to task 3' refines the plan.

Unsupported:
  - Component analysis describes a 'breakdown of the components involved' (this specific description is not found in any passage).
  - Coupling assessment describes 'how tightly components depend on each other' (this specific description is not found in any passage).
  - The skill can be triggered with '/planning architecture review for <your feature>' (the passages show '/planning authentication feature with OAuth support' as the example; the architecture-specific invocation format shown in the answer is an adaptation not directly stated in the passages).

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

## gq-009 — `audit documentation for staleness`

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

## Auditing Documentation for Staleness

You can use the **Doc Audit** tool to audit existing documentation for staleness, broken links, and drift from source code [P1].

### Tool: `doc_audit`

| Detail | Info |
|--------|------|
| **Usage** | `doc_audit()` |
| **Parameter** | `path` (string) — Project root path, defaults to `.` |
| **Group** | workflow |

The tool audits for [P1]:
- **Staleness** — documentation that is out of date
- **Broken links** — links that no longer resolve
- **Drift from source code** — documentation that no longer matches the actual code

### Related Workflow Tools

Once you've identified stale documentation, you may also find these related tools useful [P1]:

- **Security Audit** — Run security audit workflow on codebase
- **Bug Predict** — Run bug prediction workflow
- **Code Review** — Run code review workflow

### Notes

The `doc_audit` tool is sourced from `src/attune/mcp/tool_schemas.py` and belongs to the **workflow** group [P1]. It can also be referenced by the aliases: *stale documentation*, *stale docs*, *outdated documentation*, or *find stale docs* [P1].


### Judge verdicts (off → on)

| | Score | Supported | Unsupported |
| - | -: | -: | -: |
| Thinking off | 0.947 | 18 | 1 |
| Thinking on  | 0.867 | 13 | 2 |

**Reasoning (thinking off):**

> Almost every factual claim in the answer is directly supported by passage P1, which provides the tool name, description, parameter details, usage syntax, group, source file, aliases, and related topics. The only claim that goes beyond what the passages explicitly state is the interpretive suggestion that the related tools are useful "once you've identified stale documentation" — P1 merely lists them as related topics without implying a sequential workflow relationship. All other claims map cleanly to explicit text in P1.

**Reasoning (thinking on):**

> The answer is overwhelmingly faithful to P1, which is the sole relevant passage. Nearly every claim — the tool's purpose, its parameter name/type/description/default, its invocation syntax, its group, and the related tools — maps directly to text in P1. The only faithfulness issues are minor completions of truncated descriptions in P1: "Detects vulnerabili..." is extended to "Detects vulnerabilities in your codebase" (adding "in your codebase" with no textual basis), and "Analyzes code patterns and pred..." is completed as "predicts bugs" (the object "bugs" is inferred, not stated). These are small extrapolations beyond what the truncated passage explicitly confirms, so they are marked unsupported. No hallucinated parameters, flags, or tool names were introduced.

**Claims identified by judge (thinking off):**

Supported:
  - The Doc Audit tool audits existing documentation for staleness, broken links, and drift from source code.
  - The usage syntax for the tool is `doc_audit()`.
  - The `path` parameter is of type string.
  - The `path` parameter represents the project root path.
  - The `path` parameter defaults to `.`.
  - The Doc Audit tool belongs to the workflow group.
  - The tool audits for staleness — documentation that is out of date.
  - The tool audits for broken links — links that no longer resolve.
  - The tool audits for drift from source code — documentation that no longer matches the actual code.
  - Security Audit is a related tool that runs a security audit workflow on codebase.
  - Bug Predict is a related tool that runs a bug prediction workflow.
  - Code Review is a related tool that runs a code review workflow.
  - The `doc_audit` tool is sourced from `src/attune/mcp/tool_schemas.py`.
  - The `doc_audit` tool belongs to the workflow group.
  - The tool can be referenced by the alias 'stale documentation'.
  - The tool can be referenced by the alias 'stale docs'.
  - The tool can be referenced by the alias 'outdated documentation'.
  - The tool can be referenced by the alias 'find stale docs'.

Unsupported:
  - Once you've identified stale documentation, the related tools (Security Audit, Bug Predict, Code Review) may be useful for follow-up actions.

**Claims identified by judge (thinking on):**

Supported:
  - The Doc Audit tool audits existing documentation for staleness, broken links, and drift from source code.
  - The tool accepts a parameter named `path`.
  - The `path` parameter is of type string.
  - The `path` parameter description is 'Project root path'.
  - The `path` parameter has a default value of '.'
  - The tool is called as: `doc_audit()`
  - Security Audit is listed as a related tool to Doc Audit.
  - Security Audit detects vulnerabilities (partial: 'Detects vulnerabili...' per P1).
  - Bug Predict is listed as a related tool to Doc Audit.
  - Bug Predict analyzes code patterns (per P1: 'Analyzes code patterns and pred...').
  - Code Review is listed as a related tool to Doc Audit.
  - Code Review provides comprehensive code quality analysis (per P1: 'Provides comprehensive code qualit...').
  - The Doc Audit tool belongs to the 'workflow' group.

Unsupported:
  - Security Audit detects vulnerabilities 'in your codebase' — P1's description is truncated ('Detects vulnerabili...') and does not explicitly include the phrase 'in your codebase'.
  - Bug Predict 'predicts bugs' — P1's description is truncated ('Analyzes code patterns and pred...') and does not explicitly confirm the object of 'predicts' is 'bugs'.

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

Here are the best approaches for finding bugs that are easy to miss during normal development and review.

---

## 🔍 Use Bug Prediction Scanning

The **`/bug-predict`** tool scans your codebase for patterns that historically cause production incidents, analyzing code structure, complexity, and known anti-patterns to identify where failures are most likely to happen [P1].

Run it with:
```
/bug-predict <path or directory to scan>
```
[P3]

### What it catches

The scanner targets three categories of risky patterns [P1]:

| Pattern | Severity | What it finds |
|---|---|---|
| **dangerous_eval** | HIGH | `eval()`, `exec()`, `compile()` on any input — code injection vectors |
| **broad_exception** | MEDIUM | Bare `except:`, unlogged `except Exception:` — errors silently swallowed |
| **incomplete_code** | LOW | TODO, FIXME, HACK, XXX comments — unfinished code paths that break under edge cases |

### Beyond obvious patterns

It also weighs contextual risk signals [P1]:
- **Cyclomatic complexity** — deeply nested conditionals and long function bodies correlate with higher defect rates
- **Change frequency** — files modified often ("hot" files) are more likely to contain regressions
- **Code smells** — functions over 50 lines, classes with too many methods, duplicated logic

### Smart false-positive filtering

Not every match is a real bug — the scanner suppresses known-safe patterns like `eval()` inside test fixture strings and JavaScript `regex.exec()` method calls [P1].

---

## ⚙️ Automate Bug Detection in CI/CD

For catching bugs *before* they reach production on every push, integrate scanning into a CI/CD pipeline [P2]:

- **Security scanning** (e.g., bandit) runs as a pipeline stage that blocks deployment if it fails [P2]
- **Unit tests** catch regressions automatically on every commit [P2]
- Without CI, bugs can hide until someone manually runs the test suite — which might be days or weeks after the broken code was merged [P2]

---

## 📅 When to Run Bug Prediction

The best moments to scan [P1]:
- **Before merging a large PR** — catch patterns humans miss
- **During code review** — focus human attention on real risks
- **After onboarding unfamiliar code** — map risk hotspots fast
- **Before a release** — verify no new high-severity patterns crept in
- **As a periodic health check** on high-churn modules


### Judge verdicts (off → on)

| | Score | Supported | Unsupported |
| - | -: | -: | -: |
| Thinking off | 1.000 | 22 | 0 |
| Thinking on  | 0.923 | 24 | 2 |

**Reasoning (thinking off):**

> Every factual claim in the answer is directly traceable to one of the retrieved passages. The bug prediction patterns, severities, risk signals, false-positive suppression examples, CI/CD pipeline details, and recommended usage moments are all explicitly stated in P1, P2, or P3. No outside knowledge or invented details were introduced; even the specific example tools (bandit) and the exact wording about bugs hiding "days or weeks" come verbatim from P2. The answer is highly faithful to the retrieved passages.

**Reasoning (thinking on):**

> The answer is overwhelmingly faithful to the retrieved passages. Nearly all factual claims map directly to explicit statements in P1 (bug prediction concept) and P2 (CI/CD concept), with the quickstart command confirmed by P3. Two minor unsupported claims appear: (1) the characterization that type checking "catches type-related bugs" — P2 only states that the stage runs mypy or pyright without describing what class of bugs it catches; and (2) the qualifier "easy to miss locally" appended to the matrix build benefit — P2 says matrix builds catch "version-specific bugs" and "platform-specific issues" but never uses the phrase "easy to miss locally." Both are reasonable inferences but are not explicitly stated in the passages.

**Claims identified by judge (thinking off):**

Supported:
  - The /bug-predict tool scans your codebase for patterns that historically cause production incidents
  - The /bug-predict tool analyzes code structure, complexity, and known anti-patterns to identify where failures are most likely to happen
  - The /bug-predict command syntax is: /bug-predict <path or directory to scan>
  - The scanner targets a pattern called dangerous_eval with HIGH severity
  - dangerous_eval catches eval(), exec(), compile() on any input — code injection vectors
  - The scanner targets a pattern called broad_exception with MEDIUM severity
  - broad_exception catches bare except:, unlogged except Exception: — errors silently swallowed
  - The scanner targets a pattern called incomplete_code with LOW severity
  - incomplete_code catches TODO, FIXME, HACK, XXX comments — unfinished code paths that break under edge cases
  - The scanner weighs cyclomatic complexity — deeply nested conditionals and long function bodies correlate with higher defect rates
  - The scanner weighs change frequency — files modified often ('hot' files) are more likely to contain regressions
  - The scanner weighs code smells — functions over 50 lines, classes with too many methods, duplicated logic across modules
  - The scanner suppresses eval() inside test fixture strings as a known-safe pattern
  - The scanner suppresses JavaScript regex.exec() method calls as a known-safe pattern
  - Security scanning (e.g., bandit) runs as a pipeline stage that blocks deployment if it fails
  - Unit tests catch regressions automatically on every commit
  - Without CI, bugs can hide until someone manually runs the test suite — which might be days or weeks after the broken code was merged
  - Before merging a large PR is a good time to scan — catch patterns humans miss
  - During code review is a good time to scan — focus human attention on real risks
  - After onboarding unfamiliar code is a good time to scan — map risk hotspots fast
  - Before a release is a good time to scan — verify no new high-severity patterns crept in
  - As a periodic health check on high-churn modules is a good time to scan

Unsupported:


**Claims identified by judge (thinking on):**

Supported:
  - The /bug-predict tool scans your codebase for patterns that historically cause production incidents
  - The tool analyzes code structure, complexity, and known anti-patterns to identify where failures are most likely to happen
  - The tool is run with: /bug-predict <path or directory to scan>
  - The scanner looks for three categories of risk
  - The dangerous_eval pattern has HIGH severity and targets eval(), exec(), compile() on any input — code injection vectors
  - The broad_exception pattern has MEDIUM severity and targets bare except: and unlogged except Exception: — errors silently swallowed
  - The incomplete_code pattern has LOW severity and targets TODO, FIXME, HACK, XXX comments — unfinished code paths that break under edge cases
  - The scanner weighs contextual risk signals beyond pattern matching
  - Cyclomatic complexity — deeply nested conditionals and long function bodies correlate with higher defect rates
  - Change frequency — files modified often ('hot' files) are more likely to contain regressions
  - Code smells include functions over 50 lines, classes with too many methods, and duplicated logic
  - The scanner suppresses known-safe patterns such as eval() inside test fixture strings
  - The scanner suppresses JavaScript regex.exec() calls
  - The scanner suppresses broad exceptions with # INTENTIONAL: comments
  - Running checks on every commit catches bugs early before they reach production
  - Security scan stage includes bandit and pip-audit and finds vulnerabilities before they ship
  - Type check stage runs mypy or pyright
  - Unit and integration tests catch regressions before they reach main
  - Matrix builds run test suites across multiple Python versions (3.10, 3.11, 3.12, 3.13)
  - Matrix builds across operating systems catch platform-specific bugs
  - Running bug prediction before merging a large PR catches patterns humans miss
  - Running bug prediction during code review focuses human attention on real risks
  - Running bug prediction after onboarding unfamiliar code maps risk hotspots fast
  - Running bug prediction before a release verifies no new high-severity patterns crept in

Unsupported:
  - Type check (mypy or pyright) catches type-related bugs
  - Version-specific and platform-specific bugs are easy to miss locally

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

# Running a Security Audit (SAST) on Your Repository

A security audit scans your codebase for vulnerabilities that are easy to introduce and hard to spot in code review [P1]. Here's how to get started:

## Option 1: CLI Workflow

Run the following command against your source directory:

```
attune workflow run security-audit --path "src/"
```

This produces severity-grouped findings with CWE identifiers [P2].

## Option 2: Claude Code Skill

If you're working inside Claude Code, use:

```
/security-audit <path or directory to scan>
```

This returns structured results directly in your conversation [P3].

---

## What It Detects

The scan covers the following vulnerability categories [P1]:

| Category | Examples |
|----------|----------|
| **Code injection** | `eval()`, `exec()`, `compile()` on untrusted input |
| **Path traversal** | File operations without path validation |
| **Hardcoded secrets** | API keys, tokens, passwords in source control |
| **SQL/command injection** | String concatenation in queries or shell commands |
| **SSRF** | HTTP requests to user-controlled URLs |
| **Weak cryptography** | MD5/SHA1 for security, hardcoded IVs |

## Choosing a Scan Depth

| Depth | Time | What You Get |
|-------|------|-------------|
| **Quick** | ~30s | Surface scan — eval/exec, obvious secrets |
| **Standard** | ~2 min | Full pattern matching with severity ratings |
| **Deep** | ~5 min | Multi-pass review with OWASP mapping and fix suggestions |

[P1]

## Next Steps

After fixing critical issues, run `attune workflow run test-gen` [P2].


### Judge verdicts (off → on)

| | Score | Supported | Unsupported |
| - | -: | -: | -: |
| Thinking off | 1.000 | 15 | 0 |
| Thinking on  | 0.938 | 15 | 1 |

**Reasoning (thinking off):**

> Every factual claim in the answer maps directly to content in the retrieved passages. The vulnerability categories and their examples come verbatim from P1's table. The CLI command and its output description (severity-grouped findings with CWE identifiers) are taken directly from P2. The Claude Code skill command and its result description come from P3. The scan depth table (Quick/Standard/Deep with times and descriptions) is reproduced directly from P1. The "next steps" suggestion to run `attune workflow run test-gen` is explicitly stated in P2. No claims appear to be invented or drawn from outside knowledge.

**Reasoning (thinking on):**

> The vast majority of claims in the answer are directly and explicitly supported by the retrieved passages — P1 covers what the audit finds, scan depths and timings; P2 covers the CLI command and CWE output; P3 covers the /security-audit Claude Code skill. The only unsupported claim is the characterization of the tool as "SAST" in the answer's heading. None of the passages use the term "SAST" or "Static Application Security Testing," making this a label introduced from outside knowledge. All other claims — including exact command syntax, vulnerability categories, depth tiers, timings, and the follow-up test-gen recommendation — are verbatim or near-verbatim from the passages.

**Claims identified by judge (thinking off):**

Supported:
  - A security audit scans your codebase for vulnerabilities that are easy to introduce and hard to spot in code review.
  - The CLI command to run a security audit is: attune workflow run security-audit --path "src/"
  - The CLI workflow produces severity-grouped findings with CWE identifiers.
  - The Claude Code skill command is: /security-audit <path or directory to scan>
  - The Claude Code skill returns structured results directly in your conversation.
  - The scan covers code injection vulnerabilities such as eval(), exec(), compile() on untrusted input.
  - The scan covers path traversal vulnerabilities such as file operations without path validation.
  - The scan covers hardcoded secrets such as API keys, tokens, passwords in source control.
  - The scan covers SQL/command injection via string concatenation in queries or shell commands.
  - The scan covers SSRF via HTTP requests to user-controlled URLs.
  - The scan covers weak cryptography such as MD5/SHA1 for security purposes and hardcoded IVs.
  - Quick depth takes ~30s and provides a surface scan covering eval/exec and obvious secrets.
  - Standard depth takes ~2 min and provides full pattern matching with severity ratings.
  - Deep depth takes ~5 min and provides multi-pass review with OWASP mapping and fix suggestions.
  - After fixing critical issues, the next step is to run attune workflow run test-gen.

Unsupported:


**Claims identified by judge (thinking on):**

Supported:
  - A security audit scans your codebase for vulnerabilities that are easy to introduce and hard to spot in code review.
  - The CLI command to run a security audit is: attune workflow run security-audit --path "src/"
  - The CLI workflow produces severity-grouped findings with CWE identifiers.
  - Inside Claude Code, you can use the /security-audit skill with a path or directory to scan.
  - The /security-audit skill returns structured results in the Claude Code conversation.
  - The scan covers code injection vulnerabilities including eval(), exec(), and compile() on untrusted input.
  - The scan covers path traversal vulnerabilities such as file operations without path validation.
  - The scan covers hardcoded secrets including API keys, tokens, and passwords in source.
  - The scan covers SQL/command injection via string concatenation in queries or shell commands.
  - The scan covers SSRF vulnerabilities involving HTTP requests to user-controlled URLs.
  - The scan covers weak cryptography such as MD5/SHA1 and hardcoded IVs.
  - The Quick scan depth takes approximately 30 seconds and performs a surface scan for eval/exec and obvious secrets.
  - The Standard scan depth takes approximately 2 minutes and performs full pattern matching with severity ratings.
  - The Deep scan depth takes approximately 5 minutes and performs a multi-pass review with OWASP mapping and fix suggestions.
  - After addressing critical findings, the recommended next step is to run attune workflow run test-gen.

Unsupported:
  - The security audit is a SAST (Static Application Security Testing) tool.

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

## gq-031 — `my CI pipeline keeps failing`

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
> [P2] source: quickstarts/task-ci-cd-pipeline.md
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
>
> <passage>
> [P3] source: concepts/task-ci-cd-pipeline.md
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

### Answer

## Fixing Your Failing CI Pipeline

There are a few ways to approach this depending on what's causing the failures.

---

### If tests are failing specifically

The **fix-test** tool can auto-diagnose and repair failing tests. It classifies the root cause and applies a targeted fix, re-running the test after each repair attempt and retrying up to 3 times [P1].

Common root causes it handles include:

| Root cause | Auto-fixable? |
|---|---|
| Import error (module renamed/moved) | Yes |
| Mock mismatch (stale `patch()` target) | Yes |
| Assertion drift (return value changed) | Yes |
| Type error (signature changed) | Yes |
| Fixture missing | Usually |
| Environment issue (missing env var, wrong Python version) | Sometimes |

[P1]

To use it, simply say something like **"fix test_auth.py"** — providing the test name inline skips scoping questions [P1].

---

### If your pipeline configuration needs attention

A correctly structured GitHub Actions pipeline should:

- **Lint first**, then run **tests** (with `needs: lint` so tests only run if linting passes) [P2]
- Use a **matrix strategy** across Python versions (e.g., `"3.10"` and `"3.12"`) to catch version-specific bugs [P2, P3]
- Cache pip dependencies using `cache: pip` to speed up runs [P2]

Each pipeline stage acts as a **gate** — if it fails, later stages never run [P3].

---

### Protect against future failures

Once CI is stable, go to **Settings > Branches > Add branch protection rule** for `main` and enable **Require status checks to pass before merging** [P2]. This ensures no one can merge a PR that breaks CI [P2].

---

**What specifically is failing?** Sharing the error or test name will help narrow this down further.


### Judge verdicts (off → on)

| | Score | Supported | Unsupported |
| - | -: | -: | -: |
| Thinking off | 1.000 | 17 | 0 |
| Thinking on  | 0.944 | 17 | 1 |

**Reasoning (thinking off):**

> Every atomic factual claim in the answer is directly and explicitly supported by at least one of the retrieved passages. The fix-test details (root causes, retry count, inline test name skipping scoping) come directly from P1. The GitHub Actions pipeline structure (needs: lint, matrix versions, cache: pip) comes from P2. The pipeline stage gate concept comes from P3. The branch protection instructions come verbatim from P2. No claims rely on outside knowledge or invented details.

**Reasoning (thinking on):**

> The answer is overwhelmingly faithful to the retrieved passages. Nearly all claims — the Fix Test tool's behavior, retry count, root-cause table, inline test-name shortcut, GitHub Actions workflow file location, job dependency behavior, branch-protection effect, and the quick-additions table — are directly and explicitly stated in P1 or P2. The one unsupported claim is that "a failure on any one version will fail the pipeline" in the context of matrix builds: while P2 and P3 describe matrix builds across Python versions, neither passage explicitly states that a single version failure causes the entire pipeline to fail. That is a reasonable inference but goes beyond what the passages say.

**Claims identified by judge (thinking off):**

Supported:
  - The fix-test tool can auto-diagnose and repair failing tests
  - fix-test classifies the root cause and applies a targeted fix
  - fix-test re-runs the test after each repair attempt
  - fix-test retries up to 3 times
  - Import error (module renamed/moved) is auto-fixable by fix-test
  - Mock mismatch (stale patch() target) is auto-fixable by fix-test
  - Assertion drift (return value changed) is auto-fixable by fix-test
  - Type error (signature changed) is auto-fixable by fix-test
  - Fixture missing is usually auto-fixable by fix-test
  - Environment issue (missing env var, wrong Python version) is sometimes auto-fixable by fix-test
  - Saying something like 'fix test_auth.py' — providing the test name inline skips scoping questions
  - A GitHub Actions pipeline should lint first, then run tests with 'needs: lint' so tests only run if linting passes
  - A matrix strategy can be used across Python versions (e.g., '3.10' and '3.12') to catch version-specific bugs
  - pip dependencies can be cached using 'cache: pip' to speed up runs
  - Each pipeline stage acts as a gate — if it fails, later stages never run
  - To protect against future failures, go to Settings > Branches > Add branch protection rule for main and enable Require status checks to pass before merging
  - Enabling branch protection ensures no one can merge a PR that breaks CI

Unsupported:


**Claims identified by judge (thinking on):**

Supported:
  - The Fix Test tool can automatically diagnose and repair failing tests.
  - Fix Test classifies the root cause and applies a targeted fix.
  - Fix Test re-runs the test after each repair attempt.
  - Fix Test retries up to 3 times.
  - Import errors (module renamed/moved) are auto-fixable by Fix Test.
  - Stale patch() mock targets (mock mismatch) are auto-fixable by Fix Test.
  - Assertion drift (return value or output changed) is auto-fixable by Fix Test.
  - Type errors (function signature changed) are auto-fixable by Fix Test.
  - Missing fixtures are 'Usually' auto-fixable by Fix Test.
  - Environment issues are 'Sometimes' auto-fixable by Fix Test.
  - Providing the test name inline (e.g., 'fix test_yourfile.py') skips the scoping questions.
  - GitHub Actions only looks for workflow files in .github/workflows/.
  - If the test job has 'needs: lint', a lint failure will block tests from running.
  - If status checks are required before merging, any failing job will block the PR.
  - To add security scanning, add a security job with 'bandit -r src/' and 'pip-audit'.
  - To test more Python versions, add versions to the matrix list.
  - To get faster dependency installation, replace setup-python with astral-sh/setup-uv@v4.

Unsupported:
  - If you are testing across multiple Python versions, a failure on any one version (e.g., 3.10 vs 3.12) will fail the pipeline.

### Your labels

```yaml
id: gq-031
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

## gq-039 — `scope out next sprint tasks`

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
> [P2] source: concepts/tool-spec.md
> ---
> type: concept
> name: tool-spec
> tags: [spec, planning, workflow]
> source: plugin/skills/spec/SKILL.md
> ---
>
> # Spec-Driven Development
>
> Spec walks you from a rough idea to working code through
> five structured phases. Instead of jumping straight into
> implementation, you brainstorm approaches, decompose work
> into tasks, review for gaps, get explicit approval, then
> execute one task at a time with quality gates after each.
>
> ## The five phases
>
> | Phase | What happens | Interaction | Quality gate | Output |
> |-------|-------------|-------------|--------------|--------|
> | **Brainstorm** | Explore the problem space, surface constraints, generate competing approaches | Socratic Q&A -- you describe the goal, spec asks clarifying questions | At least 2 viable approaches identified | Prose summary with trade-offs |
> | **Decompose** | Break chosen approach into ordered tasks with acceptance criteria | Spec proposes tasks, you adjust scope and ordering | Every task has a testable acceptance criterion | XML task blocks saved to `.claude/plans/` |
> | **Review** | Walk through the plan task by task, check for gaps and risks | You read each task, flag concerns, suggest edits | No open questions or unaddressed risks | Annotated plan ready for sign-off |
> | **Approve** | Final summary with task count, scope, and risk overview | Single yes/no gate -- nothing runs until you say go | Explicit user approval | Locked plan |
> | **Execute** | Implement tasks one at a time, run quality gates after each | After each task: approve, redo with new instructions, or auto-run the rest | Quality score per task, severity-gated approval | Working code + state file tracking progress |
>
> ## Why spec-first matters
>
> Starting to code without a plan leads to scope creep and
> rework. Spec forces clarity upfront -- what are we building,
> what are the edge cases, what does "done" look like -- before
> a single line is written. The approval gate means you
> always see the full plan before any files change. If you
> don't like the direction, you adjust the spec, not the
> code.
>
> ## The Socratic pattern
>
> Each phase is conversational, not automatic. In brainstorm,
> spec asks about your constraints, existing code, and goals
> before proposing anything. In decompose, it explains each
> task and asks if the breakdown matches your mental model.
> In review, it highlights risks and asks if you want to
> address them. You're always driving -- spec never executes
> without your approval.
>
> ## When to use it
>
> - For any feature that touches 3+ files
> - When requirements are ambiguous or evolving
> - To produce an auditable trail of design decisions
> - When handing off implementation to another developer
> - Before large refactors where the blast radius is unclear
> - When you want quality gates on each implementation step
>
> ## What it produces
>
> | Output | Where it lives | Purpose |
> |--------|---------------|---------|
> | Plan file | `.claude/plans/{topic-slug}.md` | Prose summary + XML task blocks |
> | State file | `.claude/plans/{topic-slug}.state.json` | Tracks completed/pending tasks |
> | Implemented code | Your source tree | Files created or modified per task |
> | Quality gate results | Shown inline after each task | Pass/fail with severity and score |
>
> ## What to expect
>
> When you ask to build something, you'll have a
> conversation first -- not a wall of generated code. Spec
> guides you through scoping before anything runs. If you
> provide a detailed description upfront, the brainstorm
> phase is shorter. If you're vague, expect more questions.
>
> Runs on your Claude subscription -- no API key or
> additional cost.
>
> ## Want to learn more?
>
> - Say **"how do I use spec?"** for step-by-step
>   instructions walking through each phase
> - Say **"tell me more about spec"** for the complete
>   reference with all phases, gates, and resume behavior
> - Say **"plan a feature"** if you want high-level
>   planning without the full spec workflow
> </passage>
>
> <passage>
> [P3] source: concepts/task-code-migration.md
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

### Answer

## Scoping Next Sprint Tasks

For sprint planning and scoping work tasks, the **Planning** tool is the right fit — it's specifically designed for this use case. [P1]

### What Planning Produces for Sprint Scoping

| Output | What You Get |
|--------|-------------|
| **Task breakdown** | Ordered steps with effort estimates and dependencies [P1] |
| **Acceptance criteria** | Clear, testable definition of done per task [P1] |
| **Risk assessment** | Blockers, unknowns, and mitigation strategies [P1] |
| **Scope boundaries** | Explicit non-goals to prevent scope creep [P1] |
| **Dependency map** | What depends on what, and what can run in parallel [P1] |

### How to Use It

The Planning tool will guide you through a couple of quick questions first — what you're planning and what kind of plan you need — to keep output focused on your actual goal. If you provide both details upfront (e.g., *"plan a user authentication feature"*), the questions are skipped and it runs immediately. [P1]

### Which Planning Mode?

| Mode | Best For |
|------|----------|
| **Feature spec** | Starting a new feature or epic (~2–3 min) [P1] |
| **TDD scaffold** | Complex logic needing test coverage from the start (~1–2 min) [P1] |
| **Architecture review** | Evaluating design decisions before committing (~2–3 min) [P1] |

### Need More Structure?

If your sprint involves features touching **3+ files**, ambiguous requirements, or you want **quality gates on each implementation step**, consider using **Spec-Driven Development** instead — it adds brainstorming, formal approval, and task-by-task execution gates. [P2]

**To get started**, just describe what features or work items you're planning for the sprint and the Planning tool will take it from there. [P1]


### Judge verdicts (off → on)

| | Score | Supported | Unsupported |
| - | -: | -: | -: |
| Thinking off | 1.000 | 16 | 0 |
| Thinking on  | 0.944 | 17 | 1 |

**Reasoning (thinking off):**

> Every atomic factual claim in the answer maps directly to explicit statements in the retrieved passages. P1 covers all claims about the Planning tool — its purpose (sprint planning/scoping tasks via aliases), its outputs (task breakdown, acceptance criteria, risk assessment, scope boundaries, dependency map), its interaction model (quick questions, skipping questions when details are provided upfront), and its three modes with their time estimates and use cases. P2 covers all claims about Spec-Driven Development — the 3+ files threshold, ambiguous requirements, quality gates on each implementation step, and the brainstorm/approval/execution structure. No claims rely on outside knowledge or invented details.

**Reasoning (thinking on):**

> The answer is highly faithful to the retrieved passages. Nearly every factual claim — the Planning skill's outputs (task breakdown, acceptance criteria, risk assessment, scope boundaries, dependency map), the Feature spec mode's description and timing, the guided-questions behavior, the upfront-details shortcut, and Spec's five phases, quality gates, and output files — is verbatim or near-verbatim from P1 and P2. The only marginally unsupported claim is calling Planning the 'best' tool for sprint planning; while P1 lists 'sprint planning' as an alias, no passage makes a comparative superiority claim. Everything else is directly grounded in the passages.

**Claims identified by judge (thinking off):**

Supported:
  - The Planning tool is designed for sprint planning and scoping work tasks
  - The Planning tool produces a task breakdown with ordered steps, effort estimates, and dependencies
  - The Planning tool produces acceptance criteria described as a clear, testable definition of done per task
  - The Planning tool produces a risk assessment covering blockers, unknowns, and mitigation strategies
  - The Planning tool produces scope boundaries as explicit non-goals to prevent scope creep
  - The Planning tool produces a dependency map showing what depends on what and what can run in parallel
  - The Planning tool guides the user through a couple of quick questions first — what you're planning and what kind of plan you need
  - The quick questions keep output focused on the user's actual goal
  - If you provide both details upfront (e.g., 'plan a user authentication feature'), the questions are skipped and it runs immediately
  - The Feature spec mode is best for starting a new feature or epic and takes approximately 2–3 minutes
  - The TDD scaffold mode is best for complex logic needing test coverage from the start and takes approximately 1–2 minutes
  - The Architecture review mode is best for evaluating design decisions before committing and takes approximately 2–3 minutes
  - Spec-Driven Development is recommended when features touch 3+ files
  - Spec-Driven Development is recommended when requirements are ambiguous or evolving
  - Spec-Driven Development adds brainstorming, formal approval, and task-by-task execution gates
  - Spec-Driven Development is recommended when you want quality gates on each implementation step

Unsupported:


**Claims identified by judge (thinking on):**

Supported:
  - The Planning skill has 'sprint planning' as an alias/use case
  - The Planning skill generates a task breakdown of ordered steps with effort estimates and dependencies
  - The Planning skill generates acceptance criteria described as 'clear, testable definition of done per task'
  - The Planning skill generates a risk assessment covering blockers, unknowns, and mitigation strategies
  - The Planning skill generates scope boundaries as explicit non-goals to prevent scope creep
  - The Planning skill generates a dependency map showing what depends on what and what can run in parallel
  - Feature spec is one of the Planning modes
  - Feature spec produces goals, scope, non-goals, and a task breakdown with effort estimates
  - Feature spec takes approximately 2–3 minutes
  - When using Planning, you'll be guided through a couple of quick questions about what you're planning and what kind of plan you need
  - If you provide both details upfront (e.g. 'plan a user authentication feature'), the questions are skipped and it runs immediately
  - Spec-Driven Development is recommended for features touching 3+ files
  - Spec-Driven Development is recommended when requirements are ambiguous or evolving
  - Spec-Driven Development walks you through brainstorming, task decomposition, review, approval, and execution phases
  - Spec-Driven Development includes quality gates after each task
  - Spec-Driven Development saves a plan file for tracking progress
  - Spec-Driven Development saves a state file for tracking progress

Unsupported:
  - The Planning skill is the 'best' tool for sprint planning (passages identify it as relevant but make no comparative claim)

### Your labels

```yaml
id: gq-039
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

## Deep Review: End-to-End Review Before Merging a PR

For a comprehensive end-to-end review before merging a PR, the **Deep Review** tool is what you're looking for — it's explicitly aliased as "end-to-end review" and "PR review before merging" [P1].

### What It Does

Deep Review performs a **multi-pass deep code review** covering security, quality, and test gap analysis, delivering prioritized findings [P1].

### How to Use It

```python
deep_review(path="...")
```

Simply provide the path to the directory or file you want reviewed [P1].

### What It Analyzes

The review covers multiple dimensions, including related workflows for:
- **Security** concerns [P1]
- **Code quality** [P1]
- **Bug prediction** — scanning for patterns like dangerous `eval()` use, broad exception handling, and incomplete code paths [P2]

### Why Before Merging?

Running this kind of analysis before merging is recommended specifically to:
- **Catch patterns humans miss** during large PR reviews [P2]
- **Focus human attention on real risks** during code review [P2]
- **Verify no new high-severity patterns crept in** before a release [P2]

### Related Tools

Deep Review connects to complementary workflows including the **Security Audit** and **Code Review** tools [P1], making it a thorough, all-in-one pre-merge checkpoint.


### Judge verdicts (off → on)

| | Score | Supported | Unsupported |
| - | -: | -: | -: |
| Thinking off | 0.882 | 15 | 2 |
| Thinking on  | 0.933 | 14 | 1 |

**Reasoning (thinking off):**

> Most claims in the answer are well-supported by the retrieved passages. P1 directly supports the tool name, aliases, description (multi-pass, security/quality/test gap, prioritized findings), usage syntax, and related tools. P2 supports the bug prediction patterns and the 'when you'd use it' rationale for pre-merge review. However, the answer implies that bug prediction (eval, broad exceptions, incomplete code) is part of Deep Review's own analysis passes, when P1 only lists Bug Predict as a *related* tool — not an integrated component. Similarly, calling Deep Review a 'thorough, all-in-one pre-merge checkpoint' is an editorial characterization not directly stated in any passage.

**Reasoning (thinking on):**

> The answer is overwhelmingly faithful to the retrieved passages. Nearly every factual claim — tool aliases, the multi-pass nature of Deep Review, its coverage areas, the call signature, the path parameter's description, and the related tools with their descriptions — is drawn verbatim or near-verbatim from P1. Bug Predict's recommended use before merging a large PR is directly quoted from P2. Planning's outputs (acceptance criteria, risk assessment) are confirmed by P3. The one unsupported claim is an editorial inference: that using Planning before coding means "by the time you reach the PR stage, the scope and expectations are already clearly defined." P3 describes Planning's outputs and timing but never makes this specific causal connection to the PR stage.

**Claims identified by judge (thinking off):**

Supported:
  - Deep Review is explicitly aliased as 'end-to-end review'
  - Deep Review is explicitly aliased as 'PR review before merging'
  - Deep Review performs a multi-pass deep code review
  - Deep Review covers security, quality, and test gap analysis
  - Deep Review delivers prioritized findings
  - The usage syntax is deep_review(path='...')
  - The path parameter accepts a path to a directory or file
  - Deep Review is related to a Security Audit tool
  - Deep Review is related to a Code Review tool
  - Bug prediction scans for dangerous eval() use
  - Bug prediction scans for broad exception handling
  - Bug prediction scans for incomplete code paths
  - Running this analysis before merging is recommended to catch patterns humans miss during large PR reviews
  - Running this analysis is recommended to focus human attention on real risks during code review
  - Running this analysis is recommended to verify no new high-severity patterns crept in before a release

Unsupported:
  - Deep Review connects bug prediction as part of its own multi-pass review (the passages list Bug Predict as a related tool, not as a component of Deep Review's analysis)
  - Deep Review is described as a 'thorough, all-in-one pre-merge checkpoint' (this characterization goes beyond what the passages state)

**Claims identified by judge (thinking on):**

Supported:
  - Deep Review is listed under the alias 'end-to-end review'
  - Deep Review is listed under the alias 'PR review before merging'
  - Deep Review runs a multi-pass deep code review
  - Deep Review covers security, quality, and test gap analysis
  - Deep Review delivers prioritized findings
  - The usage syntax for Deep Review is deep_review(path='...')
  - The path parameter is the only required parameter for Deep Review
  - The path parameter accepts a path to a directory or file to review
  - Security Audit is a related tool that runs a security audit workflow on the codebase
  - Bug Predict analyzes code patterns to identify where failures are most likely to happen
  - Bug Predict is explicitly recommended for use 'before merging a large PR — catch patterns humans miss'
  - Code Review is a related tool that provides comprehensive code quality analysis
  - Planning can help define acceptance criteria before writing code
  - Planning provides risk assessments as an output

Unsupported:
  - By the time you reach the PR stage, the scope and expectations are already clearly defined [as a result of using Planning]

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

## gq-013 — `orchestrate documentation workflow`

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
> [P2] source: concepts/tool-workflow-orchestration.md
> ---
> type: concept
> name: tool-workflow-orchestration
> tags: [workflows, orchestration, routing]
> source: plugin/skills/workflow-orchestration/SKILL.md
> ---
>
> # Workflow Orchestration
>
> Workflow orchestration runs multiple analysis workflows in
> sequence and combines the results into a unified report.
> Instead of running security, code review, and test audits
> one at a time and piecing together the findings yourself,
> you describe what you want and it runs them back-to-back,
> then hands you a single combined report.
>
> ## What it runs
>
> | Workflow | What it does | Typical time |
> |----------|-------------|-------------|
> | **Security Audit** | Scans for vulnerabilities, eval/exec, path traversal, secrets | ~2 min |
> | **Code Review** | Quality, correctness, style, and architecture review | ~3 min |
> | **Bug Prediction** | Detects likely bug patterns and failure hotspots | ~1 min |
> | **Deep Review** | Multi-pass: security + quality + test gap analysis | ~5 min |
> | **Test Generation** | Creates unit tests for uncovered code | ~3 min |
> | **Test Audit** | Coverage audit and gap detection | ~2 min |
> | **Doc Audit** | Documentation freshness and gap analysis | ~1 min |
> | **Doc Generation** | Generates docs from source code | ~2 min |
> | **Performance Audit** | Bottleneck detection and optimization tips | ~2 min |
> | **Release Prep** | Health checks, changelog, dependency audits | ~3 min |
>
> ## When to use it
>
> Run workflow orchestration when one analysis isn't
> enough:
>
> - **Before a release** -- run security + test audit +
>   release prep in one pass
> - **Comprehensive review** -- run code review + bug
>   prediction + deep review for full coverage
> - **New contributor onboarding** -- run doc audit + code
>   review to understand a module's state
> - **CI gate** -- chain security + test audit to block
>   PRs with critical findings
>
> If you only need a single workflow, call it directly
> (e.g., `/security-audit src/`). Orchestration shines
> when you need two or more.
>
> ## How it works
>
> 1. You say which workflows to run (or describe what
>    you need and it picks them)
> 2. It asks which path or files to analyze
> 3. It runs each workflow in sequence on that path
> 4. It combines findings into a single scored report
>    grouped by severity
>
> The skill runs on your Claude subscription -- no API
> key or additional cost.
>
> ## Want to learn more?
>
> - Say **"tell me more"** for step-by-step instructions
> - Say **"what is security audit?"** to learn about
>   individual workflows
> - Say **"what is deep review?"** for multi-pass analysis
> </passage>
>
> <passage>
> [P3] source: concepts/meta-orchestration.md
> ---
> type: concept
> name: meta-orchestration
> tags: [architecture, workflow]
> source: src/attune/orchestration/
> ---
>
> # Concept: Meta-orchestration patterns
>
> ## What
>
> Six composition patterns for combining workflows into pipelines: Sequential, Parallel, Debate, Teaching, Refinement, and Adaptive.
>
> ## Why
>
> Complex tasks require multiple analysis passes. A release prep combines security audit, test generation, and documentation review in a single orchestrated pipeline.
>
> ## How
>
> The orchestration module composes BaseWorkflow instances using the selected pattern. Each pattern defines execution order, data flow between stages, and quality gates.
>
> ## Example
>
> Secure Release uses Sequential: security_audit -> dependency_check -> release_prep.
>
> ## Related Topics
>
> _No related topics yet._
> </passage>

### Answer

## Orchestrating the Documentation Workflow

The **Doc Orchestrator** tool handles end-to-end documentation maintenance by scouting gaps, prioritizing, generating, and updating docs [P1].

### How to Use It

**Tool:** `doc_orchestrator()`

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `path` | string | Project root path | `.` |

You can invoke it simply as `doc_orchestrator()` — it defaults to the current directory [P1].

---

### What It Does Under the Hood

The doc orchestration pipeline typically involves two sub-workflows that run in sequence [P2]:

| Workflow | What it does | Typical time |
|----------|-------------|--------------|
| **Doc Audit** | Documentation freshness and gap analysis | ~1 min |
| **Doc Generation** | Generates docs from source code | ~2 min |

Each workflow runs in sequence, and findings are combined into a single scored report [P2].

---

### When to Use It

Workflow orchestration is especially useful when a single analysis isn't enough. For example, pairing a **doc audit + code review** is recommended for **new contributor onboarding** to understand a module's state [P2].

If you only need a single workflow (e.g., just a doc audit), you can call it directly instead — orchestration shines when you need two or more workflows combined [P2].

---

### Related Tools

The Doc Orchestrator is grouped alongside other workflow tools including **Security Audit**, **Bug Predict**, and **Code Review** [P1].


### Judge verdicts (off → on)

| | Score | Supported | Unsupported |
| - | -: | -: | -: |
| Thinking off | 0.941 | 16 | 1 |
| Thinking on  | 0.895 | 17 | 2 |

**Reasoning (thinking off):**

> Most claims in the answer are well-supported by the retrieved passages. P1 directly supports the tool name, description, usage, parameters, defaults, and related tools. P2 supports the individual workflow descriptions, timing, use cases (including new contributor onboarding), and the recommendation to call a single workflow directly. However, the claim that the Doc Orchestrator pipeline *specifically* involves Doc Audit and Doc Generation running in sequence is an inference not directly supported by the passages. P1 only says it does 'scout gaps, prioritize, generate, and update docs' without naming specific sub-workflows, and P2 lists Doc Audit and Doc Generation as part of a broader list of workflows available to general orchestration — it does not state that the Doc Orchestrator tool specifically runs these two in sequence.

**Reasoning (thinking on):**

> The answer is largely faithful to the retrieved passages. Most claims about the Doc Orchestrator's description, usage syntax, parameter, and the broader orchestration behavior (sequential execution, combined report, onboarding use-case) are all directly lifted from P1 and P2. However, two claims are unsupported. First, the answer frames Doc Audit + Doc Generation as the two specific internal "stages" of the Doc Orchestrator tool — but P1 never names those sub-workflows, and P2 only lists them as options in a general orchestration menu, not as the fixed internals of doc_orchestrator(). This is an inference beyond what the passages state. Second, the answer substitutes "just a doc audit" as the example for calling a single workflow directly, whereas P2's actual example is "/security-audit src/"; the doc-audit substitution is a minor but unsupported invention.

**Claims identified by judge (thinking off):**

Supported:
  - The Doc Orchestrator tool handles end-to-end documentation maintenance
  - The Doc Orchestrator scouts gaps, prioritizes, generates, and updates docs
  - The tool is invoked as `doc_orchestrator()`
  - The `path` parameter is of type string
  - The `path` parameter describes the project root path
  - The `path` parameter defaults to `.`
  - Doc Audit performs documentation freshness and gap analysis
  - Doc Audit takes approximately 1 minute
  - Doc Generation generates docs from source code
  - Doc Generation takes approximately 2 minutes
  - Findings are combined into a single scored report
  - Pairing doc audit + code review is recommended for new contributor onboarding to understand a module's state
  - Orchestration is useful when a single analysis isn't enough
  - If you only need a single workflow, you can call it directly
  - Orchestration shines when you need two or more workflows combined
  - The Doc Orchestrator is grouped alongside Security Audit, Bug Predict, and Code Review

Unsupported:
  - The doc orchestration pipeline typically involves two sub-workflows (Doc Audit and Doc Generation) that run in sequence

**Claims identified by judge (thinking on):**

Supported:
  - The Doc Orchestrator tool handles end-to-end documentation maintenance.
  - The Doc Orchestrator scouts gaps, prioritizes, generates, and updates docs.
  - The tool is called with doc_orchestrator().
  - The tool accepts a `path` parameter of type string.
  - The `path` parameter represents the project root path.
  - The default value for `path` is `.`.
  - Doc Audit performs documentation freshness and gap analysis.
  - Doc Audit takes approximately 1 minute.
  - Doc Generation generates docs from source code.
  - Doc Generation takes approximately 2 minutes.
  - Documentation orchestration is particularly useful for new contributor onboarding.
  - Running a doc audit + code review helps understand a module's state.
  - If you only need a single workflow, you can call it directly instead of using orchestration.
  - Orchestration shines when you need two or more workflows.
  - Each workflow runs in sequence on a given path.
  - Findings are combined into a single scored report grouped by severity.
  - The Doc Orchestrator belongs to the workflow group alongside Security Audit, Bug Predict, and Code Review tools.

Unsupported:
  - The documentation workflow (Doc Orchestrator) specifically consists of two key stages: Doc Audit and Doc Generation — the passages never state that the Doc Orchestrator tool runs exactly these two sub-workflows internally.
  - The example of calling a single workflow directly is 'just a doc audit' — P2's actual example is '/security-audit src/', not a doc audit; the substitution is not supported by the passages.

### Your labels

```yaml
id: gq-013
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

### Retrieved context

> <passage>
> [P1] source: concepts/tool-refactor-plan.md
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
> [P3] source: references/tool-simplify-code.md
> ---
> type: reference
> subtype: tabular
> name: tool-simplify-code
> category: tool
> tags: [mcp, tool, workflow]
> source: src/attune/mcp/tool_schemas.py
> ---
>
> # Reference: Tool: Simplify Code
>
> Find complex code hotspots and suggest simplifications to reduce cognitive load.
>
> **Group:** workflow
>
> ## Parameters
>
> | Parameter | Type | Description | Constraints | Default |
> | --------- | ---- | ----------- | ----------- | ------- |
> | `path` | string | Directory to scan for complexity |  | . |
>
> ## Usage
>
> `simplify_code()`
>
> ## Related Topics
> - **Reference**: Tool: Security Audit — Run security audit workflow on codebase. Detects vulnerabili...
> - **Reference**: Tool: Bug Predict — Run bug prediction workflow. Analyzes code patterns and pred...
> - **Reference**: Tool: Code Review — Run code review workflow. Provides comprehensive code qualit...
> </passage>

### Answer

# Reducing Code Complexity

There are two tools well-suited for reducing code complexity in your codebase:

---

## 1. Simplify Code

The `simplify_code()` tool finds complex code hotspots and suggests simplifications to reduce cognitive load [P3]. It scans a directory (defaulting to the current directory) and can be run with no required parameters [P3].

---

## 2. Refactor Plan

A refactor plan builds a prioritized roadmap to fix structural problems, including complexity issues [P1]. Specifically, it analyzes:

- **Complexity** — high cyclomatic complexity, deep nesting, and long chains [P1]
- **Code smells** — long methods, god classes, and other structural issues [P1]
- **Dead code** — unreachable branches, unused parameters, and vestigial modules [P1]

The roadmap prioritizes fixes by **severity**, **effort**, **impact**, and **risk**, so high-severity, low-effort, high-impact items rise to the top [P1]. This is especially useful before adding features to a tangled area, or when you need data to justify refactoring time to stakeholders [P1].

---

## 3. Bug Predict (Complementary)

As a complementary step, the Bug Predict tool also flags complexity signals — such as deeply nested conditionals and long function bodies — since these correlate with higher defect rates [P2].

---

**Suggested approach:** Run `simplify_code()` for immediate hotspot detection, then use a refactor plan to build a prioritized, structured roadmap for tackling the deeper complexity issues [P1, P3].


### Judge verdicts (off → on)

| | Score | Supported | Unsupported |
| - | -: | -: | -: |
| Thinking off | 0.938 | 15 | 1 |
| Thinking on  | 0.895 | 17 | 2 |

**Reasoning (thinking off):**

> Almost all factual claims in the answer are directly supported by the retrieved passages. P3 supports all claims about `simplify_code()`. P1 supports all claims about the refactor plan's analysis categories, prioritization factors, and use cases. P2 supports the claims about Bug Predict flagging cyclomatic complexity and long function bodies as correlating with higher defect rates. The only unsupported claim is the suggested two-step approach ("Run simplify_code() first, then refactor plan"), which is an actionable recommendation synthesized by the answer author and not explicitly stated in any passage. This is a minor inference/editorial addition not found in the source material.

**Reasoning (thinking on):**

> The vast majority of claims in the answer are directly and precisely supported by the retrieved passages — tool descriptions, parameter defaults, complexity categories, prioritization factors, and the yak-shaving quote are all verbatim or near-verbatim matches to P1, P2, and P3. Two claims are unsupported: (1) the characterization of `simplify_code()` as a "quick" scan — the passages describe its function and usage but never use the word "quick"; and (2) the suggestion to run Bug Predict "alongside complexity reduction to prioritize the riskiest areas first" — P2 lists several scenarios for when to use the tool but says nothing about pairing it with complexity reduction work for prioritization purposes. This latter claim is a reasonable inference but is not grounded in any passage.

**Claims identified by judge (thinking off):**

Supported:
  - The `simplify_code()` tool finds complex code hotspots and suggests simplifications to reduce cognitive load
  - The `simplify_code()` tool scans a directory
  - The `simplify_code()` tool defaults to the current directory
  - The `simplify_code()` tool can be run with no required parameters
  - A refactor plan builds a prioritized roadmap to fix structural problems
  - A refactor plan analyzes complexity, including high cyclomatic complexity, deep nesting, and long chains
  - A refactor plan analyzes code smells, including long methods and god classes
  - A refactor plan analyzes dead code, including unreachable branches, unused parameters, and vestigial modules
  - The roadmap prioritizes fixes by severity, effort, impact, and risk
  - High-severity, low-effort, high-impact items rise to the top of the roadmap
  - A refactor plan is especially useful before adding features to a tangled area
  - A refactor plan is useful when you need data to justify refactoring time to stakeholders
  - The Bug Predict tool flags deeply nested conditionals as a complexity signal
  - The Bug Predict tool flags long function bodies as a complexity signal
  - Deeply nested conditionals and long function bodies correlate with higher defect rates

Unsupported:
  - Suggested approach: Run `simplify_code()` for immediate hotspot detection, then use a refactor plan to build a prioritized, structured roadmap for tackling the deeper complexity issues

**Claims identified by judge (thinking on):**

Supported:
  - The `simplify_code()` tool finds complex code hotspots and suggests simplifications to reduce cognitive load.
  - `simplify_code()` scans a directory, defaulting to the current directory (`.`).
  - `simplify_code()` can be invoked with no arguments: `simplify_code()`.
  - A Refactor Plan scans your code for structural problems and builds a prioritized roadmap to fix them.
  - The Refactor Plan targets high cyclomatic complexity as a complexity issue.
  - The Refactor Plan targets deep nesting as a complexity issue.
  - The Refactor Plan targets long chains as a complexity issue.
  - The Refactor Plan catches long methods as a code smell.
  - The Refactor Plan catches god classes as a code smell.
  - The Refactor Plan catches code duplication (copy-pasted blocks, near-duplicates, DRY violations).
  - The Refactor Plan catches dead code (unreachable branches, unused params, vestigial modules).
  - Refactoring without a plan can lead to yak-shaving — starting to fix one thing and ending up touching 20 files.
  - The roadmap tells you which changes deliver the most improvement per hour invested.
  - Refactor Plan items are prioritized by severity, effort, impact, and risk.
  - High-severity, low-effort, high-impact items float to the top of the Refactor Plan.
  - Risky changes get flagged so you can plan extra testing.
  - The Bug Predict tool weighs cyclomatic complexity — deeply nested conditionals and long function bodies — as a factor that correlates with higher defect rates.

Unsupported:
  - Running Bug Predict alongside complexity reduction can help you prioritize the riskiest areas first.
  - `simplify_code()` provides a 'quick' scan (the passages do not characterize it as quick).

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

Here are two tools that can help you find bugs in your code:

## Option 1: Bug Prediction

Bug prediction scans your codebase for patterns that historically cause production incidents [P1]. It looks for three categories of risk [P1]:

- **Dangerous eval** (HIGH severity) — `eval()`, `exec()`, `compile()` on any input [P1]
- **Broad exceptions** (MEDIUM severity) — bare `except:` or unlogged `except Exception:` that silently swallow errors [P1]
- **Incomplete code** (LOW severity) — TODO, FIXME, HACK, XXX comments indicating unfinished code paths [P1]

It also factors in contextual signals like cyclomatic complexity, change frequency, and code smells such as functions over 50 lines [P1].

## Option 2: Code Quality Review

A code quality review looks at your code from multiple angles at once — style, correctness, likely bugs, and structural health — giving you one unified pass instead of multiple separate reports [P3].

It can catch things like broad exceptions, mutable defaults, race conditions, unreachable code, and wrong return types [P3].

---

## How to Run Either

You can use the `/code-quality` command followed by your path or directory [P2]:

```
/code-quality <path or directory to review>
```

This covers both code quality and bug prediction in one pass [P2].

**What to expect:** You'll be asked a couple of questions to scope the work — which files to review and how deep to go — before results are returned [P3].


### Judge verdicts (off → on)

| | Score | Supported | Unsupported |
| - | -: | -: | -: |
| Thinking off | 0.933 | 14 | 1 |
| Thinking on  | 0.895 | 17 | 2 |

**Reasoning (thinking off):**

> Nearly all claims in the answer are directly supported by the retrieved passages. The one unsupported claim is that `/code-quality` covers both code quality and bug prediction in one pass. While P2 describes `/code-quality` as combining "code review and bug prediction," P2's quickstart description says it finds "quality issues, style violations, and likely bugs" — it does not explicitly state it runs the full bug prediction scanner described in P1 as a single unified pass. The answer's phrasing "covers both code quality and bug prediction in one pass" goes slightly beyond what P2 explicitly states and conflates the two distinct tools described in P1 and P3. All other claims are clearly and explicitly supported by the passages.

**Reasoning (thinking on):**

> The vast majority of claims in the answer are directly and explicitly supported by the retrieved passages — P1 covers bug prediction categories, severity levels, and contextual signals; P2 confirms the /code-quality command syntax and structured results; and P3 covers the code quality review categories, severities, and unified-pass framing. The only unsupported claims are the closing comparative recommendations ("focused bug hunt → bug prediction" vs. "broader health check → code quality"), which are reasonable inferences drawn from combining the two tool descriptions but are not explicitly stated as a comparative guide in any passage.

**Claims identified by judge (thinking off):**

Supported:
  - Bug prediction scans your codebase for patterns that historically cause production incidents.
  - Bug prediction looks for three categories of risk.
  - Dangerous eval is HIGH severity and covers eval(), exec(), compile() on any input.
  - Broad exceptions is MEDIUM severity and covers bare except: or unlogged except Exception: that silently swallow errors.
  - Incomplete code is LOW severity and covers TODO, FIXME, HACK, XXX comments indicating unfinished code paths.
  - Bug prediction factors in contextual signals like cyclomatic complexity.
  - Bug prediction factors in change frequency.
  - Bug prediction factors in code smells such as functions over 50 lines.
  - A code quality review looks at your code from multiple angles at once — style, correctness, likely bugs, and structural health.
  - A code quality review gives you one unified pass instead of multiple separate reports.
  - Code quality review can catch broad exceptions, mutable defaults, race conditions, unreachable code, and wrong return types.
  - You can use the /code-quality command followed by your path or directory.
  - The /code-quality command syntax is: /code-quality <path or directory to review>.
  - You'll be asked a couple of questions to scope the work — which files to review and how deep to go — before results are returned.

Unsupported:
  - The /code-quality command covers both code quality and bug prediction in one pass.

**Claims identified by judge (thinking on):**

Supported:
  - Bug prediction scans your codebase for patterns that historically cause production incidents.
  - Bug prediction looks for three categories of risk.
  - Dangerous eval is HIGH severity.
  - Dangerous eval involves eval(), exec(), compile() on any input.
  - Broad exceptions is MEDIUM severity.
  - Broad exceptions involves bare except: or unlogged except Exception: that silently swallow errors.
  - Incomplete code is LOW severity.
  - Incomplete code involves TODO, FIXME, HACK, XXX comments indicating unfinished code paths.
  - Bug prediction weighs contextual signals like cyclomatic complexity, change frequency, and code smells.
  - Functions over 50 lines are an example of a code smell.
  - A code quality review catches bugs as part of a broader pass.
  - Likely bugs (in code quality review) include broad exceptions, mutable defaults, and race conditions at High severity.
  - Correctness issues include unreachable code and wrong return types at High severity.
  - Security overlap includes eval/exec and unvalidated paths at Critical severity, in deep scans only.
  - You can use the /code-quality command followed by a path or directory.
  - The /code-quality command gives structured results.
  - Code quality review covers style, correctness, likely bugs, and structural health in one unified pass.

Unsupported:
  - If you want a focused bug hunt, use bug prediction.
  - If you want a broader health check that includes bugs, use code quality review.

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

# Deep Review Your PR

To run a deep review, use the `deep_review` tool, which performs a **multi-pass deep code review covering security, quality, and test gap analysis with prioritized findings** [P1].

## How to Use It

Call the tool with the path to your PR's directory or file:

```
deep_review(path="...")
```

Simply replace `"..."` with the path to the code you want reviewed (e.g., `"src/"`) [P1].

## Parameters

| Parameter | Type | Required |
|-----------|------|----------|
| `path` | string | ✅ Yes |

The `path` parameter accepts either a **directory or file path** [P1].

## What to Do After

Once the deep review is complete, you may want to follow up with related tools such as **Security Audit** or **Bug Predict** for additional analysis [P1].


### Judge verdicts (off → on)

| | Score | Supported | Unsupported |
| - | -: | -: | -: |
| Thinking off | 0.900 | 9 | 1 |
| Thinking on  | 0.909 | 10 | 1 |

**Reasoning (thinking off):**

> Almost all claims in the answer are directly and explicitly supported by passage P1, which describes the `deep_review` tool, its parameters, usage syntax, and related topics. The one unsupported claim is the recommendation to follow up with Security Audit or Bug Predict after a deep review: while P1 does list these as related topics, no passage explicitly recommends them as post-deep-review follow-up steps. The only explicit post-review follow-up suggestion in the passages (P2) is for `smart-test` after a `code-review`, not after `deep_review`.

**Reasoning (thinking on):**

> Almost all claims in the answer are directly supported by P1 (tool-deep-review.md), which explicitly documents the tool's description, parameters, usage syntax, aliases, and related topics. The one unsupported claim is the suggestion to follow up with "test generation" after a deep review: P1's related topics list Security Audit, Bug Predict, and Code Review — but not Test Generation. Test Generation appears only in P3 (tool-code-review.md) as a related topic for the separate `code_review` tool, not for `deep_review`. The answer incorrectly attributes this follow-up suggestion to P1.

**Claims identified by judge (thinking off):**

Supported:
  - The tool to use is called `deep_review`.
  - The deep review performs a multi-pass deep code review.
  - The deep review covers security, quality, and test gap analysis with prioritized findings.
  - The usage syntax is `deep_review(path="...")`.
  - The `path` parameter is of type string.
  - The `path` parameter is required.
  - The `path` parameter accepts either a directory or file path.
  - Security Audit is a related tool listed alongside deep_review.
  - Bug Predict is a related tool listed alongside deep_review.

Unsupported:
  - After the deep review is complete, you may want to follow up with related tools such as Security Audit or Bug Predict for additional analysis (the passages list them as related topics but do not recommend them as post-deep-review follow-ups; the post-review follow-up recommendation in P2 is for `smart-test` after code-review, not deep_review).

**Claims identified by judge (thinking on):**

Supported:
  - The `deep_review` tool performs a multi-pass deep code review.
  - The deep review covers security, quality, and test gap analysis with prioritized findings.
  - The usage syntax is `deep_review(path="...")`.
  - The `path` parameter is required.
  - The `path` parameter accepts a directory or file to review.
  - The deep review tool has the alias 'end-to-end review'.
  - The deep review tool has the alias 'PR review before merging'.
  - The deep review tool has the alias 'comprehensive code review'.
  - After a deep review, Security Audit is a related follow-up tool.
  - After a deep review, Bug Predict is a related follow-up tool.

Unsupported:
  - You can follow up with test generation to address any test gaps identified from the deep review.

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

## gq-008 — `prepare a release`

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
> [P3] source: references/skill-release-prep.md
> ---
> type: reference
> subtype: procedural
> name: skill-release-prep
> category: skill
> tags: [release, skill, plugin, reference]
> source: plugin/skills/release-prep/SKILL.md
> ---
>
> # Release Prep Reference
>
> Complete reference for the release prep skill --
> every check it runs, how scoring works, blocker types,
> and configuration options.
>
> ## Invocation
>
> ```
> /release-prep <version or 'check'>
> ```
>
> Or natural language:
>
> ```
> is this ready to release?
> prepare release 5.6.0
> run the release preflight
> check if I can ship this
> ```
>
> ## Guided scoping
>
> The skill asks two questions before running. Both accept
> natural language answers.
>
> | Question | Options | Default |
> |----------|---------|---------|
> | What version? | A semver string, or "check" to auto-detect | Current version in pyproject.toml |
> | What stage? | Full prep, prep check, changelog only, security only | Full prep |
>
> Runs on your Claude subscription -- no API key or
> additional cost.
>
> ## All check areas
>
> ### Health
>
> | Check | What it verifies | Blocking? | Typical issue |
> |-------|------------------|-----------|---------------|
> | Test suite | All tests pass | Yes | Failing test from recent merge |
> | Lint | ruff + black clean | Yes | Auto-fixable style violation |
> | Coverage | Above configured threshold | Advisory | Coverage dip from new code |
> | Type hints | No new mypy errors | Advisory | Missing annotation on public API |
>
> ### Security
>
> | Check | What it verifies | Blocking? | Typical issue |
> |-------|------------------|-----------|---------------|
> | eval/exec scan | No dangerous eval or exec | Yes | eval() in utility function |
> | Secrets scan | No hardcoded credentials | Yes | API key in test fixture |
> | CVE check | No known vulnerabilities in deps | Yes | Outdated dependency with CVE |
> | Path validation | File ops use validated paths | Advisory | Missing _validate_file_path() |
>
> ### Changelog
>
> | Check | What it verifies | Blocking? | Typical issue |
> |-------|------------------|-----------|---------------|
> | Version entry | Section exists for target version | Yes | Forgot to add entry |
> | Date | Release date is current or today | Advisory | Stale date from earlier prep |
> | Format | Follows Keep a Changelog structure | Advisory | Missing category headers |
> | Completeness | All merged PRs accounted for | Advisory | PR merged after changelog written |
>
> ### Dependencies
>
> | Check | What it verifies | Blocking? | Typical issue |
> |-------|------------------|-----------|---------------|
> | Pinned versions | Lower bounds above known CVEs | Yes | pydantic>=2.0.0 allows vulnerable range |
> | Compatibility | No conflicting version ranges | Yes | Two deps require different major versions |
> | Lock file | Lock file in sync with pyproject.toml | Advisory | Forgot to run uv lock |
> | Unused deps | No dependencies without imports | Advisory | Leftover from removed feature |
>
> ### Version
>
> | Check | What it verifies | Blocking? | Typical issue |
> |-------|------------------|-----------|---------------|
> | Semver validity | Version string is valid semver | Yes | Typo like 5.6.0.1 |
> | Bump type | Matches changes (breaking = major) | Advisory | Minor bump with breaking change |
> | pyproject.toml | Version updated in source | Yes | Version still says old number |
> | Dist build | `python -m build` succeeds | Yes | Missing MANIFEST.in entry |
>
> ## Output format
>
> ```markdown
> ## Release Readiness Assessment
>
> **Verdict:** GO / NO-GO
> **Version:** X.Y.Z -> A.B.C
> **Date:** YYYY-MM-DD
>
> ### Check Results
>
> | Area | Status | Summary |
> |------|--------|---------|
> | Health | PASS | 15,482 tests passing, 87% coverage |
> | Security | PASS | No critical findings |
> | Changelog | FAIL | No entry for vA.B.C |
> | Dependencies | PASS | All pinned, no CVEs |
> | Version | PASS | Semver valid, dist builds |
>
> ### Blockers
>
> | Blocker | Area | Severity | Fix |
> |---------|------|----------|-----|
> | No changelog entry | Changelog | Blocking | Add vA.B.C section |
>
> ### Advisories
>
> | Advisory | Area | Recommendation |
> |----------|------|----------------|
> | Coverage dipped 2% | Health | Add tests for new module |
>
> ### Recommendations
>
> 1. Add a vA.B.C section to CHANGELOG.md
> 2. Re-run release prep after fixing
> ```
>
> ## Scoring and verdicts
>
> The verdict is binary: GO or NO-GO.
>
> | Verdict | Meaning |
> |---------|---------|
> | **GO** | Zero blocking issues. Safe to tag and publish. |
> | **NO-GO** | One or more blocking issues. Fix before releasing. |
>
> Advisory findings do not block the release but are
> included in the report for awareness.
>
> ## Blocker types
>
> | Type | Severity | Effect |
> |------|----------|--------|
> | **Blocking** | Must fix | Prevents GO verdict |
> | **Advisory** | Should fix | Included in report, does not block |
>
> A single blocking issue is enough for a NO-GO verdict.
> The report lists all blockers so you can fix them in
> one pass.
>
> ## Configuration
>
> ### Coverage threshold
>
> Set the minimum coverage in your project config:
>
> ```yaml
> # attune.config.yml
> release_prep:
>   coverage_threshold: 80
> ```
>
> ### Excluding paths
>
> Skip directories that don't need release checks:
>
> ```yaml
> # attune.config.yml
> release_prep:
>   exclude:
>     - "benchmarks/**"
>     - "scripts/**"
>     - "docs/**"
> ```
>
> ### Skipping specific checks
>
> Disable individual check areas if they don't apply:
>
> ```yaml
> # attune.config.yml
> release_prep:
>   skip:
>     - type_hints
>     - unused_deps
> ```
>
> ## After the assessment
>
> | Goal | What to say |
> |------|-------------|
> | Fix all blockers | "fix the blockers" |
> | Update changelog | "update the changelog for this release" |
> | Re-run after fixes | "run release prep again" |
> | Tag and publish | "tag and publish" |
> | Export for CI | "export the assessment as JSON" |
> | Compare with last run | "compare with the previous assessment" |
>
> ## Want to learn more?
>
> - Say **"what is release prep?"** to go back to the
>   overview
> - Say **"how do I run release prep?"** for the
>   step-by-step guide
> - Say **"run a security audit"** to scan for
>   vulnerabilities separately
> - Say **"check my test coverage"** to analyze test
>   gaps before releasing
> </passage>

### Answer

## Preparing a Release

To prepare a release, use the **release prep** skill, which runs a preflight checklist before you publish. [P1]

### How to Invoke It

Run the following command in your Claude Code conversation: [P2]

```
/release-prep <version or 'check'>
```

You can also use natural language, such as: [P3]

```
is this ready to release?
prepare release 5.6.0
run the release preflight
check if I can ship this
```

### Guided Scoping

Before running, the skill will ask you two questions: [P3]

| Question | Options | Default |
|----------|---------|---------|
| What version? | A semver string, or "check" to auto-detect | Current version in pyproject.toml |
| What stage? | Full prep, prep check, changelog only, security only | Full prep |

### What Gets Checked

The skill checks five areas: [P1, P3]

| Check Area | What It Verifies | Blocking? |
|------------|-----------------|-----------|
| **Health** | Tests pass, lint clean, coverage above threshold | Yes if tests fail |
| **Security** | No new CVEs, no eval/exec, secrets scan clean | Yes if critical findings |
| **Changelog** | Entry exists for this version, date is current | Yes if missing |
| **Dependencies** | Pinned versions, no known vulnerabilities, compatible ranges | Yes if vulnerable dep |
| **Version** | Semver bump matches changes, pyproject.toml updated, dist builds cleanly | Yes if version mismatch |

### The Verdict

After all checks complete, you'll receive either: [P1]

- **GO** — every check passed or has only advisory warnings. Safe to tag, build, and publish.
- **NO-GO** — one or more blocking issues found. The report lists exactly what to fix before retrying.

### After the Assessment

Depending on the result, you can say: [P3]

| Goal | What to Say |
|------|-------------|
| Fix all blockers | "fix the blockers" |
| Update changelog | "update the changelog for this release" |
| Re-run after fixes | "run release prep again" |
| Tag and publish | "tag and publish" |


### Judge verdicts (off → on)

| | Score | Supported | Unsupported |
| - | -: | -: | -: |
| Thinking off | 1.000 | 24 | 0 |
| Thinking on  | 1.000 | 24 | 0 |

**Reasoning (thinking off):**

> Every atomic factual claim in the answer is directly and explicitly supported by the retrieved passages. The invocation syntax and natural language alternatives come from P2 and P3. The guided scoping questions, their options, and defaults are verbatim from P3. The five check areas and their blocking conditions are covered in both P1 and P3. The GO/NO-GO verdict descriptions are taken directly from P1 and P3. The post-assessment action phrases are listed verbatim in the "After the assessment" table in P3. No invented details, CLI flags, or outside knowledge were introduced.

**Reasoning (thinking on):**

> Every atomic factual claim in the answer is directly and explicitly supported by at least one of the retrieved passages. The command syntax, natural-language aliases, the two guided-scoping questions (including their options and defaults), all five check areas and their descriptions, the binary GO/NO-GO verdict logic, and each of the four post-assessment follow-up phrases are all verbatim or near-verbatim reproductions of content from P1, P2, and/or P3. No outside knowledge, invented details, or unsupported inferences were introduced. The one potentially inferential linkage — that "tag and publish" should be used once a GO verdict is obtained — is directly grounded in P3, which both lists "tag and publish" as a post-assessment action and defines GO as "safe to tag and publish."

**Claims identified by judge (thinking off):**

Supported:
  - To prepare a release, use the release prep skill
  - Release prep runs a preflight checklist before you publish
  - The command to invoke it is /release-prep <version or 'check'>
  - The command is run in your Claude Code conversation
  - You can use natural language such as 'is this ready to release?'
  - You can use natural language such as 'prepare release 5.6.0'
  - You can use natural language such as 'run the release preflight'
  - You can use natural language such as 'check if I can ship this'
  - Before running, the skill asks two questions
  - The first question is 'What version?' with options of a semver string or 'check' to auto-detect, defaulting to the current version in pyproject.toml
  - The second question is 'What stage?' with options of full prep, prep check, changelog only, or security only, defaulting to full prep
  - The skill checks five areas: Health, Security, Changelog, Dependencies, and Version
  - Health checks: tests pass, lint clean, coverage above threshold; blocking if tests fail
  - Security checks: no new CVEs, no eval/exec, secrets scan clean; blocking if critical findings
  - Changelog checks: entry exists for this version, date is current; blocking if missing
  - Dependencies checks: pinned versions, no known vulnerabilities, compatible ranges; blocking if vulnerable dep
  - Version checks: semver bump matches changes, pyproject.toml updated, dist builds cleanly; blocking if version mismatch
  - After all checks complete, you receive either a GO or NO-GO verdict
  - GO means every check passed or has only advisory warnings and is safe to tag, build, and publish
  - NO-GO means one or more blocking issues found and the report lists exactly what to fix before retrying
  - After the assessment, you can say 'fix the blockers' to fix all blockers
  - After the assessment, you can say 'update the changelog for this release' to update the changelog
  - After the assessment, you can say 'run release prep again' to re-run after fixes
  - After the assessment, you can say 'tag and publish' to tag and publish

Unsupported:


**Claims identified by judge (thinking on):**

Supported:
  - To prepare a release, use the /release-prep skill in Claude Code
  - The command is run with a version number or 'check': /release-prep <version or 'check'>
  - Natural language 'prepare release 5.6.0' can be used to invoke release prep
  - Natural language 'is this ready to release?' can be used to invoke release prep
  - Natural language 'run the release preflight' can be used to invoke release prep
  - Natural language 'check if I can ship this' can be used to invoke release prep
  - The skill asks two scoping questions before running
  - The first scoping question asks what version, with options of a semver string or 'check' to auto-detect, defaulting to the current version in pyproject.toml
  - The second scoping question asks what stage, with options of full prep, prep check, changelog only, or security only, defaulting to full prep
  - Release prep runs checks across five areas
  - The Health check area verifies tests pass, lint is clean, and coverage is above threshold
  - The Security check area verifies no CVEs, no eval/exec, and secrets scan is clean
  - The Changelog check area verifies an entry exists for this version and the date is current
  - The Dependencies check area verifies pinned versions, no vulnerabilities, and compatible ranges
  - The Version check area verifies semver bump validity, pyproject.toml is updated, and dist builds cleanly
  - After all checks, a GO or NO-GO verdict is issued
  - GO means every check passed or has only advisory warnings
  - GO means it is safe to tag, build, and publish
  - NO-GO means one or more blocking issues were found
  - The NO-GO report lists exactly what to fix before retrying
  - After the assessment, saying 'fix the blockers' addresses blocking issues
  - After the assessment, saying 'update the changelog for this release' can be used if the changelog is missing
  - After the assessment, saying 'run release prep again' re-runs after making fixes
  - After the assessment, saying 'tag and publish' is an available follow-up action

Unsupported:


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

## Running a Vulnerability Scan

You have two ways to run a vulnerability scan on your codebase:

### Option 1: Claude Code Skill

```
/security-audit <path or directory to scan>
```

This delivers structured results directly in your Claude Code conversation [P1].

### Option 2: CLI Workflow

```
attune workflow run security-audit --path "src/"
```

This produces severity-grouped findings with CWE identifiers [P3].

---

## What Gets Scanned

A security audit catches the following vulnerability categories [P2]:

| Category | Examples |
|----------|----------|
| **Code injection** | `eval()`, `exec()`, `compile()` on untrusted input |
| **Path traversal** | File operations without path validation |
| **Hardcoded secrets** | API keys, tokens, passwords in source control |
| **SQL/command injection** | String concatenation in queries or shell commands |
| **SSRF** | HTTP requests to user-controlled URLs |
| **Weak cryptography** | MD5/SHA1, hardcoded IVs |

---

## How Deep Should You Scan?

| Depth | Time | What you get |
|-------|------|-------------|
| **Quick** | ~30s | Surface scan — eval/exec, obvious secrets |
| **Standard** | ~2 min | Full pattern matching with severity ratings |
| **Deep** | ~5 min | Multi-pass review with OWASP mapping and fix suggestions |

[P2]

---

## When to Run It

Run a security audit **before releasing a new version**, after adding code that handles files or user input, when pulling in a new dependency, or as a CI gate on pull requests [P2].


### Judge verdicts (off → on)

| | Score | Supported | Unsupported |
| - | -: | -: | -: |
| Thinking off | 1.000 | 17 | 0 |
| Thinking on  | 1.000 | 17 | 0 |

**Reasoning (thinking off):**

> Every atomic factual claim in the answer maps directly to content in one or more of the retrieved passages. The two invocation methods (skill and CLI) are taken verbatim from P1 and P3 respectively, including their described outputs. The vulnerability category table is reproduced faithfully from P2's "What it finds" section. The depth/time/output table is reproduced faithfully from P2's "How deep it goes" section. The "When to Run It" guidance is taken word-for-word from P2's "When you'd use it" section. No claims appear to be invented or sourced from outside knowledge.

**Reasoning (thinking on):**

> Every atomic factual claim in the answer maps directly to explicit statements in the retrieved passages. The two execution methods (Claude Code skill and CLI workflow) are drawn verbatim from P1 and P3 respectively, including their output descriptions. The vulnerability category table, the depth/time/output table, and the "when to run it" list are all reproduced faithfully from P2 without adding outside knowledge, invented flags, or unverified assertions. No hallucinations or prior-knowledge drift were detected.

**Claims identified by judge (thinking off):**

Supported:
  - You can run a vulnerability scan using the Claude Code skill with the command `/security-audit <path or directory to scan>`
  - The Claude Code skill delivers structured results directly in your Claude Code conversation
  - You can run a vulnerability scan using the CLI workflow with the command `attune workflow run security-audit --path "src/"`
  - The CLI workflow produces severity-grouped findings with CWE identifiers
  - A security audit catches code injection vulnerabilities including `eval()`, `exec()`, and `compile()` on untrusted input
  - A security audit catches path traversal vulnerabilities (file operations without path validation)
  - A security audit catches hardcoded secrets (API keys, tokens, passwords in source control)
  - A security audit catches SQL/command injection via string concatenation in queries or shell commands
  - A security audit catches SSRF (HTTP requests to user-controlled URLs)
  - A security audit catches weak cryptography such as MD5/SHA1 and hardcoded IVs
  - A Quick scan takes approximately 30 seconds and provides a surface scan covering eval/exec and obvious secrets
  - A Standard scan takes approximately 2 minutes and provides full pattern matching with severity ratings
  - A Deep scan takes approximately 5 minutes and provides multi-pass review with OWASP mapping and fix suggestions
  - You should run a security audit before releasing a new version
  - You should run a security audit after adding code that handles files or user input
  - You should run a security audit when pulling in a new dependency
  - You should run a security audit as a CI gate on pull requests

Unsupported:


**Claims identified by judge (thinking on):**

Supported:
  - The command to run a security audit as a Claude Code skill is `/security-audit <path or directory to scan>`
  - The Claude Code skill runs in a Claude Code conversation and returns structured results
  - The CLI command to run a security audit is `attune workflow run security-audit --path "src/"`
  - The CLI workflow returns severity-grouped findings with CWE identifiers
  - The Code injection category covers `eval()`, `exec()`, and `compile()` on untrusted input
  - The Path traversal category covers file operations without path validation
  - The Hardcoded secrets category covers API keys, tokens, and passwords in source
  - The SQL/command injection category covers string concatenation in queries or shell commands
  - The SSRF category covers HTTP requests to user-controlled URLs
  - The Weak cryptography category covers MD5/SHA1 for security purposes and hardcoded IVs
  - The Quick depth scan takes ~30 seconds and provides a surface scan of eval/exec and obvious secrets
  - The Standard depth scan takes ~2 minutes and provides full pattern matching with severity ratings
  - The Deep depth scan takes ~5 minutes and provides a multi-pass review with OWASP mapping and fix suggestions
  - Good times to scan include before releasing a new version
  - Good times to scan include after adding code that handles files or user input
  - Good times to scan include when pulling in a new dependency
  - Good times to scan include as a CI gate on pull requests

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

Run the following command:

```
attune workflow run security-audit --path "src/"
```

This produces severity-grouped findings with CWE identifiers [P1].

### Option 2: Using the Claude Code Skill

Use the `/security-audit` command directly in your Claude Code conversation:

```
/security-audit <path or directory to scan>
```

This returns structured results in your Claude Code conversation [P2]. It scans for vulnerabilities including eval/exec usage, path traversal, hardcoded secrets, and injection risks [P2].

---

### What It Checks

The audit looks for [P3]:

| Category | Examples |
|----------|----------|
| **Code injection** | `eval()`, `exec()`, `compile()` on untrusted input |
| **Path traversal** | File operations without path validation |
| **Hardcoded secrets** | API keys, tokens, passwords in source |
| **SQL/command injection** | String concatenation in queries or shell commands |
| **SSRF** | HTTP requests to user-controlled URLs |
| **Weak cryptography** | MD5/SHA1, hardcoded IVs |

### Scan Depth Options

You can choose how deep the scan goes [P3]:

- **Quick** (~30s): Surface scan for eval/exec and obvious secrets
- **Standard** (~2 min): Full pattern matching with severity ratings
- **Deep** (~5 min): Multi-pass review with OWASP mapping and fix suggestions


### Judge verdicts (off → on)

| | Score | Supported | Unsupported |
| - | -: | -: | -: |
| Thinking off | 1.000 | 18 | 0 |
| Thinking on  | 1.000 | 13 | 0 |

**Reasoning (thinking off):**

> Every factual claim in the answer is directly and explicitly supported by one or more of the retrieved passages. The CLI command and its output description come from P1. The Claude Code skill command, its syntax, and the vulnerability categories it covers come from P2. The detailed vulnerability category table and the three scan depth options (Quick, Standard, Deep) with their times and descriptions come from P3. There are no invented details, hallucinated flags, or claims that go beyond what the passages state.

**Reasoning (thinking on):**

> Every atomic factual claim in the answer is directly reproduced from one of the three retrieved passages. The two execution methods (workflow command from P1, /security-audit skill from P2), the full vulnerability category table (P3), and the three-tier depth/time/output table (P3) are all verbatim or near-verbatim from the passages. No outside knowledge, invented flags, or unsupported inferences were introduced.

**Claims identified by judge (thinking off):**

Supported:
  - You can run a security audit using the CLI workflow with the command: attune workflow run security-audit --path "src/"
  - The CLI workflow produces severity-grouped findings with CWE identifiers
  - You can run a security audit using the /security-audit command in Claude Code
  - The /security-audit command syntax is: /security-audit <path or directory to scan>
  - The /security-audit command returns structured results in your Claude Code conversation
  - The /security-audit skill scans for eval/exec usage
  - The /security-audit skill scans for path traversal vulnerabilities
  - The /security-audit skill scans for hardcoded secrets
  - The /security-audit skill scans for injection risks
  - The audit checks for code injection including eval(), exec(), and compile() on untrusted input
  - The audit checks for path traversal via file operations without path validation
  - The audit checks for hardcoded secrets including API keys, tokens, and passwords in source
  - The audit checks for SQL/command injection via string concatenation in queries or shell commands
  - The audit checks for SSRF via HTTP requests to user-controlled URLs
  - The audit checks for weak cryptography including MD5/SHA1 and hardcoded IVs
  - The Quick scan depth takes approximately 30 seconds and performs a surface scan for eval/exec and obvious secrets
  - The Standard scan depth takes approximately 2 minutes and performs full pattern matching with severity ratings
  - The Deep scan depth takes approximately 5 minutes and performs multi-pass review with OWASP mapping and fix suggestions

Unsupported:


**Claims identified by judge (thinking on):**

Supported:
  - The command `attune workflow run security-audit --path "src/"` scans your codebase.
  - The attune workflow command produces severity-grouped findings with CWE identifiers.
  - The /security-audit skill is used in Claude Code with the syntax `/security-audit <path or directory to scan>`.
  - The /security-audit skill delivers structured results in your Claude Code conversation.
  - The security audit checks for code injection: eval(), exec(), and compile() on untrusted input.
  - The security audit checks for path traversal: file operations without path validation.
  - The security audit checks for hardcoded secrets: API keys, tokens, and passwords in source.
  - The security audit checks for SQL/command injection: string concatenation in queries or shell commands.
  - The security audit checks for SSRF: HTTP requests to user-controlled URLs.
  - The security audit checks for weak cryptography: MD5/SHA1 for security purposes and hardcoded IVs.
  - Quick depth takes ~30s and provides a surface scan covering eval/exec and obvious secrets.
  - Standard depth takes ~2 min and provides full pattern matching with severity ratings.
  - Deep depth takes ~5 min and provides multi-pass review with OWASP mapping and fix suggestions.

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

This runs directly in your Claude Code conversation and produces structured results [P3].

---

### Option 2: Use the `attune` workflow

```
attune workflow run test-gen --path src/your/module.py
```

This generates a test file with edge cases and assertions [P1]. After running it, you can verify the results with `pytest` [P1].

---

### What Gets Generated

The tool finds and covers several types of gaps [P2]:

| Output | What it covers |
|--------|---------------|
| **Coverage gap report** | Ranked list of untested functions with risk scores |
| **Generated tests** | Working pytest functions with assertions |
| **Edge cases** | Boundary values, empty inputs, None handling |
| **Error path tests** | Tests for expected exceptions and failures |
| **Parametrized tests** | `@pytest.mark.parametrize` for input combos |

[P2]

---

### How It Works

If you provide a path upfront (e.g. `generate tests for src/auth/`), it runs immediately. Otherwise, you'll be guided through a couple of quick questions to narrow down the target module and whether you want gap analysis, test generation, or both [P2].

It runs on your **Claude subscription — no API key or additional cost** [P2].


### Judge verdicts (off → on)

| | Score | Supported | Unsupported |
| - | -: | -: | -: |
| Thinking off | 1.000 | 13 | 0 |
| Thinking on  | 1.000 | 18 | 0 |

**Reasoning (thinking off):**

> Every factual claim in the answer maps directly to content in one of the retrieved passages. The two usage methods are drawn verbatim from P3 and P1, respectively. The table of outputs is taken directly from P2's "What it produces" table. The behavior around upfront path provision vs. guided questions is stated explicitly in P2's "What to expect" section, including the exact example "generate tests for src/auth/". The cost/subscription claim is also explicitly stated in P2. No hallucinated CLI flags, invented workflow names, or details beyond what the passages contain are present.

**Reasoning (thinking on):**

> Every atomic factual claim in the answer maps directly to explicit statements in the retrieved passages. The two usage commands (P1, P3), the interactive questioning flow and skip condition (P2), and all rows of the outputs and gap-type tables (P2) are reproduced faithfully without embellishment, invented flags, or outside-knowledge drift. No hallucinations were detected.

**Claims identified by judge (thinking off):**

Supported:
  - You can use the `/smart-test` skill in Claude Code by running `/smart-test <path or module to test>`
  - The `/smart-test` skill runs directly in your Claude Code conversation and produces structured results
  - You can use the `attune` workflow by running `attune workflow run test-gen --path src/your/module.py`
  - The attune workflow generates a test file with edge cases and assertions
  - After running the attune workflow, you can verify the results with `pytest`
  - The tool produces a coverage gap report: a ranked list of untested functions with risk scores
  - The tool produces generated tests: working pytest functions with assertions
  - The tool produces edge cases: boundary values, empty inputs, None handling
  - The tool produces error path tests: tests for expected exceptions and failures
  - The tool produces parametrized tests: `@pytest.mark.parametrize` for input combos
  - If you provide a path upfront (e.g. `generate tests for src/auth/`), it runs immediately
  - Otherwise, you'll be guided through a couple of quick questions to narrow down the target module and whether you want gap analysis, test generation, or both
  - It runs on your Claude subscription — no API key or additional cost

Unsupported:


**Claims identified by judge (thinking on):**

Supported:
  - The command for the smart-test skill is `/smart-test <path or module to test>`
  - The smart-test skill delivers structured results directly in your Claude Code conversation
  - The attune workflow command is `attune workflow run test-gen --path src/attune/help/engine.py`
  - The attune workflow generates a test file with edge cases and assertions
  - After running the attune workflow, you can verify results with `pytest`
  - You may be guided through a couple of quick questions first
  - The questions cover which module to target and whether you want gap analysis, test generation, or both
  - If you provide both details upfront (e.g. 'generate tests for src/auth/'), the questions are skipped and it runs immediately
  - The coverage gap report is a ranked list of untested functions with risk scores
  - Generated tests are working pytest functions with assertions
  - Edge cases cover boundary values, empty inputs, and None handling
  - Error path tests cover expected exceptions and failures
  - Parametrized tests use `@pytest.mark.parametrize` for input combos
  - Smart-test targets untested public functions
  - Smart-test targets missing if/else branches
  - Smart-test targets exception handlers never triggered
  - Smart-test targets boundary values
  - Smart-test targets input combinations never paired

Unsupported:


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

## gq-004 — `review code quality`

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
> [P3] source: quickstarts/skill-code-quality.md
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

### Answer

## Code Quality Review

Here are the ways you can run a code quality review:

---

### Option 1: Claude Code Skill

```
/code-quality <path or directory to review>
```

This gives you structured results directly in your Claude Code conversation [P3].

---

### Option 2: CLI Workflow

```
attune workflow run code-review --path "src/"
```

This returns quality findings grouped by severity [P2].

---

### What It Covers

A code quality review examines your code from multiple angles — style, correctness, likely bugs, and structural health — delivering a unified score in a single pass [P1].

| Category | Examples | Severity |
|----------|---------|----------|
| **Style** | Unused imports, line length, naming | Low |
| **Correctness** | Unreachable code, wrong return types | High |
| **Likely bugs** | Broad exceptions, mutable defaults, race conditions | High |
| **Structural** | High coupling, god classes, circular deps | Medium |
| **Security overlap** | eval/exec, unvalidated paths | Critical |

[P1]

---

### Depth Options

| Depth | Covers | Best For |
|-------|--------|----------|
| **Quick** | Style and formatting only | Pre-commit checks |
| **Standard** | + Logic errors, likely bugs | Pull requests *(default)* |
| **Deep** | + Security, architecture, test gaps | Pre-release, critical modules |

[P1]

---

### What to Expect

When you request a review, you'll be asked a couple of scoping questions — which files to review and how deep to go — to keep results focused [P1].

**Next step:** Follow up with `/smart-test` to generate tests for any flagged areas [P2].


### Judge verdicts (off → on)

| | Score | Supported | Unsupported |
| - | -: | -: | -: |
| Thinking off | 1.000 | 21 | 0 |
| Thinking on  | 1.000 | 19 | 0 |

**Reasoning (thinking off):**

> Every atomic factual claim in the answer is directly supported by at least one of the retrieved passages. The two invocation methods (skill and CLI) are taken verbatim from P3 and P2 respectively. The category table, depth table, scoping-questions behavior, and the /smart-test follow-up tip all map precisely to text in P1 and P2. The only minor editorial addition is labeling "Standard" as the default in the depth table, but P1 explicitly states "The default is standard," so that is also supported. No hallucinated CLI flags, invented workflow names, or outside-knowledge claims are present.

**Reasoning (thinking on):**

> Every atomic claim in the answer is directly and explicitly supported by at least one of the three retrieved passages. The Claude Code skill details (command syntax, capabilities, output format) come verbatim from P3. The CLI command and severity-grouped output come verbatim from P2. All table rows for categories, severities, depth levels, best-use cases, the default depth, the scoping-question behaviour, and the `/smart-test` follow-up tip are all explicitly stated in P1 and P2 respectively. No outside knowledge, invented details, or unsupported inferences were introduced.

**Claims identified by judge (thinking off):**

Supported:
  - You can run a code quality review using /code-quality <path or directory to review>
  - The /code-quality skill gives structured results in your Claude Code conversation
  - You can run a code quality review using the CLI command: attune workflow run code-review --path "src/"
  - The CLI workflow returns quality findings grouped by severity
  - A code quality review examines code from multiple angles — style, correctness, likely bugs, and structural health
  - A code quality review delivers a unified score in a single pass
  - Style issues include unused imports, line length, and naming, with Low severity
  - Correctness issues include unreachable code and wrong return types, with High severity
  - Likely bugs include broad exceptions, mutable defaults, and race conditions, with High severity
  - Structural issues include high coupling, god classes, and circular deps, with Medium severity
  - Security overlap issues include eval/exec and unvalidated paths, with Critical severity
  - The Quick depth covers style and formatting only
  - Quick depth is best for pre-commit checks
  - The Standard depth covers logic errors and likely bugs (in addition to style)
  - Standard depth is best for pull requests
  - Standard is the default depth
  - The Deep depth covers security, architecture, and test gaps (in addition to standard)
  - Deep depth is best for pre-release and critical modules
  - When you request a review, you'll be asked scoping questions about which files to review and how deep to go
  - The scoping questions keep results focused
  - You can follow up with /smart-test to generate tests for flagged areas

Unsupported:


**Claims identified by judge (thinking on):**

Supported:
  - The Claude Code skill command is `/code-quality <path or directory to review>`
  - The Claude Code skill provides code review and bug prediction
  - The Claude Code skill finds quality issues, style violations, and likely bugs
  - The Claude Code skill gives structured results in your Claude Code conversation
  - The CLI command is `attune workflow run code-review --path "src/"`
  - The CLI workflow gives AI-powered quality findings grouped by severity
  - A code quality review looks at code from multiple angles at once — style, correctness, likely bugs, and structural health
  - Style category examples include unused imports, line length, and naming, with Low severity
  - Correctness category examples include unreachable code and wrong return types, with High severity
  - Likely bugs category examples include broad exceptions, mutable defaults, and race conditions, with High severity
  - Structural category examples include high coupling, god classes, and circular deps, with Medium severity
  - Security overlap category examples include eval/exec and unvalidated paths (deep only), with Critical severity
  - Quick depth covers style and formatting only, and is best for pre-commit checks
  - Standard depth covers logic errors and likely bugs in addition to style, and is best for pull requests
  - Standard depth is the default
  - Deep depth covers security, architecture, and test gaps, and is best for pre-release and critical modules
  - When asking for a code quality review, you'll be asked a couple of questions first to scope the work
  - The scoping questions cover which files to review and how deep to go
  - After the review, you can follow up with `/smart-test` to generate tests for flagged areas

Unsupported:


### Your labels

```yaml
id: gq-004
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

## gq-006 — `predict bugs before they happen`

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
> [P2] source: quickstarts/skill-bug-predict.md
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
>
> <passage>
> [P3] source: references/tool-bug-predict.md
> ---
> type: reference
> subtype: tabular
> name: tool-bug-predict
> category: tool
> tags: [mcp, tool, workflow]
> source: src/attune/mcp/tool_schemas.py
> ---
>
> # Reference: Tool: Bug Predict
>
> Run bug prediction workflow. Analyzes code patterns and predicts potential bugs before they occur.
>
> **Group:** workflow
>
> ## Parameters
>
> | Parameter | Type | Description | Constraints | Default |
> | --------- | ---- | ----------- | ----------- | ------- |
> | `path` | string | Path to directory or file to analyze |  | required |
>
> ## Usage
>
> `bug_predict(path="...")`
>
> ## Related Topics
> - **Reference**: Tool: Security Audit — Run security audit workflow on codebase. Detects vulnerabili...
> - **Reference**: Tool: Code Review — Run code review workflow. Provides comprehensive code qualit...
> - **Reference**: Tool: Test Generation — Generate tests for code. Can batch generate tests for multip...
> </passage>

### Answer

## Predict Bugs Before They Happen

You can use the **Bug Prediction** tool to scan your codebase and identify likely bug locations before they cause production incidents. [P1]

---

### How to Run It

```
/bug-predict <path or directory to scan>
```

This produces structured results directly in your Claude Code conversation. [P2]

You can also call it programmatically:

```
bug_predict(path="...")
```

The `path` parameter (pointing to a directory or file) is required. [P3]

---

### What It Detects

The scanner looks for three categories of risk [P1]:

| Pattern | Severity | What It Finds |
|---|---|---|
| **dangerous_eval** | HIGH | `eval()`, `exec()`, `compile()` — code injection vectors |
| **broad_exception** | MEDIUM | Bare `except:` or unlogged exceptions — silently swallowed errors |
| **incomplete_code** | LOW | TODO, FIXME, HACK, XXX comments — unfinished code paths |

---

### Beyond Pattern Matching

The scanner also weighs contextual risk signals [P1]:

- **Cyclomatic complexity** — deeply nested conditionals and long function bodies
- **Change frequency** — frequently modified ("hot") files are more prone to regressions
- **Code smells** — functions over 50 lines, oversized classes, duplicated logic

---

### Smart False-Positive Filtering

Not every match is a real bug. The scanner automatically suppresses known-safe patterns such as [P1]:

- `eval()` inside test fixture strings
- JavaScript `regex.exec()` calls
- Broad exceptions marked with `# INTENTIONAL:` or `# noqa: BLE001`

---

### When to Use It

- Before merging a large PR
- During code review to focus attention on real risks
- After onboarding into unfamiliar code
- As a periodic health check on high-churn modules
- Before a release to verify no new high-severity patterns crept in [P1]


### Judge verdicts (off → on)

| | Score | Supported | Unsupported |
| - | -: | -: | -: |
| Thinking off | 1.000 | 31 | 0 |
| Thinking on  | 1.000 | 19 | 0 |

**Reasoning (thinking off):**

> Every atomic factual claim in the answer is directly and explicitly supported by the retrieved passages. P1 covers the tool's purpose, detected patterns, risk factors, false-positive filtering, and use cases. P2 confirms the `/bug-predict` command syntax and the Claude Code conversation output. P3 confirms the programmatic `bug_predict(path="...")` usage and the required `path` parameter. No details appear to have been invented or imported from outside knowledge.

**Reasoning (thinking on):**

> Every atomic factual claim in the answer is directly and explicitly supported by at least one retrieved passage. The tool description, command syntax, API call shape, risk pattern table (patterns, severities, descriptions), contextual risk signals, false-positive suppression rules, and recommended use cases are all verbatim or near-verbatim from P1, P2, and P3. The answer makes no additions, invented flags, or inferences beyond what the passages state. One minor omission (the "periodic health check on high-churn modules" use case and "version detection fallbacks" false-positive suppression) is simply not included in the answer rather than fabricated, so it raises no faithfulness concern.

**Claims identified by judge (thinking off):**

Supported:
  - You can use the Bug Prediction tool to scan your codebase and identify likely bug locations before they cause production incidents.
  - The command to run bug prediction is `/bug-predict <path or directory to scan>`.
  - The command produces structured results directly in your Claude Code conversation.
  - You can also call it programmatically using `bug_predict(path="...")`.
  - The `path` parameter is required.
  - The `path` parameter points to a directory or file.
  - The scanner looks for three categories of risk.
  - The `dangerous_eval` pattern has HIGH severity.
  - The `dangerous_eval` pattern finds `eval()`, `exec()`, `compile()` — code injection vectors.
  - The `broad_exception` pattern has MEDIUM severity.
  - The `broad_exception` pattern finds bare `except:` or unlogged exceptions — silently swallowed errors.
  - The `incomplete_code` pattern has LOW severity.
  - The `incomplete_code` pattern finds TODO, FIXME, HACK, XXX comments — unfinished code paths.
  - The scanner weighs cyclomatic complexity as a contextual risk signal.
  - Deeply nested conditionals and long function bodies correlate with higher defect rates.
  - The scanner weighs change frequency as a contextual risk signal.
  - Frequently modified ('hot') files are more prone to regressions.
  - The scanner weighs code smells as a contextual risk signal.
  - Code smells include functions over 50 lines.
  - Code smells include oversized classes (classes with too many methods).
  - Code smells include duplicated logic across modules.
  - The scanner automatically suppresses known-safe patterns.
  - `eval()` inside test fixture strings is suppressed as a known-safe pattern.
  - JavaScript `regex.exec()` calls are suppressed as a known-safe pattern.
  - Broad exceptions marked with `# INTENTIONAL:` are suppressed.
  - Broad exceptions marked with `# noqa: BLE001` are suppressed.
  - The tool should be used before merging a large PR.
  - The tool should be used during code review to focus attention on real risks.
  - The tool should be used after onboarding into unfamiliar code.
  - The tool should be used as a periodic health check on high-churn modules.
  - The tool should be used before a release to verify no new high-severity patterns crept in.

Unsupported:


**Claims identified by judge (thinking on):**

Supported:
  - The Bug Prediction tool can scan your codebase and identify likely bug locations before they cause production incidents.
  - The command to run bug prediction is `/bug-predict <path or directory to scan>`.
  - Results appear as structured output in your Claude Code conversation.
  - Bug prediction can also be called directly as `bug_predict(path="...")`.
  - The scanner looks for three categories of risk.
  - The dangerous_eval pattern has HIGH severity and catches `eval()`, `exec()`, `compile()` on any input — code injection vectors.
  - The broad_exception pattern has MEDIUM severity and catches bare `except:` and unlogged `except Exception:` — silently swallowed errors.
  - The incomplete_code pattern has LOW severity and catches TODO, FIXME, HACK, XXX comments — unfinished code paths.
  - The scanner weighs cyclomatic complexity — deeply nested conditionals and long function bodies — as a contextual risk signal.
  - The scanner weighs change frequency — frequently modified ('hot') files are more likely to contain regressions.
  - The scanner weighs code smells — functions over 50 lines, classes with too many methods, duplicated logic.
  - The scanner automatically suppresses known-safe patterns.
  - `eval()` inside test fixture strings is suppressed as a known-safe pattern.
  - JavaScript `regex.exec()` method calls are suppressed as a known-safe pattern.
  - Broad exceptions marked with `# INTENTIONAL:` or `# noqa: BLE001` are suppressed.
  - Bug prediction should be used before merging a large PR.
  - Bug prediction should be used during code review to focus attention on real risks.
  - Bug prediction should be used after onboarding unfamiliar code to map risk hotspots.
  - Bug prediction should be used before a release to verify no new high-severity patterns crept in.

Unsupported:


### Your labels

```yaml
id: gq-006
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

## gq-012 — `write documentation for my module`

### Retrieved context

> <passage>
> [P1] source: concepts/tool-doc-gen.md
> ---
> type: concept
> name: tool-doc-gen
> tags: [documentation, docstrings, readme]
> source: plugin/skills/doc-gen/SKILL.md
> ---
>
> # Doc Generation
>
> Doc-gen reads your actual source code -- function signatures,
> type hints, class hierarchies, module structure -- and generates
> documentation from it. Instead of writing docstrings by hand
> or letting them drift from the code, you get accurate docs that
> reflect the current API.
>
> ## What it generates
>
> | Doc type | What you get | Source |
> |----------|-------------|--------|
> | **Docstrings** | Google-style with Args, Returns, Raises, Examples | Function signatures and type hints |
> | **README sections** | Feature lists, usage examples, install instructions | Module exports and public API |
> | **API reference** | Full function/class signatures with descriptions | All public symbols in a module |
> | **Module overview** | Architecture summary, dependency map, entry points | Package structure and imports |
>
> ## When you'd use it
>
> Run doc-gen after creating new public APIs, before a release
> to refresh stale docs, when onboarding contributors who need
> to understand a module, or when you inherit a codebase with
> missing docstrings. It reads the source of truth so the docs
> never contradict the code.
>
> ## How it works
>
> Doc-gen is a Socratic skill. It asks you two questions before
> generating anything:
>
> 1. **What to document** -- a single file, a module, or a
>    directory tree
> 2. **What format** -- docstrings, README, API reference, or
>    a full pipeline that audits gaps and generates docs for
>    all of them
>
> Then it reads the source, generates the documentation, and
> shows you the result before applying changes.
>
> ## Example output
>
> A generated Google-style docstring looks like this:
>
> ```python
> def validate_email(email: str) -> bool:
>     """Validate email format using regex.
>
>     Args:
>         email: Email address to validate.
>
>     Returns:
>         True if email matches a valid format,
>         False otherwise.
>
>     Raises:
>         TypeError: If email is not a string.
>
>     Example:
>         >>> validate_email("user@example.com")
>         True
>     """
> ```
>
> ## Want to learn more?
>
> - Say **"tell me more"** for step-by-step instructions
> - Say **"what is code quality?"** for a broader code review
> - Say **"what is smart-test?"** to generate tests alongside docs
> </passage>
>
> <passage>
> [P2] source: quickstarts/skill-doc-gen.md
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
> [P3] source: references/skill-doc-gen.md
> ---
> type: reference
> subtype: procedural
> name: skill-doc-gen
> category: skill
> tags: [documentation, skill, plugin, reference]
> source: plugin/skills/doc-gen/SKILL.md
> ---
>
> # Doc Generation Reference
>
> Complete reference for the doc-gen skill -- every doc
> type it produces, format options, and how to control
> the output.
>
> ## Invocation
>
> ```
> /doc-gen <path>
> ```
>
> Or natural language:
>
> ```
> document src/models/
> add docstrings to auth.py
> generate a README for the cli module
> create API docs for the entire project
> ```
>
> ## Doc types
>
> The skill runs on your Claude subscription -- no API key
> or additional cost. You choose what kind of documentation
> to generate:
>
> | Doc type | What it generates | Best for |
> |----------|-------------------|----------|
> | **Docstrings** | Google-style per function/class/method | Keeping code self-documenting |
> | **README** | Feature list, usage examples, install instructions | Module or project overviews |
> | **API reference** | Full signatures with types and descriptions | Library consumers |
> | **Module overview** | Architecture summary, dependency map | Onboarding and navigation |
> | **Full pipeline** | Audit gaps, generate docs, review coverage | Pre-release documentation sweep |
>
> Set the type with natural language:
>
> ```
> add docstrings to src/auth/
> generate a README for the models package
> full documentation audit on src/
> ```
>
> ## Scoping
>
> Before generating, the skill asks:
>
> 1. **Target**: "Which file or module needs documentation?"
> 2. **Format**: "What kind of docs -- docstrings, README,
>    API reference, or a full audit?"
>
> ## Google-style docstring spec
>
> All generated docstrings follow Google style with these
> sections (each included only when applicable):
>
> | Section | When included | Format |
> |---------|---------------|--------|
> | **Summary** | Always | One-line imperative sentence |
> | **Args** | Function has parameters | `name: Description.` per line |
> | **Returns** | Function returns a value | Description of return value |
> | **Yields** | Generator function | Description of yielded values |
> | **Raises** | Function raises exceptions | `ExceptionType: When it happens.` |
> | **Example** | Public API functions | `>>> call()` with expected output |
> | **Note** | Important caveats exist | Free-form paragraph |
>
> Example of a complete generated docstring:
>
> ```python
> def create_session(
>     user_id: str,
>     ttl: int = 3600,
> ) -> Session:
>     """Create an authenticated session for a user.
>
>     Args:
>         user_id: Unique identifier for the user.
>         ttl: Session lifetime in seconds. Defaults
>             to 3600 (one hour).
>
>     Returns:
>         A new Session object with a generated token.
>
>     Raises:
>         ValueError: If user_id is empty.
>         AuthError: If the user account is locked.
>
>     Example:
>         >>> session = create_session("user-42")
>         >>> session.is_valid()
>         True
>     """
> ```
>
> ## README generation
>
> When generating a README section for a module, doc-gen
> produces:
>
> | Section | Content |
> |---------|---------|
> | **Module name** | Heading with one-line description |
> | **Overview** | What the module does and why it exists |
> | **Quick start** | Minimal usage example with imports |
> | **Public API** | Table of exported functions/classes |
> | **Configuration** | Environment variables or config options |
> | **Dependencies** | Required packages and optional extras |
>
> ## Output format
>
> Results are presented as a documentation report:
>
> ```markdown
> ## Documentation Report
>
> **Files:** 12 | **Functions Documented:** 34 | **Gaps:** 3
>
> ### Generated Docstrings
>
> | File | Function | Status |
> |------|----------|--------|
> | [session.py:15](src/auth/session.py#L15) | create_session | Added |
> | [session.py:42](src/auth/session.py#L42) | validate_token | Updated |
> | [session.py:78](src/auth/session.py#L78) | revoke_session | Added |
>
> ### Gaps Remaining
>
> | File | Missing |
> |------|---------|
> | [middleware.py](src/auth/middleware.py) | 2 private helpers |
> | [utils.py](src/auth/utils.py) | 1 module docstring |
> ```
>
> ## Source analysis
>
> Doc-gen reads these elements from your code to produce
> accurate documentation:
>
> | Source | What it extracts |
> |--------|-----------------|
> | Function signatures | Parameter names, types, defaults |
> | Type hints | Return types, generic parameters |
> | Class hierarchies | Inheritance chain, abstract methods |
> | Module `__all__` | Public API surface |
> | Existing docstrings | Preserves manual docs, fills gaps |
> | Decorator metadata | `@property`, `@staticmethod`, etc. |
> | Exception handlers | Which exceptions a function raises |
>
> ## After generating
>
> | Goal | What to say |
> |------|-------------|
> | Apply docstrings to files | "apply these to the files" |
> | Audit the rest of the project | "audit the whole project for gaps" |
> | Generate a README next | "generate a README for this module" |
> | Generate API reference | "create API docs for this package" |
> | Export as markdown | "export the docs as a markdown file" |
>
> ## Want to learn more?
>
> - Say **"what is doc-gen?"** to go back to the overview
> - Say **"how do I generate docs?"** for the step-by-step
> - Say **"review my code"** for a broader code quality review
> - Say **"what is smart-test?"** to generate tests alongside
> </passage>

### Answer

## Generating Documentation for Your Module

You can use the **doc-gen skill** to generate documentation from your source code. Here's how to get started:

## Invocation

Run the skill with:

```
/doc-gen <path>
```

Or use natural language, for example:

```
document src/models/
add docstrings to auth.py
generate a README for the cli module
```
[P3]

## What It Will Ask You

Before generating anything, the skill asks you two questions [P1, P3]:

1. **Target** — which file or module needs documentation?
2. **Format** — what kind of docs do you want?

## Documentation Types You Can Choose From

| Doc type | What you get |
|----------|-------------|
| **Docstrings** | Google-style per function/class/method |
| **README** | Feature list, usage examples, install instructions |
| **API reference** | Full signatures with types and descriptions |
| **Module overview** | Architecture summary, dependency map |
| **Full pipeline** | Audit gaps, generate all docs, review coverage |

[P3]

## What It Reads From Your Code

Doc-gen extracts information from [P3]:
- Function signatures (parameter names, types, defaults)
- Type hints and return types
- Class hierarchies and inheritance
- Module `__all__` for public API surface
- Existing docstrings (preserves manual docs, fills gaps)
- Exception handlers

## After Generating

Once docs are produced, you can [P3]:

- **Apply them** — "apply these to the files"
- **Audit the project** — "audit the whole project for gaps"
- **Generate a README** — "generate a README for this module"
- **Export as markdown** — "export the docs as a markdown file"

The skill runs on your Claude subscription with no additional API key or cost. [P3]


### Judge verdicts (off → on)

| | Score | Supported | Unsupported |
| - | -: | -: | -: |
| Thinking off | 1.000 | 24 | 0 |
| Thinking on  | 1.000 | 31 | 0 |

**Reasoning (thinking off):**

> Every factual claim in the answer maps directly to content in the retrieved passages, primarily P3 (the full reference) and P1 (the concept doc). The invocation syntax, natural language examples, two-question flow, documentation types and their descriptions, source analysis elements, after-generating actions, and the no-additional-cost note are all explicitly stated in the passages. No outside knowledge or invented details are present.

**Reasoning (thinking on):**

> Every atomic factual claim in the answer is directly and explicitly supported by at least one of the retrieved passages. Invocation syntax and natural language examples come from P2 and P3. The two pre-generation questions are stated verbatim in P1 (and echoed in P3). The doc-type table with "Best for" descriptions is taken directly from P3. The Google-style docstring details, README section contents, source analysis elements, and post-generation follow-up commands all map precisely to P3. No information was introduced from outside knowledge, inferred beyond what the passages say, or invented.

**Claims identified by judge (thinking off):**

Supported:
  - The doc-gen skill generates documentation from source code.
  - The skill can be invoked with /doc-gen <path>.
  - Natural language invocations include 'document src/models/'.
  - Natural language invocations include 'add docstrings to auth.py'.
  - Natural language invocations include 'generate a README for the cli module'.
  - Before generating anything, the skill asks two questions.
  - The first question is about the target — which file or module needs documentation.
  - The second question is about the format — what kind of docs you want.
  - Docstrings are Google-style per function/class/method.
  - README includes feature list, usage examples, and install instructions.
  - API reference includes full signatures with types and descriptions.
  - Module overview includes architecture summary and dependency map.
  - Full pipeline audits gaps, generates all docs, and reviews coverage.
  - Doc-gen reads function signatures including parameter names, types, and defaults.
  - Doc-gen reads type hints and return types.
  - Doc-gen reads class hierarchies and inheritance.
  - Doc-gen reads module __all__ for public API surface.
  - Doc-gen preserves existing manual docstrings and fills gaps.
  - Doc-gen reads exception handlers to determine which exceptions a function raises.
  - After generating, you can apply docs by saying 'apply these to the files'.
  - After generating, you can audit the project by saying 'audit the whole project for gaps'.
  - After generating, you can generate a README by saying 'generate a README for this module'.
  - After generating, you can export as markdown by saying 'export the docs as a markdown file'.
  - The skill runs on your Claude subscription with no additional API key or cost.

Unsupported:


**Claims identified by judge (thinking on):**

Supported:
  - The skill is invoked with `/doc-gen <path or module to document>`
  - 'document src/models/' is a valid natural language invocation
  - 'add docstrings to auth.py' is a valid natural language invocation
  - 'generate a README for the cli module' is a valid natural language invocation
  - Before generating anything, the skill asks two questions
  - The first question is about what to document — a single file, a module, or a directory tree
  - The second question is about what format — docstrings, README, API reference, or a full pipeline that audits gaps and generates docs for all of them
  - Docstrings doc type is best for keeping code self-documenting
  - README doc type is best for module or project overviews
  - API reference doc type is best for library consumers
  - Module overview doc type is best for onboarding and navigation
  - Full pipeline doc type is best for a pre-release documentation sweep
  - Generated docstrings follow Google style
  - Docstrings have sections for Args, Returns, Raises, and Examples
  - Each docstring section is included only when applicable
  - Generated README sections include an overview
  - Generated README sections include a quick start
  - Generated README sections include a public API table
  - Generated README sections include configuration
  - Generated README sections include dependencies
  - The skill reads function signatures from source code
  - The skill reads type hints from source code
  - The skill reads class hierarchies from source code
  - The skill reads module `__all__` from source code
  - The skill reads existing docstrings from source code
  - The skill reads decorator metadata from source code
  - The skill reads exception handlers from source code
  - Saying 'apply these to the files' will write the docstrings to the files
  - Saying 'audit the whole project for gaps' will check coverage
  - Saying 'generate a README for this module' will switch to README generation
  - Saying 'export the docs as a markdown file' will save the output as markdown

Unsupported:


### Your labels

```yaml
id: gq-012
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

## Scanning for Dangerous `eval` Calls

I can run bug prediction to look for dangerous eval patterns in your codebase. Here's what you need to know:

### What the Scanner Detects

The **`dangerous_eval`** pattern flags uses of `eval()`, `exec()`, and `compile()` on any input, as these are code injection vectors. This is classified as **HIGH severity** with **High confidence** [P1].

### How to Run It

Use the bug prediction tool:

```
bug_predict(path="...")
```

or for a broader security scan:

```
security_audit(path="...")
```

Provide the path to the directory or file you want to analyze [P3].

### ⚠️ Important: Verify HIGH Severity Findings

Before treating any `dangerous_eval` finding as a real vulnerability, **always verify it against the source code**. A known false positive exists where the scanner's regex matches `create_subprocess_exec` as containing `exec`, flagging it as `dangerous_eval` in files like `hooks/executor.py` — even when no actual `eval()` or `exec()` is present [P2].

### Known Safe Patterns (Auto-Suppressed)

The scanner automatically filters out:
- `eval()` inside **test fixture strings** (test data, not executable code)
- JavaScript **`regex.exec()`** method calls (safe, not Python's `exec()`) [P1]

---

**What path would you like me to scan?**


### Judge verdicts (off → on)

| | Score | Supported | Unsupported |
| - | -: | -: | -: |
| Thinking off | 0.923 | 12 | 1 |
| Thinking on  | 0.929 | 13 | 1 |

**Reasoning (thinking off):**

> Most claims in the answer are well-supported by the retrieved passages. P1 covers the dangerous_eval pattern details (severity, confidence, what it flags, and the auto-suppressed safe patterns). P2 covers the false-positive case with `create_subprocess_exec` in `hooks/executor.py` and the advice to verify findings. P3 covers the `security_audit` tool's parameters and usage syntax. The one unsupported claim is the `bug_predict(path="...")` usage syntax — P3 only shows the reference entry for the bug predict tool in a related topics blurb without specifying its call signature, and no passage explicitly states this syntax. The answer infers it by analogy with `security_audit`, but no passage confirms it.

**Reasoning (thinking on):**

> Almost all claims in the answer are well-supported by the passages. P1 covers the dangerous_eval pattern details (severity, confidence, what it detects, and auto-suppressed safe patterns). P2 covers the false-positive scenario involving `create_subprocess_exec` and the verification guidance. P3 covers the `security_audit(path="...")` usage. The one unsupported claim is the suggested `bug_predict(path="...")` call syntax: while P3 mentions a Bug Predict tool exists in its Related Topics section, no passage ever shows its invocation syntax. The answer appears to have inferred it by analogy from `security_audit(path="...")`, which is a reasonable inference but not directly stated in any passage.

**Claims identified by judge (thinking off):**

Supported:
  - The `dangerous_eval` pattern flags uses of `eval()`, `exec()`, and `compile()` on any input
  - These are classified as code injection vectors
  - `dangerous_eval` is classified as HIGH severity
  - `dangerous_eval` has High confidence
  - The security_audit tool accepts a `path` parameter for the directory or file to audit
  - The usage syntax for security_audit is `security_audit(path="...")`
  - Before treating any `dangerous_eval` finding as a real vulnerability, always verify it against the source code
  - A known false positive exists where the scanner's regex matches `create_subprocess_exec` as containing `exec`
  - This false positive flags it as `dangerous_eval` in files like `hooks/executor.py`
  - The false positive occurs even when no actual `eval()` or `exec()` is present
  - The scanner automatically suppresses `eval()` inside test fixture strings (test data, not executable code)
  - The scanner automatically suppresses JavaScript `regex.exec()` method calls (safe, not Python's `exec()`)

Unsupported:
  - The bug prediction tool usage syntax is `bug_predict(path="...")`

**Claims identified by judge (thinking on):**

Supported:
  - The `dangerous_eval` pattern has HIGH severity.
  - The `dangerous_eval` pattern has high confidence.
  - The scanner looks for uses of `eval()`, `exec()`, and `compile()` on any input.
  - `eval()`, `exec()`, and `compile()` on any input are code injection vectors.
  - The security audit tool detects dangerous patterns.
  - The security audit tool usage syntax is `security_audit(path="...")`.
  - The scanner's regex matches `create_subprocess_exec` as containing `exec`.
  - This produces a false positive for `dangerous_eval`.
  - There is no actual `eval()` or `exec()` usage in the false-positive case.
  - The false positive has been verified/confirmed by a prior incident.
  - You should always verify HIGH severity scanner findings against the source before treating them as real vulnerabilities.
  - `eval()` inside test fixture strings is automatically suppressed by the scanner.
  - JavaScript `regex.exec()` method calls are automatically suppressed by the scanner (they are not Python's `exec()`).

Unsupported:
  - The bug prediction tool usage syntax is `bug_predict(path="...")`.

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

Here are the ways you can run a code quality check:

### Option 1: Using the Skill in Claude Code

```
/code-quality <path or directory to review>
```

This gives you structured results directly in your Claude Code conversation [P2].

### Option 2: Using the CLI Workflow

```
attune workflow run code-review --path "src/"
```

This returns quality findings grouped by severity [P3].

---

### What Gets Analyzed

A code quality review examines your code from multiple angles at once — style, correctness, likely bugs, and structural health [P1]. Here's a breakdown of what it catches:

| Category | Examples | Severity |
|----------|---------|----------|
| **Style** | Unused imports, line length, naming | Low |
| **Correctness** | Unreachable code, wrong return types | High |
| **Likely bugs** | Broad exceptions, mutable defaults, race conditions | High |
| **Structural** | High coupling, god classes, circular deps | Medium |
| **Security overlap** | eval/exec, unvalidated paths (deep only) | Critical |

[P1]

### Choosing a Depth

| Depth | Best For |
|-------|----------|
| **Quick** | Pre-commit check, fast feedback |
| **Standard** | Pull requests, regular development |
| **Deep** | Pre-release, critical modules |

The default depth is **standard** [P1].

> **Tip:** When you ask for a code quality review, you'll be asked a couple of questions first to scope the work — which files to review and how deep to go [P1]. After the review, consider following up with `/smart-test` to generate tests for flagged areas [P3].


### Judge verdicts (off → on)

| | Score | Supported | Unsupported |
| - | -: | -: | -: |
| Thinking off | 1.000 | 16 | 0 |
| Thinking on  | 1.000 | 17 | 0 |

**Reasoning (thinking off):**

> Every factual claim in the answer is directly and explicitly supported by the retrieved passages. The `/code-quality` command and its description come from P2. The `attune workflow run code-review` CLI command, its output description, and the `/smart-test` follow-up tip come from P3. All category/severity rows in the table, the depth tiers and their best-use descriptions, the default depth being "standard," and the scoping-questions note all come verbatim or near-verbatim from P1. There are no hallucinated details, invented flags, or claims drawn from outside knowledge.

**Reasoning (thinking on):**

> Every atomic factual claim in the answer is directly and explicitly supported by the retrieved passages. The two command syntaxes, severity table, depth table, and use-case list are all verbatim or near-verbatim reproductions of content from P1, P2, and P3 respectively. No information was added from outside knowledge, and no CLI flags, category names, or workflow identifiers were invented. The answer is fully faithful to the retrieved passages.

**Claims identified by judge (thinking off):**

Supported:
  - You can run a code quality check using `/code-quality <path or directory to review>`
  - The `/code-quality` command gives structured results directly in your Claude Code conversation
  - You can run a code quality check using `attune workflow run code-review --path "src/"`
  - The CLI workflow returns quality findings grouped by severity
  - A code quality review examines your code from multiple angles at once — style, correctness, likely bugs, and structural health
  - Style issues include unused imports, line length, and naming, with Low severity
  - Correctness issues include unreachable code and wrong return types, with High severity
  - Likely bugs include broad exceptions, mutable defaults, and race conditions, with High severity
  - Structural issues include high coupling, god classes, and circular deps, with Medium severity
  - Security overlap issues include eval/exec and unvalidated paths (deep only), with Critical severity
  - The Quick depth is best for pre-commit checks and fast feedback
  - The Standard depth is best for pull requests and regular development
  - The Deep depth is best for pre-release and critical modules
  - The default depth is standard
  - When you ask for a code quality review, you'll be asked a couple of questions first to scope the work — which files to review and how deep to go
  - After the review, consider following up with `/smart-test` to generate tests for flagged areas

Unsupported:


**Claims identified by judge (thinking on):**

Supported:
  - The command syntax for the code-quality skill is `/code-quality <path or directory to review>`
  - The /code-quality skill delivers structured results in your Claude Code conversation
  - The CLI command to run a code review is `attune workflow run code-review --path "src/"`
  - The CLI workflow returns quality findings grouped by severity
  - You can follow up with `/smart-test` to generate tests for flagged areas
  - Style issues (unused imports, naming, line length) have Low severity
  - Correctness issues (unreachable code, wrong return types) have High severity
  - Likely bugs (broad exceptions, mutable defaults, race conditions) have High severity
  - Structural issues (high coupling, god classes, circular deps) have Medium severity
  - Security overlap issues (eval/exec, unvalidated paths — deep only) have Critical severity
  - Quick depth is best for pre-commit checks and fast feedback
  - Standard is the default depth
  - Standard depth is best for pull requests and regular development
  - Deep depth is best for pre-release and critical modules
  - Code quality review is useful before opening a pull request
  - Code quality review is useful after a large refactor
  - Code quality review is useful when inheriting unfamiliar code

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

Here's how to run release prep before publishing your project.

### Quickest Way to Start

Just say what you want in natural language:

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

Release prep runs a preflight checklist covering these areas [P1]:

| Check Area | What It Verifies | Blocking? |
|---|---|---|
| **Health** | Tests pass, lint clean, coverage above threshold | Yes if tests fail |
| **Security** | No new CVEs, no eval/exec, secrets scan clean | Yes if critical findings |
| **Changelog** | Entry exists for this version, date is current | Yes if missing |
| **Dependencies** | Pinned versions, no known vulnerabilities | Yes if vulnerable dep |
| **Version** | Semver bump matches changes, dist builds cleanly | Yes if version mismatch |

---

### Common Commands

| Goal | What to Type |
|---|---|
| Full preflight for a specific version | `/release-prep 5.6.0` |
| Check readiness without a version bump | `/release-prep check` |
| Just the security gate | `run the security check for release` |
| Just changelog validation | `check if the changelog is ready for release` |

[P3]

---

### Reading the Result

You'll receive either a **GO** or **NO-GO** verdict [P1]:

- **GO** — every check passed or has only advisory warnings. Safe to tag, build, and publish.
- **NO-GO** — one or more blocking issues found. The report lists exactly what to fix before retrying.

If there are blockers, you can say **"fix the blockers"** and the skill will attempt to resolve each issue, or **"re-run the check"** after making your own fixes. Once the verdict is GO, say **"tag and publish"** to proceed [P3].


### Judge verdicts (off → on)

| | Score | Supported | Unsupported |
| - | -: | -: | -: |
| Thinking off | 1.000 | 18 | 0 |
| Thinking on  | 1.000 | 27 | 0 |

**Reasoning (thinking off):**

> Every factual claim in the answer maps directly to content in the retrieved passages. The natural language invocation, the '/release-prep check' command, the two-minute timeline, all five check areas with their blocking conditions, the command table, the GO/NO-GO definitions, and the post-assessment actions ('fix the blockers', 're-run the check', 'tag and publish') are all explicitly stated in P1, P2, or P3. No outside knowledge or invented details were introduced.

**Reasoning (thinking on):**

> Every atomic factual claim in the answer maps directly to explicit text in the retrieved passages. The guided two-question flow (P3), the go/no-go verdict descriptions (P1), the blocker-handling responses (P3), and the "when to use" scenarios (P1) are all verbatim or near-verbatim reproductions of passage content. No outside knowledge, invented flags, or inferred details were introduced. The answer is fully faithful to the passages.

**Claims identified by judge (thinking off):**

Supported:
  - The fastest way to start is to say 'is this project ready to release?' in natural language.
  - You can invoke the skill directly with '/release-prep check'.
  - Running '/release-prep check' kicks off the full preflight checklist.
  - You get a go/no-go assessment in about two minutes.
  - Release prep checks Health: tests pass, lint clean, coverage above threshold, blocking if tests fail.
  - Release prep checks Security: no new CVEs, no eval/exec, secrets scan clean, blocking if critical findings.
  - Release prep checks Changelog: entry exists for this version, date is current, blocking if missing.
  - Release prep checks Dependencies: pinned versions, no known vulnerabilities, blocking if vulnerable dep.
  - Release prep checks Version: semver bump matches changes, dist builds cleanly, blocking if version mismatch.
  - The command '/release-prep 5.6.0' runs a full preflight for a specific version.
  - The command '/release-prep check' checks readiness without a version bump.
  - 'run the security check for release' runs just the security gate.
  - 'check if the changelog is ready for release' runs just changelog validation.
  - A GO verdict means every check passed or has only advisory warnings, and it is safe to tag, build, and publish.
  - A NO-GO verdict means one or more blocking issues were found and the report lists exactly what to fix before retrying.
  - Saying 'fix the blockers' causes the skill to attempt to resolve each issue.
  - Saying 're-run the check' runs the assessment again after making fixes.
  - Once the verdict is GO, saying 'tag and publish' proceeds to tag and upload.

Unsupported:


**Claims identified by judge (thinking on):**

Supported:
  - Release prep is a skill used to create a release.
  - Release prep runs a preflight checklist before you publish.
  - Release prep ensures the project is safe to ship.
  - The fastest way to get started is to say: 'is this project ready to release?'
  - You can invoke the skill directly with a target version using /release-prep 5.6.0.
  - When you invoke release prep, it asks two questions before running.
  - The first question is: What version are you releasing?
  - You can provide your target version or ask it to suggest the next bump.
  - The second question is: What stage?
  - The 'Full' stage covers health, security, changelog, deps, and version.
  - The 'Prep check' stage verifies readiness without a version bump.
  - The 'Changelog only' stage validates the changelog entry.
  - The 'Security only' stage runs the security gate.
  - Results come back as a structured go/no-go verdict.
  - GO means every check passed or has only advisory warnings.
  - GO means it is safe to tag, build, and publish.
  - NO-GO means one or more blocking issues were found.
  - The report lists exactly what to fix before retrying.
  - The checks include health, security, changelog, dependencies, and version consistency.
  - Saying 'Fix the blockers' causes the skill to attempt to resolve each issue.
  - Saying 'Update the changelog' generates a changelog entry from recent commits.
  - Saying 'Re-run the check' runs the assessment again after fixes.
  - Saying 'Tag and publish' proceeds to tag and upload if the verdict is GO.
  - Release prep is useful before bumping the version in pyproject.toml.
  - Release prep is useful before running twine upload or publishing to PyPI.
  - Release prep is useful after merging a large feature branch.
  - Release prep is useful when you're unsure if the codebase is release-ready.

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

### Generate New Tests (Smart Test)

If you want to **create tests for uncovered code**, Smart Test finds untested code and generates pytest tests to cover it [P3]. It produces:

- A **coverage gap report** — a ranked list of untested functions with risk scores [P3]
- **Generated tests** — working pytest functions with assertions [P3]
- **Edge cases** — boundary values, empty inputs, and None handling [P3]
- **Error path tests** — tests for expected exceptions and failures [P3]
- **Parametrized tests** — `@pytest.mark.parametrize` for input combinations [P3]

You can also run it directly via the CLI:

```
attune workflow run test-gen --path src/attune/help/engine.py
```
[P2]

If you provide details upfront (e.g., "generate tests for src/auth/"), the setup questions are skipped and it runs immediately [P3].

---

### Fix Failing Tests (Fix Test)

If you already have tests but they are **broken or failing**, Fix Test auto-diagnoses failing tests by classifying the root cause and applying a targeted fix [P1]. It handles root causes such as import errors, mock mismatches, assertion drift, type errors, and missing fixtures [P1].

It retries up to **3 times** and ends with either a passing test or a clear explanation of what still needs manual attention [P1].

---

**Which do you need?** Let me know the module path or failing test name and we can get started right away.


### Judge verdicts (off → on)

| | Score | Supported | Unsupported |
| - | -: | -: | -: |
| Thinking off | 1.000 | 12 | 0 |
| Thinking on  | 1.000 | 16 | 0 |

**Reasoning (thinking off):**

> Every factual claim in the answer maps directly to an explicit statement in the retrieved passages. Smart Test's capabilities (gap report, generated tests, edge cases, error path tests, parametrized tests) are all listed verbatim in P3. The CLI command is taken directly from P2. The skipping-of-questions behavior is stated in P3. Fix Test's diagnosis, root-cause categories, retry count (3), and outcome description are all explicitly stated in P1. No claim relies on outside knowledge or inference beyond what the passages state.

**Reasoning (thinking on):**

> Every atomic factual claim in the answer is directly and explicitly supported by at least one retrieved passage. The CLI command, its flags, and its described output come verbatim from P2. All Smart Test capabilities, outputs, and the interactive question flow are stated explicitly in P3. All Fix Test behaviors (auto-diagnosis, up to 3 retries, root cause categories) are stated explicitly in P1. The "no API key or additional cost" claim is supported by both P1 and P3. There are no hallucinated details, invented flags, or inferences beyond what the passages state.

**Claims identified by judge (thinking off):**

Supported:
  - Smart Test finds untested code and generates pytest tests to cover it.
  - Smart Test produces a coverage gap report — a ranked list of untested functions with risk scores.
  - Smart Test produces generated tests — working pytest functions with assertions.
  - Smart Test produces edge cases — boundary values, empty inputs, and None handling.
  - Smart Test produces error path tests — tests for expected exceptions and failures.
  - Smart Test produces parametrized tests — @pytest.mark.parametrize for input combinations.
  - The CLI command to run test generation is: attune workflow run test-gen --path src/attune/help/engine.py
  - If you provide details upfront (e.g., 'generate tests for src/auth/'), the setup questions are skipped and it runs immediately.
  - Fix Test auto-diagnoses failing tests by classifying the root cause and applying a targeted fix.
  - Fix Test handles root causes such as import errors, mock mismatches, assertion drift, type errors, and missing fixtures.
  - Fix Test retries up to 3 times.
  - Fix Test ends with either a passing test or a clear explanation of what still needs manual attention.

Unsupported:


**Claims identified by judge (thinking on):**

Supported:
  - You can auto-generate pytest tests for uncovered code.
  - The command to run is: attune workflow run test-gen --path src/attune/help/engine.py
  - Running that command produces a generated test file with edge cases and assertions.
  - The Smart Test tool finds untested code and generates pytest tests to cover it.
  - Smart Test gives you a gap analysis that shows which public functions, branches, and error paths have zero coverage.
  - Smart Test produces a coverage gap report — a ranked list of untested functions with risk scores.
  - Smart Test produces generated tests — working pytest functions with assertions.
  - Smart Test produces edge cases — boundary values, empty inputs, None handling.
  - Smart Test produces error path tests — tests for expected exceptions and failures.
  - Smart Test produces parametrized tests — @pytest.mark.parametrize for input combos.
  - You'll be guided through a couple of quick questions: which module to target and whether you want gap analysis, test generation, or both.
  - If you provide details upfront (e.g. 'generate tests for src/auth/') the questions are skipped.
  - The Fix Test tool auto-diagnoses the root cause of failing tests and applies targeted fixes.
  - Fix Test re-runs the test after each repair attempt and retries up to 3 times.
  - Fix Test handles import errors, mock mismatches, assertion drift, and type errors.
  - Both tools run on your Claude subscription with no API key or additional cost.

Unsupported:


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

## gq-021 — `clean up technical debt`

### Retrieved context

> <passage>
> [P1] source: concepts/tool-refactor-plan.md
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
> [P2] source: concepts/tool-coach.md
> ---
> type: concept
> name: tool-coach
> tags: [skill, help-system, progressive-depth]
> source: plugin/skills/coach/SKILL.md
> ---
>
> # Coach
>
> Coach is the help system's own help. It explains any
> Attune topic using progressive depth -- start with a
> short concept overview, then drill into step-by-step
> instructions, then get the full reference with every
> detail. You never leave your conversation to look
> things up.
>
> ## How progressive depth works
>
> Every topic in the knowledge base exists at three
> levels. When you ask about something, you get the
> first level. Say "tell me more" and you get the next.
>
> | Level | Template type | What you get |
> |-------|---------------|--------------|
> | 0 | **Concept** | What is it? When would I use it? |
> | 1 | **Task** | Step-by-step: how to run it right now |
> | 2 | **Reference** | Full detail -- every option, edge case, configuration |
>
> This is not just "more words." Each level is a
> different kind of document written for a different
> need. The concept tells you whether you care. The
> task gets you moving. The reference answers the
> question you'll have in six months.
>
> ## Session state
>
> The engine tracks your current topic and depth level
> across your session. Asking about the same topic
> again advances to the next level automatically. Asking
> about a different topic resets to level 0 (concept).
> Sessions expire after 4 hours of inactivity.
>
> ## When you'd use it
>
> - You heard someone mention "security audit" and want
>   to know what it does -- ask and get the concept
> - You're ready to run it -- say "tell me more" for
>   the step-by-step
> - You need to configure exclusions or understand the
>   scoring -- say "tell me more" again for the full
>   reference
> - You just finished a workflow and want to understand
>   the results -- the engine can skip straight to the
>   task level
>
> ## Example: coaching on security audits
>
> ```
> You:    what is security audit?
> Coach:  [concept] What it finds, when to use it...
>         (say "tell me more" for step-by-step)
>
> You:    tell me more
> Coach:  [task] Quick start, choosing targets, reading results...
>         (say "tell me more" for full reference)
>
> You:    tell me more
> Coach:  [reference] All checks, CWE mappings, scoring, config...
>         (full detail)
> ```
>
> ## Want to learn more?
>
> - Say **"tell me more"** for a quick-start guide to
>   using coach
> - Say **"what is progressive depth?"** to understand
>   the engine behind it
> - Say **"how does the help system work?"** for the
>   broader architecture
> </passage>
>
> <passage>
> [P3] source: concepts/tool-fix-test.md
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

### Answer

## Cleaning Up Technical Debt

The right tool for this is the **Refactor Plan**, which is designed exactly for this purpose — it's listed under the aliases "technical debt" and "code cleanup." [P1]

### What It Does

A refactor plan scans your code for structural problems and builds a **prioritized roadmap** to fix them. [P1] It catches issues that accumulate quietly, like a class that grew into a god object, copy-pasted blocks that drifted apart, or a function with deep nesting that nobody wants to touch. [P1]

### What It Analyzes

| Category | What it finds |
|----------|---------------|
| **Code smells** | Long methods, god classes, feature envy, data clumps |
| **Duplication** | Copy-pasted blocks, near-duplicates, DRY violations |
| **Complexity** | High cyclomatic complexity, deep nesting, long chains |
| **Coupling** | Circular imports, tight dependencies, shotgun surgery |
| **Naming** | Abbreviations, generic names, inconsistent conventions |
| **Dead code** | Unreachable branches, unused params, vestigial modules |

[P1]

### How It Prioritizes

Items that are **high-severity, low-effort, and high-impact** float to the top. Risky changes get flagged so you can plan extra testing. [P1]

### Why Use a Plan?

Refactoring without a plan leads to yak-shaving — you start fixing one thing and end up touching 20 files. The roadmap tells you which changes deliver the most improvement per hour invested. [P1]

---

Say **"tell me more"** to get step-by-step instructions on running a refactor plan. [P1]


### Judge verdicts (off → on)

| | Score | Supported | Unsupported |
| - | -: | -: | -: |
| Thinking off | 1.000 | 17 | 0 |
| Thinking on  | 1.000 | 16 | 0 |

**Reasoning (thinking off):**

> Every factual claim in the answer maps directly to explicit statements in passage P1. The answer accurately reproduces the tool's aliases, its analysis categories and examples, its prioritization logic, and its reasoning for why a plan is needed. Even the closing prompt to say "tell me more" is taken verbatim from P1's "Want to learn more?" section. There are no hallucinated details, invented flags, or claims drawn from outside knowledge.

**Reasoning (thinking on):**

> Every atomic factual claim in the answer is drawn directly and faithfully from passage P1. The table rows, prioritization factors, yak-shaving metaphor, the "20 files" detail, and the "tell me more" prompt are all verbatim or near-verbatim from P1. No outside knowledge, invented details, or unsupported inferences were introduced. The answer is fully faithful to the retrieved passages.

**Claims identified by judge (thinking off):**

Supported:
  - The right tool for cleaning up technical debt is the Refactor Plan.
  - The Refactor Plan is listed under the aliases 'technical debt' and 'code cleanup'.
  - A refactor plan scans your code for structural problems and builds a prioritized roadmap to fix them.
  - It catches issues that accumulate quietly, like a class that grew into a god object.
  - It catches copy-pasted blocks that drifted apart.
  - It catches a function with deep nesting that nobody wants to touch.
  - The Refactor Plan analyzes code smells, including long methods, god classes, feature envy, and data clumps.
  - The Refactor Plan analyzes duplication, including copy-pasted blocks, near-duplicates, and DRY violations.
  - The Refactor Plan analyzes complexity, including high cyclomatic complexity, deep nesting, and long chains.
  - The Refactor Plan analyzes coupling, including circular imports, tight dependencies, and shotgun surgery.
  - The Refactor Plan analyzes naming, including abbreviations, generic names, and inconsistent conventions.
  - The Refactor Plan analyzes dead code, including unreachable branches, unused params, and vestigial modules.
  - Items that are high-severity, low-effort, and high-impact float to the top.
  - Risky changes get flagged so you can plan extra testing.
  - Refactoring without a plan leads to yak-shaving — you start fixing one thing and end up touching 20 files.
  - The roadmap tells you which changes deliver the most improvement per hour invested.
  - Saying 'tell me more' gets you step-by-step instructions on running a refactor plan.

Unsupported:


**Claims identified by judge (thinking on):**

Supported:
  - A refactor plan scans your code for structural problems and builds a prioritized roadmap to fix them.
  - It catches issues that accumulate quietly, such as a class that grew into a god object.
  - It catches copy-pasted blocks that drifted apart.
  - It catches a function with deep nesting that nobody wants to touch.
  - Code smells category finds: Long methods, god classes, feature envy, data clumps.
  - Duplication category finds: Copy-pasted blocks, near-duplicates, DRY violations.
  - Complexity category finds: High cyclomatic complexity, deep nesting, long chains.
  - Coupling category finds: Circular imports, tight dependencies, shotgun surgery.
  - Naming category finds: Abbreviations, generic names, inconsistent conventions.
  - Dead code category finds: Unreachable branches, unused params, vestigial modules.
  - Issues are ranked by four factors: severity, effort, impact, and risk.
  - Items that are high-severity, low-effort, and high-impact float to the top.
  - Risky changes get flagged so you can plan extra testing.
  - Refactoring without a plan leads to yak-shaving — you start fixing one thing and end up touching 20 files.
  - The roadmap tells you which changes deliver the most improvement per hour invested.
  - Saying 'tell me more' provides step-by-step instructions on running a refactor plan.

Unsupported:


### Your labels

```yaml
id: gq-021
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

## gq-029 — `improve code quality metrics`

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

## Improving Code Quality Metrics

Here are the key ways to analyze and improve your code quality metrics:

### Run a Code Quality Review

You can perform a structured code quality review using either of these approaches:

- Use the slash command: `/code-quality <path or directory to review>` [P2]
- Run via the CLI: `attune workflow run code-review --path "src/"` [P3]

### Understand What Gets Measured

A code quality review examines your code across multiple categories [P1]:

| Category | Examples | Severity |
|----------|---------|----------|
| **Style** | Unused imports, line length, naming | Low |
| **Correctness** | Unreachable code, wrong return types | High |
| **Likely bugs** | Broad exceptions, mutable defaults, race conditions | High |
| **Structural** | High coupling, god classes, circular deps | Medium |
| **Security overlap** | eval/exec, unvalidated paths | Critical |

### Choose the Right Depth

Select the review depth based on your situation [P1]:

- **Quick** — Style and formatting only; best for pre-commit checks and fast feedback
- **Standard** *(default)* — Adds logic errors and likely bugs; best for pull requests and regular development
- **Deep** — Adds security, architecture, and test gaps; best for pre-release or critical modules

### When to Run Reviews

Key moments to check metrics include: before opening a pull request, after a large refactor, or when inheriting unfamiliar code to get a quick read on its health [P1].

### Next Steps

After getting results, follow up with `/smart-test` to generate tests for flagged areas [P3].


### Judge verdicts (off → on)

| | Score | Supported | Unsupported |
| - | -: | -: | -: |
| Thinking off | 1.000 | 14 | 0 |
| Thinking on  | 1.000 | 25 | 0 |

**Reasoning (thinking off):**

> Every factual claim in the answer is directly traceable to at least one of the retrieved passages. The slash command and CLI command are verbatim from P2 and P3 respectively. The category table, depth table, timing recommendations, and the `/smart-test` follow-up suggestion are all explicitly stated in P1 and P3. No details appear to be invented or inferred beyond what the passages state.

**Reasoning (thinking on):**

> Every atomic factual claim in the answer is directly and explicitly supported by at least one retrieved passage. The slash command syntax and CLI command are taken verbatim from P2 and P3 respectively. All depth levels, their descriptions, and the default designation come directly from P1's depth table. All quality categories, their examples, severities, and auto-fixability statuses match P1's category table exactly. The "when to use" scenarios are lifted word-for-word from P1, and the `/smart-test` follow-up recommendation is stated explicitly in P3. No outside knowledge, inference, or invented details are present in the answer.

**Claims identified by judge (thinking off):**

Supported:
  - You can use the slash command `/code-quality <path or directory to review>` to perform a code quality review
  - You can run a code review via the CLI with `attune workflow run code-review --path "src/"`
  - A code quality review examines code across multiple categories including Style, Correctness, Likely bugs, Structural, and Security overlap
  - Style issues include unused imports, line length, and naming, with Low severity
  - Correctness issues include unreachable code and wrong return types, with High severity
  - Likely bugs include broad exceptions, mutable defaults, and race conditions, with High severity
  - Structural issues include high coupling, god classes, and circular deps, with Medium severity
  - Security overlap issues include eval/exec and unvalidated paths, with Critical severity
  - The Quick depth covers style and formatting only and is best for pre-commit checks and fast feedback
  - The Standard depth adds logic errors and likely bugs and is best for pull requests and regular development
  - The Deep depth adds security, architecture, and test gaps and is best for pre-release or critical modules
  - The default review depth is Standard
  - Key moments to run reviews include before opening a pull request, after a large refactor, or when inheriting unfamiliar code
  - You can follow up with `/smart-test` to generate tests for flagged areas

Unsupported:


**Claims identified by judge (thinking on):**

Supported:
  - The slash command in Claude Code is `/code-quality <path or directory to review>`
  - The CLI workflow command is `attune workflow run code-review --path "src/"`
  - The CLI command produces quality findings grouped by severity
  - Quick depth covers style and formatting only
  - Standard depth covers logic errors and likely bugs in addition to style
  - Standard depth is the default
  - Deep depth covers security, architecture, and test gaps in addition to standard
  - A code quality review covers multiple categories at once rather than requiring separate tools
  - Style issues include unused imports, line length, and naming
  - Style issues are Low severity
  - Style issues are often auto-fixable
  - Correctness issues include unreachable code and wrong return types
  - Correctness issues are High severity
  - Likely bug issues include broad exceptions, mutable defaults, and race conditions
  - Likely bug issues are High severity
  - Likely bug issues are not auto-fixable
  - Structural issues include high coupling, god classes, and circular dependencies
  - Structural issues are Medium severity
  - Security overlap issues include eval/exec and unvalidated paths
  - Security overlap issues are Critical severity
  - Security overlap is available at the Deep level only
  - The review should be run before opening a pull request to catch issues before reviewers do
  - The review should be run after a large refactor to verify nothing degraded
  - The review should be run when inheriting unfamiliar code to get a quick read on its health
  - After reviewing, you can follow up with `/smart-test` to generate tests for flagged areas

Unsupported:


### Your labels

```yaml
id: gq-029
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

## gq-033 — `add docstrings to all my functions`

### Retrieved context

> <passage>
> [P1] source: concepts/tool-doc-gen.md
> ---
> type: concept
> name: tool-doc-gen
> tags: [documentation, docstrings, readme]
> source: plugin/skills/doc-gen/SKILL.md
> ---
>
> # Doc Generation
>
> Doc-gen reads your actual source code -- function signatures,
> type hints, class hierarchies, module structure -- and generates
> documentation from it. Instead of writing docstrings by hand
> or letting them drift from the code, you get accurate docs that
> reflect the current API.
>
> ## What it generates
>
> | Doc type | What you get | Source |
> |----------|-------------|--------|
> | **Docstrings** | Google-style with Args, Returns, Raises, Examples | Function signatures and type hints |
> | **README sections** | Feature lists, usage examples, install instructions | Module exports and public API |
> | **API reference** | Full function/class signatures with descriptions | All public symbols in a module |
> | **Module overview** | Architecture summary, dependency map, entry points | Package structure and imports |
>
> ## When you'd use it
>
> Run doc-gen after creating new public APIs, before a release
> to refresh stale docs, when onboarding contributors who need
> to understand a module, or when you inherit a codebase with
> missing docstrings. It reads the source of truth so the docs
> never contradict the code.
>
> ## How it works
>
> Doc-gen is a Socratic skill. It asks you two questions before
> generating anything:
>
> 1. **What to document** -- a single file, a module, or a
>    directory tree
> 2. **What format** -- docstrings, README, API reference, or
>    a full pipeline that audits gaps and generates docs for
>    all of them
>
> Then it reads the source, generates the documentation, and
> shows you the result before applying changes.
>
> ## Example output
>
> A generated Google-style docstring looks like this:
>
> ```python
> def validate_email(email: str) -> bool:
>     """Validate email format using regex.
>
>     Args:
>         email: Email address to validate.
>
>     Returns:
>         True if email matches a valid format,
>         False otherwise.
>
>     Raises:
>         TypeError: If email is not a string.
>
>     Example:
>         >>> validate_email("user@example.com")
>         True
>     """
> ```
>
> ## Want to learn more?
>
> - Say **"tell me more"** for step-by-step instructions
> - Say **"what is code quality?"** for a broader code review
> - Say **"what is smart-test?"** to generate tests alongside docs
> </passage>
>
> <passage>
> [P2] source: concepts/task-dependency-management.md
> ---
> type: concept
> name: task-dependency-management
> tags: [deps, packaging, python]
> source: developer-guidance
> ---
>
> # Dependency Management
>
> Every package you add to your project is a decision with
> long-term consequences. A dependency brings code you
> didn't write into your supply chain — its bugs become
> your bugs, its vulnerabilities become your vulnerabilities,
> and its license terms apply to your users. Dependency
> management is the practice of making those decisions
> deliberately instead of by accident.
>
> ## What to evaluate before adding a dependency
>
> | Concern | What to check | Risk if ignored |
> |---------|---------------|-----------------|
> | **Security** | Known CVEs, advisory history, maintainer reputation | Shipping exploitable code to production |
> | **Compatibility** | Python version support, conflicts with existing deps | Import failures, runtime crashes |
> | **License** | SPDX identifier, copyleft vs permissive, patent clauses | Legal liability, license violations |
> | **Size** | Install footprint, transitive dependency count | Bloated containers, slow CI, larger attack surface |
> | **Maintenance** | Last release date, open issue ratio, bus factor | Stuck on a dead library with no security patches |
> | **Alternatives** | Can stdlib or existing deps cover this? | Unnecessary complexity |
>
> ## When you'd think about this
>
> When adding a new package to `pyproject.toml`. When
> upgrading a pinned version. When a security scanner
> flags a CVE in your dependency tree. When CI breaks
> after a transitive dependency released a bad version.
> When preparing a release and you want to verify
> nothing risky slipped in since the last one.
>
> ## The dependency lifecycle
>
> | Phase | What happens | Tools involved |
> |-------|-------------|----------------|
> | **Evaluate** | Check license, security, maintenance | PyPI, GitHub, `pip-licenses` |
> | **Add** | Pin version, install, verify | `pyproject.toml`, `pip install` |
> | **Audit** | Scan for known vulnerabilities | `pip-audit`, security audit |
> | **Update** | Bump versions, test compatibility | Dependabot, `pip-compile` |
> | **Remove** | Drop unused deps, shrink tree | `pip-autoremove`, manual review |
>
> ## Want to learn more?
>
> - Say **"tell me more"** for step-by-step instructions
>   on adding and auditing dependencies
> - Say **"show me the reference"** for version constraint
>   syntax, lockfile management, and tooling details
> - Say **"run a security audit"** to scan your current
>   dependencies for known CVEs
> - Say **"help me prepare a release"** to verify
>   dependencies before publishing
> </passage>
>
> <passage>
> [P3] source: quickstarts/skill-doc-gen.md
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

### Answer

## Adding Docstrings to Your Functions

You can use the **doc-gen** skill to automatically generate docstrings from your source code.

### How to run it

```
/doc-gen <path or module to document>
```
[P3]

### What doc-gen will do

Doc-gen reads your actual source code — function signatures, type hints, class hierarchies, and module structure — and generates documentation from it [P1]. For docstrings specifically, it generates **Google-style** format with `Args`, `Returns`, `Raises`, and `Examples` sections, derived from your function signatures and type hints [P1].

Here's an example of what a generated docstring looks like [P1]:

```python
def validate_email(email: str) -> bool:
    """Validate email format using regex.

    Args:
        email: Email address to validate.

    Returns:
        True if email matches a valid format,
        False otherwise.

    Raises:
        TypeError: If email is not a string.

    Example:
        >>> validate_email("user@example.com")
        True
    """
```

### What to expect

Before generating anything, doc-gen will ask you two questions [P1]:

1. **What to document** — a single file, a module, or a directory tree
2. **What format** — in your case, docstrings

It will then show you the generated documentation **before applying any changes** [P1].

### When this is especially useful

Doc-gen is a good fit when you have missing docstrings or when existing ones have drifted out of sync with the code, since it reads the source of truth directly [P1].


### Judge verdicts (off → on)

| | Score | Supported | Unsupported |
| - | -: | -: | -: |
| Thinking off | 0.950 | 19 | 1 |
| Thinking on  | 0.944 | 17 | 1 |

**Reasoning (thinking off):**

> The answer is overwhelmingly faithful to the retrieved passages. Nearly every factual claim maps directly to explicit statements in P1 or P3. The command syntax comes from P3. The description of what doc-gen reads, the Google-style format with its sections, the two Socratic questions, the 'show before applying' behavior, and the example docstring are all reproduced verbatim or near-verbatim from P1. The only minor unsupported element is the answer's editorial narrowing of the second question's scope to "in your case, docstrings," which slightly misrepresents the passage (P1 lists all four format options without suggesting the user would only pick docstrings). Everything else is directly grounded in the passages.

**Reasoning (thinking on):**

> The answer is almost entirely faithful to the retrieved passages. Nearly every factual claim — the skill name, the Google-style docstring format with Args/Returns/Raises/Examples, the exact example code, the /doc-gen command syntax, the two Socratic questions and their exact options, the show-before-apply behaviour, and the claim that docs reflect the current API — is directly lifted from P1 and P3. The one unsupported claim is the editorial recommendation that the user "would choose docstrings" as the format answer; the passages list docstrings as one of several format options but never direct the user to pick it for this scenario. That is a reasonable but passage-unsubstantiated inference.

**Claims identified by judge (thinking off):**

Supported:
  - You can use the doc-gen skill to automatically generate docstrings from your source code.
  - The command to run doc-gen is `/doc-gen <path or module to document>`.
  - Doc-gen reads your actual source code — function signatures, type hints, class hierarchies, and module structure — and generates documentation from it.
  - For docstrings specifically, doc-gen generates Google-style format.
  - Google-style docstrings include Args, Returns, Raises, and Examples sections.
  - Docstrings are derived from function signatures and type hints.
  - The example generated docstring is for a function called `validate_email` that takes an `email: str` parameter and returns `bool`.
  - The example docstring includes a description 'Validate email format using regex.'
  - The example docstring includes an Args section with 'email: Email address to validate.'
  - The example docstring includes a Returns section with 'True if email matches a valid format, False otherwise.'
  - The example docstring includes a Raises section with 'TypeError: If email is not a string.'
  - The example docstring includes an Example section with `>>> validate_email("user@example.com")` returning `True`.
  - Before generating anything, doc-gen will ask two questions.
  - The first question is what to document — a single file, a module, or a directory tree.
  - The second question is what format — docstrings, README, API reference, or a full pipeline.
  - Doc-gen shows you the generated documentation before applying changes.
  - Doc-gen is useful when you have missing docstrings.
  - Doc-gen is useful when existing docstrings have drifted out of sync with the code.
  - Doc-gen reads the source of truth directly.

Unsupported:
  - The second question format options are narrowed to 'in your case, docstrings' — this is a characterization added by the answer, not stated in the passages (the passages list all format options without narrowing).

**Claims identified by judge (thinking on):**

Supported:
  - You can add docstrings automatically using the doc-gen skill
  - Doc-gen reads your actual source code and generates documentation from it
  - Doc-gen reads function signatures, type hints, and class hierarchies
  - Doc-gen produces Google-style docstrings
  - The generated docstrings include Args, Returns, Raises, and Examples sections
  - Docstrings are derived from function signatures and type hints
  - The example docstring shown is for a validate_email function with a str parameter returning bool
  - The example docstring includes an Args section with 'email: Email address to validate.'
  - The example docstring includes a Returns section indicating True or False
  - The example docstring includes a Raises section for TypeError if email is not a string
  - The example docstring includes an Example with validate_email('user@example.com') returning True
  - The command to run doc-gen is /doc-gen <path or module to document>
  - Doc-gen asks two questions before generating anything
  - The first question is what to document — a single file, a module, or a directory tree
  - The second question is what format — docstrings, README, API reference, or a full pipeline
  - Doc-gen reads the source, generates the documentation, and shows the result before applying changes
  - The generated docstrings reflect the current API and won't contradict the code

Unsupported:
  - In the user's case, they would choose 'docstrings' as the format option

### Your labels

```yaml
id: gq-033
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

## gq-035 — `publish to PyPI`

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
> [P2] source: concepts/task-package-publishing.md
> ---
> type: concept
> name: task-package-publishing
> tags: [publishing, pypi, python, packaging]
> source: developer-guidance
> ---
>
> # Package Publishing
>
> Publishing a Python package to PyPI is the final step
> in making your code available to the world. It looks
> simple -- build, upload, done -- but each stage has
> failure modes that are invisible until a user tries to
> install your package and something breaks.
>
> ## The publish cycle
>
> Every release follows the same sequence. Skipping a
> step is how broken packages end up on PyPI:
>
> | Step | What happens | What can go wrong |
> |------|-------------|-------------------|
> | **Version bump** | Update version in `pyproject.toml` | Forgetting to bump, publishing a duplicate |
> | **Clean build** | Remove old `dist/`, rebuild | Stale artifacts upload the previous version |
> | **TestPyPI** | Upload to test index, install from it | README doesn't render, missing files |
> | **Production publish** | Upload to real PyPI | Version already exists (immutable), broken links |
> | **Verify** | Install from PyPI in a fresh env | Wrong dependencies, missing extras |
>
> ## Semantic versioning
>
> Version numbers communicate intent to your users:
>
> | Component | When to bump | Example |
> |-----------|-------------|---------|
> | **Major** (X.0.0) | Breaking changes to public API | Renamed function, removed parameter |
> | **Minor** (0.X.0) | New features, backward compatible | Added a new command, new optional parameter |
> | **Patch** (0.0.X) | Bug fixes, no API changes | Fixed a crash, corrected a typo in output |
>
> Pre-release suffixes like `1.0.0rc1` or `1.0.0a1` let
> early adopters test without affecting stable users.
> PyPI treats them as separate from the release version.
>
> ## Why README rendering matters
>
> PyPI uses your `README.md` as the package description
> page. What works on GitHub may break on PyPI because:
>
> - **Relative links break.** `docs/ARCHITECTURE.md`
>   becomes `https://pypi.org/project/your-package/docs/...`
>   which does not exist. Use absolute GitHub URLs.
> - **Some HTML is stripped.** PyPI's markdown renderer
>   is stricter than GitHub's.
> - **Images need absolute URLs.** Relative image paths
>   produce broken images on the PyPI page.
>
> TestPyPI lets you preview the rendered README before
> publishing to production.
>
> ## What usually goes wrong
>
> | Problem | Root cause | How to avoid it |
> |---------|-----------|-----------------|
> | **Stale dist** | Old files in `dist/` from a previous build | Always `rm -rf dist/` before building |
> | **Wrong version on PyPI** | Built before updating `pyproject.toml` | Verify version in `dist/` filenames before uploading |
> | **"Version already exists"** | PyPI versions are immutable once uploaded | Bump to a new version; you cannot overwrite |
> | **README not rendering** | Relative links, unsupported HTML, bad markdown | Test on TestPyPI first; use absolute URLs |
> | **Missing package data** | Files not included in the sdist/wheel | Check `MANIFEST.in` or build backend config |
> | **Broken install** | Missing or incorrect dependencies in metadata | Test `pip install` in a clean virtual environment |
>
> ## When you'd think about this
>
> When you're ready to share your package with others.
> When cutting a release after a milestone. When CI
> passes and you want to make the latest version
> available. When you need to publish a security patch.
>
> ## Want to learn more?
>
> - Say **"how do I publish my package?"** for the full
>   step-by-step guide
> - Say **"show me the publishing reference"** for
>   pyproject.toml fields, build tools, and twine commands
> - Say **"I need to publish my package"** for a 5-minute
>   quickstart
> - Ask **"/release"** for pre-publish health checks,
>   changelog generation, and version bumps
> - Ask **"/security"** to scan for supply chain risks
>   before publishing
> </passage>
>
> <passage>
> [P3] source: quickstarts/task-package-publishing.md
> ---
> type: quickstart
> name: task-package-publishing
> tags: [publishing, pypi, python, packaging]
> source: developer-guidance
> ---
>
> # Quickstart: Publish Your Package
>
> Get from "ready to publish" to "verified on PyPI"
> in 5 steps.
>
> ## Step 1: Bump the version
>
> Edit `pyproject.toml`:
>
> ```toml
> [project]
> version = "2.1.0"
> ```
>
> Pick the right bump: major for breaking changes, minor
> for new features, patch for bug fixes.
>
> ## Step 2: Clean build
>
> ```bash
> rm -rf dist/ build/ *.egg-info
> python -m build
> ```
>
> Check that the filenames in `dist/` show your new
> version. If they show the old version, you forgot to
> save `pyproject.toml`.
>
> ## Step 3: Test on TestPyPI
>
> ```bash
> twine upload --repository testpypi dist/*
> ```
>
> Visit `https://test.pypi.org/project/your-package/`
> and verify the README renders correctly and the version
> is right. Then test the install:
>
> ```bash
> pip install --index-url https://test.pypi.org/simple/ \
>     --extra-index-url https://pypi.org/simple/ \
>     your-package==2.1.0
> ```
>
> ## Step 4: Publish to PyPI
>
> ```bash
> twine upload dist/*
> ```
>
> This is permanent. You cannot overwrite a version once
> it is uploaded.
>
> ## Step 5: Verify
>
> ```bash
> pip install your-package==2.1.0
> python -c "import your_package; print(your_package.__version__)"
> ```
>
> If the version matches, you are done. Tag the release:
>
> ```bash
> git tag v2.1.0
> git push origin v2.1.0
> ```
>
> **Done.** Your package is live on PyPI.
>
> ## What you get
>
> | Feature | How it works |
> |---------|-------------|
> | Installable package | Anyone can `pip install your-package` |
> | Rendered README | PyPI shows your README as the package page |
> | Versioned releases | Users pin to specific versions |
> | Extras support | `pip install 'your-package[redis]'` installs optional deps |
>
> ## Preflight checklist
>
> | Step | What to check | Common mistake | Recovery |
> |------|--------------|----------------|----------|
> | Version bump | `pyproject.toml` version is new | Forgot to bump | Fix and rebuild |
> | Clean build | `dist/` only has new version files | Stale artifacts from last build | `rm -rf dist/` and rebuild |
> | TestPyPI | README renders, install works | Relative links break on PyPI | Fix links, bump patch, re-upload |
> | Publish | Twine succeeds without errors | Token not configured | Set `TWINE_PASSWORD` env var |
> | Verify | Correct version installs and imports | Published stale dist | Bump version and re-publish |
>
> ## Want to learn more?
>
> - Say **"how do I publish my package?"** for the full
>   step-by-step guide with detailed verification tables
> - Say **"show me the publishing reference"** for every
>   pyproject.toml field, build tool, and twine command
> - Say **"what is package publishing?"** for semantic
>   versioning and what can go wrong
> - Ask **"/release"** to run pre-publish health checks
>   and generate a changelog automatically
> - Ask **"/security"** to scan dependencies for supply
>   chain vulnerabilities before publishing
> </passage>

### Answer

# Publishing to PyPI

Here's how to get your package from ready to published in 5 steps.

---

## Step 1: Bump the Version

Edit `pyproject.toml` with the new version number [P3]:

```toml
[project]
version = "2.1.0"
```

Choose the right bump type [P2]:
- **Major** (X.0.0) — breaking changes to the public API
- **Minor** (0.X.0) — new features, backward compatible
- **Patch** (0.0.X) — bug fixes, no API changes

---

## Step 2: Clean Build

```bash
rm -rf dist/ build/ *.egg-info
python -m build
```

Check that filenames in `dist/` show your new version — if they show the old version, you forgot to save `pyproject.toml` [P3]. Always remove old `dist/` before building to avoid stale artifacts uploading the previous version [P2].

---

## Step 3: Test on TestPyPI

```bash
twine upload --repository testpypi dist/*
```

Visit `https://test.pypi.org/project/your-package/` to verify the README renders correctly [P3]. This matters because relative links break on PyPI (e.g. `docs/ARCHITECTURE.md` becomes a non-existent PyPI URL), and images need absolute URLs [P2].

Then test the install:

```bash
pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    your-package==2.1.0
```
[P3]

---

## Step 4: Publish to PyPI

```bash
twine upload dist/*
```

**This is permanent.** PyPI versions are immutable once uploaded — you cannot overwrite a version that already exists [P2, P3].

---

## Step 5: Verify

```bash
pip install your-package==2.1.0
python -c "import your_package; print(your_package.__version__)"
```

If the version matches, tag the release [P3]:

```bash
git tag v2.1.0
git push origin v2.1.0
```

---

## Common Mistakes to Avoid

| Problem | How to avoid it |
|---------|----------------|
| Stale dist | Always `rm -rf dist/` before building [P2] |
| Wrong version in dist | Verify version in `dist/` filenames before uploading [P2] |
| "Version already exists" | Bump to a new version; you cannot overwrite [P2] |
| README not rendering | Test on TestPyPI first; use absolute URLs [P2] |
| Broken install | Test `pip install` in a clean virtual environment [P2] |

---

## Before You Publish

Consider running a preflight checklist first — it checks tests, security, changelog, dependencies, and version consistency, then gives you a **GO / NO-GO** verdict before you ship [P1].


### Judge verdicts (off → on)

| | Score | Supported | Unsupported |
| - | -: | -: | -: |
| Thinking off | 1.000 | 25 | 0 |
| Thinking on  | 1.000 | 42 | 0 |

**Reasoning (thinking off):**

> Every atomic factual claim in the answer is directly supported by at least one of the retrieved passages. The 5-step publish flow (version bump, clean build, TestPyPI, publish, verify) and all associated commands, URLs, and warnings come verbatim or near-verbatim from P3. The semantic versioning table (major/minor/patch), the immutability of PyPI versions, the README rendering issues (relative links, absolute image URLs), and the common-mistakes table all come from P2. The preflight checklist description (tests, security, changelog, dependencies, version consistency, GO/NO-GO verdict) comes from P1. No outside knowledge or invented details were introduced.

**Reasoning (thinking on):**

> Every atomic factual claim in the answer maps directly to at least one retrieved passage. Step-by-step commands (build, TestPyPI upload, install verification, production upload, git tagging) are all verbatim from P3. Semantic versioning rules, common failure modes, and their fixes are explicitly stated in P2. The release prep / GO/NO-GO description is explicitly covered in P1. No claims rely on outside knowledge, invented flags, or inferred details beyond what the passages state.

**Claims identified by judge (thinking off):**

Supported:
  - Edit `pyproject.toml` with the new version number
  - The version field in pyproject.toml is set under `[project]`
  - Major (X.0.0) bump is for breaking changes to the public API
  - Minor (0.X.0) bump is for new features, backward compatible
  - Patch (0.0.X) bump is for bug fixes, no API changes
  - The clean build command is `rm -rf dist/ build/ *.egg-info` followed by `python -m build`
  - Check that filenames in `dist/` show your new version
  - If dist filenames show the old version, you forgot to save `pyproject.toml`
  - Always remove old `dist/` before building to avoid stale artifacts uploading the previous version
  - The TestPyPI upload command is `twine upload --repository testpypi dist/*`
  - Visit `https://test.pypi.org/project/your-package/` to verify the README renders correctly
  - Relative links break on PyPI (e.g. `docs/ARCHITECTURE.md` becomes a non-existent PyPI URL)
  - Images need absolute URLs on PyPI
  - The test install command uses `--index-url https://test.pypi.org/simple/` and `--extra-index-url https://pypi.org/simple/`
  - The production publish command is `twine upload dist/*`
  - PyPI versions are immutable once uploaded
  - You cannot overwrite a version that already exists on PyPI
  - The verify commands are `pip install your-package==2.1.0` and `python -c "import your_package; print(your_package.__version__)"`
  - Tag the release with `git tag v2.1.0` and `git push origin v2.1.0`
  - Always `rm -rf dist/` before building to avoid stale dist
  - Verify version in `dist/` filenames before uploading
  - Test on TestPyPI first and use absolute URLs to avoid README not rendering
  - Test `pip install` in a clean virtual environment to avoid broken install
  - A preflight checklist checks tests, security, changelog, dependencies, and version consistency
  - The preflight checklist gives a GO / NO-GO verdict before you ship

Unsupported:


**Claims identified by judge (thinking on):**

Supported:
  - Edit pyproject.toml with the new version number to bump the version
  - The version example used is '2.1.0'
  - Breaking API changes require a Major (X.0.0) version bump
  - New features that are backward compatible require a Minor (0.X.0) version bump
  - Bug fixes only require a Patch (0.0.X) version bump
  - A renamed function is an example of a breaking change
  - Adding a new command is an example of a new feature
  - Fixing a crash is an example of a bug fix
  - The clean build command is: rm -rf dist/ build/ *.egg-info followed by python -m build
  - Check that filenames in dist/ show your new version
  - If filenames show the old version, you forgot to save pyproject.toml
  - Always remove old dist/ files before building
  - Stale artifacts can cause the previous version to be uploaded
  - The TestPyPI upload command is: twine upload --repository testpypi dist/*
  - The TestPyPI install verification uses --index-url https://test.pypi.org/simple/
  - The TestPyPI install verification uses --extra-index-url https://pypi.org/simple/
  - Relative links in your README will break on PyPI
  - TestPyPI lets you preview README rendering before publishing to production
  - Relative image paths become broken images on the PyPI package page
  - The publish command is: twine upload dist/*
  - PyPI versions are immutable once uploaded
  - You cannot overwrite an existing version on PyPI
  - If something goes wrong after publishing, you must bump to a new version
  - The verify install command is: pip install your-package==2.1.0
  - The verify import command is: python -c "import your_package; print(your_package.__version__)"
  - Tag the release with: git tag v2.1.0
  - Push the tag with: git push origin v2.1.0
  - Wrong version uploaded is caused by building before saving pyproject.toml
  - Fix for wrong version: check filenames in dist/ before uploading
  - The 'Version already exists' error is because PyPI versions are immutable
  - Fix for 'Version already exists': bump to a new version
  - README not rendering is caused by relative links or unsupported HTML
  - Fix for README not rendering: use absolute URLs and test on TestPyPI first
  - Broken install is caused by missing or incorrect dependencies in metadata
  - Fix for broken install: test pip install in a clean virtual environment
  - A release prep tool can verify that tests pass
  - A release prep tool checks that lint is clean
  - A release prep tool checks that coverage meets a threshold
  - A release prep tool checks that no critical CVEs exist
  - A release prep tool checks that a changelog entry is present
  - A release prep tool checks that the version bump is correct
  - A release prep tool gives a GO / NO-GO verdict

Unsupported:


### Your labels

```yaml
id: gq-035
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

## gq-036 — `identify which files are most likely to break`

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
> [P2] source: quickstarts/task-git-workflow.md
> ---
> type: quickstart
> name: task-git-workflow
> tags: [git, workflow, branching, merge]
> source: developer-guidance
> ---
>
> # I Have a Merge Conflict
>
> Five steps to resolve it and get back to work.
>
> ## Step 1: See which files conflict
>
> ```
> git status
> ```
>
> Files marked `both modified` have conflicts.
>
> ## Step 2: Open the first conflicted file
>
> Look for these markers:
>
> ```
> <<<<<<< HEAD
> your version
> =======
> their version
> >>>>>>> feature-branch
> ```
>
> Everything between `<<<<<<<` and `=======` is your code.
> Everything between `=======` and `>>>>>>>` is theirs.
>
> ## Step 3: Decide what to keep
>
> Delete the markers and keep the right code. Sometimes
> that means keeping yours, sometimes theirs, sometimes a
> blend of both.
>
> ```python
> # Before (conflicted)
> <<<<<<< HEAD
> def get_user(user_id: int) -> User:
>     return db.query(User).get(user_id)
> =======
> def get_user(user_id: int) -> User | None:
>     return db.query(User).filter_by(id=user_id).first()
> >>>>>>> feature-branch
>
> # After (resolved -- kept the better return type
> # and the safer query)
> def get_user(user_id: int) -> User | None:
>     return db.query(User).filter_by(id=user_id).first()
> ```
>
> Repeat for every conflict in the file, then repeat for
> every conflicted file.
>
> ## Step 4: Stage and commit
>
> ```
> git add .
> git commit
> ```
>
> Git fills in a merge commit message. Accept it or add a
> note about how you resolved the conflict.
>
> ## Step 5: Verify nothing broke
>
> ```
> git log --oneline --graph -5
> pytest
> ```
>
> The merge commit should appear in the log, and your tests
> should still pass.
>
> **Done.** The conflict is resolved and committed.
>
> ## Quick fixes for common situations
>
> | Situation | What to do |
> |---|---|
> | You want to abort the whole merge | `git merge --abort` |
> | You know their version is correct for a file | `git checkout --theirs path/file && git add path/file` |
> | You know your version is correct for a file | `git checkout --ours path/file && git add path/file` |
> | You accidentally committed a bad resolution | `git revert HEAD` to undo the merge commit |
> | Conflict happened during rebase, not merge | Resolve the file, then `git add . && git rebase --continue` |
>
> ## Want to learn more?
>
> - Say **"what are git workflow strategies?"** for branch
>   strategy concepts -- trunk-based, feature branch, and
>   gitflow
> - Say **"walk me through rebase and cherry-pick"** for
>   the full step-by-step guide to all git operations
> - Say **"show me all git commands"** for the complete
>   reference with risk ratings and safer alternatives
> - Try **/release** to prepare a release branch with
>   automated health checks
> - Try **/code-quality** to verify pre-commit hooks pass
>   before pushing
>
> ## Related Topics
>
> - **Concept**: Git workflow -- branch strategies, merge
>   vs rebase, and conflict resolution philosophy
> - **Task**: Git workflow -- step-by-step guides for
>   conflicts, rebasing, cherry-pick, and stash
> - **Reference**: Git workflow -- full command catalog
>   with risk ratings and safer alternatives
> </passage>
>
> <passage>
> [P3] source: references/tool-bug-predict.md
> ---
> type: reference
> subtype: tabular
> name: tool-bug-predict
> category: tool
> tags: [mcp, tool, workflow]
> source: src/attune/mcp/tool_schemas.py
> ---
>
> # Reference: Tool: Bug Predict
>
> Run bug prediction workflow. Analyzes code patterns and predicts potential bugs before they occur.
>
> **Group:** workflow
>
> ## Parameters
>
> | Parameter | Type | Description | Constraints | Default |
> | --------- | ---- | ----------- | ----------- | ------- |
> | `path` | string | Path to directory or file to analyze |  | required |
>
> ## Usage
>
> `bug_predict(path="...")`
>
> ## Related Topics
> - **Reference**: Tool: Security Audit — Run security audit workflow on codebase. Detects vulnerabili...
> - **Reference**: Tool: Code Review — Run code review workflow. Provides comprehensive code qualit...
> - **Reference**: Tool: Test Generation — Generate tests for code. Can batch generate tests for multip...
> </passage>

### Answer

## Identifying Files Most Likely to Break

You can use the **Bug Prediction** tool to find your highest-risk files. Here's what it looks for:

---

### How to Run It

```python
bug_predict(path="...")
```

Point it at a directory or file you want to analyze [P3].

---

### What It Flags

The scanner identifies three categories of risky patterns [P1]:

| Pattern | Severity | What It Means |
|---|---|---|
| `eval()`, `exec()`, `compile()` on input | HIGH | Code injection vectors |
| Bare `except:` or unlogged `except Exception:` | MEDIUM | Errors silently swallowed |
| TODO, FIXME, HACK, XXX comments | LOW | Unfinished code paths that break under edge cases |

---

### Beyond Pattern Matching

The tool also weighs contextual signals that increase the likelihood a file will break [P1]:

- **Cyclomatic complexity** — deeply nested conditionals and long function bodies correlate with higher defect rates
- **Change frequency** — files modified often ("hot" files) are more likely to contain regressions
- **Code smells** — functions over 50 lines, classes with too many methods, or duplicated logic across modules

---

### False Positive Filtering

Not every match is a real bug. The scanner automatically suppresses known-safe patterns such as `eval()` inside test fixture strings, JavaScript `regex.exec()` calls, and broad exceptions marked with `# INTENTIONAL:` or `# noqa: BLE001` [P1].

---

### When to Run It

Good moments to use this include before merging a large PR, during code review, after onboarding unfamiliar code, on high-churn modules, or before a release [P1].


### Judge verdicts (off → on)

| | Score | Supported | Unsupported |
| - | -: | -: | -: |
| Thinking off | 1.000 | 18 | 0 |
| Thinking on  | 1.000 | 17 | 0 |

**Reasoning (thinking off):**

> Every factual claim in the answer is directly and explicitly supported by the retrieved passages. The tool invocation syntax and path parameter come from P3. All pattern categories (dangerous_eval, broad_exception, incomplete_code), their severities, contextual risk signals (cyclomatic complexity, change frequency, code smells), false-positive suppression rules, and recommended usage occasions are all drawn verbatim or near-verbatim from P1. There are no hallucinated details, invented CLI flags, or unsupported assertions.

**Reasoning (thinking on):**

> Every atomic factual claim in the answer maps directly to content in the retrieved passages. The tool invocation syntax and path parameter come from P3. All three risk pattern categories (dangerous_eval, broad_exception, incomplete_code) with their severities and descriptions, the contextual risk signals (cyclomatic complexity, change frequency, code smells with their specific thresholds), the false-positive suppression examples, and the recommended use-case scenarios all appear verbatim or near-verbatim in P1. No claims rely on outside knowledge or invented details.

**Claims identified by judge (thinking off):**

Supported:
  - You can use the Bug Prediction tool to find your highest-risk files
  - The tool is run with bug_predict(path='...')
  - The path parameter points to a directory or file to analyze
  - The scanner identifies eval(), exec(), compile() on input as HIGH severity code injection vectors
  - The scanner identifies bare except: or unlogged except Exception: as MEDIUM severity errors silently swallowed
  - The scanner identifies TODO, FIXME, HACK, XXX comments as LOW severity unfinished code paths that break under edge cases
  - The tool weighs cyclomatic complexity — deeply nested conditionals and long function bodies correlate with higher defect rates
  - The tool weighs change frequency — files modified often ('hot' files) are more likely to contain regressions
  - The tool weighs code smells — functions over 50 lines, classes with too many methods, or duplicated logic across modules
  - The scanner suppresses eval() inside test fixture strings as a known-safe pattern
  - The scanner suppresses JavaScript regex.exec() calls as a known-safe pattern
  - The scanner suppresses broad exceptions marked with # INTENTIONAL: as a known-safe pattern
  - The scanner suppresses broad exceptions marked with # noqa: BLE001 as a known-safe pattern
  - Good moments to use the tool include before merging a large PR
  - Good moments to use the tool include during code review
  - Good moments to use the tool include after onboarding unfamiliar code
  - Good moments to use the tool include on high-churn modules
  - Good moments to use the tool include before a release

Unsupported:


**Claims identified by judge (thinking on):**

Supported:
  - The Bug Prediction tool can be used to find highest-risk files.
  - The tool is invoked with `bug_predict(path="...")`.
  - The path parameter accepts a directory or file to analyze.
  - The scanner looks for three categories of risky patterns.
  - `eval()`, `exec()`, and `compile()` on input are HIGH severity and represent code injection vectors.
  - Bare `except:` or unlogged `except Exception:` is MEDIUM severity and means errors are silently swallowed.
  - TODO, FIXME, HACK, and XXX comments are LOW severity and represent unfinished code paths that break under edge cases.
  - The scanner weighs contextual factors that increase the likelihood a file will break.
  - Cyclomatic complexity (deeply nested conditionals and long function bodies) correlates with higher defect rates.
  - Files modified often ('hot' files) are more likely to contain regressions.
  - Code smells include functions over 50 lines, classes with too many methods, and duplicated logic across modules.
  - The scanner suppresses `eval()` inside test fixture strings as a known-safe pattern.
  - The scanner suppresses JavaScript `regex.exec()` calls as a known-safe pattern.
  - Not every match is a real risk; the scanner filters false positives automatically.
  - The tool is especially useful before merging a large PR.
  - The tool is useful during code review to focus human attention on real risks.
  - The tool is useful before a release to verify no new high-severity patterns crept in.

Unsupported:


### Your labels

```yaml
id: gq-036
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

## gq-040 — `find potential runtime errors before deployment`

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
> [P2] source: references/tool-bug-predict.md
> ---
> type: reference
> subtype: tabular
> name: tool-bug-predict
> category: tool
> tags: [mcp, tool, workflow]
> source: src/attune/mcp/tool_schemas.py
> ---
>
> # Reference: Tool: Bug Predict
>
> Run bug prediction workflow. Analyzes code patterns and predicts potential bugs before they occur.
>
> **Group:** workflow
>
> ## Parameters
>
> | Parameter | Type | Description | Constraints | Default |
> | --------- | ---- | ----------- | ----------- | ------- |
> | `path` | string | Path to directory or file to analyze |  | required |
>
> ## Usage
>
> `bug_predict(path="...")`
>
> ## Related Topics
> - **Reference**: Tool: Security Audit — Run security audit workflow on codebase. Detects vulnerabili...
> - **Reference**: Tool: Code Review — Run code review workflow. Provides comprehensive code qualit...
> - **Reference**: Tool: Test Generation — Generate tests for code. Can batch generate tests for multip...
> </passage>
>
> <passage>
> [P3] source: concepts/task-error-handling-design.md
> ---
> type: concept
> name: task-error-handling-design
> tags: [error-handling, python, patterns]
> source: developer-guidance
> ---
>
> # Concept: Error handling design
>
> ## What
>
> Error handling design is the practice of deciding, for
> each failure point in your code, whether to catch an
> exception, propagate it, or transform it. Good error
> handling makes failures visible, debuggable, and
> recoverable. Bad error handling hides bugs.
>
> ## Why
>
> Swallowed exceptions are the most expensive kind of bug.
> They produce no stack trace, no log entry, and no test
> failure. The symptom shows up hours or days later as
> corrupted state, silent data loss, or a user report that
> "nothing happened." The cost of adding a proper handler
> is minutes; the cost of debugging a swallowed error is
> hours.
>
> ## The five strategies
>
> | Strategy | When to use | Example | Anti-pattern |
> |---|---|---|---|
> | Catch and handle | You know how to recover | `except FileNotFoundError: return default` | Catching broadly and returning `None` |
> | Catch and re-raise | You need to add context | `except ValueError as e: raise ConfigError(...) from e` | Re-raising without `from e` (loses chain) |
> | Catch, log, re-raise | You need an audit trail | `except IOError as e: logger.error(...); raise` | Logging but not re-raising (swallows error) |
> | Broad catch with justification | Failure is acceptable | `except Exception: # noqa: BLE001` with `# INTENTIONAL:` | Broad catch without a comment explaining why |
> | Cleanup / finally | Resources must be released | `finally: conn.close()` | Putting cleanup in the `except` branch only |
>
> ## Key principles
>
> - **Catch specific, propagate general.** If you do not
>   know how to handle `Exception`, let it propagate. The
>   caller might know.
> - **Log before you handle.** Once you catch and return a
>   fallback, the original error is gone unless you logged
>   it first.
> - **Chain with `from e`.** `raise NewError(...) from e`
>   preserves the original traceback. Without it, the cause
>   is lost.
> - **Never use bare `except:`.** It catches
>   `KeyboardInterrupt`, `SystemExit`, and `GeneratorExit`,
>   which should almost never be intercepted.
> - **Exception hierarchies guide granularity.** Catch
>   `ConnectionError` rather than `Exception` when you only
>   want to handle network failures.
>
> ## The cost of swallowing errors
>
> ```python
> # This silently eats every failure including bugs
> try:
>     deploy_to_production()
> except Exception:
>     pass  # "it works on my machine"
> ```
>
> When `deploy_to_production()` raises `TypeError` due to a
> code bug, this pattern hides it. The deployment appears to
> succeed. The bug surfaces in production hours later.
>
> ## Want to learn more?
>
> - "How do I design error handling for a new module?" --
>   see the **task** template for a step-by-step guide
> - "Show me all the patterns with code examples" -- see
>   the **reference** template for a full pattern catalog
> - Run `/code-quality` on your code to detect broad
>   exception catches automatically
> - Run `/security` to find error handling that masks
>   security-relevant failures
>
> ## Related Topics
>
> - **Task**: Error handling design -- step-by-step guide
>   for adding error handling to a module
> - **Reference**: Error handling design -- full pattern
>   catalog with code examples
> - **Quickstart**: Error handling design -- 5-step minimal
>   guide for adding error handling to a function
> </passage>

### Answer

## Finding Potential Runtime Errors Before Deployment

The best tool for this is **Bug Prediction**, which analyzes your codebase to identify patterns likely to cause production incidents before they occur [P1].

---

### How to Run It

```python
bug_predict(path="...")
```

Point it at the directory or file you want to analyze [P2].

---

### What It Detects

The scanner looks for three categories of runtime risk [P1]:

| Pattern | Severity | What It Means |
|---|---|---|
| **dangerous_eval** | HIGH | `eval()`, `exec()`, `compile()` on input — code injection vectors |
| **broad_exception** | MEDIUM | Bare `except:` or unlogged `except Exception:` — errors silently swallowed |
| **incomplete_code** | LOW | TODO/FIXME/HACK/XXX comments — unfinished paths that break under edge cases |

Beyond these patterns, the scanner also weighs **cyclomatic complexity**, **change frequency** (frequently modified files are more likely to have regressions), and **code smells** like functions over 50 lines [P1].

---

### Why Broad Exception Catches Are Especially Dangerous

Swallowed exceptions are particularly costly — they produce no stack trace, no log entry, and no test failure, so the symptom may appear hours or days later as corrupted state or silent data loss [P3]. The bug prediction scanner specifically flags these as **MEDIUM severity** risks [P1].

---

### Smart Filtering

Not every match is a real bug. The scanner suppresses known-safe patterns such as `eval()` inside test fixture strings, JavaScript's `regex.exec()` method calls, and broad exceptions explicitly marked with `# INTENTIONAL:` comments or `# noqa: BLE001` [P1].

---

### When to Use It

This is especially valuable [P1]:
- **Before merging a large PR** — catch patterns humans miss
- **Before a release** — verify no new high-severity patterns crept in
- **After onboarding unfamiliar code** — map risk hotspots quickly


### Judge verdicts (off → on)

| | Score | Supported | Unsupported |
| - | -: | -: | -: |
| Thinking off | 1.000 | 21 | 0 |
| Thinking on  | 1.000 | 23 | 0 |

**Reasoning (thinking off):**

> Every factual claim in the answer maps directly to an explicit statement in the retrieved passages. The tool invocation syntax, all three risk pattern categories (with their severities and descriptions), the contextual risk factors, the false-positive suppression list, the cost-of-swallowed-exceptions description, and the recommended use cases are all word-for-word or near-verbatim from P1, P2, and P3. No details appear to have been invented or inferred beyond what the passages state.

**Reasoning (thinking on):**

> Every atomic factual claim in the answer maps directly to at least one retrieved passage. The tool invocation syntax and path parameter come from P2. All three risk pattern categories (dangerous_eval, broad_exception, incomplete_code) with their severities and descriptions come from P1. The contextual risk factors (cyclomatic complexity, hot files, code smells including 50-line functions and duplicated logic) come from P1. The false-positive suppression list (test fixture evals, JS regex.exec, INTENTIONAL/noqa markers) comes from P1. The "before a release" use case comes from P1. The claims about swallowed exceptions being costly, the specific symptoms (corrupted state, silent data loss, "nothing happened"), and the deploy_to_production code example with TypeError all come verbatim from P3. No hallucinated details, invented flags, or outside-knowledge drift was detected.

**Claims identified by judge (thinking off):**

Supported:
  - Bug Prediction analyzes your codebase to identify patterns likely to cause production incidents before they occur
  - The tool is invoked with bug_predict(path='...')
  - The path parameter points to the directory or file to analyze
  - The scanner detects a pattern called dangerous_eval with HIGH severity
  - dangerous_eval covers eval(), exec(), compile() on input — code injection vectors
  - The scanner detects a pattern called broad_exception with MEDIUM severity
  - broad_exception covers bare except: or unlogged except Exception: — errors silently swallowed
  - The scanner detects a pattern called incomplete_code with LOW severity
  - incomplete_code covers TODO/FIXME/HACK/XXX comments — unfinished paths that break under edge cases
  - The scanner weighs cyclomatic complexity as a risk factor
  - The scanner weighs change frequency (frequently modified files are more likely to have regressions)
  - The scanner weighs code smells like functions over 50 lines
  - Swallowed exceptions produce no stack trace, no log entry, and no test failure
  - The symptom of swallowed exceptions may appear hours or days later as corrupted state or silent data loss
  - The scanner suppresses eval() inside test fixture strings as a known-safe pattern
  - The scanner suppresses JavaScript's regex.exec() method calls as a known-safe pattern
  - The scanner suppresses broad exceptions explicitly marked with # INTENTIONAL: comments
  - The scanner suppresses broad exceptions marked with # noqa: BLE001
  - Bug Prediction is especially valuable before merging a large PR — to catch patterns humans miss
  - Bug Prediction is especially valuable before a release — to verify no new high-severity patterns crept in
  - Bug Prediction is especially valuable after onboarding unfamiliar code — to map risk hotspots quickly

Unsupported:


**Claims identified by judge (thinking on):**

Supported:
  - The Bug Prediction tool is designed for catching risky patterns before they reach production.
  - The syntax to run it is `bug_predict(path='...')`.
  - The path parameter points to a directory or file to analyze.
  - The scanner looks for three categories of risk.
  - The `dangerous_eval` pattern has HIGH severity.
  - `dangerous_eval` catches `eval()`, `exec()`, `compile()` on any input — code injection vectors.
  - The `broad_exception` pattern has MEDIUM severity.
  - `broad_exception` catches bare `except:` or unlogged `except Exception:` — errors silently swallowed.
  - The `incomplete_code` pattern has LOW severity.
  - `incomplete_code` catches TODO, FIXME, HACK, XXX comments — unfinished paths that break under edge cases.
  - The scanner also weighs cyclomatic complexity as a risk factor.
  - The scanner also weighs change frequency (hot files) as a risk factor.
  - The scanner also weighs code smells like functions over 50 lines as a risk factor.
  - The scanner also weighs duplicated logic across modules as a risk factor.
  - Swallowed exceptions produce no stack trace, no log entry, and no test failure.
  - The symptom of a swallowed exception can appear hours or days later as corrupted state, silent data loss, or a user report that 'nothing happened'.
  - The code example with `deploy_to_production()` under a broad `except Exception: pass` is sourced from P3.
  - When `deploy_to_production()` raises a `TypeError`, the broad exception pattern hides it and the deployment appears to succeed, with the bug surfacing in production later.
  - The scanner suppresses `eval()` inside test fixture strings as a known-safe pattern.
  - The scanner suppresses JavaScript `regex.exec()` calls as a known-safe pattern.
  - The scanner suppresses broad exceptions marked with `# INTENTIONAL:` comments as a known-safe pattern.
  - The scanner suppresses broad exceptions marked with `# noqa: BLE001` as a known-safe pattern.
  - Running bug prediction before a release is one of its primary use cases — specifically to verify no new high-severity patterns crept in.

Unsupported:


### Your labels

```yaml
id: gq-040
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
