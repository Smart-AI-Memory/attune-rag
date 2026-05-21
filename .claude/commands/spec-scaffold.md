---
name: spec-scaffold
description: "Stamp out a scaffolding-only spec directory (README + requirements + design + tasks) following attune-rag's scoping-spec convention. Lower-level than /spec — /spec scopes; /spec-scaffold creates the empty shape."
argument-hint: "<spec-slug> [shape-parent-slug]"
---

Scaffold a new spec directory at `docs/specs/$ARGUMENTS` for attune-rag.

This is **scaffolding only** — the four files are stamped with status
banners that mark them non-executable until a separate `/spec` scoping
pass approves them. Used for upcoming spec dirs that should exist (so
future phases / parallel sessions don't duplicate work scaffolding
them) but cannot be scoped yet because their entry gates haven't
fired.

## Arguments

- `<spec-slug>` — kebab-case dir name under `docs/specs/`, e.g.
  `v1.0.0-release`, `api-v0.2.0-cut`.
- `[shape-parent-slug]` — optional; another spec dir under
  `docs/specs/` whose `README.md` / `requirements.md` / `design.md` /
  `tasks.md` layout this one should mirror. Default:
  `api-v0.2-public-surface`.

## Behavior

1. **Confirm intent first.** Use `AskUserQuestion` to confirm:
   - What does this spec produce? (one sentence)
   - What upstream phase / spec / PR / external event must fire
     before this can be scoped? (the activation gate)
   - What's the shape parent? (default `api-v0.2-public-surface`)
   - Should the cut happen as `0.x.y` or `x.y.0` or `x.0.0`?
     (informs the "what this means" framing)
2. **Check the dir doesn't already exist.** If
   `docs/specs/<spec-slug>/` exists, stop and tell the user — they
   may already be looking at the scaffold they want.
3. **Create the four files** with the convention below.
4. **Run `scripts/check_freeze.py`** to confirm it's freeze-clean.
   This should always pass — the scaffold is pure docs addition.
5. **Open a PR** titled
   `docs(<spec-slug>): scaffold [phase|successor] spec` with the
   activation gate quoted in the body.

## File convention

Every file starts with a one-line status banner at the top, *before*
the H1, written as a markdown blockquote:

```markdown
> **Status: scaffolding — not yet scoped; activates after
> <activation-gate>.**
```

### README.md

One-pager. Sections:

- Status banner (above).
- Frontmatter list: Owner, Created (today's date, absolute), Target
  version, Roadmap phase (if applicable), Shape parent.
- **Purpose** — one paragraph: what this spec produces and why.
- **What this spec is not** — short bulleted list of explicit
  non-goals. Catches downstream readers who confuse this spec for an
  adjacent one.
- **Inherited entry-gates** — checkbox list of conditions that must
  be true before the `/spec` scoping pass can run. Each gate names
  the upstream owner.
- **Files** table listing the four files + one-line purpose each.
- **See also** list of related specs, cross-linked with relative
  paths.

### requirements.md

Status banner. Then `## Phase 1: Requirements`. Sections:

- **Problem statement** — one paragraph.
- **Entry gates (inherited)** — reproduce the list from
  `README.md`. Yes, duplicated; the requirements file should be
  self-contained for the eventual scoping pass.
- **Scope** with sub-sections "In scope" and "Out of scope
  (Non-Goals)". Lead with the actual deliverables; list non-goals
  explicitly because the scaffold's job is to stake out boundaries.
- **User stories** — minimum 2, written from the perspective of the
  downstream consumer (not the implementer).
- **Edge cases & open questions** — table with `Question / Edge
  case` and `Placeholder resolution` columns. The placeholders are
  the things the scoping pass will pin down.
- **Affected layers** — checkbox list (attune-rag, attune-gui,
  attune-help, attune-author).

### design.md

Status banner. Then `## Phase 2: Design`. Sections:

- **What <target> means vs. <predecessor>** — a comparison table.
  Three-column: dimension, predecessor value, target value.
- **Mechanism / approach** sections — one per major design axis.
  Sketches not specs. Use blockquoted markdown for "this is the
  shape, not the final text" code/policy fragments.
- **What this design intentionally does *not* cover** — explicit
  non-mechanism list. Same role as requirements' "non-goals" but at
  the design level.

### tasks.md

Status banner (with explicit "not executable; promotes via `/spec`
scoping pass" framing). Then `## Phase 3: Tasks`. Sections:

- **Implementation order** — a small ASCII / mermaid arrow diagram.
- **Tasks** table — milestones grouped (M1, M2, ...), each row:
  `# | Task | Layer | Notes`. **No `Status` column** — this is
  scaffolding, nothing has run.
- **Dependencies** — explicit dependency list (text, not table).
- **Definition of done (placeholder)** — checkbox list. Every item
  is pinned during scoping, but the *shape* is visible.
- **Risks & mitigations (placeholder)** — two-column table.
- **Out of scope (deferred)** — cross-reference to requirements.md.
- **Follow-ups (post-cut)** — bulleted list of post-cut backlog
  candidates.

## Rules (load-bearing — do not skip)

- **Status banner on every file.** A reader landing on tasks.md or
  design.md without the README should immediately see this is
  scaffolding.
- **tasks.md must be marked non-executable.** Add "not executable;
  promotes via `/spec` scoping pass" to its banner explicitly.
- **Entry gates are reproduced in both README.md and
  requirements.md.** Duplication is intentional — the requirements
  file is the artifact the scoping pass works from.
- **No source / test / `CHANGELOG.md` changes.** Pure docs addition.
  `scripts/check_freeze.py` should pass trivially.
- **Cross-link related specs liberally.** Future readers (and
  parallel sessions) navigate by the link graph.
- **Use absolute dates, not relative.** `2026-05-21`, not
  "tomorrow" / "next week".
- **Lead each file with a frontmatter list** (Owner, Created, etc.)
  before the first paragraph. Makes the file self-contained when
  read out of context.

## Shape parent

`docs/specs/api-v0.2-public-surface/` is the canonical reference for
the 4-file shape. When in doubt, read that spec's
`requirements.md` / `design.md` / `tasks.md` and mirror their
section ordering and tone.

## Verification (before opening the PR)

- `scripts/check_freeze.py --base origin/main --head HEAD` exits 0.
- `git diff --stat origin/main..HEAD` shows only files under
  `docs/specs/<spec-slug>/`.
- No `CHANGELOG.md` change. No version bump. No classifier change.
- Every file has the status banner at the top.

## PR body template

```
## Summary

Scaffolds `docs/specs/<spec-slug>/` — <one-sentence purpose>.

**Scaffolding only.** Not yet scoped; activates after
<activation-gate>.

- **Shape parent:** `docs/specs/<shape-parent>/`
- **Source of gates and outcome:** <upstream spec or roadmap section>

## Verification

- ✅ `scripts/check_freeze.py` exits 0.
- ✅ Diff: 4 files / N insertions / 0 deletions, all under
  `docs/specs/<spec-slug>/`.
- ✅ No source, tests, scripts, or CHANGELOG changes.
```

## When NOT to use this skill

- The spec is ready to be scoped *now*. Use `/spec` directly — it
  drives the brainstorm → plan → review → execute flow on a real
  scope, not a stamped-out shape.
- The work is small enough that one `tasks.md` would do without the
  full four-file shape. A small spec lives at the appropriate scale.
- The target is a successor spec for work that already shipped.
  That's a *retrospective*, not a scaffold — different shape.
