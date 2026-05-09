# Spec: Template Path Rename

> Phase 1 (requirements) for the `template_path` rename refactor kind.
> Carried forward from the template-editor spec
> (`specs/template-editor/`) which deferred this to a follow-up.

---

## Phase 1: Requirements

**Status**: draft

### Problem statement

`attune_rag.editor.plan_rename` currently raises `NotImplementedError`
for `kind="template_path"`. The template-editor M4 rename refactor
modal supports the alias and tag kinds end-to-end, but moving a
template file (changing its rel-path within a corpus) has no path
through the system.

That gap matters because:

- Template names drift over time. A file named
  `tasks/use-attune-hub.md` might want to become `tasks/attune-hub.md`
  when the project's verb conventions change.
- Reorganizing a corpus (moving `concepts/foo.md` →
  `references/foo.md`) is currently a manual `mv` followed by a
  manual sweep of every place that path is referenced.
- Cross-corpus consistency: if `summaries.json` is path-keyed (it is)
  and the dashboard's living-docs index references the path, those
  break silently when a file is renamed.

### Scope

**In scope:**

- Renaming a single template's rel-path within a registered corpus.
- Atomic file move (tempfile + rename + rollback on partial failure).
- Updating path-keyed indexes inside the corpus root: `summaries.json`,
  any `*-index.json` files written by attune-author.
- Refreshing `Corpus.path_index` and `Corpus.alias_index` after the move.
- Returning a `RenamePlan` with `FileEdit` entries for every
  index/sidecar file the rename will touch (so the editor can preview
  the multi-file diff before the user clicks Apply).
- Surfacing a "Rename file…" entry in the editor's command palette
  (or top-bar menu) — distinct trigger from the alias/tag chip context
  menu.

**Out of scope (defer to a separate spec):**

- Cross-corpus moves (the source path is in corpus A, the target in
  corpus B). The new path must stay inside the same registered
  corpus for v1.
- `cross_links.json` updates — current cross_links are template-name
  keyed, not path-keyed, so a path-only rename leaves them untouched.
  If the rename also changes the name (the canonical case where the
  filename = the alias), that's an alias rename composed with a path
  rename — out of scope; users can do them sequentially.
- Updating attune-help's static help index, the staleness pipeline's
  feature manifests, or any external tools that hard-code template
  paths. The plan returns affected paths; the user reruns whatever
  external pipeline depends on them.
- Git history preservation — a rename is a `move`, but the corpus is
  user code, not git plumbing.

### User stories

1. *As a corpus author*, I want to rename `tasks/use-attune-hub.md` →
   `tasks/attune-hub.md` from the template editor and have every
   path-keyed reference inside the corpus update atomically.
2. *As a corpus author*, I want a multi-file preview before the
   rename happens so I can see exactly which files change.
3. *As a corpus author*, if the new path collides with an existing
   file, I want a clear error (not a silent overwrite).
4. *As a corpus author*, if the rename fails halfway through (disk
   error, permission), I want the corpus to roll back to its
   pre-rename state.

### Edge cases & open questions

| Question / Edge case | Resolution |
|----------------------|------------|
| New path is inside a directory that doesn't exist | Create the directory as part of apply; rollback removes it on failure. |
| New path equals old path | No-op plan, return empty `edits[]`. (Mirrors alias/tag behavior.) |
| New path collides with an existing template | `RenameCollisionError(new_path, owning_path=new_path)`. The editor surfaces it as a banner, same as the alias case. |
| The template currently being edited is the one being moved | After apply, the editor refreshes its `(corpus, path)` to the new path and continues. WS subscribers on the old path get a `file_changed` once + a final close so the tab can reopen on the new path. |
| `summaries.json` exists in the corpus root | Update the entry; preserve any other keys. Returns one `FileEdit` for `summaries.json`. |
| `summaries.json` doesn't exist | Skip; not an error. |
| New path tries to escape the corpus root (`../...`) | Reject with `ValueError`; surfaced as 400 in the editor route. |

### Affected layers

- [x] attune-rag (backend) — implement `_plan_template_path_rename` + `_apply_template_path_rename`
- [x] attune-gui (frontend) — new "Rename file…" trigger; rename modal already supports `kind: "template_path"` API-wise
- [ ] attune-help (mobile/docs) — none
- [ ] attune-author (authoring/infra) — none

---

## Phase 2: Design

> Stubbed. Open the design phase before implementation. Key questions
> to answer:
>
> 1. What is the canonical list of path-keyed sidecar files inside a
>    corpus root? (`summaries.json` confirmed; check for others.)
> 2. How does the editor's WS infrastructure handle the path change
>    for the active session — does it close-and-redirect, or does it
>    rebind in-place?
> 3. Is the "Rename file…" trigger on the chip context menu (where
>    alias/tag rename lives) or in a new command palette entry?
> 4. Should `apply_rename` for `template_path` require an empty
>    target directory, or only an empty target *file*?

---

## Phase 3: Tasks

> Stubbed; fill out after Phase 2 is approved.

---

## Phase 4: Implementation

> Not started.
