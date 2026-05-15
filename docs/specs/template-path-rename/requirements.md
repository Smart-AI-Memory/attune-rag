# Spec: Template Path Rename

> Phase 1 (requirements) for the `template_path` rename refactor kind.
> Carried forward from the template-editor spec
> (`specs/template-editor/`) which deferred this to a follow-up.

---

## Phase 1: Requirements

**Status**: complete

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

**Status**: approved (backend only — gui-side deferred)

### Design answers

**Q1 — Canonical path-keyed sidecars.** Only two:

- `summaries.json` (used by `DirectoryCorpus`)
- `summaries_by_path.json` (the attune-help variant of the same)

Both have flat `rel_path → summary` shape. `cross_links.json` is
out per the scope section (template-name keyed in the spec; the
attune-help variant uses a nested layout that isn't a flat
path-keyed map). No `*-index.json` files written by
attune-author exist in this repo — that's an attune-author
concern, out of scope.

**Q2 — WS active-session handling.** Cross-repo (attune-gui).
The backend emits the new path in the apply response; the gui
decides whether to close-and-redirect or rebind in-place.
Tracked separately.

**Q3 — "Rename file…" trigger location.** Cross-repo (attune-gui)
UX call. No backend implication.

**Q4 — Empty target dir vs. empty target file.** Empty target
**file** only. Target directory may exist; missing intermediate
directories are created on apply and tracked for rollback (see
`_ensure_parents` / `_undo_created_dirs` in
`src/attune_rag/editor/_rename.py`).

### RenamePlan encoding

`RenamePlan` grows a new field `moves: list[FileMove]` where
`FileMove(old_path, new_path)`. Text edits (the sidecar key
rename) live in the existing `edits: list[FileEdit]` field.
Apply applies moves first, then edits, with rollback on any
mid-flight failure.

---

## Phase 3: Tasks

**Status**: complete (in PR linked from CHANGELOG `[Unreleased]`)

1. ✅ `FileMove` dataclass + `RenamePlan.moves` field with
   `to_dict()` updates.
2. ✅ `_plan_template_path_rename` — validates corpus-root
   containment, missing-source, target collision; emits
   `FileMove` + sidecar `FileEdit`s when present.
3. ✅ Extended `apply_rename` — moves first with `mkdir -p`
   + rollback for created dirs; edits second with the existing
   tempfile + drift-check flow. Failures at either phase reverse
   prior work.
4. ✅ Tests in `tests/unit/test_editor_rename.py`:
   `test_template_path_rename_moves_file_and_emits_move`,
   `_into_new_subdir`, `_rejects_existing_target`,
   `_to_same_path_is_noop`, `_rejects_escape`,
   `_rejects_missing_source`, `_updates_summaries_sidecar`,
   `_no_sidecar_is_not_an_error`,
   `_rolls_back_when_target_appears_late`,
   `_rolls_back_move_when_edit_drifts`.
5. ✅ `FileMove` exported from `attune_rag.editor`.

---

## Phase 4: Implementation

**Status**: shipped (backend). attune-gui follow-up tracked
separately for the "Rename file…" trigger + WS path-change
handling.
