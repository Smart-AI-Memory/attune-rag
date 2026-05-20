---
type: error
name: dashboard-error
feature: dashboard
depth: error
generated_at: 2026-05-20T03:33:38.798843+00:00
source_hash: 48be0a4fd811c784bc44e073b2ac5906c205487b317ef813d32ca7c5e3b936cc
status: generated
---

# Dashboard errors

## Common error signatures

Dashboard errors fall into three categories, one per pipeline stage:

- **Refresh errors** — `build_snapshot()` fails to locate or parse `queries.yaml`. When the file is missing, `build_snapshot()` returns a partial snapshot dict that contains an `"error"` key rather than raising an exception. Downstream code that treats the snapshot as complete will encounter unexpected `None` values or missing keys.
- **Render errors** — `render()` cannot write the output file, or the HTML template is missing the sentinel strings `__ATTUNE_SNAPSHOT__` or `__ATTUNE_TITLE__`. A missing sentinel means the JSON snapshot is never embedded and the output HTML is silently incomplete.
- **Display errors** — `display()` receives a malformed or partial snapshot (for example, one produced by a failed refresh) and raises a `KeyError` or `TypeError` when Rich tries to format a missing field.

## Where errors originate

| Stage | Function | File |
|---|---|---|
| Refresh | `build_snapshot(corpus_package, queries_path)` | `src/attune_rag/dashboard/refresh.py` |
| Refresh | `main(corpus_package)` | `src/attune_rag/dashboard/refresh.py` |
| Render | `render(out, snapshot, title)` | `src/attune_rag/dashboard/render.py` |
| Display | `display(snapshot, console)` | `src/attune_rag/dashboard/show.py` |
| Display | `main(corpus_package)` | `src/attune_rag/dashboard/show.py` |

## How to diagnose

1. **Check whether the snapshot contains an error key.** A partial snapshot from `build_snapshot()` includes an `"error"` field when `queries.yaml` is missing or unreadable. Print or log the snapshot dict immediately after calling `build_snapshot()` and look for that key before passing the snapshot to `render()` or `display()`.

2. **Verify the output path is writable.** `render()` writes an HTML file to the `out` path you supply. If the parent directory does not exist or is in a protected location (the module explicitly guards against `/etc`, `/sys`, `/proc`, `/dev`, `/boot`, `/sbin`, `/bin`, and `/usr/bin`), you will get an `OSError`. Confirm that `out.parent` exists and that your process has write permission.

3. **Confirm the HTML template contains both sentinel strings.** `render()` replaces `__ATTUNE_SNAPSHOT__` with the JSON snapshot and `__ATTUNE_TITLE__` with the dashboard title. If a custom template is missing either string, the substitution silently produces broken HTML. Search the template file for both literals before running `render()`.

4. **Trace `KeyError` or `TypeError` in `display()` back to the snapshot.** If Rich raises during terminal rendering, the snapshot passed to `display()` is almost always the culprit — either it is partial (refresh failed) or a key was renamed. Compare the snapshot dict's keys against the fields `display()` accesses and fix the upstream source of the snapshot.

## Source files

- `src/attune_rag/dashboard/__init__.py`
- `src/attune_rag/dashboard/refresh.py`
- `src/attune_rag/dashboard/render.py`
- `src/attune_rag/dashboard/show.py`

**Tags:** `dashboard`, `living-docs`, `html`, `terminal`, `snapshot`, `freshness`
