---
type: error
name: dashboard-error
feature: dashboard
depth: error
generated_at: 2026-06-10T06:08:44.496038+00:00
source_hash: 48be0a4fd811c784bc44e073b2ac5906c205487b317ef813d32ca7c5e3b936cc
status: generated
---

# Dashboard errors

## Common error signatures

Dashboard failures fall into three categories, one per stage of the refresh → render → show pipeline:

- **Snapshot errors** — `build_snapshot()` returns a partial result (not an exception) when `queries.yaml` is missing. The returned `dict` still contains an error field. If you pass that partial snapshot downstream without checking it, `render()` or `display()` may produce incomplete or malformed output.

- **Render errors** — `render()` writes an HTML file to the path you supply as `out`. Failures here are typically `OSError` or `FileNotFoundError` when the destination directory does not exist, or a `KeyError` when the snapshot is missing a field that the HTML template expects (for example, if the sentinel values `__ATTUNE_SNAPSHOT__` or `__ATTUNE_TITLE__` were not substituted correctly).

- **Display errors** — `display()` uses Rich to pretty-print the snapshot to the terminal. Errors here usually mean the snapshot `dict` is malformed or empty, which traces back to a failed `build_snapshot()` call.

## Where errors originate

| Function | Module | What can go wrong |
|---|---|---|
| `build_snapshot(corpus_package, queries_path)` | `dashboard.refresh` | Returns a partial snapshot with an error field when `queries.yaml` is missing; does not raise. |
| `render(out, snapshot, title)` | `dashboard.render` | Raises on filesystem errors writing `out`, or if sentinel substitution fails. |
| `display(snapshot, console)` | `dashboard.show` | Raises if the snapshot is malformed or missing expected keys. |
| `main(corpus_package)` in `dashboard.refresh` | `dashboard.refresh` | Calls `build_snapshot()` and returns `0` on success; failures propagate from the stages above. |
| `main(corpus_package)` in `dashboard.show` | `dashboard.show` | Calls `display()` on a freshly built snapshot; failures propagate from `build_snapshot()` or `display()`. |

## How to diagnose

1. **Check whether `build_snapshot()` returned a partial snapshot.** The function does not raise when `queries.yaml` is missing — it returns a `dict` that contains an error field. Inspect the snapshot before passing it to `render()` or `display()`. A partial snapshot passed silently downstream is the most common source of confusing output.

2. **Confirm `queries_path` resolves to an existing file.** If you pass a custom `queries_path` to `build_snapshot()`, verify that the path exists and is readable before calling the function. A `None` value tells the function to locate `queries.yaml` inside the `corpus_package` package.

3. **Check that the `out` directory exists before calling `render()`.** `render()` does not create intermediate directories. An `OSError` or `FileNotFoundError` from `render()` almost always means the parent of `out` does not exist.

4. **Verify the snapshot is complete before calling `display()`.** If `display()` raises or produces garbled output, print the snapshot `dict` directly first. An empty or partial snapshot points back to step 1.

5. **Look for unsubstituted sentinels in the HTML output.** If the rendered HTML file contains the literal strings `__ATTUNE_SNAPSHOT__` or `__ATTUNE_TITLE__`, the template substitution step inside `render()` did not run correctly. Check that you are passing a non-empty snapshot and a non-empty `title`.

## Source files

- `src/attune_rag/dashboard/__init__.py`
- `src/attune_rag/dashboard/refresh.py`
- `src/attune_rag/dashboard/render.py`
- `src/attune_rag/dashboard/show.py`

**Tags:** `dashboard`, `living-docs`, `html`, `terminal`, `snapshot`, `freshness`
