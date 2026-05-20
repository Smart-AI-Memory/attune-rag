---
type: warning
name: dashboard-warning
feature: dashboard
depth: warning
generated_at: 2026-05-20T03:33:38.804318+00:00
source_hash: 48be0a4fd811c784bc44e073b2ac5906c205487b317ef813d32ca7c5e3b936cc
status: generated
---

# Dashboard cautions

## What to watch for

The dashboard pipeline runs in three stages: **refresh** → **render** → **show**. `refresh` benchmarks the corpus and writes a snapshot JSON; `render` packages that snapshot into a self-contained HTML report; `show` pretty-prints the snapshot in the terminal via Rich. Each stage has its own CLI entry point under `attune-rag dashboard`.

The risks below are specific to how these stages interact and where each function's defaults or edge-case returns can quietly produce incomplete or misleading output.

## Risk areas

### `build_snapshot()` returns a partial result on missing `queries.yaml`

`build_snapshot()` does **not** raise an exception when `queries_path` resolves to a missing `queries.yaml`. Instead, it returns a partial snapshot dict that contains an error field. If you pass that partial snapshot to `render()` or `display()` without checking for the error field first, your HTML report or terminal output will silently reflect incomplete benchmark data.

**Mitigation:** After calling `build_snapshot()`, check the returned dict for an error key before proceeding to the render or show stages.

### `render()` embeds the snapshot using sentinel substitution

`render()` writes the snapshot into the HTML template by replacing the string literal `__ATTUNE_SNAPSHOT__` and the title placeholder `__ATTUNE_TITLE__`. If the template file is modified or regenerated and those sentinel strings are altered or removed, the rendered HTML will contain an empty or broken embedded JSON payload — with no exception raised.

**Mitigation:** If you customize the dashboard HTML template, preserve `__ATTUNE_SNAPSHOT__` and `__ATTUNE_TITLE__` exactly as they appear. Treat them as load-bearing strings, not comments.

### `display()` accepts a `None` console and silences output

`display()` accepts an optional `Console` instance. Passing `None` causes it to construct a default Rich `Console`, which writes to stdout. In test environments that capture stdout differently from Rich's output stream, this can make terminal assertions unreliable or produce no visible output at all.

**Mitigation:** In tests, pass an explicit `Console` instance (for example, one constructed with `StringIO`) so you control where output goes.

### Writing output paths inside system directories

`render()` accepts an arbitrary `out: Path` for the HTML output file. The module defines `_SYSTEM_DIRS` (`/etc`, `/sys`, `/proc`, `/dev`, `/boot`, `/sbin`, `/bin`, `/usr/bin`) as protected paths, but nothing prevents you from constructing an `out` path inside one of them and passing it directly. On systems where the process has write access, this can overwrite unintended files.

**Mitigation:** Validate `out` against `_SYSTEM_DIRS` before calling `render()`, or restrict output to a known working directory in automation contexts.

## How to avoid problems

1. **Check snapshot integrity before rendering.** Inspect the dict returned by `build_snapshot()` for an error field before passing it to `render()` or `display()`. A partial snapshot produces a valid-looking but incomplete report.

2. **Pin your template sentinels.** If you maintain a custom HTML template, add a test that asserts `__ATTUNE_SNAPSHOT__` and `__ATTUNE_TITLE__` are present in the template file. This catches accidental removal during template edits.

3. **Use explicit `Console` instances in tests.** Don't rely on `display()`'s default stdout behavior in automated tests. Pass a Rich `Console` backed by a controlled stream so output capture is deterministic.

4. **Avoid private helpers.** `_SENTINEL_SNAPSHOT`, `_SENTINEL_TITLE`, and `_SYSTEM_DIRS` are underscore-prefixed and can change without notice. Reference the public functions (`build_snapshot`, `render`, `display`) instead of building logic around internal constants.

## Source files

- `src/attune_rag/dashboard/__init__.py`
- `src/attune_rag/dashboard/refresh.py`
- `src/attune_rag/dashboard/render.py`
- `src/attune_rag/dashboard/show.py`

**Tags:** `dashboard`, `living-docs`, `html`, `terminal`, `snapshot`, `freshness`
