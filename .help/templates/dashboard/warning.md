---
type: warning
name: dashboard-warning
feature: dashboard
depth: warning
generated_at: 2026-06-10T06:08:44.502587+00:00
source_hash: 48be0a4fd811c784bc44e073b2ac5906c205487b317ef813d32ca7c5e3b936cc
status: generated
---

# Dashboard Cautions

The dashboard feature is a three-stage pipeline: `build_snapshot` collects corpus data into a dict, `render` writes an HTML file with that dict embedded as JSON, and `display` pretty-prints the snapshot to the terminal. Each stage is independent, which means a stale or partial snapshot can flow silently into the later stages without raising an error.

## Risk areas

### `build_snapshot` returns a partial result on missing `queries.yaml`

When `queries_path` is `None` and the default `queries.yaml` cannot be found, `build_snapshot` does **not** raise an exception — it returns a partial snapshot dict that contains an error key. If you pass that partial result directly to `render` or `display`, you get a dashboard that appears complete but is missing benchmark data. Always inspect the returned dict for an error entry before proceeding to the next stage.

```python
snapshot = build_snapshot(corpus_package='attune_help')
if 'error' in snapshot:
    # handle missing queries.yaml before calling render or display
    ...
```

### `render` embeds the snapshot by replacing a sentinel string

`render(out, snapshot, title)` writes the snapshot into the HTML template by substituting the literal string `__ATTUNE_SNAPSHOT__` and the title sentinel `__ATTUNE_TITLE__`. If you supply a `snapshot` dict that serializes to JSON containing either of those strings, the substitution will corrupt the output silently. Avoid snapshot values that reproduce those sentinel strings.

### Passing `None` as `console` to `display` is safe, but skips your custom output target

`display(snapshot, console=None)` creates its own `Console` instance when `console` is `None`. If you are capturing terminal output in tests or redirecting to a file, pass your own `Console` explicitly — the default instance writes directly to `stdout` and ignores any redirection you have set up outside of Rich.

### `corpus_package` defaults propagate across all three stages

Both `build_snapshot` and the `main` entry points default `corpus_package` to `'attune_help'`. If you are working with a different corpus, you must pass the correct package name at every stage. Forgetting to update it in one call causes the snapshot to reflect the wrong corpus while the rest of your pipeline continues normally.

## How to avoid problems

- **Check for a partial snapshot before rendering.** Treat the presence of an error key in the `build_snapshot` return value as a hard stop, not a warning to log and continue.
- **Validate `queries_path` early.** If your corpus lives outside the default location, pass an explicit `Path` to `build_snapshot` rather than relying on the default discovery logic.
- **Pass a `Console` instance in tests.** When writing tests against `display`, construct a `Console` with `record=True` so that Rich output is captured rather than written to `stdout`.
- **Keep the sentinel strings out of snapshot data.** If your corpus metadata could contain arbitrary strings, sanitize values before they reach `render`.

## Source files

- `src/attune_rag/dashboard/__init__.py`
- `src/attune_rag/dashboard/refresh.py`
- `src/attune_rag/dashboard/render.py`
- `src/attune_rag/dashboard/show.py`

**Tags:** `dashboard`, `living-docs`, `html`, `terminal`, `snapshot`, `freshness`
