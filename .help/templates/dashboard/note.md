---
type: note
name: dashboard-note
feature: dashboard
depth: note
generated_at: 2026-06-10T06:08:44.515853+00:00
source_hash: 48be0a4fd811c784bc44e073b2ac5906c205487b317ef813d32ca7c5e3b936cc
status: generated
---

# Note: dashboard

## Context

The dashboard feature implements a three-stage pipeline for producing a "living-docs" health report against a registered corpus: **refresh → render → show**. Each stage is a separate module with its own entry point, and the stages are composable — you can run them individually or chain their outputs.

| Stage | Module | Key function | Output |
|---|---|---|---|
| refresh | `dashboard.refresh` | `build_snapshot()` | snapshot `dict` |
| render | `dashboard.render` | `render()` | HTML file at `out` |
| show | `dashboard.show` | `display()` | Rich terminal output |

## Design

The dashboard modules expose top-level functions rather than classes. You call `build_snapshot()`, `render()`, and `display()` directly without instantiating anything first.

`build_snapshot(corpus_package, queries_path)` returns a `dict[str, Any]` snapshot. If `queries.yaml` is missing, it returns a partial snapshot that includes an error entry rather than raising — downstream stages can still consume it.

`render(out, snapshot, title)` writes a self-contained HTML report to the path you supply as `out`. The snapshot is embedded as JSON using the sentinels `_SENTINEL_SNAPSHOT` (`'__ATTUNE_SNAPSHOT__'`) and `_SENTINEL_TITLE` (`'__ATTUNE_TITLE__'`) as replacement markers in the HTML template.

`display(snapshot, console)` pretty-prints the snapshot to the terminal via [Rich](https://github.com/Textualize/rich). Passing `None` for `console` causes it to construct a default `Console` internally.

Both `dashboard.refresh` and `dashboard.show` expose a `main(corpus_package)` entry point that returns `0` on success, which is the value wired to the CLI.

## Source files

- `src/attune_rag/dashboard/refresh.py`
- `src/attune_rag/dashboard/render.py`
- `src/attune_rag/dashboard/show.py`

**Tags:** `dashboard`, `living-docs`, `html`, `terminal`, `snapshot`, `freshness`
