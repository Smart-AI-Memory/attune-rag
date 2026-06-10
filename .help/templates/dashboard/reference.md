---
type: reference
name: dashboard-reference
feature: dashboard
depth: reference
generated_at: 2026-06-10T06:08:44.491483+00:00
source_hash: 48be0a4fd811c784bc44e073b2ac5906c205487b317ef813d32ca7c5e3b936cc
status: generated
---

# Dashboard reference

Build and display a living-docs dashboard for a registered corpus through a three-stage pipeline: **refresh** runs the benchmark against the corpus and emits a snapshot dict; **render** writes a self-contained HTML report with the snapshot embedded as JSON; **show** pretty-prints the snapshot to the terminal via Rich. Each stage exposes its own entry point.

## Functions

| Function | Parameters | Returns | Description |
|----------|------------|---------|-------------|
| `build_snapshot` | `corpus_package: str = 'attune_help'`, `queries_path: Path \| None = None` | `dict[str, Any]` | Return a dashboard snapshot dict. On missing queries.yaml returns partial with error. |
| `main` | `corpus_package: str = 'attune_help'` | `int` | Entry point for the refresh stage; returns `0` on success. |
| `render` | `out: Path`, `snapshot: dict[str, Any]`, `title: str = 'attune-rag dashboard'` | `Path` | Render the dashboard template to `out` with `snapshot` embedded as JSON. |
| `display` | `snapshot: dict[str, Any]`, `console: Console \| None = None` | `None` | Pretty-print a snapshot dict to the terminal using Rich. |
| `main` | `corpus_package: str = 'attune_help'` | `int` | Entry point for the show stage; returns `0` on success. |

## Constants

| Constant | Type | Value | Description |
|----------|------|-------|-------------|
| `_SENTINEL_SNAPSHOT` | `str` | `'__ATTUNE_SNAPSHOT__'` | Placeholder string replaced with the serialized snapshot when rendering the HTML template. |
| `_SENTINEL_TITLE` | `str` | `'__ATTUNE_TITLE__'` | Placeholder string replaced with the dashboard title when rendering the HTML template. |

## Source files

- `src/attune_rag/dashboard/__init__.py`
- `src/attune_rag/dashboard/refresh.py`
- `src/attune_rag/dashboard/render.py`
- `src/attune_rag/dashboard/show.py`

## Tags

`dashboard`, `living-docs`, `html`, `terminal`, `snapshot`, `freshness`
