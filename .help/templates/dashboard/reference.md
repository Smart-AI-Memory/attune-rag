---
type: reference
name: dashboard-reference
feature: dashboard
depth: reference
generated_at: 2026-05-20T03:33:38.795023+00:00
source_hash: 48be0a4fd811c784bc44e073b2ac5906c205487b317ef813d32ca7c5e3b936cc
status: generated
---

# Dashboard reference

Use the dashboard API to run a three-stage living-docs pipeline against a registered corpus: **refresh** benchmarks the corpus and produces a snapshot dict; **render** packages that snapshot into a standalone HTML report; **show** pretty-prints the snapshot to the terminal via Rich. Each stage exposes its own CLI entry point under `attune-rag dashboard`.

## Functions

| Function | Parameters | Returns | Description |
|----------|------------|---------|-------------|
| `build_snapshot` | `corpus_package: str = 'attune_help'`, `queries_path: Path \| None = None` | `dict[str, Any]` | Return a dashboard snapshot dict. On missing queries.yaml returns partial with error. |
| `render` | `out: Path`, `snapshot: dict[str, Any]`, `title: str = 'attune-rag dashboard'` | `Path` | Render the dashboard template to *out* with *snapshot* embedded as JSON. |
| `display` | `snapshot: dict[str, Any]`, `console: Console \| None = None` | `None` | Print the snapshot to the terminal using Rich. |
| `main` (refresh) | `corpus_package: str = 'attune_help'` | `int` | CLI entry point for the refresh stage; returns `0` on success. |
| `main` (show) | `corpus_package: str = 'attune_help'` | `int` | CLI entry point for the show stage; returns `0` on success. |

## Constants

| Constant | Type | Value | Description |
|----------|------|-------|-------------|
| `_SENTINEL_SNAPSHOT` | `str` | `'__ATTUNE_SNAPSHOT__'` | Placeholder replaced with the embedded snapshot JSON during rendering. |
| `_SENTINEL_TITLE` | `str` | `'__ATTUNE_TITLE__'` | Placeholder replaced with the dashboard title string during rendering. |
| `_SYSTEM_DIRS` | `frozenset` | `{'/etc', '/sys', '/proc', '/dev', '/boot', '/sbin', '/bin', '/usr/bin'}` | Directories excluded from corpus path resolution. |

## Source files

- `src/attune_rag/dashboard/__init__.py`
- `src/attune_rag/dashboard/refresh.py`
- `src/attune_rag/dashboard/render.py`
- `src/attune_rag/dashboard/show.py`

## Tags

`dashboard`, `living-docs`, `html`, `terminal`, `snapshot`, `freshness`
