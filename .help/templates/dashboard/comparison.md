---
type: comparison
name: dashboard-comparison
feature: dashboard
depth: comparison
generated_at: 2026-05-20T03:33:38.817309+00:00
source_hash: 48be0a4fd811c784bc44e073b2ac5906c205487b317ef813d32ca7c5e3b936cc
status: generated
---

# Dashboard output formats: HTML report vs terminal display

## Context

The `attune-rag dashboard` pipeline runs in three stages:

1. **Refresh** — `build_snapshot()` benchmarks the corpus and returns a snapshot dict. If `queries.yaml` is missing, it returns a partial snapshot with an embedded error rather than raising.
2. **Render** — `render()` writes an HTML file to disk with the snapshot embedded as JSON, using sentinel strings (`__ATTUNE_SNAPSHOT__`, `__ATTUNE_TITLE__`) for template substitution.
3. **Show** — `display()` pretty-prints the snapshot to the terminal using Rich.

Each stage has its own CLI entry point. You always run refresh first; render and show are independent consumers of the resulting snapshot.

---

## Feature comparison

| Capability | `render()` — HTML report | `display()` — terminal |
|---|---|---|
| **Output format** | Self-contained HTML file with embedded JSON | Rich-formatted terminal output |
| **Persistent artifact** | Yes — file written to a path you specify | No — output is ephemeral |
| **Shareable with others** | Yes — send or host the HTML file | No — terminal session only |
| **Browsable interactively** | Yes — open in any browser | No — static once printed |
| **Snapshot embedded** | Yes — full JSON inside the HTML | Rendered inline; not saved separately |
| **Custom title** | Yes — `title` parameter (default: `'attune-rag dashboard'`) | No title parameter |
| **Requires a display** | No — suitable for CI/CD pipelines | Yes — depends on a terminal and Rich |
| **Entry point** | `attune-rag dashboard render` | `attune-rag dashboard show` |
| **Source file** | `dashboard/render.py` | `dashboard/show.py` |

---

## Tradeoffs in detail

### HTML report (`render`)

`render(out, snapshot, title)` writes the dashboard to `out` and returns the path. Because the snapshot is embedded as JSON, the file is fully self-contained — no network requests, no separate data file. This makes it the right artifact for CI pipelines, pull-request summaries, or anything you need to archive or share.

The tradeoff: you need a browser to read it, and generating the file takes a write to disk. It is not useful for quick, in-session inspection.

### Terminal display (`show`)

`display(snapshot, console)` renders the same snapshot data immediately in the terminal using Rich. You can pass your own `Console` instance for output redirection or testing. There is no file I/O, so it is faster for spot-checks during development.

The tradeoff: the output disappears when the session ends. It is not suitable for sharing or archiving, and it requires a terminal environment.

---

## When to use each option

**Use `render()` when you:**
- Need a persistent, shareable artifact (CI reports, code-review dashboards, archived freshness snapshots).
- Are running in a headless or non-interactive environment.
- Want to embed the snapshot in documentation or host it as a static page.

**Use `display()` when you:**
- Are doing active development and want immediate feedback without leaving the terminal.
- Already have the snapshot in memory and do not need to persist it.
- Want to pipe output through a custom Rich `Console` (for example, in tests).

**Always run `build_snapshot()` first.** Both `render` and `show` are consumers of the snapshot dict it produces. If `queries.yaml` is absent, `build_snapshot()` returns a partial snapshot with an error field — check for that field before deciding whether the downstream output is trustworthy.

---

## Source files

- `src/attune_rag/dashboard/__init__.py`
- `src/attune_rag/dashboard/refresh.py`
- `src/attune_rag/dashboard/render.py`
- `src/attune_rag/dashboard/show.py`

**Tags:** `dashboard`, `living-docs`, `html`, `terminal`, `snapshot`, `freshness`
