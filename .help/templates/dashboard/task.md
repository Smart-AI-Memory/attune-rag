---
type: task
name: dashboard-task
feature: dashboard
depth: task
generated_at: 2026-05-20T03:33:38.790777+00:00
source_hash: 48be0a4fd811c784bc44e073b2ac5906c205487b317ef813d32ca7c5e3b936cc
status: generated
---

# Work with the attune-rag dashboard

Use the dashboard when you want to benchmark a registered corpus, inspect retrieval quality as an HTML report, or review results directly in the terminal.

The dashboard runs as a three-stage pipeline:

1. **Refresh** — runs the benchmark against the corpus and writes a JSON snapshot.
2. **Render** — embeds the snapshot in an HTML report.
3. **Show** — pretty-prints the snapshot to the terminal using Rich.

Each stage has its own CLI entry point under `attune-rag dashboard`.

## Prerequisites

- A registered corpus package (default: `attune_help`)
- A `queries.yaml` file accessible to `build_snapshot()`. Without it, the snapshot is returned in a partial state with an error field.

## Run the full pipeline

### 1. Build the snapshot

Call `build_snapshot()` to benchmark your corpus and produce a snapshot dictionary:

```python
from attune_rag.dashboard.refresh import build_snapshot

snapshot = build_snapshot(corpus_package='attune_help')
```

To use a custom queries file, pass its path:

```python
from pathlib import Path
snapshot = build_snapshot(corpus_package='attune_help', queries_path=Path('path/to/queries.yaml'))
```

If `queries.yaml` is missing, `build_snapshot()` still returns a dict — check it for an `"error"` key before proceeding.

### 2. Render the HTML report

Pass the snapshot to `render()` with an output path and an optional title:

```python
from pathlib import Path
from attune_rag.dashboard.render import render

out_path = render(
    out=Path('dashboard_report.html'),
    snapshot=snapshot,
    title='attune-rag dashboard',
)
```

`render()` embeds the snapshot as JSON inside the HTML template and returns the resolved output path.

### 3. Display results in the terminal

Pass the snapshot to `display()` to print a Rich-formatted summary:

```python
from attune_rag.dashboard.show import display

display(snapshot)
```

To direct output to a specific Rich `Console` instance, pass it as the `console` argument.

## Run from the CLI

Each stage also exposes a `main()` entry point. A successful run returns exit code `0`.

```bash
attune-rag dashboard refresh
attune-rag dashboard render
attune-rag dashboard show
```

## Verify success

- `build_snapshot()` returns a dict with no `"error"` key when `queries.yaml` is found and the benchmark completes.
- `render()` returns the output `Path` and the file exists at that location containing embedded JSON (look for the sentinel value `__ATTUNE_SNAPSHOT__` replaced with your snapshot data).
- `display()` prints the snapshot table to the terminal without raising an exception.
- CLI entry points exit with code `0`.

## Key files

| File | Purpose |
|------|---------|
| `src/attune_rag/dashboard/__init__.py` | Package entry point |
| `src/attune_rag/dashboard/refresh.py` | `build_snapshot()` and `main()` — corpus benchmarking |
| `src/attune_rag/dashboard/render.py` | `render()` — HTML report generation |
| `src/attune_rag/dashboard/show.py` | `display()` and `main()` — terminal display |
