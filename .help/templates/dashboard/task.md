---
type: task
name: dashboard-task
feature: dashboard
depth: task
generated_at: 2026-06-10T06:08:44.486758+00:00
source_hash: 48be0a4fd811c784bc44e073b2ac5906c205487b317ef813d32ca7c5e3b936cc
status: generated
---

# Work with the dashboard

Use the dashboard when you want to inspect the health of a registered corpus — it runs a three-stage pipeline: `refresh` builds a JSON snapshot by benchmarking the corpus, `render` packages that snapshot into a self-contained HTML report, and `show` pretty-prints the snapshot to your terminal.

## Prerequisites

- A registered corpus package (default: `attune_help`)
- A `queries.yaml` file accessible to the corpus package (required for a complete snapshot)
- Python environment with `attune-rag` installed

## Run the pipeline

### 1. Build a snapshot

Call `build_snapshot()` to benchmark the corpus and produce a snapshot dictionary:

```python
from attune_rag.dashboard.refresh import build_snapshot

snapshot = build_snapshot(corpus_package='attune_help')
```

If `queries.yaml` is missing, `build_snapshot()` returns a partial snapshot that includes an error field — check the returned dictionary before proceeding.

You can also supply a custom queries file:

```python
from pathlib import Path
snapshot = build_snapshot(corpus_package='attune_help', queries_path=Path('my_queries.yaml'))
```

### 2. Render an HTML report

Pass the snapshot to `render()` along with an output path:

```python
from pathlib import Path
from attune_rag.dashboard.render import render

out_path = render(out=Path('dashboard.html'), snapshot=snapshot, title='attune-rag dashboard')
```

`render()` writes the HTML file to `out` with the snapshot embedded as JSON and returns the resolved output path.

### 3. Display the snapshot in the terminal

Pass the snapshot to `display()` to pretty-print it with Rich:

```python
from attune_rag.dashboard.show import display

display(snapshot=snapshot)
```

To redirect output, pass your own `Console` instance as the `console` argument.

## Verify the results

- **Snapshot**: confirm the returned dictionary from `build_snapshot()` contains no error field (or that any error field matches an expected condition such as a missing `queries.yaml`).
- **HTML report**: open `dashboard.html` in a browser and verify the page title reads `attune-rag dashboard` and the corpus data is populated.
- **Terminal output**: confirm that `display()` prints structured snapshot content to the terminal without raising an exception.

## Key files

| File | Responsibility |
|---|---|
| `src/attune_rag/dashboard/refresh.py` | `build_snapshot()` — benchmarks the corpus and returns a snapshot dict |
| `src/attune_rag/dashboard/render.py` | `render()` — writes the HTML report with the snapshot embedded |
| `src/attune_rag/dashboard/show.py` | `display()` — prints the snapshot to the terminal via Rich |
