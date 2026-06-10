---
type: quickstart
name: dashboard-quickstart
feature: dashboard
depth: quickstart
generated_at: 2026-06-10T06:08:44.510628+00:00
source_hash: 48be0a4fd811c784bc44e073b2ac5906c205487b317ef813d32ca7c5e3b936cc
status: generated
---

# Quickstart: Dashboard

See your corpus health in the terminal with three Python calls: build a snapshot, render an HTML report, and display the results.

```python
from attune_rag.dashboard.refresh import build_snapshot
from attune_rag.dashboard.render import render
from attune_rag.dashboard.show import display
from pathlib import Path

snapshot = build_snapshot()
render(Path("dashboard.html"), snapshot)
display(snapshot)
```

Running this prints a Rich-formatted summary to your terminal and writes `dashboard.html` to the current directory.

## Prerequisites

- The project is cloned and installed locally.
- Your corpus is registered under the default `attune_help` package, or you know the package name to pass as `corpus_package`.

## Step 1: Build a snapshot

```python
from attune_rag.dashboard.refresh import build_snapshot

snapshot = build_snapshot()
print(snapshot)
```

`build_snapshot()` runs the benchmark against your corpus and returns a dict. If `queries.yaml` is missing, the dict still returns but includes an `error` key describing the partial result.

## Step 2: Render the HTML report

```python
from attune_rag.dashboard.render import render
from pathlib import Path

out_path = render(Path("dashboard.html"), snapshot)
print(out_path)  # PosixPath('dashboard.html')
```

`render()` writes an HTML file to the path you supply and returns that same path. Open `dashboard.html` in a browser to see the full report.

## Step 3: Display the snapshot in the terminal

```python
from attune_rag.dashboard.show import display

display(snapshot)
```

`display()` pretty-prints the snapshot using Rich. Pass your own `Console` instance as the second argument if you need to redirect output.

**Expected output** (abbreviated):

```
┌─────────────────────────────┐
│   attune-rag dashboard      │
│   corpus: attune_help       │
│   queries: 42   hits: 38    │
└─────────────────────────────┘
```

## Next:

Open `dashboard.html` in a browser and compare the rendered report against the terminal summary to confirm both reflect the same snapshot data.
