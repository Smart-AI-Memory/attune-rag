---
type: comparison
name: dashboard-comparison
feature: dashboard
depth: comparison
generated_at: 2026-06-10T06:08:44.518608+00:00
source_hash: 48be0a4fd811c784bc44e073b2ac5906c205487b317ef813d32ca7c5e3b936cc
status: generated
---

# Comparison: Dashboard output modes

The attune-rag dashboard runs a three-stage pipeline — **refresh → render → show** — but stages two and three give you two distinct ways to consume the result: a self-contained HTML report or a terminal summary powered by Rich. Choosing the wrong output mode wastes time; this page helps you pick quickly.

## The two output modes at a glance

| | `render` (HTML report) | `display` (terminal view) |
|---|---|---|
| **Function** | `render(out, snapshot, title)` | `display(snapshot, console)` |
| **Module** | `dashboard.render` | `dashboard.show` |
| **Output** | A `.html` file written to `out` | Rich-formatted text to stdout / a `Console` |
| **Snapshot embedded?** | Yes — as inline JSON via `_SENTINEL_SNAPSHOT` | No — reads the live `snapshot` dict |
| **Sharable without tooling?** | Yes — open the file in any browser | No — requires a terminal session |
| **Custom title** | Yes — `title` parameter (default: `'attune-rag dashboard'`) | No title parameter |
| **Programmatic output path** | Caller controls `out: Path`; returns the written `Path` | Returns `None` |
| **Best for** | CI artifacts, async review, sharing with stakeholders | Quick local inspection during development |

## Stage one is the same regardless

Both modes start from the same snapshot. `build_snapshot()` in `dashboard.refresh` produces a `dict[str, Any]` that captures the corpus benchmark results. If `queries_path` is `None` or the file is missing, `build_snapshot()` returns a partial snapshot with an error key rather than raising — so downstream stages always receive a dict.

```python
from attune_rag.dashboard.refresh import build_snapshot

snapshot = build_snapshot(corpus_package='attune_help')
```

Pass `queries_path` explicitly when your queries file is not in the default location.

## Feature-by-feature breakdown

### HTML report — `render`

```python
from pathlib import Path
from attune_rag.dashboard.render import render

out_path = render(
    out=Path("reports/dashboard.html"),
    snapshot=snapshot,
    title="attune-rag dashboard",
)
```

`render()` writes the HTML to `out` and returns the same `Path`, so you can chain it with upload or open steps. The snapshot is embedded as JSON using the `_SENTINEL_SNAPSHOT` placeholder, which means the file is fully self-contained — no separate data file, no server required.

Choose this mode when:
- you archive dashboard results in CI
- you need to share results with someone who does not have the attune-rag toolchain installed
- you want a persistent record that correlates with a specific commit or run

### Terminal view — `display`

```python
from attune_rag.dashboard.show import display

display(snapshot=snapshot)          # writes to stdout
display(snapshot=snapshot, console=my_console)  # writes to a custom Rich Console
```

`display()` returns `None` and has no file side-effect. Passing a custom `Console` is useful when you want to redirect output or control formatting in tests.

Choose this mode when:
- you are iterating locally and want immediate feedback without opening a browser
- you are inside an interactive terminal session and a file artifact would slow you down
- you are scripting and just need a human-readable summary before deciding whether to proceed

## Tradeoffs summary

| Concern | HTML (`render`) | Terminal (`display`) |
|---|---|---|
| **Persistence** | Durable file artifact | Ephemeral — lost when the session closes |
| **Shareability** | Any browser, no dependencies | Requires a terminal with Rich |
| **Speed to result** | Slightly slower — involves a file write | Instant |
| **CI integration** | Natural — save the returned `Path` as an artifact | Awkward — stdout capture only |
| **Interactivity** | Static HTML | Live terminal, supports custom `Console` |

## Use X when…

**Use `render`** when you need a durable, shareable artifact — in CI pipelines, when archiving corpus health over time, or when a stakeholder needs to review results without running attune-rag themselves.

**Use `display`** when you need fast, low-friction feedback during local development. It is the faster path to a human-readable result and disappears cleanly when you close the terminal.

**Use both together** when you want the best of each: call `display()` for an immediate summary, then call `render()` to preserve the same snapshot as a report. Both accept the same `snapshot` dict, so there is no redundant computation.

---

**Tags:** `dashboard`, `living-docs`, `html`, `terminal`, `snapshot`, `freshness`
