"""Living-docs dashboard for a registered corpus.

Three-stage pipeline:

- :mod:`attune_rag.dashboard.refresh` — runs the benchmark
  against the corpus, gathers freshness metadata, and emits a
  snapshot JSON.
- :mod:`attune_rag.dashboard.render` — renders a snapshot
  into a packaged HTML report.
- :mod:`attune_rag.dashboard.show` — pretty-prints a snapshot
  to the terminal via Rich.

Each stage has a ``main`` entry point so the trio can be run
independently as CLI subcommands, or composed (refresh → render
→ show) from a single shell.
"""
