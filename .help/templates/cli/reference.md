---
type: reference
feature: cli
depth: reference
generated_at: 2026-04-23T03:37:11.480966+00:00
source_hash: dd67ed58271857e52c84068665bf3e4f498258f5607603a6e6df7dac8dfc63fe
status: generated
---

# CLI reference

Command-line interface for debugging document retrieval and corpus inspection.

## Functions

| Function | Parameters | Returns | Description |
|----------|------------|---------|-------------|
| `build_parser()` | | `argparse.ArgumentParser` | Create argument parser for CLI commands |
| `main()` | `argv: list[str] \| None = None` | `int` | Execute CLI command and return exit code |

## Source files

- `src/attune_rag/cli.py`

## Tags

`cli`, `query`, `corpus-info`
