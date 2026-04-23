---
type: task
feature: cli
depth: task
generated_at: 2026-04-23T03:37:03.009526+00:00
source_hash: dd67ed58271857e52c84068665bf3e4f498258f5607603a6e6df7dac8dfc63fe
status: generated
---

# Work with cli

Use the CLI module when you need to debug retrieval queries or inspect corpus statistics from the command line.

## Prerequisites

- Access to the project source code
- Python environment with attune-rag installed

## Configure the command parser

1. **Open the CLI module.**
   Navigate to `src/attune_rag/cli.py`.

2. **Modify the `build_parser()` function.**
   Add new arguments, subcommands, or help text by editing the ArgumentParser configuration.

3. **Test your parser changes.**
   Run `python -m attune_rag.cli --help` to verify the help output displays correctly.

## Add new command functionality

1. **Locate the main entry point.**
   Find the `main()` function in `src/attune_rag/cli.py`.

2. **Add your command logic.**
   Handle new arguments or subcommands by extending the conditional logic in `main()`.

3. **Follow existing patterns.**
   Match the error handling, output formatting, and return value conventions used by existing commands.

## Verify your changes

Run the CLI commands to confirm they work as expected:

```bash
# Test query command
attune-rag query "your test question"

# Test corpus info command
attune-rag corpus-info
```

The commands should execute without errors and produce properly formatted output. Your changes work correctly when the CLI responds to new arguments and displays appropriate help text.
