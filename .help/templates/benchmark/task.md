---
type: task
feature: benchmark
depth: task
generated_at: 2026-04-23T03:36:40.994101+00:00
source_hash: efe3184170bbd1e763131bf4307b2835dc8fb12752af2f0f8b5cb67b4d27ad03
status: generated
---

# Work with benchmark

Run benchmark when you need to evaluate retrieval quality or gate CI builds on configurable precision, recall, and faithfulness thresholds.

## Prerequisites

- Access to the project source code
- Query files for testing (optional — benchmark includes defaults)
- Understanding of your quality thresholds for CI gates

## Steps

1. **Navigate to the benchmark module.**
   Open `src/attune_rag/benchmark.py` to examine the main entry point and available configuration options.

2. **Identify your evaluation scope.**
   Decide whether you need basic retrieval metrics (precision/recall) or comprehensive evaluation including faithfulness scoring via the `--with-faithfulness` flag.

3. **Configure your test queries.**
   Either use the default query set or specify custom query files that match your evaluation requirements.

4. **Set quality thresholds.**
   Define the minimum acceptable scores for precision, recall, and faithfulness that will determine CI pass/fail status.

5. **Execute the benchmark.**
   Run the `main()` function with your chosen configuration to generate evaluation metrics and apply threshold checks.

6. **Review the results.**
   Examine the output to verify metrics meet your quality gates and identify any areas needing improvement.

## Verification

The benchmark completes successfully when:
- All configured quality thresholds are met
- The function returns exit code 0
- Evaluation metrics are displayed in the output

## Key files

- `src/attune_rag/benchmark.py` — Main benchmark runner and configuration
