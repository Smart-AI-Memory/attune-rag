---
type: concept
name: benchmark-concept
feature: benchmark
depth: concept
generated_at: 2026-05-20T03:30:01.567597+00:00
source_hash: 82975cf88c844b87657deb87845f45f4f5fbc32319ccba10e0eb8a798867630f
status: generated
---

# Benchmark

## Overview

The benchmark module is a retrieval quality runner that measures precision and recall against a query set, optionally scores faithfulness, and fails CI when results fall below configurable thresholds.

## How it works

At its core, the benchmark runs a retrieval pipeline against a set of queries, computes precision and recall for each result, and exits with a non-zero status code if any metric falls below the thresholds you configure. When you pass `--with-faithfulness`, the runner also evaluates how faithfully the retrieved content supports the generated answers.

The single entry point is **`main()`**, which parses arguments, executes the benchmark suite, and returns `0` on success. This return value is what CI systems use to determine whether the quality gate passed.

## Key concepts

**Precision and recall** are the primary retrieval metrics. Precision measures how many retrieved results are relevant; recall measures how many relevant results were retrieved. The benchmark gates on both, so a retrieval change that improves recall at the cost of precision will still fail if either metric drops below its threshold.

**Faithfulness scoring** is optional and enabled with `--with-faithfulness`. It evaluates whether the content retrieved actually supports the answers produced, which is a separate concern from whether the right documents were retrieved at all.

**Custom query files** let you benchmark against a domain-specific query set rather than a default one, so you can target the benchmark at the workload that matters for your use case.

## When it matters

Run the benchmark when you change retrieval logic, re-index your corpus, or update embedding models. Because `main()` returns `0` only on success, you can wire it directly into a CI step and treat a non-zero exit as a build failure.
