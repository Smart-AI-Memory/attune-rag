# Corpus measurement report

- Corpus: `bundled (AttuneHelpCorpus)`
- Harness version: `0.1.0`
- Timestamp: `2026-06-03T00:00:00Z`
- Query set `baseline`: `tests/golden/queries.yaml` (sha256: `f47486df87c6`)
- Query set `paraphrased`: `tests/golden/queries_paraphrased.yaml` (sha256: `307b6fcfa0d0`)
- Mode: `keyword-only`

## Aggregate

| Set | n | P@1 | R@3 |
|-----|---|-----|-----|
| baseline | 40 | 1.0000 | 1.0000 |
| paraphrased | 80 | 0.8750 | 0.9875 |

## Per-query — baseline

| qid | difficulty | P@1 | R@3 |
|-----|-----------|-----|-----|
| gq-001 | easy | ✓ | ✓ |
| gq-002 | easy | ✓ | ✓ |
| gq-003 | easy | ✓ | ✓ |
| gq-004 | easy | ✓ | ✓ |
| gq-005 | easy | ✓ | ✓ |
| gq-006 | medium | ✓ | ✓ |
| gq-007 | medium | ✓ | ✓ |
| gq-008 | medium | ✓ | ✓ |
| gq-009 | medium | ✓ | ✓ |
| gq-010 | medium | ✓ | ✓ |
| gq-011 | medium | ✓ | ✓ |
| gq-012 | medium | ✓ | ✓ |
| gq-013 | hard | ✓ | ✓ |
| gq-014 | medium | ✓ | ✓ |
| gq-015 | medium | ✓ | ✓ |
| gq-016 | easy | ✓ | ✓ |
| gq-017 | easy | ✓ | ✓ |
| gq-018 | easy | ✓ | ✓ |
| gq-019 | easy | ✓ | ✓ |
| gq-020 | easy | ✓ | ✓ |
| gq-021 | hard | ✓ | ✓ |
| gq-022 | medium | ✓ | ✓ |
| gq-023 | medium | ✓ | ✓ |
| gq-024 | medium | ✓ | ✓ |
| gq-025 | medium | ✓ | ✓ |
| gq-026 | medium | ✓ | ✓ |
| gq-027 | medium | ✓ | ✓ |
| gq-028 | medium | ✓ | ✓ |
| gq-029 | medium | ✓ | ✓ |
| gq-030 | medium | ✓ | ✓ |
| gq-031 | medium | ✓ | ✓ |
| gq-032 | medium | ✓ | ✓ |
| gq-033 | medium | ✓ | ✓ |
| gq-034 | medium | ✓ | ✓ |
| gq-035 | hard | ✓ | ✓ |
| gq-036 | medium | ✓ | ✓ |
| gq-037 | medium | ✓ | ✓ |
| gq-038 | medium | ✓ | ✓ |
| gq-039 | medium | ✓ | ✓ |
| gq-040 | medium | ✓ | ✓ |

## Per-query — paraphrased

| qid | difficulty | P@1 | R@3 |
|-----|-----------|-----|-----|
| gqp-001a | easy | ✓ | ✓ |
| gqp-001b | easy | ✗ | ✓ |
| gqp-002a | easy | ✓ | ✓ |
| gqp-002b | easy | ✓ | ✓ |
| gqp-003a | easy | ✓ | ✓ |
| gqp-003b | easy | ✗ | ✓ |
| gqp-004a | easy | ✓ | ✓ |
| gqp-004b | easy | ✓ | ✓ |
| gqp-005a | easy | ✗ | ✓ |
| gqp-005b | easy | ✓ | ✓ |
| gqp-006a | medium | ✓ | ✓ |
| gqp-006b | medium | ✓ | ✓ |
| gqp-007a | medium | ✓ | ✓ |
| gqp-007b | medium | ✓ | ✓ |
| gqp-008a | medium | ✓ | ✓ |
| gqp-008b | medium | ✓ | ✓ |
| gqp-009a | medium | ✓ | ✓ |
| gqp-009b | medium | ✓ | ✓ |
| gqp-010a | medium | ✓ | ✓ |
| gqp-010b | medium | ✓ | ✓ |
| gqp-011a | medium | ✓ | ✓ |
| gqp-011b | medium | ✗ | ✓ |
| gqp-012a | medium | ✓ | ✓ |
| gqp-012b | medium | ✓ | ✓ |
| gqp-013a | hard | ✓ | ✓ |
| gqp-013b | hard | ✓ | ✓ |
| gqp-014a | medium | ✓ | ✓ |
| gqp-014b | medium | ✓ | ✓ |
| gqp-015a | medium | ✓ | ✓ |
| gqp-015b | medium | ✗ | ✓ |
| gqp-016a | easy | ✓ | ✓ |
| gqp-016b | easy | ✗ | ✓ |
| gqp-017a | easy | ✓ | ✓ |
| gqp-017b | easy | ✗ | ✓ |
| gqp-018a | easy | ✓ | ✓ |
| gqp-018b | easy | ✓ | ✓ |
| gqp-019a | easy | ✓ | ✓ |
| gqp-019b | easy | ✓ | ✓ |
| gqp-020a | easy | ✓ | ✓ |
| gqp-020b | easy | ✓ | ✓ |
| gqp-021a | hard | ✓ | ✓ |
| gqp-021b | hard | ✗ | ✓ |
| gqp-022a | medium | ✓ | ✓ |
| gqp-022b | medium | ✓ | ✓ |
| gqp-023a | medium | ✗ | ✗ |
| gqp-023b | medium | ✓ | ✓ |
| gqp-024a | medium | ✓ | ✓ |
| gqp-024b | medium | ✓ | ✓ |
| gqp-025a | medium | ✓ | ✓ |
| gqp-025b | medium | ✓ | ✓ |
| gqp-026a | medium | ✓ | ✓ |
| gqp-026b | medium | ✓ | ✓ |
| gqp-027a | medium | ✓ | ✓ |
| gqp-027b | medium | ✓ | ✓ |
| gqp-028a | medium | ✓ | ✓ |
| gqp-028b | medium | ✓ | ✓ |
| gqp-029a | medium | ✓ | ✓ |
| gqp-029b | medium | ✓ | ✓ |
| gqp-030a | medium | ✓ | ✓ |
| gqp-030b | medium | ✓ | ✓ |
| gqp-031a | medium | ✗ | ✓ |
| gqp-031b | medium | ✓ | ✓ |
| gqp-032a | medium | ✓ | ✓ |
| gqp-032b | medium | ✓ | ✓ |
| gqp-033a | medium | ✓ | ✓ |
| gqp-033b | medium | ✓ | ✓ |
| gqp-034a | medium | ✓ | ✓ |
| gqp-034b | medium | ✓ | ✓ |
| gqp-035a | hard | ✓ | ✓ |
| gqp-035b | hard | ✓ | ✓ |
| gqp-036a | medium | ✓ | ✓ |
| gqp-036b | medium | ✓ | ✓ |
| gqp-037a | medium | ✓ | ✓ |
| gqp-037b | medium | ✓ | ✓ |
| gqp-038a | medium | ✓ | ✓ |
| gqp-038b | medium | ✓ | ✓ |
| gqp-039a | medium | ✓ | ✓ |
| gqp-039b | medium | ✓ | ✓ |
| gqp-040a | medium | ✓ | ✓ |
| gqp-040b | medium | ✓ | ✓ |

> 💡 Run with `--with-rerank` to measure whether rerank earns its keep on your corpus — a lift on marginal queries means leave it on; a neutral result means the keyword path is already doing the work. Either outcome is informative. Typical cost: ~$0.05 for an 80-query set at Haiku pricing. See `docs/USER_CORPUS_GUIDE.md` §6.2.
