# W2.1 — Mid-phase deep-review

> Per [`docs/specs/downstream-validation/tasks.md`](./tasks.md) W2.1.
> Three-pass synthesis (security / quality / tests) over the four W2.1
> target modules. Date: 2026-05-20. Commit: `a1d9e4a` (post W1.2 log).

## Scope

| Module | LOC |
|---|---:|
| `src/attune_rag/editor/` (incl. shims) | 1,553 |
| `src/attune_rag/providers/` | 428 |
| `src/attune_rag/pipeline.py` | 451 |
| `src/attune_rag/retrieval.py` | 342 |
| **Total** | **2,774** |

## Health snapshot

| Pass | Result | Blocking? |
|---|---|---|
| Security | 0 new HIGH/CRITICAL; 4 already-triaged refs re-confirmed clean; 1 INFO docstring fix | No |
| Quality | Aggregate ~90/100 (TH 94, Doc 92, Cx 86, Style 96, API 80) | No |
| Tests | **89.78 %** aggregate line+branch coverage; 797 passed, 3 xpassed | No |

W2.1 gate: **green.** No findings need to land during the freeze; the
HIGH-rollback test gaps below are pure additions (no new public surface)
and slot naturally into W3.3's coverage push.

## Security pass

**New findings:** 1 (INFO, docstring only).
**Already-triaged refs re-confirmed clean:** W09.S.006, W09.S.007,
W09.S.008, W09.A.007.

| Severity | File:line | CWE | Finding | Suggested fix |
|---|---|---|---|---|
| INFO | `editor/rename.py:220-223` | CWE-1078 | Docstring claims `Path.resolve(strict=False)` runs "WITHOUT following symlinks" — `resolve()` does follow symlinks by default. The containment check itself is correct (compares resolved candidate to resolved root); only the comment misleads. | Correct the docstring to read "resolve symlinks so containment is checked on the canonical path" — `### Changed`, no behaviour shift. |

Confirmed clean (not re-flagged): `editor/rename.py:217`
sanitiser, `tempfile.mkstemp()` use at `:560-569`, `except Exception:`
rollback blocks at `:470, :489, :566` (all re-raise after cleanup),
`yaml.safe_load` usage in `editor/schema.py:80` and `editor/lint.py:119`,
`importlib.resources` package-internal schema load,
`providers/{claude,gemini}.py` api-key handling (never logged),
`retrieval.py` regex patterns (linear-time, no ReDoS surface),
absence of `shell=True` / `subprocess` / `eval` / `exec` / `compile` /
`pickle` / `marshal` across the entire target set.

## Quality pass

### Aggregate scores

| Dimension | Score | Notes |
|---|---:|---|
| Type hints | 94/100 | Near-universal. Only `get_provider(**kwargs)` and a handful of `_iter_entries` helpers lack annotations. |
| Docstrings | 92/100 | Every public symbol has a one-line summary. Style is closer to numpydoc/free-form than strict Google; `_stem`, `_tokenize`, `Hunk.to_dict`, `FileEdit.to_dict` lack docstrings. |
| Complexity | 86/100 | One real hotspot: `apply_rename` (`rename.py:426-515`, ~90 LOC, three rollback layers). |
| Style | 96/100 | Zero >100-char lines. No copy-paste blocks. Magic numbers are named (`MAX_CITATION_DOCUMENTS`, `_MIN_CACHE_CHARS`, `PATH_HIT_CAP`, `_MIN_STEM_LEN`). |
| API consistency | 80/100 | `__all__` in `editor/__init__.py` aligns with submodule re-exports. `providers/__init__.py:__all__` is not alphabetised; the five `editor/_*.py` shims are deliberate one-release deprecation surfaces (slated for removal at 0.3.0). |

### Top findings (severity-sorted)

1. **[MED]** `editor/rename.py:426-515` — `apply_rename` is ~90 LOC with three nested rollback layers (moves → staging → sequential rename). Each catch block duplicates `_undo_moves` + `_undo_created_dirs` cleanup. **Fix:** extract a `_RollbackState` helper / context manager so each phase calls `state.rollback()`. Internal refactor — no API change. (Phase-5 candidate.)
2. **[MED]** `providers/__init__.py:41` — `get_provider(name: str, **kwargs) -> LLMProvider` lacks a `**kwargs` annotation. **Fix:** `**kwargs: Any`. Pure type-hint completion.
3. **[MED]** `retrieval.py:114` — `_stem` has no docstring despite being load-bearing (the 0.1.22 `-ity`/`-ities` change lives here). **Fix:** one-line docstring documenting the suffix-order contract ("longest-match-first; min stem length 3") to pre-empt a future regression where someone alphabetises `_STEM_SUFFIXES`.
4. **[LOW]** `providers/gemini.py:44` — only spot using parenthesised union (`(str | None) = None`). Convert to bare `str | None = None` for consistency with the rest of the package.
5. **[LOW]** `editor/references.py:164-171`, `rename.py:586-593`, `autocomplete.py:63-76` — three near-identical `_iter_entries(corpus)` helpers (~7 LOC each). Don't lift today (the shim noise complicates it); revisit post-0.3.0.
6. **[LOW]** `editor/lint.py:62` — `_KNOWN_FRONTMATTER_KEYS` computed at module-import time. If the JSON schema were ever missing, importing `lint` would raise. Move behind an `@lru_cache`'d helper for lazy failure (parallels `_validator()`).
7. **[LOW]** `editor/rename.py:419` — `hashlib.sha256(...)` truncated to 16 hex chars for an internal id. Not security-relevant; add a "non-cryptographic content hash" comment to pre-empt audit false positives.
8. **[LOW]** `editor/rename.py:60-86` + `editor/lint.py` — `Hunk.to_dict`, `FileEdit.to_dict`, `Diagnostic.to_dict` lack docstrings.
9. **[LOW]** `editor/schema.py:90` — `validator = err.validator` would read more clearly as `keyword` (jsonschema's own term).
10. **[LOW]** `providers/__init__.py:59` — `__all__` is not alphabetised. Cosmetic.

### Notable

- `retrieval.py` is in very good shape after 0.1.22 — `MIN_ALIAS_OVERLAP` and the `-ities`/`-ity` stemming change are well-commented inline with motivation.
- `providers/claude.py` is exemplary: rationale comments at `MAX_CITATION_DOCUMENTS=200`, the cache_control single-block placement, and the V2/V3 probe references are textbook "future-me debugging" notes.
- `pipeline.py::run_and_generate` branching matrix is documented in prose right next to the branches. Tempting to extract a strategy object; current shape is actually clearer than that refactor would be — **leave alone.**
- `editor/_*.py` shims are deliberate; they have a passing equivalence test and are scheduled for removal at 0.3.0. Not technical debt today.

## Test gap pass

### Coverage (target modules only)

| Module | Stmts | Miss | Branch | BrPart | Cov |
|---|---:|---:|---:|---:|---:|
| editor/__init__.py | 7 | 0 | 0 | 0 | 100% |
| editor/_autocomplete.py | 6 | 0 | 0 | 0 | 100% |
| editor/_lint.py | 6 | 0 | 0 | 0 | 100% |
| editor/_references.py | 6 | 0 | 0 | 0 | 100% |
| editor/_rename.py | 6 | 0 | 0 | 0 | 100% |
| editor/_schema.py | 6 | 0 | 0 | 0 | 100% |
| editor/autocomplete.py | 35 | 2 | 14 | 3 | 90% |
| editor/lint.py | 126 | 7 | 50 | 5 | 93% |
| editor/references.py | 138 | 12 | 66 | 10 | 88% |
| **editor/rename.py** | 358 | 57 | 122 | 25 | **82%** |
| editor/schema.py | 60 | 4 | 14 | 2 | 92% |
| pipeline.py | 115 | 3 | 24 | 1 | 97% |
| providers/__init__.py | 22 | 2 | 6 | 0 | 93% |
| providers/base.py | 18 | 1 | 0 | 0 | 94% |
| providers/claude.py | 68 | 0 | 24 | 1 | 99% |
| providers/gemini.py | 19 | 0 | 2 | 0 | 100% |
| retrieval.py | 111 | 3 | 38 | 4 | 95% |

**Aggregate:** **89.78 %** line+branch coverage across the four W2.1
targets (1,107 stmts, 91 missing; 360 branches, 51 partial). Tests:
**797 passed, 3 xpassed** in 7.7 s. Already meeting W3.3's ~90 %
target in aggregate; gaps concentrate in `editor/rename.py` rollback
paths.

### Top gaps (priority-sorted)

1. **[HIGH]** `editor/rename.py::apply_rename` — mid-flight rollback at lines 502-512 (`os.replace` failure during *stage 3* commit) untested. Suggested: `test_apply_rolls_back_edits_when_commit_replace_fails`.
2. **[HIGH]** `editor/rename.py::_undo_moves` / `_undo_created_dirs` — `OSError` swallow is implicit, not asserted. Suggested: `test_undo_moves_tolerates_oserror`.
3. **[HIGH]** `editor/rename.py::_rewrite_yaml_block_value` — quoted list items (`- 'old'`, `- "old"`) and indented-list edges uncovered. Suggested: `test_rewrite_yaml_block_value_handles_quoted_and_indented_list_items`.
4. **[MED]** `editor/rename.py::_plan_sidecar_path_rename_edits` lines 245-254 — `OSError` / `JSONDecodeError` / key-absent continue paths uncovered. Suggested: `test_sidecar_corrupt_json_is_skipped_silently`.
5. **[MED]** `pipeline.py:408` — `prompt_variant == "baseline"` on the native-citations path never exercised. Suggested: `test_native_path_uses_join_context_when_baseline_variant`.
6. **[MED]** `editor/references.py:222` — fenced code-block exclusion in the body-ref scanner uncovered. Suggested: `test_find_references_ignores_aliases_inside_fenced_code_blocks`.
7. **[MED]** `editor/lint.py:188-219` — `_yaml_error_position` fallback when `problem_mark is None`. Suggested: `test_lint_yaml_error_without_problem_mark_returns_fallback_line`.
8. **[LOW]** `editor/autocomplete.py:34/76`, `editor/schema.py:108-129`, `retrieval.py:253/329/333`, `providers/__init__.py:36-37`, `providers/base.py:81` — assorted single-branch gaps (see test-gap pass detail).

### Shallow-but-covered (W3.3 `/test-audit` hand-off)

- **`pipeline.py::RagPipeline.run_and_generate`** — well-covered numerically, but the provider is **always a `FakeProvider`**. No live or VCR-recorded integration with `ClaudeProvider.generate_with_citations`. The `cached_prefix` tests assert the kwarg was *passed* but not that the SDK actually flagged the block for caching.
- **`providers/claude.py`** at 99 % line coverage but the entire test surface mocks `anthropic.AsyncAnthropic`. Recommend one recorded-response fixture under `tests/unit/providers/fixtures/` to catch SDK shape drift.
- **`editor/rename.py::apply_rename`** rollback paths — current tests synthesise failures with carefully timed `target.exists()` returns rather than realistic disk-fault simulation. Acceptable but worth flagging.
- **`editor/lint.py::lint_template`** — schema-error branches at 188-219 use synthetic `yaml.YAMLError` instances without `problem_mark`; a focused test would document the real-PyYAML edge case.
- **`retrieval.py::KeywordRetriever.retrieve`** — heavy scoring coverage, but no test where the retriever *raises* during `pipeline.run`. Pipeline currently lets exceptions propagate; `test_pipeline_run_propagates_retriever_exception` would document that contract.

### Recommended next-week tests (≤6 concrete cases)

These all land as **new tests against existing public surface** — no
new public symbols, no freeze-clock impact.

1. `tests/unit/test_editor_rename.py::test_apply_rolls_back_edits_when_commit_replace_fails` — closes the highest-risk rollback gap.
2. `tests/unit/test_pipeline_native_citations.py::test_native_path_uses_join_context_when_baseline_variant` — closes pipeline.py:408.
3. `tests/unit/test_retrieval.py::test_min_alias_overlap_boundary` — locks in the 0.1.22 escape-hatch contract on the golden set.
4. `tests/unit/test_editor_rename.py::test_rewrite_yaml_block_value_handles_quoted_list_items`.
5. `tests/unit/test_editor_references.py::test_find_references_skips_aliases_inside_fenced_code_blocks`.
6. `tests/unit/test_editor_rename.py::test_sidecar_corrupt_json_is_skipped_silently`.

### Additional observations

- Coverage had to be invoked with `--override-ini="addopts="` because `pyproject.toml` sets `addopts = "-ra -m 'not live'"` and pytest-cov inherits. Worth a contributor README note.
- Three `XPASS` on `tests/golden/test_golden.py::test_golden_query_top3[gq-013|021|035]` — xfailed pending embeddings (task 2.5) but now passing on the keyword retriever after 0.1.22. Consider removing the markers (or downgrading to `xfail(strict=False, ...)`) once a second clean weekly run confirms the gains stick.

## Disposition

| Class | Items |
|---|---|
| **Blocking (must fix this freeze week)** | none |
| **Fix during freeze (`### Changed` / `### Fixed`)** | Quality #2 (`**kwargs: Any`), #3 (`_stem` docstring), #4 (gemini.py union style), #6 (lazy `_KNOWN_FRONTMATTER_KEYS`), #7 (sha256 comment), #8 (to_dict docstrings), Security INFO (rename.py:220 docstring). Optional W2 housekeeping commit. |
| **W3.3 `/test-audit` follow-up** | The five shallow-but-covered notes above; the three `XPASS` markers. |
| **W3.3 `/smart-test` add (next week)** | The six concrete tests in "Recommended next-week tests". |
| **Phase-5 ticket** | Quality #1 (`_RollbackState` extraction in `apply_rename`), #5 (lift shared `_iter_entries` after 0.3.0 shim removal), #9 (`validator` → `keyword` rename), #10 (`__all__` alphabetisation). |

## Hand-off to W2.2

W2.2 runs `attune-ai:performance_audit` on retrieval + reranker hot
paths and cross-checks against [`perf-baseline.md`](./perf-baseline.md).
None of the W2.1 findings touch perf-sensitive code; W2.2 can proceed
independently.
