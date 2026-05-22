# Spec: user-corpus-onboarding — tasks

> **Status:** scoped 2026-05-22 — executable. Entry gates inherited
> from [`requirements.md`](requirements.md) §"Entry gates"; this
> file describes what to *do* once those gates open.

## Scoping decisions locked (2026-05-22)

The seven open questions inherited from
[design.md](design.md) + [requirements.md §"Open questions"](requirements.md#open-questions-for-scoping)
are answered as follows:

| # | Question | Decision | Rationale |
|---|---|---|---|
| 1 | Module name | **`attune_rag.measure_corpus`** | Matches the existing `measure_baseline_variance.py` script-naming pattern; explicit about the module's purpose. `harness` is generic; `quality` is vague. |
| 2 | CLI entry point | **Both — `python -m attune_rag.measure_corpus` AND `attune-rag-measure` console_scripts entry** | The `python -m` path is install-safe (zero PATH conflicts); the console_scripts entry is discoverable. ~3 lines of `pyproject.toml` ships both. |
| 3 | Default watermark | **R@3 ≥ 0.85** | Matches the bundled corpus's locked floor. Users who want stricter can override; users who don't think about it inherit a sensible default. P@1 watermark stays opt-in via `--watermark-p1`. |
| 4 | Guide size | **Leaner 600-line v1** | Sections 1-6 + the QueryExpander section + the override-mechanism section. The worked example (§8) defers to v1.1.0 with iteration based on user feedback. Calendar-realistic for v1.0.0; iterative on what's most likely to need revision. |
| 5 | `load_aliases_from_file` helper | **Ship as public function** | Counts as 1 symbol against the 5-symbol budget. Useful for advanced users who want to construct the dict programmatically before passing to a `DirectoryCorpus` constructor. Total surface stays at 4: `measure`, `MeasureResult`, `load_aliases_from_file`, plus `extra_aliases_file` kwarg. |
| 6 | Symmetry with summaries | **Defer to v1.1.0** | The alias mechanism is the load-bearing lever for the alias-expansion sweep's discipline; the summaries override is a different shape and use-case. No demonstrated need for the file path yet — `summaries_override.json` use is internal-to-`AttuneHelpCorpus`. v1.1.0 if usage justifies. |
| 7 | Rerank-by-default in harness | **Mirror `RagPipeline` default** (currently `on`) with `--no-rerank` flag for ablation | The harness is users' window into the same pipeline shape that ships at v1.0.0. The bundled baseline includes rerank; the harness should too. `--no-rerank` lets users do D5-style ablation on their own corpus. If D5's verdict flips the pipeline default to `off`, this flag's *default* moves with it. |

These decisions are load-bearing: subsequent milestones reference
them. Changes require a new `/spec` pass.

## Sequencing relative to D5

D5 (`docs/specs/reranker-evaluation/`) is sequenced **strictly before**
this spec's M1 — the harness inherits D5's verdict for the rerank
default (Q7). Specifically:

- M1 of this spec starts **after** D5's M3 (verdict) closes.
- M2 (the kwarg work) is independent of D5 and can run in parallel
  with D5's M1-M2.
- M3 and M4 depend on M1, so they queue behind D5.

If D5's verdict is `rerank-default-off`, this spec's harness defaults
`--rerank` to `off` and the guide's §6 documents that for user
corpora the default lifts on detection of paraphrased-gap patterns
the keyword-only floor can't close (the corpus-shape decision).

## Milestones

### M0 — Entry-gate verification (pre-flight)

| # | Task | Layer | Notes |
|---|---|---|---|
| M0.1 | Verify Phase 4 W4.2 green: read `docs/specs/downstream-validation/cadence-week-{1,2,3,4}.md` and confirm all four report `**Status:** ON TRACK`. | attune-rag | Same gate as D5 M0.1. |
| M0.2 | Verify 0.2.0 shipped to PyPI. | attune-rag | New public surface from this spec lands at v1.0.0; 0.2.0 must be on PyPI first. |
| M0.3 | Verify 7-day no-hotfix watch on 0.2.0 closed clean. | attune-rag | Same gate as D5 M0.3. |
| M0.4 | Verify D5 closed and verdict locked. | attune-rag | M1 inherits D5's verdict for `--rerank` default; cannot start until D5's `diagnostic-1.md` is committed. |

### M1 — Quality harness (`attune_rag.measure_corpus`)

| # | Task | Layer | Notes |
|---|---|---|---|
| M1.1 | Create `src/attune_rag/measure_corpus.py` with `measure(corpus_path, queries_path, paraphrased_path=None, watermark_r3=0.85, watermark_p1=None, rerank=<D5_VERDICT>) -> MeasureResult` signature. `MeasureResult` is a frozen dataclass with `p1`, `r3`, `per_difficulty_breakdown`, `per_query_table`, `report_markdown()`. | attune-rag | Pure stdlib + existing `attune_rag` imports. The `rerank=` default is whichever value D5 locks. |
| M1.2 | Implement the iteration: load corpus via `DirectoryCorpus(corpus_path)`, load queries from YAML, run `RagPipeline.run` per query, score against ground-truth top-1. | attune-rag | Reuses the scoring logic from `tests/golden/test_golden.py` — extract it to a shared module if it's not already importable. |
| M1.3 | Implement `report_markdown()`: emit the markdown report mirroring `docs/specs/release-quality-baseline/baseline-1.md`'s shape — header (corpus identifier, query-set SHA-256, harness version, retriever version, timestamp), aggregate table (P@1, R@3, mean reranker faithfulness if rerank on), per-query table (sorted by YAML order — deterministic), optional residuals section. | attune-rag | Deterministic output: same input → byte-identical bytes (modulo timestamp). |
| M1.4 | Implement the CLI entry point at `__main__`: argparse with `--corpus-path`, `--queries`, `--paraphrased`, `--output`, `--watermark-r3` (default 0.85), `--watermark-p1` (default off), `--no-rerank`, `--format` (markdown/json), `--quiet`. Non-zero exit on watermark fail. | attune-rag | `python -m attune_rag.measure_corpus ...` invokes this; the `attune-rag-measure` console script (added in M1.7) does the same. |
| M1.5 | Implement R1 self-test: when the harness runs against the bundled `attune-help` corpus with `tests/golden/queries.yaml` + `tests/golden/queries_paraphrased.yaml`, the report's aggregate table reads **exactly** P@1=1.00, R@3=1.00, paraphrased P@1=0.9125, paraphrased R@3=1.00. | attune-rag | This is R1 of requirements.md. The harness's own correctness regression-tested against the bundled corpus. |
| M1.6 | Add `tests/unit/test_measure_corpus.py` covering: (a) loading a tiny synthetic corpus + queries fixture, (b) report rendering against the fixture (golden snapshot), (c) `MeasureResult.report_markdown()` deterministic across re-runs, (d) `--watermark-r3` failure path returns non-zero exit, (e) `--no-rerank` flag passes through, (f) malformed queries YAML raises `ValueError` with file path in message. | attune-rag | Live API calls (rerank) mocked. The bundled-corpus byte-identical check is a separate, opt-in test under `tests/golden/` so unit tests stay fast. |
| M1.7 | Add `attune-rag-measure` to `[project.scripts]` in `pyproject.toml` pointing at `attune_rag.measure_corpus:main`. Update `README.md` "Public API" section with the new module + CLI. | attune-rag | One-line `pyproject.toml` addition; README link to `USER_CORPUS_GUIDE.md` follows in M4. |

### M2 — First-class `extra_aliases_file` on `DirectoryCorpus`

| # | Task | Layer | Notes |
|---|---|---|---|
| M2.1 | Extract `AttuneHelpCorpus`'s internal `_load_extra_aliases_from_file()` helper to a public function `load_aliases_from_file(path: Path \| str) -> dict[str, list[str]]` at `src/attune_rag/corpus/__init__.py`. Add to `corpus.__all__`. | attune-rag | The helper already exists internally per the alias-expansion sweep; this is a refactor that exposes it. |
| M2.2 | Add `extra_aliases_file: Path \| str \| None = None` kwarg to `DirectoryCorpus.__init__`. On non-None: call `load_aliases_from_file()`, merge with inline `extra_aliases` (inline wins per-path collision), pass to existing alias-merge logic. | attune-rag | Semantics match R4 of requirements.md exactly. `FileNotFoundError` for missing file (path in message); `ValueError` for malformed JSON. |
| M2.3 | Add `tests/unit/test_directory_corpus_extra_aliases_file.py` covering: (a) file-only load, (b) file + inline merge with inline-wins on collision, (c) missing file → `FileNotFoundError`, (d) malformed JSON → `ValueError`, (e) underscore-prefixed keys filtered (matches `AttuneHelpCorpus` behavior), (f) non-string values rejected with typed error. | attune-rag | The semantics are inherited from `AttuneHelpCorpus`; tests pin them explicitly so a future refactor can't silently change them. |
| M2.4 | Update `corpus/__init__.py` `__all__` to include `load_aliases_from_file`. Verify `tests/unit/test_api_surface.py` `EXPECTED_CORPUS_ALL` is updated. | attune-rag | The api-surface snapshot test catches the addition; update the expected constant in the same PR. |
| M2.5 | Refactor `AttuneHelpCorpus`'s internal `_load_extra_aliases_from_file()` call sites to use the new public `load_aliases_from_file()`. Net behavior unchanged; the helper just lives in a different module now. | attune-rag | Strict-dominance: the bundled corpus's R@3 = 100% / paraphrased R@3 = 100% must hold byte-identically through this refactor. |

### M3 — "Your own corpus" guide (`docs/USER_CORPUS_GUIDE.md`, leaner v1)

| # | Task | Layer | Notes |
|---|---|---|---|
| M3.1 | Create `docs/USER_CORPUS_GUIDE.md` with the leaner v1 outline: §§1-6 + §7 (QueryExpander) + §9 (override mechanism's place). §8 (worked example) deferred to v1.1.0. | attune-rag | ~600 lines target. Voice: first-person plural, present tense, concrete. Lessons from the alias-expansion sweep cited as discoveries. |
| M3.2 | Write §1 (Corpus structure): directory layout, file naming, frontmatter schema. Cross-link to `editor/template_schema.json`. | attune-rag | Reuse phrasing from the existing README "Public API" section to stay consistent. |
| M3.3 | Write §2 (Frontmatter aliases): what they are, multi-token intent (the `MIN_ALIAS_OVERLAP=2` consequence), authoring patterns, the `_tokenize()` validation step (the `bites → bit` lesson). | attune-rag | Cite `[[feedback_alias_stem_validation]]` and the M12 near-regression from the alias-sweep retro. |
| M3.4 | Write §3 (Override file pattern): when to use vs frontmatter, override-then-promote workflow, example `aliases_override.json` schema, trimming overrides after promotion. | attune-rag | Reference the `extra_aliases_file` kwarg from M2 + the new `load_aliases_from_file` helper. |
| M3.5 | Write §4 (MIN_ALIAS_OVERLAP knob): default value + rationale, when to flip to 1, when to flip higher, measuring the trade-off (point at the harness). | attune-rag | Cite the 0.1.22 freeze-override that introduced the knob. |
| M3.6 | Write §5 (Stemmer gotchas): token pipeline, `_MIN_STEM_LEN` floor, common traps (`bites → bit`, `vulnerabilities → vulnerabilit`), the "run `_tokenize()` before authoring" discipline. | attune-rag | Same source as M3.3 but in trap-list form. |
| M3.7 | Write §6 (Quality measurement): authoring `queries.yaml` (schema, difficulty tiers), authoring `queries_paraphrased.yaml` (no-token-overlap variants), running the harness (CLI + Python API), reading the report, wiring into CI (the watermark flag), strict-dominance discipline. | attune-rag | The harness from M1 is what this section documents. |
| M3.8 | Write §7 (QueryExpander, when to use it): for corpora without curated frontmatter aliases, the QueryExpander is a viable alternative path. Cite D2 measurement; link to its re-framed docstring/README section if that PR has landed. | attune-rag | If the QueryExpander re-framing hasn't shipped yet at M3 time, link to its tracking issue. |
| M3.9 | Write §9 (Override mechanism's place): pointer to attune-rag's own `aliases_override.json` — what it is, why we use it, when you might want yours to look similar. | attune-rag | The "we're shipping a package that ships with a corpus" case; relevant for packages that wrap attune-rag downstream. |
| M3.10 | Cross-link the guide from `README.md` "Public API" section, `docs/POLICY.md` (the v1.0.0 commitment paragraph), `DirectoryCorpus.__init__` docstring, the harness `--help` output footer. | attune-rag | Discoverability — the guide is only as useful as the count of paths that lead to it. |

### M4 — Documentation polish + v1.0.0 cut readiness

| # | Task | Layer | Notes |
|---|---|---|---|
| M4.1 | Add a "Quality" section to `README.md` summarizing the bundled-corpus numbers (baseline P@1=1.00, paraphrased R@3=1.00) and pointing at `USER_CORPUS_GUIDE.md` for users measuring their own corpora. | attune-rag | The README is the first thing users see; surface the framework framing here. |
| M4.2 | Verify `attune-help`-bundled documentation cross-references the guide so attune-help users who *also* want to build a corpus have the path. | attune-help | Coordinate with attune-help maintainer (Patrick); may require an attune-help PR. |
| M4.3 | Update `docs/POLICY.md` with a paragraph naming the harness + the override-file mechanism as v1.0.0 commitments. Use the framework-framing language from `v1.0.0-release/design.md`. | attune-rag | Per `[[feedback_policy_llm_metric_commitments]]`: POLICY commitments here are about *capability* (the framework works for your corpus, with these tools), not metric thresholds. Avoid committing POLICY to LLM-dependent metric floors. |
| M4.4 | Update `tests/unit/test_api_surface.py` `EXPECTED_*_ALL` constants to include the new symbols: `attune_rag.measure_corpus.measure`, `MeasureResult`, `corpus.load_aliases_from_file`. Verify the snapshot test passes. | attune-rag | Q3 enforcement: new symbols are enumerated; the symbol-budget gate (≤ 5 new) is satisfied. |
| M4.5 | Add a CHANGELOG `### Added` entry under [Unreleased] enumerating the four new public symbols + the `extra_aliases_file` kwarg + the `USER_CORPUS_GUIDE.md` doc. | attune-rag | `### Added` is legal in this spec's PRs because they land AFTER the 0.2.0 cut (post-freeze). At v1.0.0 cut time the entries roll forward into the v1.0.0 release notes. |
| M4.6 | Verify all M1-M3 PRs ran the bundled-corpus diagnostic and showed byte-identical R@3 / P@1 numbers (Q1: strict-dominance). Document the diagnostic outputs in the v1.0.0 cut PR's body. | attune-rag | The harness itself makes this trivial: `python -m attune_rag.measure_corpus --corpus-path .help/templates --queries tests/golden/queries.yaml --paraphrased tests/golden/queries_paraphrased.yaml --output /tmp/check.md`. |

## Dependencies

```
M0.1 → M0.2 → M0.3 → M0.4   (entry gates; M0.4 awaits D5 closure)

M0.4 → M1.1 → M1.2 → M1.3 → M1.4 → M1.5 → M1.6 → M1.7   (harness build)

M0.4 → M2.1 → M2.2 → M2.3 → M2.4 → M2.5                  (kwarg refactor)
       (M2 runs parallel to M1; independent code paths)

(M1.7 + M2.5) → M3.1 → M3.2 → M3.3 → ... → M3.10          (guide)

M3.10 → M4.1 → M4.2 → M4.3 → M4.4 → M4.5 → M4.6           (polish)
```

## Definition of done (Phase 5 user-corpus-onboarding closes)

- [ ] All M0 gates green.
- [ ] M1: `attune_rag.measure_corpus` module ships with `measure`,
      `MeasureResult`, CLI entry point at `__main__`,
      `attune-rag-measure` console script, full test coverage.
- [ ] M1.5 self-test: harness reproduces bundled-baseline numbers
      byte-identically (R1).
- [ ] M2: `extra_aliases_file=` kwarg on `DirectoryCorpus`;
      `load_aliases_from_file` public helper; semantics
      `AttuneHelpCorpus`-identical (R4).
- [ ] M3: `docs/USER_CORPUS_GUIDE.md` published, leaner v1 (~600
      lines), cross-linked from README/POLICY/`DirectoryCorpus`
      docstring/CLI `--help`.
- [ ] M4: README "Quality" section, POLICY paragraph,
      `test_api_surface.py` updated, CHANGELOG `### Added`.
- [ ] Symbol budget ≤ 5 new public symbols (Q3) — actual: 4
      (`measure`, `MeasureResult`, `load_aliases_from_file`,
      `extra_aliases_file` kwarg counts in `DirectoryCorpus` signature
      surface).
- [ ] All implementation PRs ran the harness against the bundled
      corpus pre-merge and showed strict-dominance (Q1).
- [ ] User-corpus-onboarding status banner across
      README/design/requirements/risks/tasks promoted to
      `complete`.

## Out-of-scope (not in this spec; tracked elsewhere)

Per [`README.md`](README.md#whats-not-in-scope) and
[`requirements.md`](requirements.md#out-of-scope-requirements-deferred):

- Embedding retriever for arbitrary corpora (revival case if
  measurement contradicts the discipline-transfers assumption —
  separate spec).
- Web UI for the harness (attune-gui territory if anyone wants it).
- LLM-driven alias authoring (future spec).
- Multi-corpus reporting (CLI + Python API only run one corpus per
  invocation).
- `QueryExpander` re-framing PR (tracked separately ahead of M3.8).
- `extra_summaries_file=` kwarg (deferred to v1.1.0 per scoping
  decision #6).
- Worked example §8 in the guide (deferred to v1.1.0 per scoping
  decision #4).
- Faithfulness-watermark capability in the harness (faithfulness
  watermarking lives in `release-quality-baseline`).

## Calendar (proposed; Phase 5 opens after 0.2.0 + 7-day watch)

If D5 closes ~2026-07-03 (per its calendar projection) and runs
inform this spec's M1:

- **M0 verification** + **M2 start (parallel)**: ~2026-07-03
- **M1 (harness)**: ~3-4 days of attention starting after D5
- **M2 (kwarg)**: ~1 day of attention; runs parallel to M1
- **M3 (guide)**: ~3-4 days of attention; starts after M1.7 + M2.5
- **M4 (polish + cut prep)**: ~1 day of attention; lands as part of
  v1.0.0 cut PR

Total user-corpus-onboarding attention budget: **~8-10 days**, which
fits within the 6-8 week Phase 5 window with room for D5 (~5-6 days)
+ perf-baseline-multi-run (~3-5 days) + the v1.0.0 cut PR itself
(~2-3 days).
