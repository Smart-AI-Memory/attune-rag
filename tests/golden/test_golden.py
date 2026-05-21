"""Golden-query regression tests.

Loads ``queries.yaml`` and asserts each query's expected
template(s) appear in the top-3 retrieval hits against the
real attune-help corpus. Skipped when the ``[attune-help]``
extra is not installed.

Also loads ``queries_paraphrased.yaml`` (the no-token-overlap
variants authored in diagnostic-1 for the alias-expansion-sweep)
as an **info-only regression input**. Each paraphrased query is
parametrized with ``xfail(strict=False)`` so it shows as XPASS
when retrieval surfaces the right entry and as XFAIL otherwise —
neither outcome breaks CI. The aggregate watermark test guards
against a catastrophic drop (e.g. corpus or stemmer change that
silently strips most paraphrased hits). The threshold-gating
decision for paraphrased R@3 is deferred to 0.4.x per
docs/specs/alias-expansion-sweep/tasks.md M13.3.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("attune_help")

import yaml  # noqa: E402

from attune_rag import RagPipeline  # noqa: E402

_QUERIES_FILE = Path(__file__).parent / "queries.yaml"
_PARAPHRASED_QUERIES_FILE = Path(__file__).parent / "queries_paraphrased.yaml"


def _load_queries() -> list[dict]:
    data = yaml.safe_load(_QUERIES_FILE.read_text(encoding="utf-8"))
    return data.get("queries", [])


def _load_paraphrased_queries() -> list[dict]:
    data = yaml.safe_load(_PARAPHRASED_QUERIES_FILE.read_text(encoding="utf-8"))
    return data.get("queries", [])


@pytest.fixture(scope="module")
def pipeline() -> RagPipeline:
    return RagPipeline()


def _entry_marks(entry: dict) -> list:
    """xfail hard queries — they document the keyword-retriever gap
    that the Phase 2 embeddings decision (task 2.5) is gated on.
    `strict=False` so if a hard query starts passing (e.g. after a
    retriever upgrade), the test doesn't fail — it passes as XPASS."""
    if entry.get("difficulty") == "hard":
        return [
            pytest.mark.xfail(
                reason="Hard queries need embeddings — task 2.5 gate",
                strict=False,
            )
        ]
    return []


@pytest.mark.parametrize(
    "entry",
    [pytest.param(e, id=e["id"], marks=_entry_marks(e)) for e in _load_queries()],
)
def test_golden_query_top3(entry: dict, pipeline: RagPipeline) -> None:
    """Each query's expected_in_top_3 must overlap with top 3 hits."""
    result = pipeline.run(entry["query"], k=3)
    hit_paths = [h.template_path for h in result.citation.hits]
    expected = set(entry.get("expected_in_top_3", []))
    actual = set(hit_paths)
    overlap = expected & actual
    assert overlap, (
        f"{entry['id']} ({entry['difficulty']}): "
        f"query={entry['query']!r} expected any of {sorted(expected)} "
        f"in top 3 but got {hit_paths}"
    )


# ---- paraphrased queries (info-only, alias-expansion-sweep M13.3) ----


# Soft floor for the aggregate watermark below.
#
# History
# - 0.3.x (M13.3): floor set at 50%, with observed R@3 at 70%. 20pp
#   headroom — only catches catastrophic regression.
# - 0.3.x (this tightening): floor raised to 85%, observed R@3 at
#   96.25% after M2-M12 swept all 12 feature clusters. 11pp headroom —
#   tight enough to flag a real cluster regression, loose enough to
#   absorb the kind of small drift that comes from corpus-level changes
#   (one new noisy alias, a borderline stemmer edit, a new template that
#   competes for content tokens).
#
# This is still NOT the gating threshold — that decision (whether to
# graduate the per-query xfails to hard assertions) is explicitly
# deferred to 0.4.x per docs/specs/alias-expansion-sweep/tasks.md M13.3.
# This is a regression guard; the gating bar will be higher and lives
# in benchmark.py with explicit stdev measurement, not here.
#
# If a future change drops R@3 below 85% legitimately (e.g., new
# paraphrased queries authored at a level the alias mechanism can't
# reach), update this constant in the same PR as the new queries land —
# don't loosen it silently to make CI green.
_PARAPHRASED_R3_FLOOR = 0.85


@pytest.mark.parametrize(
    "entry",
    [
        pytest.param(
            e,
            id=e["id"],
            marks=[
                pytest.mark.xfail(
                    reason=(
                        "Paraphrased query — info-only per "
                        "docs/specs/alias-expansion-sweep/tasks.md M13.3. "
                        "Passes show as XPASS, misses as XFAIL; neither breaks CI."
                    ),
                    strict=False,
                )
            ],
        )
        for e in _load_paraphrased_queries()
    ],
)
def test_paraphrased_query_top3(entry: dict, pipeline: RagPipeline) -> None:
    """Per-query visibility for the paraphrase regression set.

    All entries are xfail(strict=False), so the test result alone never
    breaks CI. The XPASS / XFAIL split in the test output gives a
    per-query view of paraphrase coverage; the aggregate watermark below
    is the only hard guard.
    """
    result = pipeline.run(entry["query"], k=3)
    hit_paths = [h.template_path for h in result.citation.hits]
    expected = set(entry.get("expected_in_top_3", []))
    actual = set(hit_paths)
    overlap = expected & actual
    assert overlap, (
        f"{entry['id']} ({entry['difficulty']}): "
        f"query={entry['query']!r} expected any of {sorted(expected)} "
        f"in top 3 but got {hit_paths}"
    )


def test_paraphrased_aggregate_watermark(pipeline: RagPipeline) -> None:
    """Aggregate R@3 / P@1 across the full paraphrase set.

    Acts as a regression guard against the alias-expansion mechanism
    being substantially broken — by an override-file load failure, a
    stemmer/tokenizer change that strips most overlap, the addition of
    a noisy alias that pulls adjacent features across many queries, or
    several clusters regressing at once. The floor (see
    ``_PARAPHRASED_R3_FLOOR`` above) is set below the current observed
    R@3 with enough headroom to absorb normal drift, but tight enough
    that a real degradation trips it.

    Prints the actual numbers so they're visible in ``pytest -v`` output
    and CI logs without requiring a separate diagnostic script run.
    """
    queries = _load_paraphrased_queries()
    p1_hits = 0
    r3_hits = 0
    for entry in queries:
        expected = set(entry.get("expected_in_top_3", []))
        result = pipeline.run(entry["query"], k=3)
        hit_paths = [h.template_path for h in result.citation.hits]
        if hit_paths and hit_paths[0] in expected:
            p1_hits += 1
        if expected & set(hit_paths):
            r3_hits += 1

    n = len(queries)
    p1 = p1_hits / n if n else 0.0
    r3 = r3_hits / n if n else 0.0

    # Visible in pytest -v output; CI logs this for trend tracking.
    print(
        f"\n[paraphrase-watermark] n={n} P@1={p1:.2%} ({p1_hits}/{n}) "
        f"R@3={r3:.2%} ({r3_hits}/{n})"
    )

    assert r3 >= _PARAPHRASED_R3_FLOOR, (
        f"Paraphrased R@3 dropped below regression floor: "
        f"{r3:.2%} < {_PARAPHRASED_R3_FLOOR:.0%}. "
        f"Likely causes: aliases_override.json failing to load, a "
        f"stemmer/tokenizer change stripping overlap, a noisy alias "
        f"pulling adjacent features, or several clusters regressing "
        f"at once. This floor is NOT the gating threshold — see test "
        f"docstring."
    )
