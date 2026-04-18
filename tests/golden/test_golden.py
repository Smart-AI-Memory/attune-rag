"""Golden-query regression tests.

Loads ``queries.yaml`` and asserts each query's expected
template(s) appear in the top-3 retrieval hits against the
real attune-help corpus. Skipped when the ``[attune-help]``
extra is not installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("attune_help")

import yaml  # noqa: E402

from attune_rag import RagPipeline  # noqa: E402

_QUERIES_FILE = Path(__file__).parent / "queries.yaml"


def _load_queries() -> list[dict]:
    data = yaml.safe_load(_QUERIES_FILE.read_text(encoding="utf-8"))
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
