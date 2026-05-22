"""Unit tests for the public ``attune_rag.measure_corpus`` module.

Complements ``tests/unit/test_measure_corpus.py`` which tests the
CLI (subprocess-invoked through the backward-compat
``scripts/measure_corpus.py`` shim). These tests exercise the
Python API directly: ``measure()`` + ``MeasureResult``.

No live Anthropic calls — uses a synthetic 3-doc DirectoryCorpus.
The bundled-corpus byte-identical reproduction lives in
``tests/golden/test_measure_corpus_module_bundled.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from attune_rag.measure_corpus import MeasureResult, measure


@pytest.fixture
def synthetic_corpus(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "alpha.md").write_text(
        "---\naliases: [alpha-feature, alpha-thing]\n---\n# Alpha\n", encoding="utf-8"
    )
    (root / "beta.md").write_text(
        "---\naliases: [beta-feature, beta-widget]\n---\n# Beta\n", encoding="utf-8"
    )
    (root / "gamma.md").write_text(
        "---\naliases: [gamma-feature, gamma-tool]\n---\n# Gamma\n", encoding="utf-8"
    )
    return root


@pytest.fixture
def synthetic_queries(tmp_path: Path) -> Path:
    path = tmp_path / "queries.yaml"
    yaml.safe_dump(
        {
            "queries": [
                {
                    "id": "q1",
                    "query": "alpha-feature",
                    "difficulty": "easy",
                    "expected_in_top_3": ["alpha.md"],
                },
                {
                    "id": "q2",
                    "query": "beta-widget",
                    "difficulty": "easy",
                    "expected_in_top_3": ["beta.md"],
                },
                {
                    "id": "q3",
                    "query": "gamma-tool",
                    "difficulty": "medium",
                    "expected_in_top_3": ["gamma.md"],
                },
            ]
        },
        path.open("w", encoding="utf-8"),
    )
    return path


def test_measure_returns_MeasureResult(synthetic_corpus: Path, synthetic_queries: Path) -> None:
    result = measure(corpus_path=synthetic_corpus, queries_path=synthetic_queries)
    assert isinstance(result, MeasureResult)
    assert result.p1 == 1.0
    assert result.r3 == 1.0
    assert result.n == 3
    assert result.paraphrased_p1 is None
    assert result.rerank is False
    assert "corpus" in result.corpus_label
    assert result.queries_sha and len(result.queries_sha) == 12


def test_measure_with_paraphrased(synthetic_corpus: Path, synthetic_queries: Path) -> None:
    result = measure(
        corpus_path=synthetic_corpus,
        queries_path=synthetic_queries,
        paraphrased_path=synthetic_queries,
    )
    assert result.paraphrased_p1 == 1.0
    assert result.paraphrased_n == 3
    assert result.paraphrased_per_query is not None
    assert len(result.paraphrased_per_query) == 3


def test_per_difficulty_breakdown(synthetic_corpus: Path, synthetic_queries: Path) -> None:
    result = measure(corpus_path=synthetic_corpus, queries_path=synthetic_queries)
    assert set(result.per_difficulty_breakdown.keys()) == {"easy", "medium"}
    assert result.per_difficulty_breakdown["easy"]["n"] == 2
    assert result.per_difficulty_breakdown["medium"]["n"] == 1


def test_report_markdown_deterministic(synthetic_corpus: Path, synthetic_queries: Path) -> None:
    result = measure(corpus_path=synthetic_corpus, queries_path=synthetic_queries)
    a = result.report_markdown(frozen_timestamp="2026-05-22T00:00:00Z")
    b = result.report_markdown(frozen_timestamp="2026-05-22T00:00:00Z")
    assert a == b
    assert "1.0000" in a
    assert "Timestamp: `2026-05-22T00:00:00Z`" in a


def test_watermark_failures_empty_when_pass(
    synthetic_corpus: Path, synthetic_queries: Path
) -> None:
    result = measure(corpus_path=synthetic_corpus, queries_path=synthetic_queries)
    assert result.watermark_failures(r3_floor=0.85) == []
    assert result.watermark_failures(p1_floor=0.99, r3_floor=0.85) == []


def test_watermark_failures_reports_below(synthetic_corpus: Path, synthetic_queries: Path) -> None:
    result = measure(corpus_path=synthetic_corpus, queries_path=synthetic_queries)
    failures = result.watermark_failures(r3_floor=1.5)  # impossible
    assert len(failures) == 1
    assert "baseline" in failures[0]
    assert "1.5" in failures[0]


def test_measure_corpus_path_and_bundled_mutually_exclusive(
    synthetic_corpus: Path, synthetic_queries: Path
) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        measure(
            corpus_path=synthetic_corpus,
            bundled=True,
            queries_path=synthetic_queries,
        )
    with pytest.raises(ValueError, match="exactly one"):
        measure(queries_path=synthetic_queries)


def test_measure_malformed_yaml_raises_with_path(synthetic_corpus: Path, tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("queries: [{id: q1, query: x, expected_in_top_3: ]", encoding="utf-8")
    with pytest.raises(ValueError, match="bad.yaml"):
        measure(corpus_path=synthetic_corpus, queries_path=bad)


def test_measure_missing_queries_key_raises(synthetic_corpus: Path, tmp_path: Path) -> None:
    bad = tmp_path / "no_key.yaml"
    bad.write_text("not_queries: []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="queries"):
        measure(corpus_path=synthetic_corpus, queries_path=bad)


def test_to_json_includes_aggregate_and_per_query(
    synthetic_corpus: Path, synthetic_queries: Path
) -> None:
    import json as _json

    result = measure(corpus_path=synthetic_corpus, queries_path=synthetic_queries)
    payload = _json.loads(result.to_json())
    assert payload["aggregate"] == {"p1": 1.0, "r3": 1.0, "n": 3}
    assert len(payload["per_query"]) == 3
    assert payload["rerank"] is False


def test_extra_aliases_file_passes_through(
    synthetic_corpus: Path, synthetic_queries: Path, tmp_path: Path
) -> None:
    """Smoke: the kwarg reaches DirectoryCorpus (we verified the
    DirectoryCorpus side in test_directory_corpus_extra_aliases_file)."""
    import json as _json

    aliases = tmp_path / "aliases.json"
    aliases.write_text(_json.dumps({"alpha.md": ["alpha extra alias"]}), encoding="utf-8")
    # Should not raise; the alias is merged into the corpus
    result = measure(
        corpus_path=synthetic_corpus,
        queries_path=synthetic_queries,
        extra_aliases_file=aliases,
    )
    assert result.p1 == 1.0
