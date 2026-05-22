"""Unit tests for scripts/measure_corpus.py.

Fast — uses a synthetic 3-doc DirectoryCorpus; no Anthropic calls,
no bundled-corpus dependency. The byte-identical bundled-baseline
reproduction lives in ``tests/golden/test_measure_corpus_bundled.py``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "measure_corpus.py"


@pytest.fixture
def synthetic_corpus(tmp_path: Path) -> Path:
    """Build a 3-doc markdown corpus with distinct aliases."""
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "alpha.md").write_text(
        "---\naliases: [alpha-feature, alpha-thing]\n---\n# Alpha\n\nThe alpha feature.\n",
        encoding="utf-8",
    )
    (root / "beta.md").write_text(
        "---\naliases: [beta-feature, beta-widget]\n---\n# Beta\n\nThe beta feature.\n",
        encoding="utf-8",
    )
    (root / "gamma.md").write_text(
        "---\naliases: [gamma-feature, gamma-tool]\n---\n# Gamma\n\nThe gamma feature.\n",
        encoding="utf-8",
    )
    return root


@pytest.fixture
def synthetic_queries(tmp_path: Path) -> Path:
    """A tiny query YAML that resolves cleanly against synthetic_corpus."""
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
                    "difficulty": "easy",
                    "expected_in_top_3": ["gamma.md"],
                },
            ]
        },
        path.open("w", encoding="utf-8"),
    )
    return path


def _run_script(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
    )


def test_score_queries_basic(synthetic_corpus: Path, synthetic_queries: Path) -> None:
    """score_queries returns expected per-query + aggregate shape."""
    from attune_rag import RagPipeline
    from attune_rag._scoring import score_queries
    from attune_rag.corpus import DirectoryCorpus

    pipeline = RagPipeline(corpus=DirectoryCorpus(synthetic_corpus))
    with synthetic_queries.open(encoding="utf-8") as fp:
        queries = yaml.safe_load(fp)["queries"]
    per_query, agg = score_queries(pipeline, queries, k=3)

    assert agg.n == 3
    assert agg.p1_hits == 3
    assert agg.r3_hits == 3
    assert agg.p1 == 1.0
    assert agg.r3 == 1.0
    assert [q.qid for q in per_query] == ["q1", "q2", "q3"]
    assert all(q.p1 and q.r3 for q in per_query)


def test_cli_writes_report(tmp_path: Path, synthetic_corpus: Path, synthetic_queries: Path) -> None:
    out = tmp_path / "report.md"
    result = _run_script(
        "--corpus-path",
        str(synthetic_corpus),
        "--queries",
        str(synthetic_queries),
        "--output",
        str(out),
        "--frozen-timestamp",
        "2026-05-22T00:00:00Z",
    )
    assert result.returncode == 0, result.stderr
    assert out.is_file()
    text = out.read_text(encoding="utf-8")
    assert "# Corpus measurement report" in text
    assert "1.0000" in text
    assert "Timestamp: `2026-05-22T00:00:00Z`" in text


def test_cli_deterministic_with_frozen_timestamp(
    tmp_path: Path, synthetic_corpus: Path, synthetic_queries: Path
) -> None:
    """Two runs with the same --frozen-timestamp produce byte-identical reports."""
    out_a = tmp_path / "a.md"
    out_b = tmp_path / "b.md"
    common = [
        "--corpus-path",
        str(synthetic_corpus),
        "--queries",
        str(synthetic_queries),
        "--frozen-timestamp",
        "2026-05-22T00:00:00Z",
    ]
    assert _run_script(*common, "--output", str(out_a)).returncode == 0
    assert _run_script(*common, "--output", str(out_b)).returncode == 0
    assert out_a.read_bytes() == out_b.read_bytes()


def test_cli_watermark_r3_fail(
    tmp_path: Path, synthetic_corpus: Path, synthetic_queries: Path
) -> None:
    """Watermark above achievable R@3 forces non-zero exit."""
    out = tmp_path / "report.md"
    result = _run_script(
        "--corpus-path",
        str(synthetic_corpus),
        "--queries",
        str(synthetic_queries),
        "--output",
        str(out),
        "--watermark-r3",
        "1.5",  # impossible; force failure
        "--frozen-timestamp",
        "2026-05-22T00:00:00Z",
    )
    assert result.returncode == 1
    assert "WATERMARK FAIL" in result.stderr


def test_cli_watermark_p1_pass(
    tmp_path: Path, synthetic_corpus: Path, synthetic_queries: Path
) -> None:
    """Explicit P@1 watermark at achievable level passes."""
    out = tmp_path / "report.md"
    result = _run_script(
        "--corpus-path",
        str(synthetic_corpus),
        "--queries",
        str(synthetic_queries),
        "--output",
        str(out),
        "--watermark-p1",
        "0.99",
        "--frozen-timestamp",
        "2026-05-22T00:00:00Z",
    )
    assert result.returncode == 0, result.stderr


def test_cli_paraphrased_set_appears_in_report(
    tmp_path: Path, synthetic_corpus: Path, synthetic_queries: Path
) -> None:
    """--paraphrased adds a second set + its per-query table."""
    out = tmp_path / "report.md"
    result = _run_script(
        "--corpus-path",
        str(synthetic_corpus),
        "--queries",
        str(synthetic_queries),
        "--paraphrased",
        str(synthetic_queries),  # reuse for the smoke
        "--output",
        str(out),
        "--frozen-timestamp",
        "2026-05-22T00:00:00Z",
    )
    assert result.returncode == 0, result.stderr
    text = out.read_text(encoding="utf-8")
    assert "## Per-query — baseline" in text
    assert "## Per-query — paraphrased" in text


def test_cli_malformed_queries_yaml(tmp_path: Path, synthetic_corpus: Path) -> None:
    """Malformed YAML surfaces with file path in the error."""
    bad = tmp_path / "bad.yaml"
    bad.write_text("queries: [{id: q1, query: x, expected_in_top_3: ]", encoding="utf-8")
    result = _run_script(
        "--corpus-path",
        str(synthetic_corpus),
        "--queries",
        str(bad),
        "--frozen-timestamp",
        "2026-05-22T00:00:00Z",
    )
    assert result.returncode != 0
    assert "bad.yaml" in result.stderr or "Malformed YAML" in result.stderr


def test_cli_rerank_footer_when_off(
    tmp_path: Path, synthetic_corpus: Path, synthetic_queries: Path
) -> None:
    """Footer advertises --with-rerank only in keyword-only mode."""
    out = tmp_path / "report.md"
    assert (
        _run_script(
            "--corpus-path",
            str(synthetic_corpus),
            "--queries",
            str(synthetic_queries),
            "--output",
            str(out),
            "--frozen-timestamp",
            "2026-05-22T00:00:00Z",
        ).returncode
        == 0
    )
    text = out.read_text(encoding="utf-8")
    assert "Run with `--with-rerank`" in text


def test_corpus_path_and_bundled_mutually_exclusive(
    tmp_path: Path, synthetic_corpus: Path, synthetic_queries: Path
) -> None:
    """Exactly one corpus source must be specified."""
    result = _run_script(
        "--corpus-path",
        str(synthetic_corpus),
        "--corpus-bundled",
        "--queries",
        str(synthetic_queries),
        "--frozen-timestamp",
        "2026-05-22T00:00:00Z",
    )
    assert result.returncode != 0
