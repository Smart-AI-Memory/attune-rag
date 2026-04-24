"""Tests for attune_rag.dashboard.refresh."""
from __future__ import annotations

import json
from unittest.mock import patch

from attune_rag.dashboard.refresh import build_snapshot, main

# ── helpers ──────────────────────────────────────────────────────────────────

def _fake_freshness(corpus_package: str = "attune_help") -> dict:
    return {
        f"{corpus_package}_version": "1.2.3",
        "summaries_by_path_keys": 42,
        "kinds": ["concepts", "references"],
        "kind_totals": {"concepts": 10, "references": 5},
        "features": ["security-audit", "smart-test"],
        "per_feature": {
            "security-audit": {"total": 3, "by_kind": {"concepts": 2, "references": 1}},
            "smart-test": {"total": 2, "by_kind": {"concepts": 2, "references": 0}},
        },
    }


def _fake_queries() -> list[dict]:
    return [
        {
            "id": "q1",
            "query": "how do I run a security audit",
            "difficulty": "easy",
            "expected_feature": "security-audit",
            "expected_in_top_3": ["concepts/tool-security-audit.md"],
        },
        {
            "id": "q2",
            "query": "generate tests",
            "difficulty": "medium",
            "expected_feature": "smart-test",
            "expected_in_top_3": ["concepts/tool-smart-test.md"],
        },
    ]


def _fake_bench() -> dict:
    return {
        "retriever": "FakeRetriever",
        "corpus": "fake-corpus",
        "total_queries": 2,
        "precision_at_1": 1.0,
        "recall_at_k": 1.0,
        "k": 3,
        "mean_latency_ms": 5.0,
        "max_latency_ms": 8.0,
        "per_query": [
            {
                "id": "q1",
                "query": "how do I run a security audit",
                "difficulty": "easy",
                "expected": ["concepts/tool-security-audit.md"],
                "actual": ["concepts/tool-security-audit.md"],
                "top1_match": True,
                "topk_match": True,
            },
            {
                "id": "q2",
                "query": "generate tests",
                "difficulty": "medium",
                "expected": ["concepts/tool-smart-test.md"],
                "actual": ["concepts/tool-smart-test.md"],
                "top1_match": True,
                "topk_match": True,
            },
        ],
    }


# ── tests ─────────────────────────────────────────────────────────────────────

def test_missing_queries_yaml_returns_partial(tmp_path):
    snap = build_snapshot(queries_path=tmp_path / "nonexistent.yaml")
    assert "error" in snap["retrieval"]
    assert "timestamp" in snap
    assert snap["freshness"] == {}


def test_missing_queries_yaml_exit_code_1(tmp_path, capsys):
    with patch(
        "attune_rag.dashboard.refresh._default_queries_path",
        return_value=tmp_path / "no.yaml",
    ):
        rc = main()
    out = capsys.readouterr().out
    start = out.find("{")
    assert start >= 0, "JSON must still be emitted on error"
    data = json.loads(out[start:])
    assert "error" in data["retrieval"]
    assert rc == 1


def test_snapshot_shape(tmp_path):
    queries_yaml = tmp_path / "queries.yaml"
    queries_yaml.write_text("queries: []")

    with (
        patch("attune_rag.dashboard.refresh._load_queries", return_value=_fake_queries()),
        patch("attune_rag.dashboard.refresh._run_benchmark", return_value=_fake_bench()),
        patch("attune_rag.dashboard.refresh._freshness_section", return_value=_fake_freshness()),
    ):
        snap = build_snapshot(queries_path=queries_yaml)

    assert "timestamp" in snap
    r = snap["retrieval"]
    assert r["retriever"] == "FakeRetriever"
    assert r["precision_at_1"] == 1.0
    assert r["recall_at_k"] == 1.0
    assert "per_difficulty" in r
    assert "per_feature" in r
    assert "per_query" in r
    assert len(r["per_query"]) == 2

    f = snap["freshness"]
    assert "attune_help_version" in f
    assert f["attune_help_version"] == "1.2.3"
    assert "summaries_by_path_keys" in f


def test_corpus_package_override(tmp_path):
    queries_yaml = tmp_path / "queries.yaml"
    queries_yaml.write_text("queries: []")

    def fake_freshness(corpus_package, queries):
        return _fake_freshness(corpus_package)

    with (
        patch("attune_rag.dashboard.refresh._load_queries", return_value=_fake_queries()),
        patch("attune_rag.dashboard.refresh._run_benchmark", return_value=_fake_bench()),
        patch("attune_rag.dashboard.refresh._freshness_section", side_effect=fake_freshness),
    ):
        snap = build_snapshot(corpus_package="my_custom_pkg", queries_path=queries_yaml)

    assert "my_custom_pkg_version" in snap["freshness"]
    assert snap["freshness"]["my_custom_pkg_version"] == "1.2.3"


def test_stdout_contract_starts_with_brace(tmp_path, capsys):
    with patch(
        "attune_rag.dashboard.refresh._default_queries_path",
        return_value=tmp_path / "no.yaml",
    ):
        main()
    out = capsys.readouterr().out
    brace_pos = out.find("{")
    assert brace_pos >= 0
    json.loads(out[brace_pos:])  # must be valid JSON from first {
