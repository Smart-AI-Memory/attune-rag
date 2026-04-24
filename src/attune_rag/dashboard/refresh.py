"""Generate a JSON snapshot for the attune-rag dashboard.

Emits one JSON object to stdout. structlog may write lines before the JSON;
callers should parse from the first '{' in the combined output.

Exit codes: 0 complete snapshot, 1 partial (retrieval.error set), 2 unrecoverable.
"""
from __future__ import annotations

import datetime as _dt
import importlib
import importlib.resources as _ilr
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from attune_rag.benchmark import _default_queries_path, _load_queries, _run_benchmark


def _freshness_section(corpus_package: str, queries: list[dict[str, Any]]) -> dict[str, Any]:
    corpus_mod = importlib.import_module(corpus_package)
    corpus_version = getattr(corpus_mod, "__version__", "unknown")
    templates = Path(str(_ilr.files(corpus_package).joinpath("templates")))
    kinds = sorted(p.name for p in templates.iterdir() if p.is_dir())
    kind_totals = {k: len(list((templates / k).glob("*.md"))) for k in kinds}
    features = sorted({q.get("expected_feature", "") for q in queries} - {""})
    per_feat_corpus: dict[str, Any] = {}
    for feat in features:
        cov: dict[str, int] = {}
        total = 0
        for k in kinds:
            n = len(list((templates / k).glob(f"*{feat}*.md")))
            cov[k] = n
            total += n
        per_feat_corpus[feat] = {"total": total, "by_kind": cov}

    summaries_path = templates / "summaries_by_path.json"
    summ_keys = 0
    if summaries_path.exists():
        summ_keys = len(json.loads(summaries_path.read_text()))

    return {
        f"{corpus_package}_version": corpus_version,
        "summaries_by_path_keys": summ_keys,
        "kinds": kinds,
        "kind_totals": kind_totals,
        "features": features,
        "per_feature": per_feat_corpus,
    }


def build_snapshot(
    corpus_package: str = "attune_help",
    queries_path: Path | None = None,
) -> dict[str, Any]:
    """Return a dashboard snapshot dict. On missing queries.yaml returns partial with error."""
    if queries_path is None:
        queries_path = _default_queries_path()

    if not queries_path.is_file():
        return {
            "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "retrieval": {"error": f"queries.yaml not found: {queries_path}"},
            "freshness": {},
        }

    queries = _load_queries(queries_path)
    bench = _run_benchmark(queries, 3)
    q_meta = {q["id"]: q for q in queries}

    per_difficulty: dict[str, dict] = defaultdict(lambda: {"total": 0, "top1_hit": 0, "topk_hit": 0})
    per_feature: dict[str, dict] = defaultdict(lambda: {"total": 0, "top1_hit": 0, "topk_hit": 0})
    enriched = []
    for e in bench["per_query"]:
        m = q_meta.get(e["id"], {})
        diff = m.get("difficulty", "unknown")
        feat = m.get("expected_feature", "unknown")
        per_difficulty[diff]["total"] += 1
        per_feature[feat]["total"] += 1
        if e["top1_match"]:
            per_difficulty[diff]["top1_hit"] += 1
            per_feature[feat]["top1_hit"] += 1
        if e["topk_match"]:
            per_difficulty[diff]["topk_hit"] += 1
            per_feature[feat]["topk_hit"] += 1
        enriched.append(
            {
                "id": e["id"],
                "query": e["query"],
                "feature": feat,
                "difficulty": diff,
                "expected": e["expected"],
                "actual": e["actual"][:3],
                "top1_match": e["top1_match"],
                "topk_match": e["topk_match"],
            }
        )

    freshness = _freshness_section(corpus_package, queries)
    return {
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "retrieval": {
            "retriever": bench["retriever"],
            "corpus": bench["corpus"],
            "precision_at_1": bench["precision_at_1"],
            "recall_at_k": bench["recall_at_k"],
            "mean_latency_ms": bench["mean_latency_ms"],
            "max_latency_ms": bench["max_latency_ms"],
            "total_queries": bench["total_queries"],
            "k": bench["k"],
            "per_difficulty": dict(per_difficulty),
            "per_feature": dict(per_feature),
            "per_query": enriched,
        },
        "freshness": freshness,
    }


def main(corpus_package: str = "attune_help") -> int:
    snap = build_snapshot(corpus_package)
    print(json.dumps(snap))
    if "error" in snap.get("retrieval", {}):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
