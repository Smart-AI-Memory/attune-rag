"""Diagnostic-1 — KeywordRetriever robustness to paraphrasing.

Runs `KeywordRetriever` against both the baseline golden queries and a
no-token-overlap paraphrase set, reports P@1 / R@3 per difficulty, and
prints a verdict against the decision matrix in `diagnostic-1.md`.

Pure stdlib + PyYAML (already a dev dep). No API calls.

Usage:
    python docs/specs/embedding-retriever/run_diagnostic_1.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from attune_rag.corpus.attune_help import AttuneHelpCorpus
from attune_rag.retrieval import KeywordRetriever


def _load_queries(path: Path) -> list[dict[str, Any]]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["queries"]


def _score(
    queries: list[dict[str, Any]],
    retriever: KeywordRetriever,
    corpus: AttuneHelpCorpus,
) -> dict[str, dict[str, int]]:
    by_diff: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "p1": 0, "r3": 0})
    for q in queries:
        diff = q.get("difficulty", "unknown")
        expected = set(q["expected_in_top_3"])
        hits = retriever.retrieve(q["query"], corpus, k=3)
        hit_paths = [h.entry.path for h in hits]
        by_diff[diff]["total"] += 1
        if hit_paths and hit_paths[0] in expected:
            by_diff[diff]["p1"] += 1
        if expected & set(hit_paths):
            by_diff[diff]["r3"] += 1
    return dict(by_diff)


def _render(stats: dict[str, dict[str, int]]) -> tuple[str, float, float]:
    lines = ["| difficulty | n | P@1 | R@3 |", "|---|---|---|---|"]
    totals = {"total": 0, "p1": 0, "r3": 0}
    for diff in ("easy", "medium", "hard"):
        s = stats.get(diff)
        if not s:
            continue
        totals["total"] += s["total"]
        totals["p1"] += s["p1"]
        totals["r3"] += s["r3"]
        p1 = s["p1"] / s["total"] if s["total"] else 0.0
        r3 = s["r3"] / s["total"] if s["total"] else 0.0
        lines.append(f"| {diff} | {s['total']} | {p1:.2%} | {r3:.2%} |")
    overall_p1 = totals["p1"] / totals["total"] if totals["total"] else 0.0
    overall_r3 = totals["r3"] / totals["total"] if totals["total"] else 0.0
    lines.append(
        f"| **overall** | **{totals['total']}** | " f"**{overall_p1:.2%}** | **{overall_r3:.2%}** |"
    )
    return "\n".join(lines), overall_p1, overall_r3


def _verdict(delta_p1: float) -> str:
    mag = abs(delta_p1)
    if mag <= 0.05:
        return (
            "**WEAK** — keyword is robust to paraphrasing. EmbeddingRetriever ROI is low. "
            "Defer the build, revisit only if the corpus shifts toward content where "
            "synonyms diverge from path/summary tokens."
        )
    if mag <= 0.15:
        return (
            "**MIXED** — keyword degrades materially but not catastrophically. "
            "Investigate **hybrid** (keyword + embedding blend with a small α). "
            "Pure embedding replacement is still uncertain; scope a hybrid prototype before "
            "committing to a full retriever."
        )
    return (
        "**STRONG** — keyword is brittle to paraphrasing. **EmbeddingRetriever case is real.** "
        "Proceed to design a spec: local-model first (no API dep), hybrid scoring, and a "
        "benchmark that includes this paraphrase set."
    )


def _per_query_misses(
    queries: list[dict[str, Any]],
    retriever: KeywordRetriever,
    corpus: AttuneHelpCorpus,
) -> list[tuple[str, str, list[str], list[str]]]:
    misses: list[tuple[str, str, list[str], list[str]]] = []
    for q in queries:
        expected = set(q["expected_in_top_3"])
        hits = retriever.retrieve(q["query"], corpus, k=3)
        hit_paths = [h.entry.path for h in hits]
        if not (expected & set(hit_paths)):
            misses.append((q["id"], q["query"], sorted(expected), hit_paths))
    return misses


def main() -> int:
    baseline_path = _ROOT / "tests" / "golden" / "queries.yaml"
    paraphrased_path = _ROOT / "tests" / "golden" / "queries_paraphrased.yaml"

    baseline = _load_queries(baseline_path)
    paraphrased = _load_queries(paraphrased_path)

    corpus = AttuneHelpCorpus.from_attune_help()
    retriever = KeywordRetriever()

    b_stats = _score(baseline, retriever, corpus)
    p_stats = _score(paraphrased, retriever, corpus)

    b_table, b_p1, b_r3 = _render(b_stats)
    p_table, p_p1, p_r3 = _render(p_stats)

    print("# Diagnostic-1 — KeywordRetriever paraphrase robustness\n")
    print(f"corpus: {corpus.name}  version: {corpus.version}\n")

    print("## Baseline (original golden queries)\n")
    print(b_table)
    print()

    print("## Paraphrased (no-token-overlap variants)\n")
    print(p_table)
    print()

    delta_p1 = p_p1 - b_p1
    delta_r3 = p_r3 - b_r3
    print("## Deltas\n")
    print(f"- Δ P@1 (overall): {delta_p1:+.2%}")
    print(f"- Δ R@3 (overall): {delta_r3:+.2%}")
    print()

    print("## Verdict\n")
    print(_verdict(delta_p1))
    print()

    misses = _per_query_misses(paraphrased, retriever, corpus)
    if misses:
        print(f"## Paraphrased R@3 misses ({len(misses)})\n")
        for qid, query, expected, hit_paths in misses:
            print(f"- **{qid}** — `{query}`")
            print(f"  - expected: {expected}")
            print(f"  - got: {hit_paths}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
