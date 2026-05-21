"""Diagnostic-2 — QueryExpander robustness on paraphrased queries.

Tests the second cheap-alternative hypothesis: can the existing
`QueryExpander` (Haiku-backed) close the paraphrase gap without any
new dependency? If yes, the embedding-retriever spec defers.

Approach:
  1. Load baseline (`queries.yaml`) and paraphrased (`queries_paraphrased.yaml`).
  2. For each query, call `QueryExpander.expand()` → list of synonym
     phrasings.
  3. Build the expanded query as `original + " " + " ".join(expansions)`
     — exactly the way `RagPipeline._retrieve` does it (pipeline.py:166-169).
  4. Run `KeywordRetriever` against the expanded query.
  5. Report P@1 / R@3 on both query sets, compare to D1's keyword-only
     numbers (baseline 97.5% / 100%, paraphrased 11.25% / 28.75%).

Cost: 120 Haiku calls total (40 baseline + 80 paraphrased), ~200
output tokens each, with system-prompt caching. Estimated total cost:
< $0.05 at current Haiku pricing.

Requirements:
- `ANTHROPIC_API_KEY` set in the environment.
- `attune-rag[claude]` extra installed (anthropic SDK).

Usage (recommended — your terminal, your env):
    export ANTHROPIC_API_KEY=<your key>   # or load from your secret store
    python docs/specs/embedding-retriever/run_diagnostic_2.py

The script writes expansion results to a JSON cache (`diagnostic-2.cache.json`)
so reruns are free and idempotent. Delete the cache file to re-spend tokens.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from attune_rag.corpus.attune_help import AttuneHelpCorpus
from attune_rag.retrieval import KeywordRetriever

_CACHE_PATH = Path(__file__).parent / "diagnostic-2.cache.json"


def _load_queries(path: Path) -> list[dict[str, Any]]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["queries"]


def _load_cache() -> dict[str, list[str]]:
    if _CACHE_PATH.is_file():
        try:
            return json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


def _save_cache(cache: dict[str, list[str]]) -> None:
    _CACHE_PATH.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")


def _score(
    queries: list[dict[str, Any]],
    expansions_by_query: dict[str, list[str]],
    retriever: KeywordRetriever,
    corpus: AttuneHelpCorpus,
) -> dict[str, dict[str, Any]]:
    by_diff: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"total": 0, "p1": 0, "r3": 0, "misses": []}
    )
    for q in queries:
        diff = q.get("difficulty", "unknown")
        expected = set(q["expected_in_top_3"])
        original = q["query"]
        expansions = expansions_by_query.get(original, [])
        # Mirror RagPipeline._retrieve: join original + expansions.
        if expansions:
            retrieval_query = " ".join([original, *expansions])
        else:
            retrieval_query = original
        hits = retriever.retrieve(retrieval_query, corpus, k=3)
        hit_paths = [h.entry.path for h in hits]
        by_diff[diff]["total"] += 1
        if hit_paths and hit_paths[0] in expected:
            by_diff[diff]["p1"] += 1
        if expected & set(hit_paths):
            by_diff[diff]["r3"] += 1
        else:
            by_diff[diff]["misses"].append((q["id"], original, hit_paths))
    return dict(by_diff)


def _render(stats: dict[str, dict[str, Any]], label: str) -> tuple[str, float, float]:
    lines = [f"### {label}\n", "| difficulty | n | P@1 | R@3 |", "|---|---|---|---|"]
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


def _expand_all(
    queries_lists: list[list[dict[str, Any]]],
    cache: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Call QueryExpander for each unique query, persisting to cache."""
    from attune_rag.expander import QueryExpander

    expander = QueryExpander()
    seen: set[str] = set()
    to_expand: list[str] = []
    for queries in queries_lists:
        for q in queries:
            query = q["query"]
            if query in seen:
                continue
            seen.add(query)
            if query not in cache:
                to_expand.append(query)

    if to_expand:
        print(f"Expanding {len(to_expand)} queries via Haiku...", file=sys.stderr)
        start = time.time()
        for i, query in enumerate(to_expand, 1):
            expansions = expander.expand(query)
            cache[query] = expansions
            if i % 10 == 0 or i == len(to_expand):
                _save_cache(cache)  # save progress
                elapsed = time.time() - start
                print(f"  {i}/{len(to_expand)} done ({elapsed:.1f}s elapsed)", file=sys.stderr)
        _save_cache(cache)
    else:
        print("All expansions found in cache; no API calls needed.", file=sys.stderr)

    return cache


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "ERROR: ANTHROPIC_API_KEY is not set. Set it in your environment "
            "(e.g. `export ANTHROPIC_API_KEY=...`) and rerun.",
            file=sys.stderr,
        )
        return 2

    baseline_path = _ROOT / "tests" / "golden" / "queries.yaml"
    paraphrased_path = _ROOT / "tests" / "golden" / "queries_paraphrased.yaml"
    baseline = _load_queries(baseline_path)
    paraphrased = _load_queries(paraphrased_path)

    cache = _load_cache()
    expansions = _expand_all([baseline, paraphrased], cache)

    corpus = AttuneHelpCorpus.from_attune_help()
    retriever = KeywordRetriever()

    print("# Diagnostic-2 — QueryExpander robustness\n")
    print(f"corpus: {corpus.name}  version: {corpus.version}")
    print(f"baseline queries: {len(baseline)}; paraphrased: {len(paraphrased)}")
    print(f"unique queries expanded: {len(expansions)}")
    print(f"cache: {_CACHE_PATH.relative_to(_ROOT)}\n")

    # Baseline — sanity check that expansion doesn't regress.
    print("## Baseline (`queries.yaml`)\n")
    b_stats = _score(baseline, expansions, retriever, corpus)
    b_table, b_p1, b_r3 = _render(b_stats, "Keyword + QueryExpander")
    print(b_table)
    print()
    print("**Compare to D1 keyword-only baseline:** P@1 97.50%, R@3 100.00%")
    print()

    # Paraphrased — the real test.
    print("## Paraphrased (`queries_paraphrased.yaml`)\n")
    p_stats = _score(paraphrased, expansions, retriever, corpus)
    p_table, p_p1, p_r3 = _render(p_stats, "Keyword + QueryExpander")
    print(p_table)
    print()
    print("**Compare to D1 keyword-only paraphrased:** P@1 11.25%, R@3 28.75%")
    print()

    # Verdict.
    delta_p1 = p_p1 - 0.1125  # D1 keyword-only baseline on paraphrased
    delta_r3 = p_r3 - 0.2875
    print(f"Δ P@1 vs keyword-only on paraphrased: {delta_p1:+.2%}")
    print(f"Δ R@3 vs keyword-only on paraphrased: {delta_r3:+.2%}")
    print()

    print("## Verdict\n")
    if p_r3 >= 0.75:
        verdict = (
            "**STRONG** — QueryExpander alone closes most of the paraphrase gap. "
            "Combined with the alias-expansion lever (D3), embeddings are very likely "
            "unnecessary. Spec defers; follow-up is to recommend QueryExpander in docs "
            "and consider making it default for users with `[claude]` installed."
        )
    elif p_r3 >= 0.55:
        verdict = (
            "**MIXED** — QueryExpander closes meaningful ground but not the full gap. "
            "Combine with the alias sweep (D3 generalization) to check if the two together "
            "close it. If they do, embeddings still defer."
        )
    else:
        verdict = (
            "**WEAK** — QueryExpander doesn't substantially close the gap. The expansions "
            "Haiku produces may be drifting toward feature-name vocabulary that doesn't help "
            "with semantic-only queries. Embedding case stands; alias-authoring sweep remains "
            "the primary defer-lever."
        )
    print(verdict)
    print()

    # Persist final stats to a sibling file for transparency.
    summary_path = Path(__file__).parent / "diagnostic-2.summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "baseline": {"p1": b_p1, "r3": b_r3, "n": len(baseline)},
                "paraphrased": {"p1": p_p1, "r3": p_r3, "n": len(paraphrased)},
                "d1_keyword_only_paraphrased": {"p1": 0.1125, "r3": 0.2875},
                "delta_vs_d1_paraphrased": {"p1": delta_p1, "r3": delta_r3},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Summary stats written to: {summary_path.relative_to(_ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
