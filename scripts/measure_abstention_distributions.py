"""M1/M2 measurement for docs/specs/safe-abstention-defaults.

Promotes the authoring-time abstention probe (design.md §2) to a
committed, reproducible script (requirement R5). Three sections:

1. **Distributions** — with ``KeywordRetriever(min_score=0)`` so
   nothing is filtered, record per query the raw top-1 score, the
   per-token ratio (top1 / query-token-count), and the top1−top2 gap,
   over legit and out-of-corpus negative sets on both the bundled
   attune-help corpus and the lean ``corpus_b`` fixture. Emits the
   design.md §2 medians table.
2. **Calibration (M1)** — re-derives the bundled corpus's recommended
   ``min_score`` via the same sweep as
   ``attune-rag-benchmark --calibrate-abstention``.
3. **C3 heuristics (M2)** — sweeps the calibration-free relative
   criteria (per-token τ, top1−top2 gap δ, and an absolute-OR-relative
   combination) on both corpora's legit + negative sets and reports
   whether any variant clears the both-corpora legit-recall bar
   (>= 95% legit kept on BOTH corpora, per the calibration
   convention), including the single-token-query risk case (risk §6).

The negative set is the bundled ``queries_negative.yaml`` — its
queries (cooking, geography, k8s, terraform, …) are out-of-corpus for
both fixtures, so it serves both.

Keyword-only, offline, deterministic. No API, no torch.

Usage:
    PYTHONPATH=src python3 scripts/measure_abstention_distributions.py
    PYTHONPATH=src python3 scripts/measure_abstention_distributions.py \
        --out docs/specs/safe-abstention-defaults/measurements.md
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from statistics import median
from typing import Any

import structlog

from attune_rag.benchmark import _calibrate_abstention, _load_queries
from attune_rag.corpus import AttuneHelpCorpus, DirectoryCorpus
from attune_rag.retrieval import KeywordRetriever, _tokenize

GOLDEN = Path(__file__).resolve().parent.parent / "tests" / "golden"

# Legit-kept floor a calibration-free heuristic must hold on BOTH
# corpora to ship as the no-eval-data default (matches the
# target_legit_kept default in _calibrate_abstention).
LEGIT_BAR = 0.95
# ...and it must actually buy protection: a variant "clears the bar"
# only if, at some threshold holding LEGIT_BAR on both corpora, at
# least half the out-of-corpus negatives abstain on BOTH corpora.
# Below that the heuristic is a no-op wearing a default's clothes.
MIN_USEFUL_ABSTENTION = 0.5


def query_stats(queries: list[dict[str, Any]], corpus: Any) -> list[dict[str, float]]:
    """Per-query abstention signals with filtering disabled.

    Returns one row per query: raw ``top1`` score, ``per_token``
    (top1 / query-token-count), ``gap`` (top1 − top2, = top1 when only
    one candidate), and ``n_tokens``. Queries with no candidates at
    all score 0.0 across the board.
    """
    retriever = KeywordRetriever(min_score=0.0)
    rows: list[dict[str, float]] = []
    for entry in queries:
        query = entry["query"]
        n_tokens = len(_tokenize(query))
        hits = retriever.retrieve(query, corpus, k=2) if n_tokens else []
        top1 = hits[0].score if hits else 0.0
        top2 = hits[1].score if len(hits) > 1 else 0.0
        rows.append(
            {
                "top1": top1,
                "per_token": top1 / n_tokens if n_tokens else 0.0,
                "gap": top1 - top2,
                "n_tokens": float(n_tokens),
            }
        )
    return rows


def _medians(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: median(r[key] for r in rows) for key in ("top1", "per_token", "gap")}


def distribution_table(stats: dict[str, list[dict[str, float]]]) -> list[str]:
    """The design.md §2 medians table, from live measurements."""
    med = {name: _medians(rows) for name, rows in stats.items()}
    lines = [
        "| statistic (median) | help legit | help neg | corpus_b legit | corpus_b neg |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, label in [
        ("top1", "raw top-1"),
        ("per_token", "top-1 / query-token"),
        ("gap", "top-1 − top-2 gap"),
    ]:
        cells = [
            f"{med[name][key]:.2f}" for name in ("help_legit", "help_neg", "cb_legit", "cb_neg")
        ]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return lines


def sweep_heuristic(
    legit: list[dict[str, float]],
    negs: list[dict[str, float]],
    keep: Any,
    grid: list[float],
) -> list[dict[str, float]]:
    """Sweep a keep-predicate ``keep(row, t)`` over ``grid``.

    Returns per-threshold legit-kept and negatives-abstained fractions.
    """
    out = []
    for t in grid:
        kept = sum(keep(r, t) for r in legit) / len(legit)
        abstained = sum(not keep(r, t) for r in negs) / len(negs)
        out.append({"threshold": t, "legit_kept": kept, "negatives_abstained": abstained})
    return out


def best_both_corpora(
    sweeps: dict[str, list[dict[str, float]]],
) -> dict[str, Any] | None:
    """Best threshold holding the legit bar on BOTH corpora.

    ``sweeps`` maps corpus name → sweep rows (same grid, same order).
    Eligible thresholds keep >= LEGIT_BAR legit on every corpus; the
    winner maximizes the minimum negatives-abstained across corpora.
    Returns None when no threshold is eligible — the heuristic fails
    the bar.
    """
    names = list(sweeps)
    eligible: list[dict[str, Any]] = []
    for i, row in enumerate(sweeps[names[0]]):
        per_corpus = {n: sweeps[n][i] for n in names}
        if all(r["legit_kept"] >= LEGIT_BAR for r in per_corpus.values()):
            eligible.append(
                {
                    "threshold": row["threshold"],
                    "min_negatives_abstained": min(
                        r["negatives_abstained"] for r in per_corpus.values()
                    ),
                    "per_corpus": per_corpus,
                }
            )
    if not eligible:
        return None
    return max(eligible, key=lambda e: e["min_negatives_abstained"])


def c3_verdict(stats: dict[str, list[dict[str, float]]]) -> list[str]:
    """M2: measure the C3 candidate heuristics; emit the report lines."""
    variants = {
        "per-token (top1/n_tokens >= τ)": (
            lambda r, t: r["per_token"] >= t,
            [round(0.1 * i, 1) for i in range(0, 31)],
        ),
        "gap (top1 − top2 >= δ)": (
            lambda r, t: r["gap"] >= t,
            [round(0.25 * i, 2) for i in range(0, 41)],
        ),
        # Absolute floor at the current class default OR the per-token
        # criterion — "keep if plainly strong or relatively strong".
        "combination (top1 >= 5 or per-token >= τ)": (
            lambda r, t: r["top1"] >= 5.0 or r["per_token"] >= t,
            [round(0.1 * i, 1) for i in range(0, 31)],
        ),
    }
    lines = ["", "## M2 — C3 relative-heuristic sweep (decides Q3)", ""]
    lines.append(f"Legit-recall bar: >= {LEGIT_BAR:.0%} kept on BOTH corpora.")
    lines.append("")
    for name, (keep, grid) in variants.items():
        sweeps = {
            "help": sweep_heuristic(stats["help_legit"], stats["help_neg"], keep, grid),
            "corpus_b": sweep_heuristic(stats["cb_legit"], stats["cb_neg"], keep, grid),
        }
        best = best_both_corpora(sweeps)
        lines.append(f"### {name}")
        if best is not None and best["min_negatives_abstained"] < MIN_USEFUL_ABSTENTION:
            lines.append(
                f"- **FAILS** — best eligible threshold ({best['threshold']}) "
                f"abstains only {best['min_negatives_abstained']:.1%} of "
                "negatives (min across corpora); the legit-kept-eligible "
                "range provides no useful separation."
            )
            lines.append("")
            continue
        if best is None:
            lines.append(
                "- **FAILS the bar** — no threshold keeps "
                f">= {LEGIT_BAR:.0%} legit on both corpora."
            )
            lines.append("")
            continue
        lines.append(f"- best threshold: **{best['threshold']}**")
        for corpus_name, row in best["per_corpus"].items():
            lines.append(
                f"  - {corpus_name}: legit kept {row['legit_kept']:.1%}, "
                f"negatives abstained {row['negatives_abstained']:.1%}"
            )
        lines.append(
            f"- min negatives-abstained across corpora: "
            f"**{best['min_negatives_abstained']:.1%}**"
        )
        lines.append("")
    # Risk §6: how many legit queries are single-token post-tokenize?
    for name in ("help_legit", "cb_legit"):
        singles = [r for r in stats[name] if r["n_tokens"] <= 1]
        lines.append(f"Single-token legit queries in {name}: {len(singles)}")
    return lines


def build_report() -> str:
    help_corpus = AttuneHelpCorpus.from_attune_help()
    corpus_b = DirectoryCorpus(GOLDEN / "corpus_b")

    help_legit = _load_queries(GOLDEN / "queries.yaml")
    cb_legit = _load_queries(GOLDEN / "queries_corpus_b.yaml")
    negatives = _load_queries(GOLDEN / "queries_negative.yaml")

    stats = {
        "help_legit": query_stats(help_legit, help_corpus),
        "help_neg": query_stats(negatives, help_corpus),
        "cb_legit": query_stats(cb_legit, corpus_b),
        "cb_neg": query_stats(negatives, corpus_b),
    }

    sizes = ", ".join(f"{name} n={len(rows)}" for name, rows in stats.items())
    lines = [
        "# safe-abstention-defaults — measured distributions",
        "",
        "Generated by `scripts/measure_abstention_distributions.py`",
        "(PYTHONPATH=src, keyword-only, offline). Regenerate with the",
        "`--out` flag; do not hand-edit.",
        "",
        f"Set sizes: {sizes}.",
        f"NB: at n=11 the {LEGIT_BAR:.0%} legit bar allows zero drops",
        "(10/11 = 90.9%) — the corpus_b bar is effectively 100%.",
        "",
        "## M1 — score distributions (design.md §2, re-derived)",
        "",
        *distribution_table(stats),
        "",
        "## M1 — bundled-corpus calibration (re-derived)",
        "",
    ]
    cal = _calibrate_abstention(help_legit, negatives, k=3, corpus=help_corpus)
    lines.append(
        f"Recommended bundled `min_score` = **{cal['recommended_threshold']:.0f}** "
        f"(legit kept {cal['recommended_legit_kept']:.0%}, "
        f"false-answer rate {cal['recommended_false_answer_rate']:.0%}, "
        f"target legit-kept >= {cal['target_legit_kept']:.0%})."
    )
    # R1 tension: the calibration targets 95% legit-kept, but the spec's
    # R1 binds the *chosen* default to 100% on the gate set. Report the
    # strictest R1-clean threshold and name the gate queries the
    # recommended (95%-target) threshold would drop — each is an
    # alias-remediation candidate, not a gate-set edit (R6).
    r1_rows = [r for r in cal["rows"] if r["legit_kept"] == 1.0]
    r1_best = max(r1_rows, key=lambda r: r["threshold"])
    lines.extend(
        [
            "",
            "### R1-strict view (gate set must hold 100%)",
            "",
            f"Max `min_score` keeping **100%** of the gate set = "
            f"**{r1_best['threshold']:.0f}** "
            f"(negatives abstained {r1_best['negatives_abstained']:.1%}).",
        ]
    )
    t_rec = cal["recommended_threshold"]
    dropped = [
        (entry["query"], row["top1"])
        for entry, row in zip(help_legit, stats["help_legit"], strict=False)
        if row["top1"] < t_rec
    ]
    if dropped:
        lines.append(
            f"Gate queries the recommended T={t_rec:.0f} would drop "
            "(alias-remediation candidates):"
        )
        lines.extend(f"- `{q}` (top-1 score {s:g})" for q, s in dropped)
    lines.extend(c3_verdict(stats))
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="also write the report to this markdown file",
    )
    args = parser.parse_args()
    # The calibration sweep drives RagPipeline.run per query; silence
    # its per-query info logs so stdout is just the report.
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))
    report = build_report()
    print(report)
    if args.out:
        args.out.write_text(report, encoding="utf-8")
        print(f"[written] {args.out}")


if __name__ == "__main__":
    main()
