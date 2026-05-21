"""Diagnostic-3 — Alias expansion on bug-predict.

Tests the cheap-alternative hypothesis: can hand-authored aliases close
most of the paraphrase gap for a single feature, removing the case for
embeddings *for that feature*?

Targets bug-predict because it had 9 of 80 paraphrased R@3 misses in
diagnostic-1 — the largest single-feature failure cluster.

Approach:
  1. Load the corpus as-is.
  2. For each bug-predict entry, reconstruct with augmented aliases
     (via dataclasses.replace; the entry is frozen).
  3. Wrap the corpus in a proxy that returns the augmented entries
     for bug-predict paths and delegates everything else.
  4. Re-run the paraphrase set through KeywordRetriever.
  5. Report bug-predict-specific delta + overall delta.

No new deps. No API calls. The aliases below are hand-authored under
one constraint: each alias is 2+ tokens (KeywordRetriever requires
MIN_ALIAS_OVERLAP=2 hits to credit the alias) and captures concept
language that appears in diagnostic-1's bug-predict miss list.

Usage:
    python docs/specs/embedding-retriever/run_diagnostic_3.py
"""

from __future__ import annotations

import dataclasses
import sys
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from attune_rag.corpus.attune_help import AttuneHelpCorpus
from attune_rag.corpus.base import CorpusProtocol, RetrievalEntry
from attune_rag.retrieval import KeywordRetriever

# Bug-predict entries in the corpus.
_BUG_PREDICT_PATHS = (
    "concepts/tool-bug-predict.md",
    "references/tool-bug-predict.md",
    "quickstarts/skill-bug-predict.md",
)

# Hand-authored alias additions. Drawn from concept language in
# diagnostic-1's bug-predict miss list. Each alias is 2-3 tokens so it
# can clear MIN_ALIAS_OVERLAP=2 against the paraphrased queries.
#
# Misses being targeted (paraphrased query → concept):
#   006a "landmines in this commit"           → commit risk / code landmines
#   014a "potentially harmful in my source"   → dangerous code / harmful patterns
#   015a "service fail silently"              → silent failures / hidden bugs
#   016b "weak points in my source"           → weak points / fragile code
#   027a "what part of this PR is dangerous"  → dangerous PR / PR risk
#   027b "worrisome spots in the diff"        → diff risk / worrisome code
#   036a "diff going to bite me"              → diff risk / code that bites
#   036b "what part of this commit is shaky"  → shaky code / commit risk
#   040b "what could go wrong once this is live" → production risk / what goes wrong
_BUG_PREDICT_EXTRA_ALIASES = (
    "dangerous code",
    "harmful patterns",
    "weak points",
    "fragile code",
    "code landmines",
    "danger zones",
    "what could go wrong",
    "silent failures",
    "hidden bugs",
    "PR risk review",
    "dangerous PR",
    "diff risk",
    "worrisome code",
    "shaky code",
    "commit risk",
    "production risk",
    "code that bites",
    "review risky changes",
)


def _augmented(entry: RetrievalEntry) -> RetrievalEntry:
    """Return a copy of *entry* with extra aliases appended.

    Uses ``dataclasses.replace`` because ``RetrievalEntry`` is frozen.
    The ``_tokens_cache`` field is reset so the keyword retriever
    re-tokenizes against the new alias set rather than reusing a stale
    cache from the original entry.
    """
    merged = tuple(entry.aliases) + _BUG_PREDICT_EXTRA_ALIASES
    return dataclasses.replace(entry, aliases=merged, _tokens_cache={})


class _PatchedCorpus:
    """Corpus proxy that overlays augmented bug-predict entries.

    Delegates everything else to the wrapped corpus. Satisfies
    ``CorpusProtocol`` structurally — duck-typed by ``KeywordRetriever``.
    """

    def __init__(self, inner: CorpusProtocol, overrides: dict[str, RetrievalEntry]) -> None:
        self._inner = inner
        self._overrides = overrides

    def entries(self) -> Iterable[RetrievalEntry]:
        for entry in self._inner.entries():
            yield self._overrides.get(entry.path, entry)

    def get(self, path: str) -> RetrievalEntry | None:
        if path in self._overrides:
            return self._overrides[path]
        return self._inner.get(path)

    @property
    def name(self) -> str:
        return f"{self._inner.name}+bug-predict-aliases"

    @property
    def version(self) -> str:
        return self._inner.version


def _load_queries(path: Path) -> list[dict[str, Any]]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["queries"]


def _is_bug_predict(query: dict[str, Any]) -> bool:
    return query.get("expected_feature") == "bug-predict"


def _score(
    queries: list[dict[str, Any]],
    retriever: KeywordRetriever,
    corpus: CorpusProtocol,
) -> dict[str, dict[str, Any]]:
    by_diff: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"total": 0, "p1": 0, "r3": 0, "misses": []}
    )
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
        else:
            by_diff[diff]["misses"].append((q["id"], q["query"], hit_paths))
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


def _render_bug_predict_only(
    queries: list[dict[str, Any]],
    retriever: KeywordRetriever,
    corpus: CorpusProtocol,
    label: str,
) -> tuple[str, int, int, int]:
    bp_queries = [q for q in queries if _is_bug_predict(q)]
    p1 = 0
    r3 = 0
    misses = []
    for q in bp_queries:
        expected = set(q["expected_in_top_3"])
        hits = retriever.retrieve(q["query"], corpus, k=3)
        hit_paths = [h.entry.path for h in hits]
        if hit_paths and hit_paths[0] in expected:
            p1 += 1
        if expected & set(hit_paths):
            r3 += 1
        else:
            misses.append((q["id"], q["query"], hit_paths))
    n = len(bp_queries)
    p1_pct = p1 / n if n else 0.0
    r3_pct = r3 / n if n else 0.0
    out = [f"### {label} — bug-predict only ({n} queries)\n"]
    out.append(f"- P@1: **{p1_pct:.2%}** ({p1}/{n})")
    out.append(f"- R@3: **{r3_pct:.2%}** ({r3}/{n})")
    if misses:
        out.append("")
        out.append(f"R@3 misses ({len(misses)}):")
        for qid, query, paths in misses:
            out.append(f"  - **{qid}** — `{query}` → {paths}")
    return "\n".join(out), n, p1, r3


def main() -> int:
    paraphrased_path = _ROOT / "tests" / "golden" / "queries_paraphrased.yaml"
    queries = _load_queries(paraphrased_path)

    base_corpus = AttuneHelpCorpus.from_attune_help()
    retriever = KeywordRetriever()

    # Build augmented bug-predict entries.
    overrides: dict[str, RetrievalEntry] = {}
    for path in _BUG_PREDICT_PATHS:
        e = base_corpus.get(path)
        if e is None:
            print(f"WARN: bug-predict entry missing from corpus: {path}", file=sys.stderr)
            continue
        overrides[path] = _augmented(e)

    patched_corpus = _PatchedCorpus(base_corpus, overrides)

    # Score against both corpora.
    pre_stats = _score(queries, retriever, base_corpus)
    post_stats = _score(queries, retriever, patched_corpus)

    print("# Diagnostic-3 — Alias expansion on bug-predict\n")
    print(f"corpus: {base_corpus.name}  version: {base_corpus.version}")
    print(f"queries: {paraphrased_path.relative_to(_ROOT)} ({len(queries)} total)")
    print(
        f"added aliases: {len(_BUG_PREDICT_EXTRA_ALIASES)} multi-token aliases on each "
        f"of {len(overrides)} bug-predict entries"
    )
    print()

    print("## All-corpus view (sanity check — should not regress on non-bug-predict)\n")
    pre_table, pre_p1, pre_r3 = _render(pre_stats, "Before (aliases as-is)")
    post_table, post_p1, post_r3 = _render(post_stats, "After (augmented aliases)")
    print(pre_table)
    print()
    print(post_table)
    print()
    print(f"Δ P@1 (overall): {post_p1 - pre_p1:+.2%}")
    print(f"Δ R@3 (overall): {post_r3 - pre_r3:+.2%}")
    print()

    print("## bug-predict focus\n")
    pre_bp, n_bp, pre_p1_bp, pre_r3_bp = _render_bug_predict_only(
        queries, retriever, base_corpus, "Before (aliases as-is)"
    )
    post_bp, _, post_p1_bp, post_r3_bp = _render_bug_predict_only(
        queries, retriever, patched_corpus, "After (augmented aliases)"
    )
    print(pre_bp)
    print()
    print(post_bp)
    print()
    if n_bp:
        d_p1 = (post_p1_bp - pre_p1_bp) / n_bp
        d_r3 = (post_r3_bp - pre_r3_bp) / n_bp
        print(f"Δ P@1 (bug-predict only): {d_p1:+.2%} ({post_p1_bp - pre_p1_bp:+d} queries)")
        print(f"Δ R@3 (bug-predict only): {d_r3:+.2%} ({post_r3_bp - pre_r3_bp:+d} queries)")
    print()

    print("## Verdict\n")
    if n_bp:
        d_r3 = (post_r3_bp - pre_r3_bp) / n_bp
        if d_r3 >= 0.30:
            verdict = (
                "**STRONG** — alias expansion closes most of the bug-predict paraphrase gap. "
                "If this generalizes across other feature clusters, the embedding-retriever "
                "spec defers in favor of an alias-authoring task."
            )
        elif d_r3 >= 0.15:
            verdict = (
                "**MIXED** — aliases help but don't close the gap. Embedding case remains for "
                "the residual; spec mandate shrinks to those queries."
            )
        else:
            verdict = (
                "**WEAK** — aliases barely move the needle. Embedding case stands; alias "
                "expansion is not a viable substitute."
            )
        print(verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
