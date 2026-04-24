"""Terminal display of the attune-rag dashboard using Rich."""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from typing import Any

from rich import box
from rich.console import Console
from rich.rule import Rule
from rich.table import Table
from rich.text import Text


def _pct_color(v: float, good: float = 0.8) -> str:
    if v >= good:
        return "green"
    if v >= 0.5:
        return "yellow"
    return "red"


def _fmt_pct(v: float, good: float = 0.8) -> str:
    c = _pct_color(v, good)
    return f"[{c} bold]{v*100:.1f}%[/{c} bold]"


def _ts(raw: str) -> str:
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return raw


def display(snapshot: dict[str, Any], console: Console | None = None) -> None:
    con = console or Console()
    r = snapshot.get("retrieval") or {}
    f = snapshot.get("freshness") or {}
    ts = snapshot.get("timestamp", "")

    pkg = next((k[: -len("_version")] for k in f if k.endswith("_version")), "")
    ver = f.get(f"{pkg}_version", "?") if pkg else "?"

    # ── header ──────────────────────────────────────────────────────────────
    con.print()
    con.print("[bold]attune-rag dashboard[/bold]")
    if r and not r.get("error"):
        parts = [
            f"[dim]{r.get('retriever', '?')}[/dim]",
            f"[dim]corpus: {r.get('corpus', '?')}[/dim]",
            f"[green]{pkg} {ver}[/green]" if pkg else "",
            f"[dim]k={r.get('k', '?')}[/dim]",
        ]
        con.print("  " + "  ·  ".join(p for p in parts if p))
    if ts:
        con.print(f"  [dim]Snapshot: {_ts(ts)}[/dim]")
    con.print()

    if r.get("error"):
        con.print(f"[yellow]⚠  Partial snapshot:[/yellow] {r['error']}")
        con.print()
        return

    # ── health at a glance ───────────────────────────────────────────────────
    con.print(Rule("HEALTH AT A GLANCE", style="dim"))
    con.print()

    p1 = r.get("precision_at_1", 0.0)
    rk = r.get("recall_at_k", 0.0)
    tot = r.get("total_queries", 0)
    k = r.get("k", 3)
    mean_ms = r.get("mean_latency_ms", 0.0)
    max_ms = r.get("max_latency_ms", 0.0)
    kt = f.get("kind_totals") or {}
    k_total = sum(kt.values())
    n_feat = len(f.get("features") or [])
    n_kinds = len(f.get("kinds") or [])
    summ = f.get("summaries_by_path_keys", 0)
    pf = f.get("per_feature") or {}
    n_gaps = sum(1 for v in pf.values() if v.get("total", 0) < 4)

    for label, val, sub in [
        ("P@1", _fmt_pct(p1), f"{int(p1 * tot)}/{tot} queries"),
        (f"R@{k}", _fmt_pct(rk), f"recall at top-{k}"),
        ("Latency", f"[cyan bold]{mean_ms:.1f} ms[/cyan bold]", f"max {max_ms:.1f} ms"),
    ]:
        con.print(f"  {label:<10} {val}   [dim]{sub}[/dim]")

    con.print()
    con.print(f"  {(pkg.upper() or 'CORPUS'):<10} [bold]{ver}[/bold]   [dim]{summ} path-keyed summaries[/dim]")
    con.print(f"  {'Templates':<10} [bold]{k_total}[/bold]   [dim]{n_feat} features · {n_kinds} kinds[/dim]")
    gap_color = "yellow" if n_gaps else "green"
    con.print(f"  {'Gaps':<10} [{gap_color} bold]{n_gaps}[/{gap_color} bold]   [dim]features with < 4 templates[/dim]")
    con.print()

    # ── coverage gaps ────────────────────────────────────────────────────────
    gaps = [(feat, info) for feat, info in pf.items() if info.get("total", 0) < 4]
    if gaps:
        con.print(
            f"  [yellow]⚠  Coverage gaps — {len(gaps)} "
            f"feature{'s' if len(gaps) > 1 else ''} below 4 templates[/yellow]"
        )
        for feat, info in gaps:
            by_kind = info.get("by_kind") or {}
            kinds_str = "  ".join(f"[dim]{kk}[/dim] ×{n}" for kk, n in by_kind.items() if n)
            total = info.get("total", 0)
            con.print(f"     [bold]{feat}[/bold]  {total} template{'s' if total != 1 else ''}  {kinds_str}")
        con.print()

    # ── retrieval quality ────────────────────────────────────────────────────
    con.print(Rule("RETRIEVAL QUALITY", style="dim"))
    con.print()

    pd = r.get("per_difficulty") or {}
    diff_tbl = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold dim", padding=(0, 1))
    diff_tbl.add_column("Difficulty")
    diff_tbl.add_column("Top-1", justify="right")
    diff_tbl.add_column("Top-3", justify="right")
    for d in ("easy", "medium", "hard"):
        b = pd.get(d)
        if not b:
            continue
        t1v = b["top1_hit"] / b["total"] if b["total"] else 0.0
        tkv = b["topk_hit"] / b["total"] if b["total"] else 0.0
        diff_tbl.add_row(f"{d} (n={b['total']})", _fmt_pct(t1v), _fmt_pct(tkv))
    con.print("  [bold]By difficulty[/bold]")
    con.print(diff_tbl)

    feat_rows = sorted(
        [
            (feat, v["top1_hit"] / v["total"] if v.get("total") else 0.0)
            for feat, v in (r.get("per_feature") or {}).items()
        ],
        key=lambda x: x[1],
    )
    feat_tbl = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold dim", padding=(0, 1))
    feat_tbl.add_column("Feature")
    feat_tbl.add_column("P@1", justify="right")
    for feat, p1v in feat_rows:
        feat_tbl.add_row(feat, _fmt_pct(p1v))
    con.print("  [bold]By feature (P@1) — sorted worst-first[/bold]")
    con.print(feat_tbl)

    # ── per-query results ────────────────────────────────────────────────────
    con.print(Rule("PER-QUERY RESULTS", style="dim"))
    con.print()

    DIFF_COLORS = {"easy": "green", "medium": "yellow", "hard": "red"}
    q_tbl = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold dim", padding=(0, 1))
    q_tbl.add_column("#", justify="right", style="dim", width=3)
    q_tbl.add_column("Query / Expected / Returned")
    q_tbl.add_column("Feature")
    q_tbl.add_column("Diff", justify="center")
    q_tbl.add_column("Top-1", justify="center")
    q_tbl.add_column("Top-3", justify="center")

    for i, q in enumerate(r.get("per_query") or [], 1):
        diff = (q.get("difficulty") or "unknown").lower()
        dc = DIFF_COLORS.get(diff, "white")
        expected_set = set(q.get("expected") or [])
        actual = q.get("actual") or []

        cell = Text()
        cell.append(q.get("query") or "", style="bold")

        for p in q.get("expected") or []:
            cell.append("\nexp ", style="dim")
            cell.append(p, style="dim")

        for j, p in enumerate(actual):
            in_exp = p in expected_set
            cell.append("\ngot ", style="dim")
            if in_exp:
                cell.append(p, style="green")
            elif j == 0 and not q.get("top1_match"):
                cell.append(p, style="yellow")  # wrong top-1 hit
            else:
                cell.append(p, style="dim")

        t1 = "[green]✓[/green]" if q.get("top1_match") else "[red]✕[/red]"
        tk = "[green]✓[/green]" if q.get("topk_match") else "[red]✕[/red]"
        q_tbl.add_row(str(i), cell, q.get("feature") or "", f"[{dc}]{diff}[/{dc}]", t1, tk)

    con.print(q_tbl)

    # ── corpus freshness ─────────────────────────────────────────────────────
    con.print(Rule("CORPUS FRESHNESS", style="dim"))
    con.print()
    con.print(
        f"  [bold]{pkg.upper() or 'CORPUS'} {ver}[/bold]"
        f"   [dim]{summ} path-keyed summaries[/dim]"
    )
    con.print()

    if kt:
        kind_tbl = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold dim", padding=(0, 1))
        kind_tbl.add_column("Kind")
        kind_tbl.add_column("Count", justify="right")
        for kind in sorted(kt, key=lambda kk: -kt[kk]):
            kind_tbl.add_row(kind, str(kt[kind]))
        con.print("  [bold]Total templates by kind[/bold]")
        con.print(kind_tbl)

    # ── per-feature corpus coverage ──────────────────────────────────────────
    if pf:
        kinds_all = f.get("kinds") or []
        # Only include kinds that appear in at least one feature
        active_kinds = [
            kk for kk in kinds_all
            if any((pf.get(feat, {}).get("by_kind") or {}).get(kk, 0) > 0 for feat in pf)
        ]
        cov_tbl = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold dim", padding=(0, 1))
        cov_tbl.add_column("Feature")
        cov_tbl.add_column("Total", justify="right")
        for kk in active_kinds:
            cov_tbl.add_column(kk[:4], justify="right")  # abbreviated header
        for feat in sorted(pf, key=lambda fn: -(pf[fn].get("total", 0))):
            info = pf[feat]
            by_kind = info.get("by_kind") or {}
            total = info.get("total", 0)
            total_style = "yellow bold" if total < 4 else ""
            cells = [
                Text(feat),
                Text(str(total), style=total_style),
            ] + [
                Text(str(by_kind.get(kk, 0)) if by_kind.get(kk, 0) else "·", style="dim" if not by_kind.get(kk, 0) else "")
                for kk in active_kinds
            ]
            cov_tbl.add_row(*cells)
        con.print()
        con.print("  [bold]Templates per feature[/bold]   [dim](abbreviated kind headers)[/dim]")
        con.print(cov_tbl)

    con.print()


def main(corpus_package: str = "attune_help") -> int:
    import contextlib
    import os

    from .refresh import build_snapshot

    with open(os.devnull, "w") as _null, contextlib.redirect_stdout(_null):
        snapshot = build_snapshot(corpus_package)

    display(snapshot)
    return 1 if snapshot.get("retrieval", {}).get("error") else 0


if __name__ == "__main__":
    sys.exit(main())
