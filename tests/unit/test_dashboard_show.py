"""Tests for attune_rag.dashboard.show.

The display loop builds Rich console output by interpolating snapshot
fields (corpus name, feature labels, error messages, etc.) into markup
strings. Any value containing a literal ``[…]`` would alter terminal
styling unless escaped. These tests pin that escape behavior.

The discrimination relies on a quirk of Rich rendering: when markup is
**interpreted** in a plain-text Console (color_system=None), the markup
tokens are consumed — ``[blink red]`` disappears from the output. When
markup is **escaped** (via ``rich.markup.escape``), the tokens survive
as literal text. So if the input field contains ``[blink red]`` and the
rendered output also contains ``[blink red]``, the escape did its job.
"""

from __future__ import annotations

import io

import pytest
from rich.console import Console

from attune_rag.dashboard.show import display


def _render(snapshot: dict) -> str:
    buf = io.StringIO()
    con = Console(file=buf, force_terminal=False, color_system=None, width=200)
    display(snapshot, console=con)
    return buf.getvalue()


def _snap(**retrieval_overrides) -> dict:
    base = {
        "timestamp": "2026-04-23T00:00:00Z",
        "retrieval": {
            "retriever": "KeywordRetriever",
            "corpus": "attune-help",
            "k": 3,
            "precision_at_1": 0.9,
            "recall_at_k": 0.95,
            "mean_latency_ms": 1.0,
            "max_latency_ms": 2.0,
            "total_queries": 1,
            "per_difficulty": {},
            "per_feature": {},
            "per_query": [],
        },
        "freshness": {"kind_totals": {}, "features": [], "kinds": [], "per_feature": {}},
    }
    base["retrieval"].update(retrieval_overrides)
    return base


def test_error_field_rich_markup_escaped():
    out = _render(_snap(error="boom [bold red]EVIL[/bold red] sequel"))
    # The markup tokens must survive as literal text — proof that escape
    # was applied (otherwise Rich would have consumed them).
    assert "[bold red]EVIL[/bold red]" in out
    assert "EVIL" in out


def test_retriever_name_rich_markup_escaped():
    out = _render(_snap(retriever="[blink]X[/blink]"))
    assert "[blink]X[/blink]" in out


def test_per_feature_rich_markup_escaped():
    snap = _snap(
        per_feature={"[bad]feat[/bad]": {"total": 1, "top1_hit": 1, "topk_hit": 1, "by_kind": {}}},
    )
    out = _render(snap)
    assert "[bad]feat[/bad]" in out


def test_per_query_feature_rich_markup_escaped():
    snap = _snap(
        per_query=[
            {
                "id": "q1",
                "query": "harmless",
                "feature": "[red]F[/red]",
                "difficulty": "easy",
                "expected": [],
                "actual": [],
                "top1_match": True,
                "topk_match": True,
            }
        ]
    )
    out = _render(snap)
    assert "[red]F[/red]" in out


def test_freshness_kind_rich_markup_escaped():
    snap = _snap()
    snap["freshness"]["kind_totals"] = {"[bold]concepts[/bold]": 5}
    out = _render(snap)
    assert "[bold]concepts[/bold]" in out


@pytest.mark.parametrize("marker", ["[bold]EVIL[/bold]", "[blink red]X[/blink red]"])
def test_no_markup_consumed_when_input_has_it(marker):
    # If escape weren't applied, Rich would CONSUME the markup tokens in
    # plain-text rendering, leaving only the inner text. With escape, the
    # full token survives.
    out = _render(_snap(error=f"prefix {marker} suffix"))
    assert marker in out
