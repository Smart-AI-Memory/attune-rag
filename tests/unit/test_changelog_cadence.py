"""Tests for ``scripts/changelog_cadence.py`` — Phase 4 W0.6."""

from __future__ import annotations

import importlib.util
import sys
from datetime import date
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "changelog_cadence.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("changelog_cadence", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["changelog_cadence"] = module
    spec.loader.exec_module(module)
    return module


cc = _load_module()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _changelog(
    *,
    unreleased_categories: dict[str, list[str]] | None = None,
    releases: list[tuple[str, str, dict[str, list[str]]]] | None = None,
) -> str:
    """Build a CHANGELOG.md from structured input.

    ``releases`` is a list of ``(version, date_str, categories)``.
    ``categories`` maps category name to bullet list.
    """
    lines = ["# Changelog", ""]
    lines.append("## [Unreleased]")
    lines.append("")
    if unreleased_categories:
        for cat, bullets in unreleased_categories.items():
            lines.append(f"### {cat}")
            lines.append("")
            for b in bullets:
                lines.append(f"- {b}")
            lines.append("")
    for version, date_str, categories in releases or ():
        lines.append(f"## [{version}] - {date_str}")
        lines.append("")
        for cat, bullets in categories.items():
            lines.append(f"### {cat}")
            lines.append("")
            for b in bullets:
                lines.append(f"- {b}")
            lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# parse_changelog
# ---------------------------------------------------------------------------


def test_parse_finds_unreleased_and_releases() -> None:
    text = _changelog(
        unreleased_categories={"Changed": ["a"]},
        releases=[
            ("0.1.19", "2026-05-16", {"Changed": ["b", "c"]}),
            ("0.1.18", "2026-05-10", {"Fixed": ["d"]}),
        ],
    )
    sections = cc.parse_changelog(text)
    versions = [s.version for s in sections]
    assert versions == ["Unreleased", "0.1.19", "0.1.18"]
    assert sections[0].release_date is None
    assert sections[1].release_date == date(2026, 5, 16)
    assert sections[2].release_date == date(2026, 5, 10)


def test_parse_counts_categories_correctly() -> None:
    text = _changelog(
        unreleased_categories={
            "Added": ["new thing"],
            "Changed": ["c1", "c2"],
            "Fixed": ["f1"],
        }
    )
    sections = cc.parse_changelog(text)
    unreleased = sections[0]
    assert unreleased.categories["Added"] == 1
    assert unreleased.categories["Changed"] == 2
    assert unreleased.categories["Fixed"] == 1
    assert unreleased.categories["Security"] == 0


def test_parse_ignores_unknown_categories() -> None:
    """Deprecated / Removed are valid Keep-a-Changelog categories but
    not part of the freeze-relevant set; they shouldn't inflate counts."""
    text = (
        "# Changelog\n\n"
        "## [Unreleased]\n\n"
        "### Added\n\n"
        "- a\n"
        "\n"
        "### Deprecated\n\n"
        "- d\n"
        "\n"
        "### Removed\n\n"
        "- r\n"
    )
    sections = cc.parse_changelog(text)
    assert sections[0].categories["Added"] == 1
    assert sections[0].categories.get("Deprecated", None) is None
    assert sections[0].categories.get("Removed", None) is None


def test_parse_counts_only_top_level_bullets() -> None:
    """Indented continuation lines (or nested sub-bullets) don't count
    as new entries — matches how a reviewer scans the CHANGELOG."""
    text = (
        "# Changelog\n\n"
        "## [Unreleased]\n\n"
        "### Changed\n\n"
        "- First bullet.\n"
        "  Continuation line of the first bullet.\n"
        "  - Sub-bullet that should NOT count.\n"
        "- Second bullet.\n"
    )
    sections = cc.parse_changelog(text)
    assert sections[0].categories["Changed"] == 2


def test_parse_handles_no_releases() -> None:
    """A CHANGELOG with only the Unreleased block parses cleanly."""
    text = "# Changelog\n\n## [Unreleased]\n\n### Added\n\n- new\n"
    sections = cc.parse_changelog(text)
    assert len(sections) == 1
    assert sections[0].version == "Unreleased"


def test_parse_skips_malformed_release_dates() -> None:
    """A header with an unparseable date is skipped, not crash-on."""
    text = (
        "# Changelog\n\n"
        "## [Unreleased]\n\n"
        "## [0.1.19] - not-a-date\n\n"
        "### Changed\n\n"
        "- x\n"
    )
    sections = cc.parse_changelog(text)
    # The malformed release line doesn't match _RELEASE_RE; only
    # Unreleased is recovered.
    assert [s.version for s in sections] == ["Unreleased"]


# ---------------------------------------------------------------------------
# filter_for_window
# ---------------------------------------------------------------------------


def _section(version: str, d: date | None, **counts: int) -> cc.ChangelogSection:
    full_counts = dict.fromkeys(cc.CATEGORIES, 0)
    full_counts.update(counts)
    return cc.ChangelogSection(version=version, release_date=d, categories=full_counts)


def test_filter_unreleased_always_counts() -> None:
    sections = [_section("Unreleased", None, Added=1), _section("0.1.0", date(2026, 1, 1))]
    out = cc.filter_for_window(sections, date(2026, 5, 1), date(2026, 5, 7))
    assert [s.version for s in out] == ["Unreleased"]


def test_filter_release_inside_window_included() -> None:
    sections = [_section("0.1.19", date(2026, 5, 16))]
    out = cc.filter_for_window(sections, date(2026, 5, 10), date(2026, 5, 20))
    assert len(out) == 1


def test_filter_release_outside_window_excluded() -> None:
    sections = [_section("0.1.19", date(2026, 5, 16))]
    out = cc.filter_for_window(sections, date(2026, 5, 1), date(2026, 5, 10))
    assert out == []


def test_filter_release_on_boundary_included() -> None:
    """Window is inclusive on both ends."""
    sections = [
        _section("a", date(2026, 5, 10)),
        _section("b", date(2026, 5, 17)),
    ]
    out = cc.filter_for_window(sections, date(2026, 5, 10), date(2026, 5, 17))
    assert {s.version for s in out} == {"a", "b"}


def test_filter_unreleased_and_in_window_release_both_counted() -> None:
    sections = [
        _section("Unreleased", None, Added=1),
        _section("0.1.19", date(2026, 5, 16)),
    ]
    out = cc.filter_for_window(sections, date(2026, 5, 10), date(2026, 5, 20))
    assert {s.version for s in out} == {"Unreleased", "0.1.19"}


# ---------------------------------------------------------------------------
# aggregate_counts
# ---------------------------------------------------------------------------


def test_aggregate_sums_per_category_across_sections() -> None:
    sections = [
        _section("a", date(2026, 5, 1), Added=1, Changed=2),
        _section("b", date(2026, 5, 2), Changed=1, Fixed=3),
        _section("Unreleased", None, Security=1),
    ]
    counts = cc.aggregate_counts(sections)
    assert counts == {"Added": 1, "Changed": 3, "Fixed": 3, "Security": 1}


def test_aggregate_empty_returns_all_zero() -> None:
    counts = cc.aggregate_counts([])
    assert counts == {"Added": 0, "Changed": 0, "Fixed": 0, "Security": 0}


# ---------------------------------------------------------------------------
# releases_in_window
# ---------------------------------------------------------------------------


def test_releases_in_window_excludes_unreleased_and_sorts_oldest_first() -> None:
    sections = [
        _section("Unreleased", None),
        _section("0.1.19", date(2026, 5, 16)),
        _section("0.1.18", date(2026, 5, 10)),
    ]
    assert cc.releases_in_window(sections) == ["0.1.18", "0.1.19"]


# ---------------------------------------------------------------------------
# render_report
# ---------------------------------------------------------------------------


def test_render_status_on_track_when_added_zero() -> None:
    report = cc.render_report(
        window_start=date(2026, 5, 19),
        window_end=date(2026, 5, 26),
        counts={"Added": 0, "Changed": 1, "Fixed": 0, "Security": 0},
        releases=["0.1.19"],
        freeze_week=None,
        freeze_total=None,
    )
    assert "ON TRACK" in report
    assert "RESET" not in report
    assert "2026-05-19 → 2026-05-26" in report


def test_render_status_reset_when_added_nonzero() -> None:
    report = cc.render_report(
        window_start=date(2026, 5, 19),
        window_end=date(2026, 5, 26),
        counts={"Added": 1, "Changed": 0, "Fixed": 0, "Security": 0},
        releases=[],
        freeze_week=None,
        freeze_total=None,
    )
    assert "RESET" in report


def test_render_freeze_week_appears_when_set() -> None:
    report = cc.render_report(
        window_start=date(2026, 5, 19),
        window_end=date(2026, 5, 26),
        counts={"Added": 0, "Changed": 0, "Fixed": 0, "Security": 0},
        releases=[],
        freeze_week=2,
        freeze_total=4,
    )
    assert "week 2/4" in report


def test_render_releases_annotation_appears_when_changed_and_releases_in_window() -> None:
    report = cc.render_report(
        window_start=date(2026, 5, 10),
        window_end=date(2026, 5, 20),
        counts={"Added": 0, "Changed": 2, "Fixed": 0, "Security": 0},
        releases=["0.1.19"],
        freeze_week=None,
        freeze_total=None,
    )
    assert "(releases: 0.1.19)" in report


def test_render_is_deterministic() -> None:
    """Same input → byte-identical output, so re-runs don't churn cadence files."""
    kwargs = dict(
        window_start=date(2026, 5, 19),
        window_end=date(2026, 5, 26),
        counts={"Added": 0, "Changed": 1, "Fixed": 0, "Security": 0},
        releases=["0.1.19"],
        freeze_week=1,
        freeze_total=4,
    )
    assert cc.render_report(**kwargs) == cc.render_report(**kwargs)


# ---------------------------------------------------------------------------
# cadence_summary — end-to-end
# ---------------------------------------------------------------------------


def test_cadence_summary_against_clean_unreleased() -> None:
    text = _changelog(
        unreleased_categories=None,
        releases=[("0.1.19", "2026-05-16", {"Changed": ["x"]})],
    )
    report = cc.cadence_summary(
        text,
        window_start=date(2026, 5, 10),
        window_end=date(2026, 5, 20),
    )
    assert "ON TRACK" in report
    assert "(releases: 0.1.19)" in report


def test_cadence_summary_flags_unreleased_added() -> None:
    text = _changelog(
        unreleased_categories={"Added": ["public method foo()"]},
        releases=[("0.1.19", "2026-05-16", {"Changed": ["x"]})],
    )
    report = cc.cadence_summary(
        text,
        window_start=date(2026, 5, 10),
        window_end=date(2026, 5, 20),
    )
    assert "RESET" in report


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def test_main_writes_to_stdout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(_changelog(releases=[("0.1.19", "2026-05-16", {"Changed": ["x"]})]))
    rc = cc.main(
        [
            "--changelog",
            str(changelog),
            "--end-date",
            "2026-05-20",
            "--window-days",
            "10",
        ]
    )
    assert rc == 0
    captured = capsys.readouterr()
    assert "ON TRACK" in captured.out
    assert "Cadence report" in captured.out


def test_main_writes_to_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(_changelog(unreleased_categories={"Added": ["new"]}))
    out_path = tmp_path / "out" / "cadence.md"

    rc = cc.main(
        [
            "--changelog",
            str(changelog),
            "--end-date",
            "2026-05-20",
            "--window-days",
            "7",
            "--freeze-week",
            "1",
            "--freeze-total",
            "4",
            "--out",
            str(out_path),
        ]
    )
    assert rc == 0
    body = out_path.read_text()
    assert "RESET" in body
    assert "week 1/4" in body


def test_main_invalid_window_days_exits_2(tmp_path: Path) -> None:
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text("# Changelog\n")
    rc = cc.main(
        [
            "--changelog",
            str(changelog),
            "--window-days",
            "0",
        ]
    )
    assert rc == 2


def test_main_freeze_week_without_total_exits_2(tmp_path: Path) -> None:
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text("# Changelog\n")
    rc = cc.main(
        [
            "--changelog",
            str(changelog),
            "--freeze-week",
            "1",
        ]
    )
    assert rc == 2


def test_main_missing_changelog_exits_2(tmp_path: Path) -> None:
    rc = cc.main(
        [
            "--changelog",
            str(tmp_path / "does-not-exist.md"),
        ]
    )
    assert rc == 2


def test_main_invalid_date_format_rejected(tmp_path: Path) -> None:
    """argparse converts the bad value into a SystemExit (exit 2)."""
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text("# Changelog\n")
    with pytest.raises(SystemExit) as exc_info:
        cc.main(
            [
                "--changelog",
                str(changelog),
                "--end-date",
                "not-a-date",
            ]
        )
    assert exc_info.value.code == 2
