"""Weekly changelog cadence report for the Phase 4 feature freeze.

Counts CHANGELOG entries by category (Added / Changed / Fixed /
Security) across a date window, and emits a markdown report. The
``[Unreleased]`` block always counts as "current" regardless of
window; dated releases (``## [0.1.19] - 2026-05-16``) count only if
their release date falls inside the window.

Status line is derived from the ``Added`` count:

- ``Added == 0`` → ``ON TRACK`` (freeze gate intact for this window).
- ``Added > 0`` → ``RESET`` (cadence clock resets; freeze extends).

Phase 4 deliverable W0.6 of the v1.0 roadmap (see
``docs/specs/ROADMAP-v1.md`` and
``docs/specs/downstream-validation/``). The companion workflow
(W0.7) runs this weekly via cron and commits the output to
``docs/specs/downstream-validation/cadence-week-{N}.md``.

Usage::

    # Default: 7-day window ending today.
    python scripts/changelog_cadence.py

    # Explicit window + freeze context for a weekly report.
    python scripts/changelog_cadence.py \\
        --end-date 2026-05-26 \\
        --window-days 7 \\
        --freeze-week 1 \\
        --freeze-total 4 \\
        --out docs/specs/downstream-validation/cadence-week-1.md

Pure stdlib. Safe to run in any CI image.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

CHANGELOG_PATH = "CHANGELOG.md"

CATEGORIES: tuple[str, ...] = ("Added", "Changed", "Fixed", "Security")

# Heading patterns. Release lines look like ``## [0.1.19] - 2026-05-16``
# (with various whitespace tolerances).
_UNRELEASED_RE = re.compile(r"^##\s+\[Unreleased\]\s*$", re.MULTILINE)
_RELEASE_RE = re.compile(
    r"^##\s+\[(?P<version>[^\]]+)\]\s*-\s*(?P<date>\d{4}-\d{2}-\d{2})\s*$",
    re.MULTILINE,
)
_CATEGORY_RE = re.compile(r"^###\s+(?P<cat>\w+)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class ChangelogSection:
    """One ``## [...]`` block from CHANGELOG.md.

    ``release_date`` is ``None`` for ``[Unreleased]``. ``categories``
    maps category name → bullet count.
    """

    version: str
    release_date: date | None
    categories: dict[str, int] = field(default_factory=dict)


def parse_changelog(text: str) -> list[ChangelogSection]:
    """Walk CHANGELOG.md text and return one section per ``## [...]`` block.

    Each section's category counts cover only the ``### Added``,
    ``### Changed``, ``### Fixed``, ``### Security`` subsections —
    any other ``###`` headings (e.g. ``### Deprecated``,
    ``### Removed``) are ignored, matching Keep-a-Changelog's
    standard category set plus our freeze-relevant subset.
    """
    sections: list[ChangelogSection] = []
    headings: list[tuple[int, str, date | None]] = []  # (start_pos, version, date)

    for m in _UNRELEASED_RE.finditer(text):
        headings.append((m.start(), "Unreleased", None))
    for m in _RELEASE_RE.finditer(text):
        try:
            d = datetime.strptime(m.group("date"), "%Y-%m-%d").date()
        except ValueError:
            continue
        headings.append((m.start(), m.group("version"), d))

    # Sort by file position so we can carve consecutive blocks.
    headings.sort(key=lambda h: h[0])

    for i, (pos, version, heading_date) in enumerate(headings):
        end = headings[i + 1][0] if i + 1 < len(headings) else len(text)
        block = text[pos:end]
        categories = _count_categories_in_block(block)
        sections.append(
            ChangelogSection(version=version, release_date=heading_date, categories=categories)
        )
    return sections


def _count_categories_in_block(block: str) -> dict[str, int]:
    """For one ``## [...]`` block, count bullets per ``### Category``.

    A bullet is a line starting with ``- `` at the left margin.
    Continuation lines (indented or blank) do not count as new
    bullets — only top-level bullets are counted, matching how a
    reviewer scans the CHANGELOG.
    """
    counts: dict[str, int] = dict.fromkeys(CATEGORIES, 0)
    cat_matches = list(_CATEGORY_RE.finditer(block))
    for j, m in enumerate(cat_matches):
        category = m.group("cat")
        if category not in counts:
            continue  # ignore Deprecated, Removed, etc.
        sub_start = m.end()
        sub_end = cat_matches[j + 1].start() if j + 1 < len(cat_matches) else len(block)
        sub_block = block[sub_start:sub_end]
        # Count top-level bullets — lines starting with "- " (not "  - ").
        counts[category] += sum(1 for line in sub_block.splitlines() if line.startswith("- "))
    return counts


def filter_for_window(
    sections: list[ChangelogSection], window_start: date, window_end: date
) -> list[ChangelogSection]:
    """Return sections that count for the given [start, end] window.

    ``[Unreleased]`` always counts — it's the live freeze-relevant
    block. Dated releases count only if their date falls in the
    inclusive window. Sections with parse-failed dates (kept as
    ``None``) are excluded except for the explicit Unreleased one.
    """
    out: list[ChangelogSection] = []
    for section in sections:
        if section.release_date is None:
            if section.version == "Unreleased":
                out.append(section)
            continue
        if window_start <= section.release_date <= window_end:
            out.append(section)
    return out


def aggregate_counts(sections: list[ChangelogSection]) -> dict[str, int]:
    """Sum per-category counts across the given sections."""
    out: dict[str, int] = dict.fromkeys(CATEGORIES, 0)
    for section in sections:
        for category, n in section.categories.items():
            if category in out:
                out[category] += n
    return out


def releases_in_window(sections: list[ChangelogSection]) -> list[str]:
    """Names of dated releases included in the window (sorted oldest→newest)."""
    dated = [s for s in sections if s.release_date is not None]
    dated.sort(key=lambda s: s.release_date if s.release_date is not None else date.min)
    return [s.version for s in dated]


def render_report(
    *,
    window_start: date,
    window_end: date,
    counts: dict[str, int],
    releases: list[str],
    freeze_week: int | None,
    freeze_total: int | None,
) -> str:
    """Emit a deterministic markdown report.

    Format mirrors design.md §5: a heading with the window, a
    bullet list of category counts, an optional ``(releases: ...)``
    annotation on ``Changed`` when dated releases are in scope,
    and a status line driven by the ``Added`` count.
    """
    status_core = "ON TRACK" if counts.get("Added", 0) == 0 else "RESET"
    if freeze_week is not None and freeze_total is not None:
        status_suffix = f" (freeze active, week {freeze_week}/{freeze_total})"
    else:
        status_suffix = ""

    lines: list[str] = []
    lines.append(f"# Cadence report: {window_start.isoformat()} → {window_end.isoformat()}")
    lines.append("")
    lines.append("> Auto-generated by `scripts/changelog_cadence.py`.")
    lines.append(
        "> `Added > 0` resets the Phase 4 freeze clock; status is derived "
        "from the count under `Added`."
    )
    lines.append("")
    lines.append("| Category | Count |")
    lines.append("|---|---:|")
    for cat in CATEGORIES:
        n = counts.get(cat, 0)
        suffix = ""
        if cat == "Changed" and n > 0 and releases:
            suffix = f"  (releases: {', '.join(releases)})"
        lines.append(f"| `{cat}` | {n}{suffix} |")
    lines.append("")
    lines.append(f"**Status:** {status_core}{status_suffix}")
    lines.append("")
    return "\n".join(lines) + "\n"


def cadence_summary(
    changelog_text: str,
    *,
    window_start: date,
    window_end: date,
    freeze_week: int | None = None,
    freeze_total: int | None = None,
) -> str:
    """Top-level convenience: parse → filter → aggregate → render."""
    sections = parse_changelog(changelog_text)
    windowed = filter_for_window(sections, window_start, window_end)
    counts = aggregate_counts(windowed)
    releases = releases_in_window(windowed)
    return render_report(
        window_start=window_start,
        window_end=window_end,
        counts=counts,
        releases=releases,
        freeze_week=freeze_week,
        freeze_total=freeze_total,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="changelog_cadence",
        description=(
            "Emit a markdown cadence report counting CHANGELOG entries "
            "(Added / Changed / Fixed / Security) in a date window. "
            "Used by Phase 4's weekly freeze monitor."
        ),
    )
    parser.add_argument(
        "--changelog",
        type=Path,
        default=Path(CHANGELOG_PATH),
        help=f"Path to CHANGELOG.md (default: {CHANGELOG_PATH}).",
    )
    parser.add_argument(
        "--end-date",
        type=_parse_date,
        default=None,
        help="End date of the window, YYYY-MM-DD (default: today, UTC).",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=7,
        help="Window length in days (default: 7).",
    )
    parser.add_argument(
        "--freeze-week",
        type=int,
        default=None,
        help=("If set, render 'week N/T' in the status line. " "Requires --freeze-total too."),
    )
    parser.add_argument(
        "--freeze-total",
        type=int,
        default=None,
        help="Total freeze weeks (default: not rendered).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write report to this path (default: stdout).",
    )
    args = parser.parse_args(argv)

    if args.window_days <= 0:
        print(f"error: --window-days must be > 0 (got {args.window_days})", file=sys.stderr)
        return 2
    if (args.freeze_week is None) != (args.freeze_total is None):
        print(
            "error: --freeze-week and --freeze-total must be set together",
            file=sys.stderr,
        )
        return 2

    end = args.end_date or date.today()
    start = end - timedelta(days=args.window_days - 1)

    try:
        changelog_text = args.changelog.read_text(encoding="utf-8")
    except OSError as e:
        print(f"error: could not read {args.changelog}: {e}", file=sys.stderr)
        return 2

    report = cadence_summary(
        changelog_text,
        window_start=start,
        window_end=end,
        freeze_week=args.freeze_week,
        freeze_total=args.freeze_total,
    )

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(report)
    return 0


def _parse_date(s: str) -> date:
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"invalid date {s!r}: expected YYYY-MM-DD") from e


if __name__ == "__main__":
    sys.exit(main())
