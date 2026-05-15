"""Guards against package-metadata drift.

The in-source ``__version__`` constant must match the version
declared in ``pyproject.toml`` (which is what PyPI publishes
under). Both v0.1.15 and v0.1.16 shipped with stale
``__version__`` values because the release process bumped
``pyproject.toml`` but not the constant — this test catches
that next time.
"""

from __future__ import annotations

from importlib.metadata import version as pkg_version

import attune_rag


def test_in_source_version_matches_installed_distribution() -> None:
    """``attune_rag.__version__`` must equal the installed wheel's version.

    If this fails after a release-prep PR, you forgot to bump
    ``__version__`` in ``src/attune_rag/__init__.py`` alongside
    ``pyproject.toml``.
    """
    assert attune_rag.__version__ == pkg_version("attune-rag"), (
        f"attune_rag.__version__={attune_rag.__version__!r} but "
        f"installed dist is {pkg_version('attune-rag')!r}. "
        f"Bump src/attune_rag/__init__.py to match pyproject.toml."
    )
