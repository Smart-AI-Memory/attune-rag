"""Guard published extras against the Phase 2 critical advisory ranges.

Read installed distribution metadata, including extras that are not installed.
Wheel inspection and resolver conflict controls supplement these offline guards.
"""

from __future__ import annotations

from importlib.metadata import requires

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

_RUNTIME_EXTRAS = ("attune-help", "claude", "gemini", "embeddings", "transformers")


def _requirements(extra: str = "") -> dict[str, Requirement]:
    result: dict[str, Requirement] = {}
    for raw in requires("attune-rag") or ():
        req = Requirement(raw)
        if req.marker is None or req.marker.evaluate({"extra": extra}):
            result[canonicalize_name(req.name)] = req
    return result


@pytest.mark.parametrize(
    ("package", "affected", "secure", "extras"),
    [
        ("sentence-transformers", "5.5.1", "5.6.0", ("transformers", "all")),
        ("claude-agent-sdk", "0.2.97", "0.2.121", ("claude", "all")),
        ("pyjwt", "2.13.0", "2.15.0", ("claude", "all")),
        (
            "anyio",
            "4.13.0",
            "4.14.2",
            ("claude", "gemini", "embeddings", "transformers", "all", "dev"),
        ),
    ],
)
def test_published_extras_exclude_critical_versions(
    package: str, affected: str, secure: str, extras: tuple[str, ...]
) -> None:
    for extra in extras:
        req = _requirements(extra)[package]
        assert affected not in req.specifier, (extra, req)
        assert secure in req.specifier, (extra, req)


def test_all_is_the_runtime_feature_union() -> None:
    base = _requirements()
    expected: dict[str, Requirement] = {}
    for extra in _RUNTIME_EXTRAS:
        expected.update(_requirements(extra))
    actual = _requirements("all")
    assert set(actual) == set(expected)
    for name, req in expected.items():
        assert actual[name].specifier == req.specifier
        assert actual[name].extras == req.extras
    assert "attune-author" not in actual
    assert "pytest" not in actual
    assert set(base) == {"structlog", "jinja2", "pyyaml", "rich", "jsonschema"}
