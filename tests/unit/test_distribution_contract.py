"""Behavioral guards for required release resources and retrieval parity.

Archives are synthetic and local: these tests need neither a build nor an
installed corpus. Retrieval thresholds and the query identity come from the
active repository artifacts rather than a second set of locked constants.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import site
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import check_distribution as distribution  # noqa: E402

QUERIES = REPO_ROOT / "tests/golden/queries.yaml"
THRESHOLDS = REPO_ROOT / "docs/specs/release-quality-baseline/thresholds.json"
CHECKER = REPO_ROOT / "scripts/check_distribution.py"
RESOURCES = (
    "attune_rag/py.typed",
    "attune_rag/corpus/summaries_override.json",
    "attune_rag/corpus/aliases_override.json",
    "attune_rag/editor/template_schema.json",
    "attune_rag/dashboard/templates/dashboard.html",
)
OVERRIDES = RESOURCES[1:3]
PACKAGE_CODE = "attune_rag/__init__.py"


@pytest.fixture
def source_tree(tmp_path: Path) -> tuple[Path, dict[str, bytes]]:
    source = tmp_path / "source"
    payloads = {resource: (REPO_ROOT / "src" / resource).read_bytes() for resource in RESOURCES}
    payloads[PACKAGE_CODE] = b'__version__ = "1.2.1"\n'
    for resource, content in payloads.items():
        destination = source / "src" / resource
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    return source, payloads


def _write_archives(
    tmp_path: Path,
    payloads: dict[str, bytes],
    *,
    sdist_changes: dict[str, bytes | None] | None = None,
    wheel_changes: dict[str, bytes | None] | None = None,
) -> tuple[Path, Path]:
    def changed(changes: dict[str, bytes | None] | None) -> dict[str, bytes]:
        files = dict(payloads)
        for path, content in (changes or {}).items():
            if content is None:
                files.pop(path)
            else:
                files[path] = content
        return files

    sdist = tmp_path / "attune_rag-1.2.1.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        for path, content in changed(sdist_changes).items():
            member = tarfile.TarInfo(f"attune_rag-1.2.1/src/{path}")
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))

    wheel = tmp_path / "attune_rag-1.2.1-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for path, content in changed(wheel_changes).items():
            archive.writestr(path, content)
    return sdist, wheel


def test_complete_archives_match_source_resources(
    tmp_path: Path, source_tree: tuple[Path, dict[str, bytes]]
) -> None:
    source, payloads = source_tree
    sdist, wheel = _write_archives(tmp_path, payloads)

    assert isinstance(distribution.inspect_artifacts(source, sdist, wheel), dict)


@pytest.mark.parametrize("artifact", ["sdist", "wheel"])
@pytest.mark.parametrize("resource", OVERRIDES)
@pytest.mark.parametrize("defect", ["missing", "changed"])
def test_each_override_must_survive_each_build_stage_byte_for_byte(
    tmp_path: Path,
    source_tree: tuple[Path, dict[str, bytes]],
    artifact: str,
    resource: str,
    defect: str,
) -> None:
    source, payloads = source_tree
    # A newline changes bytes while preserving valid JSON. A parser-only
    # check would miss this source/artifact discrepancy.
    replacement = None if defect == "missing" else payloads[resource] + b"\n"
    changes = {f"{artifact}_changes": {resource: replacement}}
    sdist, wheel = _write_archives(tmp_path, payloads, **changes)

    with pytest.raises(ValueError):
        distribution.inspect_artifacts(source, sdist, wheel)


@pytest.mark.parametrize("artifact", ["sdist", "wheel"])
@pytest.mark.parametrize("resource", [RESOURCES[0], *RESOURCES[3:]])
def test_other_required_resources_cannot_disappear(
    tmp_path: Path,
    source_tree: tuple[Path, dict[str, bytes]],
    artifact: str,
    resource: str,
) -> None:
    source, payloads = source_tree
    changes = {f"{artifact}_changes": {resource: None}}
    sdist, wheel = _write_archives(tmp_path, payloads, **changes)

    with pytest.raises(ValueError):
        distribution.inspect_artifacts(source, sdist, wheel)


@pytest.mark.parametrize("artifact", ["sdist", "wheel"])
@pytest.mark.parametrize("defect", ["missing", "changed"])
def test_package_code_must_match_source_in_each_artifact(
    tmp_path: Path,
    source_tree: tuple[Path, dict[str, bytes]],
    artifact: str,
    defect: str,
) -> None:
    source, payloads = source_tree
    replacement = None if defect == "missing" else b'__version__ = "0.0.0"\n'
    changes = {f"{artifact}_changes": {PACKAGE_CODE: replacement}}
    sdist, wheel = _write_archives(tmp_path, payloads, **changes)

    with pytest.raises(ValueError):
        distribution.inspect_artifacts(source, sdist, wheel)


def test_agreeing_archives_must_still_match_source_bytes(
    tmp_path: Path, source_tree: tuple[Path, dict[str, bytes]]
) -> None:
    source, payloads = source_tree
    altered = {OVERRIDES[0]: payloads[OVERRIDES[0]] + b"\n"}
    sdist, wheel = _write_archives(tmp_path, payloads, sdist_changes=altered, wheel_changes=altered)

    with pytest.raises(ValueError):
        distribution.inspect_artifacts(source, sdist, wheel)


@pytest.fixture
def passing_result() -> dict[str, Any]:
    thresholds = json.loads(THRESHOLDS.read_text(encoding="utf-8"))
    queries = yaml.safe_load(QUERIES.read_text(encoding="utf-8"))["queries"]
    return {
        "version": "1.2.1",
        "dependencies": {"attune-help": "0.13.0", "pyyaml": "6.0.2"},
        "queries_sha256": hashlib.sha256(QUERIES.read_bytes()).hexdigest(),
        "retrieval": {
            "precision_at_1": thresholds["metrics"]["precision_at_1"]["threshold"],
            "recall_at_k": thresholds["metrics"]["recall_at_3"]["threshold"],
            "k": 3,
        },
        "queries": [
            {
                "id": query["id"],
                "hits": [
                    {"path": query["expected_in_top_3"][0], "score": 0.75},
                    {"path": "concepts/alternative.md", "score": 0.5},
                    {"path": "concepts/third.md", "score": 0.25},
                ],
            }
            for query in queries
        ],
    }


def test_active_retrieval_thresholds_pass_without_faithfulness(
    passing_result: dict[str, Any],
) -> None:
    # The active baseline also contains mean_faithfulness. Artifact parity
    # must use its retrieval thresholds without requiring provider output.
    assert "mean_faithfulness" in json.loads(THRESHOLDS.read_text())["metrics"]
    assert (
        distribution.validate_results(
            passing_result, copy.deepcopy(passing_result), QUERIES, THRESHOLDS
        )
        is None
    )


@pytest.mark.parametrize("defect", ["reordered_hits", "changed_score", "missing_query"])
def test_aggregate_pass_cannot_hide_per_query_drift(
    passing_result: dict[str, Any], defect: str
) -> None:
    wheel = copy.deepcopy(passing_result)
    if defect == "reordered_hits":
        # Keep top-1 fixed, but swap the second and third positions. Even
        # genuine P@1/R@3 aggregates would remain identical.
        hits = wheel["queries"][0]["hits"]
        hits[1], hits[2] = hits[2], hits[1]
    elif defect == "changed_score":
        wheel["queries"][0]["hits"][0]["score"] += 0.000001
    else:
        wheel["queries"].pop()
    assert wheel["retrieval"] == passing_result["retrieval"]

    with pytest.raises(ValueError):
        distribution.validate_results(passing_result, wheel, QUERIES, THRESHOLDS)


@pytest.mark.parametrize("metric", ["precision_at_1", "recall_at_k"])
def test_equal_source_and_wheel_below_active_thresholds_fail(
    passing_result: dict[str, Any], metric: str
) -> None:
    passing_result["retrieval"][metric] -= 0.001

    with pytest.raises(ValueError):
        distribution.validate_results(
            passing_result, copy.deepcopy(passing_result), QUERIES, THRESHOLDS
        )


@pytest.mark.parametrize("side", ["source", "wheel"])
@pytest.mark.parametrize("metric", ["precision_at_1", "recall_at_k"])
@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), float("-inf"), -0.01, 1.01],
    ids=["nan", "positive-infinity", "negative-infinity", "negative", "above-one"],
)
def test_invalid_aggregate_metrics_fail_for_each_probe(
    passing_result: dict[str, Any], side: str, metric: str, value: float
) -> None:
    source = copy.deepcopy(passing_result)
    wheel = copy.deepcopy(passing_result)
    result = source if side == "source" else wheel
    result["retrieval"][metric] = value

    with pytest.raises(ValueError):
        distribution.validate_results(source, wheel, QUERIES, THRESHOLDS)


@pytest.mark.parametrize("field", ["version", "dependencies", "queries_sha256"])
def test_parity_requires_matching_result_identity(
    passing_result: dict[str, Any], field: str
) -> None:
    wheel = copy.deepcopy(passing_result)
    if field == "dependencies":
        wheel[field]["attune-help"] = "0.13.1"
    elif field == "version":
        wheel[field] = f"{passing_result[field]}.post1"
    else:
        wheel[field] = "0" * 64

    with pytest.raises(ValueError):
        distribution.validate_results(passing_result, wheel, QUERIES, THRESHOLDS)


def test_agreeing_result_hashes_must_match_actual_queries(
    passing_result: dict[str, Any],
) -> None:
    passing_result["queries_sha256"] = "0" * 64

    with pytest.raises(ValueError):
        distribution.validate_results(
            passing_result, copy.deepcopy(passing_result), QUERIES, THRESHOLDS
        )


def test_query_file_and_results_must_match_locked_baseline(
    tmp_path: Path, passing_result: dict[str, Any]
) -> None:
    changed_queries = tmp_path / "queries.yaml"
    changed_queries.write_bytes(QUERIES.read_bytes() + b"\n# changed query input\n")
    passing_result["queries_sha256"] = hashlib.sha256(changed_queries.read_bytes()).hexdigest()

    with pytest.raises(ValueError):
        distribution.validate_results(
            passing_result, copy.deepcopy(passing_result), changed_queries, THRESHOLDS
        )


def _run_probe(import_root: Path, cwd: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    # This would let an ordinary child import the checkout. The actual
    # isolated probe must ignore it, along with editable .pth hooks.
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    return subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            str(CHECKER),
            "_probe",
            str(import_root),
            json.dumps(site.getsitepackages()),
            str(QUERIES),
            "source",
        ],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_isolated_probe_cannot_fall_back_to_pythonpath_or_editable_checkout(
    tmp_path: Path,
) -> None:
    empty_root = tmp_path / "empty-import-root"
    empty_root.mkdir()

    result = _run_probe(empty_root, tmp_path)

    assert result.returncode != 0
    # An ordinary installed distribution, if present, must be rejected for
    # its origin; an editable-only environment cannot import attune_rag.
    assert (
        "No module named 'attune_rag'" in result.stderr
        or "Unexpected source import origin" in result.stderr
    ), result.stderr


def test_isolated_source_probe_reports_real_origin_and_locked_query_set(tmp_path: Path) -> None:
    result = _run_probe(REPO_ROOT / "src", tmp_path)

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert (
        Path(report["import_origin"]).resolve()
        == (REPO_ROOT / "src/attune_rag/__init__.py").resolve()
    )
    expected_ids = [str(query["id"]) for query in yaml.safe_load(QUERIES.read_text())["queries"]]
    assert [row["id"] for row in report["queries"]] == expected_ids
    assert report["queries_sha256"] == hashlib.sha256(QUERIES.read_bytes()).hexdigest()
    assert any(name.lower().replace("_", "-") == "attune-help" for name in report["dependencies"])


def _git(repository: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()


@pytest.fixture
def tracked_repository(source_tree: tuple[Path, dict[str, bytes]], tmp_path: Path) -> Path:
    source, _ = source_tree
    (source / "pyproject.toml").write_text(
        '[project]\nname = "distribution-fixture"\nversion = "1.2.1"\n', encoding="utf-8"
    )
    (source / "current.txt").write_text("committed content\n", encoding="utf-8")
    (source / "deleted.txt").write_text("delete after commit\n", encoding="utf-8")
    empty_hooks = tmp_path / "empty-hooks"
    empty_hooks.mkdir()
    _git(source, "init", "--quiet")
    _git(source, "add", ".")
    _git(
        source,
        "-c",
        "user.name=Distribution Test",
        "-c",
        "user.email=distribution-test@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "-c",
        f"core.hooksPath={empty_hooks}",
        "commit",
        "--quiet",
        "-m",
        "fixture baseline",
    )
    return source


def test_snapshot_uses_current_tracked_bytes_and_records_their_origin(
    tmp_path: Path, tracked_repository: Path
) -> None:
    clean = distribution.snapshot_source(tracked_repository, tmp_path / "clean")
    commit = _git(tracked_repository, "rev-parse", "HEAD")
    assert clean["base_commit"] == commit
    assert clean["tracked_changes"] is False

    (tracked_repository / "current.txt").write_text("uncommitted edit\n", encoding="utf-8")
    (tracked_repository / "deleted.txt").unlink()
    (tracked_repository / "untracked.txt").write_text("must stay outside build\n", encoding="utf-8")
    snapshot = tmp_path / "changed"

    changed = distribution.snapshot_source(tracked_repository, snapshot)

    assert (snapshot / "current.txt").read_text() == "uncommitted edit\n"
    assert not (snapshot / "deleted.txt").exists()
    assert not (snapshot / "untracked.txt").exists()
    assert not (snapshot / ".git").exists()
    assert changed["base_commit"] == commit
    assert changed["tracked_changes"] is True
    assert changed["snapshot_sha256"] != clean["snapshot_sha256"]


@pytest.mark.parametrize(
    ("defect", "expected_error"),
    [
        ("incomplete-pair", "Supply both"),
        ("malformed-sdist", "ReadError"),
        ("malformed-wheel", "BadZipFile"),
    ],
)
def test_failed_cli_run_replaces_stale_passed_receipt(
    tmp_path: Path,
    source_tree: tuple[Path, dict[str, bytes]],
    tracked_repository: Path,
    defect: str,
    expected_error: str,
) -> None:
    _, payloads = source_tree
    sdist, wheel = _write_archives(tmp_path, payloads)
    receipt = tmp_path / "receipt.json"
    receipt.write_text('{"status": "passed", "stale": true}\n', encoding="utf-8")
    artifacts = ["--sdist", str(sdist)]
    if defect != "incomplete-pair":
        artifacts += ["--wheel", str(wheel)]
        broken = sdist if defect == "malformed-sdist" else wheel
        broken.write_bytes(b"not an archive\n")

    result = subprocess.run(
        [
            sys.executable,
            "-I",
            str(CHECKER),
            "--source-root",
            str(tracked_repository),
            "--report",
            str(receipt),
            *artifacts,
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode != 0
    report = json.loads(receipt.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert "stale" not in report
    assert expected_error in report["error"]
    assert report["error"] in result.stderr


@pytest.mark.parametrize("artifact", ["sdist", "wheel"])
@pytest.mark.parametrize("collision", ["same-path", "hardlink"])
def test_report_destination_cannot_overwrite_existing_artifact(
    tmp_path: Path,
    source_tree: tuple[Path, dict[str, bytes]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    artifact: str,
    collision: str,
) -> None:
    source, payloads = source_tree
    sdist, wheel = _write_archives(tmp_path, payloads)
    target = sdist if artifact == "sdist" else wheel
    original = target.read_bytes()
    report = target
    if collision == "hardlink":
        report = tmp_path / "report.json"
        try:
            os.link(target, report)
        except OSError as exc:
            pytest.skip(f"Hardlinks unsupported in this test directory: {exc}")

    calls = []

    def successful_check(*args: Any, **kwargs: Any) -> dict[str, str]:
        calls.append((args, kwargs))
        return {"status": "passed"}

    monkeypatch.setattr(distribution, "check_distribution", successful_check)

    result = distribution.main(
        [
            "--source-root",
            str(source),
            "--output-dir",
            str(tmp_path / "output"),
            "--sdist",
            str(sdist),
            "--wheel",
            str(wheel),
            "--report",
            str(report),
        ]
    )

    assert result != 0
    assert not calls
    assert target.read_bytes() == original
    assert report.read_bytes() == original
    assert "must not overwrite a distribution artifact" in capsys.readouterr().err


@pytest.mark.parametrize("filename", ["library-1.2.1.tar.gz", "library-1.2.1-py3-none-any.whl"])
def test_report_destination_cannot_overwrite_artifact_created_during_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    filename: str,
) -> None:
    output = tmp_path / "output"
    artifact = output / filename
    original = b"newly built distribution bytes\n"
    calls = []

    def emit_artifact(source_root: Path, output_dir: Path, **kwargs: Any) -> dict[str, str]:
        calls.append((source_root, output_dir, kwargs))
        assert not artifact.exists()
        output_dir.mkdir()
        artifact.write_bytes(original)
        return {"status": "passed"}

    monkeypatch.setattr(distribution, "check_distribution", emit_artifact)

    result = distribution.main(["--output-dir", str(output), "--report", str(artifact)])

    assert result != 0
    assert len(calls) == 1
    assert artifact.read_bytes() == original
    assert "must not overwrite a distribution artifact" in capsys.readouterr().err


_PIP_OPTIONS_PROBE = """
import ensurepip
import json
import sys
from pathlib import Path

wheel = next((Path(ensurepip.__file__).parent / "_bundled").glob("pip-*.whl"))
sys.path.insert(0, str(wheel))
import pip
from pip._internal.commands import create_command

options, _ = create_command("install").parse_args(
    ["--no-index", "--no-deps", "--disable-pip-version-check", "example.whl"]
)
print(json.dumps({
    "pip_origin": pip.__file__,
    "bundled_wheel": str(wheel),
    "target": options.target_dir,
    "prefix": options.prefix_path,
    "user": options.use_user_site,
}))
"""


@pytest.mark.parametrize("poison_source", ["environment-and-config", "config-only"])
def test_subprocess_install_options_ignore_caller_pip_redirects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, poison_source: str
) -> None:
    # Exercise the actual bundled pip parser, without calling install.run
    # or fetching anything. A control subprocess proves the poisoned inputs
    # would redirect installation without _run's environment isolation.
    for name in tuple(os.environ):
        if name.startswith("PIP_"):
            monkeypatch.delenv(name)
    config_target = tmp_path / "config-target"
    config_prefix = tmp_path / "config-prefix"
    config = tmp_path / "pip.conf"
    config.write_text(
        f"[install]\ntarget = {config_target}\nprefix = {config_prefix}\nuser = true\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PIP_CONFIG_FILE", str(config))
    expected_target, expected_prefix = config_target, config_prefix
    if poison_source == "environment-and-config":
        expected_target = tmp_path / "environment-target"
        expected_prefix = tmp_path / "environment-prefix"
        monkeypatch.setenv("PIP_TARGET", str(expected_target))
        monkeypatch.setenv("PIP_PREFIX", str(expected_prefix))
        monkeypatch.setenv("PIP_USER", "true")
    args = [sys.executable, "-I", "-c", _PIP_OPTIONS_PROBE]

    control = subprocess.run(
        args, cwd=tmp_path, check=True, capture_output=True, text=True, timeout=30
    )
    poisoned = json.loads(control.stdout)
    assert poisoned["target"] == str(expected_target)
    assert poisoned["prefix"] == str(expected_prefix)
    assert poisoned["user"]  # pip may represent a parsed boolean as integer 1.

    isolated = json.loads(distribution._run(args, cwd=tmp_path))

    assert isolated["bundled_wheel"] in isolated["pip_origin"]
    assert isolated["target"] is None
    assert isolated["prefix"] is None
    assert isolated["user"] in (None, False)
    assert not expected_target.exists()
    assert not expected_prefix.exists()
