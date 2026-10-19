"""Replay release integrity checks without building, uploading, or publishing."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WHEEL = "attune_rag-1.2.1-py3-none-any.whl"
SDIST = "attune_rag-1.2.1.tar.gz"


def _workflow(name: str) -> dict[str, Any]:
    return yaml.safe_load((REPO_ROOT / ".github/workflows" / name).read_text(encoding="utf-8"))


def _named_step(job: dict[str, Any], name: str) -> dict[str, Any]:
    return next(step for step in job["steps"] if step.get("name") == name)


def _run_shell(
    script: str, cwd: Path, *, extra_env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("Replaying GitHub's bash steps requires bash")
    env = os.environ.copy()
    env["PATH"] = os.pathsep.join([str(Path(sys.executable).parent), env.get("PATH", "")])
    env.update(extra_env or {})
    # GitHub's bash runner enables errexit. Preserve it while executing the
    # literal workflow script, including its embedded Python verification.
    return subprocess.run(
        [bash, "--noprofile", "--norc", "-e", "-o", "pipefail", "-c", script],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _write_valid_bundle(root: Path) -> tuple[Path, dict[str, Any]]:
    dist = root / "dist"
    dist.mkdir()
    payloads = {WHEEL: b"validated wheel bytes\n", SDIST: b"validated sdist bytes\n"}
    for name, content in payloads.items():
        (dist / name).write_bytes(content)
    report = {
        "status": "passed",
        "artifacts": {
            name: {"path": f"/build-runner/dist/{name}", "sha256": hashlib.sha256(data).hexdigest()}
            for name, data in payloads.items()
        },
    }
    receipt = root / "validation/artifact-validation.json"
    receipt.parent.mkdir()
    receipt.write_text(json.dumps(report), encoding="utf-8")
    return receipt, report


@pytest.mark.parametrize(
    ("defect", "expected_error"),
    [
        ("valid", None),
        ("changed-artifact", "Artifact checksum mismatch"),
        ("extra-file", "Downloaded artifact set differs"),
        ("missing-file", "Downloaded artifact set differs"),
        ("changed-report", "Validation receipt checksum mismatch"),
        ("failed-report", "Artifact validation did not pass"),
        ("two-wheels", "Expected exactly one wheel and one sdist"),
    ],
)
def test_actual_publication_verifier_accepts_only_validated_bytes(
    tmp_path: Path, defect: str, expected_error: str | None
) -> None:
    receipt, report = _write_valid_bundle(tmp_path)
    expected_hash = hashlib.sha256(receipt.read_bytes()).hexdigest()
    if defect == "changed-artifact":
        (tmp_path / "dist" / WHEEL).write_bytes(b"different wheel bytes\n")
    elif defect == "extra-file":
        (tmp_path / "dist/unvalidated.whl").write_bytes(b"extra wheel\n")
    elif defect == "missing-file":
        (tmp_path / "dist" / SDIST).unlink()
    elif defect == "changed-report":
        # It remains valid JSON with the same status and artifact hashes.
        receipt.write_bytes(receipt.read_bytes() + b"\n")
    elif defect == "failed-report":
        report["status"] = "failed"
        receipt.write_text(json.dumps(report), encoding="utf-8")
        expected_hash = hashlib.sha256(receipt.read_bytes()).hexdigest()
    elif defect == "two-wheels":
        replacement = "another-1.2.1-py3-none-any.whl"
        (tmp_path / "dist" / SDIST).rename(tmp_path / "dist" / replacement)
        report["artifacts"][replacement] = report["artifacts"].pop(SDIST)
        receipt.write_text(json.dumps(report), encoding="utf-8")
        expected_hash = hashlib.sha256(receipt.read_bytes()).hexdigest()
    verify = _named_step(
        _workflow("publish.yml")["jobs"]["publish"], "Verify validated artifact bytes"
    )

    result = _run_shell(
        verify["run"], tmp_path, extra_env={"EXPECTED_REPORT_SHA256": expected_hash}
    )

    if expected_error is None:
        assert result.returncode == 0, result.stderr
        assert "verified for publication" in result.stdout
    else:
        assert result.returncode != 0
        assert expected_error in result.stderr


def test_actual_integrity_step_records_hash_of_receipt_bytes(tmp_path: Path) -> None:
    receipt, _ = _write_valid_bundle(tmp_path)
    raw = receipt.read_bytes()
    (tmp_path / "artifact-validation.json").write_bytes(raw)
    github_output = tmp_path / "github-output"
    step = _named_step(_workflow("publish.yml")["jobs"]["build"], "Record validation receipt hash")

    result = _run_shell(step["run"], tmp_path, extra_env={"GITHUB_OUTPUT": str(github_output)})

    assert result.returncode == 0, result.stderr
    assert github_output.read_text(encoding="utf-8").splitlines() == [
        f"report_sha256={hashlib.sha256(raw).hexdigest()}"
    ]


def _require_default_success(node: dict[str, Any]) -> None:
    assert node.get("continue-on-error", False) is False
    assert node.get("if") in (None, "success()", "${{ success() }}")


@pytest.mark.parametrize("workflow_name", ["publish.yml", "tests.yml"])
def test_actual_artifact_checker_step_propagates_failure(
    tmp_path: Path, workflow_name: str
) -> None:
    workflow = _workflow(workflow_name)
    jobs = [
        job
        for job in workflow["jobs"].values()
        if any("scripts/check_distribution.py" in step.get("run", "") for step in job["steps"])
    ]
    assert len(jobs) == 1
    job = jobs[0]
    check = next(
        step for step in job["steps"] if "scripts/check_distribution.py" in step.get("run", "")
    )
    _require_default_success(job)
    _require_default_success(check)
    executables = tmp_path / "executables"
    executables.mkdir()
    python = executables / "python"
    python.write_text('#!/bin/sh\nprintf "ARG:%s\\n" "$@"\nexit 19\n', encoding="utf-8")
    python.chmod(0o755)

    result = _run_shell(
        check["run"],
        tmp_path,
        extra_env={"PATH": os.pathsep.join([str(executables), os.environ.get("PATH", "")])},
    )

    assert "ARG:scripts/check_distribution.py" in result.stdout
    assert result.returncode == 19, result.stderr
    if workflow_name == "publish.yml":
        uploads = [
            step
            for step in job["steps"]
            if step.get("uses", "").startswith("actions/upload-artifact@")
        ]
        assert len(uploads) == 2
        for upload in uploads:
            assert job["steps"].index(upload) > job["steps"].index(check)
            _require_default_success(upload)


def test_publishing_requires_build_receipt_and_successful_verification() -> None:
    jobs = _workflow("publish.yml")["jobs"]
    build, publish = jobs["build"], jobs["publish"]
    assert publish["needs"] == "build"
    assert publish["environment"] == "pypi"
    assert publish["permissions"]["id-token"] == "write"
    _require_default_success(publish)
    integrity = _named_step(build, "Record validation receipt hash")
    assert integrity["id"] == "integrity"
    assert (
        build["outputs"]["validation_report_sha256"]
        == "${{ steps.integrity.outputs.report_sha256 }}"
    )
    verify = _named_step(publish, "Verify validated artifact bytes")
    assert (
        verify["env"]["EXPECTED_REPORT_SHA256"]
        == "${{ needs.build.outputs.validation_report_sha256 }}"
    )
    _require_default_success(verify)
    publisher = next(
        step
        for step in publish["steps"]
        if step.get("uses", "").startswith("pypa/gh-action-pypi-publish@")
    )
    _require_default_success(publisher)
    between = publish["steps"][
        publish["steps"].index(verify) + 1 : publish["steps"].index(publisher)
    ]
    assert publish["steps"].index(verify) < publish["steps"].index(publisher)
    assert not between, "Publish the checked bytes immediately after verification"
    downloads = [
        step
        for step in publish["steps"]
        if step.get("uses", "").startswith("actions/download-artifact@")
    ]
    assert {step["with"]["name"] for step in downloads} == {"dist", "distribution-validation"}
    assert all(publish["steps"].index(step) < publish["steps"].index(verify) for step in downloads)
