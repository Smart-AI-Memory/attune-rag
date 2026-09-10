"""Validate release resources and deterministic source/installed-wheel parity.

Build: python scripts/check_distribution.py --output-dir dist --report artifact-check.json
Validate existing bytes: add --sdist FILE --wheel FILE instead of --output-dir.
Install build and the runtime/[attune-help] dependencies in the calling environment.
Probes reuse those exact dependencies without resolving or downloading any packages.
Only the default isolated build may download build dependencies; --no-build-isolation
uses the already prepared backend. No generation or faithfulness calls are made.
Build/install subprocesses ignore caller pip configuration so installation paths
cannot be redirected by PIP_TARGET, PIP_PREFIX, PIP_USER, or config files.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import site
import subprocess
import sys
import tarfile
import tempfile
import venv
import zipfile
from pathlib import Path

REQUIRED_RESOURCES = (
    "attune_rag/py.typed",
    "attune_rag/corpus/summaries_override.json",
    "attune_rag/corpus/aliases_override.json",
    "attune_rag/editor/template_schema.json",
    "attune_rag/dashboard/templates/dashboard.html",
)
QUERIES = Path("tests/golden/queries.yaml")
THRESHOLDS = Path("docs/specs/release-quality-baseline/thresholds.json")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _run(args: list[str], *, cwd: Path) -> str:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    for name in tuple(env):
        if name.startswith("PIP_"):
            env.pop(name)
    env["PIP_CONFIG_FILE"] = os.devnull
    result = subprocess.run(args, cwd=cwd, env=env, capture_output=True, text=True, timeout=300)
    if result.returncode:
        raise ValueError(
            f"Command failed ({result.returncode}): {args!r}\n" f"{result.stdout}\n{result.stderr}"
        )
    return result.stdout


def snapshot_source(source_root: Path, destination: Path) -> dict:
    """Copy current tracked contents, honoring deletions and excluding build debris."""
    paths = _run(["git", "ls-files", "-z"], cwd=source_root).split("\0")
    manifest = {}
    for name in filter(None, paths):
        source = source_root / name
        if not source.exists():
            continue
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"Tracked build input must be a regular file: {name}")
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        manifest[name] = _sha(target.read_bytes())
    if not (destination / "pyproject.toml").is_file():
        raise ValueError("Tracked snapshot is missing pyproject.toml")
    return {
        "base_commit": _run(["git", "rev-parse", "HEAD"], cwd=source_root).strip(),
        "tracked_changes": bool(_run(["git", "diff", "HEAD", "--name-only"], cwd=source_root)),
        "snapshot_sha256": _sha(json.dumps(manifest, sort_keys=True).encode()),
    }


def inspect_artifacts(source_root: Path, sdist: Path, wheel: Path) -> dict:
    """Require exact runtime resources and package source in both distributions."""
    resources = {}
    package_source = {
        path.relative_to(source_root / "src").as_posix()
        for path in (source_root / "src" / "attune_rag").rglob("*")
        if path.is_file() and path.suffix in {".py", ".pyi"}
    }
    with tarfile.open(sdist) as archive, zipfile.ZipFile(wheel) as installed:
        members = archive.getmembers()
        roots = {member.name.split("/")[0] for member in members}
        if len(roots) != 1:
            raise ValueError("sdist must have exactly one top-level directory")
        prefix = next(iter(roots))
        for name in sorted(set(REQUIRED_RESOURCES) | package_source):
            expected = (source_root / "src" / name).read_bytes()
            sdist_name = f"{prefix}/src/{name}"
            matches = [m for m in members if m.name == sdist_name]
            if len(matches) != 1 or not matches[0].isfile():
                raise ValueError(f"sdist missing or ambiguous required resource: {name}")
            stream = archive.extractfile(matches[0])
            if stream is None or stream.read() != expected:
                raise ValueError(f"sdist resource content mismatch: {name}")
            if installed.namelist().count(name) != 1:
                raise ValueError(f"wheel missing or ambiguous required resource: {name}")
            if installed.read(name) != expected:
                raise ValueError(f"wheel resource content mismatch: {name}")
            resources[name] = _sha(expected)
    return {
        "resources": resources,
        "artifacts": {
            p.name: {"path": str(p.resolve()), "sha256": _sha(p.read_bytes())}
            for p in (sdist, wheel)
        },
    }


def validate_results(source: dict, wheel: dict, queries: Path, thresholds: Path) -> None:
    """Gate exact per-query behavior as well as the active retrieval thresholds."""
    # Load the existing gate directly, including when this driver is imported in tests.
    spec = importlib.util.spec_from_file_location(
        "distribution_thresholds", Path(__file__).with_name("check_thresholds.py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    limits = json.loads(thresholds.read_text(encoding="utf-8"))
    query_hash = _sha(queries.read_bytes())
    if limits.get("queries_sha256") != query_hash:
        raise ValueError("Query SHA mismatch with the active baseline")
    for field in ("version", "dependencies", "queries"):
        if not source.get(field) or source[field] != wheel.get(field):
            raise ValueError(f"Source/wheel {field} mismatch")
    for label, result in (("source", source), ("wheel", wheel)):
        if result.get("queries_sha256") != query_hash:
            raise ValueError(f"{label} query SHA mismatch")
        for row in result["queries"]:
            if any(not math.isfinite(hit["score"]) for hit in row["hits"]):
                raise ValueError(f"{label} contains a nonfinite retrieval score")
        for metric in ("precision_at_1", "recall_at_k"):
            value = result["retrieval"][metric]
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{label} invalid retrieval metric: {metric}")
        failures, errors = module.check(
            {**result, "queries_path": str(queries)},
            limits,
            skip_metrics=frozenset({"mean_faithfulness"}),
        )
        if failures or errors:
            raise ValueError(f"{label} retrieval threshold failure: {failures!r}; {errors!r}")


def _probe(import_root: Path, dependencies: list[str], queries_path: Path, mode: str) -> dict:
    """Run inside -I -S: no PYTHONPATH, user site, editable .pth, or checkout fallback."""
    sys.path[:0] = [str(import_root), *dependencies]
    from importlib import metadata

    import structlog
    import yaml

    import attune_rag
    from attune_rag import RagPipeline
    from attune_rag._scoring import score_queries

    origin = Path(attune_rag.__file__).resolve()
    if not origin.is_relative_to(import_root):
        raise ValueError(f"Unexpected {mode} import origin: {origin}")
    if mode == "wheel":
        distribution = metadata.distribution("attune-rag")
        if not Path(distribution.locate_file("")).resolve().is_relative_to(import_root):
            raise ValueError("Wheel metadata resolved outside the installed target")
        if distribution.version != attune_rag.__version__:
            raise ValueError("Installed wheel metadata/source version mismatch")
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(50))
    queries = yaml.safe_load(queries_path.read_text(encoding="utf-8"))["queries"]
    if not queries or len({str(q["id"]) for q in queries}) != len(queries):
        raise ValueError("Golden queries must be nonempty with unique IDs")
    rows = []

    class RecordedPipeline(RagPipeline):
        def run(self, query: str, *, k: int = 3):
            result = super().run(query, k=k)
            rows.append(
                {
                    "hits": [
                        {"path": hit.template_path, "score": hit.score}
                        for hit in result.citation.hits
                    ]
                }
            )
            return result

    _, aggregate = score_queries(RecordedPipeline(), queries, k=3)
    for query, row in zip(queries, rows, strict=True):
        row["id"] = str(query["id"])
    for name, loaded in list(sys.modules.items()):
        if name.startswith("attune_rag.") and getattr(loaded, "__file__", None):
            if not Path(loaded.__file__).resolve().is_relative_to(import_root):
                raise ValueError(f"Module escaped the {mode} target: {name}")
    return {
        "version": attune_rag.__version__,
        "import_origin": str(origin),
        "dependencies": {
            d.metadata["Name"]: d.version
            for d in metadata.distributions(path=dependencies)
            if d.metadata["Name"].lower().replace("_", "-") != "attune-rag"
        },
        "queries_sha256": _sha(queries_path.read_bytes()),
        "queries": rows,
        "retrieval": {"precision_at_1": aggregate.p1, "recall_at_k": aggregate.r3, "k": 3},
    }


def check_distribution(
    source_root: Path,
    output_dir: Path,
    *,
    sdist: Path | None = None,
    wheel: Path | None = None,
    build_isolation: bool = True,
) -> dict:
    """Build or validate an existing pair; preserve emitted artifacts in output_dir."""
    source_root = source_root.resolve()
    output_dir = output_dir.resolve()
    built_here = sdist is None
    if (sdist is None) != (wheel is None):
        raise ValueError("Supply both --sdist and --wheel, or neither")
    with tempfile.TemporaryDirectory(prefix="attune-rag-distribution-") as temporary:
        scratch = Path(temporary).resolve()
        snapshot = scratch / "source"
        snapshot.mkdir()
        source_info = snapshot_source(source_root, snapshot)
        if sdist is None:
            if output_dir.exists() and any(output_dir.iterdir()):
                raise ValueError(
                    "Build output directory must be empty; use --sdist/--wheel to validate"
                )
            output_dir.mkdir(parents=True, exist_ok=True)
            args = [sys.executable, "-m", "build", "--outdir", str(output_dir)]
            if not build_isolation:
                args += ["--no-isolation", "--skip-dependency-check"]
            # The default build lifecycle creates the wheel FROM its newly built sdist.
            _run(args, cwd=snapshot)
            sdists, wheels = list(output_dir.glob("*.tar.gz")), list(output_dir.glob("*.whl"))
            if len(sdists) != 1 or len(wheels) != 1:
                raise ValueError("Build must emit exactly one sdist and one wheel")
            sdist, wheel = sdists[0], wheels[0]
        sdist, wheel = sdist.resolve(), wheel.resolve()
        evidence = inspect_artifacts(snapshot, sdist, wheel)
        install_env = scratch / "installed"
        venv.EnvBuilder(with_pip=True, symlinks=os.name != "nt").create(install_env)
        python = install_env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        _run(
            [
                str(python),
                "-I",
                "-m",
                "pip",
                "install",
                "--no-index",
                "--no-deps",
                "--disable-pip-version-check",
                str(wheel),
            ],
            cwd=scratch,
        )
        purelib = Path(
            _run(
                [str(python), "-I", "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
                cwd=scratch,
            ).strip()
        ).resolve()
        dependencies = json.dumps(site.getsitepackages())
        results = {}
        for mode, root in (("source", snapshot / "src"), ("wheel", purelib)):
            raw = _run(
                [
                    str(python),
                    "-I",
                    "-S",
                    str(Path(__file__).resolve()),
                    "_probe",
                    str(root),
                    dependencies,
                    str(snapshot / QUERIES),
                    mode,
                ],
                cwd=scratch,
            )
            results[mode] = json.loads(raw)
        try:
            import tomllib
        except ModuleNotFoundError:
            import tomli as tomllib  # build's TOML dependency on Python 3.10
        project_version = tomllib.loads((snapshot / "pyproject.toml").read_text())["project"][
            "version"
        ]
        if project_version != results["source"]["version"]:
            raise ValueError("Source version does not match snapshot pyproject.toml")
        # Compare the sdist's own metadata independently of stale parent metadata.
        with tarfile.open(sdist) as archive:
            root = archive.getnames()[0].split("/")[0]
            from email.parser import BytesParser

            package_info = BytesParser().parsebytes(archive.extractfile(f"{root}/PKG-INFO").read())
        if package_info["Version"] != results["source"]["version"]:
            raise ValueError("Source version does not match built project metadata")
        validate_results(
            results["source"], results["wheel"], snapshot / QUERIES, snapshot / THRESHOLDS
        )
        return {
            "status": "passed",
            "source": source_info,
            **evidence,
            "probes": results,
            "faithfulness": "skipped: retrieval-only; no provider calls",
            "mode": "build" if built_here else "validate-existing",
            "build_isolation": build_isolation if built_here else None,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=Path("dist"))
    parser.add_argument("--sdist", type=Path)
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--no-build-isolation", action="store_true")
    args = parser.parse_args(argv)

    def report_conflicts() -> bool:
        candidates = [
            args.sdist,
            args.wheel,
            *args.output_dir.glob("*.whl"),
            *args.output_dir.glob("*.tar.gz"),
        ]
        return any(
            path is not None
            and (
                args.report.resolve() == path.resolve()
                or (args.report.exists() and path.exists() and args.report.samefile(path))
            )
            for path in candidates
        )

    if report_conflicts():
        print("Report destination must not overwrite a distribution artifact", file=sys.stderr)
        return 1
    try:
        report = check_distribution(
            args.source_root,
            args.output_dir,
            sdist=args.sdist,
            wheel=args.wheel,
            build_isolation=not args.no_build_isolation,
        )
    except Exception as exc:  # noqa: BLE001 — CLI boundary must emit a failed receipt, never pass.
        report = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    if report_conflicts():
        print("Report destination must not overwrite a distribution artifact", file=sys.stderr)
        return 1
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if report["status"] == "failed":
        print(report["error"], file=sys.stderr)
        return 1
    print(f"Distribution validation passed; receipt: {args.report}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "_probe":
        print(
            json.dumps(
                _probe(Path(sys.argv[2]), json.loads(sys.argv[3]), Path(sys.argv[4]), sys.argv[5]),
                allow_nan=False,
            )
        )
    else:
        raise SystemExit(main())
