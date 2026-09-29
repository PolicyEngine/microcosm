#!/usr/bin/env python3
"""Create a local BuildP native-role candidate; never calibrate or publish."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import platform
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from importlib import metadata
from pathlib import Path

from microcosm.build.us_runtime import education_assistance_source
from microcosm.build.us_runtime.spm_role_source import (
    ASEC_SPM_ROLE_SOURCES,
    derive_spm_role_source,
)
from microcosm.data.contract import ReleaseContractError
from microcosm.data.h5_enrichment import append_native_spm_role, file_sha256
from microcosm.data.source_enrichment import (
    PARENT_BUILD_ID,
    PARENT_DATASET_SHA256,
    PARENT_FILES,
    ROLE_VARIABLE,
    SOURCE_ENRICHMENT_FILE,
    SOURCE_ENRICHMENT_RELEASE_TYPE,
    SOURCE_EVIDENCE_FILE,
    SOURCE_PROVENANCE_FILE,
    validate_source_enrichment_candidate,
)

REFERENCE_EVIDENCE_SHA256 = (
    "22b5968d90fecfeef7614583e493fe10cc16bda8b5be82e6f49a5bc2102d3ce5"
)
_PRODUCER_FILES = (
    "tools/build_us_spm_role_enrichment.py",
    "packages/microcosm-build/src/microcosm/build/us_runtime/spm_role_source.py",
    "packages/microcosm-build/src/microcosm/build/us_runtime/education_assistance_source.py",
    "packages/microcosm-data/src/microcosm/data/h5_enrichment.py",
    "packages/microcosm-data/src/microcosm/data/source_enrichment.py",
    "packages/microcosm-data/src/microcosm/data/contract.py",
)


def _json_write(path: Path, value: object) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    path.chmod(0o600)


def _producer_identity() -> dict:
    import h5py

    root = Path(__file__).resolve().parents[1]

    loaded = {
        _PRODUCER_FILES[0]: __file__,
        _PRODUCER_FILES[1]: inspect.getsourcefile(derive_spm_role_source),
        _PRODUCER_FILES[2]: education_assistance_source.__file__,
        _PRODUCER_FILES[3]: inspect.getsourcefile(append_native_spm_role),
        _PRODUCER_FILES[4]: inspect.getsourcefile(validate_source_enrichment_candidate),
        _PRODUCER_FILES[5]: inspect.getsourcefile(ReleaseContractError),
    }
    for filename, actual in loaded.items():
        if actual is None or Path(actual).resolve() != (root / filename).resolve():
            raise ValueError(
                f"Producer must execute this checkout's {filename}; "
                "install this workspace's local shards first"
            )

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()

    return {
        "repository": "https://github.com/PolicyEngine/microcosm",
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "runtime": {
            "python": platform.python_version(),
            "hdf5": h5py.version.hdf5_version,
            **{
                package: metadata.version(package)
                for package in ("numpy", "pandas", "h5py", "tables", "microcosm-data")
            },
        },
        "source_files_sha256": {
            filename: file_sha256(root / filename) for filename in _PRODUCER_FILES
        },
    }


def build_candidate(
    *,
    parent_h5: Path,
    parent_release_dir: Path,
    source_paths: dict[int, Path],
    output_dir: Path,
    release_id: str,
    reference_evidence_csv: Path | None = None,
) -> dict:
    """Derive, preserve, validate, then atomically expose a new private bundle."""
    if (
        not release_id.startswith("populace-us-2024-")
        or Path(release_id).name != release_id
        or release_id == PARENT_BUILD_ID
        or "\\" in release_id
    ):
        raise ValueError("Choose a new bare US 2024 source-enrichment release ID")
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError("Output directory already exists; choose a new candidate")
    if file_sha256(parent_h5) != PARENT_DATASET_SHA256:
        raise ValueError("Only the exact reviewed BuildP parent can be enriched")
    for destination, expected in PARENT_FILES.items():
        source = parent_release_dir / destination.removeprefix("parent_")
        if file_sha256(source) != expected:
            raise ValueError(f"Parent evidence SHA-256 mismatch: {source.name}")
    if reference_evidence_csv is not None and (
        file_sha256(reference_evidence_csv) != REFERENCE_EVIDENCE_SHA256
    ):
        raise ValueError("Reference evidence CSV SHA-256 mismatch")
    producer_identity = _producer_identity()
    reconstructed = derive_spm_role_source(
        parent_h5,
        source_paths,
        expected_parent_sha256=PARENT_DATASET_SHA256,
    )
    evidence_bytes = reconstructed.evidence.to_csv(
        index=False, lineterminator="\n"
    ).encode()
    # This is an independent oracle, never the producer's input data source.
    import hashlib

    if hashlib.sha256(evidence_bytes).hexdigest() != REFERENCE_EVIDENCE_SHA256:
        raise ValueError(
            "Reconstructed source evidence differs from reviewed BuildP roles"
        )
    if (
        reference_evidence_csv is not None
        and reference_evidence_csv.read_bytes() != evidence_bytes
    ):
        raise ValueError(
            "Reconstruction differs from the supplied independent evidence"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".spm-enrichment-", dir=output_dir.parent))
    try:
        release_dir = staging / "releases" / release_id
        release_dir.mkdir(parents=True, mode=0o700)
        artifact_root = staging / "artifacts"
        artifact_root.mkdir(mode=0o700)
        candidate_h5 = artifact_root / "populace_us_2024.h5"
        preservation = append_native_spm_role(
            parent_h5,
            candidate_h5,
            reconstructed.role,
            expected_parent_sha256=PARENT_DATASET_SHA256,
        )
        evidence_path = release_dir / SOURCE_EVIDENCE_FILE
        evidence_path.write_bytes(evidence_bytes)
        evidence_path.chmod(0o400)
        _json_write(release_dir / SOURCE_PROVENANCE_FILE, reconstructed.provenance)
        (release_dir / SOURCE_PROVENANCE_FILE).chmod(0o400)
        for destination in PARENT_FILES:
            shutil.copyfile(
                parent_release_dir / destination.removeprefix("parent_"),
                release_dir / destination,
            )
            (release_dir / destination).chmod(0o400)
        dataset = {
            "filename": candidate_h5.name,
            "sha256": preservation["candidate_sha256"],
        }
        report = {
            "schema_version": 1,
            "release_type": SOURCE_ENRICHMENT_RELEASE_TYPE,
            "operation": "add_native_spm_independent_minor_role",
            "parent": {
                "build_id": PARENT_BUILD_ID,
                "repo_id": "policyengine/populace-us",
                "revision": PARENT_BUILD_ID,
                "dataset_sha256": PARENT_DATASET_SHA256,
                "calibration_diagnostics_schema_version": 5,
                "files": PARENT_FILES,
            },
            "dataset": dataset,
            "added_variable": {
                "name": ROLE_VARIABLE,
                "entity": "person",
                "dtype": "bool",
                "age_gate_applied": False,
            },
            "preservation": preservation,
            "source": {
                "person_evidence_filename": SOURCE_EVIDENCE_FILE,
                "person_evidence_sha256": file_sha256(evidence_path),
                "provenance_filename": SOURCE_PROVENANCE_FILE,
                "provenance_sha256": file_sha256(release_dir / SOURCE_PROVENANCE_FILE),
            },
            "reconciliation": reconstructed.provenance,
            "compatibility": {"status": "pending"},
        }
        _json_write(release_dir / SOURCE_ENRICHMENT_FILE, report)
        build = {
            "build_id": release_id,
            "release_type": SOURCE_ENRICHMENT_RELEASE_TYPE,
            "code": producer_identity,
            "dataset": dataset,
            "calibration": {
                "mode": "inherited",
                "parent_build_id": PARENT_BUILD_ID,
                "diagnostics_sha256": PARENT_FILES["calibration_diagnostics.json"],
                "diagnostics_schema_version": 5,
            },
            "staging": {
                "enabled": False,
                "reason": "local source-enrichment candidate; no calibration",
            },
        }
        _json_write(release_dir / "build_manifest.json", build)

        def artifact(path: Path, *, kind: str) -> dict:
            return {
                "repo_id": "policyengine/populace-us",
                "revision": release_id,
                "path": path.name,
                "sha256": file_sha256(path),
                "kind": kind,
            }

        manifest = {
            "schema_version": 1,
            "release_type": SOURCE_ENRICHMENT_RELEASE_TYPE,
            "data_package": {
                "name": "microcosm-data",
                "version": metadata.version("microcosm-data"),
            },
            "default_datasets": {"national": "populace_us_2024"},
            "build": {"build_id": release_id},
            "compatible_core_packages": [],
            "compatible_model_packages": [],
            "artifacts": {
                "populace_us_2024": artifact(candidate_h5, kind="microdata"),
                **{
                    path.name.removesuffix(".json").removesuffix(".csv"): artifact(
                        path, kind="evidence"
                    )
                    for path in sorted(release_dir.iterdir())
                    if path.is_file()
                },
            },
        }
        _json_write(release_dir / "release_manifest.json", manifest)
        validate_source_enrichment_candidate(
            release_dir, parent_h5=parent_h5, artifact_root=artifact_root
        )
        if (
            _producer_identity()["source_files_sha256"]
            != producer_identity["source_files_sha256"]
        ):
            raise ValueError(
                "Producer source files changed while building the candidate"
            )
        candidate_h5.chmod(0o400)
        if output_dir.exists() or output_dir.is_symlink():
            raise FileExistsError("Output appeared during build; refusing to overwrite")
        os.rename(staging, output_dir)
        return report
    except BaseException:
        shutil.rmtree(staging)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-h5", type=Path, required=True)
    parser.add_argument("--parent-release-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--reference-evidence-csv", type=Path)
    parser.add_argument(
        "--source-cache",
        type=Path,
        default=Path.home() / ".cache/microcosm/cps/asec_education",
        help="Directory containing the pinned complete pppub23/24/25.csv files",
    )
    args = parser.parse_args(argv)
    try:
        report = build_candidate(
            parent_h5=args.parent_h5,
            parent_release_dir=args.parent_release_dir,
            source_paths={
                year: args.source_cache / pin.member
                for year, pin in ASEC_SPM_ROLE_SOURCES.items()
            },
            output_dir=args.output_dir,
            release_id=args.release_id,
            reference_evidence_csv=args.reference_evidence_csv,
        )
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Source-enrichment candidate refused: {exc}\n")
    print(
        json.dumps(
            {"dataset": report["dataset"], "compatibility": report["compatibility"]},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
