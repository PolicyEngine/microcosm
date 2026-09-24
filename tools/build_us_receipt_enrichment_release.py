#!/usr/bin/env python3
"""Package a qualified receipt child as a local source-enrichment candidate.

Takes the pinned national default (``populace-us-2024-spm-20260915``), its
reviewed release evidence, and the output of
``tools/build_us_acs_donor_receipt_qualification.py`` run on that parent, and
writes a ``source_enrichment`` release directory whose single microdata
artifact is the qualified child H5 (microcosm#978). It derives nothing: the
three receipt inputs, their preservation proof and their aggregate counts are
the qualification's, carried verbatim in its receipt, and the release contract
replays the exhaustive H5 comparison before this tool exposes anything.

It never calibrates, certifies, stages or publishes. The candidate's
compatibility is ``pending``; certification and publication are the existing
``python -m microcosm.data.source_enrichment --certify`` and
``microcosm-publish-release`` steps. See
``docs/us-reported-receipt-source-enrichment.md``.
"""

from __future__ import annotations

import argparse
import contextlib
import errno
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

from microcosm.data import h5_boolean_append, h5_enrichment
from microcosm.data.contract import ReleaseContractError
from microcosm.data.h5_enrichment import file_sha256
from microcosm.data.source_enrichment import (
    PARENT_BUILD_ID,
    RECEIPT_COLUMNS,
    RECEIPT_DATASET_FILENAME,
    RECEIPT_OPERATION,
    RECEIPT_PARENT_BUILD_ID,
    RECEIPT_PARENT_DATASET_SHA256,
    RECEIPT_PARENT_FILES,
    RECEIPT_QUALIFICATION_FILE,
    RECEIPT_RELEASE_PRODUCER_FILES,
    SOURCE_ENRICHMENT_FILE,
    SOURCE_ENRICHMENT_RELEASE_TYPE,
    validate_source_enrichment_candidate,
)

_REPO_ID = "policyengine/populace-us"
_ARTIFACT_KEY = RECEIPT_DATASET_FILENAME.removesuffix(".h5")
_PRODUCER_FILES = RECEIPT_RELEASE_PRODUCER_FILES


def _json_write(path: Path, value: object) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    path.chmod(0o600)


def _producer_identity() -> dict:
    import h5py

    root = Path(__file__).resolve().parents[1]
    loaded = dict(
        zip(
            _PRODUCER_FILES,
            (
                __file__,
                h5_enrichment.__file__,
                h5_boolean_append.__file__,
                inspect.getsourcefile(validate_source_enrichment_candidate),
                inspect.getsourcefile(ReleaseContractError),
            ),
            strict=True,
        )
    )
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
                for package in ("numpy", "h5py", "microcosm-data")
            },
        },
        "source_files_sha256": {
            filename: file_sha256(root / filename) for filename in _PRODUCER_FILES
        },
    }


def _read_qualification(qualification_dir: Path) -> tuple[bytes, dict, Path]:
    """Return the receipt's bytes, its parsed value and the child it names."""

    receipt_path = qualification_dir / RECEIPT_QUALIFICATION_FILE
    receipt_bytes = receipt_path.read_bytes()
    receipt = json.loads(receipt_bytes)
    if not isinstance(receipt, dict) or receipt.get("operation") != RECEIPT_OPERATION:
        raise ValueError(
            f"{RECEIPT_QUALIFICATION_FILE} is not a reported-receipt qualification"
        )
    parent = receipt.get("parent") if isinstance(receipt.get("parent"), dict) else {}
    if parent.get("sha256") != RECEIPT_PARENT_DATASET_SHA256:
        raise ValueError(
            "The qualification's parent is not the pinned national default "
            f"{RECEIPT_PARENT_BUILD_ID}"
        )
    dataset = receipt.get("dataset") if isinstance(receipt.get("dataset"), dict) else {}
    if dataset.get("filename") != RECEIPT_DATASET_FILENAME:
        raise ValueError(
            f"The qualified child must be named {RECEIPT_DATASET_FILENAME}: run the "
            "qualification on the parent as populace_us_2024.h5"
        )
    child = qualification_dir / RECEIPT_DATASET_FILENAME
    if not child.is_file() or file_sha256(child) != dataset.get("sha256"):
        raise ValueError("The qualified child H5 differs from its receipt")
    return receipt_bytes, receipt, child


def _place_artifact(child: Path, destination: Path) -> None:
    """Hard-link the child beside the release; copy only across devices.

    The qualified child is read-only and hundreds of megabytes, so a link
    avoids a second copy. The contract re-hashes the placed file either way.
    """

    try:
        os.link(child, destination)
    except OSError as exc:
        if exc.errno not in {errno.EXDEV, errno.EPERM, errno.ENOTSUP}:
            raise
        shutil.copyfile(child, destination)
        destination.chmod(0o400)


def _expose_without_overwrite(staging: Path, output_dir: Path) -> None:
    """Reserve ``output_dir`` with an exclusive mkdir, then rename onto it.

    ``rename(2)`` replaces a directory only while it is empty, so this never
    overwrites anything another process put there (the same exposure the
    qualification tool uses).
    """

    occupied = "Output appeared during the build; refusing to overwrite"
    try:
        os.mkdir(output_dir, 0o700)
    except FileExistsError:
        raise FileExistsError(occupied) from None
    try:
        os.rename(staging, output_dir)
    except OSError as exc:
        with contextlib.suppress(OSError):
            os.rmdir(output_dir)
        if exc.errno in {errno.EEXIST, errno.ENOTEMPTY, errno.ENOTDIR}:
            raise FileExistsError(occupied) from exc
        raise


def build_candidate(
    *,
    parent_h5: Path,
    parent_release_dir: Path,
    qualification_dir: Path,
    output_dir: Path,
    release_id: str,
) -> dict:
    """Package, validate, then atomically expose a new local candidate."""

    if (
        not release_id.startswith("populace-us-2024-")
        or Path(release_id).name != release_id
        or release_id in {PARENT_BUILD_ID, RECEIPT_PARENT_BUILD_ID}
        or "\\" in release_id
    ):
        raise ValueError("Choose a new bare US 2024 source-enrichment release ID")
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError("Output directory already exists; choose a new candidate")
    if file_sha256(parent_h5) != RECEIPT_PARENT_DATASET_SHA256:
        raise ValueError(
            f"Only the exact pinned national default {RECEIPT_PARENT_BUILD_ID} "
            "can be the parent"
        )
    for destination, expected in RECEIPT_PARENT_FILES.items():
        source = parent_release_dir / destination.removeprefix("parent_")
        if file_sha256(source) != expected:
            raise ValueError(f"Parent evidence SHA-256 mismatch: {source.name}")
    receipt_bytes, receipt, child = _read_qualification(qualification_dir)
    producer_identity = _producer_identity()

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".receipt-enrichment-", dir=output_dir.parent)
    )
    try:
        os.chmod(staging, 0o700)
        release_dir = staging / "releases" / release_id
        release_dir.mkdir(parents=True, mode=0o700)
        artifact_root = staging / "artifacts"
        artifact_root.mkdir(mode=0o700)
        candidate_h5 = artifact_root / RECEIPT_DATASET_FILENAME
        _place_artifact(child, candidate_h5)
        receipt_path = release_dir / RECEIPT_QUALIFICATION_FILE
        receipt_path.write_bytes(receipt_bytes)
        receipt_path.chmod(0o400)
        for destination in RECEIPT_PARENT_FILES:
            shutil.copyfile(
                parent_release_dir / destination.removeprefix("parent_"),
                release_dir / destination,
            )
            (release_dir / destination).chmod(0o400)
        dataset = {
            "filename": RECEIPT_DATASET_FILENAME,
            "sha256": receipt["dataset"]["sha256"],
        }
        report = {
            "schema_version": 1,
            "release_type": SOURCE_ENRICHMENT_RELEASE_TYPE,
            "operation": RECEIPT_OPERATION,
            "parent": {
                "build_id": RECEIPT_PARENT_BUILD_ID,
                "repo_id": _REPO_ID,
                "revision": RECEIPT_PARENT_BUILD_ID,
                "dataset_sha256": RECEIPT_PARENT_DATASET_SHA256,
                "calibration_diagnostics_schema_version": 5,
                "files": RECEIPT_PARENT_FILES,
            },
            "dataset": dataset,
            "added_variables": [
                {"name": name, "entity": entity, "dtype": "bool"}
                for name, entity in RECEIPT_COLUMNS.items()
            ],
            # The qualification's own proof; the contract below recomputes it
            # from the actual parent and child H5 files and requires equality.
            "preservation": receipt["preservation"],
            "source": {
                "qualification_filename": RECEIPT_QUALIFICATION_FILE,
                "qualification_sha256": file_sha256(receipt_path),
            },
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
                "parent_build_id": RECEIPT_PARENT_BUILD_ID,
                "diagnostics_sha256": RECEIPT_PARENT_FILES[
                    "calibration_diagnostics.json"
                ],
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
                "repo_id": _REPO_ID,
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
            "default_datasets": {"national": _ARTIFACT_KEY},
            "build": {"build_id": release_id},
            "compatible_core_packages": [],
            "compatible_model_packages": [],
            "artifacts": {
                _ARTIFACT_KEY: artifact(candidate_h5, kind="microdata"),
                **{
                    path.name.removesuffix(".json"): artifact(path, kind="evidence")
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
        _expose_without_overwrite(staging, output_dir)
        return report
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parent-h5",
        type=Path,
        required=True,
        help=f"The pinned national default H5 ({RECEIPT_PARENT_BUILD_ID})",
    )
    parser.add_argument(
        "--parent-release-dir",
        type=Path,
        required=True,
        help=(
            "Directory holding the parent's release_manifest.json, "
            "calibration_diagnostics.json and us_source_coverage.json"
        ),
    )
    parser.add_argument(
        "--qualification-dir",
        type=Path,
        required=True,
        help=(
            "Output directory of tools/build_us_acs_donor_receipt_qualification.py "
            "run on the same parent"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--release-id", required=True)
    args = parser.parse_args(argv)
    try:
        report = build_candidate(
            parent_h5=args.parent_h5,
            parent_release_dir=args.parent_release_dir,
            qualification_dir=args.qualification_dir,
            output_dir=args.output_dir,
            release_id=args.release_id,
        )
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Reported-receipt enrichment candidate refused: {exc}\n")
    print(
        json.dumps(
            {"dataset": report["dataset"], "compatibility": report["compatibility"]},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
