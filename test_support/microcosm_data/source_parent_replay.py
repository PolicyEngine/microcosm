"""MOCK_METADATA_ONLY source-parent metadata, never a real H5/runtime fixture.

All hashes authenticate these tiny dummy bytes. The wheel and dataset files
are deliberately not wheels or HDF5; no model/native predicate may load them.
Only test-local transport spies use this metadata to exercise orchestration.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from microcosm.data import source_parent_replay as replay

MOCK_BUILD_ID = "populace-us-2024-MOCK_METADATA_ONLY-source-parent"
RUNTIME_KEYS = ("python", "hdf5", "microcosm-data", "numpy", "pandas", "h5py", "tables")
CURRENT_CHECKS = (
    "source_inheritance",
    "exhaustive_h5_preservation",
    "reviewed_source_evidence",
    "current_native_loaders",
    "current_stored_input_register",
    "current_claim_measurement",
    "current_producer_identity",
)


def mock_bytes(label):
    return f"MOCK_METADATA_ONLY:{label}\n".encode()


def mock_sha(label):
    return hashlib.sha256(mock_bytes(label)).hexdigest()


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def mock_legacy_receipt(dataset_sha):
    """Preserve the legacy field/check structure with explicit mock identities."""
    columns = (
        "person.person_id",
        "person.is_spm_independent_minor_role",
        "person.person_tax_unit_id",
        "person.person_spm_unit_id",
        "person.person_family_id",
        "person.person_marital_unit_id",
        "person.person_household_id",
        "person.person_source_id",
        "household.household_id",
        "household.household_weight",
        "tax_unit.tax_unit_id",
        "spm_unit.spm_unit_id",
        "family.family_id",
        "marital_unit.marital_unit_id",
    )
    return {
        "schema_version": 1,
        "status": "passed",
        "scope": "native_input_loading_only",
        "external_package_publication": "not_attested",
        "canonical_spm_model_acceptance": "not_attested",
        "dataset_sha256": dataset_sha,
        "runner_sha256": mock_sha("original_runner"),
        "packages": {
            name: {
                "version": "0.1.0+mock.metadata.only",
                "wheel_sha256": mock_sha(f"wheel:{name}"),
                "source_sha256": mock_sha(f"package_source:{name}"),
                "direct_url": {
                    "archive_info": {},
                    "url": f"mock-metadata-only://{name}",
                },
            }
            for name in (
                "policyengine",
                "policyengine-core",
                "policyengine-us",
                "spm-calculator",
            )
        },
        "loaded_source_packages": {
            name: f"MOCK_METADATA_ONLY.{name}"
            for name in ("country_loader", "native_role", "wrapper_loader")
        },
        "loaded_source_sha256": {
            name: mock_sha(f"loaded_source:{name}")
            for name in ("country_loader", "native_role", "wrapper_loader")
        },
        "checks": [
            "country:registered_person_bool_input",
            "country:complete_household_native_input_overrides_default",
            *(
                f"{loader}:{column}"
                for column in columns
                for loader in ("country", "wrapper")
            ),
        ],
    }


@dataclass(frozen=True)
class MockSourceParent:
    release_dir: Path
    artifact_root: Path
    context: replay.SourceParentReplayContext
    profile: dict
    runtime: dict
    legacy: dict
    original_bytes: dict


def make_mock_source_parent(tmp_path, *, with_claim=False):
    """Write metadata and dummy bytes; leave production profile unpatched."""
    release = tmp_path / MOCK_BUILD_ID
    release.mkdir()
    root = tmp_path / "MOCK_METADATA_ONLY_artifacts"
    root.mkdir()
    dataset = root / "MOCK_METADATA_ONLY_NOT_HDF5.h5"
    dataset.write_bytes(mock_bytes("dataset"))
    ancestor = tmp_path / "MOCK_METADATA_ONLY_ANCESTOR_NOT_HDF5.h5"
    ancestor.write_bytes(mock_bytes("ancestor_dataset"))
    checkout = tmp_path / "MOCK_METADATA_ONLY_NOT_A_SOURCE_CHECKOUT"
    checkout.mkdir()
    python = tmp_path / "MOCK_METADATA_ONLY_NOT_AN_INTERPRETER"
    python.write_bytes(mock_bytes("not_an_interpreter"))
    wheels = []
    for package in (
        "policyengine",
        "policyengine-core",
        "policyengine-us",
        "spm-calculator",
    ):
        wheel = tmp_path / f"MOCK_METADATA_ONLY_{package}.whl"
        wheel.write_bytes(mock_bytes(f"wheel:{package}"))
        wheels.append(wheel)
    runtime = {key: "0.1.0+mock.metadata.only" for key in RUNTIME_KEYS}
    legacy = mock_legacy_receipt(file_sha(dataset))
    write_json(release / "source_enrichment_compatibility.json", legacy)
    write_json(
        release / "parent_release_manifest.json",
        {"schema_version": 1, "release_type": "calibration"},
    )
    write_json(release / "parent_build_manifest.json", {"MOCK_METADATA_ONLY": True})
    build = {
        "build_id": MOCK_BUILD_ID,
        "release_type": "source_enrichment",
        "code": {
            "git_commit": "MOCK_METADATA_ONLY_NOT_A_REAL_COMMIT",
            "git_dirty": False,
            "runtime": runtime,
        },
    }
    write_json(release / "build_manifest.json", build)
    report = {
        "schema_version": 1,
        "operation": "add_native_spm_independent_minor_role",
        "dataset": {"filename": dataset.name, "sha256": file_sha(dataset)},
        "parent": {"dataset_sha256": file_sha(ancestor)},
    }
    if with_claim:
        report["compatibility"] = {
            "status": "passed",
            "publisher_claims": {
                "model": {
                    "name": "policyengine-us",
                    "specifier": ">=0.1,<0.2",
                    "basis": "publisher_claim",
                    "declared_by": "MOCK_METADATA_ONLY fixture declarer",
                }
            },
        }
    write_json(release / "source_enrichment.json", report)
    artifacts = {
        path.stem: {
            "kind": "diagnostics",
            "path": path.name,
            "repo_id": "MOCK_METADATA_ONLY/not-a-published-dataset",
            "sha256": file_sha(path),
        }
        for path in sorted(release.iterdir())
    }
    artifacts["mock_dataset"] = {
        "kind": "microdata",
        "path": dataset.name,
        "repo_id": "MOCK_METADATA_ONLY/not-a-published-dataset",
        "sha256": file_sha(dataset),
    }
    manifest = {
        "schema_version": 1,
        "release_type": "source_enrichment",
        "dataset_role": "national_default",
        "build": {"build_id": MOCK_BUILD_ID},
        "default_datasets": {"national": "mock_dataset"},
        "artifacts": artifacts,
    }
    write_json(release / "release_manifest.json", manifest)
    profile = {
        "build_id": MOCK_BUILD_ID,
        "release_manifest_sha256": file_sha(release / "release_manifest.json"),
        "build_manifest_sha256": file_sha(release / "build_manifest.json"),
        "source_report_sha256": file_sha(release / "source_enrichment.json"),
        "native_receipt_sha256": file_sha(
            release / "source_enrichment_compatibility.json"
        ),
        "dataset_sha256": file_sha(dataset),
        "ancestor_dataset_sha256": file_sha(ancestor),
        "producer_commit": build["code"]["git_commit"],
        "producer_runner_sha256": legacy["runner_sha256"],
    }
    context = replay.SourceParentReplayContext(
        original_python=python,
        original_checkout=checkout,
        original_wheels=tuple(wheels),
        ancestor_h5=ancestor,
    )
    original_bytes = {
        path: path.read_bytes() for path in (*release.iterdir(), dataset, ancestor)
    }
    return MockSourceParent(
        release, root, context, profile, runtime, legacy, original_bytes
    )


MOCK_PRODUCER_FILES = (
    "tools/build_us_spm_role_enrichment.py",
    "packages/microcosm-build/src/microcosm/build/us_runtime/spm_role_source.py",
    "packages/microcosm-build/src/microcosm/build/us_runtime/education_assistance_source.py",
    "packages/microcosm-data/src/microcosm/data/h5_enrichment.py",
    "packages/microcosm-data/src/microcosm/data/source_enrichment.py",
    "packages/microcosm-data/src/microcosm/data/contract.py",
    "packages/microcosm-data/src/microcosm/data/build_folder_manifests.py",
    "packages/microcosm-data/src/microcosm/data/source_parent_replay.py",
    "packages/microcosm-data/src/microcosm/data/source_parent_worker.py",
    "packages/microcosm-data/src/microcosm/data/compatibility_recertification.py",
    "tools/build_us_compatibility_recertification.py",
    "packages/microcosm-data/src/microcosm/data/release.py",
    "packages/microcosm-build/src/microcosm/build/release_publication.py",
)
# A mock worker's inventory names every path the current producer inventory
# binds, including the committed current Data and Diagnostics closures. Only
# the path list grows; every digest stays a MOCK_METADATA_ONLY label digest.
MOCK_PRODUCER_FILES = tuple(
    dict.fromkeys((*MOCK_PRODUCER_FILES, *replay.SOURCE_PARENT_IMPLEMENTATION_FILES))
)


def mock_current_record(source):
    """A mock worker's schema-shaped return, never real current qualification."""
    import copy

    native = copy.deepcopy(source.legacy)
    native["runner_sha256"] = mock_sha("MOCK_current_runner")
    native["checks"].insert(1, "country:defines_every_stored_model_input")
    native["stored_inputs"] = {
        "register_sha256": mock_sha("MOCK_current_register"),
        "registered_non_variables": ["person_source_id"],
    }
    compatibility = {"status": "passed"}
    original = json.loads((source.release_dir / "source_enrichment.json").read_text())
    claims = original.get("compatibility", {}).get("publisher_claims")
    if claims is not None:
        compatibility["publisher_claims"] = copy.deepcopy(claims)
        compatibility["claim_measurement"] = {
            "model": {
                "specifier": claims["model"]["specifier"],
                "measured_versions": [native["packages"]["policyengine-us"]["version"]],
            }
        }
    return {
        "schema_version": 2,
        "kind": "source_parent_current_qualification",
        "profile": dict(source.profile),
        "checks": list(CURRENT_CHECKS),
        "native_receipt": native,
        "compatibility": compatibility,
        "producer_code": {
            # A digest of dummy bytes, not a claim that any Git commit exists.
            "git_commit": mock_sha("MOCK_NOT_A_GIT_COMMIT")[:40],
            "git_dirty": False,
            "source_files_sha256": {
                path: mock_sha(f"MOCK_current_source:{path}")
                for path in MOCK_PRODUCER_FILES
            },
        },
    }


@dataclass(frozen=True)
class MockCarriedSourceParent:
    release_dir: Path
    artifact_root: Path
    parent: dict
    evidence: dict


def carry_mock_source_parent(source, tmp_path, evidence):
    """Construct only carried metadata/byte layout, never a releasable build."""
    import shutil

    child = tmp_path / "MOCK_CHILD" / "microcosm-us-2024-national"
    child.mkdir(parents=True)
    root = tmp_path / "MOCK_CHILD" / "artifacts"
    root.mkdir()
    manifest = json.loads((source.release_dir / "release_manifest.json").read_text())
    mapping = replay.build_source_parent_artifact_map(
        manifest,
        child_dataset_key="microcosm_us_2024",
        child_dataset_filename="microcosm_us_2024.h5",
    )
    release_files = {}
    root_artifacts = {}
    for row in mapping:
        original = source.release_dir / row["source_path"]
        if original.is_file():
            destination = child / row["carried_path"]
            if row["source_key"] != "build_manifest":
                release_files[row["source_path"]] = row["sha256"]
        else:
            original = source.artifact_root / row["source_path"]
            destination = root / row["carried_path"]
            root_artifacts[row["source_path"]] = row["sha256"]
        shutil.copyfile(original, destination)
    shutil.copyfile(
        source.release_dir / "release_manifest.json",
        child / "parent_release_manifest.json",
    )
    parent = {
        "build_id": source.release_dir.name,
        "repo_id": "policyengine/populace-us",
        "revision": source.release_dir.name,
        "release_type": "source_enrichment",
        "dataset_role": "national_default",
        "release_manifest_sha256": source.profile["release_manifest_sha256"],
        "build_manifest_sha256": source.profile["build_manifest_sha256"],
        "dataset": {
            "artifact": "mock_dataset",
            "filename": manifest["artifacts"]["mock_dataset"]["path"],
            "sha256": source.profile["dataset_sha256"],
        },
        "release_files": release_files,
        "root_artifacts": root_artifacts,
        "source_parent_profile": dict(source.profile),
        "source_parent_artifact_map": list(mapping),
    }
    write_json(child / replay.SOURCE_PARENT_EVIDENCE_FILE, evidence)
    return MockCarriedSourceParent(child, root, parent, evidence)


def write_mock_pending_headers(carried):
    """Only headers needed to reach context refusal, not a valid release."""
    child_build = "populace-us-2024-MOCK_METADATA_ONLY-child"
    manifest = {
        "schema_version": 1,
        "release_type": "compatibility_recertification",
        "release": carried.release_dir.name,
        "build": child_build,
        "artifacts": {
            "source_parent_evidence_v2": {
                "kind": "diagnostics",
                "path": replay.SOURCE_PARENT_EVIDENCE_FILE,
                "repo_id": "MOCK_METADATA_ONLY/not-a-published-dataset",
                "sha256": file_sha(
                    carried.release_dir / replay.SOURCE_PARENT_EVIDENCE_FILE
                ),
            }
        },
    }
    build = {
        "release_type": "compatibility_recertification",
        "release": carried.release_dir.name,
        "build": child_build,
    }
    report = {
        "schema_version": 2,
        "release_type": "compatibility_recertification",
        "transformation": "none",
        "parent": carried.parent,
        "compatibility": {"status": "pending"},
    }
    write_json(carried.release_dir / "release_manifest.json", manifest)
    write_json(carried.release_dir / "build_manifest.json", build)
    write_json(carried.release_dir / "compatibility_recertification.json", report)
