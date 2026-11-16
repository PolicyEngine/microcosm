"""Annual publication binds accepted years to exact native H5 artifacts."""

from __future__ import annotations

# ruff: noqa: F401
import hashlib
import json
import shutil

import h5py
import numpy as np
import pytest

from microcosm.data.annual_projections import validate_annual_projection_extension
from microcosm.data.contract import ReleaseContractError, validate_release_dir

from .build_folders import to_build_folder_layout
from .release import (
    RELEASE_ID,
    _bind_stored_input_evidence,
    artifact_root,
    release_dir,
)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _h5(path, year, *, person_id=1, weight=10.0):
    with h5py.File(path, "w") as store:
        store.create_dataset(
            "_time_period/table",
            data=np.array([(0, year)], dtype=[("index", "i8"), ("values", "i8")]),
        )
        for entity in (
            "person",
            "household",
            "tax_unit",
            "spm_unit",
            "family",
            "marital_unit",
        ):
            fields = [("index", "i8"), (f"{entity}_id", "i8")]
            row = [0, person_id if entity == "person" else 1]
            if entity == "person":
                fields.extend(
                    [("person_household_id", "i8"), ("employment_income", "f8")]
                )
                row.extend([1, 100.0 * (year - 2023)])
            if entity == "household":
                fields.append(("household_weight", "f8"))
                row.append(weight)
            store.create_dataset(
                f"{entity}/table", data=np.array([tuple(row)], dtype=fields)
            )


@pytest.fixture
def candidate(tmp_path):
    release = tmp_path / "release-id"
    release.mkdir()
    root = tmp_path / "artifacts"
    root.mkdir()
    mapping = {
        "populace_us_2024": {"2024": "populace_us_2024", "2025": "populace_us_2025"}
    }
    manifest = {
        "metadata": {"dataset_years": mapping},
        "artifacts": {},
        "default_datasets": {"national": "populace_us_2024"},
        "build": {
            "build_id": release.name,
            "built_with_model_package": {
                "name": "policyengine-us",
                "version": "2.6.15",
            },
            "built_with_core_package": {
                "name": "policyengine-core",
                "version": "3.32.5",
            },
        },
    }
    evidence = {
        "schema_version": 1,
        "kind": "us_annual_static_aging_candidate",
        "status": "complete",
        "base": {
            "dataset": "populace_us_2024",
            "year": 2024,
            "parent_release": release.name,
        },
        "metadata": {"dataset_years": mapping},
        "artifacts": {},
        "model": {
            "version": "2.6.15",
            "commit": "a" * 40,
            "source_tree_sha256": "b" * 64,
        },
        "runtime": {
            "versions": {"policyengine-us": "2.6.15", "policyengine-core": "3.32.5"}
        },
    }
    acceptance = {
        "schema_version": 1,
        "kind": "us_annual_projection_acceptance",
        "status": "passed",
        "years": {},
        "model": dict(evidence["model"]),
        "runtime": dict(evidence["runtime"]["versions"]),
    }
    for year in (2024, 2025):
        key = f"populace_us_{year}"
        path = root / f"{key}.h5"
        _h5(path, year)
        sha = _sha(path)
        manifest["artifacts"][key] = {
            "path": path.name,
            "sha256": sha,
            "revision": f"{release.name}-annual-20260919T220000Z-a1b2c3d4",
            "kind": "microdata",
        }
        evidence["artifacts"][key] = {
            "year": year,
            "sha256": sha,
            "rows": {
                name: 1
                for name in (
                    "person",
                    "household",
                    "tax_unit",
                    "spm_unit",
                    "family",
                    "marital_unit",
                )
            },
        }
        acceptance["years"][str(year)] = {
            "dataset": key,
            "sha256": sha,
            "checks": {
                name: "passed"
                for name in (
                    "schema",
                    "source_identity",
                    "year",
                    "demographics",
                    "input_aggregates",
                    "runtime",
                )
            },
        }
    evidence["base"]["sha256"] = manifest["artifacts"]["populace_us_2024"]["sha256"]
    projection_path = release / "projection_2025.json"
    projection_path.write_text(
        json.dumps({"base_year": 2024, "year": 2025, "factors": {}})
    )
    evidence["artifacts"]["populace_us_2025"]["projection_receipt"] = {
        "path": projection_path.name,
        "sha256": _sha(projection_path),
    }
    manifest["artifacts"]["projection_2025"] = {
        "path": f"releases/{release.name}/{projection_path.name}",
        "sha256": _sha(projection_path),
        "revision": f"{release.name}-annual-20260919T220000Z-a1b2c3d4",
        "kind": "diagnostics",
    }

    def save():
        evidence_path = release / "annual_manifest.json"
        evidence_path.write_text(json.dumps(evidence))
        acceptance["projection_manifest_sha256"] = _sha(evidence_path)
        acceptance_path = release / "annual_acceptance.json"
        acceptance_path.write_text(json.dumps(acceptance))
        for key, field, path in (
            ("annual_manifest", "annual_projection_manifest", evidence_path),
            ("annual_acceptance", "annual_projection_acceptance", acceptance_path),
        ):
            manifest["metadata"][field] = key
            manifest["artifacts"][key] = {
                "path": f"releases/{release.name}/{path.name}",
                "sha256": _sha(path),
                "revision": f"{release.name}-annual-20260919T220000Z-a1b2c3d4",
            }
        (release / "release_manifest.json").write_text(json.dumps(manifest))

    save()
    return release, root, manifest, evidence, acceptance, save


def add_annual_extension(release, root):
    """Augment an otherwise valid release without replacing its base evidence."""
    manifest = json.loads((release / "release_manifest.json").read_text())
    base_key = manifest["default_datasets"]["national"]
    base = manifest["artifacts"][base_key]
    projected = root / "annual_2025.h5"
    shutil.copyfile(root / base["path"], projected)
    with h5py.File(projected, "r+") as store:
        row = store["_time_period/table"][:]
        row["values"] = 2025
        store["_time_period/table"][:] = row
        rows = {
            entity: len(store[f"{entity}/table"])
            for entity in (
                "person",
                "household",
                "tax_unit",
                "spm_unit",
                "family",
                "marital_unit",
            )
        }
    runtime = {
        "policyengine-us": manifest["build"]["built_with_model_package"]["version"],
        "policyengine-core": manifest["build"]["built_with_core_package"]["version"],
    }
    model = {
        "version": runtime["policyengine-us"],
        "commit": "a" * 40,
        "source_tree_sha256": "b" * 64,
    }
    mapping = {base_key: {"2024": base_key, "2025": "annual_2025"}}
    evidence = {
        "schema_version": 1,
        "kind": "us_annual_static_aging_candidate",
        "status": "complete",
        "base": {
            "dataset": base_key,
            "year": 2024,
            "sha256": base["sha256"],
            "parent_release": release.name,
        },
        "metadata": {"dataset_years": mapping},
        "model": model,
        "runtime": {"versions": runtime},
        "artifacts": {},
    }
    acceptance = {
        "schema_version": 1,
        "kind": "us_annual_projection_acceptance",
        "status": "passed",
        "model": model,
        "runtime": runtime,
        "years": {},
    }
    manifest["artifacts"]["annual_2025"] = {
        "kind": "microdata",
        "path": projected.name,
        "sha256": _sha(projected),
        "repo_id": base["repo_id"],
    }
    for year, key in mapping[base_key].items():
        sha = manifest["artifacts"][key]["sha256"]
        evidence["artifacts"][key] = {"year": int(year), "sha256": sha, "rows": rows}
        acceptance["years"][year] = {
            "dataset": key,
            "sha256": sha,
            "checks": {
                name: "passed"
                for name in (
                    "schema",
                    "source_identity",
                    "year",
                    "demographics",
                    "input_aggregates",
                    "runtime",
                )
            },
        }
    projection_path = release / "projection_2025.json"
    projection_path.write_text(
        json.dumps({"base_year": 2024, "year": 2025, "factors": {}})
    )
    evidence["artifacts"]["annual_2025"]["projection_receipt"] = {
        "path": projection_path.name,
        "sha256": _sha(projection_path),
    }
    manifest["artifacts"]["projection_2025"] = {
        "kind": "diagnostics",
        "path": f"releases/{release.name}/{projection_path.name}",
        "sha256": _sha(projection_path),
        "repo_id": base["repo_id"],
    }
    evidence_path = release / "annual_manifest.json"
    evidence_path.write_text(json.dumps(evidence))
    acceptance["projection_manifest_sha256"] = _sha(evidence_path)
    acceptance_path = release / "annual_acceptance.json"
    acceptance_path.write_text(json.dumps(acceptance))
    manifest["metadata"] = {
        **manifest.get("metadata", {}),
        "dataset_years": mapping,
        "annual_projection_manifest": "annual_manifest",
        "annual_projection_acceptance": "annual_acceptance",
    }
    for key, path in (
        ("annual_manifest", evidence_path),
        ("annual_acceptance", acceptance_path),
    ):
        manifest["artifacts"][key] = {
            "kind": "diagnostics",
            "path": f"releases/{release.name}/{path.name}",
            "sha256": _sha(path),
            "repo_id": base["repo_id"],
        }
    tag = f"{release.name}-annual-20260919T220000Z-a1b2c3d4"
    for artifact in manifest["artifacts"].values():
        artifact["revision"] = tag
    (release / "release_manifest.json").write_text(json.dumps(manifest))
    return tag


#: The US releases an annual build folder involves (microcosm#1570).
US_NATIONAL_RELEASE = "microcosm-us-2024-national"
US_ANNUAL_RELEASE = "microcosm-us-2024-annual"
#: The annual build the build-folder fixture names: its base build's ID, the
#: time it was assembled and eight hex digits.
ANNUAL_BUILD = f"{RELEASE_ID}-annual-20261110T120000Z-a1b2c3d4"
_ENTITIES = ("person", "household", "tax_unit", "spm_unit", "family", "marital_unit")
_ACCEPTANCE_CHECKS = (
    "schema",
    "source_identity",
    "year",
    "demographics",
    "input_aggregates",
    "runtime",
)


def _read(path):
    return json.loads(path.read_text())


def _write(path, document):
    path.write_text(json.dumps(document))


def national_build_folder(tagged_dir, root, *, prefix="microcosm"):
    """The US calibration fixture as a build of the national release.

    Its dataset is a native single-year H5 for 2024 that an annual build can
    project, under ``<prefix>_us_2024.h5``: ``microcosm`` for a new build,
    ``populace`` for one of the builds the switch moved. The calibration
    weights take the same prefix, and the stored-input evidence is bound to
    the new H5.
    """
    dataset_key = f"{prefix}_us_2024"
    calibration_key = f"{dataset_key}_calibration"
    (root / "populace_us_2024.h5").unlink()
    dataset = root / f"{dataset_key}.h5"
    _h5(dataset, 2024)
    (root / "populace_us_2024_calibration.npz").rename(root / f"{calibration_key}.npz")
    build_path = tagged_dir / "build_manifest.json"
    build = _read(build_path)
    build["dataset"] = {"filename": dataset.name, "sha256": _sha(dataset)}
    build["calibration"]["filename"] = f"{calibration_key}.npz"
    _write(build_path, build)
    manifest_path = tagged_dir / "release_manifest.json"
    manifest = _read(manifest_path)
    renamed = {
        "populace_us_2024": dataset_key,
        "populace_us_2024_calibration": calibration_key,
    }
    manifest["artifacts"] = {
        renamed.get(key, key): (
            {
                **entry,
                "path": entry["path"].replace("populace_", f"{prefix}_"),
            }
            if key in renamed
            else entry
        )
        for key, entry in manifest["artifacts"].items()
    }
    manifest["artifacts"][dataset_key]["sha256"] = _sha(dataset)
    manifest["default_datasets"] = {"national": dataset_key}
    _write(manifest_path, manifest)
    _bind_stored_input_evidence(tagged_dir)
    national = tagged_dir.with_name(US_NATIONAL_RELEASE)
    tagged_dir.rename(national)
    return to_build_folder_layout(national, build=RELEASE_ID)


def annual_build_folder_of(national, root, *, build=ANNUAL_BUILD):
    """An annual build of the national build folder ``national``.

    The annual release's folder carries every file of the national build but
    its two manifests, a projected H5 for 2025 in ``root``, the
    candidate manifest, its passing acceptance report and the 2025 receipt.
    Both manifests name the annual release and ``build``, and the release
    manifest names the national build in ``metadata.annual_projection_base``.
    """
    annual = national.with_name(US_ANNUAL_RELEASE)
    annual.mkdir()
    for path in national.iterdir():
        if path.name not in {"release_manifest.json", "build_manifest.json"}:
            shutil.copyfile(path, annual / path.name)
    manifest = _read(national / "release_manifest.json")
    base_build = manifest["build"]
    base_key = manifest["default_datasets"]["national"]
    base = manifest["artifacts"][base_key]
    repo_id = base["repo_id"]
    with h5py.File(root / base["path"], "r") as store:
        rows = {entity: len(store[f"{entity}/table"]) for entity in _ENTITIES}
    projected_key = "microcosm_us_2025"
    mapping = {base_key: {"2024": base_key, "2025": projected_key}}
    runtime = {
        "policyengine-us": manifest["built_with_model_package"]["version"],
        "policyengine-core": manifest["built_with_core_package"]["version"],
    }
    model = {
        "version": runtime["policyengine-us"],
        "commit": "a" * 40,
        "source_tree_sha256": "b" * 64,
    }
    projected = root / f"{projected_key}.h5"
    _h5(projected, 2025)
    receipt = annual / "projection_2025.json"
    receipt.write_text(json.dumps({"base_year": 2024, "year": 2025, "factors": {}}))
    evidence = {
        "schema_version": 1,
        "kind": "us_annual_static_aging_candidate",
        "status": "complete",
        "base": {
            "dataset": base_key,
            "year": 2024,
            "parent_release": base_build,
            "path": base["path"],
            "sha256": base["sha256"],
        },
        "metadata": {"dataset_years": mapping},
        "model": model,
        "runtime": {"versions": runtime},
        "artifacts": {
            base_key: {"year": 2024, "sha256": base["sha256"], "rows": rows},
            projected_key: {
                "year": 2025,
                "sha256": _sha(projected),
                "rows": rows,
                "projection_receipt": {"path": receipt.name, "sha256": _sha(receipt)},
            },
        },
    }
    acceptance = {
        "schema_version": 1,
        "kind": "us_annual_projection_acceptance",
        "status": "passed",
        "model": model,
        "runtime": runtime,
        "years": {
            year: {
                "dataset": key,
                "sha256": evidence["artifacts"][key]["sha256"],
                "checks": {name: "passed" for name in _ACCEPTANCE_CHECKS},
            }
            for year, key in mapping[base_key].items()
        },
    }

    def declared(key, kind, path):
        return {
            "kind": kind,
            "path": path.name,
            "repo_id": repo_id,
            "sha256": _sha(path),
            "role": "dataset" if kind == "microdata" else key,
        }

    evidence_path = annual / "annual_manifest.json"
    _write(evidence_path, evidence)
    acceptance["projection_manifest_sha256"] = _sha(evidence_path)
    acceptance_path = annual / "annual_acceptance.json"
    _write(acceptance_path, acceptance)
    manifest["release"] = US_ANNUAL_RELEASE
    manifest["build"] = build
    manifest["metadata"] = {
        **manifest.get("metadata", {}),
        "dataset_years": mapping,
        "annual_projection_manifest": "annual_manifest",
        "annual_projection_acceptance": "annual_acceptance",
        "annual_projection_base": {
            "repo_id": repo_id,
            "release": US_NATIONAL_RELEASE,
            "build": base_build,
            "path": base["path"],
            "sha256": base["sha256"],
        },
    }
    manifest["artifacts"].update(
        {
            projected_key: declared(projected_key, "microdata", projected),
            "projection_2025": declared("projection_2025", "diagnostics", receipt),
            "annual_manifest": declared(
                "annual_manifest", "diagnostics", evidence_path
            ),
            "annual_acceptance": declared(
                "annual_acceptance", "diagnostics", acceptance_path
            ),
        }
    )
    _write(annual / "release_manifest.json", manifest)
    build_manifest = _read(national / "build_manifest.json")
    build_manifest.update(release=US_ANNUAL_RELEASE, build=build)
    _write(annual / "build_manifest.json", build_manifest)
    return annual


def rewrite_annual_release_manifest(annual, mutate):
    """Change an annual folder's release manifest; nothing records its digest."""
    path = annual / "release_manifest.json"
    manifest = _read(path)
    mutate(manifest)
    _write(path, manifest)


def rewrite_annual_evidence(annual, mutate):
    """Change an annual folder's candidate manifest and reseal what binds it:
    the acceptance report's ``projection_manifest_sha256`` and both files'
    digests in the release manifest."""
    evidence_path = annual / "annual_manifest.json"
    evidence = _read(evidence_path)
    mutate(evidence)
    _write(evidence_path, evidence)
    acceptance_path = annual / "annual_acceptance.json"
    acceptance = _read(acceptance_path)
    acceptance["projection_manifest_sha256"] = _sha(evidence_path)
    _write(acceptance_path, acceptance)

    def reseal(manifest):
        manifest["artifacts"]["annual_manifest"]["sha256"] = _sha(evidence_path)
        manifest["artifacts"]["annual_acceptance"]["sha256"] = _sha(acceptance_path)

    rewrite_annual_release_manifest(annual, reseal)


@pytest.fixture
def national_build(release_dir, artifact_root):  # noqa: F811
    """A build of the US national release in its build folder."""
    return national_build_folder(release_dir, artifact_root)


@pytest.fixture
def annual_build_folder(national_build, artifact_root):  # noqa: F811
    """An annual build of :func:`national_build` in the annual release."""
    return annual_build_folder_of(national_build, artifact_root)


__all__ = [name for name in globals() if not name.startswith("__")]
